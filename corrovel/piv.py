"""PIV (vélocimétrie par images de particules) par intercorrélation : troisième méthode, entre
Trackpy (particules détectées une à une puis reliées) et le flux optique (gradients de luminosité).

Principe : chaque cellule de la grille est une fenêtre d'interrogation. Pour une paire de frames,
l'intercorrélation (par FFT) de la fenêtre à t et de la même fenêtre à t + dt a son maximum au
déplacement du motif de particules qu'elle contient. Choix propres aux traceurs clairsemés en
microscopie (micro-PIV) :
- corrélation d'ENSEMBLE : les plans de corrélation de toutes les paires d'une fenêtre de temps
  sont additionnés avant de chercher le pic (Meinhart, Wereley & Santiago 2000). Avec peu de
  particules par fenêtre, une paire isolée donne un pic noyé dans le bruit ; la somme fait
  ressortir le déplacement commun. Hypothèse : écoulement à peu près stationnaire sur la fenêtre ;
- 2e passe à fenêtre décalée : la fenêtre de la 2e image est déplacée du déplacement entier
  estimé à la 1re passe (moitié de chaque côté), ce qui récupère les particules sorties de la
  fenêtre (« perte de paires ») et rend la mesure fiable jusqu'à ~1/2 cellule par intervalle ;
- position sous-pixel : ajustement gaussien sur 3 points autour du pic (parabolique en secours) ;
- validation : rapport pic principal / 2e pic (un pic à peine plus haut que le bruit n'est pas
  une mesure) puis test de médiane normalisée (Westerweel & Scarano 2005) contre les vecteurs
  aberrants isolés. Les cellules rejetées restent non mesurées (NaN) : rien n'est inventé.
Convention : déplacement (dx, dy) en px, x vers la droite, y vers le bas, du motif à t vers t + dt.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter


@dataclass
class PIVParams:
    passes: int = 2                # 1 ou 2 (2e passe à fenêtre décalée)
    min_peak_ratio: float = 1.3    # pic principal / 2e pic ; en dessous : cellule non mesurée
    median_test: float = 2.0       # seuil du test de médiane normalisée ; 0 = désactivé

    def to_dict(self) -> dict:
        return asdict(self)


_FFT_WORKERS = -1      # threads FFT ; 1 dans les processus parallèles (pas de sur-souscription)


def _patches(img: np.ndarray, y0: np.ndarray, x0: np.ndarray, size: int) -> np.ndarray:
    """Fenêtres size × size de coins haut-gauche (y0, x0) (entiers, dans l'image `img`) -> (n, size, size).
    Vue glissante de l'image indexée par les coins : bien plus rapide qu'un indexage 3-D."""
    return np.lib.stride_tricks.sliding_window_view(img, (size, size))[y0, x0]


def _zero_mean(p: np.ndarray) -> np.ndarray:
    """Retire la moyenne de chaque fenêtre (calculée sur ses seuls pixels dans l'image ; ceux hors
    de l'image, NaN, valent ensuite 0 et ne contribuent pas à la corrélation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(p, axis=(1, 2), keepdims=True)
    return np.nan_to_num(p - m, nan=0.0).astype(np.float32)


def _subpixel(line: np.ndarray) -> np.ndarray:
    """Décalage sous-pixel du sommet de profils (n, 2r + 1) centrés sur le maximum entier :
    parabole ajustée aux moindres carrés sur log(profil) (pic gaussien) si toutes les valeurs
    sont > 0, sinon sur le profil lui-même. Avec r = 1, c'est l'estimateur gaussien à 3 points
    classique ; sur un pic large, l'ajuster sur ~sa demi-largeur évite le « verrouillage » sur
    les pixels entiers (courbure sur 3 points presque plate, dominée par le bruit)."""
    n, m = line.shape
    o = np.arange(m, dtype=float) - (m - 1) / 2
    pos = (line > 0).all(axis=1)
    y = np.where(pos[:, None], np.log(np.where(line > 0, line, 1.0)), line)
    s2, s4 = float((o ** 2).sum()), float((o ** 4).sum())
    b = (y * o).sum(axis=1) / s2
    a = ((y * o ** 2).sum(axis=1) - s2 / m * y.sum(axis=1)) / (s4 - s2 ** 2 / m)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(a < 0, -b / (2 * a), 0.0)
    return np.clip(np.nan_to_num(d), -1.0, 1.0)


def _peaks(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pic de chaque plan de corrélation (n, s, s), centré (déplacement nul en s // 2).

    Retourne (dy, dx) en px (sous-pixel), le rapport pic principal / 2e pic, et un masque des
    pics trop près du bord du plan (déplacement ≥ ~1/2 fenêtre : mesure non fiable).
    Le 2e pic est le plus haut MAXIMUM LOCAL hors du pic principal, dont on exclut une zone de
    2 × sa demi-largeur à mi-hauteur (au moins 2 px) : sans cela, sur un pic large (texture lisse,
    grosses particules), une petite bosse de bruit sur son flanc passerait pour un 2e pic.
    """
    n, s, _ = corr.shape
    flat = corr.reshape(n, -1)
    k = np.argmax(flat, axis=1)
    py, px = np.unravel_index(k, (s, s))
    rows = np.arange(n)
    p1 = flat[rows, k]
    half = np.ones(n, dtype=int)                  # demi-largeur à mi-hauteur du pic principal (px)
    for d in range(1, s // 2):
        wide = ((corr[rows, py, (px + d) % s] > 0.5 * p1) | (corr[rows, py, (px - d) % s] > 0.5 * p1)
                | (corr[rows, (py + d) % s, px] > 0.5 * p1) | (corr[rows, (py - d) % s, px] > 0.5 * p1))
        grow = wide & (half == d)
        if not grow.any():
            break
        half = np.where(grow, d + 1, half)
    # Ajustement sous-pixel sur ± r points, r ≈ demi-largeur du pic (1 pour un pic étroit)
    radius = np.clip(half - 1, 1, max(1, s // 4))
    sub_y = np.zeros(n)
    sub_x = np.zeros(n)
    for r in np.unique(radius):
        g = np.flatnonzero(radius == r)
        o = np.arange(-r, r + 1)
        sub_y[g] = _subpixel(corr[g[:, None], (py[g, None] + o) % s, px[g, None]])
        sub_x[g] = _subpixel(corr[g[:, None], py[g, None], (px[g, None] + o) % s])
    dy = py - s // 2 + sub_y
    dx = px - s // 2 + sub_x
    excl = np.maximum(2, 2 * half)[:, None, None]
    iy = np.arange(s)[None, :, None]
    ix = np.arange(s)[None, None, :]
    near = (np.abs(iy - py[:, None, None]) <= excl) & (np.abs(ix - px[:, None, None]) <= excl)
    local_max = corr == maximum_filter(corr, size=(1, 3, 3), mode="wrap")
    p2 = np.where(local_max & ~near, corr, -np.inf).reshape(n, -1).max(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(p1 <= 0, 0.0, np.where(p2 > 0, p1 / p2, np.inf))
    edge = (np.abs(py - s // 2) >= s // 2 - 1) | (np.abs(px - s // 2) >= s // 2 - 1)
    return dy, dx, ratio, edge


def _ensemble_corr(stack: np.ndarray, i0: int, i1: int, y0: np.ndarray, x0: np.ndarray, size: int,
                   pad: int, sy: np.ndarray, sx: np.ndarray,
                   progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Somme, sur les paires (i, i + 1) de [i0, i1), des intercorrélations de chaque fenêtre de i
    avec la fenêtre de i + 1 ; fenêtre de i décalée de −⌊s/2⌋, celle de i + 1 de s − ⌊s/2⌋
    (s = décalage entier par cellule, 0 à la 1re passe).

    Pour chaque décalage, covariance des pixels qui se RECOUVRENT, moyennes prises sur ce
    recouvrement (comme en corrélation d'images numériques), et divisée par leur nombre :
        cov(s) = [Σ a·b − (Σ a)(Σ b) / n] / n,   sommes sur le recouvrement, n = sa taille.
    Chaque terme est une corrélation calculée par FFT (fenêtres complétées de zéros ; Σ a et Σ b
    via les masques des fenêtres, exacts aussi pour les cellules partielles du bord). Les deux
    raccourcis habituels biaisent le pic vers zéro, d'autant plus qu'il est large : ne pas
    diviser par n (le poids du recouvrement décroît avec le décalage, Westerweel 1997), et
    retirer une seule fois la moyenne de chaque fenêtre (sur une texture lisse à peu de motifs
    par fenêtre, cette moyenne dépend du contenu). Sur une texture lisse, 1.5 px était mesuré
    ~0.5 px sans ces corrections. Décalages à moins de 25 % de recouvrement : exclus.
    Retourne les plans (n, size, size), déplacement nul en size // 2.
    """
    ay, ax_ = y0 + pad - sy // 2, x0 + pad - sx // 2
    by, bx = ay + sy, ax_ + sx
    L = 2 * size
    inside = np.pad(np.ones(stack.shape[1:], np.float32), pad)
    MA = sfft.rfft2(_patches(inside, ay, ax_, size), s=(L, L), workers=_FFT_WORKERS)
    MB = sfft.rfft2(_patches(inside, by, bx, size), s=(L, L), workers=_FFT_WORKERS)
    acc = prod = None
    prev = None
    for j, i in enumerate(range(i0, i1)):
        fa = prev if prev is not None else np.pad(stack[i].astype(np.float32), pad, constant_values=np.nan)
        fb = np.pad(stack[i + 1].astype(np.float32), pad, constant_values=np.nan)
        # moyenne de fenêtre retirée d'abord : sans effet sur la covariance, évite les pertes de
        # précision (images 16 bits) dans (Σ a)(Σ b) / n
        A = sfft.rfft2(_zero_mean(_patches(fa, ay, ax_, size)), s=(L, L), workers=_FFT_WORKERS)
        B = sfft.rfft2(_zero_mean(_patches(fb, by, bx, size)), s=(L, L), workers=_FFT_WORKERS)
        spec = np.conj(A) * B
        acc = spec if acc is None else acc + spec
        sab = sfft.irfft2(np.conj(A) * MB, s=(L, L), workers=_FFT_WORKERS) * sfft.irfft2(np.conj(MA) * B, s=(L, L), workers=_FFT_WORKERS)
        prod = sab if prod is None else prod + sab
        prev = fb
        if progress:
            progress((j + 1) / (i1 - i0))
    over = sfft.irfft2(np.conj(MA) * MB, s=(L, L), workers=_FFT_WORKERS)
    good = over >= 0.25 * over.max(axis=(1, 2), keepdims=True)
    safe = np.where(good, over, 1.0)
    cov = np.where(good, (sfft.irfft2(acc, s=(L, L), workers=_FFT_WORKERS) - prod / safe) / safe, 0.0)
    crop = slice(size - size // 2, size - size // 2 + size)
    return np.fft.fftshift(cov, axes=(1, 2))[:, crop, crop]


def piv_window(stack: np.ndarray, i0: int, i1: int, cell: int, params: PIVParams,
               progress: Callable[[float], None] | None = None):
    """Déplacement (dx, dy) en px par intervalle entre frames, par cellule (ny, nx), pour la
    fenêtre de temps [i0, i1), avec le rapport de pics. Cellules non validées : NaN."""
    n, h, w = stack.shape
    ny, nx = -(-h // cell), -(-w // cell)
    Y0, X0 = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    y0, x0 = Y0.ravel(), X0.ravel()
    pad = 2 * cell
    zero = np.zeros_like(y0)
    passes = max(1, int(params.passes))
    sy = sx = zero
    for ps in range(passes):
        cb = (lambda f, ps=ps: progress((ps + f) / passes)) if progress else None
        corr = _ensemble_corr(stack, i0, i1, y0, x0, cell, pad, sy, sx, cb)
        dy, dx, ratio, edge = _peaks(corr)
        dy, dx = dy + sy, dx + sx
        ok = (ratio >= params.min_peak_ratio) & ~edge
        if ps < passes - 1:                       # décalage entier pour la passe suivante
            sy = np.where(ok, np.rint(dy), 0).astype(int)
            sx = np.where(ok, np.rint(dx), 0).astype(int)
    dx = np.where(ok, dx, np.nan).reshape(ny, nx)
    dy = np.where(ok, dy, np.nan).reshape(ny, nx)
    return dx, dy, np.where(np.isfinite(ratio), ratio, 1e3).reshape(ny, nx)


def median_test_outliers(dx: np.ndarray, dy: np.ndarray, thr: float, eps: float = 0.1) -> np.ndarray:
    """Test de médiane normalisée (Westerweel & Scarano 2005) : un vecteur est aberrant si, pour
    une de ses composantes, |écart à la médiane de ses 8 voisins| / (médiane des écarts des
    voisins à cette médiane + eps) > thr. eps = 0.1 px (bruit de mesure). Jugé seulement avec au
    moins 3 voisins mesurés."""
    ny, nx = dx.shape

    def neigh(a):
        p = np.pad(a, 1, constant_values=np.nan)
        return np.stack([p[1 + i:1 + i + ny, 1 + j:1 + j + nx]
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])

    out = np.zeros(dx.shape, dtype=bool)
    nb_n = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in (dx, dy):
            nb = neigh(a)
            nb_n = np.isfinite(nb).sum(axis=0)
            med = np.nanmedian(nb, axis=0)
            spread = np.nanmedian(np.abs(nb - med), axis=0)
            with np.errstate(invalid="ignore"):
                out |= np.abs(a - med) / (spread + eps) > thr
    return out & (nb_n >= 3) & np.isfinite(dx)


# --------------------------------------------------------------------------- fenêtres en parallèle
# Les fenêtres de temps sont indépendantes : une par processus. Le stack est placé dans une
# variable du module AVANT de créer les processus : sous Linux ils sont créés par fork et en
# héritent sans copie (copy-on-write) ; seul l'indice de fenêtre transite entre processus.

_shared: dict | None = None


def _window_task(k: int):
    global _FFT_WORKERS
    _FFT_WORKERS = 1
    s = _shared
    i0, i1 = s["wins"][k]
    return k, piv_window(s["stack"], i0, i1, s["cell"], s["params"])


def _n_procs(n_windows: int, n_cells: int, cell: int) -> int:
    """Processus : bornés par les cœurs (−1), le nombre de fenêtres et la mémoire libre (~8 plans
    de corrélation complexes de (2 cell)² par cellule et par processus)."""
    import os
    from .flow import available_memory
    k = min(max(1, (os.cpu_count() or 2) - 1), n_windows, 32)
    mem = available_memory()
    if mem is not None:
        k = min(k, max(1, int(0.5 * mem / max(1.0, 8 * 8.0 * n_cells * (2 * cell) ** 2))))
    return max(1, k)


def run_windows(stack: np.ndarray, wins: list[tuple[int, int]], cell: int, params: PIVParams,
                progress: Callable[[float, str], None] | None = None) -> list:
    """piv_window pour chaque fenêtre de temps ; en parallèle (processus) quand il y en a
    plusieurs, sinon (ou si le parallélisme échoue) à la suite, avec la progression par paire."""
    global _shared
    p = progress or (lambda *_: None)
    nt = len(wins)
    n, h, w = stack.shape
    procs = _n_procs(nt, (-(-h // cell)) * (-(-w // cell)), cell) if nt > 1 else 1
    out: list = [None] * nt
    if procs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ex = None
        try:
            _shared = {"stack": stack, "wins": wins, "cell": cell, "params": params}
            ex = ProcessPoolExecutor(max_workers=procs)
            futs = [ex.submit(_window_task, k) for k in range(nt)]
            for done, fu in enumerate(as_completed(futs), 1):
                k, r = fu.result()
                out[k] = r
                p(done / nt, f"PIV : {done}/{nt} fenêtres ({procs} processus)")
        except Exception as e:
            if type(e).__name__ in ("StopException", "RerunException"):
                raise                                  # interruption demandée par Streamlit
        finally:
            _shared = None
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
    for k in [k for k in range(nt) if out[k] is None]:
        i0, i1 = wins[k]
        out[k] = piv_window(stack, i0, i1, cell, params,
                            lambda f, k=k: p((k + f) / nt, f"PIV : fenêtre {k + 1}/{nt}"))
    return out
