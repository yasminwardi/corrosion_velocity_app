"""Champ de déplacement dense par flux optique (scikit-image), alternative à Trackpy
quand il n'y a pas de particules discrètes bien séparées."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from skimage.registration import optical_flow_ilk, optical_flow_tvl1


@dataclass
class FlowParams:
    algorithm: str = "ilk"   # "ilk" (Lucas-Kanade itératif) | "tvl1" (TV-L1) | "farneback" (OpenCV)
    radius: int = 7          # rayon de la fenêtre (ilk ; farneback : fenêtre 2 r + 1), en px
    attachment: float = 15.0  # tvl1 : plus grand = plus d'attache aux données, moins lissé

    def to_dict(self) -> dict:
        return asdict(self)


def has_opencv() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _normalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Même mise à l'échelle [0, 1] pour les deux images (robuste aux pixels aberrants)."""
    both = np.concatenate([a.ravel(), b.ravel()])
    lo, hi = np.percentile(both, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32),
            np.clip((b - lo) / (hi - lo), 0, 1).astype(np.float32))


def pair_flow(ref: np.ndarray, mov: np.ndarray, p: FlowParams) -> tuple[np.ndarray, np.ndarray]:
    """Déplacement (dx, dy) en px de chaque pixel entre ref (t) et mov (t + dt).

    Convention scikit-image : mov(r + flow(r)) ≈ ref(r), donc flow = déplacement de ref vers mov.
    """
    a, b = _normalize_pair(ref.astype(np.float32), mov.astype(np.float32))
    if p.algorithm == "ilk":
        fy, fx = optical_flow_ilk(a, b, radius=int(p.radius), prefilter=True)
    elif p.algorithm == "tvl1":
        fy, fx = optical_flow_tvl1(a, b, attachment=float(p.attachment), prefilter=True)
    elif p.algorithm == "farneback":
        # Farneback (OpenCV, C++ multithreadé) : même convention (déplacement de ref vers mov),
        # pyramide de 4 niveaux, fenêtre gaussienne 2 r + 1, polynômes de degré 7 / σ 1.5
        try:
            import cv2
        except ImportError as e:
            raise ImportError("L'algorithme Farneback demande OpenCV : pip install opencv-python-headless") from e
        f = cv2.calcOpticalFlowFarneback(a * 255, b * 255, None, 0.5, 4, 2 * int(p.radius) + 1, 5, 7, 1.5,
                                         cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        fx, fy = f[..., 0], f[..., 1]
    else:
        raise ValueError("algorithm doit être 'ilk', 'tvl1' ou 'farneback'.")
    return fx, fy


def moving_content(stack: np.ndarray, max_bg_frames: int = 200) -> np.ndarray:
    """|frame − fond fixe| : ne garde que ce qui change dans le temps (traceurs en mouvement).

    Chaque frame est d'abord ramenée à la même luminosité moyenne (photoblanchiment), puis on
    retire la médiane temporelle. Sans cela, la texture fixe des images (rayures de surface, bords
    du pit, particules collées) ne bouge pas et tire le flux optique vers zéro : sur NiCr(+), le
    flux sous-estimait d'un facteur 4–5 la vitesse des traceurs mesurée par Trackpy ; après
    soustraction, l'écart tombe à ~15 % (corrélation 0.9). Conséquence : les zones où rien ne bouge
    deviennent « sans texture », donc non mesurées (et non mesurées à vitesse nulle).
    """
    f = stack.astype(np.float32)
    n = len(f)
    means = f.reshape(n, -1).mean(axis=1)
    f *= (float(means.mean()) / np.where(means > 0, means, 1.0))[:, None, None]
    sub = f if n <= max_bg_frames else f[np.linspace(0, n - 1, max_bg_frames).astype(int)]
    bg = np.median(sub, axis=0)
    return np.abs(f - bg[None])


def block_mean(a: np.ndarray, cell: int) -> np.ndarray:
    """Moyenne par blocs cell x cell ; la dernière ligne/colonne peut être partielle."""
    h, w = a.shape
    ny, nx = -(-h // cell), -(-w // cell)
    pad = np.full((ny * cell, nx * cell), np.nan, dtype=np.float64)
    pad[:h, :w] = a
    with np.errstate(invalid="ignore"):
        return np.nanmean(pad.reshape(ny, cell, nx, cell), axis=(1, 3))


# --------------------------------------------------------------------------- calcul par paires
# Le flux d'une paire de frames ne dépend ni de la taille de cellule, ni des fenêtres, ni du pit :
# on le calcule une fois, on le garde moyenné sur de petits blocs de `base` px (exactement
# regroupables en n'importe quelle cellule multiple de `base`), et on peut le mettre en cache sur
# disque. Les paires sont indépendantes : elles se calculent en parallèle dans des processus.

def block_weights(h: int, w: int, base: int) -> np.ndarray:
    """Nombre de pixels de chaque bloc base × base (les blocs du bord droit / bas sont partiels)."""
    cy = np.minimum(base, h - np.arange(-(-h // base)) * base)
    cx = np.minimum(base, w - np.arange(-(-w // base)) * base)
    return np.outer(cy, cx).astype(np.float64)


def regroup(bm: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Moyennes de blocs de base -> moyennes de cellules de k × k blocs (pondérées par le nombre
    de pixels : identique à block_mean appliqué directement à l'image)."""
    if k == 1:
        return bm.astype(np.float64)
    ny, nx = bm.shape
    NY, NX = -(-ny // k), -(-nx // k)
    s = np.zeros((NY * k, NX * k))
    c = np.zeros_like(s)
    ok = np.isfinite(bm)
    s[:ny, :nx] = np.where(ok, bm, 0) * weights
    c[:ny, :nx] = np.where(ok, weights, 0)
    s = s.reshape(NY, k, NX, k).sum(axis=(1, 3))
    c = c.reshape(NY, k, NX, k).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def noise_sigma(img: np.ndarray) -> float:
    """Écart-type du bruit d'une image (méthode d'Immerkær 1996 : filtre laplacien qui annule les
    structures lisses ; robuste et rapide)."""
    from scipy.ndimage import convolve
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    h, w = img.shape
    s = np.abs(convolve(img.astype(np.float64), k, mode="reflect"))[1:-1, 1:-1].sum()
    return float(np.sqrt(np.pi / 2) * s / (6.0 * max(1, (w - 2) * (h - 2))))


def texture_quality(img: np.ndarray, sigma: float) -> np.ndarray:
    """Fiabilité locale du flux optique : énergie de texture AU-DELÀ du bruit, en multiples de
    l'énergie du bruit.

    λ_min = plus petite valeur propre du tenseur de structure moyenné sur la fenêtre (critère de
    Shi-Tomasi / Lucas-Kanade). Pour du bruit pur d'écart-type σ, l'énergie de gradient moyenne
    vaut σ²/2 (gradients centrés) : q = λ_min / (σ²/2) − 1 (borné à 0). q ≈ 0 : bruit seul, le
    flux n'y est pas mesurable (fond uniforme, problème d'ouverture) ; q ≳ 0.5 : texture
    exploitable (l'erreur de Lucas-Kanade décroît comme 1/√(q × taille de fenêtre)).
    """
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(img.astype(np.float64))
    ixx, iyy, ixy = (gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
    lam = 0.5 * (ixx + iyy) - np.sqrt(0.25 * (ixx - iyy) ** 2 + ixy ** 2)
    sn = noise_sigma(img)
    return np.clip(lam / max(0.5 * sn ** 2, 1e-12) - 1.0, 0, None)


@lru_cache(maxsize=64)
def noise_texture_threshold(sigma: float, cell: int, n_avg: int, pct: float = 99.9) -> float:
    """Seuil de texture « au-dessus du bruit » pour une cellule : 99.9e percentile de la qualité q
    moyennée sur une cellule (et sur n_avg paires) pour du BRUIT BLANC PUR. Comme q est rapporté
    au niveau de bruit, cette distribution ne dépend que de la fenêtre (sigma), de la taille de
    cellule et du nombre de paires moyennées : on la simule une fois (≥ 6000 cellules simulées).
    Une cellule dont la texture ne dépasse pas ce seuil est indiscernable du bruit : le flux n'y
    est pas une mesure. Taux de fausses alarmes attendu : 0.1 % des cellules sans texture.
    Hypothèse : bruit de caméra peu corrélé spatialement (à vérifier sur les images réelles)."""
    rng = np.random.default_rng(12345)
    size = int(max(512, 40 * cell))
    n = int(max(1, min(n_avg, 8)))                  # au-delà de 8 paires : seuil prudent (surestimé)
    vals = []
    for _ in range(max(1, -(-6000 // (-(-size // cell)) ** 2))):
        acc = np.zeros((-(-size // cell), -(-size // cell)))
        for _ in range(n):
            acc += block_mean(texture_quality(rng.standard_normal((size, size)), sigma), cell)
        vals.append((acc / n).ravel())
    return float(np.percentile(np.concatenate(vals), pct))


def pair_task(ref: np.ndarray, mov: np.ndarray, p: FlowParams, base: int):
    """Travail d'une paire (exécutable dans un autre processus).

    Retourne, par blocs de `base` px : le déplacement moyen (dx, dy) PONDÉRÉ par la qualité de
    texture q (les pixels de fond uniforme, où le flux n'est que du bruit, ne diluent plus la
    moyenne), la qualité moyenne q, et la norme moyenne pondérée du déplacement sur l'image.
    """
    fx, fy = pair_flow(ref, mov, p)
    a, _ = _normalize_pair(ref.astype(np.float32), mov.astype(np.float32))
    q = texture_quality(a, max(1.0, p.radius / 2))
    bq = block_mean(q, base)
    with np.errstate(invalid="ignore", divide="ignore"):
        bx = np.where(bq > 0, block_mean(q * fx, base) / bq, np.nan)
        by = np.where(bq > 0, block_mean(q * fy, base) / bq, np.nan)
    pix = float(np.sum(q * np.hypot(fx, fy)) / max(np.sum(q), 1e-12))
    return bx.astype(np.float32), by.astype(np.float32), pix, bq.astype(np.float32)


def available_memory() -> float | None:
    """Mémoire vive disponible (octets), sans dépendance obligatoire ; None si inconnue."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return float(ms.ullAvailPhys)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def n_workers(n_tasks: int, n_pixels: int, max_workers: int = 8) -> int:
    """Nombre de processus : borné par les cœurs (-1), le nombre de paires et la mémoire libre
    (~130 octets par pixel et par processus pour Lucas-Kanade, mesuré sur 1700 × 1700 px)."""
    import os
    cpu = max(1, (os.cpu_count() or 2) - 1)
    k = min(cpu, n_tasks, max_workers)
    mem = available_memory()
    if mem is not None:
        k = min(k, max(1, int(0.6 * mem / max(1.0, 130.0 * n_pixels))))
    return max(1, k)
