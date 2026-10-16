"""Champ de vitesse sur une grille régulière, moyenné par fenêtres temporelles.

Conventions :
- positions en px, centre du pixel (i, j) en (x=j, y=i) comme Trackpy ; y vers le bas ;
- vitesses stockées en unité physique (length_unit / time_unit) ;
- vitesse radiale v_r > 0 : la matière s'éloigne du centre du pit le plus proche. Avec plusieurs
  pits voisins, chaque point (cellule de grille ou déplacement individuel) est rattaché au pit le
  plus proche (découpage à la Voronoï) avant de calculer v_r par rapport à ce pit-là.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

from . import flow as flowmod
from . import piv as pivmod
from . import tracking


@dataclass
class Grid:
    height: int
    width: int
    cell: int

    def __post_init__(self):
        if self.cell < 2:
            raise ValueError("La taille de cellule doit être ≥ 2 px.")
        if self.cell > min(self.height, self.width):
            raise ValueError("La taille de cellule dépasse la taille de l'image recadrée.")

    @property
    def nx(self) -> int:
        return -(-self.width // self.cell)

    @property
    def ny(self) -> int:
        return -(-self.height // self.cell)

    def _centers(self, n: int, size: int) -> np.ndarray:
        first = np.arange(n) * self.cell
        last = np.minimum(first + self.cell, size) - 1
        return 0.5 * (first + last)

    @property
    def x(self) -> np.ndarray:
        return self._centers(self.nx, self.width)

    @property
    def y(self) -> np.ndarray:
        return self._centers(self.ny, self.height)

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip(np.floor((np.asarray(x) + 0.5) / self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((np.asarray(y) + 0.5) / self.cell).astype(int), 0, self.ny - 1)
        return ix, iy


def _nearest_pit(x: np.ndarray, y: np.ndarray, pits: list[tuple[float, float]]):
    """Pour chaque point (x, y), renvoie (dx, dy, r, idx) vers le pit le plus proche.

    x, y : tableaux de forme quelconque (grille, échantillons, voire vides). pits : liste non
    vide de (x, y) en px. Avec un seul pit, équivaut exactement à la distance à ce pit.
    """
    pits_arr = np.asarray(pits, dtype=float)              # (P, 2)
    dx_all = np.asarray(x)[..., None] - pits_arr[:, 0]    # (..., P)
    dy_all = np.asarray(y)[..., None] - pits_arr[:, 1]
    r_all = np.hypot(dx_all, dy_all)
    idx = np.argmin(r_all, axis=-1)                       # (...,)
    take = lambda a: np.take_along_axis(a, idx[..., None], axis=-1)[..., 0]
    return take(dx_all), take(dy_all), take(r_all), idx


def _normalized_conv(a: np.ndarray, sigma: float) -> np.ndarray:
    """Moyenne gaussienne pondérée par la présence de données : G*(a·m) / G*(m), m = cellules
    mesurées. Ne fait intervenir que des cellules mesurées (aucune valeur inventée à 0)."""
    m = np.isfinite(a).astype(float)
    num = gaussian_filter(np.where(m > 0, a, 0.0), sigma, mode="constant")
    den = gaussian_filter(m, sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


def gap_mask(measured: np.ndarray, max_gap: int) -> np.ndarray:
    """Cellules non mesurées à au plus `max_gap` cellules d'une cellule mesurée (petits trous)."""
    if max_gap <= 0 or measured.all() or not measured.any():
        return np.zeros_like(measured, dtype=bool)
    dist = distance_transform_edt(~measured)
    return (~measured) & (dist <= max_gap + 1e-9)


def _smooth_fill(a: np.ndarray, smooth: float = 0.0, max_gap: int = 0) -> np.ndarray:
    """Lissage et interpolation sans extrapolation.

    - smooth > 0 : moyenne gaussienne normalisée (écart-type `smooth` cellules) sur les seules
      cellules mesurées ; une cellule mesurée reste mesurée, une cellule vide reste vide ;
    - max_gap > 0 : les trous d'au plus `max_gap` cellules sont interpolés (moyenne pondérée des
      cellules mesurées voisines) ; au-delà, NaN. Ces cellules sont signalées par gap_mask.
    """
    a = np.asarray(a, dtype=float)
    measured = np.isfinite(a)
    out = _normalized_conv(a, smooth) if smooth > 0 else a.copy()
    out[~measured] = np.nan
    holes = gap_mask(measured, max_gap)
    if holes.any():
        out[holes] = _normalized_conv(a, max(1.0, max_gap / 2))[holes]
    return out


def make_windows(n_intervals: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Fenêtres [i0, i1) sur les intervalles entre frames ; la fenêtre couvre les frames i0..i1."""
    if n_intervals < 1:
        raise ValueError("Il faut au moins 2 frames.")
    if window < 1 or stride < 1:
        raise ValueError("Fenêtre et pas doivent être ≥ 1.")
    window = min(window, n_intervals)
    return [(i0, i0 + window) for i0 in range(0, n_intervals - window + 1, stride)]


@dataclass
class FieldResult:
    method: str
    x: np.ndarray                 # (nx,) centres des cellules, px
    y: np.ndarray                 # (ny,)
    windows: list[tuple[int, int]]
    t_start: np.ndarray           # (nt,)
    t_end: np.ndarray
    u: np.ndarray                 # (nt, ny, nx) unité physique
    v: np.ndarray
    count: np.ndarray             # (nt, ny, nx) nb de déplacements (trackpy) ou d'intervalles (flux)
    scale: float                  # px -> unité de longueur
    length_unit: str
    time_unit: str
    pits: list[tuple[float, float]]   # centres des pits (x, y) en px dans l'image recadrée
    cell: int
    image_shape: tuple[int, int]  # (H, W)
    series: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)
    quality: np.ndarray | None = None   # flux optique : qualité de texture par cellule (multiples du bruit) ;
                                        # PIV : rapport pic principal / 2e pic de corrélation
    pit_radii: list = field(default_factory=list)   # rayon estimé du contour de chaque pit (px), pour les figures

    @property
    def t_center(self) -> np.ndarray:
        return 0.5 * (self.t_start + self.t_end)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    @property
    def velocity_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}"

    def radial_velocity(self) -> np.ndarray:
        X, Y = np.meshgrid(self.x, self.y)
        dx, dy, r, _ = _nearest_pit(X, Y, self.pits)
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = (self.u * dx + self.v * dy) / r
        vr[:, r == 0] = np.nan
        return vr

    # ----------------------------------------------------------------- grandeurs scalaires
    # Pour les cartes de couleurs : une grandeur par cellule et par fenêtre. Une cellule est
    # « mesurée » si u et v y sont définis (Trackpy : assez de déplacements ; flux optique : assez
    # de texture). Le lissage n'utilise que les cellules mesurées, l'interpolation éventuelle ne
    # comble que les petits trous (fill = taille max en cellules) et ces cellules sont signalées
    # par interpolated_mask : rien n'est extrapolé dans les zones sans données.

    def scalar_unit(self, name: str) -> str:
        return f"1/{self.time_unit}" if name in ("vorticity", "divergence") else self.velocity_unit

    def measured_mask(self, k: int) -> np.ndarray:
        return np.isfinite(self.u[k]) & np.isfinite(self.v[k])

    def interpolated_mask(self, k: int, fill: int = 0) -> np.ndarray:
        return gap_mask(self.measured_mask(k), int(fill))

    def display_scalar(self, k: int, name: str, fill: int = 0, smooth: float = 0.0) -> np.ndarray:
        """Grandeur `name` de la fenêtre k, (ny, nx), prête à tracer ; NaN là où il n'y a pas de
        donnée.

        name : "speed" |v|, "v_radial" (> 0 : s'éloigne du pit le plus proche), "u" (v_x, > 0 vers
        la droite), "v" (v_y, > 0 vers le bas de l'image), "vorticity" (ω, > 0 : rotation
        anti-horaire à l'écran), "divergence" (∇·v, > 0 : écartement). fill : interpole les trous
        d'au plus `fill` cellules (0 = aucun) ; smooth : lissage gaussien normalisé (cellules).
        """
        u, v = self.u[k], self.v[k]
        if name in ("vorticity", "divergence"):
            u, v = _smooth_fill(u, smooth, fill), _smooth_fill(v, smooth, fill)
            h = self.cell * self.scale                     # pas de la grille, en unité de longueur
            # np.gradient propage les NaN : une dérivée n'existe que si les voisines sont renseignées
            du_dy, du_dx = np.gradient(u, h) if min(u.shape) > 1 else (np.full_like(u, np.nan),) * 2
            dv_dy, dv_dx = np.gradient(v, h) if min(v.shape) > 1 else (np.full_like(v, np.nan),) * 2
            # y vers le bas dans l'image : ω à l'écran (anti-horaire > 0) = ∂u/∂y − ∂v/∂x
            return du_dy - dv_dx if name == "vorticity" else du_dx + dv_dy
        raw = {"speed": lambda: np.hypot(u, v), "u": lambda: u, "v": lambda: v,
               "v_radial": lambda: self.radial_velocity()[k]}[name]()
        return _smooth_fill(raw, smooth, fill)

    def display_vector(self, k: int, smooth: float = 0.0, fill: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) de la fenêtre k lissés (sans extrapolation) ; NaN là où il n'y a pas de donnée."""
        return _smooth_fill(self.u[k], smooth, fill), _smooth_fill(self.v[k], smooth, fill)

    def scalar_limits(self, name: str, fill: int = 0, smooth: float = 0.0,
                      pct: float = 98.0, symmetric: bool = True) -> tuple[float, float]:
        """Échelle de couleur commune à toutes les fenêtres : 0 – p98 pour |v| ; pour les grandeurs
        signées, ± p98(|q|) si symmetric (zéro au milieu de la palette), sinon p2 – p98."""
        vals = np.concatenate([self.display_scalar(k, name, fill, smooth).ravel()
                               for k in range(len(self.windows))])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        if name == "speed":
            hi = float(np.percentile(vals, pct))
            return 0.0, hi if hi > 0 else (float(vals.max()) or 1.0)
        if not symmetric:
            lo, hi = (float(x) for x in np.percentile(vals, [100 - pct, pct]))
            return (lo, hi) if hi > lo else (lo - 1.0, lo + 1.0)
        m = float(np.percentile(np.abs(vals), pct)) or float(np.abs(vals).max()) or 1.0
        return -m, m

    def radial_profile(self, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """v_r moyenne (pondérée par count) par anneau autour du pit le plus proche, par fenêtre.

        Retourne (bords des anneaux en unité physique, profil (nt, n_bins)).
        """
        X, Y = np.meshgrid(self.x, self.y)
        _, _, r_px, _ = _nearest_pit(X, Y, self.pits)
        r = r_px * self.scale
        edges = np.linspace(0, float(r.max()) + 1e-9, n_bins + 1)
        b = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
        vr = self.radial_velocity()
        prof = np.full((len(self.windows), n_bins), np.nan)
        for k in range(len(self.windows)):
            ok = np.isfinite(vr[k]) & (self.count[k] > 0)
            w = np.where(ok, self.count[k], 0).astype(float)
            s = np.bincount(b.ravel(), weights=(np.where(ok, vr[k], 0) * w).ravel(), minlength=n_bins)
            n = np.bincount(b.ravel(), weights=w.ravel(), minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                prof[k] = np.where(n > 0, s / n, np.nan)
        return edges, prof

    def to_long_dataframe(self) -> pd.DataFrame:
        """Une ligne par (fenêtre, cellule) : pratique pour Origin / Excel / pandas."""
        nt = len(self.windows)
        X, Y = np.meshgrid(self.x, self.y)
        vr = self.radial_velocity()
        _, _, _, pit_idx = _nearest_pit(X, Y, self.pits)
        rows = []
        for k in range(nt):
            rows.append(pd.DataFrame({
                "window": k,
                "t_start": self.t_start[k], "t_end": self.t_end[k], "t_center": self.t_center[k],
                "x_px": X.ravel(), "y_px": Y.ravel(),
                f"x_{self.length_unit}": X.ravel() * self.scale,
                f"y_{self.length_unit}": Y.ravel() * self.scale,
                "u": self.u[k].ravel(), "v": self.v[k].ravel(),
                "speed": self.speed[k].ravel(), "v_radial": vr[k].ravel(),
                "nearest_pit": pit_idx.ravel(),
                "count": self.count[k].ravel(),
            }))
        return pd.concat(rows, ignore_index=True)

    def to_npz_dict(self) -> dict:
        return {
            "x_px": self.x, "y_px": self.y, "scale": self.scale,
            "t_start": self.t_start, "t_end": self.t_end, "t_center": self.t_center,
            "windows": np.asarray(self.windows), "u": self.u, "v": self.v, "count": self.count,
            "pits_px": np.asarray(self.pits), "cell_px": self.cell,
            "image_shape": np.asarray(self.image_shape),
            "length_unit": self.length_unit, "time_unit": self.time_unit, "method": self.method,
            "measured": np.isfinite(self.u) & np.isfinite(self.v),
            **({"quality": self.quality} if self.quality is not None else {}),
        }


def advect_tracers(res: FieldResult, seed_step: int = 1, fill_gaps: int = 2,
                   substeps: int = 4) -> pd.DataFrame:
    """Traceurs virtuels advectés dans le champ mesuré (lignes de courant temporelles).

    Permet de réutiliser l'analyse « par traceur » (corrovel.particles) sur un champ dense (flux
    optique), qui ne fournit pas de particules suivies individuellement : on sème des traceurs
    virtuels sur la grille, puis on les déplace fenêtre par fenêtre en intégrant le champ (u, v)
    mesuré (petits trous comblés comme pour l'affichage, voir FieldResult.display_vector). Un
    traceur qui sort de la grille ou entre dans une zone sans donnée s'arrête là — jamais
    d'extrapolation, comme le reste de l'appli. Retourne un DataFrame (particle, frame, x, y) au
    même format que les trajectoires Trackpy (corrovel.tracking.link_tracks), directement
    utilisable par particles.analyze / particles.msd.

    seed_step : un traceur tous les `seed_step` cellules (plus grand = moins de traceurs, plus
    rapide). fill_gaps : comme display_vector, comble les trous d'au plus n cellules avant
    d'advecter. substeps : nombre de pas d'Euler par fenêtre (intégration plus lisse).
    """
    nt = len(res.windows)
    cols = ["particle", "frame", "x", "y"]
    if nt == 0:
        return pd.DataFrame(columns=cols)

    xs0 = res.x[::max(1, seed_step)]
    ys0 = res.y[::max(1, seed_step)]
    X0, Y0 = np.meshgrid(xs0, ys0)
    x = X0.ravel().astype(float)
    y = Y0.ravel().astype(float)
    n = len(x)
    particle_id = np.arange(n)
    alive = np.ones(n, dtype=bool)

    rows = {"particle": list(particle_id), "frame": [res.windows[0][0]] * n,
           "x": list(x), "y": list(y)}

    t_steps = np.concatenate([[res.t_start[0]], res.t_center])
    for k in range(nt):
        if not alive.any():
            break
        u_k, v_k = res.display_vector(k, smooth=0.0, fill=fill_gaps)
        interp_u = RegularGridInterpolator((res.y, res.x), u_k, bounds_error=False, fill_value=np.nan)
        interp_v = RegularGridInterpolator((res.y, res.x), v_k, bounds_error=False, fill_value=np.nan)
        dt = (t_steps[k + 1] - t_steps[k]) / max(1, substeps)
        for _ in range(max(1, substeps)):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            pts = np.column_stack([y[idx], x[idx]])
            du, dv = interp_u(pts), interp_v(pts)
            ok = np.isfinite(du) & np.isfinite(dv)
            alive[idx[~ok]] = False
            move = idx[ok]
            x[move] += du[ok] * dt
            y[move] += dv[ok] * dt
        live = np.flatnonzero(alive)
        rows["particle"].extend(particle_id[live].tolist())
        rows["frame"].extend([res.windows[k][1]] * len(live))
        rows["x"].extend(x[live].tolist())
        rows["y"].extend(y[live].tolist())

    df = pd.DataFrame(rows)
    return df.sort_values(["particle", "frame"]).reset_index(drop=True)


def _series_row(k, w, times, speed, vr, n, n_particles=None) -> dict:
    row = {"window": k, "frame_start": w[0], "frame_end": w[1],
           "t_start": times[w[0]], "t_end": times[w[1]],
           "t_center": 0.5 * (times[w[0]] + times[w[1]]),
           "mean_speed": float(np.nanmean(speed)) if len(speed) else np.nan,
           "median_speed": float(np.nanmedian(speed)) if len(speed) else np.nan,
           "mean_v_radial": float(np.nanmean(vr)) if np.isfinite(vr).any() else np.nan,
           "n": int(n)}
    if n_particles is not None:
        row["n_particles"] = int(n_particles)
    return row


def field_from_samples(samples: pd.DataFrame, grid: Grid, windows: list[tuple[int, int]],
                       times: np.ndarray, scale: float, pits: list[tuple[float, float]],
                       min_count: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    nt = len(windows)
    u = np.full((nt, grid.ny, grid.nx), np.nan)
    v = np.full_like(u, np.nan)
    count = np.zeros((nt, grid.ny, grid.nx), dtype=int)
    rows = []
    f0 = samples["frame0"].to_numpy() if len(samples) else np.array([], int)
    for k, (i0, i1) in enumerate(windows):
        s = samples[(f0 >= i0) & (f0 < i1)]
        su = np.zeros((grid.ny, grid.nx))
        sv = np.zeros_like(su)
        n = np.zeros((grid.ny, grid.nx), dtype=int)
        if len(s):
            ix, iy = grid.index(s["x"].to_numpy(), s["y"].to_numpy())
            np.add.at(su, (iy, ix), s["u"].to_numpy() * scale)
            np.add.at(sv, (iy, ix), s["v"].to_numpy() * scale)
            np.add.at(n, (iy, ix), 1)
        good = n >= max(1, min_count)
        u[k][good] = su[good] / n[good]
        v[k][good] = sv[good] / n[good]
        count[k] = n
        # Série temporelle calculée sur les déplacements individuels (pas sur la grille)
        uu, vv = s["u"].to_numpy() * scale, s["v"].to_numpy() * scale
        dx, dy, r, _ = _nearest_pit(s["x"].to_numpy(), s["y"].to_numpy(), pits)
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = np.where(r > 0, (uu * dx + vv * dy) / r, np.nan)
        rows.append(_series_row(k, (i0, i1), times, np.hypot(uu, vv), vr, len(s),
                                s["particle"].nunique() if len(s) else 0))
    return u, v, count, pd.DataFrame(rows)


ProgressFn = Callable[[float, str], None]


def run_trackpy(stack: np.ndarray, times: np.ndarray, params: tracking.TrackpyParams, cell: int,
                window: int, stride: int, min_count: int, pits: list[tuple[float, float]], scale: float,
                length_unit: str, time_unit: str, progress: ProgressFn | None = None):
    """Détection + liaison Trackpy puis champ de vitesse. Retourne (FieldResult, tracks, samples)."""
    n, h, w = stack.shape
    grid = Grid(h, w, cell)
    wins = make_windows(n - 1, window, stride)
    p = progress or (lambda *_: None)
    p(0.0, "Préparation des images de détection")
    feat = tracking.feature_stack(stack, params.polarity, params.background)
    feats = tracking.locate_all(feat, params, progress=lambda f, t: p(0.05 + 0.75 * f, t))
    p(0.82, f"Liaison de {len(feats)} détections")
    tracks = tracking.link_tracks(feats, params)
    samples = tracking.displacements(tracks, times)
    p(0.92, "Calcul du champ sur la grille")
    u, v, count, series = field_from_samples(samples, grid, wins, times, scale, pits, min_count)
    res = FieldResult("trackpy", grid.x, grid.y, wins,
                      np.array([times[a] for a, _ in wins]), np.array([times[b] for _, b in wins]),
                      u, v, count, scale, length_unit, time_unit, list(pits), cell, (h, w), series,
                      {"method": "trackpy", "trackpy": params.to_dict(), "cell_px": cell,
                       "window_intervals": window, "stride": stride, "min_count": min_count,
                       "pits_px": [list(p_) for p_ in pits], "n_detections": int(len(feats)),
                       "n_tracks": int(tracks["particle"].nunique()) if len(tracks) else 0,
                       "n_displacements": int(len(samples))})
    p(1.0, "Terminé")
    return res, tracks, samples


def _fmt_duration(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def compute_pair_flows(stack: np.ndarray, params: flowmod.FlowParams, base: int,
                       cache_dir=None, progress: ProgressFn | None = None, workers: int | None = None,
                       tag_extra: str = "", prepare=None):
    """Flux de chaque paire de frames consécutives, moyenné sur des blocs de `base` px.

    - cache_dir : dossier où chaque paire est enregistrée dès qu'elle est calculée ; au calcul
      suivant (autre cellule, autre fenêtre, autre pit…) ou après une interruption, les paires
      déjà présentes sont relues au lieu d'être recalculées ;
    - les paires restantes sont réparties sur plusieurs processus (cœurs du processeur), en
      nombre limité par la mémoire libre ; en cas d'échec du parallélisme, calcul séquentiel.
    Retourne (bx, by, pix, bq, info) : listes par paire des moyennes de blocs de dx, dy (px,
    pondérées par la qualité de texture), norme moyenne pondérée du déplacement (px), qualité de
    texture moyenne par bloc, et un dict d'information (paires en cache, processus utilisés).
    """
    import time
    from pathlib import Path
    n, h, w = stack.shape
    m = n - 1
    p = progress or (lambda *_: None)
    out: dict[int, tuple] = {}
    cdir = None
    if cache_dir is not None:
        tag = f"{params.algorithm}_r{params.radius}_a{params.attachment:g}_b{base}{tag_extra}_v3"   # v3 : + texture
        cdir = Path(cache_dir) / f"flow_{tag}"
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(m):
            f = cdir / f"pair_{i:05d}.npz"
            if f.exists():
                try:
                    d = np.load(f)
                    out[i] = (d["bx"].astype(np.float32), d["by"].astype(np.float32), float(d["pix"]),
                              d["bq"].astype(np.float32))
                except Exception:
                    f.unlink(missing_ok=True)          # fichier abîmé (calcul interrompu) : à refaire
    cached = len(out)
    todo = [i for i in range(m) if i not in out]
    t0 = time.time()

    def done(i, r):
        out[i] = r
        if cdir is not None:
            tmp = cdir / f"pair_{i:05d}.tmp.npz"
            np.savez(tmp, bx=r[0].astype(np.float16), by=r[1].astype(np.float16), pix=r[2],
                     bq=r[3].astype(np.float32))
            tmp.replace(cdir / f"pair_{i:05d}.npz")   # écriture atomique : jamais de fichier à moitié écrit
        k = len(out) - cached
        eta = (time.time() - t0) / k * (len(todo) - k)
        p(len(out) / m, f"Flux optique {len(out)}/{m}" + (f" ({cached} en cache)" if cached else "")
          + (f" — reste ≈ {_fmt_duration(eta)}" if k < len(todo) else ""))

    p(cached / m, f"Flux optique : {cached}/{m} paires en cache, {len(todo)} à calculer…")
    if todo and prepare is not None:          # prétraitement (coûteux) seulement s'il reste des paires
        stack = prepare(stack)
    # Farneback (OpenCV) utilise déjà tous les cœurs : pas de processus supplémentaires
    nw = workers if workers is not None else (1 if params.algorithm == "farneback"
                                              else flowmod.n_workers(len(todo), h * w))
    used = 1
    if todo and nw > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ex = None
        try:
            ex = ProcessPoolExecutor(max_workers=nw)
            futs = {ex.submit(flowmod.pair_task, stack[i], stack[i + 1], params, base): i for i in todo}
            used = nw
            for fu in as_completed(futs):
                done(futs[fu], fu.result())
        except Exception as e:                        # processus indisponibles : on continue en séquentiel
            if type(e).__name__ in ("StopException", "RerunException"):
                raise                                 # interruption demandée par Streamlit
            used = 1
        finally:
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
    for i in [i for i in todo if i not in out]:
        done(i, flowmod.pair_task(stack[i], stack[i + 1], params, base))
    info = {"pairs_cached": cached, "pairs_computed": len(todo), "workers": used,
            "seconds": round(time.time() - t0, 1)}
    return ([out[i][0] for i in range(m)], [out[i][1] for i in range(m)],
            np.array([out[i][2] for i in range(m)]), [out[i][3] for i in range(m)], info)


def run_flow(stack: np.ndarray, times: np.ndarray, params: flowmod.FlowParams, cell: int,
             window: int, stride: int, pits: list[tuple[float, float]], scale: float,
             length_unit: str, time_unit: str, progress: ProgressFn | None = None,
             cache_dir=None, workers: int | None = None, min_texture: float | None = None,
             subtract_background: bool = False) -> FieldResult:
    """Flux optique entre frames consécutives puis moyenne par cellule et par fenêtre.

    - Le flux de chaque paire est calculé une seule fois (blocs de 2 px, ou 1 px si la cellule
      est impaire), en parallèle, et mis en cache dans `cache_dir` si fourni.
    - Moyenne par cellule PONDÉRÉE par la qualité de texture q (voir flow.texture_quality) : la
      vitesse d'une cellule est celle de ce qui est texturé (particules, bords), pas une moyenne
      diluée par le fond uniforme où le flux n'est que du bruit.
    - Cellules non mesurées (u = v = NaN) si leur texture moyenne q ne dépasse pas le seuil :
      min_texture fixé, ou par défaut (None) le seuil statistique flow.noise_texture_threshold
      (texture indiscernable du bruit à 99 %, pour cette taille de cellule et ce nombre de paires).
      La qualité par cellule est gardée dans FieldResult.quality.
    - Vitesse moyenne et radiale sur la ROI : moyennes pondérées par q (pixels texturés).
    - subtract_background : flux calculé sur |frame − fond fixe| (flow.moving_content), pour ne
      mesurer que ce qui bouge ; indispensable sur des images de traceurs avec texture fixe.
    """
    n, h, w = stack.shape
    grid = Grid(h, w, cell)
    wins = make_windows(n - 1, window, stride)
    for i in range(n - 1):
        if times[i + 1] - times[i] <= 0:
            raise ValueError(f"Temps non croissants entre les frames {i} et {i + 1}.")
    base = 2 if cell % 2 == 0 else 1
    bxs, bys, pix, bqs, info = compute_pair_flows(stack, params, base, cache_dir, progress, workers,
                                                  "_bgsub" if subtract_background else "",
                                                  flowmod.moving_content if subtract_background else None)
    wgt = flowmod.block_weights(h, w, base)
    xb = (np.arange(wgt.shape[1]) * base + 0.5 * (np.minimum(base, w - np.arange(wgt.shape[1]) * base) - 1))
    yb = (np.arange(wgt.shape[0]) * base + 0.5 * (np.minimum(base, h - np.arange(wgt.shape[0]) * base) - 1))
    Xb, Yb = np.meshgrid(xb, yb)
    dxb, dyb, rb, _ = _nearest_pit(Xb, Yb, pits)
    rb_safe = np.where(rb > 0, rb, np.nan)
    ui = np.empty((n - 1, grid.ny, grid.nx))
    vi = np.empty_like(ui)
    qi = np.empty_like(ui)
    pix_speed = np.empty(n - 1)
    pix_vr = np.empty(n - 1)
    k = cell // base
    for i in range(n - 1):
        f = scale / (times[i + 1] - times[i])
        bq = np.nan_to_num(bqs[i].astype(np.float64), nan=0.0)
        ui[i] = flowmod.regroup(bxs[i], wgt * bq, k) * f            # moyenne pondérée par la texture
        vi[i] = flowmod.regroup(bys[i], wgt * bq, k) * f
        qi[i] = flowmod.regroup(bq, wgt, k)                         # qualité moyenne de la cellule
        pix_speed[i] = pix[i] * f
        vr = (bxs[i] * dxb + bys[i] * dyb) / rb_safe
        ok = np.isfinite(vr) & (bq > 0)
        pix_vr[i] = float(np.sum(vr[ok] * (wgt * bq)[ok]) / np.sum((wgt * bq)[ok])) * f if ok.any() else np.nan
    nt = len(wins)
    u = np.empty((nt, grid.ny, grid.nx))
    v = np.empty_like(u)
    quality = np.empty_like(u)
    count = np.empty((nt, grid.ny, grid.nx), dtype=int)
    rows, thresholds = [], []
    for k, (i0, i1) in enumerate(wins):
        with np.errstate(invalid="ignore"):
            u[k] = np.nanmean(ui[i0:i1], axis=0)
            v[k] = np.nanmean(vi[i0:i1], axis=0)
            quality[k] = np.nanmean(qi[i0:i1], axis=0)
        thr = min_texture if min_texture is not None else flowmod.noise_texture_threshold(
            max(1.0, params.radius / 2), cell, i1 - i0)
        thresholds.append(thr)
        low = ~(quality[k] > thr)
        u[k][low] = np.nan
        v[k][low] = np.nan
        count[k] = i1 - i0
        row = _series_row(k, (i0, i1), times, pix_speed[i0:i1], pix_vr[i0:i1], (i1 - i0) * h * w)
        # Pour le flux : moyennes (pondérées par la texture) des moyennes par intervalle
        row["median_speed"] = float(np.median(pix_speed[i0:i1]))
        row["measured_cells"] = float(np.mean(~low))
        rows.append(row)
    res = FieldResult("optical_flow", grid.x, grid.y, wins,
                      np.array([times[a] for a, _ in wins]), np.array([times[b] for _, b in wins]),
                      u, v, count, scale, length_unit, time_unit, list(pits), cell, (h, w), pd.DataFrame(rows),
                      {"method": "optical_flow", "flow": params.to_dict(), "cell_px": cell,
                       "window_intervals": window, "stride": stride,
                       "texture_threshold": "statistique (99.9 % au-dessus du bruit blanc)" if min_texture is None
                       else min_texture, "texture_threshold_values": sorted(set(round(x, 4) for x in thresholds)),
                       "cell_average": "pondérée par la qualité de texture (Shi-Tomasi / bruit)",
                       "subtract_background": bool(subtract_background),
                       "pits_px": [list(p_) for p_ in pits], "compute": info},
                      quality=quality)
    return res


def run_piv(stack: np.ndarray, times: np.ndarray, params: pivmod.PIVParams, cell: int, window: int,
            stride: int, pits: list[tuple[float, float]], scale: float, length_unit: str, time_unit: str,
            progress: ProgressFn | None = None, subtract_background: bool = False) -> FieldResult:
    """PIV par intercorrélation d'ensemble : une fenêtre d'interrogation par cellule, un champ par
    fenêtre de temps (voir corrovel.piv).

    - subtract_background : corrélation calculée sur |frame − fond fixe| (flow.moving_content).
      Indispensable sur des traceurs : un motif fixe (rayures, bords du pit, particules collées)
      donne un pic de corrélation au déplacement NUL qui masque ou tire vers zéro le vrai pic ;
    - vitesse = déplacement par intervalle × échelle / intervalle de temps moyen de la fenêtre
      (la corrélation d'ensemble suppose des intervalles égaux : signalé dans params sinon) ;
    - série temporelle : moyennes sur les cellules mesurées de chaque fenêtre.
    """
    import time as _time
    n, h, w = stack.shape
    grid = Grid(h, w, cell)
    wins = make_windows(n - 1, window, stride)
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Temps non croissants entre frames.")
    p = progress or (lambda *_: None)
    t0 = _time.time()
    if subtract_background:
        p(0.0, "Soustraction du fond fixe…")
        stack = flowmod.moving_content(stack)
    X, Y = np.meshgrid(grid.x, grid.y)
    dxp, dyp, rp, _ = _nearest_pit(X, Y, pits)
    rp_safe = np.where(rp > 0, rp, np.nan)
    nt = len(wins)
    u = np.full((nt, grid.ny, grid.nx), np.nan)
    v = np.full_like(u, np.nan)
    quality = np.full_like(u, np.nan)
    count = np.zeros((nt, grid.ny, grid.nx), dtype=int)
    rows, uneven, n_low, n_out = [], [], 0, 0
    per_window = pivmod.run_windows(stack, wins, cell, params, p)
    for k, (i0, i1) in enumerate(wins):
        dx, dy, ratio = per_window[k]
        n_low += int(np.sum(~np.isfinite(dx)))
        if params.median_test > 0:
            out = pivmod.median_test_outliers(dx, dy, params.median_test)
            n_out += int(out.sum())
            dx[out] = dy[out] = np.nan
        dts = np.diff(times[i0:i1 + 1])
        if dts.std() > 0.01 * dts.mean():
            uneven.append(k)
        f = scale / dts.mean()
        u[k], v[k], quality[k] = dx * f, dy * f, ratio
        count[k] = i1 - i0
        m = np.isfinite(u[k])
        sp = np.hypot(u[k], v[k])[m]
        vr = ((u[k] * dxp + v[k] * dyp) / rp_safe)[m]
        row = _series_row(k, (i0, i1), times, sp, vr, int(m.sum()))
        row["measured_cells"] = float(m.mean())
        rows.append(row)
    return FieldResult("piv", grid.x, grid.y, wins,
                       np.array([times[a] for a, _ in wins]), np.array([times[b] for _, b in wins]),
                       u, v, count, scale, length_unit, time_unit, list(pits), cell, (h, w), pd.DataFrame(rows),
                       {"method": "piv", "piv": params.to_dict(), "cell_px": cell,
                        "interrogation_window_px": cell, "correlation": "ensemble (somme des plans de la fenêtre)",
                        "window_intervals": window, "stride": stride,
                        "subtract_background": bool(subtract_background),
                        "pits_px": [list(p_) for p_ in pits],
                        "piv_info": {"seconds": round(_time.time() - t0, 1),
                                     "rejected_low_peak_ratio": n_low, "rejected_median_test": n_out,
                                     "windows_with_uneven_dt": uneven}},
                       quality=quality)
