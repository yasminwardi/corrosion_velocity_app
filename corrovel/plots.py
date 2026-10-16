"""Figures matplotlib, prévues pour être exportées telles quelles (PNG/SVG/PDF).

Style commun (corrovel.style) : textes en anglais, étiquettes en gras « Grandeur / (unité) »,
images encadrées sans graduations quand une barre d'échelle est tracée, contour du pit en cercle
rouge pointillé.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

from dataclasses import dataclass  # noqa: E402

from . import colormaps  # noqa: E402,F401  (enregistre la palette « parula »)
from . import density as densitymod  # noqa: E402
from . import style  # noqa: E402  (applique le style commun des figures)
from .density import DensityResult  # noqa: E402
from .field import FieldResult  # noqa: E402
from .scalebar import ScaleBar, add_to_axes  # noqa: E402


def _extent(shape: tuple[int, int], scale: float) -> list[float]:
    h, w = shape
    return [-0.5 * scale, (w - 0.5) * scale, (h - 0.5) * scale, -0.5 * scale]


def _finish_image_axes(ax, scale: float, shape: tuple[int, int], scalebar: ScaleBar | None,
                       length_unit: str = "px", background: np.ndarray | None = None, color: str | None = None):
    """Panneau image : barre d'échelle si fournie (et alors sans graduations), sinon axes x, y."""
    if scalebar:
        add_to_axes(ax, scalebar, scale, shape[1], background=background, color=color)
    style.image_panel(ax, show_axes=not scalebar, length_unit=length_unit)


def _draw_pits(ax, pits: list[tuple[float, float]], scale: float, radii=None, y_of=None):
    style.draw_pits(ax, pits, scale, radii, y_of)


def _pit_radii(obj) -> list:
    return list(getattr(obj, "pit_radii", None) or [])


# Grandeurs des cartes de couleurs : libellés de l'interface (français) et des figures (anglais)
SCALAR_LABELS = {"speed": "|v|", "v_radial": "v_r (> 0 : s'éloigne du pit)", "u": "v_x",
                 "v": "v_y (> 0 vers le bas)", "vorticity": "vorticité ω (> 0 : anti-horaire)",
                 "divergence": "divergence ∇·v"}
FIG_SCALAR_NAMES = {"speed": "Speed", "v_radial": "Radial velocity", "u": "Velocity, x",
                    "v": "Velocity, y (down)", "vorticity": "Vorticity", "divergence": "Divergence"}


def scalar_fig_unit(res, quantity: str) -> str:
    return style.rate_unit(res.time_unit) if quantity in ("vorticity", "divergence") \
        else style.vel_unit(res.length_unit, res.time_unit)


@dataclass(frozen=True)
class ContourStyle:
    """Réglages du mode « contours » : carte de couleurs d'une grandeur + flèches noires."""
    quantity: str = "speed"
    levels: int = 12
    cmap: str = "parula"
    smooth: float = 0.0                       # lissage gaussien normalisé (cellules mesurées seulement)
    fill_gaps: int = 0                        # interpoler les trous d'au plus n cellules ; 0 = aucun
    arrow_step: int = 1                       # 1 flèche sur n (dans chaque direction) : affichage seulement
    limits: tuple[float, float] | None = None  # échelle de couleur ; None = calculée sur toutes les fenêtres
    symmetric: bool = True                    # grandeurs signées : échelle ± symétrique (sinon p2 – p98)
    smooth_arrows: bool = False               # flèches du champ lissé (mêmes cellules que la carte)
    y_up: bool = False                        # origine en bas à gauche, y vers le haut (style graphique)
    clean: bool = False                       # style article : sans titre ni repère du pit
    upsample: int = 4                         # interpolation (cubique) de la carte : contours lisses

    def color_limits(self, res: FieldResult) -> tuple[float, float]:
        if self.limits:
            return self.limits
        lo, hi = res.scalar_limits(self.quantity, self.fill_gaps, self.smooth, symmetric=self.symmetric)
        return (-hi, -lo) if self._flip_sign() else (lo, hi)

    def _flip_sign(self) -> bool:
        # y vers le haut : la composante verticale change de signe (positive vers le haut)
        return self.y_up and self.quantity == "v"

    def scalar(self, res: FieldResult, k: int) -> np.ndarray:
        z = res.display_scalar(k, self.quantity, self.fill_gaps, self.smooth)
        return -z if self._flip_sign() else z

    def vectors(self, res: FieldResult, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Flèches à tracer, en coordonnées d'affichage (v > 0 vers le haut si y_up)."""
        u, v = res.display_vector(k, self.smooth, self.fill_gaps) if self.smooth_arrows else (res.u[k], res.v[k])
        return u, (-v if self.y_up else v)

    def interpolated(self, res: FieldResult, k: int) -> np.ndarray:
        return res.interpolated_mask(k, self.fill_gaps)

    def y_display(self, y_px, height: int, scale: float):
        """Ordonnée d'affichage : y vers le bas (image) ou vers le haut à partir du bas de l'image."""
        y = np.asarray(y_px, dtype=float)
        return ((height - 0.5) - y) * scale if self.y_up else y * scale

    def label(self, res: FieldResult) -> str:
        """Libellé pour l'interface (français)."""
        lab = SCALAR_LABELS[self.quantity]
        if self._flip_sign():
            lab = "v_y (> 0 vers le haut)"
        if self.clean:
            lab = lab.split(" (")[0]
        return f"{lab} ({res.scalar_unit(self.quantity)})"

    def fig_label(self, res: FieldResult) -> str:
        """Libellé des figures (anglais, « Grandeur / (unité) »)."""
        name = "Velocity, y (up)" if self._flip_sign() else FIG_SCALAR_NAMES[self.quantity]
        return style.label(name, scalar_fig_unit(res, self.quantity))

    def level_values(self, res: FieldResult) -> np.ndarray:
        """Niveaux ronds (…, -0.5, 0, 0.5, …) couvrant l'échelle de couleur, ~`levels` intervalles."""
        lo, hi = self.color_limits(res)
        sym = lo < 0 < hi and abs(lo + hi) <= 1e-9 * max(abs(lo), abs(hi))   # échelle ± symétrique
        vals = MaxNLocator(nbins=max(2, self.levels), symmetric=sym).tick_values(lo, hi)
        vals = vals if len(vals) >= 3 else np.linspace(lo, hi, max(2, self.levels) + 1)
        return np.round(vals, 12)          # -3.6000000000000005 -> -3.6


def upsample_field(x: np.ndarray, y: np.ndarray, z: np.ndarray, factor: int,
                   extent: tuple[float, float, float, float] | None = None):
    """Carte prête à tracer en contours lisses, sans valeur inventée.

    - extent = (x0, x1, y0, y1) : bords de la ROI (px). La valeur de chaque cellule couvre la
      cellule elle-même : on prolonge la grille jusqu'aux bords de la ROI (demi-cellule), pour que
      la carte remplisse exactement la ROI choisie ;
    - interpolation cubique sur une grille `factor` fois plus fine (contours courbes) ;
    - les zones sans valeur (NaN) restent vides : le masque est rééchantillonné au plus proche
      voisin, et les valeurs provisoires qui y servent au calcul sont masquées à la fin.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if extent is not None:
        x = np.concatenate([[extent[0]], x, [extent[1]]])
        y = np.concatenate([[extent[2]], y, [extent[3]]])
        z = np.pad(z, 1, mode="edge")
    if factor <= 1 or min(z.shape) < 2:
        return x, y, z
    from scipy.ndimage import distance_transform_edt, zoom
    bad = ~np.isfinite(z)
    zz = z
    if bad.all():
        return x, y, z
    if bad.any():                       # provisoire, pour le calcul de l'interpolation seulement
        zz = z[tuple(distance_transform_edt(bad, return_distances=False, return_indices=True))]
    xf = np.interp(np.linspace(0, len(x) - 1, (len(x) - 1) * factor + 1), np.arange(len(x)), x)
    yf = np.interp(np.linspace(0, len(y) - 1, (len(y) - 1) * factor + 1), np.arange(len(y)), y)
    zf = zoom(zz, ((len(yf)) / zz.shape[0], (len(xf)) / zz.shape[1]), order=3, mode="nearest", grid_mode=False)
    if bad.any():
        # cellule fine -> cellule d'origine la plus proche (en indice) : vide si celle-ci est vide
        iy = np.clip(np.rint(np.interp(yf, y, np.arange(len(y)))).astype(int), 0, len(y) - 1)
        ix = np.clip(np.rint(np.interp(xf, x, np.arange(len(x)))).astype(int), 0, len(x) - 1)
        zf[bad[np.ix_(iy, ix)]] = np.nan
    return xf, yf, zf


def nice_ref(v: float) -> float:
    """Valeur ronde (1, 2, 5 × 10^k) ≤ v, pour la flèche de référence."""
    if not v or v <= 0 or not np.isfinite(v):
        return 1.0
    p = 10 ** np.floor(np.log10(v))
    return float(max(m * p for m in (1, 2, 5) if m * p <= v))


def global_vmax(res: FieldResult, pct: float = 95.0) -> float:
    s = res.speed
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    v = float(np.percentile(s, pct))
    return v if v > 0 else float(s.max()) or 1.0


def window_title(t0: float, t1: float, time_unit: str, f0: int | None = None, f1: int | None = None) -> str:
    return f"t = {t0:.4g}–{t1:.4g} {time_unit}" + (f"  (frames {f0}–{f1})" if f0 is not None else "")


def fig_quiver(res: FieldResult, k: int, background: np.ndarray | None, bg_mode: str = "image",
               vmax: float | None = None, arrow_gain: float = 1.0, show_pit: bool = True,
               figsize=(5.2, 4.4), cmap: str = "viridis", title: str | None = None,
               scalebar: ScaleBar | None = None, contour: ContourStyle | None = None, panel: str | None = None):
    """Champ de vitesse de la fenêtre k.

    bg_mode : "image" (image moyenne de la fenêtre), "speed" (norme de la vitesse), "contour"
    (carte de couleurs remplie d'une grandeur, voir ContourStyle, + flèches noires et flèche de
    référence), "none". L'échelle des flèches et des couleurs est fixe si vmax est donné :
    indispensable pour comparer les fenêtres entre elles. panel : étiquette (« Pit A »).
    """
    vmax = vmax or global_vmax(res)
    s = res.scale
    ext = _extent(res.image_shape, s)
    vu = style.vel_unit(res.length_unit, res.time_unit)
    speed_label = style.label("Speed", vu)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    X, Y = np.meshgrid(res.x * s, res.y * s)
    U = np.ma.masked_invalid(res.u[k])
    V = np.ma.masked_invalid(res.v[k])
    mag = np.ma.masked_invalid(res.speed[k])
    norm = Normalize(0, vmax)
    # longueur de flèche (en unité de longueur) pour une vitesse vmax : ~0.9 cellule
    q_scale = vmax / (0.9 * res.cell * s * arrow_gain)
    bar_bg, bar_color = None, "black"
    y_of = None

    if bg_mode == "image" and background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=ext, interpolation="nearest")
        qv = ax.quiver(X, Y, U, V, mag, cmap=cmap, norm=norm, angles="xy", scale_units="xy",
                       scale=q_scale, width=0.004, headwidth=3.5, headlength=4, pivot="mid")
        style.colorbar(fig, qv, ax, speed_label, 0, vmax)
        bar_bg, bar_color = background, None
    elif bg_mode == "speed":
        im = ax.imshow(mag, cmap="magma", norm=norm, extent=ext, interpolation="nearest")
        ax.quiver(X, Y, U, V, color="white", angles="xy", scale_units="xy", scale=q_scale,
                  width=0.004, headwidth=3.5, headlength=4, pivot="mid")
        style.colorbar(fig, im, ax, speed_label, 0, vmax)
        bar_color = "white"
    elif bg_mode == "contour":
        cs = contour or ContourStyle()
        H, W = res.image_shape
        roi = (-0.5, W - 0.5, -0.5, H - 0.5)
        xf, yf, Zf = upsample_field(res.x, res.y, cs.scalar(res, k), cs.upsample, roi)
        cf = ax.contourf(xf * s, cs.y_display(yf, H, s), np.ma.masked_invalid(Zf), levels=cs.level_values(res),
                         cmap=cs.cmap, extend="max" if cs.quantity == "speed" else "both")
        interp = cs.interpolated(res, k)
        if interp.any():                                  # cellules interpolées : « × » (comme l'aperçu animé)
            iy, ix = np.nonzero(interp)
            ax.scatter(res.x[ix] * s, cs.y_display(res.y[iy], H, s), marker="x", s=26,
                       linewidths=1.1, color="black", zorder=5)
        Ua, Va = cs.vectors(res, k)
        n = max(1, cs.arrow_step)
        Xa, Ya = np.meshgrid(res.x * s, cs.y_display(res.y, H, s))
        qv = ax.quiver(Xa[::n, ::n], Ya[::n, ::n], np.ma.masked_invalid(Ua)[::n, ::n],
                       np.ma.masked_invalid(Va)[::n, ::n], color="black", angles="xy", scale_units="xy",
                       scale=q_scale, width=0.0035, headwidth=3.5, headlength=4, pivot="mid")
        ref = nice_ref(vmax)
        ax.set_xlim(ext[0], ext[1])
        y0, y1 = (0, H * s) if cs.y_up else (ext[2], ext[3])
        if cs.clean:
            # Style article : bande blanche en haut du cadre, où se trouve la flèche de référence,
            # sans chevaucher la carte
            y1 = y1 + 0.09 * (y1 - y0)
            ax.quiverkey(qv, 0.95, 0.965, ref, f"{ref:g} {vu}", labelpos="W", coordinates="axes",
                         fontproperties={"size": 8})
        else:
            ax.quiverkey(qv, 0.97, 1.02, ref, f"{ref:g} {vu}", labelpos="W",
                         coordinates="axes", fontproperties={"size": 8})
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        lo, hi = cs.color_limits(res)
        style.colorbar(fig, cf, ax, cs.fig_label(res), lo, hi, show_max=False)
        if cs.clean:
            title = ""
            show_pit = False
        y_of = (lambda py: cs.y_display(py, H, s))
    else:
        qv = ax.quiver(X, Y, U, V, mag, cmap=cmap, norm=norm, angles="xy", scale_units="xy",
                       scale=q_scale, width=0.004, headwidth=3.5, headlength=4, pivot="mid")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        style.colorbar(fig, qv, ax, speed_label, 0, vmax)
    if show_pit:
        _draw_pits(ax, res.pits, s, _pit_radii(res), y_of)
    _finish_image_axes(ax, s, res.image_shape, scalebar, res.length_unit, bar_bg, bar_color)
    style.panel_label(ax, panel)
    if title is None:
        title = window_title(res.t_start[k], res.t_end[k], res.time_unit, *res.windows[k])
    # En mode contours, la flèche de référence est juste au-dessus du cadre, à droite : le titre
    # passe au-dessus d'elle pour ne jamais la chevaucher, même sur une image étroite
    ax.set_title(title, pad=16 if bg_mode == "contour" else 6)
    return fig


def fig_series(res: FieldResult, current_k: int | None = None, figsize=(5.2, 3.6)):
    df = res.series
    vu = style.vel_unit(res.length_unit, res.time_unit)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained")
    t = df["t_center"]
    axes[0].plot(t, df["mean_speed"], "o-", ms=3, label="Mean")
    axes[0].plot(t, df["median_speed"], "s--", ms=3, label="Median")
    axes[0].set_ylabel(style.label("Speed", vu))
    axes[0].legend()
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].plot(t, df["mean_v_radial"], "o-", ms=3, color="C3")
    axes[1].set_ylabel(style.label("Radial velocity", vu))
    axes[1].set_xlabel(style.label("Time", res.time_unit))
    if current_k is not None and 0 <= current_k < len(df):
        for ax in axes:
            ax.axvline(t.iloc[current_k], color="0.3", lw=0.8, ls=":")
    what = {"trackpy": "Trackpy displacements", "piv": "measured cells (PIV)"}.get(
        res.method, "ROI pixels (optical flow)")
    axes[0].set_title(f"Time evolution, over {what}")
    return fig


def fig_radial_profile(res: FieldResult, n_bins: int = 12, figsize=(5.2, 3.6)):
    edges, prof = res.radial_profile(n_bins)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    finite = prof[np.isfinite(prof)]
    lim = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    lim = lim or 1.0
    # Chaque fenêtre est centrée sur son temps moyen (valable aussi pour des fenêtres qui se chevauchent)
    tc = res.t_center
    if len(tc) > 1:
        mids = 0.5 * (tc[1:] + tc[:-1])
        t_edges = np.concatenate([[tc[0] - (mids[0] - tc[0])], mids, [tc[-1] + (tc[-1] - mids[-1])]])
    else:
        t_edges = np.array([res.t_start[0], res.t_end[0]])
    pm = ax.pcolormesh(t_edges, edges, np.ma.masked_invalid(prof.T), cmap="RdBu_r",
                       norm=TwoSlopeNorm(0, -lim, lim), shading="flat")
    style.colorbar(fig, pm, ax, style.label("Radial velocity", style.vel_unit(res.length_unit, res.time_unit)),
                   show_max=False)
    ax.set_xlabel(style.label("Time", res.time_unit))
    ax.set_ylabel(style.label("Distance from pit", res.length_unit))
    ax.set_title("Radial velocity (> 0: away from pit)")
    return fig


def fig_speed_tracks(steps: pd.DataFrame, background: np.ndarray | None, scale: float, length_unit: str,
                     time_unit: str, pits: list[tuple[float, float]] | None = None, pit_radii=None,
                     vmax: float | None = None, end_frame: int | None = None, particles=None,
                     min_points: int = 3, panel: str | None = None, scalebar: ScaleBar | None = None,
                     show_ends: bool = True, max_tracks: int = 3000, cmap: str = style.TRACK_CMAP,
                     figsize=(5.0, 4.3), linewidth: float = 1.3, smooth: int = 1):
    """Trajectoires colorées par la vitesse le long du chemin (figure de référence « Pit A »).

    steps : une ligne par détection (particle, frame, xs, ys en px — positions lissées —, et
    v_win — vitesse glissante sur quelques frames — ou à défaut speed — vitesse du pas suivant —,
    en unité physique), comme ParticleAnalysis.steps ou particles.track_steps. Chaque segment prend
    la couleur de la vitesse (moyenne de ses deux extrémités) : la vitesse glissante évite que le
    bruit de localisation ne moucheture les trajectoires. smooth : lissage supplémentaire, pour
    l'affichage seulement, des positions et des vitesses (moyenne glissante sur n détections).
    À la dernière position (≤ end_frame) : cercle ouvert et flèche dans la direction du mouvement.
    Contour du pit en pointillés rouges, étiquette de panneau, barre d'échelle, « Particle speed ».
    """
    S = steps if particles is None else steps[steps["particle"].isin(particles)]
    if end_frame is not None:
        S = S[S["frame"] <= end_frame]
    S = S.sort_values(["particle", "frame"])
    n_pts = S.groupby("particle")["frame"].transform("size")
    S = S[n_pts >= max(2, min_points)]
    ids = S["particle"].unique()
    if len(ids) > max_tracks:
        ids = np.random.default_rng(0).choice(ids, max_tracks, replace=False)
        S = S[S["particle"].isin(ids)]
    S = S.reset_index(drop=True)
    g = S.groupby("particle", sort=False)
    col = "v_win" if "v_win" in S.columns else "speed"
    c = g[col].ffill()                                     # bouts de trajectoire sans vitesse glissante
    c = c.groupby(S["particle"], sort=False).bfill()
    xs, ys = S["xs"], S["ys"]
    if smooth > 1:
        roll = lambda a: a.groupby(S["particle"], sort=False).rolling(int(smooth), center=True, min_periods=1) \
            .mean().reset_index(level=0, drop=True)
        xs, ys, c = roll(xs), roll(ys), roll(c)
    x = xs.to_numpy() * scale
    y = ys.to_numpy() * scale
    pid = S["particle"].to_numpy()
    sp = c.to_numpy(dtype=float)
    same = pid[1:] == pid[:-1] if len(pid) else np.zeros(0, bool)
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)[same] \
        if len(pid) > 1 else np.zeros((0, 2, 2))
    vals = (0.5 * (sp[:-1] + sp[1:]))[same] if len(pid) > 1 else np.zeros(0)
    fin = vals[np.isfinite(vals)]
    if vmax is None or vmax <= 0:
        vmax = float(np.percentile(fin, 99)) if fin.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    shape = background.shape[:2] if background is not None else None
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=_extent(shape, scale), interpolation="nearest")
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(0, vmax), linewidths=linewidth,
                        capstyle="round", joinstyle="round", zorder=3)
    lc.set_array(np.ma.masked_invalid(vals))
    ax.add_collection(lc)
    if show_ends and len(pid):
        last = np.r_[~same, True]                          # dernière détection de chaque trajectoire
        # direction : déplacement sur les 3 derniers pas (au moins 1)
        back = np.zeros(len(pid), dtype=int)
        k_idx = np.flatnonzero(last)
        start_idx = np.r_[0, k_idx[:-1] + 1]
        back[k_idx] = np.maximum(start_idx, k_idx - 3)
        xe, ye = x[k_idx], y[k_idx]
        dx, dy = xe - x[back[k_idx]], ye - y[back[k_idx]]
        nrm = np.hypot(dx, dy)
        width = (shape[1] if shape else max(1.0, np.ptp(x) / scale)) * scale
        moving = nrm > 0.5 * scale                         # immobile à la fin : cercle seul
        L = 0.035 * width
        ax.scatter(xe, ye, s=18, facecolors="none", edgecolors=style.END_COLOR, linewidths=1.0, zorder=4)
        if moving.any():
            ax.quiver(xe[moving], ye[moving], dx[moving] / nrm[moving] * L, dy[moving] / nrm[moving] * L,
                      color=style.END_COLOR, angles="xy", scale_units="xy", scale=1, pivot="tail",
                      width=0.0045, headwidth=4.2, headlength=4.8, headaxislength=4.3, zorder=5)
    if pits:
        _draw_pits(ax, pits, scale, pit_radii)
    if shape is not None:
        ext = _extent(shape, scale)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    else:
        ax.autoscale()
        ax.invert_yaxis()
    ax.set_aspect("equal")
    style.colorbar(fig, lc, ax, style.label("Particle speed", style.vel_unit(length_unit, time_unit)), 0, vmax)
    _finish_image_axes(ax, scale, shape or (0, 0), scalebar if shape else None, length_unit, background)
    style.panel_label(ax, panel)
    return fig


def fig_tracks(tracks: pd.DataFrame, background: np.ndarray, scale: float, length_unit: str,
               pits: list[tuple[float, float]] | None = None, max_tracks: int = 3000, figsize=(5.0, 4.3),
               scalebar: ScaleBar | None = None, times: np.ndarray | None = None, time_unit: str = "s",
               pit_radii=None, panel: str | None = None, vmax: float | None = None, smooth: int = 3):
    """Trajectoires Trackpy. Avec `times` : colorées par la vitesse le long du chemin (style de
    référence, voir fig_speed_tracks) ; sans : une seule couleur (pas de vitesse calculable)."""
    from .particles import track_steps
    if times is not None and len(tracks):
        steps = track_steps(tracks, times, scale, smooth)
        return fig_speed_tracks(steps, background, scale, length_unit, time_unit, pits, pit_radii, vmax,
                                panel=panel, scalebar=scalebar, max_tracks=max_tracks, figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=_extent(background.shape, scale),
              interpolation="nearest")
    if len(tracks):
        ids = tracks["particle"].unique()
        if len(ids) > max_tracks:
            ids = np.random.default_rng(0).choice(ids, max_tracks, replace=False)
        sub = tracks[tracks["particle"].isin(ids)].sort_values(["particle", "frame"])
        for _, g in sub.groupby("particle"):
            ax.plot(g["x"] * scale, g["y"] * scale, lw=1.0, color=style.END_COLOR)
    if pits:
        _draw_pits(ax, pits, scale, pit_radii)
    _finish_image_axes(ax, scale, background.shape, scalebar, length_unit, background)
    style.panel_label(ax, panel)
    n = tracks["particle"].nunique() if len(tracks) else 0
    ax.set_title(f"{n} trajectories")
    return fig


def fig_density(dens: DensityResult, k: int, background: np.ndarray | None = None,
                vmax: float | None = None, overlay: bool = True, alpha: float = 0.65,
                show_pit: bool = True, figsize=(5.2, 4.4), cmap: str = "inferno",
                title: str | None = None, scalebar: ScaleBar | None = None, pit_radii=None,
                panel: str | None = None):
    """Heatmap de densité de particules de la fenêtre k.

    overlay : superpose la carte (cellules vides transparentes) à l'image moyenne de la fenêtre.
    Échelle de couleur fixe si vmax est donné, pour comparer les fenêtres entre elles.
    """
    vmax = vmax or densitymod.global_vmax(dens)
    s = dens.scale
    ext = _extent(dens.image_shape, s)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    z = dens.density[k]
    a = 1.0
    bar_bg, bar_color = None, "white"
    if overlay and background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=ext, interpolation="nearest")
        z = np.ma.masked_less_equal(z, 0)
        a = alpha
        bar_bg, bar_color = background, None
    pm = ax.pcolormesh(dens.x_edges * s, dens.y_edges * s, z, cmap=cmap, norm=Normalize(0, vmax),
                       alpha=a, shading="flat")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    style.colorbar(fig, pm, ax, style.label("Particle density", style.unit(("particles", 1), (dens.length_unit, -2))),
                   0, vmax)
    if show_pit:
        _draw_pits(ax, dens.pits, s, pit_radii)
    _finish_image_axes(ax, s, dens.image_shape, scalebar, dens.length_unit, bar_bg, bar_color)
    style.panel_label(ax, panel)
    if title is None:
        title = "Density, " + window_title(dens.t_start[k], dens.t_end[k], dens.time_unit, *dens.windows[k])
    ax.set_title(title)
    return fig


def fig_detection(img_display: np.ndarray, features: pd.DataFrame, diameter: int, figsize=(5.2, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(img_display, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if len(features):
        ax.scatter(features["x"], features["y"], s=(diameter * 1.6) ** 2 / 4, facecolors="none",
                   edgecolors=style.PIT_COLOR, linewidths=0.8)
    ax.set_title(f"{len(features)} detections")
    style.image_panel(ax, show_axes=True, length_unit="px")
    return fig


def fig_mass_hist(features: pd.DataFrame, minmass: float, figsize=(5.2, 2.6)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    if len(features):
        ax.hist(features["mass"], bins=60, color="#2B5C7E")
    ax.axvline(minmass, color=style.PIT_COLOR, lw=1, label="Current minmass")
    ax.set_xlabel(style.label("Integrated brightness (mass)", "a.u."))
    ax.set_ylabel("Detections")
    ax.legend()
    return fig


def fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi)
    return buf.getvalue()


def fig_to_pil(fig, dpi: int = 110) -> Image.Image:
    """Rendu à taille fixe (figsize x dpi), sans recadrage automatique : utile pour le GIF."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def make_gif(res: FieldResult, backgrounds: list[np.ndarray] | None, bg_mode: str, vmax: float,
             arrow_gain: float, fps: float = 4.0, dpi: int = 110, progress=None) -> bytes:
    frames = []
    nt = len(res.windows)
    for k in range(nt):
        fig = fig_quiver(res, k, backgrounds[k] if backgrounds else None, bg_mode, vmax, arrow_gain)
        frames.append(fig_to_pil(fig, dpi))
        plt.close(fig)
        if progress:
            progress((k + 1) / nt, f"GIF {k + 1}/{nt}")
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return buf.getvalue()
