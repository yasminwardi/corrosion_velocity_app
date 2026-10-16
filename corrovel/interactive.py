"""Figures Plotly interactives pour le navigateur (par opposition à plots.py, qui produit les
figures matplotlib statiques exportées en PNG/SVG/PDF).

field_animation_figure construit une figure animée (lecture / pause / curseur) de l'évolution du
champ de vitesse dans le temps, qui joue côté navigateur sans recharger la page Streamlit ;
density_animation_figure fait de même pour la heatmap de densité de particules.
"""
from __future__ import annotations

import base64
import io

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .colormaps import plotly_scale
from .density import DensityResult
from .field import FieldResult
from . import style
from .plots import ContourStyle, nice_ref, upsample_field

FIG_FONT = dict(family=", ".join(style.FONTS))


def _cb_title(label: str) -> str:
    """Titre de barre de couleur en gras, l'unité à la ligne : « Speed<br>/ (µm s⁻¹) »."""
    return "<b>" + label.replace(" / ", "<br>/ ") + "</b>"


def _axis_title(name: str, unit: str) -> dict:
    return dict(text=f"<b>{style.label(name, unit)}</b>")


# Les figures Plotly sont envoyées au navigateur dans un seul message (limite Streamlit : 200 Mo).
# Une animation embarque une image et des flèches PAR FENÊTRE : avec des images de 1700 × 1700 px
# et 72 fenêtres, c'était ~470 Mo. L'animation sert à explorer à l'écran (~700 px de large) :
# images de fond réduites et compressées, flèches éclaircies. Les figures exportées et les vidéos
# (matplotlib) gardent la pleine résolution et toutes les flèches.
ANIM_MAX_SIDE = 720       # px : côté max des images de fond dans les animations
ANIM_MAX_ARROWS = 45      # flèches max par côté de la grille dans les animations
ANIM_MAX_CONTOUR = 200    # points max par côté pour les cartes de contours animées


def _png_uri(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _image_trace(img: np.ndarray, s: float, max_side: int = ANIM_MAX_SIDE) -> go.Image:
    """Image de fond (niveaux de gris ou RVB, en px de l'image) réduite à `max_side` px et
    compressée en JPEG, placée exactement sur les coordonnées physiques de l'image d'origine."""
    h, w = img.shape[:2]
    f = max(1.0, max(h, w) / max_side)
    nw, nh = max(1, round(w / f)), max(1, round(h / f))
    pil = Image.fromarray(img if img.ndim == 3 else np.stack([img] * 3, axis=-1))
    if (nw, nh) != (w, h):
        pil = pil.resize((nw, nh), Image.LANCZOS)
    fx, fy = w / nw, h / nh                                 # px d'origine par px de l'image réduite
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return go.Image(source=uri, x0=(0.5 * fx - 0.5) * s, y0=(0.5 * fy - 0.5) * s, dx=fx * s, dy=fy * s,
                    hoverinfo="skip")


def _speed_trace(res: FieldResult, k: int, vmax: float) -> go.Image:
    """Carte |v| à la résolution de la grille (une couleur par cellule) : quelques ko par fenêtre."""
    cmap = mpl.colormaps["magma"]
    norm = np.clip(np.nan_to_num(res.speed[k], nan=0.0) / max(vmax, 1e-9), 0.0, 1.0)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    s, c = res.scale, res.cell
    return go.Image(source=_png_uri(rgb), x0=float(res.x[0]) * s, y0=float(res.y[0]) * s, dx=c * s, dy=c * s,
                    hoverinfo="skip")


def anim_arrow_step(res: FieldResult, requested: int = 1) -> int:
    """Pas d'affichage des flèches dans une animation (au plus ANIM_MAX_ARROWS par côté)."""
    auto = int(np.ceil(max(len(res.x), len(res.y)) / ANIM_MAX_ARROWS))
    return max(1, int(requested), auto)


def _arrows_xy(x, y, u, v, scale: float, arrow_scale: float = 0.35, angle: float = np.pi / 9):
    """Tracés des flèches (tige + pointe), séparés par des NaN, pour un seul go.Scatter.

    Même géométrie que plotly.figure_factory.create_quiver, mais vectorisé : create_quiver
    recalcule toutes les pointes dans une boucle sur chaque flèche (coût quadratique), soit
    ~1 min par fenêtre pour une grille de 110 × 110 cellules.
    """
    x1, y1 = x + u * scale, y + v * scale
    head = np.hypot(x1 - x, y1 - y) * arrow_scale
    theta = np.arctan2(y1 - y, x1 - x)
    p1x, p1y = x1 - head * np.cos(theta + angle), y1 - head * np.sin(theta + angle)
    p2x, p2y = x1 - head * np.cos(theta - angle), y1 - head * np.sin(theta - angle)
    gap = np.full_like(x, np.nan)
    xs = np.concatenate([np.column_stack([x, x1, gap]).ravel(), np.column_stack([p1x, x1, p2x, gap]).ravel()])
    ys = np.concatenate([np.column_stack([y, y1, gap]).ravel(), np.column_stack([p1y, y1, p2y, gap]).ravel()])
    return xs, ys


def _quiver_trace(res: FieldResult, k: int, vmax: float, arrow_gain: float, color: str,
                  step: int = 1, cs: ContourStyle | None = None) -> go.Scatter:
    """Flèches à l'échelle globale (comme plots.fig_quiver) : une vitesse vmax donne une flèche
    d'environ 0.9 cellule, quelle que soit la fenêtre — indispensable pour rester comparable.
    step : une flèche sur `step` dans chaque direction ; cs : style contours (flèches lissées,
    y vers le haut)."""
    s = res.scale
    if cs is not None:
        X, Y = np.meshgrid(res.x * s, cs.y_display(res.y, res.image_shape[0], s))
        U, V = cs.vectors(res, k)
    else:
        X, Y = np.meshgrid(res.x * s, res.y * s)
        U, V = res.u[k], res.v[k]
    X, Y, U, V = (a[::step, ::step] for a in (X, Y, U, V))
    ok = np.isfinite(U) & np.isfinite(V)
    if not ok.any():
        return go.Scatter(x=[], y=[], mode="lines", line=dict(color=color, width=1.6),
                          hoverinfo="skip", showlegend=False)
    quiver_scale = (0.9 * res.cell * s * arrow_gain) / max(vmax, 1e-9)
    xs, ys = _arrows_xy(X[ok], Y[ok], U[ok], V[ok], quiver_scale)
    return go.Scatter(x=xs.astype(np.float32), y=ys.astype(np.float32), mode="lines",
                      line=dict(color=color, width=1.6), hoverinfo="skip", showlegend=False)


def field_animation_figure(res: FieldResult, backgrounds: list[np.ndarray] | None, bg_mode: str,
                           vmax: float, arrow_gain: float, playback_fps: float = 2.0,
                           contour: ContourStyle | None = None) -> go.Figure:
    """Figure Plotly animée : un go.Frame par fenêtre temporelle, lecture/pause/curseur.

    bg_mode : "image" (moyenne de la fenêtre, niveaux de gris), "speed" (norme de la vitesse,
    colormap magma), "contour" (carte de couleurs remplie d'une grandeur, flèches noires et
    flèche de référence, voir plots.ContourStyle), "none" (fond blanc). Échelle de couleur et de
    flèches fixe sur toutes les fenêtres, pour rester comparable entre fenêtres.
    """
    nt = len(res.windows)
    s = res.scale
    h, w = res.image_shape
    vu = style.vel_unit(res.length_unit, res.time_unit)
    radii = list(getattr(res, "pit_radii", None) or [])
    ext_x = [-0.5 * s, (w - 0.5) * s]
    ext_y = [(h - 0.5) * s, -0.5 * s]
    quiver_color = {"speed": "white", "contour": "black"}.get(bg_mode, "#E8423F")
    cs = contour or ContourStyle()
    y_up = bg_mode == "contour" and cs.y_up       # y vers le haut : seulement sans image de fond
    if y_up:
        ext_y = [0.0, h * s]
    if bg_mode == "contour":
        levels = cs.level_values(res)          # mêmes niveaux ronds que la figure matplotlib
        lo, hi = float(levels[0]), float(levels[-1])
        cscale = plotly_scale(cs.cmap)

    def bg_trace(k: int) -> go.Image | go.Contour | None:
        if bg_mode == "image" and backgrounds is not None:
            return _image_trace(backgrounds[k], s)
        if bg_mode == "speed":
            return _speed_trace(res, k, vmax)
        if bg_mode == "contour":
            up = max(1, min(cs.upsample, ANIM_MAX_CONTOUR // (max(len(res.x), len(res.y)) + 2)))
            xf, yf, z = upsample_field(res.x, res.y, cs.scalar(res, k), up, (-0.5, w - 0.5, -0.5, h - 0.5))
            z = z.astype(np.float32)
            yd = cs.y_display(yf, h, s)
            if yd[0] > yd[-1]:                    # Plotly : ordonnées croissantes
                yd, z = yd[::-1], z[::-1]
            return go.Contour(
                x=xf * s, y=yd, z=z, zmin=lo, zmax=hi, colorscale=cscale, line_smoothing=0.85,
                contours=dict(start=lo, end=hi, size=float(levels[1] - levels[0]), coloring="fill",
                              showlines=False),
                colorbar=dict(title=_cb_title(cs.fig_label(res)), thickness=14),
                hovertemplate="%{z:.3g}<extra></extra>")
        return None

    pit_trace = _pit_trace([(px, cs.y_display(py, h, 1.0) if y_up else py) for px, py in res.pits], s, radii)
    colorbar_trace = go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(size=0.0001, color=[0, vmax], cmin=0, cmax=vmax,
                                           colorscale="Viridis" if bg_mode != "speed" else "Magma",
                                           showscale=True,
                                           colorbar=dict(title=_cb_title(style.label("Speed", vu)), thickness=14)),
                                showlegend=False, hoverinfo="skip")
    step = anim_arrow_step(res, cs.arrow_step if bg_mode == "contour" else 1)

    def frame_data(k: int) -> list:
        data = []
        bg = bg_trace(k)
        if bg is not None:
            data.append(bg)
        data.append(_quiver_trace(res, k, vmax, arrow_gain, quiver_color, step,
                                  cs if bg_mode == "contour" else None))
        if bg_mode == "contour" and cs.fill_gaps:
            iy, ix = np.nonzero(cs.interpolated(res, k))     # cellules interpolées, signalées par « × »
            data.append(go.Scatter(x=res.x[ix] * s, y=cs.y_display(res.y[iy], h, s), mode="markers",
                                   marker=dict(symbol="x-thin", size=7, line=dict(width=1, color="black")),
                                   name="interpolated", showlegend=False, hoverinfo="skip"))
        if not (bg_mode == "contour" and cs.clean):
            data.append(pit_trace)
        if bg_mode != "contour":          # la carte de contours a sa propre barre de couleur
            data.append(colorbar_trace)
        return data

    def frame_title(k: int) -> str:
        return (f"<b>t = {res.t_start[k]:.4g}–{res.t_end[k]:.4g} {res.time_unit}</b>"
               f"  (window {k + 1}/{nt})")

    frames = [go.Frame(data=frame_data(k), name=str(k),
                       layout=go.Layout(title=frame_title(k))) for k in range(nt)]

    fig = go.Figure(data=frame_data(0), frames=frames)
    fig.update_layout(
        title=frame_title(0),
        xaxis=dict(range=ext_x, showgrid=False, constrain="domain", title=_axis_title("x", res.length_unit)),
        yaxis=dict(range=ext_y, showgrid=False, scaleanchor="x", title=_axis_title("y", res.length_unit)),
        font=FIG_FONT,
        plot_bgcolor="white" if bg_mode == "contour" else None,
        margin=dict(l=10, r=10, t=40, b=10),
        height=int(min(700, max(340, 720 * h / w))),
        **_animation_controls(nt, playback_fps),
    )
    if not (bg_mode == "contour" and cs.clean):
        _add_pit_outlines(fig, [(px * s, cs.y_display(py, h, s) if y_up else py * s) for px, py in res.pits],
                          radii, s)
    if bg_mode == "contour":
        # Flèche de référence (« 1 µm/s → »), dans le coin haut droit, à l'échelle des flèches
        ref = nice_ref(vmax)
        length = ref * (0.9 * res.cell * s * arrow_gain) / max(vmax, 1e-9)
        top, bottom = (ext_y[1], ext_y[0]) if not y_up else (ext_y[1], ext_y[0])
        x1, y1 = ext_x[1] - 0.03 * (ext_x[1] - ext_x[0]), top + 0.05 * (bottom - top)
        fig.add_annotation(x=x1, y=y1, ax=x1 - length, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.6, arrowcolor="black",
                           text=f"{ref:g} {vu} ", xanchor="right",
                           bgcolor="rgba(255,255,255,0.8)", font=dict(size=11, color="black"))
    return fig


def _pit_trace(pits: list[tuple[float, float]], s: float, radii=None) -> go.Scatter:
    """Centres des pits : croix rouge si le rayon est inconnu (sinon le contour est dessiné par
    _add_pit_outlines) ; étiquettes P1, P2… s'il y en a plusieurs."""
    radii = list(radii or [])
    known = [i < len(radii) and radii[i] > 0 for i in range(len(pits))]
    return go.Scatter(x=[p[0] * s for p in pits], y=[p[1] * s for p in pits],
                      mode="markers+text" if len(pits) > 1 else "markers",
                      text=[f"<b>P{i + 1}</b>" for i in range(len(pits))], textposition="top center",
                      textfont=dict(color=style.PIT_COLOR, size=11),
                      marker=dict(symbol="cross", size=[0.1 if k else 12 for k in known], color=style.PIT_COLOR,
                                  line=dict(width=1.5)),
                      showlegend=False, hoverinfo="skip")


def _add_pit_outlines(fig: go.Figure, centers: list[tuple[float, float]], radii, s: float):
    """Contour des pits en cercle rouge pointillé (forme fixe : reste affichée sur toutes les frames)."""
    for (cx, cy), r in zip(centers, list(radii or [])):
        if r and r > 0:
            fig.add_shape(type="circle", xref="x", yref="y", x0=cx - r * s, x1=cx + r * s, y0=cy - r * s,
                          y1=cy + r * s, line=dict(color=style.PIT_COLOR, width=1.5, dash="dash"), layer="above")


def density_animation_figure(dens: DensityResult, backgrounds: list[np.ndarray] | None,
                             vmax: float, playback_fps: float = 2.0, opacity: float = 0.65,
                             pit_radii=None) -> go.Figure:
    """Heatmap animée de la densité de particules, une go.Frame par fenêtre temporelle.

    backgrounds : images moyennes des fenêtres ; si fourni, la heatmap est superposée (cellules
    vides transparentes). Échelle de couleur fixe (vmax) sur toutes les fenêtres.
    """
    nt = len(dens.windows)
    s = dens.scale
    h, w = dens.image_shape
    xe, ye = dens.x_edges * s, dens.y_edges * s
    overlay = backgrounds is not None
    pit_trace = _pit_trace(dens.pits, s, pit_radii)

    def frame_data(k: int) -> list:
        data = []
        if overlay:
            data.append(_image_trace(backgrounds[k], s))
        z = dens.density[k].astype(np.float32)
        if overlay:
            z = np.where(z > 0, z, np.nan)
        data.append(go.Heatmap(
            x=xe, y=ye, z=z, zmin=0, zmax=vmax, colorscale="Inferno", opacity=opacity if overlay else 1.0,
            colorbar=dict(title=_cb_title(style.label("Particle density",
                                                      style.unit(("particles", 1), (dens.length_unit, -2)))),
                          thickness=14),
            hovertemplate="density = %{z:.3g}<extra></extra>"))
        data.append(pit_trace)
        return data

    def frame_title(k: int) -> str:
        return (f"<b>Density, t = {dens.t_start[k]:.4g}–{dens.t_end[k]:.4g} {dens.time_unit}</b>"
               f"  (window {k + 1}/{nt})")

    frames = [go.Frame(data=frame_data(k), name=str(k),
                       layout=go.Layout(title=frame_title(k))) for k in range(nt)]
    fig = go.Figure(data=frame_data(0), frames=frames)
    fig.update_layout(
        title=frame_title(0),
        xaxis=dict(range=[xe[0], xe[-1]], showgrid=False, constrain="domain",
                   title=_axis_title("x", dens.length_unit)),
        yaxis=dict(range=[ye[-1], ye[0]], showgrid=False, scaleanchor="x",
                   title=_axis_title("y", dens.length_unit)),
        font=FIG_FONT,
        margin=dict(l=10, r=10, t=40, b=10),
        height=int(min(700, max(340, 720 * h / w))),
        **_animation_controls(nt, playback_fps),
    )
    _add_pit_outlines(fig, [(px * s, py * s) for px, py in dens.pits], pit_radii, s)
    return fig


def _animation_controls(nt: int, playback_fps: float) -> dict:
    """Boutons lecture/pause et curseur de fenêtre, communs aux figures animées."""
    frame_ms = max(50, int(1000 / max(playback_fps, 0.1)))
    return dict(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", showactive=False,
            buttons=[
                dict(label="▶ Lecture", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.08, len=0.92, y=-0.12,
            currentvalue=dict(prefix="Fenêtre : ", visible=True),
            steps=[dict(label=str(k), method="animate",
                       args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                  for k in range(nt)],
        )],
    )
