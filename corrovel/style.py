"""Style commun de toutes les figures produites (PNG/SVG/PDF, vidéos) : celui des figures
d'article de référence (« Pit A » : trajectoires colorées par la vitesse).

- textes des figures en ANGLAIS (l'interface de l'application reste en français) ;
- police sans empattement (Arial / Helvetica, ou Liberation Sans, métriquement identique) ;
- étiquettes d'axes et de barres de couleur en gras, notation « Grandeur / (unité) » avec
  exposants : « Particle speed / (µm s⁻¹) » ;
- images sans graduations, encadrées, avec barre d'échelle (noire sur fond clair, blanche sur
  fond sombre) et étiquette de panneau en gras gris en haut à gauche (« Pit A ») ;
- barre de couleur pleine hauteur, valeur maximale indiquée en haut ;
- contour du pit : cercle rouge en pointillés.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

PREFERRED_FONTS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def _installed_fonts() -> list[str]:
    """Polices préférées réellement installées, DejaVu Sans (livrée avec matplotlib) en dernier :
    matplotlib n'utilise une police de secours pour un caractère absent (« ⁻ » manque à Arial /
    Liberation Sans) que si `font.family` est une LISTE de familles."""
    from matplotlib.font_manager import fontManager
    have = {f.name for f in fontManager.ttflist}
    fonts = [f for f in PREFERRED_FONTS if f in have]
    return fonts if "DejaVu Sans" in fonts else fonts + ["DejaVu Sans"]


FONTS = _installed_fonts()
PIT_COLOR = "#E0201B"           # contour du pit
END_COLOR = "#1F55D0"           # position actuelle / direction des traceurs (cercle + flèche)
PANEL_COLOR = "0.45"            # étiquette de panneau
TRACK_CMAP = "viridis"

RC = {
    "font.family": FONTS, "font.sans-serif": FONTS,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.3, "mathtext.default": "regular",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "svg.fonttype": "none",
}


def apply():
    plt.rcParams.update(RC)


apply()

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def unit(*parts: tuple[str, int]) -> str:
    """Unité composée en notation SI à exposants : unit(("µm", 1), ("s", -1)) -> « µm s⁻¹ »."""
    return " ".join(sym + ("" if p == 1 else str(p).translate(_SUP)) for sym, p in parts if sym)


def vel_unit(length_unit: str, time_unit: str) -> str:
    return unit((length_unit, 1), (time_unit, -1))


def rate_unit(time_unit: str) -> str:
    return unit((time_unit, -1))


def label(name: str, u: str | None = None) -> str:
    """« Particle speed / (µm s⁻¹) » ; sans unité : le nom seul."""
    return f"{name} / ({u})" if u else name


def image_panel(ax, show_axes: bool = False, length_unit: str = "px"):
    """Panneau image : cadre fin, sans graduations (la barre d'échelle donne les dimensions).
    show_axes : garde les graduations (image non étalonnée, ou sans barre d'échelle)."""
    if show_axes:
        ax.set_xlabel(label("x", length_unit))
        ax.set_ylabel(label("y", length_unit))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("black")


def panel_letter(ax, text: str | None):
    """Lettre de panneau d'un graphique (« (b) »), en gras, en haut à gauche, hors du cadre."""
    if text:
        ax.text(-0.02, 1.03, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
                fontweight="bold", color="black")


def panel_label(ax, text: str | None):
    if text:
        ax.text(0.025, 0.975, text, transform=ax.transAxes, ha="left", va="top", fontsize=10,
                fontweight="bold", color=PANEL_COLOR, zorder=10)


def colorbar(fig, mappable, ax, text: str, vmin: float | None = None, vmax: float | None = None,
             show_max: bool = True, **kw):
    """Barre de couleur pleine hauteur, étiquette en gras ; la valeur maximale est indiquée en
    haut quand elle ne tombe pas sur une graduation (comme « 32 » au-dessus de « 30 »)."""
    cb = fig.colorbar(mappable, ax=ax, fraction=kw.pop("fraction", 0.046), pad=kw.pop("pad", 0.03),
                      aspect=kw.pop("aspect", 22), **kw)
    cb.set_label(text)
    cb.outline.set_linewidth(0.8)
    if show_max and vmin is not None and vmax is not None and vmax > vmin:
        ticks = [t for t in MaxNLocator(nbins=4).tick_values(vmin, vmax) if vmin - 1e-9 <= t <= vmax + 1e-9]
        if not ticks or vmax - ticks[-1] > 0.03 * (vmax - vmin):
            ticks.append(vmax)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.3g}" for t in ticks])
    return cb


def contrast_color(gray_region: np.ndarray | None) -> str:
    """Noir sur fond clair, blanc sur fond sombre (région d'image uint8 sous l'élément)."""
    if gray_region is None or not np.size(gray_region):
        return "black"
    return "black" if float(np.nanmean(gray_region)) >= 110 else "white"


def draw_pits(ax, pits, scale: float, radii_px=None, y_of=None):
    """Pits : contour en cercle rouge pointillé (rayon estimé, px) ; sans rayon connu, une croix
    rouge au centre. Plusieurs pits : étiquetés P1, P2…  `y_of` : fonction d'ordonnée d'affichage
    (axe y vers le haut), sinon y × scale."""
    from matplotlib.patches import Circle
    radii = list(radii_px or [])
    multi = len(pits) > 1
    for i, (px, py) in enumerate(pits):
        x = px * scale
        y = y_of(py) if y_of else py * scale
        r = radii[i] if i < len(radii) else 0
        if r and r > 0:
            ax.add_patch(Circle((x, y), r * scale, fill=False, edgecolor=PIT_COLOR, lw=1.3, ls=(0, (3, 2)),
                                zorder=6))
        else:
            ax.plot(x, y, marker="+", color=PIT_COLOR, ms=9, mew=1.5, zorder=6)
        if multi:
            ax.annotate(f"P{i + 1}", (x, y), color=PIT_COLOR, fontsize=8, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points", zorder=6)
