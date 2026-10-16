"""Palettes supplémentaires : « parula », approximation de la palette par défaut de MATLAB.

Interpolée à partir de 9 points de la table MATLAB à 64 couleurs : visuellement très proche,
pas identique au bit près. Enregistrée auprès de matplotlib (mpl.colormaps["parula"]).
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

_PARULA_ANCHORS = [
    (0.2081, 0.1663, 0.5292), (0.0117, 0.3875, 0.8820), (0.0793, 0.5200, 0.8312),
    (0.0228, 0.6535, 0.7768), (0.2178, 0.7250, 0.6193), (0.5709, 0.7485, 0.4494),
    (0.8507, 0.7299, 0.3360), (0.9789, 0.8271, 0.1633), (0.9763, 0.9831, 0.0538),
]

PARULA = LinearSegmentedColormap.from_list("parula", _PARULA_ANCHORS, N=256)
if "parula" not in mpl.colormaps:
    mpl.colormaps.register(PARULA)


def plotly_scale(name: str, n: int = 11) -> list[list]:
    """Palette matplotlib -> colorscale Plotly ([[position, "rgb(r,g,b)"], ...])."""
    cmap = mpl.colormaps[name]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, _ = cmap(t)
        out.append([t, f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return out
