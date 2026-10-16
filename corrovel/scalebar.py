"""Barre d'échelle (« 50 µm ») incrustée sur les images (PIL) et sur les figures matplotlib.

Même rendu partout (style des figures d'article, voir corrovel.style) : barre pleine et longueur
au-dessus, sans cartouche, noires sur fond clair et blanches sur fond sombre (choisies d'après
l'image sous la barre). La longueur automatique est une valeur ronde (1, 2, 5 × 10^k) proche de
20 % de la largeur de l'image.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POSITIONS = {"bottom-right": "bas droite", "bottom-left": "bas gauche",
             "top-right": "haut droite", "top-left": "haut gauche"}


@dataclass(frozen=True)
class ScaleBar:
    um_per_px: float                  # taille d'un pixel de l'image d'origine
    length_um: float = 0.0            # 0 = automatique
    position: str = "bottom-right"

    def length_for(self, width_px: float) -> float:
        """Longueur de la barre (µm) pour une image de `width_px` pixels d'origine de large."""
        if self.length_um > 0:
            return float(self.length_um)
        target = 0.2 * width_px * self.um_per_px
        if target <= 0:
            return 1.0
        p = 10 ** np.floor(np.log10(target))
        return float(max(m * p for m in (1, 2, 5) if m * p <= target))

    @staticmethod
    def label(length_um: float) -> str:
        return f"{length_um:g} µm"


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Police des figures (Arial / Liberation Sans…, voir style.FONTS), sinon DejaVu Sans livrée
    avec matplotlib : présente sur tous les systèmes, avec le « µ »."""
    try:
        from matplotlib import font_manager
        from .style import FONTS
        fp = font_manager.FontProperties(family=FONTS, weight="bold" if bold else "normal")
        return ImageFont.truetype(font_manager.findfont(fp), size)
    except Exception:
        try:
            from matplotlib import font_manager
            return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:
            return ImageFont.load_default()


def _corner(h: int, w: int, position: str, fh: float = 0.18, fw: float = 0.35) -> tuple[slice, slice]:
    """Région d'image (lignes, colonnes) où se trouve la barre, pour choisir sa couleur."""
    rows = slice(int(h * (1 - fh)), h) if position.startswith("bottom") else slice(0, int(h * fh) + 1)
    cols = slice(int(w * (1 - fw)), w) if position.endswith("right") else slice(0, int(w * fw) + 1)
    return rows, cols


def draw_on_rgb(img: np.ndarray, sb: ScaleBar, upscale: float = 1) -> np.ndarray:
    """Incruste la barre sur une image RGB uint8 (H, W, 3). `upscale` : nombre de pixels de
    l'image par pixel d'origine (images agrandies pour la vidéo)."""
    from .style import contrast_color
    h, w = img.shape[:2]
    length = sb.length_for(w / upscale)
    bar_px = max(1, int(round(length / sb.um_per_px * upscale)))
    fnt = font(max(11, min(h, w) // 28))
    rows, cols = _corner(h, w, sb.position)
    ink = (0, 0, 0) if contrast_color(img[rows, cols].mean(axis=-1)) == "black" else (255, 255, 255)
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    text = sb.label(length)
    tb = draw.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    thick = max(3, min(h, w) // 90)
    gap, margin = max(3, thick), max(8, min(h, w) // 30)
    box_w, box_h = max(bar_px, tw), th + gap + thick
    x0 = w - margin - box_w if sb.position.endswith("right") else margin
    y0 = h - margin - box_h if sb.position.startswith("bottom") else margin
    cx = x0 + box_w / 2
    draw.text((cx - tw / 2 - tb[0], y0 - tb[1]), text, font=fnt, fill=ink)
    by = y0 + th + gap
    draw.rectangle([cx - bar_px / 2, by, cx + bar_px / 2 - 1, by + thick - 1], fill=ink)
    return np.asarray(pil)


def add_to_axes(ax, sb: ScaleBar, axis_scale: float, width_px: float, background: np.ndarray | None = None,
                color: str | None = None):
    """Barre sur un axe matplotlib dont les coordonnées valent px × `axis_scale` (µm ou px).
    Couleur : `color`, sinon noire ou blanche selon l'image `background` (uint8) sous la barre."""
    from matplotlib import font_manager
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from .style import FONTS, contrast_color
    length = sb.length_for(width_px)
    if color is None:
        region = None
        if background is not None:
            g = background if background.ndim == 2 else background[..., :3].mean(axis=-1)
            region = g[_corner(*g.shape[:2], sb.position)]
        color = contrast_color(region)
    loc = sb.position.replace("bottom", "lower").replace("top", "upper").replace("-", " ")
    bar = AnchoredSizeBar(ax.transData, length / sb.um_per_px * axis_scale, sb.label(length), loc,
                          pad=0.5, borderpad=0.5, sep=3, color=color, frameon=False,
                          size_vertical=0.009 * width_px * axis_scale, label_top=True,
                          fontproperties=font_manager.FontProperties(family=FONTS, size=8))
    ax.add_artist(bar)
