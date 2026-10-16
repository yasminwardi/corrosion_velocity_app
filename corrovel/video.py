"""Export vidéo (MP4 / GIF) : évolution du champ de vitesse et de la densité de particules
(carte par cellules ou heatmap pleine résolution à traînées), et lecture de la séquence brute.

MP4 est écrit avec imageio + imageio-ffmpeg (binaire ffmpeg statique embarqué, pas d'installation
système requise). Le writer ffmpeg d'imageio a besoin d'un vrai chemin de fichier : on écrit donc
dans un fichier temporaire puis on relit les octets (même schéma que le zip d'export dans app.py).
Toutes les vidéos acceptent une barre d'échelle optionnelle (corrovel.scalebar).
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from . import density as densitymod
from . import io as iom
from . import plots
from . import scalebar as sbmod
from .density import BaseMap, DensityResult
from .field import FieldResult
from .scalebar import ScaleBar

ProgressFn = Callable[[float, str], None]


def _write_frames(frames: list[np.ndarray], fps: float, fmt: str) -> bytes:
    """Encode une liste d'images RGB (H, W, 3) uint8 en MP4 ou GIF, retourne les octets."""
    if not frames:
        raise ValueError("Aucune image à encoder.")
    suffix = ".mp4" if fmt == "mp4" else ".gif"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"video{suffix}"
        if fmt == "mp4":
            # Dimensions paires requises par le codec H.264 (yuv420p)
            h, w = frames[0].shape[:2]
            h2, w2 = h - (h % 2), w - (w % 2)
            if (h2, w2) != (h, w):
                frames = [f[:h2, :w2] for f in frames]
            with imageio.get_writer(str(path), fps=max(fps, 0.5), codec="libx264",
                                    quality=8, macro_block_size=None) as w_:
                for fr in frames:
                    w_.append_data(fr)
        else:
            duration_ms = 1000.0 / max(fps, 0.5)
            imageio.mimwrite(str(path), frames, format="GIF", duration=duration_ms)
        return path.read_bytes()


def field_video_frames(res: FieldResult, backgrounds: list[np.ndarray] | None, bg_mode: str,
                       vmax: float, arrow_gain: float, dpi: int = 110,
                       progress: ProgressFn | None = None, scalebar: ScaleBar | None = None,
                       contour: plots.ContourStyle | None = None) -> list[np.ndarray]:
    """Rendu matplotlib (même figure que l'export statique) de chaque fenêtre temporelle."""
    frames = []
    nt = len(res.windows)
    if contour is not None and contour.limits is None:   # échelle commune, calculée une fois
        contour = replace(contour, limits=contour.color_limits(res))
    for k in range(nt):
        fig = plots.fig_quiver(res, k, backgrounds[k] if backgrounds else None, bg_mode, vmax, arrow_gain,
                               scalebar=scalebar, contour=contour)
        frames.append(np.asarray(plots.fig_to_pil(fig, dpi)))
        plt.close(fig)
        if progress:
            progress((k + 1) / nt, f"Image {k + 1}/{nt}")
    return frames


def field_video_bytes(res: FieldResult, backgrounds: list[np.ndarray] | None, bg_mode: str,
                      vmax: float, arrow_gain: float, fps: float = 4.0, fmt: str = "mp4",
                      dpi: int = 110, progress: ProgressFn | None = None,
                      scalebar: ScaleBar | None = None, contour: plots.ContourStyle | None = None) -> bytes:
    """Vidéo de l'évolution du champ de vitesse (une image par fenêtre temporelle)."""
    frames = field_video_frames(res, backgrounds, bg_mode, vmax, arrow_gain, dpi,
                                progress=lambda f, t: progress(0.85 * f, t) if progress else None,
                                scalebar=scalebar, contour=contour)
    if progress:
        progress(0.9, "Encodage vidéo…")
    data = _write_frames(frames, fps, fmt)
    if progress:
        progress(1.0, "Terminé")
    return data


def density_video_frames(dens: DensityResult, backgrounds: list[np.ndarray] | None, overlay: bool,
                         vmax: float, dpi: int = 110, progress: ProgressFn | None = None,
                         scalebar: ScaleBar | None = None) -> list[np.ndarray]:
    """Rendu matplotlib (même figure que l'export statique) de la densité de chaque fenêtre."""
    frames = []
    nt = len(dens.windows)
    for k in range(nt):
        fig = plots.fig_density(dens, k, backgrounds[k] if backgrounds else None, vmax, overlay,
                                scalebar=scalebar)
        frames.append(np.asarray(plots.fig_to_pil(fig, dpi)))
        plt.close(fig)
        if progress:
            progress((k + 1) / nt, f"Image {k + 1}/{nt}")
    return frames


def density_video_bytes(dens: DensityResult, backgrounds: list[np.ndarray] | None, overlay: bool,
                        vmax: float, fps: float = 4.0, fmt: str = "mp4", dpi: int = 110,
                        progress: ProgressFn | None = None, scalebar: ScaleBar | None = None) -> bytes:
    """Vidéo de l'évolution de la densité de particules (une image par fenêtre temporelle)."""
    frames = density_video_frames(dens, backgrounds, overlay, vmax, dpi,
                                  progress=lambda f, t: progress(0.85 * f, t) if progress else None,
                                  scalebar=scalebar)
    if progress:
        progress(0.9, "Encodage vidéo…")
    data = _write_frames(frames, fps, fmt)
    if progress:
        progress(1.0, "Terminé")
    return data


def _stamp(img: Image.Image, label: str, font) -> Image.Image:
    """Incruste `label` en haut à gauche, en gras (style des figures, comme l'étiquette « Pit A ») :
    noir sur fond clair, blanc sur fond sombre, avec un liseré de la couleur opposée."""
    from .style import contrast_color
    draw = ImageDraw.Draw(img)
    pad = max(4, img.size[1] // 60)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    region = np.asarray(img.convert("L"))[:th + 2 * pad, :tw + 2 * pad]
    dark = contrast_color(region) == "black"
    ink, edge = ((0, 0, 0), (255, 255, 255)) if dark else ((255, 255, 255), (0, 0, 0))
    draw.text((pad - bbox[0], pad - bbox[1]), label, fill=ink, font=font,
              stroke_width=max(1, th // 12), stroke_fill=edge)
    return img


def decorate(rgb: np.ndarray, label: str | None = None, scalebar: ScaleBar | None = None,
             upscale: float = 1) -> np.ndarray:
    """Ajoute à une image RGB le temps (en haut à gauche) et/ou la barre d'échelle.
    `upscale` : px de l'image par px du dataset."""
    if scalebar:
        rgb = sbmod.draw_on_rgb(rgb, scalebar, upscale)
    if label:
        rgb = np.asarray(_stamp(Image.fromarray(rgb), label, sbmod.font(max(12, rgb.shape[1] // 40), bold=True)))
    return rgb


def sequence_video_frames(stack: np.ndarray, times: np.ndarray, time_unit: str,
                          show_time: bool = True, scalebar: ScaleBar | None = None,
                          rng: tuple[float, float] | None = None) -> list[np.ndarray]:
    """Une image RGB par frame brute (recadrée), avec temps et barre d'échelle optionnels.
    rng : niveaux (noir, blanc) communs à toutes les frames (io.display_range) ; None = chaque
    frame étirée séparément."""
    frames = []
    for i in range(stack.shape[0]):
        rgb = np.asarray(Image.fromarray(iom.to_display(stack[i], rng=rng)).convert("RGB"))
        label = f"t = {times[i]:.4g} {time_unit}   (frame {i})" if show_time else None
        frames.append(decorate(rgb, label, scalebar))
    return frames


def heatmap_rgb(m: np.ndarray, vmax: float, cmap: str = "jet", upscale: int = 1) -> np.ndarray:
    """Carte (H, W) -> image RGB uint8 avec la palette `cmap`, échelle fixe 0 – vmax."""
    norm = np.clip(m / max(vmax, 1e-12), 0.0, 1.0)
    rgb = (mpl.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    if upscale > 1:
        rgb = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return rgb


def heatmap_video_bytes(base: BaseMap, n: int, trail: int, sigma: float, vmax: float,
                        cmap: str = "jet", times: np.ndarray | None = None, time_unit: str = "",
                        show_time: bool = False, fps: float = 10.0, fmt: str = "mp4",
                        min_side: int = 480, progress: ProgressFn | None = None,
                        scalebar: ScaleBar | None = None) -> bytes:
    """Vidéo de la heatmap pleine résolution à traînées (une image par frame du dataset).

    Le MP4 est agrandi (pixels dupliqués) jusqu'à `min_side` px sur le plus petit côté, pour
    rester net dans les lecteurs ; le GIF reste à la résolution native pour limiter sa taille.
    """
    frames = []
    for i, m in enumerate(densitymod.accumulated_maps(base, n, trail, sigma)):
        up = max(1, -(-min_side // min(m.shape))) if fmt == "mp4" else 1
        label = f"t = {times[i]:.4g} {time_unit}   (frame {i})" if show_time and times is not None else None
        frames.append(decorate(heatmap_rgb(m, vmax, cmap, up), label, scalebar, up))
        if progress:
            progress(0.85 * (i + 1) / n, f"Image {i + 1}/{n}")
    if progress:
        progress(0.9, "Encodage vidéo…")
    data = _write_frames(frames, fps, fmt)
    if progress:
        progress(1.0, "Terminé")
    return data


def sequence_video_bytes(stack: np.ndarray, times: np.ndarray, time_unit: str,
                         fps: float = 8.0, fmt: str = "mp4", show_time: bool = True,
                         progress: ProgressFn | None = None, scalebar: ScaleBar | None = None,
                         rng: tuple[float, float] | None = None) -> bytes:
    """Vidéo de lecture de la séquence brute (dataset dérivé), pour vérifier les données."""
    if progress:
        progress(0.05, "Préparation des images…")
    frames = sequence_video_frames(stack, times, time_unit, show_time, scalebar, rng)
    if progress:
        progress(0.7, "Encodage vidéo…")
    data = _write_frames(frames, fps, fmt)
    if progress:
        progress(1.0, "Terminé")
    return data
