"""Densité de particules (heatmap) par fenêtre temporelle, à partir des trajectoires Trackpy.

densité = nombre moyen de particules présentes par frame dans une cellule / aire de la cellule.
Une particule est comptée dans chaque frame où elle est détectée, puis on moyenne sur les frames
i0..i1 de la fenêtre : une fenêtre plus longue ne gonfle donc pas la densité. Les fenêtres sont
celles du champ de vitesse (FieldResult.windows), pour que les deux cartes restent comparables.
Seules les particules des trajectoires retenues (après filtre de longueur min) sont comptées.

Le module fournit aussi une heatmap pleine résolution (au pixel), à traînées, voir plus bas.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .field import FieldResult, Grid


@dataclass
class DensityResult:
    windows: list[tuple[int, int]]
    t_start: np.ndarray           # (nt,)
    t_end: np.ndarray
    mean_count: np.ndarray        # (nt, ny, nx) particules / frame dans la cellule (non lissé)
    density: np.ndarray           # (nt, ny, nx) particules / length_unit², lissé si sigma > 0
    bin_px: int
    sigma: float                  # lissage gaussien, en cellules
    scale: float
    length_unit: str
    time_unit: str
    image_shape: tuple[int, int]  # (H, W)
    pits: list[tuple[float, float]]

    @property
    def grid(self) -> Grid:
        return Grid(self.image_shape[0], self.image_shape[1], self.bin_px)

    @property
    def x(self) -> np.ndarray:
        return self.grid.x

    @property
    def y(self) -> np.ndarray:
        return self.grid.y

    @property
    def x_edges(self) -> np.ndarray:
        """Bords des cellules en px (la dernière cellule peut être tronquée par l'image)."""
        g = self.grid
        return np.minimum(np.arange(g.nx + 1) * g.cell, g.width) - 0.5

    @property
    def y_edges(self) -> np.ndarray:
        g = self.grid
        return np.minimum(np.arange(g.ny + 1) * g.cell, g.height) - 0.5

    @property
    def density_unit(self) -> str:
        return f"particules/{self.length_unit}²"

    def to_npz_dict(self) -> dict:
        return {
            "x_px": self.x, "y_px": self.y, "x_edges_px": self.x_edges, "y_edges_px": self.y_edges,
            "scale": self.scale, "t_start": self.t_start, "t_end": self.t_end,
            "windows": np.asarray(self.windows), "density": self.density,
            "mean_count": self.mean_count, "bin_px": self.bin_px, "sigma_cells": self.sigma,
            "image_shape": np.asarray(self.image_shape), "pits_px": np.asarray(self.pits),
            "length_unit": self.length_unit, "time_unit": self.time_unit,
        }

    def to_long_dataframe(self) -> pd.DataFrame:
        X, Y = np.meshgrid(self.x, self.y)
        rows = []
        for k in range(len(self.windows)):
            rows.append(pd.DataFrame({
                "window": k, "t_start": self.t_start[k], "t_end": self.t_end[k],
                "t_center": 0.5 * (self.t_start[k] + self.t_end[k]),
                "x_px": X.ravel(), "y_px": Y.ravel(),
                f"x_{self.length_unit}": X.ravel() * self.scale,
                f"y_{self.length_unit}": Y.ravel() * self.scale,
                "particles_per_frame": self.mean_count[k].ravel(),
                "density": self.density[k].ravel(),
            }))
        return pd.concat(rows, ignore_index=True)


def particle_density(tracks: pd.DataFrame, res: FieldResult, bin_px: int | None = None,
                     sigma: float = 0.0) -> DensityResult:
    """Carte de densité de particules par fenêtre de `res`, sur une grille de `bin_px` px."""
    h, w = res.image_shape
    grid = Grid(h, w, int(bin_px or res.cell))
    # Aire réelle de chaque cellule : les cellules du bord droit / bas peuvent être tronquées
    wx = np.diff(np.minimum(np.arange(grid.nx + 1) * grid.cell, w))
    wy = np.diff(np.minimum(np.arange(grid.ny + 1) * grid.cell, h))
    area = np.outer(wy, wx) * res.scale ** 2

    nt = len(res.windows)
    mean_count = np.zeros((nt, grid.ny, grid.nx))
    if len(tracks):
        frames = tracks["frame"].to_numpy().astype(int)
        ix_all, iy_all = grid.index(tracks["x"].to_numpy(), tracks["y"].to_numpy())
    else:
        frames = ix_all = iy_all = np.array([], int)
    for k, (i0, i1) in enumerate(res.windows):
        sel = (frames >= i0) & (frames <= i1)
        np.add.at(mean_count[k], (iy_all[sel], ix_all[sel]), 1.0)
        mean_count[k] /= (i1 - i0 + 1)
    density = mean_count / area
    if sigma > 0:
        density = np.stack([gaussian_filter(d, sigma) for d in density])
    return DensityResult(list(res.windows), res.t_start, res.t_end, mean_count, density,
                         grid.cell, float(sigma), res.scale, res.length_unit, res.time_unit,
                         (h, w), list(res.pits))


# --------------------------------------------------------------------------- heatmap pleine résolution
# Rendu « traînées » : pour chaque frame i, on somme une carte de base sur les `trail` dernières
# frames (0 = cumul depuis le début), puis on lisse au pixel près. La carte de base est soit le
# signal des particules extrait de l'image (particle_signal_stack), soit l'histogramme des
# positions Trackpy de la frame. Les particules en mouvement laissent ainsi des traînées.

BaseMap = Callable[[int], np.ndarray]


def guess_polarity(frame: np.ndarray) -> str:
    """'dark' si les objets s'écartent surtout vers le sombre (fond clair), sinon 'bright'."""
    lo, med, hi = np.percentile(frame, [0.1, 50, 99.9])
    return "dark" if med - lo > hi - med else "bright"


def particle_signal_stack(frames: np.ndarray, polarity: str = "dark", background: str = "rolling",
                          threshold: float = 3.0, binary: bool = True,
                          bg_offsets: tuple[int, ...] = (4, 8, 12)) -> np.ndarray:
    """Signal des particules frame par frame, (N, H, W), pour la heatmap pleine résolution.

    1. chaque frame est ramenée à la même luminosité moyenne (compense le photoblanchiment) ;
    2. fond soustrait : « rolling » = médiane des frames voisines (i ± bg_offsets), qui retire
       aussi ce qui évolue lentement (pit qui grossit) et ne garde que ce qui bouge ;
       « global » = médiane de toute la séquence ;
    3. seuil à `threshold` × bruit (écart-type robuste, MAD) de la frame.
    binary=True : 1 là où une particule est présente (chaque particule compte pareil, quel que
    soit son contraste) ; False : contraste au-dessus du seuil.
    """
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity doit être 'bright' ou 'dark'.")
    f = frames.astype(np.float32)                # copie : normalisée sur place ensuite
    n = len(f)
    means = f.reshape(n, -1).mean(axis=1)
    f *= (float(means.mean()) / np.where(means > 0, means, 1.0))[:, None, None]
    offs = np.array([-o for o in bg_offsets] + list(bg_offsets))
    glob = None
    if background == "global":
        glob = np.median(f if n <= 200 else f[np.linspace(0, n - 1, 200).astype(int)], axis=0)
    elif background != "rolling":
        raise ValueError("background doit être 'rolling' ou 'global'.")
    out = np.empty(f.shape, bool if binary else np.float16)
    for i in range(n):
        if glob is None:
            j = np.unique(np.clip(i + offs, 0, n - 1))
            j = j[j != i] if n > 1 else j
            bg = np.median(f[j], axis=0)
        else:
            bg = glob
        d = bg - f[i] if polarity == "dark" else f[i] - bg
        noise = 1.4826 * float(np.median(np.abs(d - np.median(d))))
        thr = threshold * noise
        out[i] = d > thr if binary else np.clip(d - thr, 0, None)
    return out


def track_frame_maps(tracks: pd.DataFrame, shape: tuple[int, int]) -> BaseMap:
    """Carte de base « positions Trackpy » : nb de particules par pixel dans la frame i."""
    h, w = shape
    if len(tracks):
        f = tracks["frame"].to_numpy().astype(int)
        order = np.argsort(f, kind="stable")
        f = f[order]
        xi = np.clip(np.rint(tracks["x"].to_numpy()[order]), 0, w - 1).astype(int)
        yi = np.clip(np.rint(tracks["y"].to_numpy()[order]), 0, h - 1).astype(int)
    else:
        f = xi = yi = np.array([], int)

    def base(i: int) -> np.ndarray:
        m = np.zeros((h, w), np.float32)
        a, b = np.searchsorted(f, i, "left"), np.searchsorted(f, i, "right")
        np.add.at(m, (yi[a:b], xi[a:b]), 1.0)
        return m
    return base


def _first_frame(i: int, trail: int) -> int:
    return 0 if trail <= 0 else max(0, i - trail + 1)


def accumulated_map(base: BaseMap, i: int, trail: int, sigma: float) -> np.ndarray:
    """Carte de la frame i : somme de base(j) pour j dans la traînée, lissée (sigma en px)."""
    acc = np.zeros_like(base(i), dtype=np.float64)
    for j in range(_first_frame(i, trail), i + 1):
        acc += base(j)
    return gaussian_filter(acc, sigma) if sigma > 0 else acc


def accumulated_maps(base: BaseMap, n: int, trail: int, sigma: float) -> Iterator[np.ndarray]:
    """Mêmes cartes que accumulated_map pour i = 0..n-1, par somme glissante (une seule lecture)."""
    acc = None
    hist: deque = deque()
    for i in range(n):
        p = base(i).astype(np.float64)
        acc = p.copy() if acc is None else acc + p
        if trail > 0:
            hist.append(p)
            if len(hist) > trail:
                acc -= hist.popleft()
        yield gaussian_filter(acc, sigma) if sigma > 0 else acc.copy()


def heatmap_vmax(base: BaseMap, n: int, trail: int, sigma: float, pct: float = 99.5,
                 n_samples: int = 24) -> float:
    """Échelle de couleur fixe pour toute la vidéo : médiane, sur des frames échantillonnées, du
    percentile `pct` de chaque carte (au-delà, la couleur sature). La médiane (et non le max)
    évite qu'une frame aberrante écrase toutes les autres. En cumul (trail = 0), on prend la
    dernière frame, la plus remplie."""
    idx = [n - 1] if trail <= 0 else np.unique(np.linspace(0, n - 1, min(n, n_samples)).astype(int))
    vals = []
    for i in idx:
        m = accumulated_map(base, int(i), trail, sigma)
        v = float(np.percentile(m, pct))
        vals.append(v if v > 0 else float(m.max()))
    return float(np.median(vals)) or 1.0


def global_vmax(dens: DensityResult, pct: float = 99.0) -> float:
    """Échelle de couleur commune à toutes les fenêtres (percentile des cellules non vides)."""
    d = dens.density[dens.density > 0]
    if d.size == 0:
        return 1.0
    v = float(np.percentile(d, pct))
    return v if v > 0 else float(d.max())
