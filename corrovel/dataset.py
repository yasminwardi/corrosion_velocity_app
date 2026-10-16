"""Sélection des frames, recadrage ROI et datasets dérivés enregistrés sur disque.

Structure d'un dataset dérivé :

    data/datasets/<nom>/
        metadata.json        paramètres de génération, indices et temps des frames
        frames/frame_00000.tif ...   frames recadrées, dtype d'origine conservé
        results/<run>/       résultats d'analyse (créés à l'export)
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile

from .io import ImageSource, natural_key, safe_name, to_gray


@dataclass(frozen=True)
class ROI:
    """Rectangle en pixels, bornes de fin exclues : image[y0:y1, x0:x1]."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def validate(self, img_w: int, img_h: int) -> "ROI":
        if not (0 <= self.x0 < self.x1 <= img_w and 0 <= self.y0 < self.y1 <= img_h):
            raise ValueError(f"ROI {self} invalide pour une image {img_w}x{img_h}.")
        return self

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "width": self.width, "height": self.height}


def square_roi(cx: float, cy: float, side: int, img_w: int, img_h: int) -> ROI:
    """Carré de `side` px centré sur (cx, cy), décalé si besoin pour rester dans l'image.

    Le côté est ramené à la plus petite dimension de l'image si nécessaire. Près d'un bord, le
    carré est décalé : le point n'est alors plus exactement au centre (voir offset côté app).
    """
    side = int(max(2, min(int(side), img_w, img_h)))
    x0 = int(round(cx - side / 2 + 0.5))
    y0 = int(round(cy - side / 2 + 0.5))
    x0 = min(max(x0, 0), img_w - side)
    y0 = min(max(y0, 0), img_h - side)
    return ROI(x0, y0, x0 + side, y0 + side)


def crop(arr: np.ndarray, roi: ROI) -> np.ndarray:
    return np.ascontiguousarray(arr[roi.y0:roi.y1, roi.x0:roi.x1, ...])


def select_indices(n_total: int, start: int, end: int, step: int = 1,
                   n_target: int | None = None) -> list[int]:
    """Indices des frames à garder (start et end inclus).

    Si n_target est donné, il remplace step : n_target frames réparties
    le plus régulièrement possible entre start et end.
    """
    if n_total <= 0:
        raise ValueError("Séquence vide.")
    if not (0 <= start <= end < n_total):
        raise ValueError(f"Il faut 0 ≤ début ≤ fin ≤ {n_total - 1} (reçu début={start}, fin={end}).")
    if n_target:
        n_avail = end - start + 1
        if n_target > n_avail:
            raise ValueError(f"{n_target} frames demandées mais seulement {n_avail} entre {start} et {end}.")
        if n_target == 1:
            return [start]
        idx = np.round(np.linspace(start, end, n_target)).astype(int)
        return [int(i) for i in idx]
    if step < 1:
        raise ValueError("Le pas doit être ≥ 1.")
    return list(range(start, end + 1, step))


def create_dataset(source: ImageSource, root: Path, name: str, roi: ROI, indices: list[int],
                   dt: float, time_unit: str, pixel_size: float | None,
                   overwrite: bool = False,
                   progress: Callable[[float, str], None] | None = None,
                   pit: tuple[float, float] | None = None, calibration_confirmed: bool = False) -> Path:
    """Recadre et copie les frames choisies dans root/datasets/<nom>/.

    pit : position du pit (x, y) en px de l'image d'origine, enregistrée dans les métadonnées
    (convertie dans les coordonnées du dataset) pour pré-placer le pit dans l'analyse.
    """
    roi.validate(source.width, source.height)
    if len(indices) < 2:
        raise ValueError("Il faut au moins 2 frames pour calculer des vitesses.")
    if dt <= 0:
        raise ValueError("L'intervalle de temps entre frames doit être > 0.")
    if pixel_size is not None and pixel_size <= 0:
        raise ValueError("La taille de pixel doit être > 0.")

    folder = Path(root) / "datasets" / safe_name(name)
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"Le dataset « {folder.name} » existe déjà.")
        shutil.rmtree(folder)
    frames_dir = folder / "frames"
    frames_dir.mkdir(parents=True)

    frame_files = []
    try:
        for k, idx in enumerate(indices):
            arr = crop(source.get(idx), roi)
            fname = f"frame_{k:05d}.tif"
            tifffile.imwrite(frames_dir / fname, arr,
                             photometric="rgb" if arr.ndim == 3 else "minisblack")
            frame_files.append(fname)
            if progress:
                progress((k + 1) / len(indices), f"Frame {k + 1}/{len(indices)}")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    meta = {
        "name": folder.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source.describe(),
        "roi": roi.to_dict(),
        "original_indices": [int(i) for i in indices],
        "frame_files": frame_files,
        "dt_original": float(dt),
        "time_unit": time_unit,
        "times": [float(i * dt) for i in indices],
        "pixel_size": None if pixel_size is None else float(pixel_size),
        "length_unit": "µm" if pixel_size is not None else "px",
        "calibration": {"confirmed": bool(calibration_confirmed and pixel_size is not None),
                        "updated": datetime.now().isoformat(timespec="seconds"), "history": []},
    }
    if pit is not None:
        meta["pit_source_px"] = [float(pit[0]), float(pit[1])]
        meta["pit_px"] = [float(pit[0]) - roi.x0, float(pit[1]) - roi.y0]
    # La liste complète des fichiers source peut être longue : on ne garde que les frames utilisées
    if "files" in meta["source"]:
        meta["source"]["files_used"] = [meta["source"]["files"][i] for i in indices]
        del meta["source"]["files"]
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder


class DerivedDataset:
    """Dataset dérivé relu depuis le disque."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        meta_path = self.folder / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json absent dans {self.folder}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.frame_paths = [self.folder / "frames" / f for f in self.meta["frame_files"]]
        missing = [p.name for p in self.frame_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Frames manquantes : {missing[:5]}")
        self.times = np.asarray(self.meta["times"], dtype=float)
        self.original_indices = list(self.meta["original_indices"])
        self.pixel_size = self.meta.get("pixel_size")
        self.length_unit = self.meta.get("length_unit", "px")
        self.time_unit = self.meta.get("time_unit", "s")

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def pit(self) -> tuple[float, float] | None:
        """Pit choisi à la création du dataset (x, y en px du dataset), s'il y en a un."""
        p = self.meta.get("pit_px")
        return (float(p[0]), float(p[1])) if p else None

    @property
    def scale(self) -> float:
        """Facteur px -> unité de longueur physique."""
        return float(self.pixel_size) if self.pixel_size else 1.0

    # ----------------------------------------------------------------- calibration
    # La calibration (µm/px, intervalle entre frames) est la seule source des unités physiques de
    # toute l'analyse. Elle peut être corrigée après coup sans régénérer les frames ; chaque
    # changement est historisé dans metadata.json. « confirmed » = vérifiée par l'utilisateur
    # (valeurs issues de l'acquisition), par opposition à une valeur par défaut.

    @property
    def calibration_confirmed(self) -> bool:
        return bool(self.meta.get("calibration", {}).get("confirmed", False))

    def calibration_issues(self) -> list[str]:
        """Problèmes de calibration qui empêchent des vitesses fiables ou comparables."""
        issues = []
        if not self.pixel_size:
            issues.append("taille de pixel inconnue : longueurs et vitesses en px, non comparables à d'autres datasets")
        dt = float(self.meta.get("dt_original", 1.0))
        if not self.calibration_confirmed:
            if dt == 1.0:
                issues.append("intervalle entre frames = 1 (valeur par défaut de l'application) : à vérifier, "
                              "une erreur fausse toutes les vitesses du même facteur")
            else:
                issues.append("calibration non confirmée (µm/px et intervalle entre frames)")
        return issues

    def set_calibration(self, pixel_size: float | None, dt: float, time_unit: str, confirmed: bool,
                        note: str = "") -> None:
        """Corrige la calibration (frames inchangées) : temps recalculés à partir des indices
        d'origine, ancienne valeur gardée dans l'historique."""
        if dt <= 0:
            raise ValueError("L'intervalle de temps entre frames doit être > 0.")
        if pixel_size is not None and pixel_size <= 0:
            raise ValueError("La taille de pixel doit être > 0.")
        m = self.meta
        cal = m.get("calibration", {})
        history = cal.get("history", [])
        history.append({"pixel_size": m.get("pixel_size"), "dt_original": m.get("dt_original"),
                        "time_unit": m.get("time_unit"), "confirmed": cal.get("confirmed", False),
                        "replaced": datetime.now().isoformat(timespec="seconds")})
        m["pixel_size"] = None if pixel_size is None else float(pixel_size)
        m["length_unit"] = "µm" if pixel_size is not None else "px"
        m["dt_original"] = float(dt)
        m["time_unit"] = time_unit
        m["times"] = [float(i * dt) for i in m["original_indices"]]
        m["calibration"] = {"confirmed": bool(confirmed), "note": note,
                            "updated": datetime.now().isoformat(timespec="seconds"), "history": history}
        (self.folder / "metadata.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        self.__init__(self.folder)

    def __len__(self) -> int:
        return len(self.frame_paths)

    def get(self, i: int) -> np.ndarray:
        return tifffile.imread(self.frame_paths[i])

    def gray_stack(self) -> np.ndarray:
        """Toutes les frames en niveaux de gris float32, forme (N, H, W).

        Lecture en parallèle (threads) : la lecture TIFF est limitée par le disque, pas le CPU,
        et chaque thread écrit dans sa propre tranche de `out`, donc pas de conflit d'écriture.
        """
        n = len(self)
        first = to_gray(self.get(0))
        out = np.empty((n,) + first.shape, dtype=np.float32)
        out[0] = first
        if n > 1:
            workers = min(32, os.cpu_count() or 4, n - 1)

            def _load(i: int) -> None:
                out[i] = to_gray(self.get(i))

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_load, range(1, n)))
        return out

    def mtime(self) -> float:
        return (self.folder / "metadata.json").stat().st_mtime


def list_datasets(root: Path) -> list[Path]:
    base = Path(root) / "datasets"
    if not base.exists():
        return []
    return sorted((p for p in base.iterdir() if (p / "metadata.json").exists()),
                  key=lambda p: natural_key(p.name))
