"""Lecture des images et des séquences (dossier d'images ou stack TIFF multi-pages)."""
from __future__ import annotations

import re
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIFF_EXTS = {".tif", ".tiff"}


def natural_key(s) -> list:
    """Tri naturel : frame_2 avant frame_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or "dataset"


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Remet une image couleur planaire (C,H,W) en (H,W,C)."""
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr)


def read_image(path, page: int = 0) -> np.ndarray:
    """Lit une image en conservant son dtype (uint8, uint16, float...)."""
    path = Path(path)
    if path.suffix.lower() in TIFF_EXTS:
        arr = tifffile.imread(path, key=page)
    else:
        with Image.open(path) as im:
            if im.mode in ("P", "PA", "CMYK", "YCbCr", "LAB", "HSV", "RGBA"):
                im = im.convert("RGB")
            elif im.mode == "LA":
                im = im.convert("L")
            arr = np.array(im)
    arr = _to_hwc(np.asarray(arr))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"{path.name} : forme d'image non gérée {arr.shape}")
    return arr


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Convertit en niveaux de gris float32 sans renormaliser les valeurs."""
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[-1] in (3, 4):
            w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            a = a[..., :3].astype(np.float32) @ w
        else:
            raise ValueError(f"Forme d'image non gérée : {a.shape}")
    elif a.ndim != 2:
        raise ValueError(f"Forme d'image non gérée : {a.shape}")
    return a.astype(np.float32, copy=False)


DISPLAY_MODES = {
    "real": "réelles (valeurs brutes)",
    "fixed": "contraste fixe pour toute la séquence",
    "auto": "contraste auto, image par image",
}


def display_range(mode: str, dtype, sample: np.ndarray) -> tuple[float, float] | None:
    """Plage de niveaux affichée en noir → blanc, la même pour toutes les images d'une séquence.

    - "real" : valeurs enregistrées telles quelles, 0 = noir, maximum du capteur = blanc. Pour les
      entiers, profondeur effective 8/10/12/14/16 bits (la plus petite qui contient le maximum de
      `sample`) : une caméra 12 bits enregistrée en 16 bits n'apparaît pas presque noire ;
    - "fixed" : étirement linéaire unique (0.5e – 99.9e percentile de `sample`, qui doit couvrir
      toute la séquence), appliqué à l'identique à chaque image : les variations de luminosité
      entre images (photoblanchiment, pit qui assombrit) restent visibles ;
    - "auto" : None, chaque image est étirée séparément (voir to_display).
    """
    if mode == "auto":
        return None
    s = np.asarray(sample, dtype=np.float64)
    s = s[np.isfinite(s)]
    dtype = np.dtype(dtype)
    if mode == "real":
        if np.issubdtype(dtype, np.unsignedinteger):
            mx = float(s.max()) if s.size else 0.0
            for bits in (8, 10, 12, 14, 16, 32, 64):
                if bits <= dtype.itemsize * 8 and mx <= 2 ** bits - 1:
                    return 0.0, float(2 ** bits - 1)
            return 0.0, float(np.iinfo(dtype).max)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        if s.size and s.min() >= 0 and s.max() <= 1:
            return 0.0, 1.0
        return (float(s.min()), float(s.max())) if s.size else (0.0, 1.0)
    if mode == "fixed":
        if not s.size:
            return 0.0, 1.0
        lo, hi = np.percentile(s, [0.5, 99.9])
        return float(lo), float(hi if hi > lo else lo + 1.0)
    raise ValueError(f"Mode d'affichage inconnu : {mode}")


def to_display(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5,
               rng: tuple[float, float] | None = None) -> np.ndarray:
    """Image uint8 pour l'affichage.

    rng = (noir, blanc) : correspondance linéaire fixe (voir display_range), sans rien adapter à
    l'image. Sans rng : étirement de contraste propre à l'image (percentiles), qui change la
    luminosité et le contraste apparents d'une image à l'autre.
    """
    g = to_gray(arr)
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
    else:
        finite = g[np.isfinite(g)]
        if finite.size == 0:
            return np.zeros(g.shape, np.uint8)
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((g - lo) / (hi - lo), 0, 1)
    return np.round(np.nan_to_num(out) * 255).astype(np.uint8)


def _tiff_page_count(path: Path) -> int:
    with tifffile.TiffFile(path) as tf:
        return len(tf.pages)


class ImageSource:
    """Séquence d'images d'origine : liste de fichiers ou stack TIFF multi-pages.

    Les frames sont lues à la demande avec un petit cache LRU.
    """

    def __init__(self, name: str, files: list[Path] | None = None, stack: Path | None = None,
                 cache_size: int = 64):
        if (files is None) == (stack is None):
            raise ValueError("Donner soit une liste de fichiers, soit un stack.")
        self.name = name
        self.files = [Path(f) for f in files] if files is not None else None
        self.stack = Path(stack) if stack is not None else None
        self._n = len(self.files) if self.files is not None else _tiff_page_count(self.stack)
        if self._n == 0:
            raise ValueError("Aucune image trouvée.")
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self.shape = None
        first = self.get(0)
        self.shape = first.shape
        self.dtype = first.dtype
        # Vérification rapide de cohérence sur la dernière frame
        self.get(self._n - 1)

    @property
    def kind(self) -> str:
        return "files" if self.files is not None else "stack"

    @property
    def height(self) -> int:
        return int(self.shape[0])

    @property
    def width(self) -> int:
        return int(self.shape[1])

    def __len__(self) -> int:
        return self._n

    def get(self, i: int) -> np.ndarray:
        i = int(i)
        if not 0 <= i < self._n:
            raise IndexError(f"Frame {i} hors limites (0..{self._n - 1}).")
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        arr = read_image(self.files[i]) if self.files is not None else read_image(self.stack, page=i)
        if self.shape is not None and arr.shape[:2] != self.shape[:2]:
            raise ValueError(
                f"Frame {i} ({self.label(i)}) : taille {arr.shape[:2]} différente "
                f"de la frame 0 {self.shape[:2]}.")
        self._cache[i] = arr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def label(self, i: int) -> str:
        return self.files[i].name if self.files is not None else f"{self.stack.name} [page {i}]"

    def describe(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "n_frames": self._n,
             "height": self.height, "width": self.width, "dtype": str(self.dtype),
             "channels": 1 if len(self.shape) == 2 else int(self.shape[2])}
        if self.files is not None:
            d["folder"] = str(self.files[0].parent)
            d["files"] = [f.name for f in self.files]
        else:
            d["stack"] = str(self.stack)
        return d


def source_from_folder(folder, name: str | None = None) -> ImageSource:
    """Construit une source à partir d'un dossier (ou d'un TIFF multi-pages).

    - plusieurs images : une image = une frame (tri naturel des noms) ;
    - un seul TIFF multi-pages : chaque page = une frame.
    """
    folder = Path(str(folder).strip().strip('"').strip("'")).expanduser()
    if folder.is_file():
        if folder.suffix.lower() in TIFF_EXTS:
            return ImageSource(name or folder.stem, stack=folder)
        raise ValueError("Un fichier unique doit être un TIFF multi-pages.")
    if not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    files = sorted((p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")),
                   key=lambda p: natural_key(p.name))
    if not files:
        raise ValueError(f"Aucune image ({', '.join(sorted(IMAGE_EXTS))}) dans {folder}")
    if len(files) == 1 and files[0].suffix.lower() in TIFF_EXTS and _tiff_page_count(files[0]) > 1:
        return ImageSource(name or folder.name, stack=files[0])
    return ImageSource(name or folder.name, files=files)


def save_uploads(uploaded: list[tuple[str, bytes]], dest: Path) -> Path:
    """Enregistre des fichiers uploadés (images ou .zip) dans dest, à plat.

    Les .zip sont extraits en ne gardant que le nom de fichier (pas de chemins).
    """
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, (fname, data) in enumerate(uploaded):
        fname = Path(fname).name
        if fname.lower().endswith(".zip"):
            tmp = dest / f".__upload_{k}.zip"
            tmp.write_bytes(data)
            try:
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        base = Path(info.filename).name
                        if (info.is_dir() or not base or base.startswith(".")
                                or "__MACOSX" in info.filename
                                or Path(base).suffix.lower() not in IMAGE_EXTS):
                            continue
                        target = dest / base
                        if target.exists():
                            raise ValueError(f"Nom de fichier en double dans le zip : {base}")
                        with zf.open(info) as src, open(target, "wb") as out:
                            shutil.copyfileobj(src, out)
                        n += 1
            finally:
                tmp.unlink(missing_ok=True)
        elif Path(fname).suffix.lower() in IMAGE_EXTS:
            (dest / fname).write_bytes(data)
            n += 1
    if n == 0:
        raise ValueError("Aucune image valide dans les fichiers envoyés.")
    return dest
