"""Suivi de particules avec Trackpy et calcul des déplacements élémentaires."""
from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
import trackpy as tp

tp.quiet()

# Seuil au-delà duquel locate_all bascule en multi-process : en-dessous, le coût de démarrage
# d'un pool de process dépasse le gain. Les deux tableaux suivants sont des globales de module,
# peuplées juste avant la création du pool : sous Linux, ProcessPoolExecutor forke, et les
# processus enfants héritent de ces tableaux déjà en mémoire (copy-on-write), sans avoir à les
# repickler pour chacune des N tâches — seul un entier (l'indice de frame) traverse l'IPC.
_PARALLEL_MIN_FRAMES = 8
_shared_feat: np.ndarray | None = None
_shared_params: "TrackpyParams | None" = None


@dataclass
class TrackpyParams:
    polarity: str = "bright"        # "bright" : particules claires ; "dark" : particules sombres
    background: str = "median"      # "none" | "median" | "mean" (fond statique soustrait)
    diameter: int = 7               # impair, en px
    minmass: float = 100.0          # dans l'échelle normalisée 0-255 de l'image de détection
    separation: float | None = None
    percentile: float = 64.0
    search_range: float = 5.0       # déplacement max entre deux frames consécutives, en px
    memory: int = 0
    min_length: int = 3             # longueur minimale des trajectoires (frames)
    subtract_drift: bool = False
    predictive: bool = False        # liaison prédictive : prévoit la position d'après la vitesse
                                    # des voisins (utile si les traceurs sont éjectés très vite)

    def to_dict(self) -> dict:
        return asdict(self)


def feature_stack(frames: np.ndarray, polarity: str = "bright", background: str = "median",
                  max_bg_frames: int = 200) -> np.ndarray:
    """Image de détection : particules claires sur fond ~0, échelle 0-255 commune à tout le stack.

    La normalisation est globale (même facteur pour toutes les frames), donc minmass
    a le même sens d'une frame à l'autre.
    """
    if frames.ndim != 3:
        raise ValueError("frames doit être de forme (N, H, W).")
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity doit être 'bright' ou 'dark'.")
    frames = frames.astype(np.float32, copy=False)
    if background == "none":
        d = frames - frames.min() if polarity == "bright" else frames.max() - frames
    elif background in ("median", "mean"):
        n = frames.shape[0]
        sub = frames if n <= max_bg_frames else frames[np.linspace(0, n - 1, max_bg_frames).astype(int)]
        bg = np.median(sub, axis=0) if background == "median" else sub.mean(axis=0)
        d = frames - bg[None] if polarity == "bright" else bg[None] - frames
    else:
        raise ValueError("background doit être 'none', 'median' ou 'mean'.")
    d = np.clip(d, 0, None)
    scale = float(np.percentile(d, 99.99))
    if scale <= 0:
        scale = float(d.max()) or 1.0
    return np.clip(d * (255.0 / scale), 0, 255).astype(np.float32)


def locate_frame(img: np.ndarray, p: TrackpyParams) -> pd.DataFrame:
    if p.diameter < 3 or p.diameter % 2 == 0:
        raise ValueError("Le diamètre Trackpy doit être un entier impair ≥ 3.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = tp.locate(img, p.diameter, minmass=p.minmass, separation=p.separation,
                      percentile=p.percentile, invert=False)
    return f


def _locate_index(i: int) -> pd.DataFrame | None:
    """Fonction de travail d'un process du pool : relit _shared_feat/_shared_params hérités
    par fork (copy-on-write), pas transmis via l'IPC."""
    f = locate_frame(_shared_feat[i], _shared_params)
    if len(f):
        f = f.copy()
        f["frame"] = i
        return f
    return None


def _locate_all_sequential(feat: np.ndarray, p: TrackpyParams,
                           progress: Callable[[float, str], None] | None = None) -> list:
    n = feat.shape[0]
    parts = []
    for i in range(n):
        f = locate_frame(feat[i], p)
        if len(f):
            f = f.copy()
            f["frame"] = i
            parts.append(f)
        if progress:
            progress((i + 1) / n, f"Détection frame {i + 1}/{n}")
    return parts


def locate_all(feat: np.ndarray, p: TrackpyParams,
               progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    global _shared_feat, _shared_params
    n = feat.shape[0]
    cols = ["y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep", "frame"]
    parts: list = []
    workers = min(os.cpu_count() or 1, n)
    # Le partage par globales repose sur fork : sous Windows / macOS (spawn), les workers
    # repartiraient de globales vides et chaque tâche échouerait — on reste alors séquentiel.
    if n >= _PARALLEL_MIN_FRAMES and workers > 1 and mp.get_start_method() == "fork":
        try:
            _shared_feat, _shared_params = feat, p   # en mémoire avant le fork des workers
            results: list = [None] * n
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                for i, res in zip(range(n), ex.map(_locate_index, range(n))):
                    results[i] = res
                    done += 1
                    if progress:
                        progress(done / n, f"Détection frame {done}/{n}")
            parts = [r for r in results if r is not None]
        except Exception:
            parts = _locate_all_sequential(feat, p, progress)
        finally:
            _shared_feat = _shared_params = None
    else:
        parts = _locate_all_sequential(feat, p, progress)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def link_tracks(features: pd.DataFrame, p: TrackpyParams) -> pd.DataFrame:
    """Relie les détections en trajectoires et filtre les trajectoires courtes.

    Liaison prédictive (p.predictive) : la position attendue de chaque particule à la frame
    suivante est extrapolée à partir de la vitesse de ses voisines (NearestVelocityPredict) ; on
    cherche alors autour de cette position plutôt qu'autour de la position actuelle. Indispensable
    quand les traceurs sont éjectés vite du pit : sans prédiction, un déplacement proche de
    search_range casse les trajectoires. La recherche adaptative réduit search_range localement
    dans les zones denses, au lieu d'échouer (SubnetOversizeException).
    """
    if features.empty:
        return features.assign(particle=pd.Series(dtype=int))
    kw = dict(memory=p.memory, adaptive_stop=max(0.5, 0.2 * p.search_range), adaptive_step=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if p.predictive:
            t = tp.predict.NearestVelocityPredict().link_df(features, p.search_range, **kw)
        else:
            t = tp.link(features, p.search_range, **kw)
        if p.min_length > 1:
            t = tp.filter_stubs(t, p.min_length)
        t = t.reset_index(drop=True)
        if p.subtract_drift and not t.empty:
            drift = tp.compute_drift(t)
            t = tp.subtract_drift(t.copy(), drift).reset_index(drop=True)
    return t


def suggest_minmass(features_all: pd.DataFrame) -> float | None:
    """Seuil minmass séparant le bruit des vraies particules.

    Les masses (luminosité intégrée) forment en général deux populations : beaucoup de petites
    détections de bruit, et les particules. Seuil d'Otsu sur log10(masse) : la valeur qui
    sépare le mieux ces deux groupes. None s'il y a trop peu de détections pour décider.
    """
    m = features_all["mass"].to_numpy() if len(features_all) else np.array([])
    m = m[np.isfinite(m) & (m > 0)]
    if m.size < 20:
        return None
    from skimage.filters import threshold_otsu
    return float(10 ** threshold_otsu(np.log10(m)))


def displacements(tracks: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Déplacements entre détections successives d'une même particule.

    Chaque ligne = un pas : position au milieu du pas (px), vitesse en px / unité de temps.
    Avec memory > 0, un pas peut couvrir plusieurs frames ; dt en tient compte.
    """
    cols = ["particle", "frame0", "frame1", "x", "y", "t", "dx", "dy", "dt", "u", "v"]
    if tracks.empty:
        return pd.DataFrame(columns=cols)
    t = tracks[["particle", "frame", "x", "y"]].sort_values(["particle", "frame"]).reset_index(drop=True)
    nxt = t.groupby("particle")[["frame", "x", "y"]].shift(-1)
    ok = nxt["frame"].notna().to_numpy()
    f0 = t["frame"].to_numpy()[ok].astype(int)
    f1 = nxt["frame"].to_numpy()[ok].astype(int)
    x0, y0 = t["x"].to_numpy()[ok], t["y"].to_numpy()[ok]
    x1, y1 = nxt["x"].to_numpy()[ok], nxt["y"].to_numpy()[ok]
    times = np.asarray(times, dtype=float)
    dt = times[f1] - times[f0]
    dx, dy = x1 - x0, y1 - y0
    out = pd.DataFrame({
        "particle": t["particle"].to_numpy()[ok].astype(int),
        "frame0": f0, "frame1": f1,
        "x": 0.5 * (x0 + x1), "y": 0.5 * (y0 + y1),
        "t": 0.5 * (times[f0] + times[f1]),
        "dx": dx, "dy": dy, "dt": dt,
        "u": dx / dt, "v": dy / dt,
    })
    return out[out["dt"] > 0].reset_index(drop=True)
