"""Groupes de traceurs à comparer et statistiques de comparaison.

Un groupe = une analyse de traceurs (ParticleAnalysis) avec un nom et un style de tracé. Deux
sources, combinables :
- les pits d'une même vidéo : chaque traceur est rattaché au pit le plus proche de sa première
  détection (colonne « pit » de particles.analyze) → « Pit A », « Pit B »… ;
- les analyses ENREGISTRÉES d'autres datasets (dossier <dataset>/analyses/<nom>/), avec la
  description de l'expérience (type de traceur, substrat, potentiel…), pour comparer des pits
  filmés séparément, des traceurs positifs / négatifs ou des conditions expérimentales.

Statistiques : intervalle de confiance par bootstrap (rééchantillonnage des traceurs), tests de
Mann-Whitney (2 groupes) ou de Kruskal-Wallis (plus de 2) avec taille d'effet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import particles as pm

PIT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Comme la figure de référence : orange plein, vert tireté, puis bleu, violet, rouge…
GROUP_STYLES = [("#F29F05", "-"), ("#2E8B3C", "--"), ("#2B5C7E", "-."), ("#8E44AD", ":"),
                ("#E8423F", (0, (6, 2, 1, 2))), ("#7A8B99", (0, (1, 1)))]


@dataclass
class Group:
    label: str
    pa: pm.ParticleAnalysis
    color: str = "#F29F05"
    linestyle: object = "-"
    meta: dict = field(default_factory=dict)      # expérience, dataset, rayons des pits (px)…


def styled(groups: list[Group]) -> list[Group]:
    for i, g in enumerate(groups):
        g.color, g.linestyle = GROUP_STYLES[i % len(GROUP_STYLES)]
    return groups


def pit_label(i: int) -> str:
    return f"Pit {PIT_LETTERS[i % len(PIT_LETTERS)]}"


def subset(pa: pm.ParticleAnalysis, particle_ids) -> pm.ParticleAnalysis:
    """Même analyse restreinte à certains traceurs (population recalculée sur eux seuls)."""
    ids = set(np.asarray(particle_ids).tolist())
    steps = pa.steps[pa.steps["particle"].isin(ids)]
    parts = pa.particles[pa.particles["particle"].isin(ids)].reset_index(drop=True)
    times = np.full(int(pa.steps["frame"].max()) + 1, np.nan)
    times[pa.steps["frame"].to_numpy().astype(int)] = pa.steps["t"].to_numpy()
    pop = pm._population(steps, times) if len(steps) else pa.population.iloc[:0]
    return pm.ParticleAnalysis(pa.params, steps, parts, pop, pa.threshold, pa.threshold_method, pa.scale,
                               pa.length_unit, pa.time_unit, pa.pits)


def merge(analyses: list[pm.ParticleAnalysis]) -> pm.ParticleAnalysis:
    """Plusieurs analyses (répétitions d'une même condition) réunies en une : traceurs renumérotés
    pour rester distincts. Les grandeurs par traceur et par pas restent valables ; la population
    par frame n'a plus de sens commun (frames de vidéos différentes) : celle de la 1re analyse."""
    steps, parts, off = [], [], 0
    for a in analyses:
        s, p = a.steps.copy(), a.particles.copy()
        s["particle"] += off
        p["particle"] += off
        off = int(max(s["particle"].max(), p["particle"].max())) + 1 if len(s) else off
        steps.append(s)
        parts.append(p)
    a0 = analyses[0]
    return pm.ParticleAnalysis(a0.params, pd.concat(steps, ignore_index=True), pd.concat(parts, ignore_index=True),
                               a0.population, a0.threshold, a0.threshold_method, a0.scale, a0.length_unit,
                               a0.time_unit, a0.pits)


def pit_groups(pa: pm.ParticleAnalysis, name: str | None = None, meta: dict | None = None) -> list[Group]:
    """Un groupe par pit de la vidéo (un seul groupe s'il n'y a qu'un pit)."""
    meta = dict(meta or {})
    if len(pa.pits) <= 1:
        return [Group(name or pit_label(0), pa, meta={**meta, "pit_index": 0})]
    out = []
    for i in range(len(pa.pits)):
        ids = pa.particles.loc[pa.particles["pit"] == i, "particle"]
        if len(ids):
            lab = pit_label(i) if not name else f"{name} · {pit_label(i)}"
            out.append(Group(lab, subset(pa, ids), meta={**meta, "pit_index": i}))
    return out


# --------------------------------------------------------------------------- analyses enregistrées

ANALYSES_DIR = "analyses"


def _safe(label: str) -> str:
    """Nom de dossier : lisible + empreinte du nom complet, pour que « NiCr (+) » et « NiCr (−) »
    (même partie lisible) ne s'écrasent pas, alors qu'un même nom réenregistré met à jour son dossier."""
    import hashlib
    from .io import safe_name
    readable = safe_name(label.replace("+", " pos ").replace("−", " neg ").replace("(-)", " neg "))
    return f"{readable}_{hashlib.sha1(label.encode('utf-8')).hexdigest()[:6]}"


def save_analysis(pa: pm.ParticleAnalysis, dataset_folder: Path, label: str, meta: dict) -> Path:
    """Enregistre l'analyse (pas, traceurs, population + description) pour la comparer plus tard."""
    folder = Path(dataset_folder) / ANALYSES_DIR / _safe(label)
    folder.mkdir(parents=True, exist_ok=True)
    pa.steps.to_parquet(folder / "steps.parquet", index=False)
    pa.particles.to_parquet(folder / "particles.parquet", index=False)
    pa.population.to_parquet(folder / "population.parquet", index=False)
    info = {**meta, "label": label, "params": pa.params.to_dict(), "threshold": pa.threshold,
            "threshold_method": pa.threshold_method, "scale": pa.scale, "length_unit": pa.length_unit,
            "time_unit": pa.time_unit, "pits": [list(p) for p in pa.pits],
            "saved": datetime.now().isoformat(timespec="seconds")}
    (folder / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return folder


def list_saved(data_root: Path) -> list[dict]:
    """Toutes les analyses enregistrées (tous datasets), avec leur description."""
    out = []
    for m in sorted(Path(data_root).glob(f"datasets/*/{ANALYSES_DIR}/*/meta.json")):
        try:
            info = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        info["path"] = str(m.parent)
        info.setdefault("dataset", m.parent.parent.parent.name)
        out.append(info)
    return out


def load_analysis(folder: Path) -> tuple[pm.ParticleAnalysis, dict]:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    params = pm.MotionParams(**meta["params"])
    pa = pm.ParticleAnalysis(params, pd.read_parquet(folder / "steps.parquet"),
                             pd.read_parquet(folder / "particles.parquet"),
                             pd.read_parquet(folder / "population.parquet"), float(meta["threshold"]),
                             meta.get("threshold_method", ""), float(meta["scale"]), meta["length_unit"],
                             meta["time_unit"], [tuple(p) for p in meta["pits"]])
    return pa, meta


# --------------------------------------------------------------------------- statistiques

def bootstrap_ci(values, stat=np.median, n_boot: int = 2000, ci: float = 95.0,
                 seed: int = 0) -> tuple[float, float, float]:
    """(estimation, borne basse, borne haute) : intervalle de confiance par bootstrap
    (rééchantillonnage avec remise des valeurs, une par traceur). Moins de 3 valeurs : pas d'IC."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    est = float(stat(v))
    if v.size < 3:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, 250):                  # par paquets : mémoire bornée
        k = min(250, n_boot - start)
        boots.append(stat(v[rng.integers(0, v.size, (k, v.size))], axis=1))
    b = np.concatenate(boots)
    a = (100 - ci) / 2
    return est, float(np.percentile(b, a)), float(np.percentile(b, 100 - a))


def proportion_ci(k: int, n: int, ci: float = 95.0) -> tuple[float, float, float]:
    """Proportion k / n et intervalle de Wilson (fiable même près de 0 ou 1)."""
    if n <= 0:
        return np.nan, np.nan, np.nan
    from scipy.stats import norm
    z = norm.ppf(0.5 + ci / 200)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def compare_samples(samples: dict[str, np.ndarray]) -> dict | None:
    """Les groupes diffèrent-ils ? 2 groupes : Mann-Whitney (taille d'effet : corrélation
    rang-bisériale, −1…1, > 0 quand le 1er groupe a les valeurs les plus grandes) ; plus :
    Kruskal-Wallis (taille d'effet : η²_H, 0…1). Valeurs non finies ignorées ; groupes de moins
    de 3 valeurs écartés."""
    from scipy.stats import kruskal, mannwhitneyu
    s = {k: np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for k, v in samples.items()}
    s = {k: v for k, v in s.items() if v.size >= 3}
    if len(s) < 2:
        return None
    labels = list(s)
    if len(s) == 2:
        a, b = s[labels[0]], s[labels[1]]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return {"test": "Mann-Whitney", "p": float(p), "effect": float(2 * u / (a.size * b.size) - 1),
                "effect_name": "rank-biserial r", "groups": labels, "n": [int(a.size), int(b.size)]}
    h, p = kruskal(*s.values())
    n = sum(v.size for v in s.values())
    return {"test": "Kruskal-Wallis", "p": float(p), "effect": float((h - len(s) + 1) / (n - len(s))),
            "effect_name": "η²_H", "groups": labels, "n": [int(v.size) for v in s.values()]}
