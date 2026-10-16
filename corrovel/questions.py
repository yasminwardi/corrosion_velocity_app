"""Calculs organisés par question de l'étude (une section de l'application chacune).

Toutes les fonctions prennent une ParticleAnalysis (particles.analyze), éventuellement restreinte
à un groupe (compare.subset). Chaque nombre important est donné avec un intervalle de confiance à
95 % (bootstrap sur les traceurs, ou intervalle de Wilson pour une proportion).

- Vitesse            : speed_summary, speed_since_first, speed_vs_distance
- Homogénéité        : homogeneity_summary, behaviour_groups
- Déplacement        : persistence, track_alpha, turning_angles, displacement_summary
- Début du mouvement : onset_summary, started_fraction
- Relation avec le pit : pit_radius_series, pit_phases, phase_stats, distance_fit,
                         speed_growth_correlation
- Traceurs / comparaison : indicators, replicate_verdict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import particles as pm
from .compare import bootstrap_ci, compare_samples, proportion_ci

MOTION_TYPES = {"confined": "confined (α < 0.7)", "diffusive": "diffusive (0.7 ≤ α ≤ 1.3)",
                "directed": "directed (α > 1.3)"}


def frame_dt(pa: pm.ParticleAnalysis) -> float:
    t = np.unique(pa.steps["t"].to_numpy())
    return float(np.median(np.diff(t))) if t.size > 1 else 1.0


def add_r_norm(pa: pm.ParticleAnalysis, radii_px) -> pm.ParticleAnalysis:
    """Distance au pit rapportée au rayon de CE pit (r / R), quand le rayon est connu : compare des
    pits de tailles différentes. Ajoute « r_norm » aux pas et « r0_norm » aux traceurs."""
    radii = np.asarray(list(radii_px or []), dtype=float)
    steps, parts = pa.steps.copy(), pa.particles.copy()
    if radii.size and (radii > 0).any():
        R = np.where(radii > 0, radii * pa.scale, np.nan)
        steps["r_norm"] = steps["r"] / R[np.clip(steps["pit"].to_numpy().astype(int), 0, R.size - 1)]
        parts["r0_norm"] = parts["r0"] / R[np.clip(parts["pit"].to_numpy().astype(int), 0, R.size - 1)]
    else:
        steps["r_norm"], parts["r0_norm"] = np.nan, np.nan
    return pm.ParticleAnalysis(pa.params, steps, parts, pa.population, pa.threshold, pa.threshold_method,
                               pa.scale, pa.length_unit, pa.time_unit, pa.pits)


# --------------------------------------------------------------------------- 1. Vitesse

def speed_summary(pa: pm.ParticleAnalysis) -> dict:
    P = pa.particles
    mob = P[P["mobile"]]
    return {"n": int(len(P)), "n_mobile": int(len(mob)),
            "mobile": proportion_ci(int(len(mob)), int(len(P))),
            "moving_speed": bootstrap_ci(mob["moving_speed"]),
            "p10": float(np.nanpercentile(mob["moving_speed"], 10)) if len(mob) else np.nan,
            "p90": float(np.nanpercentile(mob["moving_speed"], 90)) if len(mob) else np.nan,
            "max_speed": bootstrap_ci(mob["max_speed"])}


def speed_since_first(pa: pm.ParticleAnalysis, mobile_only: bool = True,
                      min_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vitesse (glissante) en fonction du temps depuis la première détection de chaque traceur
    (figure de référence (b)). Retourne (courbes individuelles, moyenne) ; la moyenne n'est donnée
    qu'aux instants où au moins `min_n` traceurs sont mesurés (au-delà elle ne représente plus la
    population)."""
    S = pa.steps
    if mobile_only:
        S = S[S["particle"].isin(pa.particles.loc[pa.particles["mobile"], "particle"])]
    S = S[np.isfinite(S["v_win"])]
    curves = S[["particle", "t_since_first", "v_win"]].rename(columns={"v_win": "speed"})
    dt = frame_dt(pa)
    b = np.rint(curves["t_since_first"] / dt).astype(int)
    g = curves.groupby(b)["speed"]
    mean = pd.DataFrame({"t_since_first": g.size().index * dt, "mean": g.mean().to_numpy(),
                         "n": g.size().to_numpy()})
    return curves, mean[mean["n"] >= min_n].reset_index(drop=True)


def speed_vs_distance(pa: pm.ParticleAnalysis, normalized: bool = False, bins: int = 12) -> pd.DataFrame:
    """Vitesse des traceurs en mouvement selon la distance au pit (médiane et quartiles par classe)."""
    col = "r_norm" if normalized and "r_norm" in pa.steps and np.isfinite(pa.steps["r_norm"]).any() else "r"
    return pm.binned(pa.steps, col, "v_win", bins, pa.steps["moving"].to_numpy())


# --------------------------------------------------------------------------- 2. Homogénéité

def _cv(x, axis=None):
    return np.std(x, axis=axis) / np.mean(x, axis=axis)


def homogeneity_summary(pa: pm.ParticleAnalysis) -> dict:
    P, pop = pa.particles, pa.population
    mob = P[P["mobile"]]
    starts = P[P["status"] == "démarre"]["onset_time"].dropna()
    half = pop.loc[pop["moving_fraction"] >= 0.5, "t"]
    bdf = pm.binned(pa.steps, "angle", "v_win", np.linspace(-180, 180, 13), pa.steps["moving"].to_numpy(), 5)
    return {"speed_cv": bootstrap_ci(mob["moving_speed"], _cv),
            "onset_spread": (float(np.percentile(starts, 10)), float(np.percentile(starts, 90)))
            if len(starts) >= 5 else (np.nan, np.nan),
            "n_starts": int(len(starts)),
            "t_half_moving": float(half.iloc[0]) if len(half) else np.nan,
            "max_moving_fraction": float(pop["moving_fraction"].max()) if len(pop) else np.nan,
            "angular_cv": float(bdf["median"].std() / bdf["median"].mean()) if len(bdf) > 2 else np.nan}


BEHAVIOUR_FEATURES = {"moving_speed": "log speed", "straightness": "straightness",
                      "cos_radial": "direction (cos θ)", "persistence": "persistence"}
# Étendue P10–P90 minimale pour qu'une caractéristique puisse séparer des groupes (unités propres :
# décades pour la vitesse, soit un facteur 1.5 ; rectitude 0…1 ; cosinus et persistance −1…1).
BEHAVIOUR_MIN_SPREAD = {"moving_speed": np.log10(1.5), "straightness": 0.1, "cos_radial": 0.2, "persistence": 0.2}


def behaviour_groups(pa: pm.ParticleAnalysis, max_k: int = 4, min_tracers: int = 30, seed: int = 0) -> dict | None:
    """Existe-t-il des groupes de traceurs aux comportements différents ?

    Mélange de gaussiennes sur les caractéristiques standardisées des traceurs mobiles (log de la
    vitesse en mouvement, rectitude, direction par rapport au pit, persistance) ; le nombre de
    groupes (1 à max_k) est choisi par le critère BIC, en ne retenant un groupe de plus que s'il
    améliore le BIC de plus de 10 (« preuve très forte », Kass & Raftery 1995) : sur une
    population homogène, de petites fluctuations ne créent pas de faux groupes. Un seul groupe =
    population homogène.
    Une caractéristique qui ne varie presque pas (étendue P10–P90 < BEHAVIOUR_MIN_SPREAD, ex.
    rectitude toujours ≈ 1) est écartée : standardisée, elle ne contiendrait que du bruit, coincé
    contre sa borne (non gaussien), que le mélange découperait en faux groupes. Aucune
    caractéristique variable : un seul groupe.
    Groupes renumérotés du plus lent au plus rapide. Moins de `min_tracers` traceurs : None.
    """
    from sklearn.mixture import GaussianMixture
    P = pa.particles[pa.particles["mobile"]].copy()
    if "persistence" not in P:
        P = P.merge(persistence(pa).rename("persistence"), left_on="particle", right_index=True, how="left")
    P = P[P["moving_speed"] > 0]
    feats = P[list(BEHAVIOUR_FEATURES)].assign(moving_speed=np.log10(P["moving_speed"]))
    ok = feats.notna().all(axis=1)
    P, feats = P[ok], feats[ok]
    if len(P) < min_tracers:
        return None
    used = [c for c in BEHAVIOUR_FEATURES
            if feats[c].quantile(0.9) - feats[c].quantile(0.1) >= BEHAVIOUR_MIN_SPREAD[c]]
    best, bics, lab = 0, [], np.zeros(len(P), int)
    if used:
        X = feats[used].to_numpy(dtype=float)
        sd = X.std(axis=0)
        Xs = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        fits = []
        for k in range(1, max_k + 1):
            if len(P) < 10 * k:
                break
            gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(Xs)
            fits.append(gm)
            bics.append(float(gm.bic(Xs)))
        for k in range(1, len(bics)):
            if bics[k] < bics[best] - 10:
                best = k
        lab = fits[best].predict(Xs)
    order = np.argsort([np.median(P["moving_speed"].to_numpy()[lab == j]) for j in range(best + 1)])
    rank = {old: new for new, old in enumerate(order)}
    P["group"] = [f"G{rank[j] + 1}" for j in lab]
    table = P.groupby("group").agg(n=("particle", "size"), speed=("moving_speed", "median"),
                                   straightness=("straightness", "median"), cos_radial=("cos_radial", "median"),
                                   persistence=("persistence", "median"), r0=("r0", "median"))
    table["share"] = table["n"] / table["n"].sum()
    return {"k": best + 1, "bic": bics, "labels": P.set_index("particle")["group"], "table": table.reset_index(),
            "features": used}


# --------------------------------------------------------------------------- 3. Déplacement

def _successive_steps(pa: pm.ParticleAnalysis, lag: int | None, moving_only: bool,
                      min_step_px: float) -> pd.DataFrame:
    """Paires de déplacements successifs (t − lag → t, t → t + lag) de chaque traceur, sur les
    positions BRUTES : les positions lissées (moyenne glissante) rendent deux pas voisins corrélés
    (ils partagent des déplacements), ce qui ferait paraître « persistante » une marche au hasard.
    Sur des intervalles disjoints, une diffusion donne bien une persistance nulle. lag : fenêtre
    de vitesse de l'analyse par défaut ; moving_only : pas centraux en mouvement seulement."""
    lag = int(lag or pa.params.window or 1)
    S = pa.steps[["particle", "frame", "x", "y", "moving"]]
    prev = S[["particle", "frame", "x", "y"]].assign(frame=S["frame"] + lag)
    nxt = S[["particle", "frame", "x", "y"]].assign(frame=S["frame"] - lag)
    M = (S.merge(prev, on=["particle", "frame"], suffixes=("", "_p"))
         .merge(nxt, on=["particle", "frame"], suffixes=("", "_n")))
    if moving_only:
        M = M[M["moving"]]
    M = M.assign(dx1=M["x"] - M["x_p"], dy1=M["y"] - M["y_p"], dx2=M["x_n"] - M["x"], dy2=M["y_n"] - M["y"])
    ok = (np.hypot(M["dx1"], M["dy1"]) > min_step_px) & (np.hypot(M["dx2"], M["dy2"]) > min_step_px)
    return M[ok]


def persistence(pa: pm.ParticleAnalysis, lag: int | None = None, moving_only: bool = True,
                min_step_px: float = 0.05) -> pd.Series:
    """Persistance de direction par traceur : moyenne du cosinus de l'angle entre deux
    déplacements successifs (sur `lag` frames chacun). 1 = va tout droit, 0 = change de direction
    au hasard (diffusion), < 0 = revient en arrière (confinement, oscillation)."""
    M = _successive_steps(pa, lag, moving_only, min_step_px)
    cos = (M["dx1"] * M["dx2"] + M["dy1"] * M["dy2"]) / (np.hypot(M["dx1"], M["dy1"]) * np.hypot(M["dx2"], M["dy2"]))
    return cos.groupby(M["particle"]).mean()


def turning_angles(pa: pm.ParticleAnalysis, moving_only: bool = True, lag: int | None = None,
                   min_step_px: float = 0.05) -> np.ndarray:
    """Angles de virage (degrés, −180…180) entre déplacements successifs : pic à 0 = mouvement
    dirigé, distribution plate = diffusion."""
    M = _successive_steps(pa, lag, moving_only, min_step_px)
    return np.degrees(np.arctan2(M["dx1"] * M["dy2"] - M["dy1"] * M["dx2"],
                                 M["dx1"] * M["dx2"] + M["dy1"] * M["dy2"])).to_numpy()


def track_alpha(pa: pm.ParticleAnalysis, min_points: int = 10, min_lags: int = 4) -> pd.DataFrame:
    """Exposant α du MSD de CHAQUE traceur (positions brutes) : MSD(τ) ∝ τ^α ajusté en log-log sur
    les retards 1…max(min_lags, n/4) frames. Type de mouvement : confiné (α < 0.7), diffusif
    (0.7–1.3), dirigé (α > 1.3). Le bruit de localisation ajoute une constante au MSD et abaisse α
    aux petits retards : un traceur presque immobile peut apparaître « confiné »."""
    rows = []
    for pid, d in pa.steps.groupby("particle", sort=False):
        if len(d) < min_points:
            continue
        f = d["frame"].to_numpy().astype(int)
        span = f.max() - f.min() + 1
        x = np.full(span, np.nan)
        y = np.full(span, np.nan)
        x[f - f.min()], y[f - f.min()] = d["x"].to_numpy(), d["y"].to_numpy()
        L = max(min_lags, len(d) // 4)
        lags, msd = [], []
        for k in range(1, min(L, span - 1) + 1):
            d2 = (x[k:] - x[:-k]) ** 2 + (y[k:] - y[:-k]) ** 2
            if np.isfinite(d2).sum() >= 3:
                lags.append(k)
                msd.append(np.nanmean(d2))
        lags, msd = np.asarray(lags, float), np.asarray(msd, float)
        okm = msd > 0
        if okm.sum() < min_lags:
            continue
        alpha = float(np.polyfit(np.log(lags[okm]), np.log(msd[okm]), 1)[0])
        rows.append((pid, alpha))
    out = pd.DataFrame(rows, columns=["particle", "alpha"])
    out["motion_type"] = np.select([out["alpha"] < 0.7, out["alpha"] > 1.3], ["confined", "directed"], "diffusive")
    return out


def displacement_summary(pa: pm.ParticleAnalysis, alphas: pd.DataFrame | None = None) -> dict:
    P = pa.particles
    mob = P[P["mobile"]]
    c = mob["cos_radial"].dropna()
    n = int(len(c))
    vr, vt = mob["v_radial_moving"].abs(), mob["v_tangential_moving"].abs()
    with np.errstate(invalid="ignore", divide="ignore"):
        share_t = (vt / (vr + vt)).dropna()
    out = {"net_displacement": bootstrap_ci(mob["net_displacement"]),
           "path_length": bootstrap_ci(mob["path_length"]),
           "straightness": bootstrap_ci(mob["straightness"]),
           "away": proportion_ci(int((c > 0.5).sum()), n), "toward": proportion_ci(int((c < -0.5).sum()), n),
           "sideways": proportion_ci(int(((c >= -0.5) & (c <= 0.5)).sum()), n),
           "tangential_share": bootstrap_ci(share_t),
           "v_tangential": bootstrap_ci(mob["v_tangential_moving"]),
           "persistence": bootstrap_ci(persistence(pa).reindex(mob["particle"]))}
    if alphas is not None and len(alphas):
        a = alphas[alphas["particle"].isin(mob["particle"])]
        out["alpha"] = bootstrap_ci(a["alpha"])
        out["motion_types"] = {k: proportion_ci(int((a["motion_type"] == k).sum()), int(len(a))) for k in MOTION_TYPES}
        out["n_alpha"] = int(len(a))
    return out


# --------------------------------------------------------------------------- 4. Début du mouvement

def onset_summary(pa: pm.ParticleAnalysis) -> dict:
    P = pa.particles
    st = P[P["status"] == "démarre"]
    already = int((P["status"] == "déjà en mouvement").sum())
    n_mob = int(P["mobile"].sum())
    t_pit = pa.params.t_pit
    out = {"n_starts": int(len(st)), "already": proportion_ci(already, n_mob),
           "onset": bootstrap_ci(st["onset_time"]),
           "spread": (float(np.percentile(st["onset_time"], 10)), float(np.percentile(st["onset_time"], 90)))
           if len(st) >= 5 else (np.nan, np.nan),
           "front": pm.front_velocity(P)}
    out["onset_rel_pit"] = bootstrap_ci(st["onset_time"] - t_pit) if t_pit is not None else (np.nan,) * 3
    half = pa.population.loc[pa.population["moving_fraction"] >= 0.5, "t"]
    out["t_half_moving"] = float(half.iloc[0]) if len(half) else np.nan
    return out


def started_fraction(pa: pm.ParticleAnalysis) -> pd.DataFrame:
    """Fraction cumulée des traceurs mobiles déjà mis en mouvement à chaque instant (un traceur
    « déjà en mouvement » compte à sa première détection)."""
    P = pa.particles[pa.particles["mobile"]]
    t_on = np.sort(P["onset_time"].fillna(P["t_first"]).to_numpy())
    if t_on.size == 0:
        return pd.DataFrame(columns=["t", "fraction"])
    return pd.DataFrame({"t": t_on, "fraction": np.arange(1, t_on.size + 1) / t_on.size})


# --------------------------------------------------------------------------- 5. Relation avec le pit

def pit_radius_series(stack: np.ndarray, pit: tuple[float, float], times: np.ndarray,
                      n_samples: int = 60) -> pd.DataFrame:
    """Rayon du pit (px, bord estimé par particles.estimate_pit_radius, marge retirée) au cours
    du temps, sur ~n_samples frames réparties ; « R_smooth » : médiane glissante sur 5 points."""
    idx = np.unique(np.linspace(0, len(stack) - 1, min(n_samples, len(stack))).astype(int))
    R = np.array([pm.estimate_pit_radius(stack[i], pit, margin_px=0.0) for i in idx])
    df = pd.DataFrame({"frame": idx, "t": np.asarray(times, float)[idx], "R_px": R})
    df["R_smooth"] = df["R_px"].rolling(5, center=True, min_periods=1).median()
    return df


def pit_phases(radius: pd.DataFrame, t_pit: float | None, t0: float, t1: float, frac: float = 0.9,
               t_end_growth: float | None = None) -> dict:
    """Phases avant / croissance / après : la croissance commence à t_pit et se termine quand le
    rayon (lissé) atteint `frac` de sa valeur finale (ou à t_end_growth si donné). Sans t_pit : une
    seule phase. Sans croissance mesurable : avant / après."""
    if t_pit is None or not (t0 < t_pit < t1):
        return {"t_pit": t_pit, "t_end_growth": None, "phases": {"whole sequence": (t0, t1)}}
    if t_end_growth is None and len(radius):
        r_final = float(np.median(radius["R_smooth"].tail(max(3, len(radius) // 10))))
        if r_final > 1.0:
            after = radius[(radius["t"] >= t_pit) & (radius["R_smooth"] >= frac * r_final)]
            if len(after):
                t_end_growth = float(after["t"].iloc[0])
    if t_end_growth is not None and t_pit < t_end_growth < t1:
        return {"t_pit": t_pit, "t_end_growth": t_end_growth,
                "phases": {"before": (t0, t_pit), "growth": (t_pit, t_end_growth), "after": (t_end_growth, t1)}}
    return {"t_pit": t_pit, "t_end_growth": None, "phases": {"before": (t0, t_pit), "after": (t_pit, t1)}}


def phase_stats(pa: pm.ParticleAnalysis, phases: dict) -> pd.DataFrame:
    """Par phase : vitesse et vitesse radiale des pas en mouvement (médiane des médianes par
    traceur, avec IC), part des pas en mouvement, nombre de traceurs."""
    rows = []
    S = pa.steps
    for name, (a, b) in phases["phases"].items():
        s = S[(S["t"] >= a) & (S["t"] < b)]
        m = s[s["moving"]]
        per = m.groupby("particle")[["v_win", "v_radial"]].median()
        v, vlo, vhi = bootstrap_ci(per["v_win"])
        r, rlo, rhi = bootstrap_ci(per["v_radial"])
        rows.append({"phase": name, "t_start": a, "t_end": b, "tracers": int(s["particle"].nunique()),
                     "moving_share": float(s["moving"].mean()) if len(s) else np.nan,
                     "speed": v, "speed_lo": vlo, "speed_hi": vhi,
                     "v_radial": r, "v_radial_lo": rlo, "v_radial_hi": rhi})
    return pd.DataFrame(rows)


def distance_fit(pa: pm.ParticleAnalysis, t_range: tuple[float, float] | None = None, normalized: bool = False,
                 bins: int = 10) -> dict | None:
    """Le mouvement change-t-il avec la distance au pit ? Ajustement en loi de puissance de la
    vitesse médiane (traceurs en mouvement) selon la distance : v ∝ r^(−n). n > 0 : plus lent loin
    du pit. Ajustement en log-log sur les classes de distance (≥ 10 pas par classe) ; IC à 95 % de
    n par l'erreur standard de la pente."""
    from scipy.stats import linregress
    S = pa.steps
    mask = S["moving"].to_numpy()
    if t_range is not None:
        mask = mask & (S["t"].to_numpy() >= t_range[0]) & (S["t"].to_numpy() < t_range[1])
    col = "r_norm" if normalized and "r_norm" in S and np.isfinite(S["r_norm"]).any() else "r"
    s = S[mask & (S[col].to_numpy() > 0)]
    if len(s) < 50:
        return None
    edges = np.geomspace(s[col].quantile(0.02), s[col].quantile(0.98), bins + 1)
    s = s[np.isfinite(s["v_win"])]
    idx = np.clip(np.digitize(s[col], edges) - 1, 0, bins - 1)
    g = s.groupby(idx)
    # Abscisse = distance MÉDIANE de chaque classe (pas le milieu de la classe) : la vitesse étant
    # une fonction monotone de r, médiane de v ↔ médiane de r, sans biais sur l'exposant.
    bdf = pd.DataFrame({"center": g[col].median(), "median": g["v_win"].median(), "n": g.size()})
    bdf = bdf[(bdf["n"] >= 10) & (bdf["median"] > 0) & (bdf["center"] > 0)].reset_index(drop=True)
    if len(bdf) < 4:
        return None
    lr = linregress(np.log(bdf["center"]), np.log(bdf["median"]))
    return {"n": float(-lr.slope), "n_ci": (float(-lr.slope - 1.96 * lr.stderr), float(-lr.slope + 1.96 * lr.stderr)),
            "r2": float(lr.rvalue ** 2), "prefactor": float(np.exp(lr.intercept)), "bins": bdf, "column": col}


def speed_growth_correlation(pa: pm.ParticleAnalysis, radius: pd.DataFrame) -> dict | None:
    """Les traceurs vont-ils plus vite quand le pit grandit vite ? Corrélation de Spearman entre
    la vitesse médiane des traceurs (par frame) et la vitesse de croissance du rayon du pit."""
    from scipy.stats import spearmanr
    if len(radius) < 6 or radius["R_smooth"].max() <= 1.0:
        return None
    t, R = radius["t"].to_numpy(), radius["R_smooth"].to_numpy() * pa.scale
    rate = np.gradient(R, t)
    pop = pa.population
    g = np.interp(pop["t"], t, rate)
    ok = np.isfinite(pop["speed_median"].to_numpy()) & np.isfinite(g)
    if ok.sum() < 6:
        return None
    rho, p = spearmanr(pop["speed_median"].to_numpy()[ok], g[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


# --------------------------------------------------------------------------- 6. Traceurs / comparaison

INDICATORS = {"moving_speed": "Speed while moving", "mobile": "Mobile tracers",
              "straightness": "Straightness", "away": "Moving away from pit",
              "onset_rel_pit": "Onset after pit appearance", "persistence": "Persistence"}


def tracer_values(pa: pm.ParticleAnalysis) -> pd.DataFrame:
    """Une ligne par traceur mobile avec les grandeurs comparées entre groupes."""
    P = pa.particles
    mob = P[P["mobile"]].copy()
    mob["away"] = (mob["cos_radial"] > 0.5).astype(float)
    t_pit = pa.params.t_pit
    mob["onset_rel_pit"] = (mob["onset_time"] - t_pit).where(mob["status"] == "démarre") if t_pit is not None else np.nan
    mob = mob.merge(persistence(pa).rename("persistence"), left_on="particle", right_index=True, how="left")
    return mob


def indicators(pa: pm.ParticleAnalysis) -> dict:
    """Indicateurs clés d'une analyse (une répétition) : médianes par traceur, parts en %."""
    P = pa.particles
    v = tracer_values(pa)
    return {"tracers": int(len(P)), "moving_speed": float(v["moving_speed"].median()) if len(v) else np.nan,
            "mobile": float(P["mobile"].mean()) if len(P) else np.nan,
            "straightness": float(v["straightness"].median()) if len(v) else np.nan,
            "away": float(v["away"].mean()) if len(v) else np.nan,
            "onset_rel_pit": float(v["onset_rel_pit"].median()) if v["onset_rel_pit"].notna().any() else np.nan,
            "persistence": float(v["persistence"].median()) if v["persistence"].notna().any() else np.nan}


def replicate_verdict(groups: dict[str, list[pm.ParticleAnalysis]], metric: str) -> dict:
    """Différence entre groupes (ex. traceurs + / −) pour une grandeur, et sa robustesse.

    - test sur les valeurs par traceur, toutes répétitions réunies (Mann-Whitney / Kruskal-Wallis) ;
    - reproductibilité : pour 2 groupes, la différence a-t-elle le même signe pour CHAQUE paire de
      répétitions (une analyse de chaque groupe) ? Variabilité entre répétitions : CV des médianes.
    Verdict : « robust » si p < 0.05 ET même signe partout (au moins 2 répétitions par groupe),
    « significant, not reproduced » si p < 0.05 mais signes contradictoires ou une seule
    répétition, « no difference detected » sinon."""
    samples, reps = {}, {}
    for lab, analyses in groups.items():
        vals = [tracer_values(a)[metric].to_numpy(dtype=float) for a in analyses]
        samples[lab] = np.concatenate(vals) if vals else np.array([])
        reps[lab] = [float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan for x in vals]
    test = compare_samples(samples)
    out = {"test": test, "replicates": reps,
           "rep_cv": {k: float(np.nanstd(r) / np.nanmean(r)) if len(r) > 1 and np.nanmean(r) else np.nan
                      for k, r in reps.items()}}
    consistent = None
    labs = list(reps)
    if len(labs) == 2:
        a, b = [x for x in reps[labs[0]] if np.isfinite(x)], [x for x in reps[labs[1]] if np.isfinite(x)]
        if a and b:
            signs = {np.sign(x - y) for x in a for y in b}
            consistent = len(signs) == 1 and 0 not in signs
    out["consistent"] = consistent
    enough = all(len(r) >= 2 for r in reps.values())
    if test is None:
        out["verdict"] = "not enough data"
    elif test["p"] < 0.05 and consistent and enough:
        out["verdict"] = "robust"
    elif test["p"] < 0.05:
        out["verdict"] = "significant, not reproduced"
    else:
        out["verdict"] = "no difference detected"
    return out


def fmt_ci(t: tuple, fmt: str = "{:.3g}", scale: float = 1.0) -> str:
    """« 3.2 [2.9 – 3.5] » ; sans IC : la valeur seule."""
    est, lo, hi = t
    if est is None or not np.isfinite(est):
        return "—"
    s = fmt.format(est * scale)
    if lo is not None and np.isfinite(lo) and hi is not None and np.isfinite(hi):
        s += f" [{fmt.format(lo * scale)} – {fmt.format(hi * scale)}]"
    return s

