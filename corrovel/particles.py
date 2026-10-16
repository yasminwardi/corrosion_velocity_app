"""Analyse particule par particule des trajectoires Trackpy.

Objectif : répondre à des questions physiques sur les traceurs, pas seulement moyenner un champ.
- Quelle est la vitesse de chaque traceur, quelle distance parcourt-il ?
- Les traceurs démarrent-ils tous en même temps ? (temps de mise en mouvement, front de propagation)
- Ont-ils tous la même vitesse ? Le mouvement est-il homogène autour du pit ?
- Mouvement brownien ou dirigé, avant / après l'apparition du pit ? (MSD)
- S'éloignent-ils ou se rapprochent-ils du pit ? (traceurs positifs / négatifs)

Conventions : positions en px (image du dataset), converties en unité physique avec `scale`.
Les vitesses et distances par particule sont calculées sur des positions lissées (moyenne
glissante sur `smooth` détections) pour ne pas confondre bruit de localisation et mouvement ;
le MSD, lui, utilise les positions brutes (le lissage le biaiserait).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .field import _nearest_pit


@dataclass(frozen=True)
class MotionParams:
    smooth: int = 3                    # lissage des positions (nb de détections, centré)
    window: int = 3                    # vitesse « glissante » mesurée sur `window` frames
    threshold: float | None = None     # seuil de mouvement (unité phys / temps) ; None = auto
    persist: int = 2                   # le seuil doit être dépassé sur `persist` points consécutifs
    min_points: int = 5                # trajectoires plus courtes ignorées dans l'analyse
    t_pit: float | None = None         # temps d'apparition du pit (même unité que les temps)
    exclude_px: float = 0.0            # ignorer les détections à moins de ce rayon (px) d'un pit :
                                       # le bord sombre du pit est souvent pris pour des particules

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParticleAnalysis:
    params: MotionParams
    steps: pd.DataFrame           # une ligne par détection (positions lissées, vitesses, état)
    particles: pd.DataFrame       # une ligne par traceur (vitesse, distance, mise en mouvement…)
    population: pd.DataFrame      # une ligne par frame (vitesse médiane, fraction mobile…)
    threshold: float              # seuil de mouvement effectivement utilisé
    threshold_method: str         # comment il a été choisi
    scale: float
    length_unit: str
    time_unit: str
    pits: list[tuple[float, float]]

    @property
    def velocity_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}"


# --------------------------------------------------------------------------- calcul principal

def _group_shift(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    return df.groupby("particle", sort=False)[cols].shift(n)


def track_steps(tracks: pd.DataFrame, times: np.ndarray, scale: float, smooth: int = 3,
                window: int = 3) -> pd.DataFrame:
    """Une ligne par détection : positions lissées (moyenne glissante centrée sur `smooth`
    détections, px), vitesse du pas vers la détection suivante (speed) et vitesse glissante sur
    `window` frames centrée sur l'instant (v_win), en unité physique. Mêmes colonnes que
    ParticleAnalysis.steps pour plots.fig_speed_tracks, sans l'analyse complète."""
    times = np.asarray(times, dtype=float)
    t = tracks[["particle", "frame", "x", "y"]].sort_values(["particle", "frame"]).reset_index(drop=True)
    t["t"] = times[t["frame"].to_numpy().astype(int)]
    g = t.groupby("particle", sort=False)
    w = max(1, int(smooth))
    for c in ("x", "y"):
        t[c + "s"] = g[c].rolling(w, center=True, min_periods=1).mean().reset_index(level=0, drop=True)
    nxt = _group_shift(t, ["xs", "ys", "t"], -1)
    nw = max(1, int(window))
    a, b = nw // 2, nw - nw // 2
    past = _group_shift(t, ["xs", "ys", "t"], a) if a else t[["xs", "ys", "t"]]
    fut = _group_shift(t, ["xs", "ys", "t"], -b)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["speed"] = np.hypot(nxt["xs"] - t["xs"], nxt["ys"] - t["ys"]) * scale / (nxt["t"] - t["t"])
        t["v_win"] = np.hypot(fut["xs"] - past["xs"], fut["ys"] - past["ys"]) * scale / (fut["t"] - past["t"])
    return t


def analyze(tracks: pd.DataFrame, times: np.ndarray, pits: list[tuple[float, float]], scale: float,
            length_unit: str, time_unit: str, params: MotionParams = MotionParams()) -> ParticleAnalysis:
    """Analyse complète des trajectoires (`tracks` : sortie de tracking.link_tracks)."""
    times = np.asarray(times, dtype=float)
    p = params
    t = tracks[["particle", "frame", "x", "y"] + (["ep"] if "ep" in tracks else [])].copy()
    if p.exclude_px > 0:
        _, _, r_raw, _ = _nearest_pit(t["x"].to_numpy(), t["y"].to_numpy(), pits)
        t = t[r_raw >= p.exclude_px]
    n_pts = t.groupby("particle")["frame"].transform("size")
    t = t[n_pts >= max(2, p.min_points)].sort_values(["particle", "frame"]).reset_index(drop=True)
    if t.empty:
        raise ValueError("Aucune trajectoire assez longue : baisse « points min » ou améliore le suivi.")
    t["t"] = times[t["frame"].to_numpy().astype(int)]
    g = t.groupby("particle", sort=False)
    w = max(1, int(p.smooth))
    t["xs"] = g["x"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    t["ys"] = g["y"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())

    # Pas élémentaires (détection -> suivante), sur positions lissées
    nxt = _group_shift(t, ["xs", "ys", "t"], -1)
    dt = nxt["t"] - t["t"]
    dx, dy = (nxt["xs"] - t["xs"]) * scale, (nxt["ys"] - t["ys"]) * scale
    t["step_len"] = np.hypot(dx, dy)
    t["speed"] = t["step_len"] / dt
    # Composantes radiale (> 0 : s'éloigne du pit le plus proche) et tangentielle (> 0 : rotation
    # dans le sens anti-horaire à l'écran autour de ce pit)
    rx, ry, r, pit_idx = _nearest_pit(t["xs"].to_numpy(), t["ys"].to_numpy(), pits)
    t["r"] = r * scale
    t["pit"] = pit_idx
    t["angle"] = np.degrees(np.arctan2(-ry, rx))                  # position autour du pit (y vers le haut)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["v_radial"] = (dx * rx + dy * ry) / r / dt              # dx, dy déjà en unité physique
        t["v_tangential"] = (dx * ry - dy * rx) / r / dt
    t["t_since_first"] = t["t"] - g["t"].transform("min")

    # Vitesse glissante sur `window` frames, centrée sur l'instant (robuste au bruit, sans
    # avancer artificiellement la mise en mouvement), et état « en mouvement »
    nw = max(1, int(p.window))
    a, b = nw // 2, nw - nw // 2
    past = _group_shift(t, ["xs", "ys", "t"], a) if a else t[["xs", "ys", "t"]]
    fut = _group_shift(t, ["xs", "ys", "t"], -b)
    t["v_win"] = np.hypot(fut["xs"] - past["xs"], fut["ys"] - past["ys"]) * scale / (fut["t"] - past["t"])
    threshold, method = _auto_threshold(t, p, scale, times)
    if p.threshold:
        threshold, method = float(p.threshold), "valeur choisie"
    # Au-dessus du seuil sur `persist` points consécutifs. Aux extrémités d'une trajectoire, la
    # fenêtre de vitesse manque de points : l'état y est inconnu (NaN) et on reprend le dernier
    # état connu (vers la fin) ou le premier (vers le début), au lieu de le compter « immobile ».
    k = max(1, p.persist)
    above = (t["v_win"] > threshold).astype(float).where(np.isfinite(t["v_win"]))
    state = above.groupby(t["particle"], sort=False).transform(
        lambda s: s.rolling(k, min_periods=k).min().shift(-(k - 1)).ffill().bfill())
    t["moving"] = state.fillna(0) == 1

    particles = _per_particle(t, scale, nw)
    population = _population(t, times)
    return ParticleAnalysis(p, t, particles, population, threshold, method, scale, length_unit,
                            time_unit, list(pits))


def _auto_threshold(t: pd.DataFrame, p: MotionParams, scale: float, times: np.ndarray) -> tuple[float, str]:
    """Seuil de « mise en mouvement », par ordre de préférence :
    1. 95e percentile des vitesses pendant la période calme du début de la séquence (premiers
       15 % du temps avant le pit, si t_pit est connu) : « un traceur bouge quand il va plus vite
       que 95 % des mouvements de repos ». Toute la période avant le pit ne convient pas : les
       traceurs peuvent déjà accélérer avant que le pit soit visible ;
    2. seuil d'Otsu sur log(vitesse) : sépare population lente et population rapide ;
    3. bruit de localisation (colonne Trackpy « ep ») : 3 écarts-types d'un faux déplacement.
    """
    v = t["v_win"].to_numpy()
    ok = np.isfinite(v) & (v > 0)
    t0 = float(times[0])
    if p.t_pit is not None and p.t_pit > t0:
        tt = t["t"].to_numpy()
        calm = ok & (tt < t0 + 0.15 * (p.t_pit - t0) + 1e-9) if (p.t_pit - t0) > 0 else ok & False
        if calm.sum() < 50:                       # période trop courte : tout l'avant-pit
            calm = ok & (tt < p.t_pit)
        if calm.sum() >= 50:
            return float(np.percentile(v[calm], 95)), "95e percentile des vitesses au repos (début de séquence)"
    if ok.sum() >= 50:
        from skimage.filters import threshold_otsu
        lv = np.log10(v[ok])
        if np.ptp(lv) > 0.3:
            return float(10 ** threshold_otsu(lv)), "seuil d'Otsu (séparation lents / rapides)"
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    ep = float(np.nanmedian(t["ep"])) if "ep" in t and np.isfinite(t["ep"]).any() else 0.5
    noise = 3 * np.sqrt(2) * ep / np.sqrt(max(1, p.smooth)) * scale / (max(1, p.window) * dt)
    return float(noise), "bruit de localisation (3σ)"


def _per_particle(t: pd.DataFrame, scale: float, window: int) -> pd.DataFrame:
    g = t.groupby("particle", sort=False)
    first, last = g.head(1).set_index("particle"), g.tail(1).set_index("particle")
    out = pd.DataFrame(index=first.index)
    out["n_points"] = g.size()
    out["t_first"], out["t_last"] = first["t"], last["t"]
    out["duration"] = out["t_last"] - out["t_first"]
    out["x0_px"], out["y0_px"] = first["xs"], first["ys"]
    out["r0"], out["r_end"] = first["r"], last["r"]
    out["pit"] = first["pit"].astype(int)          # pit le plus proche à la première détection
    out["angle0"] = first["angle"]
    out["path_length"] = g["step_len"].sum(min_count=1)
    out["net_displacement"] = np.hypot(last["xs"] - first["xs"], last["ys"] - first["ys"]) * scale
    with np.errstate(invalid="ignore", divide="ignore"):
        out["straightness"] = out["net_displacement"] / out["path_length"]
        out["mean_speed"] = out["path_length"] / out["duration"]
        out["radial_speed"] = (out["r_end"] - out["r0"]) / out["duration"]
    out["max_speed"] = g["v_win"].max()
    out["median_speed"] = g["speed"].median()
    # Vitesse pendant le mouvement seulement (la moyenne sur toute la trajectoire mélange repos
    # et mouvement) : c'est « la » vitesse d'un traceur mobile
    mv_steps = t[t["moving"]].groupby("particle")
    out["moving_speed"] = mv_steps["v_win"].median().reindex(out.index)
    # Composantes radiale et tangentielle pendant le mouvement (médianes des pas)
    out["v_radial_moving"] = mv_steps["v_radial"].median().reindex(out.index)
    out["v_tangential_moving"] = mv_steps["v_tangential"].median().reindex(out.index)
    # Direction : cosinus entre le déplacement net et la direction « s'éloigner du pit »
    dxn, dyn = last["xs"] - first["xs"], last["ys"] - first["ys"]
    out["cos_radial"] = _cos_radial(first, dxn, dyn)
    # Angle signé du déplacement par rapport à « s'éloigner du pit » : 0° s'éloigne, ±180° se
    # rapproche, ±90° tourne autour du pit
    ang = np.radians(first["angle"])
    ux, uy = np.cos(ang), -np.sin(ang)
    out["dir_angle"] = np.degrees(np.arctan2(ux * dyn - uy * dxn, dxn * ux + dyn * uy))
    # Mise en mouvement : premier instant où le traceur dépasse durablement le seuil…
    mv = t[t["moving"]]
    onset = mv.groupby("particle")["t"].min()
    out["onset_detect"] = onset.reindex(out.index)
    out["mobile"] = out["onset_detect"].notna()
    already = out["mobile"] & np.isclose(out["onset_detect"], out["t_first"])
    out["status"] = np.where(~out["mobile"], "immobile",
                             np.where(already, "déjà en mouvement", "démarre"))
    # … puis affiné par un ajustement « repos puis vitesse constante » (le lissage et la fenêtre
    # de vitesse font sinon apparaître le démarrage ~1-2 frames trop tôt)
    out["onset_time"] = out["onset_detect"]
    starts = out.index[out["status"] == "démarre"]
    if len(starts):
        out.loc[starts, "onset_time"] = _refine_onsets(t, out.loc[starts, "onset_detect"], window)
    return out.reset_index()


def _refine_onsets(t: pd.DataFrame, detected: pd.Series, window: int) -> pd.Series:
    """Instant de démarrage par ajustement d'une « charnière » sur la position brute projetée sur
    la direction du mouvement : s(t) = c avant τ, c + v (t − τ) après. τ est cherché (pas de ¼ de
    frame) autour de la détection ; l'ajustement est limité aux points proches du démarrage pour ne
    pas être faussé par une accélération ultérieure."""
    raw = t.set_index("particle")
    out = {}
    for pid, t_det in detected.items():
        d = raw.loc[[pid]]
        tt, x, y = d["t"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy()
        dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0
        keep = tt <= t_det + (3 * window + 2) * dt
        tt, x, y = tt[keep], x[keep], y[keep]
        if len(tt) < 4:
            out[pid] = t_det
            continue
        ux, uy = x[-1] - x[0], y[-1] - y[0]
        nrm = np.hypot(ux, uy) or 1.0
        s = (x - x[0]) * ux / nrm + (y - y[0]) * uy / nrm
        best, best_tau = np.inf, t_det
        for tau in np.arange(max(tt[0], t_det - 2 * window * dt), min(tt[-1], t_det + window * dt) + 1e-9, dt / 4):
            A = np.column_stack([np.ones_like(tt), np.maximum(0.0, tt - tau)])
            coef, res_, *_ = np.linalg.lstsq(A, s, rcond=None)
            sse = float(res_[0]) if res_.size else float(np.sum((A @ coef - s) ** 2))
            if coef[1] > 0 and sse < best:
                best, best_tau = sse, float(tau)
        out[pid] = best_tau
    return pd.Series(out)


def _cos_radial(first: pd.DataFrame, dxn: pd.Series, dyn: pd.Series) -> pd.Series:
    # direction radiale au point de départ ; le pit le plus proche est déjà dans first["r"]
    # (on recalcule le vecteur pit -> départ à partir de l'angle stocké, y vers le bas)
    ang = np.radians(first["angle"])
    ux, uy = np.cos(ang), -np.sin(ang)
    norm = np.hypot(dxn, dyn)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dxn * ux + dyn * uy) / norm


def _population(t: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Comportement collectif, frame par frame."""
    s = t[np.isfinite(t["speed"])]
    g = s.groupby("frame")
    pop = pd.DataFrame({
        "t": times[g.size().index.to_numpy().astype(int)],
        "n": g.size(),
        "speed_median": g["speed"].median(),
        "speed_q10": g["speed"].quantile(0.10), "speed_q25": g["speed"].quantile(0.25),
        "speed_q75": g["speed"].quantile(0.75), "speed_q90": g["speed"].quantile(0.90),
        "v_radial_median": g["v_radial"].median(),
        "moving_fraction": t.groupby("frame")["moving"].mean().reindex(g.size().index),
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        pop["speed_cv"] = g["speed"].std() / g["speed"].mean()
    return pop.reset_index()


# --------------------------------------------------------------------------- indicateurs dérivés

def front_velocity(particles: pd.DataFrame) -> dict | None:
    """Les traceurs démarrent-ils d'autant plus tard qu'ils sont loin du pit ?

    Régression robuste (Theil-Sen) du temps de mise en mouvement sur la distance initiale au pit,
    uniquement pour les traceurs observés immobiles puis en mouvement (« démarre »). Pente > 0 :
    la mise en mouvement se propage depuis le pit, à la vitesse 1 / pente.
    """
    d = particles[particles["status"] == "démarre"].dropna(subset=["r0", "onset_time"])
    if len(d) < 8 or d["r0"].nunique() < 3 or d["onset_time"].nunique() < 2:
        return None
    from scipy.stats import spearmanr, theilslopes
    slope, intercept, lo, hi = theilslopes(d["onset_time"], d["r0"])
    rho, pval = spearmanr(d["r0"], d["onset_time"])
    return {"slope": float(slope), "intercept": float(intercept), "slope_ci": (float(lo), float(hi)),
            "velocity": float(1 / slope) if slope > 0 else None, "rho": float(rho), "p": float(pval),
            "n": int(len(d))}


def binned(steps: pd.DataFrame, by: str, value: str = "speed", bins: int | np.ndarray = 10,
           mask: np.ndarray | None = None, min_count: int = 10) -> pd.DataFrame:
    """Médiane et quartiles de `value` par classe de `by` (distance au pit, angle…)."""
    s = steps if mask is None else steps[mask]
    s = s[np.isfinite(s[value]) & np.isfinite(s[by])]
    if s.empty:
        return pd.DataFrame(columns=["center", "median", "q25", "q75", "n"])
    edges = np.linspace(s[by].min(), s[by].max(), bins + 1) if np.isscalar(bins) else np.asarray(bins)
    idx = np.clip(np.digitize(s[by], edges) - 1, 0, len(edges) - 2)
    g = s.groupby(idx)[value]
    out = pd.DataFrame({"center": 0.5 * (edges[:-1] + edges[1:])[g.size().index], "median": g.median(),
                        "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
    return out[out["n"] >= min_count].reset_index(drop=True)


def msd(tracks: pd.DataFrame, times: np.ndarray, scale: float, t_range: tuple[float, float] | None = None,
        max_lag: int = 30, min_pairs: int = 20, max_cells: int = 20_000_000) -> tuple[pd.DataFrame, dict | None]:
    """Déplacement quadratique moyen d'ensemble, MSD(τ) = ⟨|r(t+τ) − r(t)|²⟩, positions brutes.

    t_range : ne garder que les positions dans [t0, t1] (avant / après le pit). Ajustement de
    MSD ∝ τ^α sur les premiers retards : α ≈ 1 diffusion (brownien), α ≈ 2 mouvement dirigé
    (balistique), α < 1 mouvement confiné / gêné.
    """
    times = np.asarray(times, dtype=float)
    tr = tracks[["particle", "frame", "x", "y"]]
    if t_range is not None:
        ft = times[tr["frame"].to_numpy().astype(int)]
        tr = tr[(ft >= t_range[0]) & (ft <= t_range[1])]
    if tr.empty:
        return pd.DataFrame(columns=["lag", "tau", "msd", "n"]), None
    pids = tr["particle"].unique()
    if len(pids) * len(times) > max_cells:            # mémoire bornée : sous-échantillon de particules
        pids = np.random.default_rng(0).choice(pids, max_cells // len(times), replace=False)
        tr = tr[tr["particle"].isin(pids)]
    X = tr.pivot_table(index="particle", columns="frame", values="x").reindex(columns=range(len(times)))
    Y = tr.pivot_table(index="particle", columns="frame", values="y").reindex(columns=range(len(times)))
    X, Y = X.to_numpy(), Y.to_numpy()
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    rows = []
    for lag in range(1, min(max_lag, len(times) - 1) + 1):
        d2 = (X[:, lag:] - X[:, :-lag]) ** 2 + (Y[:, lag:] - Y[:, :-lag]) ** 2
        n = int(np.isfinite(d2).sum())
        if n >= min_pairs:
            rows.append((lag, lag * dt, float(np.nanmean(d2)) * scale ** 2, n))
    out = pd.DataFrame(rows, columns=["lag", "tau", "msd", "n"])
    fit = None
    fitpts = out.head(10)
    fitpts = fitpts[fitpts["msd"] > 0]
    if len(fitpts) >= 3:
        alpha, logk = np.polyfit(np.log(fitpts["tau"]), np.log(fitpts["msd"]), 1)
        fit = {"alpha": float(alpha), "k": float(np.exp(logk)), "n_lags": int(len(fitpts))}
    return out, fit


def threshold_sensitivity(tracks: pd.DataFrame, times: np.ndarray, pits: list[tuple[float, float]],
                          scale: float, length_unit: str, time_unit: str, params: MotionParams,
                          base_threshold: float, factors=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Les conclusions tiennent-elles si l'on change le seuil de mouvement ? Refait l'analyse avec
    seuil × 0.5, × 1, × 2 et résume les indicateurs qui en dépendent."""
    from dataclasses import replace
    rows = []
    for f in factors:
        pa = analyze(tracks, times, pits, scale, length_unit, time_unit,
                     replace(params, threshold=base_threshold * f))
        P = pa.particles
        st_ = P[P["status"] == "démarre"]
        fv = front_velocity(P)
        half = pa.population.loc[pa.population["moving_fraction"] >= 0.5, "t"]
        rows.append({"facteur": f, "seuil": pa.threshold, "% mobiles": 100 * P["mobile"].mean(),
                     "vitesse en mvt (méd.)": P["moving_speed"].median(),
                     "50 % en mvt à t =": half.iloc[0] if len(half) else np.nan,
                     "démarrages observés": len(st_),
                     "démarrage médian": st_["onset_time"].median() if len(st_) else np.nan,
                     "ρ démarrage–distance": fv["rho"] if fv else np.nan,
                     "p": fv["p"] if fv else np.nan,
                     "% s'éloignent": 100 * (P.loc[P["mobile"], "cos_radial"] > 0.5).mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- apparition du pit

def pit_signal(stack: np.ndarray, pit: tuple[float, float], radius: float = 12.0) -> np.ndarray:
    """Intensité moyenne dans un disque autour du pit, divisée par l'intensité moyenne de la frame
    (compense le photoblanchiment). Un pit sombre qui apparaît fait chuter ce signal."""
    n, h, w = stack.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pit[0]) ** 2 + (yy - pit[1]) ** 2 <= radius ** 2
    if not disk.any():
        return np.full(n, np.nan)
    frames = stack.reshape(n, -1)
    return stack[:, disk].mean(axis=1) / np.maximum(frames.mean(axis=1), 1e-12)


def estimate_pit_radius(frame: np.ndarray, pit: tuple[float, float], max_r: int = 400,
                        dark: float = 0.6, margin_px: float = 4.0) -> float:
    """Rayon du bord du pit (px) sur une image, + une petite marge.

    Profil d'intensité moyen par anneau autour du pit, rapporté à la médiane de l'image. Le bord
    est la remontée la plus raide du profil juste après le cœur sombre du pit (le halo grisâtre
    autour du pit remonte ensuite lentement et n'est pas compté). 0 si le pit n'est pas sombre.
    """
    h, w = frame.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - pit[0], yy - pit[1]).astype(int)
    keep = r < max_r
    prof = np.bincount(r[keep], weights=frame[keep]) / np.maximum(np.bincount(r[keep]), 1)
    ref = float(np.median(frame))
    if ref <= 0 or prof[0] >= dark * ref:
        return 0.0
    from scipy.ndimage import uniform_filter1d
    prof = uniform_filter1d(prof / ref, 3, mode="nearest")   # lissage sans effet de bord en r = 0
    out = np.flatnonzero(prof >= dark)                 # sortie du cœur sombre
    stop = int(out[0]) + 5 if out.size else len(prof) - 1
    grad = np.gradient(prof[:stop + 1])
    edge = int(np.argmax(grad))                        # remontée la plus raide = bord du pit
    return float(edge + margin_px)


def detect_pit_time(signal: np.ndarray, times: np.ndarray, min_change: float = 0.1) -> tuple[float | None, str]:
    """Début de l'apparition du pit sur le signal d'intensité autour du pit.

    Le signal change d'au moins `min_change` (relatif) entre le début et son extremum, sinon rien
    n'est détecté. On ajuste alors « constant, puis variation linéaire » sur la portion qui va du
    début jusqu'à la mi-variation : la cassure donne l'instant où le pit commence à se former.
    Si elle tombe sur la première frame, le pit se formait déjà au début de la séquence.
    Retourne (temps, explication) ; temps = None si rien n'est détecté.
    """
    s, t = np.asarray(signal, dtype=float), np.asarray(times, dtype=float)
    if len(s) < 6 or not np.isfinite(s).all():
        return None, "séquence trop courte ou signal invalide"
    base = float(np.median(s[:3]))
    dev = s - base
    ext = int(np.argmax(np.abs(dev)))
    change = float(dev[ext])
    if abs(change) < min_change * max(abs(base), 1e-12):
        return None, ("aucun changement net autour du pit : le pit était peut-être déjà présent au début "
                      "de la séquence, ou n'apparaît pas dans cette ROI")
    sign = np.sign(change)
    i_half = int(np.flatnonzero(sign * (s - (base + change / 2)) >= 0)[0])
    seg = slice(0, max(i_half + 1, 4))
    ts, ss_ = t[seg], s[seg]
    dt = float(np.median(np.diff(t)))
    best, tau = np.inf, float(ts[0])
    for cand in np.arange(ts[0], ts[-1] + 1e-9, dt / 4):
        A = np.column_stack([np.ones_like(ts), np.maximum(0.0, ts - cand)])
        coef, *_ = np.linalg.lstsq(A, ss_, rcond=None)
        sse = float(np.sum((A @ coef - ss_) ** 2))
        if sign * coef[1] > 0 and sse < best:
            best, tau = sse, float(cand)
    t_half = float(t[i_half])
    if tau <= t[0] + dt:
        return float(t[0]), (f"le signal change dès le début : pit déjà en formation au début de la séquence "
                             f"(t = {t[0]:.4g}) ; la moitié de la variation est atteinte à t = {t_half:.4g}")
    return tau, f"début de la variation du signal ; la moitié de la variation est atteinte à t = {t_half:.4g}"


# --------------------------------------------------------------------------- synthèse en texte

def summary(pa: ParticleAnalysis, msd_fits: dict | None = None) -> list[str]:
    """Réponses chiffrées aux questions de l'étude (phrases prêtes à relire pour l'article)."""
    P, u = pa.particles, pa.velocity_unit
    n, mob = len(P), P[P["mobile"]]
    lines = [f"**{n} traceurs** analysés (trajectoires d'au moins {pa.params.min_points} points), dont "
             f"**{len(mob)} mobiles** ({100 * len(mob) / max(n, 1):.0f} %) ; seuil de mouvement "
             f"{pa.threshold:.3g} {u} ({pa.threshold_method})."]
    if len(mob):
        q = mob["moving_speed"].quantile([0.25, 0.5, 0.75])
        cv = mob["moving_speed"].std() / mob["moving_speed"].mean()
        lines.append(f"**Vitesse** des traceurs pendant leur mouvement : médiane {q[0.5]:.3g} {u} (50 % entre "
                     f"{q[0.25]:.3g} et {q[0.75]:.3g}) ; vitesse max (glissante) médiane "
                     f"{mob['max_speed'].median():.3g} {u}. "
                     f"Dispersion entre traceurs : CV = {cv:.2f} "
                     f"({'faible : vitesses assez homogènes' if cv < 0.3 else 'forte : vitesses hétérogènes' if cv > 0.6 else 'modérée'}).")
        dq = mob["net_displacement"].quantile([0.25, 0.5, 0.75])
        lines.append(f"**Distance** : déplacement net médian {dq[0.5]:.3g} {pa.length_unit} (50 % entre "
                     f"{dq[0.25]:.3g} et {dq[0.75]:.3g}) ; rectitude médiane {mob['straightness'].median():.2f} "
                     "(1 = trajectoire droite).")
    pop = pa.population
    if len(pop) and pop["moving_fraction"].max() > 0:
        half = pop.loc[pop["moving_fraction"] >= 0.5, "t"]
        if len(half):
            lines.append(f"**Population** : la moitié des traceurs suivis est en mouvement à partir de "
                         f"t = {half.iloc[0]:.4g} {pa.time_unit} (fraction max {100 * pop['moving_fraction'].max():.0f} %).")
    st_ = P[P["status"] == "démarre"]
    already = int((P["status"] == "déjà en mouvement").sum())
    if already > len(st_):
        lines.append(f"⚠️ {already} traceurs ({100 * already / max(n, 1):.0f} %) sont **déjà en mouvement à leur "
                     "première détection** (trajectoires commencées en cours de mouvement) : les instants de "
                     f"démarrage individuels ne portent que sur {len(st_)} traceurs et favorisent les démarrages "
                     "tardifs. Pour « quand les traceurs bougent-ils », la courbe de fraction en mouvement "
                     "est plus fiable ; allonger les trajectoires (memory, search range) réduit ce biais.")
    if len(st_) >= 3:
        oq = st_["onset_time"].quantile([0.1, 0.5, 0.9])
        lines.append(f"**Mise en mouvement** ({len(st_)} traceurs vus immobiles puis mobiles) : médiane "
                     f"t = {oq[0.5]:.4g} {pa.time_unit}, 80 % entre {oq[0.1]:.4g} et {oq[0.9]:.4g} {pa.time_unit} "
                     f"→ {'quasi simultanée' if oq[0.9] - oq[0.1] < 0.1 * (P['t_last'].max() - P['t_first'].min()) else 'étalée dans le temps'}.")
        fv = front_velocity(P)
        if fv and fv["p"] < 0.05 and fv["velocity"]:
            lines.append(f"Les traceurs **loin du pit démarrent plus tard** (Spearman ρ = {fv['rho']:.2f}, "
                         f"p = {fv['p']:.2g}) : front de mise en mouvement ≈ {fv['velocity']:.3g} {u}.")
        elif fv:
            lines.append(f"Pas de lien net entre distance au pit et instant de démarrage (ρ = {fv['rho']:.2f}, "
                         f"p = {fv['p']:.2g}).")
    if pa.params.t_pit is not None:
        if pa.params.t_pit <= float(pa.steps["t"].min()) + 1e-9:
            lines.append(f"Le pit se forme **dès le début de la séquence** (t = {pa.params.t_pit:.4g} {pa.time_unit}) : "
                         "ces données n'ont pas de période « avant le pit » ; pour comparer avant / après, "
                         "génère un dataset qui commence plus tôt.")
        else:
            lines.append(f"Pit apparu à t = {pa.params.t_pit:.4g} {pa.time_unit}.")
    if len(mob):
        c = mob["cos_radial"].dropna()
        out_, in_ = float((c > 0.5).mean()), float((c < -0.5).mean())
        lines.append(f"**Direction** : {100 * out_:.0f} % des traceurs mobiles s'éloignent du pit, "
                     f"{100 * in_:.0f} % s'en rapprochent, le reste se déplace de côté (cos moyen = {c.mean():.2f}).")
    if msd_fits:
        parts = [f"{k} : α = {v['alpha']:.2f}" for k, v in msd_fits.items() if v]
        if parts:
            lines.append("**MSD** (α ≈ 1 brownien, ≈ 2 dirigé) — " + " ; ".join(parts) + ".")
    return lines
