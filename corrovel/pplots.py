"""Figures de l'analyse particule par particule (matplotlib, prêtes pour un article).

Style commun (corrovel.style) : textes en anglais, étiquettes en gras « Grandeur / (unité) ».
Chaque figure répond à une question de l'étude :
- fig_tracks_speed    : figure de référence — trajectoires colorées par la vitesse le long du chemin ;
- fig_population      : quand les traceurs bougent-ils, et à quelle vitesse, au cours du temps ?
- fig_tracks_colored  : où vont-ils (trajectoires colorées par une propriété du traceur) ?
- fig_r_t             : s'éloignent-ils du pit, ensemble ou chacun à son rythme ?
- fig_onset           : démarrent-ils en même temps ? la mise en mouvement se propage-t-elle ?
- fig_distributions   : ont-ils tous la même vitesse, la même distance parcourue ?
- fig_space           : la vitesse dépend-elle de la distance au pit, de la direction autour du pit ?
- fig_direction       : s'éloignent-ils ou se rapprochent-ils du pit ?
- fig_msd             : mouvement brownien ou dirigé, avant / après le pit ?
- fig_tracking_quality: les trajectoires sont-elles assez longues et nombreuses pour conclure ?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from . import particles as pm
from . import plots
from . import style
from .plots import _draw_pits, _extent, _finish_image_axes
from .scalebar import ScaleBar

C_ALL, C_BEFORE, C_AFTER = "#2B5C7E", "#7A8B99", "#E8423F"
# Libellés de l'interface (français) et des figures (anglais)
COLOR_BY = {"moving_speed": "vitesse en mouvement", "onset_time": "instant de démarrage",
            "cos_radial": "direction (cos, +1 s'éloigne du pit)", "time": "temps"}
FIG_COLOR_BY = {"moving_speed": "Speed while moving", "onset_time": "Onset time",
                "cos_radial": "Direction (cos θ, +1 = away from pit)", "time": "Time"}
PERIODS_EN = {"avant le pit": "before pit", "après le pit": "after pit", "toute la séquence": "whole sequence"}


def _vu(pa: pm.ParticleAnalysis) -> str:
    return style.vel_unit(pa.length_unit, pa.time_unit)


def _color_label(pa: pm.ParticleAnalysis, color_by: str) -> str:
    unit = {"moving_speed": _vu(pa), "onset_time": pa.time_unit, "time": pa.time_unit}.get(color_by)
    return style.label(FIG_COLOR_BY.get(color_by, color_by), unit)


def _t_pit_line(ax, t_pit, label=True):
    if t_pit is not None:
        ax.axvline(t_pit, color="black", lw=1, ls="--")
        if label:
            ax.annotate("pit", (t_pit, 0), xycoords=("data", "axes fraction"), xytext=(3, 4),
                        textcoords="offset points", fontsize=8)


def fig_tracks_speed(pa: pm.ParticleAnalysis, background: np.ndarray | None, scalebar: ScaleBar | None = None,
                     pit_radii=None, vmax: float | None = None, end_frame: int | None = None,
                     only_mobile: bool = True, min_displacement: float = 0.0, panel: str | None = None,
                     show_ends: bool = True, max_tracks: int = 3000, figsize=(5.0, 4.3),
                     min_points: int = 5, smooth: int = 3):
    """Figure de référence : trajectoires (positions lissées) colorées par la vitesse le long du
    chemin, cercle + flèche à la position atteinte à `end_frame`, contour du pit, barre d'échelle.
    only_mobile : seulement les traceurs mobiles ; min_displacement : déplacement net minimal
    (unité de longueur) pour ne garder que les trajectoires qui vont quelque part ; min_points :
    trajectoires d'au moins n détections ; smooth : lissage d'affichage (voir fig_speed_tracks)."""
    P = pa.particles
    if only_mobile:
        P = P[P["mobile"]]
    if min_displacement > 0:
        P = P[P["net_displacement"] >= min_displacement]
    return plots.fig_speed_tracks(pa.steps, background, pa.scale, pa.length_unit, pa.time_unit, pa.pits,
                                  pit_radii, vmax, end_frame, P["particle"], min_points, panel, scalebar,
                                  show_ends, max_tracks, figsize=figsize, smooth=smooth)


def fig_population(pa: pm.ParticleAnalysis, figsize=(6.2, 5.6)):
    """Vitesse des traceurs (médiane, 50 % et 80 % centraux), vitesse radiale et fraction en
    mouvement, au cours du temps, avec l'instant d'apparition du pit."""
    pop, u, tp_ = pa.population, _vu(pa), pa.params.t_pit
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1.3, 1]})
    a = axes[0]
    a.fill_between(pop["t"], pop["speed_q10"], pop["speed_q90"], color=C_ALL, alpha=0.15, lw=0, label="Central 80 %")
    a.fill_between(pop["t"], pop["speed_q25"], pop["speed_q75"], color=C_ALL, alpha=0.3, lw=0, label="Central 50 %")
    a.plot(pop["t"], pop["speed_median"], color=C_ALL, lw=1.6, label="Median")
    a.axhline(pa.threshold, color="0.4", lw=0.8, ls=":", label="Motion threshold")
    a.set_ylabel(style.label("Particle speed", u))
    a.legend(fontsize=7, ncol=2)
    a.set_title("Tracer speed over time")
    b = axes[1]
    b.axhline(0, color="0.6", lw=0.8)
    b.plot(pop["t"], pop["v_radial_median"], color=C_AFTER, lw=1.4)
    b.set_ylabel(style.label("Radial velocity", u))
    c = axes[2]
    c.plot(pop["t"], 100 * pop["moving_fraction"], color="black", lw=1.4)
    c.set_ylabel(style.label("Moving", "%"))
    c.set_ylim(-3, 103)
    c.set_xlabel(style.label("Time", pa.time_unit))
    for i, ax in enumerate(axes):
        _t_pit_line(ax, tp_, label=i == len(axes) - 1)
    return fig


def _particle_segments(pa: pm.ParticleAnalysis, pids) -> tuple[list[np.ndarray], list[np.ndarray]]:
    s = pa.steps[pa.steps["particle"].isin(pids)]
    segs, tms = [], []
    for _, g in s.groupby("particle", sort=False):
        segs.append(np.column_stack([g["xs"].to_numpy(), g["ys"].to_numpy()]) * pa.scale)
        tms.append(g["t"].to_numpy())
    return segs, tms


def fig_tracks_colored(pa: pm.ParticleAnalysis, background: np.ndarray | None, color_by: str = "moving_speed",
                       max_tracks: int = 2500, scalebar: ScaleBar | None = None, cmap: str = "plasma",
                       figsize=(5.0, 4.3), only_mobile: bool = False, pit_radii=None, panel: str | None = None):
    """Trajectoires (positions lissées) colorées par traceur : vitesse, instant de démarrage,
    direction par rapport au pit — ou segment par segment selon le temps."""
    P = pa.particles if not only_mobile else pa.particles[pa.particles["mobile"]]
    if len(P) > max_tracks:
        P = P.sample(max_tracks, random_state=0)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    shape = background.shape if background is not None else None
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=_extent(shape, pa.scale), interpolation="nearest")
    segs, tms = _particle_segments(pa, P["particle"])
    if color_by == "time":
        pieces = [np.stack([s[:-1], s[1:]], axis=1) for s in segs if len(s) > 1]
        vals = np.concatenate([0.5 * (t_[:-1] + t_[1:]) for s, t_ in zip(segs, tms) if len(s) > 1]) if pieces else []
        lc = LineCollection(np.concatenate(pieces) if pieces else [], cmap="viridis", linewidths=1.2)
        lc.set_array(np.asarray(vals))
    else:
        v = P.set_index("particle")[color_by]
        lc = LineCollection(segs, cmap="coolwarm" if color_by == "cos_radial" else cmap, linewidths=1.2)
        vals = np.array([v.get(pid, np.nan) for pid in P["particle"]])
        lc.set_array(np.ma.masked_invalid(vals))
        if color_by == "cos_radial":
            lc.set_norm(Normalize(-1, 1))
        else:
            fin = vals[np.isfinite(vals)]
            if fin.size:
                lc.set_norm(Normalize(*np.percentile(fin, [2, 98])))
    ax.add_collection(lc)
    style.colorbar(fig, lc, ax, _color_label(pa, color_by), show_max=False)
    _draw_pits(ax, pa.pits, pa.scale, pit_radii)
    if shape is not None:
        ext = _extent(shape, pa.scale)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    else:
        ax.autoscale()
        ax.invert_yaxis()
    ax.set_aspect("equal")
    _finish_image_axes(ax, pa.scale, shape or (0, 0), scalebar if shape is not None else None,
                       pa.length_unit, background)
    style.panel_label(ax, panel)
    ax.set_title(f"{len(P)} trajectories")
    return fig


def fig_r_t(pa: pm.ParticleAnalysis, color_by: str = "moving_speed", max_tracks: int = 1500,
            figsize=(6.0, 4.2), cmap: str = "plasma"):
    """Distance au pit en fonction du temps, une courbe par traceur : une pente positive = le
    traceur s'éloigne ; l'instant où la courbe se redresse = sa mise en mouvement."""
    P = pa.particles
    if len(P) > max_tracks:
        P = P.sample(max_tracks, random_state=0)
    s = pa.steps[pa.steps["particle"].isin(P["particle"])]
    segs = [np.column_stack([g["t"].to_numpy(), g["r"].to_numpy()]) for _, g in s.groupby("particle", sort=False)]
    pids = [pid for pid, _ in s.groupby("particle", sort=False)]
    v = P.set_index("particle")[color_by].reindex(pids).to_numpy()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    lc = LineCollection(segs, cmap="coolwarm" if color_by == "cos_radial" else cmap, linewidths=0.8, alpha=0.85)
    lc.set_array(np.ma.masked_invalid(v))
    fin = v[np.isfinite(v)]
    lc.set_norm(Normalize(-1, 1) if color_by == "cos_radial" else Normalize(*np.percentile(fin, [2, 98])) if fin.size else Normalize())
    ax.add_collection(lc)
    ax.autoscale()
    style.colorbar(fig, lc, ax, _color_label(pa, color_by), show_max=False)
    _t_pit_line(ax, pa.params.t_pit)
    ax.set_xlabel(style.label("Time", pa.time_unit))
    ax.set_ylabel(style.label("Distance from pit", pa.length_unit))
    return fig


def fig_onset(pa: pm.ParticleAnalysis, figsize=(8.4, 3.4)):
    """(a) Quand les traceurs démarrent-ils ? (b) Démarrent-ils plus tard loin du pit ?"""
    P = pa.particles
    d = P[P["status"] == "démarre"]
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    if len(d):
        a.hist(d["onset_time"], bins=min(30, max(5, len(d) // 5)), color=C_ALL, alpha=0.85)
    _t_pit_line(a, pa.params.t_pit)
    a.set_xlabel(style.label("Onset time", pa.time_unit))
    a.set_ylabel("Tracers")
    n_al = int((P["status"] == "déjà en mouvement").sum())
    a.set_title(f"(a) {len(d)} observed onsets\n({n_al} already moving, not counted)")
    if len(d):
        b.scatter(d["r0"], d["onset_time"], s=6, color=C_ALL, alpha=0.5, lw=0)
    fv = pm.front_velocity(P)
    if fv:
        xs = np.linspace(d["r0"].min(), d["r0"].max(), 50)
        b.plot(xs, fv["intercept"] + fv["slope"] * xs, color=C_AFTER, lw=1.5)
        txt = f"Spearman ρ = {fv['rho']:.2f} (p = {fv['p']:.1g})"
        if fv["velocity"] and fv["p"] < 0.05:
            txt += f"\nfront ≈ {fv['velocity']:.3g} {_vu(pa)}"
        b.text(0.03, 0.97, txt, transform=b.transAxes, va="top", fontsize=8,
               bbox=dict(facecolor="white", alpha=0.8, lw=0))
    if pa.params.t_pit is not None:
        b.axhline(pa.params.t_pit, color="black", lw=1, ls="--")
    b.set_xlabel(style.label("Initial distance from pit", pa.length_unit))
    b.set_ylabel(style.label("Onset time", pa.time_unit))
    b.set_title("(b) Onset vs. distance from pit")
    return fig


def fig_distributions(pa: pm.ParticleAnalysis, figsize=(8.4, 3.4)):
    """(a) Vitesse : avant / après le pit (pas élémentaires) ; (b) distance parcourue par traceur."""
    S, P, u, tp_ = pa.steps, pa.particles, _vu(pa), pa.params.t_pit
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    v = S["v_win"].to_numpy()
    ok = np.isfinite(v) & (v > 0)
    hi = np.percentile(v[ok], 99) if ok.any() else 1
    bins = np.linspace(0, hi, 40)
    if tp_ is not None and (ok & (S["t"] < tp_)).sum() > 10:
        a.hist(v[ok & (S["t"] < tp_)], bins=bins, density=True, color=C_BEFORE, alpha=0.6, label="Before pit")
        a.hist(v[ok & (S["t"] >= tp_)], bins=bins, density=True, color=C_AFTER, alpha=0.6, label="After pit")
        a.legend()
    else:
        a.hist(v[ok], bins=bins, density=True, color=C_ALL, alpha=0.85)
    a.axvline(pa.threshold, color="0.3", lw=0.8, ls=":")
    a.set_xlabel(style.label("Instantaneous speed", u))
    a.set_ylabel("Probability density")
    a.set_title("(a) Speed distribution")
    mob = P[P["mobile"]]
    if len(mob):
        b.hist(mob["net_displacement"], bins=30, color=C_ALL, alpha=0.85, label="Net displacement")
        b.hist(mob["path_length"], bins=30, histtype="step", color="black", lw=1.2, label="Path length")
        b.legend()
    b.set_xlabel(style.label("Distance", pa.length_unit))
    b.set_ylabel("Moving tracers")
    b.set_title("(b) Distance travelled per tracer")
    return fig


def fig_space(pa: pm.ParticleAnalysis, figsize=(8.6, 3.6), after_pit: bool = True):
    """(a) Vitesse et vitesse radiale selon la distance au pit ; (b) vitesse médiane selon la
    direction autour du pit (homogénéité angulaire)."""
    S, u, tp_ = pa.steps, _vu(pa), pa.params.t_pit
    mask = S["moving"].to_numpy()
    if after_pit and tp_ is not None:
        mask = mask & (S["t"].to_numpy() >= tp_)
    fig = plt.figure(figsize=figsize, layout="constrained")
    a = fig.add_subplot(1, 2, 1)
    for col, color, lab in (("v_win", C_ALL, "Speed"), ("v_radial", C_AFTER, "Radial velocity")):
        bdf = pm.binned(S, "r", col, 12, mask)
        if len(bdf):
            a.fill_between(bdf["center"], bdf["q25"], bdf["q75"], color=color, alpha=0.2, lw=0)
            a.plot(bdf["center"], bdf["median"], "o-", ms=3, color=color, label=lab)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_xlabel(style.label("Distance from pit", pa.length_unit))
    a.set_ylabel(style.label("Velocity", u))
    a.legend()
    a.set_title("(a) vs. distance from pit (median, central 50 %)")
    b = fig.add_subplot(1, 2, 2, projection="polar")
    edges = np.linspace(-180, 180, 13)
    bdf = pm.binned(S, "angle", "v_win", edges, mask, min_count=5)
    if len(bdf):
        b.bar(np.radians(bdf["center"]), bdf["median"], width=np.radians(30) * 0.9, color=C_ALL, alpha=0.8)
        cv = float(bdf["median"].std() / bdf["median"].mean()) if len(bdf) > 1 else np.nan
        b.set_title(f"(b) Median speed vs. direction\naround the pit (CV = {cv:.2f})")
    b.set_theta_zero_location("E")
    b.tick_params(labelsize=7)
    return fig


def fig_direction(pa: pm.ParticleAnalysis, figsize=(8.0, 3.6)):
    """(a) Rose des directions de déplacement par rapport au pit ; (b) vitesse radiale par traceur."""
    mob = pa.particles[pa.particles["mobile"]]
    fig = plt.figure(figsize=figsize, layout="constrained")
    a = fig.add_subplot(1, 2, 1, projection="polar")
    if len(mob):
        h, e = np.histogram(np.radians(mob["dir_angle"].dropna()), bins=24, range=(-np.pi, np.pi))
        a.bar(0.5 * (e[:-1] + e[1:]), h, width=(e[1] - e[0]) * 0.95, color=C_AFTER, alpha=0.8)
    a.set_theta_zero_location("N")
    a.set_xticks(np.radians([0, 90, 180, 270]))
    a.set_xticklabels(["away", "", "toward", ""], fontsize=8)
    a.tick_params(axis="y", labelsize=7)
    a.set_title("(a) Displacement direction\nrelative to the pit")
    b = fig.add_subplot(1, 2, 2)
    if len(mob):
        rs = mob["radial_speed"].dropna()
        lim = np.percentile(np.abs(rs), 98) if len(rs) else 1
        b.hist(rs, bins=np.linspace(-lim, lim, 41), color=C_ALL, alpha=0.85)
        b.axvline(0, color="black", lw=0.8)
        b.text(0.02, 0.97, f"Away from pit: {100 * (mob['cos_radial'] > 0.5).mean():.0f} %\n"
                           f"Toward pit: {100 * (mob['cos_radial'] < -0.5).mean():.0f} %",
               transform=b.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.8, lw=0))
    b.set_xlabel(style.label("Mean radial velocity", _vu(pa)))
    b.set_ylabel("Moving tracers")
    b.set_title("(b) > 0: away from pit")
    return fig


def fig_msd(curves: dict[str, tuple[pd.DataFrame, dict | None]], length_unit: str, time_unit: str,
            figsize=(4.8, 4.0)):
    """MSD(τ) en log-log, une courbe par période, avec l'exposant α ajusté et les pentes 1 et 2."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    colors = {"avant le pit": C_BEFORE, "après le pit": C_AFTER}
    for lab, (df, fit) in curves.items():
        if df is None or df.empty:
            continue
        c = colors.get(lab, C_ALL)
        name = PERIODS_EN.get(lab, lab).capitalize()
        ax.loglog(df["tau"], df["msd"], "o", ms=3.5, color=c,
                  label=f"{name} (α = {fit['alpha']:.2f})" if fit else name)
        if fit:
            tt = df["tau"].head(fit["n_lags"])
            ax.loglog(tt, fit["k"] * tt ** fit["alpha"], "-", color=c, lw=1.2)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    if np.all(np.isfinite(xl)) and xl[0] > 0:
        x0 = xl[0] * 1.2
        y0 = 10 ** (0.5 * (np.log10(yl[0]) + np.log10(yl[1]))) if yl[0] > 0 else 1
        for slope, ls in ((1, ":"), (2, "--")):
            ax.loglog([x0, x0 * 4], [y0, y0 * 4 ** slope], ls, color="0.4", lw=0.9)
            ax.annotate(f"slope {slope}", (x0 * 4, y0 * 4 ** slope), fontsize=7, color="0.4")
    ax.set_xlabel(style.label("Lag time τ", time_unit))
    ax.set_ylabel(style.label("MSD", style.unit((length_unit, 2))))
    ax.legend()
    return fig


def fig_tracking_quality(tracks: pd.DataFrame, times: np.ndarray, time_unit: str, figsize=(8.0, 3.0)):
    """(a) Durée des trajectoires ; (b) nombre de particules suivies par frame."""
    times = np.asarray(times, dtype=float)
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    if len(tracks):
        span = tracks.groupby("particle")["frame"].agg(lambda f: f.max() - f.min() + 1)
        a.hist(span, bins=np.arange(1, span.max() + 2) - 0.5 if span.max() < 60 else 40, color=C_ALL)
        a.axvline(span.median(), color=C_AFTER, lw=1.2, label=f"Median {span.median():.0f}")
        a.legend()
        n = tracks.groupby("frame")["particle"].nunique()
        b.plot(times[n.index.to_numpy().astype(int)], n.to_numpy(), color="black", lw=1.2)
    a.set_xlabel(style.label("Track length", "frames"))
    a.set_ylabel("Tracks")
    a.set_title("(a) Track length")
    b.set_xlabel(style.label("Time", time_unit))
    b.set_ylabel("Tracked particles")
    b.set_title("(b) Tracked particles per frame")
    return fig
