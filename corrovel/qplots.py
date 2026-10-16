"""Figures des sections « questions de l'étude » (matplotlib, style de corrovel.style).

Chaque figure reçoit une liste de groupes (compare.Group : pits d'une vidéo, analyses
enregistrées…) et les superpose avec leur couleur et leur style de trait, comme la figure de
référence (b) (« Pit A mean » orange plein, « Pit B mean » vert tireté). `letter` : lettre de
panneau, « (b) ». Textes en anglais, étiquettes « Grandeur / (unité) ».
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import compare as cmp
from . import particles as pm
from . import questions as q
from . import style
from .plots import _draw_pits, _extent, _finish_image_axes

PHASE_COLORS = {"before": "#9AA5B1", "growth": "#F29F05", "after": "#2B5C7E", "whole sequence": "#9AA5B1"}
PHASE_NAMES = {"before": "Before pit", "growth": "Pit growth", "after": "After growth", "whole sequence": "Whole sequence"}


def _units(groups) -> tuple[str, str, str]:
    pa = groups[0].pa
    return style.vel_unit(pa.length_unit, pa.time_unit), pa.length_unit, pa.time_unit


def _fig(figsize=(5.0, 3.8), **kw):
    return plt.subplots(figsize=figsize, layout="constrained", **kw)


def _finish(ax, letter, legend=True):
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")
    style.panel_letter(ax, letter)


def _shade_phases(ax, phases: dict | None):
    if not phases:
        return
    for name, (a, b) in phases["phases"].items():
        if name != "whole sequence":
            ax.axvspan(a, b, color=PHASE_COLORS.get(name, "0.8"), alpha=0.12, lw=0)
    for t in (phases.get("t_pit"), phases.get("t_end_growth")):
        if t is not None:
            ax.axvline(t, color="black", lw=0.9, ls="--")


def _t(g: cmp.Group, t: np.ndarray, align_pit: bool) -> np.ndarray:
    tp = g.pa.params.t_pit
    return t - tp if align_pit and tp is not None else t


# --------------------------------------------------------------------------- 1. Vitesse

def fig_speed_since_first(groups, mobile_only: bool = True, max_time: float | None = None, individual: bool = True,
                          max_individual: int = 150, letter: str | None = None, figsize=(5.0, 3.9)):
    """Figure de référence (b) : vitesse de chaque traceur depuis sa première détection (traits
    fins) et moyenne par groupe (trait épais)."""
    from matplotlib import patheffects
    vu, _, T = _units(groups)
    fig, ax = _fig(figsize)
    ymax, xmax = 0.0, 0.0
    rng = np.random.default_rng(0)
    data = [(g, *q.speed_since_first(g.pa, mobile_only)) for g in groups]
    for g, curves, _ in data:                      # traits fins d'abord, tous groupes
        if individual and len(curves):
            ids = curves["particle"].unique()
            if len(ids) > max_individual:
                ids = rng.choice(ids, max_individual, replace=False)
            sub = curves[curves["particle"].isin(ids)]
            segs = [np.column_stack([d["t_since_first"], d["speed"]]) for _, d in sub.groupby("particle")]
            # transparence selon le nombre de courbes : la moyenne reste lisible avec des centaines de traceurs
            ax.add_collection(LineCollection(segs, colors=g.color, linewidths=0.5,
                                             alpha=float(np.clip(12 / len(segs), 0.04, 0.35))))
            ymax = max(ymax, float(np.nanpercentile(curves["speed"], 99.5)))
    halo = [patheffects.Stroke(linewidth=4.4, foreground="white"), patheffects.Normal()]
    for g, _, mean in data:                        # moyennes par-dessus, liseré blanc
        if len(mean):
            ax.plot(mean["t_since_first"], mean["mean"], color=g.color, ls=g.linestyle, lw=2.4, zorder=5,
                    path_effects=halo, label=f"{g.label} mean")
            xmax = max(xmax, float(mean["t_since_first"].max()))
            ymax = max(ymax, float(mean["mean"].max()))
    ax.set_xlim(0, max_time or xmax or 1)
    ax.set_ylim(0, 1.05 * ymax if ymax > 0 else 1)
    ax.set_xlabel(style.label("Time since first detection", T))
    ax.set_ylabel(style.label("Particle speed", vu))
    _finish(ax, letter)
    return fig


def fig_speed_distribution(groups, letter: str | None = None, figsize=(5.0, 3.8)):
    """Distribution des vitesses en mouvement (une valeur par traceur), médiane en pointillés."""
    vu, _, _ = _units(groups)
    fig, ax = _fig(figsize)
    vals = {g.label: g.pa.particles.loc[g.pa.particles["mobile"], "moving_speed"].dropna().to_numpy() for g in groups}
    allv = np.concatenate([v for v in vals.values() if v.size]) if any(v.size for v in vals.values()) else np.array([1.0])
    bins = np.linspace(0, np.percentile(allv, 99), 30)
    for g in groups:
        v = vals[g.label]
        if v.size:
            ax.hist(v, bins, density=True, histtype="step", color=g.color, ls=g.linestyle, lw=1.8,
                    label=f"{g.label} (n = {v.size})")
            ax.axvline(np.median(v), color=g.color, ls=":", lw=1.2)
    ax.set_xlabel(style.label("Speed while moving", vu))
    ax.set_ylabel("Probability density")
    _finish(ax, letter)
    return fig


def fig_speed_time(groups, align_pit: bool = False, letter: str | None = None, figsize=(5.0, 3.8)):
    """Vitesse médiane des traceurs au cours du temps (bande : 50 % central)."""
    vu, _, T = _units(groups)
    fig, ax = _fig(figsize)
    for g in groups:
        pop = g.pa.population
        if not len(pop):
            continue
        t = _t(g, pop["t"].to_numpy(), align_pit)
        ax.fill_between(t, pop["speed_q25"], pop["speed_q75"], color=g.color, alpha=0.15, lw=0)
        ax.plot(t, pop["speed_median"], color=g.color, ls=g.linestyle, lw=1.6, label=g.label)
        if not align_pit and g.pa.params.t_pit is not None:
            ax.axvline(g.pa.params.t_pit, color=g.color, lw=0.9, ls="--")
    if align_pit:
        ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(style.label("Time since pit appearance" if align_pit else "Time", T))
    ax.set_ylabel(style.label("Particle speed", vu))
    ax.set_ylim(bottom=0)
    _finish(ax, letter)
    return fig


def fig_speed_distance(groups, normalized: bool = False, letter: str | None = None, figsize=(5.0, 3.8)):
    """Vitesse des traceurs en mouvement selon la distance au pit (médiane, 50 % central)."""
    vu, L, _ = _units(groups)
    fig, ax = _fig(figsize)
    for g in groups:
        bdf = q.speed_vs_distance(g.pa, normalized)
        if len(bdf):
            ax.fill_between(bdf["center"], bdf["q25"], bdf["q75"], color=g.color, alpha=0.15, lw=0)
            ax.plot(bdf["center"], bdf["median"], "o", color=g.color, ls=g.linestyle, ms=3.5, lw=1.6, label=g.label)
    ax.set_xlabel("Distance from pit, r / R" if normalized else style.label("Distance from pit", L))
    ax.set_ylabel(style.label("Particle speed", vu))
    ax.set_ylim(bottom=0)
    _finish(ax, letter)
    return fig


# --------------------------------------------------------------------------- 2. Homogénéité

def fig_moving_fraction(groups, align_pit: bool = False, letter: str | None = None, figsize=(5.0, 3.6)):
    _, _, T = _units(groups)
    fig, ax = _fig(figsize)
    for g in groups:
        pop = g.pa.population
        if len(pop):
            ax.plot(_t(g, pop["t"].to_numpy(), align_pit), 100 * pop["moving_fraction"], color=g.color,
                    ls=g.linestyle, lw=1.6, label=g.label)
    if align_pit:
        ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_ylim(-3, 103)
    ax.set_xlabel(style.label("Time since pit appearance" if align_pit else "Time", T))
    ax.set_ylabel(style.label("Moving tracers", "%"))
    _finish(ax, letter)
    return fig


def fig_behaviour_groups(pa: pm.ParticleAnalysis, bg: dict, letter: str | None = None, figsize=(8.6, 3.6)):
    """Groupes de comportement (mélange de gaussiennes) dans l'espace des caractéristiques."""
    P = pa.particles.merge(q.persistence(pa).rename("persistence"), left_on="particle", right_index=True, how="left")
    P = P[P["particle"].isin(bg["labels"].index)].copy()
    P["group"] = bg["labels"].reindex(P["particle"]).to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    vu = style.vel_unit(pa.length_unit, pa.time_unit)
    for i, (name, d) in enumerate(P.groupby("group")):
        c = cmp.GROUP_STYLES[i % len(cmp.GROUP_STYLES)][0]
        lab = f"{name} (n = {len(d)})"
        a.scatter(d["moving_speed"], d["straightness"], s=10, color=c, alpha=0.7, lw=0, label=lab)
        b.scatter(d["cos_radial"], d["persistence"], s=10, color=c, alpha=0.7, lw=0, label=lab)
    sp = P["moving_speed"][P["moving_speed"] > 0]
    if len(sp) and sp.max() / sp.min() > 10:
        a.set_xscale("log")
    a.set_xlabel(style.label("Speed while moving", vu))
    a.set_ylabel("Straightness")
    b.set_xlabel("Direction, cos θ (+1 = away from pit)")
    b.set_ylabel("Direction persistence")
    a.legend(loc="lower right")
    style.panel_letter(a, letter)
    return fig


def fig_angular_speed(groups, letter: str | None = None, figsize=(4.4, 4.2)):
    """Vitesse médiane selon la direction autour du pit : un cercle = mouvement homogène."""
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(projection="polar")
    edges = np.linspace(-180, 180, 13)
    for g in groups:
        bdf = pm.binned(g.pa.steps, "angle", "v_win", edges, g.pa.steps["moving"].to_numpy(), 5)
        if len(bdf):
            th = np.radians(np.r_[bdf["center"], bdf["center"].iloc[0]])
            ax.plot(th, np.r_[bdf["median"], bdf["median"].iloc[0]], color=g.color, ls=g.linestyle, lw=1.8,
                    marker="o", ms=3, label=g.label)
    ax.set_theta_zero_location("E")
    ax.tick_params(labelsize=7)
    if len(groups) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12))
    style.panel_letter(ax, letter)
    return fig


def fig_tracks_categorical(pa: pm.ParticleAnalysis, background: np.ndarray | None, categories: pd.Series,
                           names: dict[str, str] | None = None, scalebar=None, pit_radii=None,
                           panel: str | None = None, figsize=(5.0, 4.3)):
    """Trajectoires colorées par catégorie (groupe de comportement, type de mouvement…)."""
    fig, ax = _fig(figsize)
    shape = background.shape[:2] if background is not None else None
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=_extent(shape, pa.scale), interpolation="nearest")
    S = pa.steps[pa.steps["particle"].isin(categories.index)]
    cats = [c for c in pd.unique(categories) if isinstance(c, str)]
    for i, c in enumerate(sorted(cats)):
        ids = categories.index[categories == c]
        segs = [np.column_stack([d["xs"], d["ys"]]) * pa.scale for _, d in S[S["particle"].isin(ids)].groupby("particle")]
        col = cmp.GROUP_STYLES[i % len(cmp.GROUP_STYLES)][0]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=1.2, label=f"{(names or {}).get(c, c)} ({len(ids)})"))
    _draw_pits(ax, pa.pits, pa.scale, pit_radii)
    if shape is not None:
        ext = _extent(shape, pa.scale)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    else:
        ax.autoscale()
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    _finish_image_axes(ax, pa.scale, shape or (0, 0), scalebar if shape else None, pa.length_unit, background)
    style.panel_label(ax, panel)
    return fig


# --------------------------------------------------------------------------- 3. Déplacement

def fig_displacement(groups, letter: str | None = None, figsize=(8.6, 3.5)):
    """(gauche) déplacement net, (droite) chemin parcouru, par traceur mobile."""
    _, L, _ = _units(groups)
    fig, (a, b) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for ax, col, name in ((a, "net_displacement", "Net displacement"), (b, "path_length", "Path length")):
        vals = [g.pa.particles.loc[g.pa.particles["mobile"], col].dropna().to_numpy() for g in groups]
        allv = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.array([1.0])
        bins = np.linspace(0, np.percentile(allv, 99), 30)
        for g, v in zip(groups, vals):
            if v.size:
                ax.hist(v, bins, density=True, histtype="step", color=g.color, ls=g.linestyle, lw=1.8, label=g.label)
                ax.axvline(np.median(v), color=g.color, ls=":", lw=1.2)
        ax.set_xlabel(style.label(name, L))
        ax.set_ylabel("Probability density")
    if len(groups) > 1:
        b.legend(loc="upper right")
    style.panel_letter(a, letter)
    return fig


def fig_direction_rose(groups, letter: str | None = None, figsize=(4.4, 4.2)):
    """Direction du déplacement net par rapport au pit (0° en haut = s'éloigne du pit)."""
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(projection="polar")
    edges = np.radians(np.linspace(-180, 180, 25))
    for g in groups:
        d = np.radians(g.pa.particles.loc[g.pa.particles["mobile"], "dir_angle"].dropna())
        if len(d):
            h, _ = np.histogram(d, bins=edges)
            h = h / h.sum()
            th = np.r_[0.5 * (edges[:-1] + edges[1:]), 0.5 * (edges[0] + edges[1])]
            ax.plot(th, np.r_[h, h[0]], color=g.color, ls=g.linestyle, lw=1.8, label=g.label)
            ax.fill(th, np.r_[h, h[0]], color=g.color, alpha=0.12)
    ax.set_theta_zero_location("N")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["away", "", "toward", ""], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    if len(groups) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12))
    style.panel_letter(ax, letter)
    return fig


def fig_radial_tangential(groups, letter: str | None = None, figsize=(4.8, 4.2)):
    """Composantes radiale et tangentielle de la vitesse de chaque traceur (en mouvement)."""
    vu, _, _ = _units(groups)
    fig, ax = _fig(figsize)
    lim = 0.0
    for g in groups:
        P = g.pa.particles[g.pa.particles["mobile"]]
        ax.scatter(P["v_radial_moving"], P["v_tangential_moving"], s=9, color=g.color, alpha=0.6, lw=0, label=g.label)
        if len(P):
            lim = max(lim, float(np.nanpercentile(np.abs(P[["v_radial_moving", "v_tangential_moving"]]), 99)))
    lim = lim or 1.0
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(style.label("Radial velocity (> 0 away)", vu))
    ax.set_ylabel(style.label("Tangential velocity (> 0 ccw)", vu))
    _finish(ax, letter)
    return fig


def fig_alpha(groups, alphas: dict[str, pd.DataFrame], letter: str | None = None, figsize=(5.0, 3.8)):
    """Exposant α du MSD de chaque traceur : < 0.7 confiné, 0.7–1.3 diffusif, > 1.3 dirigé."""
    fig, ax = _fig(figsize)
    bins = np.linspace(0, 2.5, 26)
    ax.axvspan(0, 0.7, color="0.9", lw=0)
    ax.axvspan(1.3, 2.5, color="0.9", lw=0)
    for x, txt in ((0.35, "confined"), (1.0, "diffusive"), (1.9, "directed")):
        ax.text(x, 0.98, txt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="0.35")
    for g in groups:
        a = alphas.get(g.label)
        if a is not None and len(a):
            ax.hist(np.clip(a["alpha"], 0, 2.5), bins, density=True, histtype="step", color=g.color,
                    ls=g.linestyle, lw=1.8, label=f"{g.label} (n = {len(a)})")
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("MSD exponent α (per tracer)")
    ax.set_ylabel("Probability density")
    _finish(ax, letter)
    return fig


def fig_turning_angles(groups, letter: str | None = None, figsize=(5.0, 3.6)):
    """Angle entre deux pas successifs : pic à 0° = mouvement dirigé ; plat = diffusion."""
    fig, ax = _fig(figsize)
    bins = np.linspace(-180, 180, 37)
    for g in groups:
        a = q.turning_angles(g.pa)
        if a.size:
            ax.hist(a, bins, density=True, histtype="step", color=g.color, ls=g.linestyle, lw=1.8, label=g.label)
    ax.axhline(1 / 360, color="0.5", lw=0.8, ls=":")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(style.label("Turning angle", "°"))
    ax.set_ylabel("Probability density")
    _finish(ax, letter)
    return fig


# --------------------------------------------------------------------------- 4. Début du mouvement

def fig_onset_hist(groups, rel_pit: bool = True, letter: str | None = None, figsize=(5.0, 3.6)):
    """Instants de démarrage (traceurs vus immobiles puis en mouvement)."""
    _, _, T = _units(groups)
    fig, ax = _fig(figsize)
    vals = {}
    for g in groups:
        P = g.pa.particles
        v = P.loc[P["status"] == "démarre", "onset_time"].dropna().to_numpy()
        tp = g.pa.params.t_pit
        vals[g.label] = v - tp if rel_pit and tp is not None else v
    allv = np.concatenate([v for v in vals.values() if v.size]) if any(v.size for v in vals.values()) else np.array([0.0, 1.0])
    bins = np.linspace(allv.min(), allv.max() + 1e-9, 30)
    for g in groups:
        v = vals[g.label]
        if v.size:
            ax.hist(v, bins, histtype="step", color=g.color, ls=g.linestyle, lw=1.8, label=f"{g.label} (n = {v.size})")
    if rel_pit:
        ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(style.label("Onset time after pit appearance" if rel_pit else "Onset time", T))
    ax.set_ylabel("Tracers")
    _finish(ax, letter)
    return fig


def fig_onset_distance(groups, letter: str | None = None, figsize=(5.0, 3.8)):
    """Démarrage selon la distance initiale au pit, avec la droite de propagation (Theil-Sen)."""
    vu, L, T = _units(groups)
    fig, ax = _fig(figsize)
    for g in groups:
        P = g.pa.particles
        d = P[P["status"] == "démarre"]
        ax.scatter(d["r0"], d["onset_time"], s=8, color=g.color, alpha=0.5, lw=0)
        fv = pm.front_velocity(P)
        lab = g.label
        if fv:
            xs = np.linspace(d["r0"].min(), d["r0"].max(), 50)
            lab += f": ρ = {fv['rho']:.2f}, p = {fv['p']:.1g}"
            if fv["velocity"] and fv["p"] < 0.05:
                lab += f", front {fv['velocity']:.3g} {vu}"
            ax.plot(xs, fv["intercept"] + fv["slope"] * xs, color=g.color, ls=g.linestyle, lw=1.8)
        ax.plot([], [], color=g.color, ls=g.linestyle, lw=1.8, label=lab)
    ax.set_xlabel(style.label("Initial distance from pit", L))
    ax.set_ylabel(style.label("Onset time", T))
    ax.legend(loc="upper left", fontsize=7)
    style.panel_letter(ax, letter)
    return fig


def fig_onset_map(pa: pm.ParticleAnalysis, background: np.ndarray | None, scalebar=None, pit_radii=None,
                  panel: str | None = None, figsize=(5.0, 4.3)):
    """Position de départ de chaque traceur, colorée par son instant de démarrage."""
    fig, ax = _fig(figsize)
    shape = background.shape[:2] if background is not None else None
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=255, extent=_extent(shape, pa.scale), interpolation="nearest")
    d = pa.particles[pa.particles["status"] == "démarre"]
    sc = ax.scatter(d["x0_px"] * pa.scale, d["y0_px"] * pa.scale, c=d["onset_time"], cmap=style.TRACK_CMAP, s=22,
                    edgecolors="black", linewidths=0.4, zorder=4)
    _draw_pits(ax, pa.pits, pa.scale, pit_radii)
    if shape is not None:
        ext = _extent(shape, pa.scale)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    if len(d):
        style.colorbar(fig, sc, ax, style.label("Onset time", pa.time_unit), float(d["onset_time"].min()),
                       float(d["onset_time"].max()))
    _finish_image_axes(ax, pa.scale, shape or (0, 0), scalebar if shape else None, pa.length_unit, background)
    style.panel_label(ax, panel)
    return fig


def fig_started_fraction(groups, rel_pit: bool = True, letter: str | None = None, figsize=(5.0, 3.6)):
    """Part cumulée des traceurs mobiles déjà en mouvement au cours du temps."""
    _, _, T = _units(groups)
    fig, ax = _fig(figsize)
    for g in groups:
        sf = q.started_fraction(g.pa)
        if len(sf):
            ax.step(_t(g, sf["t"].to_numpy(), rel_pit), 100 * sf["fraction"], where="post", color=g.color,
                    ls=g.linestyle, lw=1.8, label=g.label)
    if rel_pit:
        ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_ylim(-3, 103)
    ax.set_xlabel(style.label("Time since pit appearance" if rel_pit else "Time", T))
    ax.set_ylabel(style.label("Tracers already started", "%"))
    _finish(ax, letter)
    return fig


# --------------------------------------------------------------------------- 5. Relation avec le pit

def fig_pit_kinetics(times: np.ndarray, signal: np.ndarray, radius: pd.DataFrame, phases: dict, scale: float,
                     length_unit: str, time_unit: str, letter: str | None = None, figsize=(5.4, 4.6)):
    """(haut) intensité autour du pit, (bas) rayon du pit, phases avant / croissance / après."""
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained")
    a.plot(times, signal, color="black", lw=1.3)
    a.set_ylabel("Relative intensity\naround pit")
    if len(radius):
        b.plot(radius["t"], radius["R_px"] * scale, "o", ms=3, color="0.6")
        b.plot(radius["t"], radius["R_smooth"] * scale, color=PHASE_COLORS["growth"], lw=1.8)
    b.set_ylabel(style.label("Pit radius", length_unit))
    b.set_xlabel(style.label("Time", time_unit))
    for ax in (a, b):
        _shade_phases(ax, phases)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[n], alpha=0.3) for n in phases["phases"]
               if n != "whole sequence"]
    if handles:
        a.legend(handles, [PHASE_NAMES[n] for n in phases["phases"] if n != "whole sequence"], loc="best", fontsize=7)
    style.panel_letter(a, letter)
    return fig


def fig_phase_timeseries(pa: pm.ParticleAnalysis, phases: dict, letter: str | None = None, figsize=(5.4, 4.6)):
    """Vitesse et vitesse radiale des traceurs au cours du temps, phases du pit en fond."""
    vu = style.vel_unit(pa.length_unit, pa.time_unit)
    pop = pa.population
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained")
    if len(pop):
        a.fill_between(pop["t"], pop["speed_q25"], pop["speed_q75"], color="#2B5C7E", alpha=0.15, lw=0)
        a.plot(pop["t"], pop["speed_median"], color="#2B5C7E", lw=1.5)
        b.plot(pop["t"], pop["v_radial_median"], color="#E8423F", lw=1.4)
    b.axhline(0, color="0.5", lw=0.8)
    a.set_ylabel(style.label("Particle speed", vu))
    b.set_ylabel(style.label("Radial velocity", vu))
    b.set_xlabel(style.label("Time", pa.time_unit))
    for ax in (a, b):
        _shade_phases(ax, phases)
    style.panel_letter(a, letter)
    return fig


def fig_distance_phases(pa: pm.ParticleAnalysis, phases: dict, normalized: bool = False, letter: str | None = None,
                        figsize=(5.0, 3.9)):
    """Vitesse selon la distance au pit, par phase, en log-log, avec la loi de puissance ajustée."""
    vu = style.vel_unit(pa.length_unit, pa.time_unit)
    fig, ax = _fig(figsize)
    for name, rng in phases["phases"].items():
        fit = q.distance_fit(pa, rng, normalized)
        if fit is None:
            continue
        bdf, c = fit["bins"], PHASE_COLORS.get(name, "0.4")
        ax.plot(bdf["center"], bdf["median"], "o", color=c, ms=4)
        xs = np.geomspace(bdf["center"].min(), bdf["center"].max(), 30)
        ax.plot(xs, fit["prefactor"] * xs ** (-fit["n"]), color=c, lw=1.6,
                label=f"{PHASE_NAMES.get(name, name)}: n = {fit['n']:.2f} [{fit['n_ci'][0]:.2f}, {fit['n_ci'][1]:.2f}]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")          # « 5 » plutôt que « 5×10⁰ » sur un axe log court
    two_five = FuncFormatter(lambda v, _: f"{v:g}" if v > 0 and round(v / 10 ** np.floor(np.log10(v))) in (2, 5)
                             else "")
    ax.autoscale_view()
    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        axis.set_major_formatter(plain)
        # axe court : graduations secondaires chiffrées (toutes sous une décade, 2 et 5 sous deux décades)
        ratio = hi / lo if lo > 0 else np.inf
        axis.set_minor_formatter(plain if ratio < 10 else two_five if ratio < 100 else NullFormatter())
    ax.set_xlabel("Distance from pit, r / R" if normalized else style.label("Distance from pit", pa.length_unit))
    ax.set_ylabel(style.label("Particle speed", vu))
    ax.legend(loc="lower left", fontsize=7, title="v ∝ r⁻ⁿ", title_fontsize=7)
    style.panel_letter(ax, letter)
    return fig


def fig_radial_map(pa: pm.ParticleAnalysis, phases: dict | None = None, n_t: int = 30, n_r: int = 12,
                   letter: str | None = None, figsize=(5.4, 3.8)):
    """Vitesse radiale des traceurs en mouvement selon la distance au pit et le temps."""
    S = pa.steps[pa.steps["moving"] & np.isfinite(pa.steps["v_radial"])]
    vu = style.vel_unit(pa.length_unit, pa.time_unit)
    fig, ax = _fig(figsize)
    if len(S) >= 20:
        te = np.linspace(S["t"].min(), S["t"].max() + 1e-9, n_t + 1)
        re = np.linspace(0, S["r"].quantile(0.98), n_r + 1)
        it = np.clip(np.digitize(S["t"], te) - 1, 0, n_t - 1)
        ir = np.digitize(S["r"], re) - 1
        ok = (ir >= 0) & (ir < n_r)
        Z = np.full((n_r, n_t), np.nan)
        df = pd.DataFrame({"it": it[ok], "ir": ir[ok], "v": S["v_radial"].to_numpy()[ok]})
        med = df.groupby(["ir", "it"])["v"].agg(["median", "size"])
        med = med[med["size"] >= 5]
        Z[med.index.get_level_values(0), med.index.get_level_values(1)] = med["median"]
        lim = float(np.nanpercentile(np.abs(Z), 98)) if np.isfinite(Z).any() else 1.0
        pm_ = ax.pcolormesh(te, re, np.ma.masked_invalid(Z), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim or -1, lim or 1))
        style.colorbar(fig, pm_, ax, style.label("Radial velocity", vu), show_max=False)
    _shade_phases(ax, {**phases, "phases": {}} if phases else None)
    ax.set_xlabel(style.label("Time", pa.time_unit))
    ax.set_ylabel(style.label("Distance from pit", pa.length_unit))
    style.panel_letter(ax, letter)
    return fig


# --------------------------------------------------------------------------- 6. Traceurs / comparaison

def fig_indicator_strips(groups: dict[str, list[pm.ParticleAnalysis]], styles: dict[str, tuple], unit: str,
                         time_unit: str, letter: str | None = None, figsize=(9.0, 5.4)):
    """Une case par grandeur : valeurs par traceur (points) et médiane de chaque répétition
    (gros losanges) pour chaque groupe — la reproductibilité se lit dans la dispersion des losanges."""
    metrics = [("moving_speed", style.label("Speed while moving", unit), False),
               ("straightness", "Straightness", False), ("persistence", "Direction persistence", False),
               ("onset_rel_pit", style.label("Onset after pit", time_unit), False),
               ("away", style.label("Moving away from pit", "%"), True), ("mobile", style.label("Mobile tracers", "%"), True)]
    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    rng = np.random.default_rng(0)
    labels = list(groups)
    for ax, (m, name, is_share) in zip(axes.ravel(), metrics):
        for i, lab in enumerate(labels):
            col = styles[lab][0]
            reps = groups[lab]
            if is_share:
                vals = [100 * q.indicators(a)[m] for a in reps]
                ax.bar(i, np.nanmean(vals) if vals else np.nan, width=0.55, color=col, alpha=0.35)
                ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)), vals, marker="D", s=30,
                           color=col, edgecolors="black", linewidths=0.6, zorder=3)
            else:
                per = [q.tracer_values(a)[m].to_numpy(dtype=float) for a in reps]
                allv = np.concatenate(per) if per else np.array([])
                allv = allv[np.isfinite(allv)]
                ax.scatter(i + rng.uniform(-0.22, 0.22, allv.size), allv, s=4, color=col, alpha=0.3, lw=0)
                meds = [np.nanmedian(p) for p in per if np.isfinite(p).any()]
                ax.scatter(np.full(len(meds), i) + rng.uniform(-0.08, 0.08, len(meds)), meds, marker="D", s=34,
                           color=col, edgecolors="black", linewidths=0.6, zorder=3)
                if allv.size:
                    ax.hlines(np.median(allv), i - 0.3, i + 0.3, color="black", lw=1.5, zorder=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(name)
    style.panel_letter(axes[0, 0], letter)
    return fig


def fig_aligned_population(groups: dict[str, list[pm.ParticleAnalysis]], styles: dict[str, tuple], unit: str,
                           time_unit: str, letter: str | None = None, figsize=(5.0, 3.8)):
    """Vitesse médiane des traceurs alignée sur l'apparition du pit : une courbe fine par
    répétition, la moyenne des répétitions en trait épais."""
    fig, ax = _fig(figsize)
    for lab, reps in groups.items():
        col, ls = styles[lab]
        curves = []
        for a in reps:
            pop, tp = a.population, a.params.t_pit
            if not len(pop):
                continue
            t = pop["t"].to_numpy() - (tp if tp is not None else 0.0)
            ax.plot(t, pop["speed_median"], color=col, lw=0.7, alpha=0.4)
            curves.append((t, pop["speed_median"].to_numpy()))
        if curves:
            lo = max(c[0].min() for c in curves)
            hi = min(c[0].max() for c in curves)
            if hi > lo:
                grid = np.linspace(lo, hi, 80)
                mean = np.nanmean([np.interp(grid, t, v) for t, v in curves], axis=0)
                ax.plot(grid, mean, color=col, ls=ls, lw=2.4, label=f"{lab} mean ({len(curves)})")
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(style.label("Time since pit appearance", time_unit))
    ax.set_ylabel(style.label("Particle speed (median)", unit))
    ax.set_ylim(bottom=0)
    _finish(ax, letter)
    return fig
