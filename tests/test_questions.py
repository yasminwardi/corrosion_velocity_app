"""Calculs et figures des sections « questions de l'étude » (corrovel.questions, compare, qplots) :
vérité synthétique connue."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrovel import compare as cmp  # noqa: E402
from corrovel import particles as pm  # noqa: E402
from corrovel import plots, qplots  # noqa: E402
from corrovel import questions as q  # noqa: E402

PIT = (100.0, 100.0)


def radial_tracks(n=60, n_frames=50, pit=PIT, speed=2.0, onset=5, seed=0, pid0=0, r_range=(20, 60),
                  speeds=None):
    """Traceurs qui partent radialement du pit à `speed` px/frame après la frame `onset`."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r0, ang = rng.uniform(*r_range), rng.uniform(0, 2 * np.pi)
        v = speed if speeds is None else speeds[i]
        for f in range(n_frames):
            r = r0 + v * max(0, f - onset)
            rows.append((pid0 + i, f, pit[0] + r * np.cos(ang) + 0.05 * rng.standard_normal(),
                         pit[1] + r * np.sin(ang) + 0.05 * rng.standard_normal()))
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


def analyze(tracks, pits=(PIT,), n_frames=50, **kw):
    return pm.analyze(tracks, np.arange(n_frames, dtype=float), list(pits), 0.5, "µm", "s", pm.MotionParams(**kw))


def test_tangential_velocity_on_swirl():
    """Rotation autour du pit à ω = 0.02 rad/frame, r = 40 px : |v_t| = ω r × 0.5 µm/px = 0.4 µm/s,
    v radiale nulle. Angle croissant dans les coordonnées image (y vers le bas) = horaire à l'écran
    → v tangentielle < 0 (convention : > 0 anti-horaire à l'écran)."""
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(30):
        a0 = rng.uniform(0, 2 * np.pi)
        for f in range(50):
            a = a0 + 0.02 * f
            rows.append((pid, f, PIT[0] + 40 * np.cos(a), PIT[1] + 40 * np.sin(a)))
    pa = analyze(pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]), threshold=0.1)
    mob = pa.particles[pa.particles["mobile"]]
    assert len(mob) == 30
    assert mob["v_tangential_moving"].median() == pytest.approx(-0.4, rel=0.05)
    assert mob["v_radial_moving"].abs().median() < 0.02
    d = q.displacement_summary(pa)
    assert d["tangential_share"][0] > 0.9


def test_track_alpha_separates_brownian_and_directed():
    rng = np.random.default_rng(3)
    rows = []
    for pid in range(40):                                       # browniens, D grand devant le bruit
        xy = np.cumsum(rng.normal(0, 1.5, (60, 2)), axis=0) + 100
        rows += [(pid, f, *xy[f]) for f in range(60)]
    for pid in range(40, 80):                                   # dirigés, 1.5 px/frame
        a = rng.uniform(0, 2 * np.pi)
        rows += [(pid, f, 100 + 1.5 * f * np.cos(a), 100 + 1.5 * f * np.sin(a)) for f in range(60)]
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    pa = analyze(tracks, n_frames=60)
    al = q.track_alpha(pa).set_index("particle")
    assert al.loc[al.index < 40, "alpha"].median() == pytest.approx(1.0, abs=0.25)
    assert (al.loc[al.index >= 40, "motion_type"] == "directed").all()
    assert (al.loc[al.index < 40, "motion_type"] == "diffusive").mean() > 0.7
    pers = q.persistence(pa, moving_only=False)       # diffusion : persistance nulle (pas de biais du lissage)
    assert pers[pers.index >= 40].median() > 0.95 and abs(pers[pers.index < 40].median()) < 0.1
    turn = q.turning_angles(pa, moving_only=False)
    assert turn.size > 1000


def test_behaviour_groups_homogeneous_vs_bimodal():
    pa1 = analyze(radial_tracks(n=80, speed=2.0, seed=4), threshold=0.2)
    bg1 = q.behaviour_groups(pa1)
    assert bg1 is not None and bg1["k"] == 1
    speeds = np.r_[np.full(50, 0.6), np.full(50, 4.0)]
    pa2 = analyze(radial_tracks(n=100, speeds=speeds, seed=5), threshold=0.1)
    bg2 = q.behaviour_groups(pa2)
    assert bg2["k"] >= 2
    lab = bg2["labels"]
    slow, fast = lab[lab.index < 50], lab[lab.index >= 50]
    assert (slow == "G1").mean() > 0.9 and (fast == "G1").mean() < 0.1            # G1 = le plus lent


def test_speed_since_first_mean_matches_truth():
    """Traceurs déjà en mouvement à leur première détection, vitesses 0.5…1.5 µm/s : la moyenne de la
    figure (b) vaut la moyenne des vitesses (1 µm/s) à chaque instant."""
    speeds = np.linspace(1.0, 3.0, 40)                     # px/frame -> 0.5…1.5 µm/s
    pa = analyze(radial_tracks(n=40, speeds=speeds, onset=0, seed=6), threshold=0.1)
    curves, mean = q.speed_since_first(pa)
    assert curves["particle"].nunique() == 40
    mid = mean[(mean["t_since_first"] >= 5) & (mean["t_since_first"] <= 40)]
    assert np.allclose(mid["mean"], 1.0, rtol=0.05)
    assert (mid["n"] == 40).all()


def test_pit_groups_and_subset():
    t1 = radial_tracks(n=30, pit=(60.0, 60.0), r_range=(10, 30), seed=7)
    t2 = radial_tracks(n=20, pit=(240.0, 60.0), r_range=(10, 30), seed=8, pid0=100)
    pa = analyze(pd.concat([t1, t2]), pits=[(60.0, 60.0), (240.0, 60.0)], threshold=0.2)
    G = cmp.styled(cmp.pit_groups(pa))
    assert [g.label for g in G] == ["Pit A", "Pit B"]
    assert [len(g.pa.particles) for g in G] == [30, 20]
    assert G[0].color == cmp.GROUP_STYLES[0][0] and G[1].linestyle == "--"
    assert set(G[1].pa.steps["particle"]) == set(range(100, 120))
    assert G[1].pa.population["n"].max() == 20
    named = cmp.pit_groups(pa, "NiCr")
    assert [g.label for g in named] == ["NiCr · Pit A", "NiCr · Pit B"]
    merged = cmp.merge([G[0].pa, G[1].pa])
    assert merged.particles["particle"].is_unique and len(merged.particles) == 50


def test_save_load_roundtrip(tmp_path):
    pa = analyze(radial_tracks(seed=9), threshold=0.2, t_pit=4.0)
    ds = tmp_path / "datasets" / "ds1"
    folder = cmp.save_analysis(pa, ds, "NiCr (+) 1.6 V", {"dataset": "ds1", "experiment": {"tracers": "positive"},
                                                          "pit_radii_px": [12.0]})
    listed = cmp.list_saved(tmp_path)
    assert len(listed) == 1 and listed[0]["path"] == str(folder) and listed[0]["label"] == "NiCr (+) 1.6 V"
    pa2, meta = cmp.load_analysis(folder)
    pd.testing.assert_frame_equal(pa2.particles, pa.particles)
    pd.testing.assert_frame_equal(pa2.steps, pa.steps)
    assert pa2.params == pa.params and pa2.threshold == pa.threshold and pa2.pits == pa.pits
    assert meta["experiment"]["tracers"] == "positive"
    pn = q.add_r_norm(pa2, meta["pit_radii_px"])
    assert np.allclose(pn.particles["r0_norm"], pa.particles["r0"] / (12.0 * 0.5))
    # « NiCr (+) » et « NiCr (−) » : dossiers distincts ; même nom réenregistré : même dossier
    other = cmp.save_analysis(pa, ds, "NiCr (−) 1.6 V", {"dataset": "ds1"})
    assert other != folder and cmp.save_analysis(pa, ds, "NiCr (+) 1.6 V", {"dataset": "ds1"}) == folder
    assert len(cmp.list_saved(tmp_path)) == 2


def test_compare_samples_and_ci():
    rng = np.random.default_rng(10)
    a, b = rng.normal(1.0, 0.3, 200), rng.normal(1.0, 0.3, 200)
    assert cmp.compare_samples({"a": a, "b": b})["p"] > 0.05
    t = cmp.compare_samples({"a": a, "b": b + 0.3})
    assert t["test"] == "Mann-Whitney" and t["p"] < 1e-6 and t["effect"] < -0.3
    assert cmp.compare_samples({"a": a, "b": b, "c": b + 1})["test"] == "Kruskal-Wallis"
    est, lo, hi = cmp.bootstrap_ci(rng.normal(5.0, 1.0, 400))
    assert lo < 5.0 < hi and hi - lo < 0.4
    p, lo, hi = cmp.proportion_ci(0, 20)
    assert p == 0 and lo == 0 and 0.1 < hi < 0.2                                 # Wilson : pas d'IC nul


def test_pit_phases_and_distance_fit():
    t = np.arange(60, dtype=float)
    R = np.clip((t - 10) * 1.5, 0, 30)                              # croissance de t = 10 à t = 30
    radius = pd.DataFrame({"t": t, "R_px": R, "R_smooth": R})
    ph = q.pit_phases(radius, 10.0, 0.0, 59.0, frac=0.9)
    assert ph["t_end_growth"] == pytest.approx(28.0, abs=1.0)
    assert list(ph["phases"]) == ["before", "growth", "after"]
    assert list(q.pit_phases(radius, None, 0.0, 59.0)["phases"]) == ["whole sequence"]
    # v = 50 r^-1.5 sur des pas « en mouvement »
    rng = np.random.default_rng(11)
    r = rng.uniform(5, 100, 4000)
    steps = pd.DataFrame({"particle": np.arange(4000) % 200, "frame": np.arange(4000) % 60, "t": t[np.arange(4000) % 60],
                          "r": r, "v_win": 50 * r ** -1.5 * np.exp(0.05 * rng.standard_normal(4000)), "moving": True})
    pa = pm.ParticleAnalysis(pm.MotionParams(), steps, pd.DataFrame(), pd.DataFrame(), 0.1, "", 1.0, "µm", "s", [PIT])
    fit = q.distance_fit(pa)
    assert fit["n"] == pytest.approx(1.5, abs=0.05) and fit["n_ci"][0] < 1.5 < fit["n_ci"][1]


def test_replicate_verdict():
    def rep(speed, seed):
        return analyze(radial_tracks(n=40, speed=speed, seed=seed), threshold=0.2, t_pit=4.0)
    fast = [rep(3.0, 20), rep(3.2, 21)]
    slow = [rep(1.5, 22), rep(1.6, 23)]
    v = q.replicate_verdict({"positive": fast, "negative": slow}, "moving_speed")
    assert v["verdict"] == "robust" and v["consistent"] is True
    v1 = q.replicate_verdict({"positive": fast[:1], "negative": slow[:1]}, "moving_speed")
    assert v1["verdict"] == "significant, not reproduced"                       # une seule répétition
    same = q.replicate_verdict({"a": [rep(2.0, 24), rep(2.0, 25)], "b": [rep(2.0, 26), rep(2.0, 27)]}, "moving_speed")
    assert same["verdict"] in ("no difference detected", "significant, not reproduced")
    assert q.fmt_ci((3.21, 2.9, 3.5)) == "3.21 [2.9 – 3.5]" and q.fmt_ci((np.nan, 1, 2)) == "—"


def test_question_figures_render():
    t1 = radial_tracks(n=40, pit=(60.0, 60.0), r_range=(10, 40), seed=12)
    t2 = radial_tracks(n=40, pit=(240.0, 60.0), r_range=(10, 40), seed=13, pid0=100, speed=1.0)
    pa = q.add_r_norm(analyze(pd.concat([t1, t2]), pits=[(60.0, 60.0), (240.0, 60.0)], threshold=0.2, t_pit=4.0),
                      [8.0, 8.0])
    G = cmp.styled(cmp.pit_groups(pa))
    alphas = {g.label: q.track_alpha(g.pa) for g in G}
    t = np.arange(50, dtype=float)
    R = np.clip((t - 4) * 1.0, 0, 20)
    radius = pd.DataFrame({"frame": t.astype(int), "t": t, "R_px": R, "R_smooth": R})
    phases = q.pit_phases(radius, 4.0, 0.0, 49.0)
    bg = np.full((150, 320), 128, np.uint8)
    cats = alphas["Pit A"].set_index("particle")["motion_type"]
    figs = [qplots.fig_speed_since_first(G, letter="(b)"), qplots.fig_speed_distribution(G),
            qplots.fig_speed_time(G), qplots.fig_speed_time(G, align_pit=True), qplots.fig_speed_distance(G, True),
            qplots.fig_moving_fraction(G), qplots.fig_angular_speed(G), qplots.fig_displacement(G),
            qplots.fig_direction_rose(G), qplots.fig_radial_tangential(G), qplots.fig_alpha(G, alphas),
            qplots.fig_turning_angles(G), qplots.fig_onset_hist(G), qplots.fig_onset_distance(G),
            qplots.fig_started_fraction(G), qplots.fig_onset_map(pa, bg),
            qplots.fig_tracks_categorical(pa, bg, cats, q.MOTION_TYPES),
            qplots.fig_pit_kinetics(t, np.linspace(1, 0.8, 50), radius, phases, 0.5, "µm", "s"),
            qplots.fig_phase_timeseries(pa, phases), qplots.fig_distance_phases(pa, phases),
            qplots.fig_radial_map(pa, phases),
            qplots.fig_indicator_strips({"positive": [G[0].pa], "negative": [G[1].pa]},
                                        {"positive": cmp.GROUP_STYLES[0], "negative": cmp.GROUP_STYLES[1]}, "µm s⁻¹", "s"),
            qplots.fig_aligned_population({"positive": [G[0].pa], "negative": [G[1].pa]},
                                          {"positive": cmp.GROUP_STYLES[0], "negative": cmp.GROUP_STYLES[1]}, "µm s⁻¹", "s")]
    legend = [tx.get_text() for tx in figs[0].axes[0].get_legend().get_texts()]
    assert legend == ["Pit A mean", "Pit B mean"]
    assert any(tx.get_text() == "(b)" for tx in figs[0].axes[0].texts)
    for f in figs:
        texts = [a.get_xlabel() + a.get_ylabel() for a in f.axes]
        assert not any(c in "".join(texts) for c in "éèàç"), texts                # figures en anglais
        assert len(plots.fig_to_bytes(f, "png", 50)) > 1000
        plt.close(f)
