"""Tests sur séquences synthétiques dont la vitesse est connue.

Lancer : python -m pytest -q  (depuis la racine du projet)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrovel import dataset, field, flow, io, plots, tracking, video  # noqa: E402

PIT = (100.0, 90.0)
SPEED_PX = 2.0          # px par frame
DT = 0.5                # s entre frames d'origine
PIX = 0.8               # µm/px


def radial_particles_sequence(n_frames=30, h=180, w=200, seed=0):
    """Particules claires éjectées radialement depuis PIT à vitesse constante."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = 1000 + 80 * rng.standard_normal((h, w))
    n_p = 14
    ang = np.linspace(0, 2 * np.pi, n_p, endpoint=False) + 0.1
    r0 = rng.uniform(12, 20, n_p)
    frames = []
    for t in range(n_frames):
        img = bg.copy()
        r = r0 + SPEED_PX * t
        for a, rr in zip(ang, r):
            cx, cy = PIT[0] + rr * np.cos(a), PIT[1] + rr * np.sin(a)
            if 5 < cx < w - 5 and 5 < cy < h - 5:
                img += 1500 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2))
        img += 20 * rng.standard_normal((h, w))
        frames.append(np.clip(img, 0, 65535).astype(np.uint16))
    return frames


def write_folder(frames, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        tifffile.imwrite(folder / f"img_{i}.tif", f)   # noms non paddés : teste le tri naturel
    return folder


def test_natural_sort_and_source(tmp_path):
    frames = radial_particles_sequence(12)
    src = io.source_from_folder(write_folder(frames, tmp_path / "raw"))
    assert len(src) == 12
    assert [f.name for f in src.files][:3] == ["img_0.tif", "img_1.tif", "img_2.tif"]
    assert src.files[10].name == "img_10.tif"
    np.testing.assert_array_equal(src.get(10), frames[10])


def test_stack_source(tmp_path):
    frames = radial_particles_sequence(6)
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, np.stack(frames))
    src = io.source_from_folder(p)
    assert src.kind == "stack" and len(src) == 6
    np.testing.assert_array_equal(src.get(4), frames[4])


def test_select_indices():
    assert dataset.select_indices(100, 10, 20, 5) == [10, 15, 20]
    assert dataset.select_indices(100, 0, 9, 1, n_target=4) == [0, 3, 6, 9]
    with pytest.raises(ValueError):
        dataset.select_indices(10, 5, 3)
    with pytest.raises(ValueError):
        dataset.select_indices(10, 0, 3, n_target=5)


def make_derived(tmp_path, step=1):
    frames = radial_particles_sequence(30)
    src = io.source_from_folder(write_folder(frames, tmp_path / "raw"))
    roi = dataset.ROI(20, 10, 190, 170)
    idx = dataset.select_indices(len(src), 0, 29, step)
    folder = dataset.create_dataset(src, tmp_path, "test ds", roi, idx, DT, "s", PIX)
    return frames, roi, idx, dataset.DerivedDataset(folder)


def test_calibration_fix_and_flags(tmp_path):
    """Calibration corrigée sans régénérer : temps recalculés, historique gardé, alertes justes."""
    frames = radial_particles_sequence(6)
    src = io.source_from_folder(write_folder(frames, tmp_path / "raw"))
    roi = dataset.ROI(0, 0, src.width, src.height)
    folder = dataset.create_dataset(src, tmp_path, "cal", roi, [0, 2, 4], 1.0, "s", None)
    ds = dataset.DerivedDataset(folder)
    issues = " ".join(ds.calibration_issues())
    assert "taille de pixel inconnue" in issues and "valeur par défaut" in issues
    ds.set_calibration(0.48, 0.1, "s", confirmed=True, note="objectif 20x")
    ds2 = dataset.DerivedDataset(folder)
    assert ds2.pixel_size == 0.48 and ds2.length_unit == "µm" and ds2.calibration_issues() == []
    np.testing.assert_allclose(ds2.times, [0.0, 0.2, 0.4])          # indices d'origine 0, 2, 4 × 0.1 s
    hist = ds2.meta["calibration"]["history"]
    assert hist[-1]["dt_original"] == 1.0 and hist[-1]["pixel_size"] is None
    np.testing.assert_array_equal(ds2.get(1), ds.get(1))             # frames inchangées
    with pytest.raises(ValueError):
        ds2.set_calibration(0.48, 0.0, "s", True)


def test_square_roi_and_pit(tmp_path):
    """Carré centré sur le pit (décalé près d'un bord, côté borné à l'image), et pit enregistré
    dans les métadonnées, en coordonnées du dataset."""
    r = dataset.square_roi(100, 90, 40, 200, 180)
    assert (r.width, r.height) == (40, 40)
    # pit au centre, à ½ px près (côté pair : le centre tombe entre deux pixels)
    assert abs((r.x0 + r.x1 - 1) / 2 - 100) <= 0.5 and abs((r.y0 + r.y1 - 1) / 2 - 90) <= 0.5
    edge = dataset.square_roi(5, 90, 40, 200, 180)
    assert edge.x0 == 0 and edge.width == 40                                 # décalé, reste dedans
    assert dataset.square_roi(100, 90, 500, 200, 180).width == 180           # côté borné
    frames = radial_particles_sequence(6)
    src = io.source_from_folder(write_folder(frames, tmp_path / "raw"))
    roi = dataset.square_roi(PIT[0], PIT[1], 60, src.width, src.height)
    folder = dataset.create_dataset(src, tmp_path, "carre", roi, list(range(6)), DT, "s", PIX, pit=PIT)
    ds = dataset.DerivedDataset(folder)
    assert ds.get(0).shape == (60, 60)
    assert ds.pit == (PIT[0] - roi.x0, PIT[1] - roi.y0)
    assert all(abs(c - 29.5) <= 0.5 for c in ds.pit)                         # centre du carré 60 px


def test_create_dataset(tmp_path):
    frames, roi, idx, ds = make_derived(tmp_path, step=2)
    assert ds.name == "test_ds"
    assert len(ds) == len(idx) == 15
    np.testing.assert_array_equal(ds.get(3), frames[idx[3]][roi.y0:roi.y1, roi.x0:roi.x1])
    assert ds.get(0).dtype == np.uint16
    meta = json.loads((ds.folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["times"][1] == pytest.approx(2 * DT)
    with pytest.raises(FileExistsError):
        dataset.create_dataset(io.source_from_folder(tmp_path / "raw"), tmp_path, "test ds",
                               roi, idx, DT, "s", PIX)


def test_trackpy_recovers_radial_speed(tmp_path):
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    p = tracking.TrackpyParams(polarity="bright", background="median", diameter=7, minmass=50,
                               search_range=5, memory=0, min_length=4)
    res, tracks, samples = field.run_trackpy(stack, ds.times, p, cell=20, window=5, stride=5,
                                             min_count=1, pits=[pit], scale=ds.scale,
                                             length_unit=ds.length_unit, time_unit=ds.time_unit)
    expected = SPEED_PX * PIX / DT          # µm/s
    assert tracks["particle"].nunique() >= 10
    med = float(np.median(np.hypot(samples["u"], samples["v"]) * PIX))
    assert med == pytest.approx(expected, rel=0.05)
    assert np.nanmedian(res.series["mean_v_radial"]) == pytest.approx(expected, rel=0.1)
    assert np.isfinite(res.u).any()
    # Les vecteurs de la grille pointent vers l'extérieur du pit
    vr = res.radial_velocity()
    assert np.nanmedian(vr) > 0.8 * expected
    edges, prof = res.radial_profile(8)
    assert prof.shape == (len(res.windows), 8)
    df = res.to_long_dataframe()
    assert len(df) == len(res.windows) * len(res.x) * len(res.y)
    assert set(df["nearest_pit"].unique()) == {0}   # un seul pit : toujours le pit 0


def test_radial_velocity_nearest_pit():
    """Avec deux pits éloignés, v_r de chaque cellule doit se calculer par rapport au pit le
    plus proche, pas toujours le premier (découpage à la Voronoï)."""
    x = np.array([2.0, 10.0, 20.0, 28.0])
    y = np.array([5.0])
    # Vitesse construite pour "s'éloigner du pit le plus proche" partout : +1 côté pit0 (x=0),
    # -1 côté pit1 (x=30) -> v_r doit valoir +1 pour les 4 cellules si l'assignation est correcte.
    u = np.array([[[1.0, 1.0, -1.0, -1.0]]])
    v = np.zeros_like(u)
    count = np.ones_like(u, dtype=int)
    res = field.FieldResult(
        method="optical_flow", x=x, y=y, windows=[(0, 1)],
        t_start=np.array([0.0]), t_end=np.array([1.0]),
        u=u, v=v, count=count, scale=1.0, length_unit="px", time_unit="s",
        pits=[(0.0, 5.0), (30.0, 5.0)], cell=8, image_shape=(10, 30),
    )
    vr = res.radial_velocity()
    np.testing.assert_allclose(vr[0, 0], [1.0, 1.0, 1.0, 1.0], atol=1e-9)
    df = res.to_long_dataframe()
    assert list(df["nearest_pit"]) == [0, 0, 1, 1]
    edges, prof = res.radial_profile(4)
    assert prof.shape == (1, 4)
    assert np.nanmean(prof) == pytest.approx(1.0, abs=1e-9)


def test_optical_flow_recovers_translation():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((260, 260))
    from scipy.ndimage import gaussian_filter, shift
    base = gaussian_filter(base, 3)
    shift_px = 1.5
    frames = np.stack([shift(base, (0, shift_px * t), order=3, mode="wrap") for t in range(6)])[:, 30:-30, 30:-30]
    times = np.arange(6) * DT
    res = field.run_flow(frames.astype(np.float32), times, flow.FlowParams("ilk", radius=7), cell=20,
                         window=5, stride=5, pits=[(100, 100)], scale=PIX, length_unit="µm", time_unit="s")
    inner_u = res.u[0, 2:-2, 2:-2]
    inner_v = res.v[0, 2:-2, 2:-2]
    assert np.median(inner_u) == pytest.approx(shift_px * PIX / DT, rel=0.05)
    assert abs(np.median(inner_v)) < 0.05 * shift_px * PIX / DT


def test_farneback_recovers_translation():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter, shift
    base = gaussian_filter(rng.standard_normal((260, 260)), 3)
    frames = np.stack([shift(base, (0.5 * t, 1.5 * t), order=3, mode="wrap") for t in range(4)])[:, 30:-30, 30:-30]
    res = field.run_flow(frames.astype(np.float32), np.arange(4) * DT, flow.FlowParams("farneback", radius=7),
                         cell=20, window=3, stride=3, pits=[(100, 100)], scale=PIX, length_unit="µm", time_unit="s")
    assert np.median(res.u[0, 2:-2, 2:-2]) == pytest.approx(1.5 * PIX / DT, rel=0.03)
    assert np.median(res.v[0, 2:-2, 2:-2]) == pytest.approx(0.5 * PIX / DT, rel=0.05)
    assert res.params["compute"]["workers"] == 1                     # OpenCV gère lui-même les cœurs


def test_regroup_matches_block_mean():
    """Regrouper des blocs de 2 px en cellules de k × 2 px = block_mean direct, bords partiels compris."""
    rng = np.random.default_rng(4)
    for h, w, cell in ((37, 51, 8), (40, 40, 10), (33, 29, 6)):
        a = rng.standard_normal((h, w))
        bm = flow.block_mean(a, 2)
        np.testing.assert_allclose(flow.regroup(bm, flow.block_weights(h, w, 2), cell // 2),
                                   flow.block_mean(a, cell), atol=1e-12)


def test_flow_parallel_and_cache(tmp_path):
    """Résultat identique en séquentiel, en parallèle et relu du cache (autre cellule, autre fenêtre)."""
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter, shift
    base = gaussian_filter(rng.standard_normal((150, 150)), 3)
    frames = np.stack([shift(base, (0, 1.2 * t), order=3, mode="wrap") for t in range(5)])[:, 20:-20, 20:-20]
    times = np.arange(5) * DT
    kw = dict(pits=[(50, 50)], scale=PIX, length_unit="µm", time_unit="s")
    p = flow.FlowParams("ilk", radius=5)
    seq = field.run_flow(frames.astype(np.float32), times, p, 10, 2, 2, workers=1, **kw)
    par = field.run_flow(frames.astype(np.float32), times, p, 10, 2, 2, workers=3, cache_dir=tmp_path, **kw)
    np.testing.assert_allclose(par.u, seq.u, atol=1e-9)
    assert par.params["compute"]["pairs_computed"] == 4 and len(list(tmp_path.rglob("pair_*.npz"))) == 4
    again = field.run_flow(frames.astype(np.float32), times, p, 20, 4, 4, cache_dir=tmp_path, **kw)
    assert again.params["compute"] == {**again.params["compute"], "pairs_cached": 4, "pairs_computed": 0}
    fresh = field.run_flow(frames.astype(np.float32), times, p, 20, 4, 4, workers=1, **kw)
    np.testing.assert_allclose(again.u, fresh.u, atol=2e-3 * np.abs(fresh.u).max())   # cache en float16
    assert np.median(again.u) == pytest.approx(1.2 * PIX / DT, rel=0.05)


def test_block_mean_partial():
    a = np.arange(25, dtype=float).reshape(5, 5)
    b = flow.block_mean(a, 2)
    assert b.shape == (3, 3)
    assert b[0, 0] == pytest.approx(np.mean([0, 1, 5, 6]))
    assert b[2, 2] == pytest.approx(24)


def test_grid_consistency():
    g = field.Grid(50, 45, 10)
    assert (g.ny, g.nx) == (5, 5)
    assert g.x[0] == pytest.approx(4.5) and g.x[-1] == pytest.approx(42.0)
    ix, iy = g.index(np.array([0.0, 9.4, 9.6, 44.0]), np.array([0.0, 0, 0, 49]))
    assert list(ix) == [0, 0, 1, 4] and iy[-1] == 4


def test_locate_all_parallel_matches_sequential():
    """locate_all doit donner le même résultat (même détections par frame) qu'on passe par le
    pool de process (>= _PARALLEL_MIN_FRAMES) ou la boucle séquentielle — preuve que le
    réordonnancement sous ProcessPoolExecutor ne change rien au résultat."""
    n = max(tracking._PARALLEL_MIN_FRAMES + 4, 12)
    frames = radial_particles_sequence(n_frames=n, h=100, w=120)
    stack = np.stack(frames).astype(np.float32)
    feat = tracking.feature_stack(stack, "bright", "median")
    p = tracking.TrackpyParams(diameter=7, minmass=50.0)

    seq_parts = tracking._locate_all_sequential(feat, p)
    seq_df = pd.concat(seq_parts, ignore_index=True) if seq_parts else pd.DataFrame()

    par_df = tracking.locate_all(feat, p)

    assert len(par_df) == len(seq_df) > 0
    seq_counts = seq_df.groupby("frame").size().sort_index()
    par_counts = par_df.groupby("frame").size().sort_index()
    pd.testing.assert_series_equal(seq_counts, par_counts)
    np.testing.assert_allclose(sorted(seq_df["x"]), sorted(par_df["x"]))


def test_plots_and_gif(tmp_path):
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    p = tracking.TrackpyParams(minmass=50, min_length=4)
    res, tracks, _ = field.run_trackpy(stack, ds.times, p, 20, 5, 5, 1, [pit], ds.scale, "µm", "s")
    bgs = [io.to_display(stack[a:b + 1].mean(0)) for a, b in res.windows]
    for mode in ("image", "speed", "none"):
        f = plots.fig_quiver(res, 0, bgs[0], mode, plots.global_vmax(res))
        assert len(plots.fig_to_bytes(f, "png", 100)) > 1000
        assert len(plots.fig_to_bytes(f, "svg")) > 1000
    for f in (plots.fig_series(res, 1), plots.fig_radial_profile(res),
              plots.fig_tracks(tracks, bgs[0], ds.scale, "µm", [pit])):
        assert len(plots.fig_to_bytes(f, "pdf")) > 1000
    gif = plots.make_gif(res, bgs, "image", plots.global_vmax(res), 1.0)
    assert gif[:3] == b"GIF"


def test_field_video(tmp_path):
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    p = tracking.TrackpyParams(minmass=50, min_length=4)
    res, _, _ = field.run_trackpy(stack, ds.times, p, 20, 5, 5, 1, [pit], ds.scale, "µm", "s")
    bgs = [io.to_display(stack[a:b + 1].mean(0)) for a, b in res.windows]
    mp4 = video.field_video_bytes(res, bgs, "image", plots.global_vmax(res), 1.0, fps=3, fmt="mp4")
    assert mp4[4:8] == b"ftyp" and len(mp4) > 500
    gif = video.field_video_bytes(res, bgs, "image", plots.global_vmax(res), 1.0, fps=3, fmt="gif")
    assert gif[:6] == b"GIF89a" and len(gif) > 500


def test_sequence_video(tmp_path):
    _, _, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    mp4 = video.sequence_video_bytes(stack, ds.times, ds.time_unit, fps=5, fmt="mp4")
    assert mp4[4:8] == b"ftyp" and len(mp4) > 500
    gif = video.sequence_video_bytes(stack, ds.times, ds.time_unit, fps=5, fmt="gif", show_time=False)
    assert gif[:6] == b"GIF89a" and len(gif) > 500
    rng = io.display_range("real", ds.get(0).dtype, stack[:, ::4, ::4])
    frames = video.sequence_video_frames(stack, ds.times, ds.time_unit, show_time=False, rng=rng)
    np.testing.assert_array_equal(frames[3][..., 0], io.to_display(stack[3], rng=rng))     # pas de retouche par frame


def test_display_modes_keep_real_values():
    """Affichage « réel » : niveaux proportionnels aux valeurs enregistrées, identiques pour toutes
    les frames ; « fixe » : un seul étirement pour toute la séquence ; « auto » : étirement par
    image, qui efface les vraies différences de luminosité entre images."""
    u8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    np.testing.assert_array_equal(io.to_display(u8, rng=io.display_range("real", np.uint8, u8)), u8)
    assert io.display_range("real", np.uint16, np.array([0.0, 4000.0])) == (0.0, 4095.0)      # caméra 12 bits
    assert io.display_range("real", np.uint16, np.array([0.0, 63000.0])) == (0.0, 65535.0)
    assert io.display_range("auto", np.uint16, np.array([0.0, 1.0])) is None
    rng_ = np.random.default_rng(0)
    bright = (50000 + 2000 * rng_.standard_normal((40, 40))).astype(np.uint16)
    dark = (bright * 0.9).astype(np.uint16)                                              # 10 % plus sombre
    for mode in ("real", "fixed"):
        r = io.display_range(mode, np.uint16, np.concatenate([bright.ravel(), dark.ravel()]).astype(float))
        assert io.to_display(dark, rng=r).mean() < io.to_display(bright, rng=r).mean() - 5   # visible
    auto_b, auto_d = io.to_display(bright).mean(), io.to_display(dark).mean()
    assert abs(auto_b - auto_d) < 3                                                      # effacée par l'auto


def test_figure_style_units_and_labels():
    from corrovel import style
    assert style.vel_unit("µm", "s") == "µm s⁻¹"
    assert style.unit(("particles", 1), ("µm", -2)) == "particles µm⁻²"
    assert style.unit(("µm", 2)) == "µm²"
    assert style.label("Particle speed", "µm s⁻¹") == "Particle speed / (µm s⁻¹)"
    assert style.label("Tracks") == "Tracks"
    assert "DejaVu Sans" in style.FONTS                                  # secours pour « ⁻ »


def test_speed_tracks_reference_figure(tmp_path):
    """Figure de référence : segments colorés par la vitesse, barre « Particle speed / (µm s⁻¹) »
    (sans glyphe manquant), étiquette de panneau, contour du pit en pointillés, cercle + flèche."""
    import warnings
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    from corrovel import particles as pm, pplots
    from corrovel.scalebar import ScaleBar
    tracks, _ = synthetic_tracer_tracks()
    times = np.arange(60, dtype=float)
    pa = pm.analyze(tracks, times, [(100.0, 100.0)], 0.5, "µm", "s", pm.MotionParams())
    bg = np.full((200, 200), 200, np.uint8)
    fig = pplots.fig_tracks_speed(pa, bg, ScaleBar(0.5), [12.0], panel="Pit A", min_displacement=2.0)
    ax, cax = fig.axes[0], fig.axes[1]
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    assert len(lc.get_segments()) > 100 and np.isfinite(lc.get_array()).all()
    assert cax.get_ylabel() == "Particle speed / (µm s⁻¹)"
    assert any(t.get_text() == "Pit A" for t in ax.texts)
    assert any(isinstance(p, Circle) and p.get_linestyle() != "-" for p in ax.patches)   # pointillés
    assert len(ax.get_xticks()) == 0                                     # panneau image, barre d'échelle
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        png = plots.fig_to_bytes(fig, "png", 80)
    assert len(png) > 1000 and not [x for x in w if "missing from font" in str(x.message)]
    # Mêmes trajectoires à partir des trajectoires brutes (page champ de vitesse)
    f2 = plots.fig_tracks(tracks, bg, 0.5, "µm", [(100.0, 100.0)], times=times, time_unit="s", pit_radii=[12.0])
    assert f2.axes[1].get_ylabel() == "Particle speed / (µm s⁻¹)"
    steps = pm.track_steps(tracks, times, 0.5)
    assert {"xs", "ys", "speed", "v_win"} <= set(steps.columns)
    assert np.nanmedian(steps.loc[steps["frame"] > 45, "v_win"]) == pytest.approx(1.0, rel=0.1)  # 2 px/s × 0.5


def test_piv_pipeline(tmp_path):
    """PIV de bout en bout sur la séquence radiale : vitesse radiale retrouvée, et toute la suite
    (figures, animation, exports) accepte le résultat."""
    from corrovel import interactive, piv
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    res = field.run_piv(stack, ds.times, piv.PIVParams(), cell=40, window=5, stride=5, pits=[pit], scale=ds.scale,
                        length_unit=ds.length_unit, time_unit=ds.time_unit, subtract_background=True)
    expected = SPEED_PX * PIX / DT
    assert res.method == "piv" and np.isfinite(res.u).any()
    assert np.nanmedian(res.radial_velocity()) > 0.7 * expected                         # s'éloigne du pit
    bgs = [io.to_display(stack[a:b + 1].mean(0)) for a, b in res.windows]
    vmax = plots.global_vmax(res)
    for mode in ("image", "speed", "contour", "none"):
        assert len(plots.fig_to_bytes(plots.fig_quiver(res, 0, bgs[0], mode, vmax), "png", 60)) > 1000
    assert len(plots.fig_to_bytes(plots.fig_series(res, 1), "png", 60)) > 1000
    assert len(interactive.field_animation_figure(res, bgs, "image", vmax, 1.0).frames) == len(res.windows)
    df = res.to_long_dataframe()
    assert len(df) == len(res.windows) * len(res.x) * len(res.y)
    assert "quality" in res.to_npz_dict()


def test_particle_density_exact():
    """2 particules par frame dans la cellule (0, 0), 1 seule dans la moitié des frames de la
    cellule (0, 1) : densité = particules / frame / aire, cellule tronquée du bord comprise."""
    res = field.FieldResult(
        method="trackpy", x=np.array([4.5, 12.0]), y=np.array([4.5]), windows=[(0, 3)],
        t_start=np.array([0.0]), t_end=np.array([3.0]),
        u=np.zeros((1, 1, 2)), v=np.zeros((1, 1, 2)), count=np.ones((1, 1, 2), dtype=int),
        scale=0.5, length_unit="µm", time_unit="s", pits=[(5.0, 5.0)], cell=10, image_shape=(10, 15),
    )
    rows = [(f, x, 3.0) for f in range(4) for x in (2.0, 6.0)]        # cellule 0 : 2 / frame
    rows += [(f, 12.0, 3.0) for f in (0, 2)]                          # cellule 1 : 0.5 / frame
    rows += [(9, 2.0, 3.0)]                                           # hors fenêtre : ignorée
    tracks = pd.DataFrame(rows, columns=["frame", "x", "y"])
    from corrovel import density
    d = density.particle_density(tracks, res, bin_px=10)
    np.testing.assert_allclose(d.mean_count[0], [[2.0, 0.5]])
    # aires : 10x10 px et 10x5 px (cellule tronquée), à 0.5 µm/px
    np.testing.assert_allclose(d.density[0], [[2.0 / 25.0, 0.5 / 12.5]])
    np.testing.assert_allclose(d.x_edges, [-0.5, 9.5, 14.5])
    smooth = density.particle_density(tracks, res, bin_px=10, sigma=1.0)
    assert smooth.density.shape == d.density.shape and np.isfinite(smooth.density).all()
    assert len(d.to_long_dataframe()) == 2


def test_density_figures_and_video(tmp_path):
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    p = tracking.TrackpyParams(minmass=50, min_length=4)
    res, tracks, _ = field.run_trackpy(stack, ds.times, p, 20, 5, 5, 1, [pit], ds.scale, "µm", "s")
    bgs = [io.to_display(stack[a:b + 1].mean(0)) for a, b in res.windows]
    from corrovel import density, interactive
    dens = density.particle_density(tracks, res, bin_px=15, sigma=0.5)
    assert dens.density.shape == (len(res.windows), dens.grid.ny, dens.grid.nx)
    assert dens.density.sum() > 0
    vmax = density.global_vmax(dens)
    for overlay in (True, False):
        f = plots.fig_density(dens, 0, bgs[0], vmax, overlay)
        assert len(plots.fig_to_bytes(f, "png", 100)) > 1000
        fig = interactive.density_animation_figure(dens, bgs if overlay else None, vmax)
        assert len(fig.frames) == len(res.windows)
    mp4 = video.density_video_bytes(dens, bgs, True, vmax, fps=3, fmt="mp4")
    assert mp4[4:8] == b"ftyp" and len(mp4) > 500
    gif = video.density_video_bytes(dens, None, False, vmax, fps=3, fmt="gif")
    assert gif[:6] == b"GIF89a" and len(gif) > 500


def test_fine_heatmap(tmp_path):
    """La somme glissante (vidéo) doit donner les mêmes cartes que le calcul direct (aperçu)."""
    from corrovel import density
    rng = np.random.default_rng(3)
    stack = rng.random((9, 20, 30)).astype(np.float32)
    base = stack.__getitem__
    for trail in (0, 1, 4):
        maps = list(density.accumulated_maps(base, len(stack), trail, 1.5))
        for i in (0, 3, 8):
            np.testing.assert_allclose(maps[i], density.accumulated_map(base, i, trail, 1.5), rtol=1e-9)
    np.testing.assert_allclose(density.accumulated_map(base, 8, 0, 0), stack.sum(0), rtol=1e-6)
    np.testing.assert_allclose(density.accumulated_map(base, 8, 3, 0), stack[6:9].sum(0), rtol=1e-6)
    # Positions Trackpy -> nb de particules par pixel
    tracks = pd.DataFrame({"frame": [0, 0, 1], "x": [2.2, 2.4, 29.9], "y": [5.0, 5.1, 19.6]})
    tb = density.track_frame_maps(tracks, (20, 30))
    assert tb(0)[5, 2] == 2 and tb(0).sum() == 2 and tb(1)[19, 29] == 1 and tb(5).sum() == 0
    vmax = density.heatmap_vmax(base, len(stack), 4, 1.5)
    assert vmax > 0
    rgb = video.heatmap_rgb(maps[-1], vmax, "jet")
    assert rgb.shape == (20, 30, 3) and rgb.dtype == np.uint8
    mp4 = video.heatmap_video_bytes(base, len(stack), 4, 1.5, vmax, "jet", np.arange(9.0), "s",
                                    show_time=True, fps=5, fmt="mp4")
    assert mp4[4:8] == b"ftyp"
    gif = video.heatmap_video_bytes(base, len(stack), 4, 1.5, vmax, fps=5, fmt="gif")
    assert gif[:6] == b"GIF89a"


def test_particle_signal_dark_bleaching():
    """Particules sombres sur fond clair qui pâlit (photoblanchiment) et pit qui grossit
    lentement : seule la particule mobile doit être détectée, pas le fond ni le pit."""
    from corrovel import density
    rng = np.random.default_rng(5)
    n, h, w = 30, 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for t in range(n):
        img = 50000 * (1 - 0.005 * t) + 30 * rng.standard_normal((h, w))     # fond qui pâlit
        img[np.hypot(xx - 20, yy - 30) < 5 + 0.1 * t] -= 20000                 # pit qui grossit
        cx = 40 + t                                                             # particule mobile
        img[np.hypot(xx - cx, yy - 15) < 2.5] -= 15000
        frames.append(img)
    stack = np.stack(frames).astype(np.float32)
    assert density.guess_polarity(stack[0]) == "dark"
    sig = density.particle_signal_stack(stack, "dark", "rolling", 3.0, binary=True)
    assert sig.dtype == bool and sig.shape == stack.shape
    k = 15
    assert sig[k, 15, 40 + k]                       # particule détectée à sa position
    assert sig[k][np.hypot(xx - 20, yy - 30) < 4].mean() < 0.05    # intérieur du pit : ignoré
    far = (np.abs(yy - 15) > 6) & (np.hypot(xx - 20, yy - 30) > 12)       # loin particule et pit
    assert sig[k][far].mean() < 0.01                                        # fond : (quasi) rien
    sig_c = density.particle_signal_stack(stack, "dark", "global", 3.0, binary=False)
    assert sig_c.dtype == np.float16 and float(sig_c[k, 15, 40 + k]) > 0


def test_scalebar():
    """Longueur automatique ronde, et barre blanche de la bonne longueur en pixels (avec et sans
    agrandissement de l'image)."""
    from corrovel.scalebar import ScaleBar, draw_on_rgb
    sb = ScaleBar(0.48)
    assert sb.length_for(862) == 50.0            # 20 % de 413.8 µm -> 82.8 -> 50
    assert ScaleBar(0.48, 30).length_for(862) == 30.0
    for up in (1, 2):
        img = np.zeros((400 * up, 600 * up, 3), np.uint8)
        out = draw_on_rgb(img, ScaleBar(0.5, 40, "bottom-left"), upscale=up)
        white = (out == 255).all(axis=2)
        rows = white.sum(axis=1)
        assert rows.max() == 80 * up               # 40 µm / 0.5 µm/px = 80 px d'origine
        assert white[: 200 * up].sum() == 0        # en bas : rien dans la moitié haute
    f = plots.fig_tracks(pd.DataFrame(columns=["particle", "frame", "x", "y"]), np.zeros((50, 60), np.uint8),
                         0.48, "µm", scalebar=sb)
    assert len(plots.fig_to_bytes(f, "svg")) > 1000


def _analytic_field(u_fn, v_fn, n=12, cell=2):
    x = np.arange(n) * cell + 0.5
    X, Y = np.meshgrid(x, x)
    u, v = u_fn(X, Y)[None], v_fn(X, Y)[None]
    return field.FieldResult(method="optical_flow", x=x, y=x, windows=[(0, 1)],
                             t_start=np.array([0.0]), t_end=np.array([1.0]), u=u, v=v,
                             count=np.ones_like(u, dtype=int), scale=1.0, length_unit="µm",
                             time_unit="s", pits=[(x.mean(), x.mean())], cell=cell,
                             image_shape=(n * cell, n * cell))


def test_contour_scalars():
    """Rotation solide horaire à l'écran (y vers le bas) : ω = -2a ; expansion : ∇·v = 2b ;
    remplissage des cellules vides ; niveaux ronds et symétriques."""
    a, b, c = 0.3, 0.2, 12.5
    rot = _analytic_field(lambda X, Y: -a * (Y - c), lambda X, Y: a * (X - c))
    w_ = rot.display_scalar(0, "vorticity")
    np.testing.assert_allclose(w_, -2 * a, atol=1e-9)
    np.testing.assert_allclose(rot.display_scalar(0, "divergence"), 0, atol=1e-9)
    exp = _analytic_field(lambda X, Y: b * (X - c), lambda X, Y: b * (Y - c))
    np.testing.assert_allclose(exp.display_scalar(0, "divergence"), 2 * b, atol=1e-9)
    assert exp.scalar_unit("divergence") == "1/s" and exp.scalar_unit("speed") == "µm/s"
    assert np.nanmin(exp.display_scalar(0, "v_radial")) > -1e-9          # tout s'éloigne du pit
    exp.u[0, 3, 4] = np.nan                                              # cellule vide
    assert np.isfinite(exp.display_scalar(0, "speed", fill=True)).all()
    assert np.isnan(exp.display_scalar(0, "speed", fill=False)[3, 4])
    lv = plots.ContourStyle("vorticity", levels=10).level_values(rot)
    assert lv[0] == -lv[-1] and 0 in lv                                  # symétrique, zéro inclus
    assert plots.nice_ref(7.3) == 5 and plots.nice_ref(0.034) == 0.02


def test_contour_article_style():
    """Style « référence » : y vers le haut (v_y change de signe, pas l'image), flèches lissées sur
    toute la grille, carte interpolée lisse, échelle ajustée aux données."""
    from corrovel import interactive
    exp = _analytic_field(lambda X, Y: 0.2 * (X - 12.5), lambda X, Y: 0.1 * (Y - 3.0))
    exp.u[0, 2, 3] = np.nan
    base = plots.ContourStyle("v", symmetric=False)
    up = plots.ContourStyle("v", symmetric=False, y_up=True, smooth_arrows=True, clean=True)
    np.testing.assert_allclose(up.scalar(exp, 0), -base.scalar(exp, 0))
    lo, hi = base.color_limits(exp)
    assert up.color_limits(exp) == pytest.approx((-hi, -lo)) and lo > -1.0       # pas symétrique
    assert up.y_display(np.array([0.0, 23.5]), 24, 1.0).tolist() == [23.5, 0.0]  # haut de l'image -> grand y
    U, V = up.vectors(exp, 0)
    assert np.isnan(U[2, 3]) and np.isfinite(np.delete(U.ravel(), 2 * U.shape[1] + 3)).all()   # vide reste vide
    filled = plots.ContourStyle("v", smooth_arrows=True, fill_gaps=1)
    assert np.isfinite(filled.vectors(exp, 0)[0][2, 3]) and filled.interpolated(exp, 0)[2, 3]  # interpolé, signalé
    z = exp.display_scalar(0, "u")
    xf, yf, zf = plots.upsample_field(exp.x, exp.y, z, 4)
    ok = np.isfinite(z)
    np.testing.assert_allclose(zf[::4, ::4][ok], z[ok], atol=1e-9)             # passe par les mesures
    assert np.isnan(zf[2 * 4, 3 * 4])                                          # cellule vide : reste vide
    xe, ye, _ = plots.upsample_field(exp.x, exp.y, z, 4, (-0.5, 23.5, -0.5, 23.5))
    assert (xe[0], xe[-1], ye[0], ye[-1]) == (-0.5, 23.5, -0.5, 23.5)           # couvre exactement la ROI
    f = plots.fig_quiver(exp, 0, None, "contour", 2.0, 1.0, contour=up)
    ax = f.axes[0]
    assert ax.get_title() == "" and ax.get_ylim()[0] == 0 and ax.get_ylim()[1] > 24 * 1.0
    fig = interactive.field_animation_figure(exp, None, "contour", 2.0, 1.0, contour=up)
    assert list(fig.layout.yaxis.range) == [0.0, 24.0] and len(fig.data) == 2   # contours + flèches, sans pit


def test_contour_mode_outputs(tmp_path):
    from corrovel import interactive
    exp = _analytic_field(lambda X, Y: 0.2 * (X - 12.5), lambda X, Y: 0.2 * (Y - 12.5))
    for q in plots.SCALAR_LABELS:
        cs = plots.ContourStyle(quantity=q, cmap="parula", arrow_step=2)
        f = plots.fig_quiver(exp, 0, None, "contour", 2.0, 1.0, contour=cs)
        assert len(plots.fig_to_bytes(f, "png", 80)) > 1000
        fig = interactive.field_animation_figure(exp, None, "contour", 2.0, 1.0, contour=cs)
        assert fig.data[0].type == "contour" and len(fig.layout.annotations) == 1
    gif = video.field_video_bytes(exp, None, "contour", 2.0, 1.0, fps=2, fmt="gif",
                                  contour=plots.ContourStyle("v_radial"))
    assert gif[:6] == b"GIF89a"


def synthetic_tracer_tracks(n=80, n_frames=60, pit=(100.0, 100.0), seed=0):
    """Traceurs immobiles (bruit de localisation 0.1 px) qui démarrent radialement à 2 px/frame,
    d'autant plus tard qu'ils sont loin du pit : démarrage à la frame 10 + r0 / 4 (front 4 px/frame)."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for pid in range(n):
        r0, ang = rng.uniform(20, 80), rng.uniform(0, 2 * np.pi)
        on = 10 + r0 / 4
        truth.append((pid, r0, on))
        for f in range(n_frames):
            r = r0 + 2.0 * max(0.0, f - on)
            x, y = pit[0] + r * np.cos(ang), pit[1] + r * np.sin(ang)
            rows.append((pid, f, x + 0.1 * rng.standard_normal(), y + 0.1 * rng.standard_normal()))
    return (pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]),
            pd.DataFrame(truth, columns=["particle", "r0_px", "onset_frame"]))


def test_particle_analysis_recovers_truth():
    """Échelle 0.5 µm/px, 1 s/frame : vitesse vraie 1 µm/s, front 2 µm/s, tous s'éloignent du pit."""
    from corrovel import particles as pm
    tracks, truth = synthetic_tracer_tracks()
    times = np.arange(60, dtype=float)
    pa = pm.analyze(tracks, times, [(100.0, 100.0)], 0.5, "µm", "s", pm.MotionParams())
    P = pa.particles.merge(truth, on="particle")
    assert P["mobile"].all() and (P["status"] == "démarre").all()
    assert np.median(np.abs(P["onset_time"] - P["onset_frame"])) <= 1.5        # démarrage à ±1-2 frames
    assert P["moving_speed"].median() == pytest.approx(1.0, rel=0.1)          # 2 px/frame × 0.5 µm/px
    assert (P["cos_radial"] > 0.95).all()                                      # tous s'éloignent
    assert P["r0"].corr(P["r0_px"] * 0.5) > 0.999
    fv = pm.front_velocity(pa.particles)
    assert fv["p"] < 1e-6 and fv["velocity"] == pytest.approx(2.0, rel=0.15)  # 4 px/frame × 0.5
    assert pa.population["moving_fraction"].iloc[0] == 0 and pa.population["moving_fraction"].iloc[-1] == 1
    # MSD : quasi immobile avant tout démarrage (α ≈ 0), mouvement dirigé à la fin (α ≈ 2)
    _, fit_before = pm.msd(tracks, times, 0.5, (0, 14))
    _, fit_after = pm.msd(tracks, times, 0.5, (35, 59))
    assert fit_before["alpha"] < 0.3 and fit_after["alpha"] == pytest.approx(2.0, abs=0.15)
    text = " ".join(pm.summary(pa, {"avant": fit_before, "après": fit_after}))
    assert "100 % des traceurs mobiles s'éloignent" in text and "démarrent plus tard" in text


def test_particle_figures():
    from corrovel import particles as pm, pplots
    tracks, _ = synthetic_tracer_tracks()
    times = np.arange(60, dtype=float)
    pa = pm.analyze(tracks, times, [(100.0, 100.0)], 0.5, "µm", "s", pm.MotionParams(t_pit=8.0))
    msds = {"avant le pit": pm.msd(tracks, times, 0.5, (0, 8)), "après le pit": pm.msd(tracks, times, 0.5, (8, 59))}
    bg = np.full((200, 200), 128, np.uint8)
    figs = [pplots.fig_population(pa), pplots.fig_r_t(pa), pplots.fig_onset(pa), pplots.fig_distributions(pa),
            pplots.fig_space(pa), pplots.fig_direction(pa), pplots.fig_msd(msds, "µm", "s"),
            pplots.fig_tracking_quality(tracks, times, "s")]
    figs += [pplots.fig_tracks_colored(pa, bg, c) for c in pplots.COLOR_BY]
    for f in figs:
        assert len(plots.fig_to_bytes(f, "png", 60)) > 1000
    sens = pm.threshold_sensitivity(tracks, times, [(100.0, 100.0)], 0.5, "µm", "s", pa.params, pa.threshold)
    assert list(sens["facteur"]) == [0.5, 1.0, 2.0] and (sens["% s'éloignent"] > 95).all()


def test_particle_analysis_from_advected_optical_flow_field():
    """Flux optique : pas de trajectoires réelles, mais advect_tracers en reconstruit à partir du
    champ dense, et toute la chaîne d'analyse par traceur (conçue pour Trackpy) doit tourner dessus
    sans modification — c'est le point du design : advect_tracers produit le même format que
    tracking.link_tracks."""
    from corrovel import particles as pm, pplots
    pit = (0.0, 100.0)
    x, y = np.arange(0, 200, 10, dtype=float), np.arange(0, 200, 10, dtype=float)
    X, Y = np.meshgrid(x, y)
    nt = 20
    r = np.hypot(X - pit[0], Y - pit[1])
    r_safe = np.where(r > 0, r, 1.0)
    u = np.broadcast_to(((X - pit[0]) / r_safe) * 1.5, (nt,) + X.shape).copy()   # 1.5 µm/s radial, sortant
    v = np.broadcast_to(((Y - pit[1]) / r_safe) * 1.5, (nt,) + X.shape).copy()
    count = np.ones_like(u, dtype=int)
    res = field.FieldResult(method="optical_flow", x=x, y=y, windows=[(i, i + 1) for i in range(nt)],
                            t_start=np.arange(nt, dtype=float), t_end=np.arange(1, nt + 1, dtype=float),
                            u=u, v=v, count=count, scale=1.0, length_unit="µm", time_unit="s",
                            pits=[pit], cell=10, image_shape=(200, 200))

    tracks = field.advect_tracers(res, seed_step=1, fill_gaps=0, substeps=2)
    assert set(tracks.columns) == {"particle", "frame", "x", "y"}

    times = np.arange(nt + 1, dtype=float)
    pa = pm.analyze(tracks, times, [pit], res.scale, res.length_unit, res.time_unit,
                    pm.MotionParams(min_points=3))
    assert len(pa.particles) > 0
    mob = pa.particles[pa.particles["mobile"]]
    assert len(mob) > 0
    assert (mob["cos_radial"] > 0.9).mean() > 0.9             # champ radial sortant : quasi tous s'éloignent

    msds = {"tout": pm.msd(tracks, times, res.scale)}
    bg = np.full(res.image_shape, 128, np.uint8)
    figs = [pplots.fig_population(pa), pplots.fig_r_t(pa), pplots.fig_distributions(pa),
           pplots.fig_space(pa), pplots.fig_direction(pa), pplots.fig_msd(msds, "µm", "s"),
           pplots.fig_tracks_colored(pa, bg, "moving_speed")]
    for f in figs:
        assert len(plots.fig_to_bytes(f, "png", 60)) > 1000


def test_pit_time_detection():
    from corrovel import particles as pm
    n, h, w = 40, 50, 60
    rng = np.random.default_rng(2)
    stack = 1000 + 5 * rng.standard_normal((n, h, w))
    stack *= np.linspace(1, 0.85, n)[:, None, None]                           # photoblanchiment
    yy, xx = np.mgrid[:h, :w]
    for i in range(18, n):                                                     # le pit apparaît à i = 18
        stack[i][(xx - 30) ** 2 + (yy - 25) ** 2 < (2 + 0.3 * (i - 18)) ** 2] *= 0.3
    sig = pm.pit_signal(stack, (30, 25), radius=8)
    t, why = pm.detect_pit_time(sig, np.arange(n) * 2.0)
    assert t == pytest.approx(36.0, abs=2.0), why                              # frame 18 × 2 s
    t0, why0 = pm.detect_pit_time(np.ones(n), np.arange(n) * 1.0)
    assert t0 is None and "déjà présent" in why0
    early = np.concatenate([np.linspace(1, 0.4, 15), np.full(25, 0.4)])         # pit qui se forme dès t = 0
    t1, why1 = pm.detect_pit_time(early, np.arange(n) * 1.0)
    assert t1 == 0 and "déjà en formation" in why1
    # Rayon : pit sombre de 20 px entouré d'un halo qui remonte lentement jusqu'à 60 px
    img = np.full((200, 200), 1000.0)
    r = np.hypot(*np.mgrid[:200, :200] - 100.0)
    img[r < 60] = 1000 - 300 * (60 - r[r < 60]) / 40
    img[r < 20] = 300
    assert pm.estimate_pit_radius(img, (100, 100)) == pytest.approx(24, abs=3)   # bord 20 px + marge


def test_field_animation_figure(tmp_path):
    _, roi, _, ds = make_derived(tmp_path, step=1)
    stack = ds.gray_stack()
    pit = (PIT[0] - roi.x0, PIT[1] - roi.y0)
    p = tracking.TrackpyParams(minmass=50, min_length=4)
    res, _, _ = field.run_trackpy(stack, ds.times, p, 20, 5, 5, 1, [pit], ds.scale, "µm", "s")
    bgs = [io.to_display(stack[a:b + 1].mean(0)) for a, b in res.windows]
    from corrovel import interactive
    for mode in ("image", "speed", "none"):
        fig = interactive.field_animation_figure(res, bgs, mode, plots.global_vmax(res), 1.0)
        assert len(fig.frames) == len(res.windows)
