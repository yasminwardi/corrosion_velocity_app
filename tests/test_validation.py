"""Validation NUMÉRIQUE : les méthodes retrouvent-elles des mouvements synthétiques connus ?

Distinct des tests logiciels (test_pipeline.py) : chaque test fabrique une séquence dont la vitesse,
la direction et les trajectoires sont connues, la fait analyser par la chaîne complète (Trackpy ou
flux optique -> champ sur grille -> grandeurs affichées) et compare au résultat attendu.
Ces tests ne disent rien de la validité sur les images réelles (validation expérimentale).

Lancer : python -m pytest -q tests/test_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrovel import field, flow, plots, tracking  # noqa: E402

PIX, DT = 0.5, 0.2                        # µm/px, s entre frames
TP = tracking.TrackpyParams(polarity="bright", background="none", diameter=7, minmass=50,
                            search_range=6, memory=1, min_length=3)


def particle_movie(vel, n_frames=12, h=160, w=200, n_p=70, seed=0, xlim=None, drop=0.0, return_pos=False):
    """Particules gaussiennes claires (σ 1.5 px) sur fond bruité ; vel(x, y) -> (dx, dy) px/frame.
    drop : probabilité qu'une particule soit invisible sur une frame (détection manquée)."""
    rng = np.random.default_rng(seed)
    x0, x1 = xlim or (12, w - 12)
    pos = np.column_stack([rng.uniform(x0, x1, n_p), rng.uniform(12, h - 12, n_p)])
    yy, xx = np.mgrid[0:h, 0:w]
    frames, truth = [], []
    for _ in range(n_frames):
        truth.append(pos.copy())
        img = 1000 + 20 * rng.standard_normal((h, w))
        for (px, py) in pos:
            if rng.random() >= drop and 4 < px < w - 4 and 4 < py < h - 4:
                sl = (slice(max(0, int(py) - 6), int(py) + 7), slice(max(0, int(px) - 6), int(px) + 7))
                img[sl] += 1500 * np.exp(-((xx[sl] - px) ** 2 + (yy[sl] - py) ** 2) / (2 * 1.5 ** 2))
        frames.append(img)
        d = np.array([vel(px, py) for px, py in pos])
        pos = pos + d
    stack = np.stack(frames).astype(np.float32)
    return (stack, truth) if return_pos else stack


def trackpy_field(stack, cell=40, pits=((100.0, 80.0),), times=None, scale=PIX, params=TP, min_count=3):
    times = np.arange(len(stack)) * DT if times is None else times
    res, tracks, _ = field.run_trackpy(stack, times, params, cell=cell, window=len(stack) - 1,
                                       stride=len(stack) - 1, min_count=min_count, pits=list(pits),
                                       scale=scale, length_unit="µm", time_unit="s")
    return res, tracks


def texture_movie(dx, dy, n=4, size=200, seed=1, flat_right=False):
    from scipy.ndimage import gaussian_filter, shift
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.standard_normal((size + 60, size + 60)), 3) * 50 + 1000
    frames = []
    for t in range(n):
        f = shift(base, (dy * t, dx * t), order=3, mode="wrap")[30:-30, 30:-30]
        if flat_right:
            f[:, size // 2:] = 1000.0
        frames.append(f + 2.0 * rng.standard_normal((size, size)))
    return np.stack(frames).astype(np.float32)


def flow_field(stack, cell=20, scale=PIX, dt=DT, algo="ilk", **kw):
    return field.run_flow(stack, np.arange(len(stack)) * dt, flow.FlowParams(algo, radius=7), cell,
                          len(stack) - 1, len(stack) - 1, pits=[(100, 100)], scale=scale, length_unit="µm",
                          time_unit="s", workers=1, **kw)


# --------------------------------------------------------------------------- Trackpy

def test_trackpy_uniform_translation():
    res, _ = trackpy_field(particle_movie(lambda x, y: (1.5, -0.8)))
    m = res.measured_mask(0)
    assert m.mean() > 0.8
    u, v = res.u[0][m], res.v[0][m]
    assert np.median(u) == pytest.approx(1.5 * PIX / DT, rel=0.02)           # 3.75 µm/s vers la droite
    assert np.median(v) == pytest.approx(-0.8 * PIX / DT, rel=0.03)          # 2 µm/s vers le HAUT (v < 0)
    # cellule par cellule : quelques erreurs de liaison (particules proches) -> dispersion réelle
    assert (np.abs(u / (1.5 * PIX / DT) - 1) < 0.10).mean() >= 0.85


def test_trackpy_radial_direction():
    c = (100.0, 80.0)

    def radial(x, y):
        r = np.hypot(x - c[0], y - c[1]) or 1.0
        return 1.2 * (x - c[0]) / r, 1.2 * (y - c[1]) / r
    res, _ = trackpy_field(particle_movie(radial, n_p=120), cell=32, pits=(c,))
    m = res.measured_mask(0)
    vr = res.radial_velocity()[0][m]
    sp = res.speed[0][m]
    assert (vr / sp > 0.9).mean() > 0.9                                      # flèches dirigées vers l'extérieur
    assert np.median(vr) == pytest.approx(1.2 * PIX / DT, rel=0.1)


def _recovered(tracks, truth, tol=1.5):
    """Fraction des vraies particules (frame 0) retrouvées à moins de tol px."""
    f0 = tracks[tracks["frame"] == 0][["x", "y"]].to_numpy()
    if len(f0) == 0:
        return 0.0
    d = np.hypot(truth[0][:, None, 0] - f0[None, :, 0], truth[0][:, None, 1] - f0[None, :, 1])
    return float((d.min(axis=1) < tol).mean())


def test_trackpy_zero_motion_and_median_background_pitfall():
    still, truth = particle_movie(lambda x, y: (0.0, 0.0), return_pos=True)
    res, tracks = trackpy_field(still)
    m = res.measured_mask(0)
    assert m.mean() > 0.8 and np.median(res.speed[0][m]) < 0.05 * PIX / DT   # bruit de localisation seulement
    assert _recovered(tracks, truth) > 0.7
    # Piège : un fond « médiane temporelle » contient les particules immobiles et les efface ; la
    # normalisation amplifie alors le bruit, que Trackpy détecte à la place des vraies particules
    from dataclasses import replace
    _, tr_med = trackpy_field(still, params=replace(TP, background="median"))
    assert _recovered(tr_med, truth) < 0.3


def test_trackpy_units_scale_with_displacement_dt_and_pixel():
    a, _ = trackpy_field(particle_movie(lambda x, y: (1.0, 0.0), seed=3))
    b, _ = trackpy_field(particle_movie(lambda x, y: (2.0, 0.0), seed=3))
    c, _ = trackpy_field(particle_movie(lambda x, y: (1.0, 0.0), seed=3), times=np.arange(12) * 2 * DT)
    d, _ = trackpy_field(particle_movie(lambda x, y: (1.0, 0.0), seed=3), scale=2 * PIX)
    ua = np.nanmedian(a.u[0])
    assert np.nanmedian(b.u[0]) == pytest.approx(2 * ua, rel=0.05)           # déplacement × 2 -> vitesse × 2
    assert np.nanmedian(c.u[0]) == pytest.approx(ua / 2, rel=0.05)           # Δt × 2 -> vitesse / 2
    assert np.nanmedian(d.u[0]) == pytest.approx(2 * ua, rel=0.05)           # µm/px × 2 -> vitesse × 2


def test_trackpy_sparse_region_not_invented():
    res, _ = trackpy_field(particle_movie(lambda x, y: (1.0, 0.5), xlim=(12, 90)), cell=40)
    m = res.measured_mask(0)
    assert m[:, :2].mean() > 0.7 and not m[:, -2:].any()                      # moitié droite : aucune particule
    z = res.display_scalar(0, "speed", fill=0, smooth=1.0)
    assert np.isnan(z[:, -2:]).all()                                         # rien d'inventé, même lissé
    zi = res.display_scalar(0, "speed", fill=1)
    holes = res.interpolated_mask(0, 1)
    assert not holes[:, -1].any() and np.array_equal(np.isfinite(zi), m | holes)   # 1 cellule max, signalée


def test_trackpy_missed_detections_with_memory():
    res, _ = trackpy_field(particle_movie(lambda x, y: (1.5, 0.0), drop=0.15, seed=5))
    m = res.measured_mask(0)
    assert np.median(res.u[0][m]) == pytest.approx(1.5 * PIX / DT, rel=0.05)


# --------------------------------------------------------------------------- flux optique

@pytest.mark.parametrize("algo", ["ilk", pytest.param("farneback", marks=pytest.mark.skipif(
    not flow.has_opencv(), reason="OpenCV absent"))])
def test_flow_uniform_translation_and_units(algo):
    a = flow_field(texture_movie(1.0, 0.4), algo=algo)
    b = flow_field(texture_movie(2.0, 0.8), algo=algo)
    m = a.measured_mask(0)
    assert m.mean() > 0.8                                    # texture faible : quelques cellules masquées
    assert np.nanmedian(a.u[0]) == pytest.approx(1.0 * PIX / DT, rel=0.05)
    assert np.nanmedian(a.v[0]) == pytest.approx(0.4 * PIX / DT, rel=0.08)       # vers le BAS : v > 0
    assert np.nanmedian(b.u[0]) == pytest.approx(2 * np.nanmedian(a.u[0]), rel=0.05)


def test_flow_zero_motion():
    r = flow_field(texture_movie(0.0, 0.0))
    assert r.measured_mask(0).mean() > 0.8
    assert np.nanmedian(r.speed[0]) < 0.08 * PIX / DT        # plancher de bruit < 0.08 px/frame


def test_flow_untextured_region_masked():
    r = flow_field(texture_movie(1.0, 0.0, flat_right=True))
    m = r.measured_mask(0)
    assert m[:, :4].mean() > 0.8 and not m[:, 6:].any()                        # fond uniforme : pas de mesure
    assert np.nanmedian(r.u[0][:, 1:4]) == pytest.approx(1.0 * PIX / DT, rel=0.05)


def test_flow_moving_blob_not_diluted_by_flat_background():
    """Une particule texturée qui se déplace sur un fond uniforme : la vitesse de sa cellule est
    celle de la particule (moyenne pondérée par la texture), pas une moyenne diluée par le fond."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:120, 0:120]
    blob = gaussian_filter(rng.standard_normal((40, 40)), 1.5) * 400
    frames = []
    for t in range(4):
        img = np.full((120, 120), 1000.0)
        x0 = 40 + 1.0 * t
        sl_y, sl_x = slice(40, 80), slice(int(x0), int(x0) + 40)
        from scipy.ndimage import shift
        img[sl_y, 30:90] += shift(np.pad(blob, ((0, 0), (10, 10))), (0, x0 - int(x0) + int(x0) - 30 - 10), order=3)[:, :60]
        frames.append(img + 2.0 * rng.standard_normal((120, 120)))
    r = flow_field(np.stack(frames).astype(np.float32), cell=60)
    cell = r.u[0][0:2, 0:2]
    assert np.nanmax(cell) == pytest.approx(1.0 * PIX / DT, rel=0.15)


def test_flow_static_texture_bias_and_background_subtraction():
    """Traceurs qui se déplacent devant une texture FIXE (rayures, bords) : sans soustraction du
    fond, le flux optique est tiré vers zéro ; avec soustraction, il retrouve la vitesse vraie.
    (Effet observé sur NiCr(+) : ×0.25 sans soustraction, ×0.85 avec, par rapport à Trackpy.)"""
    from scipy.ndimage import gaussian_filter
    movie = particle_movie(lambda x, y: (1.5, 0.0), n_p=120, seed=7)
    rng = np.random.default_rng(8)
    scratches = gaussian_filter(rng.standard_normal(movie.shape[1:]), (0.8, 6)) * 4000   # rayures fixes
    movie = movie + scratches[None]
    raw = flow_field(movie, cell=40)
    sub = flow_field(movie, cell=40, subtract_background=True)
    true = 1.5 * PIX / DT
    r_raw, r_sub = np.nanmedian(raw.u[0]) / true, np.nanmedian(sub.u[0]) / true
    assert r_raw < 0.85                                                         # biais vers zéro (ici ~0.75)
    assert abs(r_sub - 1) < 0.2 and abs(r_sub - 1) < abs(r_raw - 1)             # corrigé par la soustraction


def test_flow_sparse_tracers_not_diluted_in_time():
    """Traceurs clairsemés : une cellule n'est traversée que pendant quelques paires de la fenêtre.
    Sa vitesse doit être celle du traceur (moyenne temporelle pondérée par la texture), pas une
    moyenne diluée par les paires où rien n'y passait."""
    movie = particle_movie(lambda x, y: (2.0, 0.0), n_p=6, n_frames=12, seed=11)
    r = flow_field(movie, cell=40, subtract_background=True)
    m = r.measured_mask(0)
    assert m.any()
    assert np.nanmedian(r.u[0]) == pytest.approx(2.0 * PIX / DT, rel=0.2)


# --------------------------------------------------------------------------- PIV

def piv_field(stack, cell=20, scale=PIX, dt=DT, subtract_background=False, **kw):
    from corrovel import piv
    return field.run_piv(np.asarray(stack, dtype=np.float32), np.arange(len(stack)) * dt, piv.PIVParams(**kw),
                         cell, len(stack) - 1, len(stack) - 1, pits=[(100, 100)], scale=scale, length_unit="µm",
                         time_unit="s", subtract_background=subtract_background)


def test_piv_uniform_translation_and_units():
    """Texture lisse (cas difficile pour la PIV : pic de corrélation large). Sans correction du
    recouvrement, de la moyenne et sans ajustement sous-pixel adapté à la largeur du pic, ce cas
    était biaisé de ~40 % vers zéro : mêmes tolérances que le flux optique."""
    a = piv_field(texture_movie(1.0, 0.4))
    b = piv_field(texture_movie(2.0, 0.8))
    assert a.measured_mask(0).mean() > 0.7
    assert np.nanmedian(a.u[0]) == pytest.approx(1.0 * PIX / DT, rel=0.05)
    assert np.nanmedian(a.v[0]) == pytest.approx(0.4 * PIX / DT, rel=0.08)                # vers le BAS : v > 0
    assert np.nanmedian(b.u[0]) == pytest.approx(2 * np.nanmedian(a.u[0]), rel=0.05)
    assert a.method == "piv" and a.quality is not None


def test_piv_large_displacement_second_pass():
    """7 px par intervalle dans des cellules de 24 px : la 2e passe à fenêtre décalée retrouve le
    déplacement (la 1re seule perd des particules sorties de la fenêtre)."""
    r = piv_field(texture_movie(7.0, 3.0, n=5), cell=24, passes=2)
    assert r.measured_mask(0).mean() > 0.6
    assert np.nanmedian(r.u[0]) == pytest.approx(7.0 * PIX / DT, rel=0.02)
    assert np.nanmedian(r.v[0]) == pytest.approx(3.0 * PIX / DT, rel=0.02)


def test_piv_sparse_particles_ensemble_and_zero_motion():
    """Traceurs clairsemés : la corrélation d'ensemble (somme des plans de toutes les paires)
    retrouve la vitesse ; aucun mouvement -> vitesse nulle, pas de biais."""
    r = piv_field(particle_movie(lambda x, y: (1.5, -0.8), n_frames=12), cell=40)
    assert r.measured_mask(0).mean() > 0.8
    assert np.nanmedian(r.u[0]) == pytest.approx(1.5 * PIX / DT, rel=0.02)
    assert np.nanmedian(r.v[0]) == pytest.approx(-0.8 * PIX / DT, rel=0.03)
    z = piv_field(particle_movie(lambda x, y: (0.0, 0.0), n_frames=12), cell=40)
    assert abs(np.nanmedian(z.u[0])) < 0.02 * PIX / DT and abs(np.nanmedian(z.v[0])) < 0.02 * PIX / DT


def test_piv_untextured_region_masked():
    r = piv_field(texture_movie(1.0, 0.0, n=5, flat_right=True))
    nx = r.u.shape[2]
    assert not np.isfinite(r.u[0][:, nx // 2 + 1:]).any()                               # rien d'inventé
    assert np.isfinite(r.u[0][:, :nx // 2 - 1]).mean() > 0.7


def test_piv_static_texture_needs_background_subtraction():
    """Texture FIXE (rayures) : elle donne un pic de corrélation au déplacement nul qui écrase
    celui des traceurs (×0.17 ici) ; avec le fond soustrait, la vitesse vraie est retrouvée."""
    from scipy.ndimage import gaussian_filter
    movie = particle_movie(lambda x, y: (1.5, 0.0), n_p=120, seed=7)
    rng = np.random.default_rng(8)
    movie = movie + (gaussian_filter(rng.standard_normal(movie.shape[1:]), (0.8, 6)) * 4000)[None]
    true = 1.5 * PIX / DT
    raw = np.nanmedian(piv_field(movie, cell=40).u[0]) / true
    sub = np.nanmedian(piv_field(movie, cell=40, subtract_background=True).u[0]) / true
    assert raw < 0.5
    assert sub == pytest.approx(1.0, abs=0.05)


def test_piv_parallel_windows_match_sequential():
    """Les fenêtres calculées dans des processus séparés donnent exactement le même champ."""
    from corrovel import piv
    st = np.asarray(particle_movie(lambda x, y: (1.5, -0.8), n_frames=13), dtype=np.float32)
    wins = field.make_windows(len(st) - 1, 4, 4)
    par = piv.run_windows(st, wins, 40, piv.PIVParams())
    for (i0, i1), (dx, dy, ratio) in zip(wins, par):
        sdx, sdy, sratio = piv.piv_window(st, i0, i1, 40, piv.PIVParams())
        np.testing.assert_allclose(dx, sdx, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, sdy, rtol=1e-5, atol=1e-6)


def test_piv_median_test_removes_isolated_outlier():
    from corrovel import piv
    dx = np.full((5, 5), 2.0)
    dy = np.zeros((5, 5))
    dx[2, 2] = 6.0                                                                       # vecteur aberrant isolé
    out = piv.median_test_outliers(dx, dy, 2.0)
    assert out[2, 2] and out.sum() == 1


# --------------------------------------------------------------------------- advection (flux optique -> traceurs virtuels)

def test_advect_tracers_follows_uniform_flow_and_stops_at_no_data():
    """Un champ uniforme connu doit advecter les traceurs virtuels à la bonne vitesse ; un traceur
    qui entre dans une zone sans donnée doit s'arrêter là, pas continuer (jamais d'extrapolation)."""
    x, y = np.arange(10) * 10.0, np.arange(8) * 10.0
    nt = 6
    u = np.full((nt, 8, 10), 2.0)             # 2 unités/s vers la droite
    v = np.zeros((nt, 8, 10))
    u[:, :, 6:] = np.nan                      # moitié droite de la grille : pas de donnée
    v[:, :, 6:] = np.nan
    count = np.where(np.isfinite(u), 1, 0)
    res = field.FieldResult(method="optical_flow", x=x, y=y, windows=[(i, i + 1) for i in range(nt)],
                            t_start=np.arange(nt, dtype=float), t_end=np.arange(1, nt + 1, dtype=float),
                            u=u, v=v, count=count, scale=1.0, length_unit="px", time_unit="s",
                            pits=[(0.0, 0.0)], cell=10, image_shape=(80, 100))

    tracks = field.advect_tracers(res, seed_step=1, fill_gaps=0, substeps=2)
    assert list(tracks.columns) == ["particle", "frame", "x", "y"]

    # Traceur semé loin du bord (x=0) : traverse les 6 fenêtres, ~2 unités/s * 1 s par fenêtre
    # pleine (le tout premier pas ne couvre qu'une demi-fenêtre, voir advect_tracers).
    t0 = tracks[tracks["particle"] == 0].sort_values("frame")
    assert len(t0) == nt + 1
    np.testing.assert_allclose(np.diff(t0["x"].to_numpy())[1:], 2.0, atol=1e-9)
    assert (t0["y"] == 0.0).all()                                       # pas de dérive verticale

    # Traceur semé juste avant la frontière sans-donnée (x=50) : doit s'arrêter en entrant dedans,
    # pas continuer de l'autre côté.
    near = tracks[(tracks["particle"] != 0)]
    seed_x = near.groupby("particle")["x"].first()
    pid_edge = seed_x[seed_x == 50.0].index[0]
    tp = tracks[tracks["particle"] == pid_edge].sort_values("frame")
    assert tp["x"].max() < 60.0                                         # jamais entré dans la zone NaN
    assert len(tp) < nt + 1                                             # trajectoire interrompue plus tôt


# --------------------------------------------------------------------------- figures

def test_figure_orientation_and_common_scales():
    """y vers le bas (image) ou vers le haut (graphique) : un mouvement vers le bas de l'image est
    dessiné vers le bas dans les deux cas ; mêmes niveaux de couleur et même échelle de flèches pour
    deux datasets quand on les impose."""
    down, _ = trackpy_field(particle_movie(lambda x, y: (0.0, 1.0)))
    fast, _ = trackpy_field(particle_movie(lambda x, y: (0.0, 2.0), seed=4))
    for y_up in (False, True):
        cs = plots.ContourStyle("v", y_up=y_up, limits=(-10.0, 10.0))
        f = plots.fig_quiver(down, 0, None, "contour", 5.0, 1.0, contour=cs)
        ax = f.axes[0]
        q = next(c for c in ax.collections if c.__class__.__name__ == "Quiver")
        V = np.ma.filled(q.V, np.nan)
        inverted = ax.get_ylim()[0] > ax.get_ylim()[1]
        visually_down = (np.nanmedian(V) > 0) if inverted else (np.nanmedian(V) < 0)
        assert visually_down
        assert np.nanmedian(cs.scalar(down, 0)) == pytest.approx((-1 if y_up else 1) * PIX / DT, rel=0.1)
    cs = plots.ContourStyle("speed", limits=(0.0, 12.0))
    f1 = plots.fig_quiver(down, 0, None, "contour", 8.0, 1.0, contour=cs)
    f2 = plots.fig_quiver(fast, 0, None, "contour", 8.0, 1.0, contour=cs)
    q1 = next(c for c in f1.axes[0].collections if c.__class__.__name__ == "Quiver")
    q2 = next(c for c in f2.axes[0].collections if c.__class__.__name__ == "Quiver")
    assert q1.scale == q2.scale                                                 # même échelle de flèches
    np.testing.assert_array_equal(cs.level_values(down), cs.level_values(fast))  # mêmes niveaux de couleur
