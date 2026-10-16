"""Interface web : ROI, sélection de frames, datasets dérivés, champ de vitesse et vidéos.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import base64
import io as pyio
import json
import os
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

if not hasattr(st, "fragment"):          # streamlit < 1.37 : même fonction, ancien nom
    st.fragment = st.experimental_fragment

from corrovel import compare as cmpmod
from corrovel import dataset as dsm
from corrovel import density as densitymod
from corrovel import field as fieldmod
from corrovel import flow as flowmod
from corrovel import interactive as interactivemod
from corrovel import io as iom
from corrovel import particles as pm
from corrovel import piv as pivmod
from corrovel import plots, pplots, qplots, style, tracking
from corrovel import questions as qmod
from corrovel import video as videomod
from corrovel.scalebar import POSITIONS as SB_POSITIONS
from corrovel.scalebar import ScaleBar

APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("CORROVEL_DATA", APP_DIR / "data")).expanduser().resolve()
(DATA_ROOT / "raw").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "datasets").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Champ de vitesse – pitting corrosion", page_icon="🔬", layout="wide")

ss = st.session_state
TIME_UNITS = ["s", "min", "h", "frame"]

# Dashboard à 4 sections ; "Données" a trois sous-vues (préparation en 3 étapes internes) et
# "Analyse" une sous-vue par question de l'article (après ses réglages).
SECTIONS = ["📂 Données", "🌀 Champ de vitesse", "🔬 Analyse", "📤 Export"]
ANALYSIS_VIEWS = ["⚙️ Réglages", "Vitesse", "Homogénéité", "Déplacement", "Début du mouvement",
                  "Relation avec le pit", "Traceurs"]
SUBVIEWS = {"📂 Données": ["Import", "ROI & frames", "Datasets"], "🔬 Analyse": ANALYSIS_VIEWS}
MAX_ANIM_WINDOWS = 80   # au-delà, l'aperçu animé n'embarque plus d'image de fond (payload navigateur)
VIDEO_FORMATS = {"mp4": ["mp4"], "gif": ["gif"], "les deux": ["mp4", "gif"]}
VIDEO_MIME = {"mp4": "video/mp4", "gif": "image/gif"}
# Méthode choisie (radio) -> libellé ; méthode d'un résultat (FieldResult.method) -> libellé
METHOD_LABELS = {"trackpy": "Trackpy (suivi de particules)", "flow": "Flux optique dense",
                 "piv": "PIV (intercorrélation)"}
RESULT_LABELS = {"trackpy": "Trackpy", "optical_flow": "Flux optique", "piv": "PIV"}


# --------------------------------------------------------------------------- thème / CSS

def inject_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
    div[data-testid="stMetric"] {
        background: #F2F6F8; border: 1px solid rgba(43,92,126,.15);
        border-radius: 12px; padding: .9rem 1rem .6rem 1rem;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; opacity: .75; }
    div[data-testid="stRadio"] > div[role="radiogroup"] { gap: .4rem; flex-wrap: wrap; }
    div[data-testid="stRadio"] label {
        background: #F2F6F8; border: 1px solid #E1E8ED;
        padding: .42rem 1rem; border-radius: 999px; transition: all .15s ease;
    }
    div[data-testid="stRadio"] label:hover { border-color: #2B5C7E; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; }
    .corrovel-card {
        background: #F2F6F8; border: 1px solid rgba(43,92,126,.15);
        border-radius: 12px; padding: .6rem .9rem; margin-bottom: .55rem; font-size: .9rem;
    }
    .corrovel-card b { color: #2B5C7E; }
    .corrovel-hero {
        background: linear-gradient(135deg, #2B5C7E 0%, #3E7EA6 100%);
        color: white; border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
    }
    .corrovel-hero h1 { color: white; margin: 0 0 .2rem 0; font-size: 1.5rem; }
    .corrovel-hero p { margin: 0; opacity: .9; font-size: .92rem; }
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- état persistant
# Streamlit efface l'état d'un widget quand sa page n'est pas affichée. On garde donc la valeur
# dans une clé « persistante » et le widget utilise une clé préfixée par « _ ».

def _persist(key: str):
    ss[key] = ss["_" + key]


def w(key: str, default=None, clamp: tuple | None = None) -> dict:
    """Arguments de widget liés à une valeur persistante ss[key]."""
    if key not in ss:
        ss[key] = default
    if clamp is not None and ss[key] is not None:
        lo, hi = clamp
        ss[key] = type(ss[key])(min(max(ss[key], lo), hi))
    ss["_" + key] = ss[key]
    return {"key": "_" + key, "on_change": _persist, "args": (key,)}


def goto(section: str, subview: str | None = None):
    ss.section = section
    if section in SUBVIEWS:
        ss.subview = subview or SUBVIEWS[section][0]


def cached_fig(name: str, sig: tuple, build):
    """Garde la dernière figure `name` en session tant que `sig` ne change pas : les figures
    animées sont longues à construire et chaque interaction relance toute la page."""
    c = ss.get("_figcache_" + name)
    if c is None or c[0] != sig:
        c = (sig, build())
        ss["_figcache_" + name] = c
    return c[1]


def scalebar_ui(ds: dsm.DerivedDataset):
    """Réglages de la barre d'échelle (une fois par page). La taille de pixel vient uniquement de
    la calibration du dataset : une barre en µm sur des axes en px serait incohérente."""
    with st.expander("📏 Barre d'échelle et style des figures"):
        a, b = st.columns([2, 1])
        a.text_input("Étiquette de panneau (ex. « Pit A »)", **w("fig_panel", ""),
                     help="En gras gris, en haut à gauche des images (champ, trajectoires, densité), comme "
                          "dans les figures d'article. Vide : pas d'étiquette. Les textes des figures sont en anglais.")
        b.checkbox("Contour du pit", **w("fig_pit_outline", True),
                   help="Cercle rouge en pointillés au bord du pit (rayon estimé sur la dernière frame). "
                        "Décoché, ou bord non détecté : une croix au centre.")
        if not ds.pixel_size:
            st.caption("Pas de barre d'échelle : ce dataset n'a pas de taille de pixel. Renseigne la "
                       "calibration (encadré ⚠️ en haut de page).")
            return
        a, b, c = st.columns(3)
        a.checkbox("Afficher la barre d'échelle", **w("sb_on", True))
        b.number_input("Longueur (µm), 0 = auto", 0.0, step=10.0, format="%.4g", **w("sb_len", 0.0),
                       help="Auto : valeur ronde (1, 2, 5, 10, 20, 50…) proche de 20 % de la largeur.")
        c.selectbox("Position", list(SB_POSITIONS), format_func=SB_POSITIONS.get,
                    **w("sb_pos", "bottom-right"))
        st.caption(f"Calibration du dataset : {ds.pixel_size:g} µm/px.")


def current_scalebar(ds: dsm.DerivedDataset) -> ScaleBar | None:
    if not ss.get("sb_on", True) or not ds.pixel_size:
        return None
    return ScaleBar(float(ds.pixel_size), float(ss.get("sb_len") or 0.0), ss.get("sb_pos", "bottom-right"))


def current_panel() -> str | None:
    t = str(ss.get("fig_panel") or "").strip()
    return t or None


def figure_pit_radii(path: str, mtime: float, pits) -> list:
    """Rayon du contour de chaque pit (px) pour les figures ; vide si le contour est désactivé."""
    if not ss.get("fig_pit_outline", True):
        return []
    return [pit_radius_cached(path, mtime, (float(x), float(y))) for x, y in pits]


def invalidate_results(ds: dsm.DerivedDataset):
    """Après un changement de calibration : les résultats calculés avec l'ancienne ne sont plus
    valables (unités, temps). Le cache du flux optique, en px par paire, reste valable."""
    r = ss.get("result")
    if r and r.get("ds_path") == str(ds.folder):
        ss.pop("result", None)
    for k in [k for k in ss.keys() if str(k).startswith("_figcache_")]:
        ss.pop(k, None)
    for k in ("export_zip", "pa_zip", "field_video", "density_video", "heatmap_video", "seq_preview"):
        ss.pop(k, None)


def calibration_box(ds: dsm.DerivedDataset, expanded: bool | None = None):
    """État de la calibration du dataset et correction (unique endroit où on la modifie)."""
    issues = ds.calibration_issues()
    m = ds.meta
    state = (f"{ds.pixel_size:g} µm/px" if ds.pixel_size else "µm/px inconnu") + \
            f", Δt = {m['dt_original']:g} {ds.time_unit} entre frames d'origine"
    title = ("⚠️ Calibration à vérifier — " if issues else "✅ Calibration confirmée — ") + state
    with st.expander(title, expanded=bool(issues) if expanded is None else expanded):
        for msg in issues:
            st.warning(msg)
        if issues:
            st.caption("Toutes les vitesses, distances et temps de l'analyse en dépendent. Indique les valeurs de "
                       "l'acquisition (microscope / caméra) ; les frames ne sont pas modifiées.")
        a, b, c = st.columns(3)
        k = ds.name
        pix = a.number_input("Taille de pixel (µm/px), 0 = inconnue", 0.0, format="%.6g",
                             value=float(ds.pixel_size or 0.0), key=f"cal_px_{k}")
        dt = b.number_input("Intervalle entre frames d'origine", 1e-9, format="%.6g",
                            value=float(m["dt_original"]), key=f"cal_dt_{k}")
        tu = c.selectbox("Unité de temps", TIME_UNITS, index=TIME_UNITS.index(ds.time_unit)
                         if ds.time_unit in TIME_UNITS else 0, key=f"cal_tu_{k}")
        conf = st.checkbox("Valeurs vérifiées (issues de l'acquisition)", value=ds.calibration_confirmed,
                           key=f"cal_ok_{k}")
        note = st.text_input("Source / remarque", m.get("calibration", {}).get("note", ""),
                             placeholder="ex. objectif 20×, caméra 2048 px, 10 images/s", key=f"cal_note_{k}")
        if st.button("Enregistrer la calibration", key=f"cal_save_{k}", type="primary" if issues else "secondary"):
            try:
                ds.set_calibration(float(pix) if pix > 0 else None, float(dt), tu, bool(conf and pix > 0), note)
                invalidate_results(ds)
                st.success("Calibration enregistrée. Les résultats calculés avant doivent être relancés.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))


PLOTLY_LIMIT_MB = 150.0   # Streamlit refuse les messages > 200 Mo : marge de sécurité


def show_animation(fig, name: str, sig: tuple, **kw):
    """Affiche une figure Plotly après avoir vérifié son poids (une figure trop lourde fait
    échouer toute la page avec MessageSizeError). Poids mis en cache avec la figure."""
    mb = cached_fig("size_" + name, sig, lambda: len(fig.to_json()) / 1e6)
    if mb > PLOTLY_LIMIT_MB:
        st.warning(f"Animation non affichée : {mb:.0f} Mo, trop lourde pour le navigateur. Augmente le pas entre "
                   "fenêtres (moins de fenêtres) ou choisis le fond « aucun ». Les figures fenêtre par fenêtre "
                   "et les vidéos ne sont pas concernées.")
        return
    if mb > 40:
        st.caption(f"Animation de {mb:.0f} Mo : l'affichage peut prendre quelques secondes.")
    st.plotly_chart(fig, **kw)


def show_fig(fig, **kw):
    st.pyplot(fig, **kw)
    plt.close(fig)


def progress_cb(bar):
    def cb(frac: float, text: str):
        bar.progress(min(max(float(frac), 0.0), 1.0), text=text)
    return cb


# --------------------------------------------------------------------------- source / dataset

def set_source(src: iom.ImageSource):
    ss.source = src
    n, h, wd = len(src), src.height, src.width
    ss.roi_x0, ss.roi_y0, ss.roi_x1, ss.roi_y1 = 0, 0, wd, h
    ss.ref_frame = n - 1          # dernière frame : les pits y sont les plus visibles
    ss.frame_start, ss.frame_end = 0, n - 1
    ss.frame_step, ss.frame_target = 1, min(n, 50)
    ss.pit_cx, ss.pit_cy, ss.sq_size = float(wd // 2), float(h // 2), min(wd, h) // 2
    ss.pop("ds_name", None)
    ss.pop("last_created", None)


def current_roi() -> dsm.ROI:
    return dsm.ROI(int(ss.roi_x0), int(ss.roi_y0), int(ss.roi_x1), int(ss.roi_y1))


@st.cache_resource(max_entries=2, show_spinner="Chargement du dataset en mémoire…")
def load_stack(path: str, mtime: float) -> np.ndarray:
    return dsm.DerivedDataset(Path(path)).gray_stack()


@st.cache_resource(max_entries=4, show_spinner="Préparation des images de détection…")
def load_feature_stack(path: str, mtime: float, polarity: str, background: str) -> np.ndarray:
    return tracking.feature_stack(load_stack(path, mtime), polarity, background)


def display_mode() -> str:
    return ss.get("display_mode", "real")


def _range_sample(frames: list[np.ndarray]) -> np.ndarray:
    """Pixels sous-échantillonnés (pour les percentiles) + vrais min et max (pour la profondeur
    effective en mode « réelles »), sans garder toutes les images en mémoire."""
    g = [iom.to_gray(f) for f in frames]
    sub = np.concatenate([a[::4, ::4].ravel() for a in g])
    return np.concatenate([sub, [min(float(np.nanmin(a)) for a in g), max(float(np.nanmax(a)) for a in g)]])


@st.cache_data(max_entries=8, show_spinner=False)
def ds_range(path: str, mtime: float, mode: str) -> tuple[float, float] | None:
    """Niveaux d'affichage (noir, blanc) communs à toutes les frames du dataset (estimés sur 12
    frames réparties, lues sur le disque : pas besoin de charger tout le dataset)."""
    if mode == "auto":
        return None
    ds = dsm.DerivedDataset(Path(path))
    idx = np.unique(np.linspace(0, len(ds) - 1, min(12, len(ds))).astype(int))
    frames = [ds.get(int(i)) for i in idx]
    return iom.display_range(mode, frames[0].dtype, _range_sample(frames))


def ds_display(path: str, mtime: float, arr: np.ndarray) -> np.ndarray:
    return iom.to_display(arr, rng=ds_range(path, mtime, display_mode()))


def src_range(src: iom.ImageSource) -> tuple[float, float] | None:
    """Niveaux d'affichage communs à toute la séquence d'origine (estimés sur 8 frames réparties)."""
    mode = display_mode()
    if mode == "auto":
        return None
    key = (src.name, len(src), src.width, src.height, mode)
    c = ss.get("_src_range")
    if c is None or c[0] != key:
        idx = np.unique(np.linspace(0, len(src) - 1, min(8, len(src))).astype(int))
        frames = [src.get(int(i)) for i in idx]
        c = (key, iom.display_range(mode, frames[0].dtype, _range_sample(frames)))
        ss["_src_range"] = c
    return c[1]


@st.cache_data(max_entries=512, show_spinner=False)
def _window_background(path: str, mtime: float, f0: int, f1: int, mode: str) -> np.ndarray:
    return iom.to_display(load_stack(path, mtime)[f0:f1 + 1].mean(axis=0), rng=ds_range(path, mtime, mode))


def window_background(path: str, mtime: float, f0: int, f1: int) -> np.ndarray:
    """Image moyenne des frames f0..f1, affichée avec le mode d'affichage choisi."""
    return _window_background(path, mtime, f0, f1, display_mode())


def active_dataset() -> dsm.DerivedDataset | None:
    p = ss.get("active_ds")
    if not p:
        return None
    try:
        return dsm.DerivedDataset(Path(p))
    except Exception as e:  # dossier supprimé ou modifié à la main
        st.error(f"Impossible d'ouvrir le dataset actif : {e}")
        ss.active_ds = None
        return None


# --------------------------------------------------------------------------- barre latérale

def sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Champ de vitesse")
        st.caption("Pitting corrosion — séquences d'images")
        st.divider()
        src = ss.get("source")
        if src is not None:
            seq = f"<b>Séquence</b><br>{src.name}<br><span style='opacity:.65'>{len(src)} frames · {src.width}×{src.height} px</span>"
        else:
            seq = "<b>Séquence</b><br><span style='opacity:.65'>aucune</span>"
        st.markdown(f"<div class='corrovel-card'>{seq}</div>", unsafe_allow_html=True)
        if ss.get("active_ds"):
            ds_name = Path(ss.active_ds).name
            ds = "<b>Dataset actif</b><br>" + ds_name
        else:
            ds = "<b>Dataset actif</b><br><span style='opacity:.65'>aucun</span>"
        st.markdown(f"<div class='corrovel-card'>{ds}</div>", unsafe_allow_html=True)
        if ss.get("result"):
            label = RESULT_LABELS.get(ss.result["res"].method, ss.result["res"].method)
            st.markdown(f"<div class='corrovel-card'><b>Dernier calcul</b><br>{label} · "
                       f"{len(ss.result['res'].windows)} fenêtres</div>", unsafe_allow_html=True)
        st.divider()
        st.radio("Affichage des images", list(iom.DISPLAY_MODES), format_func=iom.DISPLAY_MODES.get,
                 **w("display_mode", "real"),
                 help="Uniquement l'affichage : les calculs utilisent toujours les valeurs brutes des images.\n\n"
                      "• Réelles : niveaux de gris proportionnels aux valeurs enregistrées (0 = noir, maximum du "
                      "capteur = blanc), sans aucune retouche.\n\n"
                      "• Contraste fixe : un seul étirement pour toute la séquence ; les variations réelles de "
                      "luminosité d'une image à l'autre restent visibles.\n\n"
                      "• Contraste auto : chaque image étirée séparément (plus lisible, mais contraste et "
                      "luminosité changent d'une image à l'autre et ne reflètent plus les valeurs).")
        st.divider()
        st.caption(f"Données : `{DATA_ROOT}`")


# =========================================================================== 1. Import

def page_import():
    st.header("Importer la séquence d'origine")
    mode = st.radio("Origine des images", ["Dossier sur le serveur", "Envoyer des fichiers"],
                    horizontal=True, **w("import_mode", "Dossier sur le serveur"))

    if mode == "Dossier sur le serveur":
        st.caption("Un dossier d'images (une image = une frame, triées par nom avec tri naturel) "
                   "ou un fichier TIFF multi-pages. Aucune copie n'est faite.")
        path = st.text_input("Chemin du dossier ou du TIFF", placeholder="/chemin/vers/sequence",
                             **w("import_path", ""))
        raw_dirs = [p.name for p in sorted((DATA_ROOT / "raw").iterdir(), key=lambda p: iom.natural_key(p.name))
                    if p.is_dir()]
        pick = None
        if raw_dirs:
            pick = st.selectbox("…ou une séquence déjà envoyée sur le serveur", ["—"] + raw_dirs)
        if st.button("Charger la séquence", type="primary", key="btn_load"):
            target = path.strip() if path.strip() else (DATA_ROOT / "raw" / pick if pick and pick != "—" else "")
            if not target:
                st.error("Indique un chemin ou choisis une séquence déjà envoyée.")
            else:
                try:
                    with st.spinner("Lecture de la séquence…"):
                        set_source(iom.source_from_folder(target))
                    st.success(f"Séquence chargée : {len(ss.source)} frames.")
                except Exception as e:
                    st.error(f"Chargement impossible : {e}")
    else:
        st.caption("Plusieurs images, ou un .zip contenant les images. Les fichiers sont copiés "
                   f"dans `{DATA_ROOT / 'raw'}`.")
        files = st.file_uploader("Images ou archive .zip", accept_multiple_files=True,
                                 type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "zip"])
        name = st.text_input("Nom de la séquence", **w("upload_name", "sequence"))
        if st.button("Importer", type="primary", disabled=not files, key="btn_upload"):
            dest = DATA_ROOT / "raw" / iom.safe_name(name)
            if dest.exists() and any(dest.iterdir()):
                st.error(f"Le dossier {dest} existe déjà. Choisis un autre nom.")
            else:
                try:
                    with st.spinner("Copie des fichiers…"):
                        iom.save_uploads([(f.name, f.getvalue()) for f in files], dest)
                        set_source(iom.source_from_folder(dest, iom.safe_name(name)))
                    st.success(f"Séquence importée : {len(ss.source)} frames.")
                except Exception as e:
                    st.error(f"Import impossible : {e}")

    src = ss.get("source")
    if src is None:
        st.info("Aucune séquence chargée.")
        return
    st.subheader("Aperçu")
    c1, c2 = st.columns([3, 1])
    with c2:
        d = src.describe()
        st.markdown(f"**{d['n_frames']}** frames  \n**{d['width']} × {d['height']}** px  \n"
                    f"dtype **{d['dtype']}**, {d['channels']} canal(aux)  \n"
                    f"type : {'fichiers séparés' if d['kind'] == 'files' else 'TIFF multi-pages'}")
        st.button("Passer à la ROI →", on_click=goto, args=(SECTIONS[0], "ROI & frames"),
                  type="primary", key="nav_import_roi")
    with c1:
        i = st.slider("Frame", 0, len(src) - 1, **w("view_frame", 0, (0, len(src) - 1)))
        try:
            st.image(iom.to_display(src.get(i), rng=src_range(src)), caption=f"{i} : {src.label(i)}",
                     width="stretch")
        except Exception as e:
            st.error(str(e))


# =========================================================================== 2. ROI et frames

def _png_uri(img: np.ndarray) -> str:
    buf = pyio.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def roi_figure(disp: np.ndarray, roi: dsm.ROI, max_side: int = 1400) -> go.Figure:
    h, wd = disp.shape
    f = max(1, int(np.ceil(max(h, wd) / max_side)))
    small = disp[::f, ::f]
    fig = go.Figure()
    fig.add_trace(go.Image(source=_png_uri(small), x0=(f - 1) / 2, y0=(f - 1) / 2, dx=f, dy=f,
                           hoverinfo="skip"))
    # Grille de points invisibles : la sélection rectangulaire de Plotly a besoin de points
    g = max(1, max(h, wd) // 60)
    gx, gy = np.meshgrid(np.arange(0, wd, g), np.arange(0, h, g))
    fig.add_trace(go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers",
        marker=dict(size=6, opacity=0),
        selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
        hovertemplate="x=%{x}, y=%{y}<extra></extra>", showlegend=False))
    fig.add_shape(type="rect", x0=roi.x0 - 0.5, y0=roi.y0 - 0.5, x1=roi.x1 - 0.5, y1=roi.y1 - 0.5,
                  line=dict(color="#E8423F", width=2))
    aspect = h / wd
    fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=0, b=0),
                      height=int(min(720, max(320, 760 * aspect))))
    fig.update_xaxes(range=[-0.5, wd - 0.5], showgrid=False, constrain="domain")
    fig.update_yaxes(range=[h - 0.5, -0.5], showgrid=False, scaleanchor="x", constrain="domain")
    return fig


def on_roi_select():
    src = ss.get("source")
    state = ss.get("roi_chart")
    if src is None or not state:
        return
    sel = state.get("selection") or {}
    box = None
    for b in reversed(sel.get("box") or []):
        x, y = b.get("x"), b.get("y")
        if x and y and len(x) >= 2 and len(y) >= 2:
            # Rectangle continu : on garde les pixels dont le centre est dedans
            box = (int(np.ceil(min(x))), int(np.ceil(min(y))),
                   int(np.floor(max(x))) + 1, int(np.floor(max(y))) + 1)
            break
    if box is None:
        pts = [p for p in (sel.get("points") or []) if "x" in p and "y" in p]
        if not pts:
            return
        xs = [int(p["x"]) for p in pts]
        ys = [int(p["y"]) for p in pts]
        box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    x0 = int(np.clip(box[0], 0, src.width - 1))
    y0 = int(np.clip(box[1], 0, src.height - 1))
    x1 = int(np.clip(box[2], x0 + 1, src.width))
    y1 = int(np.clip(box[3], y0 + 1, src.height))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    ss.roi_x0, ss.roi_y0, ss.roi_x1, ss.roi_y1 = x0, y0, x1, y1


def reset_roi():
    src = ss.source
    ss.roi_x0, ss.roi_y0, ss.roi_x1, ss.roi_y1 = 0, 0, src.width, src.height


def on_roi_pit_select():
    """Clic sur l'image (mode carré) : le pit, donc le centre du carré, va au point cliqué."""
    state = ss.get("roi_pit_chart")
    pts = [p for p in (((state or {}).get("selection") or {}).get("points") or []) if "x" in p and "y" in p]
    if pts:
        ss.pit_cx, ss.pit_cy = float(pts[-1]["x"]), float(pts[-1]["y"])


def _square_side_px(W: int, H: int, pix: float) -> int:
    """Côté du carré demandé, en px de l'image d'origine (saisi en µm si la taille de pixel est connue)."""
    if pix > 0:
        return int(round(float(ss.get("sq_side_um") or 0) / pix)) or min(W, H) // 2
    return int(ss.get("sq_size") or min(W, H) // 2)


def roi_square_controls(W: int, H: int, pix: float) -> dsm.ROI:
    """Mode « carré centré sur le pit » : position du pit, côté, et ROI carrée qui en découle."""
    st.markdown("**Pit** — clique dessus sur l'image, puis ajuste au pixel près si besoin")
    a, b = st.columns(2)
    a.number_input("x du pit (px)", 0.0, float(W - 1), step=1.0, **w("pit_cx", float(W // 2), (0.0, float(W - 1))))
    b.number_input("y du pit (px)", 0.0, float(H - 1), step=1.0, **w("pit_cy", float(H // 2), (0.0, float(H - 1))))
    side_max = min(W, H)
    if pix > 0:
        hi = side_max * pix
        default = float(min(hi, max(10.0, np.floor(hi / 2 / 10) * 10)))
        st.number_input("Côté du carré (µm)", round(4 * pix, 4), float(hi), step=10.0, format="%.4g",
                        **w("sq_side_um", default, (round(4 * pix, 4), float(hi))),
                        help="Même valeur pour toutes les séquences → images de même champ et de même taille.")
    else:
        st.number_input("Côté du carré (px)", 4, side_max, step=2, **w("sq_size", side_max // 2, (4, side_max)))
        st.caption("Indique la taille de pixel (en haut) pour saisir le côté en µm.")
    roi = dsm.square_roi(ss.pit_cx, ss.pit_cy, _square_side_px(W, H, pix), W, H)
    ss.roi_x0, ss.roi_y0, ss.roi_x1, ss.roi_y1 = roi.x0, roi.y0, roi.x1, roi.y1
    size = f"{roi.width} × {roi.height} px" + (f" = {roi.width * pix:.4g} × {roi.height * pix:.4g} µm" if pix > 0 else "")
    st.caption(f"ROI carrée : {size}, x {roi.x0}–{roi.x1}, y {roi.y0}–{roi.y1}.")
    off = float(np.hypot((roi.x0 + roi.x1 - 1) / 2 - ss.pit_cx, (roi.y0 + roi.y1 - 1) / 2 - ss.pit_cy))
    if off > 1:
        st.info(f"Pit près d'un bord : le carré est décalé de {off:.0f} px pour rester dans l'image "
                "(le pit n'est plus exactement au centre). Réduis le côté pour le recentrer.")
    return roi


def roi_rect_controls(W: int, H: int) -> dsm.ROI:
    """Mode « rectangle libre » : coordonnées éditables (le tracé se fait sur l'image)."""
    st.markdown("**ROI en pixels** (fin exclue)")
    a, b = st.columns(2)
    a.number_input("x début", 0, W - 1, step=1, **w("roi_x0", 0, (0, W - 1)))
    b.number_input("x fin", 1, W, step=1, **w("roi_x1", W, (1, W)))
    a.number_input("y début", 0, H - 1, step=1, **w("roi_y0", 0, (0, H - 1)))
    b.number_input("y fin", 1, H, step=1, **w("roi_y1", H, (1, H)))
    st.button("Image entière", on_click=reset_roi, key="btn_roi_full")
    return current_roi()


def page_roi():
    st.header("ROI et sélection des frames")
    src = ss.get("source")
    if src is None:
        st.info("Charge d'abord une séquence.")
        st.button("Importer une séquence", on_click=goto, args=(SECTIONS[0], "Import"), key="nav_roi_import")
        return
    n, H, W = len(src), src.height, src.width

    st.subheader("Échelles physiques")
    a, b, c = st.columns(3)
    dt = a.number_input("Temps entre deux frames d'origine", min_value=1e-9, format="%.6g",
                        **w("dt", 1.0))
    tu = b.selectbox("Unité de temps", TIME_UNITS, **w("time_unit", "s"))
    pix = c.number_input("Taille de pixel (µm/px), 0 = inconnue", min_value=0.0, format="%.6g",
                         **w("pixel_size", 0.0))
    st.caption("Les vitesses seront en µm/" + tu if pix > 0 else "Les vitesses seront en px/" + tu)

    st.subheader("ROI")
    mode = st.radio("Forme de la ROI", ["square", "rect"], horizontal=True,
                    format_func={"square": "Carré centré sur le pit (recommandé pour un article)",
                                 "rect": "Rectangle libre"}.get, **w("roi_mode", "square"))
    c1, c2 = st.columns([3, 2])
    with c2:   # réglages d'abord : l'image de gauche affiche la ROI qui en résulte
        roi = roi_square_controls(W, H, pix) if mode == "square" else roi_rect_controls(W, H)
    with c1:
        ref = st.slider("Frame affichée (la dernière par défaut : les pits y sont les plus visibles)",
                        0, n - 1, **w("ref_frame", n - 1, (0, n - 1)))
        disp = iom.to_display(src.get(ref), rng=src_range(src))
        if mode == "square":
            st.caption("Clique sur le pit : la ROI est le carré rouge centré sur lui.")
            st.plotly_chart(pit_figure(disp, [{"x": ss.pit_cx, "y": ss.pit_cy}], grid_div=160,
                                       box=(roi.x0 - 0.5, roi.x1 - 0.5, roi.y0 - 0.5, roi.y1 - 0.5)),
                            key="roi_pit_chart", on_select=on_roi_pit_select, selection_mode="points",
                            config={"displaylogo": False})
        else:
            st.caption("Outil de sélection rectangulaire actif : trace un rectangle sur l'image. "
                       "Le cadre rouge est la ROI enregistrée.")
            st.plotly_chart(roi_figure(disp, roi), key="roi_chart",
                            on_select=on_roi_select, selection_mode="box", width="stretch",
                            config={"displaylogo": False, "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"]})
    with c2:
        try:
            roi.validate(W, H)
            roi_ok = True
            st.image(iom.to_display(dsm.crop(src.get(ref), roi), rng=src_range(src)),
                     caption=f"Aperçu de la ROI ({roi.width} × {roi.height} px), frame {ref}", width="stretch")
        except ValueError:
            roi_ok = False
            st.error("ROI invalide : il faut x début < x fin et y début < y fin.")

    st.divider()
    st.subheader("Frames à conserver")
    a, b, c, d = st.columns(4)
    start = a.number_input("Frame de début", 0, n - 1, step=1, **w("frame_start", 0, (0, n - 1)))
    end = b.number_input("Frame de fin", 0, n - 1, step=1, **w("frame_end", n - 1, (0, n - 1)))
    sel_mode = c.radio("Sous-échantillonnage", ["Pas fixe", "Nombre total de frames"],
                       **w("frame_mode", "Pas fixe"))
    if sel_mode == "Pas fixe":
        step = d.number_input("Garder 1 frame sur", 1, max(1, n - 1), step=1,
                              **w("frame_step", 1, (1, max(1, n - 1))))
        target = None
    else:
        step = 1
        target = d.number_input("Nombre de frames", 2, max(2, n), step=1,
                                **w("frame_target", min(n, 50), (2, max(2, n))))
    indices = None
    try:
        indices = dsm.select_indices(n, int(start), int(end), int(step), int(target) if target else None)
        steps_used = sorted(set(np.diff(indices).tolist())) if len(indices) > 1 else []
        msg = f"**{len(indices)} frames** retenues : {indices[:6]}{' …' if len(indices) > 6 else ''}"
        if len(steps_used) > 1:
            msg += f"  \nÉcarts non constants ({steps_used}) : les temps réels sont utilisés pour les vitesses."
        st.markdown(msg)
        if len(indices) < 2:
            st.warning("Il faut au moins 2 frames pour calculer des vitesses.")
    except ValueError as e:
        st.error(str(e))

    st.subheader("Générer le dataset")
    roi = current_roi()
    auto = f"{src.name}_roi{roi.x0}-{roi.y0}_{roi.width}x{roi.height}_f{int(start)}-{int(end)}"
    auto += f"_n{int(target)}" if target else f"_s{int(step)}"
    a, b = st.columns([3, 1])
    name = a.text_input("Nom du dataset (vide = nom automatique)", placeholder=iom.safe_name(auto),
                        **w("ds_name", ""))
    overwrite = b.checkbox("Écraser s'il existe", **w("overwrite", False))
    cal_ok = st.checkbox("Taille de pixel et intervalle entre frames vérifiés (valeurs de l'acquisition)",
                         **w("cal_ok", False),
                         help="Sans cette confirmation, le dataset est marqué « calibration à vérifier » : "
                              "Δt = 1 est la valeur par défaut de l'application, pas une mesure.")
    final_name = iom.safe_name(name if name.strip() else auto)
    exists = (DATA_ROOT / "datasets" / final_name).exists()
    if exists and not overwrite and ss.get("last_created") != str(DATA_ROOT / "datasets" / final_name):
        st.warning(f"Un dataset « {final_name} » existe déjà. Change le nom ou coche « Écraser ».")
    can = roi_ok and indices is not None and len(indices) >= 2 and (overwrite or not exists)
    if st.button("Générer le dataset", type="primary", disabled=not can, key="btn_generate"):
        bar = st.progress(0.0, text="Recadrage…")
        try:
            folder = dsm.create_dataset(src, DATA_ROOT, final_name, roi, indices, float(dt), tu,
                                        float(pix) if pix > 0 else None, overwrite=overwrite,
                                        progress=progress_cb(bar),
                                        pit=(ss.pit_cx, ss.pit_cy) if mode == "square" else None,
                                        calibration_confirmed=bool(cal_ok))
            ss.active_ds = str(folder)
            ss.ds_view = folder.name
            ss.last_created = str(folder)
            ss.pop("result", None)
            bar.empty()
        except Exception as e:
            bar.empty()
            st.error(f"Génération impossible : {e}")
    if ss.get("last_created"):
        st.success(f"Dataset enregistré dans `{ss.last_created}` et sélectionné pour l'analyse.")
        a, b, _ = st.columns([1, 1, 3])
        a.button("Voir le dataset", on_click=goto, args=(SECTIONS[0], "Datasets"), key="nav_roi_datasets")
        b.button("Aller au champ de vitesse →", on_click=goto, args=(SECTIONS[1],), type="primary",
                key="nav_roi_velocity")


# =========================================================================== 3. Datasets

def page_datasets():
    st.header("Datasets générés")
    folders = dsm.list_datasets(DATA_ROOT)
    if not folders:
        st.info("Aucun dataset pour l'instant.")
        st.button("Créer un dataset", on_click=goto, args=(SECTIONS[0], "ROI & frames"), key="nav_datasets_roi")
        return
    names = [f.name for f in folders]
    default = Path(ss.active_ds).name if ss.get("active_ds") and Path(ss.active_ds).name in names else names[-1]
    if ss.get("ds_view") not in names:
        ss.ds_view = default
    name = st.selectbox("Dataset", names, **w("ds_view", default))
    try:
        ds = dsm.DerivedDataset(DATA_ROOT / "datasets" / name)
    except Exception as e:
        st.error(f"Dataset illisible : {e}")
        return
    m = ds.meta
    c1, c2 = st.columns([3, 2])
    with c2:
        roi = m["roi"]
        oi = ds.original_indices
        st.markdown(
            f"**{len(ds)} frames**, {roi['width']} × {roi['height']} px  \n"
            f"ROI : x {roi['x0']}–{roi['x1']}, y {roi['y0']}–{roi['y1']}  \n"
            f"Frames d'origine : {oi[0]} → {oi[-1]}  \n"
            f"Δt d'origine : {m['dt_original']:g} {ds.time_unit}  \n"
            f"Taille de pixel : {str(ds.pixel_size) + ' µm/px' if ds.pixel_size else 'inconnue (px)'}  \n"
            + (f"Pit (centre du carré) : x {ds.pit[0]:.0f}, y {ds.pit[1]:.0f} px  \n" if ds.pit else "")
            + f"Source : {m['source']['name']}  \nCréé le {m['created']}")
        is_active = ss.get("active_ds") == str(ds.folder)
        if is_active:
            st.success("Dataset actif pour l'analyse.")
        else:
            if st.button("Utiliser pour l'analyse", type="primary", key="btn_use_ds"):
                ss.active_ds = str(ds.folder)
                ss.pop("result", None)
                st.rerun()
        st.button("Aller au champ de vitesse →", on_click=goto, args=(SECTIONS[1],), disabled=not is_active,
                  key="nav_datasets_velocity")
        with st.expander("metadata.json"):
            st.json(m)
        st.caption(f"`{ds.folder}`")
    calibration_box(ds)
    with c1:
        k = st.slider("Frame du dataset", 0, len(ds) - 1, **w("ds_frame", 0, (0, len(ds) - 1)))
        rgb = np.asarray(Image.fromarray(ds_display(str(ds.folder), ds.mtime(), ds.get(k))).convert("RGB"))
        st.image(videomod.decorate(rgb, scalebar=current_scalebar(ds)), width="stretch",
                 caption=f"frame {k} (origine {oi[k]}), t = {ds.times[k]:g} {ds.time_unit}")

    scalebar_ui(ds)
    with st.expander("🎬 Vidéo de la séquence brute (aperçu / export rapide)"):
        st.caption("Lecture de toutes les frames du dataset, pour vérifier visuellement la séquence "
                   "avant de lancer un calcul de champ de vitesse.")
        a, b, c = st.columns(3)
        fps = a.number_input("Images / s", 1.0, 60.0, step=1.0, **w("seqvid_fps", 8.0))
        fmt = b.selectbox("Format", ["mp4", "gif"], **w("seqvid_fmt", "mp4"))
        show_time = c.checkbox("Incruster le temps", **w("seqvid_time", True))
        if st.button("Générer la vidéo", key="btn_seqvid"):
            bar = st.progress(0.0, text="Préparation…")
            try:
                stack = load_stack(str(ds.folder), ds.mtime())
                data = videomod.sequence_video_bytes(stack, ds.times, ds.time_unit, fps=float(fps),
                                                     fmt=fmt, show_time=show_time,
                                                     progress=progress_cb(bar),
                                                     scalebar=current_scalebar(ds),
                                                     rng=ds_range(str(ds.folder), ds.mtime(), display_mode()))
                ss.seq_preview = {"data": data, "fmt": fmt, "name": f"{ds.name}_sequence.{fmt}"}
                bar.empty()
            except Exception as e:
                bar.empty()
                st.error(f"Vidéo impossible : {e}")
        sv = ss.get("seq_preview")
        if sv:
            if sv["fmt"] == "mp4":
                st.video(sv["data"])
            else:
                st.image(sv["data"])
            st.download_button("Télécharger", sv["data"], file_name=sv["name"],
                               mime="video/mp4" if sv["fmt"] == "mp4" else "image/gif", key="dl_seqvid")


# =========================================================================== Champ de vitesse

def trackpy_params_ui(nframes: int) -> tracking.TrackpyParams:
    a, b = st.columns(2)
    pol = a.selectbox("Particules", ["bright", "dark"],
                      format_func=lambda s: "claires sur fond sombre" if s == "bright" else "sombres sur fond clair",
                      **w("tp_polarity", "bright"))
    bg = b.selectbox("Fond soustrait", ["median", "mean", "none"],
                     format_func={"median": "médiane temporelle", "mean": "moyenne temporelle",
                                  "none": "aucun"}.get, **w("tp_background", "median"))
    a, b, c = st.columns(3)
    diam = a.number_input("Diamètre (px, impair)", 3, 101, step=2, **w("tp_diameter", 7, (3, 101)))
    minmass = b.number_input("minmass", 0.0, step=10.0, format="%.4g", **w("tp_minmass", 100.0))
    sep = c.number_input("Séparation min (px), 0 = diamètre+1", 0.0, step=1.0, **w("tp_separation", 0.0))
    a, b, c = st.columns(3)
    sr = a.number_input("Déplacement max / frame (px)", 0.5, step=0.5, **w("tp_search", 5.0))
    mem = b.number_input("Mémoire (frames)", 0, 20, step=1, **w("tp_memory", 3, (0, 20)),
                         help="Une particule peut disparaître ce nombre de frames sans casser sa trajectoire "
                              "(défocalisation, détection manquée). 3–4 allonge fortement les trajectoires.")
    minlen = c.number_input("Longueur min des trajectoires", 1, max(1, nframes), step=1,
                            **w("tp_minlen", min(3, nframes), (1, max(1, nframes))))
    a, b, c = st.columns(3)
    pct = a.number_input("percentile", 0.0, 100.0, step=1.0, **w("tp_percentile", 64.0))
    drift = b.checkbox("Soustraire la dérive globale", **w("tp_drift", False),
                       help="Retire le mouvement d'ensemble (dérive de la platine). "
                            "Attention : retire aussi un écoulement collectif réel.")
    pred = c.checkbox("Liaison prédictive", **w("tp_predictive", False),
                      help="Prévoit la position suivante d'après la vitesse des voisines : utile si les "
                           "traceurs sont éjectés très vite (déplacement proche du max par frame).")
    if bg in ("median", "mean"):
        st.caption("⚠️ Fond « médiane / moyenne temporelle » : une particule immobile pendant plus de la moitié "
                   "de la séquence fait partie du fond et disparaît (vérifié sur données synthétiques ; le "
                   "bruit amplifié est alors détecté à sa place). Pour étudier la mise en mouvement de "
                   "traceurs d'abord immobiles, préfère « aucun ».")
    if diam % 2 == 0:
        st.error("Le diamètre doit être impair.")
    return tracking.TrackpyParams(polarity=pol, background=bg, diameter=int(diam), minmass=float(minmass),
                                  separation=float(sep) if sep > 0 else None, percentile=float(pct),
                                  search_range=float(sr), memory=int(mem), min_length=int(minlen),
                                  subtract_drift=bool(drift), predictive=bool(pred))


def _set_minmass(value: float):
    ss.tp_minmass = float(round(value, 1))


def piv_params_ui(cell: int) -> tuple[pivmod.PIVParams, bool]:
    a, b, c = st.columns(3)
    passes = a.selectbox("Passes", [2, 1], format_func={1: "1 passe", 2: "2 passes (fenêtre décalée)"}.get,
                         **w("piv_passes", 2),
                         help="2e passe : la fenêtre de la 2e image est décalée du déplacement trouvé à la 1re, "
                              "ce qui garde les particules dans la fenêtre et rend la mesure fiable jusqu'à "
                              "~½ cellule par intervalle (~¼ avec une seule passe).")
    ratio = b.number_input("Rapport de pics min", 1.0, 5.0, step=0.1, format="%.2g", **w("piv_ratio", 1.3),
                           help="Pic de corrélation principal / 2e pic. En dessous, le déplacement n'est pas "
                                "distinguable du bruit : cellule non mesurée. 1.2–1.5 est usuel ; plus haut = plus "
                                "strict (moins de cellules, plus fiables).")
    med = c.number_input("Test de médiane (0 = aucun)", 0.0, 10.0, step=0.5, **w("piv_median", 2.0),
                         help="Retire les vecteurs aberrants isolés : écart à la médiane des 8 voisins, rapporté "
                              "à leur dispersion (Westerweel & Scarano 2005). 2 est la valeur usuelle.")
    bgsub = st.checkbox("Soustraire le fond fixe (recommandé pour des traceurs)", **w("piv_bgsub", True),
                        help="Corrélation calculée sur ce qui change dans le temps. Sans cela, un motif fixe "
                             "(rayures, bords du pit, particules collées) donne un pic de corrélation au "
                             "déplacement nul qui masque ou tire vers zéro le déplacement des traceurs.")
    st.caption(f"Fenêtre d'interrogation = cellule ({cell} px) : déplacement mesurable jusqu'à ~{cell // 2} px "
               f"par intervalle avec 2 passes. Prévois quelques particules par cellule et par fenêtre de temps ; "
               "sinon agrandis les cellules ou les fenêtres.")
    return pivmod.PIVParams(passes=int(passes), min_peak_ratio=float(ratio), median_test=float(med)), bool(bgsub)


# --------------------------------------------------------------------------- pit(s)
# Un ou plusieurs pits (par exemple des pits voisins) : on clique sur l'image pour en ajouter
# (même technique que le sélecteur de ROI), avec une liste éditable en dessous pour corriger les
# coordonnées. La vitesse radiale est ensuite calculée, par cellule, par rapport au pit le plus
# proche (voir corrovel.field._nearest_pit). Les pits servent aussi de centre au cadrage article.
# Chaque dataset garde ses pits (ss.pits_saved) ; les identifiants sont uniques entre datasets,
# car ils nomment les widgets de coordonnées.

PIT_COLORS = ["#E8423F", "#2B5C7E", "#E8A23F", "#4CAF50", "#9C27B0", "#00BCD4"]


def _new_pit_id() -> int:
    ss.pit_uid = ss.get("pit_uid", 0) + 1
    return ss.pit_uid


def ensure_pits(W: int, H: int, path: str, default: tuple[float, float] | None = None):
    """Pits du dataset `path` ; au premier passage : le pit choisi à l'étape ROI s'il existe
    (`default`), sinon le centre de l'image."""
    if ss.get("pits_for") != path:
        saved = ss.setdefault("pits_saved", {})
        if ss.get("pits_for"):
            saved[ss.pits_for] = ss.pits
        x, y = default if default is not None else (float(W // 2), float(H // 2))
        ss.pits = saved.get(path) or [{"id": _new_pit_id(), "x": float(x), "y": float(y)}]
        ss.pits_for = path


def add_pit(x: float, y: float):
    ss.pits.append({"id": _new_pit_id(), "x": float(x), "y": float(y)})


def remove_pit(pid: int):
    if len(ss.pits) > 1:
        ss.pits = [p for p in ss.pits if p["id"] != pid]


def pit_figure(disp: np.ndarray, pits: list[dict], max_side: int = 1200,
               box: tuple[float, float, float, float] | None = None, grid_div: int = 120) -> go.Figure:
    """Image cliquable avec les pits ; `box` = (x0, x1, y0, y1) trace la ROI en rouge."""
    h, wd = disp.shape
    f = max(1, int(np.ceil(max(h, wd) / max_side)))
    small = disp[::f, ::f]
    fig = go.Figure()
    fig.add_trace(go.Image(source=_png_uri(small), x0=(f - 1) / 2, y0=(f - 1) / 2, dx=f, dy=f,
                           hoverinfo="skip"))
    # Grille de points invisibles : le clic (sélection de points) de Plotly a besoin de points
    g = max(1, max(h, wd) // grid_div)
    gx, gy = np.meshgrid(np.arange(0, wd, g), np.arange(0, h, g))
    fig.add_trace(go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=10, opacity=0),
        selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
        hovertemplate="x=%{x}, y=%{y}<extra></extra>", showlegend=False))
    for i, p in enumerate(pits):
        color = PIT_COLORS[i % len(PIT_COLORS)]
        fig.add_trace(go.Scatter(x=[p["x"]], y=[p["y"]], mode="markers+text", text=[f"P{i + 1}"],
                                 textposition="top center", textfont=dict(color=color, size=12),
                                 marker=dict(symbol="cross", size=14, color=color, line=dict(width=1.5)),
                                 showlegend=False, hoverinfo="skip"))
    if box is not None:
        fig.add_shape(type="rect", x0=box[0], x1=box[1], y0=box[2], y1=box[3],
                      line=dict(color="#E8423F", width=2))
    aspect = h / wd
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=int(min(600, max(260, 620 * aspect))))
    fig.update_xaxes(range=[-0.5, wd - 0.5], showgrid=False, constrain="domain")
    fig.update_yaxes(range=[h - 0.5, -0.5], showgrid=False, scaleanchor="x", constrain="domain")
    return fig


def on_pit_select():
    state = ss.get("pit_chart")
    if not state:
        return
    pts = [p for p in ((state.get("selection") or {}).get("points") or []) if "x" in p and "y" in p]
    if not pts:
        return
    p = pts[-1]
    add_pit(p["x"], p["y"])


def pits_editor(W: int, H: int, background: np.ndarray):
    st.markdown("**Pit(s)** — dernière frame affichée (les pits y sont les plus visibles) : clique "
               "sur l'image pour ajouter un point (un ou plusieurs pits voisins), ajuste les "
               "coordonnées si besoin. La vitesse radiale de chaque cellule est calculée par rapport "
               "au pit le plus proche.")
    st.plotly_chart(pit_figure(background, ss.pits), key="pit_chart", on_select=on_pit_select,
                    selection_mode="points", config={"displaylogo": False})
    for i, p in enumerate(ss.pits):
        pid = p["id"]
        c1, c2, c3 = st.columns([1, 1, 1])
        for ax_, col, hi in (("x", c1, W - 1), ("y", c2, H - 1)):
            key = f"pit_{ax_}_{pid}"
            if key not in ss:          # valeur initiale ; ensuite l'état du widget fait foi
                ss[key] = p[ax_]
            p[ax_] = float(col.number_input(f"{ax_} (pit {i + 1})", 0.0, float(hi), step=1.0, key=key))
        c3.button("Supprimer", key=f"pit_del_{pid}", on_click=remove_pit, args=(pid,),
                 disabled=len(ss.pits) <= 1)
    st.button("+ Ajouter un pit (centre de l'image)", on_click=add_pit, args=(W / 2, H / 2),
             key="btn_add_pit")


def page_velocity():
    st.header("Champ de vitesse")
    ds = active_dataset()
    if ds is None:
        st.info("Sélectionne ou génère d'abord un dataset.")
        st.button("Voir les datasets", on_click=goto, args=(SECTIONS[0], "Datasets"), key="nav_velocity_datasets")
        return
    path, mtime = str(ds.folder), ds.mtime()
    stack = load_stack(path, mtime)
    N, H, W = stack.shape
    calibration_box(ds)
    st.caption(f"Dataset **{ds.name}** : {N} frames, {W} × {H} px, "
               f"unités {ds.length_unit}/{ds.time_unit}")

    method = st.radio("Méthode", list(METHOD_LABELS), horizontal=True, format_func=METHOD_LABELS.get,
                      **w("method", "trackpy"))
    if method == "trackpy":
        st.caption("Adapté quand des particules discrètes sont visibles (traceurs, piqûres ponctuelles…). "
                   "Le champ est la moyenne, par cellule et par fenêtre de temps, des déplacements des "
                   "particules suivies. Pas sûr du choix ? Regarde une frame : si tu vois des taches "
                   "séparées qui se déplacent, prends Trackpy.")
    elif method == "piv":
        st.caption("Chaque cellule est une fenêtre d'interrogation : le déplacement est le pic de "
                   "l'intercorrélation entre deux frames, cumulée sur toutes les paires de la fenêtre de "
                   "temps (corrélation d'ensemble, robuste avec peu de traceurs). Pas besoin de détecter ni "
                   "de relier les particules ; plus robuste que le flux optique aux grands déplacements "
                   "(jusqu'à ~½ cellule par intervalle) et au bruit. Donne un champ par cellule, pas de "
                   "trajectoires individuelles.")
    else:
        st.caption("Estime le déplacement local de ce qui est texturé entre deux images. Avec « soustraire le "
                   "fond fixe », seul ce qui bouge est mesuré : une cellule sans traceur en mouvement reste "
                   "« non mesurée ». Pour des traceurs bien séparés, Trackpy donne en plus la vitesse de chaque "
                   "particule.")

    colp, colv = st.columns([1, 1])
    with colp:
        st.subheader("Grille et temps")
        a, b, c = st.columns(3)
        cell = a.number_input("Taille de cellule (px)", 2, max(2, min(H, W)), step=2,
                              **w("cell", min(16, max(2, min(H, W) // 4)), (2, max(2, min(H, W)))))
        win = b.number_input("Fenêtre (intervalles)", 1, max(1, N - 1), step=1,
                             **w("window", min(5, N - 1), (1, max(1, N - 1))),
                             help="Nombre d'intervalles entre frames moyennés pour un champ.")
        stride = c.number_input("Pas entre fenêtres", 1, max(1, N - 1), step=1,
                                **w("stride", min(5, N - 1), (1, max(1, N - 1))),
                                help="Égal à la fenêtre : fenêtres disjointes. Plus petit : fenêtres glissantes.")
        min_count = 1
        if method == "trackpy":
            a, _, _ = st.columns(3)
            min_count = a.number_input("Déplacements min / cellule", 1, 1000, step=1,
                                       **w("min_count", 5, (1, 1000)),
                                       help="Cellules avec moins de déplacements : laissées vides (non mesurées). "
                                            "Avec 2–3 déplacements, la moyenne d'une cellule est très bruitée ; "
                                            "5 est un minimum raisonnable.")
        nwin = len(fieldmod.make_windows(N - 1, int(win), int(stride)))
        st.caption(f"{nwin} fenêtres temporelles, grille {-(-W // int(cell))} × {-(-H // int(cell))} cellules.")

        ensure_pits(W, H, path, ds.pit)
        pits_editor(W, H, ds_display(path, mtime, stack[-1]))

        st.subheader("Paramètres de la méthode")
        if method == "trackpy":
            tp_params = trackpy_params_ui(N)
        elif method == "piv":
            piv_params, piv_bgsub = piv_params_ui(int(cell))
        else:
            a, b = st.columns(2)
            algs = ["ilk", "tvl1"] + (["farneback"] if flowmod.has_opencv() else [])
            if ss.get("flow_alg") not in algs:
                ss.flow_alg = "ilk"
            alg = a.selectbox("Algorithme", algs,
                              format_func={"ilk": "Lucas-Kanade itératif (ilk)", "tvl1": "TV-L1 (lent)",
                                           "farneback": "Farneback (OpenCV) — ×15-20 plus rapide"}.get,
                              **w("flow_alg", "ilk"))
            if alg in ("ilk", "farneback"):
                rad = b.number_input("Rayon de fenêtre (px)", 1, 50, step=1, **w("flow_radius", 7, (1, 50)))
                fl_params = flowmod.FlowParams(alg, radius=int(rad))
            else:
                att = b.number_input("Attache aux données", 0.1, 200.0, step=1.0, **w("flow_att", 15.0))
                fl_params = flowmod.FlowParams("tvl1", attachment=float(att))
            flow_bgsub = st.checkbox("Soustraire le fond fixe (recommandé pour des traceurs)", **w("flow_bgsub", True),
                                     help="Flux calculé sur ce qui change dans le temps. Sans cela, la texture fixe "
                                          "(rayures, bords du pit) tire le flux vers zéro : sur NiCr(+), ×0.25 de la "
                                          "vitesse Trackpy sans soustraction, ×0.85 avec. Les zones immobiles "
                                          "deviennent alors « non mesurées ».")
            st.caption("Vérification sur NiCr(+) (frames 40–60, cellules de 24 px), Trackpy comme référence "
                       "indépendante : avec fond soustrait, corrélation 0.9 et vitesses ~15 % plus faibles "
                       "(Lucas-Kanade 0.87, Farneback 0.81). Les vitesses du flux optique restent à ±15 % près.")
            if alg != "farneback" and not flowmod.has_opencv():
                st.caption("Pour un flux optique ×15-20 plus rapide (Farneback), installe OpenCV dans ton "
                           "environnement : `pip install opencv-python-headless`, puis relance l'app.")
            if H * W > 1_500_000 and alg != "farneback":
                st.info(f"Image de {W} × {H} px : compte ~15-20 s par paire de frames avec Lucas-Kanade "
                        f"(≈ {(N - 1) * 17 / 60:.0f} min pour {N - 1} paires sur un cœur, moins en parallèle). "
                        "Le résultat est gardé en cache : les calculs suivants (autre cellule, fenêtre, pit) "
                        "sont immédiats. Pour des traceurs, Trackpy est souvent plus adapté et prend quelques secondes.")

    with colv:
        if method == "trackpy":
            st.subheader("Aperçu de la détection")
            fi = st.slider("Frame testée", 0, N - 1, **w("tp_preview", 0, (0, N - 1)))
            if tp_params.diameter % 2 == 1:
                feat = load_feature_stack(path, mtime, tp_params.polarity, tp_params.background)
                f = tracking.locate_frame(feat[fi], tp_params)
                show_det = st.toggle("Afficher l'image de détection (fond soustrait)", **w("tp_show_feat", False))
                img = feat[fi].astype(np.uint8) if show_det else ds_display(path, mtime, stack[fi])
                show_fig(plots.fig_detection(img, f, tp_params.diameter))
                # Histogramme sans seuil minmass pour voir où couper
                f_all = tracking.locate_frame(feat[fi], tracking.TrackpyParams(
                    **{**tp_params.to_dict(), "minmass": 0.0}))
                show_fig(plots.fig_mass_hist(f_all, tp_params.minmass))
                st.caption("Ajuste minmass pour séparer les vraies particules du bruit (pic à faible masse).")
                sug = tracking.suggest_minmass(f_all)
                if sug is not None:
                    st.button(f"Utiliser le minmass suggéré : {sug:.3g}", key="btn_minmass_sug",
                              on_click=_set_minmass, args=(sug,),
                              help="Seuil d'Otsu sur log(masse) : sépare au mieux bruit et particules sur cette frame.")
                guess = densitymod.guess_polarity(stack[fi])
                if guess != tp_params.polarity:
                    st.warning("L'image ressemble plutôt à des particules "
                               f"{'sombres sur fond clair' if guess == 'dark' else 'claires sur fond sombre'} : "
                               "vérifie le réglage « Particules ».")
        else:
            st.subheader("Grille de calcul")
            fig, ax = plt.subplots(figsize=(5, 4.2), layout="constrained")
            ax.imshow(ds_display(path, mtime, stack[-1]), cmap="gray", vmin=0, vmax=255)
            for xg in np.arange(0, W, int(cell)):
                ax.axvline(xg - 0.5, color="w", lw=0.3, alpha=0.5)
            for yg in np.arange(0, H, int(cell)):
                ax.axhline(yg - 0.5, color="w", lw=0.3, alpha=0.5)
            for i, p in enumerate(ss.pits):
                ax.plot(p["x"], p["y"], "+", color=PIT_COLORS[i % len(PIT_COLORS)], ms=12, mew=1.5)
            ax.set_title("Dernière frame, grille et pit(s)")
            show_fig(fig)

    # Réglages qui déterminent le calcul : gardés avec le résultat, pour signaler un résultat affiché
    # qui ne correspond plus aux réglages visibles
    settings = {"méthode": method, "taille de cellule (px)": int(cell), "fenêtre (intervalles)": int(win),
                "pas entre fenêtres": int(stride), "pit(s)": tuple((round(p["x"], 1), round(p["y"], 1)) for p in ss.pits)}
    if method == "trackpy":
        settings.update({"déplacements min / cellule": int(min_count),
                         **{f"trackpy : {k}": v for k, v in tp_params.to_dict().items()}})
    elif method == "piv":
        settings.update({**{f"piv : {k}": v for k, v in piv_params.to_dict().items()},
                         "fond fixe soustrait": bool(piv_bgsub)})
    else:
        settings.update({**{f"flux : {k}": v for k, v in fl_params.to_dict().items()},
                         "fond fixe soustrait": bool(flow_bgsub)})

    st.divider()
    if st.button("Calculer le champ de vitesse", type="primary", key="btn_compute",
                 disabled=(method == "trackpy" and tp_params.diameter % 2 == 0)):
        bar = st.progress(0.0, text="Démarrage…")
        try:
            pits_list = [(p["x"], p["y"]) for p in ss.pits]
            common = dict(cell=int(cell), window=int(win), stride=int(stride), pits=pits_list,
                          scale=ds.scale, length_unit=ds.length_unit, time_unit=ds.time_unit,
                          progress=progress_cb(bar))
            if method == "trackpy":
                res, tracks, samples = fieldmod.run_trackpy(stack, ds.times, tp_params,
                                                            min_count=int(min_count), **common)
            elif method == "piv":
                res = fieldmod.run_piv(stack, ds.times, piv_params, subtract_background=piv_bgsub, **common)
                tracks = samples = None
            else:
                res = fieldmod.run_flow(stack, ds.times, fl_params, cache_dir=ds.folder / "cache",
                                        subtract_background=bool(flow_bgsub), **common)
                tracks = samples = None
            res.params.update({"dataset": ds.name, "dataset_path": path,
                               "length_unit": ds.length_unit, "time_unit": ds.time_unit,
                               "pixel_size": ds.pixel_size})
            ss.result = {"res": res, "tracks": tracks, "samples": samples, "ds_path": path,
                         "created": datetime.now().strftime("%Y%m%d_%H%M%S"), "settings": settings}
            ss.res_window = 0
            ss.vmax_user = 0.0
            ss.pop("export_zip", None)
            ss.pop("field_video", None)
            ss.pop("density_video", None)
            ss.pop("dens_video_export", None)
            bar.empty()
        except Exception as e:
            bar.empty()
            st.error(f"Calcul impossible : {e}")

    r = ss.get("result")
    if r and r["ds_path"] != path:
        st.warning("Le dernier résultat a été calculé sur un autre dataset. Relance le calcul.")
        r = None
    if r and r.get("settings") and r["settings"] != settings:
        changed = [f"{k} : {r['settings'].get(k)} → {v}" for k, v in settings.items() if r["settings"].get(k) != v]
        st.warning("⚠️ Les résultats ci-dessous correspondent à des réglages différents de ceux affichés "
                   f"({'; '.join(changed[:4])}{' …' if len(changed) > 4 else ''}). Clique sur « Calculer le "
                   "champ de vitesse » pour les mettre à jour"
                   + (" (le flux optique déjà calculé est relu du cache : quasi instantané)." if method == "flow" else "."))
    scalebar_ui(ds)
    if r:
        show_results(r, ds, path, mtime)
    st.divider()
    show_heatmap(ds, path, mtime, r)


BG_LABELS = {"image": "image moyenne", "speed": "norme de la vitesse",
             "contour": "carte de couleurs (contours) + flèches", "none": "aucun"}
CONTOUR_CMAPS = ["parula", "viridis", "RdBu_r", "coolwarm", "jet", "turbo"]


def current_contour(res: fieldmod.FieldResult) -> plots.ContourStyle:
    """Réglages du mode contours (lus dans ss : communs à l'affichage et à l'export). L'échelle
    de couleur est fixée ici, une fois pour toutes les fenêtres."""
    q = ss.get("ct_quantity", "v_radial")
    lim = float(ss.get("ct_lim") or 0.0)
    cs = plots.ContourStyle(
        quantity=q, levels=int(ss.get("ct_levels", 12)), cmap=ss.get("ct_cmap", "parula"),
        smooth=float(ss.get("ct_smooth", 0.0)), fill_gaps=int(ss.get("ct_gap", 0)),
        arrow_step=int(ss.get("ct_step", 1)),
        limits=None if lim <= 0 else ((0.0, lim) if q == "speed" else (-lim, lim)),
        symmetric=bool(ss.get("ct_symmetric", True)), smooth_arrows=bool(ss.get("ct_smooth_arrows", False)),
        y_up=bool(ss.get("ct_yup", False)), clean=bool(ss.get("ct_clean", False)))
    return cs if cs.limits else replace(cs, limits=cs.color_limits(res))


def _apply_article_style(nx: int):
    """Préréglage « comme la référence » : carte lisse, flèches régulières (~20 par ligne),
    origine en bas à gauche, sans titre, échelle de couleur ajustée aux données."""
    ss.ct_yup = ss.ct_smooth_arrows = ss.ct_clean = True
    ss.ct_symmetric = False
    ss.ct_levels, ss.ct_cmap = 10, "parula"
    ss.ct_step = max(1, round(nx / 20))
    # Mise en page seulement : le traitement des données (lissage, interpolation) n'est pas modifié


def contour_ui(res: fieldmod.FieldResult) -> plots.ContourStyle:
    st.button("✨ Style article (comme la figure de référence)", key="btn_article_style",
              on_click=_apply_article_style, args=(len(res.x),),
              help="Carte lisse, flèches régulières, origine en bas à gauche, sans titre ni repère du pit, "
                   "échelle de couleur ajustée aux données, flèche de référence dans le cadre.")
    a, b, c, d = st.columns(4)
    a.selectbox("Grandeur en couleur", list(plots.SCALAR_LABELS), format_func=plots.SCALAR_LABELS.get,
                **w("ct_quantity", "v_radial"))
    b.slider("Nombre de niveaux", 4, 30, **w("ct_levels", 12, (4, 30)))
    c.selectbox("Palette", CONTOUR_CMAPS, **w("ct_cmap", "parula"),
                help="parula : proche de la palette par défaut de MATLAB.")
    d.number_input("Lissage (cellules), 0 = aucun", 0.0, 5.0, step=0.5, **w("ct_smooth", 0.0),
                   help="Moyenne gaussienne normalisée sur les seules cellules mesurées : une cellule vide "
                        "reste vide. Le lissage change les valeurs affichées : à indiquer dans la légende.")
    a, b, c, d = st.columns(4)
    a.number_input("Interpoler les trous de (cellules), 0 = aucun", 0, 3, step=1, **w("ct_gap", 0, (0, 3)),
                   help="Comble seulement les petits trous (≤ n cellules d'une mesure) ; ces cellules sont "
                        "hachurées sur la figure. Les grandes zones sans donnée restent blanches.")
    b.number_input("1 flèche sur", 1, 10, step=1, **w("ct_step", 1, (1, 10)),
                   help="Allège les flèches quand la grille est fine.")
    c.number_input("Échelle de couleur (±), 0 = auto", 0.0, format="%.4g", **w("ct_lim", 0.0))
    d.checkbox("Échelle symétrique autour de 0", **w("ct_symmetric", True),
               help="Grandeurs signées : zéro au milieu de la palette. Décoché : la palette couvre les "
                    "valeurs présentes (2e – 98e percentile), comme sur la figure de référence.")
    a, b, c, _ = st.columns(4)
    a.checkbox("Flèches lissées", **w("ct_smooth_arrows", False),
               help="Flèches du champ lissé (même lissage que la carte), sur les mêmes cellules. "
                    "Décoché : mesures brutes.")
    b.checkbox("Origine en bas à gauche", **w("ct_yup", False),
               help="Axe y vers le haut, comme un graphique (l'image n'est pas retournée : seules les "
                    "graduations changent, et v_y devient positive vers le haut).")
    c.checkbox("Style épuré (sans titre ni pit)", **w("ct_clean", False))
    cs = current_contour(res)
    lo, hi = cs.limits
    measured = float(np.mean(np.isfinite(res.u) & np.isfinite(res.v)))
    crit = {"trackpy": f"au moins {res.params.get('min_count', 1)} déplacements par cellule",
            "piv": (f"pic de corrélation net (rapport ≥ {res.params.get('piv', {}).get('min_peak_ratio', 1.3):g}) "
                    "et test de médiane")}.get(res.method, "texture au-dessus du bruit (seuil statistique 99.9 %)")
    st.caption(f"Couleur : {cs.label(res)}, de {lo:.3g} à {hi:.3g}, identique pour toutes les fenêtres. "
               f"Cellules mesurées : {100 * measured:.0f} % ({crit}) ; les autres restent blanches"
               + (f", sauf les trous ≤ {cs.fill_gaps} cellule(s), interpolés et hachurés" if cs.fill_gaps else "")
               + (f". Lissage {cs.smooth:g} cellule(s) (sur les cellules mesurées)" if cs.smooth else "")
               + (" ; flèches lissées." if cs.smooth_arrows else " ; flèches = mesures brutes."))
    if measured < 0.5:
        st.warning(f"Seulement {100 * measured:.0f} % des cellules sont mesurées : la carte est lacunaire. "
                   "Pour la compléter honnêtement : agrandis les cellules ou les fenêtres de temps"
                   + (", ou ajuste la détection (minmass, diamètre, mémoire)." if res.method == "trackpy" else "."))
    return cs


def show_results(r: dict, ds: dsm.DerivedDataset, path: str, mtime: float):
    res: fieldmod.FieldResult = r["res"]
    st.subheader("Résultats")
    nt = len(res.windows)
    auto_vmax = plots.global_vmax(res)
    cols = st.columns(4)
    cols[0].metric("Fenêtres", nt)
    if res.method == "trackpy":
        cols[1].metric("Trajectoires", res.params.get("n_tracks", 0))
        cols[2].metric("Déplacements", res.params.get("n_displacements", 0))
        filled = float(np.mean(res.count >= res.params.get("min_count", 1)))
        cols[3].metric("Cellules renseignées", f"{100 * filled:.0f} %")
        if res.params.get("n_displacements", 0) == 0:
            st.error("Aucun déplacement mesuré. Baisse minmass, augmente le déplacement max "
                     "ou réduis la longueur min des trajectoires.")
            return
    else:
        cols[1].metric("Intervalles", len(ds) - 1)
        cols[2].metric(f"|v| p95 ({res.velocity_unit})", f"{auto_vmax:.3g}")
        pi = res.params.get("piv_info")
        if pi:
            cols[3].metric("Cellules mesurées", f"{100 * float(np.mean(np.isfinite(res.u))):.0f} %")
            st.caption(f"PIV : {pi['seconds']:.0f} s ; rejetées : {pi['rejected_low_peak_ratio']} cellule·fenêtres "
                       f"au pic de corrélation trop faible, {pi['rejected_median_test']} au test de médiane "
                       "(vecteurs aberrants). Elles restent non mesurées.")
            if pi.get("windows_with_uneven_dt"):
                st.warning("Intervalles de temps inégaux dans certaines fenêtres : la corrélation d'ensemble "
                           "suppose des intervalles égaux, les vitesses y sont approximatives.")
        ci = res.params.get("compute")
        if ci:
            mode = (" (OpenCV, tous les cœurs)" if res.params.get("flow", {}).get("algorithm") == "farneback"
                    else f" sur {ci['workers']} processus en parallèle" if ci["workers"] > 1 else " (un seul cœur)")
            st.caption(f"Flux optique : {ci['pairs_computed']} paires calculées" + (mode if ci["pairs_computed"] else "")
                       + (f", {ci['pairs_cached']} relues du cache" if ci["pairs_cached"] else "")
                       + f", en {ci['seconds']:.0f} s. Le flux de chaque paire est gardé dans le dossier du "
                       "dataset (cache/) : changer la cellule, les fenêtres ou le pit ne le recalcule plus.")

    st.markdown("##### Affichage")
    a, b, c, d = st.columns(4)
    bg_mode = a.selectbox("Fond", list(BG_LABELS), format_func=BG_LABELS.get, **w("bg_mode", "image"))
    gain = b.slider("Longueur des flèches", 0.2, 5.0, step=0.1, **w("arrow_gain", 1.0))
    vmax_user = c.number_input(f"Échelle max ({res.velocity_unit}), 0 = auto", 0.0, format="%.4g",
                               **w("vmax_user", 0.0))
    play_fps = d.slider("Vitesse de lecture (fenêtres/s)", 0.5, 8.0, step=0.5, **w("play_fps", 2.0))
    vmax = vmax_user if vmax_user > 0 else auto_vmax
    st.caption(f"Échelle des flèches{' et des couleurs' if bg_mode != 'contour' else ''} fixe pour toutes "
               f"les fenêtres : 0 – {vmax:.3g} {res.velocity_unit} (auto = 95e percentile de |v| sur tout "
               "le calcul).")
    cs = contour_ui(res) if bg_mode == "contour" else None

    bgs = [window_background(path, mtime, f0, f1) for f0, f1 in res.windows]
    res.pit_radii = figure_pit_radii(path, mtime, res.pits)

    st.markdown("### 🎬 Évolution du champ de vitesse")
    st.caption("Lecture/curseur directement dans le navigateur — joue côté client, sans recalcul.")
    anim_bg_mode = bg_mode
    if nt > MAX_ANIM_WINDOWS and bg_mode in ("image", "speed"):
        anim_bg_mode = "none"
        st.info(f"{nt} fenêtres : l'aperçu animé masque le fond image/vitesse pour rester fluide "
               "dans le navigateur (les figures détaillées ci-dessous restent complètes).")
    sig = (r["created"], anim_bg_mode, vmax, gain, play_fps, cs, tuple(res.pit_radii))
    anim = cached_fig("field_anim", sig,
                      lambda: interactivemod.field_animation_figure(res, bgs, anim_bg_mode, vmax, gain,
                                                                    playback_fps=play_fps, contour=cs))
    step = interactivemod.anim_arrow_step(res, cs.arrow_step if cs else 1)
    if step > (cs.arrow_step if cs else 1):
        st.caption(f"Animation : 1 flèche sur {step} et images de fond réduites, pour rester légère dans le "
                   "navigateur (affichage seulement ; figures et vidéos exportées : toutes les flèches, pleine résolution).")
    show_animation(anim, "field_anim", sig, config={"displaylogo": False})

    st.markdown("### Vue détaillée, fenêtre par fenêtre")
    k = st.slider("Fenêtre", 0, nt - 1, **w("res_window", 0, (0, nt - 1))) if nt > 1 else 0
    c1, c2 = st.columns(2)
    with c1:
        show_fig(plots.fig_quiver(res, k, bgs[k], bg_mode, vmax, gain, scalebar=current_scalebar(ds),
                                  contour=cs, panel=current_panel()))
    with c2:
        show_fig(plots.fig_series(res, k))
    c1, c2 = st.columns(2)
    with c1:
        if nt > 1:
            show_fig(plots.fig_radial_profile(res))
        else:
            st.info("Profil radial en fonction du temps : il faut au moins 2 fenêtres.")
    with c2:
        if res.method == "trackpy":
            show_fig(plots.fig_tracks(r["tracks"], window_background(path, mtime, len(ds) - 1, len(ds) - 1),
                                      res.scale, res.length_unit, res.pits, scalebar=current_scalebar(ds),
                                      times=ds.times, time_unit=res.time_unit, pit_radii=res.pit_radii,
                                      panel=current_panel()))
            st.caption("Trajectoires colorées par la vitesse le long du chemin ; cercle + flèche : dernière "
                       "position et direction. Filtres et réglages : 🔬 Analyse → Vitesse, figure (e).")
    if res.method == "trackpy":
        show_density(r, ds, k, bgs)
    with st.expander("Série temporelle (tableau)"):
        st.dataframe(res.series, width="stretch", hide_index=True)
    a, b, _ = st.columns([1.3, 1, 2])
    if res.method == "trackpy":
        a.button("🔬 Analyser les particules une à une →", on_click=goto, args=(SECTIONS[2],), type="primary",
                 key="nav_velocity_particles")
    b.button("Exporter les résultats →", on_click=goto, args=(SECTIONS[3],), key="nav_velocity_export")


# --------------------------------------------------------------------------- densité de particules
# Réglages gardés dans ss (dens_bin, dens_sigma, dens_overlay, dens_vmax_user) : la page Export
# les relit pour produire exactement les mêmes cartes et vidéos que l'aperçu.

def compute_density(r: dict) -> densitymod.DensityResult:
    res = r["res"]
    H, W = res.image_shape
    b = int(min(max(int(ss.get("dens_bin") or res.cell), 2), max(2, min(H, W))))
    return densitymod.particle_density(r["tracks"], res, b, float(ss.get("dens_sigma") or 0.0))


def density_vmax(dens: densitymod.DensityResult) -> float:
    return ss.get("dens_vmax_user") or densitymod.global_vmax(dens)


@st.fragment   # ses réglages ne relancent que cette section, pas tout le champ de vitesse
def show_density(r: dict, ds: dsm.DerivedDataset, k: int, bgs: list[np.ndarray]):
    res: fieldmod.FieldResult = r["res"]
    nt = len(res.windows)
    H, W = res.image_shape
    st.markdown("### 🟣 Densité de particules")
    st.caption("Nombre moyen de particules suivies présentes par frame dans chaque cellule, divisé par "
               "l'aire de la cellule ; une carte par fenêtre temporelle (mêmes fenêtres que le champ de "
               "vitesse). Seules les trajectoires retenues (longueur min) sont comptées.")
    a, b, c, d = st.columns(4)
    a.number_input("Taille de cellule (px)", 2, max(2, min(H, W)), step=2,
                   **w("dens_bin", res.cell, (2, max(2, min(H, W)))))
    b.number_input("Lissage gaussien (cellules)", 0.0, 10.0, step=0.5, **w("dens_sigma", 0.0),
                   help="Écart-type du lissage, en nombre de cellules. 0 = aucun lissage.")
    overlay = c.checkbox("Superposer à l'image moyenne", **w("dens_overlay", True))
    d.number_input(f"Échelle max (particules/{res.length_unit}²), 0 = auto", 0.0, format="%.4g",
                   **w("dens_vmax_user", 0.0))
    dens = compute_density(r)
    vmax = density_vmax(dens)
    st.caption(f"Échelle de couleur fixe pour toutes les fenêtres : 0 – {vmax:.3g} {dens.density_unit} "
               "(auto = 99e percentile des cellules non vides).")

    anim_bgs = bgs if overlay and nt <= MAX_ANIM_WINDOWS else None
    play_fps = ss.get("play_fps", 2.0)
    radii = list(getattr(res, "pit_radii", None) or [])
    anim = cached_fig("density_anim", (r["created"], dens.bin_px, dens.sigma, anim_bgs is not None, vmax,
                                        play_fps, tuple(radii)),
                      lambda: interactivemod.density_animation_figure(dens, anim_bgs, vmax, play_fps,
                                                                      pit_radii=radii))
    show_animation(anim, "density_anim", (r["created"], dens.bin_px, dens.sigma, anim_bgs is not None, vmax, play_fps),
                   config={"displaylogo": False}, key="density_anim")

    sb = current_scalebar(ds)
    c1, c2 = st.columns(2)
    with c1:
        show_fig(plots.fig_density(dens, k, bgs[k], vmax, overlay, scalebar=sb, pit_radii=radii,
                                   panel=current_panel()))
        st.caption(f"Fenêtre {k} (même curseur que la vue détaillée ci-dessus).")
    with c2:
        st.markdown("**🎬 Vidéo de la densité**")
        fmt_choice = st.radio("Format", list(VIDEO_FORMATS), horizontal=True, **w("dens_vid_fmt", "mp4"))
        fps = st.number_input("Images / s", 0.5, 30.0, step=0.5, **w("dens_vid_fps", 4.0))
        sig = (r["created"], dens.bin_px, dens.sigma, overlay, vmax, float(fps), sb)
        if st.button("Générer la vidéo de densité", key="btn_dens_video", disabled=nt < 2):
            fmts = VIDEO_FORMATS[fmt_choice]
            bar = st.progress(0.0, text="Préparation…")
            try:
                files = []
                for j, fmt in enumerate(fmts):
                    data = videomod.density_video_bytes(
                        dens, bgs if overlay else None, overlay, vmax, fps=float(fps), fmt=fmt,
                        progress=lambda f, t: bar.progress((j + f) / len(fmts), text=t), scalebar=sb)
                    files.append((fmt, data, f"{ds.name}_density.{fmt}"))
                ss.density_video = {"sig": sig, "files": files}
                bar.empty()
            except Exception as e:
                bar.empty()
                st.error(f"Vidéo impossible : {e}")
        if nt < 2:
            st.caption("Il faut au moins 2 fenêtres pour une vidéo.")
        dv = ss.get("density_video")
        if dv:
            if dv["sig"] != sig:
                st.caption("Réglages modifiés depuis la génération : régénère la vidéo pour les appliquer.")
            for fmt, data, name in dv["files"]:
                if fmt == "mp4":
                    st.video(data)
                else:
                    st.image(data)
                st.download_button(f"Télécharger ({fmt})", data, file_name=name, mime=VIDEO_MIME[fmt],
                                   key=f"dl_dens_{fmt}")


# --------------------------------------------------------------------------- heatmap pleine résolution
# Carte au pixel, à traînées (voir corrovel.density) : avec les sources tirées de l'image, ne
# dépend pas du champ de vitesse, donc disponible même sans calcul. Réglages dans ss (hm_*),
# relus par la page Export.

HEATMAP_CMAPS = ["jet", "turbo", "inferno", "hot", "viridis"]
HM_SOURCE_LABELS = {"presence": "Présence des particules (image, recommandé)",
                    "intensity": "Contraste des particules (image)",
                    "tracks": "Positions Trackpy (dernier calcul)"}
HM_UNITS = {"presence": "nb de frames où une particule est présente, lissé",
            "intensity": "contraste sommé, u.a.", "tracks": "particules / px, lissé"}
HM_POLARITY_LABELS = {"auto": "détection automatique", "dark": "sombres sur fond clair",
                      "bright": "claires sur fond sombre"}
HM_BG_LABELS = {"rolling": "glissant (frames voisines)", "global": "fixe (médiane de la séquence)"}


def heatmap_sources(r: dict | None) -> list[str]:
    return ["presence", "intensity"] + (["tracks"] if r and r["res"].method == "trackpy" else [])


@st.cache_resource(max_entries=2, show_spinner="Extraction des particules frame par frame…")
def load_particle_signal(path: str, mtime: float, polarity: str, background: str, threshold: float,
                         binary: bool) -> np.ndarray:
    return densitymod.particle_signal_stack(load_stack(path, mtime), polarity, background,
                                            threshold, binary)


@st.cache_data(max_entries=16, show_spinner="Calcul de l'échelle de couleur…")
def heatmap_vmax_cached(key: tuple, _base, n: int, trail: int, sigma: float, pct: float) -> float:
    # `key` identifie la carte de base (non hachable) : dataset, source, réglages, calcul
    return densitymod.heatmap_vmax(_base, n, trail, sigma, pct)


def heatmap_setup(ds: dsm.DerivedDataset, path: str, mtime: float, r: dict | None) -> dict:
    """Carte de base et réglages de la heatmap, communs à l'aperçu et à l'export."""
    srcs = heatmap_sources(r)
    source = ss.get("hm_source") if ss.get("hm_source") in srcs else srcs[0]
    n = len(ds)
    polarity = ss.get("hm_polarity", "auto")
    if polarity == "auto":
        polarity = densitymod.guess_polarity(load_stack(path, mtime)[0])
    if source == "tracks":
        base = densitymod.track_frame_maps(r["tracks"], r["res"].image_shape)
        origin = r["created"]
    else:
        bg, thr = ss.get("hm_bg", "rolling"), float(ss.get("hm_thr", 3.0))
        sig_stack = load_particle_signal(path, mtime, polarity, bg, thr, source == "presence")
        base = sig_stack.__getitem__
        origin = (polarity, bg, thr)
    trail = int(min(max(int(ss.get("hm_trail", min(15, n))), 0), n))
    sigma = float(ss.get("hm_sigma", 2.0))
    pct = float(ss.get("hm_pct", 99.5))
    key = (path, mtime, source, origin, trail, sigma, pct)
    vmax = ss.get("hm_vmax_user") or heatmap_vmax_cached(key, base, n, trail, sigma, pct)
    cmap = ss.get("hm_cmap", "jet")
    show_time = bool(ss.get("hm_time", False))
    return {"base": base, "n": n, "source": source, "polarity": polarity,
            "background": ss.get("hm_bg", "rolling"), "threshold": float(ss.get("hm_thr", 3.0)),
            "trail": trail,
            "sigma": sigma, "pct": pct, "vmax": float(vmax), "cmap": cmap, "show_time": show_time,
            "unit": HM_UNITS[source], "scalebar": current_scalebar(ds),
            "sig": key + (vmax, cmap, show_time, current_scalebar(ds))}


def heatmap_image(hm: dict, i: int) -> np.ndarray:
    """Image RGB de la heatmap à la frame i (aperçu et PNG exporté), barre d'échelle comprise."""
    m = densitymod.accumulated_map(hm["base"], i, hm["trail"], hm["sigma"])
    return videomod.decorate(videomod.heatmap_rgb(m, hm["vmax"], hm["cmap"]), scalebar=hm["scalebar"])


def heatmap_video(hm: dict, ds: dsm.DerivedDataset, fmt: str, fps: float, progress) -> bytes:
    return videomod.heatmap_video_bytes(hm["base"], hm["n"], hm["trail"], hm["sigma"], hm["vmax"],
                                        hm["cmap"], ds.times, ds.time_unit, hm["show_time"],
                                        fps=fps, fmt=fmt, progress=progress, scalebar=hm["scalebar"])


@st.fragment   # ses réglages ne relancent que cette section, pas tout le champ de vitesse
def show_heatmap(ds: dsm.DerivedDataset, path: str, mtime: float, r: dict | None):
    N = len(ds)
    st.markdown("### 🔥 Heatmap de densité pleine résolution")
    st.caption("Pour chaque frame, les particules des dernières frames (traînée) sont additionnées "
               "puis lissées au pixel près : les trajectoires apparaissent comme des traînées "
               "colorées. Avec les sources tirées de l'image, aucun calcul préalable n'est nécessaire "
               "(le photoblanchiment est compensé).")
    srcs = heatmap_sources(r)
    if ss.get("hm_source") not in srcs:
        ss.hm_source = srcs[0]
    a, b, c, d = st.columns(4)
    source = a.selectbox("Source", srcs, format_func=HM_SOURCE_LABELS.get, **w("hm_source", srcs[0]))
    b.number_input("Traînée (frames), 0 = cumul", 0, N, step=1, **w("hm_trail", min(15, N), (0, N)),
                   help="Nombre de frames additionnées pour chaque image. 0 : cumul depuis le début.")
    c.number_input("Lissage gaussien (px)", 0.0, 20.0, step=0.5, **w("hm_sigma", 2.0))
    d.selectbox("Palette", HEATMAP_CMAPS, **w("hm_cmap", "jet"))
    if source != "tracks":
        a, b, c, _ = st.columns(4)
        a.selectbox("Particules ", list(HM_POLARITY_LABELS), format_func=HM_POLARITY_LABELS.get,
                    **w("hm_polarity", "auto"))
        b.selectbox("Fond soustrait ", list(HM_BG_LABELS), format_func=HM_BG_LABELS.get,
                    **w("hm_bg", "rolling"),
                    help="Glissant : retire aussi ce qui évolue lentement (pit qui grossit), "
                         "ne garde que ce qui bouge.")
        c.number_input("Seuil (× bruit)", 0.0, 20.0, step=0.5, **w("hm_thr", 3.0),
                       help="Un pixel est compté s'il s'écarte du fond de plus de seuil × écart-type "
                            "du bruit de la frame. Plus haut : moins de bruit, moins de particules faibles.")
    a, b, c, _ = st.columns(4)
    a.slider("Saturation (percentile)", 90.0, 100.0, step=0.1, **w("hm_pct", 99.5),
             help="Les pixels au-dessus de ce percentile ont la couleur maximale.")
    b.number_input("Échelle max, 0 = auto", 0.0, format="%.4g", **w("hm_vmax_user", 0.0))
    c.checkbox("Incruster le temps", **w("hm_time", False))
    if "tracks" not in srcs:
        st.caption("Lance un calcul Trackpy pour pouvoir aussi utiliser les positions des particules suivies.")

    hm = heatmap_setup(ds, path, mtime, r)
    st.caption(f"Échelle de couleur fixe pour toute la vidéo : 0 – {hm['vmax']:.3g} ({hm['unit']}).")
    c1, c2 = st.columns([3, 2])
    with c1:
        i = st.slider("Frame ", 0, N - 1, **w("hm_frame", N - 1, (0, N - 1)))
        st.image(heatmap_image(hm, i), width="stretch",
                 caption=f"frame {i}, t = {ds.times[i]:g} {ds.time_unit}")
    with c2:
        st.markdown("**🎬 Vidéo / GIF de la heatmap**")
        fmt_choice = st.radio("Format", list(VIDEO_FORMATS), horizontal=True, **w("hm_vid_fmt", "mp4"))
        fps = st.number_input("Images / s", 1.0, 60.0, step=1.0, **w("hm_vid_fps", 10.0))
        sig = hm["sig"] + (float(fps),)
        if st.button("Générer la vidéo de la heatmap", key="btn_hm_video", disabled=N < 2):
            fmts = VIDEO_FORMATS[fmt_choice]
            bar = st.progress(0.0, text="Préparation…")
            try:
                files = []
                for j, fmt in enumerate(fmts):
                    data = heatmap_video(hm, ds, fmt, float(fps),
                                         lambda f, t: bar.progress((j + f) / len(fmts), text=t))
                    files.append((fmt, data, f"{ds.name}_heatmap.{fmt}"))
                ss.heatmap_video = {"sig": sig, "files": files}
                bar.empty()
            except Exception as e:
                bar.empty()
                st.error(f"Vidéo impossible : {e}")
        st.caption("Pour l'ajouter au zip des résultats : page 📤 Export (option « Inclure la heatmap »).")
        hv = ss.get("heatmap_video")
        if hv:
            if hv["sig"] != sig:
                st.caption("Réglages modifiés depuis la génération : régénère la vidéo pour les appliquer.")
            for fmt, data, name in hv["files"]:
                if fmt == "mp4":
                    st.video(data)
                else:
                    st.image(data)
                st.download_button(f"Télécharger ({fmt}, {len(data) / 1e6:.1f} Mo)", data, file_name=name,
                                   mime=VIDEO_MIME[fmt], key=f"dl_hm_{fmt}")


# =========================================================================== Analyse des traceurs
# Analyse traceur par traceur (corrovel.particles) : trajectoires Trackpy, ou traceurs virtuels
# advectés dans le champ du flux optique / PIV. Calculs par question (corrovel.questions),
# figures (corrovel.qplots, corrovel.pplots), comparaison de groupes (corrovel.compare).

TRACER_TYPES = {"positive": "positifs (+)", "negative": "négatifs (−)", "neutral": "neutres",
                "none": "sans traceur", "unknown": "non précisé"}
PIT_MODES = {"auto": "détection automatique", "manual": "valeur manuelle", "none": "pas de pit / inconnu"}


def load_experiment(ds: dsm.DerivedDataset) -> dict:
    """Description de l'expérience (traceurs, substrat…), enregistrée dans le dossier du dataset
    pour pouvoir ensuite comparer plusieurs expériences."""
    p = ds.folder / "experiment.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    name = (ds.meta.get("source", {}).get("name", "") + " " + ds.name).lower()
    guess = "positive" if "(+)" in name else "negative" if "(-)" in name or "_-_" in name else "unknown"
    return {"tracers": guess, "substrate": "NiCr" if "nicr" in name else "", "electrolyte": "", "notes": ""}


@st.cache_data(max_entries=16, show_spinner="Signal autour du pit…")
def pit_signal_cached(path: str, mtime: float, pit: tuple[float, float], radius: float) -> np.ndarray:
    return pm.pit_signal(load_stack(path, mtime), pit, radius)


@st.cache_data(max_entries=16, show_spinner=False)
def pit_radius_cached(path: str, mtime: float, pit: tuple[float, float]) -> float:
    stack = load_stack(path, mtime)
    return pm.estimate_pit_radius(stack[-1], pit)


def _png(fig, dpi: int = 130) -> bytes:
    data = plots.fig_to_bytes(fig, "png", dpi)
    plt.close(fig)
    return data


def show_png(name: str, sig: tuple, build):
    """Figure matplotlib mise en cache (PNG) : les onglets ne sont pas redessinés à chaque clic."""
    st.image(cached_fig("png_" + name, sig, lambda: _png(build())))


def fig_pit_signal(times: np.ndarray, sig: np.ndarray, detected: float | None, t_pit: float | None,
                   time_unit: str):
    fig, ax = plt.subplots(figsize=(6.0, 2.4), layout="constrained")
    ax.plot(times, sig, color="black", lw=1.2)
    if detected is not None:
        ax.axvline(detected, color=style.PIT_COLOR, lw=1, ls=":", label=f"Detected: {detected:.4g} {time_unit}")
    if t_pit is not None:
        ax.axvline(t_pit, color="black", lw=1, ls="--", label=f"Used: {t_pit:.4g} {time_unit}")
    ax.set_xlabel(style.label("Time", time_unit))
    ax.set_ylabel("Relative intensity\nat pit")
    if detected is not None or t_pit is not None:
        ax.legend(frameon=False, fontsize=8)
    return fig


def speed_fig_settings(n_frames: int) -> dict:
    """Réglages de la figure « vitesse le long des trajectoires » (🔬 Analyse → Vitesse), réutilisés à l'export."""
    return {"vmax": float(ss.get("paf_vmax") or 0) or None,
            "end_frame": int(min(max(int(ss.get("paf_end", n_frames - 1)), 0), n_frames - 1)),
            "min_displacement": float(ss.get("paf_disp") or 0.0),
            "min_points": int(ss.get("paf_minpts", 5)), "smooth": int(ss.get("paf_smooth", 3)),
            "only_mobile": bool(ss.get("paf_mobile", True)), "show_ends": bool(ss.get("paf_ends", True))}


def fig_tracks_speed_for(pa: pm.ParticleAnalysis, ds: dsm.DerivedDataset, path: str, mtime: float, radii: list):
    """Figure de référence (trajectoires colorées par la vitesse) sur l'image de la frame de fin :
    les cercles tombent sur les particules visibles."""
    kw = speed_fig_settings(len(ds))
    bg_end = window_background(path, mtime, kw["end_frame"], kw["end_frame"])
    return pplots.fig_tracks_speed(pa, bg_end, current_scalebar(ds), radii, panel=current_panel(), **kw)


def particle_analysis_from_state(ds: dsm.DerivedDataset, r: dict):
    """Analyse des traceurs avec les réglages enregistrés (⚙️ Réglages), commune à toutes les
    sections et à l'export. Retourne (pa, msds, tracks) ; ValueError sans trajectoire exploitable."""
    res = r["res"]
    is_tp = res.method == "trackpy"
    seed_step = int(ss.get("adv_seed_step", 1))
    tracks = r["tracks"] if is_tp else cached_fig(
        "adv_tracks", (r["created"], seed_step), lambda: fieldmod.advect_tracers(res, seed_step=seed_step))
    path, mtime = str(ds.folder), ds.mtime()
    pits = [tuple(p) for p in res.pits]
    mode = ss.get("pa_pit_mode", "auto")
    t_pit = None
    if mode == "auto":
        t_pit, _ = pm.detect_pit_time(pit_signal_cached(path, mtime, pits[0], float(ss.get("pa_sig_r", 15))),
                                      ds.times)
    elif mode == "manual":
        t_pit = ss.get("pa_t_pit")
    excl = ss.get("pa_excl")
    mp = pm.MotionParams(smooth=int(ss.get("pa_smooth", 3)), window=int(ss.get("pa_window", 3)),
                         threshold=float(ss.get("pa_thr") or 0) or None, persist=int(ss.get("pa_persist", 2)),
                         min_points=int(ss.get("pa_minpts", 5)), t_pit=t_pit,
                         exclude_px=float(excl if excl is not None else round(pit_radius_cached(path, mtime, pits[0]))))
    key = (r["created"], mp, tuple(pits), None if is_tp else seed_step)
    pa = cached_fig("pa", key, lambda: pm.analyze(tracks, ds.times, pits, res.scale, res.length_unit,
                                                  res.time_unit, mp))
    t0, t1 = float(ds.times[0]), float(ds.times[-1])
    periods = ({"avant le pit": (t0, t_pit), "après le pit": (t_pit, t1)}
               if t_pit is not None and t0 < t_pit < t1 else {"toute la séquence": (t0, t1)})
    msds = cached_fig("msd", (r["created"], t_pit, mp.exclude_px, None if is_tp else seed_step), lambda: {
        lab: pm.msd(tracks[tracks["particle"].isin(pa.particles["particle"])], ds.times, res.scale, rng)
        for lab, rng in periods.items()})
    return pa, msds, tracks



# =========================================================================== Analyse, par question
# Une sous-section par question de l'article. Chaque section = « Calculs » (tableau avec
# intervalles de confiance à 95 %, phrases de réponse, test entre groupes) puis « Figures »
# (lettres (a), (b)…), et un zip de la section. Les groupes comparés sont les pits de la vidéo
# (« Pit A », « Pit B »…) et les analyses enregistrées choisies dans ⚙️ Réglages.

QUESTIONS = {
    "Vitesse": "Quelles sont les vitesses des traceurs ? Quelle est leur distribution ? Comment évoluent-elles "
               "dans le temps et dans l'espace ?",
    "Homogénéité": "Tous les traceurs bougent-ils en même temps ? Ont-ils des vitesses comparables ? Existe-t-il "
                   "des groupes de particules aux comportements différents ?",
    "Déplacement": "Quelle distance chaque traceur parcourt-il ? Dans quelle direction ? Son mouvement est-il "
                   "rectiligne, radial, diffusif ou plus complexe ?",
    "Début du mouvement": "Quand les traceurs commencent-ils à bouger ? Peut-on identifier une mise en mouvement "
                          "collective ou progressive ?",
    "Relation avec le pit": "Que se passe-t-il avant, pendant et après l'apparition d'un pit ? Le mouvement "
                            "change-t-il en fonction de la distance au pit ?",
    "Traceurs": "Les traceurs positifs et négatifs se comportent-ils différemment ? Les différences observées "
                "sont-elles robustes et reproductibles ? (Plus généralement : selon le substrat, la solution, le "
                "potentiel appliqué…)",
}
SECTION_SLUGS = {"Vitesse": "vitesse", "Homogénéité": "homogeneite", "Déplacement": "deplacement",
                 "Début du mouvement": "debut_du_mouvement", "Relation avec le pit": "relation_avec_le_pit",
                 "Traceurs": "traceurs"}
WIDE_FIGS = {"displacement", "behaviour", "indicators"}       # figures larges : une ligne entière
MOTION_FR = {"confined": "confiné", "diffusive": "diffusif", "directed": "dirigé"}
PHASE_FR = {"before": "avant le pit", "growth": "croissance du pit", "after": "après la croissance",
            "whole sequence": "toute la séquence"}
VERDICT_FR = {"robust": "✅ différence robuste et reproductible",
              "significant, not reproduced": "⚠️ significative, mais pas reproduite dans chaque répétition",
              "no difference detected": "pas de différence détectée", "not enough data": "données insuffisantes"}
GROUP_FIELDS = {"tracers": "type de traceur", "substrate": "substrat", "potential_V": "potentiel appliqué",
                "electrolyte": "électrolyte / solution", "concentration": "concentration",
                "tracer_size": "taille des traceurs", "label": "nom de l'analyse", "dataset": "dataset"}


@st.cache_resource(max_entries=32, show_spinner="Chargement de l'analyse enregistrée…")
def load_saved_cached(path: str, mtime: float):
    spa, meta = cmpmod.load_analysis(Path(path))
    return qmod.add_r_norm(spa, meta.get("pit_radii_px")), meta


def _saved_mtime(path: str) -> float:
    return (Path(path) / "meta.json").stat().st_mtime


@st.cache_data(max_entries=8, show_spinner="Rayon du pit au cours du temps…")
def pit_radius_series_cached(path: str, mtime: float, pit: tuple[float, float]) -> pd.DataFrame:
    return qmod.pit_radius_series(load_stack(path, mtime), pit, dsm.DerivedDataset(Path(path)).times)


def _unique_labels(groups: list) -> list:
    seen: dict[str, int] = {}
    for g in groups:
        n = seen.get(g.label, 0)
        seen[g.label] = n + 1
        if n:
            g.label = f"{g.label} ({n + 1})"
    return groups


def analysis_ctx() -> dict | None:
    """Analyse en cours (réglages de ⚙️ Réglages) et groupes comparés : un par pit de la vidéo, plus
    les analyses enregistrées choisies. None sans champ de vitesse ; ValueError sans trajectoire."""
    ds, r = active_dataset(), ss.get("result")
    if not r or ds is None or r["ds_path"] != str(ds.folder):
        return None
    res = r["res"]
    path, mtime = str(ds.folder), ds.mtime()
    pits = [(float(x), float(y)) for x, y in res.pits]
    pa0, msds, tracks = particle_analysis_from_state(ds, r)
    R_px = [pit_radius_cached(path, mtime, p) for p in pits]     # rayon réel (r / R), contour affiché ou non
    key = (r["created"], pa0.params, tuple(pits),
           None if res.method == "trackpy" else int(ss.get("adv_seed_step", 1)), tuple(R_px))
    pa = cached_fig("pa_norm", key, lambda: qmod.add_r_norm(pa0, R_px))
    name = str(ss.get("pa_name") or "").strip() or None
    saved = tuple((p, _saved_mtime(p)) for p in ss.get("cmp_saved", []) if (Path(p) / "meta.json").exists())
    csig = key + (name, saved)

    def build_groups():
        G, warn = cmpmod.pit_groups(pa, name, {"source": "current", "dataset": ds.name}), []
        for p, m in saved:
            try:
                spa, smeta = load_saved_cached(p, m)
            except Exception as e:
                warn.append(f"Analyse enregistrée illisible ({Path(p).name}) : {e}")
                continue
            G += cmpmod.pit_groups(spa, smeta.get("label"), {**smeta, "source": "saved"})
        return cmpmod.styled(_unique_labels(G)), warn

    groups, warn = cached_fig("groups", csig, build_groups)
    for msg in warn:
        st.warning(msg)
    radii = figure_pit_radii(path, mtime, pits)
    return {"ds": ds, "r": r, "res": res, "path": path, "mtime": mtime, "pa": pa, "msds": msds, "tracks": tracks,
            "R_px": R_px, "radii": radii, "groups": groups, "exp": load_experiment(ds), "csig": csig,
            "sig": csig + (tuple(radii), current_panel(), display_mode()), "sb": current_scalebar(ds),
            "panel": current_panel(), "has_r": bool(np.isfinite(pa.steps["r_norm"]).any())}


def calc(name: str, ctx: dict, build):
    """Calcul d'une section, gardé tant que l'analyse et les groupes ne changent pas."""
    return cached_fig("calc_" + name, ctx["csig"], build)


def q_header(view: str) -> dict | None:
    st.header(view)
    st.markdown(f"*{QUESTIONS[view]}*")
    try:
        ctx = analysis_ctx()
    except ValueError as e:
        st.error(str(e))
        return None
    if ctx is None:
        st.info("Calcule d'abord un champ de vitesse sur le dataset actif (page Champ de vitesse), puis règle "
                "l'analyse dans ⚙️ Réglages.")
        st.button("Aller au champ de vitesse", on_click=goto, args=(SECTIONS[1],), key=f"nav_q_{view}")
        return None
    pa = ctx["pa"]
    other = {(g.pa.length_unit, g.pa.time_unit) for g in ctx["groups"]} - {(pa.length_unit, pa.time_unit)}
    if other:
        st.warning("Certains groupes comparés n'ont pas les mêmes unités que l'analyse en cours "
                   f"({', '.join('/'.join(u) for u in other)}) : leurs valeurs ne sont pas directement comparables.")
    st.caption("Groupes comparés : " + ", ".join(f"**{g.label}**" for g in ctx["groups"])
               + " — réglages, enregistrement et choix des comparaisons : ⚙️ Réglages.")
    return ctx


def _letter(i: int) -> str:
    return f"({chr(97 + i)})"


def _with_letter(build, i: int):
    fig = build()
    style.panel_letter(fig.axes[0], _letter(i))
    return fig


def section_files(figs: list[tuple], tables: dict[str, pd.DataFrame], prefix: str,
                  formats=("png", "svg"), dpi: int = 300) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for i, (name, _, build, _) in enumerate(figs):
        fig = _with_letter(build, i)
        for fmt in formats:
            out[f"{prefix}/figures/{chr(97 + i)}_{name}.{fmt}"] = plots.fig_to_bytes(fig, fmt, dpi)
        plt.close(fig)
    for tname, df in tables.items():
        out[f"{prefix}/{tname}.csv"] = df.to_csv(index=False).encode("utf-8")
    return out


def render_section(ctx: dict, section: str, figs: list[tuple], tables: dict[str, pd.DataFrame]):
    """Figures de la section (2 par ligne, les larges seules), lettres (a), (b)… dans l'ordre, puis
    zip de la section. figs : (nom, légende, constructeur, éléments de signature propres)."""
    st.subheader("Figures")
    rows, cur = [], []
    for i, f in enumerate(figs):
        if f[0] in WIDE_FIGS:
            rows += [cur] if cur else []
            rows.append([i])
            cur = []
        else:
            cur.append(i)
            if len(cur) == 2:
                rows.append(cur)
                cur = []
    rows += [cur] if cur else []
    for row in rows:
        wide = len(row) == 1 and figs[row[0]][0] in WIDE_FIGS
        cols = [st.container()] if wide else st.columns(2)
        for col, i in zip(cols, row):
            name, caption, build, extra = figs[i]
            with col:
                show_png(f"q_{section}_{name}", ctx["sig"] + tuple(extra), lambda b=build, k=i: _with_letter(b, k))
                st.caption(f"**{_letter(i)}** {caption}")
    slug = SECTION_SLUGS[section]
    if st.button("📦 Préparer le zip de la section (figures PNG 300 dpi + SVG, calculs CSV)", key=f"zipbtn_{slug}"):
        with st.spinner("Figures…"):
            buf = pyio.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in section_files(figs, tables, slug).items():
                    zf.writestr(name, data)
            ss[f"zip_{slug}"] = buf.getvalue()
    if ss.get(f"zip_{slug}"):
        st.download_button("Télécharger le zip", ss[f"zip_{slug}"], file_name=f"{ctx['ds'].name}_{slug}.zip",
                           mime="application/zip", type="primary", key=f"zipdl_{slug}")


def _test_sentence(samples: dict, what: str) -> str | None:
    t = cmpmod.compare_samples(samples)
    if not t:
        return None
    verdict = "différence significative" if t["p"] < 0.05 else "pas de différence significative"
    return (f"Entre groupes ({what}) : **{verdict}** — {t['test']}, p = {t['p']:.2g}, "
            f"taille d'effet {t['effect_name']} = {t['effect']:.2f}.")


def _spearman(df: pd.DataFrame, a: str, b: str, min_n: int = 8) -> tuple[float, float] | None:
    d = df[[a, b]].dropna()
    if len(d) < min_n or d[a].nunique() < 3:
        return None
    from scipy.stats import spearmanr
    rho, p = spearmanr(d[a], d[b])
    return float(rho), float(p)


def _f(x, fmt: str = "{:.4g}", none: str = "—") -> str:
    try:
        return fmt.format(x) if x is not None and np.isfinite(x) else none
    except TypeError:
        return none


# --------------------------------------------------------------------------- ⚙️ Réglages

def page_q_settings():
    st.header("⚙️ Réglages de l'analyse des traceurs")
    ds, r = active_dataset(), ss.get("result")
    if not r or ds is None or r["ds_path"] != str(ds.folder):
        st.info("Calcule d'abord un champ de vitesse sur le dataset actif (page Champ de vitesse).")
        st.button("Aller au champ de vitesse", on_click=goto, args=(SECTIONS[1],), key="nav_particles_velocity")
        return
    res: fieldmod.FieldResult = r["res"]
    calibration_box(ds)
    path, mtime = str(ds.folder), ds.mtime()
    pits = [(float(x), float(y)) for x, y in res.pits]
    is_tp = res.method == "trackpy"
    if not is_tp:
        st.info(f"🌀 **{RESULT_LABELS.get(res.method, res.method)} : pas de particules suivies individuellement.** Les "
                "« traceurs » de l'analyse sont des **traceurs virtuels**, reconstruits en intégrant le champ de "
                "vitesse mesuré dans le temps (lignes de courant) — pas des particules réellement détectées. Les "
                "réponses décrivent donc le champ mesuré.")
        a, _ = st.columns([1, 3])
        a.number_input("1 traceur virtuel tous les N cellules", 1, 10, step=1, **w("adv_seed_step", 1, (1, 10)),
                       help="Plus petit = plus de traceurs virtuels (plus lent, mieux échantillonné).")
    st.caption("Ces réglages valent pour toutes les sections : Vitesse, Homogénéité, Déplacement, Début du "
               "mouvement, Relation avec le pit et Traceurs.")

    # ---------------------------------------------------------------- expérience
    exp = load_experiment(ds)
    with st.expander("🧪 Expérience (enregistrée avec le dataset, pour comparer les expériences)",
                     expanded=not (ds.folder / "experiment.json").exists()):
        k = ds.name
        a, b, c = st.columns(3)
        tracers = a.selectbox("Traceurs", list(TRACER_TYPES), format_func=TRACER_TYPES.get,
                              index=list(TRACER_TYPES).index(exp.get("tracers", "unknown")), key=f"exp_tr_{k}")
        substrate = b.text_input("Substrat", exp.get("substrate", ""), placeholder="inox 316L, NiCr…", key=f"exp_sub_{k}")
        electrolyte = c.text_input("Électrolyte / solution", exp.get("electrolyte", ""), placeholder="NaCl",
                                   key=f"exp_el_{k}")
        a, b, c = st.columns(3)
        pot = a.text_input("Potentiel appliqué (V)", _f(exp.get("potential_V"), "{:g}", ""), placeholder="1.6",
                           key=f"exp_pot_{k}")
        conc = b.text_input("Concentration", exp.get("concentration", ""), placeholder="1000 mM", key=f"exp_conc_{k}")
        size = c.text_input("Taille des traceurs", exp.get("tracer_size", ""), placeholder="1 µm", key=f"exp_size_{k}")
        notes = st.text_input("Notes", exp.get("notes", ""), key=f"exp_no_{k}")
        try:
            pot_v = float(pot.replace(",", ".")) if pot.strip() else None
        except ValueError:
            pot_v = None
            st.warning("Potentiel : un nombre est attendu (en V).")
        exp = {"tracers": tracers, "substrate": substrate, "electrolyte": electrolyte, "potential_V": pot_v,
               "concentration": conc, "tracer_size": size, "notes": notes}
        if st.button("Enregistrer la description", key="btn_exp_save"):
            (ds.folder / "experiment.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success("Enregistré dans le dossier du dataset.")

    # ---------------------------------------------------------------- apparition du pit
    st.subheader("Apparition du pit")
    c1, c2 = st.columns([2, 1])
    with c2:
        radius = st.number_input("Rayon de mesure autour du pit (px)", 3, 300, step=1, **w("pa_sig_r", 15, (3, 300)))
        mode = st.radio("Instant d'apparition", list(PIT_MODES), format_func=PIT_MODES.get, **w("pa_pit_mode", "auto"))
    sig = pit_signal_cached(path, mtime, pits[0], float(radius))
    detected, why = pm.detect_pit_time(sig, ds.times)
    with c2:
        t_pit = None
        if mode == "auto":
            t_pit = detected
            st.caption(f"Détection : {why}." if detected is not None else f"⚠️ {why}.")
        elif mode == "manual":
            t_lo, t_hi = float(ds.times[0]), float(ds.times[-1])
            default = detected if detected is not None else 0.5 * (t_lo + t_hi)
            t_pit = st.number_input(f"t d'apparition ({ds.time_unit})", t_lo, t_hi, format="%.4g",
                                    **w("pa_t_pit", float(default), (t_lo, t_hi)))
    with c1:
        show_png("pit_signal", (path, mtime, pits[0], radius, detected, t_pit),
                 lambda: fig_pit_signal(ds.times, sig, detected, t_pit, ds.time_unit))
        st.caption("Intensité moyenne autour du pit (divisée par celle de l'image, ce qui compense le "
                   "photoblanchiment) : un pit sombre qui se forme la fait chuter. Plusieurs pits : mesurée sur P1.")

    # ---------------------------------------------------------------- réglages
    auto_excl = pit_radius_cached(path, mtime, pits[0])
    with st.expander("⚙️ Réglages de l'analyse"):
        a, b, c, d = st.columns(4)
        a.number_input("Lissage des positions (points)", 1, 15, step=1, **w("pa_smooth", 3, (1, 15)),
                       help="Moyenne glissante des positions avant de calculer vitesses et distances "
                            "(sépare le bruit de localisation du vrai mouvement).")
        b.number_input("Fenêtre de vitesse (frames)", 1, 15, step=1, **w("pa_window", 3, (1, 15)))
        c.number_input("Durée min au-dessus du seuil (points)", 1, 10, step=1, **w("pa_persist", 2, (1, 10)))
        d.number_input("Points min par trajectoire", 3, 200, step=1, **w("pa_minpts", 5, (3, 200)))
        a, b, _ = st.columns([1, 1, 2])
        a.number_input(f"Seuil de mouvement ({res.velocity_unit}), 0 = auto", 0.0, format="%.4g", **w("pa_thr", 0.0))
        b.number_input("Exclure autour du pit (px)", 0.0, 1000.0, step=5.0, **w("pa_excl", float(round(auto_excl))),
                       help=f"Le bord sombre du pit est souvent pris pour des particules. Estimé sur la "
                            f"dernière frame : {auto_excl:.0f} px.")

    try:
        ctx = analysis_ctx()
    except ValueError as e:
        st.error(str(e))
        return
    pa, msds, tracks = ctx["pa"], ctx["msds"], ctx["tracks"]

    # ---------------------------------------------------------------- nom, enregistrement, comparaisons
    st.subheader("Nom, enregistrement et comparaisons")
    a, b = st.columns([2, 1])
    a.text_input("Nom de cette analyse (légendes des figures)", **w("pa_name", ""), placeholder="Pit A, NiCr (+) 1.6 V…",
                 help="Vide : « Pit A ». Plusieurs pits dans la vidéo : un groupe par pit, « Pit A », « Pit B »… "
                      "(préfixés par ce nom s'il est donné). Les traceurs sont rattachés au pit le plus proche de "
                      "leur première détection.")
    label = str(ss.get("pa_name") or "").strip() or ds.name
    b.markdown("&nbsp;")
    if b.button("💾 Enregistrer pour comparer", key="btn_save_analysis",
                help="Garde cette analyse (traceurs + description de l'expérience ci-dessus) dans le dossier du "
                     "dataset, pour la superposer à d'autres pits ou expériences et pour la section Traceurs."):
        (ds.folder / "experiment.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False), encoding="utf-8")
        folder = cmpmod.save_analysis(pa, ds.folder, label, {"dataset": ds.name, "dataset_path": path,
                                                               "experiment": exp, "pit_radii_px": ctx["R_px"],
                                                               "method": res.method})
        st.success(f"Analyse « {label} » enregistrée (avec la description de l'expérience) dans `{folder}`.")
    saved = cmpmod.list_saved(DATA_ROOT)
    info = {s["path"]: s for s in saved}
    ss["cmp_saved"] = [p for p in ss.get("cmp_saved", []) if p in info]
    st.multiselect("Superposer des analyses enregistrées (autres pits, autres datasets)", list(info),
                   format_func=lambda p: (f"{info[p].get('label')} — {info[p].get('dataset')} "
                                          f"({TRACER_TYPES.get((info[p].get('experiment') or {}).get('tracers'), '?')})"),
                   **w("cmp_saved", []),
                   help="Les groupes choisis sont superposés dans toutes les sections (« Pit A mean », « Pit B "
                        "mean »…) et comparés par des tests statistiques.")
    st.caption("Groupes comparés dans les sections : " + ", ".join(f"**{g.label}**" for g in ctx["groups"]) + ".")

    # ---------------------------------------------------------------- résumé
    st.subheader("Résumé")
    P = pa.particles
    mob = P[P["mobile"]]
    fits = {lab: f for lab, (_, f) in msds.items()}
    cols = st.columns(4)
    cols[0].metric("Traceurs analysés", len(P), f"{100 * len(mob) / max(len(P), 1):.0f} % mobiles", delta_color="off")
    cols[1].metric(f"Vitesse en mouvement ({res.velocity_unit})", f"{mob['moving_speed'].median():.3g}" if len(mob) else "—")
    cols[2].metric(f"Seuil de mouvement ({res.velocity_unit})", f"{pa.threshold:.3g}")
    cols[3].metric("Pits", len(pits))
    with st.container(border=True):
        for line in pm.summary(pa, fits):
            st.markdown(line)
    if st.checkbox("Tester la robustesse au seuil (refait l'analyse avec seuil × 0.5, × 1, × 2)", key="pa_sens"):
        sens = cached_fig("sens", ctx["csig"], lambda: pm.threshold_sensitivity(
            tracks, ds.times, pits, res.scale, res.length_unit, res.time_unit, pa.params, pa.threshold))
        st.dataframe(sens.round(3), hide_index=True, width="stretch")
        st.caption("Une conclusion n'est solide que si elle reste vraie sur les trois lignes.")

    with st.expander("✅ Qualité du suivi et tableau par traceur"):
        show_png("qual", ctx["sig"], lambda: pplots.fig_tracking_quality(tracks, ds.times, res.time_unit))
        span = tracks.groupby("particle")["frame"].agg(lambda f: f.max() - f.min() + 1)
        st.caption(f"Durée médiane des trajectoires : {span.median():.0f} frames. Si elle est courte (< 10), les "
                   "trajectoires se cassent : augmente « Mémoire » (3–5) et « Déplacement max », ou ajuste minmass."
                   if is_tp else f"Durée médiane des traceurs virtuels : {span.median():.0f} fenêtres. Un traceur "
                   "virtuel s'arrête quand il sort de la zone mesurée (jamais d'extrapolation) : réduis « 1 traceur "
                   "virtuel tous les N cellules » pour en semer plus.")
        vu, lu, T = res.velocity_unit, res.length_unit, res.time_unit
        show_cols = {"particle": "traceur", "pit": "pit", "status": "état", "onset_time": f"démarrage ({T})",
                     "moving_speed": f"vitesse en mvt ({vu})", "max_speed": f"vitesse max ({vu})",
                     "net_displacement": f"déplacement net ({lu})", "path_length": f"chemin ({lu})",
                     "straightness": "rectitude", "r0": f"distance initiale au pit ({lu})", "cos_radial": "cos direction",
                     "v_radial_moving": f"v radiale ({vu})", "v_tangential_moving": f"v tangentielle ({vu})",
                     "t_first": f"1re détection ({T})", "duration": f"durée ({T})"}
        tab = P[[c for c in show_cols if c in P]].assign(pit=lambda d: d["pit"].map(cmpmod.pit_label))
        st.dataframe(tab.rename(columns=show_cols).round(3), hide_index=True, width="stretch")

    st.subheader("Télécharger tout")
    st.caption("Les figures et calculs des sections Vitesse → Relation avec le pit (PNG 300 dpi + SVG, CSV) et un "
               "résumé texte, avec les options choisies dans chaque section. Aussi inclus dans le zip de la page "
               "Export. La section Traceurs a son propre zip.")
    if st.button("Préparer le zip complet de l'analyse", key="btn_pa_zip"):
        with st.spinner("Figures…"):
            buf = pyio.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in particle_report_ctx(ctx).items():
                    zf.writestr(f"{ds.name}_analyse/{name}", data)
            ss.pa_zip = buf.getvalue()
    if ss.get("pa_zip"):
        st.download_button("Télécharger", ss.pa_zip, file_name=f"{ds.name}_analyse.zip",
                           mime="application/zip", type="primary", key="dl_pa_zip")


# --------------------------------------------------------------------------- 1. Vitesse
# Chaque section est décrite par une fonction (calculs -> tableaux, figures -> constructeurs),
# réutilisée pour l'affichage, le zip de la section et le zip complet.

def speed_opts() -> dict:
    return {"mobile_only": bool(ss.get("qv_mobile", True)), "max_time": float(ss.get("qv_tmax") or 0.0),
            "align_pit": bool(ss.get("qv_align", False)), "normalized": bool(ss.get("qv_norm", False))}


def speed_section(ctx: dict, o: dict) -> tuple[dict, list]:
    G, pa = ctx["groups"], ctx["pa"]
    vu = pa.velocity_unit
    summ = calc("speed", ctx, lambda: {g.label: qmod.speed_summary(g.pa) for g in G})
    rho = calc("speed_rho", ctx, lambda: {g.label: _spearman(g.pa.particles[g.pa.particles["mobile"]], "r0",
                                                             "moving_speed") for g in G})
    rows = []
    for g in G:
        s, rr = summ[g.label], rho[g.label]
        rows.append({"groupe": g.label, "traceurs": s["n"], "mobiles (%)": qmod.fmt_ci(s["mobile"], "{:.0f}", 100),
                     f"vitesse en mouvement, médiane ({vu})": qmod.fmt_ci(s["moving_speed"]),
                     f"80 % des traceurs entre ({vu})": f"{s['p10']:.3g} – {s['p90']:.3g}" if np.isfinite(s["p10"]) else "—",
                     f"vitesse max, médiane ({vu})": qmod.fmt_ci(s["max_speed"]),
                     "vitesse ↔ distance au pit (Spearman)": f"ρ = {rr[0]:.2f} (p = {rr[1]:.2g})" if rr else "—"})
    norm = o["normalized"] and ctx["has_r"]
    figs = [("since_first", "Vitesse de chaque traceur depuis sa première détection (traits fins) et moyenne de chaque "
             "groupe (trait épais), comme la figure de référence (b).",
             lambda: qplots.fig_speed_since_first(G, o["mobile_only"], o["max_time"] or None),
             (o["mobile_only"], o["max_time"])),
            ("distribution", "Distribution des vitesses en mouvement (une valeur par traceur) ; pointillés : médianes.",
             lambda: qplots.fig_speed_distribution(G), ()),
            ("time", "Vitesse médiane au cours du temps (bande : 50 % central des traceurs)"
             + (", temps aligné sur l'apparition du pit." if o["align_pit"] else " ; tirets : apparition du pit."),
             lambda: qplots.fig_speed_time(G, o["align_pit"]), (o["align_pit"],)),
            ("distance", "Vitesse des traceurs en mouvement selon la distance au pit (médiane, 50 % central)"
             + (", distance rapportée au rayon du pit (r / R)." if norm else "."),
             lambda: qplots.fig_speed_distance(G, norm), (norm,)),
            ("tracks", "Trajectoires colorées par la vitesse (analyse en cours) : cercle + flèche = position et "
             "direction à la frame choisie ; pointillés rouges = bord du pit.",
             lambda: fig_tracks_speed_for(pa, ctx["ds"], ctx["path"], ctx["mtime"], ctx["radii"]),
             (tuple(sorted(speed_fig_settings(len(ctx["ds"])).items())), ctx["sb"]))]
    return {"vitesse": pd.DataFrame(rows)}, figs


def speed_figure_controls(ctx: dict):
    """Réglages de la figure « trajectoires colorées par la vitesse »."""
    ds, res = ctx["ds"], ctx["res"]
    st.markdown("**Figure des trajectoires (e)**")
    a, b, c, d = st.columns(4)
    a.number_input(f"Échelle max ({res.velocity_unit}), 0 = auto", 0.0, format="%.4g", **w("paf_vmax", 0.0),
                   help="Auto : 99e percentile des vitesses affichées. Fixe-la pour comparer plusieurs pits.")
    b.number_input(f"Déplacement net min ({res.length_unit})", 0.0, step=1.0, format="%.4g", **w("paf_disp", 0.0),
                   help="Ne garde que les traceurs qui vont quelque part (retire ceux qui piétinent sur place).")
    c.number_input("Points min / trajectoire", 2, 1000, step=1, **w("paf_minpts", 5, (2, 1000)))
    d.number_input("Lissage d'affichage (points)", 1, 25, step=1, **w("paf_smooth", 3, (1, 25)))
    a, b, c = st.columns([2, 1, 1])
    a.slider("Trajectoires jusqu'à la frame", 0, len(ds) - 1, **w("paf_end", len(ds) - 1, (0, len(ds) - 1)),
             help="Cercle + flèche : position et direction des traceurs à cette frame, qui sert aussi d'image de fond.")
    b.checkbox("Seulement les mobiles", **w("paf_mobile", True))
    c.checkbox("Cercle + flèche", **w("paf_ends", True))
    st.caption("Étiquette de panneau (« Pit A »), barre d'échelle et contour du pit : encadré « Barre d'échelle et "
               "style des figures » (page Champ de vitesse ou Export).")


def page_q_speed():
    ctx = q_header("Vitesse")
    if not ctx:
        return
    G, pa, T = ctx["groups"], ctx["pa"], ctx["pa"].time_unit
    vu = pa.velocity_unit
    with st.expander("Options des figures"):
        a, b, c, d = st.columns(4)
        a.checkbox("(a) Traceurs mobiles seulement", **w("qv_mobile", True))
        b.number_input(f"(a) Durée max ({T}), 0 = auto", 0.0, format="%.4g", **w("qv_tmax", 0.0))
        c.checkbox("(c) Temps aligné sur le pit", **w("qv_align", False))
        d.checkbox("(d) Distance r / R", **w("qv_norm", False), disabled=not ctx["has_r"],
                   help="Distance rapportée au rayon du pit : compare des pits de tailles différentes.")
        speed_figure_controls(ctx)
    tables, figs = speed_section(ctx, speed_opts())
    st.subheader("Calculs")
    st.dataframe(tables["vitesse"], hide_index=True, width="stretch")
    summ = calc("speed", ctx, lambda: {g.label: qmod.speed_summary(g.pa) for g in G})
    for g in G:
        s = summ[g.label]
        if s["n_mobile"]:
            st.markdown(f"**{g.label}** : {qmod.fmt_ci(s['mobile'], '{:.0f}', 100)} % des traceurs bougent ; vitesse "
                        f"en mouvement médiane **{qmod.fmt_ci(s['moving_speed'])} {vu}**, 80 % des traceurs mobiles "
                        f"entre {s['p10']:.3g} et {s['p90']:.3g} {vu}.")
        else:
            st.markdown(f"**{g.label}** : aucun traceur mobile.")
    t = _test_sentence({g.label: g.pa.particles.loc[g.pa.particles["mobile"], "moving_speed"] for g in G},
                       "vitesse en mouvement")
    if t:
        st.markdown(t)
    st.caption("Entre crochets : intervalle de confiance à 95 % (bootstrap sur les traceurs ; intervalle de Wilson "
               "pour un pourcentage). ρ de Spearman : lien entre la vitesse d'un traceur et sa distance initiale au "
               "pit (< 0 : plus lent loin du pit).")
    render_section(ctx, "Vitesse", figs, tables)


# --------------------------------------------------------------------------- 2. Homogénéité

def homogeneity_opts() -> dict:
    return {"align_pit": bool(ss.get("qh_align", False))}


def homogeneity_section(ctx: dict, o: dict) -> tuple[dict, list]:
    G, pa = ctx["groups"], ctx["pa"]
    T = pa.time_unit
    summ = calc("homog", ctx, lambda: {g.label: qmod.homogeneity_summary(g.pa) for g in G})
    bg = calc("behaviour", ctx, lambda: qmod.behaviour_groups(pa))
    rows = []
    for g in G:
        h = summ[g.label]
        rows.append({"groupe": g.label, "CV des vitesses": qmod.fmt_ci(h["speed_cv"], "{:.2f}"),
                     f"80 % des démarrages entre ({T})": "{:.4g} – {:.4g}".format(*h["onset_spread"])
                     if np.isfinite(h["onset_spread"][0]) else "—",
                     f"50 % en mouvement à t = ({T})": _f(h["t_half_moving"], none="jamais"),
                     "en mouvement au maximum (%)": _f(100 * h["max_moving_fraction"], "{:.0f}"),
                     "CV angulaire": _f(h["angular_cv"], "{:.2f}")})
    tables = {"homogeneite": pd.DataFrame(rows)}
    if bg:
        tables["groupes_de_comportement"] = bg["table"]
    figs = [("moving_fraction", "Part des traceurs en mouvement au cours du temps.",
             lambda: qplots.fig_moving_fraction(G, o["align_pit"]), (o["align_pit"],)),
            ("started", "Part cumulée des traceurs déjà mis en mouvement : une montée raide = démarrage simultané.",
             lambda: qplots.fig_started_fraction(G, o["align_pit"]), (o["align_pit"],)),
            ("distribution", "Distribution des vitesses en mouvement : étroite = vitesses comparables ; plusieurs "
             "bosses = plusieurs groupes.", lambda: qplots.fig_speed_distribution(G), ()),
            ("angular", "Vitesse médiane selon la direction autour du pit : un cercle = mouvement homogène autour du pit.",
             lambda: qplots.fig_angular_speed(G), ())]
    if bg and bg["k"] > 1:
        names = {r_["group"]: f"{r_['group']}: {r_['speed']:.2g} {style.vel_unit(pa.length_unit, T)}"
                 for _, r_ in bg["table"].iterrows()}
        figs += [("behaviour", "Groupes de comportement (mélange de gaussiennes) : vitesse, rectitude, direction, "
                  "persistance de chaque traceur mobile (analyse en cours).",
                  lambda: qplots.fig_behaviour_groups(pa, bg), ()),
                 ("behaviour_tracks", "Trajectoires colorées par groupe de comportement (analyse en cours).",
                  lambda: qplots.fig_tracks_categorical(
                      pa, window_background(ctx["path"], ctx["mtime"], 0, len(ctx["ds"]) - 1), bg["labels"], names,
                      ctx["sb"], ctx["radii"], ctx["panel"]), (ctx["sb"],))]
    return tables, figs


def page_q_homogeneity():
    ctx = q_header("Homogénéité")
    if not ctx:
        return
    G, pa = ctx["groups"], ctx["pa"]
    with st.expander("Options des figures"):
        st.checkbox("Temps aligné sur l'apparition du pit", **w("qh_align", False))
    tables, figs = homogeneity_section(ctx, homogeneity_opts())
    summ = calc("homog", ctx, lambda: {g.label: qmod.homogeneity_summary(g.pa) for g in G})
    bg = calc("behaviour", ctx, lambda: qmod.behaviour_groups(pa))
    st.subheader("Calculs")
    st.dataframe(tables["homogeneite"], hide_index=True, width="stretch")
    for g in G:
        h = summ[g.label]
        cv = h["speed_cv"][0]
        if not np.isfinite(cv):
            continue
        txt = ("vitesses assez homogènes" if cv < 0.3 else "vitesses très hétérogènes" if cv > 0.6
               else "dispersion modérée des vitesses")
        lo, hi = h["onset_spread"]
        st.markdown(f"**{g.label}** : CV des vitesses = {qmod.fmt_ci(h['speed_cv'], '{:.2f}')} → {txt} ; "
                    + (f"80 % des démarrages entre t = {lo:.4g} et {hi:.4g} {pa.time_unit}."
                       if np.isfinite(lo) else "pas assez de démarrages observés pour juger de leur simultanéité."))
    t = _test_sentence({g.label: g.pa.particles.loc[g.pa.particles["mobile"], "moving_speed"] for g in G},
                       "vitesse en mouvement")
    if t:
        st.markdown(t)
    st.markdown("**Groupes de comportement** (analyse en cours, tous pits réunis)")
    if bg is None:
        st.info("Pas assez de traceurs mobiles (≥ 30) pour chercher des groupes.")
    elif bg["k"] == 1:
        st.markdown("Un seul groupe : la population est **homogène** ("
                    + ("un 2e groupe n'améliore pas assez le modèle, critère BIC)." if bg["features"] else
                       "aucune caractéristique ne varie assez d'un traceur à l'autre pour séparer des groupes)."))
    else:
        st.markdown(f"**{bg['k']} groupes** de traceurs aux comportements différents (critère BIC, preuve forte), "
                    "du plus lent au plus rapide :")
    if bg is not None:
        names = {"moving_speed": "vitesse", "straightness": "rectitude", "cos_radial": "direction par rapport au pit",
                 "persistence": "persistance"}
        unused = [names[f] for f in names if f not in bg["features"]]
        st.caption("Caractéristiques utilisées : " + (", ".join(names[f] for f in bg["features"]) or "aucune")
                   + (f" ; écartées car presque identiques d'un traceur à l'autre : {', '.join(unused)}." if unused
                      else "."))
        tab = bg["table"].rename(columns={"group": "groupe", "n": "traceurs", "share": "part",
                                          "speed": f"vitesse ({pa.velocity_unit})", "straightness": "rectitude",
                                          "cos_radial": "direction (cos)", "persistence": "persistance",
                                          "r0": f"distance initiale au pit ({pa.length_unit})"})
        st.dataframe(tab.round(3), hide_index=True, width="stretch")
    st.caption("CV : écart-type / moyenne des vitesses en mouvement (une valeur par traceur). CV angulaire : variation "
               "de la vitesse médiane selon la direction autour du pit (0 = parfaitement homogène). Groupes : mélange "
               "de gaussiennes sur log(vitesse), rectitude, direction et persistance ; un groupe de plus n'est retenu "
               "que s'il améliore le BIC de plus de 10.")
    render_section(ctx, "Homogénéité", figs, tables)


# --------------------------------------------------------------------------- 3. Déplacement

def displacement_calc(ctx: dict) -> tuple[dict, dict]:
    G = ctx["groups"]
    alphas = calc("alphas", ctx, lambda: {g.label: qmod.track_alpha(g.pa) for g in G})
    summ = calc("displ", ctx, lambda: {g.label: qmod.displacement_summary(g.pa, alphas.get(g.label)) for g in G})
    return alphas, summ


def displacement_section(ctx: dict) -> tuple[dict, list]:
    G, pa = ctx["groups"], ctx["pa"]
    lu = pa.length_unit
    alphas, summ = displacement_calc(ctx)
    rows = []
    for g in G:
        d = summ[g.label]
        row = {"groupe": g.label, f"déplacement net ({lu})": qmod.fmt_ci(d["net_displacement"]),
               f"chemin parcouru ({lu})": qmod.fmt_ci(d["path_length"]),
               "rectitude": qmod.fmt_ci(d["straightness"], "{:.2f}"),
               "s'éloignent du pit (%)": qmod.fmt_ci(d["away"], "{:.0f}", 100),
               "se rapprochent (%)": qmod.fmt_ci(d["toward"], "{:.0f}", 100),
               "de côté (%)": qmod.fmt_ci(d["sideways"], "{:.0f}", 100),
               "part tangentielle": qmod.fmt_ci(d["tangential_share"], "{:.2f}"),
               "persistance": qmod.fmt_ci(d["persistence"], "{:.2f}")}
        if "alpha" in d:
            row["α médian"] = qmod.fmt_ci(d["alpha"], "{:.2f}")
            for k_, v_ in d["motion_types"].items():
                row[f"{MOTION_FR[k_]} (%)"] = qmod.fmt_ci(v_, "{:.0f}", 100)
        rows.append(row)
    cur = [alphas[g.label] for g in G if g.meta.get("source") == "current" and g.label in alphas]
    cats = pd.concat(cur).set_index("particle")["motion_type"] if cur else pd.Series(dtype=str)
    figs = [("displacement", "Déplacement net (gauche) et chemin parcouru (droite) de chaque traceur mobile ; "
             "pointillés : médianes.", lambda: qplots.fig_displacement(G), ()),
            ("direction", "Direction du déplacement net par rapport au pit (vers le haut : s'éloigne du pit).",
             lambda: qplots.fig_direction_rose(G), ()),
            ("radial_tangential", "Composantes radiale et tangentielle de la vitesse de chaque traceur : sur l'axe "
             "horizontal = mouvement radial ; loin de l'axe = rotation autour du pit.",
             lambda: qplots.fig_radial_tangential(G), ()),
            ("alpha", "Exposant α du MSD de chaque traceur (trajectoires ≥ 10 points) : confiné / diffusif / dirigé.",
             lambda: qplots.fig_alpha(G, alphas), ()),
            ("msd", "MSD de l'ensemble des traceurs (analyse en cours), avant / après le pit : α ≈ 1 brownien, "
             "α ≈ 2 dirigé.", lambda: pplots.fig_msd(ctx["msds"], pa.length_unit, pa.time_unit), ()),
            ("turning", "Angle entre deux pas successifs : pic à 0° = trajectoires rectilignes ; plat = diffusion.",
             lambda: qplots.fig_turning_angles(G), ())]
    if len(cats):
        figs.append(("motion_tracks", "Trajectoires colorées par type de mouvement (analyse en cours).",
                     lambda: qplots.fig_tracks_categorical(
                         pa, window_background(ctx["path"], ctx["mtime"], 0, len(ctx["ds"]) - 1), cats,
                         qmod.MOTION_TYPES, ctx["sb"], ctx["radii"], ctx["panel"]), (ctx["sb"],)))
    tables = {"deplacement": pd.DataFrame(rows)}
    if any(len(a) for a in alphas.values()):
        tables["alpha_par_traceur"] = pd.concat([a.assign(groupe=lab) for lab, a in alphas.items()])
    return tables, figs


def page_q_displacement():
    ctx = q_header("Déplacement")
    if not ctx:
        return
    G, pa = ctx["groups"], ctx["pa"]
    tables, figs = displacement_section(ctx)
    _, summ = displacement_calc(ctx)
    st.subheader("Calculs")
    st.dataframe(tables["deplacement"], hide_index=True, width="stretch")
    for g in G:
        d = summ[g.label]
        s = d["straightness"][0]
        if not np.isfinite(s):
            continue
        shape = "quasi rectilignes" if s > 0.8 else "sinueuses" if s < 0.4 else "modérément sinueuses"
        txt = (f"**{g.label}** : déplacement net médian {qmod.fmt_ci(d['net_displacement'])} {pa.length_unit}, "
               f"trajectoires {shape} (rectitude {s:.2f}) ; {100 * d['away'][0]:.0f} % s'éloignent du pit, "
               f"{100 * d['toward'][0]:.0f} % s'en rapprochent")
        tang = d["tangential_share"][0]
        if np.isfinite(tang):
            txt += (f" ; rotation autour du pit {'négligeable' if tang < 0.25 else 'notable' if tang < 0.5 else 'dominante'}"
                    f" (part tangentielle {tang:.2f})")
        if d.get("n_alpha"):
            main = max(d["motion_types"], key=lambda k_: d["motion_types"][k_][0])
            txt += (f" ; mouvement majoritairement **{MOTION_FR[main]}** ({100 * d['motion_types'][main][0]:.0f} % de "
                    f"{d['n_alpha']} traceurs, α médian {d['alpha'][0]:.2f})")
        st.markdown(txt + ".")
    t = _test_sentence({g.label: g.pa.particles.loc[g.pa.particles["mobile"], "straightness"] for g in G}, "rectitude")
    if t:
        st.markdown(t)
    fits = {lab: f for lab, (_, f) in ctx["msds"].items() if f}
    if fits:
        st.markdown("MSD d'ensemble (analyse en cours) : " + " ; ".join(f"{k} α = {f['alpha']:.2f}" for k, f in fits.items())
                    + ".")
    st.caption("Rectitude : déplacement net / chemin parcouru (1 = ligne droite). Part tangentielle : |v tangentielle| / "
               "(|v radiale| + |v tangentielle|). Persistance : cosinus moyen de l'angle entre deux pas successifs "
               "(1 = tout droit, 0 = au hasard). α par traceur : le bruit de localisation abaisse α des traceurs presque "
               "immobiles (ils peuvent apparaître « confinés »).")
    render_section(ctx, "Déplacement", figs, tables)


# --------------------------------------------------------------------------- 4. Début du mouvement

def onset_opts() -> dict:
    return {"rel_pit": bool(ss.get("qo_rel", True))}


def onset_calc(ctx: dict) -> dict:
    return calc("onset", ctx, lambda: {g.label: qmod.onset_summary(g.pa) for g in ctx["groups"]})


def onset_section(ctx: dict, o: dict) -> tuple[dict, list]:
    G, pa = ctx["groups"], ctx["pa"]
    T, vu = pa.time_unit, pa.velocity_unit
    summ = onset_calc(ctx)
    rows = []
    for g in G:
        s = summ[g.label]
        fv = s["front"]
        vci = "—"
        if fv and fv["velocity"]:
            lo, hi = fv["slope_ci"]
            vci = f"{fv['velocity']:.3g}" + (f" [{1 / hi:.3g} – {1 / lo:.3g}]" if lo > 0 and hi > 0 else "")
        rows.append({"groupe": g.label, "démarrages observés": s["n_starts"],
                     "déjà en mouvement à la 1re détection (%)": qmod.fmt_ci(s["already"], "{:.0f}", 100),
                     f"démarrage médian ({T})": qmod.fmt_ci(s["onset"]),
                     f"démarrage après le pit ({T})": qmod.fmt_ci(s["onset_rel_pit"]),
                     f"80 % des démarrages entre ({T})": "{:.4g} – {:.4g}".format(*s["spread"])
                     if np.isfinite(s["spread"][0]) else "—",
                     f"50 % en mouvement à t = ({T})": _f(s["t_half_moving"], none="jamais"),
                     "démarrage ↔ distance (Spearman)": f"ρ = {fv['rho']:.2f} (p = {fv['p']:.2g})" if fv else "—",
                     f"vitesse du front ({vu})": vci})
    figs = [("onset_hist", "Instants de démarrage" + (" par rapport à l'apparition du pit (tirets)." if o["rel_pit"]
                                                      else "."),
             lambda: qplots.fig_onset_hist(G, o["rel_pit"]), (o["rel_pit"],)),
            ("onset_distance", "Démarrage selon la distance initiale au pit : une pente positive = mise en mouvement "
             "qui se propage depuis le pit (vitesse du front = 1 / pente).", lambda: qplots.fig_onset_distance(G), ()),
            ("started", "Part cumulée des traceurs déjà en mouvement : montée raide = démarrage collectif ; montée "
             "lente = démarrage progressif.", lambda: qplots.fig_started_fraction(G, o["rel_pit"]), (o["rel_pit"],)),
            ("onset_map", "Carte des démarrages : position de départ de chaque traceur colorée par son instant de "
             "démarrage (analyse en cours).",
             lambda: qplots.fig_onset_map(pa, window_background(ctx["path"], ctx["mtime"], 0, 0), ctx["sb"],
                                          ctx["radii"], ctx["panel"]), (ctx["sb"],))]
    return {"debut_du_mouvement": pd.DataFrame(rows)}, figs


def page_q_onset():
    ctx = q_header("Début du mouvement")
    if not ctx:
        return
    G, pa = ctx["groups"], ctx["pa"]
    with st.expander("Options des figures"):
        st.checkbox("Temps par rapport à l'apparition du pit", **w("qo_rel", True))
    tables, figs = onset_section(ctx, onset_opts())
    summ = onset_calc(ctx)
    st.subheader("Calculs")
    st.dataframe(tables["debut_du_mouvement"], hide_index=True, width="stretch")
    dur = float(pa.steps["t"].max() - pa.steps["t"].min()) or 1.0
    for g in G:
        o = summ[g.label]
        if o["n_starts"] < 5:
            st.markdown(f"**{g.label}** : trop peu de démarrages observés ({o['n_starts']}) pour conclure.")
            continue
        fv = o["front"]
        if o["spread"][1] - o["spread"][0] < 0.1 * dur:
            kind = "**collective** (quasi simultanée)"
        elif fv and fv["p"] < 0.05 and fv["slope"] > 0:
            kind = f"**progressive** : elle se propage depuis le pit à ≈ {fv['velocity']:.3g} {pa.velocity_unit}"
        else:
            kind = "**étalée dans le temps**, sans lien net avec la distance au pit"
        st.markdown(f"**{g.label}** : démarrage médian {qmod.fmt_ci(o['onset'])} {pa.time_unit}"
                    + (f" ({qmod.fmt_ci(o['onset_rel_pit'])} {pa.time_unit} après le pit)"
                       if np.isfinite(o["onset_rel_pit"][0]) else "") + f" ; mise en mouvement {kind}.")
        if o["already"][0] > 0.5:
            st.caption(f"⚠️ {g.label} : {100 * o['already'][0]:.0f} % des traceurs mobiles bougent déjà à leur première "
                       "détection : les démarrages observés surreprésentent les démarrages tardifs. La courbe de part "
                       "cumulée (c) est plus fiable.")
    rel = all(g.pa.params.t_pit is not None for g in G)
    t = _test_sentence({g.label: g.pa.particles.loc[g.pa.particles["status"] == "démarre", "onset_time"]
                        - (g.pa.params.t_pit if rel else 0.0) for g in G},
                       "instant de démarrage" + (" par rapport au pit" if rel else ""))
    if t:
        st.markdown(t)
    st.caption("« Démarrage observé » : traceur vu immobile puis en mouvement (son instant de démarrage est mesuré). "
               "Vitesse du front : 1 / pente de la régression robuste (Theil-Sen) du démarrage sur la distance "
               "initiale au pit.")
    render_section(ctx, "Début du mouvement", figs, tables)


# --------------------------------------------------------------------------- 5. Relation avec le pit

def pit_opts(ctx: dict) -> dict:
    pa, ds = ctx["pa"], ctx["ds"]
    pit0 = tuple(map(float, ctx["res"].pits[0]))
    radius = pit_radius_series_cached(ctx["path"], ctx["mtime"], pit0)
    frac = float(ss.get("qp_frac", 0.9))
    t_end = float(ss.get("qp_tend") or 0.0) or None
    phases = qmod.pit_phases(radius, pa.params.t_pit, float(ds.times[0]), float(ds.times[-1]), frac, t_end)
    return {"phases": phases, "radius": radius,
            "signal": pit_signal_cached(ctx["path"], ctx["mtime"], pit0, float(ss.get("pa_sig_r", 15))),
            "normalized": bool(ss.get("qp_norm", False)) and ctx["has_r"],
            "phase_key": (phases["t_pit"], phases["t_end_growth"], frac)}


def pit_section(ctx: dict, o: dict) -> tuple[dict, list, tuple]:
    G, pa, ds = ctx["groups"], ctx["pa"], ctx["ds"]
    phases, radius, norm = o["phases"], o["radius"], o["normalized"]
    vu, T = pa.velocity_unit, pa.time_unit

    def compute():
        return (qmod.phase_stats(pa, phases),
                {name: qmod.distance_fit(pa, rng, norm) for name, rng in phases["phases"].items()})

    ps, fits = cached_fig("calc_pit", ctx["csig"] + (o["phase_key"], norm), compute)
    phase_tab = pd.DataFrame({
        "phase": ps["phase"].map(lambda n: PHASE_FR.get(n, n)), f"début ({T})": ps["t_start"],
        f"fin ({T})": ps["t_end"], "traceurs": ps["tracers"],
        "pas en mouvement (%)": (100 * ps["moving_share"]).round(0),
        f"vitesse en mouvement ({vu})": [qmod.fmt_ci(t) for t in zip(ps["speed"], ps["speed_lo"], ps["speed_hi"])],
        f"vitesse radiale ({vu})": [qmod.fmt_ci(t) for t in zip(ps["v_radial"], ps["v_radial_lo"], ps["v_radial_hi"])]})
    fit_tab = pd.DataFrame([{"phase": PHASE_FR.get(n, n), "exposant n (v ∝ r⁻ⁿ)": _f(f["n"], "{:.2f}") if f else "—",
                             "IC 95 %": "{:.2f} – {:.2f}".format(*f["n_ci"]) if f else "—",
                             "R²": _f(f["r2"], "{:.2f}") if f else "—",
                             "distance": ("r / R" if f["column"] == "r_norm" else f"r ({pa.length_unit})") if f else "—"}
                            for n, f in fits.items()])
    tables = {"phases": phase_tab, "loi_distance": fit_tab,
              "rayon_du_pit": radius.assign(R_phys=radius["R_smooth"] * pa.scale)}
    figs = [("pit_kinetics", "Intensité autour du pit (haut) et rayon du pit (bas) au cours du temps ; fonds : avant le "
             "pit, croissance, après la croissance.",
             lambda: qplots.fig_pit_kinetics(ds.times, o["signal"], radius, phases, pa.scale, pa.length_unit, T),
             (o["phase_key"],)),
            ("phase_timeseries", "Vitesse (haut) et vitesse radiale (bas) des traceurs au cours du temps, phases du pit "
             "en fond.", lambda: qplots.fig_phase_timeseries(pa, phases), (o["phase_key"],)),
            ("distance_phases", "Vitesse selon la distance au pit pour chaque phase (log-log) et loi de puissance "
             "ajustée v ∝ r⁻ⁿ : n > 0 = plus lent loin du pit.",
             lambda: qplots.fig_distance_phases(pa, phases, norm), (o["phase_key"], norm)),
            ("radial_map", "Vitesse radiale selon la distance au pit et le temps (rouge : s'éloigne, bleu : se rapproche).",
             lambda: qplots.fig_radial_map(pa, phases), (o["phase_key"],)),
            ("r_t", "Distance au pit de chaque traceur au cours du temps (couleur : vitesse en mouvement).",
             lambda: pplots.fig_r_t(pa), ()),
            ("aligned", "Vitesse médiane alignée sur l'apparition du pit, pour chaque groupe.",
             lambda: qplots.fig_speed_time(G, True), ())]
    return tables, figs, (ps, fits)


def page_q_pit():
    ctx = q_header("Relation avec le pit")
    if not ctx:
        return
    pa = ctx["pa"]
    with st.expander("Options : phases du pit et distance"):
        a, b, c = st.columns(3)
        a.slider("Fin de croissance : rayon ≥ … × rayon final", 0.5, 0.99, step=0.01, **w("qp_frac", 0.9))
        b.number_input(f"Fin de croissance imposée ({pa.time_unit}), 0 = auto", 0.0, format="%.4g", **w("qp_tend", 0.0))
        c.checkbox("Distance r / R", **w("qp_norm", False), disabled=not ctx["has_r"],
                   help="Distance rapportée au rayon du pit.")
        st.caption("Apparition du pit : ⚙️ Réglages. La croissance va de l'apparition jusqu'à ce que le rayon atteigne "
                   "la fraction choisie de son rayon final.")
    o = pit_opts(ctx)
    phases, radius = o["phases"], o["radius"]
    tables, figs, (ps, fits) = pit_section(ctx, o)
    st.subheader("Calculs")
    R = radius["R_smooth"].to_numpy() * pa.scale
    a, b, c, d = st.columns(4)
    a.metric(f"Apparition du pit ({pa.time_unit})", _f(phases["t_pit"]))
    b.metric(f"Fin de croissance ({pa.time_unit})", _f(phases["t_end_growth"]))
    r_final = float(np.median(R[-max(3, len(R) // 10):])) if len(R) else np.nan
    c.metric(f"Rayon final ({pa.length_unit})", _f(r_final, "{:.3g}"))
    if phases["t_end_growth"] and phases["t_pit"] is not None and len(R):
        r_start = float(np.interp(phases["t_pit"], radius["t"], R))
        d.metric(f"Croissance moyenne ({style.vel_unit(pa.length_unit, pa.time_unit)})",
                 _f((r_final - r_start) / (phases["t_end_growth"] - phases["t_pit"]), "{:.3g}"))
    st.markdown("**Avant, pendant, après** (traceurs en mouvement : médiane des vitesses médianes par traceur, IC 95 %)")
    st.dataframe(tables["phases"], hide_index=True, width="stretch")
    sp = {PHASE_FR.get(n, n): v for n, v in zip(ps["phase"], ps["speed"]) if np.isfinite(v)}
    if len(sp) > 1:
        st.markdown("Vitesse en mouvement : " + " → ".join(f"{k} **{v:.3g}**" for k, v in sp.items())
                    + f" {pa.velocity_unit}.")
    st.markdown("**Dépendance à la distance au pit** : loi de puissance v ∝ r⁻ⁿ par phase")
    st.dataframe(tables["loi_distance"], hide_index=True, width="stretch")
    for n_, f in fits.items():
        if f:
            lo, hi = f["n_ci"]
            what = ("ralentit en s'éloignant du pit" if lo > 0 else "accélère en s'éloignant du pit" if hi < 0
                    else "ne dépend pas nettement de la distance")
            st.markdown(f"{PHASE_FR.get(n_, n_).capitalize()} : n = {f['n']:.2f} [{lo:.2f} – {hi:.2f}] → le mouvement "
                        f"**{what}**.")
    corr = cached_fig("calc_growth_corr", ctx["csig"] + (len(radius),),
                      lambda: qmod.speed_growth_correlation(pa, radius))
    if corr:
        st.markdown(f"Vitesse des traceurs ↔ vitesse de croissance du pit : ρ de Spearman = {corr['rho']:.2f} "
                    f"(p = {corr['p']:.2g}, {corr['n']} frames)"
                    + (" → **les traceurs vont plus vite quand le pit grandit vite**."
                       if corr["p"] < 0.05 and corr["rho"] > 0 else "."))
    if len(ctx["res"].pits) > 1:
        st.caption("Plusieurs pits : apparition, rayon et phases mesurés sur le premier pit (P1).")
    if phases["t_pit"] is None:
        st.info("Instant d'apparition du pit inconnu (⚙️ Réglages) : pas de découpage avant / pendant / après.")
    render_section(ctx, "Relation avec le pit", figs, tables)


# --------------------------------------------------------------------------- 6. Traceurs

def _group_value(meta: dict, field_: str) -> str:
    if field_ in ("label", "dataset"):
        return str(meta.get(field_) or "?")
    v = (meta.get("experiment") or {}).get(field_)
    if field_ == "tracers":
        return TRACER_TYPES.get(v, "?")
    if field_ == "potential_V" and v not in (None, ""):
        try:
            return f"{float(v):g} V"
        except (TypeError, ValueError):
            return str(v)
    return str(v) if v not in (None, "") else "?"


def tracers_figs(reps: dict, styles: dict, unit: str, T: str) -> list:
    def merged():
        return [cmpmod.Group(lab, cmpmod.merge(a), *styles[lab]) for lab, a in reps.items()]
    return [("indicators", "Valeurs par traceur (points) et médiane de chaque répétition (losanges) ; trait : médiane "
             "de toutes les répétitions. Parts en % : une barre par groupe, un losange par répétition.",
             lambda: qplots.fig_indicator_strips(reps, styles, unit, T), ()),
            ("aligned", "Vitesse médiane alignée sur l'apparition du pit : traits fins = répétitions, épais = moyenne.",
             lambda: qplots.fig_aligned_population(reps, styles, unit, T), ()),
            ("since_first", "Vitesse depuis la première détection, toutes répétitions réunies par groupe.",
             lambda: qplots.fig_speed_since_first(merged()), ()),
            ("direction", "Direction du déplacement par rapport au pit, par groupe.",
             lambda: qplots.fig_direction_rose(merged()), ())]


def page_q_tracers():
    st.header("Traceurs")
    st.markdown(f"*{QUESTIONS['Traceurs']}*")
    saved = cmpmod.list_saved(DATA_ROOT)
    try:
        ctx = analysis_ctx()
    except ValueError:
        ctx = None
    if not saved and ctx is None:
        st.info("Enregistre des analyses (⚙️ Réglages → « 💾 Enregistrer pour comparer »), par exemple une avec des "
                "traceurs positifs et une avec des traceurs négatifs, en décrivant chaque expérience.")
        return
    info = {s["path"]: s for s in saved}
    # Sélection gardée d'une visite à l'autre ; les analyses enregistrées depuis y sont ajoutées.
    known = set(ss.get("qt_known", []))
    ss["qt_sel"] = list(dict.fromkeys([p for p in ss.get("qt_sel", []) if p in info]
                                      + [p for p in info if p not in known]))
    ss["qt_known"] = list(info)
    a, b = st.columns([1, 2])
    field_ = a.selectbox("Comparer selon", list(GROUP_FIELDS), format_func=GROUP_FIELDS.get, **w("qt_field", "tracers"))
    sel = b.multiselect("Analyses enregistrées (une par répétition)", list(info),
                        format_func=lambda p: f"{info[p].get('label')} — {info[p].get('dataset')}",
                        **w("qt_sel", list(info)))
    include_cur = ctx is not None and st.checkbox("Inclure l'analyse en cours (non enregistrée)", **w("qt_cur", False))
    if not saved:
        st.caption("Aucune analyse enregistrée pour l'instant (⚙️ Réglages → « 💾 Enregistrer pour comparer »).")

    sel_m = tuple((p, _saved_mtime(p)) for p in sel if (Path(p) / "meta.json").exists())
    reps: dict[str, list] = {}
    rows = []
    for p, m in sel_m:
        try:
            spa, meta = load_saved_cached(p, m)
        except Exception as e:
            st.warning(f"{Path(p).name} : {e}")
            continue
        reps.setdefault(_group_value(meta, field_), []).append(spa)
        rows.append({"analyse": meta.get("label"), "dataset": meta.get("dataset"),
                     GROUP_FIELDS[field_]: _group_value(meta, field_), **qmod.indicators(spa)})
    if include_cur:
        cur_meta = {"label": str(ss.get("pa_name") or "") or ctx["ds"].name, "dataset": ctx["ds"].name,
                    "experiment": ctx["exp"]}
        reps.setdefault(_group_value(cur_meta, field_), []).append(ctx["pa"])
        rows.append({"analyse": f"{cur_meta['label']} (en cours)", "dataset": ctx["ds"].name,
                     GROUP_FIELDS[field_]: _group_value(cur_meta, field_), **qmod.indicators(ctx["pa"])})
    if not rows:
        st.info("Choisis au moins deux analyses à comparer.")
        return
    any_pa = next(iter(reps.values()))[0]
    vu, T = any_pa.velocity_unit, any_pa.time_unit
    ind_fr = {"tracers": "traceurs", "moving_speed": f"vitesse en mouvement ({vu})", "mobile": "mobiles (part)",
              "straightness": "rectitude", "away": "s'éloignent du pit (part)",
              "onset_rel_pit": f"démarrage après le pit ({T})", "persistence": "persistance"}
    st.subheader("Analyses comparées")
    table = pd.DataFrame(rows).rename(columns=ind_fr)
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption("Une ligne par analyse (répétition) : médianes sur les traceurs mobiles, parts entre 0 et 1.")
    if len(reps) < 2:
        st.info(f"Un seul groupe pour « {GROUP_FIELDS[field_]} » ({', '.join(reps)}). Il en faut au moins deux : décris "
                "chaque expérience (⚙️ Réglages → 🧪 Expérience) puis enregistre à nouveau ses analyses.")
        return
    tsig = ("tracers", field_, sel_m, include_cur, ctx["csig"] if (ctx and include_cur) else None)
    styles = {lab: cmpmod.GROUP_STYLES[i % len(cmpmod.GROUP_STYLES)] for i, lab in enumerate(reps)}

    def verdicts():
        out = []
        for metric in qmod.INDICATORS:
            if metric == "mobile":
                continue
            v = qmod.replicate_verdict(reps, metric)
            t = v["test"]
            out.append({"grandeur": ind_fr[metric],
                        "médiane de chaque répétition": " | ".join(
                            f"{k}: " + ", ".join(_f(x, "{:.3g}") for x in r) for k, r in v["replicates"].items()),
                        "test": t["test"] if t else "—", "p": _f(t["p"], "{:.2g}") if t else "—",
                        "taille d'effet": f"{t['effect_name']} = {t['effect']:.2f}" if t else "—",
                        "même sens pour chaque paire de répétitions": {True: "oui", False: "non", None: "—"}[v["consistent"]],
                        "CV entre répétitions": ", ".join(f"{k}: {c:.2f}" for k, c in v["rep_cv"].items()
                                                          if np.isfinite(c)) or "—",
                        "verdict": VERDICT_FR[v["verdict"]]})
        return pd.DataFrame(out)

    st.subheader("Calculs")
    st.markdown("Répétitions par groupe : " + ", ".join(f"**{k}** : {len(v)}" for k, v in reps.items()) + ".")
    vt = cached_fig("calc_tracers", tsig, verdicts)
    st.dataframe(vt, hide_index=True, width="stretch")
    for _, row in vt.iterrows():
        if row["verdict"] == VERDICT_FR["robust"]:
            eff = row["taille d'effet"]
            st.markdown(f"**{row['grandeur'].capitalize()}** : différence robuste et reproductible entre "
                        f"{', '.join(reps)} ({row['test']}, p = {row['p']}, {eff}).")
    st.caption("Test sur les valeurs par traceur, toutes répétitions réunies (Mann-Whitney pour 2 groupes, "
               "Kruskal-Wallis au-delà). Une différence est jugée **robuste et reproductible** si p < 0,05 ET si elle a "
               "le même sens pour chaque paire de répétitions (au moins 2 répétitions par groupe). Variabilité entre "
               "répétitions : CV des médianes des répétitions. Robustesse au seuil de mouvement : ⚙️ Réglages.")
    figs = tracers_figs(reps, styles, style.vel_unit(any_pa.length_unit, T), T)
    render_section({"ds": SimpleNamespace(name="comparaison"), "sig": tsig}, "Traceurs", figs,
                   {"analyses": table, "tests": vt})


# --------------------------------------------------------------------------- zip complet

def particle_report_ctx(ctx: dict, formats=("png", "svg"), dpi: int = 300) -> dict[str, bytes]:
    """Figures et calculs des sections Vitesse → Relation avec le pit (options actuelles de chaque
    section), tableau par traceur, qualité du suivi, MSD et résumé texte."""
    pa = ctx["pa"]
    out: dict[str, bytes] = {}
    pit_tables, pit_figs, _ = pit_section(ctx, pit_opts(ctx))
    sections = {"vitesse": speed_section(ctx, speed_opts()),
                "homogeneite": homogeneity_section(ctx, homogeneity_opts()),
                "deplacement": displacement_section(ctx),
                "debut_du_mouvement": onset_section(ctx, onset_opts()),
                "relation_avec_le_pit": (pit_tables, pit_figs)}
    for slug, (tables, figs) in sections.items():
        out.update(section_files(figs, tables, slug, formats, dpi))
    fig = pplots.fig_tracking_quality(ctx["tracks"], ctx["ds"].times, pa.time_unit)
    for fmt in formats:
        out[f"qualite_suivi.{fmt}"] = plots.fig_to_bytes(fig, fmt, dpi)
    plt.close(fig)
    out["traceurs.csv"] = pa.particles.to_csv(index=False).encode()
    out["population.csv"] = pa.population.to_csv(index=False).encode()
    for lab, (df, _) in ctx["msds"].items():
        if df is not None and not df.empty:
            out[f"msd_{lab.replace(' ', '_')}.csv"] = df.to_csv(index=False).encode()
    fits = {lab: f for lab, (_, f) in ctx["msds"].items()}
    exp = ctx["exp"]
    text = ["# Analyse des traceurs", "",
            f"Traceurs : {TRACER_TYPES.get(exp.get('tracers'), '?')} — substrat : {exp.get('substrate') or '?'} — "
            f"solution : {exp.get('electrolyte') or '?'} {exp.get('concentration') or ''} — potentiel : "
            f"{_f(exp.get('potential_V'), '{:g}', '?')} V", "",
            "Groupes : " + ", ".join(g.label for g in ctx["groups"]), ""]
    text += ["- " + line for line in pm.summary(pa, fits)]
    text += ["", "## Paramètres", "```json",
             json.dumps({"analyse": pa.params.to_dict(), "seuil": pa.threshold, "méthode du seuil": pa.threshold_method,
                         "expérience": exp}, indent=2, ensure_ascii=False, default=str), "```"]
    out["resume.md"] = "\n".join(text).encode("utf-8")
    return out


# =========================================================================== Export

def write_fig(fig, base: Path, formats: list[str], dpi: int):
    for fmt in formats:
        fig.savefig(base.with_suffix("." + fmt), dpi=dpi)
    plt.close(fig)


def page_export():
    st.header("Export des résultats")
    r = ss.get("result")
    ds = active_dataset()
    if not r or ds is None or r["ds_path"] != str(ds.folder):
        st.info("Calcule d'abord un champ de vitesse sur le dataset actif.")
        st.button("Aller au calcul", on_click=goto, args=(SECTIONS[1],), key="nav_export_velocity")
        return
    res: fieldmod.FieldResult = r["res"]
    calibration_box(ds)
    path, mtime = str(ds.folder), ds.mtime()
    res.pit_radii = figure_pit_radii(path, mtime, res.pits)
    panel = current_panel()
    nt = len(res.windows)
    vmax = ss.get("vmax_user") or plots.global_vmax(res)
    bg_mode = ss.get("bg_mode", "image")
    gain = ss.get("arrow_gain", 1.0)
    cs = current_contour(res) if bg_mode == "contour" else None

    fond = BG_LABELS.get(bg_mode, bg_mode) + (f" : {plots.SCALAR_LABELS[cs.quantity]}" if cs else "")
    st.caption(f"Réglages d'affichage repris de l'étape précédente : fond « {fond} », échelle 0 – {vmax:.3g} "
               f"{res.velocity_unit}, longueur des flèches ×{gain:g}.")
    scalebar_ui(ds)
    sb = current_scalebar(ds)
    a, b, c = st.columns(3)
    formats = a.multiselect("Formats des figures", ["png", "svg", "pdf"], **w("exp_formats", ["png", "svg"]))
    dpi = b.selectbox("Résolution PNG (dpi)", [150, 300, 600], **w("exp_dpi", 300))
    per_window = c.checkbox(f"Une figure par fenêtre ({nt})", **w("exp_per_window", True))

    st.markdown("##### 🎬 Vidéo du champ de vitesse")
    a, b, c = st.columns(3)
    make_field_video = a.checkbox("Générer", **w("exp_field_video", nt > 1))
    field_fmt = b.radio("Format", ["mp4", "gif", "les deux"], horizontal=True, **w("exp_field_fmt", "mp4"))
    fps = c.number_input("Images / s", 0.5, 30.0, step=0.5, **w("exp_fps", 4.0))

    is_tp = res.method == "trackpy"
    make_dens_video = make_particles = False
    st.markdown("##### 🔬 Analyse des traceurs")
    make_particles = st.checkbox("Inclure l'analyse des traceurs (sections Vitesse → Relation avec le pit : "
                                 "figures, calculs, résumé)", **w("exp_particles", True),
                                 help="Avec les réglages et options choisis dans 🔬 Analyse (dossier analyse/)."
                                      + ("" if is_tp else f" {RESULT_LABELS.get(res.method, res.method)} : traceurs virtuels advectés dans le "
                                         "champ mesuré, pas des particules détectées."))
    if is_tp:
        st.markdown("##### 🟣 Densité de particules")
        st.caption("Cartes (figures), données (density.npz, density_long.csv) et vidéo, avec les réglages "
                   "de densité choisis sur la page du champ de vitesse.")
        a, b, c = st.columns(3)
        make_dens_video = a.checkbox("Générer la vidéo", **w("exp_dens_video", nt > 1))
        dens_fmt = b.radio("Format  ", list(VIDEO_FORMATS), horizontal=True, **w("exp_dens_fmt", "mp4"))
        dens_fps = c.number_input("Images / s  ", 0.5, 30.0, step=0.5, **w("exp_dens_fps", 4.0))

    st.markdown("##### 🔥 Heatmap pleine résolution")
    st.caption("Vidéo + image de la frame affichée, avec les réglages de heatmap choisis sur la page du "
               "champ de vitesse. Une vidéo déjà générée là-bas avec les mêmes réglages est reprise telle quelle.")
    a, b, c = st.columns(3)
    make_hm_video = a.checkbox("Inclure la heatmap", **w("exp_hm_video", True))
    hm_fmt = b.radio("Format   ", list(VIDEO_FORMATS), horizontal=True, **w("exp_hm_fmt", "mp4"))
    hm_fps = c.number_input("Images / s   ", 1.0, 60.0, step=1.0, **w("exp_hm_fps", 10.0))

    st.markdown("##### 🎬 Vidéo de la séquence brute")
    a, b, c = st.columns(3)
    make_seq_video = a.checkbox("Générer", **w("exp_seq_video", False))
    seq_fmt = b.radio("Format ", ["mp4", "gif", "les deux"], horizontal=True, **w("exp_seq_fmt", "mp4"))
    seq_fps = c.number_input("Images / s ", 1.0, 60.0, step=1.0, **w("exp_seq_fps", 8.0))

    if st.button("Enregistrer sur le serveur et préparer le zip", type="primary", disabled=not formats,
                key="btn_export"):
        run_dir = ds.folder / "results" / f"{r['created']}_{res.method}"
        figdir = run_dir / "figures"
        (figdir / "fields").mkdir(parents=True, exist_ok=True)
        bar = st.progress(0.0, text="Données…")
        try:
            import skimage
            import trackpy as tp_
            dens = compute_density(r) if is_tp else None
            dens_vmax = density_vmax(dens) if is_tp else None
            dens_overlay = bool(ss.get("dens_overlay", True))
            params = {**res.params,
                      "display": {"background": bg_mode, "vmax": vmax, "arrow_gain": gain,
                                  "contour": None if cs is None else {
                                      "quantity": cs.quantity, "unit": res.scalar_unit(cs.quantity),
                                      "levels": [float(x) for x in cs.level_values(res)], "cmap": cs.cmap,
                                      "smooth_cells": cs.smooth, "interpolated_gaps_max_cells": cs.fill_gaps,
                                      "measured_cells_fraction": float(np.mean(np.isfinite(res.u))),
                                      "arrow_step": cs.arrow_step},
                                  "scalebar": None if sb is None else {
                                      "um_per_px": sb.um_per_px, "length_um": sb.length_um or "auto",
                                      "position": sb.position}},
                      "dataset_metadata": ds.meta,
                      "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                   "trackpy": tp_.__version__, "scikit-image": skimage.__version__},
                      "exported": datetime.now().isoformat(timespec="seconds")}
            if is_tp:
                params["density"] = {"bin_px": dens.bin_px, "sigma_cells": dens.sigma,
                                     "overlay": dens_overlay, "vmax": dens_vmax, "unit": dens.density_unit}
            hm = heatmap_setup(ds, path, mtime, r) if make_hm_video else None
            if hm:
                params["heatmap"] = {k_: hm[k_] for k_ in ("source", "polarity", "background", "threshold",
                                                           "trail", "sigma", "pct", "vmax", "cmap", "unit")}
            (run_dir / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
            np.savez_compressed(run_dir / "field.npz", **res.to_npz_dict())
            res.to_long_dataframe().to_csv(run_dir / "field_long.csv", index=False)
            res.series.to_csv(run_dir / "series.csv", index=False)
            if res.method == "trackpy":
                r["tracks"].to_csv(run_dir / "tracks.csv", index=False)
                r["samples"].to_csv(run_dir / "displacements.csv", index=False)
                np.savez_compressed(run_dir / "density.npz", **dens.to_npz_dict())
                dens.to_long_dataframe().to_csv(run_dir / "density_long.csv", index=False)
                (figdir / "density").mkdir(exist_ok=True)
            bar.progress(0.08, text="Figures de synthèse…")
            write_fig(plots.fig_series(res), figdir / "series", formats, dpi)
            if nt > 1:
                write_fig(plots.fig_radial_profile(res), figdir / "radial_profile", formats, dpi)
            if res.method == "trackpy":
                write_fig(plots.fig_tracks(r["tracks"], window_background(path, mtime, len(ds) - 1, len(ds) - 1),
                                           res.scale, res.length_unit, res.pits, scalebar=sb, times=ds.times,
                                           time_unit=res.time_unit, pit_radii=res.pit_radii, panel=panel),
                          figdir / "tracks", formats, dpi)
            bgs = [window_background(path, mtime, f0, f1) for f0, f1 in res.windows]
            ks = range(nt) if per_window else [int(ss.get("res_window", 0))]
            for j, k in enumerate(ks):
                write_fig(plots.fig_quiver(res, k, bgs[k], bg_mode, vmax, gain, scalebar=sb, contour=cs, panel=panel),
                          figdir / "fields" / f"field_w{k:03d}", formats, dpi)
                if is_tp:
                    write_fig(plots.fig_density(dens, k, bgs[k], dens_vmax, dens_overlay, scalebar=sb,
                                                pit_radii=res.pit_radii, panel=panel),
                              figdir / "density" / f"density_w{k:03d}", formats, dpi)
                bar.progress(0.08 + 0.32 * (j + 1) / len(ks), text=f"Champ {j + 1}/{len(ks)}")

            field_formats = {"mp4": ["mp4"], "gif": ["gif"], "les deux": ["mp4", "gif"]}[field_fmt]
            field_downloads = []
            if make_field_video and nt > 1:
                for j, fmt in enumerate(field_formats):
                    data = videomod.field_video_bytes(
                        res, bgs, bg_mode, vmax, gain, fps=float(fps), fmt=fmt, scalebar=sb, contour=cs,
                        progress=lambda f, t: bar.progress(
                            0.4 + 0.15 * (j + f) / len(field_formats), text=t))
                    out = run_dir / f"field_evolution.{fmt}"
                    out.write_bytes(data)
                    field_downloads.append((fmt, data, out.name))
            ss.field_video = field_downloads or None

            dens_downloads = []
            if make_dens_video and nt > 1:
                dens_formats = VIDEO_FORMATS[dens_fmt]
                for j, fmt in enumerate(dens_formats):
                    data = videomod.density_video_bytes(
                        dens, bgs if dens_overlay else None, dens_overlay, dens_vmax,
                        fps=float(dens_fps), fmt=fmt, scalebar=sb,
                        progress=lambda f, t: bar.progress(
                            0.55 + 0.1 * (j + f) / len(dens_formats), text=t))
                    out = run_dir / f"density_evolution.{fmt}"
                    out.write_bytes(data)
                    dens_downloads.append((fmt, data, out.name))
            ss.dens_video_export = dens_downloads or None

            hm_downloads = []
            if hm:
                fi = int(min(max(int(ss.get("hm_frame", hm["n"] - 1)), 0), hm["n"] - 1))
                Image.fromarray(heatmap_image(hm, fi)).save(run_dir / f"heatmap_frame{fi:03d}.png")
                # Vidéo déjà générée sur la page du champ de vitesse avec les mêmes réglages : reprise
                hv = ss.get("heatmap_video")
                ready = dict((f_, d_) for f_, d_, _ in hv["files"]) \
                    if hv and hv["sig"] == hm["sig"] + (float(hm_fps),) else {}
                hm_formats = VIDEO_FORMATS[hm_fmt]
                for j, fmt in enumerate(hm_formats):
                    data = ready.get(fmt) or heatmap_video(hm, ds, fmt, float(hm_fps), lambda f, t: bar.progress(
                        0.65 + 0.15 * (j + f) / len(hm_formats), text=t))
                    out = run_dir / f"heatmap.{fmt}"
                    out.write_bytes(data)
                    hm_downloads.append((fmt, data, out.name))
            ss.hm_video_export = hm_downloads or None

            seq_formats = {"mp4": ["mp4"], "gif": ["gif"], "les deux": ["mp4", "gif"]}[seq_fmt]
            seq_downloads = []
            if make_seq_video:
                stack = load_stack(path, mtime)
                for j, fmt in enumerate(seq_formats):
                    data = videomod.sequence_video_bytes(
                        stack, ds.times, ds.time_unit, fps=float(seq_fps), fmt=fmt, scalebar=sb,
                        rng=ds_range(path, mtime, display_mode()),
                        progress=lambda f, t: bar.progress(
                            0.8 + 0.15 * (j + f) / len(seq_formats), text=t))
                    out = run_dir / f"sequence.{fmt}"
                    out.write_bytes(data)
                    seq_downloads.append((fmt, data, out.name))
            ss.seq_video = seq_downloads or None

            if make_particles:
                bar.progress(0.95, text="Analyse des traceurs…")
                try:
                    actx = analysis_ctx()
                    for name, data in (particle_report_ctx(actx, tuple(formats), int(dpi)) if actx else {}).items():
                        (run_dir / "analyse" / name).parent.mkdir(parents=True, exist_ok=True)
                        (run_dir / "analyse" / name).write_bytes(data)
                except ValueError as e:
                    st.warning(f"Analyse des traceurs non exportée : {e}")

            bar.progress(0.97, text="Archive zip…")
            buf = pyio.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(run_dir.rglob("*")):
                    if p.is_file():
                        zf.write(p, Path(run_dir.name) / p.relative_to(run_dir))
            ss.export_zip = {"data": buf.getvalue(), "name": run_dir.name + ".zip", "dir": str(run_dir)}
            bar.empty()
        except Exception as e:
            bar.empty()
            st.error(f"Export impossible : {e}")

    ez = ss.get("export_zip")
    if ez:
        st.success(f"Résultats enregistrés dans `{ez['dir']}`.")
        st.download_button("Télécharger le zip", ez["data"], file_name=ez["name"], mime="application/zip",
                           type="primary", key="dl_zip")
        for fmt, data, name in (ss.get("field_video") or []):
            mime = "video/mp4" if fmt == "mp4" else "image/gif"
            st.download_button(f"Vidéo du champ ({fmt})", data, file_name=name, mime=mime,
                               key=f"dl_field_{fmt}")
        for fmt, data, name in (ss.get("dens_video_export") or []):
            st.download_button(f"Vidéo de la densité ({fmt})", data, file_name=name, mime=VIDEO_MIME[fmt],
                               key=f"dl_dens_exp_{fmt}")
        for fmt, data, name in (ss.get("hm_video_export") or []):
            st.download_button(f"Heatmap pleine résolution ({fmt})", data, file_name=name,
                               mime=VIDEO_MIME[fmt], key=f"dl_hm_exp_{fmt}")
        for fmt, data, name in (ss.get("seq_video") or []):
            mime = "video/mp4" if fmt == "mp4" else "image/gif"
            st.download_button(f"Vidéo de la séquence ({fmt})", data, file_name=name, mime=mime,
                               key=f"dl_seq_{fmt}")

    st.subheader("Téléchargements rapides")
    a, b, c = st.columns(3)
    a.download_button("Série temporelle (CSV)", res.series.to_csv(index=False).encode(),
                      file_name=f"{ds.name}_series.csv", mime="text/csv")
    npz = pyio.BytesIO()
    np.savez_compressed(npz, **res.to_npz_dict())
    b.download_button("Champ complet (NPZ)", npz.getvalue(), file_name=f"{ds.name}_field.npz",
                      mime="application/octet-stream")
    k = int(ss.get("res_window", 0))
    k = min(max(k, 0), nt - 1)
    f0, f1 = res.windows[k]
    fig = plots.fig_quiver(res, k, window_background(path, mtime, f0, f1), bg_mode, vmax, gain, scalebar=sb,
                           contour=cs, panel=panel)
    png = plots.fig_to_bytes(fig, "png", 300)
    plt.close(fig)
    c.download_button(f"Champ de la fenêtre {k} (PNG)", png, file_name=f"{ds.name}_field_w{k:03d}.png",
                      mime="image/png")
    with st.expander("Lire field.npz en Python"):
        st.code('d = np.load("field.npz")\n'
                'u, v = d["u"], d["v"]          # (fenêtre, y, x), en unité physique\n'
                'x = d["x_px"] * d["scale"]      # centres des cellules\n'
                't = d["t_center"]', language="python")


# =========================================================================== dispatch

PAGES = {"Import": page_import, "ROI & frames": page_roi, "Datasets": page_datasets,
         "⚙️ Réglages": page_q_settings, "Vitesse": page_q_speed, "Homogénéité": page_q_homogeneity,
         "Déplacement": page_q_displacement, "Début du mouvement": page_q_onset,
         "Relation avec le pit": page_q_pit, "Traceurs": page_q_tracers}


def main():
    inject_css()
    sidebar()
    st.markdown("<div class='corrovel-hero'><h1>🔬 Champ de vitesse — pitting corrosion</h1>"
               "<p>Import → ROI & frames → dataset dérivé → champ de vitesse → export &amp; vidéo</p>"
               "</div>", unsafe_allow_html=True)

    if "section" not in ss:
        ss.section = SECTIONS[0]
    st.radio("section_nav", SECTIONS, key="section", label_visibility="collapsed", horizontal=True)

    if ss.section in SUBVIEWS:
        subviews = SUBVIEWS[ss.section]
        if ss.get("subview") not in subviews:
            ss.subview = subviews[0]
        st.radio("subview_nav", subviews, key="subview", label_visibility="collapsed", horizontal=True)
        PAGES[ss.subview]()
    elif ss.section == SECTIONS[1]:
        page_velocity()
    else:
        page_export()


main()
