# Champ de vitesse – pitting corrosion

Interface web (Streamlit) pour analyser une séquence d'images de pitting corrosion :
import, ROI autour du pit, sélection des frames, dataset dérivé enregistré sur disque,
champ de vitesse en fonction du temps (Trackpy ou flux optique) avec aperçu animé dans le
navigateur, export des figures, des données et de vidéos (MP4/GIF) du champ ou de la séquence brute.

## Installation

Python ≥ 3.10.

```bash
cd corrosion_velocity_app
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Testé avec streamlit 1.50, plotly 7.1, trackpy 0.7, scikit-image 0.24, numpy 2.0, pandas 2.3.
L'export vidéo MP4 utilise `imageio-ffmpeg`, qui embarque un binaire ffmpeg statique : pas
besoin d'installer ffmpeg sur le système.

## Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. Les données vont dans `data/`
(modifiable avec la variable d'environnement `CORROVEL_DATA=/chemin/vers/stockage`).

Pour l'utiliser depuis un autre poste du labo : `streamlit run app.py --server.address 0.0.0.0`,
puis `http://<ip-de-la-machine>:8501`. Il n'y a pas d'authentification : à réserver au réseau local.

## Déroulé

L'appli est un tableau de bord à 3 sections (onglets en haut de page) :

1. **📂 Données**, en 3 sous-vues :
   - *Import* : chemin d'un dossier sur le serveur (une image = une frame, tri naturel des noms :
     `img_2` avant `img_10`) ou d'un TIFF multi-pages ; ou envoi de fichiers / d'un .zip, copiés
     dans `data/raw/<nom>/`. Formats : tif, tiff, png, jpg, bmp. Le dtype (8/16 bits) est conservé.
   - *ROI & frames* : tracer un rectangle sur l'image (outil de sélection actif par défaut), ou
     saisir les coordonnées, ou « carré centré sur un point ». Choisir début, fin, et soit un pas
     fixe, soit un nombre total de frames. Renseigner Δt entre frames d'origine et la taille de
     pixel (µm/px ; 0 = px), puis générer le dataset dérivé.
   - *Datasets* : visualiser un dataset, le choisir pour l'analyse, et prévisualiser/exporter une
     vidéo rapide (MP4/GIF) de la séquence brute pour vérifier les données avant de calculer quoi
     que ce soit.
2. **🌀 Champ de vitesse** : choisir la méthode (Trackpy ou flux optique), la grille, la fenêtre
   temporelle et le centre du pit, calculer. Les résultats s'ouvrent sur un **aperçu animé** du
   champ (lecture/pause/curseur, joue dans le navigateur sans recalcul), puis la vue détaillée
   fenêtre par fenêtre (figures exportables).
3. **📤 Export** : figures (png/svg/pdf), données (CSV/NPZ), et vidéos MP4/GIF — du champ de
   vitesse et/ou de la séquence brute. Tout est écrit dans
   `data/datasets/<nom>/results/<date>_<méthode>/` et proposé en zip.

## Structure d'un dataset dérivé

```
data/datasets/<nom>/
├── metadata.json            ROI, indices d'origine, temps, Δt, µm/px, source
├── frames/frame_00000.tif   frames recadrées, dtype d'origine
└── results/<run>/
    ├── params.json          tous les paramètres + versions des librairies
    ├── field.npz            u, v, count (fenêtre, y, x), centres des cellules, temps
    ├── field_long.csv       une ligne par (fenêtre, cellule)
    ├── series.csv           |v| moyen/médian et v_r moyen par fenêtre
    ├── tracks.csv, displacements.csv   (Trackpy)
    ├── field_evolution.mp4 / .gif      évolution du champ de vitesse
    ├── sequence.mp4 / .gif             lecture de la séquence brute (dataset dérivé)
    └── figures/             series, radial_profile, tracks, fields/field_wXXX (png/svg/pdf)
```

## Méthodes

**Trackpy** (particules discrètes). Image de détection = frame − fond (médiane temporelle par
défaut), normalisée sur tout le stack en 0–255 : `minmass` a donc le même sens pour toutes les frames.
`tp.locate` par frame, `tp.link`, `tp.filter_stubs`. Chaque pas d'une trajectoire donne un
déplacement ; vitesse = déplacement / différence des temps réels (un pas irrégulier ou `memory > 0`
sont gérés). Le champ d'une fenêtre = moyenne des vitesses des déplacements tombant dans chaque cellule
(cellules sous le seuil « déplacements min » laissées vides). La série temporelle est calculée sur les
déplacements individuels, pas sur la grille.

**Flux optique** (texture qui bouge, pas de particules séparées). `skimage.registration`
(`optical_flow_ilk` ou `optical_flow_tvl1`) entre frames consécutives, moyenne par cellule puis par
fenêtre.

**Vitesse radiale** : v_r = v · (r − r_pit) / |r − r_pit|, positive quand la matière s'éloigne du pit.
Le profil v_r(distance, temps) est une moyenne par anneau pondérée par le nombre de déplacements.

Conventions : x vers la droite, y vers le bas (coordonnées image), centre du pixel (i, j) en (x=j, y=i).
L'échelle des flèches et des couleurs est fixe pour toutes les fenêtres (95e percentile de |v| par
défaut) pour que les figures soient comparables entre elles.

## Limites

- Flux optique : la série temporelle moyenne tous les pixels de la ROI, zones immobiles comprises ;
  elle sous-estime fortement la vitesse de particules isolées. Sur la séquence synthétique de test
  (particules à 3,2 µm/s), il donne 0,35–0,58 µm/s. Pour des particules éjectées, utiliser Trackpy.
- Trackpy : la qualité dépend surtout de `minmass`, du diamètre et du déplacement max. Utiliser
  l'aperçu de détection et l'histogramme des masses. Des particules plus proches que `séparation`
  (diamètre + 1 par défaut) fusionnent en une seule détection.
- Le fond médian suppose un fond à peu près statique. Si le pit grandit beaucoup pendant la
  séquence, sa bordure peut ressortir comme des « particules » : restreindre la plage de frames ou
  passer à `mean`/`none`.
- « Soustraire la dérive globale » retire aussi un écoulement collectif réel.
- Le dataset actif est chargé en mémoire en float32 (300 frames de 512×512 ≈ 300 Mo).
- Pas de vérification que les frames sont recalées entre elles : une dérive de platine apparaît
  comme une vitesse.

## Tests

```bash
python -m pytest -q
```

Séquences synthétiques à vitesse connue : éjection radiale (Trackpy doit retrouver 3,2 µm/s à 5 %
près et v_r ≈ |v|), translation (le flux optique doit retrouver le déplacement à 5 % près),
tri naturel, stack TIFF, sélection des frames, création du dataset, grille, figures et GIF.
