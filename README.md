# Champ de vitesse et analyse des traceurs – pitting corrosion

Interface web (Streamlit) pour analyser des vidéos de microscopie de piqûration (pitting corrosion)
avec des particules traceuses : import et recadrage de la séquence, **champ de vitesse** au cours
du temps (Trackpy, flux optique ou PIV), puis **analyse traceur par traceur organisée selon les
questions de l'étude** (vitesse, homogénéité, déplacement, début du mouvement, relation avec le
pit, comparaison des traceurs / expériences), avec calculs chiffrés, intervalles de confiance,
tests statistiques et figures prêtes pour un article.

L'interface est en français ; les figures exportées sont en anglais, au style d'article
(« Particle speed / (µm s⁻¹) »).

---

## Sommaire

- [Installation](#installation)
- [Lancement](#lancement)
- [Vue d'ensemble](#vue-densemble)
- [1. 📂 Données](#1--données)
- [2. 🌀 Champ de vitesse](#2--champ-de-vitesse)
- [3. 🔬 Analyse — une section par question de l'étude](#3--analyse--une-section-par-question-de-létude)
- [4. 📤 Export](#4--export)
- [Style des figures](#style-des-figures)
- [Organisation des fichiers](#organisation-des-fichiers)
- [Méthodes de calcul](#méthodes-de-calcul)
- [Limites et pièges](#limites-et-pièges)
- [Tests](#tests)

---

## Installation

Python ≥ 3.10.

```bash
cd corrosion_velocity_app
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Testé avec Python 3.10, streamlit 1.65, numpy 2.2, pandas 2.3, scipy 1.15, trackpy 0.7,
scikit-image 0.25, scikit-learn 1.7, matplotlib 3.10, plotly 7.1.

- **OpenCV** (`opencv-python-headless`, dans `requirements.txt`) est optionnel : il active
  l'algorithme de flux optique Farneback, 15 à 20 fois plus rapide que Lucas-Kanade.
- **scikit-learn** sert aux groupes de comportement (section Homogénéité).
- L'export vidéo MP4 utilise `imageio-ffmpeg`, qui embarque un binaire ffmpeg : pas besoin
  d'installer ffmpeg sur le système.

## Lancement

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. Les données vont dans `data/` (modifiable avec la
variable d'environnement `CORROVEL_DATA=/chemin/vers/stockage`).

Depuis un autre poste du labo : `streamlit run app.py --server.address 0.0.0.0`, puis
`http://<ip-de-la-machine>:8501`. Il n'y a pas d'authentification : à réserver au réseau local.

---

## Vue d'ensemble

L'application est un tableau de bord à 4 sections (onglets en haut de page) :

| Section | Contenu |
|---|---|
| **📂 Données** | Import, ROI & frames, Datasets (préparer une séquence recadrée et étalonnée) |
| **🌀 Champ de vitesse** | Calcul du champ (Trackpy / flux optique / PIV), aperçu animé, densité et heatmap |
| **🔬 Analyse** | ⚙️ Réglages, puis une sous-section par question : Vitesse, Homogénéité, Déplacement, Début du mouvement, Relation avec le pit, Traceurs |
| **📤 Export** | Figures, données, vidéos et analyse complète, écrites sur disque et proposées en zip |

---

## 1. 📂 Données

- **Import** : chemin d'un dossier sur le serveur (une image = une frame, tri naturel des noms :
  `img_2` avant `img_10`) ou d'un TIFF multi-pages ; ou envoi de fichiers / d'un `.zip`, copiés dans
  `data/raw/<nom>/`. Formats : tif, tiff, png, jpg, bmp. Le dtype (8/16 bits) est conservé.
- **ROI & frames** : tracer un rectangle sur l'image, saisir les coordonnées, ou « carré centré sur
  un point » (le pit). Choisir début, fin et un pas fixe ou un nombre total de frames. Renseigner
  le Δt entre frames d'origine et la taille de pixel (µm/px), puis générer le **dataset dérivé**.
- **Datasets** : visualiser un dataset, le choisir pour l'analyse, prévisualiser / exporter une
  vidéo rapide de la séquence brute pour vérifier les données avant tout calcul.

**Affichage des images** (barre latérale) : les images sont montrées telles qu'enregistrées.

| Mode | Effet |
|---|---|
| réelles (défaut) | valeurs brutes, 0 = noir, maximum du capteur = blanc (profondeur 8/10/12/14/16 bits détectée) |
| contraste fixe | un seul étirement pour toute la séquence : les variations de luminosité (photoblanchiment, pit qui s'assombrit) restent visibles |
| contraste auto | chaque image étirée séparément (pratique pour voir, mais la luminosité apparente change d'une image à l'autre) |

---

## 2. 🌀 Champ de vitesse

On choisit la grille (taille de cellule), la fenêtre temporelle (nombre d'intervalles moyennés) et
le pas entre fenêtres, puis le ou les **pits** (cliquer sur l'image ; plusieurs pits possibles).

### Les trois méthodes

| Méthode | Quand l'utiliser | Ce qu'elle donne |
|---|---|---|
| **Trackpy** | Particules discrètes visibles, bien séparées | Trajectoires individuelles + champ moyen par cellule |
| **Flux optique** (Lucas-Kanade, TV-L1, Farneback) | Texture qui bouge, particules mal séparées | Champ dense ; traceurs **virtuels** pour l'analyse |
| **PIV** (intercorrélation) | Traceurs clairsemés, grands déplacements, bruit | Champ par cellule ; traceurs **virtuels** pour l'analyse |

Aides au réglage de Trackpy : aperçu de la détection sur une frame choisie, histogramme des masses
avec **minmass suggéré** (seuil d'Otsu sur log(masse)), alerte si la polarité (particules claires ou
sombres) semble inversée.

### Affichage des résultats

- **Aperçu animé** du champ (lecture / pause / curseur), joué dans le navigateur sans recalcul.
- **Vue détaillée** fenêtre par fenêtre, avec plusieurs fonds : image moyenne, norme de la vitesse,
  ou **carte de couleurs (contours) + flèches** d'une grandeur : |v|, vitesse radiale v_r
  (> 0 : s'éloigne du pit), v_x, v_y, **vorticité** ω (> 0 : rotation anti-horaire) ou
  **divergence**. Les cellules comblées par interpolation sont marquées « × » (dans les images et
  dans la vidéo), pour ne jamais confondre mesure et interpolation.
- Échelle des couleurs et des flèches **fixe pour toutes les fenêtres** : les figures sont
  comparables entre elles.
- Série temporelle (|v| moyen / médian, v_r moyen) et profil radial v_r(distance, temps).
- **Densité de particules** (Trackpy) : nombre moyen de particules présentes par frame dans chaque
  cellule, divisé par l'aire de la cellule, sur les mêmes fenêtres que le champ.
- **Heatmap pleine résolution** (au pixel, avec traînée) : présence ou contraste des particules
  détectés sur l'image, ou positions Trackpy ; vidéo MP4/GIF.

---

## 3. 🔬 Analyse — une section par question de l'étude

### Principe commun

- **Traceurs** : avec Trackpy, ce sont les particules réellement suivies. Avec le flux optique ou la
  PIV, il n'y a pas de particules individuelles : l'application sème des **traceurs virtuels** sur
  la grille et les fait avancer dans le champ mesuré (lignes de courant). Les réponses décrivent
  alors le champ mesuré, pas des particules observées.
- **Groupes comparés**, superposés dans toutes les figures :
  - **plusieurs pits dans la même vidéo** : chaque traceur est rattaché au pit le plus proche de sa
    première détection → groupes « Pit A », « Pit B »… ;
  - **pits ou expériences filmés séparément** : analyses enregistrées (bouton « 💾 Enregistrer pour
    comparer ») puis choisies dans « Superposer des analyses enregistrées ».
- **Chaque section** contient :
  - **Calculs** : un tableau par groupe, chaque nombre avec son **intervalle de confiance à 95 %**
    (ex. « 3.75 [3.70 – 3.82] µm/s »), des phrases de réponse en français et un **test entre
    groupes** (p et taille d'effet) ;
  - **Figures** lettrées (a), (b), (c)…, en anglais, au style d'article ;
  - un **zip de la section** (figures PNG 300 dpi + SVG, tableaux CSV).

### ⚙️ Réglages

Réglages communs à toutes les sections :

- **🧪 Expérience** : type de traceurs (positifs, négatifs, neutres…), substrat, électrolyte /
  solution, potentiel appliqué (V), concentration, taille des traceurs, notes. Enregistré avec le
  dataset (`experiment.json`) et avec chaque analyse enregistrée ; sert à regrouper les analyses
  dans la section Traceurs.
- **Apparition du pit** : courbe d'intensité autour du pit (relative à l'image), instant
  d'apparition détecté automatiquement, saisi à la main, ou « pas de pit / inconnu ».
- **Réglages de l'analyse** : lissage des positions, fenêtre de vitesse, seuil de mouvement
  (0 = automatique), durée minimale au-dessus du seuil, points minimum par trajectoire, rayon
  exclu autour du pit (le bord sombre du pit est souvent pris pour des particules ; estimé
  automatiquement sur la dernière frame).
- **Nom de l'analyse** (légendes des figures), **enregistrement** pour comparaison, **choix des
  analyses à superposer**.
- **Résumé** chiffré, **test de robustesse au seuil** (analyse refaite avec seuil × 0.5, × 1, × 2 :
  une conclusion n'est solide que si elle tient sur les trois lignes), **qualité du suivi**
  (longueur des trajectoires, particules suivies par frame) et **tableau par traceur**.
- Zip complet de l'analyse (toutes les sections + résumé texte).

---

### Vitesse

**Question.** Quelles sont les vitesses des traceurs ? Quelle est leur distribution ? Comment
évoluent-elles dans le temps et dans l'espace ?

**Pourquoi ces calculs.** La vitesse moyenne sur toute une trajectoire mélange les périodes de repos
et de mouvement ; on retient donc la **vitesse en mouvement** de chaque traceur (médiane de sa
vitesse sur les instants où il bouge). Une valeur par traceur permet une distribution et un
intervalle de confiance honnêtes (les instants d'un même traceur ne sont pas indépendants).

**Calculs** (par groupe) :

| Grandeur | Méthode |
|---|---|
| Traceurs mobiles (%) | proportion, IC de Wilson |
| Vitesse en mouvement, médiane | une valeur par traceur, IC par bootstrap |
| 80 % des traceurs entre… | 10e – 90e percentiles des vitesses en mouvement |
| Vitesse max, médiane | maximum de chaque traceur, médiane, IC bootstrap |
| Vitesse ↔ distance au pit | corrélation de Spearman (ρ < 0 : plus lent loin du pit) |
| Différence entre groupes | Mann-Whitney (2 groupes) ou Kruskal-Wallis (plus) |

**Figures** :
- (a) **vitesse de chaque traceur depuis sa première détection** (traits fins) et **moyenne de chaque
  groupe** (trait épais, « Pit A mean », « Pit B mean ») ;
- (b) distribution des vitesses en mouvement ;
- (c) vitesse médiane au cours du temps (bande : 50 % central), option temps aligné sur le pit ;
- (d) vitesse selon la distance au pit (option r / R : distance rapportée au rayon du pit) ;
- (e) trajectoires sur l'image, colorées par la vitesse, avec cercle + flèche (position et direction
  à la frame choisie) et contour du pit.

---

### Homogénéité

**Question.** Tous les traceurs bougent-ils en même temps ? Ont-ils des vitesses comparables ?
Existe-t-il des groupes de particules aux comportements différents ?

**Pourquoi ces calculs.** L'homogénéité a trois aspects distincts : dans le temps (démarrent-ils
ensemble ?), en vitesse (même vitesse ?), dans l'espace (même vitesse dans toutes les directions
autour du pit ?). Pour les groupes de comportement, on cherche des populations distinctes sans les
imposer à l'avance, avec un critère qui ne crée pas de faux groupes sur une population homogène.

**Calculs** :

| Grandeur | Méthode |
|---|---|
| CV des vitesses | écart-type / moyenne des vitesses en mouvement, IC bootstrap |
| 80 % des démarrages entre… | 10e – 90e percentiles des instants de démarrage |
| 50 % en mouvement à t = | premier instant où la moitié des traceurs bouge |
| En mouvement au maximum (%) | part maximale de traceurs en mouvement simultanément |
| CV angulaire | variation de la vitesse médiane selon la direction autour du pit (secteurs de 30°) |
| Groupes de comportement | mélange de gaussiennes (voir [Méthodes](#statistiques)) |

**Figures** : part des traceurs en mouvement au cours du temps ; part cumulée des traceurs déjà mis
en mouvement ; distribution des vitesses ; vitesse selon la direction autour du pit (diagramme
polaire) ; si plusieurs groupes de comportement : nuage des caractéristiques et trajectoires
colorées par groupe.

---

### Déplacement

**Question.** Quelle distance chaque traceur parcourt-il ? Dans quelle direction ? Son mouvement
est-il rectiligne, radial, diffusif ou plus complexe ?

**Pourquoi ces calculs.** La distance se décrit par le **déplacement net** (où il arrive) et le
**chemin parcouru** (ce qu'il a fait), leur rapport donne la **rectitude**. La direction se décompose
par rapport au pit : composante **radiale** (s'éloigner / se rapprocher) et **tangentielle**
(tourner autour). La nature du mouvement se lit dans l'exposant α du déplacement quadratique moyen
(MSD ∝ τ^α) : α ≈ 1 diffusion (brownien), α ≈ 2 mouvement dirigé, α < 1 mouvement confiné.

**Calculs** :

| Grandeur | Méthode |
|---|---|
| Déplacement net, chemin parcouru | par traceur mobile, médiane, IC bootstrap |
| Rectitude | déplacement net / chemin (1 = ligne droite) |
| S'éloignent / se rapprochent / de côté (%) | cosinus entre déplacement net et direction « s'éloigner du pit » (> 0.5 / < −0.5 / entre les deux), IC de Wilson |
| Part tangentielle | \|v tangentielle\| / (\|v radiale\| + \|v tangentielle\|) : rotation autour du pit |
| Persistance | cosinus moyen de l'angle entre deux déplacements successifs (1 = tout droit, 0 = au hasard) |
| α par traceur et type de mouvement | MSD de chaque traceur (≥ 10 points) : confiné (α < 0.7), diffusif (0.7–1.3), dirigé (> 1.3) |
| MSD d'ensemble | avant / après l'apparition du pit |

**Figures** : déplacement net et chemin parcouru ; rose des directions par rapport au pit ;
composantes radiale / tangentielle ; histogramme des α par traceur ; MSD d'ensemble ; angles de
virage entre pas successifs ; trajectoires colorées par type de mouvement.

---

### Début du mouvement

**Question.** Quand les traceurs commencent-ils à bouger ? Peut-on identifier une mise en mouvement
collective ou progressive ?

**Pourquoi ces calculs.** Si le pit déclenche le mouvement, les traceurs devraient démarrer après son
apparition, et plus tard s'ils sont loin (un **front** qui se propage depuis le pit). Un démarrage
collectif se voit à une dispersion faible des instants de démarrage ; un démarrage progressif, à une
relation entre l'instant de démarrage et la distance au pit.

**Calculs** :

| Grandeur | Méthode |
|---|---|
| Démarrages observés | traceurs vus immobiles puis en mouvement (statut « démarre ») |
| Déjà en mouvement à la 1re détection (%) | IC de Wilson ; s'il est élevé, les démarrages observés surreprésentent les démarrages tardifs |
| Démarrage médian, et par rapport au pit | IC bootstrap |
| Dispersion | 10e – 90e percentiles des démarrages |
| Démarrage ↔ distance | Spearman ; **vitesse du front** = 1 / pente (régression robuste de Theil-Sen), avec IC |
| Verdict | **collective** si 80 % des démarrages tiennent dans 10 % de la durée ; **progressive** si le démarrage dépend significativement de la distance (p < 0.05, pente > 0) ; sinon **étalée** |

**Figures** : histogramme des démarrages (par rapport au pit) ; démarrage selon la distance au pit
avec ajustement ; part cumulée des traceurs démarrés ; **carte des démarrages** (position de départ
de chaque traceur colorée par son instant de démarrage).

---

### Relation avec le pit

**Question.** Que se passe-t-il avant, pendant et après l'apparition d'un pit ? Le mouvement
change-t-il en fonction de la distance au pit ?

**Pourquoi ces calculs.** Pour parler d'« avant / pendant / après », il faut des phases mesurées :
l'apparition vient du signal d'intensité, la croissance du **rayon du pit mesuré au cours du temps**.
La dépendance à la distance est décrite par une loi de puissance v ∝ r⁻ⁿ (n > 0 : plus lent loin du
pit), ajustée phase par phase pour voir si elle change.

**Calculs** :
- instant d'apparition, fin de croissance (rayon ≥ 90 % du rayon final, fraction réglable ou instant
  imposé), rayon final, vitesse moyenne de croissance ;
- par phase (**avant le pit / croissance / après la croissance**) : nombre de traceurs, part des pas
  en mouvement, vitesse en mouvement et vitesse radiale (IC 95 %) ;
- par phase : exposant **n** de v ∝ r⁻ⁿ avec IC 95 % et R², distance en r ou r / R ;
- corrélation (Spearman) entre la vitesse des traceurs et la **vitesse de croissance du pit**.

**Figures** : intensité autour du pit et rayon du pit au cours du temps (phases en fond) ; vitesse
et vitesse radiale au cours du temps ; vitesse selon la distance par phase (log-log, loi ajustée) ;
carte vitesse radiale × distance × temps ; distance au pit de chaque traceur au cours du temps ;
vitesse médiane alignée sur l'apparition du pit, pour chaque groupe.

Avec plusieurs pits, apparition, rayon et phases sont mesurés sur le premier pit (P1).

---

### Traceurs

**Question.** Les traceurs positifs et négatifs se comportent-ils différemment ? Les différences
observées sont-elles robustes et reproductibles ? (Plus généralement : selon le substrat, la
solution, le potentiel appliqué…)

**Pourquoi ces calculs.** Une différence significative sur une seule expérience peut venir de cette
expérience-là. Une différence n'est jugée **robuste et reproductible** que si elle est
statistiquement significative **et** va dans le même sens pour chaque paire de répétitions.

**Utilisation** : enregistrer chaque expérience (⚙️ Réglages → décrire l'expérience → « 💾 Enregistrer
pour comparer »), puis choisir le critère de regroupement : type de traceur (par défaut), substrat,
potentiel, électrolyte, concentration, taille des traceurs, nom ou dataset.

**Calculs** :
- tableau des analyses comparées : traceurs, vitesse en mouvement, part de mobiles, rectitude, part
  qui s'éloigne du pit, démarrage après le pit, persistance ;
- pour chaque grandeur : médiane de chaque répétition, test sur les valeurs par traceur
  (Mann-Whitney / Kruskal-Wallis) et taille d'effet, même sens pour chaque paire de répétitions,
  variabilité entre répétitions (CV), **verdict** :

| Verdict | Condition |
|---|---|
| ✅ différence robuste et reproductible | p < 0.05, même sens pour chaque paire de répétitions, ≥ 2 répétitions par groupe |
| ⚠️ significative, mais pas reproduite | p < 0.05 mais une seule répétition, ou sens contradictoires |
| pas de différence détectée | p ≥ 0.05 |

**Figures** : valeurs par traceur et médianes de chaque répétition, par grandeur ; vitesse alignée
sur l'apparition du pit (répétitions et moyenne) ; vitesse depuis la première détection par groupe ;
directions par rapport au pit par groupe.

---

## 4. 📤 Export

Choix des formats (png / svg / pdf) et de la résolution. Tout est écrit dans
`data/datasets/<nom>/results/<date>_<méthode>/` et proposé en zip :

- champ de vitesse (NPZ, CSV long), série temporelle, trajectoires et déplacements (Trackpy),
  densité ;
- figures : série, profil radial, trajectoires, champ de chaque fenêtre, densité ;
- vidéos MP4 / GIF : évolution du champ, de la densité, heatmap pleine résolution, séquence brute ;
- **analyse des traceurs** (dossier `analyse/`) : figures et calculs des sections Vitesse →
  Relation avec le pit, tableau par traceur, population, MSD, résumé texte.

`params.json` garde tous les paramètres et les versions des librairies.

---

## Style des figures

Toutes les figures et vidéos suivent le même style d'article :
- textes en **anglais**, étiquettes en gras « Grandeur / (unité) » avec exposants
  (« Particle speed / (µm s⁻¹) ») ;
- lettres de panneau (a), (b)… ; étiquette d'image en haut à gauche (« Pit A », réglable) ;
- images sans graduations, avec **barre d'échelle** (noire sur fond clair, blanche sur fond sombre) ;
- **contour du pit** en cercle rouge pointillé (rayon estimé) ;
- groupes : orange plein, vert tireté, puis bleu, violet, rouge… (comme « Pit A mean / Pit B mean »).

Barre d'échelle, étiquette de panneau et contour du pit se règlent dans l'encadré « 📏 Barre
d'échelle et style des figures ».

---

## Organisation des fichiers

```
data/
├── raw/<nom>/                      fichiers envoyés par le navigateur
└── datasets/<nom>/
    ├── metadata.json               ROI, indices d'origine, temps, Δt, µm/px, source
    ├── experiment.json             description de l'expérience (traceurs, substrat, potentiel…)
    ├── frames/frame_00000.tif      frames recadrées, dtype d'origine
    ├── analyses/<nom>_<id>/        analyses enregistrées pour comparaison
    │   ├── steps.parquet           une ligne par détection (positions, vitesses, état)
    │   ├── particles.parquet       une ligne par traceur
    │   ├── population.parquet      comportement collectif frame par frame
    │   └── meta.json               réglages, unités, pits, description de l'expérience
    └── results/<date>_<méthode>/   export
        ├── params.json, field.npz, field_long.csv, series.csv
        ├── tracks.csv, displacements.csv, density.npz      (Trackpy)
        ├── *.mp4 / *.gif                                     vidéos
        ├── figures/                                          figures du champ et de la densité
        └── analyse/<section>/figures, *.csv, resume.md      analyse des traceurs
```

Code :

```
app.py                  interface Streamlit
corrovel/
├── io.py, dataset.py   lecture des images, ROI, datasets dérivés
├── tracking.py         Trackpy (détection, liens, déplacements)
├── flow.py             flux optique
├── piv.py              PIV par intercorrélation
├── field.py            champ sur grille, vitesse radiale, traceurs virtuels
├── density.py          densité et heatmap
├── particles.py        analyse traceur par traceur (vitesses, démarrage, MSD, pit)
├── questions.py        calculs des sections de l'Analyse
├── compare.py          groupes, analyses enregistrées, statistiques
├── plots.py, pplots.py, qplots.py, interactive.py   figures (matplotlib, Plotly)
├── style.py, scalebar.py, colormaps.py              style commun des figures
└── video.py            vidéos MP4 / GIF
```

---

## Méthodes de calcul

### Conventions

x vers la droite, y vers le bas (coordonnées image), centre du pixel (i, j) en (x = j, y = i).
Vitesse radiale v_r > 0 : la matière s'éloigne du pit. Avec plusieurs pits, chaque point est
rattaché au pit le plus proche (découpage de Voronoï). Vitesse tangentielle > 0 : rotation
anti-horaire à l'écran. Rien n'est extrapolé : une cellule sans mesure reste « non mesurée ».

### Trackpy

Image de détection = frame − fond (médiane temporelle par défaut), normalisée sur tout le stack :
`minmass` a le même sens pour toutes les frames. `tp.locate`, `tp.link`, `tp.filter_stubs`, option
« soustraire la dérive globale ». Vitesse d'un pas = déplacement / différence des temps réels. Le
champ d'une fenêtre = moyenne des vitesses des déplacements tombant dans chaque cellule (cellules
sous « déplacements min » laissées vides).

### Flux optique

`skimage.registration` (Lucas-Kanade itératif `ilk`, `tvl1`) ou Farneback (OpenCV) entre frames
consécutives, moyenne par cellule puis par fenêtre.
- **Soustraire le fond fixe** (recommandé pour des traceurs) : le flux est calculé sur ce qui change
  dans le temps ; sans cela, la texture fixe (rayures, bord du pit) tire le flux vers zéro.
- **Fiabilité** : la texture de chaque pixel est comparée au bruit de l'image (critère de
  Shi-Tomasi rapporté au bruit). La moyenne par cellule est pondérée par cette texture, et une
  cellule dont la texture ne dépasse pas le seuil du bruit pur (simulé) est « non mesurée ».
- Le flux de chaque paire est gardé en cache : changer la cellule, la fenêtre ou le pit est immédiat.

### PIV

Chaque cellule est une fenêtre d'interrogation ; le déplacement est le pic de l'intercorrélation
(FFT) entre deux frames.
- **corrélation d'ensemble** : les plans de corrélation de toutes les paires d'une fenêtre de temps
  sont additionnés avant de chercher le pic (robuste avec peu de traceurs) ;
- **2e passe à fenêtre décalée** : fiable jusqu'à ~½ cellule par intervalle ;
- position sous-pixel par ajustement gaussien (parabolique en secours) ;
- validation : rapport pic principal / 2e pic (réglable, 1.3 par défaut), puis test de médiane
  normalisée contre les vecteurs aberrants ; cellules rejetées laissées non mesurées ;
- option « soustraire le fond fixe », comme pour le flux optique.

### Traceurs virtuels (flux optique, PIV)

Un traceur tous les N cellules, avancé fenêtre par fenêtre en intégrant le champ (u, v) mesuré
(4 sous-pas d'Euler par fenêtre ; trous d'au plus 2 cellules comblés). Un traceur qui sort de la
zone mesurée s'arrête. Format identique aux trajectoires Trackpy : toute l'analyse s'applique.

### Analyse traceur par traceur

- **Positions lissées** (moyenne glissante sur `lissage` détections) pour les vitesses et distances :
  le bruit de localisation n'est pas pris pour du mouvement. Le MSD, la persistance et les angles de
  virage utilisent les positions **brutes** (le lissage les biaiserait).
- **Vitesse glissante** : déplacement sur `fenêtre` frames / temps écoulé.
- **Seuil de mouvement** automatique, dans l'ordre de préférence :
  1. 95e percentile des vitesses pendant la période calme du début (premiers 15 % du temps avant
     le pit) ;
  2. seuil d'Otsu sur log(vitesse) (sépare lents et rapides) ;
  3. 3 écarts-types du bruit de localisation Trackpy.
- **En mouvement** : seuil dépassé sur au moins `durée min` points consécutifs. **Statut** d'un
  traceur : immobile, démarre (vu immobile puis en mouvement), déjà en mouvement (dès sa première
  détection).
- **Instant de démarrage** affiné par un ajustement « repos puis vitesse constante » (le lissage le
  ferait sinon apparaître 1 à 2 frames trop tôt).
- **Persistance** : cosinus moyen entre deux déplacements successifs de `fenêtre` frames, sur des
  intervalles disjoints (une marche au hasard donne bien 0).
- **α par traceur** : ajustement log-log du MSD sur les retards 1 … max(4, n/4) frames.

### Pit

- **Signal d'apparition** : intensité moyenne dans un disque autour du pit, divisée par l'intensité
  moyenne de l'image (compense le photoblanchiment). Détection : variation d'au moins 10 %, puis
  ajustement « constant puis linéaire » jusqu'à la mi-variation ; la cassure donne l'instant
  d'apparition.
- **Rayon du pit** : profil d'intensité par anneaux autour du pit ; le bord est la remontée la plus
  raide après le cœur sombre. Mesuré sur ~60 frames, lissé par médiane glissante.
- **Loi de puissance v ∝ r⁻ⁿ** : classes de distance logarithmiques (≥ 10 pas par classe, ≥ 4
  classes), vitesse médiane contre distance médiane de chaque classe, régression log-log ; IC de n
  par l'erreur standard de la pente.

### Statistiques

| Outil | Usage |
|---|---|
| Bootstrap (2000 rééchantillonnages des traceurs) | IC 95 % des médianes, du CV… |
| Intervalle de Wilson | IC 95 % des pourcentages (fiable près de 0 % ou 100 %) |
| Mann-Whitney (taille d'effet : corrélation rang-bisériale r, > 0 si le 1er groupe a les plus grandes valeurs) | 2 groupes |
| Kruskal-Wallis (taille d'effet : η²_H) | plus de 2 groupes |
| Spearman | liens monotones (vitesse ↔ distance, démarrage ↔ distance, vitesse ↔ croissance du pit) |
| Theil-Sen | pente robuste du démarrage selon la distance (vitesse du front) |
| Mélange de gaussiennes + BIC | groupes de comportement |

**Groupes de comportement** : mélange de gaussiennes sur les caractéristiques standardisées des
traceurs mobiles (log de la vitesse en mouvement, rectitude, direction par rapport au pit,
persistance). Une caractéristique qui varie trop peu d'un traceur à l'autre est écartée (elle ne
contiendrait que du bruit) ; la page indique celles qui ont été utilisées. Un groupe
supplémentaire n'est retenu que s'il améliore le BIC de plus de 10 (« preuve très forte ») : une
population homogène donne un seul groupe. Il faut au moins 30 traceurs mobiles.

---

## Limites et pièges

- **Dossier d'import** : toutes les images du dossier sont prises comme frames et doivent avoir la
  même taille. Une capture d'écran ou une image annotée laissée dans le dossier provoque l'erreur
  « taille différente de la frame 0 » : la déplacer ailleurs avant l'import.
- **Traceurs virtuels** (flux optique, PIV) : ils décrivent le champ mesuré, pas des particules
  observées ; leurs « démarrages » et « types de mouvement » sont à interpréter avec prudence.
- **Flux optique** : même avec le fond soustrait, les vitesses sont environ 15 % plus faibles que
  celles de Trackpy sur les traceurs NiCr(+) (corrélation 0.9). Pour des particules bien séparées,
  préférer Trackpy.
- **Trackpy** : la qualité dépend surtout de `minmass`, du diamètre et du déplacement max ; deux
  particules plus proches que la « séparation » fusionnent.
- **α par traceur** : le bruit de localisation abaisse α des traceurs presque immobiles (ils peuvent
  apparaître « confinés »).
- **Démarrages** : si beaucoup de traceurs bougent déjà à leur première détection, les démarrages
  observés surreprésentent les démarrages tardifs ; la part cumulée est plus fiable.
- **Reproductibilité** : un verdict « robuste » demande au moins 2 répétitions par groupe ; les
  groupes comparés doivent avoir les mêmes unités (µm, s).
- **Plusieurs pits** : la cinétique du pit (apparition, rayon, phases) est mesurée sur P1.
- Le fond médian suppose un fond à peu près statique ; si le pit grandit beaucoup, sa bordure peut
  ressortir comme des « particules » (utiliser « Exclure autour du pit »).
- « Soustraire la dérive globale » retire aussi un écoulement collectif réel.
- Pas de recalage des frames : une dérive de la platine apparaît comme une vitesse.
- Le dataset actif est chargé en mémoire (300 frames de 512 × 512 ≈ 300 Mo).

---

## Tests

```bash
python -m pytest -q
```

- `tests/test_validation.py` — **validation numérique** : séquences synthétiques à mouvement connu
  (translation, éjection radiale, zones sans texture, grands déplacements, traceurs clairsemés) ;
  Trackpy, flux optique, PIV et traceurs virtuels doivent retrouver la vitesse et la direction.
- `tests/test_pipeline.py` — chaîne logicielle : import, datasets, champ, figures, vidéos, analyse
  traceur par traceur (vitesse, démarrage, front, MSD), détection du pit.
- `tests/test_questions.py` — calculs des sections de l'Analyse contre une vérité connue : vitesse
  tangentielle sur une rotation, α brownien / dirigé, persistance nulle pour une diffusion, groupes
  de comportement (1 groupe si homogène, 2 si bimodal), figure « vitesse depuis la première
  détection », pits A / B, enregistrement / rechargement d'une analyse, tests statistiques, phases
  du pit, exposant de la loi de puissance, verdict de reproductibilité, rendu de toutes les figures.
