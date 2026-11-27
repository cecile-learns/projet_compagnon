# Projet Compagnon Immobilier

## Description

Le projet aide un particulier à estimer le prix d'un logement ancien et à le comparer à des ventes réelles, en France métropolitaine hors Alsace-Moselle.

- **Données** : ventes DVF+ 2014-2025 (Cerema) et sources socio-économiques et territoriales.
- **Modélisation** : prévision du prix médian au m² d'un territoire, puis du prix de chaque vente. Le modèle retenu, H7v4, estime un bien ; il ne prévoit pas le marché.
- **Outil** : estimation et ventes comparables, présentées dans une application Streamlit.

**Axe annonces, abandonné.** Le sujet prévoyait aussi l'exploitation d'annonces (texte, images, chatbot). Le jeu fourni porte sur le Haut-Rhin, que DVF+ ne couvre pas, sur une courte période, avec des prix demandés et non payés, et sans métadonnées. Les ventes comparables DVF+ en tiennent lieu.

## Organisation du dépôt

```
projet_compagnon/
├── README.md
├── .gitignore
├── requirements.txt                ← dépendances (pip install -r requirements.txt)
├── pyproject.toml                  ← installation du socle (pip install -e .) et réglages de pytest
├── config.py                       ← chemins communs à tous les codes
├── data/                           ← local, jamais sur GitHub
│   ├── raw/                        ← sources brutes (00_telechargement)
│   └── processed/                  ← fichiers créés par nos codes
├── 00_telechargement/
│   └── telechargement.py           ← téléchargement des sources depuis le Google Drive
├── 01_preparation/
│   ├── preparation.py              ← préparation des sources, 12 blocs lançables un par un
│   └── feature_engineering.py      ← table de modélisation, 11 sections lançables une par une
├── 02_audit_eda/
│   ├── 1_audits_sources.ipynb               ← audit de chaque source brute
│   ├── 1_eda_sources.ipynb                  ← analyse exploratoire des sources : cible, relations, synthèse
│   ├── 2_audit_eda_table_modelisation.ipynb ← audit et EDA de la table de modélisation, historique des passages
│   ├── 3_temporalite_prix_national.ipynb    ← SARIMA sur le prix national : enseignements pour formuler les hypothèses
│   └── 04_baseline.ipynb                    ← baseline de la cible (prix N-1), lit la table sans l'écrire
├── socle/                          ← code partagé, importé par les notebooks et les scripts
│   ├── commun.py                   ← table, découpage, prévision naïve, évaluation et tests contre la naïve
│   ├── gabarit_boosting.py         ← gradient boosting (même grille pour toutes les hypothèses)
│   ├── gabarit_hierarchique.py     ← modèle hiérarchique (effet aléatoire par sous-bassin)
│   ├── gabarit_pont.py             ← modèle pont, base du tour 2
│   └── gabarit_keras.py            ← réseau de neurones du modèle par vente
├── tests/                          ← un fichier de contrôles par module du socle (pytest)
├── 03_modelisation/
│   ├── modelisation_template.ipynb ← modèle de notebook à copier pour une hypothèse
│   ├── modelisation_parcours_court.py ← parcours court : télécharge le kit, lance les modèles et le test 2025, tout tracé dans MLflow
│   ├── 1_tour1/                    ← H3 : attractivité, logement et ménages, emploi, puis H3 globale
│   ├── 2_prepa_tour2/              ← scripts P6, P7, P9 : préparations propres au tour 2
│   ├── 3_tour2/                    ← H4 (pont), H5 (volume et élan), H6 (capacité d'achat), H5-H6 (comparaison), H7 (modèle par vente)
│   └── 4_evaluation_2025/          ← test final sur 2025, ouvert une seule fois : deux scripts, puis un notebook qui compare les résultats
├── 04_outils/                      ← modèle final et sa fiche, lignes d'entrée, ventes comparables, mise à jour de DVF+
├── 05_application/                 ← application Streamlit
└── models/
    └── h7v4/                       ← le modèle livré : réseau, paramètres, fiche et empreintes (versionné)
```

**Lecture des noms de dossiers.** Un dossier numéroté est une étape du pipeline, à lancer dans l'ordre des numéros (y compris les sous-dossiers de `03_modelisation`). Un dossier sans numéro (`socle`, `tests`, `data`, `models`) sert de support aux étapes, sans ordre d'exécution. Un dossier qui commence par un point (`.git`, `.streamlit`, environnement virtuel) appartient à l'outillage.

**Lecture des formats.** Un seul format par type de fichier : les préparations, les gabarits et les scripts sont des `.py` ; les analyses exploratoires et les modélisations sont des notebooks, car elles produisent des résultats à commenter et des synthèses.

GitHub contient le code et, en plus, ce dont l'outil a besoin pour fonctionner sans le dossier `data/` : les fichiers de données de `05_application/donnees/` (tables des communes, ventes comparables, lignes d'entrée 2026, fichiers du modèle, tables de la page d'exploration) et le modèle livré de `models/h7v4/`. Les données de travail (sources brutes, fichiers préparés, table de modélisation) ne sont jamais dans GitHub : elles sont sur un Google Drive partagé, lisible par toute personne ayant le lien : https://drive.google.com/drive/folders/1bVlZgMTxjtoWMveHXJEsxY6LPRo6U6nZ

Les rapports (rédigés en Word, remis en PDF) sont tenus à part, sur le Google Drive de l'équipe.

## Installation

Depuis la racine du projet, dans un environnement virtuel (Python 3.12) :

```bash
python3 -m venv .venv && source .venv/bin/activate   # une seule fois, puis « source .venv/bin/activate » à chaque session
pip install -r requirements.txt   # dépendances
pip install -e .                  # socle : rend « from commun import ... » disponible partout
```

`pip install -e .` ne se fait qu'une fois. Le mode `-e` (éditable) est nécessaire : le socle lit `config.py` et `data/` à la racine du projet, et toute modification du socle est prise en compte sans réinstaller.

Toutes les commandes se lancent depuis la racine du projet.

`py7zr` (lecture des archives 7z de la livraison du Cerema) n'est nécessaire que pour l'étape `extraire` de la mise à jour de DVF+ (voir « 6. Outils ») : `pip install py7zr`. Il n'est pas dans `requirements.txt`, le reste du projet n'en a pas besoin.

## Parcours court

Un seul fichier lance l'ensemble, depuis la racine du projet, environnement virtuel activé :

```bash
pip install -r requirements.txt
pip install -e .
python 03_modelisation/modelisation_parcours_court.py
```

Il enchaîne trois étapes, toutes tracées dans MLflow :

1. **Téléchargement du kit** (9 fichiers déjà préparés, `data/processed/`) : aucune préparation ni feature engineering à lancer.
2. **Modèles des hypothèses**, jugés sur 2023 et 2024 : H3 (modèle complet), H4 (pont), H5-H6 (pont, taux d'effort, volume et élan) et H7 (modèle par vente : Elastic Net et réseaux Keras, évaluations B et C).
3. **Test final 2025** : les deux scripts de `4_evaluation_2025`, dont les résultats sont aussi tracés.

**Durée : environ 1 h 30 à 2 h**, selon les ressources de la machine (cœurs et mémoire). H7 en est la plus grande part. Le script affiche la durée de chaque étape.

```bash
python 03_modelisation/modelisation_parcours_court.py --etapes 1 2            # sans le test 2025
python 03_modelisation/modelisation_parcours_court.py --etapes 3              # test 2025 seul
python 03_modelisation/modelisation_parcours_court.py --etapes 2 --hypotheses h7 h3   # certaines hypothèses, dans cet ordre
mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db            # lire les résultats
```

Dans MLflow, deux expériences : **Compagnon immo - hypothèses** (étape 2) et **Compagnon immo - évaluation 2025** (étape 3). Chaque run est un modèle jugé contre la naïve, pour un type de bien et une année ; les comparaisons entre modèles restent dans les notebooks.

Le parcours court n'optimise pas le réseau (il utilise `meilleurs_reglages_keras.json` du kit) et ne rejoue pas les audits ni les notebooks. Le **parcours long** reconstruit tout depuis les sources et passe par les notebooks (étapes ci-dessous).

## Ordre d'exécution

### 1. Télécharger les données

```bash
python 00_telechargement/telechargement.py              # liste les sources disponibles
python 00_telechargement/telechargement.py taux lovac   # télécharge seulement ces sources
python 00_telechargement/telechargement.py tout         # télécharge toutes les sources, y compris les fichiers déjà préparés du kit
python 00_telechargement/telechargement.py parcours_court   # le kit du parcours court (9 fichiers)
```

Un fichier déjà téléchargé n'est pas retéléchargé. Si un téléchargement échoue (Google Drive limite parfois les gros téléchargements), il suffit de relancer la même commande plus tard.

**DVF+ : deux parcours au choix**

- **Rapide** : `python 00_telechargement/telechargement.py dvf_residentiel` télécharge DVF+ déjà filtré (un seul fichier, 1,4 Go). Le filtrage du bloc 2 est alors sauté.
- **Complet** : `python 00_telechargement/telechargement.py dvf` télécharge DVF+ brut (97 fichiers départementaux, plusieurs Go), filtré ensuite par le bloc 2.

**Kit du parcours court** : `parcours_court` télécharge seulement les fichiers déjà préparés dont la modélisation a besoin (table de modélisation, Filosofi par commune, table de liaison, taux des crédits par mois, taux d'effort et volume et élan par sous-bassin, ventes d'apprentissage et de test 2025, réglages optimisés du réseau Keras), dans `data/processed/`. Il évite la préparation et le feature engineering.

**Parcours long (reconstruction depuis les sources)** : ne pas utiliser `tout`, qui télécharge aussi les fichiers déjà préparés du kit. Télécharger seulement les sources brutes, plus DVF+ (voir ci-dessus) :

```bash
python 00_telechargement/telechargement.py passage_geo bassins_vie communes superficie_arrondissements filosofi menages logement delinquance lovac equipements taux bpe bpe_evolution emploi
```

Le même principe vaut pour l'échantillon d'audit de DVF+ (`02_audit_eda/1_audits_sources.ipynb`) : `python 00_telechargement/telechargement.py dvf_audit` le télécharge directement, sinon il est tiré à partir de DVF+ brut.

### 2. Préparer les données

```bash
python 01_preparation/preparation.py            # les 12 blocs, dans l'ordre
python 01_preparation/preparation.py 5 6 9      # seulement ces blocs
python 01_preparation/preparation.py liste      # liste des blocs
```

Chaque bloc lit ses entrées dans `data/raw/` (ou les fichiers d'un bloc précédent dans `data/processed/`) et écrit les siens. Un bloc qui a besoin du résultat d'un autre le dit et s'arrête, avec la commande à lancer d'abord. Les explications sont en commentaire dans `preparation.py`, en tête de chaque bloc.

### 3. Audits et analyse exploratoire

Les audits et l'analyse exploratoire sont des notebooks, sans version script : ils contiennent le code, les résultats et leurs interprétations.

- `02_audit_eda/1_audits_sources.ipynb` : audit de chaque source telle qu'elle est livrée (origine, structure, variables, qualité). Les audits faits après le premier rapport y sont rangés dans la partie de leur source, avec leur date.
- `02_audit_eda/1_eda_sources.ipynb` : la cible, les relations à l'intérieur de chaque source, les relations avec la cible, et la synthèse. Les cartes ont besoin d'un accès à internet (contours des départements et des communes).
- `02_audit_eda/2_audit_eda_table_modelisation.ipynb` : audit et EDA de la table de modélisation, à lancer après l'étape 4. Il se relance à chaque changement de la table ; sa synthèse garde, pour chaque passage, la date et les enseignements tirés.
- `02_audit_eda/3_temporalite_prix_national.ipynb` : SARIMA sur le prix national. Ce n'est pas un modèle de l'outil : l'analyse sert à tirer des enseignements pour formuler les hypothèses de modélisation.
- `02_audit_eda/04_baseline.ipynb` : la baseline de la cible (prix N-1 comme prévision), sur le découpage du socle. Il lit la table de modélisation et ne produit aucun fichier : aucun notebook ni script ne doit écrire dans `data/processed/` sous le nom d'un fichier de production.

### 4. Feature engineering et table de modélisation

```bash
python 01_preparation/feature_engineering.py            # les 11 sections, dans l'ordre
python 01_preparation/feature_engineering.py 3 3bis     # seulement ces sections
python 01_preparation/feature_engineering.py liste      # liste des sections
```

Les choix sont expliqués en commentaire dans `feature_engineering.py` ; les constats de l'EDA qui les justifient sont dans les notebooks de `02_audit_eda`.

Pour chaque source : rattachement des communes aux sous-bassins, agrégation, jointure à gauche sur la cible, export `fe_cible_<source>_sbv.parquet`. Puis fusion des sources et construction de `table_modelisation_bv.parquet` (voir plus bas).

**Raccourci** : la table de modélisation déjà construite peut être téléchargée directement, sans relancer la préparation ni le feature engineering :

```bash
python 00_telechargement/telechargement.py table_modelisation
```

Elle est rangée dans `data/processed/`, là où `commun.py` la lit. L'équipe, elle, reconstruit toujours la table avec le pipeline.

Une fois la table construite ou téléchargée, contrôles du socle (environ 5 minutes) :

```bash
pytest                                  # tous les contrôles
pytest tests/test_gabarit_pont.py       # les contrôles d'un seul module
```

Les contrôles lisent `data/processed/table_modelisation_bv.parquet` (ceux de Keras utilisent des données inventées) et ne modifient aucun fichier. Résultat attendu : `31 passed`.

### 5. Modélisation

Les hypothèses sont des notebooks (`1_tour1`, `3_tour2`) ; les préparations du tour 2 et les deux tests finaux sont des scripts (`2_prepa_tour2`, `4_evaluation_2025`) ; les résultats du test 2025 sont comparés dans le notebook `4_evaluation_2025/evaluation_2025.ipynb`. Ordre des sous-dossiers de `03_modelisation` : `1_tour1`, puis `2_prepa_tour2`, `3_tour2` et `4_evaluation_2025`. Chaque notebook affiche ses résultats au fur et à mesure, avec leurs commentaires et une synthèse. Pour une nouvelle hypothèse, copier `modelisation_template.ipynb`.

Le notebook H7 (modèle par vente) est enregistré avec ses sorties : son exécution complète dépasse deux heures, il peut être lu sans être relancé.

Chaque notebook importe `socle/commun.py`, qui fixe pour toute l'équipe :

- la table et le masque des lignes communes (`charger_table`) ;
- le découpage temporel (`decouper`, `fenetres`, `plis_temporels`) ;
- la prévision naïve (prix N = prix N-1) ;
- le tableau commun des erreurs (`evaluer`) et le test contre la naïve (`tester` : bootstrap par bassin de vie et Wilcoxon apparié) ;
- l'évaluation du modèle par vente (P11) : `charger_ventes`, `evaluer_ventes` et `tester_ventes` (biens vendus), `panier_fixe` et `evaluer_panier` (panier fixe, comparaison avec les modèles agrégés).

Les gabarits du socle (boosting, hiérarchique, pont, Keras) donnent à chacun le même modèle, avec les mêmes réglages, pour que les résultats soient comparables d'une hypothèse à l'autre.

Les scripts complètent les notebooks. P6, P7 et P9 se lancent avant les notebooks de `3_tour2`, qui lisent leurs fichiers :

```bash
python 03_modelisation/2_prepa_tour2/P6_table_periode_longue.py     # contrôles de la période longue (aucun fichier produit)
python 03_modelisation/2_prepa_tour2/P7_capacite_achat.py           # taux d'effort : fe_capacite_achat_sbv.parquet
python 03_modelisation/2_prepa_tour2/P9_volume_elan.py              # volume et élan : fe_volume_elan_sbv.parquet
python 03_modelisation/3_tour2/H7_optimisation.py --etape 1         # optimisation du réseau de H7 (étapes 1, 2, 3)
python 03_modelisation/4_evaluation_2025/modele_hierarchique_test_final.py   # test final 2025, modèle agrégé
python 03_modelisation/4_evaluation_2025/modele_keras_test_final.py           # test final 2025, modèle par vente
```

Les deux scripts de test final écrivent leurs résultats dans `data/processed/` (fichiers `*_test_final_resultats.csv`) ; `evaluation_2025.ipynb` les lit et s'arrête si l'un des deux manque : **lancer les deux scripts avant le notebook**.

L'optimisation (longue) produit `meilleurs_reglages_keras.json`, que H7 lit ensuite. Pour la sauter : `python 00_telechargement/telechargement.py reglages_keras`. Les deux tests 2025 vérifient d'abord qu'ils redonnent le résultat de 2024 ; sinon ils s'arrêtent sans ouvrir 2025.

### 6. Outils

Le modèle final et les lignes d'entrée de l'outil, dans cet ordre :

```bash
python 04_outils/fourchette_h7v4.py          # fourchette d'erreur, tirée du test 2025
python 04_outils/modele_final_h7v4.py        # modèle final (appris sur toutes les années) : écrit models/h7v4/, copie les deux fichiers lus par l'application
python 04_outils/controle_lignes_2025.py     # contrôle de la construction des lignes sur 2025
python 04_outils/produire_lignes_2026.py     # lignes d'entrée de 2026
python 04_outils/produire_ventes_comparables.py   # ventes 2023-2025 pour les ventes comparables et la fourchette de l'outil
python 04_outils/controle_estimer_2025.py    # contrôle de estimer() sur les ventes 2025 (comparaison avec la prévision naïve)
```

`lignes_annee.py` n'est pas à lancer : il contient la construction des lignes, utilisée par `controle_lignes_2025.py`, `controle_estimer_2025.py` et `produire_lignes_2026.py`. `fourchette_h7v4.py`, `modele_final_h7v4.py`, `produire_lignes_2026.py` et `produire_ventes_comparables.py` refusent d'écraser un fichier qu'ils ont déjà produit.

Scripts de support, lancés seulement si leur source change :

```bash
python 04_outils/preparer_communes_app.py          # tables des communes lues par la page Outil
python 04_outils/preparer_eda_app.py               # fichiers lus par la page Exploration des données
python 04_outils/produire_ref_trimestres_2025.py   # référence des trimestres 2025, lue par controle_lignes_2025.py
```

**Règle sur les données de l'application.** L'application n'a pas accès à `data/` : elle ne lit que fichiers de `05_application/donnees/`, versionnés dans Git. Chacun est produit par un script de `04_outils`, qui lit `data/processed/` et écrit dans ce dossier. On ne les modifie jamais à la main.

#### Le modèle livré : `models/h7v4/`

Le dossier contient le réseau H7v4 (`h7v4.keras`) et son fichier de paramètres (`h7v4_parametres.json` : préparation du bien, mise à l'échelle, table des sous-bassins, variables, réglages, fourchette d'erreur). Les deux vont ensemble. `MODEL_CARD.md` en est la fiche (ce que le modèle estime, ses limites, ses chiffres), générée à partir du fichier de paramètres par `04_outils/fiche_modele.py`. `SHA256SUMS` donne l'empreinte de chacun des deux fichiers ; pour vérifier qu'une copie est intacte :

```bash
cd models/h7v4 && sha256sum -c SHA256SUMS
```

Le modèle estime un bien, il ne prévoit pas le marché. Appris sur 2017-2025, il ne peut pas être jugé sur 2025 : le test de 2025 utilise le même réseau appris jusqu'en 2024 (`03_modelisation/4_evaluation_2025`). `modele_final_h7v4.py` refuse d'écraser un modèle déjà présent dans `models/h7v4/`.

#### Mise à jour de DVF+

`04_outils/mettre_a_jour_dvf.sh` récupère une nouvelle livraison du Cerema et la met au périmètre du projet. Il met à jour les données seulement : le modèle n'est pas réappris, ni les lignes d'entrée ni les ventes comparables régénérées.

```bash
bash 04_outils/mettre_a_jour_dvf.sh --help
bash 04_outils/mettre_a_jour_dvf.sh tout archive_cerema.zip                  # extraire, assembler, filtrer
bash 04_outils/mettre_a_jour_dvf.sh extraire archive_cerema.zip --annee 2026 --mois-max 6   # étape 1 seule (py7zr requis)
bash 04_outils/mettre_a_jour_dvf.sh assembler                                # étape 2 seule
bash 04_outils/mettre_a_jour_dvf.sh filtrer                                  # étape 3 seule
```

- `extraire` (`extraire_dvf_plus.py`) : lit les archives 7z de la livraison (dumps PostgreSQL) et extrait les mutations de l'année demandée ;
- `assembler` : réunit les extraits des régions en un fichier `dvf_plus_assemble.csv` ;
- `filtrer` (`filtrer_dvf_plus.py`) : exclut les DOM, retire les accolades des codes commune (`{33125}` devient `33125`) et applique le filtre du périmètre de `01_preparation/preparation.py` (vente d'un seul logement ancien, VEFA exclues). Il écrit `data/processed/dvf_plus_residentiel_<année>.parquet` et refuse d'écraser un fichier existant.

Les fichiers intermédiaires sont rangés dans `data/raw/dvf_plus_maj/` (variable `DVF_MAJ_DOSSIER` pour changer de dossier) et ne sont jamais versionnés.

#### Contrôle de vraisemblance 2026

Les ventes de 2026 ne font pas partie du projet. Après une mise à jour de DVF+, elles ne servent qu'à une indication : comparer les estimations de H7v4 aux prix réellement constatés. Ce n'est pas un test (six mois de données seulement).

### 7. Application

```bash
cd 05_application
streamlit run app.py
```

Le lancement depuis `05_application` est nécessaire pour que le thème de `.streamlit/` soit lu.

## Sources de données

| Nom court                                                                                                | Source                                                                                                                | Producteur                                    | Période             |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------- |
| `dvf`                                                                                                    | DVF+ (mutations immobilières)                                                                                         | Cerema                                        | 2014-2025           |
| `dvf_residentiel`                                                                                        | DVF+ filtré (bloc 2)                                                                                                  | équipe                                        | 2014-2025           |
| `dvf_audit`                                                                                              | Échantillon d'audit de DVF+ (100 000 mutations)                                                                       | équipe                                        | 2014-2025           |
| `table_modelisation`                                                                                     | Table de modélisation déjà construite (raccourci, voir étape 4)                                                       | équipe                                        | 2015-2025           |
| `filosofi_commune`, `liaison_bv_scindes`, `taux_credit_mois`, `ventes_apprentissage`, `ventes_test_2025` | Fichiers déjà préparés (blocs 1, 3 bis, 5 et 11), kit du parcours court                                               | équipe                                        | —                   |
| `fe_capacite_achat`, `fe_volume_elan`                                                                    | Taux d'effort (P7) et volume et élan (P9) par sous-bassin, kit du parcours court                                       | équipe                                        | —                   |
| `reglages_keras`                                                                                         | Réglages optimisés du réseau de H7 (`meilleurs_reglages_keras.json`), kit du parcours court                           | équipe                                        | —                   |
| `passage_geo`                                                                                            | Table de passage des communes 2003-2026                                                                               | Insee                                         | géographie 2026     |
| `bassins_vie`                                                                                            | Bassins de vie 2022                                                                                                   | Insee                                         | géographie 2026     |
| `bassins_vie_2019_2022`                                                                                  | Bassins de vie, fichier des années 2019 à 2022                                                                        | Insee                                         | 2019-2022           |
| `communes`                                                                                               | Référentiel des communes (dont la superficie et la grille de densité)                                                 | data.gouv.fr (Insee, IGN)                     | géographie 2026     |
| `superficie_arrondissements`                                                                             | Superficie des 45 arrondissements de Paris, Lyon et Marseille                                                         | API Découpage administratif (geo.api.gouv.fr) | géographie actuelle |
| `filosofi`                                                                                               | Revenus disponibles par commune (Filosofi)                                                                            | Insee                                         | 2014-2021           |
| `menages`                                                                                                | Couples, familles, ménages (recensement)                                                                              | Insee                                         | 2019-2022           |
| `logement`                                                                                               | Logement (recensement)                                                                                                | Insee                                         | 2022                |
| `delinquance`                                                                                            | Délinquance enregistrée par commune                                                                                   | SSMSI                                         | 2016-2025           |
| `lovac`                                                                                                  | Logements privés vacants                                                                                              | Cerema                                        | 2020-2026           |
| `equipements`                                                                                            | Indicateurs d'équipements (BPE 2015 et 2020), table construite par l'équipe                                           | Insee / équipe                                | 2015, 2020          |
| `bpe`                                                                                                    | Base permanente des équipements 2025                                                                                  | Insee                                         | 2025                |
| `bpe_evolution`                                                                                          | Base permanente des équipements en évolution (nombre d'équipements par type et par territoire, dont le bassin de vie) | Insee                                         | 2015, 2020, 2025    |
| `taux`                                                                                                   | Taux des crédits immobiliers aux particuliers                                                                         | Banque de France                              | 1998-2026           |
| `emploi`                                                                                                 | Établissements employeurs et salariés du secteur privé, par commune et par activité                                   | Urssaf                                        | 2006-2025           |
| `logement_millesimes`                                                                                    | Logement (recensement), bases IRIS des millésimes antérieurs                                                          | Insee                                         | 2016-2021, 2023     |
| `annonces_ventes`                                                                                        | Annonces de vente du Haut-Rhin (axe 2)                                                                                | échantillon fourni                            | 2019-2023           |
| `annonces_locations`                                                                                     | Annonces de location du Haut-Rhin (axe 2)                                                                             | échantillon fourni                            | —                   |

## Fichiers produits par la préparation

Tous dans `data/processed/`, au format Parquet (les codes géographiques restent en texte, avec leur zéro initial).

| Bloc  | Fichier                                                | Une ligne =                                                     |
| ----- | ------------------------------------------------------ | --------------------------------------------------------------- |
| 1     | `liaison_commune_bv2022.parquet`                       | un code commune (depuis 2003)                                   |
| 1     | `liaison_bv_scindes.parquet`                           | un code commune, avec son sous-bassin de vie                    |
| 1     | `communes_superficie_2026.parquet`                     | une commune 2026 (Paris, Lyon, Marseille par arrondissement)    |
| 2     | `dvf_plus_residentiel.parquet`                         | une vente                                                       |
| 3     | `dvf_plus_echantillon_apprentissage.parquet`           | une vente de l'échantillon                                      |
| 3 bis | `ventes_apprentissage.parquet`                         | une vente de l'échantillon, prête pour le modèle par vente (H7) |
| 3 bis | `ventes_test_2025.parquet`, `ids_ventes_test_2025.csv` | une vente de 2025, réservée au test final du niveau 2           |
| 4     | `cible_commune_{mois,trimestre,annee}.parquet`         | une commune × une période                                       |
| 4     | `cible_bv_{mois,trimestre,annee}.parquet`              | un bassin de vie × une période                                  |
| 4     | `cible_sbv_{mois,trimestre,annee}.parquet`             | un sous-bassin de vie × une période                             |
| 5     | `filosofi_commune_2014_2021.parquet`                   | une commune × une année                                         |
| 6     | `menages_commune_2022.parquet`                         | une commune                                                     |
| 6     | `menages_commune_2019_2022.parquet`                    | une commune × une année de recensement                          |
| 7     | `logement_commune_2022.parquet`                        | une commune                                                     |
| 8     | `delinquance_commune_annee.parquet`                    | une commune × une année                                         |
| 9     | `lovac_commune_annee.parquet`                          | une commune × un millésime                                      |
| 10    | `equipements_commune_millesime.parquet`                | une commune × un millésime BPE                                  |
| 11    | `taux_credit_{mois,trimestre,annee}.parquet`           | une période                                                     |
| 12    | `equipements_commune_2025.parquet`                     | une commune                                                     |

**Conventions communes**

- Paris, Lyon et Marseille sont codées **par arrondissement**, comme dans DVF+.
- Toute source se rattache au bassin de vie et au sous-bassin par la table de liaison du bloc 1 (colonne `CODGEO_INI`).
- Au bassin de vie et au sous-bassin, les taux et les parts se recalculent en additionnant les effectifs bruts conservés dans chaque fichier, jamais en faisant la moyenne des taux communaux.
- L'année 2025 est réservée au test.

**Fichiers produits par les préparations du tour 2** (`03_modelisation/2_prepa_tour2/`), dans `data/processed/` :

| Script | Fichier | Une ligne = |
| ------ | ------- | ----------- |
| `P6_table_periode_longue.py` | aucun (contrôles seulement) | — |
| `P7_capacite_achat.py` | `fe_capacite_achat_sbv.parquet` | une année × un sous-bassin de vie × un type de bien |
| `P9_volume_elan.py` | `fe_volume_elan_sbv.parquet` | une année × un sous-bassin de vie × un type de bien |

## Maille de modélisation : le sous-bassin de vie

Les bassins de vie de 100 000 habitants ou plus sont découpés selon la grille de densité des communes : `_1` (grands centres et centres intermédiaires), `_2` (ceintures urbaines et petites villes), `_3` (rural). Les bassins plus petits gardent leur code. Chaque arrondissement de Paris, Lyon et Marseille forme son propre sous-bassin.

Un prix médian n'est retenu qu'à partir de **20 ventes** dans le sous-bassin, l'année et le type de bien.

## Table de modélisation

`data/processed/table_modelisation_bv.parquet`, produite par `01_preparation/feature_engineering.py` (section 8), lue par `commun.charger_table()`.

- **Une ligne** = une année × un sous-bassin de vie × un type de bien (maison ou appartement). Index à 3 niveaux : `annee`, `sous_bassin_de_vie`, `type_bien`.
- **Cible** : `prix`, prix médian au m² de l'année N.
- **Variables explicatives** : prises sur l'année N-1 ou en photo fixe (recensement, LOVAC, BPE, Filosofi) ; `prix_n1` (prix de l'année précédente) et `maison` (indicatrice du type de bien) comprises.
- **Hors modèle** : `n_ventes` (ventes de l'année N, sert seulement au seuil), `BV2022` (bassin de vie, identifiant du bootstrap de `commun.tester`), variables de contrôle qualité (`part_indicateurs_estimes`, `part_logements_taux_impute`, `part_menages_couverts`).
- **Lignes exportées** : seulement celles où `prix` et `prix_n1` sont renseignés.

## Échantillon d'apprentissage DVF+

`dvf_plus_echantillon_apprentissage.parquet` : 907 306 ventes d'un seul logement ancien (maison ou appartement), 2014-2024, France métropolitaine hors DOM. Aucune vente de 2025 (année du test).

**Méthode :** une vente sur dix a été tirée au hasard dans chaque département et chaque année, avec au moins 300 ventes à chaque fois (bloc 3 de la préparation). Le tirage est reproductible : graine fixée à 42.

```python
import pandas as pd

df = pd.read_parquet("../data/processed/dvf_plus_echantillon_apprentissage.parquet")
```

Vérification au premier chargement : `len(df)` doit afficher `907306`.

**Important : les poids `w_h`.** L'échantillon n'est pas proportionnel : les petits départements y sont surreprésentés. Pour tout calcul portant sur plusieurs départements ou plusieurs années (moyenne, médiane, part de maisons, modèle), il faut utiliser la colonne `w_h`. Par exemple :

```python
import numpy as np
np.average(df["libtypbien"] == "UNE MAISON", weights=df["w_h"])
```

## Ventes pour le modèle par vente (bloc 3 bis)

Deux fichiers pour le modèle par vente (H7, niveau 2), produits par le bloc 3 bis de la préparation :

- `ventes_apprentissage.parquet` : **904 949 ventes** (2014-2024), l'échantillon du bloc 3 après les règles d'exclusion du bloc 4 (valeur foncière renseignée et positive, prix entre 50 et 50 000 €/m²). 2 357 ventes retirées (0,26 %, entre 0,21 et 0,37 % selon l'année) ; poids `w_h` inchangés.
- `ventes_test_2025.parquet` : **80 673 ventes** de 2025, tirées une seule fois (même règle, graine 2025). **Ne pas l'ouvrir avant le test final du niveau 2.**

Colonnes : identifiants (`idmutation`, `coddep`, `code_commune`, `BV2022`, `sous_bassin_de_vie`, jamais des variables du modèle), date (`anneemut`, `moismut`), bien (`type_bien`, `sbati`, `sterr`, `nblocdep`, `nb_pieces`, `geompar_x`, `geompar_y`), cible `prix_m2` et poids `w_h`. `valeurfonc` n'est pas gardée : avec `sbati`, elle redonne la cible.

Le prix N-1 du territoire, les variables « où » et le seuil de 20 ventes sont ajoutés au moment de la modélisation, par `commun.charger_ventes`.

## Livrables

- Rapport d'exploration et de visualisation des données
- Rapport de modélisation
- Rapport final
- Dépôt GitHub associé
- Modèle H7v4 enregistré et documenté (`models/h7v4/`) et fonctions de l'outil

L'application Streamlit est un démonstrateur, hors livrables.

## Équipe

- Cécile
- Christian
- Gabrielle
- Theophyle