# Projet Compagnon Immobilier

## Description

L'objectif du projet est de développer une solution innovante permettant aux acheteurs et investisseurs immobiliers d'explorer, comparer et estimer les prix des logements sur différents territoires.

Le projet s'appuie sur deux axes principaux :

### 1. Analyse des données immobilières et territoriales

- Prédiction de l'évolution des prix des logements selon les territoires.
- Exploitation des ventes immobilières historiques.
- Intégration de facteurs socio-économiques (démographie, éducation, criminalité, transports, services, etc.).

### 2. Exploitation des annonces immobilières

- Extraction d'informations à partir des descriptions d'annonces.
- Analyse des images des biens immobiliers.
- Développement d'un chatbot intelligent capable de comprendre les requêtes des utilisateurs et de proposer les annonces les plus pertinentes.

L'ambition du projet est de combiner des approches de machine learning classiques (analyse tabulaire et modélisation prédictive) avec le traitement du langage naturel afin de fournir aux acheteurs un outil d'aide à la décision fiable et interactif.

## Organisation du dépôt

```
projet_compagnon/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py              ← chemins communs à tous les codes
├── data/                  ← local, jamais sur GitHub
│   ├── raw/               ← sources brutes (00_telechargement)
│   └── processed/         ← fichiers créés par nos codes (01_preparation)
├── 00_telechargement/
│   └── telechargement.py
├── 01_preparation/
│   ├── preparation.py     ← version courte, commentée
│   └── preparation.ipynb  ← même code, avec les explications
├── 02_audit_eda/
│   ├── audits.ipynb       ← audit de chaque source
│   ├── eda.ipynb          ← analyse exploratoire : cible, relations, synthèse
│   └── audits_eda.py      ← version script des deux notebooks
└── 03_modelisation/       ← à venir
```

GitHub ne contient que du code. Les données sont sur un Google Drive partagé, lisible par toute personne ayant le lien : https://drive.google.com/drive/folders/1bVlZgMTxjtoWMveHXJEsxY6LPRo6U6nZ

Les rapports (rédigés en Word, remis en PDF) sont tenus à part, sur le Google Drive de l'équipe.

## Installation

```bash
pip install -r requirements.txt
```

Toutes les commandes se lancent depuis la racine du projet.

## Ordre d'exécution

### 1. Télécharger les données

```bash
python 00_telechargement/telechargement.py              # liste les sources disponibles
python 00_telechargement/telechargement.py taux lovac   # télécharge seulement ces sources
python 00_telechargement/telechargement.py tout         # télécharge toutes les sources
```

Un fichier déjà téléchargé n'est pas retéléchargé. Si un téléchargement échoue (Google Drive limite parfois les gros téléchargements), il suffit de relancer la même commande plus tard.

**DVF+ : deux parcours au choix**

- **Rapide** : `python 00_telechargement/telechargement.py dvf_residentiel` télécharge DVF+ déjà filtré (un seul fichier, 1,4 Go). Le filtrage du bloc 2 est alors sauté.
- **Complet** : `python 00_telechargement/telechargement.py dvf` télécharge DVF+ brut (97 fichiers départementaux, plusieurs Go), filtré ensuite par le bloc 2.

Le même principe vaut pour l'échantillon d'audit de DVF+ (`02_audit_eda/audits.ipynb`) : `python 00_telechargement/telechargement.py dvf_audit` le télécharge directement, sinon il est tiré à partir de DVF+ brut.

### 2. Préparer les données

```bash
python 01_preparation/preparation.py
```

ou exécuter `01_preparation/preparation.ipynb`, qui contient le même code avec les explications.

### 3. Audits et analyse exploratoire

```bash
python 02_audit_eda/audits_eda.py           # audits puis EDA
python 02_audit_eda/audits_eda.py audits    # audits seulement
python 02_audit_eda/audits_eda.py eda       # EDA seulement
```

ou exécuter `02_audit_eda/audits.ipynb` et `02_audit_eda/eda.ipynb`, qui contiennent les interprétations. Le script affiche les tableaux dans la console et enregistre les graphiques dans `02_audit_eda/figures/`. Les cartes de l'EDA ont besoin d'un accès à internet (contours des départements et des communes).

## Sources de données

| Nom court | Source | Producteur | Période |
|---|---|---|---|
| `dvf` | DVF+ (mutations immobilières) | Cerema | 2014-2025 |
| `dvf_residentiel` | DVF+ filtré (bloc 2) | équipe | 2014-2025 |
| `dvf_audit` | Échantillon d'audit de DVF+ (100 000 mutations) | équipe | 2014-2025 |
| `passage_geo` | Table de passage des communes 2003-2026 | Insee | géographie 2026 |
| `bassins_vie` | Bassins de vie 2022 | Insee | géographie 2026 |
| `filosofi` | Revenus disponibles par commune (Filosofi) | Insee | 2014-2021 |
| `menages` | Couples, familles, ménages (recensement) | Insee | 2019-2022 |
| `logement` | Logement (recensement) | Insee | 2022 |
| `delinquance` | Délinquance enregistrée par commune | SSMSI | 2016-2025 |
| `lovac` | Logements privés vacants | Cerema | 2020-2026 |
| `equipements` | Indicateurs d'équipements (BPE 2015 et 2020), table construite par l'équipe | Insee / équipe | 2015, 2020 |
| `bpe` | Base permanente des équipements 2025 | Insee | 2025 |
| `bpe_evolution` | Base permanente des équipements en évolution (nombre d'équipements par type et par territoire, dont le bassin de vie) | Insee | 2015, 2020, 2025 |
| `communes` | Référentiel des communes (dont la superficie) | data.gouv.fr (Insee, IGN) | géographie 2026 |
| `taux` | Taux des crédits immobiliers aux particuliers | Banque de France | 1998-2026 |
| `annonces_ventes` | Annonces de vente du Haut-Rhin (axe 2) | échantillon fourni | 2019-2023 |
| `annonces_locations` | Annonces de location du Haut-Rhin (axe 2) | échantillon fourni | — |

## Fichiers produits par la préparation

Tous dans `data/processed/`, au format Parquet (les codes géographiques restent en texte, avec leur zéro initial).

| Bloc | Fichier | Une ligne = |
|---|---|---|
| 1 | `liaison_commune_bv2022.parquet` | un code commune (depuis 2003) |
| 2 | `dvf_plus_residentiel.parquet` | une vente |
| 3 | `dvf_plus_echantillon_apprentissage.parquet` | une vente de l'échantillon |
| 4 | `cible_commune_{mois,trimestre,annee}.parquet` | une commune × une période |
| 4 | `cible_bv_{mois,trimestre,annee}.parquet` | un bassin de vie × une période |
| 5 | `filosofi_commune_2014_2021.parquet` | une commune × une année |
| 6 | `menages_commune_2022.parquet` | une commune |
| 6 | `menages_bv_2019_2024.parquet` | un bassin de vie × une année |
| 7 | `logement_commune_2022.parquet` | une commune |
| 8 | `delinquance_commune_annee.parquet` | une commune × une année |
| 9 | `lovac_commune_annee.parquet` | une commune × un millésime |
| 10 | `equipements_commune_millesime.parquet` | une commune × un millésime BPE |
| 11 | `taux_credit_{mois,trimestre,annee}.parquet` | une période |

**Conventions communes**

- Paris, Lyon et Marseille sont codées **par arrondissement**, comme dans DVF+.
- Toute source se rattache au bassin de vie par la table de liaison du bloc 1 (colonne `CODGEO_INI`).
- Au bassin de vie, les taux et les parts se recalculent en additionnant les effectifs bruts conservés dans chaque fichier, jamais en faisant la moyenne des taux communaux.
- L'année 2025 est réservée au test.

## Échantillon d'apprentissage DVF+

`dvf_plus_echantillon_apprentissage.parquet` : 835 288 ventes d'un seul logement ancien (maison ou appartement), 2014-2023, France métropolitaine hors DOM, 53 colonnes.

**Méthode :** une vente sur dix a été tirée au hasard dans chaque département et chaque année, avec au moins 300 ventes à chaque fois (bloc 3 de la préparation). Le tirage est reproductible : graine fixée à 42.

```python
import pandas as pd

df = pd.read_parquet("../data/processed/dvf_plus_echantillon_apprentissage.parquet")
```

Vérification au premier chargement : `df.shape` doit afficher `(835288, 53)`.

**Important : les poids `w_h`.** L'échantillon n'est pas proportionnel : les petits départements y sont surreprésentés. Pour tout calcul portant sur plusieurs départements ou plusieurs années (moyenne, médiane, part de maisons, modèle), il faut utiliser la colonne `w_h`. Par exemple :

```python
import numpy as np
np.average(df["libtypbien"] == "UNE MAISON", weights=df["w_h"])
```

## Livrables

- Rapport d'exploration et de visualisation des données
- Rapport de modélisation
- Rapport final
- Dépôt GitHub associé

## Équipe

- Cécile
- Christian
- Gabrielle
- Theophyle
