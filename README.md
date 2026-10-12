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

## Sources de données

### Annonces immobilières

- Ventes
- Locations

### Données complémentaires

- DVF (Demandes de Valeurs Foncières)
- INSEE
- Data.gouv
- Données territoriales et socio-économiques

## Échantillon DVF+ (apprentissage)

**Fichier :** `data/dvf_plus_echantillon_apprentissage.parquet`, soit 835 288 ventes d'un seul logement ancien (maison ou appartement), sur la période 2014-2023, en France métropolitaine hors DOM. Il comporte 53 colonnes et pèse 33 Mo.

**Méthode :** une vente sur dix a été tirée au hasard dans chaque département et chaque année, avec au moins 300 ventes à chaque fois. Le notebook `description_dataset/echantillonnage_dvf_plus.ipynb` détaille la méthode.

### Installer les bibliothèques

Le format Parquet nécessite `pyarrow`, qui figure dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

### Lire l'échantillon

Depuis un notebook du dossier `description_dataset/` :

```python
import pandas as pd

df = pd.read_parquet("../data/dvf_plus_echantillon_apprentissage.parquet")
```

Pas besoin de `sep` ni de `dtype` : le format Parquet conserve les types des colonnes, donc les codes géographiques (`coddep`, `l_codinsee`) restent en texte avec leur zéro initial.

Vérification au premier chargement : `df.shape` doit afficher `(835288, 53)`.

### Important : les poids `w_h`

L'échantillon n'est pas proportionnel : les petits départements y sont surreprésentés. Pour tout calcul portant sur plusieurs départements ou plusieurs années (moyenne, médiane, part de maisons, modèle), il faut utiliser la colonne `w_h`. Par exemple :

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
- Gabrielle
- Theophyle
- Christian