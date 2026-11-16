# =============================================================================
# 01_preparation/preparer_communes_app.py : tables des communes pour l'application (page Outil)
# =============================================================================
# Pourquoi : l'application en ligne n'a pas accès à data/ ; elle lit des copies réduites, placées dans
# 05_application/donnees/. Ce script en produit deux, toutes deux liées aux communes :
#
# 1. communes_sous_bassins.parquet : la table de localiser()
#    Une ligne par code commune (tout code ayant existé depuis 2003, plus les arrondissements de Paris,
#    Lyon et Marseille), avec :
#      - CODGEO_INI         : le code saisi, tel qu'on le cherche (clé de la table) ;
#      - code_commune       : le code utilisé dans les ventes (commune 2026, ou arrondissement) ;
#      - sous_bassin_de_vie : le territoire du modèle ;
#      - BV2022             : le bassin de vie (recherche élargie des ventes comparables).
#    Même règle que preparation.py (bloc 5) pour code_commune : un arrondissement garde son propre code.
#
# 2. communes_recherche.parquet : la liste déroulante de la page Outil
#    Une ligne par commune proposée à l'utilisateur, avec son libellé « nom (code postal) » et son code INSEE.
#      - Source : communes-france-2026.csv, déjà utilisé par preparation.py (aucun téléchargement en plus).
#        Il donne pour chaque commune son code postal principal : on n'affiche que celui-là, car la liste
#        complète contient les codes CEDEX (plus de 80 pour Bordeaux), inutiles pour un particulier.
#      - Paris, Lyon et Marseille sont remplacées par leurs arrondissements (« Paris 8e arrondissement
#        (75008) ») : le modèle et les ventes sont à l'arrondissement.
#      - France métropolitaine seulement (périmètre de l'outil, plan de développement).
#      - Deux communes de même nom et même code postal (cas rare) : le code INSEE est ajouté au libellé.
#
# À lancer depuis la racine du projet :  python 01_preparation/preparer_communes_app.py
# Entrées : data/processed/liaison_bv_scindes.parquet, data/raw/communes-france-2026.csv
# Sorties : 05_application/donnees/communes_sous_bassins.parquet, communes_recherche.parquet

import sys
from pathlib import Path

import pandas as pd

# config.py se trouve à la racine du projet
RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from config import DATA_OUT, DATA_RAW

DONNEES_APP = RACINE / "05_application" / "donnees"
DONNEES_APP.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. TABLE DE localiser() : code commune -> sous-bassin et bassin de vie
# =============================================================================
liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet",
                          columns=["CODGEO_INI", "CODGEO_2026", "BV2022", "sous_bassin_de_vie"])

# Code de la commune dans les ventes : le code 2026, sauf pour un arrondissement, qui garde le sien
arrondissement = liaison["CODGEO_INI"].str.match(r"^(751\d\d|6938\d|132\d\d)$")
liaison["code_commune"] = liaison["CODGEO_2026"].where(~arrondissement, liaison["CODGEO_INI"])

communes = liaison[["CODGEO_INI", "code_commune", "sous_bassin_de_vie", "BV2022"]]

# Contrôles : un code = une ligne, et les arrondissements de contrôle ont leur propre sous-bassin
assert communes["CODGEO_INI"].is_unique, "un code commune apparaît deux fois"
print("1. Communes :", len(communes),
      "| sans sous-bassin :", communes["sous_bassin_de_vie"].isna().sum(),
      "| sans bassin de vie :", communes["BV2022"].isna().sum())
print(communes[communes["CODGEO_INI"].isin(["33063", "75101", "69381", "13201", "75056"])].to_string(index=False))

communes.to_parquet(DONNEES_APP / "communes_sous_bassins.parquet", index=False)


# =============================================================================
# 2. LISTE DÉROULANTE DE LA PAGE OUTIL : « nom (code postal) » -> code INSEE
# =============================================================================
ref = pd.read_csv(DATA_RAW / "communes-france-2026.csv",
                  usecols=["code_insee", "nom_standard", "nom_sans_accent", "code_postal", "dep_code"],
                  dtype=str)

# France métropolitaine : départements 01 à 95, 2A et 2B (les DOM ont un code à 3 chiffres)
ref = ref[ref["dep_code"].str.len() == 2].copy()

# --- Paris, Lyon, Marseille : la ville entière est remplacée par ses arrondissements ---
# Code postal d'un arrondissement : 750xx pour Paris (75116 est aussi utilisé pour le 16e, on garde 75016),
# 690xx pour Lyon, 130xx pour Marseille, où xx est le numéro de l'arrondissement.
VILLES = {
    "75056": ("Paris", 75101, 20, "750"),      # 75101 à 75120
    "69123": ("Lyon", 69381, 9, "690"),        # 69381 à 69389
    "13055": ("Marseille", 13201, 16, "130"),  # 13201 à 13216
}
lignes = []
for code_ville in VILLES:
    nom, premier_code, nombre, debut_cp = VILLES[code_ville]
    for numero in range(1, nombre + 1):
        rang = "1er" if numero == 1 else str(numero) + "e"
        lignes.append({"code_insee": str(premier_code + numero - 1),
                       "nom_standard": nom + " " + rang + " arrondissement",
                       "nom_sans_accent": nom.lower() + " " + str(numero).zfill(2),
                       "code_postal": debut_cp + str(numero).zfill(2),
                       "dep_code": code_ville[:2]})
ref = ref[~ref["code_insee"].isin(list(VILLES))]
ref = pd.concat([ref, pd.DataFrame(lignes)], ignore_index=True)

# --- Libellé affiché : « Bordeaux (33000) » ---
ref["libelle"] = ref["nom_standard"] + " (" + ref["code_postal"] + ")"
# Deux communes au même libellé (même nom, même code postal) : on ajoute le code INSEE pour les distinguer
double = ref["libelle"].duplicated(keep=False)
ref.loc[double, "libelle"] = ref.loc[double, "libelle"] + " – " + ref.loc[double, "code_insee"]
print("2. Libellés complétés par le code INSEE (homonymes au même code postal) :", double.sum())

# Ordre alphabétique sans tenir compte des accents (« Écully » rangé avec les E)
recherche = ref.sort_values("nom_sans_accent")[["libelle", "code_insee"]].reset_index(drop=True)

# Contrôles : libellés uniques, et chaque commune proposée est connue de localiser()
assert recherche["libelle"].is_unique, "deux communes ont le même libellé"
inconnues = recherche[~recherche["code_insee"].isin(communes["CODGEO_INI"])]
print("   Communes proposées :", len(recherche), "| inconnues de localiser() :", len(inconnues))
if len(inconnues) > 0:
    print(inconnues.head(10).to_string(index=False))
print(recherche[recherche["libelle"].str.startswith(("Bordeaux", "Paris 8e", "Lyon 3e", "Marseille 1er",
                                                      "Saint-Denis (93"))].to_string(index=False))

recherche.to_parquet(DONNEES_APP / "communes_recherche.parquet", index=False)
print("Enregistré dans", DONNEES_APP)
