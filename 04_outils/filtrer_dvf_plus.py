# =============================================================================
# filtrer_dvf_plus.py : filtre de la préparation (bloc 2) appliqué à des mutations DVF+ extraites d'une livraison
# =============================================================================
# Étape 3 de 04_outils/mettre_a_jour_dvf.sh. Lit le fichier assemblé par extraire_dvf_plus.py
# (dvf_plus_assemble.csv, séparateur |) et lui applique les règles de la préparation :
#   - départements d'outre-mer exclus ;
#   - codes commune nettoyés : la livraison SQL les écrit entre accolades ({33125}), la préparation les attend
#     sans (33125) ;
#   - filtre_perimetre du bloc 2 de 01_preparation/preparation.py (vente d'un seul logement ancien, une seule
#     commune), importé de ce fichier : une seule définition des règles.
# Écrit un Parquet de même contenu que dvf_plus_residentiel.parquet (mêmes colonnes, hors géométries et listes
# d'identifiants que l'extraction n'a pas gardées), sans rien modifier du pipeline : le fichier est une sortie à
# part, que la préparation ne lit pas.
# Lancement, depuis la racine du projet :
#   python 04_outils/filtrer_dvf_plus.py data/raw/dvf_plus_maj/dvf_plus_assemble.csv
#   (option --sortie pour choisir le fichier de sortie ; par défaut data/processed/dvf_plus_residentiel_<année>.parquet)
# Le script refuse d'écraser un fichier existant.

import argparse
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RACINE), str(RACINE / "01_preparation")]
from config import DATA_OUT                          # noqa: E402
from preparation import filtre_perimetre             # noqa: E402

COLONNES_REQUISES = ["anneemut", "moismut", "coddep", "libnatmut", "nbcomm", "l_codinsee",
                     "libtypbien", "nblocmai", "nblocapt", "nblocact"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", type=Path, help="dvf_plus_assemble.csv, produit par extraire_dvf_plus.py assembler")
parser.add_argument("--sortie", type=Path, help="fichier Parquet à écrire")
args = parser.parse_args()

assert args.csv.exists(), f"fichier introuvable : {args.csv}"
types_texte = {"coddep": str, "l_codinsee": str}
morceaux, n_brut, n_hors_dom = [], 0, 0
for bloc in pd.read_csv(args.csv, sep="|", chunksize=200_000, dtype=types_texte, low_memory=False):
    manquantes = [c for c in COLONNES_REQUISES if c not in bloc.columns]
    assert not manquantes, f"colonnes absentes du fichier : {manquantes}"
    n_brut += len(bloc)
    bloc = bloc[~bloc["coddep"].fillna("").str.startswith("97")].copy()          # DOM exclus
    n_hors_dom += len(bloc)
    bloc["l_codinsee"] = bloc["l_codinsee"].str.strip("{}")                       # format SQL : {33125}
    morceaux.append(filtre_perimetre(bloc))
pop = pd.concat(morceaux, ignore_index=True)

annees = sorted(int(a) for a in pop["anneemut"].unique())
sortie = args.sortie or DATA_OUT / ("dvf_plus_residentiel_" + "_".join(str(a) for a in annees) + ".parquet")
if sortie.exists():
    sys.exit(f"{sortie} existe déjà : supprime-le ou renomme-le avant de relancer (pas d'écrasement).")
assert len(pop) > 0, "aucune vente retenue : fichier vide ou colonnes inattendues"
assert pop["idmutation"].is_unique, "identifiants de mutation en double"
assert pop["l_codinsee"].str.fullmatch(r"\d[\dAB]\d{3}").all(), "code commune au format inattendu après nettoyage"

sortie.parent.mkdir(parents=True, exist_ok=True)
pop.to_parquet(sortie, index=False)
print(f"Mutations lues : {n_brut} | hors DOM : {n_hors_dom} | ventes d'un seul logement ancien retenues : {len(pop)}")
print(f"Années : {annees} | mois : {sorted(int(m) for m in pop['moismut'].unique())}")
print(pop.groupby("libtypbien").size().to_string())
print(f"Écrit : {sortie}")
