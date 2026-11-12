# =============================================================================
# reconstituer_ventes_h7.py : régénère les deux fichiers du bloc 3 bis de preparation.py
#   - ventes_apprentissage.parquet (échantillon 2014-2024 après règles d'exclusion)
#   - ventes_test_2025.parquet     (tirage 2025, graine 2025) + ids_ventes_test_2025.csv
# Le code du tirage et de preparer_ventes est COPIÉ de preparation.py (lignes 399-454), sans
# modification : mêmes graines, mêmes règles. À placer dans 01_preparation/ et à lancer depuis la
# racine du projet :  python 01_preparation/reconstituer_ventes_h7.py
# Entrées (dans data/processed) : dvf_plus_residentiel.parquet, dvf_plus_echantillon_apprentissage.parquet,
# liaison_bv_scindes.parquet.  Ne pas versionner ce fichier : c'est un dépannage.
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_OUT

FICHIER_POP = DATA_OUT / "dvf_plus_residentiel.parquet"
SORTIES = [DATA_OUT / "ventes_apprentissage.parquet", DATA_OUT / "ventes_test_2025.parquet",
           DATA_OUT / "ids_ventes_test_2025.csv"]
for s in SORTIES:
    assert not s.exists(), f"{s.name} existe déjà : on ne l'écrase pas (le supprimer à la main si voulu)"

taux, plancher = 0.10, 300

liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet")
COLONNES_VENTES = ["idmutation", "anneemut", "moismut", "coddep", "code_commune", "BV2022",
                   "sous_bassin_de_vie", "type_bien", "prix_m2", "sbati", "sterr",
                   "nblocdep", "nb_pieces", "geompar_x", "geompar_y", "w_h"]


def preparer_ventes(d):
    """Règles d'exclusion, nombre de pièces, type de bien et rattachement territorial."""
    d = d.copy()
    d["prix_m2"] = d["valeurfonc"] / d["sbati"]
    d = d[d["valeurfonc"].notna() & (d["valeurfonc"] > 0) & d["prix_m2"].between(50, 50000)].copy()
    d["type_bien"] = d["libtypbien"].map({"UNE MAISON": "maison", "UN APPARTEMENT": "appartement"})
    pieces = (d[[f"nbmai{i}pp" for i in range(1, 6)]].to_numpy()
              + d[[f"nbapt{i}pp" for i in range(1, 6)]].to_numpy())
    d["nb_pieces"] = pieces.argmax(axis=1) + 1
    d = d.merge(liaison[["CODGEO_INI", "CODGEO_2026", "BV2022", "sous_bassin_de_vie"]],
                left_on="l_codinsee", right_on="CODGEO_INI", how="left", validate="many_to_one")
    arrondissement = d["l_codinsee"].str.match(r"^(751\d\d|6938\d|132\d\d)$")
    d["code_commune"] = d["CODGEO_2026"].where(~arrondissement, d["l_codinsee"])
    return d[COLONNES_VENTES]


# --- Apprentissage (2014-2024) : l'échantillon du bloc 3, déjà tiré et enregistré ---
ech = pd.read_parquet(DATA_OUT / "dvf_plus_echantillon_apprentissage.parquet")
ventes_app = preparer_ventes(ech)

# --- Test 2025 : tirage unique (graine 2025) ---
strates = (pd.read_parquet(FICHIER_POP, columns=["coddep", "anneemut"])
             .value_counts().rename("N_h").reset_index())
test = strates[strates["anneemut"] == 2025].copy()
test["n_h"] = np.minimum(test["N_h"], np.maximum(plancher, np.round(test["N_h"] * taux))).astype(int)
test["w_h"] = test["N_h"] / test["n_h"]
rng_test = np.random.default_rng(2025)
morceaux = []
for dep in sorted(test["coddep"].unique()):
    d = pd.read_parquet(FICHIER_POP, filters=[("coddep", "==", dep), ("anneemut", "==", 2025)])
    d = d.merge(test[["coddep", "anneemut", "n_h", "w_h"]], on=["coddep", "anneemut"])
    d["alea"] = rng_test.random(len(d))
    d["rang"] = d["alea"].rank(method="first")
    morceaux.append(d[d["rang"] <= d["n_h"]].drop(columns=["alea", "rang", "n_h"]))
ventes_test = preparer_ventes(pd.concat(morceaux, ignore_index=True))

# --- Contrôles, puis écriture ---
for nom, v in [("apprentissage", ventes_app), ("test 2025", ventes_test)]:
    print(f"{nom} : {len(v)} ventes | sans sous-bassin : {v['sous_bassin_de_vie'].isna().sum()}"
          f" | idmutation uniques : {v['idmutation'].is_unique} | années : {sorted(v['anneemut'].unique())}")
assert sorted(ventes_test["anneemut"].unique()) == [2025]
ventes_app.to_parquet(SORTIES[0], index=False)
ventes_test.to_parquet(SORTIES[1], index=False)
ventes_test[["idmutation"]].to_csv(SORTIES[2], index=False)
print("Fichiers écrits :", [s.name for s in SORTIES])
print("À comparer avec Cécile : nombre de ventes test 2025 =", len(ventes_test))
