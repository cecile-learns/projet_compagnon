# =============================================================================
# 04_outils/preparer_eda_app.py : fichiers lus par la page Exploration des données (Christian)
# =============================================================================
# L'application en ligne n'a pas accès à data/processed : ce script copie, réduits aux colonnes utiles,
# les fichiers que la page EDA lit, dans 05_application/donnees/.
#   eda_territoires.parquet : prix médian et nombre de ventes par (année, sous-bassin, type de bien),
#                             d'après la table de modélisation (sous-bassins d'au moins 20 ventes) ;
#   eda_seuil.parquet       : part des territoires et des ventes gardés selon le seuil de ventes,
#                             d'après fe_cible_toutes_sources_sbv.parquet (avant seuil).
# Lancement (depuis 04_outils) :  python preparer_eda_app.py

from pathlib import Path

import pandas as pd

from commun import DATA_OUT

DONNEES_APP = Path(__file__).resolve().parents[1] / "05_application" / "donnees"
SEUILS = list(range(5, 101, 5))
TYPES = {"maison": "n_maison", "appartement": "n_appart"}
CLES = ["annee", "sous_bassin_de_vie", "type_bien"]

DONNEES_APP.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Prix et ventes par territoire (table de modélisation)
# -----------------------------------------------------------------------------
table = pd.read_parquet(DATA_OUT / "table_modelisation_bv.parquet")
# « annee » peut être à la fois dans l'index et dans les colonnes : l'index fait foi
table = table.drop(columns=[c for c in table.columns if c in table.index.names]).reset_index()
manquantes = sorted(set(CLES + ["prix", "n_ventes"]) - set(table.columns))
assert not manquantes, f"colonnes absentes de table_modelisation_bv : {manquantes}"
territoires = table[CLES + ["prix", "n_ventes"]].copy()
territoires["sous_bassin_de_vie"] = territoires["sous_bassin_de_vie"].astype(str)
assert not territoires.duplicated(CLES).any(), "doublons année x sous-bassin x type"
territoires.to_parquet(DONNEES_APP / "eda_territoires.parquet", index=False)
print("eda_territoires.parquet :", territoires.shape,
      "| années", int(territoires["annee"].min()), "à", int(territoires["annee"].max()))

# -----------------------------------------------------------------------------
# 2. Effet du seuil de ventes (cible avant seuil)
# -----------------------------------------------------------------------------
cible = pd.read_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv.parquet")
attendues = ["sous_bassin_de_vie", "annee"] + list(TYPES.values())
absentes = sorted(set(attendues) - set(cible.columns))
assert not absentes, f"colonnes absentes de fe_cible_toutes_sources_sbv : {absentes}"
lignes = []
for type_bien, colonne in TYPES.items():
    n = cible[["annee", colonne]].rename(columns={colonne: "n"}).assign(n=lambda d: d["n"].fillna(0))
    for annee, g in n.groupby("annee"):
        presents = g[g["n"] > 0]
        for seuil in SEUILS:
            gardes = presents[presents["n"] >= seuil]
            lignes.append({"annee": int(annee), "type_bien": type_bien, "seuil": seuil,
                           "territoires": len(presents), "territoires_gardes": len(gardes),
                           "ventes": float(presents["n"].sum()), "ventes_gardees": float(gardes["n"].sum())})
seuil = pd.DataFrame(lignes)
seuil.to_parquet(DONNEES_APP / "eda_seuil.parquet", index=False)
print("eda_seuil.parquet :", seuil.shape)
print("\nContrôle (seuil de 20 ventes, dernière année) : territoires gardés / total, ventes gardées")
derniere = seuil[(seuil["seuil"] == 20) & (seuil["annee"] == seuil["annee"].max())]
print(derniere.to_string(index=False))
