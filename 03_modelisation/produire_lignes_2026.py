# =============================================================================
# produire_lignes_2026.py : lignes d'entrée de H7v4 pour 2026 (bloc A)
# =============================================================================
# Produit lignes_2026.parquet : une ligne par (sous-bassin, type de bien) ayant au moins 20 ventes en 2025,
# avec toutes les variables explicatives de H7v4 (sauf le bloc du bien, saisi par l'utilisateur).
# Prérequis : construire_lignes(2025) validée par controle_lignes_2025.py (0 écart), ventes_test_2025.parquet,
# fe_cible_toutes_sources_sbv.parquet, taux de crédit de décembre 2025 dans taux_credit_mois.parquet.
# Lancement (depuis 03_modelisation) :  python produire_lignes_2026.py

import pandas as pd

import commun
from lignes_annee import construire_lignes, COLONNES

ANNEE = 2026
SORTIE = commun.DATA_OUT / "lignes_2026.parquet"

lignes = construire_lignes(ANNEE, exclure_incomplets=True)

# Contrôles : aucune case vide hors evol_prix_n2_n1 (H7v4 la remplace par 0 + indicatrice)
obligatoires = [c for c in COLONNES if c != "evol_prix_n2_n1"]
assert lignes[obligatoires].notna().all().all(), "case vide dans une variable obligatoire"
assert (lignes["annee"] == ANNEE).all()
assert not lignes.index.duplicated().any(), "doublon (sous-bassin, type de bien)"

print("\nLignes 2026 :", len(lignes))
print(lignes.groupby("type_bien").size().to_string())
print("evol_prix_n2_n1 vide :", int(lignes["evol_prix_n2_n1"].isna().sum()))
print("trimestre_national = 1 :", int(lignes["trimestre_national"].sum()))
print("\nRésumé des variables :")
print(lignes[COLONNES].describe().T[["min", "50%", "max"]].round(2).to_string())

if SORTIE.exists():
    raise SystemExit(f"{SORTIE} existe déjà : supprime-le ou renomme-le avant de relancer (pas d'écrasement).")
lignes.to_parquet(SORTIE)
print("\nÉcrit :", SORTIE)
