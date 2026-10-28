"""
Feature Engineering - Traitement des données et création des features
======================================================================
Auteurs      : Cécile & Théophile
Date         : 28/10/2026
Description  : Fichier de feature engineering regroupant les traitements
               pour la Délinquance, LOVAC, les Équipements (BPE),
               le Parc de logements (Recensement INSEE), Filosofi et DVF+.
"""

# ==============================================================================
# IMPORTS ET CONFIGURATION GENERALES
# ==============================================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ajout du dossier parent au path Python pour importer la configuration
sys.path.insert(0, str(Path.cwd().parent))
from config import DATA_RAW, DATA_OUT


# ==============================================================================
# SECTION 1 : FEATURE ENGINEERING - DELINQUANCE
# ==============================================================================
print("--- [1/6] TRAITEMENT : DELINQUANCE ---")

# 1.1 Chargement des données
df_delinq = pd.read_parquet(DATA_OUT / "delinquance_commune_annee.parquet")
cles_delinq = ["code_commune", "annee"]

# 1.2 Regroupement en familles d'indicateurs
FAMILLES_DELINQ = {
    "vols_vehicules":            ["nb_vols_de_vehicule", "nb_vols_dans_les_vehicules"],
    "vols_violents":             ["nb_vols_avec_armes", "nb_vols_violents_sans_arme"],
    "vols_sans_violence":        ["nb_vols_sans_violence_contre_des_personnes"],
    "cambriolages":              ["nb_cambriolages_de_logement"],
    "destructions":              ["nb_destructions_et_degradations_volontaires"],
    "stupefiants":               ["nb_usage_de_stupefiants", "nb_trafic_de_stupefiants"],
    "violences_intrafamiliales": ["nb_violences_physiques_intrafamiliales"],
    "violences_sexuelles":       ["nb_violences_sexuelles"],
}

fe_delinq = df_delinq[cles_delinq + ["insee_pop", "insee_log"]].copy()
for famille, colonnes in FAMILLES_DELINQ.items():
    fe_delinq["nb_" + famille] = df_delinq[colonnes].sum(axis=1, min_count=1)

# 1.3 Décalage temporal (Faits en N -> Prix expliqués en N+1)
fe_delinq["annee_faits"] = df_delinq["annee"]
fe_delinq["annee"] = df_delinq["annee"] + 1

# 1.4 Indicateur de fiabilité
fe_delinq["part_indicateurs_estimes"] = df_delinq["n_indicateurs_estimes"] / 15 * 100

# 1.5 Fusion avec les tables cibles et export
variables_delinq = ["annee_faits", "insee_pop", "insee_log", "part_indicateurs_estimes"] + ["nb_" + f for f in FAMILLES_DELINQ]
df_fe_delinq = fe_delinq[cles_delinq + variables_delinq]

def taux_lisses(table, maille, familles, fenetre=3):
    """
    Calcule les taux annuels et lissés sur une fenêtre glissante par maille/année.
    """
    table = table.copy()
    table["estimes_x_pop"] = table["part_indicateurs_estimes"] * table["insee_pop"]
    colonnes = ["nb_" + f for f in familles] + ["insee_pop", "insee_log", "estimes_x_pop"]
    
    agr = table.groupby([maille, "annee"])[colonnes].sum().reset_index()
    agr = agr.sort_values([maille, "annee"]).reset_index(drop=True)
    
    agr["part_indicateurs_estimes"] = agr["estimes_x_pop"] / agr["insee_pop"]
    
    for f in familles:
        denom = "insee_log" if f == "cambriolages" else "insee_pop"
        faits_n = agr.groupby(maille)["nb_" + f].rolling(fenetre, min_periods=fenetre).sum().reset_index(level=0, drop=True)
        denom_n = agr.groupby(maille)[denom].rolling(fenetre, min_periods=fenetre).sum().reset_index(level=0, drop=True)
        agr["taux1_" + f] = agr["nb_" + f] / agr[denom] * 1000
        agr["taux3_" + f] = faits_n / denom_n * 1000
        
    return agr.drop(columns="estimes_x_pop")

cible = pd.read_parquet(DATA_OUT / "cible_sbv_annee.parquet")
liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet")

com_fe_delinq = pd.merge(liaison, df_fe_delinq, left_on="CODGEO_INI", right_on="code_commune").drop(columns="code_commune")
df_final_delinq = pd.merge(com_fe_delinq, cible, on=["sous_bassin_de_vie", "annee"])

df_final_delinq.to_parquet(DATA_OUT / "fe_cible_delinquance_commune.parquet", index=False)
print("-> Exporté : fe_cible_delinquance_commune.parquet")


# ==============================================================================
# SECTION 2 : FEATURE ENGINEERING - LOVAC (LOGEMENTS VACANTS)
# ==============================================================================
print("--- [2/6] TRAITEMENT : LOVAC ---")

# 2.1 Chargement des données
df_lovac = pd.read_parquet(DATA_OUT / "lovac_commune_annee.parquet")

# 2.2 Préparation et renommage
fe_lovac = df_lovac.rename(columns={"annee_vente_associee": "annee"})
fe_lovac = fe_lovac[["code_commune", "annee", "millesime", "taux_vacance_complete", "taux_impute", "apres_rupture"]].copy()

# 2.3 Extraction de la photo fixe (Millésime 2023)
MILLESIME_PHOTO_LOVAC = 2023
df_fe_lovac = fe_lovac.loc[
    fe_lovac["millesime"] == MILLESIME_PHOTO_LOVAC,
    ["code_commune", "millesime", "taux_vacance_complete", "taux_impute"]
]

# 2.4 Fusion et export
com_fe_lovac = pd.merge(liaison, df_fe_lovac, left_on="CODGEO_INI", right_on="code_commune").drop(columns="code_commune")
df_final_lovac = pd.merge(com_fe_lovac, cible, on=["sous_bassin_de_vie"])

df_final_lovac.to_parquet(DATA_OUT / "fe_cible_vacance_commune.parquet", index=False)
print("-> Exporté : fe_cible_vacance_commune.parquet")


# ==============================================================================
# SECTION 3 : FEATURE ENGINEERING - EQUIPEMENTS (BPE)
# ==============================================================================
print("--- [3/6] TRAITEMENT : EQUIPEMENTS (BPE) ---")

# 3.1 Chargement
equip = pd.read_parquet(DATA_OUT / "equipements_commune_2025.parquet")

# 3.2 Fusion et export avec la cible
com_fe_equip = pd.merge(equip, liaison, left_on="code_commune", right_on="CODGEO_INI")
df_final_equip = pd.merge(com_fe_equip, cible, on=["sous_bassin_de_vie"]).drop(columns=["CODGEO_INI", "CODGEO_2026"])

df_final_equip.to_parquet(DATA_OUT / "fe_cible_equipement_communes.parquet", index=False)
print("-> Exporté : fe_cible_equipement_communes.parquet")


# ==============================================================================
# SECTION 4 : FEATURE ENGINEERING - PARC DE LOGEMENTS (RECENSEMENT)
# ==============================================================================
print("--- [4/6] TRAITEMENT : PARC DE LOGEMENTS (INSEE) ---")

# 4.1 Chargement
df_logement = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet")

# 4.2 Sélection des variables d'effectifs
INDICATEURS_LOGEMENT = {
    "taux_vacance":       ["P22_LOGVAC", "P22_LOG"],
    "part_rsec":          ["P22_RSECOCC", "P22_LOG"],
    "part_apparts":       ["P22_APPART", "P22_LOG"],
    "log_nb_logements":   ["P22_LOG"],
    "part_moins40m2":     ["P22_RP_M30M2", "P22_RP_3040M2", "P22_RP"],
    "part_avant1946":     ["P22_RP_ACH1919", "P22_RP_ACH1945", "P22_RP"],
    "part_2006_2019":     ["P22_RP_ACH2019", "P22_RP"],
    "anciennete_moyenne": ["P22_ANEM_RP", "P22_RP"],
    "taille_menage":      ["P22_NPER_RP", "P22_RP"],
    "part_sans_voiture":  ["P22_RP_VOIT1P", "P22_MEN"],
    "part_parking":       ["P22_RP_GARL", "P22_MEN"],
    "part_suroccupes":    ["C22_RP_SUROCC_MOD", "C22_RP_SUROCC_ACC", "P22_RP"],
}
EFFECTIFS_LOGEMENT = sorted({c for cols in INDICATEURS_LOGEMENT.values() for c in cols})

fe_logement = df_logement[["code_commune", "annee_rp"] + EFFECTIFS_LOGEMENT].copy()

# 4.3 Fusion et export
com_fe_logement = pd.merge(fe_logement, liaison, left_on="code_commune", right_on="CODGEO_INI").drop(columns="code_commune")
df_final=pd.merge(com_fe,cible, left_on=["sous_bassin_de_vie", "annee_rp"], right_on=["sous_bassin_de_vie", "annee"])
df_final.to_parquet(DATA_OUT / "fe_cible_logement_commune.parquet", index=False)
print("-> Exporté : fe_cible_logement_communes.parquet")