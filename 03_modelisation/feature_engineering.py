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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_OUT

# Chargement des tables de référence communes
cible = pd.read_parquet(DATA_OUT / "cible_sbv_annee.parquet")
liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet")


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
variables_delinq = ["annee_faits", "insee_pop", "insee_log", "part_indicateurs_estimes"] + ["nb_" + f for f in FAMILLES_DELINQ]
df_fe_delinq = fe_delinq[cles_delinq + variables_delinq]

# 1.5 Rattachement aux sous-bassins
com_fe_delinq = pd.merge(liaison, df_fe_delinq, left_on="CODGEO_INI", right_on="code_commune").drop(columns=["code_commune"])

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

sbv_fe_delinq = taux_lisses(com_fe_delinq, maille="sous_bassin_de_vie", familles=list(FAMILLES_DELINQ.keys()))

# 1.6 Jointure à la cible (Left join)
df_final_delinq = pd.merge(cible, sbv_fe_delinq, on=["sous_bassin_de_vie", "annee"], how="left")
assert len(df_final_delinq) == len(cible), "Erreur de granulométrie sur la délinquance"

df_final_delinq.to_parquet(DATA_OUT / "fe_cible_delinquance_sbv.parquet", index=False)
print("-> Exporté : fe_cible_delinquance_sbv.parquet")


# ==============================================================================
# SECTION 2 : FEATURE ENGINEERING - LOVAC (LOGEMENTS VACANTS)
# ==============================================================================
print("--- [2/6] TRAITEMENT : LOVAC ---")

# 2.1 Chargement des données
df_lovac = pd.read_parquet(DATA_OUT / "lovac_commune_annee.parquet")
df_logement_p22 = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet", columns=["code_commune", "P22_LOG"])

# 2.2 Préparation et filtre sur la photo fixe (Millésime 2023)
MILLESIME_PHOTO_LOVAC = 2023
fe_lovac = df_lovac[df_lovac["millesime"] == MILLESIME_PHOTO_LOVAC].copy()

# 2.3 Jointure pour récupérer P22_LOG et rattachement aux sous-bassins
fe_lovac = pd.merge(fe_lovac, df_logement_p22, on="code_commune", how="left")
com_fe_lovac = pd.merge(liaison, fe_lovac, left_on="CODGEO_INI", right_on="code_commune").drop(columns=["code_commune"])

# 2.4 Calcul de la moyenne pondérée par P22_LOG à l'échelle du sous-bassin
com_fe_lovac["vacance_complete_x_p22"] = com_fe_lovac["taux_vacance_complete"] * com_fe_lovac["P22_LOG"]
com_fe_lovac["impute_x_p22"] = com_fe_lovac["taux_impute"] * com_fe_lovac["P22_LOG"]

sbv_lovac = com_fe_lovac.groupby("sous_bassin_de_vie").agg(
    somme_vacance_complete=("vacance_complete_x_p22", "sum"),
    somme_impute=("impute_x_p22", "sum"),
    total_p22_log=("P22_LOG", "sum")
).reset_index()

sbv_lovac["taux_vacance_complete"] = sbv_lovac["somme_vacance_complete"] / sbv_lovac["total_p22_log"]
sbv_lovac["taux_impute"] = sbv_lovac["somme_impute"] / sbv_lovac["total_p22_log"]

sbv_lovac = sbv_lovac[["sous_bassin_de_vie", "taux_vacance_complete", "taux_impute"]]

# 2.5 Jointure à la cible
df_final_lovac = pd.merge(cible, sbv_lovac, on="sous_bassin_de_vie", how="left")
assert len(df_final_lovac) == len(cible), "Erreur de granulométrie sur LOVAC"

df_final_lovac.to_parquet(DATA_OUT / "fe_cible_vacance_sbv.parquet", index=False)
print("-> Exporté : fe_cible_vacance_sbv.parquet")

# ==============================================================================
# SECTION 3 : FEATURE ENGINEERING - EQUIPEMENTS (BPE)
# ==============================================================================
print("--- [3/6] TRAITEMENT : EQUIPEMENTS (BPE) ---")

# 3.1 Chargement
equip = pd.read_parquet(DATA_OUT / "equipements_commune_2025.parquet")

# 3.2 Rattachement et agrégation par sous-bassin
com_fe_equip = pd.merge(equip, liaison, left_on="code_commune", right_on="CODGEO_INI")

# Exclusion des clés de métadonnées / jointures et du millésime
cols_a_exclure = ["code_commune", "CODGEO_INI", "CODGEO_2026", "sous_bassin_de_vie", "millesime_bpe"]
cols_equip = [c for c in equip.columns if c not in cols_a_exclure]

sbv_equip = com_fe_equip.groupby("sous_bassin_de_vie")[cols_equip].sum().reset_index()

# Conservation du millésime (valeur unique / première valeur disponible)
if "millesime_bpe" in equip.columns:
    millesime_val = equip["millesime_bpe"].iloc[0]
    sbv_equip["millesime_bpe"] = millesime_val

# 3.3 Jointure à la cible
df_final_equip = pd.merge(cible, sbv_equip, on="sous_bassin_de_vie", how="left")
assert len(df_final_equip) == len(cible), "Erreur de granulométrie sur les équipements"

# Imputation par 0 sur les comptes d'équipements
cols_nb_equip = [c for c in df_final_equip.columns if c.startswith("nb_equip_")]
df_final_equip[cols_nb_equip] = df_final_equip[cols_nb_equip].fillna(0)"

df_final_equip.to_parquet(DATA_OUT / "fe_cible_equipement_sbv.parquet", index=False)
print("-> Exporté : fe_cible_equipement_sbv.parquet")


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

# 4.3 Rattachement et agrégation des effectifs par sous-bassin
com_fe_logement = pd.merge(fe_logement, liaison, left_on="code_commune", right_on="CODGEO_INI").drop(columns=["code_commune"])
sbv_logement = com_fe_logement.groupby("sous_bassin_de_vie")[EFFECTIFS_LOGEMENT].sum()

# 4.4 Calcul des 12 indicateurs à l'échelle du sous-bassin
def indicateurs_logement(e):
    """Calcule les 12 indicateurs à partir d'effectifs additionnés à une maille."""
    e = e.copy()
    for d in ["P22_LOG", "P22_RP", "P22_MEN"]:
        e[d] = e[d].where(e[d] > 0)   # dénominateur nul -> valeur vide au lieu d'un infini

    r = pd.DataFrame(index=e.index)
    r["taux_vacance"] = e["P22_LOGVAC"] / e["P22_LOG"] * 100
    r["part_rsec"] = e["P22_RSECOCC"] / e["P22_LOG"] * 100
    r["part_apparts"] = e["P22_APPART"] / e["P22_LOG"] * 100
    r["log_nb_logements"] = np.log10(e["P22_LOG"])
    r["part_moins40m2"] = (e["P22_RP_M30M2"] + e["P22_RP_3040M2"]) / e["P22_RP"] * 100
    r["part_avant1946"] = (e["P22_RP_ACH1919"] + e["P22_RP_ACH1945"]) / e["P22_RP"] * 100
    r["part_2006_2019"] = e["P22_RP_ACH2019"] / e["P22_RP"] * 100
    r["anciennete_moyenne"] = e["P22_ANEM_RP"] / e["P22_RP"]
    r["taille_menage"] = e["P22_NPER_RP"] / e["P22_RP"]
    r["part_sans_voiture"] = 100 - e["P22_RP_VOIT1P"] / e["P22_MEN"] * 100
    r["part_parking"] = e["P22_RP_GARL"] / e["P22_MEN"] * 100
    r["part_suroccupes"] = (e["C22_RP_SUROCC_MOD"] + e["C22_RP_SUROCC_ACC"]) / e["P22_RP"] * 100
    return r

df_ind_logement = indicateurs_logement(sbv_logement).reset_index()

# 4.5 Jointure à la cible
df_final_logement = pd.merge(cible, df_ind_logement, on="sous_bassin_de_vie", how="left")
assert len(df_final_logement) == len(cible), "Erreur de granulométrie sur le parc de logements"

df_final_logement.to_parquet(DATA_OUT / "fe_cible_logement_sbv.parquet", index=False)
print("-> Exporté : fe_cible_logement_sbv.parquet")


# ==============================================================================
# SECTION 5 : FEATURE ENGINEERING - FILOSOFI (REVENUS ET PAUVRETE)
# ==============================================================================
print("--- [5/6] TRAITEMENT : FILOSOFI ---")

# 5.1 Chargement
df_filosofi = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet")

# 5.2 Filtre millésime 2021 et exclusion des agrégats PLM (Paris, Lyon, Marseille)
MILLESIME_FILOSOFI = 2021

fe_filosofi = df_filosofi.loc[
    df_filosofi["annee"] == MILLESIME_FILOSOFI,
    ["code_commune", "niveau_vie_median", "NBMEN"]
].copy()

# Paris (75056), Lyon (69123), Marseille (13055) : on retire les villes mères
CODES_PLM_DOUBLONS = ["75056", "69123", "13055"]
fe_filosofi = fe_filosofi[~fe_filosofi["code_commune"].isin(CODES_PLM_DOUBLONS)]
fe_filosofi = fe_filosofi.rename(columns={"NBMEN": "nb_menages_fiscaux"})

# 5.3 Rattachement aux sous-bassins
com_fe_filosofi = pd.merge(liaison, fe_filosofi, left_on="CODGEO_INI", right_on="code_commune").drop(columns=["code_commune"])

# 5.4 Agrégation au sous-bassin (Moyenne des médianes pondérée par les ménages)
com_fe_filosofi["masse_niveau_vie"] = com_fe_filosofi["niveau_vie_median"] * com_fe_filosofi["nb_menages_fiscaux"]

sbv_filosofi = com_fe_filosofi.groupby("sous_bassin_de_vie").agg(
    masse_niveau_vie=("masse_niveau_vie", "sum"),
    nb_menages_fiscaux=("nb_menages_fiscaux", "sum")
).reset_index()

sbv_filosofi["niveau_vie_median"] = sbv_filosofi["masse_niveau_vie"] / sbv_filosofi["nb_menages_fiscaux"]
sbv_filosofi["millesime_filosofi"] = MILLESIME_FILOSOFI

sbv_filosofi = sbv_filosofi[["sous_bassin_de_vie", "niveau_vie_median", "nb_menages_fiscaux", "millesime_filosofi"]]

# 5.5 Jointure à la cible
df_final_filosofi = pd.merge(cible, sbv_filosofi, on="sous_bassin_de_vie", how="left")
assert len(df_final_filosofi) == len(cible), "Erreur de granulométrie sur Filosofi"

df_final_filosofi.to_parquet(DATA_OUT / "fe_cible_filosofi_sbv.parquet", index=False)
print("-> Exporté : fe_cible_filosofi_sbv.parquet")


# ==============================================================================
# SECTION 6 : FEATURE ENGINEERING - DVF+ (TRANSACTIONS IMMOBILIERES)
# ==============================================================================
print("--- [6/6] TRAITEMENT : DVF+ ---")

# 6.1 Chargement des données
df_dvf = pd.read_parquet(DATA_OUT / "cible_commune_annee.parquet")
cles_dvf = ["code_commune", "annee"]

# 6.2 Sélection et renommage des variables
fe_dvf = df_dvf[["code_commune", "annee", "n_maison", "n_appart"]].rename(
    columns={
        "n_maison": "n_ventes_maison_n1",
        "n_appart": "n_ventes_appart_n1"
    }
)

# 6.3 Décalage temporel (Ventes de N-1 -> Prix de N)
fe_dvf["annee_ventes"] = df_dvf["annee"]
fe_dvf["annee"] = df_dvf["annee"] + 1

variables_dvf = ["annee_ventes", "n_ventes_maison_n1", "n_ventes_appart_n1"]
df_fe_dvf = fe_dvf[cles_dvf + variables_dvf]

# 6.4 Rattachement aux sous-bassins
com_fe_dvf = pd.merge(liaison, df_fe_dvf, left_on="CODGEO_INI", right_on="code_commune").drop(columns=["code_commune"])

# 6.5 Agrégation par sous-bassin x année
# On somme les volumes de ventes par sous-bassin et on conserve l'année des ventes (min/max identiques)
sbv_dvf = com_fe_dvf.groupby(["sous_bassin_de_vie", "annee"]).agg(
    annee_ventes=("annee_ventes", "min"),
    n_ventes_maison_n1=("n_ventes_maison_n1", "sum"),
    n_ventes_appart_n1=("n_ventes_appart_n1", "sum")
).reset_index()

# 6.6 Jointure à la cible et imputation des valeurs manquantes
df_final_dvf = pd.merge(cible, sbv_dvf, on=["sous_bassin_de_vie", "annee"], how="left")

cols_ventes = [c for c in df_final_dvf.columns if c.startswith("n_ventes_")]
df_final_dvf[cols_ventes] = df_final_dvf[cols_ventes].fillna(0)

assert len(df_final_dvf) == len(cible), "Erreur de granulométrie sur DVF+"

df_final_dvf.to_parquet(DATA_OUT / "fe_cible_dvf_sbv.parquet", index=False)
print("-> Exporté : fe_cible_dvf_sbv.parquet")