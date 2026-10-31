"""
Feature engineering - une table par source, jointe à la cible au sous-bassin de vie
===================================================================================
Auteurs      : Cécile, Théophile & Gabrielle (jointures aux sous-bassins)
Date         : 29/10/2026
Description  : délinquance, LOVAC, équipements (BPE 2025), densité de population,
               parc de logements (recensement 2022), Filosofi et DVF+ (volume de ventes).
               Pour chaque source : préparation à la commune, rattachement aux
               sous-bassins (liaison_bv_scindes), agrégation, jointure à gauche
               sur la cible (cible_sbv_annee), export fe_cible_<source>_sbv.parquet.
               Constats et justifications : notebook feature_engineering.ipynb
               (même code, mêmes noms de fichiers).
               Taux des crédits : non retenus dans le modèle annuel (voir notebook).
"""

# ==============================================================================
# IMPORTS ET CONFIGURATION
# ==============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# config.py se trouve à la racine du projet, dossier parent de ce fichier
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_OUT

# Tables de référence communes à toutes les sources
cible = pd.read_parquet(DATA_OUT / "cible_sbv_annee.parquet")
liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet")


# ==============================================================================
# FONCTIONS
# ==============================================================================
def taux_lisses(table, maille, familles, fenetre=3):
    """Taux annuels (taux1_) et lissés sur `fenetre` années (taux3_) par maille x année,
    pour 1 000 habitants (1 000 logements pour les cambriolages).
    Faits et dénominateurs sont additionnés avant le calcul des taux.
    Premières années : la fenêtre grandit (1 an, puis 2 ans, puis `fenetre` ans)."""
    table = table.copy()
    table["estimes_x_pop"] = table["part_indicateurs_estimes"] * table["insee_pop"]
    colonnes = ["nb_" + f for f in familles] + ["insee_pop", "insee_log", "estimes_x_pop"]
    # 1. Faits et dénominateurs additionnés par maille et par année
    agr = table.groupby([maille, "annee"])[colonnes].sum().reset_index()
    agr = agr.sort_values([maille, "annee"]).reset_index(drop=True)
    # 2. Fiabilité : moyenne pondérée par la population
    agr["part_indicateurs_estimes"] = agr["estimes_x_pop"] / agr["insee_pop"]
    # 3. Taux annuel et taux lissé (années passées uniquement ; fenêtre qui grandit au début)
    for f in familles:
        denom = "insee_log" if f == "cambriolages" else "insee_pop"
        faits_n = agr.groupby(maille)["nb_" + f].rolling(fenetre, min_periods=1).sum().reset_index(level=0, drop=True)
        denom_n = agr.groupby(maille)[denom].rolling(fenetre, min_periods=1).sum().reset_index(level=0, drop=True)
        agr["taux1_" + f] = agr["nb_" + f] / agr[denom] * 1000
        agr["taux3_" + f] = faits_n / denom_n * 1000
    return agr.drop(columns="estimes_x_pop")


def indicateurs_logement(e):
    """Calcule les 12 indicateurs à partir d'effectifs additionnés à une maille
    (mêmes formules que le bloc 7 de la préparation)."""
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


def joindre_cible(table_sbv, cles, nom_fichier):
    """Jointure à gauche sur la cible, contrôle du nombre de lignes, export."""
    final = pd.merge(cible, table_sbv, on=cles, how="left", validate="many_to_one")
    assert len(final) == len(cible), f"Nombre de lignes modifié par la jointure : {nom_fichier}"
    final.to_parquet(DATA_OUT / nom_fichier, index=False)
    print("-> Exporté :", nom_fichier, final.shape)
    return final


# ==============================================================================
# SECTION 1 : DELINQUANCE (commune x année -> sous-bassin x année)
# ==============================================================================
print("--- [1/6] DELINQUANCE ---")

df_delinq = pd.read_parquet(DATA_OUT / "delinquance_commune_annee.parquet")

# 1.1 Regroupement en familles (même nature, même unité de compte)
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
fe_delinq = df_delinq[["code_commune", "annee", "insee_pop", "insee_log"]].copy()
for famille, colonnes in FAMILLES_DELINQ.items():
    # min_count=1 : la somme reste vide si tous les indicateurs de la famille sont vides
    fe_delinq["nb_" + famille] = df_delinq[colonnes].sum(axis=1, min_count=1)

# 1.2 Décalage d'un an : faits de N -> prix de N+1
fe_delinq["annee_faits"] = df_delinq["annee"]
fe_delinq["annee"] = df_delinq["annee"] + 1

# 1.3 Fiabilité : part des 15 indicateurs estimés (secret statistique)
fe_delinq["part_indicateurs_estimes"] = df_delinq["n_indicateurs_estimes"] / 15 * 100

# 1.4 Rattachement aux sous-bassins, taux lissés sur 3 ans
com_delinq = pd.merge(liaison, fe_delinq, left_on="CODGEO_INI", right_on="code_commune")
sbv_delinq = taux_lisses(com_delinq, maille="sous_bassin_de_vie", familles=list(FAMILLES_DELINQ))
sbv_delinq["annee_faits"] = sbv_delinq["annee"] - 1

# Une seule variable par famille : le taux lissé sur 3 ans (plus stable dans les petits sous-bassins),
# sur 1 an pour les prix 2017 et 2 ans pour les prix 2018 (fenêtre qui grandit).
# Nombres de faits et dénominateurs (insee_pop, insee_log) n'ont servi qu'au calcul des taux ;
# taux annuels (taux1_) retirés
DELINQ_GARDEES = (["sous_bassin_de_vie", "annee", "annee_faits", "part_indicateurs_estimes"]
                  + ["taux3_" + f for f in FAMILLES_DELINQ])
sbv_delinq = sbv_delinq[DELINQ_GARDEES]

# 1.5 Jointure à la cible et export
joindre_cible(sbv_delinq, ["sous_bassin_de_vie", "annee"], "fe_cible_delinquance_sbv.parquet")


# ==============================================================================
# SECTION 2 : LOVAC (photo fixe, millésime 2023)
# ==============================================================================
print("--- [2/6] LOVAC ---")

df_lovac = pd.read_parquet(DATA_OUT / "lovac_commune_annee.parquet")
p22_log = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet", columns=["code_commune", "P22_LOG"])

# 2.1 Photo du millésime 2023 (situation au 1er janvier 2022), avant la rupture de série
MILLESIME_PHOTO_LOVAC = 2023
fe_lovac = df_lovac.loc[df_lovac["millesime"] == MILLESIME_PHOTO_LOVAC,
                        ["code_commune", "taux_vacance_complete", "taux_impute"]]

# 2.2 Poids : nombre de logements du recensement (parc privé inconnu pour les communes masquées)
fe_lovac = pd.merge(fe_lovac, p22_log, on="code_commune", how="left")
com_lovac = pd.merge(liaison, fe_lovac, left_on="CODGEO_INI", right_on="code_commune")

# 2.3 Moyenne pondérée par P22_LOG au sous-bassin
com_lovac["vacance_x_log"] = com_lovac["taux_vacance_complete"] * com_lovac["P22_LOG"]
com_lovac["impute_x_log"] = com_lovac["taux_impute"].astype(float) * com_lovac["P22_LOG"]
sbv_lovac = com_lovac.groupby("sous_bassin_de_vie")[["vacance_x_log", "impute_x_log", "P22_LOG"]].sum()
sbv_lovac["taux_vacance_complete"] = sbv_lovac["vacance_x_log"] / sbv_lovac["P22_LOG"]
sbv_lovac["part_logements_taux_impute"] = sbv_lovac["impute_x_log"] / sbv_lovac["P22_LOG"] * 100
sbv_lovac["millesime_lovac"] = MILLESIME_PHOTO_LOVAC
sbv_lovac = sbv_lovac[["taux_vacance_complete", "part_logements_taux_impute", "millesime_lovac"]].reset_index()

# 2.4 Jointure à la cible (même photo pour toutes les années) et export
joindre_cible(sbv_lovac, ["sous_bassin_de_vie"], "fe_cible_vacance_sbv.parquet")


# ==============================================================================
# SECTION 3 : EQUIPEMENTS (BPE 2025 par commune, photo fixe)
# ==============================================================================
print("--- [3/6] EQUIPEMENTS ---")

equip = pd.read_parquet(DATA_OUT / "equipements_commune_2025.parquet")
cols_equip = [c for c in equip.columns if c.startswith("nb_equip_")]

# 3.1 Rattachement et somme par sous-bassin
com_equip = pd.merge(liaison, equip, left_on="CODGEO_INI", right_on="code_commune")
sbv_equip = com_equip.groupby("sous_bassin_de_vie")[cols_equip].sum()

# 3.2 Population du sous-bassin : habitants de 2024 (SSMSI), l'année la plus proche de la BPE 2025
#     (sert aussi à la densité, section 3 bis)
pop_2024 = pd.read_parquet(DATA_OUT / "delinquance_commune_annee.parquet",
                           columns=["code_commune", "annee", "insee_pop"])
pop_2024 = pop_2024.loc[pop_2024["annee"] == 2024, ["code_commune", "insee_pop"]]
com_pop = pd.merge(liaison, pop_2024, left_on="CODGEO_INI", right_on="code_commune")
pop_sbv = com_pop.groupby("sous_bassin_de_vie")["insee_pop"].sum()

# 3.3 Taux pour 10 000 habitants ; sous-bassin sans équipement recensé = 0 (règle de l'EDA complémentaire)
#     Les nombres bruts ne sont pas exportés
nb_sbv = sbv_equip.reindex(pop_sbv.index, fill_value=0)
taux_equip = nb_sbv.div(pop_sbv.where(pop_sbv > 0), axis=0) * 10_000
taux_equip.columns = [c.replace("nb_equip_", "taux_equip_") for c in cols_equip]
taux_equip["millesime_bpe"] = 2025
taux_equip = taux_equip.reset_index()

# 3.4 Jointure à la cible (même photo pour toutes les années) et export
final_equip = joindre_cible(taux_equip, ["sous_bassin_de_vie"], "fe_cible_equipement_sbv.parquet")
print("Sous-bassins de la cible sans population :",
      final_equip.loc[final_equip["taux_equip_commerces"].isna(), "sous_bassin_de_vie"].nunique())


# ==============================================================================
# SECTION 3 BIS : DENSITÉ DE POPULATION (photo fixe)
# ==============================================================================
print("--- [3 bis] DENSITÉ DE POPULATION ---")

superficie = pd.read_parquet(DATA_OUT / "communes_superficie_2026.parquet")

# 3b.1 Arrondissements de Paris, Lyon, Marseille : absents du fichier des superficies.
#      Superficie de la ville répartie entre ses arrondissements au prorata de leur population 2024
#      (chaque arrondissement prend la densité de sa ville). La ville entière est retirée.
VILLES_ARRONDISSEMENTS = {"75056": r"^751\d\d$", "69123": r"^6938\d$", "13055": r"^132\d\d$"}
morceaux = [superficie[~superficie["code_commune"].isin(VILLES_ARRONDISSEMENTS)]]
for ville, motif in VILLES_ARRONDISSEMENTS.items():
    arr = pop_2024[pop_2024["code_commune"].str.match(motif)].copy()
    sup_ville = superficie.loc[superficie["code_commune"] == ville, "superficie_km2"].iloc[0]
    arr["superficie_km2"] = sup_ville * arr["insee_pop"] / arr["insee_pop"].sum()
    morceaux.append(arr[["code_commune", "superficie_km2"]])
superficie_com = pd.concat(morceaux, ignore_index=True)
assert superficie_com["code_commune"].is_unique, "Commune en double dans les superficies"
print("Arrondissements ajoutés :", len(superficie_com) - len(morceaux[0]), "(45 attendus)")

# 3b.2 Superficie additionnée au sous-bassin, densité = habitants 2024 / km²
com_sup = pd.merge(liaison, superficie_com, left_on="CODGEO_INI", right_on="code_commune")
sup_sbv = com_sup.groupby("sous_bassin_de_vie")["superficie_km2"].sum(min_count=1)
sbv_densite = pd.DataFrame({"densite_population": pop_sbv / sup_sbv.reindex(pop_sbv.index).where(lambda x: x > 0)})
sbv_densite = sbv_densite.rename_axis("sous_bassin_de_vie").reset_index()

# 3b.3 Jointure à la cible (même photo pour toutes les années) et export
final_densite = joindre_cible(sbv_densite, ["sous_bassin_de_vie"], "fe_cible_densite_sbv.parquet")
densite = final_densite.drop_duplicates("sous_bassin_de_vie")["densite_population"]
print(densite.describe().round(0).to_string())
print("Sous-bassins de la cible sans densité :", densite.isna().sum())


# ==============================================================================
# SECTION 4 : PARC DE LOGEMENTS (recensement 2022, photo fixe)
# ==============================================================================
print("--- [4/6] PARC DE LOGEMENTS ---")

df_logement = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet")

# 4.1 Effectifs nécessaires à chaque indicateur retenu (numérateur puis dénominateur)
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

# 4.2 Effectifs additionnés au sous-bassin, puis les 12 indicateurs
com_logement = pd.merge(liaison, df_logement[["code_commune"] + EFFECTIFS_LOGEMENT],
                        left_on="CODGEO_INI", right_on="code_commune")
sbv_logement = indicateurs_logement(com_logement.groupby("sous_bassin_de_vie")[EFFECTIFS_LOGEMENT].sum())
sbv_logement["millesime_rp"] = 2022
sbv_logement = sbv_logement.reset_index()

# 4.3 Jointure à la cible (même photo pour toutes les années) et export
joindre_cible(sbv_logement, ["sous_bassin_de_vie"], "fe_cible_logement_sbv.parquet")


# ==============================================================================
# SECTION 5 : FILOSOFI (millésime 2021, photo fixe)
# ==============================================================================
print("--- [5/6] FILOSOFI ---")

df_filosofi = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet")

# 5.1 Millésime 2021 ; Paris, Lyon, Marseille : ville entière retirée, arrondissements gardés
MILLESIME_FILOSOFI = 2021
fe_filosofi = df_filosofi.loc[df_filosofi["annee"] == MILLESIME_FILOSOFI,
                              ["code_commune", "niveau_vie_median", "NBMEN"]]
fe_filosofi = fe_filosofi[~fe_filosofi["code_commune"].isin(["75056", "69123", "13055"])]
fe_filosofi = fe_filosofi.rename(columns={"NBMEN": "nb_menages_fiscaux"})

# 5.2 Moyenne des médianes communales pondérée par les ménages fiscaux,
#     sur les seules communes dont la médiane ET le nombre de ménages sont connus
com_filosofi = pd.merge(liaison, fe_filosofi, left_on="CODGEO_INI", right_on="code_commune")
connue = com_filosofi["niveau_vie_median"].notna() & com_filosofi["nb_menages_fiscaux"].notna()
com_filosofi["poids"] = com_filosofi["nb_menages_fiscaux"].where(connue)
com_filosofi["masse_niveau_vie"] = (com_filosofi["niveau_vie_median"] * com_filosofi["nb_menages_fiscaux"]).where(connue)
sbv_filosofi = com_filosofi.groupby("sous_bassin_de_vie")[["masse_niveau_vie", "poids"]].sum(min_count=1)
sbv_filosofi["niveau_vie_median"] = sbv_filosofi["masse_niveau_vie"] / sbv_filosofi["poids"]

# 5.3 Part des ménages couverts (contrôle qualité) : ménages fiscaux des communes à médiane connue
#     / ménages du recensement 2022 (Filosofi masque aussi le nombre de ménages des petites communes).
#     Définitions un peu différentes : la part peut dépasser légèrement 100 %. Le nombre n'est pas exporté
men_rp = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet", columns=["code_commune", "P22_MEN"])
com_men = pd.merge(liaison, men_rp, left_on="CODGEO_INI", right_on="code_commune")
men_sbv = com_men.groupby("sous_bassin_de_vie")["P22_MEN"].sum().reindex(sbv_filosofi.index)
sbv_filosofi["part_menages_couverts"] = sbv_filosofi["poids"] / men_sbv.where(men_sbv > 0) * 100
sbv_filosofi["millesime_filosofi"] = MILLESIME_FILOSOFI
sbv_filosofi = sbv_filosofi[["niveau_vie_median", "part_menages_couverts", "millesime_filosofi"]].reset_index()

# 5.4 Jointure à la cible (même photo pour toutes les années) et export
final_filosofi = joindre_cible(sbv_filosofi, ["sous_bassin_de_vie"], "fe_cible_filosofi_sbv.parquet")
couverts = final_filosofi.drop_duplicates("sous_bassin_de_vie")["part_menages_couverts"]
print(couverts.describe().round(1).to_string())
print("Sous-bassins à plus de 100 % :", (couverts > 100).sum())


# ==============================================================================
# SECTION 6 : DVF+ (volume de ventes N-1)
# ==============================================================================
print("--- [6/6] DVF+ ---")

df_dvf = pd.read_parquet(DATA_OUT / "cible_commune_annee.parquet")

# 6.1 Nombres de ventes ; décalage d'un an : ventes de N-1 -> prix de N
fe_dvf = df_dvf[["code_commune", "annee", "n_maison", "n_appart"]].rename(
    columns={"n_maison": "n_ventes_maison_n1", "n_appart": "n_ventes_appart_n1"})
fe_dvf["annee"] = df_dvf["annee"] + 1

# 6.2 Rattachement et somme par sous-bassin x année
com_dvf = pd.merge(liaison, fe_dvf, left_on="CODGEO_INI", right_on="code_commune")
cols_ventes = ["n_ventes_maison_n1", "n_ventes_appart_n1"]
sbv_dvf = com_dvf.groupby(["sous_bassin_de_vie", "annee"])[cols_ventes].sum().reset_index()

# 6.3 Jointure à la cible ; aucune vente = 0, mais seulement pour les années
#     dont l'année N-1 existe dans DVF+ (pas de ventes 2013 : prix 2014 laissés vides)
final_dvf = pd.merge(cible, sbv_dvf, on=["sous_bassin_de_vie", "annee"], how="left", validate="one_to_one")
annee_n1_connue = final_dvf["annee"] >= df_dvf["annee"].min() + 1
final_dvf.loc[annee_n1_connue, cols_ventes] = final_dvf.loc[annee_n1_connue, cols_ventes].fillna(0)
final_dvf["annee_ventes"] = final_dvf["annee"] - 1
assert len(final_dvf) == len(cible), "Nombre de lignes modifié par la jointure : DVF+"
final_dvf.to_parquet(DATA_OUT / "fe_cible_dvf_sbv.parquet", index=False)
print("-> Exporté : fe_cible_dvf_sbv.parquet", final_dvf.shape)


# ==============================================================================
# SECTION 7 : FUSION DES SEPT TABLES (une ligne par sous-bassin x année, comme la cible)
# ==============================================================================
print("--- [7/8] FUSION DES SOURCES ---")
 
FICHIERS_FE = [
    "fe_cible_delinquance_sbv.parquet",
    "fe_cible_vacance_sbv.parquet",
    "fe_cible_equipement_sbv.parquet",
    "fe_cible_densite_sbv.parquet",
    "fe_cible_logement_sbv.parquet",
    "fe_cible_filosofi_sbv.parquet",
    "fe_cible_dvf_sbv.parquet",
]
CLES_SBV = ["sous_bassin_de_vie", "annee"]
cols_cible = [c for c in cible.columns if c not in CLES_SBV]   # colonnes de la cible, gardées une seule fois
assert not cible.duplicated(CLES_SBV).any(), "La cible a des doublons sur sous-bassin x année"
 
table_finale = cible.copy()
for nom in FICHIERS_FE:
    t = pd.read_parquet(DATA_OUT / nom).drop(columns=cols_cible)
    # Une colonne présente dans deux sources serait renommée _x / _y en silence : on l'interdit
    deja_la = (set(t.columns) & set(table_finale.columns)) - set(CLES_SBV)
    assert not deja_la, f"{nom} : colonnes déjà présentes {sorted(deja_la)}"
    table_finale = pd.merge(table_finale, t, on=CLES_SBV, how="left", validate="one_to_one")
 
assert len(table_finale) == len(cible), "Nombre de lignes modifié par la fusion des sources"
table_finale.to_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv.parquet", index=False)
print("-> Exporté : fe_cible_toutes_sources_sbv.parquet", table_finale.shape)
 
# Contrôle : part de valeurs vides par variable (hors cible), de la plus vide à la moins vide
vides = table_finale.drop(columns=CLES_SBV + cols_cible).isna().mean().mul(100).round(1)
print(vides[vides > 0].sort_values(ascending=False).to_string())
 
 
# ==============================================================================
# SECTION 8 : TABLE DE MODÉLISATION (format long, prix N-1, index, export)
# ==============================================================================
print("--- [8/8] TABLE DE MODÉLISATION ---")
 
# 8.1 Passage en format long : une ligne par sous-bassin x année x type de bien
#     n_ventes = ventes de l'année N : sert au seuil, JAMAIS une feature (fuite)
#     prix = nom attendu par commun.py ; une seule indicatrice (maison) : maison + appartement = 1 toujours
df = pd.read_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv.parquet")
 
# Colonnes propres à un type de bien -> nom commun en format long
CORRESPONDANCE = {
    "maison":      {"n_ventes": "n_maison", "nb_ventes_n1": "n_ventes_maison_n1", "prix": "prix_m2_med_maison"},
    "appartement": {"n_ventes": "n_appart", "nb_ventes_n1": "n_ventes_appart_n1", "prix": "prix_m2_med_appart"},
}
CLES_SBV = ["sous_bassin_de_vie", "annee"]
 
colonnes_specifiques = [ancien for mapping in CORRESPONDANCE.values() for ancien in mapping.values()]
# Variables du territoire : identiques pour les deux types de bien
colonnes_territoire = [c for c in df.columns if c not in colonnes_specifiques + CLES_SBV]
 
blocs = []
for type_bien, mapping in CORRESPONDANCE.items():
    bloc = df[CLES_SBV].copy()
    bloc["type_bien"] = type_bien
    bloc["maison"] = int(type_bien == "maison")      # seule indicatrice du type de bien
    for nouveau, ancien in mapping.items():
        bloc[nouveau] = df[ancien]
    blocs.append(pd.concat([bloc, df[colonnes_territoire]], axis=1))
 
table_longue = pd.concat(blocs, ignore_index=True)
 
# Contrôles : lignes exactement doublées, une seule ligne par sous-bassin x année x type
assert len(table_longue) == 2 * len(df), "Le nombre de lignes n'a pas doublé"
assert not table_longue.duplicated(CLES_SBV + ["type_bien"]).any(), "Doublons sous-bassin x année x type"
print(table_longue.shape)
 
 
# 8.2 Seuil de 20 ventes, AVANT prix_n1 : sinon un prix peu fiable deviendrait le prix N-1 de l'année suivante
#     La ligne est gardée, seul son prix est vidé
SEUIL_VENTES = 20
 
# n_ventes vide = aucune vente connue : traité comme sous le seuil
sous_seuil = ~(table_longue["n_ventes"] >= SEUIL_VENTES)
table_longue.loc[sous_seuil, "prix"] = np.nan
 
print("Prix vidés (moins de", SEUIL_VENTES, "ventes) :", int(sous_seuil.sum()))
print(table_longue.assign(prix_connu=table_longue["prix"].notna())
      .pivot_table(index="annee", columns="type_bien", values="prix_connu", aggfunc="sum"))
 
 
# 8.3 prix_n1 : prix de l'année précédente, même sous-bassin, même type de bien
#     Jointure sur l'année précédente (pas de décalage de lignes : une année absente ne fait rien glisser)
prix_annee_precedente = (table_longue[["sous_bassin_de_vie", "type_bien", "annee", "prix"]]
                         .assign(annee=lambda t: t["annee"] + 1)      # le prix de N-1 sert pour l'année N
                         .rename(columns={"prix": "prix_n1"}))
 
table_longue = table_longue.merge(prix_annee_precedente, on=["sous_bassin_de_vie", "type_bien", "annee"],
                                  how="left", validate="one_to_one")
 
assert len(table_longue) == 2 * len(df), "Nombre de lignes modifié par le calcul de prix_n1"
print("prix_n1 renseigné :", int(table_longue["prix_n1"].notna().sum()), "lignes sur", len(table_longue))
 
 
# 8.4 Index à 3 niveaux (annee, sous_bassin_de_vie, type_bien) : les codes de sous-bassin contiennent déjà un "_",
#     une chaîne "2020_01004_1_maison" ne permettrait pas de les retrouver.
#     type_bien seulement dans l'index (commun.py fait groupby("type_bien") : ambigu s'il était aussi en colonne) ;
#     annee dans l'index ET en colonne (commun.decouper fait table["annee"]).
#     Colonnes retirées (liste écrite) : les 4 millésimes, annee_faits et annee_ventes (= annee - 1)
NIVEAUX_INDEX = ["annee", "sous_bassin_de_vie", "type_bien"]
COLONNES_A_RETIRER = ["millesime_lovac", "millesime_bpe", "millesime_rp", "millesime_filosofi",
                      "annee_faits", "annee_ventes"]
 
absentes = sorted(set(COLONNES_A_RETIRER) - set(table_longue.columns))
assert not absentes, f"Colonnes à retirer introuvables : {absentes}"
 
table_longue = (table_longue
                .set_index(NIVEAUX_INDEX, drop=False)
                .drop(columns=["sous_bassin_de_vie", "type_bien"] + COLONNES_A_RETIRER)   # annee reste en colonne
                .sort_index())
 
assert table_longue.index.is_unique, "L'index annee x sous-bassin x type n'est pas unique"
print(table_longue.shape)
 
 
# 8.5 Cible, colonnes hors modèle, features ; ligne utilisable = prix et prix_n1 renseignés
CIBLE = "prix"
ANNEE_EN_FEATURE = False      # à aligner sur la décision de l'étape 0 de la modélisation
HORS_MODELE = ["n_ventes"] + ([] if ANNEE_EN_FEATURE else ["annee"])
FEATURES = [c for c in table_longue.columns if c not in [CIBLE] + HORS_MODELE]
 
assert "prix_n1" in FEATURES and "maison" in FEATURES
assert "n_ventes" not in FEATURES, "n_ventes (ventes de l'année N) ne doit jamais être une feature"
assert "appartement" not in table_longue.columns
 
utilisable = table_longue[CIBLE].notna() & table_longue["prix_n1"].notna()
 
non_numeriques = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(table_longue[c])]
print(len(FEATURES), "features | non numériques :", non_numeriques)
print("Lignes utilisables :", int(utilisable.sum()), "sur", len(table_longue))
 
 
# 8.6 Contrôles : lignes doublées, index unique, prix_n1 2024 = prix 2023, 823 / 1 777 sous-bassins en 2024 (baseline)
# Lignes doublées et index unique
assert len(table_longue) == 2 * len(df)
assert table_longue.index.is_unique
 
# prix_n1 2024 = prix 2023 : un exemple lisible...
prix_2023 = table_longue.xs(2023, level="annee")["prix"].dropna()
sbv_ex, type_ex = prix_2023.index[0]
print(f"Exemple {sbv_ex} / {type_ex} : prix 2023 =", table_longue.loc[(2023, sbv_ex, type_ex), "prix"],
      "| prix_n1 2024 =", table_longue.loc[(2024, sbv_ex, type_ex), "prix_n1"])
assert table_longue.loc[(2024, sbv_ex, type_ex), "prix_n1"] == table_longue.loc[(2023, sbv_ex, type_ex), "prix"]
 
# ... puis sur toutes les lignes de 2024
prix_n1_2024 = table_longue.xs(2024, level="annee")["prix_n1"]
attendu = table_longue.xs(2023, level="annee")["prix"].reindex(prix_n1_2024.index)
assert prix_n1_2024.equals(attendu), "prix_n1 2024 différent du prix 2023"
 
# Nombre de sous-bassins utilisables en 2024, par type de bien (baseline : 823 et 1 777)
utilisables_2024 = utilisable[utilisable.index.get_level_values("annee") == 2024].groupby("type_bien").sum()
print(utilisables_2024.to_string())
assert utilisables_2024.to_dict() == {"appartement": 823, "maison": 1777}, "Écart avec la baseline (823 / 1 777)"
 
 
# 8.7 Export sous le nom lu par commun.py (FICHIER_TABLE), lignes utilisables seulement
table_modelisation = table_longue[utilisable]
table_modelisation.to_parquet(DATA_OUT / "table_modelisation_bv.parquet", index=True)
print("-> Exporté : table_modelisation_bv.parquet", table_modelisation.shape)
print(table_modelisation.groupby(level=["annee", "type_bien"]).size().unstack())