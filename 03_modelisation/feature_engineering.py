"""
Feature engineering - une table par source, jointe à la cible au sous-bassin de vie
===================================================================================
Auteurs      : Cécile & Théophile
Date         : 29/10/2026
Description  : délinquance, LOVAC, équipements (BPE 2025), parc de logements
               (recensement 2022), Filosofi et DVF+ (volume de ventes).
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
    Le taux lissé reste vide tant que la fenêtre n'est pas complète (premières années)."""
    table = table.copy()
    table["estimes_x_pop"] = table["part_indicateurs_estimes"] * table["insee_pop"]
    colonnes = ["nb_" + f for f in familles] + ["insee_pop", "insee_log", "estimes_x_pop"]
    # 1. Faits et dénominateurs additionnés par maille et par année
    agr = table.groupby([maille, "annee"])[colonnes].sum().reset_index()
    agr = agr.sort_values([maille, "annee"]).reset_index(drop=True)
    # 2. Fiabilité : moyenne pondérée par la population
    agr["part_indicateurs_estimes"] = agr["estimes_x_pop"] / agr["insee_pop"]
    # 3. Taux annuel et taux lissé (années passées uniquement, fenêtre complète)
    for f in familles:
        denom = "insee_log" if f == "cambriolages" else "insee_pop"
        faits_n = agr.groupby(maille)["nb_" + f].rolling(fenetre, min_periods=fenetre).sum().reset_index(level=0, drop=True)
        denom_n = agr.groupby(maille)[denom].rolling(fenetre, min_periods=fenetre).sum().reset_index(level=0, drop=True)
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

# 1.4 Rattachement aux sous-bassins, taux annuels et lissés sur 3 ans
com_delinq = pd.merge(liaison, fe_delinq, left_on="CODGEO_INI", right_on="code_commune")
sbv_delinq = taux_lisses(com_delinq, maille="sous_bassin_de_vie", familles=list(FAMILLES_DELINQ))
sbv_delinq["annee_faits"] = sbv_delinq["annee"] - 1

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
sbv_equip = com_equip.groupby("sous_bassin_de_vie")[cols_equip].sum().reset_index()

# 3.2 Jointure à la cible ; sous-bassin sans équipement recensé = 0 (règle de l'EDA complémentaire)
final_equip = pd.merge(cible, sbv_equip, on="sous_bassin_de_vie", how="left", validate="many_to_one")
final_equip[cols_equip] = final_equip[cols_equip].fillna(0)
final_equip["millesime_bpe"] = 2025
assert len(final_equip) == len(cible), "Nombre de lignes modifié par la jointure : équipements"
final_equip.to_parquet(DATA_OUT / "fe_cible_equipement_sbv.parquet", index=False)
print("-> Exporté : fe_cible_equipement_sbv.parquet", final_equip.shape)


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
sbv_filosofi = sbv_filosofi.rename(columns={"poids": "nb_menages_fiscaux"})
sbv_filosofi["millesime_filosofi"] = MILLESIME_FILOSOFI
sbv_filosofi = sbv_filosofi[["niveau_vie_median", "nb_menages_fiscaux", "millesime_filosofi"]].reset_index()

# 5.3 Jointure à la cible (même photo pour toutes les années) et export
joindre_cible(sbv_filosofi, ["sous_bassin_de_vie"], "fe_cible_filosofi_sbv.parquet")


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

# Tables produites par les sections 1 à 6 (une ligne par sous-bassin x année, comme la cible)
FICHIERS_FE = [
    "fe_cible_delinquance_sbv.parquet",
    "fe_cible_vacance_sbv.parquet",
    "fe_cible_equipement_sbv.parquet",
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
# SECTION 7 : Fusion, encodage du type de bien, création d'index
# ==============================================================================
# Tables produites par les sections précédentes (une ligne par sous-bassin x année, comme la cible)
FICHIERS_FE = [
    "fe_cible_delinquance_sbv.parquet",
    "fe_cible_vacance_sbv.parquet",
    "fe_cible_equipement_sbv.parquet",
    "fe_cible_logement_sbv.parquet",
    "fe_cible_filosofi_sbv.parquet",
    "fe_cible_dvf_sbv.parquet",
]
CLES_SBV = ["sous_bassin_de_vie", "annee"]
cols_cible = [c for c in cible.columns if c not in CLES_SBV]   # colonnes de la cible, gardées une seule fois
assert not cible.duplicated(CLES_SBV).any(), "La cible a des doublons sur sous-bassin x année"

# Fusion de toutes les tables
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


"""
Passage de la table "large" (une ligne par sous-bassin de vie x année,
avec des colonnes séparées pour maisons et appartements) à une table
"longue" (une ligne par sous-bassin x année x type de bien), avec
encodage binaire du type de bien.

Entrée  : fe_cible_toutes_sources_sbv.parquet
Sortie  : fe_cible_toutes_sources_sbv_encodee.parquet
"""

# ---------------------------------------------------------------------------
# 1. Lecture
df = pd.read_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv.parquet")

# 2. Correspondance entre les colonnes d'origine et les colonnes cibles
# Trois variables existent en double dans la table d'origine (une version
# "maison", une version "appart"). On les ramène chacune à une seule colonne.

CORRESPONDANCE = {
    "maison": {
        "nb_biens": "n_maison",
        "nb_ventes_n1": "n_ventes_maison_n1",
        "prix_m2_med": "prix_m2_med_maison",
    },
    "appartement": {
        "nb_biens": "n_appart",
        "nb_ventes_n1": "n_ventes_appart_n1",
        "prix_m2_med": "prix_m2_med_appart",
    },
}

CLES = ["sous_bassin_de_vie", "annee"]

# Colonnes spécifiques à un type de bien (à ne pas recopier telles quelles).
colonnes_specifiques = [
    ancienne
    for mapping in CORRESPONDANCE.values()
    for ancienne in mapping.values()
]

# Toutes les autres colonnes (criminalité, INSEE, équipements...) décrivent le
# territoire et non le type de bien : elles seront dupliquées à l'identique
# sur les deux lignes.
colonnes_communes = [
    c for c in df.columns if c not in colonnes_specifiques and c not in CLES
]

# 3. Construction d'un bloc par type de bien
blocs = []
for type_bien, mapping in CORRESPONDANCE.items():
    bloc = df[CLES].copy()

    # Encodage binaire (one-hot) : 1 pour le type de la ligne, 0 pour l'autre.
    bloc["maison"] = int(type_bien == "maison")
    bloc["appartement"] = int(type_bien == "appartement")

    # Variables propres au type de bien, renommées avec un nom commun.
    for nouveau_nom, ancien_nom in mapping.items():
        bloc[nouveau_nom] = df[ancien_nom]

    # Variables du territoire, identiques pour les deux types de bien.
    for c in colonnes_communes:
        bloc[c] = df[c]

    # Ordre d'affichage : maison en premier (0), appartement ensuite (1).
    bloc["_ordre"] = 0 if type_bien == "maison" else 1
    blocs.append(bloc)

# 4. Empilement et tri
# Tri par sous-bassin, année, puis type de bien : on obtient pour chaque
# couple sous-bassin x année une ligne maison suivie d'une ligne appartement.
table_longue = (
    pd.concat(blocs)
    .sort_values(CLES + ["_ordre"])
    .drop(columns="_ordre")
    .reset_index(drop=True)
)

# 5. Contrôles
# Le nombre de lignes doit avoir exactement doublé.
assert len(table_longue) == 2 * len(df)
# Une et une seule des deux colonnes d'encodage vaut 1 sur chaque ligne.
assert (table_longue["maison"] + table_longue["appartement"] == 1).all()
# Aucune clé (sous-bassin, année, type) en double.
assert not table_longue.duplicated(CLES + ["maison"]).any()

# 6. Génération d'un index annee_sbv_typebien et suppression des colonnes constantes
# La création de l'index nous permettra de supprimer la colonne du sbv et de ne pas avoir de lignes en doublons/ vérifier qu'il n'y a pas de doublons.
type_bien = table_longue["maison"].map({1: "maison", 0: "appartement"})
table_longue.index = (
    table_longue["annee"].astype(str)
    + "_" + table_longue["sous_bassin_de_vie"]
    + "_" + type_bien
)
table_longue.index.name = "annee_sbv_typebien"

# L'index doit identifier chaque ligne de façon unique.
assert table_longue.index.is_unique

# Les colonnes constantes, par définition, n'apportent pas d'informations.
colonnes_constantes = [c for c in table_longue.columns if table_longue[c].nunique(dropna=True) <= 1]
table_longue = table_longue.drop(columns=colonnes_constantes)

table_longue=table_longue.drop(columns="sous_bassin_de_vie")

# 7. Export
# Les valeurs manquantes (prix absent, premier nb_ventes_n1 de chaque
# sous-bassin) restent vides : elles ne sont pas imputées ici.
table_longue.to_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv_encodee.parquet", index=True)
print("-> Exporté : fe_cible_toutes_sources_sbv_encodee.parquet", table_longue.shape)