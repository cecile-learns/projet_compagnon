"""
Feature engineering - une table par source, jointe à la cible au sous-bassin de vie
===================================================================================
Auteurs      : Cécile, Théophile, Gabrielle (jointures aux sous-bassins & emploi salarié privé)
               & Christian (superficie des arrondissements)
Date         : 29/10/2026, mis à jour le 04/11/2026
Description  : délinquance, LOVAC, équipements (BPE 2025), densité de population,
               parc de logements (recensement 2022), Filosofi, ménages (recensements
               2019-2022), DVF+ (volume de ventes) et emploi salarié privé (Urssaf,
               exporté à part, pas encore branché à la fusion).
               Pour chaque source : préparation à la commune, rattachement aux
               sous-bassins (liaison_bv_scindes), agrégation, jointure à gauche
               sur la cible (cible_sbv_annee), export fe_cible_<source>_sbv.parquet.
               Constats et justifications : notebook feature_engineering.ipynb
               (même code, mêmes noms de fichiers ; ce .py est la version de référence,
               toute modification se fait dans les deux).
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
from config import DATA_RAW, DATA_OUT

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

def indicateurs_menages(e):
    """Calcule les 5 indicateurs des ménages à partir d'effectifs additionnés à une maille,
    arrondis à 2 décimales. Parts en % : suffixe _pct_men (pour 100 ménages) ou
    _pct_fam (pour 100 familles) ; taille moyenne en personnes par ménage."""
    men = e["C_MEN"].where(e["C_MEN"] > 0)   # dénominateur nul -> valeur vide au lieu d'un infini
    fam = e["C_FAM"].where(e["C_FAM"] > 0)

    r = pd.DataFrame(index=e.index)
    r["taux_personnes_seules_pct_men"] = e["C_MENPSEUL"] / men * 100
    r["part_familles_monoparentales_pct_fam"] = e["C_FAMMONO"] / fam * 100
    r["part_familles_nombreuses_pct_fam"] = (e["C_NE24F3"] + e["C_NE24F4P"]) / fam * 100
    r["part_couples_sans_enfants_pct_fam"] = e["C_COUPSENF"] / fam * 100
    r["taille_moyenne_menage"] = e["C_PMEN"] / men
    return r.round(2)

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

# 3b.1 Paris, Lyon, Marseille : la table des superficies (bloc 1 de la préparation) donne déjà
#      les 45 arrondissements avec leur superficie réelle, et plus la ville entière
VILLES_ENTIERES = ["75056", "69123", "13055"]
assert not superficie["code_commune"].isin(VILLES_ENTIERES).any(), "Ville entière présente : superficie comptée deux fois"
n_arrondissements = superficie["code_commune"].str.match(r"^(751\d\d|6938\d|132\d\d)$").sum()
assert n_arrondissements == 45, f"{n_arrondissements} arrondissements au lieu de 45 : relancer le bloc 1 de la préparation"
superficie_com = superficie
print("Arrondissements avec leur superficie :", n_arrondissements, "(45 attendus)")

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

# 3b.4 Contrôle : la densité varie d'un arrondissement à l'autre d'une même ville
#      (avec une superficie répartie au prorata de la population, tous auraient la densité de leur ville)
dens = sbv_densite.set_index("sous_bassin_de_vie")["densite_population"]
resume = []
for ville, motif, n_attendu in [("Paris", r"^751\d\d$", 20), ("Lyon", r"^6938\d$", 9), ("Marseille", r"^132\d\d$", 16)]:
    d = dens[dens.index.astype(str).str.match(motif)]
    assert len(d) == n_attendu, f"{ville} : {len(d)} arrondissements au lieu de {n_attendu}"
    assert d.round(0).nunique() > 1, f"{ville} : densités identiques, superficie des arrondissements à vérifier"
    resume.append({"ville": ville, "arrondissements": len(d),
                   "densite_min": round(d.min()), "densite_max": round(d.max())})
print(pd.DataFrame(resume).to_string(index=False))


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
# SECTION 5 BIS : MÉNAGES (recensements 2019-2022, prolongés à 2024)
# ==============================================================================
print("--- [5 bis] MÉNAGES ---")

df_menages = pd.read_parquet(DATA_OUT / "menages_commune_2019_2022.parquet")

EFFECTIFS_MENAGES = ["C_MEN", "C_MENPSEUL", "C_PMEN", "C_FAM",
                     "C_FAMMONO", "C_COUPSENF", "C_NE24F3", "C_NE24F4P"]
RATIOS_MENAGES = ["taux_personnes_seules_pct_men", "part_familles_monoparentales_pct_fam",
                  "part_familles_nombreuses_pct_fam", "part_couples_sans_enfants_pct_fam",
                  "taille_moyenne_menage"]
# Évolution 2020-2022 bornée pour 2024 (en points de %, en personnes pour la taille)
BORNES_MENAGES = {
    "taux_personnes_seules_pct_men": 1.5,
    "part_familles_monoparentales_pct_fam": 1.5,
    "part_familles_nombreuses_pct_fam": 1.0,
    "part_couples_sans_enfants_pct_fam": 1.5,
    "taille_moyenne_menage": 0.03,
}

# 5b.1 Rattachement aux sous-bassins (codes commune de chaque millésime, via CODGEO_INI),
#      effectifs additionnés par sous-bassin x année, puis ratios (jamais de moyenne de ratios)
com_menages = pd.merge(liaison, df_menages, left_on="CODGEO_INI", right_on="code_commune")
print("Lignes commune x année sans sous-bassin, retirées :", len(df_menages) - len(com_menages))
eff_menages = com_menages.groupby(["sous_bassin_de_vie", "annee"])[EFFECTIFS_MENAGES].sum()
sbv_menages = indicateurs_menages(eff_menages).reset_index()

# 5b.1 bis Stabilité des indicateurs entre recensements (justifie 2023 = 2022 et le prolongement de 2024)
#          Écart absolu par sous-bassin entre deux années, comparé à un seuil
#          (points de % pour les parts, personnes pour la taille des ménages)
def ecarts_entre_annees(table, debut, fin, seuils):
    """Écart absolu de chaque indicateur entre deux années, par sous-bassin, comparé à un seuil."""
    pivot = table[table["annee"].isin([debut, fin])].pivot(index="sous_bassin_de_vie", columns="annee",
                                                            values=RATIOS_MENAGES)
    lignes = []
    for r in RATIOS_MENAGES:
        ecart = (pivot[(r, fin)] - pivot[(r, debut)]).abs().dropna()
        lignes.append({"indicateur": r,
                       "seuil": seuils[r],
                       "écart médian": round(ecart.median(), 3),
                       "écart max": round(ecart.max(), 2),
                       "sous-bassins au-delà du seuil (%)": round((ecart > seuils[r]).mean() * 100, 1)})
    print(f"{debut} -> {fin} : {len(pivot)} sous-bassins comparés")
    return pd.DataFrame(lignes).set_index("indicateur")


SEUILS_1AN = {"taux_personnes_seules_pct_men": 1.5, "part_familles_monoparentales_pct_fam": 1.5,
              "part_familles_nombreuses_pct_fam": 1.0, "part_couples_sans_enfants_pct_fam": 1.5,
              "taille_moyenne_menage": 0.02}
SEUILS_2ANS = {"taux_personnes_seules_pct_men": 2.0, "part_familles_monoparentales_pct_fam": 2.0,
               "part_familles_nombreuses_pct_fam": 1.5, "part_couples_sans_enfants_pct_fam": 2.0,
               "taille_moyenne_menage": 0.03}
print(ecarts_entre_annees(sbv_menages, 2021, 2022, SEUILS_1AN).to_string())
print(ecarts_entre_annees(sbv_menages, 2020, 2022, SEUILS_2ANS).to_string())

# 5b.2 2023 : copie de 2022 (pas encore de recensement publié) ;
#      2024 : 2022 + évolution 2020-2022, bornée pour éviter les valeurs extrêmes
r_2020 = sbv_menages[sbv_menages["annee"] == 2020].set_index("sous_bassin_de_vie")[RATIOS_MENAGES]
r_2022 = sbv_menages[sbv_menages["annee"] == 2022].set_index("sous_bassin_de_vie")[RATIOS_MENAGES]
communs = r_2022.index.intersection(r_2020.index)
evolution = r_2022.loc[communs] - r_2020.loc[communs]
for col, borne in BORNES_MENAGES.items():
    evolution[col] = evolution[col].clip(lower=-borne, upper=borne)
an_2023 = r_2022.assign(annee=2023).reset_index()
an_2024 = (r_2022.loc[communs] + evolution).round(2).assign(annee=2024).reset_index()
sbv_menages = pd.concat([sbv_menages, an_2023, an_2024], ignore_index=True)
sbv_menages = sbv_menages.sort_values(["sous_bassin_de_vie", "annee"]).reset_index(drop=True)
assert not sbv_menages.duplicated(["sous_bassin_de_vie", "annee"]).any(), "Doublons sous-bassin x année (ménages)"
print("Sous-bassins sans 2024 (absents en 2020) :", sbv_menages["sous_bassin_de_vie"].nunique() - len(an_2024))

# 5b.3 Années de prix hors 2019-2024 : recensement de l'année la plus proche
#      (2019 pour les prix 2014-2018, 2024 pour les prix 2025), gardé dans annee_rp_menages
annees_menages = pd.DataFrame({"annee": sorted(cible["annee"].unique())})
annees_menages["annee_rp_menages"] = annees_menages["annee"].clip(sbv_menages["annee"].min(),
                                                                  sbv_menages["annee"].max())
print(annees_menages.to_string(index=False))
sbv_menages = pd.merge(annees_menages, sbv_menages.rename(columns={"annee": "annee_rp_menages"}),
                       on="annee_rp_menages")
sbv_menages = sbv_menages[["sous_bassin_de_vie", "annee", "annee_rp_menages"] + RATIOS_MENAGES]

# 5b.4 Jointure à la cible (sous-bassin x année) et export
final_menages = joindre_cible(sbv_menages, ["sous_bassin_de_vie", "annee"], "fe_cible_menages_sbv.parquet")
print("Lignes de la cible sans ménages :", final_menages["taux_personnes_seules_pct_men"].isna().sum())
print(final_menages[RATIOS_MENAGES].describe().round(2).to_string())


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
# SECTION 6 BIS : EMPLOI SALARIÉ PRIVÉ (Urssaf, commune x activité x année -> sous-bassin x année)
# ==============================================================================
print("--- [6 bis] EMPLOI ---")

df_emploi = pd.read_parquet(DATA_RAW / "etablissements_effectifs_commune.parquet")
ANNEES_EMPLOI = sorted(int(c[-4:]) for c in df_emploi.columns if c.startswith("Effectifs salariés"))
print(df_emploi.shape, "| années", ANNEES_EMPLOI[0], "à", ANNEES_EMPLOI[-1])

# 6b.1 Code commune sur 5 caractères : zéro initial perdu pour une partie des lignes des départements
#      01 à 09 (audit, 5.2 : 574 communes écrites des deux façons)
df_emploi["code_commune"] = df_emploi["Code commune"].astype(str).str.zfill(5)
assert not df_emploi.duplicated(["code_commune", "Code APE"]).any(), "Doublons commune x activité après correction du code"

# 6b.2 Cases vides (audit, 5.1) : sans établissement -> 0 salarié (l'activité n'existait pas cette année-là) ;
#      avec établissement -> secret statistique, laissé vide
salaries = pd.DataFrame({a: df_emploi[f"Effectifs salariés {a}"] for a in ANNEES_EMPLOI})
etablissements = pd.DataFrame({a: df_emploi[f"Nombre d'établissements {a}"] for a in ANNEES_EMPLOI})
salaries = salaries.mask(salaries.isna() & ~(etablissements > 0), 0.0)
secret = salaries.isna()
active = etablissements > 0                   # cellules avec au moins un établissement
print("Cellules sous secret, toutes années :", int(secret.to_numpy().sum()))

# 6b.3 Rattachement de chaque cellule commune x activité à son sous-bassin (CODGEO_INI)
lien_emploi = liaison[["CODGEO_INI", "sous_bassin_de_vie"]].drop_duplicates()
assert not lien_emploi["CODGEO_INI"].duplicated().any(), "Un code commune rattaché à deux sous-bassins"
sbv_cellule = df_emploi["code_commune"].map(lien_emploi.set_index("CODGEO_INI")["sous_bassin_de_vie"])
rattachee = sbv_cellule.notna()
part_non_rattachee = salaries[~rattachee].sum().sum() / salaries.sum().sum()
print(f"Salariés non rattachés à un sous-bassin : {part_non_rattachee:.2%}",
      f"({df_emploi.loc[~rattachee, 'code_commune'].nunique()} codes)")
assert part_non_rattachee < 0.01, "Plus de 1 % des salariés sans sous-bassin : vérifier les codes commune"

# 6b.4 Somme au sous-bassin x année (le secret compte pour 0 dans le niveau : un établissement, négligeable)
#      et part des cellules actives sous secret (contrôle qualité, hors modèle)
groupe = sbv_cellule[rattachee]
sal_sbv = salaries[rattachee].groupby(groupe).sum()
part_secret = secret[rattachee].groupby(groupe).sum() / active[rattachee].groupby(groupe).sum().clip(lower=1) * 100

# 6b.5 Évolution sur un an, à champ constant : seules les cellules connues les deux années comptent.
#      Les absences valent 0 : on n'écarte que le secret, jamais une ouverture ou une fermeture
cst, cst_prec = {}, {}
for a in ANNEES_EMPLOI[1:]:
    connues = rattachee & salaries[a].notna() & salaries[a - 1].notna()
    cst[a] = salaries.loc[connues, a].groupby(sbv_cellule[connues]).sum()
    cst_prec[a] = salaries.loc[connues, a - 1].groupby(sbv_cellule[connues]).sum()
cst, cst_prec = pd.DataFrame(cst), pd.DataFrame(cst_prec)
evolution = (cst / cst_prec.where(cst_prec > 0) - 1) * 100
evolution = evolution.mask((cst == 0) & (cst_prec == 0), 0.0)   # aucun salarié les deux années : pas d'évolution

# Sous-bassins de la cible sans aucun employeur privé : 0 salarié, pas d'évolution, pas de secret
sbv_cible = pd.Index(cible["sous_bassin_de_vie"].unique(), name="sous_bassin_de_vie")
print("Sous-bassins de la cible sans employeur privé :", len(sbv_cible.difference(sal_sbv.index)))
sans_employeur = ~sbv_cible.isin(sal_sbv.index)
sal_sbv = sal_sbv.reindex(sbv_cible, fill_value=0.0)
part_secret = part_secret.reindex(sbv_cible, fill_value=0.0)
evolution = evolution.reindex(sbv_cible)
evolution.loc[sans_employeur] = 0.0

# 6b.6 Niveau rapporté au parc de logements du recensement 2022 (même dénominateur que log_nb_logements)
p22_log_emploi = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet", columns=["code_commune", "P22_LOG"])
com_parc = pd.merge(liaison, p22_log_emploi, left_on="CODGEO_INI", right_on="code_commune")
parc_sbv = com_parc.groupby("sous_bassin_de_vie")["P22_LOG"].sum()


def en_long(t, nom):
    """Sous-bassin en lignes x millésime en colonnes -> une ligne par sous-bassin x millésime."""
    return (t.rename_axis("sous_bassin_de_vie").rename_axis(columns=None).reset_index()
            .melt(id_vars="sous_bassin_de_vie", var_name="annee_emploi", value_name=nom))


long_emploi = (en_long(sal_sbv, "salaries")
               .merge(en_long(evolution, "evol_emploi_n2_n1"), on=["sous_bassin_de_vie", "annee_emploi"], how="left")
               .merge(en_long(part_secret, "part_emploi_secret"), on=["sous_bassin_de_vie", "annee_emploi"], how="left"))
long_emploi["annee_emploi"] = long_emploi["annee_emploi"].astype(int)
long_emploi["emploi_par_logement"] = (long_emploi["salaries"]
                                      / long_emploi["sous_bassin_de_vie"].map(parc_sbv.where(parc_sbv > 0)))

# 6b.7 Décalage : salariés au 31/12 de N-1 -> prix de N (convention « connu en N-1 ») ;
#      le 31/12/N-1 n'est publié qu'en mai de N : variante prudente (N-2) exportée à part, testée dans H3c
DECALAGE_EMPLOI = 1
VARIABLES_EMPLOI = ["emploi_par_logement", "evol_emploi_n2_n1", "part_emploi_secret"]


def emploi_decale(decalage):
    t = long_emploi.assign(annee=long_emploi["annee_emploi"] + decalage)
    return t[["sous_bassin_de_vie", "annee", "annee_emploi"] + VARIABLES_EMPLOI]


# Une table par variante de calendrier, sous-bassin x année de prix (lues par le notebook H3c)
for d in (1, 2):
    emploi_decale(d).to_parquet(DATA_OUT / f"fe_emploi_decalage{d}_sbv.parquet", index=False)
    print(f"-> Exporté : fe_emploi_decalage{d}_sbv.parquet")

# 6b.8 Jointure à la cible (sous-bassin x année) et export
final_emploi = joindre_cible(emploi_decale(DECALAGE_EMPLOI), ["sous_bassin_de_vie", "annee"], "fe_cible_emploi_sbv.parquet")
print("Cases vides par année de prix (%) :")
print(final_emploi.groupby("annee")[VARIABLES_EMPLOI].apply(lambda g: g.isna().mean() * 100).round(2).to_string())
print(final_emploi[VARIABLES_EMPLOI].describe().round(2).to_string())

# ==============================================================================
# SECTION 7 : FUSION DES HUIT TABLES (une ligne par sous-bassin x année, comme la cible)
# ==============================================================================
print("--- [7/8] FUSION DES SOURCES ---")
 
FICHIERS_FE = [
    "fe_cible_delinquance_sbv.parquet",
    "fe_cible_vacance_sbv.parquet",
    "fe_cible_equipement_sbv.parquet",
    "fe_cible_densite_sbv.parquet",
    "fe_cible_logement_sbv.parquet",
    "fe_cible_filosofi_sbv.parquet",
    "fe_cible_menages_sbv.parquet",
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


# 8.3 bis BV2022 : bassin de vie de chaque sous-bassin. Identifiant pour le bootstrap par bassin de vie
#          de commun.tester (les sous-bassins d'un même bassin ne sont pas indépendants) ; jamais une feature (8.5).
#          Ajouté avant 8.4, tant que sous_bassin_de_vie est encore une colonne
#          Lu dans le code du sous-bassin (et non dans la colonne BV2022 de la liaison : les sous-bassins viennent
#          des bassins de communes-france-2026, BV2022 du fichier Insee, qui ne concordent pas toujours)
# Bassin de vie lu dans le code du sous-bassin : "01004_1" -> "01004" ; bassin non scindé -> son propre code
table_longue["BV2022"] = table_longue["sous_bassin_de_vie"].str.split("_").str[0]

# Arrondissements de Paris, Lyon, Marseille : le sous-bassin porte le code de l'arrondissement,
# on lui donne le bassin de vie de sa ville (table de liaison, une ligne par code)
arrondissement = table_longue["sous_bassin_de_vie"].str.match(r"^(751\d\d|6938\d|132\d\d)$")
bv_des_communes = liaison.set_index("CODGEO_INI")["BV2022"]
table_longue.loc[arrondissement, "BV2022"] = table_longue.loc[arrondissement, "sous_bassin_de_vie"].map(bv_des_communes)

# Contrôles : chaque sous-bassin a un et un seul bassin de vie
assert table_longue["BV2022"].notna().all(), "Des sous-bassins n'ont pas de bassin de vie"
assert table_longue.groupby("sous_bassin_de_vie")["BV2022"].nunique().eq(1).all(), "Un sous-bassin a plusieurs bassins de vie"
print("Bassins de vie :", table_longue["BV2022"].nunique(),
      "| sous-bassins :", table_longue["sous_bassin_de_vie"].nunique(),
      "| arrondissements :", int(arrondissement.sum() / table_longue["annee"].nunique() / 2))
 
 
# 8.4 Index à 3 niveaux (annee, sous_bassin_de_vie, type_bien) : les codes de sous-bassin contiennent déjà un "_",
#     une chaîne "2020_01004_1_maison" ne permettrait pas de les retrouver.
#     type_bien seulement dans l'index (commun.py fait groupby("type_bien") : ambigu s'il était aussi en colonne) ;
#     annee dans l'index ET en colonne (commun.decouper fait table["annee"]).
#     Colonnes retirées (liste écrite) : les 4 millésimes, annee_rp_menages (année du recensement utilisé),
#     annee_faits et annee_ventes (= annee - 1),
#     une variable par couple redondant (audit de la table, version 1 : |rho| de 0,78 à 0,88 avec la variable gardée)
#     et les deux tailles des ménages (audit, version 2 : |rho| de 0,92 à 0,99 avec la part de personnes seules, gardée)
NIVEAUX_INDEX = ["annee", "sous_bassin_de_vie", "type_bien"]
COLONNES_A_RETIRER = ["millesime_lovac", "millesime_bpe", "millesime_rp", "millesime_filosofi",
                      "annee_rp_menages", "annee_faits", "annee_ventes",
                      "part_suroccupes", "part_parking", "taux3_destructions",   # redondantes (audit, version 1)
                      "taille_menage", "taille_moyenne_menage"]                  # redondantes (audit, version 2)
 
absentes = sorted(set(COLONNES_A_RETIRER) - set(table_longue.columns))
assert not absentes, f"Colonnes à retirer introuvables : {absentes}"
 
table_longue = (table_longue
                .set_index(NIVEAUX_INDEX, drop=False)
                .drop(columns=["sous_bassin_de_vie", "type_bien"] + COLONNES_A_RETIRER)   # annee reste en colonne
                .sort_index())
 
assert table_longue.index.is_unique, "L'index annee x sous-bassin x type n'est pas unique"
print(table_longue.shape)
 
 
# 8.5 Cible, colonnes hors modèle, features ; ligne utilisable = prix et prix_n1 renseignés
#     Hors modèle : n_ventes (fuite), BV2022 (identifiant) et, selon l'étape 0 de la modélisation, annee
CIBLE = "prix"
ANNEE_EN_FEATURE = False      # à aligner sur la décision de l'étape 0 de la modélisation
HORS_MODELE = ["n_ventes", "BV2022"] + ([] if ANNEE_EN_FEATURE else ["annee"])
FEATURES = [c for c in table_longue.columns if c not in [CIBLE] + HORS_MODELE]
 
assert "prix_n1" in FEATURES and "maison" in FEATURES
assert "n_ventes" not in FEATURES, "n_ventes (ventes de l'année N) ne doit jamais être une feature"
assert "BV2022" not in FEATURES, "BV2022 est un identifiant, jamais une feature"
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