# =============================================================================
# 01 - PRÉPARATION DES DONNÉES - projet Compagnon Immobilier
# =============================================================================
# Version courte de preparation.ipynb (explications détaillées dans le notebook).
# À lancer depuis la racine du projet, après 00_telechargement :
#   python 01_preparation/preparation.py
# Entrées : data/raw/  |  Sorties : data/processed/

import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# config.py se trouve dans le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_OUT

DATA_OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BLOC 1 - TABLE DE LIAISON COMMUNE -> BASSIN DE VIE 2022
# =============================================================================
# Une ligne par code commune ayant existé depuis 2003 (+ arrondissements de
# Paris, Lyon, Marseille), avec son code 2026 et son bassin de vie 2022.
# Toute source s'y rattache par son code commune, sur CODGEO_INI.

# --- Lecture des fichiers Insee (titres sur 5 lignes, codes en texte) ---
passage = pd.read_excel(DATA_RAW / "table_passage_geo2003_geo2026.xlsx",
                        sheet_name="Table de passage", skiprows=5, dtype=str)
fichier_bv = DATA_RAW / "BV2022_2022_2026.xlsx"
compo = pd.read_excel(fichier_bv, sheet_name="Composition_communale", skiprows=5, dtype=str)
bassins = pd.read_excel(fichier_bv, sheet_name="BV2022", skiprows=5, dtype=str)

# --- Communes scindées : filtrage sur la table de passage initiale ---
scindee = passage["CODGEO_INI"].duplicated(keep=False)
identite = passage["CODGEO_INI"] == passage["CODGEO_2026"]
cas_52427 = (passage["CODGEO_INI"] == "52427") & (
    passage["CODGEO_2026"] == "52331"
)
passage = passage[~scindee | identite | cas_52427]

# --- Ajout des arrondissements (correction de la borne sup pour Lyon) ---
villes = {
    "75056": ("Paris", range(75101, 75121)),  # 75101 à 75120
    "69123": ("Lyon", range(69381, 69391)),  # 69381 à 69390
    "13055": ("Marseille", range(13201, 13217)),  # 13201 à 13216
}

lignes = []
for code_ville, (nom, arrondissements) in villes.items():
    for code_arr in arrondissements:
        lignes.append({
            "CODGEO_INI": str(code_arr),
            "CODGEO_2026": code_ville,
            "LIBGEO_2026": nom,
        })

df_arrondissements = pd.DataFrame(lignes)

# Sécurité : on évite les doublons si les arrondissements existent déjà dans passage
passage = passage[~passage["CODGEO_INI"].isin(df_arrondissements["CODGEO_INI"])]
passage = pd.concat([passage, df_arrondissements], ignore_index=True)

# --- Ajout du bassin de vie et de sa description ---
liaison = passage.merge(compo[["CODGEO", "BV2022"]], left_on="CODGEO_2026", right_on="CODGEO",
                        how="left", validate="many_to_one").drop(columns="CODGEO")
liaison = liaison.merge(bassins[["BV2022", "LIBBV2022", "TYPE_BV2022"]], on="BV2022",
                        how="left", validate="many_to_one")

# --- Contrôles et enregistrement ---
print("Bloc 1 - lignes :", len(liaison),
      "| un code = une ligne :", liaison["CODGEO_INI"].is_unique,
      "| codes sans bassin :", liaison["BV2022"].isna().sum())
liaison.to_parquet(DATA_OUT / "liaison_commune_bv2022.parquet", index=False)


# =============================================================================
# BLOC 2 - FILTRAGE DE DVF+
# =============================================================================
# Périmètre fixé par l'audit : ventes classiques (VEFA exclues) d'une seule
# maison ou d'un seul appartement (dépendances acceptées), sur une seule commune,
# France métropolitaine sans les DOM. Résultat : un seul fichier Parquet.
# Deux parcours :
#  - complet : DVF+ brut téléchargé (source "dvf"), filtré ici ;
#  - rapide  : fichier filtré téléchargé depuis le Drive (source "dvf_residentiel").
# Si le fichier existe déjà, le filtrage est sauté (le supprimer pour le refaire).

fichier_pop = DATA_OUT / "dvf_plus_residentiel.parquet"
dossier_dep = DATA_OUT / "dvf_plus_residentiel_departements"   # temporaire

if fichier_pop.exists():
    print("Bloc 2 - population déjà présente : filtrage sauté")
else:
    # --- Fichiers départementaux, sans les DOM (93 attendus) ---
    DOM = {"dvf_plus_d971", "dvf_plus_d972", "dvf_plus_d973", "dvf_plus_d974"}
    fichiers = [f for f in sorted((DATA_RAW / "DVF+").glob("dvf_plus_d*.csv")) if f.stem not in DOM]

    # --- Règles de filtrage ---
    def filtre_perimetre(df):
        """Garde les ventes d'un seul logement ancien, sur une seule commune."""
        nature_ok = df["libnatmut"] == "Vente"
        une_commune = df["nbcomm"] == 1
        maison_ok = ((df["libtypbien"] == "UNE MAISON") & (df["nblocmai"] == 1)
                     & (df["nblocapt"] == 0) & (df["nblocact"] == 0))
        appartement_ok = ((df["libtypbien"] == "UN APPARTEMENT") & (df["nblocapt"] == 1)
                          & (df["nblocmai"] == 0) & (df["nblocact"] == 0))
        return df.loc[nature_ok & une_commune & (maison_ok | appartement_ok)]

    # --- Lecture par blocs, filtrage, un fichier temporaire par département ---
    dossier_dep.mkdir(exist_ok=True)
    types_texte = {"coddep": str, "l_codinsee": str}
    bilan = []
    for fichier in fichiers:
        morceaux = []
        n_brut = 0
        for bloc in pd.read_csv(fichier, sep="|", chunksize=200_000,
                                dtype=types_texte, low_memory=False):
            n_brut += len(bloc)
            morceaux.append(filtre_perimetre(bloc))
        pop = pd.concat(morceaux, ignore_index=True)
        pop.to_parquet(dossier_dep / f"{fichier.stem}.parquet", index=False)
        bilan.append({"fichier": fichier.stem, "brut": n_brut, "retenu": len(pop)})

    # --- Bilan par département (pour le rapport) ---
    bilan = pd.DataFrame(bilan)
    bilan["part_retenue_%"] = (bilan["retenu"] / bilan["brut"] * 100).round(1)
    bilan.to_csv(DATA_OUT / "bilan_filtrage.csv", index=False)

    # --- Réunion en un seul fichier ---
    # Type commun par colonne (texte si un département a du texte, sinon décimal),
    # puis écriture département par département pour limiter la mémoire.
    parquets_dep = sorted(dossier_dep.glob("*.parquet"))
    types = {}
    for p in parquets_dep:
        for champ in pq.read_schema(p):
            ancien = types.get(champ.name)
            if ancien is None or ancien == champ.type:
                types[champ.name] = champ.type
            elif pa.types.is_large_string(ancien) or pa.types.is_large_string(champ.type) \
                    or pa.types.is_string(ancien) or pa.types.is_string(champ.type):
                types[champ.name] = pa.large_string()
            else:
                types[champ.name] = pa.float64()
    colonnes = pq.read_schema(parquets_dep[0]).names
    schema = pa.schema([(c, types[c]) for c in colonnes])
    with pq.ParquetWriter(fichier_pop, schema) as fichier_unique:
        for p in parquets_dep:
            fichier_unique.write_table(pq.read_table(p, columns=colonnes).cast(schema))
    shutil.rmtree(dossier_dep)

    print("Bloc 2 - départements :", len(bilan),
          "| mutations brutes :", bilan["brut"].sum(),
          "| ventes retenues :", bilan["retenu"].sum())


# =============================================================================
# BLOC 3 - ÉCHANTILLON D'APPRENTISSAGE (2014-2024)
# =============================================================================
# Tirage stratifié : une strate = un département x une année.
# 10 % des ventes de chaque strate, au moins 300 (sans dépasser la strate).
# Poids w_h = N_h / n_h : l'échantillon n'est pas proportionnel,
# toute statistique sur plusieurs départements ou années doit utiliser w_h.
# 2025 est réservée au test : aucune vente de 2025 n'est tirée.

# --- Nombre de ventes par strate (N_h) ---
strates = (pd.read_parquet(fichier_pop, columns=["coddep", "anneemut"])
             .value_counts().rename("N_h").reset_index())

# --- Nombre à tirer (n_h) et poids (w_h) par strate ---
taux, plancher = 0.10, 300
app = strates[strates["anneemut"] <= 2024].copy()
app["n_h"] = np.minimum(app["N_h"], np.maximum(plancher, np.round(app["N_h"] * taux))).astype(int)
app["w_h"] = app["N_h"] / app["n_h"]

# --- Tirage : un nombre au hasard par vente, les n_h plus petits par strate ---
rng = np.random.default_rng(42)   # graine fixée : tirage reproductible
morceaux = []
for dep in sorted(app["coddep"].unique()):
    d = pd.read_parquet(fichier_pop, filters=[("coddep", "==", dep)])   # un département à la fois
    d = d[d["anneemut"] <= 2024]
    d = d.merge(app[["coddep", "anneemut", "n_h", "w_h"]], on=["coddep", "anneemut"])
    d["alea"] = rng.random(len(d))
    d["rang"] = d.groupby(["coddep", "anneemut"])["alea"].rank(method="first")
    morceaux.append(d[d["rang"] <= d["n_h"]].drop(columns=["alea", "rang", "n_h"]))
ech = pd.concat(morceaux, ignore_index=True)

# --- Retrait des 11 colonnes écartées par l'audit, puis enregistrement ---
a_retirer = ["idmutinvar", "idopendata", "codservch", "refdoc", "nbartcgi", "l_artcgi",
             "l_section", "l_idpar", "l_idparmut", "l_idlocmut", "l_dcnt"]
ech = ech.drop(columns=a_retirer)
ech.to_parquet(DATA_OUT / "dvf_plus_echantillon_apprentissage.parquet", index=False)
ech[["idmutation"]].to_csv(DATA_OUT / "ids_echantillon_apprentissage.csv", index=False)

# --- Contrôles ---
verif = ech.groupby(["coddep", "anneemut"]).size().rename("tire").reset_index()
verif = verif.merge(app[["coddep", "anneemut", "n_h"]], on=["coddep", "anneemut"])
print("Bloc 3 - échantillon :", len(ech),
      "| strates en écart :", (verif["tire"] != verif["n_h"]).sum(),
      "| idmutation uniques :", ech["idmutation"].is_unique)


# =============================================================================
# BLOC 4 - TABLES CIBLES : PRIX MÉDIAN AU M2
# =============================================================================
# Six tables : 2 mailles (commune, bassin de vie) x 3 pas de temps (mois,
# trimestre, année). Une ligne = une zone x une période, avec par type de bien
# le nombre de ventes et le prix médian au m2 (recalculé à partir des ventes).
# Toutes les ventes retenues, pas l'échantillon. Aucun seuil de ventes appliqué :
# le seuil retenu (20 ventes par bassin, par année et par type de bien, EDA A2)
# est appliqué à la modélisation, à partir de n_maison et n_appart.
# Géographie constante : code commune 2026. Tables par commune : arrondissements
# de Paris, Lyon et Marseille séparés. Tables par bassin de vie : arrondissements
# rattachés au bassin de leur ville (par la table de liaison du bloc 1).

# --- Population, colonnes utiles seulement ---
colonnes = ["idmutation", "anneemut", "moismut", "l_codinsee", "libtypbien", "valeurfonc", "sbati"]
pop = pd.read_parquet(fichier_pop, columns=colonnes)

# --- Prix au m2 et règles d'exclusion de l'EDA ---
pop["prix_m2"] = pop["valeurfonc"] / pop["sbati"]
pop = pop[pop["valeurfonc"].notna() & (pop["valeurfonc"] > 0)
          & (pop["prix_m2"] >= 50) & (pop["prix_m2"] <= 50000)].copy()
pop["type_bien"] = pop["libtypbien"].map({"UNE MAISON": "maison", "UN APPARTEMENT": "appart"})

# --- Rattachement à la commune 2026 et au bassin de vie (table du bloc 1) ---
liaison = pd.read_parquet(DATA_OUT / "liaison_commune_bv2022.parquet")
pop = pop.merge(liaison[["CODGEO_INI", "CODGEO_2026", "BV2022"]],
                left_on="l_codinsee", right_on="CODGEO_INI", how="left", validate="many_to_one")
arrondissement = pop["l_codinsee"].str.match(r"^(751\d\d|6938\d|132\d\d)$")
pop["code_commune"] = pop["CODGEO_2026"].where(~arrondissement, pop["l_codinsee"])

# --- Périodes ---
pop["annee"] = pop["anneemut"]
pop["mois"] = pop["moismut"]
pop["trimestre"] = (pop["moismut"] - 1) // 3 + 1


# --- Calcul d'une table : zone x période, maisons et appartements en colonnes ---
def table_cible(pop, maille, periode):
    """Prix médian au m2 et nombre de ventes par zone x période, maisons et appartements."""
    cles = [maille] + periode
    t = (pop.groupby(cles + ["type_bien"])["prix_m2"]
            .agg(n="count", prix_m2_med="median")
            .unstack("type_bien"))
    t.columns = [f"{stat}_{typ}" for stat, typ in t.columns]
    t = t.reset_index()
    for col in ["n_maison", "n_appart"]:
        t[col] = t[col].fillna(0).astype(int)
    t[["prix_m2_med_maison", "prix_m2_med_appart"]] = t[["prix_m2_med_maison", "prix_m2_med_appart"]].round(0)
    return t[cles + ["n_appart", "n_maison", "prix_m2_med_appart", "prix_m2_med_maison"]]


# --- Les six tables, avec contrôle : aucune vente perdue ---
print("Bloc 4 - ventes après exclusion :", len(pop), "| ventes sans bassin :", pop["BV2022"].isna().sum())
mailles = {"commune": "code_commune", "bv": "BV2022"}
pas_de_temps = {"mois": ["annee", "mois"], "trimestre": ["annee", "trimestre"], "annee": ["annee"]}
for nom_maille, maille in mailles.items():
    for nom_pas, periode in pas_de_temps.items():
        t = table_cible(pop, maille, periode)
        total = t["n_appart"].sum() + t["n_maison"].sum()
        t.to_parquet(DATA_OUT / f"cible_{nom_maille}_{nom_pas}.parquet", index=False)
        print(f"  cible_{nom_maille}_{nom_pas} : {len(t)} lignes | aucune vente perdue : {total == len(pop)}")


# =============================================================================
# BLOC 5 - FILOSOFI : REVENUS DES MÉNAGES PAR COMMUNE (2014-2021)
# =============================================================================
# Un fichier par millésime : Excel de 2014 à 2017 (en-tête ligne 6), CSV ";"
# de 2018 à 2021 (virgule décimale et "s" pour le secret en 2020-2021).
# On garde les colonnes présentes les 8 années. Une ligne = commune x année.

dossier_filo = DATA_RAW / "Filo_disp_com"


def en_nombre(df, cols):
    """Convertit des colonnes en nombres (virgule -> point, 's' -> vide)."""
    return df[cols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False),
                                                  errors="coerce"))


# --- Lecture de chaque millésime, année retirée des noms (Q214 -> Q2) ---
tableaux = []
for f in sorted(dossier_filo.glob("FILO*DISP_COM*")):
    annee = int(re.search(r"20\d\d", f.name).group())
    aa = str(annee)[2:]
    if f.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(f, sheet_name="ENSEMBLE", header=5, dtype={"CODGEO": str})
    else:
        df = pd.read_csv(f, sep=";", dtype={"CODGEO": str})
    df.columns = [c[:-2] if c.endswith(aa) and c != "CODGEO" else c for c in df.columns]
    df["annee"] = annee
    # 2014 : PACT reconstruit (PTSAC 2014 = PTSA + PCHO à partir de 2015)
    if annee == 2014:
        x = en_nombre(df, ["PTSAC", "PBEN"])
        df["PACT"] = x["PTSAC"] + x["PBEN"]
    tableaux.append(df)

# --- Colonnes communes aux 8 années, empilement, conversion en nombres ---
communes = set(tableaux[0].columns)
for df in tableaux[1:]:
    communes = communes & set(df.columns)
communes = ["CODGEO", "annee"] + sorted(communes - {"CODGEO", "annee"})
filo = pd.concat([df[communes] for df in tableaux], ignore_index=True)
cols_num = [c for c in filo.columns if c != "CODGEO"]
filo[cols_num] = en_nombre(filo, cols_num)

# --- Renommage pour la jointure, contrôles et enregistrement ---
filo = filo.rename(columns={"CODGEO": "code_commune", "Q2": "niveau_vie_median"})
print("Bloc 5 - millésimes :", len(tableaux),
      "| lignes :", len(filo),
      "| doublons commune x année :", filo.duplicated(["code_commune", "annee"]).sum())
filo.to_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet", index=False)


# =============================================================================
# BLOC 6 - MÉNAGES : STRUCTURE DES MÉNAGES PAR COMMUNE (RECENSEMENT 2022)
# =============================================================================
# Base IRIS additionnée par commune (Paris, Lyon, Marseille par arrondissement).
# On additionne les effectifs puis on divise, jamais de moyenne des ratios.
# Effectifs bruts conservés pour recalculer les ratios au bassin de vie.
# Ratios entre 0 et 1 ; vides pour une commune sans ménage.

# --- Lecture, codes en texte ---
chemin_menages = (DATA_RAW / "base-ic-couples-familles-menages-2022_csv"
                  / "base-ic-couples-familles-menages-2022.CSV")
codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
menages = pd.read_csv(chemin_menages, sep=";", dtype=codes_texte)
effectifs = ["C22_MEN", "C22_PMEN", "C22_MENPSEUL", "C22_MENFAMMONO",
             "C22_NE24F3", "C22_NE24F4P", "C22_MENCOUPSENF"]

# --- Somme des IRIS par commune ---
menages_com = menages.groupby("COM")[effectifs].sum().reset_index()

# --- Ratios (dénominateur vide si aucun ménage) ---
men = menages_com["C22_MEN"].where(menages_com["C22_MEN"] > 0)
menages_com["taux_personnes_seules"] = menages_com["C22_MENPSEUL"] / men
menages_com["part_familles_monoparentales"] = menages_com["C22_MENFAMMONO"] / men
menages_com["part_familles_nombreuses"] = (menages_com["C22_NE24F3"] + menages_com["C22_NE24F4P"]) / men
menages_com["taille_moyenne_menage"] = menages_com["C22_PMEN"] / men
menages_com["part_couples_sans_enfants"] = menages_com["C22_MENCOUPSENF"] / men

# --- Enregistrement ---
menages_com = menages_com.rename(columns={"COM": "code_commune"})
menages_com["annee_rp"] = 2022
print("Bloc 6 - communes :", len(menages_com),
      "| une commune = une ligne :", menages_com["code_commune"].is_unique,
      "| communes sans ménage :", men.isna().sum())
menages_com.to_parquet(DATA_OUT / "menages_commune_2022.parquet", index=False)

# Base retravaillée pour inclure 2019 à 2022 en données brutes INSEE
ANNEES = [2019, 2020, 2021, 2022]
A_SUPPRIMER = ["IRIS", "TYP_IRIS", "MODIF_IRIS", "LAB_IRIS"]

frames = []
for an in ANNEES:
    chemin = DATA_RAW / "base-ic-couples-familles-menages-2022_csv" / f"base-ic-couples-familles-menages-{an}.CSV"
    df = pd.read_csv(chemin, sep=";", dtype={"IRIS": str, "COM": str}, low_memory=False)

    # Suppression des colonnes IRIS superflues
    df = df.drop(columns=[c for c in A_SUPPRIMER if c in df.columns])

    # Normalisation des noms de colonnes (suppression des suffixes d'année C19_ -> C_)
    suffixe = str(an)[2:]
    df.columns = [re.sub(rf"^([CP]){suffixe}_", r"\1_", c) for c in df.columns]

    # Conservation uniquement des colonnes numériques + le code commune COM
    cols_num = df.select_dtypes(include="number").columns.tolist()
    
    # Agrégation par commune
    communes = df.groupby("COM", as_index=False)[cols_num].sum(min_count=1)
    communes.insert(1, "annee", an)
    frames.append(communes)
    print(f"{an}: {len(df)} IRIS -> {len(communes)} communes")

# Concaténation de toutes les années
final = pd.concat(frames, ignore_index=True, sort=False)
final = final.rename(columns={"COM": "code_commune"}).sort_values(["code_commune", "annee"]).reset_index(drop=True)

liaison = pd.read_parquet(DATA_OUT / "liaison_commune_bv2022.parquet")

# Jointure à gauche sur le code commune
res = final.merge(
    liaison, 
    how="left", 
    left_on="code_commune", 
    right_on="CODGEO_INI"
).drop(columns=["CODGEO_INI"])

# Nettoyage et filtrage des communes sans bassin de vie attribué
res = res.dropna(subset=["BV2022"])
print(f"Lignes après jointure BV : {len(res)} | Sans BV : {res['BV2022'].isna().sum()}")

#Calcul des ratios par bassin de vie
COLS = [
    "C_MEN", "C_MENPSEUL", "C_PMEN", "C_FAM", 
    "C_FAMMONO", "C_COUPSENF", "C_NE24F3", "C_NE24F4P"
]

# Agrégation par bassin de vie et par année
g = res.groupby(["BV2022", "annee"])
bv = g[COLS].sum().reset_index()
bv.insert(2, "nb_communes", g.size().values)

# Conditions de garde pour éviter la division par zéro
men_ok = bv["C_MEN"] > 0
fam_ok = men_ok & (bv["C_FAM"] > 0)

def ratio(num, den, garde, x100=True):
    out = num / den.where(den > 0)
    return np.where(garde, out * (100 if x100 else 1), np.nan)

# Calcul des ratios
bv["taux_personnes_seules"]    = ratio(bv["C_MENPSEUL"], bv["C_MEN"], men_ok)
bv["part_familles_monoparent"] = ratio(bv["C_FAMMONO"], bv["C_FAM"], fam_ok)
bv["part_familles_nombreuses"] = ratio(bv["C_NE24F3"] + bv["C_NE24F4P"], bv["C_FAM"], fam_ok)
bv["taille_moyenne_menages"]   = ratio(bv["C_PMEN"], bv["C_MEN"], men_ok, x100=False)
bv["part_couples_sans_enfant"] = ratio(bv["C_COUPSENF"], bv["C_FAM"], fam_ok)

RATIOS = [
    "taux_personnes_seules", "part_familles_monoparent", 
    "part_familles_nombreuses", "taille_moyenne_menages", 
    "part_couples_sans_enfant"
]

# Formatage final en rettrapant les infos sur les BV
infos_bv=liaison[["CODGEO_INI", "CODGEO_2026","BV2022","TYPE_BV2022"]]
bv = bv.merge(infos_bv, on="BV2022", how="left")
cols_finales = ["CODGEO_INI", "CODGEO_2026","BV2022","TYPE_BV2022", "annee", "nb_communes", "C_MEN", "C_FAM"] + RATIOS
ratios = bv[cols_finales].round({"C_MEN": 0, "C_FAM": 0, **{r: 2 for r in RATIOS}})

# Test d'évolution 2021 à 2022
# Restriction aux années 2021 et 2022
df_21_22 = ratios[ratios["annee"].isin([2021, 2022])]

# 1. Pivotement pour avoir 2021 et 2022 côte à côte par BV
df_21_22 = ratios[ratios["annee"].isin([2021, 2022])].dropna(subset=["BV2022"])
pivot = df_21_22.drop_duplicates(subset=["BV2022", "annee"]).pivot(
    index="BV2022", columns="annee"
)

# 2. Seuils de significativité pratique (à ajuster selon tes critères)
SEUILS = {
    "taux_personnes_seules": 1.5,  # 1 point de %
    "part_familles_monoparent": 1.5,  # 1 point de %
    "part_familles_nombreuses": 1.0,  # 1 point de %
    "taille_moyenne_menages": 0.02,  # 0.05 personne/ménage
    "part_couples_sans_enfant": 1.5,  # 1 point de %
}

bilan_bv = []
nb_total_bv = len(pivot)

for r in RATIOS:
    s_2021 = pivot[r][2021]
    s_2022 = pivot[r][2022]

    # Variation absolue (|2022 - 2021|)
    diff_abs = (s_2022 - s_2021).abs()
    seuil = SEUILS[r]

    # Nombre et % de BV où la différence dépasse le seuil
    bv_depassant = (diff_abs > seuil).sum()
    pct_depassant = (bv_depassant / nb_total_bv) * 100

    bilan_bv.append(
        {
            "Ratio": r,
            "Seuil critique": f"> {seuil} pt" if "part" in r or "taux" in r else f"> {seuil} pers.",
            "Écart Max observé": diff_abs.max().round(2),
            "Écart Médian": diff_abs.median().round(3),
            "Nb BV > Seuil": bv_depassant,
            "% BV avec écart significatif": round(pct_depassant, 2),
            "% BV stables (duplication valide)": round(100 - pct_depassant, 2),
        }
    )

df_bilan_bv = pd.DataFrame(bilan_bv)
#print(f"Total des bassins de vie analysés : {nb_total_bv}\n")
# print(df_bilan_bv.to_string(index=False))

# le test démontre que la stabilité des indicateurs entre 2021 et 2022 confirme la pertinence de la réplication des données 2022 pour 2023

#Dupplication de 2022 à 2023
# Isoler les données de 2022
ratios_2022 = ratios[ratios["annee"] == 2022].copy()

# Remplacer la valeur de la colonne annee par 2023
ratios_2022["annee"] = 2023

# Empiler avec le DataFrame d'origine
ratios = pd.concat([ratios, ratios_2022], ignore_index=True)

# Trier le résultat
ratios = ratios.sort_values(["BV2022", "annee"]).reset_index(drop=True)

# Analyse sur l'évolution sur 2ans par bassin de vie
# 1. Pivotement pour aligner 2020 et 2022
df_20_22 = ratios[ratios["annee"].isin([2020, 2022])].dropna(subset=["BV2022"])
pivot_2ans = df_20_22.drop_duplicates(subset=["BV2022", "annee"]).pivot(
    index="BV2022", columns="annee"
)

# 2. Seuils adaptés pour une fenêtre de 2 ans
SEUILS_2ANS = {
    "taux_personnes_seules": 2.0,       # 2 points de %
    "part_familles_monoparent": 2.0,   # 2 points de %
    "part_familles_nombreuses": 1.5,   # 1.5 point de %
    "taille_moyenne_menages": 0.03,    # 0.03 pers./ménage
    "part_couples_sans_enfant": 2.0,   # 2 points de %
}

bilan_2ans = []
nb_total_bv = len(pivot_2ans)

for r in RATIOS:
    s_2020 = pivot_2ans[r][2020]
    s_2022 = pivot_2ans[r][2022]

    # Variation absolue sur 2 ans (|2022 - 2020|)
    diff_abs = (s_2022 - s_2020).abs()
    seuil = SEUILS_2ANS[r]

    bv_depassant = (diff_abs > seuil).sum()
    pct_depassant = (bv_depassant / nb_total_bv) * 100

    bilan_2ans.append(
        {
            "Ratio": r,
            "Seuil critique (2 ans)": f"> {seuil} pt" if "part" in r or "taux" in r else f"> {seuil} pers.",
            "Écart Max (2020-2022)": diff_abs.max().round(2),
            "Écart Médian (2020-2022)": diff_abs.median().round(3),
            "Nb BV > Seuil": bv_depassant,
            "% BV avec écart significatif": round(pct_depassant, 2),
            "% BV stables (duplication 2 ans valide)": round(100 - pct_depassant, 2),
        }
    )

df_bilan_2ans = pd.DataFrame(bilan_2ans)
#print(f"Total des bassins de vie analysés : {nb_total_bv}\n")
# print(df_bilan_2ans.to_string(index=False))

#Choix d'une extrapolation lissée de 2022 à 2024
# Isolation des données 2020 et 2022
r_2020 = ratios[ratios["annee"] == 2020].set_index("BV2022")
r_2022 = ratios[ratios["annee"] == 2022].set_index("BV2022")

# Alignement des bassins de vie communs
bv_communs = r_2022.index.intersection(r_2020.index)
r_2020 = r_2020.loc[bv_communs]
r_2022 = r_2022.loc[bv_communs]

# Calcul du delta sur 2 ans (2020-2022)
deltas = r_2022[RATIOS] - r_2020[RATIOS]

# Application des bornes de sécurité (clipping)
BORNES = {
    "taux_personnes_seules": 1.5,
    "part_familles_monoparent": 1.5,
    "part_familles_nombreuses": 1.0,
    "taille_moyenne_menages": 0.03,
    "part_couples_sans_enfant": 1.5,
}

deltas_bornes = deltas.copy()
for col, lim in BORNES.items():
    deltas_bornes[col] = deltas_bornes[col].clip(lower=-lim, upper=lim)

# Construction de la table 2024
r_2024 = r_2022.copy()
r_2024[RATIOS] = (r_2022[RATIOS] + deltas_bornes).round(2)
r_2024["annee"] = 2024


# Reconstitution table finale complète
ratios_complets = pd.concat([ratios, r_2024.reset_index()], ignore_index=True)
ratios_complets = ratios_complets.sort_values(["BV2022", "annee"]).reset_index(drop=True)

fichier_final = DATA_OUT / "ratios_menages_bv_2019_2024.csv"

ratios_complets.to_csv(fichier_final, sep=";", index=False, encoding="utf-8-sig", decimal=",")

# =============================================================================
# BLOC 7 - PARC DE LOGEMENT PAR COMMUNE (RECENSEMENT 2022)
# =============================================================================
# Base IRIS nettoyée (colonnes outre-mer, chauffage, LAB_IRIS, TYP_IRIS ; IRIS
# sans logement), additionnée par commune (Paris, Lyon, Marseille par
# arrondissement). 19 indicateurs en % ou en moyenne. Effectifs bruts conservés
# pour recalculer les indicateurs au bassin de vie.

# --- Lecture, codes en texte ---
chemin_logement = DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV"
codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
logement = pd.read_csv(chemin_logement, sep=";", dtype=codes_texte)

# --- Colonnes inutiles et IRIS sans logement ---
colonnes_dom = ["P22_RP_ELEC", "P22_RP_EAUCH", "P22_RP_BDWC", "P22_RP_CHOS", "P22_RP_CLIM",
                "P22_RP_TTEGOU", "P22_RP_HABFOR", "P22_RP_CASE", "P22_RP_MIBOIS", "P22_RP_MIDUR"]
colonnes_chauffage = ["P22_RP_CGAZV", "P22_RP_CFIOUL", "P22_RP_CELEC", "P22_RP_CGAZB", "P22_RP_CAUT"]
logement = logement.drop(columns=colonnes_dom + colonnes_chauffage + ["LAB_IRIS", "TYP_IRIS"])
logement = logement[logement["P22_LOG"] > 0]

# --- Somme des IRIS par commune ---
colonnes_effectifs = [c for c in logement.columns if c.startswith("P22_") or c.startswith("C22_")]
commune = logement.groupby("COM")[colonnes_effectifs].sum()
commune["nb_iris"] = logement.groupby("COM")["IRIS"].count()
commune = commune.reset_index()


# --- 19 indicateurs (vides si le dénominateur vaut 0) ---
def ratio(numerateur, denominateur):
    """Part en % ; vide si le dénominateur vaut 0."""
    return (numerateur / denominateur.replace(0, np.nan)) * 100


c = commune
log_, rp, men = c["P22_LOG"], c["P22_RP"], c["P22_MEN"]
c["taux_vacance"] = ratio(c["P22_LOGVAC"], log_)
c["part_rsec"] = ratio(c["P22_RSECOCC"], log_)
c["part_maisons"] = ratio(c["P22_MAISON"], log_)
c["part_apparts"] = ratio(c["P22_APPART"], log_)
c["part_moins40m2"] = ratio(c["P22_RP_M30M2"] + c["P22_RP_3040M2"], rp)
c["part_100m2_plus"] = ratio(c["P22_RP_100120M2"] + c["P22_RP_120M2P"], rp)
c["nb_pieces_moyen"] = c["P22_NBPI_RP"] / rp.replace(0, np.nan)
c["part_avant1946"] = ratio(c["P22_RP_ACH1919"] + c["P22_RP_ACH1945"], rp)
c["part_2006_2019"] = ratio(c["P22_RP_ACH2019"], rp)
c["part_proprio"] = ratio(c["P22_RP_PROP"], rp)
c["part_locataires"] = ratio(c["P22_RP_LOC"], rp)
c["part_hlm"] = ratio(c["P22_RP_LOCHLMV"], rp)
c["part_emmenages_2ans"] = ratio(c["P22_MEN_ANEM0002"], men)
c["anciennete_moyenne"] = c["P22_ANEM_RP"] / rp.replace(0, np.nan)
c["taille_menage"] = c["P22_NPER_RP"] / rp.replace(0, np.nan)
c["part_sans_voiture"] = 100 - ratio(c["P22_RP_VOIT1P"], men)
c["part_2voitures"] = ratio(c["P22_RP_VOIT2P"], men)
c["part_parking"] = ratio(c["P22_RP_GARL"], men)
c["part_suroccupes"] = ratio(c["C22_RP_SUROCC_MOD"] + c["C22_RP_SUROCC_ACC"], rp)
c = c.copy()   # réorganise le tableau en mémoire après l'ajout des colonnes
indicateurs = ["taux_vacance", "part_rsec", "part_maisons", "part_apparts",
               "part_moins40m2", "part_100m2_plus", "nb_pieces_moyen",
               "part_avant1946", "part_2006_2019",
               "part_proprio", "part_locataires", "part_hlm",
               "part_emmenages_2ans", "anciennete_moyenne",
               "taille_menage", "part_sans_voiture", "part_2voitures", "part_parking", "part_suroccupes"]

# --- Colonnes d'aide ---
c["dep"] = np.where(c["COM"].str[:2] == "97", c["COM"].str[:3], c["COM"].str[:2])
c["alsace_moselle"] = c["dep"].isin(["57", "67", "68"])   # absente de DVF+ : signalée, pas supprimée
c["petite_commune"] = c["P22_LOG"] < 50                    # indicateurs fragiles
c["log_nb_logements"] = np.log10(c["P22_LOG"])
c["annee_rp"] = 2022

# --- Contrôle : parts entre 0 et 100 ---
for col in indicateurs:
    if col.startswith("part_") or col == "taux_vacance":
        hors_bornes = ((c[col] < -0.01) | (c[col] > 100.01)).sum()
        if hors_bornes > 0:
            print("Attention :", col, "a", hors_bornes, "valeurs hors de [0, 100]")

# --- Enregistrement : identifiants, effectifs bruts, indicateurs, colonnes d'aide ---
effectifs_utiles = ["P22_LOG", "P22_RP", "P22_MEN",
                    "P22_LOGVAC", "P22_RSECOCC", "P22_MAISON", "P22_APPART",
                    "P22_RP_M30M2", "P22_RP_3040M2", "P22_RP_100120M2", "P22_RP_120M2P", "P22_NBPI_RP",
                    "P22_RP_ACH1919", "P22_RP_ACH1945", "P22_RP_ACH2019",
                    "P22_RP_PROP", "P22_RP_LOC", "P22_RP_LOCHLMV",
                    "P22_MEN_ANEM0002", "P22_ANEM_RP", "P22_NPER_RP",
                    "P22_RP_VOIT1P", "P22_RP_VOIT2P", "P22_RP_GARL",
                    "C22_RP_SUROCC_MOD", "C22_RP_SUROCC_ACC"]
colonnes_finales = (["COM", "dep", "annee_rp", "nb_iris", "log_nb_logements"]
                    + effectifs_utiles + indicateurs + ["alsace_moselle", "petite_commune"])
resultat = c[colonnes_finales].rename(columns={"COM": "code_commune"})
print("Bloc 7 - communes :", len(resultat),
      "| une commune = une ligne :", resultat["code_commune"].is_unique)
resultat.to_parquet(DATA_OUT / "logement_commune_2022.parquet", index=False)


# =============================================================================
# BLOC 8 - DÉLINQUANCE ENREGISTRÉE PAR COMMUNE (SSMSI, 2016-2025)
# =============================================================================
# Paris, Lyon, Marseille : arrondissements gardés, ville entière retirée.
# Valeurs masquées (secret statistique) remplacées par le complément
# départemental, jamais par 0. Format large : une ligne = commune x année.
# Nombres, population et logements conservés pour recalculer les taux au
# bassin de vie. Taux pour 1 000 habitants (1 000 logements pour les cambriolages).

delinq = pd.read_parquet(DATA_RAW / "dataset_delinquance_communale.parquet")

# --- Retrait de la ville entière de Paris, Lyon et Marseille ---
delinq = delinq[~delinq["CODGEO_2026"].isin(["75056", "69123", "13055"])]

# --- Nombre publié, sinon complément départemental ---
delinq["nombre_estime"] = delinq["nombre"].fillna(delinq["complement_info_nombre"])


# --- Format large : une colonne par indicateur ---
def nom_court(libelle):
    """'Vols d'accessoires sur véhicules' -> 'vols_d_accessoires_sur_vehicules'"""
    sans_accent = unicodedata.normalize("NFKD", libelle).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sans_accent.lower()).strip("_")


large = delinq.pivot(index=["CODGEO_2026", "annee"], columns="indicateur", values="nombre_estime")
large.columns = ["nb_" + nom_court(c) for c in large.columns]
denominateurs = delinq.groupby(["CODGEO_2026", "annee"])[["insee_pop", "insee_log"]].first()
masques = ((delinq["est_diffuse"] == "ndiff")
           .groupby([delinq["CODGEO_2026"], delinq["annee"]]).sum()
           .rename("n_indicateurs_estimes"))
large = large.join(denominateurs).join(masques).reset_index()

# --- Taux pour 1 000 (vides si aucun habitant ou logement) ---
pop = large["insee_pop"].where(large["insee_pop"] > 0)
log = large["insee_log"].where(large["insee_log"] > 0)
for col in [c for c in large.columns if c.startswith("nb_")]:
    denominateur = log if col == "nb_cambriolages_de_logement" else pop
    large["taux_" + col[3:]] = large[col] / denominateur * 1000

# --- Enregistrement ---
large = large.rename(columns={"CODGEO_2026": "code_commune"})
print("Bloc 8 - lignes :", len(large),
      "| une commune x année = une ligne :", not large.duplicated(["code_commune", "annee"]).any(),
      "| nombres vides :", large.filter(like="nb_").isna().sum().sum())
large.to_parquet(DATA_OUT / "delinquance_commune_annee.parquet", index=False)


# =============================================================================
# BLOC 9 - LOVAC : LOGEMENTS PRIVÉS VACANTS PAR COMMUNE (2020-2026)
# =============================================================================
# Millésime N = situation au 1er janvier N-1 ; pour une vente de l'année N,
# on utilise le millésime N. Secret statistique ("s") -> valeur vide.
# Format long : une ligne = commune x millésime. Taux cachés complétés par le
# taux de l'intercommunalité (EPCI), signalés par taux_impute.

# --- Lecture (tout en texte à cause des "s"), secret -> vide, conversion en nombres ---
lovac = pd.read_excel(DATA_RAW / "lovac-open-data-2020-a-2026.xlsx", sheet_name="COM open data", dtype=str)
colonnes_chiffres = [c for c in lovac.columns if c.startswith("pp_") or c.startswith("ff_")]
for col in colonnes_chiffres:
    lovac[col] = pd.to_numeric(lovac[col].replace("s", np.nan))

# --- Format long : un millésime à la fois ---
morceaux = []
for annee in range(2020, 2027):
    suffixe = str(annee)[2:]
    morceau = pd.DataFrame({
        "COM": lovac["CODGEO_26"],
        "EPCI": lovac["EPCI_26"],
        "millesime": annee,
        "vacants": lovac["pp_vacant_" + suffixe],
        "vacants_2ans": lovac["pp_vacant_plus_2ans_" + suffixe],
    })
    # Pas de parc privé total pour le millésime 2026
    if "ff_pp_total_" + suffixe in lovac.columns:
        morceau["parc_prive"] = lovac["ff_pp_total_" + suffixe]
    else:
        morceau["parc_prive"] = np.nan
    morceaux.append(morceau)
lovac_long = pd.concat(morceaux, ignore_index=True)
lovac_long["annee_situation"] = lovac_long["millesime"] - 1
lovac_long["annee_vente_associee"] = lovac_long["millesime"]

# --- Indicateurs en % (ratio défini au bloc 7) et signalements ---
lovac_long["taux_vacance_lovac"] = ratio(lovac_long["vacants"], lovac_long["parc_prive"])
lovac_long["taux_vacance_2ans"] = ratio(lovac_long["vacants_2ans"], lovac_long["parc_prive"])
lovac_long["part_longue_parmi_vacants"] = ratio(lovac_long["vacants_2ans"], lovac_long["vacants"])
lovac_long["apres_rupture"] = lovac_long["millesime"] >= 2024        # fin de la taxe d'habitation
lovac_long["dep"] = np.where(lovac_long["COM"].str[:2] == "97",
                             lovac_long["COM"].str[:3], lovac_long["COM"].str[:2])
lovac_long["alsace_moselle"] = lovac_long["dep"].isin(["57", "67", "68"])
lovac_long["mayotte"] = lovac_long["dep"] == "976"

# --- Taux cachés : taux de l'EPCI (sommes des communes connues, même millésime) ---
connues = lovac_long[lovac_long["taux_vacance_lovac"].notna()]
taux_epci = connues.groupby(["EPCI", "millesime"])[["vacants", "parc_prive"]].sum().reset_index()
taux_epci["taux_vacance_epci"] = ratio(taux_epci["vacants"], taux_epci["parc_prive"])
lovac_long = lovac_long.merge(taux_epci[["EPCI", "millesime", "taux_vacance_epci"]],
                              on=["EPCI", "millesime"], how="left")
lovac_long["taux_impute"] = lovac_long["taux_vacance_lovac"].isna() & lovac_long["taux_vacance_epci"].notna()
lovac_long["taux_vacance_complete"] = lovac_long["taux_vacance_lovac"].fillna(lovac_long["taux_vacance_epci"])

# --- Contrôles et enregistrement ---
hors_bornes = sum(((lovac_long[col] < 0) | (lovac_long[col] > 100)).sum()
                  for col in ["taux_vacance_lovac", "taux_vacance_2ans",
                              "part_longue_parmi_vacants", "taux_vacance_complete"])
lovac_long = lovac_long.rename(columns={"COM": "code_commune"})
print("Bloc 9 - lignes :", len(lovac_long),
      "| une commune x millésime = une ligne :", not lovac_long.duplicated(["code_commune", "millesime"]).any(),
      "| taux hors [0, 100] :", hors_bornes,
      "| taux complétés par l'EPCI :", lovac_long["taux_impute"].sum())
lovac_long.to_parquet(DATA_OUT / "lovac_commune_annee.parquet", index=False)


# =============================================================================
# BLOC 10 - ÉQUIPEMENTS (BPE 2015 ET 2020) PAR COMMUNE
# =============================================================================
# Source : table construite par CB (ancienne table cible + BPE 2015 et 2020).
# Prix et nombres de ventes retirés (remplacés par le bloc 4).
# Une ligne = commune (géographie 2026, arrondissements séparés) x millésime BPE.
# 152 indicateurs : 1 présent, 0 absent, vide inconnu (gardé tel quel).
# BPE 2015 -> ventes 2016-2020 ; BPE 2020 -> ventes 2021-2025.

# --- Lecture des seules colonnes utiles ---
colonnes_utiles = ["code_commune", "code_commune_2026", "bpe_annee_source"]
equip = pd.read_csv(DATA_RAW / "cible_commune_annee_type_BPE_152_indicateurs.csv",
                    usecols=lambda c: c in colonnes_utiles or c.startswith("bpe_presence_"),
                    dtype={"code_commune": str, "code_commune_2026": str})
colonnes_presence = [c for c in equip.columns if c.startswith("bpe_presence_")]

# --- Lignes avec un millésime BPE, code commune en géographie 2026 ---
equip = equip.dropna(subset=["bpe_annee_source"]).copy()
equip["millesime_bpe"] = equip["bpe_annee_source"].astype(int)
arrondissement = equip["code_commune"].str.match(r"^(751\d\d|6938\d|132\d\d)$")
equip["code_commune_geo2026"] = equip["code_commune_2026"].where(~arrondissement, equip["code_commune"])

# --- Contrôle puis une ligne par commune x millésime ---
cles = ["code_commune_geo2026", "millesime_bpe"]
ecarts = equip.groupby(cles)[colonnes_presence].nunique(dropna=False).max(axis=1)
equip = equip.drop_duplicates(cles)[cles + colonnes_presence]
equip = equip.rename(columns={"code_commune_geo2026": "code_commune"})

# --- Enregistrement ---
print("Bloc 10 - lignes :", len(equip),
      "| une commune x millésime = une ligne :", not equip.duplicated(["code_commune", "millesime_bpe"]).any(),
      "| communes où les indicateurs diffèrent :", (ecarts > 1).sum())
equip.to_parquet(DATA_OUT / "equipements_commune_millesime.parquet", index=False)


# =============================================================================
# BLOC 11 - TAUX DES CRÉDITS IMMOBILIERS (BANQUE DE FRANCE, 1998-2026)
# =============================================================================
# Série nationale mensuelle, en %, observée en fin de mois.
# Trois tables : mois, trimestre et année (moyenne des taux mensuels),
# avec les mêmes clés de période que les tables cibles du bloc 4.

# --- Lecture : 6 lignes de métadonnées, pas d'en-tête ---
taux = pd.read_csv(DATA_RAW / "Taux d'intérêt annuel des crédits immobiliers.csv",
                   sep=";", skiprows=6, header=None, names=["date", "taux"])

# --- Conversion (virgule décimale), ordre chronologique, périodes ---
taux["taux"] = pd.to_numeric(taux["taux"].str.replace(",", ".", regex=False))
taux["date"] = pd.to_datetime(taux["date"])
taux = taux.sort_values("date").reset_index(drop=True)
taux["annee"] = taux["date"].dt.year
taux["mois"] = taux["date"].dt.month
taux["trimestre"] = (taux["mois"] - 1) // 3 + 1

# --- Trois tables et enregistrement ---
taux_mois = taux[["annee", "mois", "taux"]]
taux_trimestre = taux.groupby(["annee", "trimestre"], as_index=False)["taux"].mean().round(2)
taux_annee = taux.groupby("annee", as_index=False)["taux"].mean().round(2)
taux_mois.to_parquet(DATA_OUT / "taux_credit_mois.parquet", index=False)
taux_trimestre.to_parquet(DATA_OUT / "taux_credit_trimestre.parquet", index=False)
taux_annee.to_parquet(DATA_OUT / "taux_credit_annee.parquet", index=False)
print("Bloc 11 - mois :", len(taux_mois),
      "| mois en double :", taux.duplicated(["annee", "mois"]).sum(),
      "| trimestres :", len(taux_trimestre), "| années :", len(taux_annee))
