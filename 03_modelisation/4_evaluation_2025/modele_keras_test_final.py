# =============================================================================
# modele_keras_test_final.py : test final 2025 du candidat par vente (H7v4, réseau optimisé)
# =============================================================================
# DÉCLARATION, écrite avant d'ouvrir 2025 (12/11/2026)
#   Candidat unique du modèle par vente : H7v4, réseau Keras optimisé (réglages de
#   meilleurs_reglages_keras.json), variables de H7v3 + volume et élan + capacité d'achat
#   + trimestres précédents. Seul modèle par vente retenu contre la naïve en 2023 et 2024
#   pour les deux types de bien (évaluation B), le meilleur des versions.
#   Référence : la naïve (prix médian N-1 du sous-bassin).
#   Deux évaluations, comme dans H7 : B (chaque vente) et C (médiane du sous-bassin, panier fixe).
#   Critère de réussite pour chacune : gain >= 3 %, IC 95 % haut < 0 et rang-bisériale < 0,
#   par type de bien. Résultat rapporté tel quel, aucune variante sur 2025.
#
# Ce que fait ce fichier :
#   1. prépare les ventes et la table exactement comme le notebook H7 (seuils fixés sur 2017-2022) ;
#      les ventes de 2025 viennent de ventes_test_2025.parquet (bloc 3 bis de la préparation) ;
#   2. CONTROLE_2024 = True : relance H7v4 sur la fenêtre 2024 et vérifie qu'il redonne le gain
#      de H7 (maisons, évaluation B) ; sinon le script s'arrête AVANT d'ouvrir 2025 ;
#   3. apprend sur les ventes 2017-2024 et prévoit 2025 (une seule ouverture) ;
#   4. evaluer_ventes / tester_ventes / evaluer_panier de commun.py, biais, règle ; CSV (le parcours court
#      le trace ensuite dans MLflow).
#
# Lancement, depuis la racine du projet :  python 03_modelisation/4_evaluation_2025/modele_keras_test_final.py
# (le socle doit être installé : pip install -e . depuis la racine, une seule fois)
# Durée : deux entraînements du réseau si CONTROLE_2024 = True, un seul sinon.

import json
import time
import warnings

import numpy as np
import pandas as pd

import commun
from commun import (CIBLE, CLES, charger_table, charger_ventes_brutes, fenetres, rattacher_ventes,
                    evaluer_ventes, tester_ventes, panier_fixe, evaluer_panier, mediane_ponderee,
                    _par_sous_bassin)
from gabarit_keras import prevoir_keras

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
NOM = "H7v4 – Keras – H7v3 + volume, élan, capacité d'achat et trimestres (test 2025)"
ANNEE_DEBUT_H7 = 2017
SEUIL_GAIN = 3
CONTROLE_2024 = True
GAIN_H7V4_MAISONS_2024 = 7.29          # H7v4 optimisé, maisons 2024, évaluation B (résultat de H7)
TOLERANCE = 0.5                        # points de % : le réseau varie un peu d'un entraînement à l'autre

# -----------------------------------------------------------------------------
# 1. Données : table, ventes 2014-2024 (apprentissage) et ventes 2025 (test)
# -----------------------------------------------------------------------------
table = charger_table()
ventes_brutes = charger_ventes_brutes()                                    # 2014-2024
ventes_test = pd.read_parquet(commun.DATA_OUT / "ventes_test_2025.parquet")
assert ventes_brutes["anneemut"].max() == 2024 and (ventes_test["anneemut"] == 2025).all()

toutes = pd.concat([ventes_brutes, ventes_test], ignore_index=True)
ventes = rattacher_ventes(toutes.rename(columns={"anneemut": "annee"}), table)
assert ventes["prix_n1"].notna().all() and ventes["BV2022"].notna().all()
print("Ventes rattachées par année :", ventes.groupby("annee").size().loc[ANNEE_DEBUT_H7:].to_dict())

# Constats de H7 : seuils fixés sur l'apprentissage de la première fenêtre (ventes 2017-2022)
app_constat = ventes[(ventes["annee"] >= ANNEE_DEBUT_H7) & (ventes["annee"] <= 2022)]

# -----------------------------------------------------------------------------
# 2. Variables du bien (copie de H7, section 2.1)
# -----------------------------------------------------------------------------
SEUIL_STERR = app_constat.loc[app_constat["type_bien"] == "maison", "sterr"].quantile(0.99)
BORNES_X, BORNES_Y = (100_000, 1_250_000), (6_030_000, 7_120_000)
COLONNES_PIECES = ["pieces_1", "pieces_2", "pieces_3", "pieces_4", "pieces_5_plus"]
COLONNES_DEP = ["dep_0", "dep_1", "dep_2", "dep_3_plus"]
BLOC_BIEN = ["maison", "log_sbati", "log1p_sterr"] + COLONNES_PIECES + COLONNES_DEP
BLOC_COORD = ["geompar_x", "geompar_y"]


def coordonnees_douteuses(v):
    return (~v["geompar_x"].between(*BORNES_X)) | (~v["geompar_y"].between(*BORNES_Y))


valides = app_constat[~coordonnees_douteuses(app_constat)]
CENTRES = valides.groupby("sous_bassin_de_vie")[BLOC_COORD].median()


def preparer_bien(v):
    v = v.copy()
    v["log_sbati"] = np.log(v["sbati"])
    terrain = v["sterr"].where(v["type_bien"] == "maison", 0).clip(upper=SEUIL_STERR)
    v["log1p_sterr"] = np.log1p(terrain)
    for k, col in enumerate(COLONNES_PIECES, start=1):
        v[col] = (v["nb_pieces"] == k).astype(int)
    dependances = v["nblocdep"].clip(upper=3)
    for k, col in enumerate(COLONNES_DEP):
        v[col] = (dependances == k).astype(int)
    douteuses = coordonnees_douteuses(v)
    for c in BLOC_COORD:
        v.loc[douteuses, c] = v.loc[douteuses, "sous_bassin_de_vie"].map(CENTRES[c])
    return v


ventes = preparer_bien(ventes)
ventes_brutes = preparer_bien(ventes_brutes)          # pour le panier fixe (biens vendus en N-1 <= 2024)
assert ventes[BLOC_BIEN].notna().all().all(), "case vide dans le bloc du bien"

# -----------------------------------------------------------------------------
# 3. Territoire et prix N-1 (copie de H7, sections 4.4 et 4.7)
# -----------------------------------------------------------------------------
PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
        "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
VACANCE = ["taux_vacance_complete"]
DENSITE = ["densite_population"]
LOG_TERRITOIRE = [v for v in PARC + VACANCE + DENSITE
                  if v in ["densite_population", "part_sans_voiture", "part_moins40m2", "part_rsec", "part_apparts"]]
BLOC_TERRITOIRE = ["log_" + v if v in LOG_TERRITOIRE else v for v in PARC + VACANCE + DENSITE]


def preparer_territoire(t):
    t = t.copy()
    assert (t[LOG_TERRITOIRE] > 0).all().all(), "valeur nulle ou négative dans une variable passée en log"
    for v in LOG_TERRITOIRE:
        t["log_" + v] = np.log(t[v])
    return t


table = preparer_territoire(table)
ventes = preparer_territoire(ventes)
table["log_prix_n1"] = np.log(table["prix_n1"])
ventes["log_prix_n1"] = np.log(ventes["prix_n1"])

# -----------------------------------------------------------------------------
# 4. Le « quand » du niveau 2 : P9 et P7 (copie de H7, section 4.10)
# -----------------------------------------------------------------------------
CLES_INDICATEURS = ["annee", "sous_bassin_de_vie", "type_bien"]


def cles_de(t):
    if "sous_bassin_de_vie" in t.columns:
        return t[CLES_INDICATEURS].reset_index(drop=True)
    return t.index.to_frame(index=False)[CLES_INDICATEURS]


def ajouter(t, valeurs, colonnes):
    t = t.copy()
    k = cles_de(t).merge(valeurs[CLES_INDICATEURS + colonnes], on=CLES_INDICATEURS, how="left", validate="many_to_one")
    t[colonnes] = k[colonnes].to_numpy()
    return t


P9 = pd.read_parquet(commun.DATA_OUT / "fe_volume_elan_sbv.parquet")
COLONNES_P9 = ["evol_prix_n2_n1", "rotation_parc_type"]
P7 = pd.read_parquet(commun.DATA_OUT / "fe_capacite_achat_sbv.parquet")


def preparer_quand_niveau2(t):
    t = ajouter(ajouter(t, P9, COLONNES_P9), P7, ["taux_effort"])
    t["evol_n2_inconnue"] = t["evol_prix_n2_n1"].isna().astype(int)
    t["evol_prix_n2_n1"] = t["evol_prix_n2_n1"].fillna(0)
    for v in ["nb_ventes_n1", "rotation_parc_type", "taux_effort"]:
        assert (t[v] > 0).all(), f"{v} : valeur vide, nulle ou négative"
        t["log_" + v] = np.log(t[v])
    return t


table = preparer_quand_niveau2(table)
ventes = preparer_quand_niveau2(ventes)
BLOC_H5 = ["log_nb_ventes_n1", "log_rotation_parc_type", "evol_prix_n2_n1", "evol_n2_inconnue"]
BLOC_H6 = ["log_taux_effort"]

# -----------------------------------------------------------------------------
# 5. Trimestres précédents (copie de H7, section 4.13) : ventes 2014-2024 seulement
# -----------------------------------------------------------------------------
SEUIL_TRIMESTRE = 30
TRIMESTRES = [3, 4]


def stats_trimestre(v, lieu):
    g = v.groupby(lieu + ["anneemut", "trimestre"])
    return pd.DataFrame({"mediane": g.apply(lambda e: mediane_ponderee(e["prix_m2"], e["w_h"]), include_groups=False),
                         "volume": g["w_h"].sum(), "ventes": g.size()}).reset_index()


def glissements(s, lieu):
    sorties = []
    for q in TRIMESTRES:
        e = s[s["trimestre"] == q].drop(columns="trimestre")
        n1 = e.assign(annee=e["anneemut"] + 1).drop(columns="anneemut")
        n2 = e.assign(annee=e["anneemut"] + 2).drop(columns="anneemut")
        m = n1.merge(n2, on=lieu + ["annee"], suffixes=("_n1", "_n2"))
        sorties.append(pd.DataFrame({f"gliss_prix_t{q}": (m["mediane_n1"] / m["mediane_n2"] - 1).to_numpy() * 100,
                                     f"gliss_volume_t{q}": (m["volume_n1"] / m["volume_n2"] - 1).to_numpy() * 100,
                                     f"fiable_t{q}": (m[["ventes_n1", "ventes_n2"]].min(axis=1) >= SEUIL_TRIMESTRE).to_numpy()},
                                    index=pd.MultiIndex.from_frame(m[lieu + ["annee"]])))
    return pd.concat(sorties, axis=1).reset_index()


assert ventes_brutes["anneemut"].max() == 2024, "aucune vente de 2025 ne doit servir aux trimestres"
vb = ventes_brutes.assign(departement=ventes_brutes["sous_bassin_de_vie"].str[:2],
                          trimestre=(ventes_brutes["moismut"] - 1) // 3 + 1)
G_DEP = glissements(stats_trimestre(vb, ["departement", "type_bien"]), ["departement", "type_bien"])
G_NAT = glissements(stats_trimestre(vb, ["type_bien"]), ["type_bien"])
BLOC_TRIMESTRES = [f"gliss_{x}_t{q}" for q in TRIMESTRES for x in ("prix", "volume")]


def preparer_trimestres(t):
    t = t.copy()
    k = cles_de(t).assign(departement=lambda d: d["sous_bassin_de_vie"].str[:2])
    k = k.merge(G_DEP, on=["departement", "type_bien", "annee"], how="left", validate="many_to_one")
    k = k.merge(G_NAT, on=["type_bien", "annee"], how="left", validate="many_to_one", suffixes=("", "_nat"))
    for q in TRIMESTRES:
        national = ~k[f"fiable_t{q}"].eq(True)
        for x in ("prix", "volume"):
            c = f"gliss_{x}_t{q}"
            k.loc[national, c] = k.loc[national, c + "_nat"]
    t[BLOC_TRIMESTRES] = k[BLOC_TRIMESTRES].to_numpy(dtype=float)
    t["trimestre_national"] = (~k[[f"fiable_t{q}" for q in TRIMESTRES]].eq(True).all(axis=1)).astype(int).to_numpy()
    return t


table = preparer_trimestres(table)
ventes = preparer_trimestres(ventes)

VARIABLES = (["sous_bassin_de_vie"] + BLOC_BIEN + BLOC_TERRITOIRE + ["log_prix_n1"]
             + BLOC_H5 + BLOC_H6 + BLOC_TRIMESTRES + ["trimestre_national"])
periode = ventes[ventes["annee"] >= ANNEE_DEBUT_H7]
assert periode[VARIABLES].notna().all().all(), "case vide dans les variables de H7v4 (2017-2025)"
print(len(VARIABLES), "variables de H7v4")

# -----------------------------------------------------------------------------
# 6. Réglages optimisés du réseau (comme H7, section 3.1)
# -----------------------------------------------------------------------------
with open(commun.DATA_OUT / "meilleurs_reglages_keras.json", encoding="utf-8") as f:
    OPTIMISATION_KERAS = json.load(f)["reglages"]
assert OPTIMISATION_KERAS["organisation"] == "commun"
REGLAGES = {k: OPTIMISATION_KERAS[k] for k in ["dropout", "learning_rate", "dim_embedding", "perte"]}
REGLAGES["couches"] = tuple(OPTIMISATION_KERAS["couches"])
if OPTIMISATION_KERAS["cible"] == "evolution":
    REGLAGES["reference"] = "log_prix_n1"
print("Réglages :", REGLAGES)


def entrainer(annee):
    """Fenêtre de annee : apprentissage sur les ventes 2017..annee-1, prévisions des ventes et du panier fixe."""
    app, ev = fenetres(ventes, annees_jugees=[annee], annee_debut=ANNEE_DEBUT_H7)[annee]
    debut = time.time()
    prevision, m = prevoir_keras(app, ev, VARIABLES, "prix_m2", poids="w_h", **REGLAGES)
    panier = panier_fixe(ventes_brutes, table, annee)          # biens vendus en annee-1, datés de annee
    prevision_panier = m.predict(panier)
    print(f"Fenêtre {annee} : apprentissage {int(app['annee'].min())}-{int(app['annee'].max())} "
          f"({len(app)} ventes) | jugée {annee} ({len(ev)} ventes) | {(time.time() - debut) / 60:.1f} min")
    return ev, prevision, panier, prevision_panier


# -----------------------------------------------------------------------------
# 7. Contrôle de reproduction sur 2024 (évaluation B)
# -----------------------------------------------------------------------------
if CONTROLE_2024:
    ev24, prev24, _, _ = entrainer(2024)
    t = _par_sous_bassin(ev24, prev24)
    print("\nContrôle 2024 (doit redonner H7v4 optimisé, évaluation B) :")
    for type_bien, e in t.groupby("type_bien"):
        gain = (1 - e["modele"].mean() / e["naive"].mean()) * 100
        print(f"  {type_bien} : gain {gain:.2f} %")
        if type_bien == "maison":
            assert abs(gain - GAIN_H7V4_MAISONS_2024) < TOLERANCE, \
                f"le réseau ne redonne pas H7v4 ({gain:.2f} % au lieu de {GAIN_H7V4_MAISONS_2024} %) : 2025 n'est PAS ouvert"
    print("Contrôle réussi : le réseau est bien H7v4.")

# -----------------------------------------------------------------------------
# 8. Test 2025 : une seule ouverture
# -----------------------------------------------------------------------------
ev25, prev25, panier25, prev_panier25 = entrainer(2025)
resultats, tests = [], []
evaluer_ventes(NOM, ev25, prev25, resultats)
tester_ventes(NOM, ev25, prev25, tests)
evaluer_panier(NOM, panier25, prev_panier25, table, resultats, tests)

cles = ["modèle", "type de bien", "année jugée"]
tableau = pd.DataFrame(resultats).merge(pd.DataFrame(tests), on=cles, suffixes=("", " (test)"))
tableau["évaluation"] = tableau["modèle"].str.extract(r"\((B|C), ", expand=False)

# Biais B : écart moyen pondéré prévision / prix de la vente (%), comme biais() de H7
ecart = np.log(prev25 / ev25["prix_m2"])
biais_b = {tb: (np.exp(np.average(ecart.loc[e.index], weights=e["w_h"])) - 1) * 100
           for tb, e in ev25.groupby("type_bien")}
# Biais C : médiane des prévisions du panier moins prix médian réel 2025 (€/m²)
d = pd.DataFrame({"p": prev_panier25.loc[panier25.index], "w": panier25["w_h"], **{c: panier25[c] for c in CLES}})
med = d.groupby(table.index.names).apply(lambda g: mediane_ponderee(g["p"], g["w"]), include_groups=False)
evaluation_c = table.loc[table.index.intersection(med.index)]
erreur_c = med.loc[evaluation_c.index] - evaluation_c[CIBLE]
biais_c = erreur_c.groupby(evaluation_c.index.get_level_values("type_bien")).mean().to_dict()

tableau["biais"] = [f"{biais_b[r['type de bien']]:+.1f} %" if r["évaluation"] == "B"
                    else f"{biais_c[r['type de bien']]:+.0f} €/m²" for _, r in tableau.iterrows()]
tableau["réussi"] = ((tableau["IC 95 % haut (€/m²)"] < 0) & (tableau["rang-bisériale"] < 0)
                     & (tableau["gain sur la naïve (%)"] >= SEUIL_GAIN))

print("\n=== Test 2025 : H7v4 contre la naïve ===")
print(tableau[["évaluation", "type de bien", "MAE (€/m²)", "MAE naïve (€/m²)", "gain sur la naïve (%)",
               "IC 95 % bas (€/m²)", "IC 95 % haut (€/m²)", "rang-bisériale", "bassins où le modèle gagne (%)",
               "biais", "réussi"]].round(2).to_string(index=False))

sortie = commun.DATA_OUT / "modele_keras_test_final_resultats.csv"
tableau.to_csv(sortie, index=False)
print("Résultats enregistrés :", sortie)
