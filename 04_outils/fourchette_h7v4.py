# =============================================================================
# fourchette_h7v4.py : fourchette d'erreur de l'outil, tirée du test 2025 de H7v4 (bloc B, étape 2)
# =============================================================================
# Ce que fait ce fichier (une seule fois) :
#   1-6. prépare les ventes et la table exactement comme modele_keras_test_final.py (sections 1 à 6, recopiées
#        sans changement : aucun nouveau choix sur le modèle) ;
#   7.   contrôle sur 2024 : le réseau réentraîné doit redonner H7v4 (sinon arrêt, rien n'est enregistré) ;
#   8.   apprend sur 2017-2024, prévoit chaque vente de 2025 et ENREGISTRE ces prévisions ;
#   9.   calcule la fourchette « 8 ventes sur 10 » par type de bien et l'ENREGISTRE.
# Choix de méthode (13/11/2026) :
#   - erreur d'une vente = prix réel / prévision - 1 (s'applique à la prévision d'un bien au prix inconnu) ;
#   - fourchette asymétrique : entre le 10e et le 90e centile des erreurs (garde le biais du modèle visible) ;
#   - centiles pondérés par w_h (« 8 ventes sur 10 » de toutes les ventes 2025, pas de l'échantillon) ;
#   - contrôle 2024 gardé.
# Pas de MLflow : rien n'est ajouté au suivi des expériences.
# Le réseau varie un peu d'un entraînement à l'autre : les prévisions sont très proches de celles du
# test officiel du 12/11, pas identiques.
#
# Produit (data/processed) :
#   previsions_2025_h7v4.parquet : une ligne par vente 2025 (prix réel, prévision, erreur, poids)
#   fourchette_h7v4.json         : bornes basse et haute de l'erreur, par type de bien (lu par estimer())
# Lancement, depuis la racine du projet :  python 04_outils/fourchette_h7v4.py
# Durée : deux entraînements du réseau (2024 puis 2025).

import json
import time
import warnings

import numpy as np
import pandas as pd

import commun
from commun import (charger_table, charger_ventes_brutes, fenetres, rattacher_ventes,
                    mediane_ponderee, _par_sous_bassin)
from gabarit_keras import prevoir_keras

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
ANNEE_DEBUT_H7 = 2017
CONTROLE_2024 = True
GAIN_H7V4_MAISONS_2024 = 7.29          # H7v4 optimisé, maisons 2024, évaluation B (MLflow)
TOLERANCE = 0.5                        # points de % : le réseau varie un peu d'un entraînement à l'autre
COUVERTURE = 0.80                      # part des ventes dans la fourchette (« 8 ventes sur 10 »)
SORTIE_PREVISIONS = commun.DATA_OUT / "previsions_2025_h7v4.parquet"
SORTIE_FOURCHETTE = commun.DATA_OUT / "fourchette_h7v4.json"

# Pas d'écrasement : un nouvel entraînement donnerait une fourchette un peu différente
for f in [SORTIE_PREVISIONS, SORTIE_FOURCHETTE]:
    if f.exists():
        raise SystemExit(f"{f.name} existe déjà : supprime-le ou renomme-le avant de relancer (pas d'écrasement).")


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
# (le panier fixe de l'évaluation C n'est pas utile ici : ventes_brutes ne sert qu'aux trimestres)
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
    """Fenêtre de annee : apprentissage sur les ventes 2017..annee-1, prévision de chaque vente de annee."""
    app, ev = fenetres(ventes, annees_jugees=[annee], annee_debut=ANNEE_DEBUT_H7)[annee]
    debut = time.time()
    prevision, _ = prevoir_keras(app, ev, VARIABLES, "prix_m2", poids="w_h", **REGLAGES)
    print(f"Fenêtre {annee} : apprentissage {int(app['annee'].min())}-{int(app['annee'].max())} "
          f"({len(app)} ventes) | jugée {annee} ({len(ev)} ventes) | {(time.time() - debut) / 60:.1f} min")
    return ev, prevision


# -----------------------------------------------------------------------------
# 7. Contrôle de reproduction sur 2024 (évaluation B)
# -----------------------------------------------------------------------------
if CONTROLE_2024:
    ev24, prev24 = entrainer(2024)
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
# 8. Prévisions 2025, vente par vente (même fenêtre que le test officiel)
# -----------------------------------------------------------------------------
ev25, prev25 = entrainer(2025)
previsions = ev25[["annee", "sous_bassin_de_vie", "type_bien", "prix_m2", "prix_n1", "w_h"]].copy()
previsions["prevision"] = prev25.loc[ev25.index].to_numpy()
previsions["erreur"] = previsions["prix_m2"] / previsions["prevision"] - 1      # + : vendu au-dessus de la prévision
assert previsions[["prevision", "erreur"]].notna().all().all() and (previsions["prevision"] > 0).all()
previsions.to_parquet(SORTIE_PREVISIONS)
print(f"\nPrévisions enregistrées : {SORTIE_PREVISIONS.name} ({len(previsions)} ventes)")

# -----------------------------------------------------------------------------
# 9. Fourchette « 8 ventes sur 10 », par type de bien
# -----------------------------------------------------------------------------
def centile_pondere(valeurs, poids, q):
    """Première valeur où le poids cumulé atteint la part q du poids total (même règle que mediane_ponderee)."""
    ordre = np.argsort(valeurs)
    v, w = np.asarray(valeurs)[ordre], np.asarray(poids)[ordre]
    cumul = np.cumsum(w)
    return float(v[np.searchsorted(cumul, q * cumul[-1])])


q_bas, q_haut = (1 - COUVERTURE) / 2, 1 - (1 - COUVERTURE) / 2           # 10e et 90e centiles
fourchette = {"modele": "H7v4", "annee_test": 2025, "couverture": COUVERTURE,
              "erreur": "prix réel / prévision - 1", "ponderation": "w_h", "types": {}}
print(f"\n=== Fourchette : {COUVERTURE:.0%} des ventes 2025 (pondérées) ===")
for type_bien, e in previsions.groupby("type_bien"):
    bas = centile_pondere(e["erreur"], e["w_h"], q_bas)
    haut = centile_pondere(e["erreur"], e["w_h"], q_haut)
    dedans = e["erreur"].between(bas, haut)
    part = np.average(dedans, weights=e["w_h"])                            # contrôle : proche de 80 %
    mediane = float(mediane_ponderee(e["erreur"], e["w_h"]))
    fourchette["types"][type_bien] = {"erreur_basse": bas, "erreur_haute": haut, "erreur_mediane": mediane,
                                      "part_dans_la_fourchette": float(part), "ventes": int(len(e))}
    print(f"  {type_bien:<12} de {bas:+.1%} à {haut:+.1%} | erreur médiane {mediane:+.1%} | "
          f"dans la fourchette : {part:.1%} des ventes pondérées ({len(e)} ventes)")
    assert abs(part - COUVERTURE) < 0.02, f"{type_bien} : la fourchette ne contient pas {COUVERTURE:.0%} des ventes"

with open(SORTIE_FOURCHETTE, "w", encoding="utf-8") as f:
    json.dump(fourchette, f, ensure_ascii=False, indent=2)
print(f"\nFourchette enregistrée : {SORTIE_FOURCHETTE.name}")
print("Lecture : prix estimé x (1 + erreur_basse) à prix estimé x (1 + erreur_haute).")
