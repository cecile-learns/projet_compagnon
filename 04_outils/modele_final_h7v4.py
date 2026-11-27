# =============================================================================
# modele_final_h7v4.py : modèle final de l'outil (bloc B, étapes 3 et 4)
# =============================================================================
# Ce que fait ce fichier (une seule fois) :
#   1-6. prépare les ventes et la table exactement comme modele_keras_test_final.py (sections 1 à 6, recopiées
#        sans changement : aucun nouveau choix sur le modèle) ;
#   7.   réapprend H7v4 sur les ventes 2017-2025, avec les réglages optimisés inchangés
#        (gabarit : arrêt réglé sur 2025, puis réseau neuf sur toute la période) ;
#   8.   enregistre le réseau (.keras) et un fichier de paramètres (.json) : préparation du bien,
#        mise à l'échelle, table des sous-bassins, liste des variables, réglages et fourchette ;
#   9.   contrôle : le modèle rechargé depuis les deux fichiers redonne les mêmes prévisions.
# Pas de MLflow. La fourchette n'est pas recalculée : elle est lue dans fourchette_h7v4.json
# (produit par fourchette_h7v4.py) et recopiée dans le fichier de paramètres.
#
# Produit dans models/h7v4/ (versionné dans Git : c'est le modèle livré) :
#   h7v4.keras              : le réseau
#   h7v4_parametres.json    : tout ce qu'il faut pour préparer un bien et lire la prévision (lu par estimer())
#   SHA256SUMS, MODEL_CARD.md : empreintes et fiche du modèle (écrites par fiche_modele.py)
# Les deux premiers fichiers sont ensuite copiés dans 05_application/donnees/, où l'application les lit.
# Lancement, depuis la racine du projet :  python 04_outils/modele_final_h7v4.py
# Durée : un entraînement du réseau.

import json
import shutil
import time
import warnings
from datetime import date

import keras
import numpy as np
import pandas as pd

import commun
from config import RACINE
from fiche_modele import ecrire_fiche
from commun import (charger_table, charger_ventes_brutes, rattacher_ventes, mediane_ponderee)
from gabarit_keras import ReseauKeras

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
ANNEE_DEBUT_H7 = 2017
ANNEE_FIN = 2025                       # dernière année d'apprentissage du modèle final
FOURCHETTE = commun.DATA_OUT / "fourchette_h7v4.json"
DOSSIER_MODELE = RACINE / "models" / "h7v4"                     # le modèle livré (versionné)
DOSSIER_APPLICATION = RACINE / "05_application" / "donnees"      # copie lue par l'application
SORTIE_RESEAU = DOSSIER_MODELE / "h7v4.keras"
SORTIE_PARAMETRES = DOSSIER_MODELE / "h7v4_parametres.json"
QUAND_EN_LOG = ["nb_ventes_n1", "rotation_parc_type", "taux_effort"]     # comme preparer_quand_niveau2

assert FOURCHETTE.exists(), "lancer d'abord fourchette_h7v4.py (fourchette_h7v4.json absent)"
for f in [SORTIE_RESEAU, SORTIE_PARAMETRES]:
    if f.exists():
        raise SystemExit(f"{f} existe déjà : supprime-le ou renomme-le avant de relancer (pas d'écrasement).")
DOSSIER_MODELE.mkdir(parents=True, exist_ok=True)


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

# -----------------------------------------------------------------------------
# 7. Réapprentissage sur 2017-2025
# -----------------------------------------------------------------------------
colonnes = VARIABLES + ["annee"]                  # l'année ne sert qu'au découpage du gabarit, jamais de variable
assert periode["annee"].max() == ANNEE_FIN
debut = time.time()
reseau = ReseauKeras(**REGLAGES)
reseau.fit(periode[colonnes], periode["prix_m2"], sample_weight=periode["w_h"])
print(f"Modèle final : apprentissage {ANNEE_DEBUT_H7}-{ANNEE_FIN} ({len(periode)} ventes) | arrêt réglé sur "
      f"{reseau.annee_arret_} : {reseau.meilleure_epoque_} époques | {(time.time() - debut) / 60:.1f} min")


# -----------------------------------------------------------------------------
# 8. Enregistrement : le réseau (.keras) et ses paramètres (.json)
# -----------------------------------------------------------------------------
with open(FOURCHETTE, encoding="utf-8") as f:
    fourchette = json.load(f)

parametres = {
    "modele": "H7v4",
    "date": date.today().isoformat(),
    "apprentissage": {"annees": [ANNEE_DEBUT_H7, ANNEE_FIN], "ventes": int(len(periode)),
                      "annee_arret": int(reseau.annee_arret_), "epoques": int(reseau.meilleure_epoque_)},
    "versions": {"keras": keras.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    # Réglages du réseau (ReseauKeras)
    "reglages": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in REGLAGES.items()},
                 "colonne_embedding": reseau.colonne_embedding, "cible_log": reseau.cible_log},
    # Ce que le réseau a appris de l'échelle des données (sans ces valeurs, ses prévisions sont fausses)
    "echelle": {"variables": reseau.variables_,                              # ordre des entrées du réseau
                "moyennes": reseau.moy_X_.to_dict(), "ecarts_types": reseau.ect_X_.to_dict(),
                "moyenne_cible": reseau.moy_y_, "ecart_type_cible": reseau.ect_y_},
    "sous_bassins": reseau.numeros_,                                         # numéro dans l'embedding (0 = inconnu)
    # Préparation d'un bien et d'une ligne de territoire, comme H7 (sections 2 à 5)
    "preparation": {"variables_h7v4": VARIABLES,
                    "seuil_sterr": float(SEUIL_STERR),                      # terrain des maisons plafonné
                    "colonnes_pieces": COLONNES_PIECES, "colonnes_dependances": COLONNES_DEP,
                    "territoire_en_log": LOG_TERRITOIRE, "quand_en_log": QUAND_EN_LOG,
                    "evol_inconnue": "evol_prix_n2_n1 vide -> 0 et evol_n2_inconnue = 1"},
    "fourchette": fourchette,
}
reseau.modele_.save(SORTIE_RESEAU)
with open(SORTIE_PARAMETRES, "w", encoding="utf-8") as f:
    json.dump(parametres, f, ensure_ascii=False, indent=1)
print(f"Enregistré dans {DOSSIER_MODELE} : {SORTIE_RESEAU.name}, {SORTIE_PARAMETRES.name}")
ecrire_fiche(DOSSIER_MODELE)                 # SHA256SUMS et MODEL_CARD.md, d'après le fichier de paramètres


# -----------------------------------------------------------------------------
# 9. Contrôle : le modèle rechargé redonne les mêmes prévisions
# -----------------------------------------------------------------------------
def charger_modele(fichier_reseau, fichier_parametres):
    """Reconstruit un ReseauKeras prêt à prévoir à partir des deux fichiers (même code dans estimation.py)."""
    with open(fichier_parametres, encoding="utf-8") as f:
        p = json.load(f)
    r = p["reglages"]
    m = ReseauKeras(**{k: (tuple(v) if k == "couches" else v) for k, v in r.items()})
    e = p["echelle"]
    m.variables_ = e["variables"]
    m.moy_X_ = pd.Series(e["moyennes"])[m.variables_]
    m.ect_X_ = pd.Series(e["ecarts_types"])[m.variables_]
    m.moy_y_, m.ect_y_ = e["moyenne_cible"], e["ecart_type_cible"]
    m.numeros_ = p["sous_bassins"]
    m.modele_ = keras.saving.load_model(fichier_reseau, compile=False)     # prévoir seulement, pas réentraîner
    return m, p


recharge, _ = charger_modele(SORTIE_RESEAU, SORTIE_PARAMETRES)
controle = periode[periode["annee"] == ANNEE_FIN]
avant, apres = reseau.predict(controle[colonnes]), recharge.predict(controle[colonnes])
ecart = float((apres / avant - 1).abs().max())
print(f"Contrôle sur les {len(controle)} ventes de {ANNEE_FIN} : écart relatif maximal {ecart:.2e}")
assert ecart < 1e-5, "le modèle rechargé ne redonne pas les mêmes prévisions"
print("Contrôle réussi : le modèle rechargé redonne les mêmes prévisions.")


# -----------------------------------------------------------------------------
# 10. Copie dans l'application : elle lit ses fichiers dans 05_application/donnees/
# -----------------------------------------------------------------------------
DOSSIER_APPLICATION.mkdir(parents=True, exist_ok=True)
for fichier in (SORTIE_RESEAU, SORTIE_PARAMETRES):
    shutil.copy2(fichier, DOSSIER_APPLICATION / fichier.name)
print(f"Copié dans {DOSSIER_APPLICATION} : {SORTIE_RESEAU.name}, {SORTIE_PARAMETRES.name}")
