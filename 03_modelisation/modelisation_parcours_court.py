#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# 03_modelisation/modelisation_parcours_court.py : parcours court du projet
# =============================================================================
# Un seul fichier, trois étapes, tout tracé dans MLflow :
#   1. télécharge le kit du parcours court (00_telechargement/telechargement.py, groupe parcours_court) ;
#   2. lance les modèles des hypothèses (H3, H4, H5-H6, H7) sur les années jugées 2023 et 2024 ;
#   3. lance l'évaluation 2025 (les deux scripts de 4_evaluation_2025) et trace leurs résultats.
#
# Lancement, depuis la racine du projet :
#     python 03_modelisation/modelisation_parcours_court.py                  # les trois étapes
#     python 03_modelisation/modelisation_parcours_court.py --etapes 1 2     # sans l'évaluation 2025
#     python 03_modelisation/modelisation_parcours_court.py --etapes 3       # évaluation 2025 seule
#     python 03_modelisation/modelisation_parcours_court.py --etapes 2 --hypotheses h7 h3   # certaines hypothèses, dans cet ordre
# Interface MLflow :
#     mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db
#
# Ce que ce fichier NE fait PAS : choisir les variables ou les réglages. Les modèles reprennent à l'identique ceux
# des notebooks (mêmes variables, mêmes grilles, mêmes gabarits de socle/). Il ne touche pas aux déclarations
# (4_evaluation_2025) : les scripts de test sont lancés tels quels.
#
# Suivi MLflow : un run par (hypothèse, modèle, type de bien, année jugée), toujours contre la naïve (prix N-1).
# Les comparaisons entre modèles restent dans les notebooks. Les erreurs (evaluer) et les tests (tester) sont dans
# le même run. Tags : hypothese, modele, type_bien, annee_jugee, evaluation (agrege / B / C), lancement (date et heure).
# Deux expériences : « Compagnon immo - hypothèses » (étape 2) et « Compagnon immo - évaluation 2025 » (étape 3).

import argparse
import re
import runpy
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))                      # config.py
sys.path.insert(0, str(RACINE / "socle"))            # commun.py et les gabarits
sys.path.insert(0, str(RACINE / "00_telechargement"))

from config import DATA_OUT

DOSSIER_2025 = RACINE / "03_modelisation" / "4_evaluation_2025"
SCRIPTS_2025 = {"modele_hierarchique_test_final.py": "modele_hierarchique_test_final_resultats.csv",
                "modele_keras_test_final.py": "modele_keras_test_final_resultats.csv"}
EXPERIENCE_HYPOTHESES = "Compagnon immo - hypothèses"
EXPERIENCE_2025 = "Compagnon immo - évaluation 2025"
SEUIL_GAIN = 3                                       # % de gain sur la naïve (règle de l'équipe)
LANCEMENT = datetime.now().strftime("%Y-%m-%d %H:%M")


# =============================================================================
# Suivi MLflow
# =============================================================================
# Noms courts des mesures (MLflow refuse €, % et parenthèses dans les noms)
NOMS_MESURES = {"MAE (€/m²)": "mae", "MAE en % du prix": "mae_pct", "MAE naïve (€/m²)": "mae_naive",
                "gain sur la naïve (%)": "gain_naive_pct", "bassins où le modèle gagne (%)": "territoires_gagnes_pct",
                "écart de MAE avec la naïve (€/m²)": "ecart_naive", "IC 95 % bas (€/m²)": "ic_bas",
                "IC 95 % haut (€/m²)": "ic_haut", "p-value Wilcoxon": "p_wilcoxon",
                "rang-bisériale": "rang_biseriale", "bassins de vie": "bassins_de_vie",
                "évolution médiane du prix 2025 (%)": "evolution_mediane_prix_pct"}
CLES_RUN = ["modèle", "type de bien", "année jugée"]


def _mlflow(experience):
    """Importe mlflow à la demande (inutile pour l'étape 1) et sélectionne l'expérience."""
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(DATA_OUT / 'mlflow.db').as_posix()}")   # data/processed : hors Git
    mlflow.set_experiment(experience)
    return mlflow


def type_evaluation(nom):
    """agrege (table), B (ventes) ou C (panier fixe), d'après le suffixe du nom du modèle."""
    if nom.endswith("(B, ventes)"):
        return "B"
    if nom.endswith("(C, panier fixe)"):
        return "C"
    return "agrege"


def tracer(experience, hypothese, resultats, tests=None):
    """Un run MLflow par ligne de `resultats` ; les lignes de `tests` correspondantes sont ajoutées au même run.

    resultats, tests : listes de dictionnaires remplies par commun.evaluer et commun.tester (ou DataFrames),
    toujours calculées contre la naïve.
    """
    mlflow = _mlflow(experience)
    res = pd.DataFrame(resultats)
    if tests is not None and len(tests):
        res = res.merge(pd.DataFrame(tests), on=CLES_RUN, how="left", suffixes=("", " (test)"))
    for _, ligne in res.iterrows():
        mesures = {NOMS_MESURES.get(k, re.sub(r"[^A-Za-z0-9_]+", "_", k)): float(v)
                   for k, v in ligne.items()
                   if k not in CLES_RUN and not str(k).endswith("(test)")
                   and isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
                   and not pd.isna(v)}
        tags = {"hypothese": hypothese, "modele": ligne["modèle"], "type_bien": ligne["type de bien"],
                "annee_jugee": str(ligne["année jugée"]),
                "evaluation": type_evaluation(ligne["modèle"]), "lancement": LANCEMENT}
        if "réussi" in ligne.index and not pd.isna(ligne["réussi"]):
            tags["reussi"] = str(bool(ligne["réussi"]))
        nom_run = " | ".join([hypothese, ligne["modèle"], ligne["type de bien"], str(ligne["année jugée"])])
        with mlflow.start_run(run_name=nom_run):
            mlflow.set_tags(tags)
            mlflow.log_metrics(mesures)


# =============================================================================
# Étape 1 : kit du parcours court
# =============================================================================
def etape_telechargement():
    import telechargement
    telechargement.DATA_RAW.mkdir(parents=True, exist_ok=True)       # comme le programme principal de telechargement.py
    telechargement.DATA_OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== Étape 1 : téléchargement du kit (parcours court) ===")
    echecs = [cle for cle in telechargement.GROUPES["parcours_court"] if not telechargement.telecharger(cle)]
    assert not echecs, f"Téléchargement incomplet : {echecs}"


# =============================================================================
# Étape 2 : modèles des hypothèses
# =============================================================================
def mesurer(hypothese, nom, table_fenetres, previsions):
    """evaluer + tester de commun.py pour un modèle, contre la naïve, puis trace dans MLflow.

    previsions : {(nom, année): Série en €/m²}. table_fenetres : {année: (apprentissage, évaluation)}.
    """
    import commun
    res, tst = [], []
    for annee, (app, ev) in table_fenetres.items():
        commun.evaluer(nom, ev, previsions[(nom, annee)], res)
        commun.tester(nom, ev, previsions[(nom, annee)], tst)
    tracer(EXPERIENCE_HYPOTHESES, hypothese, res, tst)
    tableau = pd.DataFrame(res).merge(pd.DataFrame(tst), on=CLES_RUN, suffixes=("", " (test)"))
    gain = tableau["gain sur la naïve (%)"]
    tableau["année réussie"] = ((tableau["IC 95 % haut (€/m²)"] < 0) & (tableau["rang-bisériale"] < 0)
                                & (gain >= SEUIL_GAIN))
    print(f"\n{nom} (contre la naïve)")
    print(tableau[["type de bien", "année jugée", "MAE (€/m²)", "MAE naïve (€/m²)", "gain sur la naïve (%)",
                   "année réussie"]].round(1).to_string(index=False))
    return tableau


# --- H3 : modèle complet du niveau 1 (blocs retenus en H3a, H3b, H3c) ---------
def hypothese_h3():
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    import commun
    from commun import CIBLE, charger_table, fenetres, plis_temporels
    from gabarit_boosting import BoostingEvolution, prevoir_boosting
    from gabarit_hierarchique import ModeleHierarchique
    from gabarit_pont import MODELE_COMPLET, REFERENCE_VARS, GRILLE_EN

    HYP = "H3"
    GRILLE_RIDGE = {"alpha": list(np.logspace(-4, 3, 15))}
    GRILLE_LASSO = {"alpha": list(np.logspace(-5, -1, 9))}
    LOG = ["nb_ventes_n1", "densite_population", "niveau_vie_median", "part_sans_voiture", "part_moins40m2",
           "part_rsec", "part_apparts", "prix_n1"]                       # présentes dans les variables du modèle complet
    print("\n=== Étape 2 / H3 : modèle complet du niveau 1 ===")
    table = charger_table()
    fen = fenetres(table)                                               # apprentissage dès 2020, comme H3
    previsions = {}

    def en_log(X):
        X = X.copy()
        cols = [c for c in LOG if c in X.columns]
        assert (X[cols] > 0).all().all(), "valeur nulle ou négative dans une variable passée en log"
        X[cols] = np.log(X[cols])
        return X

    class Hierarchique(RegressorMixin, BaseEstimator):
        """Enveloppe du modèle hiérarchique : log de la cible, prévisions en €/m²."""
        def __init__(self, modele=None):
            self.modele = modele

        def fit(self, X, y):
            self.modele_ = deepcopy(self.modele).fit(X, np.log(y))
            return self

        def predict(self, X):
            return np.exp(np.asarray(self.modele_.predict(X)))

    def lancer(nom, modele, variables, log=False, grille=None):
        nom = f"{HYP} – {nom}"
        for annee, (app, ev) in fen.items():
            if isinstance(modele, BoostingEvolution):
                prevision, _ = prevoir_boosting(app, ev, variables, cible=modele.cible, grille=grille)
            else:
                prevision = pd.Series(np.nan, index=ev.index)
                for type_bien, e in app.groupby("type_bien"):
                    idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
                    X_app, X_ev = e[variables], ev.loc[idx, variables]
                    if isinstance(modele, ModeleHierarchique):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")           # convergence du modèle mixte (bord de l'espace)
                            m = Hierarchique(modele).fit(en_log(X_app), e[CIBLE])
                        prevision.loc[idx] = m.predict(en_log(X_ev))
                        continue
                    m = Pipeline([("echelle", StandardScaler()), ("modele", clone(modele))])
                    prefixe = "modele__"
                    if log:
                        X_app, X_ev = en_log(X_app), en_log(X_ev)
                        m = TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
                        prefixe = "regressor__modele__"
                    if grille:
                        m = GridSearchCV(m, {prefixe + k: v for k, v in grille.items()},
                                         cv=plis_temporels(e), scoring="neg_mean_absolute_error")
                    m.fit(X_app, e[CIBLE])
                    prevision.loc[idx] = m.predict(X_ev)
            assert prevision.notna().all(), f"{nom} : prévisions manquantes"
            previsions[(nom, annee)] = prevision
        mesurer(HYP, nom, fen, previsions)

    lancer("Régression linéaire – prix N-1 seul", LinearRegression(), REFERENCE_VARS, log=True)
    lancer("Ridge – modèle complet", Ridge(), MODELE_COMPLET, log=True, grille=GRILLE_RIDGE)
    lancer("Lasso – modèle complet", Lasso(max_iter=50_000), MODELE_COMPLET, log=True, grille=GRILLE_LASSO)
    lancer("Elastic Net – modèle complet", ElasticNet(max_iter=50_000), MODELE_COMPLET, log=True, grille=GRILLE_EN)
    lancer("Hiérarchique – modèle complet", ModeleHierarchique(), MODELE_COMPLET, log=True)
    lancer("Gradient boosting – modèle complet", BoostingEvolution(), MODELE_COMPLET)
    lancer("Arbre – modèle complet", DecisionTreeRegressor(random_state=commun.GRAINE), MODELE_COMPLET,
           grille={"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50]})
    lancer("Forêt – modèle complet",
           RandomForestRegressor(n_estimators=200, random_state=commun.GRAINE, n_jobs=-1), MODELE_COMPLET,
           grille={"max_depth": [5, 10], "min_samples_leaf": [20, 50]})


# --- H4 : le modèle pont (Elastic Net, période longue) ------------------------
def hypothese_h4():
    from commun import charger_table, fenetres
    from gabarit_pont import ANNEE_DEBUT_PONT, previsions_pont

    HYP = "H4"
    print("\n=== Étape 2 / H4 : modèle pont ===")
    table = charger_table()
    previsions = {}
    for nom, debut in [(f"{HYP} – Pont (apprentissage dès {ANNEE_DEBUT_PONT})", ANNEE_DEBUT_PONT),
                       (f"{HYP} – Pont (apprentissage dès 2020, période courte)", 2020)]:
        for annee, p in previsions_pont(table, annee_debut=debut).items():
            assert p.notna().all(), f"{nom} : prévisions manquantes en {annee}"
            previsions[(nom, annee)] = p
        # Les deux modèles sont jugés sur les mêmes lignes de la table : les fenêtres de la période longue
        mesurer(HYP, nom, fenetres(table, annee_debut=ANNEE_DEBUT_PONT), previsions)


# --- H5-H6 : pont + capacité d'achat (H6) + volume et élan (H5) ----------------
def hypothese_h5h6():
    import commun
    from commun import charger_table, fenetres
    from gabarit_pont import ANNEE_DEBUT_PONT, previsions_pont

    HYP = "H5-H6"
    TAUX_EFFORT = ["taux_effort"]
    BLOC_H5 = ["nb_ventes_n1", "rotation_parc_type", "evol_prix_n2_n1"]
    LOG_EN_PLUS = ["taux_effort", "nb_ventes_n1"]
    COLONNES_P9 = ["rotation_parc", "rotation_parc_type", "evol_ventes_n2_n1", "evol_prix_n2_n1"]
    NOM_PONT, NOM_CAPACITE, NOM_NOUVEAU = (f"{HYP} – Pont", f"{HYP} – Pont + taux d'effort",
                                           f"{HYP} – Pont + taux d'effort + volume et élan")
    print("\n=== Étape 2 / H5-H6 : pont, capacité d'achat, volume et élan ===")

    fichiers = [commun.DATA_OUT / "fe_volume_elan_sbv.parquet", commun.DATA_OUT / "fe_capacite_achat_sbv.parquet"]
    manquants = [f.name for f in fichiers if not f.exists()]
    assert not manquants, (f"Tables absentes de data/processed : {manquants}. "
                           "Elles sont produites par les notebooks P9 et P7 du dossier de préparation.")

    table_commune = charger_table()
    cles = pd.DataFrame({"annee": table_commune["annee"].to_numpy(),
                         "sous_bassin_de_vie": table_commune.index.get_level_values("sous_bassin_de_vie"),
                         "type_bien": table_commune.index.get_level_values("type_bien")})
    p9 = pd.read_parquet(fichiers[0])
    p7 = pd.read_parquet(fichiers[1])[["annee", "sous_bassin_de_vie", "type_bien", "taux_effort"]]
    joint = (cles.merge(p9, on=["annee", "sous_bassin_de_vie", "type_bien"], how="left", validate="one_to_one")
                 .merge(p7, on=["annee", "sous_bassin_de_vie", "type_bien"], how="left", validate="one_to_one"))
    assert len(joint) == len(table_commune), "la jointure a changé le nombre de lignes"
    table_jointe = table_commune.copy()
    for c in COLONNES_P9 + TAUX_EFFORT:
        table_jointe[c] = joint[c].to_numpy()
    complet = table_jointe[COLONNES_P9 + TAUX_EFFORT].notna().all(axis=1)       # mêmes lignes pour les trois modèles
    table = table_jointe[complet]
    periode = table[table["annee"] >= ANNEE_DEBUT_PONT]
    assert (periode["taux_effort"] > 0).all() and (periode["nb_ventes_n1"] > 0).all(), "log impossible"
    fen = fenetres(table, annee_debut=ANNEE_DEBUT_PONT)

    previsions = {}
    warnings.filterwarnings("ignore")                  # convergence de l'Elastic Net sur les plis les plus petits
    for nom, bloc in [(NOM_PONT, []), (NOM_CAPACITE, TAUX_EFFORT), (NOM_NOUVEAU, TAUX_EFFORT + BLOC_H5)]:
        prev = previsions_pont(table, annee_debut=ANNEE_DEBUT_PONT, bloc=bloc,
                               log_en_plus=[c for c in LOG_EN_PLUS if c in bloc])
        for annee, p in prev.items():
            assert p.notna().all(), f"{nom} : prévisions manquantes en {annee}"
            previsions[(nom, annee)] = p
    for nom in (NOM_PONT, NOM_CAPACITE, NOM_NOUVEAU):
        mesurer(HYP, nom, fen, previsions)


def hypothese_h7():
    """H7 : modèle par vente. Un Elastic Net et des réseaux Keras de plus en plus riches, jugés contre la naïve sur
    les ventes (B) et sur le panier fixe (C). Les comparaisons deux à deux restent dans le notebook H7."""
    import json

    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import KFold, ParameterGrid
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    import commun
    from commun import (charger_table, charger_ventes, charger_ventes_brutes, evaluer_panier, evaluer_ventes,
                        fenetres, mediane_ponderee, panier_fixe, tester_ventes)
    from gabarit_keras import prevoir_keras
    from gabarit_pont import GRILLE_EN

    HYP = "H7"
    ANNEE_DEBUT_H7 = 2017
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print("\n=== Étape 2 / H7 : modèle par vente (Elastic Net et réseaux Keras) ===")

    entrees = [commun.DATA_OUT / n for n in ("fe_volume_elan_sbv.parquet", "fe_capacite_achat_sbv.parquet",
                                             "meilleurs_reglages_keras.json", "ventes_apprentissage.parquet")]
    manquants = [f.name for f in entrees if not f.exists()]
    assert not manquants, f"Fichiers absents de data/processed : {manquants} (étape 1 du script)."

    # --- Chargement ---------------------------------------------------------------------------------------
    table = charger_table()
    ventes_brutes = charger_ventes_brutes()
    ventes = charger_ventes(table)
    assert ventes["prix_n1"].notna().all() and ventes["BV2022"].notna().all()
    assert ventes["annee"].max() <= commun.ANNEES_JUGEES[-1], "aucune vente de 2025 ne doit être chargée"
    assert "valeurfonc" not in ventes.columns, "valeurfonc (cible) ne doit pas être dans les ventes"
    app_constat = fenetres(ventes, annee_debut=ANNEE_DEBUT_H7)[commun.ANNEES_JUGEES[0]][0]

    # --- Variables du bien : seuils et centres fixés d'après les années d'apprentissage de la 1re fenêtre ---
    SEUIL_STERR = app_constat.loc[app_constat["type_bien"] == "maison", "sterr"].quantile(0.99)
    BORNES_X, BORNES_Y = (100_000, 1_250_000), (6_030_000, 7_120_000)       # Lambert 93, France métropolitaine
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
    ventes_brutes = preparer_bien(ventes_brutes)                              # pour le panier fixe (évaluation C)

    # --- Territoire (log des variables asymétriques) --------------------------------------------------------
    PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
            "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
    VACANCE, DENSITE = ["taux_vacance_complete"], ["densite_population"]
    LOG_TERRITOIRE = [v for v in PARC + VACANCE + DENSITE
                      if v in ["densite_population", "part_sans_voiture", "part_moins40m2", "part_rsec", "part_apparts"]]
    BLOC_TERRITOIRE = ["log_" + v if v in LOG_TERRITOIRE else v for v in PARC + VACANCE + DENSITE]

    def preparer_territoire(t):
        t = t.copy()
        assert (t[LOG_TERRITOIRE] > 0).all().all(), "valeur nulle ou négative dans une variable passée en log"
        for v in LOG_TERRITOIRE:
            t["log_" + v] = np.log(t[v])
        return t

    table, ventes = preparer_territoire(table), preparer_territoire(ventes)
    assert ventes[BLOC_TERRITOIRE].notna().all().all(), "case vide dans le bloc du territoire"
    table["log_prix_n1"] = np.log(table["prix_n1"])
    ventes["log_prix_n1"] = np.log(ventes["prix_n1"])

    # --- Volume et élan (P9), capacité d'achat (P7) -----------------------------------------------------------
    CLES_IND = ["annee", "sous_bassin_de_vie", "type_bien"]

    def cles_de(t):
        if "sous_bassin_de_vie" in t.columns:
            return t[CLES_IND].reset_index(drop=True)
        return t.index.to_frame(index=False)[CLES_IND]

    def ajouter(t, valeurs, colonnes):
        t = t.copy()
        k = cles_de(t).merge(valeurs[CLES_IND + colonnes], on=CLES_IND, how="left", validate="many_to_one")
        t[colonnes] = k[colonnes].to_numpy()
        return t

    P9 = pd.read_parquet(entrees[0])
    P7 = pd.read_parquet(entrees[1])
    COLONNES_P9 = ["evol_prix_n2_n1", "rotation_parc_type"]

    def preparer_quand_niveau2(t):
        t = ajouter(ajouter(t, P9, COLONNES_P9), P7, ["taux_effort"])
        t["evol_n2_inconnue"] = t["evol_prix_n2_n1"].isna().astype(int)      # élan vide : 0 et indicatrice
        t["evol_prix_n2_n1"] = t["evol_prix_n2_n1"].fillna(0)
        for v in ["nb_ventes_n1", "rotation_parc_type", "taux_effort"]:
            assert (t[v] > 0).all(), f"{v} : valeur vide, nulle ou négative"
            t["log_" + v] = np.log(t[v])
        return t

    table, ventes = preparer_quand_niveau2(table), preparer_quand_niveau2(ventes)
    BLOC_H5 = ["log_nb_ventes_n1", "log_rotation_parc_type", "evol_prix_n2_n1", "evol_n2_inconnue"]
    BLOC_H6 = ["log_taux_effort"]
    BASE_V3 = ["sous_bassin_de_vie"] + BLOC_BIEN + BLOC_TERRITOIRE + ["log_prix_n1"]
    assert ventes[BLOC_H5 + BLOC_H6].notna().all().all() and table[BLOC_H5 + BLOC_H6].notna().all().all()

    # --- Trimestres précédents (glissements des trimestres 3 et 4 de N-1, par département) -----------------------
    SEUIL_TRIMESTRE, TRIMESTRES = 30, [3, 4]

    def stats_trimestre(v, lieu):
        g = v.groupby(lieu + ["anneemut", "trimestre"])
        return pd.DataFrame({"mediane": g.apply(lambda e: mediane_ponderee(e["prix_m2"], e["w_h"]), include_groups=False),
                             "volume": g["w_h"].sum(), "ventes": g.size()}).reset_index()

    def glissements(st, lieu):
        sorties = []
        for q in TRIMESTRES:
            e = st[st["trimestre"] == q].drop(columns="trimestre")
            n1 = e.assign(annee=e["anneemut"] + 1).drop(columns="anneemut")
            n2 = e.assign(annee=e["anneemut"] + 2).drop(columns="anneemut")
            m = n1.merge(n2, on=lieu + ["annee"], suffixes=("_n1", "_n2"))
            sorties.append(pd.DataFrame(
                {f"gliss_prix_t{q}": (m["mediane_n1"] / m["mediane_n2"] - 1).to_numpy() * 100,
                 f"gliss_volume_t{q}": (m["volume_n1"] / m["volume_n2"] - 1).to_numpy() * 100,
                 f"fiable_t{q}": (m[["ventes_n1", "ventes_n2"]].min(axis=1) >= SEUIL_TRIMESTRE).to_numpy()},
                index=pd.MultiIndex.from_frame(m[lieu + ["annee"]])))
        return pd.concat(sorties, axis=1).reset_index()

    assert ventes_brutes["anneemut"].max() <= commun.ANNEES_JUGEES[-1], "aucune vente de 2025 ne doit servir"
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
            national = ~k[f"fiable_t{q}"].eq(True)           # département absent ou trop petit : valeur nationale
            for x in ("prix", "volume"):
                c = f"gliss_{x}_t{q}"
                k.loc[national, c] = k.loc[national, c + "_nat"]
        t[BLOC_TRIMESTRES] = k[BLOC_TRIMESTRES].to_numpy(dtype=float)
        t["trimestre_national"] = (~k[[f"fiable_t{q}" for q in TRIMESTRES]].eq(True).all(axis=1)).astype(int).to_numpy()
        return t

    table, ventes = preparer_trimestres(table), preparer_trimestres(ventes)
    assert ventes[BLOC_TRIMESTRES].notna().all().all() and table[BLOC_TRIMESTRES].notna().all().all()
    FEN = fenetres(ventes, annee_debut=ANNEE_DEBUT_H7)

    # --- Réglages du réseau (H7_optimisation.py, année de validation 2022 : jamais une année jugée) ---------------
    with open(entrees[2], encoding="utf-8") as f:
        OPTIMISATION_KERAS = json.load(f)["reglages"]
    assert OPTIMISATION_KERAS["organisation"] == "commun", "H7 n'utilise qu'un réseau pour les deux types de bien"
    EVOLUTION = OPTIMISATION_KERAS["cible"] == "evolution"
    REGLAGES_KERAS = {k: OPTIMISATION_KERAS[k] for k in ["dropout", "learning_rate", "dim_embedding", "perte"]}
    REGLAGES_KERAS["couches"] = tuple(OPTIMISATION_KERAS["couches"])

    # --- Elastic Net par vente ----------------------------------------------------------------------------------
    class ElasticNetVentes(RegressorMixin, BaseEstimator):
        """Elastic Net par vente : apprend log(prix au m²), rend des €/m². Sous-bassin codé par TargetEncoder."""

        def __init__(self, colonne_lieu="sous_bassin_de_vie", alpha=1e-3, l1_ratio=0.5, random_state=commun.GRAINE):
            self.colonne_lieu = colonne_lieu
            self.alpha = alpha
            self.l1_ratio = l1_ratio
            self.random_state = random_state

        def _variables(self, X, code):
            Xn = X[self.variables_].astype(float)
            return Xn if code is None else Xn.assign(code_lieu=code)

        def fit(self, X, y, sample_weight=None):
            cible = np.log(np.asarray(y, dtype=float))
            self.avec_lieu_ = self.colonne_lieu in X.columns
            self.variables_ = [c for c in X.columns if c not in ("annee", self.colonne_lieu)]
            code = None
            if self.avec_lieu_:
                self.encodeur_ = TargetEncoder(target_type="continuous",
                                               cv=KFold(5, shuffle=True, random_state=self.random_state))
                code = self.encodeur_.fit_transform(X[[self.colonne_lieu]], cible).ravel()
            Xn = self._variables(X, code)
            self.echelle_ = StandardScaler().fit(Xn, sample_weight=sample_weight)
            self.modele_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=50_000, precompute=True)
            self.modele_.fit(self.echelle_.transform(Xn), cible, sample_weight=sample_weight)
            return self

        def predict(self, X):
            code = self.encodeur_.transform(X[[self.colonne_lieu]]).ravel() if self.avec_lieu_ else None
            Z = self.echelle_.transform(self._variables(X, code))
            return pd.Series(np.exp(self.modele_.predict(Z)), index=X.index)

    def prevoir_elastic_ventes(app, ev, variables, grille=None):
        """Grille évaluée sur la dernière année d'apprentissage, réapprentissage sur toute la fenêtre."""
        grille = GRILLE_EN if grille is None else grille
        colonnes = list(variables) + ["annee"]
        derniere = app["annee"].max()
        ent, arr = app[app["annee"] < derniere], app[app["annee"] == derniere]
        essais = []
        for reglage in ParameterGrid(grille):
            m = ElasticNetVentes(**reglage).fit(ent[colonnes], ent["prix_m2"], sample_weight=ent["w_h"])
            mae = np.average((arr["prix_m2"] - m.predict(arr[colonnes])).abs(), weights=arr["w_h"])
            essais.append({**reglage, "MAE": mae})
        essais = pd.DataFrame(essais).sort_values("MAE")
        meilleur = {k: essais.iloc[0][k] for k in grille}
        modele = ElasticNetVentes(**meilleur).fit(app[colonnes], app["prix_m2"], sample_weight=app["w_h"])
        return modele.predict(ev[colonnes]), modele

    # --- Lancement d'un modèle : B (ventes) et C (panier fixe), contre la naïve, tracé dans MLflow ---------------
    def lancer(nom, famille, variables, **reglages):
        nom = f"{HYP} – {nom}" if not nom.startswith("H7") else nom
        res, tst = [], []
        if famille == "keras":
            reglages = {**REGLAGES_KERAS, **reglages}
            if EVOLUTION and "log_prix_n1" in variables:
                reglages.setdefault("reference", "log_prix_n1")
        for annee, (app, ev) in FEN.items():
            debut = time.time()
            panier = panier_fixe(ventes_brutes, table, annee)
            if famille == "keras":
                prevision, m = prevoir_keras(app, ev, variables, "prix_m2", poids="w_h", **reglages)
            else:
                prevision, m = prevoir_elastic_ventes(app, ev, variables, **reglages)
            prevision_panier = m.predict(panier)
            evaluer_ventes(nom, ev, prevision, res)
            tester_ventes(nom, ev, prevision, tst)
            evaluer_panier(nom, panier, prevision_panier, table, res, tst)
            print(f"{nom} | fenêtre {annee} : {(time.time() - debut) / 60:.1f} min")
        tracer(EXPERIENCE_HYPOTHESES, HYP, res, tst)
        r = pd.DataFrame(res)[["modèle", "type de bien", "année jugée", "MAE (€/m²)", "MAE naïve (€/m²)",
                               "gain sur la naïve (%)"]]
        print(f"\n{nom} ({len(variables)} variables)")
        print(r.round(1).to_string(index=False))

    # --- Modèles : d'abord le « où », puis le « quand » ----------------------------------------------------------
    lancer("Elastic Net – sous-bassin + bien", "elastic", ["sous_bassin_de_vie"] + BLOC_BIEN)
    lancer("H7v1 – Keras – sous-bassin + bien", "keras", ["sous_bassin_de_vie"] + BLOC_BIEN)
    lancer("H7v1b – Keras – sous-bassin + bien + coordonnées", "keras", ["sous_bassin_de_vie"] + BLOC_BIEN + BLOC_COORD)
    lancer("H7v2a – Keras – sous-bassin + bien + territoire", "keras",
           ["sous_bassin_de_vie"] + BLOC_BIEN + BLOC_TERRITOIRE)
    lancer("H7v2b – Keras – bien + territoire, sans embedding", "keras",
           BLOC_BIEN + BLOC_TERRITOIRE, colonne_embedding=None)
    lancer("Elastic Net – sous-bassin + bien + territoire + prix N-1", "elastic", BASE_V3)
    lancer("H7v3 – Keras – sous-bassin + bien + territoire + prix N-1", "keras", BASE_V3)
    lancer("H7vH5 – Keras – H7v3 + volume et élan", "keras", BASE_V3 + BLOC_H5)
    lancer("H7vH6 – Keras – H7v3 + capacité d'achat", "keras", BASE_V3 + BLOC_H6)
    lancer("H7vT – Keras – H7v3 + trimestres précédents", "keras", BASE_V3 + BLOC_TRIMESTRES + ["trimestre_national"])
    lancer("H7v4 – Keras – H7v3 + volume, élan, capacité d'achat et trimestres", "keras",
           BASE_V3 + BLOC_H5 + BLOC_H6 + BLOC_TRIMESTRES + ["trimestre_national"])


HYPOTHESES = {"h3": hypothese_h3, "h4": hypothese_h4, "h5h6": hypothese_h5h6, "h7": hypothese_h7}


def etape_hypotheses(choix=None):
    print("\n=== Étape 2 : modèles des hypothèses (années jugées 2023 et 2024) ===")
    for cle in (choix or list(HYPOTHESES)):
        HYPOTHESES[cle]()


# =============================================================================
# Étape 3 : évaluation 2025
# =============================================================================
def etape_evaluation_2025():
    """Lance les deux scripts de test final tels quels (ils se retrouvent avec leurs déclarations et leur contrôle
    de reproduction, qui s'arrête avant d'ouvrir 2025 si le résultat 2024 ne se reproduit pas), puis trace leurs
    tableaux dans MLflow (les scripts eux-mêmes n'utilisent pas MLflow)."""
    print("\n=== Étape 3 : évaluation 2025 ===")
    for script, csv in SCRIPTS_2025.items():
        print(f"\n--- {script} ---")
        try:
            runpy.run_path(str(DOSSIER_2025 / script), run_name="__main__")
        except SystemExit as sortie:
            if sortie.code not in (None, 0):
                raise RuntimeError(f"{script} s'est arrêté (code {sortie.code}) : 2025 non évalué, rien n'est tracé.")
        fichier = DATA_OUT / csv
        assert fichier.exists(), f"{csv} absent de data/processed après {script}"
        tableau = pd.read_csv(fichier)
        for _, lignes in tableau.groupby(tableau["modèle"].map(type_evaluation)):
            tracer(EXPERIENCE_2025, "test 2025", lignes.to_dict("records"))
        print(f"Tracé dans MLflow : {len(tableau)} lignes de {csv}")


# =============================================================================
# Programme principal
# =============================================================================
if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Parcours court : kit, modèles des hypothèses, évaluation 2025, tout tracé dans MLflow")
    parseur.add_argument("--etapes", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
                         help="1 : kit, 2 : hypothèses, 3 : évaluation 2025 (défaut : les trois)")
    parseur.add_argument("--hypotheses", nargs="+", choices=list(HYPOTHESES), default=None,
                         help="étape 2 seulement : h3 h4 h5h6 h7 (défaut : toutes, dans cet ordre)")
    args = parseur.parse_args()
    depart = time.time()

    def chrono(etape, debut):
        print(f"\n[durée] {etape} : {(time.time() - debut) / 60:.1f} min "
              f"(total depuis le début : {(time.time() - depart) / 60:.1f} min)")

    for numero, nom, fonction in [(1, "étape 1 (kit)", etape_telechargement),
                                  (2, "étape 2 (hypothèses)", lambda: etape_hypotheses(args.hypotheses)),
                                  (3, "étape 3 (évaluation 2025)", etape_evaluation_2025)]:
        if numero in args.etapes:
            debut = time.time()
            fonction()
            chrono(nom, debut)
    print("\nTerminé. Interface : mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db")