# =============================================================================
# 03_modelisation/gabarit_pont.py : gabarit du modèle pont (H4, tour 2)
# =============================================================================
# Pourquoi : le modèle pont est la base de toutes les hypothèses du tour 2 (H5, H6, H7). Chacun
# doit le relancer à l'identique, y ajouter son bloc de variables, puis comparer le résultat
# au pont avec la règle de décision 1.5. Ce fichier évite de recopier le notebook H4.
#
# Le modèle pont (décision H4) = le modèle complet du niveau 1 appris sur la période longue :
#   - Elastic Net sur le log du prix, variables asymétriques en log, mise à l'échelle dans un pipeline ;
#   - variables : prix N-1 + PARC + vacance complète + densité (MODELE_COMPLET) ;
#   - apprentissage dès 2017 (ANNEE_DEBUT_PONT) ;
#   - réglage (alpha, l1_ratio) par plis temporels, critère MAE, mêmes grilles que H4.
#
# Ce que fait ce fichier : une enveloppe fit / predict pour UN type de bien, et une fonction
# `previsions_pont` qui boucle sur les fenêtres et les deux types de bien.
# Ce qu'il ne fait PAS : choisir les variables d'une hypothèse, évaluer, tester. Cela reste
# dans le notebook, avec `evaluer` et `tester` de commun.py (décisions 1.6 de la feuille de route).
#
# Ajouter un bloc (H5, H6, H7) :
#     prev = previsions_pont(table, bloc=["niveau_vie_median"])                # bloc ajouté au pont
#     prev = previsions_pont(table, bloc=[...], log_en_plus=["niveau_vie_median"])   # si le bloc est asymétrique
# Les colonnes du bloc doivent être des variables N-1 de la table, sans case vide (masque commun).

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commun import CIBLE, fenetres, plis_temporels

ANNEE_DEBUT_PONT = 2017        # début d'apprentissage du pont (période longue, décision H4)

# Variables du modèle complet du niveau 1 (H3 globale), reprises à l'identique
PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
        "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
VACANCE = ["taux_vacance_complete"]
DENSITE = ["densite_population"]
REFERENCE_VARS = ["prix_n1"]
MODELE_COMPLET = REFERENCE_VARS + PARC + VACANCE + DENSITE      # = le modèle pont
LOG = ["densite_population", "part_sans_voiture", "part_moins40m2", "part_rsec", "part_apparts", "prix_n1"]
GRILLE_EN = {"alpha": list(np.logspace(-5, -1, 9)), "l1_ratio": [0.2, 0.5, 0.8]}


def en_log(X, colonnes_log):
    """Passe en log les colonnes asymétriques présentes dans X (elles doivent être strictement positives)."""
    X = X.copy()
    cols = [c for c in colonnes_log if c in X.columns]
    assert (X[cols] > 0).all().all(), f"log impossible : valeurs <= 0 dans {cols}"
    X[cols] = np.log(X[cols])
    return X


class ModelePont:
    """Elastic Net sur le log du prix, pour UN type de bien, réglé par plis temporels.

    - variables : liste des colonnes (par défaut MODELE_COMPLET) ; `bloc` y ajoute les colonnes d'une hypothèse.
    - log_en_plus : colonnes du bloc à passer en log (LOG est toujours appliqué aux colonnes présentes).
    - fit(apprentissage) : DataFrame de la table (colonnes `annee`, CIBLE et les variables), UN type de bien.
      Les plis temporels (commun.plis_temporels) sont construits sur cet apprentissage.
    - predict(evaluation) : Série des prévisions en €/m², au même index que `evaluation`.

    Après fit : coefs_ (coefficients standardisés), coef_prix_n1_ (coefficient du prix N-1 en échelle
    d'origine du log, plus proche de 1 attendu), reglage_ ({alpha, l1_ratio}), n_lignes_.
    """

    def __init__(self, variables=None, bloc=None, log_en_plus=None):
        base = list(MODELE_COMPLET if variables is None else variables)
        self.variables = base + [c for c in (bloc or []) if c not in base]
        self.colonnes_log = list(LOG) + list(log_en_plus or [])
        assert "prix_n1" in self.variables, "le prix N-1 doit rester dans le modèle"

    def fit(self, apprentissage):
        pipe = Pipeline([("echelle", StandardScaler()), ("modele", ElasticNet(max_iter=50_000))])
        m = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
        m = GridSearchCV(m, {"regressor__modele__" + k: v for k, v in GRILLE_EN.items()},
                         cv=plis_temporels(apprentissage), scoring="neg_mean_absolute_error")
        m.fit(en_log(apprentissage[self.variables], self.colonnes_log), apprentissage[CIBLE])
        self.modele_ = m
        pipe_f = m.best_estimator_.regressor_
        self.coefs_ = pd.Series(pipe_f["modele"].coef_, index=self.variables)
        self.coef_prix_n1_ = self.coefs_["prix_n1"] / pipe_f["echelle"].scale_[self.variables.index("prix_n1")]
        self.reglage_ = {k.split("__")[-1]: v for k, v in m.best_params_.items()}
        self.n_lignes_ = len(apprentissage)
        return self

    def predict(self, evaluation):
        X = en_log(evaluation[self.variables], self.colonnes_log)
        return pd.Series(self.modele_.predict(X), index=evaluation.index)


def previsions_pont(table, annees_jugees=None, annee_debut=ANNEE_DEBUT_PONT, variables=None,
                    bloc=None, log_en_plus=None, modeles=None):
    """Prévisions du pont (ou du pont + bloc) pour chaque année jugée : {année: Série en €/m²}.

    Une fenêtre par année jugée (commun.fenetres) et un modèle par type de bien. Les Séries ont
    l'index de l'évaluation : elles passent telles quelles dans `evaluer` et `tester`.
    `modeles`, s'il est donné, est un dict vide rempli avec {(année, type de bien): ModelePont ajusté}
    pour lire les coefficients et le réglage.
    """
    sortie = {}
    for annee, (app, ev) in fenetres(table, annees_jugees=annees_jugees, annee_debut=annee_debut).items():
        prevision = pd.Series(np.nan, index=ev.index)
        for type_bien, e in app.groupby("type_bien"):
            idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
            m = ModelePont(variables, bloc, log_en_plus).fit(e)
            prevision.loc[idx] = m.predict(ev.loc[idx])
            if modeles is not None:
                modeles[(annee, type_bien)] = m
        sortie[annee] = prevision
    return sortie
