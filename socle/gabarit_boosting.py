# =============================================================================
# socle/gabarit_boosting.py : gabarit du gradient boosting (P3)
# =============================================================================
# Pourquoi : chacun fait au moins un gradient boosting au tour 1. Avec le même gabarit
# (même modèle, même grille, même réglage), les résultats sont comparables d'une
# hypothèse à l'autre : une différence vient des variables, pas des réglages.
#
# Ce que fait ce fichier :
# - BoostingEvolution : un gradient boosting qui apprend l'ÉVOLUTION du prix
#   (log(prix / prix N-1)) et rend une prévision en €/m² (prix N-1 x évolution prévue) ;
# - GRILLE_REFERENCE : la grille d'hyperparamètres commune ;
# - prevoir_boosting : pour UNE fenêtre, règle le modèle sur les plis temporels de
#   commun.py, le réapprend sur toute la fenêtre et prévoit l'année jugée,
#   un modèle par type de bien.
# Ce qu'il ne fait PAS : choisir les variables, appeler evaluer / tester. Cela reste
# dans le notebook de chaque hypothèse.
#
# Pourquoi l'évolution plutôt que le niveau :
# un arbre ne prévoit que des moyennes de prix déjà vus dans ses feuilles. En prévoyant
# directement le niveau, il perd la précision du prix N-1 de chaque territoire.
# En prévoyant l'évolution, il part du prix N-1 exact et n'a plus qu'à prévoir de
# combien le prix va bouger. L'option cible="niveau" garde l'autre version, pour comparer.
#
# Pourquoi pas la fonction lancer du template : elle met toujours une mise à l'échelle
# (StandardScaler) avant le modèle. Elle transformerait prix_n1, qu'on ne pourrait plus
# utiliser pour remettre la prévision en €/m². Pour un arbre, elle ne sert à rien.
#
# Utilisation dans un notebook (remplace la cellule 4.4 du template) :
#     from gabarit_boosting import prevoir_boosting
#     for annee, (app, ev) in FENETRES.items():
#         prevision, modeles = prevoir_boosting(app, ev, ["prix_n1"] + VARIABLES_BLOC)
#         evaluer(nom, ev, prevision, resultats)
#         tester(nom, ev, prevision, tests)
#
# Vérification (« c'est fini quand ») : tests/test_gabarit_boosting.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from commun import CIBLE, GRAINE, plis_temporels

# Grille de référence : 8 combinaisons, réglées sur les plis temporels.
# Volontairement petite : 2 000 à 7 000 lignes par type de bien, et des plis d'un an.
GRILLE_REFERENCE = {
    "learning_rate": [0.05, 0.1],     # taille de chaque pas de correction
    "max_depth": [3, 5],              # profondeur des arbres : interactions entre variables
    "min_samples_leaf": [20, 50],     # au moins 20 ou 50 territoires par feuille : limite le sur-apprentissage
}


class BoostingEvolution(RegressorMixin, BaseEstimator):
    """Gradient boosting qui prévoit le prix N en €/m².

    cible="evolution" (par défaut) : apprend log(prix / prix N-1), puis prévoit
        prix N-1 x exp(évolution prévue). La colonne prix_n1 doit être dans X.
    cible="niveau" : apprend directement le prix (version de comparaison).

    fit(X, y) reçoit le prix en €/m², et predict(X) rend des €/m² : GridSearchCV
    compare donc les réglages sur la MAE en €/m², comme evaluer.
    """

    def __init__(self, cible="evolution", learning_rate=0.1, max_depth=3, min_samples_leaf=20,
                 max_iter=300, random_state=GRAINE):
        self.cible = cible
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        if self.cible == "evolution":
            assert "prix_n1" in X.columns, "La cible 'evolution' a besoin de prix_n1 dans les variables"
            y_appris = np.log(y.to_numpy() / X["prix_n1"].to_numpy())
        else:
            y_appris = y.to_numpy()
        # early_stopping=False : l'arrêt précoce de scikit-learn met de côté des lignes tirées
        # au hasard, toutes années mélangées ; ce serait une fuite temporelle.
        self.modele_ = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_iter=self.max_iter,
            early_stopping=False, random_state=self.random_state)
        self.modele_.fit(X, y_appris)
        return self

    def predict(self, X):
        p = self.modele_.predict(X)
        if self.cible == "evolution":
            return X["prix_n1"].to_numpy() * np.exp(p)
        return p


def prevoir_boosting(apprentissage, evaluation, variables, cible="evolution", grille=None):
    """Règle, réapprend et prévoit l'année jugée d'UNE fenêtre, un modèle par type de bien.

    apprentissage, evaluation : une fenêtre de commun.fenetres().
    variables : liste des variables du modèle (avec prix_n1 pour la cible 'evolution').
    grille : GRILLE_REFERENCE par défaut ; {} pour ne pas régler (réglages par défaut).

    Renvoie (prevision, modeles) :
    - prevision : pd.Series en €/m², même index que evaluation, prête pour evaluer et tester ;
    - modeles : {type de bien: GridSearchCV ou modèle ajusté}, pour l'interprétation.
    """
    if grille is None:
        grille = GRILLE_REFERENCE
    prevision = pd.Series(np.nan, index=evaluation.index)
    modeles = {}
    for type_bien, e in apprentissage.groupby("type_bien"):
        # Lignes de l'année jugée pour ce type de bien
        lignes = evaluation.index[evaluation.index.get_level_values("type_bien") == type_bien]
        X_app = e[variables]          # même ordre que e : les plis donnent des positions dans e
        X_ev = evaluation.loc[lignes, variables]

        modele = BoostingEvolution(cible=cible)
        if grille:
            # Réglage sur les plis temporels : chaque pli valide une année, jamais l'année jugée
            modele = GridSearchCV(modele, grille, cv=plis_temporels(e),
                                  scoring="neg_mean_absolute_error")
        # Avec une grille, GridSearchCV réapprend le meilleur réglage sur toute la fenêtre (refit=True)
        modele.fit(X_app, e[CIBLE])
        prevision.loc[lignes] = modele.predict(X_ev)
        modeles[type_bien] = modele

    assert prevision.notna().all(), "Prévisions manquantes : un type de bien sans apprentissage ?"
    return prevision, modeles
