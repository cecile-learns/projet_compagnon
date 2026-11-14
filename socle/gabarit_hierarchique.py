# =============================================================================
# socle/gabarit_hierarchique.py : gabarit du modèle hiérarchique (P4)
# =============================================================================
# Pourquoi : le même sous-bassin revient chaque année. Un effet aléatoire par sous-bassin
# (ordonnée à l'origine propre à chaque territoire) en tient compte, au lieu de traiter
# les lignes comme indépendantes.
#
# Ce que fait ce fichier : une enveloppe fit / predict autour de statsmodels MixedLM,
# utilisable dans la boucle des fenêtres de commun.py (comme un modèle scikit-learn).
# Ce qu'il ne fait PAS : choisir les variables, passer en log, séparer maisons et appartements.
# Cela reste dans le notebook (décisions 1.6 de la feuille de route) : le gabarit prend
# les X et y tels qu'on les lui donne, pour UN type de bien à la fois.

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


class ModeleHierarchique:
    """Régression à ordonnée aléatoire par groupe (par défaut : le sous-bassin de vie).

    prix_i = b0 + b1*x1_i + ... + u_groupe(i) + e_i      avec u ~ N(0, s2_u), e ~ N(0, s2_e)

    - fit(X, y) : X est un DataFrame dont l'index contient le niveau `groupe`.
    - predict(X) : pour un groupe vu à l'apprentissage, effets fixes + effet aléatoire du groupe ;
      pour un groupe absent de l'apprentissage, effets fixes seulement (u = 0, la moyenne de la loi).
    """

    def __init__(self, groupe="sous_bassin_de_vie", reml=True):
        self.groupe = groupe
        self.reml = reml            # REML : estimation de la variance moins biaisée que le maximum de vraisemblance

    def _etiquettes(self, X):
        return X.index.get_level_values(self.groupe)

    def _standardiser(self, X):
        # Les variables sont centrées-réduites sur l'apprentissage seulement (pas de fuite) :
        # prix_n1 vaut plusieurs milliers, les parts valent 0 à 1 ; sans cela l'optimiseur converge mal.
        # Les prévisions ne changent pas (transformation affine des variables).
        Z = (X[self.variables_] - self.moyennes_) / self.ecarts_
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X, y):
        self.variables_ = list(X.columns)
        self.moyennes_ = X.mean()
        self.ecarts_ = X.std(ddof=0).replace(0, 1.0)     # variable constante : pas de division par 0
        exog = self._standardiser(X)
        modele = MixedLM(endog=y.to_numpy(), exog=exog.to_numpy(),
                         groups=np.asarray(self._etiquettes(X)))
        with warnings.catch_warnings(record=True) as avertissements:
            warnings.simplefilter("always")
            self.resultat_ = modele.fit(reml=self.reml)
        self.avertissements_ = [str(a.message) for a in avertissements]
        self.converge_ = bool(self.resultat_.converged)
        self.colonnes_exog_ = list(exog.columns)
        self.effets_aleatoires_ = {g: float(v.iloc[0]) for g, v in self.resultat_.random_effects.items()}
        # Diagnostic de l'effet de groupe : si sa variance est ~0 devant la variance résiduelle (ICC ~ 0),
        # le territoire n'apporte rien au-delà des effets fixes et le modèle revient à la régression linéaire.
        # C'est aussi ce qui fait échouer la convergence « propre » : l'optimum est à la frontière (variance = 0).
        self.variance_groupe_ = float(np.asarray(self.resultat_.cov_re).ravel()[0])
        self.variance_residuelle_ = float(self.resultat_.scale)
        self.icc_ = self.variance_groupe_ / (self.variance_groupe_ + self.variance_residuelle_)
        return self

    def predict(self, X):
        exog = self._standardiser(X)
        fixe = exog.to_numpy() @ np.asarray(self.resultat_.fe_params)
        aleatoire = np.array([self.effets_aleatoires_.get(g, 0.0) for g in self._etiquettes(X)])
        return pd.Series(fixe + aleatoire, index=X.index)

    def part_groupes_inconnus(self, X):
        """Part des lignes de X dont le groupe est absent de l'apprentissage (effets fixes seulement)."""
        return float(np.mean([g not in self.effets_aleatoires_ for g in self._etiquettes(X)]))
