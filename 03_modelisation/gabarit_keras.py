# =============================================================================
# 03_modelisation/gabarit_keras.py : gabarit du réseau de neurones Keras (P12)
# =============================================================================
# Pourquoi : un réseau doit passer dans la même boucle que les autres modèles (fenêtres,
# evaluer / tester, evaluer_ventes / tester_ventes, importance par permutation), avec des
# précautions contre les fuites écrites une seule fois.
#
# Ce que fait ce fichier :
# - ReseauKeras : un réseau dense (ReLU, dropout, sortie linéaire), avec en option un embedding
#   pour une colonne catégorielle (le sous-bassin) ; interface fit / predict en €/m² ;
# - prevoir_keras : apprend sur une fenêtre et prévoit l'année jugée (un seul modèle).
# Ce qu'il ne fait PAS : choisir les variables, les transformer ligne à ligne (log de la surface,
# indicatrices), évaluer. Cela reste dans le notebook (H7, étape 2.4).
#
# Décisions (cahier des charges P12, 06/11/2026) :
# - l'année de chaque ligne est lue dans X (colonne 'annee' ou niveau d'index 'annee') ; elle ne
#   sert qu'au découpage, jamais de variable ;
# - early stopping sur la DERNIÈRE année d'apprentissage, jamais sur des lignes tirées au hasard
#   (fuite temporelle) ; restore_best_weights ;
# - reapprendre=True : réseau neuf réentraîné sur toute la fenêtre, avec le nombre d'époques trouvé ;
# - mise à l'échelle des entrées, log puis standardisation de la cible, table des sous-bassins :
#   calculées sur les seules lignes d'apprentissage du réseau ; sous-bassin inconnu -> numéro 0 ;
# - poids (w_h) normalisés par leur moyenne : la perte de Keras reste à l'échelle naturelle
#   (sinon elle est multipliée par le poids moyen) ; un modèle se juge sur NOS mesures, pas sur cette perte ;
# - pas de GridSearchCV (temps de calcul) : 2 ou 3 configurations comparées sur les années jugées.
#
# Vérification : partie 5 de test.py (données inventées, jamais les ventes réelles).

import numpy as np
import pandas as pd
import keras
from keras import Input, Model, layers
from sklearn.base import BaseEstimator, RegressorMixin

from commun import GRAINE


class ReseauKeras(RegressorMixin, BaseEstimator):
    """Réseau dense qui prévoit un prix en €/m².

    fit(X, y, sample_weight=None)
        X : DataFrame des variables, avec l'année (colonne 'annee' ou niveau d'index 'annee') et,
            si colonne_embedding n'est pas None, la colonne catégorielle (ex. sous_bassin_de_vie).
            Toutes les autres colonnes sont des variables numériques, sans case vide.
        y : prix en €/m² (strictement positifs si cible_log).
        sample_weight : poids des lignes (w_h), facultatif.
    predict(X) -> pd.Series en €/m², même index que X.

    Attributs après fit : meilleure_epoque_, historique_ (pertes par époque, échelle normalisée),
    variables_ (colonnes numériques), annee_arret_ (année utilisée pour l'early stopping).
    """

    def __init__(self, colonne_embedding="sous_bassin_de_vie", dim_embedding=8, couches=(128, 64, 32),
                 dropout=0.1, learning_rate=1e-3, batch_size=512, epochs_max=300, patience=15,
                 cible_log=True, reapprendre=True, graine=GRAINE, verbose=0):
        self.colonne_embedding = colonne_embedding
        self.dim_embedding = dim_embedding
        self.couches = couches
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs_max = epochs_max
        self.patience = patience
        self.cible_log = cible_log
        self.reapprendre = reapprendre
        self.graine = graine
        self.verbose = verbose

    # ------------------------------------------------------------------ outils
    @staticmethod
    def _annees(X):
        """Année de chaque ligne : colonne 'annee', sinon niveau d'index 'annee'."""
        if "annee" in X.columns:
            return X["annee"].to_numpy()
        if "annee" in (X.index.names or []):
            return X.index.get_level_values("annee").to_numpy()
        raise ValueError("ReseauKeras : l'année est introuvable (colonne ou niveau d'index 'annee').")

    def _ajuster_transformations(self, X, y):
        """Apprend, sur les lignes reçues seulement : moyennes et écarts-types, table des sous-bassins."""
        self.variables_ = [c for c in X.columns if c not in ("annee", self.colonne_embedding)]
        non_num = [c for c in self.variables_ if not pd.api.types.is_numeric_dtype(X[c])]
        if non_num:
            raise ValueError(f"ReseauKeras : variables non numériques {non_num} (à encoder dans le notebook).")
        vides = [c for c in self.variables_ if X[c].isna().any()]
        if vides:
            raise ValueError(f"ReseauKeras : cases vides dans {vides} (à traiter dans le notebook).")
        Xn = X[self.variables_].astype(float)
        self.moy_X_ = Xn.mean()
        self.ect_X_ = Xn.std(ddof=0).replace(0, 1.0)        # variable constante : pas de division par 0
        t = self._transformer_cible_brute(y)
        self.moy_y_, self.ect_y_ = float(t.mean()), float(t.std()) or 1.0
        if self.colonne_embedding is not None:
            valeurs = sorted(pd.unique(X[self.colonne_embedding]))
            self.numeros_ = {v: i + 1 for i, v in enumerate(valeurs)}     # 0 = inconnu

    def _transformer_cible_brute(self, y):
        y = np.asarray(y, dtype=float)
        if self.cible_log:
            if (y <= 0).any():
                raise ValueError("ReseauKeras : cible_log=True demande des prix strictement positifs.")
            return np.log(y)
        return y

    def _cible(self, y):
        return ((self._transformer_cible_brute(y) - self.moy_y_) / self.ect_y_).astype("float32")

    def _entrees(self, X):
        num = ((X[self.variables_].astype(float) - self.moy_X_) / self.ect_X_).to_numpy("float32")
        if self.colonne_embedding is None:
            return [num]
        sbv = X[self.colonne_embedding].map(self.numeros_).fillna(0).astype("int32").to_numpy()
        return [num, sbv]

    def _construire(self):
        """Un réseau neuf : entrées numériques (+ embedding), couches denses ReLU + dropout, sortie linéaire."""
        keras.backend.clear_session()            # repart d'un état propre (évite l'accumulation entre modèles)
        keras.utils.set_random_seed(self.graine)
        e_num = Input(shape=(len(self.variables_),), name="numeriques")
        entrees, x = [e_num], e_num
        if self.colonne_embedding is not None:
            e_cat = Input(shape=(1,), dtype="int32", name="categorie")
            v = layers.Embedding(len(self.numeros_) + 1, self.dim_embedding, name="embedding")(e_cat)
            x = layers.Concatenate()([e_num, layers.Flatten()(v)])
            entrees.append(e_cat)
        for n in self.couches:
            x = layers.Dense(n, activation="relu")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        sortie = layers.Dense(1, activation="linear", name="prix")(x)
        modele = Model(entrees, sortie)
        modele.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return modele

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, y, sample_weight=None):
        annees = self._annees(X)
        distinctes = np.unique(annees)
        if len(distinctes) < 2:
            raise ValueError("ReseauKeras : au moins deux années sont nécessaires "
                             "(la dernière sert à l'early stopping).")
        y = np.asarray(y, dtype=float)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # 1. Entraînement sur les années avant la dernière, arrêt réglé sur la dernière
        self.annee_arret_ = int(distinctes.max())
        ent, arr = annees < self.annee_arret_, annees == self.annee_arret_
        self._ajuster_transformations(X[ent], y[ent])
        w_moy = w[ent].mean()
        modele = self._construire()
        arret = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                              restore_best_weights=True)
        h = modele.fit(self._entrees(X[ent]), self._cible(y[ent]), sample_weight=w[ent] / w_moy,
                       validation_data=(self._entrees(X[arr]), self._cible(y[arr]), w[arr] / w_moy),
                       epochs=self.epochs_max, batch_size=self.batch_size, callbacks=[arret],
                       verbose=self.verbose)
        self.historique_ = {"perte": h.history["loss"], "perte_arret": h.history["val_loss"]}
        self.meilleure_epoque_ = int(np.argmin(h.history["val_loss"])) + 1

        # 2. Option : réseau neuf, toute la fenêtre, nombre d'époques trouvé (transformations recalculées)
        if self.reapprendre:
            self._ajuster_transformations(X, y)
            modele = self._construire()
            modele.fit(self._entrees(X), self._cible(y), sample_weight=w / w.mean(),
                       epochs=self.meilleure_epoque_, batch_size=self.batch_size, verbose=self.verbose)
        self.modele_ = modele
        return self

    def predict(self, X):
        p = self.modele_.predict(self._entrees(X), batch_size=8192, verbose=0).ravel()
        p = p.astype(float) * self.ect_y_ + self.moy_y_
        if self.cible_log:
            p = np.exp(p)
        return pd.Series(p, index=X.index)


def prevoir_keras(apprentissage, evaluation, variables, cible, poids=None, **reglages):
    """Apprend un ReseauKeras sur une fenêtre et prévoit l'année jugée.

    apprentissage, evaluation : DataFrames avec l'année (colonne ou index 'annee').
    variables : colonnes du modèle (la colonne d'embedding comprise si elle est utilisée ;
                'annee' est ajoutée automatiquement si c'est une colonne).
    cible : nom de la colonne du prix (ex. 'prix_m2' pour les ventes, 'prix' pour la table).
    poids : nom de la colonne des poids (ex. 'w_h'), ou None.
    Renvoie (prevision, modele) : prevision en €/m², même index que evaluation.
    """
    colonnes = list(variables) + (["annee"] if "annee" in apprentissage.columns and "annee" not in variables else [])
    modele = ReseauKeras(**reglages)
    w = None if poids is None else apprentissage[poids]
    modele.fit(apprentissage[colonnes], apprentissage[cible], sample_weight=w)
    return modele.predict(evaluation[colonnes]), modele
