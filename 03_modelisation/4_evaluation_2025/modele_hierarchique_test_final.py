# =============================================================================
# modele_hierarchique_test_final.py : test final 2025 du candidat agrégé du niveau 2
# =============================================================================
# DÉCLARATION, écrite avant d'ouvrir 2025 (12/11/2026)
#   Candidat unique des modèles agrégés : modèle hiérarchique, ordonnée aléatoire par sous-bassin
#   (sans pente aléatoire), variables du modèle pont + taux d'effort (H6, modèle 2.2 « ordonnée seule »).
#   Choisi par la règle de décision sur 2023-2024 : seul modèle agrégé à battre la naïve une année
#   (maisons 2024) et meilleur gain sur le pont ; version la plus simple (la pente n'est pas retenue).
#   Référence : la naïve (prix N = prix N-1).
#   Critère de réussite (une seule année jugée) : gain >= 3 %, IC 95 % haut du bootstrap < 0 et
#   rang-bisériale < 0, pour chaque type de bien. Résultat rapporté tel quel, aucune variante sur 2025.
#
# Ce que fait ce fichier :
#   1. reconstruit le modèle à l'identique de H6 et vérifie qu'il redonne le résultat de H6 sur 2024
#      (contrôle : si l'écart est trop grand, le script s'arrête AVANT d'ouvrir 2025) ;
#   2. réapprend sur les prix 2017-2024 et prévoit 2025 (une seule ouverture) ;
#   3. evaluer / tester de commun.py contre la naïve, règle de décision, biais ;
#   4. enregistre les résultats dans un CSV (le parcours court les trace ensuite dans MLflow).
#
# Lancement, depuis la racine du projet :  python 03_modelisation/4_evaluation_2025/modele_hierarchique_test_final.py
# (le socle doit être installé : pip install -e . depuis la racine, une seule fois)

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.regression.mixed_linear_model import MixedLM

import commun
from commun import CIBLE, charger_table, fenetres, evaluer, tester
from gabarit_pont import ANNEE_DEBUT_PONT, LOG, MODELE_COMPLET, en_log
from gabarit_hierarchique import ModeleHierarchique

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
NOM = "H6 – Hiérarchique – ordonnée seule + taux d'effort (test 2025)"
VARIABLES = MODELE_COMPLET + ["taux_effort"]          # mêmes variables que H6, modèle 2.2
COLONNES_LOG = LOG + ["taux_effort"]                  # mêmes passages en log que H6
SEUIL_GAIN = 3
GAIN_H6_MAISONS_2024 = 9.47                           # résultat de H6 (ordonnée seule, maisons 2024)
TOLERANCE = 0.1                                       # points de % d'écart admis au contrôle


# -----------------------------------------------------------------------------
# Modèle : copie de ModeleHierarchiquePente de H6, utilisé avec pente=None
# -----------------------------------------------------------------------------
class ModeleHierarchiquePente(ModeleHierarchique):
    """Ordonnée aléatoire par sous-bassin (pente=None), comme dans H6."""

    def __init__(self, groupe="sous_bassin_de_vie", reml=True, pente=None):
        super().__init__(groupe=groupe, reml=reml)
        self.pente = pente

    def fit(self, X, y):
        self.variables_ = list(X.columns)
        self.moyennes_ = X.mean()
        self.ecarts_ = X.std(ddof=0).replace(0, 1.0)
        exog = self._standardiser(X)
        if self.pente is None:
            exog_re = np.ones((len(X), 1))
        else:
            exog_re = np.column_stack([np.ones(len(X)), exog[self.pente].to_numpy()])
        modele = MixedLM(endog=np.asarray(y), exog=exog.to_numpy(),
                         groups=np.asarray(self._etiquettes(X)), exog_re=exog_re)
        self.resultat_ = modele.fit(reml=self.reml)
        self.converge_ = bool(self.resultat_.converged)
        self.coefs_fixes_ = pd.Series(np.asarray(self.resultat_.fe_params), index=exog.columns)
        self.effets_aleatoires_ = {g: np.asarray(v, dtype=float) for g, v in self.resultat_.random_effects.items()}
        return self

    def predict(self, X):
        exog = self._standardiser(X)
        fixe = exog.to_numpy() @ self.coefs_fixes_.to_numpy()
        nb_effets = 1 if self.pente is None else 2
        u = np.array([self.effets_aleatoires_.get(g, np.zeros(nb_effets)) for g in self._etiquettes(X)])
        aleatoire = u[:, 0]
        if self.pente is not None:
            aleatoire = aleatoire + u[:, 1] * exog[self.pente].to_numpy()
        return pd.Series(fixe + aleatoire, index=X.index)


def prevoir(app, ev):
    """Un modèle par type de bien ; cible et variables asymétriques en log ; prévision en €/m²."""
    prevision = pd.Series(np.nan, index=ev.index)
    modeles = {}
    for type_bien, e in app.groupby("type_bien"):
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        X_app = en_log(e[VARIABLES], COLONNES_LOG)
        X_ev = en_log(ev.loc[idx, VARIABLES], COLONNES_LOG)
        m = ModeleHierarchiquePente(pente=None).fit(X_app, np.log(e[CIBLE]))
        prevision.loc[idx] = np.exp(m.predict(X_ev)).to_numpy()
        modeles[type_bien] = m
    assert prevision.notna().all(), "prévisions manquantes"
    return prevision, modeles


# -----------------------------------------------------------------------------
# 1. Table : table commune + taux d'effort (P7), comme H6
# -----------------------------------------------------------------------------
capacite = pd.read_parquet(commun.DATA_OUT / "fe_capacite_achat_sbv.parquet")
capacite = capacite.set_index(["annee", "sous_bassin_de_vie", "type_bien"])["taux_effort"]
table = charger_table()
table["taux_effort"] = capacite.reindex(table.index).to_numpy()

periode = table[table["annee"] >= ANNEE_DEBUT_PONT]
assert periode["taux_effort"].notna().all() and (periode["taux_effort"] > 0).all(), \
    "taux d'effort manquant ou nul (2025 compris) : vérifier P7"
print("Lignes 2017-2025 :", len(periode), "| dont 2025 :", int((periode["annee"] == 2025).sum()))


# -----------------------------------------------------------------------------
# 2. Contrôle de reproduction sur 2024 (avant d'ouvrir 2025)
# -----------------------------------------------------------------------------
app24, ev24 = fenetres(table, annees_jugees=[2024], annee_debut=ANNEE_DEBUT_PONT)[2024]
prev24, _ = prevoir(app24, ev24)
print("\nContrôle 2024 (doit redonner H6) :")
for type_bien, e in ev24.groupby("type_bien"):
    mae, mae_naive = mean_absolute_error(e[CIBLE], prev24.loc[e.index]), mean_absolute_error(e[CIBLE], e["prix_n1"])
    gain = (1 - mae / mae_naive) * 100
    print(f"  {type_bien} : MAE {mae:.1f} | naïve {mae_naive:.1f} | gain {gain:.2f} %")
    if type_bien == "maison":
        assert abs(gain - GAIN_H6_MAISONS_2024) < TOLERANCE, \
            f"le modèle ne redonne pas H6 ({gain:.2f} % au lieu de {GAIN_H6_MAISONS_2024} %) : 2025 n'est PAS ouvert"
print("Contrôle réussi : le modèle est bien celui de H6.")


# -----------------------------------------------------------------------------
# 3. Test 2025 : apprentissage sur les prix 2017-2024, une seule ouverture
# -----------------------------------------------------------------------------
app25, ev25 = fenetres(table, annees_jugees=[2025], annee_debut=ANNEE_DEBUT_PONT)[2025]
assert app25["annee"].max() == 2024 and (ev25["annee"] == 2025).all()
print(f"\nTest 2025 : apprentissage {int(app25['annee'].min())}-{int(app25['annee'].max())} "
      f"({len(app25)} lignes) | jugée 2025 ({len(ev25)} lignes)")

prev25, modeles25 = prevoir(app25, ev25)
resultats, tests = [], []
evaluer(NOM, ev25, prev25, resultats)
tester(NOM, ev25, prev25, tests)

cles = ["modèle", "type de bien", "année jugée"]
tableau = pd.DataFrame(resultats).merge(pd.DataFrame(tests), on=cles, suffixes=("", " (test)"))
erreur = prev25 - ev25[CIBLE]
tb = ev25.index.get_level_values("type_bien")
tableau["biais (€/m²)"] = tableau["type de bien"].map(erreur.groupby(tb).mean())
tableau["évolution médiane du prix 2025 (%)"] = tableau["type de bien"].map(
    ((ev25[CIBLE] / ev25["prix_n1"] - 1) * 100).groupby(tb).median())
tableau["réussi"] = ((tableau["IC 95 % haut (€/m²)"] < 0) & (tableau["rang-bisériale"] < 0)
                     & (tableau["gain sur la naïve (%)"] >= SEUIL_GAIN))

print("\n=== Test 2025 : candidat agrégé contre la naïve ===")
print(tableau[["type de bien", "MAE (€/m²)", "MAE naïve (€/m²)", "gain sur la naïve (%)",
               "IC 95 % bas (€/m²)", "IC 95 % haut (€/m²)", "rang-bisériale", "bassins où le modèle gagne (%)",
               "biais (€/m²)", "évolution médiane du prix 2025 (%)", "réussi"]].round(2).to_string(index=False))
print("\nConvergence du modèle :", {tb_: m.converge_ for tb_, m in modeles25.items()})

sortie = commun.DATA_OUT / "modele_hierarchique_test_final_resultats.csv"
tableau.to_csv(sortie, index=False)
print("Résultats enregistrés :", sortie)
