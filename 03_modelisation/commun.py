# =============================================================================
# 03_modelisation/commun.py : partie commune à toutes les modélisations
# =============================================================================
# Fixe la table, le découpage, la prévision naïve, le calcul des erreurs
# et le test de comparaison avec la naïve.
# Chaque notebook de modélisation l'importe : personne ne réécrit ces fonctions.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

# config.py se trouve à la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_OUT

CIBLE = "prix"               # niveau du prix médian au m² (décision du 26/10)
ANNEE_DEBUT = 2020           # première année d'apprentissage (période arrêtée le 23/10)
ANNEE_VALIDATION = 2024      # réglage des modèles
ANNEE_TEST = 2025            # évaluation finale, une seule fois
FICHIER_TABLE = DATA_OUT / "table_modelisation_bv.parquet"
N_TIRAGES = 1000             # nombre de tirages du bootstrap
GRAINE = 42                  # graine fixée : bootstrap reproductible


def charger_table():
    """Table de référence : une ligne par bassin x année x type de bien, variables en N-1."""
    return pd.read_parquet(FICHIER_TABLE)


def decouper(table, annee_eval=ANNEE_VALIDATION):
    """Apprentissage : de ANNEE_DEBUT à l'année avant annee_eval ; évaluation : annee_eval."""
    apprentissage = table[(table["annee"] >= ANNEE_DEBUT) & (table["annee"] < annee_eval)]
    evaluation = table[table["annee"] == annee_eval]
    return apprentissage, evaluation


def prevision_naive(evaluation):
    """Prix N = prix N-1."""
    return evaluation["prix_n1"]


def evaluer(nom, evaluation, prevision, resultats):
    """Ajoute au tableau commun l'erreur du modèle et son gain sur la naïve, par type de bien."""
    for type_bien, e in evaluation.groupby("type_bien"):
        p, naif = prevision.loc[e.index], prevision_naive(e)
        mae, mae_naif = mean_absolute_error(e[CIBLE], p), mean_absolute_error(e[CIBLE], naif)
        gagne = (np.abs(e[CIBLE] - p) < np.abs(e[CIBLE] - naif)).mean() * 100
        resultats.append({"modèle": nom, "type de bien": type_bien,
                          "MAE (€/m²)": mae, "MAE en % du prix": (np.abs(e[CIBLE] - p) / e[CIBLE]).mean() * 100,
                          "MAE naïve (€/m²)": mae_naif,
                          "gain sur la naïve (%)": (1 - mae / mae_naif) * 100,
                          "bassins où le modèle gagne (%)": gagne})


def tester(nom, evaluation, prevision, tests):
    """Le modèle fait-il vraiment mieux que la naïve ? Deux tests, par type de bien.

    On compare, bassin par bassin, l'erreur du modèle et celle de la naïve :
    écart = |erreur du modèle| - |erreur de la naïve| (négatif : le modèle fait mieux).
    - Bootstrap : combien le modèle gagne ou perd en moyenne (€/m²), avec un
      intervalle de confiance à 95 % (bassins tirés au hasard avec remise).
    - Wilcoxon apparié : le modèle fait-il mieux dans la plupart des bassins ?
      Taille d'effet : corrélation rang-bisériale, entre -1 (le modèle fait mieux
      partout) et +1 (le modèle fait moins bien partout).
    """
    rng = np.random.default_rng(GRAINE)
    for type_bien, e in evaluation.groupby("type_bien"):
        ecart = (np.abs(e[CIBLE] - prevision.loc[e.index])
                 - np.abs(e[CIBLE] - prevision_naive(e))).to_numpy()

        # Bootstrap : écart moyen sur N_TIRAGES échantillons de bassins
        tirages = rng.integers(0, len(ecart), size=(N_TIRAGES, len(ecart)))
        moyennes = ecart[tirages].mean(axis=1)
        ic_bas, ic_haut = np.percentile(moyennes, [2.5, 97.5])

        # Wilcoxon apparié (bassins à écart nul retirés) et corrélation rang-bisériale
        non_nul = ecart[ecart != 0]
        if len(non_nul) > 0:
            p_value = stats.wilcoxon(non_nul).pvalue
            rangs = stats.rankdata(np.abs(non_nul))
            r = (rangs[non_nul > 0].sum() - rangs[non_nul < 0].sum()) / rangs.sum()
        else:                       # modèle identique à la naïve : rien à tester
            p_value, r = np.nan, np.nan

        tests.append({"modèle": nom, "type de bien": type_bien,
                      "écart de MAE avec la naïve (€/m²)": ecart.mean(),
                      "IC 95 % bas (€/m²)": ic_bas, "IC 95 % haut (€/m²)": ic_haut,
                      "p-value Wilcoxon": p_value,
                      "rang-bisériale": r})