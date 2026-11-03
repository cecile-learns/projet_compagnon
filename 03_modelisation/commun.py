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

# --- Ajouts P1 (feuille de route du 03/11) ------------------------------------
ANNEES_JUGEES = [2023, 2024]  # une fenêtre par année jugée (feuille de route, P1)
N_ANNEES_MIN_PLI = 1          # années d'apprentissage minimales par pli (décision 1.6) :
                              # 2 plis pour la fenêtre 2023, 3 pour la fenêtre 2024

# Colonnes qui ne sont PAS des variables explicatives : elles ne comptent pas pour le masque.
# - cible et comptages de ventes ;
# - variables de contrôle qualité (hors modèle, voir 1.4 de la feuille de route) ;
# - 'annee' et 'maison' : identifiants.
# À COMPLÉTER PAR L'ÉQUIPE : les variables non retenues (ex. structure des ménages)
# doivent être ajoutées ici, sinon une case vide dans l'une d'elles retirerait des lignes pour rien.
HORS_MASQUE = ["annee", "maison", CIBLE, "n_ventes", "nb_ventes_n1",
               "part_indicateurs_estimes", "part_logements_taux_impute", "part_menages_couverts"]


def masque_lignes_communes(table, variables=None):
    """Vrai pour les lignes sans case vide dans les variables explicatives (masque commun).

    Décision d'équipe (1.6) : les lignes avec une case vide sont retirées UNE fois, ici, pour que
    tous les modèles et la naïve soient jugés sur les mêmes lignes.
    Par défaut, toutes les colonnes de la table sauf HORS_MASQUE ; `variables` permet d'en fixer la liste.
    """
    if variables is None:
        variables = [c for c in table.columns if c not in HORS_MASQUE]
    return table[variables].notna().all(axis=1)


def charger_table(masquer=True):
    """Table de référence : une ligne par bassin x année x type de bien, variables en N-1.

    masquer=True (par défaut) retire les lignes à case vide (voir masque_lignes_communes).
    masquer=False rend la table brute, pour un audit.
    """
    table = pd.read_parquet(FICHIER_TABLE)
    return table[masque_lignes_communes(table)] if masquer else table


def decouper(table, annee_eval=ANNEE_VALIDATION, annee_debut=None):
    """Apprentissage : de annee_debut à l'année avant annee_eval ; évaluation : annee_eval.

    annee_debut vaut ANNEE_DEBUT par défaut (2020) : même résultat qu'avant (P5, période longue du tour 2).
    """
    if annee_debut is None:
        annee_debut = ANNEE_DEBUT
    apprentissage = table[(table["annee"] >= annee_debut) & (table["annee"] < annee_eval)]
    evaluation = table[table["annee"] == annee_eval]
    return apprentissage, evaluation


def fenetres(table, annees_jugees=None, annee_debut=None):
    """Une fenêtre par année jugée : {année: (apprentissage, évaluation)}.

    L'apprentissage ne contient que des années strictement antérieures à l'année jugée.
    """
    if annees_jugees is None:
        annees_jugees = ANNEES_JUGEES
    return {a: decouper(table, annee_eval=a, annee_debut=annee_debut) for a in annees_jugees}


def plis_temporels(apprentissage, n_annees_min=N_ANNEES_MIN_PLI):
    """Plis du réglage à l'intérieur d'une fenêtre : un pli par année de validation.

    Pour chaque année a (hors les n_annees_min premières) : apprentissage = les années avant a,
    validation = l'année a. Renvoie une liste de (indices_apprentissage, indices_validation),
    au format de GridSearchCV(cv=...).
    ATTENTION : les indices sont des POSITIONS dans `apprentissage`. X et y passés à GridSearchCV
    doivent donc être extraits de ce même tableau, dans le même ordre (pas de tri ni de filtre entre-temps).
    """
    annees = apprentissage["annee"].to_numpy()
    distinctes = np.sort(np.unique(annees))
    plis = []
    for a in distinctes[n_annees_min:]:
        plis.append((np.flatnonzero(annees < a), np.flatnonzero(annees == a)))
    return plis


def prevision_naive(evaluation):
    """Prix N = prix N-1."""
    return evaluation["prix_n1"]


def annee_jugee(evaluation):
    """Année jugée d'une évaluation : un entier, ou un texte si l'évaluation couvre plusieurs années."""
    annees = sorted(evaluation["annee"].unique())
    return int(annees[0]) if len(annees) == 1 else ", ".join(str(int(a)) for a in annees)


def evaluer(nom, evaluation, prevision, resultats):
    """Ajoute au tableau commun l'erreur du modèle et son gain sur la naïve, par type de bien."""
    for type_bien, e in evaluation.groupby("type_bien"):
        p, naif = prevision.loc[e.index], prevision_naive(e)
        mae, mae_naif = mean_absolute_error(e[CIBLE], p), mean_absolute_error(e[CIBLE], naif)
        gagne = (np.abs(e[CIBLE] - p) < np.abs(e[CIBLE] - naif)).mean() * 100
        resultats.append({"modèle": nom, "type de bien": type_bien, "année jugée": annee_jugee(e),
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

        tests.append({"modèle": nom, "type de bien": type_bien, "année jugée": annee_jugee(e),
                      "écart de MAE avec la naïve (€/m²)": ecart.mean(),
                      "IC 95 % bas (€/m²)": ic_bas, "IC 95 % haut (€/m²)": ic_haut,
                      "p-value Wilcoxon": p_value,
                      "rang-bisériale": r})