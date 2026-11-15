# =============================================================================
# socle/commun.py : partie commune à toutes les modélisations
# =============================================================================
# Fixe la table, le découpage, la prévision naïve, le calcul des erreurs
# et le test de comparaison avec la naïve.
# P11 (fin du fichier) : chargement des ventes de P10 et évaluation du modèle par vente
# (B : biens vendus, C : panier fixe).
# Chaque notebook de modélisation l'importe : personne ne réécrit ces fonctions.
# Contrôles : tests/test_commun.py (pytest tests/ depuis la racine du projet).

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
ANNEE_VALIDATION = 2024      # dernière année avant le test : année évaluée par défaut dans decouper, borne des analyses (2025 exclue)
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
# - 'annee' et 'maison' : identifiants ; 'BV2022' : bassin de vie, identifiant du bootstrap de tester (P2).
# Toute autre colonne de la table est une variable explicative : une case vide y retire la ligne (masque commun).

HORS_MASQUE = ["annee", "maison", CIBLE, "n_ventes", "nb_ventes_n1",
               "part_indicateurs_estimes", "part_logements_taux_impute", "part_menages_couverts",
               "BV2022"]


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

    On compare, sous-bassin par sous-bassin, l'erreur du modèle et celle de la naïve :
    écart = |erreur du modèle| - |erreur de la naïve| (négatif : le modèle fait mieux).
    - Bootstrap : combien le modèle gagne ou perd en moyenne (€/m²), avec un
      intervalle de confiance à 95 %. On tire des bassins de vie avec remise, chacun
      avec tous ses sous-bassins (P2) : les sous-bassins d'un même bassin ne sont pas
      indépendants, les tirer un par un donnerait un intervalle trop étroit.
    - Wilcoxon apparié : le modèle fait-il mieux dans la plupart des sous-bassins ?
      Taille d'effet : corrélation rang-bisériale, entre -1 (le modèle fait mieux
      partout) et +1 (le modèle fait moins bien partout).
    """
    assert "BV2022" in evaluation.columns, "Colonne BV2022 absente : relancer le feature engineering"
    rng = np.random.default_rng(GRAINE)
    for type_bien, e in evaluation.groupby("type_bien"):
        ecart = (np.abs(e[CIBLE] - prevision.loc[e.index])
                 - np.abs(e[CIBLE] - prevision_naive(e)))

        # Bootstrap par bassin de vie : somme des écarts et nombre de sous-bassins de chaque bassin ;
        # écart moyen par sous-bassin de chaque échantillon (même mesure que ecart.mean())
        par_bassin = ecart.groupby(e["BV2022"].to_numpy()).agg(["sum", "count"])
        sommes, comptes = par_bassin["sum"].to_numpy(), par_bassin["count"].to_numpy()
        tirages = rng.integers(0, len(par_bassin), size=(N_TIRAGES, len(par_bassin)))
        moyennes = sommes[tirages].sum(axis=1) / comptes[tirages].sum(axis=1)
        ic_bas, ic_haut = np.percentile(moyennes, [2.5, 97.5])

        # Wilcoxon apparié (sous-bassins à écart nul retirés) et corrélation rang-bisériale
        ecart = ecart.to_numpy()
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
                      "rang-bisériale": r,
                      "bassins de vie": len(par_bassin)})


# =============================================================================
# P11 : ÉVALUATION DU MODÈLE PAR VENTE (H7)
# =============================================================================
# Le modèle par vente prévoit le prix au m² de chaque vente de l'échantillon (P10).
# Deux évaluations (plan de modélisation, partie 6) :
# - B, biens vendus : erreur de chaque vente, moyenne pondérée par w_h dans chaque sous-bassin,
#   puis mesures et tests sur les sous-bassins (evaluer_ventes, tester_ventes) ;
#   naïve = prix médian N-1 du sous-bassin de la vente.
# - C, panier fixe : le modèle prévoit en N le prix des biens vendus en N-1 ; la médiane pondérée
#   de ces prévisions est comparée au prix médian réel de N, comme pour les modèles agrégés
#   (panier_fixe, evaluer_panier). C'est la comparaison à information égale avec ces modèles.
# Les ventes ne sont gardées que dans les sous-bassins de la table de modélisation (seuil de
# 20 ventes en N et en N-1, masque commun) : mêmes territoires que les modèles agrégés.

FICHIER_VENTES = DATA_OUT / "ventes_apprentissage.parquet"
CLES = ["annee", "sous_bassin_de_vie", "type_bien"]


def charger_ventes_brutes(fichier=FICHIER_VENTES):
    """Ventes de l'échantillon (P10) telles qu'enregistrées, sans rattachement (colonne anneemut).

    À passer à panier_fixe (évaluation C).
    """
    return pd.read_parquet(fichier)


def charger_ventes(table, fichier=FICHIER_VENTES):
    """Ventes de l'échantillon (P10), rattachées aux sous-bassins de la table.

    Chaque vente reçoit les variables de son sous-bassin pour son année (prix_n1, BV2022, variables
    de territoire). Le prix médian de l'année N (CIBLE) et le nombre de ventes de N (n_ventes) ne
    sont PAS ajoutés : inconnus au moment de prévoir. La cible de la vente est prix_m2.
    """
    ventes = charger_ventes_brutes(fichier).rename(columns={"anneemut": "annee"})
    return rattacher_ventes(ventes, table)


def rattacher_ventes(ventes, table):
    """Garde les ventes des sous-bassins présents dans la table pour leur année, avec leurs variables."""
    territoires = table.drop(columns=[CIBLE, "n_ventes", "annee"]).reset_index()
    v = ventes.drop(columns=["BV2022"], errors="ignore").merge(
        territoires, on=CLES, how="inner", validate="many_to_one")
    return v.reset_index(drop=True)


def _par_sous_bassin(ventes, prevision):
    """Erreurs absolues du modèle et de la naïve, moyennes pondérées par w_h dans chaque sous-bassin."""
    w = ventes["w_h"]
    err = (ventes["prix_m2"] - prevision.loc[ventes.index]).abs()
    err_naive = (ventes["prix_m2"] - ventes["prix_n1"]).abs()
    sommes = pd.DataFrame({"modele": err * w, "naive": err_naive * w,
                           "pct": err / ventes["prix_m2"] * w, "w": w,
                           "BV2022": ventes["BV2022"]})
    sommes[CLES] = ventes[CLES]
    t = sommes.groupby(CLES).agg(modele=("modele", "sum"), naive=("naive", "sum"), pct=("pct", "sum"),
                                 w=("w", "sum"), BV2022=("BV2022", "first"))
    for c in ["modele", "naive", "pct"]:
        t[c] = t[c] / t["w"]
    t["pct"] *= 100
    return t.reset_index()


def evaluer_ventes(nom, ventes, prevision, resultats):
    """Évaluation B : ajoute au tableau commun l'erreur du modèle par vente et son gain sur la naïve.

    MAE d'un sous-bassin = moyenne de l'erreur de ses ventes, pondérée par w_h ; MAE affichée =
    moyenne sur les sous-bassins, chacun comptant pour un, comme pour les modèles agrégés
    (cohérent avec l'écart moyen testé par tester_ventes).
    """
    for type_bien, t in _par_sous_bassin(ventes, prevision).groupby("type_bien"):
        mae, mae_naif = t["modele"].mean(), t["naive"].mean()
        resultats.append({"modèle": f"{nom} (B, ventes)", "type de bien": type_bien,
                          "année jugée": annee_jugee(t),
                          "MAE (€/m²)": mae, "MAE en % du prix": t["pct"].mean(),
                          "MAE naïve (€/m²)": mae_naif,
                          "gain sur la naïve (%)": (1 - mae / mae_naif) * 100,
                          "bassins où le modèle gagne (%)": (t["modele"] < t["naive"]).mean() * 100})


def tester_ventes(nom, ventes, prevision, tests):
    """Évaluation B : mêmes tests que tester (bootstrap par bassin de vie, Wilcoxon), sur les sous-bassins.

    Réutilise tester tel quel : pour chaque sous-bassin, tester calcule
    |CIBLE - prévision| - |CIBLE - prix_n1| ; avec CIBLE = 0, prévision = MAE du modèle et
    prix_n1 = MAE de la naïve, cet écart est exactement MAE du modèle - MAE de la naïve.
    """
    t = _par_sous_bassin(ventes, prevision)
    evaluation = pd.DataFrame({CIBLE: 0.0, "prix_n1": t["naive"], "type_bien": t["type_bien"],
                               "annee": t["annee"], "BV2022": t["BV2022"]})
    tester(f"{nom} (B, ventes)", evaluation, t["modele"], tests)


def mediane_ponderee(valeurs, poids):
    """Médiane pondérée : première valeur où le poids cumulé atteint la moitié du poids total."""
    ordre = np.argsort(valeurs)
    v, w = np.asarray(valeurs)[ordre], np.asarray(poids)[ordre]
    cumul = np.cumsum(w)
    return v[np.searchsorted(cumul, cumul[-1] / 2)]


def panier_fixe(ventes_brutes, table, annee):
    """Évaluation C : biens vendus en annee - 1, datés de annee et rattachés aux variables de annee.

    ventes_brutes : les ventes de charger_ventes_brutes (pas celles de charger_ventes).
    Le modèle prévoit le prix de ces biens en annee ; evaluer_panier compare la médiane des prévisions
    au prix médian réel de annee.
    """
    panier = ventes_brutes[ventes_brutes["anneemut"] == annee - 1].copy()
    panier["anneemut"] = annee
    return rattacher_ventes(panier.rename(columns={"anneemut": "annee"}), table)


def evaluer_panier(nom, panier, prevision, table, resultats, tests):
    """Évaluation C : médiane pondérée des prévisions par sous-bassin, puis evaluer et tester communs."""
    w = panier["w_h"]
    mediane = (pd.DataFrame({"p": prevision.loc[panier.index], "w": w, **{c: panier[c] for c in CLES}})
               .groupby(table.index.names)
               .apply(lambda g: mediane_ponderee(g["p"], g["w"]), include_groups=False))
    evaluation = table.loc[table.index.intersection(mediane.index)]
    evaluer(f"{nom} (C, panier fixe)", evaluation, mediane.loc[evaluation.index], resultats)
    tester(f"{nom} (C, panier fixe)", evaluation, mediane.loc[evaluation.index], tests)
    