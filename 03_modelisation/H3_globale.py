#!/usr/bin/env python
# coding: utf-8

# # Modélisation – H3 globale : naïve (1.0), Ridge / Lasso / Elastic Net (1.2), hiérarchique (1.3), arbres et boosting (1.4)
# **Contributeurs :** Christian · **Date :** 05/11/2026
# 
# Notebook construit à partir du template commun de l'équipe. Il passe par `commun.py` pour tout ce qui doit être identique d'une hypothèse à l'autre : la table, le masque des lignes, les fenêtres, la naïve, `evaluer` et `tester`.
# 
# **Questions :** (1.0) faut-il un modèle commun ou deux modèles séparés pour les maisons et les appartements ? Une fois les blocs retenus réunis (modèle complet) : (1.2) que reste-t-il comme gain au-delà du prix N-1, et quelles variables survivent ? (1.3) l'effet propre à chaque sous-bassin apporte-t-il quelque chose ? (1.4) un arbre, une forêt ou un gradient boosting font-ils mieux ?
# 
# **Conclusions des hypothèses intégrées :** H3a (Théophyle) : aucune variable d'attractivité retenue ; H3b (Cécile) : parc, vacance et densité retenus ; H3c (Gabrielle) : emploi retenu pour les maisons avec le gradient boosting seulement (à ajouter en 1.4, voir section 4.7).

# ## 0. Paramètres de l'hypothèse
# Le **modèle complet** est : prix N-1 + les blocs retenus en H3a, H3b et H3c.
# - **H3b retenu** (Cécile) : **parc** (9 indicateurs), **vacance** (LOVAC) et **densité**. Ménages et revenus sont écartés : les ménages font perdre l'année 2023 aux deux types de bien, les revenus ne réduisent pas l'erreur davantage que le parc.
# - **H3a** (Théophyle) : **aucune variable retenue**. Deux pistes sont signalées, à tester sans les retenir d'office : les **équipements pour les maisons** (gain de 1,4 à 2,9 %, sous le seuil de 3 %) et la **régularisation sur le prix N-1**. La densité n'entre qu'une fois, dans H3b. Les équipements sont donc testés en **variante** (4.3), la délinquance n'est pas reprise (effet instable d'une année à l'autre).
# - **H3c** (Gabrielle) : emploi retenu pour les maisons avec le gradient boosting seulement : il entre en 1.4, pas dans les modèles linéaires.
# - `nb_ventes_n1` n'appartient à aucun bloc : hors modèle.
# 
# `LOG` et `LOG1P` : mêmes règles que le template. `prix_n1` passe en log avec la cible.

# In[1]:


HYPOTHESE = "H3"

# --- H3b retenu (conclusion de Cécile) : parc, vacance, densité ---
PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
        "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
VACANCE = ["taux_vacance_complete"]
DENSITE = ["densite_population"]
# --- H3a : rien de retenu ; les équipements sont testés en variante (piste de Théophyle) ---
EQUIPEMENTS = ["taux_equip_commerces", "taux_equip_sante", "taux_equip_services", "taux_equip_sports",
               "taux_equip_tourisme", "taux_equip_transports"]

VARIABLES_BLOC = PARC + VACANCE + DENSITE + EQUIPEMENTS      # tous les blocs testés (constat rapide, section 2)

# Variables asymétriques de la table (audit complémentaire, version 2)
LOG = ["nb_ventes_n1", "densite_population", "niveau_vie_median",
       "taux3_vols_sans_violence", "taux3_vols_vehicules",
       "taux3_violences_sexuelles", "taux3_cambriolages", "taux3_violences_intrafamiliales",
       "taux_equip_commerces", "taux_equip_services", "taux_equip_sante", "taux_equip_sports",
       "part_sans_voiture", "part_moins40m2", "part_rsec", "part_apparts",
       "part_familles_monoparentales_pct_fam"]
LOG1P = ["taux3_vols_violents", "taux3_stupefiants", "taux_equip_tourisme", "taux_equip_transports"]

# Variables de l'hypothèse seulement ; prix_n1 passe en log avec la cible
VARIABLES_LOG = [v for v in LOG if v in VARIABLES_BLOC] + ["prix_n1"]
VARIABLES_LOG1P = [v for v in LOG1P if v in VARIABLES_BLOC]
print("log :", VARIABLES_LOG, "| log(1 + x) :", VARIABLES_LOG1P)


# ## 1. Import et chargement
# `charger_table()` retire une seule fois les lignes à case vide (masque commun) : tous les modèles et la naïve sont jugés sur les mêmes lignes. `fenetres()` donne une fenêtre par année jugée (2023 et 2024), avec l'apprentissage sur les années d'avant.

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import commun
from commun import CIBLE, charger_table, fenetres, plis_temporels, evaluer, tester
from gabarit_hierarchique import ModeleHierarchique
from gabarit_boosting import BoostingEvolution, prevoir_boosting

table = charger_table()
FENETRES = fenetres(table)

absentes = sorted(set(VARIABLES_BLOC + VARIABLES_LOG + VARIABLES_LOG1P) - set(table.columns))
assert not absentes, f"Variables absentes de la table : {absentes}"
print("Table :", table.shape)
for a, (app, ev) in FENETRES.items():
    print(f"Fenêtre {a} : apprentissage {sorted(app['annee'].unique())} ({len(app)} lignes) | jugée {a} ({len(ev)} lignes)")


# ## 2. Constat rapide
# Distribution des variables du bloc et lien avec le prix (rho de Spearman), sur les années d'apprentissage de la première fenêtre seulement : l'année jugée n'est jamais regardée avant la modélisation.

# In[3]:


app_constat = FENETRES[commun.ANNEES_JUGEES[0]][0]
print(app_constat[VARIABLES_BLOC].describe().round(2).to_string())

liens = []
for type_bien, e in app_constat.groupby("type_bien"):
    for v in VARIABLES_BLOC:
        liens.append({"type de bien": type_bien, "variable": v, "rho": stats.spearmanr(e[v], e[CIBLE]).statistic})
pd.DataFrame(liens).pivot(index="variable", columns="type de bien", values="rho").round(2)


# ### 2.2 Redondances dans le bloc et lien avec l'écart au prix N-1
# - **Redondances** : couples de variables du bloc dont |rho| dépasse 0,7 ; deux variables aussi liées portent la même information.
# - **Écart au prix N-1** : `prix_n1` porte déjà le niveau du prix. Ce que le bloc peut apporter, c'est ce que `prix_n1` n'explique pas : l'évolution relative prix / prix N-1 − 1. On mesure son lien avec chaque variable (rho de Spearman).
# - Années d'apprentissage de la première fenêtre seulement. Un même territoire revient chaque année : les rho décrivent, ils ne se testent pas.

# In[4]:


couples = []
for type_bien, e in app_constat.groupby("type_bien"):
    rho = e[VARIABLES_BLOC].corr(method="spearman")
    for i, a in enumerate(VARIABLES_BLOC):
        for b in VARIABLES_BLOC[i + 1:]:
            if abs(rho.loc[a, b]) > 0.7:
                couples.append({"type de bien": type_bien, "variable 1": a, "variable 2": b, "rho": round(rho.loc[a, b], 2)})
print(pd.DataFrame(couples).to_string(index=False) if couples else "Aucun couple au-delà de 0,7")

evolution_app = app_constat[CIBLE] / app_constat["prix_n1"] - 1
liens_evolution = []
for type_bien, e in app_constat.groupby("type_bien"):
    for v in VARIABLES_BLOC:
        liens_evolution.append({"type de bien": type_bien, "variable": v,
                                "rho": stats.spearmanr(e[v], evolution_app.loc[e.index]).statistic})
pd.DataFrame(liens_evolution).pivot(index="variable", columns="type de bien", values="rho").round(2)


# ## 3. Fonctions du notebook
# Chaque modèle est lancé en une ligne : `lancer(nom, modèle, variables, log=..., grille=...)`.
# - `nom` : le nom du modèle dans les tableaux (le préfixe de l'hypothèse est ajouté automatiquement) ;
# - `modèle` : le modèle à utiliser (régression linéaire, Lasso, gradient boosting `BoostingEvolution()`, modèle hiérarchique) ;
# - `variables` : les colonnes de la table données au modèle ;
# - `log=True` : la cible et les variables de `VARIABLES_LOG` passent en log, celles de `VARIABLES_LOG1P` en log(1 + x) ; à réserver aux modèles qui en ont besoin (régressions, Lasso, modèle hiérarchique), jamais aux arbres ;
# - `grille` : les valeurs d'un réglage à tester (par exemple `alpha` pour le Lasso).
# 
# Ce que fait `lancer`, pour chacune des deux années jugées (2023, puis 2024) et pour chaque type de bien :
# 1. elle prend les années d'apprentissage, avant l'année jugée ;
# 2. elle met les variables à l'échelle (sauf pour le gradient boosting et le modèle hiérarchique) et entraîne le modèle ; avec une grille, elle choisit d'abord le meilleur réglage, chaque année d'apprentissage servant tour à tour de validation (`plis_temporels` de `commun.py`) ;
# 3. elle prévoit le prix de l'année jugée, remis en €/m² si le modèle a appris en log ;
# 4. elle compare la prévision au prix réel et à la naïve (prix N = prix N-1) avec `evaluer` et `tester` de `commun.py`.
# 
# Elle affiche ensuite l'erreur du modèle (MAE, en €/m²), celle de la naïve, le gain sur la naïve (négatif : le modèle fait moins bien) et la part des sous-bassins où le modèle gagne. Elle garde les modèles entraînés (`ajustes`, pour la section 6) et leurs prévisions (`previsions`, pour la section 5.2). Relancer une cellule remplace les résultats du modèle au lieu de les ajouter une seconde fois.
# 
# Le modèle hiérarchique (`gabarit_hierarchique.py`) passe par une enveloppe : il n'est pas mis dans le pipeline (il met lui-même ses variables à l'échelle) et n'accepte pas de grille.
# 
# Le gradient boosting (`BoostingEvolution` de `gabarit_boosting.py`, gabarit commun P3) passe par `prevoir_boosting` : ni mise à l'échelle ni log, réglage avec la grille commune (`GRILLE_REFERENCE`) sur les plis temporels quand `grille` n'est pas donnée (`grille={}` : pas de réglage). Il apprend l'évolution log(prix / prix N-1) et rend un prix en €/m² (prix N-1 × évolution prévue) : la cible jugée reste le prix. `prix_n1` doit être dans les variables.

# In[5]:


resultats, tests, ajustes, previsions = [], [], {}, {}


def en_log(X):
    """Variables asymétriques en log ; log(1 + x) pour celles qui ont des zéros."""
    X = X.copy()
    cols = [c for c in VARIABLES_LOG if c in X.columns]
    cols1p = [c for c in VARIABLES_LOG1P if c in X.columns]
    assert (X[cols] > 0).all().all(), "valeur nulle ou négative dans une variable de VARIABLES_LOG : la passer dans LOG1P"
    X[cols] = np.log(X[cols])
    X[cols1p] = np.log1p(X[cols1p])
    return X


class Hierarchique(RegressorMixin, BaseEstimator):
    """Enveloppe scikit-learn du modèle hiérarchique : log de la cible si demandé, prévisions en €/m²."""
    def __init__(self, modele=None, log=False):
        self.modele = modele
        self.log = log

    def fit(self, X, y):
        self.modele_ = deepcopy(self.modele).fit(X, np.log(y) if self.log else y)
        return self

    def predict(self, X):
        p = np.asarray(self.modele_.predict(X))
        return np.exp(p) if self.log else p


def lancer(nom, modele, variables, log=False, grille=None):
    """Ajuste un modèle par fenêtre et par type de bien, puis evaluer et tester ; affiche le résultat."""
    nom = f"{HYPOTHESE} – {nom}"
    # Relancer une cellule remplace les résultats du modèle au lieu de les ajouter une seconde fois
    resultats[:] = [r for r in resultats if r["modèle"] != nom]
    tests[:] = [t for t in tests if t["modèle"] != nom]
    for annee, (app, ev) in FENETRES.items():
        if isinstance(modele, BoostingEvolution):
            # Gabarit commun P3 : réglage, un modèle par type de bien et prévision en €/m² faits par prevoir_boosting
            assert not log, "pas de log pour le boosting : le gabarit transforme lui-même la cible"
            prevision, modeles = prevoir_boosting(app, ev, variables, cible=modele.cible, grille=grille)
            for type_bien, m in modeles.items():
                idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
                ajustes[(nom, annee, type_bien)] = (m, ev.loc[idx, variables], ev.loc[idx, CIBLE])
        else:
            prevision = pd.Series(np.nan, index=ev.index)
            for type_bien, e in app.groupby("type_bien"):
                idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
                X_app, X_ev = e[variables], ev.loc[idx, variables]
                if isinstance(modele, ModeleHierarchique):
                    assert not grille, "pas de grille pour le modèle hiérarchique"
                    if log:
                        X_app, X_ev = en_log(X_app), en_log(X_ev)
                    m = Hierarchique(modele, log).fit(X_app, e[CIBLE])
                    prevision.loc[idx] = m.predict(X_ev)
                    ajustes[(nom, annee, type_bien)] = (m, X_ev, ev.loc[idx, CIBLE])
                    continue
                m = Pipeline([("echelle", StandardScaler()), ("modele", clone(modele))])
                prefixe = "modele__"
                if log:
                    X_app, X_ev = en_log(X_app), en_log(X_ev)
                    m = TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
                    prefixe = "regressor__modele__"
                if grille:
                    # plis recalculés sur e (positions dans e, même ordre que X_app)
                    m = GridSearchCV(m, {prefixe + k: v for k, v in grille.items()},
                                     cv=plis_temporels(e), scoring="neg_mean_absolute_error")
                m.fit(X_app, e[CIBLE])
                prevision.loc[idx] = m.predict(X_ev)
                ajustes[(nom, annee, type_bien)] = (m, X_ev, ev.loc[idx, CIBLE])
        assert prevision.notna().all(), f"{nom} : prévisions manquantes"
        previsions[(nom, annee)] = prevision          # gardée pour comparer les modèles entre eux (section 5.2)
        evaluer(nom, ev, prevision, resultats)
        tester(nom, ev, prevision, tests)

    # Résultat du modèle, affiché tout de suite : erreur du modèle, erreur de la naïve, gain
    res = pd.DataFrame(resultats)
    res = res[res["modèle"] == nom][["type de bien", "année jugée", "MAE (€/m²)", "MAE naïve (€/m²)",
                                     "gain sur la naïve (%)", "bassins où le modèle gagne (%)"]]
    print(nom, f"({len(variables)} variables)")
    print(res.round(1).to_string(index=False))
    return nom


# ## 4. Modèles
# Du plus simple au plus complexe. Tous les modèles sont gardés, y compris ceux qui perdent. Les régressions, les régularisées et le modèle hiérarchique sont faits en log (cible et variables asymétriques), la référence aussi ; les arbres et le gradient boosting n'ont pas besoin du log.
# 
# **Réglages (choix de ce notebook, pas de la feuille de route)** : `alpha` sur une échelle logarithmique réglé sur les plis temporels ; `l1_ratio` de l'Elastic Net à 0,2 / 0,5 / 0,8 ; grilles courtes pour l'arbre et la forêt. Le gradient boosting suit la grille commune du gabarit.

# In[6]:


REFERENCE_VARS = ["prix_n1"]
MODELE_COMPLET = REFERENCE_VARS + PARC + VACANCE + DENSITE        # prix N-1 + blocs retenus (H3b)
AVEC_EQUIPEMENTS = MODELE_COMPLET + EQUIPEMENTS                   # variante : piste H3a
GRILLE_RIDGE = {"alpha": list(np.logspace(-4, 3, 15))}
GRILLE_LASSO = {"alpha": list(np.logspace(-5, -1, 9))}
GRILLE_EN = {"alpha": list(np.logspace(-5, -1, 9)), "l1_ratio": [0.2, 0.5, 0.8]}
print(len(MODELE_COMPLET), "variables dans le modèle complet ;", len(AVEC_EQUIPEMENTS), "avec les équipements")


# ### 4.1 Référence : prix N-1 seul
# La naïve (prix N = prix N-1) est déjà dans chaque ligne du tableau (« MAE naïve »). Le modèle « prix N-1 seul » est la référence que le modèle complet doit battre (section 5.2).

# In[7]:


REFERENCE = lancer("Régression linéaire – prix N-1 seul", LinearRegression(), REFERENCE_VARS, log=True)


# ### 4.1 bis Étape 1.0 : modèle commun ou séparé pour maisons et appartements
# La référence ci-dessus est **séparée** : un modèle par type de bien, comme tout le notebook. On la compare à une version **commune** : un seul modèle pour les deux types, avec un indicateur « maison » (`maison`, déjà dans la table). Même cible en log, mêmes lignes, mêmes fenêtres ; seule la séparation change. Le modèle commun entre dans les tableaux des sections 5 et 5.2.

# In[8]:


def lancer_commun(nom):
    """Régression log(prix) ~ log(prix N-1) + maison, un seul modèle pour les deux types de bien."""
    nom = f"{HYPOTHESE} – {nom}"
    resultats[:] = [r for r in resultats if r["modèle"] != nom]
    tests[:] = [t for t in tests if t["modèle"] != nom]
    for annee, (app, ev) in FENETRES.items():
        X_app = pd.DataFrame({"log_prix_n1": np.log(app["prix_n1"]), "maison": app["maison"]})
        X_ev = pd.DataFrame({"log_prix_n1": np.log(ev["prix_n1"]), "maison": ev["maison"]})
        m = LinearRegression().fit(X_app, np.log(app[CIBLE]))
        prevision = pd.Series(np.exp(m.predict(X_ev)), index=ev.index)
        previsions[(nom, annee)] = prevision
        evaluer(nom, ev, prevision, resultats)
        tester(nom, ev, prevision, tests)
    return nom

REFERENCE_COMMUNE = lancer_commun("Régression linéaire – prix N-1 seul, commune (indicateur maison)")

# Commun contre séparé : écart de MAE (positif : le modèle commun est moins bon)
r = pd.DataFrame(resultats)
mae = r[r["modèle"].isin([REFERENCE, REFERENCE_COMMUNE])].pivot_table(
    index=["année jugée", "type de bien"], columns="modèle", values="MAE (€/m²)")
ecart = pd.DataFrame({"MAE séparée": mae[REFERENCE], "MAE commune": mae[REFERENCE_COMMUNE]})
ecart["commune - séparée (€/m²)"] = ecart["MAE commune"] - ecart["MAE séparée"]
ecart.round(2)


# **Lecture (étape 1.0).** Le modèle commun fait un peu mieux pour les maisons (−5 à −6 €/m² de MAE) et un peu moins bien pour les appartements (+10 €/m²), les deux années. L'écart est de 3 à 5 % de l'erreur et va dans les deux sens : ni l'un ni l'autre ne s'impose. **Choix retenu pour la suite : modèles séparés** (un par type de bien et par fenêtre). Raisons : la séparation laisse à chaque type sa propre pente sur `prix_n1` et ses propres coefficients, c'est le fonctionnement du template d'équipe et des notebooks H3a, H3b et H3c, et le commun ne gagne que sur un type de bien. *Décision à confirmer par le propriétaire du profil (décision 1.6 : « décidé par Chauve-Souris sur la naïve »).*

# ### 4.2 Étape 1.2 : Ridge, Lasso, Elastic Net sur le modèle complet

# In[9]:


RIDGE = lancer("Ridge – modèle complet", Ridge(), MODELE_COMPLET, log=True, grille=GRILLE_RIDGE)
LASSO = lancer("Lasso – modèle complet", Lasso(max_iter=50_000), MODELE_COMPLET, log=True, grille=GRILLE_LASSO)
ELASTIC = lancer("Elastic Net – modèle complet", ElasticNet(max_iter=50_000), MODELE_COMPLET, log=True, grille=GRILLE_EN)


# ### 4.3 Variante : modèle complet + équipements (piste H3a), et régularisation sur le prix N-1 seul
# - **Équipements :** Théophyle signale un gain faible des équipements pour les maisons ; on les ajoute au modèle complet pour voir s'ils tiennent quand le reste est là.
# - **Régularisation sur le prix N-1 seul :** H3a observe que le gain du Lasso ne vient pas de l'attractivité mais de la régularisation du prix N-1, et propose de le confirmer avec un Lasso sur le prix N-1 seul. C'est fait ici, avec Ridge et Lasso.

# In[10]:


RIDGE_EQ = lancer("Ridge – complet + équipements", Ridge(), AVEC_EQUIPEMENTS, log=True, grille=GRILLE_RIDGE)
LASSO_EQ = lancer("Lasso – complet + équipements", Lasso(max_iter=50_000), AVEC_EQUIPEMENTS, log=True, grille=GRILLE_LASSO)
ELASTIC_EQ = lancer("Elastic Net – complet + équipements", ElasticNet(max_iter=50_000), AVEC_EQUIPEMENTS, log=True, grille=GRILLE_EN)
RIDGE_REF = lancer("Ridge – prix N-1 seul", Ridge(), REFERENCE_VARS, log=True, grille=GRILLE_RIDGE)
LASSO_REF = lancer("Lasso – prix N-1 seul", Lasso(max_iter=50_000), REFERENCE_VARS, log=True, grille=GRILLE_LASSO)


# ### 4.4 Étape 1.3 : modèle hiérarchique (P4)
# Une ordonnée à l'origine propre à chaque sous-bassin (`gabarit_hierarchique.py`), en log, sur le modèle complet. L'ICC mesure la part de la variance qui tient au sous-bassin : proche de 0, l'effet propre du territoire n'apporte rien au-delà du prix N-1 et le modèle se comporte comme une régression simple. Des avertissements de convergence sont attendus dans ce cas (solution sur le bord).

# In[11]:


import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # convergence du modèle mixte : relevée par m.modele_.converge_ (ci-dessous)
    HIERARCHIQUE = lancer("Hiérarchique – modèle complet", ModeleHierarchique(), MODELE_COMPLET, log=True)
icc = pd.DataFrame({(a, t): {"ICC": ajustes[(HIERARCHIQUE, a, t)][0].modele_.icc_,
                             "convergence": ajustes[(HIERARCHIQUE, a, t)][0].modele_.converge_}
                    for a in commun.ANNEES_JUGEES for t in ["appartement", "maison"]}).T
icc.index.names = ["année jugée", "type de bien"]
icc.round(4)


# ### 4.5 Étape 1.4 : arbre, forêt et gradient boosting (P3)
# - **Gradient boosting (gabarit commun)** : il apprend l'évolution du prix (log(prix / prix N-1)) et rend une prévision en €/m². Grille commune du gabarit.
# - **Arbre et forêt** : prévoient directement le niveau du prix, comme le demande la feuille de route. Ils rendent la moyenne des prix de chaque feuille et perdent la précision du prix N-1 de chaque sous-bassin (raison pour laquelle le gabarit apprend l'évolution) : ils servent de contraste. Pas de log, pas d'imputation nécessaire (table masquée).

# In[12]:


BOOSTING = lancer("Gradient boosting – modèle complet", BoostingEvolution(), MODELE_COMPLET)
ARBRE = lancer("Arbre – modèle complet", DecisionTreeRegressor(random_state=commun.GRAINE), MODELE_COMPLET,
               grille={"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50]})
FORET = lancer("Forêt – modèle complet", RandomForestRegressor(n_estimators=200, random_state=commun.GRAINE, n_jobs=-1),
               MODELE_COMPLET, grille={"max_depth": [5, 10], "min_samples_leaf": [20, 50]})


# ### 4.6 Emploi (H3c) pour les maisons : à ajouter
# Gabrielle retient le bloc emploi (`evol_emploi_n2_n1`, `emploi_par_logement`) pour les maisons avec le gradient boosting. Ces deux colonnes sont dans `fe_cible_emploi_sbv.parquet`, pas dans la table de modélisation : le boosting avec emploi se lance ici une fois ce fichier joint (année et sous-bassin). Pas fait dans cette version.

# ## 5. Mesures et tests
# Trois tableaux, du plus simple au plus exigeant :
# 1. **`res` : les erreurs.** MAE du modèle et de la naïve en €/m², gain sur la naïve en %, part des sous-bassins où le modèle gagne. C'est une description : un modèle peut avoir une MAE plus basse par hasard.
# 2. **`tst` : les tests contre la naïve.** Ils disent si l'écart avec la naïve est plus grand que le hasard.
#    - **Écart de MAE avec la naïve** : erreur du modèle moins erreur de la naïve, en €/m² (négatif : le modèle fait mieux).
#    - **IC 95 %** (bootstrap par bassin de vie) : la fourchette dans laquelle se trouve cet écart, en tenant compte du hasard. Si toute la fourchette est sous 0, le modèle fait vraiment mieux ; si elle contient 0, on ne peut pas conclure ; si elle est toute au-dessus de 0, il fait vraiment moins bien.
#    - **p-value de Wilcoxon** : le modèle fait-il mieux (ou moins bien) dans la plupart des sous-bassins ? Avec plusieurs centaines de sous-bassins, elle est presque toujours très petite : on la lit avec la taille d'effet.
#    - **Rang-bisériale** (taille d'effet) : de −1 (le modèle fait mieux partout) à +1 (il fait moins bien partout) ; autour de 0, pas de différence nette.
# 3. **`decision` : la règle de décision.** Une année est réussie si l'IC est entièrement sous 0, la rang-bisériale négative et le gain d'au moins 3 %. Un modèle est retenu s'il réussit la majorité des années jugées (ici, les deux).

# In[13]:


res = pd.DataFrame(resultats)
res.round(2)


# In[14]:


tst = pd.DataFrame(tests)
tst.round(3)


# ### 5.1 Règle de décision contre la naïve

# In[15]:


SEUIL_GAIN = 3      # % de gain sur la MAE de la comparaison, fixé avant les résultats


def regle_decision(res, tst, comparaison="naïve"):
    """Années réussies et décision, par modèle et type de bien."""
    cles = ["modèle", "type de bien", "année jugée"]
    regle = res[cles + [f"gain sur la {comparaison} (%)"]].merge(
        tst[cles + ["IC 95 % haut (€/m²)", "rang-bisériale"]], on=cles)
    regle["année réussie"] = ((regle["IC 95 % haut (€/m²)"] < 0) & (regle["rang-bisériale"] < 0)
                              & (regle[f"gain sur la {comparaison} (%)"] >= SEUIL_GAIN))
    decision = regle.groupby(["modèle", "type de bien"], sort=False)["année réussie"].agg(["sum", "count"])
    decision["retenu"] = decision["sum"] > decision["count"] / 2
    return decision.rename(columns={"sum": "années réussies", "count": "années jugées"})


regle_decision(res, tst)


# **Comment lire le tableau des tests.** Prenons « Ridge – modèle complet », appartements, 2023 : l'écart de MAE avec la naïve est positif (le modèle fait un peu moins bien) et l'IC 95 % contient 0 : on ne peut pas conclure à une différence. La même ligne en 2024 a un IC entièrement au-dessus de 0 : le modèle fait vraiment moins bien que la naïve.
# 
# **Résultats contre la naïve :** aucun modèle n'est retenu, pour les deux types de bien (0 année réussie sur 2 partout, y compris le hiérarchique, le gradient boosting, l'arbre et la forêt). Le gain sur la naïve est négatif dans tous les cas : de −1 % (appartements 2023, régularisées) à −83 % (maisons 2023, forêt). Le modèle « prix N-1 seul » perd aussi (−12 % à −66 %).

# ### 5.2 Le bloc apporte-t-il quelque chose au prix N-1 seul ?
# La question propre à l'hypothèse : les variables du bloc améliorent-elles la prévision **par rapport au modèle « prix N-1 seul »** ?
# 
# On applique exactement les mêmes mesures et la même règle de décision, en remplaçant la naïve par la prévision du modèle « prix N-1 seul ». Les fonctions `evaluer` et `tester` lisent la prévision de comparaison dans la colonne `prix_n1` : on leur donne une copie de l'année jugée où cette colonne contient la prévision de la référence.
# 
# Lecture : un gain positif et un intervalle du bootstrap entièrement sous 0 veulent dire que le modèle fait vraiment mieux que la référence.

# In[16]:


resultats_ref, tests_ref = [], []
modeles_compares = [n for n in dict.fromkeys(k[0] for k in previsions) if n != REFERENCE]

for nom in modeles_compares:
    for annee, (app, ev) in FENETRES.items():
        ev_ref = ev.copy()
        ev_ref["prix_n1"] = previsions[(REFERENCE, annee)]     # la comparaison se fait avec la référence
        evaluer(nom, ev_ref, previsions[(nom, annee)], resultats_ref)
        tester(nom, ev_ref, previsions[(nom, annee)], tests_ref)

# Mêmes tableaux que contre la naïve, avec « référence » à la place de « naïve »
res_ref = pd.DataFrame(resultats_ref).rename(columns=lambda c: c.replace("naïve", "référence"))
tst_ref = pd.DataFrame(tests_ref).rename(columns=lambda c: c.replace("naïve", "référence"))
display(res_ref[["modèle", "type de bien", "année jugée", "MAE (€/m²)", "MAE référence (€/m²)",
                 "gain sur la référence (%)", "bassins où le modèle gagne (%)"]].round(2))
display(tst_ref[["modèle", "type de bien", "année jugée", "écart de MAE avec la référence (€/m²)",
                 "IC 95 % bas (€/m²)", "IC 95 % haut (€/m²)", "p-value Wilcoxon", "rang-bisériale"]].round(3))


# #### Règle de décision contre la référence
# Même règle que contre la naïve, avec le modèle « prix N-1 seul » à la place de la naïve.

# In[17]:


regle_decision(res_ref, tst_ref, comparaison="référence")


# **Comment lire le tableau des tests contre la référence.** Prenons « Lasso – modèle complet », maisons, 2024 : le gain sur « prix N-1 seul » est de 8,3 % et l'IC 95 % est entièrement sous 0 (haut à −14 €/m²) : le modèle fait vraiment mieux que la référence.
# 
# **Résultats contre le modèle « prix N-1 seul » :**
# - **Modèle complet (parc + vacance + densité) :** Ridge, Lasso, Elastic Net et hiérarchique sont retenus pour les deux types de bien (2 années sur 2). Gains de 7 à 11 %.
# - **Piste H3a, équipements en plus :** les trois régularisées restent retenues ; pour les maisons le gain monte un peu (8 à 10 % contre 7 à 8 %), pour les appartements il ne change pas. Écart de quelques €/m² : à ne pas sur-lire.
# - **Régularisation seule (prix N-1 seul) :** Ridge apporte 1 à 2 % ; Lasso apporte 6 % (2023) et 9 % (2024) aux appartements, et rien aux maisons (0 à 1 %). Pour les appartements, une grande part du gain du modèle complet vient donc de la régularisation du prix N-1, comme le suggérait H3a ; pour les maisons, il vient bien des variables du bloc.
# - **Gradient boosting :** gain de 2 à 5 % ; retenu pour les maisons en 2024 seulement (2023 : 1,8 %, sous le seuil), pas pour les appartements. **Arbre et forêt :** moins bons que la référence (−4 à −20 %), comme attendu quand on prévoit le niveau plutôt que l'évolution.
# - **Modèle commun :** moins bon que le séparé pour les appartements (−5 %), un peu meilleur pour les maisons (+2 à +3 %).
# - **Réserve :** ces gains sont mesurés contre une référence qui perd elle-même contre la naïve. Ils ne disent pas que le modèle est utilisable.

# ## 6. Interprétation

# ### 6.1 Contexte : le marché s'est-il retourné ?
# Les modèles apprennent sur les années de hausse 2020-2022, puis sont jugés sur 2023 et 2024. Pour lire leurs résultats, on mesure l'évolution du prix d'une année sur l'autre (prix / prix N-1 − 1), en moyenne et en médiane, par année et par type de bien, jusqu'à 2024. L'année 2025, réservée au test final, n'est pas regardée.

# In[18]:


jusqua_2024 = table[table["annee"] <= commun.ANNEE_VALIDATION]
evolution = (jusqua_2024[CIBLE] / jusqua_2024["prix_n1"] - 1) * 100
retournement = (evolution.groupby([jusqua_2024["annee"], jusqua_2024.index.get_level_values("type_bien")])
                .agg(["mean", "median"]).round(1)
                .rename(columns={"mean": "évolution moyenne (%)", "median": "évolution médiane (%)"}))
retournement.unstack()


# ### 6.2 Importance des variables
# Importance par permutation, sur l'année jugée : on mélange au hasard les valeurs d'une variable entre les sous-bassins, 10 fois, et on mesure de combien l'erreur du modèle change (moyenne et écart-type). Une hausse : la variable était utile ; 0 : le modèle ne s'en sert pas ; une baisse : sur cette année, la variable dessert le modèle. Une variable compte si sa valeur dépasse nettement deux écarts-types. Choisir le modèle, l'année jugée et le type de bien.

# In[19]:


choix = (LASSO, 2024, "maison")        # modèle, année jugée, type de bien étudiés (à adapter)
m, X_ev, y_ev = ajustes[choix]
imp = permutation_importance(m, X_ev, y_ev, scoring="neg_mean_absolute_error",
                             n_repeats=10, random_state=commun.GRAINE)
pd.DataFrame({"hausse de la MAE (€/m²)": imp.importances_mean,
              "écart-type (€/m²)": imp.importances_std},
             index=X_ev.columns).sort_values("hausse de la MAE (€/m²)", ascending=False).round(1)


# ### 6.3 Analyse des erreurs selon le contrôle qualité
# L'erreur est-elle plus grande là où les données sont estimées ou complétées ? Pour chaque variable de contrôle qualité : groupe « part nulle », puis quartiles parmi les parts non nulles. Par groupe : erreur moyenne en €/m² et en % du prix (l'erreur en €/m² grandit avec le niveau des prix), et erreur moyenne signée (positive : le modèle surestime). Rho de Spearman entre la variable et l'erreur, en €/m² et en %. Même modèle qu'en 6.2.

# In[20]:


CONTROLE = ["part_indicateurs_estimes", "part_logements_taux_impute", "part_menages_couverts"]

m, X_ev, y_ev = ajustes[choix]
erreur = pd.Series(m.predict(X_ev), index=X_ev.index) - y_ev       # signée : positive = prix surestimé
erreur_pct = erreur.abs() / y_ev * 100                             # erreur absolue en % du prix réel
controle = table.loc[X_ev.index, CONTROLE]

for v in CONTROLE:
    x = controle[v]
    groupe = pd.Series("part nulle", index=x.index)
    non_nul = x > 0
    if non_nul.sum() >= 4:
        groupe[non_nul] = pd.qcut(x[non_nul], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
    tableau = pd.DataFrame({"MAE (€/m²)": erreur.abs(), "erreur en % du prix": erreur_pct,
                            "erreur signée (€/m²)": erreur}).groupby(groupe).mean()
    tableau["lignes"] = groupe.value_counts()
    rho, p = stats.spearmanr(x, erreur.abs())
    rho_pct, _ = stats.spearmanr(x, erreur_pct)
    print(f"\n{v} : rho avec l'erreur en €/m² = {rho:.2f} (p = {p:.3f}) | rho avec l'erreur en % = {rho_pct:.2f}")
    print(tableau.round(1).to_string())


# ### 6.4 Quelles variables survivent au Lasso ?
# Coefficients du Lasso « modèle complet » (variables transformées et mises à l'échelle, cible en log), année jugée 2024, par type de bien. Un coefficient nul : la variable est écartée à l'alpha retenu.

# In[21]:


def coefficients(nom, annee, type_bien):
    m = ajustes[(nom, annee, type_bien)][0]
    est = m.best_estimator_ if hasattr(m, "best_estimator_") else m
    return pd.Series(est.regressor_.named_steps["modele"].coef_, index=MODELE_COMPLET)

coef = pd.DataFrame({t: coefficients(LASSO, 2024, t) for t in ["appartement", "maison"]})
print({t: f"{int((coef[t] != 0).sum())} variables sur {len(coef)}" for t in coef})
coef.loc[(coef != 0).any(axis=1)].sort_values("maison", key=abs, ascending=False).round(4)


# ### 6.5 Contributions de chaque variable (équivalent SHAP) pour le modèle retenu pour le T1
# Modèle étudié : **Elastic Net – modèle complet**, année jugée 2024, par type de bien (choix fixé avant les résultats : le plus simple parmi des régularisées à quelques €/m² les unes des autres ; le Ridge sert de contrôle en 6.6).
# 
# Pour un modèle linéaire, la valeur SHAP d'une variable est exactement **coefficient × (valeur centrée-réduite)** : on la calcule sans la bibliothèque `shap`, avec le même résultat. La cible est en log, une contribution de 0,05 se lit « environ +5 % sur le prix ». On donne la contribution absolue moyenne de chaque variable sur l'année jugée (son importance).

# In[22]:


MODELE_T1 = ELASTIC          # Elastic Net – modèle complet (choix fixé avant les résultats)
TYPES_T1 = ["appartement", "maison"]

def contributions(nom, annee, type_bien):
    """Contributions linéaires (équivalent SHAP, en log du prix) de chaque variable sur l'année jugée."""
    m, X_ev, _ = ajustes[(nom, annee, type_bien)]
    est = m.best_estimator_.regressor_ if hasattr(m, "best_estimator_") else m.regressor_
    Z = est.named_steps["echelle"].transform(X_ev)            # variables transformées, centrées-réduites
    coef = est.named_steps["modele"].coef_
    contrib = pd.DataFrame(Z * coef, index=X_ev.index, columns=X_ev.columns)
    # Contrôle : intercept + somme des contributions = prévision en log
    assert np.allclose(est.named_steps["modele"].intercept_ + contrib.sum(axis=1), est.predict(X_ev))
    return contrib

tab = {}
for tb in TYPES_T1:
    c = contributions(MODELE_T1, 2024, tb)
    tab[tb] = c.abs().mean() * 100
shap_tab = pd.DataFrame(tab).sort_values("maison", ascending=False).rename_axis("importance : contribution absolue moyenne (%)")
shap_tab.round(2)


# **Lecture.** `prix_n1` porte presque toute la prévision : sa contribution absolue moyenne est d'environ 40 % pour les appartements et 35 % pour les maisons, contre 2 % au plus pour chaque autre variable (résidences secondaires, densité, vacance en tête ; quatre variables sont mises à zéro pour les appartements : part_sans_voiture, part_apparts, log_nb_logements, anciennete_moyenne). Autrement dit, le modèle prévoit surtout *l'écart de niveau entre les territoires*, déjà contenu dans le prix N-1, et les variables du territoire n'y changent presque rien. Ces contributions décrivent le modèle sur une année et un type de bien, sans test.

# ### 6.6 Diagnostic T1 : part commune et part propre de l'erreur
# **Question :** l'erreur de 2024 vient-elle d'un décalage **commun à tous les territoires** (le « quand » : le marché a changé de phase) ou de l'erreur **propre à chaque territoire** (le « où ») ?
# 
# **Calcul**, par type de bien et par année jugée :
# - erreur d'un territoire = prévision − prix réel (positive : surestimation) ;
# - **part commune** = erreur moyenne de tous les territoires ; **part propre** = erreur du territoire − part commune ;
# - on retire la part commune à la prévision et on compare à la naïve avec `evaluer` et `tester`, de deux façons : contre la **naïve telle quelle** (comme demandé par la feuille de route), et contre la **naïve elle aussi privée de sa part commune** (comparaison à égalité : la naïve a aussi une part commune, la lui laisser avantagerait le modèle).
# 
# **Limite à retenir :** la part commune utilise l'erreur réelle de l'année jugée. C'est un outil de diagnostic, pas un modèle : on ne peut pas la connaître avant l'année. Le Ridge sert de contrôle, pour vérifier que la lecture ne dépend pas du modèle choisi.

# In[23]:


resultats_t1, tests_t1, parts = [], [], []
for nom_modele, etiquette in [(ELASTIC, "Elastic Net"), (RIDGE, "Ridge")]:
    for annee, (app, ev) in FENETRES.items():
        prev = previsions[(nom_modele, annee)]
        type_bien = ev.index.get_level_values("type_bien")
        err = prev - ev[CIBLE]                                              # signée : positive = surestimation
        part_commune = err.groupby(type_bien).transform("mean")             # erreur moyenne de tous les territoires
        prev_sans = prev - part_commune                                     # prévision sans la part commune
        err_naive = ev["prix_n1"] - ev[CIBLE]
        part_commune_naive = err_naive.groupby(type_bien).transform("mean")
        ev_naive_sans = ev.copy()
        ev_naive_sans["prix_n1"] = ev["prix_n1"] - part_commune_naive      # naïve sans sa part commune
        nom1 = f"{HYPOTHESE} – T1 – {etiquette} sans part commune, contre naïve"
        nom2 = f"{HYPOTHESE} – T1 – {etiquette} sans part commune, contre naïve sans part commune"
        evaluer(nom1, ev, prev_sans, resultats_t1); tester(nom1, ev, prev_sans, tests_t1)
        evaluer(nom2, ev_naive_sans, prev_sans, resultats_t1); tester(nom2, ev_naive_sans, prev_sans, tests_t1)
        for tb in TYPES_T1:
            sel = type_bien == tb
            parts.append({"modèle": etiquette, "année jugée": annee, "type de bien": tb,
                          "part commune du modèle (€/m²)": part_commune[sel].iloc[0],
                          "part commune de la naïve (€/m²)": part_commune_naive[sel].iloc[0],
                          "MAE modèle (€/m²)": err[sel].abs().mean(),
                          "MAE modèle sans part commune (€/m²)": (err - part_commune)[sel].abs().mean(),
                          "MAE naïve (€/m²)": err_naive[sel].abs().mean(),
                          "MAE naïve sans part commune (€/m²)": (err_naive - part_commune_naive)[sel].abs().mean()})
parts = pd.DataFrame(parts).set_index(["modèle", "année jugée", "type de bien"])
parts.round(1)


# In[24]:


rt, tt = pd.DataFrame(resultats_t1), pd.DataFrame(tests_t1)
cles = ["modèle", "type de bien", "année jugée"]
t1 = rt[cles + ["gain sur la naïve (%)"]].merge(tt[cles + ["IC 95 % bas (€/m²)", "IC 95 % haut (€/m²)", "rang-bisériale"]], on=cles)
t1["comparaison"] = t1["modèle"].str.contains("contre naïve sans part commune").map({True: "naïve sans part commune (égalité)", False: "naïve telle quelle"})
t1["modèle"] = t1["modèle"].str.extract(r"T1 – (Elastic Net|Ridge)")
t1["règle 1.5 remplie"] = (t1["IC 95 % haut (€/m²)"] < 0) & (t1["rang-bisériale"] < 0) & (t1["gain sur la naïve (%)"] >= SEUIL_GAIN)
t1.sort_values(["année jugée", "type de bien", "modèle", "comparaison"]).round(2).reset_index(drop=True)


# **Lecture du T1 (2024, Elastic Net ; le Ridge donne la même lecture).**
# - **La part commune pèse un quart de l'erreur.** Le modèle surestime de **+150 €/m²** en moyenne pour les appartements et de **+174 €/m²** pour les maisons ; la naïve, de +41 et +71 €/m². En retirant la part commune, la MAE du modèle passe de 197 à **148 €/m²** (appartements, −25 %) et de 197 à **141 €/m²** (maisons, −29 %).
# - **Mais même sans part commune, le modèle ne bat pas la naïve en 2024.** Appartements : 148 €/m² contre 148 pour la naïve (égalité : gain de −0,03 %, IC contenant 0) ; maisons : 141 contre 129 (gain de −8,8 %, IC entièrement au-dessus de 0 : le modèle fait vraiment moins bien). À égalité, en privant aussi la naïve de sa part commune, c'est encore moins favorable (−2,0 % et −18,9 %).
# - **2023 est plus nuancé.** Pour les appartements, le modèle sans part commune bat la naïve telle quelle (+9,0 %, IC entièrement sous 0 ; Ridge +5,6 %), mais pas à égalité (+3,1 %, IC contenant 0). Pour les maisons, non (−5,5 %). La règle 1.5 est donc remplie une année sur deux, pour les appartements seulement : pas retenu.
# - **Lecture selon la grille de la feuille de route :** « même sans la part commune, il ne bat pas la naïve » → **le « où » n'apporte rien de stable au-delà du prix N-1** ; le niveau 2 est utile et le gain attendu vient du « quand » (phase du marché). Cette lecture est à confirmer collectivement ; elle repose sur deux années jugées, toutes deux pendant le retournement.

# 
# ## 7. Conclusion
# **Questions :** (1.0) modèle commun ou séparé ? Une fois les blocs retenus réunis : (1.2) quel gain reste-t-il au-delà du prix N-1 et quelles variables survivent ? (1.3) l'effet propre du sous-bassin apporte-t-il quelque chose ? (1.4) un arbre, une forêt ou un boosting font-ils mieux ?
# 
# **0. Modèle commun ou séparé (1.0) :** séparé (section 4.1 bis). Le commun gagne de 5 à 6 €/m² pour les maisons et perd de 10 €/m² pour les appartements : l'écart est faible et va dans les deux sens. *À confirmer.*
# 
# **1. Contre la naïve : aucun modèle n'est retenu** (0 année réussie sur 2, tous modèles confondus : régularisées, hiérarchique, arbre, forêt, boosting). Gain sur la naïve de −1 % à −83 %. Cause, commune à H3a, H3b et H3c : les modèles apprennent sur 2020-2022 (+5 à +10 % par an) et sont jugés sur 2023 (+1 à +3 %) puis 2024 (−1 à −3 %) ; ils prolongent la hausse et surestiment les prix. L'erreur signée est positive partout (+100 à +340 €/m² selon les groupes de contrôle qualité). Le modèle « prix N-1 seul » perd déjà : le problème ne vient pas des blocs.
# 
# **2. Contre « prix N-1 seul » : le modèle complet apporte un gain réel mais modeste (7 à 11 %).**
# - Ridge, Lasso, Elastic Net et le hiérarchique sont retenus pour les deux types de bien. Le gradient boosting l'est pour les maisons en 2024 seulement ; l'arbre et la forêt ne le sont pas.
# - **Où vient le gain :** pour les maisons, des variables du bloc (le Lasso sur le prix N-1 seul ne gagne rien) ; pour les appartements, en grande partie de la régularisation du prix N-1 (Lasso sur le prix N-1 seul : 6 à 9 %), ce qui confirme la remarque de H3a.
# - **Équipements (piste H3a) :** un petit gain de plus pour les maisons (8 à 10 % contre 7 à 8 %), rien pour les appartements.
# 
# **3. Variables (1.2).** `prix_n1` domine très largement (sections 6.2 et 6.5 : environ 35 à 40 % de contribution moyenne contre 2 % au plus pour chaque autre variable). Le Lasso garde 8 variables sur 12 pour les appartements et les 12 pour les maisons (section 6.4). Après `prix_n1`, les plus fortes sont la densité, la part de résidences secondaires et la vacance. En 2024, seules `prix_n1` et la part de résidences secondaires aident (importance par permutation) : les autres ont un effet appris pendant la hausse qui dessert le modèle pendant le retournement.
# 
# **4. Hiérarchique (1.3).** L'ICC est de 0 à 0,0008 : l'effet propre du sous-bassin n'apporte rien au-delà du prix N-1 et le modèle se comporte comme la régression simple (même MAE à 1-2 €/m² près). Un modèle (maisons 2023) n'a pas convergé, ce qui est attendu quand la variance de groupe est quasi nulle.
# 
# **5. Arbres et boosting (1.4).** Le gradient boosting (évolution du prix) fait mieux que l'arbre et la forêt (niveau du prix) mais moins bien que les régularisées. L'emploi pour les maisons (H3c) n'est pas encore ajouté (section 4.6).
# 
# **6. Qualité des données :** l'erreur n'est pas concentrée dans les territoires aux données estimées ou complétées ; rapportée au prix, elle est de 8 à 10 % dans tous les groupes (section 6.3).
# 
# **Modèle retenu pour le diagnostic T1 : Elastic Net – modèle complet** (règle fixée avant les résultats : le plus simple parmi des régularisées à 2-4 % les unes des autres ; il est le meilleur ou à égalité sur les deux types de bien et les deux années ; le Ridge sert de contrôle). Appartements 2024 : 197 €/m² contre 148 pour la naïve ; maisons 2024 : 197 contre 129.
# 
# **7. Diagnostic T1 (section 6.6).** Le modèle surestime de +150 €/m² (appartements) et +174 €/m² (maisons) en 2024, un décalage commun à tous les territoires qui pèse un quart de l'erreur (MAE 197 → 148 et 141 €/m² sans part commune). Même sans elle, le modèle ne bat pas la naïve en 2024 (appartements : égalité ; maisons : −8,8 %). **Le « où » n'apporte rien de stable au-delà du prix N-1 ; le gain attendu est dans le « quand ».** À lire collectivement avec les autres chiffres.
# 
# **À décider avec l'équipe avant d'ouvrir 2025 :** aucun modèle n'est retenu contre la naïve, donc la règle « test 2025 pour le modèle retenu » ne désigne personne. Soit le meilleur modèle du niveau 1 (Elastic Net complet) sert de candidat, soit la naïve reste la référence du niveau 1 et le test confirme qu'aucun modèle ne la bat. À noter avant l'ouverture du test.
# 
# **Limites :** deux années jugées, toutes deux pendant le retournement ; plusieurs variantes comparées sur les mêmes années (celle qui ressort a pu en profiter) ; l'importance des variables porte sur une année et un type de bien ; l'emploi n'est pas inclus ; plusieurs `alpha` tombent sur le bord de la grille (maisons : 1e-4 et 1e-5, proche d'une régression non pénalisée).
# 
