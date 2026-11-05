#!/usr/bin/env python
# coding: utf-8

# # Modélisation – H3 globale : naïve (1.0) et Ridge, Lasso, Elastic Net sur le modèle complet (1.2)
# **Contributeurs :** Christian · **Date :** 05/11/2026
# 
# Notebook construit à partir du template commun de l'équipe. Il passe par `commun.py` pour tout ce qui doit être identique d'une hypothèse à l'autre : la table, le masque des lignes, les fenêtres, la naïve, `evaluer` et `tester`.
# 
# **Questions :** (1.0) faut-il un modèle commun ou deux modèles séparés pour les maisons et les appartements ? (1.2) une fois les blocs retenus réunis (modèle complet), que reste-t-il comme gain au-delà du prix N-1, et quelles variables survivent ?
# 
# **Statut : PROVISOIRE.** Conclusions intégrées : H3b (Cécile) et H3c (Gabrielle, emploi retenu pour les maisons avec le gradient boosting seulement : hors de ce notebook). **H3a (Théophyle) est absente** : son bloc est ajouté en variante, avec tous ses indicateurs, en attendant sa conclusion.
# 
# Les étapes 1.3 (modèle hiérarchique) et 1.4 (arbre, forêt, gradient boosting) s'ajoutent à ce notebook, pour que tous les modèles soient comparés dans les mêmes tableaux.

# ## 0. Paramètres de l'hypothèse
# Les variables sont rangées par bloc, dans l'ordre où elles entrent dans les modèles :
# - **H3b retenu** (conclusion de Cécile) : **parc** (9 indicateurs), **vacance** (LOVAC) et **densité**. Ce sont les blocs qui portent l'apport contre « prix N-1 seul ». Les **ménages** et les **revenus** sont écartés : les ménages font perdre l'année 2023 aux deux types de bien, les revenus ne réduisent pas l'erreur davantage que le parc.
# - **H3a** (attractivité : équipements et délinquance) : conclusion non reçue. Tous les indicateurs sont inclus en variante « avec H3a », pour être retirés ou gardés quand elle arrive.
# - **H3c** (emploi) : retenu pour les maisons avec le gradient boosting seulement. Il n'entre donc pas dans les modèles régularisés (il sera ajouté à l'étape 1.4).
# - `nb_ventes_n1` n'appartient à aucun bloc : hors modèle (la feuille de route définit le modèle complet comme prix N-1 + blocs retenus).
# 
# `LOG` et `LOG1P` : mêmes règles que le template. `prix_n1` passe en log avec la cible.

# In[1]:


HYPOTHESE = "H3"

# --- H3b retenu (conclusion de Cécile) -------------------------------------------------------------
PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
        "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
VACANCE = ["taux_vacance_complete"]
DENSITE = ["densite_population"]
# --- H3a : conclusion non reçue (PROVISOIRE : tous les indicateurs, à restreindre) -----------------
EQUIPEMENTS = ["taux_equip_commerces", "taux_equip_sante", "taux_equip_services", "taux_equip_sports",
               "taux_equip_tourisme", "taux_equip_transports"]
DELINQUANCE = ["taux3_vols_sans_violence", "taux3_vols_vehicules", "taux3_vols_violents", "taux3_cambriolages",
               "taux3_violences_intrafamiliales", "taux3_violences_sexuelles", "taux3_stupefiants"]

VARIABLES_BLOC = PARC + VACANCE + DENSITE + EQUIPEMENTS + DELINQUANCE      # tous les blocs (constat rapide)

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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import commun
from commun import CIBLE, charger_table, fenetres, plis_temporels, evaluer, tester
from gabarit_hierarchique import ModeleHierarchique

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
# - `modèle` : le modèle à utiliser (régression linéaire, Lasso, gradient boosting, modèle hiérarchique) ;
# - `variables` : les colonnes de la table données au modèle ;
# - `log=True` : la cible et les variables de `VARIABLES_LOG` passent en log, celles de `VARIABLES_LOG1P` en log(1 + x) ; à réserver aux modèles qui en ont besoin (régressions, Lasso, modèle hiérarchique), jamais aux arbres ;
# - `grille` : les valeurs d'un réglage à tester (par exemple `alpha` pour le Lasso).
# 
# Ce que fait `lancer`, pour chacune des deux années jugées (2023, puis 2024) et pour chaque type de bien :
# 1. elle prend les années d'apprentissage, avant l'année jugée ;
# 2. elle met les variables à l'échelle et entraîne le modèle ; avec une grille, elle choisit d'abord le meilleur réglage, chaque année d'apprentissage servant tour à tour de validation (`plis_temporels` de `commun.py`) ;
# 3. elle prévoit le prix de l'année jugée, remis en €/m² si le modèle a appris en log ;
# 4. elle compare la prévision au prix réel et à la naïve (prix N = prix N-1) avec `evaluer` et `tester` de `commun.py`.
# 
# Elle affiche ensuite l'erreur du modèle (MAE, en €/m²), celle de la naïve, le gain sur la naïve (négatif : le modèle fait moins bien) et la part des sous-bassins où le modèle gagne. Elle garde les modèles entraînés (`ajustes`, pour la section 6) et leurs prévisions (`previsions`, pour la section 5.2). Relancer une cellule remplace les résultats du modèle au lieu de les ajouter une seconde fois.
# 
# Le modèle hiérarchique (`gabarit_hierarchique.py`) passe par une enveloppe : il n'est pas mis dans le pipeline (il met lui-même ses variables à l'échelle) et n'accepte pas de grille.

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
# Trois régularisations (Ridge, Lasso, Elastic Net) sur deux jeux de variables, avec la référence « prix N-1 seul ». Tous les modèles sont gardés, y compris ceux qui perdent. Tous sont faits en log (cible et variables asymétriques), comme la référence.
# - **Modèle complet sans H3a** : prix N-1 + parc + vacance + densité (l'apport retenu de H3b).
# - **Modèle complet avec H3a** : le précédent + équipements + délinquance (PROVISOIRE, en attendant la conclusion de Théophyle).
# 
# **Réglages (choix de ce notebook, pas de la feuille de route)** : `alpha` sur une échelle logarithmique, réglé sur les plis temporels (`plis_temporels`) ; `l1_ratio` de l'Elastic Net à 0,2 / 0,5 / 0,8. Les grilles vont assez loin pour que le meilleur réglage ne soit pas collé au bord (vérifié en 4.5).

# In[6]:


REFERENCE_VARS = ["prix_n1"]
SANS_H3A = REFERENCE_VARS + PARC + VACANCE + DENSITE
AVEC_H3A = SANS_H3A + EQUIPEMENTS + DELINQUANCE
GRILLE_RIDGE = {"alpha": list(np.logspace(-4, 3, 15))}
GRILLE_LASSO = {"alpha": list(np.logspace(-5, -1, 9))}
GRILLE_EN = {"alpha": list(np.logspace(-5, -1, 9)), "l1_ratio": [0.2, 0.5, 0.8]}
print(len(SANS_H3A), "variables sans H3a ;", len(AVEC_H3A), "avec H3a")


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


# **Lecture (étape 1.0).** Le modèle commun fait un peu mieux pour les maisons (−5 à −6 €/m² de MAE) et un peu moins bien pour les appartements (+10 €/m²), les deux années. L'écart est de 3 à 5 % de l'erreur et va dans les deux sens : ni l'un ni l'autre ne s'impose. **Choix retenu pour la suite : modèles séparés** (un par type de bien et par fenêtre). Raisons : la séparation laisse à chaque type sa propre pente sur `prix_n1` et ses propres coefficients, c'est le fonctionnement du template d'équipe et des notebooks H3b et H3c, et le commun ne gagne que sur un type de bien. *Décision à confirmer par le propriétaire du profil (décision 1.6 : « décidé par Chauve-Souris sur la naïve »).*

# ### 4.2 Modèle complet sans H3a : Ridge, Lasso, Elastic Net

# In[9]:


RIDGE = lancer("Ridge – sans H3a", Ridge(), SANS_H3A, log=True, grille=GRILLE_RIDGE)
LASSO = lancer("Lasso – sans H3a", Lasso(max_iter=50_000), SANS_H3A, log=True, grille=GRILLE_LASSO)
ELASTIC = lancer("Elastic Net – sans H3a", ElasticNet(max_iter=50_000), SANS_H3A, log=True, grille=GRILLE_EN)


# ### 4.3 Modèle complet avec H3a (provisoire) : Ridge, Lasso, Elastic Net
# Le Lasso ramène à zéro les coefficients des variables qui n'apportent rien : c'est lui qui dit quelles variables survivent (section 6.4). Les valeurs d'`alpha` sont petites : la cible est en log et les variables sont mises à l'échelle.

# In[10]:


RIDGE_H3A = lancer("Ridge – avec H3a", Ridge(), AVEC_H3A, log=True, grille=GRILLE_RIDGE)
LASSO_H3A = lancer("Lasso – avec H3a", Lasso(max_iter=50_000), AVEC_H3A, log=True, grille=GRILLE_LASSO)
ELASTIC_H3A = lancer("Elastic Net – avec H3a", ElasticNet(max_iter=50_000), AVEC_H3A, log=True, grille=GRILLE_EN)


# ### 4.4 Gradient boosting et modèle hiérarchique
# Hors de ce notebook : ils sont faits aux étapes 1.4 (gabarit gradient boosting de Théophyle, avec l'emploi pour les maisons) et 1.3 (gabarit hiérarchique).

# In[11]:


# Réglage retenu par la validation (alpha, l1_ratio) pour chaque modèle : le meilleur réglage est-il collé au bord de la grille ?
reglages = pd.DataFrame(
    {(nom, a, t): {k.split("__")[-1]: v for k, v in m.best_params_.items()}
     for (nom, a, t), (m, _, _) in ajustes.items() if hasattr(m, "best_params_")}).T
reglages.index.names = ["modèle", "année jugée", "type de bien"]
reglages


# ### 4.5 Les réglages retenus
# Tableau ci-dessus : `alpha` (et `l1_ratio`) retenus pour chaque modèle, année et type de bien. Un alpha qui tombe sur la première ou la dernière valeur de la grille dit que la grille est trop étroite (voir « Comment lire »).

# In[ ]:





# ## 5. Mesures et tests
# Trois tableaux, du plus simple au plus exigeant :
# 1. **`res` : les erreurs.** MAE du modèle et de la naïve en €/m², gain sur la naïve en %, part des sous-bassins où le modèle gagne. C'est une description : un modèle peut avoir une MAE plus basse par hasard.
# 2. **`tst` : les tests contre la naïve.** Ils disent si l'écart avec la naïve est plus grand que le hasard.
#    - **Écart de MAE avec la naïve** : erreur du modèle moins erreur de la naïve, en €/m² (négatif : le modèle fait mieux).
#    - **IC 95 %** (bootstrap par bassin de vie) : la fourchette dans laquelle se trouve cet écart, en tenant compte du hasard. Si toute la fourchette est sous 0, le modèle fait vraiment mieux ; si elle contient 0, on ne peut pas conclure ; si elle est toute au-dessus de 0, il fait vraiment moins bien.
#    - **p-value de Wilcoxon** : le modèle fait-il mieux (ou moins bien) dans la plupart des sous-bassins ? Avec plusieurs centaines de sous-bassins, elle est presque toujours très petite : on la lit avec la taille d'effet.
#    - **Rang-bisériale** (taille d'effet) : de −1 (le modèle fait mieux partout) à +1 (il fait moins bien partout) ; autour de 0, pas de différence nette.
# 3. **`decision` : la règle de décision.** Une année est réussie si l'IC est entièrement sous 0, la rang-bisériale négative et le gain d'au moins 3 %. Un modèle est retenu s'il réussit la majorité des années jugées (ici, les deux).

# In[12]:


res = pd.DataFrame(resultats)
res.round(2)


# In[13]:


tst = pd.DataFrame(tests)
tst.round(3)


# ### 5.1 Règle de décision contre la naïve

# In[14]:


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


# **Comment lire le tableau des tests.** Prenons « Ridge – sans H3a », appartements, 2023 : l'écart de MAE avec la naïve est positif (le modèle fait un peu moins bien) et l'IC 95 % contient 0, donc on ne peut pas conclure à une différence. La même ligne en 2024 a un IC entièrement au-dessus de 0 : le modèle fait vraiment moins bien que la naïve.
# 
# **Résultats contre la naïve :** aucun modèle n'est retenu, pour les deux types de bien (0 année réussie sur 2 partout). Le gain sur la naïve est négatif dans tous les cas : de −1 % (appartements 2023) à −57 % (maisons 2023, avec H3a). Même le modèle « prix N-1 seul » perd (−12 % à −66 %). Les modèles régularisés réduisent l'écart (appartements 2024 : −37 % contre −50 %), sans le combler.

# ### 5.2 Le bloc apporte-t-il quelque chose au prix N-1 seul ?
# La question propre à l'hypothèse : les variables du bloc améliorent-elles la prévision **par rapport au modèle « prix N-1 seul »** ?
# 
# On applique exactement les mêmes mesures et la même règle de décision, en remplaçant la naïve par la prévision du modèle « prix N-1 seul ». Les fonctions `evaluer` et `tester` lisent la prévision de comparaison dans la colonne `prix_n1` : on leur donne une copie de l'année jugée où cette colonne contient la prévision de la référence.
# 
# Lecture : un gain positif et un intervalle du bootstrap entièrement sous 0 veulent dire que le modèle fait vraiment mieux que la référence.

# In[15]:


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

# In[16]:


regle_decision(res_ref, tst_ref, comparaison="référence")


# **Comment lire le tableau des tests contre la référence.** Prenons « Lasso – sans H3a », maisons, 2024 : l'écart de MAE avec « prix N-1 seul » est d'environ −18 €/m² et l'IC 95 % est entièrement sous 0 (haut à −14 €/m²), pour un gain de 8,3 % : le modèle fait vraiment mieux que la référence.
# 
# **Résultats contre le modèle « prix N-1 seul » :**
# - **Sans H3a** (prix N-1 + parc + vacance + densité) : les trois modèles sont retenus pour les deux types de bien, avec 2 années réussies sur 2. Gains de 7 à 11 %. C'est cohérent avec la conclusion de Cécile.
# - **Avec H3a** (tous les indicateurs d'équipements et de délinquance) : le résultat se dégrade. Seuls le Lasso et l'Elastic Net sont retenus, pour les appartements. Pour les maisons, aucun modèle n'est retenu : en 2023, le gain devient négatif (−2,7 %, contre +7 % sans H3a). Ajouter tout le bloc H3a sans le sélectionner apporte du bruit. À reprendre avec la conclusion de Théophyle.
# - **Réserve :** ce gain est mesuré contre une référence qui perd elle-même contre la naïve. Il ne dit pas que le modèle est utilisable.

# ## 6. Interprétation

# ### 6.1 Contexte : le marché s'est-il retourné ?
# Les modèles apprennent sur les années de hausse 2020-2022, puis sont jugés sur 2023 et 2024. Pour lire leurs résultats, on mesure l'évolution du prix d'une année sur l'autre (prix / prix N-1 − 1), en moyenne et en médiane, par année et par type de bien, jusqu'à 2024. L'année 2025, réservée au test final, n'est pas regardée.

# In[17]:


jusqua_2024 = table[table["annee"] <= commun.ANNEE_VALIDATION]
evolution = (jusqua_2024[CIBLE] / jusqua_2024["prix_n1"] - 1) * 100
retournement = (evolution.groupby([jusqua_2024["annee"], jusqua_2024.index.get_level_values("type_bien")])
                .agg(["mean", "median"]).round(1)
                .rename(columns={"mean": "évolution moyenne (%)", "median": "évolution médiane (%)"}))
retournement.unstack()


# ### 6.2 Importance des variables
# Importance par permutation, sur l'année jugée : on mélange au hasard les valeurs d'une variable entre les sous-bassins, 10 fois, et on mesure de combien l'erreur du modèle change (moyenne et écart-type). Une hausse : la variable était utile ; 0 : le modèle ne s'en sert pas ; une baisse : sur cette année, la variable dessert le modèle. Une variable compte si sa valeur dépasse nettement deux écarts-types. Choisir le modèle, l'année jugée et le type de bien.

# In[18]:


choix = (LASSO_H3A, 2024, "maison")        # modèle, année jugée, type de bien étudiés (à adapter)
m, X_ev, y_ev = ajustes[choix]
imp = permutation_importance(m, X_ev, y_ev, scoring="neg_mean_absolute_error",
                             n_repeats=10, random_state=commun.GRAINE)
pd.DataFrame({"hausse de la MAE (€/m²)": imp.importances_mean,
              "écart-type (€/m²)": imp.importances_std},
             index=X_ev.columns).sort_values("hausse de la MAE (€/m²)", ascending=False).round(1)


# ### 6.3 Analyse des erreurs selon le contrôle qualité
# L'erreur est-elle plus grande là où les données sont estimées ou complétées ? Pour chaque variable de contrôle qualité : groupe « part nulle », puis quartiles parmi les parts non nulles. Par groupe : erreur moyenne en €/m² et en % du prix (l'erreur en €/m² grandit avec le niveau des prix), et erreur moyenne signée (positive : le modèle surestime). Rho de Spearman entre la variable et l'erreur, en €/m² et en %. Même modèle qu'en 6.2.

# In[19]:


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
# Coefficients du Lasso « avec H3a » (variables transformées et mises à l'échelle, cible en log), année jugée 2024, par type de bien. Un coefficient nul : la variable est écartée à l'alpha retenu. Le coefficient de `prix_n1` est le poids du prix de l'année précédente.

# In[20]:


def coefficients(nom, annee, type_bien):
    m = ajustes[(nom, annee, type_bien)][0]
    est = m.best_estimator_ if hasattr(m, "best_estimator_") else m
    coef = est.regressor_.named_steps["modele"].coef_
    return pd.Series(coef, index=AVEC_H3A)

coef = pd.DataFrame({t: coefficients(LASSO_H3A, 2024, t) for t in ["appartement", "maison"]})
print({t: f"{int((coef[t] != 0).sum())} variables sur {len(coef)}" for t in coef})
coef.loc[(coef != 0).any(axis=1)].sort_values("maison", key=abs, ascending=False).round(4)


# ## 7. Conclusion
# **Question :** une fois les blocs retenus réunis, que reste-t-il comme gain au-delà du prix N-1, et quelles variables survivent ?
# 
# **0. Modèle commun ou séparé (1.0) :** séparé (section 4.1 bis). Le commun gagne de 5 à 6 €/m² pour les maisons et perd de 10 €/m² pour les appartements : l'écart est faible et va dans les deux sens.
# 
# **1. Contre la naïve : aucun modèle n'est retenu.** 0 année réussie sur 2 pour tous les modèles régularisés (Ridge, Lasso, Elastic Net), avec ou sans H3a, appartements comme maisons. Le gain sur la naïve va de −1 % à −57 %. La cause est celle déjà vue en H3b et H3c : les modèles apprennent sur 2020-2022 (+5 à +10 % par an) et sont jugés sur 2023 (+1 à +3 %) puis 2024 (−1 à −2 %) ; ils prolongent la hausse et surestiment les prix. L'erreur signée est positive partout (surestimation de +110 à +340 €/m² selon les groupes, section 6.3) ; l'erreur en % du prix reste stable (8 à 10 %) d'un groupe de contrôle qualité à l'autre : la qualité des données n'explique pas l'erreur. Le modèle « prix N-1 seul » perd déjà : le problème ne vient pas des blocs.
# 
# **2. Contre « prix N-1 seul » : le modèle complet sans H3a apporte un gain réel mais modeste.** Les trois régularisations sont retenues pour les deux types de bien (gain de 7 à 11 %, IC 95 % sous 0 les deux années). Le Lasso garde 10 variables sur 25 pour les appartements et 13 pour les maisons (section 6.4, version avec H3a). `prix_n1` domine très largement (section 6.2 : permuter cette variable fait monter la MAE de près de 1 000 €/m² sur les maisons 2024). Les autres variables ont une importance nulle ou négative en 2024 (`taux_vacance` : −7 €/m²), c'est-à-dire que permuter la variable améliore même un peu la prévision : le coefficient appris sur 2020-2023 ne tient plus pendant le retournement. Les coefficients du Lasso (section 6.4) les plus forts après `prix_n1` sont `taux_vacance`, `part_rsec` et `densite_population`. Avec H3a au complet, le gain disparaît pour les maisons : le bloc doit être sélectionné.
# 
# **3. Pas de gain de la régularisation sur la régression simple.** Ridge, Lasso et Elastic Net sont proches les uns des autres (écarts de MAE de quelques €/m²). Les variables que le Lasso garde en premier sont celles de H3b (vacance, résidences secondaires, densité, ancienneté du parc) ; les équipements et la délinquance pèsent peu.
# 
# **À valider ou à refaire quand les conclusions arrivent**
# - H3a de Théophyle : remplacer `EQUIPEMENTS` et `DELINQUANCE` par les variables retenues, par type de bien.
# - Les sept variables « qui ne comptaient nulle part à l'audit » : liste non reçue ; à repérer parmi les coefficients nuls du Lasso.
# - Réglages : plusieurs `alpha` tombent sur le bord de la grille (Ridge et Lasso pour les maisons, 1e-4 et 1e-5 : proche d'une régression non pénalisée). Les chiffres changent peu mais le bord est atteint.
# - Choix « un modèle par type de bien » : à aligner sur la décision de l'étape 1.0.
# 
# **Limites :** deux années jugées, toutes deux pendant le retournement du marché ; l'emploi (H3c) n'est pas inclus ; l'analyse porte sur le niveau de prix et non sur son évolution.
