# =============================================================================
# 03_modelisation/test.py : contrôles « c'est fini quand » des préparations communes
# =============================================================================
# Remplace test_commun_p1.py, test_tester_p2.py et test_gabarit_hierarchique.py.
# À lancer depuis 03_modelisation :  python test.py
# Chaque contrôle est un assert : le script s'arrête au premier échec.
# Il ne modifie aucun fichier.
#   Partie 1 : P1 et P5 (masque, fenêtres, plis, tableau commun, non-régression)   -- Christian
#   Partie 2 : P2 (test contre la naïve, bootstrap par bassin de vie)               -- Cécile
#   Partie 3 : P4 (gabarit hiérarchique)                                            -- Christian

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

import commun
from commun import (ANNEES_JUGEES, CIBLE, charger_table, decouper, fenetres, plis_temporels,
                    masque_lignes_communes, prevision_naive, evaluer, tester)
from gabarit_hierarchique import ModeleHierarchique


def titre(t):
    print(f"\n--- {t}")


def prix_n1_seul(app, ev):
    """Prévision du modèle « prix N-1 seul », un modèle par type de bien."""
    prev = pd.Series(np.nan, index=ev.index)
    for type_bien, e in app.groupby("type_bien"):
        m = LinearRegression().fit(e[["prix_n1"]], e[CIBLE])
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
    assert prev.notna().all()
    return prev


brute = charger_table(masquer=False)
table = charger_table()
f = fenetres(table)


# =============================================================================
# PARTIE 1 : P1 ET P5
# =============================================================================

# ---------------------------------------------------------------- masque
titre("P1 -- Masque des lignes communes")
retirees = brute[~masque_lignes_communes(brute)]
print("lignes brutes :", len(brute), "| gardées :", len(table), "| retirées :", len(retirees))
print("lignes retirées par année :", retirees["annee"].value_counts().sort_index().to_dict())
assert len(table) + len(retirees) == len(brute)
variables = [c for c in table.columns if c not in commun.HORS_MASQUE]
assert table[variables].notna().all().all(), "il reste une case vide dans une variable explicative"
fenetre_niveau1 = retirees[retirees["annee"].between(commun.ANNEE_DEBUT, ANNEES_JUGEES[-1])]
print("dont sur 2020-2024 :", len(fenetre_niveau1))

# ---------------------------------------------------------------- fenêtres
titre("P1 -- Fenêtres")
assert list(f) == ANNEES_JUGEES
for a, (app, ev) in f.items():
    annees_app = sorted(app["annee"].unique())
    print(f"fenêtre {a} : apprentissage {annees_app} ({len(app)} lignes) | jugée {sorted(ev['annee'].unique())} ({len(ev)} lignes)")
    assert annees_app == list(range(commun.ANNEE_DEBUT, a)), "l'apprentissage doit être exactement ANNEE_DEBUT..a-1"
    assert (ev["annee"] == a).all() and app["annee"].max() < a
assert sorted(f[2023][0]["annee"].unique()) == [2020, 2021, 2022]      # critère de la feuille de route

# ---------------------------------------------------------------- plis
titre("P1 -- Plis temporels")
for a, (app, ev) in f.items():
    plis = plis_temporels(app)
    print(f"fenêtre {a} : {len(plis)} plis")
    for tr, va in plis:
        a_tr, a_va = app["annee"].to_numpy()[tr], app["annee"].to_numpy()[va]
        assert len(np.unique(a_va)) == 1, "un pli = une année de validation"
        assert a_tr.max() < a_va.min(), "l'apprentissage du pli doit précéder la validation"
        assert not set(tr) & set(va), "aucune ligne dans les deux ensembles"
        print(f"   apprentissage {sorted(set(a_tr))} -> validation {int(a_va[0])}")
assert len(plis_temporels(f[2023][0])) == 2 and len(plis_temporels(f[2024][0])) == 3   # décision 1.6

# ---------------------------------------------------------------- compatibilité GridSearchCV
titre("P1 -- Compatibilité GridSearchCV")
app, ev = f[2024]
for type_bien, e in app.groupby("type_bien"):
    # `e` garde l'ordre de `app` : les plis sont recalculés sur ce sous-ensemble, jamais réutilisés d'un autre
    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=plis_temporels(e),
                      scoring="neg_mean_absolute_error")
    gs.fit(e[["prix_n1"]], e[CIBLE])
    print(type_bien, "meilleur alpha :", gs.best_params_)

# ---------------------------------------------------------------- tableau commun « prix N-1 seul »
titre("P1 -- Tableau commun : modèle « prix N-1 seul », 2 années x 2 types de bien")
resultats, tests = [], []
for a, (app, ev) in f.items():
    prev = prix_n1_seul(app, ev)
    evaluer("P1 -- prix N-1 seul", ev, prev, resultats)
    tester("P1 -- prix N-1 seul", ev, prev, tests)
res = pd.DataFrame(resultats)
print(res.round(2).to_string())
assert len(res) == 4 and "année jugée" in res.columns and len(tests) == 4
assert sorted(res["année jugée"]) == [2023, 2023, 2024, 2024]
assert "année jugée" in tests[0]

# ---------------------------------------------------------------- non-régression
titre("P5 -- Non-régression : comportement d'avant")
app0, ev0 = decouper(table)                                   # valeurs par défaut
app1, ev1 = decouper(table, annee_eval=2024, annee_debut=2020)
assert app0.equals(app1) and ev0.equals(ev1)
assert app0["annee"].min() == 2020 and ev0["annee"].unique().tolist() == [2024]
app2, _ = decouper(table, annee_eval=2024, annee_debut=2022)  # P5 : le début est bien un paramètre
assert app2["annee"].min() == 2022
print("Tous les contrôles P1/P5 passent.")


# =============================================================================
# PARTIE 2 : P2 (bootstrap par bassin de vie)
# =============================================================================

def tester_ancien(evaluation, prevision):
    """Ancien bootstrap (sous-bassins tirés un par un), recopié ici pour la seule comparaison."""
    rng = np.random.default_rng(commun.GRAINE)
    lignes = {}
    for type_bien, e in evaluation.groupby("type_bien"):
        ecart = (np.abs(e[CIBLE] - prevision.loc[e.index]) - np.abs(e[CIBLE] - prevision_naive(e))).to_numpy()
        tirages = rng.integers(0, len(ecart), size=(commun.N_TIRAGES, len(ecart)))
        bas, haut = np.percentile(ecart[tirages].mean(axis=1), [2.5, 97.5])
        lignes[type_bien] = (ecart.mean(), haut - bas)
    return lignes


titre("P2 -- Bassin de vie de chaque sous-bassin")
assert "BV2022" in table.columns, "Colonne BV2022 absente : relancer le feature engineering"
assert "BV2022" in commun.HORS_MASQUE, "BV2022 doit être dans HORS_MASQUE (identifiant)"
sbv = table.index.get_level_values("sous_bassin_de_vie")
assert table.groupby(sbv)["BV2022"].nunique().eq(1).all(), "un sous-bassin a plusieurs bassins de vie"
print("sous-bassins :", sbv.nunique(), "| bassins de vie :", table["BV2022"].nunique())

titre("P2 -- Ancien et nouveau bootstrap, modèle « prix N-1 seul »")
lignes = []
for a, (app, ev) in f.items():
    prev = prix_n1_seul(app, ev)
    ancien = tester_ancien(ev, prev)
    nouveau = []
    tester("P2", ev, prev, nouveau)
    for t in nouveau:
        moy_ancien, larg_ancien = ancien[t["type de bien"]]
        larg_nouveau = t["IC 95 % haut (€/m²)"] - t["IC 95 % bas (€/m²)"]
        lignes.append({"année jugée": a, "type de bien": t["type de bien"],
                       "bassins de vie": t["bassins de vie"],
                       "écart moyen (€/m²)": t["écart de MAE avec la naïve (€/m²)"],
                       "largeur IC ancien": larg_ancien, "largeur IC par bassin": larg_nouveau,
                       "rapport des largeurs": larg_nouveau / larg_ancien})
        assert np.isclose(t["écart de MAE avec la naïve (€/m²)"], moy_ancien), "l'écart moyen ne doit pas changer"
comparaison = pd.DataFrame(lignes)
print(comparaison.round(2).to_string(index=False))
assert (comparaison["rapport des largeurs"] > 0.8).all(), "intervalle par bassin nettement plus étroit : à vérifier"
print("Tous les contrôles P2 passent.")


# =============================================================================
# PARTIE 3 : P4 (gabarit hiérarchique)
# =============================================================================
titre("P4 -- Modèle hiérarchique « prix N-1 seul »")
resultats, tests = [], []
for a, (app, ev) in f.items():
    prev = pd.Series(np.nan, index=ev.index)
    for type_bien, e in app.groupby("type_bien"):
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        m = ModeleHierarchique().fit(e[["prix_n1"]], e[CIBLE])
        prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
        print(f"{a} {type_bien:11s} convergé={m.converge_} | groupes vus={len(m.effets_aleatoires_)} "
              f"| lignes jugées à groupe inconnu={m.part_groupes_inconnus(ev.loc[idx, ['prix_n1']]):.1%}"
              f" | avertissements={len(m.avertissements_)} | ICC={m.icc_:.4f}")
    assert prev.notna().all()
    evaluer("P4 -- hiérarchique, prix N-1 seul", ev, prev, resultats)
    tester("P4 -- hiérarchique, prix N-1 seul", ev, prev, tests)
print(pd.DataFrame(resultats).round(2).to_string())

# Critère : un territoire absent de l'apprentissage reçoit seulement les effets fixes, sans erreur
app, ev = f[2024]
e = app[app.index.get_level_values("type_bien") == "maison"]
sbv_retire = e.index.get_level_values("sous_bassin_de_vie")[0]
e_sans = e[e.index.get_level_values("sous_bassin_de_vie") != sbv_retire]
m = ModeleHierarchique().fit(e_sans[["prix_n1"]], e_sans[CIBLE])
x_nouveau = e[e.index.get_level_values("sous_bassin_de_vie") == sbv_retire][["prix_n1"]]
p = m.predict(x_nouveau)
assert sbv_retire not in m.effets_aleatoires_ and p.notna().all()
beta = np.asarray(m.resultat_.fe_params)
attendu = beta[0] + beta[1] * (x_nouveau["prix_n1"] - m.moyennes_["prix_n1"]) / m.ecarts_["prix_n1"]
assert np.allclose(p, attendu), "pour un territoire inconnu la prévision doit être les effets fixes seuls"
print(f"Territoire nouveau ({sbv_retire}) : prévision = effets fixes seuls, OK")
print("Tous les contrôles P4 passent.")

print("\n=== Tous les contrôles (P1, P2, P4, P5) passent. ===")
