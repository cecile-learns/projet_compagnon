# =============================================================================
# tests/test_commun.py : contrôles de socle/commun.py
# =============================================================================
# Lancement depuis la racine du projet :  pytest tests/test_commun.py
# Lit data/processed/table_modelisation_bv.parquet ; ne modifie aucun fichier.
#   P1  : masque des lignes communes, fenêtres, plis temporels, tableau commun   -- Christian
#   P5  : non-régression du découpage                                             -- Christian
#   P2  : test contre la naïve, bootstrap par bassin de vie                       -- Cécile
#   P11 : évaluation du modèle par vente (ventes inventées)                       -- Cécile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

import commun
from commun import (ANNEES_JUGEES, CIBLE, charger_table, decouper, fenetres, plis_temporels,
                    masque_lignes_communes, prevision_naive, evaluer, tester,
                    rattacher_ventes, evaluer_ventes, tester_ventes, mediane_ponderee,
                    panier_fixe, evaluer_panier)


# -----------------------------------------------------------------------------
# Données partagées par les tests du fichier (lues une seule fois)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def brute():
    return charger_table(masquer=False)


@pytest.fixture(scope="module")
def table():
    return charger_table()


@pytest.fixture(scope="module")
def f(table):
    return fenetres(table)


def prix_n1_seul(app, ev):
    """Prévision du modèle « prix N-1 seul », un modèle par type de bien."""
    prev = pd.Series(np.nan, index=ev.index)
    for type_bien, e in app.groupby("type_bien"):
        m = LinearRegression().fit(e[["prix_n1"]], e[CIBLE])
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
    assert prev.notna().all()
    return prev


# =============================================================================
# P1 ET P5
# =============================================================================
def test_masque_lignes_communes(brute, table):
    retirees = brute[~masque_lignes_communes(brute)]
    assert len(table) + len(retirees) == len(brute)
    variables = [c for c in table.columns if c not in commun.HORS_MASQUE]
    assert table[variables].notna().all().all(), "il reste une case vide dans une variable explicative"


def test_fenetres(f):
    assert list(f) == ANNEES_JUGEES
    for a, (app, ev) in f.items():
        annees_app = sorted(app["annee"].unique())
        assert annees_app == list(range(commun.ANNEE_DEBUT, a)), "l'apprentissage doit être exactement ANNEE_DEBUT..a-1"
        assert (ev["annee"] == a).all() and app["annee"].max() < a
    assert sorted(f[2023][0]["annee"].unique()) == [2020, 2021, 2022]      # critère de la feuille de route


def test_plis_temporels(f):
    for a, (app, ev) in f.items():
        for tr, va in plis_temporels(app):
            a_tr, a_va = app["annee"].to_numpy()[tr], app["annee"].to_numpy()[va]
            assert len(np.unique(a_va)) == 1, "un pli = une année de validation"
            assert a_tr.max() < a_va.min(), "l'apprentissage du pli doit précéder la validation"
            assert not set(tr) & set(va), "aucune ligne dans les deux ensembles"
    assert len(plis_temporels(f[2023][0])) == 2 and len(plis_temporels(f[2024][0])) == 3   # décision 1.6


def test_compatibilite_gridsearch(f):
    app, ev = f[2024]
    for type_bien, e in app.groupby("type_bien"):
        # `e` garde l'ordre de `app` : les plis sont recalculés sur ce sous-ensemble, jamais réutilisés d'un autre
        gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=plis_temporels(e),
                          scoring="neg_mean_absolute_error")
        gs.fit(e[["prix_n1"]], e[CIBLE])
        assert gs.best_params_["alpha"] in [0.1, 1.0, 10.0]


def test_tableau_commun_prix_n1_seul(f):
    """Modèle « prix N-1 seul », 2 années x 2 types de bien."""
    resultats, tests = [], []
    for a, (app, ev) in f.items():
        prev = prix_n1_seul(app, ev)
        evaluer("P1 -- prix N-1 seul", ev, prev, resultats)
        tester("P1 -- prix N-1 seul", ev, prev, tests)
    res = pd.DataFrame(resultats)
    assert len(res) == 4 and "année jugée" in res.columns and len(tests) == 4
    assert sorted(res["année jugée"]) == [2023, 2023, 2024, 2024]
    assert "année jugée" in tests[0]


def test_non_regression_decouper(table):
    """P5 : comportement d'avant conservé, et le début de l'apprentissage est un paramètre."""
    app0, ev0 = decouper(table)                                   # valeurs par défaut
    app1, ev1 = decouper(table, annee_eval=2024, annee_debut=2020)
    assert app0.equals(app1) and ev0.equals(ev1)
    assert app0["annee"].min() == 2020 and ev0["annee"].unique().tolist() == [2024]
    app2, _ = decouper(table, annee_eval=2024, annee_debut=2022)
    assert app2["annee"].min() == 2022


# =============================================================================
# P2 : BOOTSTRAP PAR BASSIN DE VIE
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


def test_bassin_de_vie_de_chaque_sous_bassin(table):
    assert "BV2022" in table.columns, "Colonne BV2022 absente : relancer le feature engineering"
    assert "BV2022" in commun.HORS_MASQUE, "BV2022 doit être dans HORS_MASQUE (identifiant)"
    sbv = table.index.get_level_values("sous_bassin_de_vie")
    assert table.groupby(sbv)["BV2022"].nunique().eq(1).all(), "un sous-bassin a plusieurs bassins de vie"


def test_bootstrap_par_bassin_de_vie(f):
    """Même écart moyen que l'ancien bootstrap, intervalle pas nettement plus étroit."""
    for a, (app, ev) in f.items():
        prev = prix_n1_seul(app, ev)
        ancien = tester_ancien(ev, prev)
        nouveau = []
        tester("P2", ev, prev, nouveau)
        for t in nouveau:
            moy_ancien, larg_ancien = ancien[t["type de bien"]]
            larg_nouveau = t["IC 95 % haut (€/m²)"] - t["IC 95 % bas (€/m²)"]
            assert np.isclose(t["écart de MAE avec la naïve (€/m²)"], moy_ancien), "l'écart moyen ne doit pas changer"
            assert larg_nouveau / larg_ancien > 0.8, "intervalle par bassin nettement plus étroit : à vérifier"


# =============================================================================
# P11 : ÉVALUATION DU MODÈLE PAR VENTE
# =============================================================================
# Ventes inventées : 5 par ligne de la table, prix et poids au hasard, plus 3 ventes
# dans un sous-bassin absent de la table. On vérifie le rattachement et les calculs.
@pytest.fixture(scope="module")
def ventes_inventees(table):
    rng = np.random.default_rng(0)
    cles = table.index.to_frame(index=False)
    brutes = cles.loc[cles.index.repeat(5)].reset_index(drop=True).rename(columns={"annee": "anneemut"})
    brutes["prix_m2"] = rng.uniform(1000, 5000, len(brutes))
    brutes["w_h"] = rng.uniform(1, 20, len(brutes))
    brutes["idmutation"] = range(len(brutes))
    extra = brutes.iloc[:3].copy()
    extra["sous_bassin_de_vie"] = "inconnu"
    return pd.concat([brutes, extra], ignore_index=True)


@pytest.fixture(scope="module")
def ventes(ventes_inventees, table):
    return rattacher_ventes(ventes_inventees.rename(columns={"anneemut": "annee"}), table)


def test_rattachement_des_ventes(ventes_inventees, ventes):
    assert len(ventes) == len(ventes_inventees) - 3, "les ventes hors table doivent être retirées, les autres gardées"
    assert CIBLE not in ventes and "n_ventes" not in ventes, "prix médian et ventes de N : inconnus au moment de prévoir"
    assert "prix_n1" in ventes and "BV2022" in ventes


def test_evaluation_B_biens_vendus(ventes):
    ev = ventes[ventes["annee"] == ANNEES_JUGEES[-1]]
    res = []
    evaluer_ventes("naïve", ev, ev["prix_n1"], res)
    assert all(abs(r["gain sur la naïve (%)"]) < 1e-9 for r in res), "prévoir la naïve doit donner un gain nul"

    p = ev["prix_m2"] * np.random.default_rng(1).uniform(0.9, 1.1, len(ev))
    res, tst = [], []
    evaluer_ventes("bruitée", ev, p, res)
    tester_ventes("bruitée", ev, p, tst)
    for type_bien, e in ev.groupby("type_bien"):
        sbv, w = e["sous_bassin_de_vie"], e["w_h"]
        m = ((e["prix_m2"] - p.loc[e.index]).abs() * w).groupby(sbv).sum() / w.groupby(sbv).sum()
        n = ((e["prix_m2"] - e["prix_n1"]).abs() * w).groupby(sbv).sum() / w.groupby(sbv).sum()
        r = next(x for x in tst if x["type de bien"] == type_bien)
        assert np.isclose(r["écart de MAE avec la naïve (€/m²)"], (m - n).mean()), "écart testé faux"
        assert np.isclose(next(x for x in res if x["type de bien"] == type_bien)["MAE (€/m²)"], m.mean())


def test_evaluation_C_panier_fixe(ventes_inventees, table):
    assert mediane_ponderee([1, 2, 3], [1, 1, 1]) == 2 and mediane_ponderee([1, 2, 3], [1, 1, 10]) == 3
    panier = panier_fixe(ventes_inventees, table, ANNEES_JUGEES[-1])
    assert len(panier) > 0 and (panier["annee"] == ANNEES_JUGEES[-1]).all()
    vrai = table[CIBLE].reindex(pd.MultiIndex.from_frame(panier[table.index.names])).to_numpy()
    res, tst = [], []
    evaluer_panier("parfait", panier, pd.Series(vrai, index=panier.index), table, res, tst)
    assert all(r["MAE (€/m²)"] < 1e-9 for r in res), "prévoir le prix réel doit donner une erreur nulle"


def test_recopier_n1_redonne_la_naive(table):
    """Critère de la feuille de route (C) : ventes au prix médian de leur territoire et de leur année,
    la médiane pondérée des ventes de N-1 est alors exactement le prix N-1 de la table."""
    rng = np.random.default_rng(2)
    copies = table.index.to_frame(index=False).rename(columns={"annee": "anneemut"})
    copies["prix_m2"] = table[CIBLE].to_numpy()
    copies["w_h"] = rng.uniform(1, 20, len(copies))
    copies = copies.loc[copies.index.repeat(3)].reset_index(drop=True)
    copies["idmutation"] = range(len(copies))
    panier = panier_fixe(copies, table, ANNEES_JUGEES[-1])
    res, tst = [], []
    evaluer_panier("recopie de N-1", panier, panier["prix_m2"], table, res, tst)
    for r in res:
        assert np.isclose(r["MAE (€/m²)"], r["MAE naïve (€/m²)"]), "recopier les prix de N-1 doit redonner la naïve"
        assert abs(r["gain sur la naïve (%)"]) < 1e-9
