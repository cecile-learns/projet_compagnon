# =============================================================================
# 03_modelisation/test_commun_p1.py : contrôles « c'est fini quand » de P1 (et P5)
# =============================================================================
# À lancer depuis 03_modelisation :  python test_commun_p1.py
# Chaque contrôle est un assert : le script s'arrête au premier échec.
# Il ne modifie aucun fichier.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

import commun
from commun import (ANNEES_JUGEES, charger_table, decouper, fenetres, plis_temporels,
                    masque_lignes_communes, prevision_naive, evaluer, tester)


def titre(t):
    print(f"\n--- {t}")


brute = charger_table(masquer=False)
table = charger_table()

# ---------------------------------------------------------------- masque
titre("Masque des lignes communes")
retirees = brute[~masque_lignes_communes(brute)]
print("lignes brutes :", len(brute), "| gardées :", len(table), "| retirées :", len(retirees))
print("lignes retirées par année :", retirees["annee"].value_counts().sort_index().to_dict())
assert len(table) + len(retirees) == len(brute)
variables = [c for c in table.columns if c not in commun.HORS_MASQUE]
assert table[variables].notna().all().all(), "il reste une case vide dans une variable explicative"
# le masque ne doit retirer aucune ligne 2020-2024 pour une autre raison que les variables explicatives
fenetre_niveau1 = retirees[retirees["annee"].between(commun.ANNEE_DEBUT, ANNEES_JUGEES[-1])]
print("dont sur 2020-2024 :", len(fenetre_niveau1))

# ---------------------------------------------------------------- fenêtres
titre("Fenêtres")
f = fenetres(table)
assert list(f) == ANNEES_JUGEES
for a, (app, ev) in f.items():
    annees_app = sorted(app["annee"].unique())
    print(f"fenêtre {a} : apprentissage {annees_app} ({len(app)} lignes) | jugée {sorted(ev['annee'].unique())} ({len(ev)} lignes)")
    assert annees_app == list(range(commun.ANNEE_DEBUT, a)), "l'apprentissage doit être exactement ANNEE_DEBUT..a-1"
    assert (ev["annee"] == a).all() and app["annee"].max() < a
assert sorted(f[2023][0]["annee"].unique()) == [2020, 2021, 2022]      # critère de la feuille de route

# ---------------------------------------------------------------- plis
titre("Plis temporels")
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
titre("Compatibilité GridSearchCV")
app, ev = f[2024]
for type_bien, e in app.groupby("type_bien"):
    # `e` garde l'ordre de `app` : les plis sont recalculés sur ce sous-ensemble, jamais réutilisés d'un autre
    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=plis_temporels(e),
                      scoring="neg_mean_absolute_error")
    gs.fit(e[["prix_n1"]], e["prix"])
    print(type_bien, "meilleur alpha :", gs.best_params_)

# ---------------------------------------------------------------- tableau commun « prix N-1 seul »
titre("Tableau commun : modèle « prix N-1 seul », 2 années x 2 types de bien")
resultats, tests = [], []
for a, (app, ev) in f.items():
    prev = pd.Series(np.nan, index=ev.index)
    for type_bien, e in app.groupby("type_bien"):           # un modèle par type de bien : PROVISOIRE, voir note
        m = LinearRegression().fit(e[["prix_n1"]], e["prix"])
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
    assert prev.notna().all()
    evaluer("P1 -- prix N-1 seul", ev, prev, resultats)
    tester("P1 -- prix N-1 seul", ev, prev, tests)
res = pd.DataFrame(resultats)
print(res.round(2).to_string())
assert len(res) == 4 and "année jugée" in res.columns and len(tests) == 4
assert sorted(res["année jugée"]) == [2023, 2023, 2024, 2024]
assert "année jugée" in tests[0]

# ---------------------------------------------------------------- non-régression
titre("Non-régression : comportement d'avant")
app0, ev0 = decouper(table)                                   # valeurs par défaut
app1, ev1 = decouper(table, annee_eval=2024, annee_debut=2020)
assert app0.equals(app1) and ev0.equals(ev1)
assert app0["annee"].min() == 2020 and ev0["annee"].unique().tolist() == [2024]
app2, _ = decouper(table, annee_eval=2024, annee_debut=2022)  # P5 : le début est bien un paramètre
assert app2["annee"].min() == 2022
print("OK")
print("\nTous les contrôles P1/P5 passent.")
