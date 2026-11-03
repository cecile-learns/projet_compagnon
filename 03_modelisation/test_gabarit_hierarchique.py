# 03_modelisation/test_gabarit_hierarchique.py : « c'est fini quand » de P4
# À lancer depuis 03_modelisation :  python test_gabarit_hierarchique.py
import numpy as np
import pandas as pd

from commun import charger_table, fenetres, evaluer, tester
from gabarit_hierarchique import ModeleHierarchique

table = charger_table()
resultats, tests = [], []
for a, (app, ev) in fenetres(table).items():
    prev = pd.Series(np.nan, index=ev.index)
    for type_bien, e in app.groupby("type_bien"):
        idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
        m = ModeleHierarchique().fit(e[["prix_n1"]], e["prix"])
        prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
        print(f"{a} {type_bien:11s} convergé={m.converge_} | groupes vus={len(m.effets_aleatoires_)} "
              f"| lignes jugées à groupe inconnu={m.part_groupes_inconnus(ev.loc[idx, ['prix_n1']]):.1%}"
              f" | avertissements={len(m.avertissements_)} | ICC={m.icc_:.4f}")
    assert prev.notna().all()
    evaluer(f"P4 -- hiérarchique, prix N-1 seul", ev, prev, resultats)
    tester(f"P4 -- hiérarchique, prix N-1 seul", ev, prev, tests)
print(pd.DataFrame(resultats).round(2).to_string())

# Critère : un territoire absent de l'apprentissage reçoit seulement les effets fixes, sans erreur
app, ev = fenetres(table)[2024]
e = app[app.index.get_level_values("type_bien") == "maison"]
sbv_retire = e.index.get_level_values("sous_bassin_de_vie")[0]
e_sans = e[e.index.get_level_values("sous_bassin_de_vie") != sbv_retire]
m = ModeleHierarchique().fit(e_sans[["prix_n1"]], e_sans["prix"])
x_nouveau = e[e.index.get_level_values("sous_bassin_de_vie") == sbv_retire][["prix_n1"]]
p = m.predict(x_nouveau)
assert sbv_retire not in m.effets_aleatoires_ and p.notna().all()
beta = np.asarray(m.resultat_.fe_params)
attendu = beta[0] + beta[1] * (x_nouveau["prix_n1"] - m.moyennes_["prix_n1"]) / m.ecarts_["prix_n1"]
assert np.allclose(p, attendu), "pour un territoire inconnu la prévision doit être les effets fixes seuls"
print(f"\nTerritoire nouveau ({sbv_retire}) : prévision = effets fixes seuls, OK")
print("Tous les contrôles P4 passent.")
