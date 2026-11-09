# 03_modelisation/test_gabarit_pont.py : « c'est fini quand » du gabarit pont (H4)
# À lancer depuis 03_modelisation :  python test_gabarit_pont.py
import warnings
import numpy as np
import pandas as pd

from commun import charger_table, fenetres, evaluer, tester
from gabarit_pont import ModelePont, previsions_pont, ANNEE_DEBUT_PONT

warnings.filterwarnings("ignore")      # convergence de l'Elastic Net sur les plis les plus petits
table = charger_table()
ANNEES = [2023, 2024]

# MAE publiées par H4 (période 2017-) : le gabarit doit les retrouver
H4 = {("complet", "appartement", 2023): 158.7, ("complet", "appartement", 2024): 187.8,
      ("complet", "maison", 2023): 130.4, ("complet", "maison", 2024): 180.2,
      ("N-1 seul", "appartement", 2023): 167.5, ("N-1 seul", "appartement", 2024): 202.0,
      ("N-1 seul", "maison", 2023): 137.2, ("N-1 seul", "maison", 2024): 194.5}


def passer_par_commun(nom, prev):
    resultats, tests = [], []
    for a, (app, ev) in fenetres(table, annees_jugees=ANNEES, annee_debut=ANNEE_DEBUT_PONT).items():
        assert prev[a].notna().all() and (prev[a] > 0).all(), "prévisions manquantes ou négatives"
        evaluer(nom, ev, prev[a], resultats)
        tester(nom, ev, prev[a], tests)
    return pd.DataFrame(resultats), pd.DataFrame(tests)


# 1. « prix N-1 seul » passe par evaluer et tester, et retrouve H4
r1, t1 = passer_par_commun("pont -- prix N-1 seul", previsions_pont(table, annees_jugees=ANNEES, variables=["prix_n1"]))
for _, l in r1.iterrows():
    attendu = H4[("N-1 seul", l["type de bien"], l["année jugée"])]
    assert abs(l["MAE (€/m²)"] - attendu) < 0.1, (l["type de bien"], l["année jugée"], l["MAE (€/m²)"], attendu)
print("1. prix N-1 seul : evaluer et tester passent, MAE identiques à H4")

# 2. Le pont (modèle complet) retrouve H4
ajustes = {}
r2, t2 = passer_par_commun("pont", previsions_pont(table, annees_jugees=ANNEES, modeles=ajustes))
for _, l in r2.iterrows():
    attendu = H4[("complet", l["type de bien"], l["année jugée"])]
    assert abs(l["MAE (€/m²)"] - attendu) < 0.1, (l["type de bien"], l["année jugée"], l["MAE (€/m²)"], attendu)
print("2. pont : MAE identiques à H4 (158,7 / 187,8 appartements ; 130,4 / 180,2 maisons)")
print(r2.round(1).to_string())
print("   coefficient du prix N-1 :", {k: round(m.coef_prix_n1_, 3) for k, m in ajustes.items()})
print("   réglage :", {k: m.reglage_ for k, m in ajustes.items()})

# 3. Ajouter un bloc : mêmes lignes, prévisions différentes, tout passe par evaluer et tester
bloc = ["niveau_vie_median"]
prev_b = previsions_pont(table, annees_jugees=ANNEES, bloc=bloc, log_en_plus=bloc)
r3, t3 = passer_par_commun("pont + niveau de vie", prev_b)
prev_p = previsions_pont(table, annees_jugees=ANNEES)
assert all(prev_b[a].index.equals(prev_p[a].index) for a in ANNEES)
assert not np.allclose(prev_b[2024], prev_p[2024]), "le bloc doit changer les prévisions"
m = ModelePont(bloc=bloc)
assert m.variables[-1] == "niveau_vie_median" and len(m.variables) == len(ModelePont().variables) + 1
assert ModelePont(bloc=["prix_n1"]).variables == ModelePont().variables, "un doublon ne doit pas être ajouté"
print("3. bloc ajouté : même index, prévisions modifiées, evaluer et tester passent")
print(r3[["type de bien", "année jugée", "MAE (€/m²)", "gain sur la naïve (%)"]].round(1).to_string())

# 4. Le prix N-1 est obligatoire
try:
    ModelePont(variables=["densite_population"])
    raise SystemExit("le gabarit aurait dû refuser un modèle sans prix N-1")
except AssertionError:
    print("4. modèle sans prix N-1 refusé")
print("\nTous les contrôles du gabarit pont passent.")
