# =============================================================================
# tests/test_gabarit_pont.py : contrôles de socle/gabarit_pont.py (H4)
# =============================================================================
# Lancement depuis la racine du projet :  pytest tests/test_gabarit_pont.py
# Lit data/processed/table_modelisation_bv.parquet ; ne modifie aucun fichier.
# Le gabarit doit retrouver les MAE publiées par H4 (période 2017-).

import numpy as np
import pandas as pd
import pytest

from commun import charger_table, fenetres, evaluer, tester
from gabarit_pont import ModelePont, previsions_pont, ANNEE_DEBUT_PONT

# Convergence de l'Elastic Net sur les plis les plus petits : avertissements attendus
pytestmark = pytest.mark.filterwarnings("ignore")

ANNEES = [2023, 2024]
H4 = {("complet", "appartement", 2023): 158.7, ("complet", "appartement", 2024): 187.8,
      ("complet", "maison", 2023): 130.4, ("complet", "maison", 2024): 180.2,
      ("N-1 seul", "appartement", 2023): 167.5, ("N-1 seul", "appartement", 2024): 202.0,
      ("N-1 seul", "maison", 2023): 137.2, ("N-1 seul", "maison", 2024): 194.5}


@pytest.fixture(scope="module")
def table():
    return charger_table()


@pytest.fixture(scope="module")
def prev_pont(table):
    """Prévisions du pont (modèle complet), utilisées par plusieurs tests."""
    return previsions_pont(table, annees_jugees=ANNEES)


def passer_par_commun(table, nom, prev):
    resultats, tests = [], []
    for a, (app, ev) in fenetres(table, annees_jugees=ANNEES, annee_debut=ANNEE_DEBUT_PONT).items():
        assert prev[a].notna().all() and (prev[a] > 0).all(), "prévisions manquantes ou négatives"
        evaluer(nom, ev, prev[a], resultats)
        tester(nom, ev, prev[a], tests)
    return pd.DataFrame(resultats), pd.DataFrame(tests)


def verifier_h4(resultats, version):
    for _, l in resultats.iterrows():
        attendu = H4[(version, l["type de bien"], l["année jugée"])]
        assert abs(l["MAE (€/m²)"] - attendu) < 0.1, (l["type de bien"], l["année jugée"], l["MAE (€/m²)"], attendu)


def test_prix_n1_seul_retrouve_h4(table):
    r, _ = passer_par_commun(table, "pont -- prix N-1 seul",
                             previsions_pont(table, annees_jugees=ANNEES, variables=["prix_n1"]))
    verifier_h4(r, "N-1 seul")


def test_pont_complet_retrouve_h4(table, prev_pont):
    """158,7 / 187,8 pour les appartements ; 130,4 / 180,2 pour les maisons."""
    r, _ = passer_par_commun(table, "pont", prev_pont)
    verifier_h4(r, "complet")


def test_ajout_d_un_bloc(table, prev_pont):
    """Mêmes lignes, prévisions différentes, tout passe par evaluer et tester."""
    bloc = ["niveau_vie_median"]
    prev_b = previsions_pont(table, annees_jugees=ANNEES, bloc=bloc, log_en_plus=bloc)
    passer_par_commun(table, "pont + niveau de vie", prev_b)
    assert all(prev_b[a].index.equals(prev_pont[a].index) for a in ANNEES)
    assert not np.allclose(prev_b[2024], prev_pont[2024]), "le bloc doit changer les prévisions"
    m = ModelePont(bloc=bloc)
    assert m.variables[-1] == "niveau_vie_median" and len(m.variables) == len(ModelePont().variables) + 1
    assert ModelePont(bloc=["prix_n1"]).variables == ModelePont().variables, "un doublon ne doit pas être ajouté"


def test_prix_n1_obligatoire():
    with pytest.raises(AssertionError):
        ModelePont(variables=["densite_population"])
