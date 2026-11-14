# =============================================================================
# tests/test_gabarit_hierarchique.py : contrôles de socle/gabarit_hierarchique.py (P4)
# =============================================================================
# Lancement depuis la racine du projet :  pytest tests/test_gabarit_hierarchique.py
# Lit data/processed/table_modelisation_bv.parquet ; ne modifie aucun fichier.   -- Christian

import numpy as np
import pandas as pd
import pytest

from commun import CIBLE, charger_table, fenetres, evaluer, tester
from gabarit_hierarchique import ModeleHierarchique


@pytest.fixture(scope="module")
def f():
    return fenetres(charger_table())


def test_prix_n1_seul_passe_par_evaluer_et_tester(f):
    resultats, tests = [], []
    for a, (app, ev) in f.items():
        prev = pd.Series(np.nan, index=ev.index)
        for type_bien, e in app.groupby("type_bien"):
            idx = ev.index[ev.index.get_level_values("type_bien") == type_bien]
            m = ModeleHierarchique().fit(e[["prix_n1"]], e[CIBLE])
            prev.loc[idx] = m.predict(ev.loc[idx, ["prix_n1"]])
        assert prev.notna().all()
        evaluer("P4 -- hiérarchique, prix N-1 seul", ev, prev, resultats)
        tester("P4 -- hiérarchique, prix N-1 seul", ev, prev, tests)
    assert len(resultats) == 4 and len(tests) == 4


def test_territoire_inconnu_recoit_les_effets_fixes_seuls(f):
    """Un territoire absent de l'apprentissage reçoit seulement les effets fixes, sans erreur."""
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
