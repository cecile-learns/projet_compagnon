# =============================================================================
# tests/test_gabarit_boosting.py : contrôles de socle/gabarit_boosting.py (P3)
# =============================================================================
# Lancement depuis la racine du projet :  pytest tests/test_gabarit_boosting.py
# Lit data/processed/table_modelisation_bv.parquet ; ne modifie aucun fichier.
# Remplace la vérification qui se trouvait en fin de gabarit_boosting.py (« prix N-1 seul »
# sur les deux fenêtres, cible niveau puis évolution).
#
# Valeurs de référence : MAE obtenues le 14/11/2026 dans l'environnement de requirements.txt
# (Python 3.12, scikit-learn 1.9.1). Le gradient boosting découpe les variables en intervalles,
# et ce découpage varie légèrement d'une version de scikit-learn à l'autre (jusqu'à 1 €/m² observé
# avec scikit-learn 1.7.2). Si ce test échoue après une mise à jour de scikit-learn, vérifier
# l'écart, puis mettre à jour REFERENCE.

import pytest

from commun import charger_table, fenetres, evaluer, tester
from gabarit_boosting import GRILLE_REFERENCE, prevoir_boosting

REFERENCE = {("niveau", "appartement", 2023): 197.17, ("niveau", "maison", 2023): 196.75,
             ("evolution", "appartement", 2023): 182.58, ("evolution", "maison", 2023): 171.45,
             ("niveau", "appartement", 2024): 239.94, ("niveau", "maison", 2024): 236.42,
             ("evolution", "appartement", 2024): 222.78, ("evolution", "maison", 2024): 216.66}
TOLERANCE = 0.1    # €/m², comme pour le gabarit pont (arrondi des valeurs de référence)
REGLAGE_RETENU = {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 50}


@pytest.fixture(scope="module")
def f():
    return fenetres(charger_table())


@pytest.mark.parametrize("cible", ["niveau", "evolution"])
def test_prix_n1_seul_retrouve_la_reference(f, cible):
    resultats, tests = [], []
    for annee, (app, ev) in f.items():
        prevision, modeles = prevoir_boosting(app, ev, ["prix_n1"], cible=cible)
        assert prevision.notna().all() and (prevision > 0).all()
        evaluer(cible, ev, prevision, resultats)
        tester(cible, ev, prevision, tests)
        for type_bien, m in modeles.items():
            assert set(m.best_params_) == set(GRILLE_REFERENCE)
            if cible == "evolution":
                assert m.best_params_ == REGLAGE_RETENU, (annee, type_bien, m.best_params_)
    assert len(tests) == 4
    for r in resultats:
        attendu = REFERENCE[(cible, r["type de bien"], r["année jugée"])]
        assert abs(r["MAE (€/m²)"] - attendu) < TOLERANCE, (r["type de bien"], r["année jugée"], r["MAE (€/m²)"], attendu)


def test_evolution_exige_prix_n1(f):
    app, ev = f[2023]
    with pytest.raises(AssertionError):
        prevoir_boosting(app, ev, ["densite_population"], cible="evolution", grille={})


def test_sans_grille(f):
    """grille={} : pas de réglage, un modèle ajusté par type de bien, prévisions complètes."""
    app, ev = f[2023]
    prevision, modeles = prevoir_boosting(app, ev, ["prix_n1"], grille={})
    assert prevision.notna().all() and set(modeles) == {"appartement", "maison"}
    assert not hasattr(modeles["maison"], "best_params_")
