# =============================================================================
# tests/test_gabarit_keras.py : contrôles de socle/gabarit_keras.py (P12)
# =============================================================================
# Lancement depuis la racine du projet :  pytest tests/test_gabarit_keras.py
# Sans TensorFlow / Keras installé, ces tests sont signalés « skipped » (sautés), pas en échec.
# Ventes inventées (jamais les ventes réelles, règle de la feuille de route) : 40 sous-bassins
# avec chacun son niveau de prix, un marché qui monte puis baisse, un effet surface et un effet
# maison. La règle est log-linéaire et connue : un réseau qui fonctionne doit battre la naïve.
# Ne lit ni n'écrit aucun fichier.                                                 -- Cécile

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

keras = pytest.importorskip("keras")
from gabarit_keras import ReseauKeras, prevoir_keras  # noqa: E402  (après la vérification de Keras)

VARS = ["log_prix_n1", "log_sbati", "maison", "sous_bassin_de_vie", "annee"]
RAPIDE = dict(couches=(32, 16), epochs_max=60, patience=8, batch_size=128)


# -----------------------------------------------------------------------------
# Données inventées et réseaux partagés par les tests du fichier
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def donnees():
    rng = np.random.default_rng(12)
    niveau = pd.Series(rng.lognormal(np.log(2500), 0.5, 40), index=[f"sbv_{i}" for i in range(40)])
    tendance = {2016: 1.00, 2017: 1.02, 2018: 1.08, 2019: 1.15, 2020: 1.20, 2021: 1.17}
    n = 8_000
    inv = pd.DataFrame({"annee": rng.choice([2017, 2018, 2019, 2020, 2021], n),
                        "sous_bassin_de_vie": rng.choice(niveau.index, n),
                        "maison": rng.integers(0, 2, n), "sbati": rng.uniform(20, 150, n).round(),
                        "w_h": rng.uniform(1, 10, n)})
    lieu = inv["sous_bassin_de_vie"].map(niveau)
    inv["prix_n1"] = lieu * inv["annee"].sub(1).map(tendance)
    inv["prix_m2"] = (lieu * inv["annee"].map(tendance) * (1 - 0.002 * (inv["sbati"] - 60))
                      * np.where(inv["maison"] == 1, 0.9, 1.0) * rng.lognormal(0, 0.15, n))
    inv["log_sbati"] = np.log(inv["sbati"])
    inv["log_prix_n1"] = np.log(inv["prix_n1"])
    return inv[inv["annee"] < 2021], inv[inv["annee"] == 2021]       # apprentissage, année jugée


@pytest.fixture(scope="module")
def reseau(donnees):
    """Réseau sans réapprentissage : early stopping sur la dernière année d'apprentissage."""
    app, _ = donnees
    return ReseauKeras(reapprendre=False, **RAPIDE).fit(app[VARS], app["prix_m2"], sample_weight=app["w_h"])


@pytest.fixture(scope="module")
def reseau_reappris(donnees):
    """Réseau réappris sur toute la fenêtre avec le nombre d'époques trouvé."""
    app, _ = donnees
    return ReseauKeras(reapprendre=True, **RAPIDE).fit(app[VARS], app["prix_m2"], sample_weight=app["w_h"])


# =============================================================================
# TESTS
# =============================================================================
def test_decoupage_interne_et_transformations_sans_fuite(donnees, reseau):
    app, _ = donnees
    m = reseau
    ent = app[app["annee"] < 2020]
    assert m.annee_arret_ == 2020, "l'early stopping doit utiliser la dernière année d'apprentissage"
    assert "annee" not in m.variables_ and "sous_bassin_de_vie" not in m.variables_, "année et sous-bassin : pas des variables numériques"
    assert np.isclose(m.moy_y_, np.log(ent["prix_m2"]).mean()), "cible standardisée sur les seules années d'entraînement"
    assert np.allclose(m.moy_X_["log_sbati"], ent["log_sbati"].mean())
    assert set(m.numeros_) == set(ent["sous_bassin_de_vie"]) and min(m.numeros_.values()) == 1, "numéro 0 réservé à l'inconnu"
    assert 1 <= m.meilleure_epoque_ <= len(m.historique_["perte_arret"]) <= RAPIDE["epochs_max"]


def test_previsions_en_euros_index_conserve_sous_bassin_inconnu(donnees, reseau):
    _, ev = donnees
    m = reseau
    p = m.predict(ev[VARS])
    assert isinstance(p, pd.Series) and p.index.equals(ev.index) and np.isfinite(p).all() and (p > 0).all()
    inconnu = ev[VARS].head(5).assign(sous_bassin_de_vie="jamais_vu")
    assert np.isfinite(m.predict(inconnu)).all(), "un sous-bassin inconnu doit recevoir le numéro 0, sans erreur"
    # Perte à l'échelle naturelle : poids normalisés (sinon multipliée par le poids moyen, ~5,5 ici)
    assert m.historique_["perte_arret"][m.meilleure_epoque_ - 1] < 1.0, "perte d'arrêt trop haute : poids non normalisés ?"


def test_reapprendre_toute_la_fenetre(donnees, reseau, reseau_reappris):
    app, _ = donnees
    assert np.isclose(reseau_reappris.moy_y_, np.log(app["prix_m2"]).mean()), "après réapprentissage : transformations sur toute la fenêtre"
    assert reseau_reappris.meilleure_epoque_ == reseau.meilleure_epoque_, "même graine : même nombre d'époques trouvé"


def test_reproductibilite(donnees, reseau_reappris):
    app, ev = donnees
    m3 = ReseauKeras(reapprendre=True, **RAPIDE).fit(app[VARS], app["prix_m2"], sample_weight=app["w_h"])
    ecart = (reseau_reappris.predict(ev[VARS]) - m3.predict(ev[VARS])).abs().max()
    assert ecart < 1.0, "deux entraînements avec la même graine doivent donner les mêmes prévisions (à 1 €/m² près)"


@pytest.mark.parametrize("cas", ["une seule année", "année absente", "case vide"])
def test_garde_fous(donnees, cas):
    app, _ = donnees
    X_mauvais = {"une seule année": app[app["annee"] == 2019][VARS],
                 "année absente": app[VARS].drop(columns="annee"),
                 "case vide": app[VARS].assign(log_sbati=np.nan)}[cas]
    with pytest.raises(ValueError):
        ReseauKeras(**RAPIDE).fit(X_mauvais, app.loc[X_mauvais.index, "prix_m2"])


def test_compatible_scikit_learn(reseau):
    assert clone(reseau).get_params()["couches"] == RAPIDE["couches"], "compatible scikit-learn (clone, get_params)"


def test_le_reseau_bat_la_naive(donnees):
    """Sur l'année jugée, sans et avec embedding."""
    app, ev = donnees
    mae = lambda prev: np.average((ev["prix_m2"] - prev).abs(), weights=ev["w_h"])  # noqa: E731
    p_emb, _ = prevoir_keras(app, ev, VARS, "prix_m2", poids="w_h", **RAPIDE)
    p_sans, _ = prevoir_keras(app, ev, ["log_prix_n1", "log_sbati", "maison"], "prix_m2", poids="w_h",
                              colonne_embedding=None, **RAPIDE)
    assert mae(p_emb) < mae(ev["prix_n1"]) and mae(p_sans) < mae(ev["prix_n1"]), "le réseau doit battre la naïve sur ces données"
