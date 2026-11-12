# =============================================================================
# H5H6_pont_capacite_volume.py : pont + taux d'effort + volume et élan (2023, 2024)
# =============================================================================
# Pourquoi : avant d'ouvrir le test 2025, vérifier si ajouter le bloc volume et élan (H5)
# au pont + taux d'effort (H6) améliore la prévision. Si oui, ce modèle devient un candidat
# de plus pour le test 2025 ; sinon, il reste documenté comme essai non retenu.
#
# Ce que fait ce fichier :
#   1. charge la table commune, y joint P9 (volume et élan) et P7 (taux d'effort) ;
#   2. garde les mêmes lignes pour tous les modèles (indicateurs renseignés) ;
#   3. lance trois modèles avec gabarit_pont (Elastic Net, apprentissage dès 2017) :
#        - pont ;
#        - pont + taux d'effort (= H6, modèle 2.1) ;
#        - pont + taux d'effort + volume et élan (le nouveau modèle) ;
#   4. mesure chaque modèle contre la naïve, puis le nouveau contre le pont + taux d'effort ;
#   5. applique la règle de décision (gain >= 3 %, IC haut < 0, rang-bisériale < 0,
#      sur la majorité des années jugées) ;
#   6. contrôle si le réglage (alpha) tombe au bord de la grille ;
#   7. enregistre les résultats dans MLflow et dans un CSV.
# Le test 2025 n'est PAS ouvert ici.
#
# Lancement (depuis le dossier qui contient commun.py et gabarit_pont.py) :
#     python H5H6_pont_capacite_volume.py

import inspect
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import commun
from commun import CIBLE, charger_table, fenetres, evaluer, tester
from gabarit_pont import ANNEE_DEBUT_PONT, GRILLE_EN, previsions_pont

warnings.filterwarnings("ignore")      # convergence de l'Elastic Net sur les plus petits plis
pd.set_option("display.width", 220)

# -----------------------------------------------------------------------------
# Paramètres
# -----------------------------------------------------------------------------
HYPOTHESE = "H5-H6"
EXPERIENCE_MLFLOW = "Compagnon immobilier v2"
SEUIL_GAIN = 3                                       # % (règle de décision de l'équipe)

TAUX_EFFORT = ["taux_effort"]                        # bloc H6
BLOC_H5 = ["nb_ventes_n1", "rotation_parc_type", "evol_prix_n2_n1"]   # bloc H5 retenu (rapport 7.1)
LOG_EN_PLUS = ["taux_effort", "nb_ventes_n1"]        # variables asymétriques passées en log (comme H5 et H6)
COLONNES_P9 = ["rotation_parc", "rotation_parc_type", "evol_ventes_n2_n1", "evol_prix_n2_n1"]

NOM_PONT = f"{HYPOTHESE} – Pont"
NOM_CAPACITE = f"{HYPOTHESE} – Pont + taux d'effort"
NOM_NOUVEAU = f"{HYPOTHESE} – Pont + taux d'effort + volume et élan"


# -----------------------------------------------------------------------------
# 1. Table : table commune + P9 (volume et élan) + P7 (taux d'effort)
# -----------------------------------------------------------------------------
table_commune = charger_table()

# Clés de jointure lues dans la table, sans dépendre de la forme de son index (comme H5)
cles = pd.DataFrame({"annee": table_commune["annee"].to_numpy(),
                     "sous_bassin_de_vie": table_commune.index.get_level_values("sous_bassin_de_vie"),
                     "type_bien": table_commune.index.get_level_values("type_bien")})

p9 = pd.read_parquet(commun.DATA_OUT / "p9_volume_elan.parquet")
p7 = pd.read_parquet(commun.DATA_OUT / "fe_capacite_achat_sbv.parquet")[
    ["annee", "sous_bassin_de_vie", "type_bien", "taux_effort"]]

joint = (cles.merge(p9, on=["annee", "sous_bassin_de_vie", "type_bien"], how="left", validate="one_to_one")
             .merge(p7, on=["annee", "sous_bassin_de_vie", "type_bien"], how="left", validate="one_to_one"))
assert len(joint) == len(table_commune), "la jointure a changé le nombre de lignes"

table_jointe = table_commune.copy()
for c in COLONNES_P9 + TAUX_EFFORT:
    table_jointe[c] = joint[c].to_numpy()

# Mêmes lignes pour les trois modèles : indicateurs P9 et taux d'effort renseignés
complet = table_jointe[COLONNES_P9 + TAUX_EFFORT].notna().all(axis=1)
table = table_jointe[complet]
periode = table[table["annee"] >= ANNEE_DEBUT_PONT]
assert (periode["taux_effort"] > 0).all() and (periode["nb_ventes_n1"] > 0).all(), "log impossible"

print("Table commune :", table_commune.shape, "| lignes gardées :", table.shape,
      "| retirées (indicateur vide) :", int((~complet).sum()))
FENETRES = fenetres(table, annee_debut=ANNEE_DEBUT_PONT)
for a, (app, ev) in FENETRES.items():
    print(f"Fenêtre {a} : apprentissage {int(app['annee'].min())}-{int(app['annee'].max())} "
          f"({len(app)} lignes) | jugée {a} ({len(ev)} lignes)")


# -----------------------------------------------------------------------------
# 2. Les trois modèles (gabarit pont : mêmes réglages, mêmes fenêtres)
# -----------------------------------------------------------------------------
previsions, ajustes = {}, {}
for nom, bloc in [(NOM_PONT, []),
                  (NOM_CAPACITE, TAUX_EFFORT),
                  (NOM_NOUVEAU, TAUX_EFFORT + BLOC_H5)]:
    modeles = {}
    prev = previsions_pont(table, annee_debut=ANNEE_DEBUT_PONT, bloc=bloc,
                           log_en_plus=[c for c in LOG_EN_PLUS if c in bloc], modeles=modeles)
    for annee, p in prev.items():
        assert p.notna().all(), f"{nom} : prévisions manquantes en {annee}"
        previsions[(nom, annee)] = p
    for cle, m in modeles.items():
        ajustes[(nom,) + cle] = m
    print("Modèle lancé :", nom)


# -----------------------------------------------------------------------------
# 3. Mesures : contre la naïve, puis le nouveau contre ses références
# -----------------------------------------------------------------------------
def mesurer(nom, reference=None):
    """evaluer + tester de commun.py ; reference = nom d'un modèle dont la prévision remplace le prix N-1."""
    res, tst = [], []
    for annee, (app, ev) in FENETRES.items():
        ev_ref = ev.copy()
        if reference is not None:
            ev_ref["prix_n1"] = previsions[(reference, annee)]
        evaluer(nom, ev_ref, previsions[(nom, annee)], res)
        tester(nom, ev_ref, previsions[(nom, annee)], tst)
    cles_t = ["modèle", "type de bien", "année jugée"]
    tableau = pd.DataFrame(res).merge(pd.DataFrame(tst), on=cles_t, suffixes=("", " (test)"))
    tableau = tableau.rename(columns=lambda c: c.replace("naïve", "référence"))
    tableau["référence"] = "naive" if reference is None else reference.replace(f"{HYPOTHESE} – ", "").lower()
    # Biais : prévision moins prix réel, moyenne par type de bien
    biais = []
    for _, r in tableau.iterrows():
        ev = FENETRES[r["année jugée"]][1]
        idx = ev.index[ev.index.get_level_values("type_bien") == r["type de bien"]]
        biais.append((previsions[(nom, r["année jugée"])].loc[idx] - ev.loc[idx, CIBLE]).mean())
    tableau["biais (€/m²)"] = biais
    tableau["année réussie"] = ((tableau["IC 95 % haut (€/m²)"] < 0) & (tableau["rang-bisériale"] < 0)
                                & (tableau["gain sur la référence (%)"] >= SEUIL_GAIN))
    return tableau


tableaux = [mesurer(NOM_PONT), mesurer(NOM_CAPACITE), mesurer(NOM_NOUVEAU),
            mesurer(NOM_NOUVEAU, reference=NOM_CAPACITE),       # question du jour
            mesurer(NOM_NOUVEAU, reference=NOM_PONT)]
resultats = pd.concat(tableaux, ignore_index=True)

COLONNES = ["modèle", "référence", "type de bien", "année jugée", "MAE (€/m²)", "MAE référence (€/m²)",
            "gain sur la référence (%)", "IC 95 % haut (€/m²)", "rang-bisériale", "biais (€/m²)", "année réussie"]
print("\n=== Résultats (2023 et 2024) ===")
print(resultats[COLONNES].round(2).to_string(index=False))

decision = (resultats.groupby(["modèle", "référence", "type de bien"], sort=False)["année réussie"]
            .agg(["sum", "count"]).rename(columns={"sum": "années réussies", "count": "années jugées"}))
decision["retenu"] = decision["années réussies"] > decision["années jugées"] / 2
print("\n=== Règle de décision ===")
print(decision.to_string())


# -----------------------------------------------------------------------------
# 4. Contrôle du réglage : alpha au bord de la grille ?
# -----------------------------------------------------------------------------
bornes = (min(GRILLE_EN["alpha"]), max(GRILLE_EN["alpha"]))
print("\n=== Réglage retenu (alpha, l1_ratio) ===")
for (nom, annee, type_bien), m in ajustes.items():
    alpha = m.reglage_["alpha"]
    bord = "  <- au bord de la grille" if alpha in bornes else ""
    print(f"{nom} | {annee} | {type_bien} : alpha = {alpha:.1e}, l1_ratio = {m.reglage_['l1_ratio']}{bord}")

# Poids du bloc dans le nouveau modèle (coefficients standardisés)
coefs = pd.DataFrame({(a, tb): m.coefs_[["prix_n1"] + TAUX_EFFORT + BLOC_H5]
                      for (nom, a, tb), m in ajustes.items() if nom == NOM_NOUVEAU})
print("\n=== Coefficients standardisés du nouveau modèle ===")
print(coefs.round(4).to_string())


# -----------------------------------------------------------------------------
# 5. Sauvegarde : CSV et MLflow
# -----------------------------------------------------------------------------
sortie = commun.DATA_OUT / "h5h6_pont_capacite_volume_resultats.csv"
resultats.to_csv(sortie, index=False)
print("\nRésultats enregistrés :", sortie)

# Si evaluer / tester de commun.py enregistrent déjà dans MLflow, on n'enregistre pas une 2e fois
auto = "mlflow" in inspect.getsource(commun.evaluer) + inspect.getsource(commun.tester)
if auto:
    print("MLflow : commun.py enregistre déjà les résultats (pas de doublon).")
else:
    import mlflow

    mlflow.set_experiment(EXPERIENCE_MLFLOW)
    lancement = datetime.now().strftime("%Y-%m-%d %H:%M")
    for _, r in resultats.iterrows():
        modele = r["modèle"]
        nom_run = f"{HYPOTHESE} | {modele} | {r['type de bien']} | {r['année jugée']} | contre : {r['référence']}"
        with mlflow.start_run(run_name=nom_run):
            mlflow.set_tags({"hypothese": HYPOTHESE, "modele": modele, "type_bien": r["type de bien"],
                             "annee_jugee": str(r["année jugée"]), "reference": r["référence"],
                             "lancement": lancement})
            mlflow.log_metrics({"mae": r["MAE (€/m²)"], "mae_reference": r["MAE référence (€/m²)"],
                                "gain_naive_pct": r["gain sur la référence (%)"],
                                "ic_haut": r["IC 95 % haut (€/m²)"], "rang_biseriale": r["rang-bisériale"],
                                "biais": r["biais (€/m²)"]})
    print(f"MLflow : {len(resultats)} runs enregistrés dans « {EXPERIENCE_MLFLOW} » (tracking : {mlflow.get_tracking_uri()}).")
