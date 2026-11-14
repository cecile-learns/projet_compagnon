# =============================================================================
# P6_table_periode_longue.py : vérification de la période longue (prix 2017 à 2025)
# =============================================================================
# Contributeur : Théophyle · Date : 06/11/2026
#
# Pourquoi : le tour 2 apprend sur plusieurs phases du marché (stabilité 2017-2018,
#   hausse 2019-2022, recul 2023) au lieu de la seule hausse 2020-2022.
#   H4, H5, H6 et H7 en dépendent.
#
# Constat : table_modelisation_bv.parquet couvre déjà les prix 2015 à 2025, avec le même
#   seuil de 20 ventes et les mêmes colonnes. La délinquance ne commence qu'avec les prix
#   2017 : le masque commun de charger_table() retire 2015 et 2016 et garde 2017-2025.
#   La période est un paramètre de commun.py : fenetres(table, annee_debut=2017) (P5).
#   Il n'y a donc pas de nouvelle table : ce script vérifie les conditions de la feuille
#   de route et ne produit aucun fichier.
#
# Lancement, depuis la racine du projet :  python 03_modelisation/2_prepa_tour2/P6_table_periode_longue.py
# (le socle doit être installé : pip install -e . depuis la racine, une seule fois)

import commun
from commun import charger_table, decouper, fenetres

ANNEE_DEBUT_LONGUE = 2017            # premiers prix avec la délinquance (feuille de route, P6)

brute = charger_table(masquer=False)  # toutes les lignes, avant le masque commun
table = charger_table()               # lignes sans case vide : celles des modèles

# Le masque commun retire 2015 et 2016 et garde 2017 à 2025
assert table["annee"].min() == ANNEE_DEBUT_LONGUE, "Le masque commun ne commence pas en 2017"
assert table["annee"].max() == 2025, "Le masque commun ne va pas jusqu'à 2025"

# Fenêtres de la période longue : mêmes années jugées (2023 et 2024), apprentissage dès 2017
for annee, (app, ev) in fenetres(table, annee_debut=ANNEE_DEBUT_LONGUE).items():
    annees_app = sorted(int(a) for a in app["annee"].unique())
    assert annees_app[0] == ANNEE_DEBUT_LONGUE and annees_app[-1] == annee - 1, \
        "Fenêtre " + str(annee) + " : apprentissage inattendu " + str(annees_app)
    print("Fenêtre", annee, ": apprentissage", annees_app[0], "à", annees_app[-1],
          "(", len(app), "lignes ) | jugée", annee, "(", len(ev), "lignes )")

# L'apprentissage long, limité aux années 2020 et suivantes, doit être exactement l'apprentissage actuel
for annee in commun.ANNEES_JUGEES:
    app_actuel, _ = decouper(table, annee_eval=annee)
    app_long, _ = decouper(table, annee_eval=annee, annee_debut=ANNEE_DEBUT_LONGUE)
    assert app_long[app_long["annee"] >= commun.ANNEE_DEBUT].equals(app_actuel), \
        "Fenêtre " + str(annee) + " : la période longue diffère de la table actuelle sur les années communes"

print("P6 : contrôles de la période longue passés (aucun fichier à exporter).")
