# =============================================================================
# 04_outils/controle_estimer_2025.py : contrôle de estimer() sur les ventes 2025 (bloc B, étape 5)
# =============================================================================
# Pourquoi : le plan demande d'essayer estimer() d'abord sur les lignes 2025, dont on connaît les vrais prix,
# avant les lignes 2026.
#
# Ce que fait ce fichier :
#   1. reconstruit les lignes 2025 (lignes_annee.construire_lignes, contrôlée par controle_lignes_2025.py) ;
#   2. prépare chaque vente 2025 avec la MÊME fonction que l'outil (estimation.preparer_entrees) ;
#   3. prévoit avec le réseau de l'outil (05_application/donnees/h7v4.keras) ;
#   4. compare à la naïve (prix médian 2024 du sous-bassin) avec la mesure de l'évaluation B du rapport ;
#   5. vérifie que localiser() redonne le sous-bassin des ventes (échantillon de communes).
# Limite : le réseau de l'outil a appris sur 2017-2025 ; 2025 fait donc partie de son apprentissage.
# Le gain mesuré ici n'est PAS un test (le test officiel est celui du rapport 2, tableau 20) : ce contrôle
# vérifie seulement que la préparation de l'outil est juste. Un gain négatif ou très faible signalerait
# une erreur de préparation (variable mal transformée, mauvaise ligne).
# Lancement, depuis la racine du projet :  python 04_outils/controle_estimer_2025.py
# Données : data/processed (local seulement) ; ce contrôle ne fait pas partie de l'application.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import commun
from lignes_annee import construire_lignes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_application"))
from estimation import charger_modele, preparer_entrees, prevoir      # noqa: E402
from localisation import localiser                           # noqa: E402

ECHANTILLON_COMMUNES = 200

# 1. Ventes 2025 et lignes 2025
ventes = pd.read_parquet(commun.DATA_OUT / "ventes_test_2025.parquet")
lignes = construire_lignes(2025).droplevel("annee")
cles = pd.MultiIndex.from_frame(ventes[["sous_bassin_de_vie", "type_bien"]])
avec_ligne = cles.isin(lignes.index)
print(f"Ventes 2025 : {len(ventes)} | avec une ligne 2025 : {int(avec_ligne.sum())} "
      f"| sans ligne (non estimées) : {int((~avec_ligne).sum())}")
ventes = ventes[avec_ligne].reset_index(drop=True)

# 2-3. Préparation de l'outil et prévision
reseau, parametres = charger_modele()
X = preparer_entrees(ventes, lignes, parametres)
vides = X.isna().any()
assert not vides.any(), f"cases vides dans : {list(vides[vides].index)}"
prevision = prevoir(reseau, parametres, X).to_numpy()
assert np.isfinite(prevision).all() and (prevision > 0).all(), "prévision vide, infinie ou négative"
inconnus = ~X["sous_bassin_de_vie"].isin(parametres["sous_bassins"].keys())
print("Ventes dans un sous-bassin inconnu du réseau :", int(inconnus.sum()))

# 4. Comparaison à la naïve, avec la MÊME mesure que l'évaluation B du rapport (commun.evaluer_ventes) :
#    MAE de chaque sous-bassin (ventes pondérées par w_h), puis moyenne des sous-bassins, chacun comptant pour un.
#    Biais : médiane pondérée de (prévision / prix réel - 1). Une moyenne de ce rapport serait tirée vers le haut
#    par les ventes à très bas prix au m² (rapport très grand) : la médiane ne l'est pas.
ventes["annee"] = 2025
ventes["prix_n1"] = np.exp(X["log_prix_n1"].to_numpy())          # naïve : prix médian 2024 du sous-bassin
prevision = pd.Series(prevision, index=ventes.index)
resultats = []
commun.evaluer_ventes("H7v4 de l'outil", ventes, prevision, resultats)
print("\nÉvaluation B sur 2025, apprentissage compris (pas un test ; test officiel : rapport 2, tableau 20) :")
for r in resultats:
    m = (ventes["type_bien"] == r["type de bien"]).to_numpy()
    biais = commun.mediane_ponderee(prevision[m] / ventes.loc[m, "prix_m2"] - 1, ventes.loc[m, "w_h"]) * 100
    biais_naive = commun.mediane_ponderee(ventes.loc[m, "prix_n1"] / ventes.loc[m, "prix_m2"] - 1,
                                          ventes.loc[m, "w_h"]) * 100
    print(f"  {r['type de bien']:<12} MAE H7v4 {r['MAE (€/m²)']:5.0f} | naïve {r['MAE naïve (€/m²)']:5.0f} "
          f"| gain {r['gain sur la naïve (%)']:+.1f} % | biais médian H7v4 {biais:+.1f} %, naïve {biais_naive:+.1f} % "
          f"| {int(m.sum())} ventes")
    assert r["gain sur la naïve (%)"] > 0, (f"{r['type de bien']} : H7v4 ne bat pas la naïve sur des ventes "
                                            "qu'il a apprises : préparation à revoir")

# 5. localiser() redonne le sous-bassin des ventes
communes = ventes[["code_commune", "sous_bassin_de_vie"]].drop_duplicates("code_commune")
communes = communes.sample(min(ECHANTILLON_COMMUNES, len(communes)), random_state=0)
differents = [c for c, s in zip(communes["code_commune"], communes["sous_bassin_de_vie"])
              if localiser(c)["sous_bassin_de_vie"] != s]
print(f"\nlocaliser() sur {len(communes)} communes : {len(differents)} sous-bassin(s) différent(s)",
      differents[:10])
assert not differents, "localiser() ne redonne pas le sous-bassin des ventes"
print("\nContrôle réussi : estimer() prépare les ventes 2025 comme le modèle les attend.")