# =============================================================================
# 05_application/estimation.py : estimer(bien), bloc B (Cécile)
# =============================================================================
# Pourquoi : l'outil donne le prix estimé d'un bien en 2026 avec le modèle retenu, H7v4 (réseau par vente).
#
# Ce que fait estimer(bien), dans l'ordre (plan de développement, étape 5) :
#   1. localise la commune (localisation.py, Théophyle) ; Alsace-Moselle : « non couvert par DVF » ;
#   2. prend la ligne 2026 du sous-bassin et du type de bien (lignes_2026.parquet) ;
#   3. prépare le bien comme dans H7 (log de la surface, terrain plafonné, pièces, dépendances) ;
#   4. prévoit le prix au m² avec le réseau rechargé, puis le prix total (prix au m² x surface) ;
#   5. remplit le contexte avec la même ligne : prix médian 2025, évolution 2024 -> 2025, ventes 2025.
#
# Contrat (contrats.py) : estimer(bien) -> prix_m2, prix_total, sous_bassin_de_vie, annee, contexte, message.
# Pas de fourchette ici : elle vient des prix réels des biens similaires (comparables, Théophyle).
#
# Données (donnees/, dans GitHub car l'application ne fonctionne pas sans elles) :
#   h7v4.keras, h7v4_parametres.json : produits par 04_outils/modele_final_h7v4.py
#   lignes_2026.parquet              : produit par 04_outils/produire_lignes_2026.py
# Application autonome (choix de Cécile, 17/11) : elle n'importe rien du socle. Les quelques lignes qui servent
# à prévoir sont recopiées de socle/gabarit_keras.py (ReseauKeras.predict et _entrees) ; si le gabarit change,
# reporter le changement ici.
# La fourchette d'erreur rangée dans h7v4_parametres.json n'est pas utilisée (page Résultats et rapport).
#
# Vérification rapide : python estimation.py (depuis 05_application) ; quelques biens tests.
# Contrôle sur les ventes 2025 : python 04_outils/controle_estimer_2025.py (depuis la racine du projet).

import json

import keras
import numpy as np
import pandas as pd
import streamlit as st

from commun_app import DONNEES, charger_parquet
from localisation import localiser

ANNEE = 2026
FICHIER_RESEAU = DONNEES / "h7v4.keras"
FICHIER_PARAMETRES = DONNEES / "h7v4_parametres.json"
FICHIER_LIGNES = "lignes_2026.parquet"
DEPARTEMENTS_NON_COUVERTS = ("57", "67", "68")            # Alsace-Moselle : absente de DVF (livre foncier)
PIECES_MAX = 5                                            # comme dans les ventes : 5 = 5 pièces ou plus
DEPENDANCES_MAX = 3                                       # comme dans H7 : 3 = 3 dépendances ou plus


# -----------------------------------------------------------------------------
# 1. Chargement du modèle et des lignes 2026 (une seule fois par session)
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_modele(fichier_reseau=FICHIER_RESEAU, fichier_parametres=FICHIER_PARAMETRES):
    """Le réseau enregistré et ses paramètres (échelle, sous-bassins, préparation), lus une seule fois."""
    with open(fichier_parametres, encoding="utf-8") as f:
        parametres = json.load(f)
    reseau = keras.saving.load_model(fichier_reseau, compile=False)     # prévoir seulement, pas réentraîner
    return reseau, parametres


def prevoir(reseau, parametres, X):
    """Prix au m² prévus pour les lignes de X (recopié de ReseauKeras.predict, socle/gabarit_keras.py).
    1. met chaque variable à l'échelle apprise (moyenne et écart-type de l'apprentissage) ;
    2. donne à chaque sous-bassin son numéro dans l'embedding (0 = inconnu) ;
    3. le réseau prévoit le rapport prix / prix N-1, en log et à l'échelle : on le remet en €/m²."""
    e, r = parametres["echelle"], parametres["reglages"]
    variables = e["variables"]
    moyennes = pd.Series(e["moyennes"])[variables]
    ecarts = pd.Series(e["ecarts_types"])[variables]
    numeriques = ((X[variables].astype(float) - moyennes) / ecarts).to_numpy("float32")
    entrees = [numeriques]
    if r["colonne_embedding"] is not None:
        numeros = X[r["colonne_embedding"]].map(parametres["sous_bassins"]).fillna(0).astype("int32").to_numpy()
        entrees.append(numeros)
    p = reseau.predict(entrees, batch_size=8192, verbose=0).ravel().astype(float)
    p = p * e["ecart_type_cible"] + e["moyenne_cible"]
    if r["cible_log"]:
        p = np.exp(p)
    if r.get("reference") is not None:                 # rapport prévu x prix de référence (prix N-1) -> €/m²
        p = p * np.exp(X[r["reference"]].to_numpy(dtype=float))
    return pd.Series(p, index=X.index)


def charger_lignes():
    """Lignes 2026, indexées par (sous_bassin_de_vie, type_bien)."""
    return charger_parquet(FICHIER_LIGNES).droplevel("annee")


# -----------------------------------------------------------------------------
# 2. Préparation : les variables de H7v4, dans l'ordre du modèle
# -----------------------------------------------------------------------------
def preparer_entrees(biens, lignes, parametres):
    """biens : tableau avec sous_bassin_de_vie, type_bien, sbati, nb_pieces, sterr, nblocdep.
    lignes : lignes d'une année, indexées par (sous_bassin_de_vie, type_bien).
    Renvoie les variables de H7v4, une ligne par bien (cases vides si le territoire n'a pas de ligne).
    Sert à estimer() (un bien) et au contrôle sur 2025 (toutes les ventes)."""
    prep = parametres["preparation"]
    b = biens.reset_index(drop=True)
    cles = pd.MultiIndex.from_frame(b[["sous_bassin_de_vie", "type_bien"]])
    t = lignes.reindex(cles).reset_index(drop=True)       # la ligne du territoire de chaque bien

    X = pd.DataFrame(index=b.index)
    X["sous_bassin_de_vie"] = b["sous_bassin_de_vie"].astype(str)

    # Le bien (comme H7, section 2.1)
    X["maison"] = (b["type_bien"] == "maison").astype(int)
    X["log_sbati"] = np.log(b["sbati"].astype(float))
    terrain = b["sterr"].astype(float).where(b["type_bien"] == "maison", 0).clip(upper=prep["seuil_sterr"])
    X["log1p_sterr"] = np.log1p(terrain)
    pieces = b["nb_pieces"].clip(upper=PIECES_MAX)
    for k, col in enumerate(prep["colonnes_pieces"], start=1):
        X[col] = (pieces == k).astype(int)
    dependances = b["nblocdep"].clip(upper=DEPENDANCES_MAX)
    for k, col in enumerate(prep["colonnes_dependances"]):
        X[col] = (dependances == k).astype(int)

    # Le territoire et le prix N-1 (comme H7, sections 4.4 et 4.7)
    for v in prep["territoire_en_log"]:
        X["log_" + v] = np.log(t[v])
    X["log_prix_n1"] = np.log(t["prix_n1"])

    # Le « quand » : élan vide -> 0 + indicatrice, puis logs (comme H7, section 4.10)
    X["evol_n2_inconnue"] = t["evol_prix_n2_n1"].isna().astype(int)
    X["evol_prix_n2_n1"] = t["evol_prix_n2_n1"].fillna(0)
    for v in prep["quand_en_log"]:
        X["log_" + v] = np.log(t[v])

    # Les autres variables sont reprises telles quelles de la ligne (parc, vacance, trimestres...)
    for c in prep["variables_h7v4"]:
        if c not in X.columns:
            X[c] = t[c]
    return X[prep["variables_h7v4"]]


# -----------------------------------------------------------------------------
# 3. estimer(bien)
# -----------------------------------------------------------------------------
def _sans_estimation(sous_bassin, message):
    """Réponse quand il n'y a pas d'estimation : chiffres vides et message pour l'utilisateur."""
    return {"prix_m2": None, "prix_total": None, "sous_bassin_de_vie": sous_bassin, "annee": ANNEE,
            "contexte": None, "message": message}


def estimer(bien):
    """Prix estimé en 2026 du bien saisi (contrat : contrats.py)."""
    # 1. Localisation
    code = str(bien["commune"]).strip().upper().zfill(5)
    if code[:2] in DEPARTEMENTS_NON_COUVERTS:
        return _sans_estimation(None, "Commune d'Alsace ou de Moselle : non couverte par DVF "
                                      "(la publicité foncière y relève du livre foncier). Pas d'estimation possible.")
    lieu = localiser(code)
    sous_bassin = lieu["sous_bassin_de_vie"]
    if sous_bassin is None:
        return _sans_estimation(None, lieu["message"])

    # 2. Ligne 2026 du sous-bassin et du type de bien
    lignes = charger_lignes()
    if (sous_bassin, bien["type_bien"]) not in lignes.index:
        type_pluriel = "maisons" if bien["type_bien"] == "maison" else "appartements"
        return _sans_estimation(sous_bassin, f"Territoire non couvert : trop peu de ventes de {type_pluriel} "
                                             f"en 2025 dans ce secteur pour estimer un prix.")
    reseau, parametres = charger_modele()
    if sous_bassin not in parametres["sous_bassins"]:
        return _sans_estimation(sous_bassin, "Territoire inconnu du modèle : pas d'estimation possible.")

    # 3-4. Préparation du bien et prévision
    saisie = pd.DataFrame([{**bien, "sous_bassin_de_vie": sous_bassin}])
    X = preparer_entrees(saisie, lignes, parametres)
    assert X.notna().all().all(), "case vide dans les variables du bien"
    prix_m2 = float(prevoir(reseau, parametres, X).iloc[0])

    # 5. Contexte du territoire (même ligne)
    ligne = lignes.loc[(sous_bassin, bien["type_bien"])]
    evolution = ligne["evol_prix_n2_n1"]
    contexte = {"prix_median_2025": round(float(ligne["prix_n1"])),
                "evolution_pct": None if pd.isna(evolution) else round(float(evolution), 1),
                "nb_ventes_2025": int(ligne["nb_ventes_n1"])}
    return {"prix_m2": round(prix_m2), "prix_total": round(prix_m2 * bien["sbati"], -3),
            "sous_bassin_de_vie": sous_bassin, "annee": ANNEE, "contexte": contexte, "message": None}


# -----------------------------------------------------------------------------
# Vérification : quelques biens tests (plan, section 5) et le respect du contrat
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from contrats import verifier_format

    BIENS = {
        "maison 90 m², Bordeaux": {"commune": "33063", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4,
                                   "sterr": 400.0, "nblocdep": 1},
        "T3 de 65 m², Bordeaux": {"commune": "33063", "type_bien": "appartement", "sbati": 65.0, "nb_pieces": 3,
                                  "sterr": 0.0, "nblocdep": 0},
        "T2 de 40 m², Paris 8e": {"commune": "75108", "type_bien": "appartement", "sbati": 40.0, "nb_pieces": 2,
                                  "sterr": 0.0, "nblocdep": 0},
        "maison 8 pièces (plafond 5)": {"commune": "33063", "type_bien": "maison", "sbati": 200.0, "nb_pieces": 8,
                                        "sterr": 1500.0, "nblocdep": 5},
        "appartement à Mulhouse": {"commune": "68224", "type_bien": "appartement", "sbati": 60.0, "nb_pieces": 3,
                                   "sterr": 0.0, "nblocdep": 0},
        "maison, sous-bassin 49069": {"commune": "49069", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4,
                                      "sterr": 400.0, "nblocdep": 1},
        "Paris sans arrondissement": {"commune": "75056", "type_bien": "appartement", "sbati": 40.0,
                                      "nb_pieces": 2, "sterr": 0.0, "nblocdep": 0},
        "commune inconnue": {"commune": "99999", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4,
                             "sterr": 400.0, "nblocdep": 1},
    }
    for nom, bien in BIENS.items():
        r = verifier_format("estimer", estimer(bien))
        print(f"{nom} : prix {r['prix_m2']} €/m², total {r['prix_total']} €, sous-bassin {r['sous_bassin_de_vie']}"
              f" | contexte {r['contexte']} | message {r['message']}")

    # Même saisie, même résultat
    assert estimer(BIENS["T3 de 65 m², Bordeaux"]) == estimer(BIENS["T3 de 65 m², Bordeaux"])
    # Un bien plus grand coûte plus cher au total
    plus_grand = {**BIENS["T3 de 65 m², Bordeaux"], "sbati": 80.0}
    assert estimer(plus_grand)["prix_total"] > estimer(BIENS["T3 de 65 m², Bordeaux"])["prix_total"]
    print("Contrôles réussis.")
