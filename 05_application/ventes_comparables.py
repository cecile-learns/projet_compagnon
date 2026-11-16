# =============================================================================
# 05_application/ventes_comparables.py : comparables(bien, k=8), bloc C (Théophyle)
# =============================================================================
# Pourquoi : l'estimation du modèle seule ne suffit pas à un acheteur. comparables() lui montre des ventes
# réelles : les plus ressemblantes à son bien, et la fourchette des prix des biens équivalents du secteur.
#
# Contrat (contrats.py) : comparables(bien, k=8) -> dictionnaire
#   ventes     : les k ventes les plus ressemblantes (colonnes COLONNES_COMPARABLES), triées par distance
#   maille     : "sous-bassin", ou "bassin de vie" si le sous-bassin a moins de MIN_VENTES_PROCHES ventes du type
#   fourchette : prix au m² des ventes équivalentes (du 12,5e au 87,5e centile), ou None s'il y en a trop peu
#   message    : texte ou None
#
# 1. Ventes les plus ressemblantes (plus proches voisins)
#    Même type de bien, même sous-bassin (bassin de vie si moins de 5 ventes). Ressemblance mesurée sur :
#      - la surface, en log (60 -> 70 m² compte autant que 120 -> 140 m²) ;
#      - le nombre de pièces ;
#      - le terrain, en log(1 + x), pour les maisons seulement.
#    Chaque variable est mise à l'échelle (centrée, divisée par son écart-type) sur les ventes du secteur :
#    sinon le terrain, en centaines de m², écraserait les pièces. Distance = distance euclidienne.
#    À distance égale, la vente la plus récente passe d'abord.
#
# 2. Fourchette des biens équivalents (règle proposée par Cécile, contrat du 15/11)
#    Ventes du même type des NB_ANNEES_FOURCHETTE dernières années. On essaie les niveaux dans l'ordre
#    (NIVEAUX_APPARTEMENT, NIVEAUX_MAISON) et on garde le premier qui a au moins SEUIL_FOURCHETTE ventes.
#    Bornes : du 12,5e au 87,5e centile du prix au m² (les 75 % du milieu). La surface saisie n'intervient pas.
#
# Données : donnees/ventes_comparables.parquet (04_outils/produire_ventes_comparables.py, Cécile).
# Colonnes lues : code_commune, nom_commune, sous_bassin_de_vie, nom_sous_bassin, BV2022, nom_bassin,
#                 type_bien, annee, mois, sbati, nb_pieces (1 à 5, 5 = 5 ou plus ; vide si inconnu), sterr, prix_m2.
# Vérification : python ventes_comparables.py (depuis 05_application).

import numpy as np
import pandas as pd

from commun_app import charger_parquet
from contrats import COLONNES_COMPARABLES, NB_ANNEES_FOURCHETTE, PART_FOURCHETTE, SEUIL_FOURCHETTE
from localisation import localiser

FICHIER = "ventes_comparables.parquet"
MIN_VENTES_PROCHES = 5          # en dessous, recherche des ventes proches élargie au bassin de vie

# Niveaux de la fourchette, du plus précis au plus large : (échelle, pièces)
#   pièces : "même" (même nombre de pièces), "±1" (une pièce de plus ou de moins), "toutes"
# Ordre validé avec Cécile par le test du 16/11 : changer de taille élargit peu la fourchette (quelques points),
# changer de lieu peut l'élargir beaucoup (Paris 8e : 67 % -> 115 % en passant à tout Paris).
# On reste donc dans le sous-bassin, toutes tailles comprises, avant de passer au bassin de vie.
NIVEAUX_APPARTEMENT = [("sous-bassin", "même"), ("sous-bassin", "±1"), ("sous-bassin", "toutes"),
                       ("bassin de vie", "même"), ("bassin de vie", "±1"), ("bassin de vie", "toutes")]
NIVEAUX_MAISON = [("sous-bassin", "toutes"), ("bassin de vie", "toutes")]   # la taille joue peu sur le prix au m²

# -----------------------------------------------------------------------------
# 1. Lecture et sélection des ventes
# -----------------------------------------------------------------------------
def charger_ventes():
    """Toutes les ventes de l'outil (lues une seule fois grâce au cache de charger_parquet)."""
    return charger_parquet(FICHIER)


def ventes_du_secteur(ventes, type_bien, echelle, lieu):
    """Ventes d'un type de bien dans un sous-bassin ou un bassin de vie (codes en texte)."""
    colonne = "sous_bassin_de_vie" if echelle == "sous-bassin" else "BV2022"
    # Comparaison directe sur les colonnes « category » du fichier : rapide, sans tout convertir en texte
    garde = (ventes["type_bien"] == type_bien) & (ventes[colonne] == lieu)
    return ventes[garde]


def classe_pieces(nb_pieces):
    """Nombre de pièces ramené à la classe du fichier : 1, 2, 3, 4, ou 5 (5 pièces ou plus)."""
    return min(int(nb_pieces), 5)


# -----------------------------------------------------------------------------
# 2. Ventes les plus ressemblantes
# -----------------------------------------------------------------------------
def variables_ressemblance(table, type_bien):
    """Variables de la distance : log de la surface, pièces, et log(1 + terrain) pour une maison."""
    variables = pd.DataFrame({"surface": np.log(table["sbati"].astype(float)),
                              "pieces": table["nb_pieces"].astype(float)}, index=table.index)
    if type_bien == "maison":
        variables["terrain"] = np.log1p(table["sterr"].astype(float))
    return variables


def plus_proches(candidates, bien, k):
    """Les k ventes les plus ressemblantes au bien, avec leur distance, au format du contrat."""
    # Ventes au nombre de pièces inconnu : on ne peut pas mesurer leur ressemblance, elles sont écartées
    candidates = candidates[candidates["nb_pieces"].notna()]
    if len(candidates) == 0:
        return pd.DataFrame(columns=COLONNES_COMPARABLES)

    # Le bien saisi, avec les mêmes variables que les ventes (pièces ramenées à la classe 5 et plus)
    saisie = pd.DataFrame({"sbati": [bien["sbati"]], "nb_pieces": [classe_pieces(bien["nb_pieces"])],
                           "sterr": [bien["sterr"]]})
    x_ventes = variables_ressemblance(candidates, bien["type_bien"])
    x_bien = variables_ressemblance(saisie, bien["type_bien"])

    # Mise à l'échelle sur les ventes du secteur ; une variable constante n'est pas divisée par 0
    moyennes = x_ventes.mean()
    ecarts = x_ventes.std(ddof=0).replace(0, 1.0)
    z_ventes = (x_ventes - moyennes) / ecarts
    z_bien = (x_bien.iloc[0] - moyennes) / ecarts

    # Distance euclidienne de chaque vente au bien
    distance = np.sqrt(((z_ventes - z_bien) ** 2).sum(axis=1))

    resultat = candidates.assign(distance=distance.round(3))
    resultat = resultat.sort_values(["distance", "annee", "mois"], ascending=[True, False, False]).head(k)

    # Format du contrat : la date est le 1er du mois de la vente (DVF+ ne donne ici que le mois)
    dates = pd.to_datetime(pd.DataFrame({"year": resultat["annee"].astype(int),
                                         "month": resultat["mois"].astype(int), "day": 1}))
    return pd.DataFrame({"commune": resultat["nom_commune"].astype(str), "date": dates,
                         "sbati": resultat["sbati"].astype(float), "nb_pieces": resultat["nb_pieces"].astype(int),
                         "sterr": resultat["sterr"].astype(float), "prix_m2": resultat["prix_m2"].astype(float),
                         "distance": resultat["distance"].astype(float)}).reset_index(drop=True)[COLONNES_COMPARABLES]


# -----------------------------------------------------------------------------
# 3. Fourchette des biens équivalents
# -----------------------------------------------------------------------------
def nom_niveau(type_bien, pieces, classe):
    """Libellé affiché : « T3 », « T2 à T4 », « T5 et plus », « appartements », « maisons »."""
    if type_bien == "maison":
        return "maisons"
    if pieces == "toutes":
        return "appartements"

    def t(n):
        return "T5 et plus" if n == 5 else "T" + str(n)

    if pieces == "même":
        return t(classe)
    bas, haut = max(classe - 1, 1), min(classe + 1, 5)
    return t(bas) + " à " + t(haut)


def fourchette_equivalents(ventes, bien, lieu):
    """Premier niveau qui a au moins SEUIL_FOURCHETTE ventes ; None si aucun."""
    type_bien = bien["type_bien"]
    classe = classe_pieces(bien["nb_pieces"])

    # Ventes des NB_ANNEES_FOURCHETTE dernières années du fichier
    derniere = int(ventes["annee"].max())
    recentes = ventes[ventes["annee"] > derniere - NB_ANNEES_FOURCHETTE]

    niveaux = NIVEAUX_MAISON if type_bien == "maison" else NIVEAUX_APPARTEMENT
    for echelle, pieces in niveaux:
        code = lieu["sous_bassin_de_vie"] if echelle == "sous-bassin" else lieu["bassin_de_vie"]
        secteur = ventes_du_secteur(recentes, type_bien, echelle, code)
        if pieces == "même":
            secteur = secteur[secteur["nb_pieces"] == classe]
        elif pieces == "±1":
            secteur = secteur[(secteur["nb_pieces"] >= classe - 1) & (secteur["nb_pieces"] <= classe + 1)]

        if len(secteur) >= SEUIL_FOURCHETTE:
            part_hors = (100 - PART_FOURCHETTE) / 2 / 100          # 12,5 % de chaque côté
            nom_lieu = secteur["nom_sous_bassin" if echelle == "sous-bassin" else "nom_bassin"].astype(str).iloc[0]
            return {"nb_ventes": int(len(secteur)), "niveau": nom_niveau(type_bien, pieces, classe),
                                        # bassin non découpé : sous-bassin = bassin, on ne parle pas d'élargissement
                    "echelle": "sous-bassin" if lieu["sous_bassin_de_vie"] == lieu["bassin_de_vie"] else echelle,
                    "lieu": nom_lieu,
                    # bornes arrondies à la dizaine d'euros : pas de fausse précision
                    "basse_m2": float(round(secteur["prix_m2"].quantile(part_hors), -1)),
                    "haute_m2": float(round(secteur["prix_m2"].quantile(1 - part_hors), -1))}
    return None


# -----------------------------------------------------------------------------
# 4. La fonction du contrat
# -----------------------------------------------------------------------------
def comparables(bien, k=8):
    """Ventes les plus ressemblantes au bien et fourchette des biens équivalents."""
    vide = pd.DataFrame(columns=COLONNES_COMPARABLES)
    lieu = localiser(bien["commune"])
    if lieu["sous_bassin_de_vie"] is None:
        return {"ventes": vide, "maille": "sous-bassin", "fourchette": None, "message": lieu["message"]}

    ventes = charger_ventes()

    # Ventes proches : sous-bassin, ou bassin de vie s'il y a trop peu de ventes du type
    maille = "sous-bassin"
    candidates = ventes_du_secteur(ventes, bien["type_bien"], "sous-bassin", lieu["sous_bassin_de_vie"])
    if len(candidates) < MIN_VENTES_PROCHES:
        maille = "bassin de vie"
        candidates = ventes_du_secteur(ventes, bien["type_bien"], "bassin de vie", lieu["bassin_de_vie"])

    fourchette = fourchette_equivalents(ventes, bien, lieu)
    message = None
    if fourchette is None:
        message = ("Pas assez de ventes équivalentes pour donner une fourchette de prix (moins de "
                   + str(SEUIL_FOURCHETTE) + ", même à l'échelle du bassin de vie).")

    return {"ventes": plus_proches(candidates, bien, k), "maille": maille, "fourchette": fourchette,
            "message": message}


# -----------------------------------------------------------------------------
# Vérification : quelques biens tests, et le respect du contrat
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from contrats import verifier_format

    pd.set_option("display.width", 160)
    BIENS = {
        "maison 90 m², Bordeaux": {"commune": "33063", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4,
                                   "sterr": 400.0, "nblocdep": 1},
        "T3 de 65 m², Bordeaux": {"commune": "33063", "type_bien": "appartement", "sbati": 65.0, "nb_pieces": 3,
                                  "sterr": 0.0, "nblocdep": 0},
        "T2 de 40 m², Paris 8e": {"commune": "75108", "type_bien": "appartement", "sbati": 40.0, "nb_pieces": 2,
                                  "sterr": 0.0, "nblocdep": 0},
        "maison, petite commune": {"commune": "01001", "type_bien": "maison", "sbati": 110.0, "nb_pieces": 5,
                                   "sterr": 1500.0, "nblocdep": 2},
        "T6, petite commune": {"commune": "01001", "type_bien": "appartement", "sbati": 140.0, "nb_pieces": 6,
                               "sterr": 0.0, "nblocdep": 0},
        "commune inconnue": {"commune": "99999", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4,
                             "sterr": 400.0, "nblocdep": 0},
    }
    for nom in BIENS:
        resultat = verifier_format("comparables", comparables(BIENS[nom]))
        print("\n=====", nom, "| maille :", resultat["maille"], "| message :", resultat["message"])
        print("fourchette :", resultat["fourchette"])
        print(resultat["ventes"].to_string(index=False))

    # Même saisie, même résultat
    bien = BIENS["maison 90 m², Bordeaux"]
    assert comparables(bien)["ventes"].equals(comparables(bien)["ventes"])
    print("\nContrôles réussis.")
