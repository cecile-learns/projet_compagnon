# =============================================================================
# 05_application/contrats.py : contrats entre les blocs de l'outil et fonctions provisoires
# =============================================================================
# Les pages n'appellent QUE les fonctions de ce fichier. Tant qu'une vraie fonction n'est pas prête,
# c'est sa version provisoire qui répond, au bon format : personne n'attend personne.
# Un changement de format est annoncé à l'équipe AVANT d'être fait.
#
# Branchement : chaque vraie fonction est écrite par son responsable dans son propre fichier
#   localiser            -> localisation.py   (Théophyle)
#   estimer              -> estimation.py     (Cécile)
#   comparables          -> ventes_comparables.py (Théophyle)
#   expliquer, expliquer_llm -> explication.py (Théophyle)
# Quand elle est prête, on passe son interrupteur à True dans BRANCHEES (section 4).
#
# ---------------------------------------------------------------------------
# FORMATS (font foi)
# ---------------------------------------------------------------------------
# bien (saisie de l'utilisateur) : dictionnaire
#   commune    : code INSEE en texte, 5 caractères ("33063") ; la page Outil traduit le nom choisi en code
#   type_bien  : "maison" ou "appartement"
#   sbati      : surface habitable en m² (nombre > 0)
#   nb_pieces  : nombre de pièces principales (entier >= 1)
#   sterr      : surface du terrain en m² (0 pour un appartement)
#   nblocdep   : nombre de dépendances (entier >= 0)
#
# localiser(commune) -> dictionnaire
#   commune, sous_bassin_de_vie, bassin_de_vie (textes ; None si commune inconnue), message (texte ou None)
#
# estimer(bien) -> dictionnaire
#   prix_m2, prix_total            : estimation 2026 (None si pas d'estimation)
#   sous_bassin_de_vie, annee      : territoire et année estimée (2026)
#   contexte : dictionnaire prix_median_2025, evolution_pct (2024 -> 2025, None si inconnue), nb_ventes_2025
#   message                        : texte ou None (sous-bassin non couvert, etc.)
#   (pas de fourchette : elle vient des prix réels des biens similaires, voir comparables)
#
# comparables(bien, k=8) -> dictionnaire
#   ventes      : tableau (DataFrame), une ligne par vente, colonnes COLONNES_COMPARABLES, triées par distance
#   maille      : "sous-bassin" ou "bassin de vie" (recherche élargie si moins de 5 ventes) ; elle ne concerne
#                 que le tableau des ventes, la fourchette a son propre niveau (voir `niveau`)
#   fourchette  : dictionnaire, ou None s'il n'y a pas assez de ventes équivalentes (au moins SEUIL_FOURCHETTE
#                 à chaque niveau essayé), avec alors le message « pas assez de ventes comparables »
#                 nb_ventes        : nombre de ventes équivalentes retenues (>= SEUIL_FOURCHETTE),
#                                    ventes des NB_ANNEES_FOURCHETTE dernières années
#                 niveau           : à quoi le bien est comparé, ex. "T3" ou "maisons"
#                 echelle, lieu    : "sous-bassin" ou "bassin de vie", et le nom de ce territoire
#                                    (c'est l'échelle de la fourchette, distincte de `maille` ci-dessus)
#                 basse_m2, haute_m2 : bornes en €/m² : du 12,5e au 87,5e centile du prix au m² des ventes
#                                    retenues (PART_FOURCHETTE % du milieu). La surface saisie n'intervient pas.
#   message     : texte ou None
#
# expliquer(estimation, comparables) -> texte   (modèle de phrase fixe)
# expliquer_llm(estimation, comparables) -> texte (Groq ; repli sur expliquer en cas d'erreur ou de délai)

import pandas as pd

TYPES_BIEN = ("maison", "appartement")
CHAMPS_BIEN = ("commune", "type_bien", "sbati", "nb_pieces", "sterr", "nblocdep")
CHAMPS_LOCALISATION = ("commune", "sous_bassin_de_vie", "bassin_de_vie", "message")
CHAMPS_ESTIMATION = ("prix_m2", "prix_total", "sous_bassin_de_vie", "annee", "contexte", "message")
CHAMPS_CONTEXTE = ("prix_median_2025", "evolution_pct", "nb_ventes_2025")
CHAMPS_COMPARABLES = ("ventes", "maille", "fourchette", "message")
CHAMPS_FOURCHETTE = ("nb_ventes", "niveau", "echelle", "lieu", "basse_m2", "haute_m2")
SEUIL_FOURCHETTE = 30      # ventes équivalentes minimum à chaque niveau, sinon pas de fourchette
PART_FOURCHETTE = 75       # part des ventes du milieu (en %) comprise entre basse_m2 et haute_m2
NB_ANNEES_FOURCHETTE = 3   # fourchette calculée sur les ventes de ces dernières années (période à confirmer)
COLONNES_COMPARABLES = ["commune", "date", "sbati", "nb_pieces", "sterr", "prix_m2", "distance"]


# -----------------------------------------------------------------------------
# 1. Contrôle du bien saisi
# -----------------------------------------------------------------------------
def verifier_bien(bien):
    """Renvoie la liste des problèmes du bien saisi (liste vide = bien correct)."""
    problemes = [f"champ manquant : {c}" for c in CHAMPS_BIEN if c not in bien]
    if problemes:
        return problemes
    if not (isinstance(bien["commune"], str) and len(bien["commune"]) == 5):
        problemes.append("commune : code INSEE en texte de 5 caractères attendu")
    if bien["type_bien"] not in TYPES_BIEN:
        problemes.append("type_bien : 'maison' ou 'appartement' attendu")
    if not bien["sbati"] > 0:
        problemes.append("sbati : surface positive attendue")
    if not bien["nb_pieces"] >= 1:
        problemes.append("nb_pieces : au moins 1 pièce")
    if bien["sterr"] < 0 or (bien["type_bien"] == "appartement" and bien["sterr"] != 0):
        problemes.append("sterr : positif, et 0 pour un appartement")
    if bien["nblocdep"] < 0:
        problemes.append("nblocdep : entier positif ou nul")
    return problemes


# -----------------------------------------------------------------------------
# 2. Fonctions provisoires (chiffres fixes, au bon format)
# -----------------------------------------------------------------------------
def _localiser_provisoire(commune):
    if commune == "00000":
        return {"commune": commune, "sous_bassin_de_vie": None, "bassin_de_vie": None,
                "message": "Commune inconnue : vérifiez le nom saisi."}
    return {"commune": commune, "sous_bassin_de_vie": "330631", "bassin_de_vie": "33063", "message": None}


def _prix_m2_provisoire(bien):
    return 3500.0 if bien["type_bien"] == "maison" else 4200.0


def _estimer_provisoire(bien):
    prix_m2 = _prix_m2_provisoire(bien)
    return {"prix_m2": prix_m2, "prix_total": prix_m2 * bien["sbati"],
            "sous_bassin_de_vie": "330631", "annee": 2026,
            "contexte": {"prix_median_2025": prix_m2 * 0.98, "evolution_pct": 1.5, "nb_ventes_2025": 1200},
            "message": "Estimation provisoire : chiffres fixes, en attendant le modèle."}


def _comparables_provisoire(bien, k=8):
    ventes = pd.DataFrame({
        "commune": ["Bordeaux", "Mérignac", "Pessac"],
        "date": pd.to_datetime(["2025-03-14", "2025-09-02", "2026-02-20"]),
        "sbati": [85.0, 92.0, 88.0],
        "nb_pieces": [4, 4, 5],
        "sterr": [300.0, 450.0, 380.0] if bien["type_bien"] == "maison" else [0.0, 0.0, 0.0],
        "prix_m2": [3400.0, 3650.0, 3520.0],
        "distance": [0.12, 0.25, 0.31],
    })[COLONNES_COMPARABLES]
    prix_m2 = _prix_m2_provisoire(bien)
    fourchette = {"nb_ventes": 48, "niveau": "T3" if bien["type_bien"] == "appartement" else "maisons",
                  "echelle": "sous-bassin", "lieu": "Exemple-sur-Garonne",
                  "basse_m2": round(prix_m2 * 0.85, -1), "haute_m2": round(prix_m2 * 1.15, -1)}
    return {"ventes": ventes.head(k), "maille": "sous-bassin", "fourchette": fourchette,
            "message": "Ventes et fourchette inventées (provisoire)."}


def _nombre(x):
    """3500.0 -> '3 500' (espace pour les milliers, à la française)."""
    return f"{x:,.0f}".replace(",", " ")


def _expliquer_provisoire(estimation, comparables):
    if estimation["prix_m2"] is None:
        return "Pas d'estimation pour ce territoire."
    texte = (f"Le prix estimé en {estimation['annee']} est de {_nombre(estimation['prix_total'])} € "
             f"soit environ {_nombre(estimation['prix_m2'])} €/m². ")
    f = comparables["fourchette"]
    if f is None:
        return texte + "À titre d'information : pas assez de ventes comparables pour donner une fourchette de prix."
    return texte + (f"À titre d'information, sur les {NB_ANNEES_FOURCHETTE} dernières années, {f['nb_ventes']} biens "
                    f"équivalents ont été vendus dans le {f['echelle']} de {f['lieu']}. Pour {PART_FOURCHETTE} % "
                    f"d'entre eux, le prix au m² était compris entre {_nombre(f['basse_m2'])} "
                    f"et {_nombre(f['haute_m2'])} €.")


def _expliquer_llm_provisoire(estimation, comparables):
    return _expliquer_provisoire(estimation, comparables)


# -----------------------------------------------------------------------------
# 3. Contrôle des formats rendus (contrôle « respect des contrats »)
# -----------------------------------------------------------------------------
def verifier_format(nom, resultat):
    """Lève une erreur si le résultat d'une fonction ne respecte pas son contrat."""
    attendus = {"localiser": CHAMPS_LOCALISATION, "estimer": CHAMPS_ESTIMATION,
                "comparables": CHAMPS_COMPARABLES}
    if nom in attendus:
        manquants = [c for c in attendus[nom] if c not in resultat]
        assert not manquants, f"{nom} : champs manquants {manquants}"
    if nom == "estimer" and resultat["contexte"] is not None:
        manquants = [c for c in CHAMPS_CONTEXTE if c not in resultat["contexte"]]
        assert not manquants, f"estimer, contexte : champs manquants {manquants}"
    if nom == "comparables":
        assert list(resultat["ventes"].columns) == COLONNES_COMPARABLES, "comparables : colonnes du tableau"
        assert resultat["maille"] in ("sous-bassin", "bassin de vie"), "comparables : maille"
        f = resultat["fourchette"]
        if f is None:
            assert resultat["message"], "comparables : sans fourchette, un message doit expliquer pourquoi"
        else:
            manquants = [c for c in CHAMPS_FOURCHETTE if c not in f]
            assert not manquants, f"comparables, fourchette : champs manquants {manquants}"
            assert f["nb_ventes"] >= SEUIL_FOURCHETTE, f"comparables, fourchette : moins de {SEUIL_FOURCHETTE} ventes"
            assert f["echelle"] in ("sous-bassin", "bassin de vie"), "comparables, fourchette : echelle"
            assert f["basse_m2"] <= f["haute_m2"], "comparables, fourchette : basse_m2 supérieure à haute_m2"
    if nom in ("expliquer", "expliquer_llm"):
        assert isinstance(resultat, str), f"{nom} : texte attendu"
    return resultat


# -----------------------------------------------------------------------------
# 4. Branchement : provisoire ou vraie fonction
# -----------------------------------------------------------------------------
# Passer un interrupteur à True quand la vraie fonction est livrée (annoncé à l'équipe).
BRANCHEES = {"localiser": True, "estimer": True, "comparables": True, "expliquer": True, "expliquer_llm": True}


def localiser(commune):
    if BRANCHEES["localiser"]:
        from localisation import localiser as f
    else:
        f = _localiser_provisoire
    return verifier_format("localiser", f(commune))


def estimer(bien):
    if BRANCHEES["estimer"]:
        from estimation import estimer as f
    else:
        f = _estimer_provisoire
    return verifier_format("estimer", f(bien))


def comparables(bien, k=8):
    if BRANCHEES["comparables"]:
        from ventes_comparables import comparables as f
    else:
        f = _comparables_provisoire
    return verifier_format("comparables", f(bien, k=k))


def expliquer(estimation, comparables_):
    if BRANCHEES["expliquer"]:
        from explication import expliquer as f
    else:
        f = _expliquer_provisoire
    return verifier_format("expliquer", f(estimation, comparables_))


def expliquer_llm(estimation, comparables_):
    if BRANCHEES["expliquer_llm"]:
        from explication import expliquer_llm as f
    else:
        f = _expliquer_llm_provisoire
    return verifier_format("expliquer_llm", f(estimation, comparables_))
    