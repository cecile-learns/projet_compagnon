# =============================================================================
# 05_application/explication.py : expliquer() et expliquer_llm(), bloc C (Théophyle)
# =============================================================================
# Pourquoi : la page Outil affiche des chiffres (prix estimé, contexte du secteur, fourchette, ventes proches).
# expliquer() les reprend en un court paragraphe, lisible par un particulier.
#
# C'est le « modèle de phrase fixe » du plan : un texte à trous, sans modèle de langage. Il sert :
#   - d'explication par défaut ;
#   - de repli quand expliquer_llm() (Groq) échoue ou met trop de temps.
# Il doit donc toujours marcher, quel que soit le cas (pas d'estimation, pas de fourchette...).
#
# Contrat (contrats.py) : expliquer(estimation, comparables) -> texte
#   estimation  : sortie de estimer()     (prix_m2, prix_total, annee, contexte, message...)
#   comparables : sortie de comparables() (ventes, maille, fourchette, message)
#
# Règles du plan (contrôle « explication fidèle ») :
#   - n'utiliser QUE les chiffres reçus, ne rien recalculer d'autre que des arrondis ;
#   - aucune adresse (les ventes n'en ont pas, et le texte ne cite que des communes) ;
#   - toujours rappeler la limite du modèle.
#
# Vérification : python explication.py (depuis 05_application), sur des cas inventés.

import os
import re

import streamlit as st

import contrats

LIMITE = ("Le modèle prolonge l'évolution récente des prix : il n'anticipe pas un retournement du marché, "
          "et ne connaît ni l'état du bien, ni sa rue, ni ses travaux.")


# -----------------------------------------------------------------------------
# 1. Mise en forme des nombres, à la française
# -----------------------------------------------------------------------------
def nombre(x):
    """3500.0 -> « 3 500 »."""
    return format(x, ",.0f").replace(",", " ")


def euros(x):
    """3500.0 -> « 3 500 € »."""
    return nombre(x) + " €"


def pourcentage(x):
    """1.53 -> « +1,5 % »."""
    return format(x, "+.1f").replace(".", ",") + " %"


# -----------------------------------------------------------------------------
# 2. Les morceaux du texte, un par information
# -----------------------------------------------------------------------------
def phrase_estimation(estimation):
    """Le prix estimé, ou la raison de l'absence d'estimation."""
    if estimation["prix_m2"] is None:
        texte = "Nous ne pouvons pas estimer ce bien"
        message = estimation["message"]
        if message:
            # « Moins de 20 ventes... » -> « ... : moins de 20 ventes... » (minuscule après les deux-points)
            texte = texte + " : " + message[0].lower() + message[1:].rstrip(".")
        return texte + "."
    return ("Ce bien est estimé à " + euros(estimation["prix_total"]) + " en " + str(estimation["annee"])
            + ", soit environ " + euros(estimation["prix_m2"]) + " par m².")


def phrase_contexte(estimation):
    """Le prix médian du secteur en 2025 et son évolution sur un an (si l'estimation existe)."""
    contexte = estimation["contexte"]
    if estimation["prix_m2"] is None or contexte is None:
        return None
    texte = ("Dans ce secteur, le prix médian était de " + euros(contexte["prix_median_2025"])
             + " par m² en 2025")
    if contexte["evolution_pct"] is not None:
        texte = texte + " (" + pourcentage(contexte["evolution_pct"]) + " sur un an)"
    return texte + "."


def phrase_fourchette(estimation, comparables):
    """Les prix réels des biens équivalents, et la place de l'estimation par rapport à eux."""
    f = comparables["fourchette"]
    if f is None:
        return "Il n'y a pas assez de ventes équivalentes dans ce secteur pour donner une fourchette de prix."

    texte = ("Sur les " + str(contrats.NB_ANNEES_FOURCHETTE) + " dernières années, " + str(f["nb_ventes"])
             + " biens équivalents (" + f["niveau"] + ") ont été vendus dans le " + f["echelle"]
             + " « " + f["lieu"] + " » : pour " + str(contrats.PART_FOURCHETTE) + " % d'entre eux, le prix "
             + "était compris entre " + euros(f["basse_m2"]) + " et " + euros(f["haute_m2"]) + " par m²")

  
    texte = texte + "."

    # Place de l'estimation : le modèle et les ventes passées peuvent ne pas concorder
    if estimation["prix_m2"] is not None:
        if estimation["prix_m2"] > f["haute_m2"]:
            texte = (texte + " L'estimation est au-dessus de la plupart de ces ventes : le modèle tient compte "
                     "des caractéristiques du bien et de l'évolution récente du secteur.")
        elif estimation["prix_m2"] < f["basse_m2"]:
            texte = (texte + " L'estimation est en dessous de la plupart de ces ventes : le modèle tient compte "
                     "des caractéristiques du bien et de l'évolution récente du secteur.")
        else:
            texte = texte + " L'estimation se situe dans cette fourchette."
    return texte


def phrase_ventes(comparables):
    """Les ventes les plus proches : combien, où, et l'écart de leurs prix au m²."""
    ventes = comparables["ventes"]
    if len(ventes) == 0:
        return "Aucune vente comparable n'a été trouvée."
    texte = ("Les " + str(len(ventes)) + " ventes les plus ressemblantes se sont faites entre "
             + euros(ventes["prix_m2"].min()) + " et " + euros(ventes["prix_m2"].max()) + " par m²")
    if comparables["maille"] == "bassin de vie":
        texte = texte + " ; peu de ventes dans le secteur, la recherche a été élargie au bassin de vie"
    return texte + "."


# -----------------------------------------------------------------------------
# 3. La fonction du contrat
# -----------------------------------------------------------------------------
def expliquer(estimation, comparables):
    """Paragraphe d'explication, assemblé à partir des chiffres reçus ; la limite est toujours rappelée."""
    morceaux = [phrase_estimation(estimation),
                phrase_contexte(estimation),
                phrase_fourchette(estimation, comparables),
                phrase_ventes(comparables),
                LIMITE]
    # Les morceaux absents (None) sont sautés, les autres mis bout à bout, séparés par une espace
    texte = ""
    for morceau in morceaux:
        if morceau:
            texte = texte + morceau + " "
    return texte.strip()


# =============================================================================
# expliquer_llm(estimation, comparables) : la même explication, rédigée par un modèle de langage (Groq)
# =============================================================================
# Principe (plan de développement, partie « génération » du RAG) : le modèle de langage ne calcule rien.
# Il reçoit le texte de expliquer() comme seule source de faits, et le reformule de façon plus naturelle.
#
# Garde-fous :
#   - consigne stricte : n'utiliser que les faits fournis, aucun chiffre nouveau, aucune adresse, garder la limite ;
#   - contrôle après coup : chaque nombre du texte rédigé doit figurer dans les faits ; sinon, on jette le texte ;
#   - repli automatique sur expliquer() en cas d'erreur, de délai trop long, de clé absente ou de texte refusé ;
#   - même saisie, même résultat : température 0 et réponse gardée en mémoire (cache) pour un même texte de faits ;
#   - la clé n'est jamais dans le code : elle est lue dans les secrets de Streamlit
#     (05_application/.streamlit/secrets.toml en local, exclu de Git ; les « secrets » de l'hébergeur en ligne),
#     ou à défaut dans la variable d'environnement GROQ_API_KEY.

MODELE = "openai/gpt-oss-120b"       # modèle de Groq ; vérifier sur console.groq.com qu'il est toujours proposé
DELAI_MAX = 10                       # secondes ; au-delà, on se replie sur la phrase type

CONSIGNE = (
    "Tu rédiges, en français, l'explication d'une estimation immobilière pour un particulier qui veut acheter "
    "un logement. On te donne des FAITS déjà calculés. Règles strictes :\n"
    "1. N'utilise que les faits fournis. N'ajoute aucun chiffre, aucune date, aucun pourcentage, aucun nom de lieu "
    "qui n'y figure pas, et ne refais aucun calcul.\n"
    "2. Recopie les nombres exactement comme dans les faits (mêmes chiffres, mêmes espaces, mêmes unités).\n"
    "3. Ne donne aucune adresse, aucune rue, aucun conseil d'achat ou d'investissement.\n"
    "4. Garde la limite du modèle (il n'anticipe pas un retournement du marché).\n"
    "5. Un seul paragraphe de 4 à 6 phrases, vouvoiement, ton clair et neutre, sans liste ni titre."
)


def lire_cle():
    """Clé Groq : secrets de Streamlit, sinon variable d'environnement ; None si absente."""
    try:
        cle = st.secrets["GROQ_API_KEY"]
    except Exception:          # pas de fichier de secrets, ou clé absente du fichier
        cle = os.environ.get("GROQ_API_KEY")
    return cle or None


def nombres(texte):
    """Nombres d'un texte, écrits sans espaces : « 315 000 € » -> « 315000 », « +1,5 % » -> « 1,5 »."""
    trouves = re.findall(r"\d[\d   ]*(?:,\d+)?", texte)
    resultat = set()
    for n in trouves:
        resultat.add(n.replace(" ", "").replace(" ", "").replace(" ", ""))
    return resultat


def texte_fidele(texte, faits):
    """Vrai si chaque nombre du texte rédigé figure dans les faits (contrôle « explication fidèle »)."""
    return nombres(texte) <= nombres(faits)


@st.cache_data(show_spinner=False)
def rediger(faits):
    """Appel à Groq ; renvoie le texte rédigé, ou None en cas de problème (le repli est fait par l'appelant)."""
    cle = lire_cle()
    if cle is None:
        return None
    try:
        from groq import Groq     # importé ici : sans la bibliothèque, l'outil marche quand même (phrase type)
        client = Groq(api_key=cle, timeout=DELAI_MAX, max_retries=0)
               # gpt-oss « réfléchit » avant de répondre : réflexion courte (rapide), et assez de jetons pour la réflexion + le texte
        reponse = client.chat.completions.create(
            model=MODELE, temperature=0, max_tokens=1500, reasoning_effort="low",
            messages=[{"role": "system", "content": CONSIGNE},
                      {"role": "user", "content": "FAITS :\n" + faits}])
        texte = reponse.choices[0].message.content.strip()
    except Exception:              # réseau, délai, clé refusée, quota, bibliothèque absente...
        return None
    if not texte or not texte_fidele(texte, faits):
        return None
    return texte


def expliquer_llm(estimation, comparables):
    """Explication rédigée par Groq à partir de la phrase type ; la phrase type elle-même en cas de problème."""
    faits = expliquer(estimation, comparables)
    texte = rediger(faits)
    if texte is None:
        return faits
    return texte


# -----------------------------------------------------------------------------
# Vérification : les cas limites, avec les fonctions provisoires de contrats.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import copy

    bien = {"commune": "33063", "type_bien": "maison", "sbati": 90.0, "nb_pieces": 4, "sterr": 400.0, "nblocdep": 1}
    estimation = contrats._estimer_provisoire(bien)
    proches = contrats._comparables_provisoire(bien)

    # Cas 1 : tout est là (estimation dans la fourchette)
    cas = {"1. cas normal": (estimation, proches)}

    # Cas 2 : estimation au-dessus de la fourchette
    haute = copy.deepcopy(estimation)
    haute["prix_m2"], haute["prix_total"] = 4500.0, 4500.0 * 90
    cas["2. estimation au-dessus"] = (haute, proches)

    # Cas 3 : pas de fourchette (pas assez de ventes équivalentes)
    sans_fourchette = copy.deepcopy(proches)
    sans_fourchette["fourchette"] = None
    cas["3. pas de fourchette"] = (estimation, sans_fourchette)

    # Cas 4 : pas d'estimation (sous-bassin non couvert)
    sans_estimation = copy.deepcopy(estimation)
    sans_estimation.update({"prix_m2": None, "prix_total": None, "contexte": None,
                            "message": "Moins de 20 ventes en 2025 dans ce secteur."})
    cas["4. pas d'estimation"] = (sans_estimation, proches)

    # Cas 5 : recherche élargie au bassin de vie, évolution inconnue
    elargie = copy.deepcopy(proches)
    elargie["maille"] = "bassin de vie"
    sans_evolution = copy.deepcopy(estimation)
    sans_evolution["contexte"]["evolution_pct"] = None
    cas["5. bassin de vie, évolution inconnue"] = (sans_evolution, elargie)

    # Cas 6 : aucune vente comparable
    aucune = copy.deepcopy(proches)
    aucune["ventes"] = aucune["ventes"].head(0)
    cas["6. aucune vente"] = (estimation, aucune)

    for nom in cas:
        texte = contrats.verifier_format("expliquer", expliquer(cas[nom][0], cas[nom][1]))
        print("\n" + nom + "\n" + texte)
        assert LIMITE in texte, "la limite doit toujours être rappelée"

    # Même saisie, même résultat
    assert expliquer(estimation, proches) == expliquer(estimation, proches)

    # Contrôle de fidélité : un nombre inventé fait rejeter le texte rédigé
    faits = expliquer(estimation, proches)
    assert texte_fidele("Ce bien est estimé à 315 000 € en 2026.", faits)
    assert not texte_fidele("Ce bien est estimé à 320 000 € en 2026.", faits)

    # Version Groq (si la clé est trouvée) : texte rédigé, ou phrase type en repli
    print("\nClé Groq trouvée :", lire_cle() is not None)
    texte_llm = contrats.verifier_format("expliquer_llm", expliquer_llm(estimation, proches))
    if texte_llm == faits:
        print("expliquer_llm : repli sur la phrase type (pas de clé, erreur, délai ou texte non fidèle).")
    else:
        print("expliquer_llm, texte rédigé par Groq :\n" + texte_llm)
    print("\nContrôles réussis.")
