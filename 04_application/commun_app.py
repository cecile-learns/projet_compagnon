# =============================================================================
# 04_application/commun_app.py : éléments communs aux quatre pages
# =============================================================================
# Comme commun.py pour la modélisation : ce qui doit être identique partout est écrit une seule fois ici.
#   1. Chemins     : le dossier des fichiers lus par l'application
#   2. Chargement  : lecture des fichiers, gardée en mémoire (cache) pour ne pas relire à chaque clic
#   3. Affichage   : style, en-tête de page, messages
# Une page n'écrit pas son propre style : elle appelle entete() et afficher_message().

from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Chemins
# -----------------------------------------------------------------------------
# L'application en ligne n'a pas accès à data/processed : les fichiers qu'elle lit sont copiés,
# réduits aux colonnes utiles, dans 04_application/donnees/.
DOSSIER_APP = Path(__file__).resolve().parent
DONNEES = DOSSIER_APP / "donnees"


# -----------------------------------------------------------------------------
# 2. Chargement
# -----------------------------------------------------------------------------
@st.cache_data
def charger_parquet(nom):
    """Lit donnees/<nom> une seule fois par session de l'application (cache des tableaux)."""
    chemin = DONNEES / nom
    if not chemin.exists():
        raise FileNotFoundError(f"{nom} absent de {DONNEES}")
    return pd.read_parquet(chemin)


@st.cache_data
def charger_csv(nom):
    """Même chose pour les CSV de résultats (page Résultats, page EDA)."""
    chemin = DONNEES / nom
    if not chemin.exists():
        raise FileNotFoundError(f"{nom} absent de {DONNEES}")
    return pd.read_csv(chemin)


# -----------------------------------------------------------------------------
# 3. Affichage
# -----------------------------------------------------------------------------

def entete(titre, sous_titre=None):
    """En-tête identique sur les quatre pages."""
    st.title(titre)
    if sous_titre:
        st.caption(sous_titre)
    st.divider()


def afficher_message(message, niveau="info"):
    """Messages des cas limites, même présentation partout (niveau : info, avertissement, erreur)."""
    if not message:
        return
    {"info": st.info, "avertissement": st.warning, "erreur": st.error}[niveau](message)


def en_construction(responsable):
    """Bandeau provisoire des pages pas encore écrites (retiré par chacun sur sa page)."""
    st.info(f"Page en construction · {responsable}")
