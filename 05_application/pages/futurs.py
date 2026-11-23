# =============================================================================
# 05_application/pages/futur.py : Développements futurs possibles (Gabrielle)
# =============================================================================
# Ce que fait la page : raconte le chemin des données, des sources brutes à la table de modélisation.
# Cinq onglets, chacun suit le même parcours que la page EDA : un message clé (« À retenir »), puis des
# questions dans l'ordre où le lecteur se les pose, puis le détail dans un menu repliable.
#   1. Méthodologie         : répartition du travail, gabarits signés, fusion en un fichier, organisation du dépôt
#   2. Téléchargement       : 00_telechargement/telechargement.py et le Drive du projet
#   3. Préparation          : 01_preparation/preparation.py, bloc par bloc, avec des mini-simulateurs
#   4. Feature engineering  : 01_preparation/feature_engineering.py, sections 1 à 6 bis
#   5. Fusion               : sections 7 et 8 (table finale), puis maille et seuil (repris de la page EDA)
#
# Données lues : donnees/eda_seuil.parquet (onglet Fusion, produit par 04_outils/preparer_eda_app.py).
# Tout le reste est du texte écrit ici : la page s'affiche même sans données.
# Règles de l'application : aucune couleur ni police ici (thème commun), seulement les fonctions de commun_app.

import altair as alt
import pandas as pd
import streamlit as st

from commun_app import DONNEES, afficher_message, entete

LIEN_DRIVE = "https://drive.google.com/drive/folders/1bVlZgMTxjtoWMveHXJEsxY6LPRo6U6nZ"


# -----------------------------------------------------------------------------
# 0. Petites fonctions (mêmes conventions que la page EDA)
# -----------------------------------------------------------------------------
@st.cache_data
def _lire_parquet(nom, date_fichier):
    # date_fichier fait partie de la clé de cache : un parquet régénéré est relu sans redémarrer le serveur
    return pd.read_parquet(DONNEES / nom)


def charger_parquet(nom):
    chemin = DONNEES / nom
    if not chemin.exists():
        raise FileNotFoundError(nom + " absent de " + str(DONNEES))
    return _lire_parquet(nom, chemin.stat().st_mtime_ns)


def fr(x, d=0):
    """Nombre à la française : espace pour les milliers, virgule pour les décimales."""
    return format(x, ",." + str(d) + "f").replace(",", " ").replace(".", ",")


def signe(x, d=0):
    return ("+" if x >= 0 else "−") + fr(abs(x), d)


def cle(titre, sous_titre=None):
    """Le message à retenir de l'onglet."""
    with st.container(border=True):
        st.caption("À RETENIR")
        st.markdown("### " + titre)
        if sous_titre:
            st.write(sous_titre)


def question(numero, titre):
    st.caption("QUESTION " + str(numero))
    st.subheader(titre)


def carte(titre, texte, legende=None):
    with st.container(border=True):
        st.markdown("**" + titre + "**")
        st.write(texte)
        if legende:
            st.caption(legende)


entete("Développements futurs possibles",
       "Idées et pistes pour la suite du projet")

onglets = st.tabs(["Client : agence immobilière", "Client :  "])


# =============================================================================
# 1. MÉTHODOLOGIE
# =============================================================================
