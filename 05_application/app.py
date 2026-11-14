# =============================================================================
# 05_application/app.py : point d'entrée de l'application Compagnon Immobilier
# =============================================================================
# Lancement (depuis le dossier 05_application, pour que le thème de .streamlit/ soit lu) :  streamlit run app.py
# Ce fichier ne contient que la configuration et la navigation : chaque page est dans
# son propre fichier (pages/), écrit par une seule personne.
#   Méthodologie : Gabrielle · EDA : Christian · Résultats de modélisation : Cécile · Outil : Théophyle

import streamlit as st

# Configuration commune : une seule fois, ici, pour toutes les pages
st.set_page_config(page_title="Compagnon Immobilier", page_icon="🏠", layout="wide")


# Navigation : l'ordre de la liste est l'ordre du menu ; l'outil s'ouvre en premier
pages = [
    st.Page("pages/methodologie.py", title="Méthodologie", icon="🧭"),
    st.Page("pages/eda.py", title="Exploration des données", icon="📊"),
    st.Page("pages/resultats.py", title="Résultats de modélisation", icon="📈"),
    st.Page("pages/outil.py", title="Outil", icon="🏠", default=True),
]
st.navigation(pages).run()
