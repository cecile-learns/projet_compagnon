# =============================================================================
# 05_application/methodologie_outil.py : onglet « Méthodologie et résultats » de la page Outil
# =============================================================================
# Contenu écrit par Cécile (d'abord prévu pour la page Modélisation), intégré à la page Outil (Théophyle).
# Il répond à : « que peut-on illustrer dans le simulateur, et que ne peut-on pas affirmer ? ».
# Contenu repris du comportement réel de l'outil (contrats.py, outil.py) et des résultats du rapport.
#
# Utilisation (pages/outil.py) :
#     from methodologie_outil import onglet_simulateur
#     with onglet_methodo:
#         onglet_simulateur()
# Ne dépend que de streamlit : ni commun_app, ni données.

import streamlit as st


def question(titre):
    st.caption("QUESTION")
    st.subheader(titre)


def carte(titre, texte, legende=None):
    with st.container(border=True, height="stretch"):
        st.markdown("**" + titre + "**")
        st.write(texte)
        if legende:
            st.caption(legende)


def retenir(phrase):
    st.markdown("**À retenir.** " + phrase)


def onglet_simulateur():
    question("Que peut-on illustrer dans le simulateur, et que ne peut-on pas affirmer ?")
    st.write("Le simulateur reprend le réseau H7v4, réappris sur les ventes 2017-2025. L'utilisateur décrit un bien "
             "(commune, type, surface, pièces, terrain, dépendances) ; l'outil affiche l'estimation 2026 en euros et "
             "au m², le contexte du sous-bassin (prix médian 2025, évolution sur un an), la fourchette des biens "
             "équivalents et les ventes les plus proches.")
    gauche, droite = st.columns(2)
    with gauche:
        with st.container(border=True, height="stretch"):
            st.caption("LES RÉSULTATS PERMETTENT D'ILLUSTRER")
            st.markdown("**1. Un bien précis reçoit une estimation propre.**")
            st.write("À sous-bassin égal, deux biens différents (surface, terrain) n'ont pas la même estimation. C'est ce "
                     "que mesure l'évaluation B : +5 à +8 % face à un prix unique pour tout le sous-bassin.")
            st.markdown("**2. Le territoire reste la base.**")
            st.write("Le réseau apprend un écart au prix N-1 du sous-bassin : le prix N-1 est ce qui change le plus le résultat.")
            st.markdown("**3. L'estimation se lit à côté d'un repère de marché.**")
            st.write("La fourchette vient de ventes réelles de biens équivalents sur trois ans (les 75 % du milieu, à partir "
                     "de 30 ventes). Estimation et fourchette peuvent ne pas concorder : l'estimation tient compte du bien.")
    with droite:
        with st.container(border=True, height="stretch"):
            st.caption("LES RÉSULTATS NE PERMETTENT PAS D'AFFIRMER")
            st.markdown("**1. Où ira le marché.**")
            st.write("Sur la médiane d'un territoire, aucun modèle ne bat la naïve. L'évolution sur un an affichée par "
                     "l'outil décrit le passé : ce n'est pas une prévision de tendance.")
            st.markdown("**2. Une estimation précise.**")
            st.write("L'erreur moyenne par vente reste de l'ordre de 700 €/m² au test 2025 (703 et 672 contre 751 et 730 pour "
                     "la naïve). La fourchette décrit le marché local, ce n'est pas une marge d'erreur du modèle.")
            st.markdown("**3. Que l'estimation 2026 sera juste.**")
            st.write("Aucune vente de 2026 n'est jugée. La version de l'outil, réapprise sur 2017-2025, n'a pas de test propre : "
                     "le test 2025 porte sur le réseau appris jusqu'en 2024.")
    retenir("le simulateur illustre ce que le modèle sait faire, estimer un bien dans son territoire, et non anticiper le marché.")

    st.markdown("#### Pour la démonstration")
    colonnes = st.columns(3)
    with colonnes[0]:
        carte("1. Comparer deux biens", "Choisir une commune, puis estimer deux biens de surface ou de terrain différents.")
    with colonnes[1]:
        carte("2. Lire l'estimation face à la fourchette", "Une estimation hors fourchette n'est pas une erreur : elle tient compte du bien.")
    with colonnes[2]:
        carte("3. Lire le contexte comme un repère", "Prix médian 2025 et évolution sur un an décrivent le secteur, ils ne prédisent rien.")
