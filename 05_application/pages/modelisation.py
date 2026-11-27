# =============================================================================
# 05_application/pages/modelisation.py : page Modélisation (Cécile)
# =============================================================================
# Page de présentation orale, 3 à 4 minutes : le modèle retenu, H7v4, et l'hypothèse à laquelle il répond.
# Une seule page qui se lit de haut en bas, sans onglets ni clic, en quatre blocs :
#   1. L'hypothèse        H7 : prévoir le prix de chaque vente fait-il mieux que prévoir la médiane d'un territoire ?
#   2. Le modèle          H7v4 : ce qu'il reçoit, ce qu'il apprend, comment il a été jugé
#   3. La fiabilité       le gain sur la naïve (B et C) et la fourchette d'erreur
#   4. L'interprétation   ce que le modèle permet, et ce qu'il ne permet pas
# Le cheminement complet (niveau 1, pont, signaux « quand », variantes de H7, mesures, code) est dans le rapport ;
# les anciens onglets détaillés ne sont pas repris dans l'application.
#
# Données lues : aucune. Les chiffres sont repris du rapport de recherche (tableau 20, test 2025) et de la fiche du
# modèle (models/h7v4/MODEL_CARD.md, fourchette d'erreur issue de h7v4_parametres.json), comparaison avec les autres
# modèles par vente sur 2023 et 2024 (rapport, tableaux 18 et 33 ; notebook H7, 5.3), importance par permutation
# (notebook H7, 6.2). La page s'affiche donc sans le
# dossier data/.
# Règles de l'application : aucune couleur ni police ici (thème commun), seulement les fonctions de commun_app.
# Les graphiques utilisent les couleurs du thème (alt.Color sur une colonne de texte, jamais de code couleur).

import altair as alt
import pandas as pd
import streamlit as st

from commun_app import entete


# =============================================================================
# 0. Petites fonctions (mêmes conventions que les pages EDA et Préparation)
# =============================================================================
def fr(x, d=0):
    """Nombre à la française : espace pour les milliers, virgule pour les décimales."""
    return format(x, ",." + str(d) + "f").replace(",", " ").replace(".", ",")


def signe(x, d=0):
    return ("+" if x >= 0 else "−") + fr(abs(x), d)


def cle(titre, sous_titre=None):
    """Le message à retenir du bloc."""
    with st.container(border=True):
        st.caption("À RETENIR")
        st.markdown("### " + titre)
        if sous_titre:
            st.write(sous_titre)


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
    """Une phrase sous chaque graphique : ce que le public doit en retenir."""
    st.markdown("**À retenir.** " + phrase)


def definition(terme, texte):
    st.caption("**" + terme + "** : " + texte)


def projection(graphique):
    """Tailles de texte lisibles en projection, appliquées une seule fois au graphique complet."""
    return (graphique.configure_axis(labelFontSize=14, titleFontSize=14)
            .configure_legend(labelFontSize=14, titleFontSize=14, orient="top")
            .configure_header(labelFontSize=15, titleFontSize=15)
            .configure_view(stroke=None))


def montrer(graphique):
    st.altair_chart(projection(graphique), width="stretch")


def ordre_lignes(df, colonne):
    """Les valeurs de `colonne` dans l'ordre de la colonne `ordre` (altair trierait sinon par ordre alphabétique)."""
    return list(df.sort_values("ordre")[colonne].drop_duplicates())


def barres_gain(df, y, x="Gain", couleur="Verdict", ordre_couleur=None, titre_x="Gain sur la naïve (%)",
                seuil=3, hauteur=150, facettes=None, decalage=None):
    """Barres horizontales de gains (positif : le modèle fait mieux), avec les repères 0 et seuil de 3 %.

    decalage : colonne qui place plusieurs barres côte à côte pour chaque ligne (ex. le type de bien).
    """
    echelle = alt.Scale(domain=ordre_couleur) if ordre_couleur else alt.Undefined
    bas, haut = min(0, df[x].min()), max(seuil or 0, df[x].max())
    marge = (haut - bas) * 0.2          # place pour les étiquettes de valeur, de chaque côté
    domaine = alt.Scale(domain=[bas - marge, haut + marge], nice=False)
    graduations = [v for v in range(int(bas // 10) * 10 - 10, int(haut // 10) * 10 + 20, 10)
                   if bas - marge <= v <= haut + marge]
    codage = dict(y=alt.Y(y + ":N", sort=ordre_lignes(df, y), title=None, axis=alt.Axis(labelLimit=330)),
                  x=alt.X(x + ":Q", title=titre_x, scale=domaine, axis=alt.Axis(values=graduations)),
                  color=alt.Color(couleur + ":N", scale=echelle, title=None))
    if decalage:
        codage["yOffset"] = alt.YOffset(decalage + ":N")
    barres = alt.Chart().mark_bar().encode(**codage)
    texte_pos = (alt.Chart().mark_text(align="left", dx=4, fontSize=14)
                 .encode(y=codage["y"], x=codage["x"], text="texte:N",
                         **({"yOffset": codage["yOffset"]} if decalage else {}))
                 .transform_filter(alt.datum[x] >= 0))
    texte_neg = (alt.Chart().mark_text(align="right", dx=-4, fontSize=14)
                 .encode(y=codage["y"], x=codage["x"], text="texte:N",
                         **({"yOffset": codage["yOffset"]} if decalage else {}))
                 .transform_filter(alt.datum[x] < 0))
    zero = alt.Chart().mark_rule(color="gray").encode(x=alt.datum(0))
    couches = [barres, texte_pos, texte_neg, zero]
    if seuil is not None:
        couches.append(alt.Chart().mark_rule(color="gray", strokeDash=[5, 4]).encode(x=alt.datum(seuil)))
    return alt.layer(*couches, data=df).properties(height=hauteur)


# =============================================================================
# 1. Contenu repris du rapport et de la fiche du modèle
# =============================================================================
HYPOTHESE_H7 = ("H7 : prévoir le prix de chaque vente plutôt que la médiane du territoire fait mieux, à information "
                "égale.")

# --- Test 2025 (rapport, tableau 20 ; notebook evaluation_2025) : H7v4, gain sur la naïve (%) ---
# (évaluation, type de bien, MAE du modèle, MAE de la naïve, gain %, règle de décision réussie)
# B : chaque vente jugée contre le prix médian N-1 de son sous-bassin ; C : médiane du sous-bassin.
H7V4_2025 = [
    ("B · chaque vente", "Appartements", 703.3, 751.4, 6.4, True),
    ("B · chaque vente", "Maisons", 672.1, 729.8, 7.9, True),
    ("C · médiane du sous-bassin", "Appartements", 185.8, 140.7, -32.1, False),
    ("C · médiane du sous-bassin", "Maisons", 159.0, 103.4, -53.9, False),
]
EVALUATIONS = ["B · chaque vente", "C · médiane du sous-bassin"]

# --- Fourchette d'erreur (fiche du modèle, models/h7v4/MODEL_CARD.md, issue de h7v4_parametres.json) ---
# Erreur d'une vente = prix réel / prévision − 1, ventes de 2025, modèle appris jusqu'en 2024 ;
# 10e et 90e centiles pondérés : 80 % des ventes tombent entre les deux bornes.
# (type de bien, ventes, borne basse %, médiane %, borne haute %)
FOURCHETTE = [
    ("Appartements", 34545, -34.3, 2.4, 49.8),
    ("Maisons", 45393, -42.7, 2.0, 53.2),
]

# Part des sous-bassins où H7v4 fait mieux que la naïve, test 2025, vente par vente (décomposition du rapport, 3.8).
PART_SOUS_BASSINS = 72


# --- Autres références, années jugées 2023 et 2024 (pas le test 2025) : gain sur la naïve (%), chaque vente (B) ---
# (modèle par vente, appartements 2023, maisons 2023, appartements 2024, maisons 2024)
CAS = ["Appartements 2023", "Maisons 2023", "Appartements 2024", "Maisons 2024"]
MODELES_VENTE = [
    ("Régression hédonique (apprise sur l'année N-1)", -3.9, -3.4, -6.1, -4.4),
    ("Elastic Net (bien, territoire, prix N-1)", 2.0, 2.3, 1.8, 2.6),
    ("H7v4", 5.6, 6.7, 5.4, 7.3),
]

# --- Importance par permutation, année jugée 2024 (notebook H7, 6.2) : hausse de l'erreur moyenne par vente (€/m²)
# quand les informations d'une famille sont mélangées entre les ventes. (famille, appartements, maisons)
PERMUTATION = [
    ("Prix de l'an dernier", 1559, 912),
    ("Le bien", 146, 125),
    ("Le lieu", 115, 68),
    ("Le « quand »", 102, 90),
]


# =============================================================================
# 2. Graphiques
# =============================================================================
def graphique_gains():
    lignes = []
    for ordre, (evaluation, type_bien, _m, _n, gain, reussi) in enumerate(H7V4_2025):
        lignes.append({"Évaluation": evaluation, "Type": type_bien, "Gain": gain,
                       "ordre": EVALUATIONS.index(evaluation),
                       "Verdict": "Année réussie" if reussi else "Non réussie", "texte": signe(gain, 1)})
    df = pd.DataFrame(lignes)
    return barres_gain(df, "Évaluation", couleur="Type", ordre_couleur=["Appartements", "Maisons"], hauteur=260,
                       decalage="Type")

def graphique_permutation():
    lignes = []
    for ordre, (famille, appartements, maisons) in enumerate(PERMUTATION):
        for type_bien, valeur in (("Appartements", appartements), ("Maisons", maisons)):
            lignes.append({"Famille": famille, "Type": type_bien, "Hausse": valeur, "ordre": ordre,
                           "texte": "+" + fr(valeur)})
    df = pd.DataFrame(lignes)
    codage = dict(y=alt.Y("Famille:N", sort=ordre_lignes(df, "Famille"), title=None, axis=alt.Axis(labelLimit=330)),
                  x=alt.X("Hausse:Q", title="Hausse de l'erreur par vente (€/m²)",
                          scale=alt.Scale(domain=[0, df["Hausse"].max() * 1.15]),
                          axis=alt.Axis(tickCount=5, labelExpr="replace(format(datum.value, ',d'), ',', ' ')")),
                  yOffset=alt.YOffset("Type:N"))
    barres = alt.Chart().mark_bar().encode(color=alt.Color("Type:N", title=None), **codage)
    textes = alt.Chart().mark_text(align="left", dx=4, fontSize=14).encode(text="texte:N", **codage)
    return alt.layer(barres, textes, data=df).properties(height=260)


# =============================================================================
# 3. Page
# =============================================================================
entete("Modélisation", "Le modèle retenu, H7v4 : l'hypothèse à laquelle il répond, ce qu'il fait, sa fiabilité et "
                       "ce qu'on peut en conclure.")

# --- 1. L'hypothèse ---------------------------------------------------------
question("Prévoir chaque vente fait-il mieux que prévoir la médiane d'un territoire ?")
st.info(HYPOTHESE_H7)
colonnes = st.columns(2)
with colonnes[0]:
    carte("Ce que nous attendions",
          "Oui sur le prix d'un bien : le modèle connaît le bien (surface, pièces, terrain), ce que la médiane d'un "
          "territoire ignore.")
with colonnes[1]:
    carte("Ce que nous n'attendions pas",
          "Qu'il anticipe mieux le retournement du marché : il lui manquerait pour cela un signal d'avance sur la "
          "phase du marché.")

# --- 2. Le modèle -----------------------------------------------------------
st.divider()
question("Quel est le modèle retenu ?")
st.write("**H7v4** est un réseau de neurones qui estime le prix au m² d'un logement ancien vendu seul. Il reçoit trois "
         "familles d'informations et apprend l'écart au prix de l'année précédente.")
colonnes = st.columns(3)
with colonnes[0]:
    carte("Le bien", "Type, surface, nombre de pièces, terrain, dépendances.")
with colonnes[1]:
    carte("Le territoire", "Le sous-bassin de vie, son prix de l'année précédente, son parc de logements, la vacance et la "
                            "densité.")
with colonnes[2]:
    carte("Le « quand »", "Volume de ventes, élan des prix, capacité d'achat (le taux d'effort), prix et volume de "
                              "la fin de l'année précédente.")
chiffres = st.columns(3)
chiffres[0].metric("Variables d'entrée", "35", "bien, territoire, « quand »", delta_color="off")
chiffres[1].metric("Ventes d'apprentissage", "773 106", "2017 à 2025", delta_color="off")
chiffres[2].metric("Sous-bassins connus", "1 810", "France métropolitaine hors Alsace-Moselle", delta_color="off")
st.caption("Pour le test, le modèle est appris jusqu'en 2024 puis jugé sur 2025, avec un seul "
           "candidat déclaré avant. Le modèle livré est réappris jusqu'en 2025.")

# --- 3. La fiabilité --------------------------------------------------------
st.divider()
question("Peut-on s'y fier ?")
with st.container(border=True):
    st.caption("LA RÉFÉRENCE")
    st.markdown("### La prévision naïve : le prix de l'an dernier reconduit")
    st.write("Elle prévoit que le prix de l'année N sera celui de l'année N-1. Un modèle n'a d'intérêt que s'il fait "
             "mieux. Pour juger chaque vente (B), la naïve donne à toutes les ventes d'un sous-bassin son prix médian "
             "de l'année précédente ; pour juger la médiane du sous-bassin (C), elle reprend ce prix médian N-1.")
chiffres = st.columns(3)
chiffres[0].metric("Gain sur la naïve, appartements", signe(6.4, 1) + " %", "chaque vente, test 2025", delta_color="off")
chiffres[1].metric("Gain sur la naïve, maisons", signe(7.9, 1) + " %", "chaque vente, test 2025", delta_color="off")
chiffres[2].metric("Sous-bassins où le modèle gagne", str(PART_SOUS_BASSINS) + " %", "test 2025, chaque vente",
                   delta_color="off")
definition("Gain sur la naïve", "positif, le modèle fait mieux que reconduire le prix N-1. Trait pointillé : seuil de 3 %. "
                                "B juge chaque vente ; C juge la médiane du sous-bassin, comme les modèles agrégés.")
montrer(graphique_gains())
retenir("sur chaque vente, le modèle fait mieux que la naïve de 6 à 8 % ; sur la médiane d'un territoire, il fait "
        "moins bien.")

st.markdown("#### Et face à d'autres modèles par vente ?")
st.table(pd.DataFrame([[signe(v, 1) + " %" for v in gains] for _nom, *gains in MODELES_VENTE],
                      index=[ligne[0] for ligne in MODELES_VENTE], columns=CAS))
retenir("H7v4 gagne 3,6 à 4,7 points de plus que l'Elastic Net ; une régression hédonique, qui connaît le bien, fait "
        "moins bien que la naïve.")
st.caption("Gain sur la naïve, chaque vente, années jugées 2023 et 2024 (ces deux références n'ont pas été jugées sur "
           "2025). L'Elastic Net ne reçoit pas les indicateurs du « quand » : l'écart avec H7v4 vient à la fois du "
           "réseau et de ces indicateurs.")

st.markdown("#### L'erreur à attendre sur une vente")
colonnes = st.columns(2)
for colonne, (type_bien, _n, bas, mediane, haut) in zip(colonnes, FOURCHETTE):
    colonne.metric(type_bien + " : 80 % des ventes", signe(bas) + " % à " + signe(haut) + " %",
                   "médiane " + signe(mediane, 1) + " %", delta_color="off")
retenir("pour 80 % des ventes de 2025, l'erreur reste entre −34 % et +50 % pour un appartement, entre −43 % et "
        "+53 % pour une maison ; la médiane est proche de 0.")
st.caption("Erreur d'une vente = prix réel / prévision − 1, mesurée sur les ventes de 2025 avec le modèle appris jusqu'en "
           "2024 (10e et 90e centiles pondérés).")

# --- 4. L'interprétation ----------------------------------------------------
st.divider()
question("Que faut-il en conclure ?")
cle("Le modèle estime un bien, il ne prévoit pas le marché",
    "L'hypothèse est confirmée sur la vente, pas sur le territoire : le gain est réel mais modéré, et il ne dit "
    "rien de l'évolution du marché.")
st.markdown("#### De quoi le modèle a-t-il besoin pour être juste ?")
montrer(graphique_permutation())
retenir("le prix de l'an dernier donne le niveau, le bien fait l'ajustement principal.")
st.caption("Importance par permutation, année jugée 2024 : on mélange une famille d'informations entre les ventes et on "
           "mesure de combien l'erreur augmente. Les informations liées se partagent l'importance : le lieu est déjà en "
           "partie porté par le prix de l'an dernier.")

colonnes = st.columns(3)
with colonnes[0]:
    carte("Le territoire dit où, pas quand",
          "Il explique les écarts de prix entre territoires, mais le modèle construit sur les seuls territoires reste "
          "moins bon que la naïve. Le seul signal du « quand » qui aide nettement est le taux d'effort, qui contient le taux des "
          "crédits.")
with colonnes[1]:
    carte("Un gain réel, une marge large",
          "Gagner 6 à 8 % ne rend pas l'estimation exacte : une vente sur cinq sort de la fourchette de −34 % à +50 % "
          "(appartements) ou de −43 % à +53 % (maisons). L'outil donne une estimation et des ventes comparables, pas un prix.")
with colonnes[2]:
    carte("Une réserve à connaître",
          "La naïve ignore le bien. Un modèle linéaire qui le connaît gagne déjà environ 2 % : une partie du gain de "
          "H7v4 vient de l'information, le reste de la forme du réseau et des indicateurs du « quand » (2023 et 2024). "
          "Le test porte sur 2025 seule, un marché presque plat.")