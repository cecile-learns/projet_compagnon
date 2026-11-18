# =============================================================================
# 05_application/pages/modelisation.py : page Modélisation (Cécile)
# =============================================================================
# Ce que fait la page : un choix dans la colonne de gauche, « Méthodologie » ou « Résultats ».
# Chaque onglet répond à une question que le public se pose, dans l'ordre de l'oral.
#
#   Méthodologie : comment les modèles ont été construits, puis jugés
#     1. Cible et référence      que prédit-on, et par rapport à quoi ? (la prévision naïve)
#     2. Deux niveaux            comment la démarche a progressé : le « où », la transition, le « quand »
#     3. Fenêtres et test 2025   comment éviter d'apprendre sur le futur
#     4. Mesures et décision     comment lire un résultat, quand un modèle est retenu
#     5. Modèle par vente        ce que comparent les évaluations B et C
#     6. Construction du code    code commun, gabarits, tests pytest, parcours court et parcours long
#   Résultats : les comparaisons d'abord, la réponse en conclusion
#     1. Territoire              niveau 1 : que gagne-t-on avec le « où » ?
#     2. Biais                   pourquoi les modèles se trompent, ce que change la période longue
#     3. Signaux « quand »       H5 (volume, élan) et H6 (capacité d'achat)
#     4. Vente par vente         H7 : le réseau de neurones et ses deux évaluations
#     5. Test 2025               l'année tenue en réserve
#     6. Réponse                 la réponse à la question de recherche et son niveau de certitude
#   L'onglet « Simulateur » est mis de côté : voir 05_application/en_attente/onglet_simulateur_modelisation.py
#
# Données lues : aucune. L'affichage est indépendant de l'entraînement des modèles : tous les chiffres sont
# repris du rapport de recherche (rendu 2, tableaux cités en commentaire) et recoupés avec les sorties des notebooks
# (H3_globale, H4_pont, H5H6, H6, H7, evaluation_2025). La page s'affiche donc sans le dossier data/.
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
    """Le message à retenir de l'onglet."""
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
    """Une phrase sous chaque graphique ou tableau clé : ce que le public doit en retenir."""
    st.markdown("**À retenir.** " + phrase)


def definition(terme, texte):
    """Une mesure est définie à l'endroit où elle sert à lire un résultat."""
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

    facettes : dict {"row": colonne, "column": colonne} pour un graphique en panneaux.
    decalage : colonne qui place plusieurs barres côte à côte pour chaque ligne (ex. le type de bien).
    """
    echelle = alt.Scale(domain=ordre_couleur) if ordre_couleur else alt.Undefined
    bas, haut = min(0, df[x].min()), max(seuil or 0, df[x].max())
    marge = (haut - bas) * 0.2          # place pour les étiquettes de valeur, de chaque côté
    domaine = alt.Scale(domain=[bas - marge, haut + marge], nice=False)
    codage = dict(y=alt.Y(y + ":N", sort=ordre_lignes(df, y), title=None, axis=alt.Axis(labelLimit=330)),
                  x=alt.X(x + ":Q", title=titre_x, scale=domaine),
                  color=alt.Color(couleur + ":N", scale=echelle, title=None))
    if decalage:
        codage["yOffset"] = alt.YOffset(decalage + ":N")
    barres = alt.Chart().mark_bar().encode(**codage)
    texte_pos = (alt.Chart().mark_text(align="left", dx=4, fontSize=13)
                 .encode(y=codage["y"], x=codage["x"], text="texte:N",
                         **({"yOffset": codage["yOffset"]} if decalage else {}))
                 .transform_filter(alt.datum[x] >= 0))
    texte_neg = (alt.Chart().mark_text(align="right", dx=-4, fontSize=13)
                 .encode(y=codage["y"], x=codage["x"], text="texte:N",
                         **({"yOffset": codage["yOffset"]} if decalage else {}))
                 .transform_filter(alt.datum[x] < 0))
    zero = alt.Chart().mark_rule(color="gray").encode(x=alt.datum(0))
    couches = [barres, texte_pos, texte_neg, zero]
    if seuil is not None:
        couches.append(alt.Chart().mark_rule(color="gray", strokeDash=[5, 4]).encode(x=alt.datum(seuil)))
    graphique = alt.layer(*couches).properties(height=hauteur)
    if facettes:
        return graphique.facet(data=df, **facettes).resolve_scale(x="shared")
    return alt.layer(*couches, data=df).properties(height=hauteur)


# =============================================================================
# 1. Contenu repris du rapport de recherche
# =============================================================================
QUESTION = ("Les caractéristiques d'un territoire et l'historique de ses ventes permettent-ils de prévoir le prix "
            "médian au m² de ses logements anciens l'année suivante, mieux qu'en reconduisant le prix de l'année "
            "précédente ?")

# --- Fenêtres temporelles (rapport 4.2, 4.5, 6.1 ; notebooks H3_globale section 8, scripts de 4_evaluation_2025) ---
# (fenêtre, début d'apprentissage, fin d'apprentissage, année jugée, rôle de l'année jugée)
FENETRES = [
    ("Niveau 1 · juger 2023", 2020, 2022, 2023, "Année jugée"),
    ("Niveau 1 · juger 2024", 2020, 2023, 2024, "Année jugée"),
    ("Niveau 1 · test final", 2020, 2024, 2025, "Test 2025 (1 ouverture)"),
    ("Niveau 2 · juger 2023", 2017, 2022, 2023, "Année jugée"),
    ("Niveau 2 · juger 2024", 2017, 2023, 2024, "Année jugée"),
    ("Niveau 2 · test final", 2017, 2024, 2025, "Test 2025 (1 ouverture)"),
]
ROLES = ["Apprentissage", "Année jugée", "Test 2025 (1 ouverture)"]

# --- Lire un intervalle : trois cas réels du test 2025 (notebook evaluation_2025) ---
# (modèle, type de bien, borne basse, borne haute) : écart de MAE avec la naïve en €/m², IC 95 % du bootstrap
INTERVALLES_LECTURE = [
    ("H7v4, chaque vente (maisons)", -64.75, -51.19, "Mieux que la naïve"),
    ("Hiérarchique (appartements)", -0.34, 12.60, "Indécis : contient 0"),
    ("Hiérarchique (maisons)", 3.04, 9.10, "Moins bien que la naïve"),
]
VERDICTS_INTERVALLE = ["Mieux que la naïve", "Indécis : contient 0", "Moins bien que la naïve"]

# --- Synthèse (rapport, tableau 21) : gain sur la naïve (%), appartements / maisons ---
SYNTHESE = [
    ("Niveau 1 · Elastic Net complet", {2023: (-1.1, -41.5), 2024: (-32.9, -52.6), 2025: (-14.7, -9.6)}),
    ("Modèle pont (H4)", {2023: (4.6, -16.7), 2024: (-26.9, -39.5)}),
    ("Pont + taux d'effort (H6)", {2023: (-0.3, -10.6), 2024: (-10.0, 2.8)}),
    ("Hiérarchique + taux d'effort (H6)", {2023: (-2.0, -10.4), 2024: (2.0, 9.5), 2025: (-4.1, -5.6)}),
    ("Réseau H7v4 · médiane du sous-bassin (C)", {2023: (-20.0, -65.0), 2024: (-32.0, -22.0), 2025: (-32.1, -53.9)}),
    ("Réseau H7v4 · chaque vente (B)", {2023: (5.6, 6.7), 2024: (5.4, 7.3), 2025: (6.4, 7.9)}),
]
# Années réussies selon la règle de décision (gain d'au moins 3 %, IC sous 0, rang-bisériale négative) :
# le réseau en évaluation B (toutes les années) et, seul succès sur la médiane, le hiérarchique pour les maisons en 2024.
REUSSIES = {("Hiérarchique + taux d'effort (H6)", "Maisons", 2024)}
REUSSIES |= {("Réseau H7v4 · chaque vente (B)", t, a) for t in ("Appartements", "Maisons") for a in (2023, 2024, 2025)}

# --- Niveau 1 (rapport, tableau 9) : MAE en €/m², naïve et Elastic Net du modèle complet ---
# (type de bien, année, naïve, Elastic Net)
NIVEAU1_MAE = [
    ("Appartements", "2023", 166.3, 168.2), ("Appartements", "2024", 148.0, 196.7), ("Appartements", "Test 2025", 141.8, 162.7),
    ("Maisons", "2023", 111.7, 158.0), ("Maisons", "2024", 129.2, 197.2), ("Maisons", "Test 2025", 103.5, 113.4),
]

# --- Biais 2024 en €/m² (prévision moins prix réel ; positif : le modèle surestime) ---
# Naïve, niveau 1 et pont : rapport, tableau 10 (lignes de H4). Pont + taux d'effort : tableau 19 (lignes de H5-H6).
BIAIS_2024 = [
    ("Naïve", 40.7, 71.1),
    ("Niveau 1", 149.5, 173.6),
    ("Pont", 136.0, 150.6),
    ("Pont + effort", 92.0, 48.0),
]

# --- Signaux « quand » (rapport, tableaux 12 et 14) : gain sur le modèle pont (%), 2023 / 2024 ---
# (signal, famille, (appartements 2023, 2024), (maisons 2023, 2024))
SIGNAUX = [
    ("Nombre de ventes N-1", "H5 · volume et élan", (-6.1, 2.3), (-5.4, 3.5)),
    ("Rotation du parc (même type)", "H5 · volume et élan", (-2.5, 4.3), (-8.6, 12.1)),
    ("Élan du prix N-2 → N-1", "H5 · volume et élan", (0.4, -3.9), (11.4, -14.0)),
    ("Les trois réunis", "H5 · volume et élan", (-1.5, 0.5), (3.2, 3.5)),
    ("Taux d'effort", "H6 · capacité d'achat", (-5.1, 13.3), (5.3, 30.3)),
]

# --- H7 (rapport, tableau 17, réseau optimisé) : gain sur la naïve (%), 2023 / 2024 ---
# (version, B appartements, C appartements, B maisons, C maisons), chaque couple = (2023, 2024)
H7 = [
    ("H7v1 · sous-bassin + bien", (-9.6, -7.9), (-174, -184), (-2.2, 1.6), (-207, -98)),
    ("H7v2a · + territoire", (-9.2, -6.3), (-175, -174), (-1.8, 3.1), (-213, -94)),
    ("H7v3 · + prix N-1", (4.6, 5.0), (-28, -60), (6.3, 4.8), (-75, -97)),
    ("H7vH5 · + volume et élan", (5.1, 5.1), (-16, -47), (6.3, 5.7), (-80, -65)),
    ("H7vH6 · + capacité d'achat", (5.7, 5.1), (-20, -38), (7.0, 6.6), (-42, -32)),
    ("H7vT · + trimestres", (4.8, 5.1), (-28, -37), (6.6, 5.2), (-68, -80)),
    ("H7v4 · tout le « quand »", (5.6, 5.4), (-20, -32), (6.7, 7.3), (-65, -22)),
]

# --- Optimisation du réseau (rapport, tableau 16) : quart le plus cher des appartements, validation 2022 ---
# erreur moyenne par vente en €/m² ; la naïve vaut 1 913
OPTIMISATION = [("Naïve", 1913), ("Réglages par défaut", 2089), ("1. Apprendre l'évolution", 1851),
                ("2. Perte absolue", 1867), ("3. Hyperparamètres", 1857)]

# --- Test 2025 (rapport, tableau 20 ; notebook evaluation_2025 ; H3_globale section 8) ---
# (candidat, évaluation, type, MAE modèle, MAE naïve, gain %, IC bas, IC haut, biais, réussi)
TEST_2025 = [
    ("Niveau 1 · Elastic Net complet", "Médiane du sous-bassin", "Appartements", 162.7, 141.8, -14.7, 6.0, 43.0, "+41 €/m²", False),
    ("Niveau 1 · Elastic Net complet", "Médiane du sous-bassin", "Maisons", 113.4, 103.5, -9.6, 6.0, 15.0, "+37 €/m²", False),
    ("Hiérarchique + taux d'effort", "Médiane du sous-bassin", "Appartements", 147.6, 141.8, -4.1, -0.34, 12.60, "−24 €/m²", False),
    ("Hiérarchique + taux d'effort", "Médiane du sous-bassin", "Maisons", 109.2, 103.5, -5.6, 3.04, 9.10, "−21 €/m²", False),
    ("H7v4 · chaque vente (B)", "Chaque vente", "Appartements", 703.3, 751.4, 6.4, -56.77, -39.68, "+0,7 %", True),
    ("H7v4 · chaque vente (B)", "Chaque vente", "Maisons", 672.1, 729.8, 7.9, -64.75, -51.19, "+3,9 %", True),
    ("H7v4 · médiane du sous-bassin (C)", "Médiane du sous-bassin", "Appartements", 185.8, 140.7, -32.1, 31.10, 59.12, "−14 €/m²", False),
    ("H7v4 · médiane du sous-bassin (C)", "Médiane du sous-bassin", "Maisons", 159.0, 103.4, -53.9, 48.74, 62.51, "−1 €/m²", False),
]
ORDRE_CANDIDATS = ["Niveau 1 · Elastic Net complet", "Hiérarchique + taux d'effort",
                   "H7v4 · médiane du sous-bassin (C)", "H7v4 · chaque vente (B)"]


# =============================================================================
# 2. Tableaux et graphiques
# =============================================================================
def graphique_fenetres():
    lignes = []
    for ordre, (nom, debut, fin, cible, role) in enumerate(FENETRES):
        lignes.append({"fenetre": nom, "debut": debut - 0.5, "fin": fin + 0.5, "role": "Apprentissage",
                       "etiquette": str(debut) + "-" + str(fin), "ordre": ordre})
        lignes.append({"fenetre": nom, "debut": cible - 0.5, "fin": cible + 0.5, "role": role,
                       "etiquette": str(cible), "ordre": ordre})
    df = pd.DataFrame(lignes)
    df["milieu"] = (df["debut"] + df["fin"]) / 2
    x = alt.X("debut:Q", scale=alt.Scale(domain=[2016.5, 2025.5], nice=False),
              axis=alt.Axis(values=list(range(2017, 2026)), format="d", title="Année des prix", labelFontSize=14))
    y = alt.Y("fenetre:N", sort=ordre_lignes(df, "fenetre"), title=None,
              axis=alt.Axis(labelLimit=260, labelFontSize=14))
    rectangles = (alt.Chart(df).mark_rect(stroke="white", strokeWidth=2)
                  .encode(x=x, x2="fin:Q", y=y,
                          color=alt.Color("role:N", scale=alt.Scale(domain=ROLES), title=None)))
    textes = (alt.Chart(df).mark_text(fontSize=13, color="white")
              .encode(x=alt.X("milieu:Q", scale=alt.Scale(domain=[2016.5, 2025.5], nice=False)), y=y, text="etiquette:N"))
    return (rectangles + textes).properties(height=250)


def tableau_synthese():
    lignes = []
    for ordre, (modele, annees) in enumerate(SYNTHESE):
        for annee, gains in annees.items():
            for type_bien, gain in zip(("Appartements", "Maisons"), gains):
                reussi = (modele, type_bien, annee) in REUSSIES
                etiquette = "2025 (test)" if annee == 2025 else str(annee)
                lignes.append({"Modèle": modele, "Type": type_bien, "Année": etiquette, "Gain": gain,
                               "Verdict": "Année réussie" if reussi else "Non réussie", "ordre": ordre,
                               "texte": signe(gain, 1 if abs(gain) < 100 else 0)})
    return pd.DataFrame(lignes)


def graphique_biais():
    lignes = []
    for ordre, (etape, appart, maison) in enumerate(BIAIS_2024):
        for type_bien, biais in (("Appartements", appart), ("Maisons", maison)):
            lignes.append({"Étape": etape, "Type": type_bien, "Biais": biais, "ordre": ordre, "texte": signe(biais)})
    df = pd.DataFrame(lignes)
    x = alt.X("Étape:N", sort=ordre_lignes(df, "Étape"), title=None,
              axis=alt.Axis(labelAngle=0, labelLimit=190, labelFontSize=14))
    barres = alt.Chart(df).mark_bar().encode(
        x=x, xOffset=alt.XOffset("Type:N"), y=alt.Y("Biais:Q", title="Biais 2024 (€/m²)"),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Appartements", "Maisons"]), title=None))
    textes = alt.Chart(df).mark_text(dy=-8, fontSize=14).encode(
        x=x, xOffset=alt.XOffset("Type:N"), y="Biais:Q", text="texte:N")
    return (barres + textes).properties(height=330)


def graphique_niveau1(type_bien):
    lignes = []
    for t, annee, naive, modele in NIVEAU1_MAE:
        if t == type_bien:
            lignes.append({"Année jugée": annee, "Modèle": "Prévision naïve", "MAE": naive, "texte": fr(naive)})
            lignes.append({"Année jugée": annee, "Modèle": "Elastic Net complet", "MAE": modele, "texte": fr(modele)})
    df = pd.DataFrame(lignes)
    x = alt.X("Année jugée:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14))
    barres = alt.Chart(df).mark_bar().encode(
        x=x, xOffset=alt.XOffset("Modèle:N"), y=alt.Y("MAE:Q", title="Erreur moyenne, MAE (€/m²)"),
        color=alt.Color("Modèle:N", scale=alt.Scale(domain=["Prévision naïve", "Elastic Net complet"]), title=None))
    textes = alt.Chart(df).mark_text(dy=-8, fontSize=14).encode(
        x=x, xOffset=alt.XOffset("Modèle:N"), y="MAE:Q", text="texte:N")
    return (barres + textes).properties(height=300)


def tableau_signaux(annee):
    lignes = []
    indice = 0 if annee == 2023 else 1
    for ordre, (signal, famille, appart, maison) in enumerate(SIGNAUX):
        for type_bien, valeurs in (("Appartements", appart), ("Maisons", maison)):
            gain = valeurs[indice]
            lignes.append({"Signal": signal, "Famille": famille, "Type": type_bien, "Gain": gain,
                           "ordre": ordre, "texte": signe(gain, 1)})
    return pd.DataFrame(lignes)


def tableau_h7(type_bien):
    lignes = []
    for ordre, (version, b_app, c_app, b_mai, c_mai) in enumerate(H7):
        b, c = (b_app, c_app) if type_bien == "Appartements" else (b_mai, c_mai)
        for evaluation, valeurs in (("B · chaque vente", b), ("C · médiane du sous-bassin", c)):
            for annee, gain in zip(("2023", "2024"), valeurs):
                lignes.append({"Version": version, "Évaluation": evaluation, "Année": annee, "Gain": gain,
                               "ordre": ordre, "texte": signe(gain, 1 if abs(gain) < 20 else 0)})
    return pd.DataFrame(lignes)


def graphique_h7(type_bien):
    df = tableau_h7(type_bien)
    y = alt.Y("Version:N", sort=ordre_lignes(df, "Version"), title=None,
              axis=alt.Axis(labelLimit=260))
    codage = dict(y=y, yOffset=alt.YOffset("Année:N"), x=alt.X("Gain:Q", title="Gain sur la naïve (%)"))
    barres = alt.Chart().mark_bar().encode(color=alt.Color("Année:N", title=None), **codage)
    texte_pos = (alt.Chart().mark_text(align="left", dx=4, fontSize=12).encode(text="texte:N", **codage)
                 .transform_filter(alt.datum.Gain >= 0))
    texte_neg = (alt.Chart().mark_text(align="right", dx=-4, fontSize=12).encode(text="texte:N", **codage)
                 .transform_filter(alt.datum.Gain < 0))
    zero = alt.Chart().mark_rule(color="gray").encode(x=alt.datum(0))
    graphique = alt.layer(barres, texte_pos, texte_neg, zero).properties(height=330, width=330)
    return graphique.facet(column=alt.Column("Évaluation:N", title=None,
                                             sort=["B · chaque vente", "C · médiane du sous-bassin"]),
                           data=df).resolve_scale(x="independent")


def graphique_optimisation():
    """Points (et non barres) : l'axe ne part pas de zéro, le trait pointillé est la naïve."""
    df = pd.DataFrame([{"Étape": e, "Erreur": v, "ordre": i, "texte": fr(v),
                        "Verdict": "Moins bien que la naïve" if v > 1913 else ("Naïve" if e == "Naïve" else "Mieux que la naïve")}
                       for i, (e, v) in enumerate(OPTIMISATION)])
    y = alt.Y("Étape:N", sort=ordre_lignes(df, "Étape"), title=None,
              axis=alt.Axis(labelLimit=300, labelFontSize=14))
    x = alt.X("Erreur:Q", title="Erreur moyenne par vente (€/m²)", scale=alt.Scale(domain=[1780, 2160], nice=False))
    points = alt.Chart().mark_circle(size=320).encode(
        y=y, x=x, color=alt.Color("Verdict:N", scale=alt.Scale(
            domain=["Naïve", "Moins bien que la naïve", "Mieux que la naïve"]), title=None))
    textes = alt.Chart().mark_text(align="left", dx=14, fontSize=14).encode(y=y, x=x, text="texte:N")
    repere = alt.Chart().mark_rule(color="gray", strokeDash=[5, 4]).encode(x=alt.datum(1913))
    return alt.layer(points, textes, repere, data=df).properties(height=220)


def graphique_intervalles(lignes):
    """Un intervalle par ligne, un repère à 0 : entièrement sous 0, le modèle fait mieux ; au-dessus, moins bien."""
    df = pd.DataFrame([{"Modèle": m, "bas": b, "haut": h, "Lecture": v, "ordre": i}
                       for i, (m, b, h, v) in enumerate(lignes)])
    y = alt.Y("Modèle:N", sort=ordre_lignes(df, "Modèle"), title=None,
              axis=alt.Axis(labelLimit=330))
    barres = alt.Chart(df).mark_bar(height=16).encode(
        y=y, x=alt.X("bas:Q", title="Écart de MAE avec la naïve (€/m²), intervalle à 95 %"), x2="haut:Q",
        color=alt.Color("Lecture:N", scale=alt.Scale(domain=VERDICTS_INTERVALLE), title=None))
    zero = alt.Chart().mark_rule(color="gray", strokeDash=[5, 4], size=2).encode(x=alt.datum(0))
    return alt.layer(barres, zero, data=df).properties(height=40 * len(lignes) + 40)


def graphique_test_2025():
    lignes = []
    for candidat, evaluation, type_bien, _m, _n, gain, _b, _h, _biais, reussi in TEST_2025:
        lignes.append({"Candidat": candidat, "Type": type_bien, "Gain": gain, "ordre": ORDRE_CANDIDATS.index(candidat),
                       "Verdict": "Année réussie" if reussi else "Non réussie", "texte": signe(gain, 1)})
    df = pd.DataFrame(lignes)
    return barres_gain(df, "Candidat", ordre_couleur=["Année réussie", "Non réussie"], hauteur=250,
                       decalage="Type")


# =============================================================================
# 3. Méthodologie
# =============================================================================
def onglet_cible_reference():
    question("Que cherche-t-on à prédire, et par rapport à quoi ?")
    st.info(QUESTION)

    st.markdown("#### Le point de départ : la table préparée")
    colonnes = st.columns(4)
    with colonnes[0]:
        carte("La cible", "Le prix médian au m² de l'année N, pour les logements anciens vendus seuls (maison ou appartement).")
    with colonnes[1]:
        carte("L'unité", "Une ligne par sous-bassin de vie, par année et par type de bien. Appartements et maisons sont modélisés à part.")
    with colonnes[2]:
        carte("Les variables", "Connues en fin d'année N-1 : le prix N-1, des photos fixes du territoire, puis des indicateurs annuels.")
    with colonnes[3]:
        carte("Le seuil", "Une médiane n'est gardée qu'avec au moins 20 ventes en N et en N-1.")
    chiffres = st.columns(4)
    chiffres[0].metric("Ventes retenues", "9,68 M", "59,1 % des mutations", delta_color="off")
    chiffres[1].metric("Lignes de la table", "28 450", "prix de 2015 à 2025", delta_color="off")
    chiffres[2].metric("Appartements / maisons", "8 891 / 19 559", "lignes", delta_color="off")
    chiffres[3].metric("Territoires prédits en 2024", "823 / 1 777", "appartements / maisons", delta_color="off")

    st.markdown("#### La référence : la prévision naïve")
    gauche, droite = st.columns([3, 2])
    with gauche:
        with st.container(border=True):
            st.markdown("### Prix de l'année N = prix de l'année N-1")
            st.write("La prévision naïve reconduit le prix de l'an dernier. Elle sert de référence à tous les modèles : "
                     "**un modèle n'a d'intérêt que s'il fait mieux.**")
        st.write("Nous prévoyons le prix lui-même, pas son évolution : l'évolution s'en déduit par différence avec le "
                 "prix N-1, et la référence reste simple. Chaque résultat de la page se lit donc comme un gain ou "
                 "une perte face à cette référence.")
    with droite:
        st.metric("Erreur de la naïve en 2024, appartements", "5,6 %", help="Rapport, tableau 3 : erreur de la naïve au sous-bassin de vie.")
        st.metric("Erreur de la naïve en 2024, maisons", "5,9 %")
        st.caption("La référence n'est pas faible : elle se trompe déjà de 5 à 6 % du prix.")
    retenir("tout est jugé contre le prix de l'an dernier reconduit ; la table préparée donne un prix médian par "
            "sous-bassin, par année et par type de bien.")


def onglet_deux_niveaux():
    question("Comment la démarche a-t-elle progressé ?")
    st.write("Le défi n'est pas seulement de décrire les territoires (le « où »), mais de savoir dans quelle phase est "
             "le marché (le « quand »). La démarche procède donc en deux niveaux, avec une transition entre les deux.")
    st.info("**H1 et H2** (le territoire explique les écarts de prix ; une photo par source suffit) ont été confirmées "
            "pendant l'exploration, avant toute modélisation. Elles ne sont pas testées par un modèle du niveau 1.")

    colonnes = st.columns(4)
    with colonnes[0]:
        carte("NIVEAU 1 · LE « OÙ »",
              "Que gagne-t-on en ajoutant le territoire au prix N-1 ? Trois blocs sont testés un à un : attractivité "
              "(H3a), logement et ménages (H3b), emploi (H3c), puis réunis.",
              "Apprentissage 2020-2023 · années jugées 2023 et 2024")
    with colonnes[1]:
        carte("TRANSITION",
              "L'erreur est coupée en deux : une part commune à tous les territoires, une part propre à chacun. "
              "Le modèle complet est ensuite réappris sur une période plus longue (H4, le « pont »).",
              "Mêmes variables, apprentissage dès 2017")
    with colonnes[2]:
        carte("NIVEAU 2 · LE « QUAND »",
              "Des indicateurs qui changent chaque année : volume et élan des prix (H5), capacité d'achat (H6), "
              "puis le prix de chaque vente (H7).",
              "Apprentissage dès 2017 · mêmes années jugées")
    with colonnes[3]:
        carte("TEST FINAL 2025",
              "Une année tenue en réserve, ouverte une seule fois par niveau, pour un candidat déclaré avant.",
              "Un candidat agrégé et un candidat par vente")

    st.markdown("#### Chaque hypothèse ajoute un bloc, et se compare à deux références")
    st.table(pd.DataFrame([
        ("H1", "Le territoire explique l'essentiel des écarts de prix avec les autres.","", "EDA"),
        ("H2", "Les caractéristiques lentes d'un territoire peuvent être prises à un seul millésime" ," ", "EDA"),
        ("H3", "Ajouté au prix N-1, le territoire n'apporte qu'un petit gain de prévision","Densité, équipements, délinquance", "Prix N-1 seul"),
        ("H3a", "L'attractivité d'un territoire améliore la prévision du prix.","Densité, équipements, délinquance", "Prix N-1 seul"),
        ("H3b", "Le parc de logements, la vacance, les ménages et les revenus améliore la prévision?" ,"Parc de logements, vacance, ménages, revenus", "Prix N-1 seul"),
        ("H3c", "L'évolution de l'emploi améliore la prévision des prix","Évolution de l'emploi salarié privé en N-1", "Prix N-1 seul"),
        ("H4","Apprendre sur plusieurs phases du marché réduit l'erreur systématique du modèle et rend les variables de territoires plus utiles" ,"Rien : le modèle complet est appris dès 2017", "Niveau 1 appris dès 2020"),
        ("H5","Le volume de ventes et l'élan récent des prix annoncent le sens de l'évolution", "Nombre de ventes, rotation du parc, élan des prix", "Modèle pont"),
        ("H6", "La capacité d'achat locale, c'est à dire la mensualité d'un achat type rapportée au revenu, améliore la prévision", "Taux d'effort : mensualité d'un achat type rapportée au revenu", "Modèle pont"),
        ("H7","Prévoir le prix de chaque vente plutôt que la médiane du territoire fait mieux à information égale", "Le bien (surface, pièces, terrain), le sous-bassin, le prix N-1, puis le « quand »",
         "Naïve du sous-bassin ; Elastic Net par vente"),
    ], columns=["Référence","Hypothèse", "Ce qui est ajouté", "Comparé à"]).set_index("Référence"))
    st.caption("Chaque modèle est comparé à la naïve, pour répondre à la question de recherche, et au modèle de l'étape "
               "précédente, pour isoler l'apport du bloc : seul le bloc change.")

    

def onglet_fenetres():
    question("Comment éviter qu'un modèle apprenne sur le futur ?")
    st.write("Pas de validation croisée au hasard : elle mélangerait les années. Pour chaque année jugée, le modèle est "
             "réappris sur les seules années qui la précèdent, prétraitement compris. Le réglage se fait à l'intérieur "
             "de chaque fenêtre, sur des plis temporels.")
    montrer(graphique_fenetres())
    retenir("le niveau 1 apprend sur 2020 et après, le niveau 2 dès 2017 ; 2023 et 2024 sont jugées, 2025 n'est ouverte "
            "qu'une fois.")

    st.markdown("#### Pourquoi 2025 reste fermée jusqu'au bout")
    colonnes = st.columns(3)
    with colonnes[0]:
        carte("Un test, une ouverture",
              "2025 n'est ouvert qu'une fois par niveau, pour un seul candidat par famille de modèle.")
    with colonnes[1]:
        carte("Un candidat déclaré avant",
              "Les deux candidats du niveau 2 ont été déclarés le 12/11/2026, avant l'ouverture, avec la règle de réussite. "
              "Aucune variante n'a été essayée sur 2025.")
    with colonnes[2]:
        carte("Un contrôle avant l'ouverture",
              "Chaque script refait d'abord la fenêtre 2024 et doit retrouver le résultat attendu ; sinon il s'arrête sans "
              "ouvrir 2025.")
    st.markdown("**Les deux candidats du niveau 2**")
    st.table(pd.DataFrame([
        ("Modèle agrégé", "Hiérarchique, une ordonnée propre à chaque sous-bassin, variables du pont + taux d'effort",
         "Médiane du sous-bassin"),
        ("Modèle par vente", "H7v4 : réseau Keras optimisé, bien + sous-bassin + prix N-1 + tout le « quand »",
         "B : chaque vente ; C : médiane du sous-bassin"),
    ], columns=["Famille", "Candidat", "Ce qui est jugé"]).set_index("Famille"))
    st.caption("Le niveau 1 avait ouvert 2025 avant eux, pour l'Elastic Net du modèle complet (réappris sur 2020-2024).")
    retenir("2025 joue le rôle d'un examen final : choisi avant, ouvert une fois, rapporté tel quel.")


def onglet_mesures():
    question("Comment lit-on un résultat, et quand un modèle est-il retenu ?")
    st.markdown("#### Quatre mesures")
    colonnes = st.columns(4)
    with colonnes[0]:
        carte("MAE", "Erreur absolue moyenne, en €/m². Plus elle est basse, mieux c'est.")
    with colonnes[1]:
        carte("Gain sur la naïve", "1 − MAE du modèle / MAE de la naïve. Positif : le modèle fait mieux ; négatif : moins bien.")
    with colonnes[2]:
        carte("Biais", "Prévision moins prix réel, en moyenne. Positif : le modèle surestime les prix.")
    with colonnes[3]:
        carte("Territoires gagnés", "Part des sous-bassins où l'erreur du modèle est plus petite que celle de la naïve.")

    st.markdown("#### Un résultat peut être dû au hasard : deux tests")
    colonnes = st.columns(2)
    with colonnes[0]:
        carte("Bootstrap par bassin de vie",
              "1 000 tirages de bassins de vie entiers, avec tous leurs sous-bassins (ils ne sont pas indépendants). "
              "On obtient un intervalle à 95 % de l'écart de MAE avec la naïve.")
    with colonnes[1]:
        carte("Wilcoxon apparié et rang-bisériale",
              "Le modèle fait-il mieux dans la plupart des sous-bassins ? La rang-bisériale va de −1 (mieux partout) "
              "à +1 (moins bien partout).")
    montrer(graphique_intervalles(INTERVALLES_LECTURE))
    retenir("si l'intervalle est entièrement sous 0, le modèle fait vraiment mieux ; s'il contient 0, on ne peut pas "
            "conclure ; entièrement au-dessus, il fait vraiment moins bien.")
    

    st.markdown("#### La règle de décision, fixée avant les résultats")
    with st.container(border=True):
        st.markdown("Une année est réussie pour un type de bien si **les trois conditions** sont remplies :")
        st.markdown("- le **gain sur la naïve** atteint au moins **3 %** ;\n"
                    "- l'**intervalle du bootstrap** est entièrement sous 0 ;\n"
                    "- la **rang-bisériale** est négative.")
        st.write("Un modèle est retenu s'il réussit la majorité des années jugées : ici, 2023 et 2024, soit les deux.")
    retenir("le gain doit être net, démontré et en faveur du modèle ; un gain de quelques pour cent qui peut s'expliquer "
            "par le hasard ne suffit pas.")


def onglet_modele_par_vente():
    question("Que compare-t-on quand on prévoit chaque vente ?")
    st.write("Le modèle par vente prévoit le prix au m² de chaque bien vendu, à partir du bien (surface, terrain, pièces), "
             "de son sous-bassin et des mêmes indicateurs que les autres modèles. Deux évaluations sont possibles, et "
             "elles ne répondent pas à la même question.")
    gauche, droite = st.columns(2)
    with gauche:
        with st.container(border=True, height="stretch"):
            st.caption("ÉVALUATION B")
            st.markdown("### L'erreur sur chaque vente")
            st.write("Le modèle prévoit le prix de chaque vente de l'année jugée. L'erreur est moyennée par sous-bassin.")
            st.markdown("**Référence :** le prix médian N-1 du sous-bassin, donné à toutes ses ventes. Cette référence "
                        "ignore les caractéristiques du bien, que le réseau connaît.")
    with droite:
        with st.container(border=True, height="stretch"):
            st.caption("ÉVALUATION C")
            st.markdown("### La médiane du sous-bassin")
            st.write("Le modèle prévoit en N le prix des biens vendus en N-1. La médiane de ces prévisions est comparée au "
                     "prix médian réel du sous-bassin.")
            st.markdown("**Référence :** le prix N-1 du sous-bassin, comme pour les modèles agrégés. C'est la comparaison à "
                        "information égale.")
    retenir("B dit si le modèle prévoit mieux le prix d'un bien ; seule C dit s'il prévoit mieux le prix d'un territoire, "
            "la question de recherche.")
    st.caption("Les deux évaluations n'ont pas la même unité : leurs erreurs ne se comparent pas entre elles, seulement "
               "leur gain sur la naïve correspondante.")


def onglet_construction():
    question("Comment le travail est-il construit pour être comparable, contrôlé et rejouable ?")
    st.write("Quatre personnes modélisent des hypothèses différentes. Pour qu'une différence de résultat vienne des variables "
             "et non du code, tout ce qui doit être identique est écrit une seule fois dans un dossier partagé, le socle, puis "
             "contrôlé par des tests.")

    st.markdown("#### Le socle : écrit une fois, importé partout")
    st.table(pd.DataFrame([
        ("commun.py", "La table, le découpage par fenêtres et plis temporels, la prévision naïve, les erreurs, le bootstrap par "
                      "bassin de vie et l'évaluation par vente (B et C)", "Toutes les hypothèses"),
        ("gabarit_boosting.py", "Gradient boosting qui apprend l'évolution du prix, même grille pour tous", "Niveau 1 (H3)"),
        ("gabarit_hierarchique.py", "Modèle hiérarchique : une ordonnée propre à chaque sous-bassin", "Candidat agrégé du test 2025"),
        ("gabarit_pont.py", "Le modèle pont : Elastic Net du niveau 1 appris dès 2017, où l'on ajoute un bloc de variables", "H4, H5, H6"),
        ("gabarit_keras.py", "Réseau de neurones avec précautions contre les fuites, dont l'arrêt précoce sur la dernière année d'apprentissage",
         "H7"),
    ], columns=["Fichier", "Ce qu'il fait", "Utilisé par"]).set_index("Fichier"))
    
    retenir("deux hypothèses ne diffèrent que par leurs variables : même découpage, même naïve, mêmes mesures, mêmes tests.")

    st.markdown("#### Les tests : vérifier le code avant de croire un résultat")
    colonnes = st.columns(3)
    with colonnes[0]:
        carte("Un fichier de tests par module", "28 fonctions `test_*` dans `tests/`, lancées par `pytest` depuis la racine du projet.")
    with colonnes[1]:
        carte("Ce qu'ils vérifient",
              "Dans chaque pli temporel, l'apprentissage précède la validation ; le gabarit pont retrouve les MAE publiées de H4 "
              "(2023 / 2024 : 158,7 / 187,8 en appartements, 130,4 / 180,2 en maisons) ; dans l'évaluation C, des ventes qui "
              "recopient le prix N-1 redonnent exactement la naïve.")
    with colonnes[2]:
        carte("Des ventes inventées pour le réseau",
              "Les tests de Keras utilisent des ventes fabriquées à règle connue, jamais les ventes réelles : un réseau qui "
              "fonctionne doit battre la naïve. Sans TensorFlow, ils sont sautés, pas en échec.")
  

    st.markdown("#### Deux parcours pour rejouer le travail")
    gauche, droite = st.columns(2)
    with gauche:
        with st.container(border=True, height="stretch"):
            st.caption("PARCOURS COURT")
            st.markdown("### Un seul script, environ 1 h 30 à 2 h")
            st.markdown("1. Télécharge un kit de 9 fichiers déjà préparés.\n"
                        "2. Lance les modèles H3, H4, H5-H6 et H7, jugés sur 2023 et 2024.\n"
                        "3. Lance le test 2025 (les deux scripts de `4_evaluation_2025`).")
            st.write("Tout est tracé dans MLflow : un run par hypothèse, modèle, type de bien et année jugée, toujours contre la naïve. "
                     "Le réseau n'est pas réoptimisé : il reprend les réglages du kit.")
            st.code("python 03_modelisation/modelisation_parcours_court.py", language="bash")
    with droite:
        with st.container(border=True, height="stretch"):
            st.caption("PARCOURS LONG")
            st.markdown("### Tout reconstruire depuis les sources")
            st.markdown("1. Télécharger les sources brutes (DVF+ filtré ou brut).\n"
                        "2. Préparer les sources (12 blocs), puis la table de modélisation (11 sections).\n"
                        "3. Audits et EDA, puis les notebooks de modélisation, du niveau 1 au modèle par vente.\n"
                        "4. Test 2025, deux scripts puis un notebook de comparaison.")
            st.write("C'est le parcours qui montre comment chaque choix a été fait et commenté, y compris l'optimisation du réseau.")
            st.code("python 00_telechargement/telechargement.py dvf_residentiel ...", language="bash")
    retenir("le parcours court vérifie les résultats en une session ; le parcours long montre d'où ils viennent.")
   


# =============================================================================
# 4. Résultats
# =============================================================================
def onglet_reponse():
    question("Peut-on prévoir le prix mieux qu'en reconduisant celui de l'an dernier ?")
    with st.container(border=True):
        st.caption("RÉPONSE À LA QUESTION DE RECHERCHE")
        gauche, droite = st.columns(2)
        with gauche:
            st.markdown("### Pour le prix médian d'un territoire : non")
            st.write("Aucun de nos modèles ne fait mieux que la naïve, ni en 2023, ni en 2024, ni en 2025 : pas pendant "
                     "le retournement du marché, pas dans un marché stable.")
            st.caption("Seul succès isolé : le modèle hiérarchique pour les maisons en 2024 (+9,5 %), une année sur trois.")
        with droite:
            st.markdown("### Pour le prix d'une vente : oui")
            st.write("Le réseau H7v4 fait mieux que la naïve de 5 à 8 %, trois années sur trois, pour les appartements "
                     "comme pour les maisons.")
            st.caption("Cette naïve ignore les caractéristiques du bien : une part du gain peut venir de cette information.")

    st.markdown("#### Le gain de chaque modèle sur la naïve, année par année")
    definition("Gain sur la naïve", "positif, le modèle fait mieux que reconduire le prix N-1 ; négatif, il fait moins bien. "
                                    "Trait pointillé : seuil de 3 %.")
    df = tableau_synthese()
    montrer(barres_gain(df, "Modèle", ordre_couleur=["Année réussie", "Non réussie"], hauteur=118,
                        facettes={"row": alt.Row("Type:N", title=None), "column": alt.Column("Année:N", title=None)}))
    retenir("presque toutes les barres sont à gauche de 0 : sur la médiane d'un territoire, nos modèles font moins bien "
            "que la naïve ; seul le réseau jugé sur chaque vente est réussi chaque année.")

    st.markdown("#### Le niveau de certitude")
    colonnes = st.columns(3)
    with colonnes[0]:
        carte("Trois années seulement", "Chacune dans une phase différente : retournement en 2023 et 2024, marché presque plat en 2025.")
    with colonnes[1]:
        carte("Beaucoup de modèles comparés", "Les comparaisons multiples ne sont pas corrigées : un succès isolé peut être dû au hasard.")
    with colonnes[2]:
        carte("Des associations, pas des causes", "Le modèle par vente apprend sur un échantillon d'environ 10 % des ventes.")
    with st.expander("Le tableau complet"):
        tableau = df.pivot_table(index=["Modèle", "Type"], columns="Année", values="Gain", sort=False)
        st.dataframe(tableau.map(lambda v: "—" if pd.isna(v) else signe(v, 1) + " %"), width="stretch")


def onglet_territoire():
    question("Le territoire (le « où ») améliore-t-il la prévision ?")
    st.write("Chaque bloc de variables de territoire est ajouté au prix N-1 et comparé au modèle « prix N-1 seul » de la "
             "même méthode, appris sur 2020-2023.")
    colonnes = st.columns(3)
    with colonnes[0]:
        with st.container(border=True, height="stretch"):
            st.badge("Non retenue", color="gray")
            st.markdown("**H3a · attractivité**")
            st.write("Les équipements apportent +1,4 à +2,9 % aux maisons, sous le seuil de 3 %. La délinquance perd en "
                     "2023 et gagne en 2024 : son effet n'est pas stable.")
    with colonnes[1]:
        with st.container(border=True, height="stretch"):
            st.badge("Retenue", color="primary")
            st.markdown("**H3b · logement et ménages**")
            st.write("Le parc de logements et la vacance portent le gain (+6 à +7 % pour les maisons). Ménages et revenus "
                     "n'ajoutent rien.")
    with colonnes[2]:
        with st.container(border=True, height="stretch"):
            st.badge("Retenue (maisons)", color="primary")
            st.markdown("**H3c · emploi**")
            st.write("Gradient boosting : +3,7 % et +6,7 %. Le gain passe par la tendance nationale ; avec l'emploi connu "
                     "en début d'année (N-2), aucune année n'est réussie.")
    st.caption("Gains contre « prix N-1 seul ». Le modèle complet qui réunit les blocs retenus gagne de 7 à 11 %.")

    st.markdown("#### Mais contre la naïve, le modèle complet perd les trois années")
    definition("MAE", "erreur absolue moyenne en €/m² ; plus elle est basse, mieux c'est. Le test 2025 est ouvert une seule fois.")
    choix = st.segmented_control("Type de bien", ["Appartements", "Maisons"], default="Maisons", key="modelisation_niveau1",
                                 label_visibility="collapsed")
    if choix is None:
        choix = "Maisons"
    montrer(graphique_niveau1(choix))
    retenir("le territoire améliore un peu « prix N-1 seul » (+7 à +11 %), mais le modèle complet reste moins bon que "
            "la naïve en 2023, 2024 et 2025 : il dit où les prix sont élevés, pas comment ils évoluent.")


def onglet_biais():
    question("Pourquoi les modèles se trompent-ils, et que change une période d'apprentissage plus longue ?")
    st.write("Le niveau 1 n'a connu que la hausse de 2020-2022 (+5 à +10 % par an). Quand le marché se retourne, il "
             "prolonge cette hausse et surestime les prix presque partout.")
    definition("Biais", "prévision moins prix réel, en moyenne, en €/m². Positif : le modèle surestime les prix.")
    montrer(graphique_biais())
    retenir("appris pendant la hausse, tous les modèles surestiment en 2024 ; apprendre dès 2017 puis ajouter le taux "
            "d'effort réduit ce décalage, sans l'éliminer.")
    st.caption("Niveau 1 : appris dès 2020 ; pont : appris dès 2017 ; pont + effort : avec le taux d'effort (H6). Naïve, niveau 1 et pont : lignes de H4. Dernière barre : lignes communes de H5-H6 (312 lignes de moins). "
               "Sur les seules lignes de H6, le biais des maisons passe de +151 à +50 €/m².")

    colonnes = st.columns(3)
    colonnes[0].metric("MAE 2024 des maisons", "197 → 180 €/m²", "apprendre dès 2017 plutôt que 2020", delta_color="off")
    colonnes[1].metric("Erreur réduite dans", "8 cas sur 8", "2 types de bien, 2 années, 2 modèles", delta_color="off")
    colonnes[2].metric("Gain sur la naïve en 2021-2022", "+16 à +32 %", "contrôle : le marché montait", delta_color="off")
    retenir("le modèle pont gagne quand le marché monte et perd quand il recule : il manque un signal sur la phase du marché.")

    with st.expander("Le diagnostic : le décalage commun pèse un quart de l'erreur"):
        st.write("L'erreur de 2024 est séparée en une part commune à tous les territoires et une part propre à chacun. "
                 "Sans la part commune, l'erreur du modèle complet du niveau 1 baisse d'un quart, mais il ne bat toujours "
                 "pas la naïve.")
        st.table(pd.DataFrame([
            ("Appartements", "197", "148", "148"), ("Maisons", "197", "141", "129"),
        ], columns=["MAE 2024 (€/m²)", "Modèle complet", "Sans la part commune", "Naïve"]).set_index("MAE 2024 (€/m²)"))
        st.caption("La part commune utilise l'erreur réelle de l'année jugée : c'est un outil de diagnostic, pas un modèle utilisable.")


def onglet_signaux():
    question("Quels signaux du « quand » aident, et lesquels s'inversent d'une année à l'autre ?")
    st.write("Le niveau 2 ajoute des indicateurs qui changent chaque année dans chaque territoire. Ils sont ajoutés au "
             "modèle pont et comparés à lui.")
    definition("Taux d'effort (H6)", "mensualité d'un achat type (60 m² en appartement, 100 m² en maison, au prix N-1, sur "
                                      "25 ans, au taux des crédits de décembre N-1) rapportée au revenu du territoire. Tout est connu avant l'année prévue.")
    definition("Gain sur le pont", "positif, l'ajout améliore le modèle pont ; négatif, il le dégrade. Trait pointillé : seuil de 3 %.")
    choix = st.segmented_control("Année jugée", [2023, 2024], default=2024, key="modelisation_signaux_annee",
                                 label_visibility="collapsed")
    if choix is None:
        choix = 2024
    df = tableau_signaux(choix)
    df["Verdict"] = df["Type"]
    montrer(barres_gain(df, "Signal", couleur="Type", ordre_couleur=["Appartements", "Maisons"], titre_x="Gain sur le modèle pont (%)",
                        hauteur=300, decalage="Type"))
    retenir("volume, rotation et élan changent de signe d'une année à l'autre ; le taux d'effort, qui contient le taux "
            "des crédits, est le seul signal qui aide nettement, surtout en 2024.")

    colonnes = st.columns(2)
    with colonnes[0]:
        with st.container(border=True, height="stretch"):
            st.badge("Non confirmée", color="gray")
            st.markdown("**H5 · volume et élan**")
            st.write("Le bloc est retenu de justesse pour les maisons (+3,2 % et +3,5 %), pas pour les appartements. "
                     "Il n'annonce pas le retournement : le biais du pont reste presque inchangé.")
    with colonnes[1]:
        with st.container(border=True, height="stretch"):
            st.badge("Confirmée pour les maisons", color="primary")
            st.markdown("**H6 · capacité d'achat**")
            st.write("Pour les maisons, le biais 2024 passe de +151 à +50 €/m² et la MAE de 180 à 126 €/m². Pour les "
                     "appartements, 2023 perd (−5 %). Le taux d'effort ne bat pas la naïve et ne tient pas en 2025.")
    st.caption("Réunir H5 et H6 n'apporte rien. Une indicatrice des prix 2021-2022 corrigerait le biais de 2023, mais elle est "
               "écartée : on ne sait pas d'avance quelles années seront exceptionnelles.")


def onglet_vente_par_vente():
    question("Prévoir le prix de chaque vente plutôt que la médiane d'un territoire change-t-il le résultat ?")
    colonnes = st.columns(3)
    colonnes[0].metric("H7v4, chaque vente (B)", "+5,4 à +7,3 %", "2023 et 2024, les deux types de bien", delta_color="off")
    colonnes[1].metric("H7v4, médiane du sous-bassin (C)", "−20 à −65 %", "aucun réseau ne bat la naïve", delta_color="off")
    colonnes[2].metric("Le prix N-1 change tout", "−9 → +5 %", "appartements 2023, avant et après le prix N-1", delta_color="off")
    definition("B et C", "B juge chaque vente contre le prix médian N-1 de son sous-bassin ; C juge la médiane du sous-bassin, "
                         "comme les modèles agrégés.")
    choix = st.segmented_control("Type de bien", ["Appartements", "Maisons"], default="Maisons", key="modelisation_h7",
                                 label_visibility="collapsed")
    if choix is None:
        choix = "Maisons"
    montrer(graphique_h7(choix))
    retenir("dès que le réseau connaît le prix N-1, il bat la naïve vente par vente ; sur la médiane d'un territoire, "
            "aucune version ne la bat, et le gain n'est plus un gain.")

    st.markdown("#### Pourquoi le réseau apprend l'écart au prix N-1")
    st.write("Avec les réglages par défaut, le réseau perdait contre la naïve sur le quart le plus cher des appartements : "
             "il ramenait les prix élevés vers le milieu. Lui faire apprendre l'évolution par rapport au prix N-1 règle le problème.")
    montrer(graphique_optimisation())
    retenir("apprendre l'écart au prix N-1 plutôt que le prix fait passer le quart le plus cher des appartements de "
            "−176 à +62 €/m² face à la naïve ; perte et hyperparamètres n'affinent qu'un peu.")
    st.caption("Validation 2022, une année qui n'est jamais jugée : ces erreurs ne se comparent pas à celles des tableaux ci-dessus.")

    with st.expander("Le réseau reste moins bon que l'Elastic Net sur la médiane d'un territoire"):
        st.write("Pour les maisons en 2024, l'Elastic Net par vente fait 121 €/m² d'erreur sur la médiane d'un sous-bassin, "
                 "contre 157 pour H7v4. Le réseau apprend l'évolution moyenne de 2017 à l'année d'avant, surtout la hausse : il "
                 "surestime les prix sans signal du « quand ». La capacité d'achat corrige l'essentiel de ce biais.")


def onglet_test_2025():
    question("Que donne l'année tenue en réserve ?")
    colonnes = st.columns(3)
    colonnes[0].metric("Prix médian en 2025", "+1,5 % / +0,9 %", "appartements / maisons", delta_color="off")
    colonnes[1].metric("Taux des crédits, décembre 2024", "3,15 %", "contre 3,63 % un an plus tôt", delta_color="off")
    colonnes[2].metric("Situation du marché", "Presque plat", "ni la hausse, ni le retournement", delta_color="off")
    st.caption("Dans un marché stable, la naïve est par construction très difficile à battre.")

    definition("Gain sur la naïve", "positif, le candidat fait mieux que reconduire le prix N-1. Trait pointillé : seuil de 3 %.")
    montrer(graphique_test_2025())
    retenir("seul le réseau jugé sur chaque vente réussit en 2025 ; sur la médiane d'un territoire, le niveau 2 se "
            "rapproche de la naïve sans la battre, et le niveau 1 en est plus loin.")

    st.markdown("#### L'intervalle de l'écart avec la naïve")
    intervalles = []
    for candidat, _e, type_bien, _m, _n, _g, bas, haut, _b, _r in TEST_2025:
        lecture = ("Mieux que la naïve" if haut < 0 else
                   "Moins bien que la naïve" if bas > 0 else "Indécis : contient 0")
        intervalles.append((candidat + " · " + type_bien.lower(), bas, haut, lecture))
    montrer(graphique_intervalles(intervalles))
    retenir("seul le modèle hiérarchique des appartements frôle la naïve ; tous les autres intervalles du niveau 1, du "
            "modèle hiérarchique des maisons et du réseau sur la médiane sont entièrement du mauvais côté de 0.")
    st.caption("Échelles différentes : B porte sur l'erreur de chaque vente (autour de 700 €/m²), les autres sur la médiane "
               "d'un territoire (100 à 190 €/m²). Seule la position par rapport à 0 se compare.")

    tableau = pd.DataFrame([
        (c, t, fr(m, 1), fr(n, 1), signe(g, 1) + " %", b, "Oui" if r else "Non")
        for c, _e, t, m, n, g, _bas, _haut, b, r in TEST_2025
    ], columns=["Candidat", "Type", "MAE modèle (€/m²)", "MAE naïve (€/m²)", "Gain", "Biais", "Réussi"])
    with st.expander("Le tableau du test 2025"):
        st.dataframe(tableau, hide_index=True, width="stretch")
        st.caption("Le biais de B est en % du prix, celui des autres lignes en €/m² : ils ne se comparent pas. Le niveau 1 "
                   "a ses propres bornes d'intervalle arrondies.")
    colonnes = st.columns(2)
    with colonnes[0]:
        carte("Le niveau 2 fait mieux que le niveau 1",
              "En 2025, le hiérarchique fait 148 €/m² contre 163 pour l'Elastic Net du niveau 1 (appartements), 109 contre "
              "113 pour les maisons. Son écart à la naïve est 3,6 fois plus petit pour les appartements.")
    with colonnes[1]:
        carte("Le biais change de signe",
              "Le hiérarchique sous-estime en 2025 (−24 et −21 €/m²) alors que le pont + taux d'effort surestimait en "
              "2023 et 2024. Hypothèse, non vérifiée : l'effet du crédit cher ne se renverse pas aussi vite que le marché.")


# =============================================================================
# 5. Page
# =============================================================================
with st.sidebar:
    st.divider()
    partie = st.radio("Afficher", ["Méthodologie", "Résultats"], key="modelisation_partie")

if partie == "Méthodologie":
    entete("Modélisation", "Méthodologie. Ce que nous cherchions à prédire, comment la démarche a progressé, comment les "
                           "modèles ont été jugés.")
    onglets = st.tabs(["Cible et référence", "Deux niveaux", "Fenêtres et test 2025", "Mesures et décision",
                       "Modèle par vente", "Construction du code"])
    with onglets[0]:
        onglet_cible_reference()
    with onglets[1]:
        onglet_deux_niveaux()
    with onglets[2]:
        onglet_fenetres()
    with onglets[3]:
        onglet_mesures()
    with onglets[4]:
        onglet_modele_par_vente()
    with onglets[5]:
        onglet_construction()
else:
    entete("Modélisation", "Résultats. Les comparaisons étape par étape, jusqu'au test 2025, puis la réponse à la question "
                           "de recherche.")
    onglets = st.tabs(["Territoire", "Biais", "Signaux « quand »", "Vente par vente", "Test 2025", "Réponse"])
    with onglets[0]:
        onglet_territoire()
    with onglets[1]:
        onglet_biais()
    with onglets[2]:
        onglet_signaux()
    with onglets[3]:
        onglet_vente_par_vente()
    with onglets[4]:
        onglet_test_2025()
    with onglets[5]:
        onglet_reponse()
