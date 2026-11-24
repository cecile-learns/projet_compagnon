# =============================================================================
# 05_application/pages/futurs.py : page Développements futurs (Gabrielle)
# =============================================================================
# Ce que fait la page : présente la suite possible du projet, en 3 à 4 minutes d'oral.
# Contenu repris du rapport final, partie V (« Ouverture ») et tableau 23.
# Deux onglets, un par contexte de suite (rapport, 5.1 et 5.2), chacun suit le parcours des autres pages :
# un message clé (« À retenir »), puis des questions dans l'ordre où le lecteur se les pose.
#   1. Une application à vendre  : ce qu'il faut demander aux futurs utilisateurs avant d'investir
#   2. Un client identifié       : trois directions techniques, dont le chantier du « quand »
# Sous les onglets, un menu repliable : les pistes d'amélioration du travail existant (rapport, 5.3).
#
# Données lues : donnees/eda_territoires.parquet (même fichier que la page EDA, produit par
# 04_outils/preparer_eda_app.py), pour le graphique « les ventes bougent avant les prix ».
# Sans ce fichier, la page s'affiche quand même : seul le graphique est remplacé par un message.
# Règles de l'application : aucune couleur ni police ici (thème commun), seulement les fonctions de commun_app.

import altair as alt
import pandas as pd
import streamlit as st

from commun_app import DONNEES, afficher_message, entete


# -----------------------------------------------------------------------------
# 0. Petites fonctions (mêmes conventions que les pages EDA et Modélisation)
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


def carte(titre, texte, legende=None, badge=None, couleur_badge="gray"):
    with st.container(border=True, height="stretch"):
        if badge:
            st.badge(badge, color=couleur_badge)
        st.markdown("**" + titre + "**")
        st.write(texte)
        if legende:
            st.caption(legende)


# -----------------------------------------------------------------------------
# 1. Contenu repris du rapport final (partie V)
# -----------------------------------------------------------------------------
# 5.1 : trois questions à poser en entretien (titre, constat, question posée)
QUESTIONS_ENTRETIEN = [
    ("Lire l'estimation",
     "Un prix estimé pour 2026 n'est pas une prévision de tendance. Nous l'écrivons dans l'outil, mais rien ne dit "
     "que la distinction passe.",
     "L'utilisateur comprend-il ce que dit le chiffre ?"),
    ("Choisir la fourchette",
     "Nous affichons la dispersion des prix de biens équivalents plutôt que l'erreur du modèle, trop large pour être "
     "utile. C'est une décision prise à la place de l'utilisateur.",
     "Préfère-t-il une fourchette large mais honnête sur le modèle, ou une fourchette qui décrit le marché ?"),
    ("Accepter un refus",
     "L'outil n'estime rien dans 45,5 % des sous-bassins d'appartements, faute de ventes.",
     "Ce refus est-il vu comme un gage de sérieux, ou comme une panne ?"),
]

# 5.1 : les interlocuteurs possibles (nom, statut)
INTERLOCUTEURS = [("Particulier qui achète", "visé aujourd'hui"), ("Agent immobilier", "à interroger"),
                  ("Notaire", "à interroger"), ("Investisseur", "à interroger")]

# 5.2 : trois directions, par ordre de proximité avec l'existant (badge, titre, texte, ce que ça lève)
DIRECTIONS = [
    ("1 · Le plus immédiat", "Se raccorder aux données du client",
     "La base d'une agence contient ce qui manque à DVF+ : descriptions, photos, diagnostics, délai de vente, et "
     "surtout l'écart entre prix demandé et prix payé.",
     "Lève la limite la plus lourde : DVF+ ne voit ni l'état du bien, ni l'étage, ni la performance énergétique."),
    ("2 · Architecture prête", "Étendre l'outil aux annonces",
     "Avec des annonces sur un territoire couvert par DVF+, l'assistant prévu au départ redevient possible : "
     "comprendre une recherche en langage libre, proposer des annonces, confronter chaque prix demandé à "
     "l'estimation issue des ventes réelles.",
     "L'axe 2 a été abandonné faute de données, pas faute d'intérêt. Les quatre blocs de l'outil ont déjà leurs "
     "formats d'échange."),
    ("3 · Le plus ambitieux", "Servir les professionnels",
     "Un professionnel ne demande pas le prix d'un bien : il demande où en est le marché, s'il risque de se "
     "retourner, et ce que cela change pour une estimation aujourd'hui.",
     "Part de notre résultat négatif : c'est la question que nous n'avons pas résolue."),
]

# 5.2 : ce que nos travaux disent du « quand » (titre, texte)
CONSTATS = [
    ("Le prix N-1 porte presque tout", "Il ne reste qu'environ 2 % d'évolution propre à prévoir."),
    ("Le territoire dit « où », pas « quand »", "Il décrit où les prix sont hauts, jamais quand ils changent."),
    ("Le crédit n'aide que dans sa phase", "La capacité d'achat a corrigé le retournement de 2024, pas la "
                                           "stabilisation de 2025."),
]

# 5.3, tableau 23 : pistes d'amélioration du travail existant
PISTES = pd.DataFrame([
    ("Actualiser les données à chaque livraison semestrielle de DVF+",
     "Les données s'arrêtent à 2025 ; les caractéristiques du territoire vieillissent plus vite encore"),
    ("Juger le modèle sur une année jamais ouverte", "2025 a été lu deux fois ; trois années jugées seulement"),
    ("Analyser le réseau retenu avec la méthode SHAP", "Son importance de variables n'est mesurée que par permutation"),
    ("Apprendre sur davantage de ventes", "Le modèle apprend sur environ une vente sur dix"),
    ("Corriger les comparaisons multiples",
     "De nombreux modèles comparés ; un gain proche du seuil de 3 % pourrait tenir au hasard"),
    ("Apparier les ventes au diagnostic de performance énergétique",
     "DVF+ ne voit aucune caractéristique qualitative du bien"),
    ("Mesurer l'écart entre surface saisie et surface bâtie",
     "L'utilisateur saisit une surface habitable, le modèle a appris sur une surface bâtie"),
    ("Corriger les territoires sans estimation", "Les maisons d'un sous-bassin et cinq codes de liaison restent non couverts"),
], columns=["Piste", "Limite qu'elle lève"])


# -----------------------------------------------------------------------------
# 2. Graphique : les ventes bougent avant les prix (France entière)
# -----------------------------------------------------------------------------
def series_marche():
    """Indices base 100 (première année) : prix au m² par type de bien et nombre de ventes, France entière.
    Même calcul que la page EDA : médiane de chaque sous-bassin, moyenne pondérée par le nombre de ventes."""
    t = charger_parquet("eda_territoires.parquet")
    t = t.assign(prix_x_ventes=t["prix"] * t["n_ventes"])
    prix = t.groupby(["annee", "type_bien"]).agg(px=("prix_x_ventes", "sum"), n=("n_ventes", "sum")).reset_index()
    prix["valeur"] = prix["px"] / prix["n"]
    prix["serie"] = prix["type_bien"].map({"appartement": "Prix des appartements", "maison": "Prix des maisons"})
    ventes = t.groupby("annee")["n_ventes"].sum().reset_index().rename(columns={"n_ventes": "valeur"})
    ventes["serie"] = "Nombre de ventes"
    df = pd.concat([prix[["annee", "serie", "valeur"]], ventes[["annee", "serie", "valeur"]]])
    df["annee"] = df["annee"].astype(int)
    df = df.sort_values(["serie", "annee"]).reset_index(drop=True)
    df["indice"] = 100 * df["valeur"] / df.groupby("serie")["valeur"].transform("first")
    return df


def graphique_marche(df):
    ordre = ["Nombre de ventes", "Prix des appartements", "Prix des maisons"]
    debut, fin = int(df["annee"].min()), int(df["annee"].max())
    x = alt.X("annee:Q", title=None, axis=alt.Axis(format="d", values=list(range(debut, fin + 1))),
              scale=alt.Scale(domain=[debut - 0.2, fin + 0.2], nice=False))
    lignes = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
        x=x, y=alt.Y("indice:Q", title="Indice (100 = " + str(debut) + ")", scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=ordre), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("annee:Q", title="Année", format="d"), alt.Tooltip("serie:N", title="Série"),
                 alt.Tooltip("indice:Q", title="Indice", format=".1f")])
    sommets = df.loc[df.groupby("serie")["indice"].idxmax()]
    sommets = sommets[(sommets["annee"] > debut) & (sommets["annee"] < fin)]
    cercles = alt.Chart(sommets).mark_point(size=220, filled=False, strokeWidth=2, color="gray").encode(
        x=x, y="indice:Q")
    repere = alt.Chart(pd.DataFrame({"indice": [100]})).mark_rule(strokeDash=[3, 3], opacity=0.5).encode(y="indice:Q")
    return (repere + lignes + cercles).properties(height=300), sommets


# -----------------------------------------------------------------------------
# 3. Page
# -----------------------------------------------------------------------------
entete("Développements futurs", "Deux suites possibles, selon que le projet trouve ou non un client")

st.write("Le projet est pédagogique : personne ne l'a commandé. Pour envisager la suite, il faut se donner un "
         "contexte. Deux hypothèses sont plausibles, et elles ne mènent pas aux mêmes prochaines étapes.")

onglets = st.tabs(["Une application à vendre", "Un client déjà identifié"])

# =============================================================================
# Onglet 1 : une application à vendre (rapport, 5.1)
# =============================================================================
with onglets[0]:
    cle("Le modèle est prêt. Ce qui reste à faire porte sur l'**usage**.",
        "Ergonomie, présentation, formulation des résultats. Avant d'y investir, il faut vérifier auprès de futurs "
        "utilisateurs que l'outil répond à un besoin réel, et de la façon attendue.")

    question(1, "Que faut-il demander aux futurs utilisateurs ?")
    st.write("Trois questions à poser en entretien. Aucune ne se tranche depuis les données.")
    colonnes = st.columns(3)
    for colonne, (titre, constat, demande) in zip(colonnes, QUESTIONS_ENTRETIEN):
        with colonne:
            carte(titre, constat, legende="**Question d'entretien :** " + demande)

    question(2, "À qui s'adresse l'outil ?")
    st.write("Les entretiens décideraient aussi de l'interlocuteur. Chacun attend une précision, une maille et une "
             "sortie différentes.")
    colonnes = st.columns(len(INTERLOCUTEURS))
    for colonne, (nom, statut) in zip(colonnes, INTERLOCUTEURS):
        with colonne.container(border=True):
            st.markdown("**" + nom + "**")
            st.badge(statut, color="primary" if statut == "visé aujourd'hui" else "gray")

# =============================================================================
# Onglet 2 : un client déjà identifié (rapport, 5.2)
# =============================================================================
with onglets[1]:
    cle("Avec un client, le besoin est donné : le travail redevient **technique**.",
        "Par exemple une agence immobilière, un réseau, un acteur du financement. Trois directions s'ouvrent, de la "
        "plus proche de l'existant à la plus ambitieuse.")

    question(1, "Par quoi commencer ?")
    colonnes = st.columns(3)
    for colonne, (badge, titre, texte, leve) in zip(colonnes, DIRECTIONS):
        with colonne:
            carte(titre, texte, legende=leve, badge=badge,
                  couleur_badge="primary" if badge.startswith("3") else "gray")

    question(2, "Pourquoi la question des professionnels est-elle la plus intéressante ?")
    st.write("Parce que nos résultats disent précisément pourquoi elle est difficile, et par où commencer.")
    colonnes = st.columns(3)
    for colonne, (titre, texte) in zip(colonnes, CONSTATS):
        with colonne:
            carte(titre, texte)

    st.markdown("#### Une piste ouverte : les ventes bougent avant les prix")
    try:
        marche = series_marche()
    except FileNotFoundError as erreur:
        afficher_message(str(erreur) + " : lancer 04_outils/preparer_eda_app.py.", "avertissement")
    else:
        graphique, sommets = graphique_marche(marche)
        st.altair_chart(graphique, use_container_width=True)
        annees = dict(zip(sommets["serie"], sommets["annee"].astype(int)))
        lecture = "Cercles : année du sommet de chaque série."
        if "Nombre de ventes" in annees and len(annees) > 1:
            prix = [a for s, a in annees.items() if s != "Nombre de ventes"]
            lecture = ("Les ventes culminent en " + str(annees["Nombre de ventes"]) + ", les prix en "
                       + " et ".join(str(a) for a in sorted(set(prix))) + ". " + lecture)
        st.caption(lecture + " France entière : médiane de chaque sous-bassin, moyenne pondérée par les ventes "
                   "(même calcul que la page Exploration des données).")
    st.write("Pour les maisons, le test de causalité le confirme (p = 0,001, sans effet en retour). Ajouter les "
             "volumes n'améliore pas la prévision du prix **en moyenne**, mais ce n'était pas la bonne question.")
    with st.container(border=True):
        st.caption("LE CHANTIER QUE NOUS OUVRIRIONS EN PREMIER")
        st.markdown("### Détecter un changement de régime, plutôt que prévoir un niveau de prix")
        st.write("Ce sont deux problèmes différents, qui se traitent avec des outils différents. Un signal qui "
                 "n'améliore pas une prévision moyenne peut très bien annoncer un retournement.")

# =============================================================================
# Commun aux deux scénarios : améliorer le travail livré (rapport, 5.3)
# =============================================================================
st.divider()
with st.expander("Quel que soit le scénario : pistes d'amélioration du travail existant"):
    st.table(PISTES.set_index("Piste"))
    st.caption("Rapport final, tableau 23. Les pistes sont classées par la limite qu'elles lèvent.")
