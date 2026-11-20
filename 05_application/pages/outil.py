# =============================================================================
# 05_application/pages/outil.py : page Outil (Théophyle)
# =============================================================================
# Ce que fait la page : l'utilisateur décrit un bien ; la page enchaîne les quatre blocs de l'outil
#   localiser -> estimer -> comparables -> expliquer (ou expliquer_llm)
# et affiche le prix estimé en 2026, le contexte du sous-bassin, la fourchette des biens équivalents,
# les ventes les plus proches et une explication.
#
# Règles (plan de développement et squelette de l'application) :
#   - la page n'appelle que les fonctions de contrats.py : tant qu'un bloc n'est pas livré, c'est sa version
#     provisoire qui répond, au bon format ;
#   - aucune couleur ni police ici : tout vient du thème commun (.streamlit/config.toml) ;
#   - les cas limites (commune inconnue, pas d'estimation, pas assez de ventes) affichent un message clair.
#
# Données lues : donnees/communes_recherche.parquet (liste des communes, 01_preparation/preparer_communes_app.py).
# Sans ce fichier, la page demande le code INSEE à la place de la liste.

import pandas as pd
import streamlit as st

import contrats
from commun_app import afficher_message, charger_parquet, entete
from methodologie_outil import onglet_simulateur

# -----------------------------------------------------------------------------
# 1. Petites fonctions d'affichage
# -----------------------------------------------------------------------------
def nombre(x):
    """3500.0 -> « 3 500 » (espace pour les milliers, à la française)."""
    return format(x, ",.0f").replace(",", " ")


def euros(x):
    return nombre(x) + " €"


def liste_communes():
    """Liste « nom (code postal) » -> code INSEE ; None si le fichier n'est pas encore dans donnees/."""
    try:
        return charger_parquet("communes_recherche.parquet")
    except FileNotFoundError:
        return None
    
@st.cache_data
def noms_sous_bassins():
    """Code du sous-bassin -> nom lisible (« 33063_1 » -> « Bordeaux (centres) »), tiré des ventes comparables.
    Calculé une seule fois : le petit dictionnaire est gardé en mémoire."""
    ventes = charger_parquet("ventes_comparables.parquet")
    paires = ventes[["sous_bassin_de_vie", "nom_sous_bassin"]].astype(str).drop_duplicates("sous_bassin_de_vie")
    return paires.set_index("sous_bassin_de_vie")["nom_sous_bassin"].to_dict()

# -----------------------------------------------------------------------------
# 2. En-tête et options de la colonne de gauche
# -----------------------------------------------------------------------------
entete("Estimer un bien", "Prix estimé en 2026 et ventes comparables, à partir des ventes DVF+")

with st.sidebar:
    avec_llm = st.toggle("Explication rédigée par un modèle de langage", value=True,
                         help="Sinon, une phrase type. En cas d'erreur du modèle de langage, la phrase type est utilisée.")
    nb_comparables = st.slider("Nombre de ventes comparables", min_value=5, max_value=10, value=8)

# -----------------------------------------------------------------------------
# Onglets : l'estimation, puis la méthodologie et les résultats (contenu de Cécile)
# -----------------------------------------------------------------------------
onglet_estimer, onglet_methodo = st.tabs(["Estimer un bien", "Méthodologie et résultats"])

# L'onglet Méthodologie est rempli EN PREMIER : la partie « Estimer » utilise st.stop() quand la saisie
# est incomplète, ce qui arrête toute la page ; rempli après, cet onglet resterait vide.
with onglet_methodo:
    onglet_simulateur()

with onglet_estimer:

    # -----------------------------------------------------------------------------
    # 3. Formulaire : le bien
    # -----------------------------------------------------------------------------
    # Le type de bien est choisi hors du formulaire : il décide si le champ « terrain » s'affiche.
    type_bien = st.radio("Type de bien", ["maison", "appartement"], horizontal=True,
                        format_func=lambda t: t.capitalize())

    with st.form("formulaire_bien"):   # nom différent de st.session_state["bien"], où l'on garde le bien saisi:
        communes = liste_communes()
        if communes is not None:
            nom_commune = st.selectbox("Commune", communes["libelle"], index=None,
                                    placeholder="Tapez le nom de la commune, par exemple Bordeaux")
            code_commune = None
            if nom_commune is not None:
                code_commune = communes.loc[communes["libelle"] == nom_commune, "code_insee"].iloc[0]
        else:
            code_commune = st.text_input("Code INSEE de la commune", placeholder="33063")
            nom_commune = code_commune

        colonne_1, colonne_2 = st.columns(2)
        with colonne_1:
            sbati = st.number_input("Surface habitable (m²)", min_value=9, max_value=1000, value=70, step=1)
            nb_pieces = st.number_input("Nombre de pièces principales", min_value=1, max_value=20, value=3, step=1)
        with colonne_2:
            if type_bien == "maison":
                sterr = st.number_input("Surface du terrain (m²)", min_value=0, max_value=100000, value=400, step=10)
            else:
                sterr = 0
            nblocdep = st.number_input("Nombre de dépendances (garage, cave, abri…)", min_value=0, max_value=10,
                                    value=0, step=1)

        valide = st.form_submit_button("Estimer", type="primary")

    # Le bien est gardé en mémoire : changer une option à gauche ne fait pas disparaître le résultat
    if valide:
        st.session_state["bien"] = {"commune": code_commune, "type_bien": type_bien, "sbati": float(sbati),
                                    "nb_pieces": int(nb_pieces), "sterr": float(sterr), "nblocdep": int(nblocdep)}
        st.session_state["nom_commune"] = nom_commune

    if "bien" not in st.session_state:
        st.info("Décrivez le bien, puis cliquez sur « Estimer ».")
        st.stop()

    bien = st.session_state["bien"]
    nom_commune = st.session_state["nom_commune"]


    # -----------------------------------------------------------------------------
    # 4. Contrôles et localisation
    # -----------------------------------------------------------------------------
    if not bien["commune"]:
        afficher_message("Choisissez une commune.", "avertissement")
        st.stop()

    problemes = contrats.verifier_bien(bien)
    if problemes:
        afficher_message("Saisie à corriger : " + " ; ".join(problemes), "erreur")
        st.stop()

    lieu = contrats.localiser(bien["commune"])
    if lieu["sous_bassin_de_vie"] is None:
        afficher_message(lieu["message"], "avertissement")
        st.stop()
    afficher_message(lieu["message"], "info")


    # -----------------------------------------------------------------------------
    # 5. Estimation, ventes comparables, explication
    # -----------------------------------------------------------------------------
    estimation = contrats.estimer(bien)
    ventes_proches = contrats.comparables(bien, k=nb_comparables)

    phrase_type = contrats.expliquer(estimation, ventes_proches)
    explication = phrase_type
    if avec_llm:
        explication = contrats.expliquer_llm(estimation, ventes_proches)
    # Vrai seulement si le texte vient vraiment du modèle de langage (en cas de repli, c'est la phrase type)
    genere_par_ia = explication != phrase_type


    # -----------------------------------------------------------------------------
    # 6. Résultat : le prix estimé et le contexte du territoire
    # -----------------------------------------------------------------------------
    st.subheader(nom_commune + " · " + type_bien + " de " + nombre(bien["sbati"]) + " m²")
    afficher_message(estimation["message"], "info")

    if estimation["prix_m2"] is not None:
        contexte = estimation["contexte"]
        colonnes = st.columns(3)
        colonnes[0].metric("Prix estimé en " + str(estimation["annee"]), euros(estimation["prix_total"]))
        colonnes[1].metric("Soit, au m²", euros(estimation["prix_m2"]))
        if contexte is not None:
            evolution = None
            if contexte["evolution_pct"] is not None:
                evolution = format(contexte["evolution_pct"], "+.1f").replace(".", ",") + " % sur un an"
            colonnes[2].metric("Prix médian 2025 du secteur, au m²", euros(contexte["prix_median_2025"]),
                            delta=evolution, delta_color="off")
            # Nom lisible du secteur ; le code seulement si le nom est introuvable
            code_secteur = estimation["sous_bassin_de_vie"]
            nom_secteur = noms_sous_bassins().get(code_secteur, code_secteur)
            st.caption("Secteur : sous-bassin « " + nom_secteur + " », "
                       + nombre(contexte["nb_ventes_2025"]) + " ventes en 2025.")


    # -----------------------------------------------------------------------------
    # 7. Fourchette des biens équivalents
    # -----------------------------------------------------------------------------
    st.subheader("Prix des biens équivalents")
    fourchette = ventes_proches["fourchette"]
    if fourchette is None:
        afficher_message(ventes_proches["message"], "avertissement")
    else:
        basse, haute = fourchette["basse_m2"], fourchette["haute_m2"]
        st.write("Sur les " + str(contrats.NB_ANNEES_FOURCHETTE) + " dernières années, **" + str(fourchette["nb_ventes"])
                + " ventes** de biens équivalents (" + fourchette["niveau"] + ") ont eu lieu dans le "
                + fourchette["echelle"] + " « " + fourchette["lieu"] + " ». Pour " + str(contrats.PART_FOURCHETTE)
                + " % d'entre elles, le prix au m² était compris entre **" + euros(basse) + "** et **"
                + euros(haute) + "**.")
        # L'estimation vient du modèle, la fourchette des ventes passées : elles peuvent ne pas concorder
        if estimation["prix_m2"] is not None:
            if estimation["prix_m2"] > haute:
                afficher_message("L'estimation est au-dessus de la plupart des ventes équivalentes : le modèle tient "
                                "compte des caractéristiques du bien (surface, terrain, dépendances) et de l'évolution "
                                "récente du secteur.", "info")
            elif estimation["prix_m2"] < basse:
                afficher_message("L'estimation est en dessous de la plupart des ventes équivalentes : le modèle tient "
                                "compte des caractéristiques du bien (surface, terrain, dépendances) et de l'évolution "
                                "récente du secteur.", "info")


    # -----------------------------------------------------------------------------
    # 8. Ventes les plus proches
    # -----------------------------------------------------------------------------
    st.subheader("Ventes les plus proches")
    ventes = ventes_proches["ventes"]
    if ventes_proches["maille"] == "bassin de vie":
        afficher_message("Peu de ventes dans le secteur : la recherche a été élargie au bassin de vie.", "info")
    if ventes_proches["fourchette"] is not None:
        afficher_message(ventes_proches["message"], "info")
    if len(ventes) == 0:
        afficher_message("Aucune vente comparable trouvée.", "avertissement")
    else:
        tableau = ventes.drop(columns="distance")
        if type_bien == "appartement":
            tableau = tableau.drop(columns="sterr")
        st.dataframe(
            tableau, hide_index=True, use_container_width=True,
            column_config={
                "commune": st.column_config.TextColumn("Commune"),
                "date": st.column_config.DateColumn("Date de vente", format="MM/YYYY"),   # DVF+ ne donne que le mois,
                "sbati": st.column_config.NumberColumn("Surface (m²)", format="%.0f"),
                "nb_pieces": st.column_config.NumberColumn("Pièces", format="%d"),
                "sterr": st.column_config.NumberColumn("Terrain (m²)", format="%.0f"),
                "prix_m2": st.column_config.NumberColumn("Prix au m² (€)", format="%.0f"),
            })
        st.caption("Ventes du même type, classées de la plus ressemblante à la moins ressemblante "
                "(surface, pièces, terrain). Ni adresse ni parcelle ne sont affichées.")


    # -----------------------------------------------------------------------------
    # 9. Explication et limites
    # -----------------------------------------------------------------------------
    st.subheader("En résumé")
    st.write(explication)
    if genere_par_ia:
        st.caption("*Texte généré par IA.*")
    st.caption("Limite : le modèle estime un bien à partir des prix de l'année précédente et de l'évolution récente. "
            "Il n'anticipe pas un retournement du marché. La fourchette décrit des ventes passées, ce n'est pas "
            "une garantie de prix.")
