# =============================================================================
# 05_application/pages/eda.py : page Exploration des données (Christian)
# =============================================================================
# Ce que fait la page : reprend les quatre thèmes du rendu 1 et du rendu 2, avec des filtres.
# Chaque onglet suit le même parcours : un message clé (« À retenir »), puis des questions dans l'ordre où le lecteur
# se les pose, puis le détail dans un menu repliable.
#   1. Sources          : quelles données, sur quelles années, avec quelles limites (texte du rapport 2, partie 2)
#   2. Phases du marché : prix et ventes par année (filtres : type de bien, années, département)
#   3. Territoires      : le « où », prix par département (filtres : type de bien, année)
#   4. Maille et seuil  : pourquoi un seuil, ce qu'il écarte (filtre : année)
#
# Données lues (produites par 04_outils/preparer_eda_app.py, copiées dans donnees/) :
#   eda_territoires.parquet et eda_seuil.parquet.
# Règles de l'application : aucune couleur ni police ici (thème commun), seulement les fonctions de commun_app.

import pandas as pd
import json

import altair as alt
import streamlit as st

from commun_app import DONNEES, afficher_message, entete


@st.cache_data
def _lire_parquet(nom, date_fichier):
    # date_fichier fait partie de la clé de cache : un parquet régénéré est relu sans redémarrer le serveur
    return pd.read_parquet(DONNEES / nom)


def charger_parquet(nom):
    chemin = DONNEES / nom
    if not chemin.exists():
        raise FileNotFoundError(nom + " absent de " + str(DONNEES))
    return _lire_parquet(nom, chemin.stat().st_mtime_ns)

TYPES = ["maison", "appartement"]

# Départements métropolitains : les deux premiers caractères du code du sous-bassin de vie
DEPARTEMENTS = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence", "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes", "09": "Ariège", "10": "Aube", "11": "Aude",
    "12": "Aveyron", "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs", "26": "Drôme",
    "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère", "30": "Gard", "31": "Haute-Garonne", "32": "Gers",
    "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire",
    "38": "Isère", "39": "Jura", "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère",
    "49": "Maine-et-Loire", "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord",
    "60": "Oise", "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhône",
    "70": "Haute-Saône", "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres",
    "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse", "85": "Vendée",
    "86": "Vienne", "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis", "94": "Val-de-Marne", "95": "Val-d'Oise",
}

# Sources : texte repris du rapport de recherche (rendu 2, partie 2)
# (nom, ce qu'elle apporte, limite en quelques mots, précision sur la limite)
SOURCES = [
    ("DVF+ (Cerema, DGFiP)", "Les ventes de logements : date, valeur foncière, type, surfaces, commune. 2014 à 2025.",
     "Ventes réalisées seulement", "Ni les invendus, ni les prix demandés. L'Alsace, la Moselle et Mayotte sont absentes (livre foncier)."),
    ("Taux des crédits immobiliers (Banque de France)", "Taux moyen des crédits nouveaux à l'habitat, chaque mois depuis 1998.",
     "Valeur nationale", "La même pour tous les territoires un mois donné."),
    ("Niveau de vie (Filosofi, Insee)", "Niveau de vie médian des habitants par commune, de 2014 à 2021.",
     "10 % de communes masquées", "Masquées en 2021 : aucun indicateur sous 50 ménages et 100 personnes. 2021 est le dernier millésime."),
    ("Délinquance enregistrée (SSMSI)", "Faits enregistrés par la police et la gendarmerie, 15 indicateurs, de 2016 à 2025.",
     "46 % des lignes masquées", "Faits enregistrés, pas délinquance réelle. Une ligne masquée reçoit la moyenne du département."),
    ("Parc de logements (recensement, Insee)", "Statut d'occupation, taille, époque de construction, vacance, résidences secondaires, par IRIS. Le recensement 2022 agrège des enquêtes de 2020 à 2024.",
     "Estimations fragiles", "Surtout dans les petits IRIS (estimations pondérées)."),
    ("Logements vacants (LOVAC)", "Logements privés vacants par commune, millésimes 2020 à 2026 ; le millésime N décrit le 1er janvier N-1.",
     "Parc privé seul", "Série rompue à partir du millésime 2025 (7,2 % de vacance contre environ 11 % avant). Parc privé total masqué pour environ 53 % des communes."),
    ("Équipements (BPE, Insee)", "Commerces, services, santé, transport, sport, tourisme ; exhaustive. 2015, 2020 et 2025.",
     "Maille bassin de vie seulement", "Le fichier comparable dans le temps (2015, 2020, 2025) ne donne que le bassin de vie. L'enseignement est exclu."),
    ("Emploi salarié privé (Urssaf)", "Salariés et établissements employeurs au 31 décembre, de 2006 à 2025.",
     "Varie chaque année", "Elle change chaque année dans chaque territoire : elle sert à tester l'hypothèse H3c. Seul le secteur privé est couvert."),
]

# Années couvertes (frise de l'onglet Sources) : (source, début, fin, forme, libellé de la période)
# La frise commence en 2005 : la série des taux, qui commence en 1998, est coupée à gauche.
# Périodes : Tableau 2 du rapport de recherche (LOVAC : millésimes 2020 à 2026).
FRISE_DEBUT = 2005
PERIODES = [
    ("DVF+", 2014, 2025, "barre", "2014 à 2025"),
    ("Taux des crédits", FRISE_DEBUT, 2025, "barre", "depuis 1998, chaque mois"),
    ("Niveau de vie", 2014, 2021, "barre", "2014 à 2021"),
    ("Délinquance", 2016, 2025, "barre", "2016 à 2025"),
    ("Parc de logements", 2016, 2022, "barre", "2016 à 2022"),
    ("Logements vacants", 2020, 2026, "barre", "millésimes 2020 à 2026"),
    ("Équipements", 2015, 2015, "point", "2015"),
    ("Équipements", 2020, 2020, "point", "2020"),
    ("Équipements", 2025, 2025, "point", "2025"),
    ("Emploi salarié privé", 2006, 2025, "barre", "2006 à 2025"),
]

ECARTEES = [      # partie 2.9 du rapport de recherche
    ("Annonces immobilières du Haut-Rhin", "Un seul département, période courte (2019-2023), prix affichés et non prix de vente. "
                                           "Réservées au second axe du projet."),
]

# Tableau 3 du rapport de recherche (rendu 2, partie 3.2) : (bassin de vie, sous-bassin)
TABLEAU_3 = {
    "Territoires à prédire en 2024": {"Appartements": (692, 823), "Maisons": (1606, 1777)},
    "Lignes d'entraînement 2020-2023": {"Appartements": (2962, 3508), "Maisons": (6433, 7124)},
}
MAILLES = pd.DataFrame([
    ("Bassin de vie", "692 / 1 606", "2 962 / 6 433", "5,6 % / 6,0 %"),
    ("Sous-bassin", "823 / 1 777", "3 508 / 7 124", "5,6 % / 5,9 %"),
], columns=["Maille", "Territoires 2024 (appart. / maisons)", "Lignes 2020-2023 (appart. / maisons)",
            "Erreur de la naïve 2024 (appart. / maisons)"])


PHASES = [("Stabilité", 2015, 2018), ("Hausse", 2018, 2022), ("Retournement", 2022, 2024)]   # lecture sur la France entière
SEUIL_RETENU = 20
BASSIN_PARIS = "75056"      # sous-bassins 75056_1, _2 et _3 : bassin de vie de Paris
CENTRES_ALSACE = "57-67-68"  # sous-bassins au code d'un centre alsacien ou mosellan ; leurs ventes sont en 54, 88 ou 90


def departement(code):
    if code == BASSIN_PARIS:
        return "Bassin de vie de Paris"
    if code == CENTRES_ALSACE:
        return "Sous-bassins au nom d'un centre alsacien ou mosellan"
    return DEPARTEMENTS.get(code, "département " + code)


@st.cache_data
def fond_departements():
    """Contours des départements métropolitains (donnees/departements_metropole.geojson), lus une seule fois."""
    chemin = DONNEES / "departements_metropole.geojson"
    if not chemin.exists():
        raise FileNotFoundError("departements_metropole.geojson absent de " + str(DONNEES))
    return json.loads(chemin.read_text(encoding="utf-8"))


@st.cache_data
def territoires():
    t = charger_parquet("eda_territoires.parquet").copy()
    t["departement"] = t["sous_bassin_de_vie"].str[:2]
    # Les trois sous-bassins du bassin de vie de Paris couvrent l'Île-de-France : ils ne sont pas dans le 75
    t.loc[t["sous_bassin_de_vie"].str.startswith(BASSIN_PARIS + "_"), "departement"] = BASSIN_PARIS
    # DVF+ ne couvre pas 57, 67 et 68 : un sous-bassin à ce code vient de communes voisines (54, 88, 90), pas d'Alsace-Moselle
    t.loc[t["departement"].isin(["57", "67", "68"]), "departement"] = CENTRES_ALSACE
    return t


def fr(x, d=0):
    """Nombre à la française : espace insécable pour les milliers, virgule pour les décimales."""
    return format(x, ",." + str(d) + "f").replace(",", " ").replace(".", ",")


def signe(x, d=0):
    return ("+" if x >= 0 else "−") + fr(abs(x), d)


def cle(titre, sous_titre=None):
    """Le message à retenir de l'onglet, calculé sur les données affichées."""
    with st.container(border=True):
        st.caption("À RETENIR")
        st.markdown("### " + titre)
        if sous_titre:
            st.write(sous_titre)


def question(numero, titre):
    st.caption("QUESTION " + str(numero))
    st.subheader(titre)


entete("Exploration des données", "Sources, phases du marché, territoires et maille")

onglets = st.tabs(["Sources", "Phases du marché", "Territoires", "Maille et seuil"])

# -----------------------------------------------------------------------------
# 1. Sources
# -----------------------------------------------------------------------------
with onglets[0]:
    with st.container(border=True):
        st.caption("À RETENIR")
        st.markdown("### DVF+ donne les prix. **Sept sources publiques** décrivent les territoires, avec des limites "
                    "qu'il faut connaître.")
        st.write("Elles ne couvrent pas les mêmes années, et plusieurs sont en partie masquées : 10 % des communes "
                 "pour le niveau de vie en 2021, 46 % des lignes pour la délinquance, environ 53 % des communes pour le parc "
                 "privé total des logements vacants.")

    question(1, "Quelles données, sur quelles années ?")
    st.write("Les ventes (DVF+) couvrent 2014 à 2025. Chaque source décrit les territoires sur sa propre période.")
    frise = pd.DataFrame(PERIODES, columns=["source", "debut", "fin", "forme", "periode"])
    ordre_sources = list(dict.fromkeys(frise["source"]))
    echelle = alt.Scale(domain=[FRISE_DEBUT, 2026.6], nice=False)
    axe = alt.Axis(format="d", values=[2005, 2010, 2015, 2020, 2025])
    axe_y = alt.Y("source:N", title=None, sort=ordre_sources, axis=alt.Axis(labelLimit=220))
    info = [alt.Tooltip("source:N", title="Source"), alt.Tooltip("periode:N", title="Période")]
    barres = alt.Chart(frise[frise["forme"] == "barre"]).mark_bar(size=14).encode(
        x=alt.X("debut:Q", title=None, scale=echelle, axis=axe), x2="fin:Q", y=axe_y, tooltip=info)
    points = alt.Chart(frise[frise["forme"] == "point"]).mark_point(filled=True, size=130, opacity=1).encode(
        x=alt.X("debut:Q", title=None, scale=echelle, axis=axe), y=axe_y, tooltip=info)
    st.altair_chart((barres + points).properties(height=300), use_container_width=True)
    st.caption("Un point : une année de référence. La série des taux commence en 1998 : la frise est coupée à gauche, en 2005. "
               "LOVAC : le millésime N décrit le 1er janvier N-1.")

    question(2, "Que faut-il retenir de chaque source ?")
    st.write("Chaque carte donne ce que la source apporte, puis sa limite principale.")
    for debut in range(0, len(SOURCES), 3):
        colonnes = st.columns(3)
        for colonne, (nom, apport, limite, precision) in zip(colonnes, SOURCES[debut:debut + 3]):
            with colonne.container(border=True):
                st.markdown("**" + nom + "**")
                st.write(apport)
                st.badge(limite, color="primary")
                st.caption(precision)

    question(3, "Qu'avons-nous écarté, et pourquoi ?")
    for nom, pourquoi in ECARTEES:
        with st.container(border=True):
            st.markdown("**" + nom + "**")
            st.caption(pourquoi)
    st.caption("Source : rapport de recherche, partie 2.")

# -----------------------------------------------------------------------------
# 2. Phases du marché
# -----------------------------------------------------------------------------
with onglets[1]:
    try:
        t = territoires()
    except FileNotFoundError as erreur:
        afficher_message(str(erreur) + " : lancer 04_outils/preparer_eda_app.py.", "avertissement")
    else:
        zone_cle = st.container()        # remplie plus bas, une fois les filtres connus
        question(1, "Que s'est-il passé depuis 2015 ?")
        st.write("Trois phases : des prix presque stables jusqu'en 2018, une hausse de 2019 à 2022, puis un "
                 "retournement en 2023-2024. Les ventes baissent avant les prix.")
        colonne_1, colonne_2, colonne_3 = st.columns(3)
        with colonne_1:
            types = st.multiselect("Type de bien", TYPES, default=TYPES, key="phases_types",
                                   format_func=lambda x: x.capitalize())
        with colonne_2:
            annee_min, annee_max = int(t["annee"].min()), int(t["annee"].max())
            periode = st.slider("Années", annee_min, annee_max, (annee_min, annee_max), key="phases_annees")
        with colonne_3:
            codes = sorted(t["departement"].unique())
            choix = st.selectbox("Département", ["Tous"] + codes, key="phases_dep",
                                 format_func=lambda c: "Tous" if c == "Tous" else c + " – " + departement(c))
        vue = t[t["type_bien"].isin(types) & t["annee"].between(*periode)]
        if choix != "Tous":
            vue = vue[vue["departement"] == choix]
        if vue.empty or vue["annee"].nunique() < 2:
            afficher_message("Choisir au moins un type de bien et deux années avec des ventes.", "avertissement")
        else:
            debut, fin = int(vue["annee"].min()), int(vue["annee"].max())
            # Prix de l'année : moyenne des prix des sous-bassins pondérée par leur nombre de ventes
            donnees = (vue.assign(prix_x_ventes=vue["prix"] * vue["n_ventes"]).groupby(["annee", "type_bien"])
                       .agg(prix_x_ventes=("prix_x_ventes", "sum"), ventes=("n_ventes", "sum")).reset_index())
            donnees["prix"] = donnees["prix_x_ventes"] / donnees["ventes"]
            prix = donnees.assign(serie=donnees["type_bien"].map({"appartement": "Appartements", "maison": "Maisons"})
                                  ).sort_values(["serie", "annee"]).reset_index(drop=True)
            prix["annee"] = prix["annee"].astype(int)
            prix["indice"] = 100 * prix["prix"] / prix.groupby("serie")["prix"].transform("first")
            prix["valeur"] = prix["prix"].round(0).astype(int).map(fr) + " €/m²"
            ventes = vue.groupby("annee")["n_ventes"].sum().reset_index().sort_values("annee").reset_index(drop=True)
            ventes["annee"] = ventes["annee"].astype(int)
            ventes["indice"] = 100 * ventes["n_ventes"] / ventes["n_ventes"].iloc[0]
            ventes["serie"] = "Ventes"
            ventes["valeur"] = ventes["n_ventes"].astype(int).map(fr) + " ventes"

            def sommets(df):
                """Année du maximum de chaque série, seulement si ce n'est ni la première ni la dernière année."""
                lignes = []
                for _, groupe in df.groupby("serie"):
                    ligne = groupe.loc[groupe["indice"].idxmax()]
                    if debut < ligne["annee"] < fin:
                        lignes.append(ligne)
                if not lignes:
                    return pd.DataFrame(columns=["annee", "indice", "serie", "texte"])
                res = pd.DataFrame(lignes).sort_values("indice", ascending=False).drop_duplicates("annee")
                return res.assign(texte="Sommet " + res["annee"].astype(str))[["annee", "indice", "serie", "texte"]]

            def fin_de_serie(df, nom):
                """Dernière valeur de chaque série, écrite au bout de la courbe."""
                derniere = df.sort_values("annee").groupby("serie").tail(1).copy()
                derniere["texte"] = derniere["indice"].map(lambda v: signe(v - 100) + " % ") + nom(derniere)
                return derniere

            echelle_x = alt.Scale(domain=[debut - 0.2, fin + 2.4], nice=False)      # place pour les étiquettes de fin
            axe_x = alt.Axis(format="d", values=list(range(debut, fin + 1)))
            x_annee = alt.X("annee:Q", title=None, axis=axe_x, scale=echelle_x)

            def panneau(df, couleur, hauteur, titre_y, nom):
                base = alt.Chart(df)
                bas_y, haut_y = min(df["indice"].min(), 100), df["indice"].max()
                marge = (haut_y - bas_y) * 0.06
                echelle_y = alt.Scale(domain=[bas_y - marge, haut_y + (haut_y - bas_y) * 0.22], nice=False)
                lignes = base.mark_line(point=True).encode(
                    x=x_annee, y=alt.Y("indice:Q", title=titre_y, scale=echelle_y), color=couleur,
                    tooltip=[alt.Tooltip("annee:Q", title="Année", format="d"), alt.Tooltip("serie:N", title="Série"),
                             alt.Tooltip("indice:Q", title="Indice", format=".1f"), alt.Tooltip("valeur:N", title="Valeur")])
                if "Maisons" in set(df["serie"]):
                    lignes = lignes.encode(strokeDash=alt.StrokeDash(
                        "serie:N", legend=None, scale=alt.Scale(domain=["Appartements", "Maisons"], range=[[1, 0], [6, 4]])))
                repere = alt.Chart(pd.DataFrame({"indice": [100]})).mark_rule(strokeDash=[3, 3], opacity=0.5).encode(y=alt.Y("indice:Q", scale=echelle_y))
                bout = alt.Chart(fin_de_serie(df, nom)).mark_text(align="left", dx=10, fontWeight="bold").encode(
                    x=x_annee, y=alt.Y("indice:Q", scale=echelle_y), text="texte:N", color=couleur)
                haut = alt.Chart(sommets(df))
                cercles = haut.mark_point(size=150, filled=False, strokeWidth=2, color="gray").encode(x=x_annee, y=alt.Y("indice:Q", scale=echelle_y))
                noms_sommets = haut.mark_text(dy=-12, fontWeight="bold").encode(x=x_annee, y=alt.Y("indice:Q", scale=echelle_y), text="texte:N")
                couches = repere + lignes + bout + cercles + noms_sommets
                if choix == "Tous":        # les phases se lisent sur la France entière
                    phases = pd.DataFrame(PHASES, columns=["nom", "debut", "fin"])
                    phases = phases[(phases["fin"] > debut) & (phases["debut"] < fin)].copy()
                    phases["debut"], phases["fin"] = phases["debut"].clip(lower=debut), phases["fin"].clip(upper=fin)
                    phases["milieu"] = (phases["debut"] + phases["fin"]) / 2
                    fond = alt.Chart(phases[phases["nom"] == "Hausse"]).mark_rect(opacity=0.10).encode(
                        x=alt.X("debut:Q", scale=echelle_x), x2="fin:Q")
                    couches = fond + couches
                    if titre_y.startswith("Prix"):
                        noms = alt.Chart(phases).mark_text(baseline="top", dy=2).encode(
                            x=alt.X("milieu:Q", scale=echelle_x), y=alt.value(0), text="nom:N")
                        couches = couches + noms
                return couches.properties(height=hauteur)

            couleur_prix = alt.Color("serie:N", legend=None, scale=alt.Scale(domain=["Appartements", "Maisons"]))
            couleur_ventes = alt.Color("serie:N", legend=None, scale=alt.Scale(domain=["Ventes"]))
            st.write("**Prix au m²** (indice : 100 = " + str(debut) + "). Plein : appartements, tireté : maisons.")
            st.altair_chart(panneau(prix, couleur_prix, 260, "Prix, indice (base 100)", lambda d: d["serie"].str.lower()),
                            use_container_width=True)
            st.write("**Nombre de ventes**, tous types de bien choisis (indice : 100 = " + str(debut) + ")")
            st.altair_chart(panneau(ventes, couleur_ventes, 190, "Ventes, indice (base 100)", lambda d: d["serie"].str.lower()),
                            use_container_width=True)
            st.caption("Lecture : un indice de 120 veut dire +20 % depuis " + str(debut) + ". Les deux graphiques sont séparés pour "
                       "que les prix, qui varient peu, ne soient pas écrasés par les ventes, qui varient beaucoup. "
                       + ("Zone claire : phase de hausse."
                          + (" 2025 n'appartient à aucune des trois phases du rapport, qui décrit une année stable (+1,5 % pour "
                             "les appartements et +0,9 % pour les maisons, en médiane des sous-bassins : une autre mesure que la "
                             "moyenne pondérée affichée ici)." if fin == 2025 else "")
                          if choix == "Tous" else "Les phases du rapport décrivent la France entière : elles ne s'affichent pas pour un département."))
            with st.expander("Voir les valeurs (€/m² et nombre de ventes)"):
                valeurs = pd.concat([prix[["annee", "serie", "prix"]].rename(columns={"prix": "v"}),
                                     ventes[["annee", "serie", "n_ventes"]].rename(columns={"n_ventes": "v"})])
                valeurs = valeurs.pivot(index="annee", columns="serie", values="v").round(0)
                valeurs.columns = [c if c == "Ventes" else "Prix – " + c.lower() + " (€/m²)" for c in valeurs.columns]
                valeurs = valeurs.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else fr(v)))
                valeurs.index = valeurs.index.astype(str)
                valeurs.index.name = "Année"
                st.dataframe(valeurs, use_container_width=True)
            st.caption("Prix : médiane au m² de chaque sous-bassin, puis moyenne des sous-bassins pondérée par leur nombre de "
                       "ventes (un grand marché pèse plus qu'un petit). Ce n'est donc pas un prix médian national. "
                       "Seuls les sous-bassins d'au moins 20 ventes dans l'année et le type de bien sont comptés.")

            # Message clé, calculé sur les filtres choisis
            def variation(df, serie):
                s = df[df["serie"] == serie]["indice"]
                return s.iloc[-1] - 100
            variations = [(nom, variation(prix, nom)) for nom in ["Appartements", "Maisons"] if nom in set(prix["serie"])]
            verbe = "gagnent" if all(v >= 0 for _, v in variations) else "évoluent de"
            lieu = "" if choix == "Tous" else " (" + departement(choix) + ")"
            titre = ("De " + str(debut) + " à " + str(fin) + ", les prix au m² " + verbe + " "
                     + " et ".join("**" + signe(v) + " %** pour les " + nom.lower() for nom, v in variations) + lieu + ".")
            phrases = []
            sommet_ventes = sommets(ventes)
            sommet_prix = sommets(prix)
            if not sommet_ventes.empty:
                phrase = "Les ventes atteignent leur sommet en " + str(int(sommet_ventes["annee"].iloc[0]))
                if not sommet_prix.empty and int(sommet_prix["annee"].iloc[0]) != int(sommet_ventes["annee"].iloc[0]):
                    phrase += ", les prix en " + str(int(sommet_prix["annee"].iloc[0]))
                phrases.append(phrase + ".")
            if fin == 2025 and debut <= 2024:
                dernier, avant = ventes["n_ventes"].iloc[-1], ventes["n_ventes"].iloc[-2]
                phrases.append("En 2025, les ventes " + ("repartent" if dernier >= avant else "reculent") + " ("
                               + signe(100 * (dernier / avant - 1)) + " % sur un an), hors des trois phases du rapport.")
            with zone_cle:
                cle(titre, " ".join(phrases) or None)

# -----------------------------------------------------------------------------
# 3. Territoires : le « où »
# -----------------------------------------------------------------------------
with onglets[2]:
    try:
        t = territoires()
    except FileNotFoundError as erreur:
        afficher_message(str(erreur) + " : lancer 04_outils/preparer_eda_app.py.", "avertissement")
    else:
        zone_cle = st.container()
        question(1, "Où les prix sont-ils élevés ?")
        st.write("La localisation est le facteur le plus fort du prix. Les variables de territoire disent où les prix "
                 "sont élevés, pas quand ils changent.")
        colonne_1, colonne_2 = st.columns(2)
        with colonne_1:
            type_bien = st.radio("Type de bien", TYPES, horizontal=True, key="ou_type",
                                 format_func=lambda x: x.capitalize())
        with colonne_2:
            annees = sorted(t["annee"].unique())
            annee = st.select_slider("Année", annees, value=annees[-1], key="ou_annee")
        vue = t[(t["type_bien"] == type_bien) & (t["annee"] == annee)]
        tout = vue.groupby("departement").agg(prix=("prix", "median"), sous_bassins=("prix", "size"),
                                              ventes=("n_ventes", "sum")).reset_index()
        tout = tout[tout["sous_bassins"] >= 3]      # un département à 1 ou 2 sous-bassins ne dit rien du département
        tout["Département"] = tout["departement"].map(lambda c: c + " – " + departement(c))
        paris = tout[tout["departement"] == BASSIN_PARIS]
        centres = tout[tout["departement"] == CENTRES_ALSACE]
        par_dep = tout[~tout["departement"].isin([BASSIN_PARIS, CENTRES_ALSACE])].sort_values("prix", ascending=False)   # carte et classement
        if par_dep.empty:
            afficher_message("Aucun territoire pour ce choix.", "avertissement")
        else:
            haut, bas = par_dep.iloc[0], par_dep.iloc[-1]
            carte_affichee, sans_valeur = False, []
            gauche, droite = st.columns([1, 1])
            with gauche:
                try:
                    fond = fond_departements()
                except FileNotFoundError as erreur:
                    afficher_message(str(erreur) + " : à placer dans 05_application/donnees.", "avertissement")
                else:
                    carte = par_dep[["departement", "Département", "prix", "sous_bassins", "ventes"]].copy()
                    carte["prix"] = carte["prix"].round(0)
                    chaleur = alt.Chart(alt.Data(values=fond["features"])).mark_geoshape(stroke="white", strokeWidth=0.6).transform_lookup(
                        lookup="properties.code",
                        from_=alt.LookupData(carte, "departement", ["Département", "prix", "sous_bassins", "ventes"]),
                    ).encode(
                        color=alt.Color("prix:Q", title="Prix médian au m² (€)",
                                        scale=alt.Scale(type="quantile", scheme=alt.SchemeParams(name="blues", count=6)),
                                        legend=alt.Legend(orient="bottom", format="d", type="symbol", symbolType="square", columns=3)),
                        tooltip=[alt.Tooltip("properties.nom:N", title="Département"), alt.Tooltip("prix:Q", title="Prix au m² (€)", format="d"),
                                 alt.Tooltip("sous_bassins:Q", title="Sous-bassins"), alt.Tooltip("ventes:Q", title="Ventes")],
                    )
                    socle_gris = alt.Chart(alt.Data(values=fond["features"])).mark_geoshape(fill="lightgray", stroke="white", strokeWidth=0.6)
                    st.altair_chart((socle_gris + chaleur).project(type="conicConformal", parallels=[44, 49], rotate=[-3, 0]).properties(height=620),
                                    use_container_width=True)
                    presents = set(carte["departement"])
                    sans_valeur = sorted(f["properties"]["code"] + " – " + f["properties"]["nom"]
                                         for f in fond["features"] if f["properties"]["code"] not in presents)
                    carte_affichee = True
                    bornes = [par_dep["prix"].min()] + [par_dep["prix"].quantile(k / 6) for k in range(1, 6)] + [par_dep["prix"].max()]
                    alsace = [c for c in ("57", "67", "68") if any(d.startswith(c + " ") for d in sans_valeur)]
                    st.caption("Six classes de même effectif, bornes en €/m² : " + " | ".join(fr(v) for v in bornes) + ". En gris : "
                               + str(len(sans_valeur)) + " départements sans valeur"
                               + (", dont l'Alsace-Moselle, non couverte par DVF+" if alsace else "")
                               + ". Survole un département pour voir son prix, ses sous-bassins et ses ventes.")
            with droite:
                maxi = float(haut["prix"])

                def classement(df, titre_classement):
                    df = df.assign(texte=df["prix"].round(0).astype(int).map(fr) + " €")
                    barres = alt.Chart(df).mark_bar().encode(
                        y=alt.Y("Département:N", sort=None, title=None, axis=alt.Axis(labelLimit=170)),
                        x=alt.X("prix:Q", title=None, axis=None, scale=alt.Scale(domain=[0, maxi * 1.22])))
                    texte = barres.mark_text(align="left", dx=4).encode(text="texte:N")
                    st.write("**" + titre_classement + "**")
                    st.altair_chart((barres + texte).properties(height=140), use_container_width=True)

                classement(par_dep.head(5), "Les 5 plus chers")
                classement(par_dep.tail(5).iloc[::-1], "Les 5 moins chers")
                st.caption(str(len(par_dep)) + " départements classés, prix médian au m² en €. Le tableau complet est dans « Détails et méthode ».")
                if not paris.empty:
                    st.caption("Bassin de vie de Paris (sous-bassins 75056_1, _2 et _3) : " + fr(float(paris["prix"].iloc[0]))
                               + " €/m². Il couvre l'Île-de-France : présenté à part, hors carte et hors classement. "
                               "« 75 – Paris » ne regroupe que les arrondissements.")
                if not centres.empty:
                    st.caption(str(int(centres["sous_bassins"].iloc[0])) + " sous-bassins portent le code d'un centre d'Alsace-Moselle (57, 67 ou 68) "
                               "mais leurs ventes sont dans les départements voisins (54, 88, 90) : ils sont hors carte et hors classement.")

            question(2, "Comment lire la carte ?")
            st.write("Couleur : prix médian au m² des sous-bassins du département, en six classes de même effectif. En gris : "
                     "département sans valeur, c'est-à-dire moins de 3 sous-bassins d'au moins 20 ventes, ou aucun "
                     "sous-bassin rattaché.")
            with st.expander("Détails et méthode"):
                st.write("**Détail par département**")
                configuration = {
                    "prix": st.column_config.ProgressColumn("Prix médian au m² (€)", format="%.0f", min_value=0,
                                                            max_value=float(haut["prix"])),
                    "sous_bassins": st.column_config.NumberColumn("Sous-bassins", format="%d"),
                    "ventes": st.column_config.NumberColumn("Ventes", format="%d"),
                }
                st.dataframe(tout.sort_values("prix", ascending=False)[["Département", "prix", "sous_bassins", "ventes"]],
                             hide_index=True, column_config=configuration, use_container_width=True)
                if sans_valeur:
                    st.write("**" + str(len(sans_valeur)) + " départements sans valeur** : " + ", ".join(sans_valeur))
                st.write("**Méthode**")
                points = [
                    "- Le prix d'un département est la médiane des prix au m² de ses sous-bassins, sans pondérer par le "
                    "nombre de ventes. Seuls les sous-bassins d'au moins 20 ventes sont comptés, et les départements "
                    "d'au moins 3 sous-bassins.",
                    "- Un sous-bassin est rattaché au département de son chef-lieu : les grands bassins découpés par "
                    "densité (codes se terminant par _1, _2 ou _3) peuvent s'étendre sur plusieurs départements.",
                    "- Le bassin de vie de Paris (codes 75056_x) n'est pas un département : il est présenté à part. Il compte "
                    "440 communes (rapport, partie 3.2) et est découpé par densité en trois sous-bassins.",
                    "- L'Alsace, la Moselle et Mayotte sont absentes de DVF+ : la publicité foncière y relève du livre foncier "
                    "(rapport, partie 2.1). Quelques sous-bassins portent pourtant le code d'un centre alsacien ou mosellan "
                    "(ex. Sarrebourg, 57630) : leurs ventes viennent de communes voisines, en Meurthe-et-Moselle, dans les Vosges ou "
                    "dans le Territoire de Belfort. Ils sont écartés du classement par département.",
                ]
                sans_sous_bassin = [c for c in ("92", "93", "94") if any(d.startswith(c + " ") for d in sans_valeur)]
                if sans_sous_bassin:
                    points.append("- Les départements " + ", ".join(sans_sous_bassin) + " n'ont aucun sous-bassin à leur nom dans "
                                  "les données : ils sont en gris.")
                if carte_affichee:
                    points.append("- Fond de carte : IGN, Admin Express (Licence ouverte), via github.com/gregoiredavid/france-geojson.")
                st.markdown("\n".join(points))
            # Message clé
            nom_type = "maisons" if type_bien == "maison" else "appartements"
            titre = ("En " + str(int(annee)) + ", le prix médian au m² des " + nom_type + " va de **" + fr(bas["prix"]) + " €** ("
                     + departement(bas["departement"]) + ") à **" + fr(haut["prix"]) + " €** (" + departement(haut["departement"])
                     + ") : un écart de **×\u00a0" + fr(haut["prix"] / bas["prix"], 1) + "**.")
            sous_titre = str(len(par_dep)) + " départements ont une valeur."
            if haut["departement"] == "75" and len(par_dep) > 1:
                sous_titre = ("Paris (75) tire cet écart : sans lui, le rapport tombe à ×\u00a0"
                              + fr(par_dep.iloc[1]["prix"] / bas["prix"], 1) + ".")
            with zone_cle:
                cle(titre, sous_titre)

# -----------------------------------------------------------------------------
# 4. Maille et seuil
# -----------------------------------------------------------------------------
with onglets[3]:
    zone_cle = st.container()
    ORDRE_SERIES = ["Ventes conservées", "Sous-bassins gardés"]      # mêmes couleurs partout dans l'onglet

    question(1, "Pourquoi un seuil de ventes ?")
    st.write("Une médiane calculée sur peu de ventes saute d'une année à l'autre, même quand le marché bouge à peine.")
    # Les deux barres partent de 0. La teinte claire montre la part incertaine ("15 à 18 points" : de 15 jusqu'à 18).
    ecarts = pd.DataFrame([("Hausse réelle du marché", 3, 3, "≈ 3 % : le marché bouge à peine"),
                           ("Médiane avec 20 à 29 ventes", 15, 18, "15 à 18 points : 5 à 6 fois plus que le marché")],
                          columns=["libelle", "plein", "fin", "texte"])
    ecarts["zero"] = 0
    echelle_pts = alt.Scale(domain=[0, 40])
    axe_pts = alt.Axis(values=[0, 5, 10, 15, 20, 25, 30, 35, 40], labelExpr="datum.value > 20 ? '' : datum.label", format="d", title="Points (échelle de 0 à 20)")
    y_pts = alt.Y("libelle:N", title=None, sort=None, axis=alt.Axis(labelLimit=300))
    couleur_pts = alt.Color("libelle:N", legend=None)
    clair = alt.Chart(ecarts).mark_bar(size=22, opacity=0.4).encode(
        y=y_pts, x=alt.X("plein:Q", scale=echelle_pts, axis=axe_pts), x2="fin:Q", color=couleur_pts)
    plein = alt.Chart(ecarts).mark_bar(size=22).encode(
        y=y_pts, x=alt.X("zero:Q", scale=echelle_pts, axis=axe_pts), x2="plein:Q", color=couleur_pts)
    etiquettes = alt.Chart(ecarts).mark_text(align="left", dx=6, fontSize=12).encode(
        y=alt.Y("libelle:N", sort=None), x=alt.X("fin:Q", scale=echelle_pts, axis=axe_pts), text="texte:N")
    st.altair_chart((clair + plein + etiquettes).properties(height=130), use_container_width=True)
    st.caption("Teinte claire : la médiane varie de 15 jusqu'à 18 points selon les cas.")
    st.caption("La médiane varie de 5 à 6 fois plus que le marché : nous avons fixé un seuil de 20 ventes par année et par "
               "type de bien.")

    question(2, "Pourquoi le sous-bassin plutôt que le bassin de vie ?")
    st.write("Découper plus fin donne plus de territoires qui passent le seuil, sans changer l'erreur de la prévision naïve.")
    ordre_mailles = ["Bassin de vie", "Sous-bassin"]
    colonnes = st.columns(2)
    for colonne, (panneau_nom, valeurs) in zip(colonnes, TABLEAU_3.items()):
        lignes = pd.DataFrame([(type_nom, maille, v) for type_nom, (a, b) in valeurs.items()
                               for maille, v in zip(ordre_mailles, (a, b))], columns=["type", "maille", "valeur"])
        lignes["texte"] = lignes["valeur"].map(fr)
        graphique = alt.Chart(lignes).mark_bar().encode(
            y=alt.Y("type:N", title=None, sort=["Appartements", "Maisons"]),
            yOffset=alt.YOffset("maille:N", sort=ordre_mailles),
            x=alt.X("valeur:Q", title=None, axis=None, scale=alt.Scale(domain=[0, lignes["valeur"].max() * 1.2])),
            color=alt.Color("maille:N", title=None, scale=alt.Scale(domain=ordre_mailles), legend=alt.Legend(orient="top")))
        texte = alt.Chart(lignes).mark_text(align="left", dx=4).encode(
            y=alt.Y("type:N", sort=["Appartements", "Maisons"]), yOffset=alt.YOffset("maille:N", sort=ordre_mailles),
            x="valeur:Q", text="texte:N")
        with colonne:
            st.write("**" + panneau_nom + "**")
            st.altair_chart((graphique + texte).properties(height=170), use_container_width=True)
            st.caption("Avec le sous-bassin : " + " ; ".join(nom.lower() + " " + signe(100 * (b / a - 1)) + " %"
                                                              for nom, (a, b) in valeurs.items()))
    st.caption("Erreur de la prévision naïve en 2024 : 5,6 % dans les deux mailles pour les appartements, 6,0 % contre 5,9 % "
               "pour les maisons. Seuil de 20 ventes en N et en N-1 (rapport de recherche, tableau 3).")
    with st.expander("Voir le tableau 3 du rapport de recherche"):
        st.dataframe(MAILLES, hide_index=True, use_container_width=True)

    question(3, "Qu'est-ce qu'on écarte avec un seuil de " + str(SEUIL_RETENU) + " ventes ?")
    try:
        seuils = charger_parquet("eda_seuil.parquet")
    except FileNotFoundError as erreur:
        afficher_message(str(erreur) + " : lancer 04_outils/preparer_eda_app.py.", "avertissement")
    else:
        st.write("Un sous-bassin est écarté s'il a moins de " + str(SEUIL_RETENU) + " ventes dans l'année pour ce type de bien.")
        annees = sorted(seuils["annee"].unique())
        annee = st.select_slider("Année", annees, value=annees[-1], key="seuil_annee")
        vue = seuils[seuils["annee"] == annee].sort_values(["type_bien", "seuil"]).copy()
        vue["part_sous_bassins"] = 100 * vue["territoires_gardes"] / vue["territoires"]
        vue["part_ventes"] = 100 * vue["ventes_gardees"] / vue["ventes"]
        au_seuil = vue[vue["seuil"] == SEUIL_RETENU].set_index("type_bien")
        if au_seuil.empty:
            afficher_message("Aucune donnée pour ce choix.", "avertissement")
        else:
            colonnes = st.columns(2)
            for colonne, type_nom, libelle in zip(colonnes, ["appartement", "maison"], ["Appartements", "Maisons"]):
                if type_nom not in au_seuil.index:
                    continue
                ligne = au_seuil.loc[type_nom]
                gardes, total = int(ligne["territoires_gardes"]), int(ligne["territoires"])
                ecartees_ventes = 100 - ligne["part_ventes"]
                pesent = "moins de 0,05 %" if ecartees_ventes < 0.05 else fr(ecartees_ventes, 1) + " %"
                barres_type = pd.DataFrame({"serie": ORDRE_SERIES, "part": [ligne["part_ventes"], ligne["part_sous_bassins"]]})
                barres_type["texte"] = barres_type["part"].map(lambda v: fr(v, 1) + " %")
                graphique = alt.Chart(barres_type).mark_bar().encode(
                    y=alt.Y("serie:N", title=None, sort=ORDRE_SERIES[::-1], axis=alt.Axis(labelLimit=200)),
                    x=alt.X("part:Q", title=None, scale=alt.Scale(domain=[0, 115]), axis=alt.Axis(values=[0, 25, 50, 75, 100], format="d")),
                    color=alt.Color("serie:N", legend=None, scale=alt.Scale(domain=ORDRE_SERIES)))
                with colonne.container(border=True):
                    st.write("**" + libelle + "**")
                    texte = alt.Chart(barres_type).mark_text(align="left", dx=4).encode(
                        y=alt.Y("serie:N", sort=ORDRE_SERIES[::-1]), x="part:Q", text="texte:N")
                    st.altair_chart((graphique + texte).properties(height=100),
                                    use_container_width=True)
                    st.caption(fr(gardes) + " sous-bassins gardés sur " + fr(total) + " qui ont au moins une vente. Les "
                               + fr(total - gardes) + " écartés pèsent " + pesent + " des ventes.")

            question(4, "Et si on changeait le seuil ?")
            st.write("Plus le seuil monte, plus la médiane est fiable, mais moins il reste de sous-bassins. Les ventes, "
                     "elles, restent presque toutes couvertes : la zone colorée est l'écart entre les deux.")
            colonnes = st.columns(2)
            for colonne, type_nom, libelle in zip(colonnes, ["appartement", "maison"], ["Appartements", "Maisons"]):
                donnees_type = vue[vue["type_bien"] == type_nom]
                if donnees_type.empty:
                    continue
                large = donnees_type.assign(**{"Ventes conservées": donnees_type["part_ventes"],
                                               "Sous-bassins gardés": donnees_type["part_sous_bassins"]})
                long = large.melt(id_vars=["seuil"], value_vars=ORDRE_SERIES, var_name="serie", value_name="part")
                echelle_seuil = alt.Scale(domain=[5, 140], nice=False)
                x_seuil = alt.X("seuil:Q", title="Seuil (ventes minimum par an)", scale=echelle_seuil,
                                axis=alt.Axis(values=[5] + list(range(10, 101, 10)), format="d"))
                y_part = alt.Y("part:Q", title="Part (%)", scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(values=[0, 25, 50, 75, 100]))
                couleur = alt.Color("serie:N", legend=None, scale=alt.Scale(domain=ORDRE_SERIES))
                zone = alt.Chart(large.assign(serie="Sous-bassins gardés")).mark_area(opacity=0.18).encode(
                    x=alt.X("seuil:Q", scale=echelle_seuil), y=alt.Y("Sous-bassins gardés:Q", scale=alt.Scale(domain=[0, 105])),
                    y2="Ventes conservées:Q", color=couleur)
                lignes = alt.Chart(long).mark_line(point=True).encode(
                    x=x_seuil, y=y_part, color=couleur,
                    tooltip=[alt.Tooltip("seuil:Q", title="Seuil"), alt.Tooltip("serie:N", title="Série"),
                             alt.Tooltip("part:Q", title="Part (%)", format=".1f")])
                trait = alt.Chart(pd.DataFrame({"seuil": [SEUIL_RETENU]})).mark_rule(strokeDash=[4, 4]).encode(x="seuil:Q")
                valeurs = long[long["seuil"] == SEUIL_RETENU].assign(texte=lambda d: d["part"].map(lambda v: fr(v, 1) + " %"))
                ecriture = alt.Chart(valeurs)
                haut = ecriture.transform_filter(alt.datum.serie == "Ventes conservées").mark_text(align="left", dx=8, dy=-10, fontWeight="bold").encode(
                    x="seuil:Q", y="part:Q", text="texte:N", color=couleur)
                bas = ecriture.transform_filter(alt.datum.serie == "Sous-bassins gardés").mark_text(align="left", dx=8, dy=16, fontWeight="bold").encode(
                    x="seuil:Q", y="part:Q", text="texte:N", color=couleur)
                bouts = alt.Chart(long[long["seuil"] == long["seuil"].max()].assign(
                    etiquette=lambda d: d["serie"].map({"Ventes conservées": "Ventes", "Sous-bassins gardés": "Sous-bassins"}))).mark_text(align="left", dx=8, fontWeight="bold").encode(
                    x="seuil:Q", y="part:Q", text="etiquette:N", color=couleur)
                couches = zone + lignes + trait + haut + bas + bouts
                ecart = float(valeurs.loc[valeurs["serie"] == "Ventes conservées", "part"].iloc[0]
                              - valeurs.loc[valeurs["serie"] == "Sous-bassins gardés", "part"].iloc[0])
                with colonne:
                    st.write("**" + libelle + "**")
                    st.altair_chart(couches.properties(height=300), use_container_width=True)
                    if ecart >= 10:
                        st.caption("Au seuil de " + str(SEUIL_RETENU) + " : " + fr(ecart) + " points d'écart entre la part des ventes "
                                   "conservées et la part des sous-bassins gardés.")
            st.caption("Trait pointillé : seuil retenu, " + str(SEUIL_RETENU) + " ventes.")
            with st.expander("Voir les valeurs par seuil"):
                detail = vue[vue["seuil"].isin([10, 20, 30, 50, 100])].assign(
                    **{"Type de bien": lambda d: d["type_bien"].str.capitalize(), "Seuil": lambda d: d["seuil"].astype(int),
                       "Sous-bassins gardés": lambda d: d["territoires_gardes"].astype(int),
                       "Part des sous-bassins (%)": lambda d: d["part_sous_bassins"].map(lambda v: fr(v, 1)),
                       "Part des ventes conservées (%)": lambda d: d["part_ventes"].map(lambda v: fr(v, 1))})
                st.dataframe(detail[["Type de bien", "Seuil", "Sous-bassins gardés", "Part des sous-bassins (%)",
                                     "Part des ventes conservées (%)"]].sort_values(["Type de bien", "Seuil"]),
                             hide_index=True, use_container_width=True)

            # Message clé, calculé sur l'année choisie
            if "appartement" in au_seuil.index:
                a = au_seuil.loc["appartement"]
                titre = ("Un seuil de " + str(SEUIL_RETENU) + " ventes écarte **" + fr(100 - a["part_sous_bassins"], 1)
                         + " %** des sous-bassins d'appartements, mais seulement **" + fr(100 - a["part_ventes"], 1) + " %** des ventes.")
                sous_titre = None
                if "maison" in au_seuil.index:
                    m = au_seuil.loc["maison"]
                    sous_titre = ("Pour les maisons, le seuil écarte " + fr(int(m["territoires"] - m["territoires_gardes"]))
                                  + " sous-bassins sur " + fr(int(m["territoires"])) + " : " + fr(m["part_ventes"], 1)
                                  + " % des ventes conservées.")
                with zone_cle:
                    cle("En " + str(int(annee)) + " : " + titre[0].lower() + titre[1:], sous_titre)
    st.caption("Sources : eda_seuil.parquet (ventes agrégées par sous-bassin de vie) ; rapport de recherche, tableau 3.")
