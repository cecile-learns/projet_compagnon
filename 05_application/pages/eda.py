# =============================================================================
# 05_application/pages/eda.py : page Exploration des données (Christian)
# =============================================================================
# Ce que fait la page : un parcours guidé en quatre scènes pour la démonstration (environ 5 minutes).
# Chaque scène répond à la question posée par la précédente :
#   1 Quand     : le marché a changé de régime (prix et ventes, trois phases)
#   2 Où        : le territoire fixe le niveau du prix (carte par département)
#   3 Le changement : mais il dit peu du changement (prix N-1, évolution sur un an, taux)
#   4 Ce que cela impose : référence naïve, sous-bassin et seuil de 20 ventes, modèle par vente
# Chaque scène a la même trame : « À retenir », les graphiques, « Ce qu'on en déduit », la source.
#
# Données lues (produites par 04_outils/preparer_eda_app.py, copiées dans donnees/) :
#   eda_territoires.parquet, eda_seuil.parquet et departements_metropole.geojson.
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

# Années couvertes par chaque source : (source, début, fin, forme, période, limite principale)
# La frise commence en 2005 : la série des taux, qui commence en 1998, est coupée à gauche. Tableau 2 du rapport de recherche.
FRISE_DEBUT = 2005
PERIODES = [
    ("DVF+ (prix)", 2014, 2025, "barre", "2014 à 2025", "Ventes réalisées seulement ; Alsace, Moselle et Mayotte absentes"),
    ("Taux des crédits", FRISE_DEBUT, 2025, "barre", "depuis 1998, chaque mois", "Valeur nationale, la même pour tous les territoires"),
    ("Niveau de vie", 2014, 2021, "barre", "2014 à 2021", "10 % de communes masquées ; 2021 est le dernier millésime"),
    ("Délinquance", 2016, 2025, "barre", "2016 à 2025", "46 % des lignes masquées ; faits enregistrés, pas délinquance réelle"),
    ("Parc de logements", 2016, 2022, "barre", "2016 à 2022", "Estimations fragiles dans les petits IRIS"),
    ("Logements vacants", 2020, 2026, "barre", "millésimes 2020 à 2026", "Parc privé seul ; série rompue à partir du millésime 2025"),
    ("Équipements", 2015, 2015, "point", "2015", "Maille bassin de vie seulement"),
    ("Équipements", 2020, 2020, "point", "2020", "Maille bassin de vie seulement"),
    ("Équipements", 2025, 2025, "point", "2025", "Maille bassin de vie seulement"),
    ("Emploi salarié privé", 2006, 2025, "barre", "2006 à 2025", "Secteur privé seul"),
]
LIMITES = [
    "**DVF+ ne contient que des ventes réalisées** : ni invendus, ni prix demandés. L'Alsace, la Moselle et Mayotte en sont absentes.",
    "**Plusieurs sources sont en partie masquées** : 10 % des communes pour le niveau de vie, 46 % des lignes pour la délinquance.",
    "**Elles ne couvrent pas les mêmes années** : le niveau de vie s'arrête en 2021, les équipements ne sont connus qu'en 2015, 2020 et 2025.",
    "**Les taux des crédits sont nationaux** : la même valeur pour tous les territoires un mois donné.",
]


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


def _simplifier(points, tolerance):
    """Douglas-Peucker itératif : garde les points qui s'écartent de plus de `tolerance` degrés du tracé simplifié."""
    if len(points) < 5:
        return points
    garde = [False] * len(points)
    garde[0] = garde[-1] = True
    pile = [(0, len(points) - 1)]
    while pile:
        debut, fin = pile.pop()
        (x1, y1), (x2, y2) = points[debut], points[fin]
        dx, dy = x2 - x1, y2 - y1
        norme = (dx * dx + dy * dy) ** 0.5 or 1e-12
        pire, indice = 0.0, None
        for i in range(debut + 1, fin):
            distance = abs(dy * (points[i][0] - x1) - dx * (points[i][1] - y1)) / norme
            if distance > pire:
                pire, indice = distance, i
        if indice is not None and pire > tolerance:
            garde[indice] = True
            pile.extend([(debut, indice), (indice, fin)])
    return [p for p, g in zip(points, garde) if g]


def _alleger_anneau(anneau, tolerance=0.02):
    points = [(x, y) for x, y in anneau]
    if len(points) > 4 and points[0] == points[-1]:
        # anneau fermé : on le coupe en deux pour que la droite de référence ne soit pas réduite à un point
        milieu = len(points) // 2
        simple = _simplifier(points[:milieu + 1], tolerance)[:-1] + _simplifier(points[milieu:], tolerance)
    else:
        simple = _simplifier(points, tolerance)
    arrondi = []
    for x, y in simple:
        p = [round(x, 3), round(y, 3)]
        if not arrondi or p != arrondi[-1]:
            arrondi.append(p)
    return arrondi if len(arrondi) >= 4 else None


@st.cache_data
def fond_departements():
    """Contours des départements métropolitains (donnees/departements_metropole.geojson), lus une seule fois puis allégés :
    le tracé est simplifié (écart inférieur à 2 km) pour que les cartes s'affichent vite, y compris sur un portable."""
    chemin = DONNEES / "departements_metropole.geojson"
    if not chemin.exists():
        raise FileNotFoundError("departements_metropole.geojson absent de " + str(DONNEES))
    fond = json.loads(chemin.read_text(encoding="utf-8"))
    for f in fond["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            anneaux = [_alleger_anneau(r) for r in g["coordinates"]]
            g["coordinates"] = [r for r in anneaux if r is not None] or g["coordinates"]
        else:
            polygones = []
            for polygone in g["coordinates"]:
                anneaux = [_alleger_anneau(r) for r in polygone]
                if anneaux[0] is not None:
                    polygones.append([r for r in anneaux if r is not None])
            g["coordinates"] = polygones
    return fond


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


# =============================================================================
# PARCOURS GUIDÉ : quatre scènes pour la démonstration (environ 5 minutes)
# =============================================================================
# Chiffres du rapport (non recalculables avec les fichiers de l'application) : mutations, ventes retenues, taux.
MUTATIONS_BRUTES = 16_375_738     # rapport, partie 2 : mutations DVF+ avant filtres
VENTES_RETENUES = 9_680_495       # rapport, partie 2 : ventes de logements retenues

SCENES = [
    ("1 · Quand", "Que s'est-il passé depuis 2015 ?"),
    ("2 · Où", "Où les prix sont-ils élevés ?"),
    ("3 · Le changement", "Le territoire dit-il comment les prix changent ?"),
    ("4 · Ce que cela impose", "Qu'est-ce que cela change pour la prévision ?"),
]


def aller(delta):
    """Passe à la scène voisine (appelé par les boutons, avant le rendu suivant)."""
    courant = SCENES_NOMS.index(st.session_state.get("eda_scene", SCENES_NOMS[0]))
    st.session_state["eda_scene"] = SCENES_NOMS[min(max(courant + delta, 0), len(SCENES_NOMS) - 1)]


SCENES_NOMS = [nom for nom, _ in SCENES]


def indices_nationaux(t):
    """Prix (moyenne des sous-bassins pondérée par les ventes) et ventes, par année, France entière."""
    d = t.assign(pv=t["prix"] * t["n_ventes"]).groupby(["annee", "type_bien"]).agg(
        pv=("pv", "sum"), ventes=("n_ventes", "sum")).reset_index()
    d["prix"] = d["pv"] / d["ventes"]
    prix = d.assign(serie=d["type_bien"].map({"appartement": "Appartements", "maison": "Maisons"})).sort_values(["serie", "annee"])
    prix["indice"] = 100 * prix["prix"] / prix.groupby("serie")["prix"].transform("first")
    ventes = t.groupby("annee")["n_ventes"].sum().reset_index().sort_values("annee")
    ventes["indice"] = 100 * ventes["n_ventes"] / ventes["n_ventes"].iloc[0]
    ventes["serie"] = "Ventes"
    return prix.reset_index(drop=True), ventes.reset_index(drop=True)


def prix_par_departement(t, type_bien, annee):
    """Prix médian des sous-bassins par département ; Paris et les faux centres alsaciens sont mis à part."""
    vue = t[(t["type_bien"] == type_bien) & (t["annee"] == annee)]
    tout = vue.groupby("departement").agg(prix=("prix", "median"), sous_bassins=("prix", "size"),
                                          ventes=("n_ventes", "sum")).reset_index()
    tout = tout[tout["sous_bassins"] >= 3]
    tout["Département"] = tout["departement"].map(lambda c: c + " – " + departement(c))
    paris = tout[tout["departement"] == BASSIN_PARIS]
    par_dep = tout[~tout["departement"].isin([BASSIN_PARIS, CENTRES_ALSACE])].sort_values("prix", ascending=False)
    return par_dep, paris


def carte_departements(par_dep, hauteur):
    fond = fond_departements()
    carte = par_dep[["departement", "Département", "prix", "sous_bassins", "ventes"]].copy()
    carte["prix"] = carte["prix"].round(0)
    chaleur = alt.Chart(alt.Data(values=fond["features"])).mark_geoshape(stroke="white", strokeWidth=0.6).transform_lookup(
        lookup="properties.code",
        from_=alt.LookupData(carte, "departement", ["Département", "prix", "sous_bassins", "ventes"]),
    ).encode(
        color=alt.Color("prix:Q", title="Prix médian au m² (€)",
                        scale=alt.Scale(type="quantile", scheme=alt.SchemeParams(name="blues", count=6)),
                        legend=None),
        tooltip=[alt.Tooltip("properties.nom:N", title="Département"), alt.Tooltip("prix:Q", title="Prix au m² (€)", format="d"),
                 alt.Tooltip("sous_bassins:Q", title="Sous-bassins"), alt.Tooltip("ventes:Q", title="Ventes")])
    socle = alt.Chart(alt.Data(values=fond["features"])).mark_geoshape(fill="lightgray", stroke="white", strokeWidth=0.6)
    return (socle + chaleur).project(type="conicConformal", parallels=[44, 49], rotate=[-3, 0]).properties(height=hauteur)


def barres_classement(df, titre):
    maxi = float(df["prix"].max())
    df = df.assign(texte=df["prix"].round(0).astype(int).map(fr) + " €")
    barres = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Département:N", sort=None, title=None, axis=alt.Axis(labelLimit=170, labelOverlap=False, labelFontSize=11)),
        x=alt.X("prix:Q", title=None, axis=None, scale=alt.Scale(domain=[0, maxi * 1.8])))
    texte = barres.mark_text(align="left", dx=4).encode(text="texte:N")
    st.write("**" + titre + "**")
    st.altair_chart((barres + texte).properties(height=120), use_container_width=True)


CONCLUSIONS = [
    "Un marché qui change",
    "Le territoire fixe le niveau",
    "Le changement est le défi",
    "Trois choix de modélisation",
]


def fil_rouge(i):
    """Ligne de repères : conclusions des scènes vues, scène en cours, scènes à venir."""
    colonnes = st.columns(len(CONCLUSIONS))
    for j, (colonne, texte) in enumerate(zip(colonnes, CONCLUSIONS)):
        if j < i:
            colonne.markdown("✓ " + texte)
        elif j == i:
            colonne.markdown("**▶ " + texte + "**")
        else:
            colonne.caption(texte)


def donc(texte):
    with st.container(border=True):
        st.markdown("**Donc** " + texte)


def bandeau_chiffres(t):
    st.caption(fr(VENTES_RETENUES) + " ventes de logements retenues (sur " + fr(MUTATIONS_BRUTES) + " mutations)  ·  "
               + fr(t["sous_bassin_de_vie"].nunique()) + " sous-bassins de vie  ·  " + str(int(t["annee"].min())) + " à "
               + str(int(t["annee"].max())) + "  ·  DVF+ et sept sources sur les territoires")


def evolutions(t, annee):
    p = t.pivot_table(index=["sous_bassin_de_vie", "type_bien"], columns="annee", values="prix")
    d = p[[annee - 1, annee]].dropna().reset_index()
    d.columns = ["sous_bassin_de_vie", "type_bien", "avant", "apres"]
    d["Type"] = d["type_bien"].map({"appartement": "Appartements", "maison": "Maisons"})
    d["evolution"] = 100 * (d["apres"] / d["avant"] - 1)
    return d


def bloc_sources(t):
    """Sources et limites, dans un encadré replié de la première scène."""
    gauche, droite = st.columns([3, 2])
    with gauche:
        frise = pd.DataFrame(PERIODES, columns=["source", "debut", "fin", "forme", "periode", "limite"])
        ordre = list(dict.fromkeys(frise["source"]))
        echelle = alt.Scale(domain=[FRISE_DEBUT, 2026.6], nice=False)
        axe = alt.Axis(format="d", values=[2005, 2010, 2015, 2020, 2025])
        axe_y = alt.Y("source:N", title=None, sort=ordre, axis=alt.Axis(labelLimit=200))
        info = [alt.Tooltip("source:N", title="Source"), alt.Tooltip("periode:N", title="Période"), alt.Tooltip("limite:N", title="Limite")]
        barres = alt.Chart(frise[frise["forme"] == "barre"]).mark_bar(size=12).encode(
            x=alt.X("debut:Q", title=None, scale=echelle, axis=axe), x2="fin:Q", y=axe_y, tooltip=info)
        points = alt.Chart(frise[frise["forme"] == "point"]).mark_point(filled=True, size=110, opacity=1).encode(
            x=alt.X("debut:Q", title=None, scale=echelle, axis=axe), y=axe_y, tooltip=info)
        st.write("**Les années couvertes par chaque source**")
        st.altair_chart((barres + points).properties(height=250), use_container_width=True)
        st.caption("Un point : une année de référence. La série des taux commence en 1998 (frise coupée à 2005). Survole une barre pour voir sa limite.")
    with droite:
        st.markdown("**Quatre limites à connaître**")
        st.markdown("\n".join("- " + ligne for ligne in LIMITES))


def entree_donnees(t):
    """Ouverture de l'histoire : ce que sont les données, en une ligne, avec le détail replié."""
    st.caption("**Les données** : " + fr(VENTES_RETENUES) + " ventes de logements retenues (DVF+), " + fr(t["sous_bassin_de_vie"].nunique())
               + " sous-bassins de vie, " + str(int(t["annee"].min())) + " à " + str(int(t["annee"].max()))
               + ", décrits par sept autres sources. Limites : ventes réalisées seulement, sources en partie masquées, années différentes.")
    with st.expander("Sources et limites"):
        bloc_sources(t)


def scene_quand(t):
    prix, ventes = indices_nationaux(t)
    debut, fin = int(prix["annee"].min()), int(prix["annee"].max())
    v = {s: prix[prix["serie"] == s]["indice"].iloc[-1] - 100 for s in ["Appartements", "Maisons"]}
    pic_v = int(ventes.loc[ventes["indice"].idxmax(), "annee"])
    pic_p = int(prix.groupby("annee")["indice"].mean().idxmax())
    entree_donnees(t)
    cle("De " + str(debut) + " à " + str(fin) + ", les prix gagnent " + signe(v["Appartements"]) + " % (appartements) et "
        + signe(v["Maisons"]) + " % (maisons) : stables, puis en hausse, puis en repli.")
    echelle_x = alt.Scale(domain=[debut - 0.2, fin + 2.6], nice=False)
    x_annee = alt.X("annee:Q", title=None, axis=alt.Axis(format="d", values=list(range(debut, fin + 1))), scale=echelle_x)
    phases = pd.DataFrame(PHASES, columns=["nom", "debut", "fin"])
    phases["milieu"] = (phases["debut"] + phases["fin"]) / 2
    fond = alt.Chart(phases[phases["nom"] == "Hausse"]).mark_rect(opacity=0.12).encode(x=alt.X("debut:Q", scale=echelle_x), x2="fin:Q")
    noms = alt.Chart(phases).mark_text(baseline="top", dy=2, fontWeight="bold").encode(x=alt.X("milieu:Q", scale=echelle_x), y=alt.value(0), text="nom:N")

    def panneau(df, hauteur, titre_y, couleur, avec_noms):
        bas_y, haut_y = min(df["indice"].min(), 100), df["indice"].max()
        echelle_y = alt.Scale(domain=[bas_y - (haut_y - bas_y) * 0.06, haut_y + (haut_y - bas_y) * 0.2], nice=False)
        lignes = alt.Chart(df).mark_line(point=True).encode(
            x=x_annee, y=alt.Y("indice:Q", title=titre_y, scale=echelle_y), color=couleur,
            tooltip=[alt.Tooltip("annee:Q", title="Année", format="d"), alt.Tooltip("serie:N", title="Série"), alt.Tooltip("indice:Q", title="Indice", format=".1f")])
        if "Maisons" in set(df["serie"]):
            lignes = lignes.encode(strokeDash=alt.StrokeDash("serie:N", legend=None, scale=alt.Scale(domain=["Appartements", "Maisons"], range=[[1, 0], [6, 4]])))
        derniere = df.sort_values("annee").groupby("serie").tail(1).assign(texte=lambda d: d["indice"].map(lambda x: signe(x - 100) + " %"))
        bout = alt.Chart(derniere).mark_text(align="left", dx=10, fontWeight="bold").encode(x=x_annee, y=alt.Y("indice:Q", scale=echelle_y), text="texte:N", color=couleur)
        repere = alt.Chart(pd.DataFrame({"indice": [100]})).mark_rule(strokeDash=[3, 3], opacity=0.5).encode(y=alt.Y("indice:Q", scale=echelle_y))
        couches = fond + repere + lignes + bout
        return (couches + noms if avec_noms else couches).properties(height=hauteur)

    gauche, droite = st.columns(2)
    with gauche:
        st.write("**Prix au m²** (indice, 100 = " + str(debut) + "). Plein : appartements, tireté : maisons.")
        st.altair_chart(panneau(prix, 300, "Prix, indice (base 100)", alt.Color("serie:N", legend=None, scale=alt.Scale(domain=["Appartements", "Maisons"])), True), use_container_width=True)
    with droite:
        st.write("**Nombre de ventes** (indice, 100 = " + str(debut) + ")")
        st.altair_chart(panneau(ventes, 300, "Ventes, indice (base 100)", alt.Color("serie:N", legend=None, scale=alt.Scale(domain=["Ventes"])), False), use_container_width=True)
    st.caption("Les ventes atteignent leur sommet en " + str(pic_v) + ", les prix en " + str(pic_p) + " : les ventes reculent avant les prix.")
    donc("le passé récent ne suffit pas à prévoir : le marché a changé de régime. Reste à savoir ce qui distingue un territoire d'un autre.")
    st.caption("Source : DVF+. Prix de l'année : moyenne des prix des sous-bassins (au moins 20 ventes), pondérée par leurs ventes.")


def scene_ou(t):
    annee = int(t["annee"].max())
    vues = {}
    for type_bien in ["maison", "appartement"]:
        par_dep, paris = prix_par_departement(t, type_bien, annee)
        vues[type_bien] = (par_dep, paris, par_dep.iloc[0], par_dep.iloc[-1])
    ecart = {k: v[2]["prix"] / v[3]["prix"] for k, v in vues.items()}
    cle("En " + str(annee) + ", le territoire fixe le niveau du prix : jusqu'à ×\u00a0" + fr(ecart["maison"], 1) + " entre départements pour les maisons, ×\u00a0"
        + fr(ecart["appartement"], 1) + " pour les appartements.")
    colonnes = st.columns(2)
    for colonne, type_bien, titre in zip(colonnes, ["maison", "appartement"], ["Maisons", "Appartements"]):
        par_dep, paris, haut, bas = vues[type_bien]
        with colonne:
            st.write("**" + titre + "**")
            carte, classement = st.columns([5, 4])
            with carte:
                st.altair_chart(carte_departements(par_dep, 330), use_container_width=True)
            with classement:
                barres_classement(par_dep.head(5), "Les 5 plus chers")
                barres_classement(par_dep.tail(5).iloc[::-1], "Les 5 moins chers")
                if not paris.empty:
                    st.caption("Bassin de vie de Paris : " + fr(float(paris["prix"].iloc[0])) + " €/m²")
    st.caption("Prix médian au m² des sous-bassins de chaque département, six classes de même effectif, de la plus claire à la plus foncée. "
               "Gris sur la carte : moins de 3 sous-bassins d'au moins 20 ventes, ou Alsace-Moselle (absente de DVF+).")
    donc("décrire les territoires (niveau de vie, délinquance, équipements, parc de logements, emploi) dit où les prix sont élevés. Dit-il aussi comment ils changent ?")
    st.caption("Source : DVF+.")


TAUX_RAPPORT = [(2021, 1.10, "1,10 %"), (2023, 3.6, "> 3,6 %"), (2025, 3.0, "≈ 3 %")]   # rapport de recherche


def graphique_taux(t):
    """Évolution annuelle des prix, et trois points de taux du rapport : le taux monte avant que les prix ne baissent."""
    prix, _ = indices_nationaux(t)
    prix = prix.sort_values(["serie", "annee"])
    prix["evolution"] = prix.groupby("serie")["prix"].pct_change() * 100
    prix = prix.dropna(subset=["evolution"]).query("annee >= 2019").rename(columns={"evolution": "valeur"})
    nom_taux = "Taux des crédits"
    taux = pd.DataFrame(TAUX_RAPPORT, columns=["annee", "valeur", "texte"]).assign(serie=nom_taux)
    series = ["Appartements", "Maisons", nom_taux]
    x = alt.X("annee:Q", title=None, scale=alt.Scale(domain=[2018.7, 2025.4], nice=False), axis=alt.Axis(format="d", values=list(range(2019, 2026))))
    y = alt.Y("valeur:Q", title="% (évolution des prix ; taux)")
    couleur = alt.Color("serie:N", legend=alt.Legend(orient="top", title=None, labelLimit=0, columns=1, symbolType="stroke", symbolStrokeWidth=3), scale=alt.Scale(domain=series))
    trait = alt.StrokeDash("serie:N", legend=None, scale=alt.Scale(domain=series, range=[[1, 0], [6, 4], [2, 2]]))
    info = [alt.Tooltip("annee:Q", title="Année", format="d"), alt.Tooltip("serie:N", title="Série"), alt.Tooltip("valeur:Q", title="%", format=".2f")]
    lignes = alt.Chart(pd.concat([prix[["annee", "serie", "valeur"]], taux[["annee", "serie", "valeur"]]])).mark_line(point=True).encode(
        x=x, y=y, color=couleur, strokeDash=trait, tooltip=info)
    etiq = alt.Chart(taux).mark_text(dy=-14, fontWeight="bold").encode(x=x, y=y, text="texte:N", color=alt.Color("serie:N", legend=None, scale=alt.Scale(domain=series)))
    zero = alt.Chart(pd.DataFrame({"valeur": [0]})).mark_rule(strokeDash=[3, 3], opacity=0.5).encode(y=y)
    return (zero + lignes + etiq).properties(height=300)


def correlation_minimale(t):
    """Plus faible corrélation entre le prix d'une année et celui de l'année précédente, sur toutes les années disponibles."""
    p = t.pivot_table(index=["sous_bassin_de_vie", "type_bien"], columns="annee", values="prix")
    annees = sorted(p.columns)
    return min(p[[a, b]].dropna().corr().iloc[0, 1] for a, b in zip(annees[:-1], annees[1:]))


def scene_changement(t):
    annee = int(t["annee"].max())
    d = evolutions(t, annee)
    corr = {k: d.loc[d["type_bien"] == k, ["avant", "apres"]].corr().iloc[0, 1] for k in ["appartement", "maison"]}
    c_a, c_m = fr(corr["appartement"], 2), fr(corr["maison"], 2)
    mediane = d.groupby("Type")["evolution"].median()
    lien = c_a if c_a == c_m else c_a + " et " + c_m
    mini = fr(correlation_minimale(t), 2)
    annees = sorted(t["annee"].unique())
    cle("Le niveau est connu d'avance (corrélation d'au moins " + mini + " d'une année à la suivante, depuis " + str(int(annees[1])) + ") : le vrai défi est le changement, de "
        + signe(mediane["Appartements"], 1) + " % en médiane en " + str(annee) + " pour les appartements.")
    gauche, droite = st.columns(2)
    with gauche:
        borne = [float(d[["avant", "apres"]].min().min()) * 0.9, float(d[["avant", "apres"]].max().max()) * 1.1]
        echelle = alt.Scale(type="log", domain=borne, nice=False)
        axe = alt.Axis(format="d", grid=False, values=[500, 1000, 2000, 5000, 10000])
        points = alt.Chart(d).mark_circle(size=22, opacity=0.4).encode(
            x=alt.X("avant:Q", title="Prix au m² en " + str(annee - 1) + " (€, échelle log)", scale=echelle, axis=axe),
            y=alt.Y("apres:Q", title="Prix au m² en " + str(annee) + " (€)", scale=echelle, axis=axe),
            color=alt.Color("Type:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("sous_bassin_de_vie:N", title="Sous-bassin"), alt.Tooltip("Type:N"), alt.Tooltip("avant:Q", format="d"), alt.Tooltip("apres:Q", format="d")])
        diag = alt.Chart(pd.DataFrame({"x": borne, "y": borne})).mark_line(strokeDash=[4, 4], color="gray").encode(x=alt.X("x:Q", scale=echelle), y=alt.Y("y:Q", scale=echelle))
        note = alt.Chart(pd.DataFrame({"x": [borne[0] * 1.15], "y": [borne[1] * 0.8], "t": ["Corrélation " + lien + " : presque sur la diagonale"]})).mark_text(
            align="left", fontWeight="bold").encode(x=alt.X("x:Q", scale=echelle), y=alt.Y("y:Q", scale=echelle), text="t:N")
        st.write("**Le niveau : presque aligné sur l'an dernier** (" + str(annee - 1) + " contre " + str(annee) + ")")
        st.altair_chart((points + diag + note).properties(height=300), use_container_width=True)
    with droite:
        st.write("**Le changement : le taux monte, puis les prix reculent**")
        st.altair_chart(graphique_taux(t), use_container_width=True)
    donc("une référence simple (prix N = prix N-1) est déjà très difficile à battre, et le contexte du marché doit entrer en jeu : le taux est lié "
         "à l'évolution des prix (corrélation de −0,88), avec un retard d'environ six mois.")
    st.caption("Sources : DVF+ (prix, évolution annuelle moyenne pondérée par les ventes) ; taux des crédits, Banque de France (valeurs de fin 2021, fin 2023 et 2025).")


def scene_impose(t):
    annee = int(t["annee"].max())
    seuils = charger_parquet("eda_seuil.parquet")
    s = seuils[(seuils["annee"] == annee) & (seuils["seuil"] == SEUIL_RETENU) & (seuils["type_bien"] == "appartement")].iloc[0]
    ecartes = 100 - 100 * s["territoires_gardes"] / s["territoires"]
    ventes_ecartees = 100 - 100 * s["ventes_gardees"] / s["ventes"]
    cle("Trois constats, trois choix de modélisation.")
    entetes = st.columns([5, 1, 6, 1, 4])
    entetes[0].caption("CE QUE L'EXPLORATION MONTRE")
    entetes[2].caption("CE QUE NOUS AVONS CHOISI")
    entetes[4].caption("OÙ LE VOIR")
    lignes = [
        ("Le changement est le vrai défi : le prix de l'an dernier explique presque tout.",
         "**Une référence naïve** : reconduire le prix de l'an dernier. Le modèle doit faire mieux sur l'écart à N-1.", "naive"),
        ("Le territoire fixe le niveau du prix.",
         "**Le sous-bassin de vie** comme maille, avec un seuil de " + str(SEUIL_RETENU) + " ventes par année et par type de bien : il écarte **"
         + fr(ecartes, 1) + " %** des sous-bassins d'appartements, mais seulement **" + fr(ventes_ecartees, 1) + " %** des ventes.", "rapport"),
        ("Le marché a changé de régime : le passé récent ne suffit pas.",
         "**Un modèle par vente**, qui apprend avec le contexte du marché et du territoire.", "modele"),
    ]
    for constat, choix, lien in lignes:
        with st.container(border=True):
            c = st.columns([5, 1, 6, 1, 4], vertical_alignment="center")
            c[0].markdown(constat)
            c[1].markdown("→")
            c[2].markdown(choix)
            c[3].markdown("→")
            if lien == "naive":
                c[4].page_link("pages/modelisation.py", label="Voir la modélisation", icon="📈")
            elif lien == "modele":
                c[4].page_link("pages/modelisation.py", label="Résultats du modèle", icon="📈")
                c[4].page_link("pages/outil.py", label="Essayer l'outil", icon="🏠")
            else:
                c[4].write("")
    donc("le modèle améliore la référence naïve de **+6,4 %** et **+7,9 %** sur le test 2025 : le détail est dans Modélisation, "
         "et l'Outil permet d'estimer un bien.")
    st.caption("Source : eda_seuil.parquet, année " + str(annee) + ", seuil de " + str(SEUIL_RETENU) + " ventes.")


def navigation_haut(i):
    colonnes = st.columns([8, 1.7, 1.5], vertical_alignment="center")
    colonnes[0].segmented_control("Scène", SCENES_NOMS, key="eda_scene", label_visibility="collapsed", selection_mode="single")
    colonnes[1].button("← Précédent", on_click=aller, args=(-1,), disabled=i == 0, key="eda_prec")
    colonnes[2].button("Suivant →", on_click=aller, args=(1,), disabled=i == len(SCENES) - 1, key="eda_suiv", type="primary")


def parcours_guide():
    try:
        t = territoires()
    except FileNotFoundError as erreur:
        afficher_message(str(erreur) + " : lancer 04_outils/preparer_eda_app.py.", "avertissement")
        return
    if st.session_state.get("eda_scene") not in SCENES_NOMS:
        st.session_state["eda_scene"] = SCENES_NOMS[0]
    i = SCENES_NOMS.index(st.session_state["eda_scene"])
    navigation_haut(i)
    fil_rouge(i)
    # Une zone réservée par scène : les zones inactives sont vidées, la scène active s'affiche dans la sienne.
    # Sans cela, Streamlit réutilise les éléments de l'ancienne scène à la même position et les laisse affichés,
    # grisés et mélangés à la nouvelle, pendant que la carte se calcule.
    zones = [st.empty() for _ in SCENES]
    for j, zone in enumerate(zones):
        if j != i:
            zone.empty()
    with zones[i].container():
        [scene_quand, scene_ou, scene_changement, scene_impose][i](t)


entete("Exploration des données", "Un parcours en quatre scènes : quand, où, le changement, ce que cela impose")
parcours_guide()
