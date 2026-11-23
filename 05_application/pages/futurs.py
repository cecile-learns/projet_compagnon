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
       "Idées et pistes pour améliorer le projet")

onglets = st.tabs(["Méthodologie", "Téléchargement", "Préparation", "Feature engineering", "Fusion"])


# =============================================================================
# 1. MÉTHODOLOGIE
# =============================================================================
with onglets[0]:
    cle("**Tout le monde fait de tout** : chaque étape du projet est partagée entre les quatre, avec des "
        "quantités de travail comparables.",
        "Chacun part du même gabarit, le complète avec ses initiales et le dépose à un endroit précis du dépôt. "
        "Une personne fusionne ensuite les notebooks en un seul fichier Python, la version de référence.")

    # --- Question 1 : la répartition ---
    question(1, "Comment le travail est-il réparti ?")
    st.write("Pas de spécialiste de l'EDA ni de spécialiste du modèle : à chaque étape, chacun prend une part "
             "de la même nature que les autres. Tout le monde pratique toutes les méthodes du projet, et "
             "personne ne porte seul une étape plus lourde.")
    colonnes = st.columns(4)
    for colonne, (emoji, etape, part) in zip(colonnes, [
            ("🔎", "Audit et EDA", "une ou plusieurs sources chacun"),
            ("🧪", "Feature engineering", "une ou plusieurs sources chacun, ramenées au sous-bassin"),
            ("📈", "Modélisation", "une hypothèse par tour et par personne"),
            ("🖥️", "Application", "une page Streamlit par personne")]):
        with colonne.container(border=True):
            st.markdown("### " + emoji)
            st.markdown("**" + etape + "**")
            st.caption(part)
    st.caption("Équipe : Cécile Naudet (CN), Théophyle Chevet (TC), Gabrielle Richert (GR), Christian Bayili (CB). "
               "Nom d'un fichier : initiales_étape_sujet, par exemple GR_eda_source.ipynb.")

    # --- Question 2 : du gabarit au fichier unique ---
    question(2, "Comment le travail de chacun devient-il un seul fichier ?")
    st.graphviz_chart("""
    digraph {
        rankdir=LR; node [shape=box, style=rounded, fontsize=11];
        gabarit [label="Gabarit commun\\n(EDA, feature engineering,\\nmodélisation, page Streamlit)", shape=note];
        cn [label="CN_eda_source.ipynb"]; tc [label="TC_eda_source.ipynb"]; gr [label="GR_eda_source.ipynb"]; cb [label="CB_eda_source.ipynb"];
        depot [label="Dossier de l'étape\\ndans le dépôt", shape=folder];
        fusion [label="Une personne fusionne"];
        py [label="Un seul fichier .py\\n(version de référence)", shape=note];
        gabarit -> cn; gabarit -> tc; gabarit -> gr; gabarit -> cb;
        cn -> depot; tc -> depot; gr -> depot; cb -> depot;
        depot -> fusion -> py;
    }""", use_container_width=True)
    colonnes = st.columns(4)
    for colonne, (titre, texte) in zip(colonnes, [
            ("1. Copier le gabarit", "Le gabarit de l'étape fixe les sections, les noms de variables et les "
                                     "contrôles : tous les fichiers ont la même forme."),
            ("2. Signer", "Le nom du fichier commence par les initiales de son auteur (GR_eda_source.ipynb) : "
                          "on sait qui l'a écrit, et deux personnes ne touchent jamais le même fichier."),
            ("3. Pousser au bon endroit", "Dans le dossier de l'étape où en est le projet (audit, préparation, "
                                          "modélisation, application)."),
            ("4. Fusionner", "Une personne réunit les notebooks en un seul fichier Python. Les constats restent "
                             "dans les notebooks et dans le rapport.")]):
        with colonne.container(border=True):
            st.markdown("**" + titre + "**")
            st.caption(texte)
    st.write("C'est ainsi que `preparation.py` et `feature_engineering.py` sont nés : une section par source, "
             "écrite par une personne à partir du même gabarit, puis réunie dans un seul script où chaque "
             "section se lance seule.")

    # --- Question 3 : le dépôt ---
    question(3, "Comment le dépôt GitHub est-il organisé ?")
    st.write("GitHub ne contient que du code. Les données sont sur le Drive partagé, les rapports sur le Drive de "
             "l'équipe. Toutes les commandes se lancent depuis la racine du projet.")
    gauche, droite = st.columns([1, 1])
    with gauche:
        st.code(
            "projet_compagnon/\n"
            "├── README.md, requirements.txt, pyproject.toml\n"
            "├── config.py                  chemins communs à tous les codes\n"
            "├── data/                      local, jamais sur GitHub\n"
            "│   ├── raw/                   sources brutes\n"
            "│   └── processed/             fichiers créés par nos codes\n"
            "├── 00_telechargement/         téléchargement depuis le Drive\n"
            "├── 01_preparation/            preparation.py (12 blocs)\n"
            "│                              feature_engineering.py (11 sections)\n"
            "├── 02_audit_eda/              audits et EDA des sources,\n"
            "│                              audit de la table, SARIMA national\n"
            "├── socle/                     code partagé : commun.py, gabarits\n"
            "│                              (boosting, hiérarchique, pont, Keras)\n"
            "├── tests/                     un fichier de contrôles par module (pytest)\n"
            "├── 03_modelisation/           modelisation_template.ipynb\n"
            "│   ├── 1_tour1/               H3 : attractivité, logement, emploi, H3 globale\n"
            "│   ├── 2_prepa_tour2/         P6, P7, P9\n"
            "│   ├── 3_tour2/               H4 à H7\n"
            "│   └── 4_evaluation_2025/     test final, ouvert une seule fois\n"
            "├── 04_outils/                 modèle final, lignes d'entrée, ventes comparables\n"
            "└── 05_application/           application Streamlit",
            language=None)
    with droite:
        st.graphviz_chart("""
        digraph {
            rankdir=TB; node [shape=box, style=rounded, fontsize=11]; edge [fontsize=9];
            drive [label="Drive du projet"];
            prep [label="01 · préparation\\npuis feature engineering"];
            table [label="table_modelisation_bv.parquet", shape=note];
            audit [label="02 · audits et EDA"];
            socle [label="socle/ · commun.py et gabarits", shape=folder];
            modele [label="03 · modélisation\\ntour 1 → prépa tour 2 → tour 2 → test 2025"];
            outils [label="04 · outils"];
            appli [label="05 · application"];
            drive -> prep [label=" 00 · téléchargement"];
            prep -> audit; prep -> table -> modele -> outils -> appli;
            socle -> modele [style=dashed];
        }""", use_container_width=True)
    colonnes = st.columns(3)
    for colonne, (titre, texte) in zip(colonnes, [
            ("Dossier numéroté", "Une étape du pipeline, à lancer dans l'ordre des numéros, y compris les "
                                 "sous-dossiers de 03_modelisation."),
            ("Dossier sans numéro", "Un support des étapes, sans ordre d'exécution : socle, tests, data."),
            ("Dossier avec un point", "L'outillage : .git, .streamlit, environnement virtuel.")]):
        with colonne.container(border=True):
            st.markdown("**" + titre + "**")
            st.caption(texte)
    st.write("**Un seul format par type de fichier** : les préparations, les gabarits et les scripts sont des `.py` ; "
             "les analyses exploratoires et les modélisations sont des notebooks, car elles produisent des résultats "
             "à commenter et des synthèses.")
    st.caption("Le socle s'installe une fois (pip install -e .) : « from commun import … » marche alors partout, et "
               "une modification du socle est prise en compte sans réinstaller. pytest vérifie ses modules.")

# =============================================================================
# 2. TÉLÉCHARGEMENT
# =============================================================================
# (clé, fichier, type, rangé dans, rôle, parcours court)
CATALOGUE = [
    ("dvf", "DVF+", "dossier", "raw", "Ventes, un CSV par département (le plus lourd)", False),
    ("filosofi", "Filo_disp_com", "dossier", "raw", "Niveau de vie par commune, 2014 à 2021", False),
    ("menages", "base-ic-couples-familles-menages-2022_csv", "dossier", "raw", "Recensement, ménages, 2019 à 2022", False),
    ("logement", "base-ic-logement-2022_csv", "dossier", "raw", "Recensement, parc de logements 2022", False),
    ("logement_millesimes", "logement_millesimes", "dossier", "raw", "Recensement logement 2016 à 2021 (audits, EDA)", False),
    ("bpe", "BPE25.parquet", "fichier", "raw", "Équipements 2025", False),
    ("bpe_evolution", "DS_BPE_EVOLUTION_2025_data.csv", "fichier", "raw", "Équipements 2015, 2020, 2025 par bassin de vie", False),
    ("bassins_vie", "BV2022_2022_2026.xlsx", "fichier", "raw", "Composition des bassins de vie 2022", False),
    ("bassins_vie_2019_2022", "bassins_vie_2019_2022.xlsx", "fichier", "raw", "Bassins de vie, années 2019 à 2022", False),
    ("passage_geo", "table_passage_geo2003_geo2026.xlsx", "fichier", "raw", "Codes commune 2003 → 2026", False),
    ("equipements", "cible_commune_annee_type_BPE_152_indicateurs.csv", "fichier", "raw", "BPE 2015 et 2020 (table de CB)", False),
    ("lovac", "lovac-open-data-2020-a-2026.xlsx", "fichier", "raw", "Logements privés vacants", False),
    ("delinquance", "dataset_delinquance_communale.parquet", "fichier", "raw", "Délinquance enregistrée", False),
    ("taux", "Taux d'intérêt annuel des crédits immobiliers.csv", "fichier", "raw", "Taux des crédits (Banque de France)", False),
    ("emploi", "etablissements_effectifs_commune.parquet", "fichier", "raw", "Emploi salarié privé (Urssaf)", False),
    ("communes", "communes-france-2026.csv", "fichier", "raw", "Référentiel des communes 2026 (superficie, densité)", False),
    ("superficie_arrondissements", "superficie_arrondissements_geoapi.json", "api", "raw",
     "Superficie des 45 arrondissements (geo.api.gouv.fr)", False),
    ("annonces_ventes", "ech_annonces_ventes_68.csv", "fichier", "raw", "Annonces de vente du Haut-Rhin (axe 2)", False),
    ("annonces_locations", "ech_annonces_locations_68.csv", "fichier", "raw", "Annonces de location du Haut-Rhin (axe 2)", False),
    ("dvf_residentiel", "dvf_plus_residentiel.parquet", "fichier", "processed", "DVF+ déjà filtré (bloc 2) : parcours rapide", False),
    ("dvf_audit", "dvf_plus_echantillon_100k.csv", "fichier", "processed", "Échantillon d'audit de 100 000 mutations", False),
    ("table_modelisation", "table_modelisation_bv.parquet", "fichier", "processed", "Table de modélisation déjà construite", True),
    ("filosofi_commune", "filosofi_commune_2014_2021.parquet", "fichier", "processed", "Filosofi par commune (entrée de P7)", True),
    ("liaison_bv_scindes", "liaison_bv_scindes.parquet", "fichier", "processed", "Commune → sous-bassin (entrée de P7)", True),
    ("taux_credit_mois", "taux_credit_mois.parquet", "fichier", "processed", "Taux des crédits par mois (entrée de P7)", True),
    ("fe_capacite_achat", "fe_capacite_achat_sbv.parquet", "fichier", "processed", "Capacité d'achat (P7)", True),
    ("fe_volume_elan", "fe_volume_elan_sbv.parquet", "fichier", "processed", "Volume et élan (P9)", True),
    ("ventes_apprentissage", "ventes_apprentissage.parquet", "fichier", "processed", "Ventes 2014-2024, modèle par vente (H7)", True),
    ("ventes_test_2025", "ventes_test_2025.parquet", "fichier", "processed", "Ventes 2025 réservées au test final", True),
    ("reglages_keras", "meilleurs_reglages_keras.json", "fichier", "processed", "Réglages optimisés du réseau Keras (H7)", True),
]

with onglets[1]:
    cle("Une commande, toutes les sources : le script télécharge depuis **le Drive du projet** et ne refait "
        "jamais un fichier déjà complet.",
        "Les données ne sont pas dans Git. Chacun les récupère au même endroit, avec les mêmes noms : "
        "le code des autres trouve ses fichiers sans rien changer.")
    st.link_button("📂 Ouvrir le Drive du projet", LIEN_DRIVE)

    # --- Question 1 : deux parcours ---
    question(1, "Faut-il tout télécharger ?")
    st.write("Non. Deux parcours, selon ce qu'on veut refaire.")
    complet, court = st.columns(2)
    with complet.container(border=True):
        st.markdown("**🐢 Parcours complet**")
        st.write("Toutes les sources brutes, puis tout le pipeline : préparation, feature engineering, fusion. "
                 "Pour vérifier ou modifier une étape.")
        st.code("python 00_telechargement/telechargement.py tout", language="bash")
        st.caption("DVF+ brut est le plus lourd : un fichier par département. Le fichier déjà filtré "
                   "(dvf_residentiel) permet de sauter le bloc 2.")
    with court.container(border=True):
        st.markdown("**🐇 Parcours court**")
        st.write("9 fichiers déjà préparés, rangés dans data/processed/ : de quoi lancer les hypothèses H3 à H7 "
                 "et l'évaluation 2025 sans refaire le pipeline.")
        st.code("python 00_telechargement/telechargement.py parcours_court", language="bash")
        st.caption("Raccourci pour un examinateur : l'équipe reconstruit la table avec le pipeline.")

    # --- Question 2 : le catalogue ---
    question(2, "Que contient le Drive ?")
    st.write("Le script tient un **catalogue** : pour chaque source, un nom court (celui qu'on tape en ligne de "
             "commande), son identifiant sur le Drive, son nom local et son dossier.")
    filtre = st.segmented_control("Afficher", ["Tout", "Sources brutes", "Fichiers préparés", "Parcours court"],
                                  default="Tout", key="telech_filtre")
    catalogue = pd.DataFrame(CATALOGUE, columns=["Nom court", "Fichier local", "Type", "Dossier", "Rôle", "court"])
    if filtre == "Sources brutes":
        catalogue = catalogue[catalogue["Dossier"] == "raw"]
    elif filtre == "Fichiers préparés":
        catalogue = catalogue[catalogue["Dossier"] == "processed"]
    elif filtre == "Parcours court":
        catalogue = catalogue[catalogue["court"]]
    catalogue["Dossier"] = "data/" + catalogue["Dossier"] + "/"
    st.dataframe(catalogue.drop(columns="court"), hide_index=True, use_container_width=True)
    st.caption(str(len(catalogue)) + " entrées affichées sur " + str(len(CATALOGUE)) + ". Type « dossier » : un dossier "
               "du Drive téléchargé en entier. Type « api » : la seule source qui ne vient pas du Drive.")

    # --- Question 3 : comment ça marche ---
    question(3, "Que se passe-t-il quand on lance le script ?")
    colonnes = st.columns(4)
    for colonne, (titre, texte) in zip(colonnes, [
            ("1. Lire la demande", "Sans argument : la liste des sources. « tout » : toutes. Un groupe "
                                   "(parcours_court) est remplacé par ses fichiers, sans doublon."),
            ("2. Vérifier les noms", "Un nom inconnu arrête tout **avant** le premier téléchargement : "
                                     "pas de demi-téléchargement pour une faute de frappe."),
            ("3. Télécharger", "Fichier déjà présent : sauté. Dossier : relancé, mais seuls les fichiers "
                               "manquants arrivent (resume=True)."),
            ("4. Faire le bilan", "Les sources incomplètes sont listées, avec la commande exacte à relancer.")]):
        with colonne.container(border=True):
            st.markdown("**" + titre + "**")
            st.caption(texte)


# =============================================================================
# 3. PRÉPARATION
# =============================================================================
# (numéro, emoji, surnom, ce qu'il fait, entrées, sorties, le piège évité, le contrôle)
BLOCS = [
    ("1", "🗺️", "L'annuaire des communes",
     "Relie chaque code commune ayant existé depuis 2003 à son code 2026, à son bassin de vie 2022 et à son "
     "**sous-bassin**. Toutes les sources passeront par cette table pour rejoindre le territoire. Il calcule aussi "
     "la superficie de chaque commune.",
     "table de passage 2003 → 2026, bassins de vie 2022, communes-france-2026.csv, superficie des arrondissements (API)",
     "liaison_commune_bv2022, liaison_bv_scindes, communes_superficie_2026",
     "Les communes ont fusionné ou se sont scindées depuis 2003 : une vente de 2015 porte parfois un code qui "
     "n'existe plus. Les arrondissements de Paris, Lyon et Marseille, absents de la table Insee, sont ajoutés un par un.",
     "Un code = une ligne ; aucun code sans sous-bassin ; la somme des superficies des arrondissements est à moins de "
     "10 % de celle de la ville (détecte une erreur d'unité hectares / km²)."),
    ("2", "🧹", "Le tamis",
     "Garde, dans DVF+ brut, les **ventes classiques d'un seul logement** : une maison ou un appartement (dépendances "
     "acceptées), sur une seule commune, en métropole. La VEFA, les échanges, les ventes de plusieurs logements et les "
     "locaux d'activité partent.",
     "DVF+ : un CSV par département (DOM exclus)",
     "dvf_plus_residentiel.parquet, bilan_filtrage.csv",
     "DVF+ brut ne tient pas en mémoire : lecture par morceaux de 200 000 lignes, un département à la fois. Une même "
     "colonne n'a pas le même type d'un département à l'autre : un schéma commun est construit avant la réunion.",
     "Bilan par département : mutations brutes, ventes retenues, part retenue. Si le fichier filtré existe déjà "
     "(parcours rapide), le bloc est sauté."),
    ("3", "🎲", "Le tirage au sort",
     "Tire un **échantillon stratifié** des ventes 2014-2024 : une strate = un département × une année, 10 % des ventes, "
     "au moins 300 (sans dépasser la strate). Chaque vente porte son poids w_h = N_h / n_h.",
     "dvf_plus_residentiel.parquet (bloc 2)",
     "dvf_plus_echantillon_apprentissage.parquet, liste des idmutation",
     "Avec un plancher de 300, un petit département est sur-représenté : toute statistique sur plusieurs départements "
     "doit être pondérée par w_h. Aucune vente de 2025 n'est tirée : 2025 est réservée au test.",
     "Graine fixée (42) : le tirage est reproductible. Aucune strate en écart, idmutation uniques."),
    ("3bis", "🔒", "Le coffre-fort du test",
     "Prépare les ventes du **modèle par vente** (H7) : l'échantillon d'apprentissage nettoyé, et un tirage unique "
     "des ventes 2025, rangé à part pour le test final. Le nombre de pièces est reconstitué.",
     "échantillon (bloc 3), DVF+ filtré (bloc 2), liaison (bloc 1)",
     "ventes_apprentissage.parquet, ventes_test_2025.parquet",
     "La valeur foncière n'est **pas** gardée : divisée par la surface, elle redonne exactement la cible (fuite). "
     "Le test 2025 a sa propre graine (2025) et n'est tiré qu'une fois.",
     "Assertion : aucune colonne valeurfonc ; le test ne contient que 2025."),
    ("4", "🎯", "La cible",
     "Calcule le **prix médian au m²** et le nombre de ventes, par type de bien, à trois mailles (commune, bassin de vie, "
     "sous-bassin) et trois pas de temps (mois, trimestre, année) : neuf tables.",
     "DVF+ filtré (bloc 2) : toutes les ventes, pas l'échantillon ; liaison (bloc 1)",
     "cible_<maille>_<pas>.parquet, dont cible_sbv_annee",
     "Ventes sans valeur foncière ou hors 50 – 50 000 €/m² écartées. Aucun seuil de ventes ici : il est appliqué à la "
     "fusion, à partir des nombres de ventes gardés dans la table.",
     "Aucune vente perdue : la somme des ventes de chaque table égale le nombre de ventes retenues."),
    ("5", "💶", "Les revenus (Filosofi)",
     "Empile **huit millésimes** de niveau de vie par commune (2014 à 2021) en une seule table commune × année.",
     "Filo_disp_com : Excel de 2014 à 2017, CSV de 2018 à 2021",
     "filosofi_commune_2014_2021.parquet",
     "Huit fichiers, trois formats : en-tête en ligne 6, virgule décimale, « s » pour le secret statistique, année "
     "collée au nom de colonne (Q214 → Q2). Seules les colonnes présentes les huit années sont gardées.",
     "Aucun doublon commune × année."),
    ("6", "👨‍👩‍👧", "Les ménages",
     "Additionne les IRIS du recensement par commune : structure des ménages 2022, puis **effectifs** 2019 à 2022 par "
     "commune × année.",
     "base-ic-couples-familles-menages, millésimes 2019 à 2022",
     "menages_commune_2022.parquet, menages_commune_2019_2022.parquet",
     "On garde des effectifs, pas des ratios : ils s'additionnent à n'importe quelle maille. Les ratios sont calculés "
     "après, au sous-bassin (feature engineering).",
     "Une commune × année = une ligne ; aucun effectif vide."),
    ("7", "🏘️", "Le parc de logements",
     "Additionne les IRIS du recensement 2022 par commune et calcule **19 indicateurs** : vacance, résidences "
     "secondaires, taille, âge du parc, propriétaires, voiture, suroccupation…",
     "base-ic-logement-2022",
     "logement_commune_2022.parquet",
     "Les effectifs bruts sont gardés à côté des indicateurs, pour les recalculer au sous-bassin. L'Alsace-Moselle "
     "(absente de DVF+) est signalée, pas supprimée ; les communes de moins de 50 logements aussi.",
     "Toutes les parts entre 0 et 100 %."),
    ("8", "🚨", "La délinquance",
     "Met en forme les faits enregistrés par commune et par année (2016-2025) : une colonne par indicateur, "
     "avec la population et les logements pour calculer des taux.",
     "dataset_delinquance_communale.parquet (SSMSI)",
     "delinquance_commune_annee.parquet",
     "Une valeur masquée (secret statistique) reçoit le complément départemental, **jamais 0** : un zéro ferait croire "
     "à une commune sans délit. Le nombre d'indicateurs estimés est gardé pour mesurer la fiabilité.",
     "Une commune × année = une ligne."),
    ("9", "🏚️", "Les logements vacants (LOVAC)",
     "Passe les millésimes 2020 à 2026 en format long et calcule les taux de vacance du parc privé.",
     "lovac-open-data-2020-a-2026.xlsx",
     "lovac_commune_annee.parquet",
     "Le millésime N décrit le 1er janvier N-1. Un taux caché est complété par celui de l'intercommunalité, et signalé "
     "(taux_impute). Une rupture de série est marquée à partir du millésime 2024.",
     "Aucun taux hors de [0, 100] ; une commune × millésime = une ligne."),
    ("10", "🏪", "Les équipements 2015 et 2020",
     "Reprend 152 indicateurs de présence d'équipements (1 présent, 0 absent) par commune et millésime.",
     "table construite par CB",
     "equipements_commune_millesime.parquet",
     "Les prix de l'ancienne table sont retirés : la cible vient du bloc 4. Remplacé au feature engineering par "
     "le bloc 12, plus récent.",
     "Une commune × millésime = une ligne."),
    ("11", "📉", "Les taux de crédit",
     "Range la série mensuelle de la Banque de France en trois tables : mois, trimestre, année.",
     "Taux d'intérêt annuel des crédits immobiliers.csv",
     "taux_credit_mois, _trimestre, _annee",
     "Mêmes clés de période que les tables cibles du bloc 4. Une valeur nationale, la même partout : non retenue dans "
     "le modèle annuel par sous-bassin.",
     "Aucun mois en double."),
    ("12", "🏥", "Les équipements 2025",
     "Compte les équipements par commune dans **six domaines** : services, commerces, santé, transports, sports, "
     "tourisme (enseignement exclu).",
     "BPE25.parquet, BPE en évolution",
     "equipements_commune_2025.parquet",
     "Seuls les types d'équipement présents dans la BPE en évolution sont comptés : les nombres restent comparables à "
     "ceux de l'EDA. Une commune absente compte 0 équipement.",
     "Aucun équipement perdu entre la BPE et la table par commune."),
]

with onglets[2]:
    cle("**Douze blocs**, une source par bloc : chacun lit ses fichiers, les nettoie, se contrôle et écrit un "
        "fichier à la commune.",
        "Le bloc 1 est la pièce maîtresse : la table de liaison qui ramène toute source, quel que soit son code commune, "
        "à son sous-bassin de vie. Chaque bloc se lance seul et dit ce qui lui manque.")

    # --- Question 1 : l'ordre ---
    question(1, "Dans quel ordre ?")
    st.write("Trois temps suffisent. Les blocs 5 à 11 ne dépendent de personne : ils se lancent à tout moment.")
    st.graphviz_chart("""
    digraph {
        rankdir=LR; node [shape=box, style=rounded, fontsize=12];
        t1 [label="Temps 1\\n\\n1 · liaison\\n2 · tamis DVF+"];
        t2 [label="Temps 2\\n\\n3 · échantillon\\n4 · cible\\n12 · BPE 2025"];
        t3 [label="Temps 3\\n\\n3 bis · ventes H7"];
        libres [label="À tout moment\\n\\n5 · 6 · 7 · 8 · 9 · 10 · 11"];
        t1 -> t2 -> t3;
    }""", use_container_width=True)

    # --- Question 2 : chaque bloc ---
    question(2, "Que fait chaque bloc ?")
    choix = st.pills("Choisir un bloc", [b[0] for b in BLOCS], default="1", key="prep_bloc",
                     format_func=lambda n: next(b[1] + " " + n for b in BLOCS if b[0] == n))
    if choix is None:
        st.caption("Cliquez sur un bloc pour l'ouvrir.")
    else:
        numero, emoji, surnom, fait, entrees, sorties, piege, controle = next(b for b in BLOCS if b[0] == choix)
        with st.container(border=True):
            st.markdown("#### " + emoji + " Bloc " + numero + " · " + surnom)
            st.write(fait)
            gauche, droite = st.columns(2)
            gauche.markdown("**📥 Entrée**  \n" + entrees)
            droite.markdown("**📤 Sortie** (data/processed/)  \n" + sorties)
            st.markdown("**⚠️ Le piège évité**  \n" + piege)
            st.markdown("**✅ Le contrôle**  \n" + controle)


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
# (section, source, temps, variables produites, règle clé)
SECTIONS_FE = [
    ("1", "Délinquance", "Chaque année · faits N → prix N+1",
     "Un taux lissé sur 3 ans par famille (8 familles), part d'indicateurs estimés",
     "Faits et habitants additionnés avant de diviser ; pour 1 000 habitants (1 000 logements pour les cambriolages)"),
    ("2", "LOVAC", "Photo fixe · millésime 2023", "Taux de vacance, part des logements au taux complété",
     "Millésime d'avant la rupture de série ; moyenne pondérée par le nombre de logements"),
    ("3", "Équipements", "Photo fixe · BPE 2025", "Équipements pour 10 000 habitants, par domaine",
     "Habitants de 2024, l'année la plus proche ; sous-bassin sans équipement = 0"),
    ("3 bis", "Densité", "Photo fixe", "Habitants 2024 par km²",
     "Paris, Lyon, Marseille : superficie réelle de chaque arrondissement"),
    ("4", "Parc de logements", "Photo fixe · recensement 2022", "12 indicateurs (vacance, taille, âge du parc…)",
     "Effectifs additionnés au sous-bassin, mêmes formules qu'au bloc 7"),
    ("5", "Filosofi", "Photo fixe · millésime 2021", "Niveau de vie médian, part des ménages couverts",
     "Moyenne des médianes communales pondérée par les ménages fiscaux"),
    ("5 bis", "Ménages", "Chaque année · 2019 à 2024", "5 ratios (personnes seules, familles…)",
     "2023 = copie de 2022 ; 2024 = 2022 + évolution 2020-2022 bornée"),
    ("6", "DVF+ (volume)", "Chaque année · ventes N-1 → prix N", "Nombre de ventes de l'année précédente, par type",
     "Aucune vente = 0, sauf pour 2014 (pas de ventes 2013 : vide)"),
    ("6 bis", "Emploi salarié privé", "Chaque année · 31/12 de N-1 → prix N", "Emploi par logement, évolution sur un an, part sous secret",
     "Exporté à part (variante N-2 testée dans H3c) : pas encore branché à la fusion"),
]

with onglets[3]:
    cle("Chaque source est ramenée **au sous-bassin de vie et à l'année du prix**, avec la même recette.",
        "Une source = une section = un fichier fe_cible_<source>_sbv.parquet, qui a exactement les lignes de la cible. "
        "Trois règles reviennent partout : additionner avant de diviser, n'utiliser que ce qui est connu en N-1, "
        "et dire quand une source est une photo fixe.")

    # --- Question 1 : la recette ---
    question(1, "Quelle est la recette ?")
    colonnes = st.columns(5)
    for colonne, (emoji, titre, texte) in zip(colonnes, [
            ("📥", "Lire", "la table de la source, à la commune (sortie de la préparation)"),
            ("🔗", "Rattacher", "chaque commune à son sous-bassin, par la table de liaison"),
            ("➕", "Additionner", "les effectifs au sous-bassin, puis calculer les taux"),
            ("⏱️", "Caler dans le temps", "décaler d'un an, ou garder une photo fixe"),
            ("🧷", "Joindre", "à gauche sur la cible : le nombre de lignes ne bouge pas")]):
        with colonne.container(border=True):
            st.markdown("### " + emoji)
            st.markdown("**" + titre + "**")
            st.caption(texte)

    # --- Question 2 : les sections ---
    question(2, "Que produit chaque section ?")
    st.dataframe(pd.DataFrame(SECTIONS_FE, columns=["Section", "Source", "Dans le temps", "Variables", "Règle clé"]),
                 hide_index=True, use_container_width=True)
    st.caption("Photo fixe : une seule valeur par sous-bassin, la même pour toutes les années. Elle dit où les prix "
               "sont élevés, pas quand ils changent.")


    # --- Question 3 : le calendrier ---
    question(4, "Que sait-on au moment de prévoir ?")
    st.write("Le modèle prévoit le prix de l'année N avec ce qui est **connu en N-1**. Chaque source est décalée "
             "en conséquence.")
    annee_prix = st.select_slider("Prix de l'année", list(range(2017, 2026)), value=2024, key="fe_annee")
    debut_lissage = max(2016, annee_prix - 3)
    calendrier = pd.DataFrame([
        ("Prix de l'année précédente", "prix " + str(annee_prix - 1), "Chaque année"),
        ("Délinquance", "faits " + (str(debut_lissage) + " à " if debut_lissage < annee_prix - 1 else "")
         + str(annee_prix - 1) + " (" + str(annee_prix - debut_lissage) + " an" + ("s" if annee_prix - debut_lissage > 1 else "")
         + " de lissage)", "Chaque année"),
        ("Volume de ventes", "ventes " + str(annee_prix - 1), "Chaque année"),
        ("Ménages", "recensement " + str(min(max(annee_prix, 2019), 2024))
         + (" (copie de 2022)" if annee_prix == 2023 else " (2022 prolongé)" if annee_prix >= 2024 else ""), "Chaque année"),
        ("Emploi salarié privé", "salariés au 31/12/" + str(annee_prix - 1), "Chaque année (hors fusion)"),
        ("Logements vacants", "millésime 2023", "Photo fixe"),
        ("Équipements", "BPE 2025", "Photo fixe"),
        ("Densité", "habitants 2024", "Photo fixe"),
        ("Parc de logements", "recensement 2022", "Photo fixe"),
        ("Niveau de vie", "Filosofi 2021", "Photo fixe"),
    ], columns=["Variable", "Donnée utilisée pour le prix " + str(annee_prix), "Type"])
    st.dataframe(calendrier, hide_index=True, use_container_width=True)
    st.caption("Délinquance : le lissage porte sur 3 ans, mais la fenêtre grandit au début de la série "
               "(les faits commencent en 2016) : 1 an pour les prix 2017, 2 ans pour 2018.")


# =============================================================================
# 5. FUSION (sections 7 et 8)
# =============================================================================
with onglets[4]:
    cle("Huit tables, **une ligne par sous-bassin × année × type de bien** : la table de modélisation.",
        "Le seuil de 20 ventes est appliqué avant de calculer le prix de l'année précédente, et seules les lignes "
        "qui ont un prix et un prix N-1 sont gardées : 823 sous-bassins d'appartements et 1 777 de maisons en 2024.")

    # --- Question 1 : assembler ---
    question(1, "Comment les huit tables s'assemblent-elles ?")
    gauche, droite = st.columns([3, 2])
    with gauche:
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [shape=box, style=rounded, fontsize=10];
            cible [label="cible_sbv_annee", shape=note];
            subgraph cluster_fe { label="fe_cible_<source>_sbv"; fontsize=10;
                d [label="délinquance"]; v [label="vacance"]; e [label="équipements"]; de [label="densité"];
                l [label="logement"]; f [label="filosofi"]; m [label="ménages"]; dv [label="volume DVF+"]; }
            toutes [label="fe_cible_toutes_sources_sbv\\n(section 7)"];
            table [label="table_modelisation_bv\\n(section 8)", shape=note];
            cible -> toutes; d -> toutes; v -> toutes; e -> toutes; de -> toutes;
            l -> toutes; f -> toutes; m -> toutes; dv -> toutes; toutes -> table;
        }""", use_container_width=True)
    with droite:
        st.write("**Trois garde-fous à chaque jointure**")
        st.markdown(
            "- Jointure sur **sous-bassin × année**, `validate=\"one_to_one\"` : une ligne ne peut pas se dédoubler.\n"
            "- Une colonne présente dans deux sources serait renommée _x / _y **en silence** : c'est interdit, "
            "le script s'arrête.\n"
            "- Le nombre de lignes final doit être celui de la cible.")

    # --- Question 2 : du large au long ---
    question(2, "Pourquoi passer en format long ?")
    st.write("Jusqu'ici, une ligne porte les maisons **et** les appartements. Le modèle veut une ligne par type de bien : "
             "les colonnes propres au type deviennent communes, les variables du territoire sont recopiées.")
    large = pd.DataFrame({"sous_bassin_de_vie": ["01004_1"], "annee": [2024], "n_maison": [45], "n_appart": [12],
                          "prix_m2_med_maison": [2100], "prix_m2_med_appart": [2600], "densite_population": [350]})
    long_ex = pd.DataFrame({"sous_bassin_de_vie": ["01004_1"] * 2, "annee": [2024] * 2,
                            "type_bien": ["maison", "appartement"], "maison": [1, 0], "n_ventes": [45, 12],
                            "prix": [2100, 2600], "densite_population": [350, 350]})
    st.write("Avant (une ligne)")
    st.dataframe(large, hide_index=True, use_container_width=True)
    st.write("Après (deux lignes)")
    st.dataframe(long_ex, hide_index=True, use_container_width=True)
    st.caption("Exemple fictif. Une seule indicatrice, « maison » : maison + appartement = 1 toujours, une seconde "
               "colonne serait redondante. n_ventes sert au seuil et n'est **jamais** une variable du modèle "
               "(c'est le nombre de ventes de l'année à prévoir : une fuite).")

    # --- Question 3 : seuil avant prix N-1 ---
    question(3, "Pourquoi le seuil passe-t-il avant le prix de l'année précédente ?")
    st.write("Un sous-bassin a 12 ventes d'appartements en 2023 : sous le seuil, son prix médian n'est pas fiable. "
             "Que devient le prix N-1 de 2024 ?")
    ordre = st.segmented_control("Ordre des étapes", ["Seuil, puis prix N-1", "Prix N-1, puis seuil"],
                                 default="Seuil, puis prix N-1", key="fusion_ordre")
    seuil_avant = ordre != "Prix N-1, puis seuil"
    exemple = pd.DataFrame({"annee": [2023, 2024], "n_ventes": [12, 25], "prix": [None if seuil_avant else 2600, 2650],
                            "prix_n1": [None, None if seuil_avant else 2600]})
    exemple["ligne utilisable"] = exemple["prix"].notna() & exemple["prix_n1"].notna()
    st.dataframe(exemple, hide_index=True, use_container_width=True)
    if seuil_avant:
        afficher_message("Le prix 2023 est vidé (12 ventes < 20) : 2024 n'a pas de prix N-1 fiable, la ligne n'est pas "
                         "utilisée. Le modèle ne part jamais d'un prix peu sûr.", "info")
    else:
        afficher_message("Le prix 2023, calculé sur 12 ventes, devient le point de départ de 2024 : le modèle s'appuierait "
                         "sur une médiane qui peut sauter de 15 points sans que le marché bouge.", "avertissement")
    st.caption("Exemple fictif. La ligne sous le seuil est gardée, seul son prix est vidé.")

    # --- Question 4 : ce qui sort ---
    question(4, "Que contient la table finale ?")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**🗂️ Index à trois niveaux**")
        st.write("année × sous-bassin × type de bien")
        st.caption("Pas de clé collée « 2020_01004_1_maison » : le code du sous-bassin contient déjà un « _ », "
                   "on ne saurait plus le retrouver.")
    with c2.container(border=True):
        st.markdown("**🚫 Hors modèle**")
        st.write("n_ventes (fuite), BV2022 (identifiant), annee")
        st.caption("BV2022 sert au bootstrap par bassin de vie : les sous-bassins d'un même bassin ne sont pas "
                   "indépendants.")
    with c3.container(border=True):
        st.markdown("**✂️ Colonnes retirées**")
        st.write("4 millésimes, 3 années de référence, 5 variables redondantes")
        st.caption("part_suroccupes, part_parking, taux3_destructions (|ρ| de 0,78 à 0,88), taille_menage, "
                   "taille_moyenne_menage (|ρ| de 0,92 à 0,99) : audit de la table de modélisation.")
    st.write("**Contrôles avant export** : lignes exactement doublées par le passage en long, index unique, prix N-1 "
             "de 2024 = prix 2023 sur toutes les lignes, et **823 / 1 777** sous-bassins utilisables en 2024 "
             "(la référence du rapport). Un écart arrête le script.")
    st.caption("Source : feature_engineering.py, sections 7 et 8.")