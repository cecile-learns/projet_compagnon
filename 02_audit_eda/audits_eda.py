# =============================================================================
# 02 - AUDITS ET ANALYSE EXPLORATOIRE - projet Compagnon Immobilier
# =============================================================================
# Version script de audits.ipynb et eda.ipynb (textes et interprétations dans
# les notebooks). À lancer depuis la racine du projet, après 00_telechargement
# et 01_preparation :
#   python 02_audit_eda/audits_eda.py           -> audits puis EDA
#   python 02_audit_eda/audits_eda.py audits    -> audits seulement
#   python 02_audit_eda/audits_eda.py eda       -> EDA seulement
# Les tableaux s'affichent dans la console ; les graphiques sont enregistrés
# dans 02_audit_eda/figures/, nommés d'après la partie et la section.
# Audit 8 (équipements et sources complémentaires) et EDA B4 / C6 : en attente
# des fichiers de Christian, absents de ce script.

import gc
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas de fenêtre : les graphiques sont enregistrés en fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# config.py se trouve dans le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_OUT

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# -----------------------------------------------------------------------------
# Titres des sections et enregistrement des graphiques
# -----------------------------------------------------------------------------
FIGURES = Path(__file__).resolve().parent / "figures"
_fig = {"partie": "", "section": "", "n": 0}


def _nom_court(texte):
    """'B2.1 Distribution du taux de vacance' -> 'b2_1_distribution_du_taux_de_vacance'"""
    sans_accent = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sans_accent.lower()).strip("_")[:80]


def enregistrer_figures(*args, **kwargs):
    """Remplace plt.show() : enregistre les graphiques ouverts, puis les ferme."""
    FIGURES.mkdir(exist_ok=True)
    for num in plt.get_fignums():
        _fig["n"] += 1
        nom = f"{_fig['partie']}_{_fig['section']}_{_fig['n']}.png"
        plt.figure(num).savefig(FIGURES / nom, bbox_inches="tight")
    plt.close("all")


plt.show = enregistrer_figures


def afficher_section(texte):
    """Affiche le titre de la section dans la console et nomme ses graphiques."""
    enregistrer_figures()   # graphique éventuellement resté ouvert
    print("\n" + "=" * 79 + "\n" + texte + "\n" + "=" * 79)
    _fig["section"], _fig["n"] = _nom_court(texte), 0


# -----------------------------------------------------------------------------
# Tailles d'effet utilisées dans les analyses
# -----------------------------------------------------------------------------
def eta2_kruskal(groupes):
    """Kruskal-Wallis + taille d'effet eta² (part de la variabilité des rangs expliquée par les groupes)."""
    H, p = stats.kruskal(*groupes)
    k, n = len(groupes), sum(len(g) for g in groupes)
    return H, p, (H - k + 1) / (n - k)


def r_mannwhitney(a, b):
    """Mann-Whitney + taille d'effet r rang-bisérial (de -1 à 1).
    r > 0 : les valeurs de a ont tendance à être plus élevées que celles de b."""
    U, p = stats.mannwhitneyu(a, b)
    return U, p, 2 * U / (len(a) * len(b)) - 1


# -----------------------------------------------------------------------------
# Parties à lancer
# -----------------------------------------------------------------------------
PARTIES = sys.argv[1:] or ["audits", "eda"]
_inconnues = [p for p in PARTIES if p not in ("audits", "eda")]
if _inconnues:
    print("Partie(s) inconnue(s) :", ", ".join(_inconnues), "| choix possibles : audits, eda")
    sys.exit(1)
NOMS_INITIAUX = set(globals()) | {"NOMS_INITIAUX"}


# #############################################################################
# AUDITS DES SOURCES
# #############################################################################
if "audits" in PARTIES:
    _fig["partie"] = "audits"

    # =========================================================================
    # 1. DVF+
    # =========================================================================

    # -------------------------------------------------------------------------
    # Chargement par échantillonnage
    # -------------------------------------------------------------------------
    afficher_section('1. DVF+ | Chargement par échantillonnage')
    chemin_audit = DATA_OUT / "dvf_plus_echantillon_100k.csv"
    deja_present = chemin_audit.exists()
    if deja_present:
        print("Échantillon d'audit déjà présent :", chemin_audit.name, "-> le tirage n'est pas refait")

    if not deja_present:
        #Effectis et quotas proportionnels pour les fichiers DVF+ (sans charger les CSV en mémoire)
        from pathlib import Path
        import pandas as pd
        import numpy as np

        dossier = DATA_RAW / "DVF+"
        fichiers = sorted(dossier.glob("dvf_plus_d*.csv"))

        # Compter les lignes sans charger les CSV en mémoire
        effectifs = {}

        for fichier in fichiers:
            with open(fichier, "rb") as f:
                effectifs[fichier.name] = sum(1 for _ in f) - 1

        repartition = pd.DataFrame.from_dict(
            effectifs,
            orient="index",
            columns=["Effectif"]
        )

        total = repartition["Effectif"].sum()
        objectif = 100_000

        # Calculer les proportions
        repartition["Proportion (%)"] = (
            repartition["Effectif"] / total * 100
        ).round(3)

        # Calculer les quotas et ajuster les arrondis
        quotas_exacts = repartition["Effectif"] / total * objectif

        repartition["Quota"] = np.floor(quotas_exacts).astype(int)

        reste = objectif - repartition["Quota"].sum()

        indices = (
            (quotas_exacts - repartition["Quota"])
            .sort_values(ascending=False)
            .head(reste)
            .index
        )

        repartition.loc[indices, "Quota"] += 1

        display(repartition.sort_values("Effectif", ascending=False))

        print("Total des lignes sources :", total)
        print("Total des quotas :", repartition["Quota"].sum())

    if not deja_present:
        sortie = chemin_audit
        rng = np.random.default_rng(42)

        premier_bloc = True
        total_extrait = 0
        colonnes_reference = None

        for fichier in fichiers:
            n_total = int(repartition.loc[fichier.name, "Effectif"])
            quota = int(repartition.loc[fichier.name, "Quota"])

            # Tirer les positions aléatoires sans remise
            positions = np.sort(
                rng.choice(n_total, size=quota, replace=False)
            )

            debut = 0

            for bloc in pd.read_csv(
                fichier,
                sep="|",
                chunksize=10_000,
                low_memory=False
            ):
                # Vérifier que les colonnes sont identiques
                if colonnes_reference is None:
                    colonnes_reference = bloc.columns.tolist()
                elif bloc.columns.tolist() != colonnes_reference:
                    raise ValueError(
                        f"Colonnes différentes dans {fichier.name}"
                    )

                fin = debut + len(bloc)

                indices = positions[
                    (positions >= debut) & (positions < fin)
                ] - debut

                if len(indices) > 0:
                    extrait = bloc.iloc[indices].copy()

                    extrait["departement_source"] = (
                        fichier.stem.removeprefix("dvf_plus_d")
                    )

                    extrait.to_csv(
                        sortie,
                        mode="w" if premier_bloc else "a",
                        header=premier_bloc,
                        index=False,
                        sep="|"
                    )

                    premier_bloc = False
                    total_extrait += len(extrait)

                debut = fin

            if debut != n_total:
                raise ValueError(
                    f"Nombre de lignes incohérent dans {fichier.name}"
                )

            print(f"{fichier.name} : {quota} lignes extraites")

        print("\nTotal extrait :", total_extrait)
        print("Fichier créé :", sortie)

    df = pd.read_csv(chemin_audit, sep="|", low_memory=False)

    # -------------------------------------------------------------------------
    # A. Identité
    # -------------------------------------------------------------------------

    # --- 2. Dimensions, unité d'observations ---
    afficher_section("1. DVF+ | 2. Dimensions, unité d'observations")
    print(df.shape)
    print(df.info())
    display(df.head())

    #Vérification de l'unicité des identifiants de mutation
    display(df["idmutation"].nunique())

    # --- 3. Temporalité ---
    afficher_section('1. DVF+ | 3. Temporalité')
    # Bornes temporelles observées
    print("Première date :", df["datemut"].min())
    print("Dernière date :", df["datemut"].max())

    # Répartition des mutations par année
    print("\nMutations par année :")
    print(df["anneemut"].value_counts().sort_index())

    # -------------------------------------------------------------------------
    # B. Variables
    # -------------------------------------------------------------------------

    # --- 3. Dictionnaire des variables ---
    afficher_section('1. DVF+ | 3. Dictionnaire des variables')
    #Vérification des variables de type str selon indications de la documentation DVF+
    for col in ['coddep', 'libnatmut', 'libtypbien', 'departement_source']:
        print(f"\nVariable : {col}")
        print(df[col].value_counts())
    display(df['vefa'].value_counts())

    # 1. Variables qualitatives textuelles
    df_str_cat = df[['coddep', 'libnatmut', 'libtypbien']].copy()

    # 2. Variable qualitative booléenne
    df_cat = df[['vefa']].copy()

    # 3. Variables quantitatives discrètes (dénombrements)
    df_num_discrete = df[[
        'nbartcgi', 'nbdispo', 'nblot', 'nbcomm',
        'nbsection', 'nbpar', 'nbparmut', 'nbsuf',
        'nbvolmut', 'nblocmut', 'nblocmai', 'nblocapt',
        'nblocdep', 'nblocact',
        'nbapt1pp', 'nbapt2pp', 'nbapt3pp', 'nbapt4pp', 'nbapt5pp',
        'nbmai1pp', 'nbmai2pp', 'nbmai3pp', 'nbmai4pp', 'nbmai5pp'
    ]].copy()

    # 4. Variables quantitatives continues (valeurs et surfaces)
    df_num_quanti = df[[
        'valeurfonc', 'sterr', 'sbati', 'sbatmai', 'sbatapt', 'sbatact',
        'sapt1pp', 'sapt2pp', 'sapt3pp', 'sapt4pp', 'sapt5pp',
        'smai1pp', 'smai2pp', 'smai3pp', 'smai4pp', 'smai5pp'
    ]].copy()


    print("Qualitatives textuelles :", df_str_cat.shape)
    print("Qualitative booléenne   :", df_cat.shape)
    print("Quantitatives discrètes :", df_num_discrete.shape)
    print("Quantitatives continues :", df_num_quanti.shape)
    for col in df_num_discrete :
        print(f"\nVariable : {col}")
        print(df[col].value_counts())

    display(df_num_quanti.describe().T)

    # --- 3.4. Synthèse et implications pour Compagnon Immo ---
    afficher_section('1. DVF+ | 3.4. Synthèse et implications pour Compagnon Immo')
    #Réalisation d'un sous échantillon résidentiel à partir de l'échantillon initial de 100 000 lignes
    # Nature des transactions retenues
    nature_ok = df["libnatmut"].isin([
        "Vente",
        "Vente en l'état futur d'achèvement"
    ])

    # Une seule maison, aucun appartement ni local d'activité
    maison_ok = (
        (df["libtypbien"] == "UNE MAISON")
        & (df["nblocmai"] == 1)
        & (df["nblocapt"] == 0)
        & (df["nblocact"] == 0)
    )

    # Un seul appartement, aucune maison ni local d'activité
    appartement_ok = (
        (df["libtypbien"] == "UN APPARTEMENT")
        & (df["nblocapt"] == 1)
        & (df["nblocmai"] == 0)
        & (df["nblocact"] == 0)
    )

    # Sous-échantillon résidentiel
    df_residentiel = df.loc[
        nature_ok & (maison_ok | appartement_ok)
    ].copy()

    print("Dimensions :", df_residentiel.shape)
    print("Part de l'échantillon initial :",
          round(len(df_residentiel) / len(df) * 100, 2), "%")



    # 1. Variables qualitatives textuelles
    df_str_cat = df_residentiel[[
        'coddep', 'libnatmut', 'libtypbien'
    ]].copy()

    # 2. Variable qualitative booléenne
    df_cat = df_residentiel[['vefa']].copy()

    # 3. Variables quantitatives discrètes (dénombrements)
    df_num_discrete = df_residentiel[[
        'nbartcgi', 'nbdispo', 'nblot', 'nbcomm',
        'nbsection', 'nbpar', 'nbparmut', 'nbsuf',
        'nbvolmut', 'nblocmut', 'nblocmai', 'nblocapt',
        'nblocdep', 'nblocact',
        'nbapt1pp', 'nbapt2pp', 'nbapt3pp', 'nbapt4pp', 'nbapt5pp',
        'nbmai1pp', 'nbmai2pp', 'nbmai3pp', 'nbmai4pp', 'nbmai5pp'
    ]].copy()

    # 4. Variables quantitatives continues (valeurs et surfaces)
    df_num_quanti = df_residentiel[[
        'valeurfonc', 'sterr', 'sbati', 'sbatmai', 'sbatapt', 'sbatact',
        'sapt1pp', 'sapt2pp', 'sapt3pp', 'sapt4pp', 'sapt5pp',
        'smai1pp', 'smai2pp', 'smai3pp', 'smai4pp', 'smai5pp'
    ]].copy()

    # Vérification
    print("Qualitatives textuelles :", df_str_cat.shape)
    print("Qualitative booléenne   :", df_cat.shape)
    print("Quantitatives discrètes :", df_num_discrete.shape)
    print("Quantitatives continues :", df_num_quanti.shape)

    # -------------------------------------------------------------------------
    # C. Qualité
    # -------------------------------------------------------------------------

    # --- 4. Doublons ---
    afficher_section('1. DVF+ | 4. Doublons')
    # Nombre de lignes strictement identiques
    display(df_residentiel.duplicated().sum())
    # Nombre de répétitions de l'identifiant
    display(df_residentiel.duplicated(subset=["idmutation"]).sum())
    print("Doublons exacts :", df_residentiel.duplicated().sum())
    print("Doublons idmutation :", df_residentiel.duplicated(subset=["idmutation"]).sum())

    # --- 5. Valeurs manquantes ---
    afficher_section('1. DVF+ | 5. Valeurs manquantes')
    #Vérification des quantité de valeur manquante dans le dataset
    missing = (df_residentiel.isna().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)

    display(missing)
    display(pd.crosstab(
        df_residentiel["nbparmut"] == 0,
        df_residentiel["l_idparmut"].isna(),
        margins=True
    )
    )
    print("Les 46,43 % de valeurs manquantes de l_idparmut correspondent exclusivement aux mutations pour lesquelles nbparmut = 0.")

    # --- 6 Distributions et valeurs extrêmes ---
    afficher_section('1. DVF+ | 6 Distributions et valeurs extrêmes')
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Visualisation des variables catégorielles
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flatten()

    for i, col in enumerate(df_str_cat.columns[:3]):
        sns.countplot(
            data=df_str_cat,
            y=col,
            order=df_str_cat[col].value_counts().index,
            ax=axes[i]
        )
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    # Masquer les emplacements inutilisés
    for ax in axes[len(df_str_cat.columns[:3]):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    #Visualisation des variables quantitatives discrètes
    # Visualisation des variables quantitatives discrètes
    fig, axes = plt.subplots(6, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, col in enumerate(df_num_discrete.columns):
        sns.countplot(
            data=df_num_discrete,
            y=col,
            order=df_num_discrete[col].value_counts().sort_index().index,
            ax=axes[i]
        )
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    # Visualisation des variables quantitatives continues
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, col in enumerate(df_num_quanti.columns):
        sns.histplot(
            data=df_num_quanti,
            x=col,
            bins=30,
            ax=axes[i]
        )
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    axes = axes.flatten()

    for i, col in enumerate(df_num_quanti.columns):
        sns.boxplot(
            data=df_num_quanti,
            x=col,
            ax=axes[i]
        )
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()

    # Vérification des valeurs nulles et des valeurs extrêmes
    colonnes = ["valeurfonc", "sbati", "sterr"]

    for col in colonnes:
        print(f"\nVariable : {col}")
        print("Valeurs nulles :", (df_residentiel[col] == 0).sum())
        print("Minimum :", df_residentiel[col].min())
        print("Quantile 95 % :", df_residentiel[col].quantile(0.95))
        print("Quantile 99 % :", df_residentiel[col].quantile(0.99))
        print("Maximum :", df_residentiel[col].max())
    display(pd.crosstab(
        df_residentiel["libtypbien"],
        df_residentiel["sterr"] == 0,
        normalize="index"
    ).mul(100).round(2))

    display(df_residentiel.groupby("libtypbien")["sterr"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
    ).round(2))

    # =========================================================================
    # 2. BASSINS DE VIE ET TABLE DE PASSAGE DES COMMUNES
    # =========================================================================

    # -------------------------------------------------------------------------
    # Bassins de vie 2022
    # -------------------------------------------------------------------------

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('2. Bassins de vie et table de passage des communes | 1. Situer la source et ouvrir les données')
    bv2022=pd.read_excel(DATA_RAW / 'BV2022_2022_2026.xlsx', sheet_name='BV2022', skiprows=5)
    compo_communale2022=pd.read_excel(DATA_RAW / 'BV2022_2022_2026.xlsx', sheet_name='Composition_communale', skiprows=5)

    display(bv2022.head())

    display(compo_communale2022.head())

    #Nombre de valeurs uniques par variable
    print('valeurs uniques dans les compositions communales 2022 :\n', compo_communale2022.nunique())
    print('valeurs uniques dans les bassins de vie 2022 :\n', bv2022.nunique())
    # Nombre d'annonces par département 
    print(compo_communale2022["DEP"].value_counts())

    print('nombre de communes totales de 2022 à 2026')
    print(compo_communale2022["DEP"].value_counts().sum())
    codgeo_bv2022=compo_communale2022.merge(bv2022, on=["BV2022", 'LIBBV2022'],how="left")

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('2. Bassins de vie et table de passage des communes | 2. Comprendre la structure et une ligne')
    display(codgeo_bv2022.head())

    print(codgeo_bv2022.shape) 
    display(codgeo_bv2022.head())
    codgeo_bv2022.info() 


    # La clé semble être le code géographique et on vérifie si elle est bien unique et jamais vide
    print("code géographique unique :", codgeo_bv2022["CODGEO"].is_unique)
    print("code géographique vide   :", codgeo_bv2022["CODGEO"].isna().sum())

    # On regarde s'il y a des doublons exacts dans le dataset sans prendre en compte l'Index
    display(codgeo_bv2022.duplicated().sum()) 

    # --- 3. Comprendre les variables ---
    afficher_section('2. Bassins de vie et table de passage des communes | 3. Comprendre les variables')
    codgeo_bv2022.info()

    display(codgeo_bv2022.nunique())

    # Mesurer les NaN par variable 
    manquants = pd.DataFrame({"nb_NaN": codgeo_bv2022.isna().sum(),
                              "pct_NaN": (codgeo_bv2022.isna().mean() * 100).round(1)})
    manquants = manquants[manquants["nb_NaN"] > 0].sort_values("pct_NaN", ascending=False)
    display(manquants)

    # --- Distribution des variables sur les bassins de vie ---
    afficher_section('2. Bassins de vie et table de passage des communes | Distribution des variables sur les bassins de vie')
    sns.set_theme(style="whitegrid")

    bv22 = codgeo_bv2022.copy()
    bv22["NB_COM"] = bv22["NB_COM"].astype(int)

    # Un bassin = une ligne
    b22 = bv22.drop_duplicates("BV2022")

    # Répartition des types de bassin 2022
    labels = {1: "Rural", 2: "Intermédiaire", 3: "Dense"}
    t = b22.TYPE_BV2022.map(labels).value_counts()
    t.plot.bar(title="Bassins 2022 par type", figsize=(6, 4))
    plt.show()

    # 3. Top 15 des plus grands bassins (en nb de communes)
    top = b22.nlargest(15, "NB_COM")
    sns.barplot(data=top, y="LIBBV2022", x="NB_COM", color="steelblue")
    plt.title("15 plus grands bassins de vie (2022)")
    plt.show()
    # 4. Nombre de bassins par département / région
    bv_dep = b22.groupby("DEP").size().sort_values(ascending=False).head(20)
    bv_dep.plot.bar(figsize=(10, 4), title="Bassins par département (top 20)")
    plt.show()

    # -------------------------------------------------------------------------
    # Exploitation du fichier codgeo_bv2022.csv
    # -------------------------------------------------------------------------

    # --- Structure de la table de passage des communes ---
    afficher_section('2. Bassins de vie et table de passage des communes | Structure de la table de passage des communes')
    import pandas as pd
    import numpy as np

    passage=pd.read_excel(DATA_RAW / 'table_passage_geo2003_geo2026.xlsx', sheet_name='Table de passage', skiprows=5, dtype=str)
    scissions=pd.read_excel(DATA_RAW / 'table_passage_geo2003_geo2026.xlsx', sheet_name='Liste des scissions', skiprows=5, dtype=str)
    fusions=pd.read_excel(DATA_RAW / 'table_passage_geo2003_geo2026.xlsx', sheet_name='Liste des fusions', skiprows=5, dtype=str)

    display(passage.head())

    print(passage.shape)

    passage.info()

    # --- Comprendre les variables ---
    afficher_section('2. Bassins de vie et table de passage des communes | Comprendre les variables')
    print('valeurs uniques dans les compositions communales 2022 :\n', passage.nunique())

    print('valeurs uniques dans les compositions communales 2022 :\n', compo_communale2022.nunique())

    doublons = passage[passage.duplicated(subset=['CODGEO_INI'], keep=False)].sort_values('CODGEO_INI')
    print(f"{len(doublons)} lignes en doublon sur 'CODGEO_INI'")
    print(doublons)

    # --- Ajout des arrondissements de Paris, Lyon et Marseille ---

    # --- Gestion des communes scindées ---
    afficher_section('2. Bassins de vie et table de passage des communes | Gestion des communes scindées')
    #fusion des tables des passages avec la composition communale 2022
    passage_compo_communale2022 = passage.merge(compo_communale2022, left_on="CODGEO_2026", right_on="CODGEO", how="left").drop(columns='CODGEO')
    display(passage_compo_communale2022.head())

    # 1. Doublons de CODGEO_INI : les communes qui ont été scindées
    dbl_ini = passage_compo_communale2022[passage_compo_communale2022.duplicated("CODGEO_INI", keep=False)].sort_values("CODGEO_INI")
    dbl_ini_bv_idem = dbl_ini[dbl_ini.duplicated('BV2022', keep=False)]
    print(f"{len(dbl_ini_bv_idem)} lignes en doublon sur CODGEO_INI et sur BV2022 ({dbl_ini_bv_idem['CODGEO_INI'].nunique()} codes)")
    print(dbl_ini_bv_idem)
    differences=dbl_ini[~dbl_ini['CODGEO_2026'].isin(dbl_ini_bv_idem['CODGEO_2026'])]
    print(differences)

    display(scissions.loc[scissions['COM_FIN'].isin(['52266', '55068', '76601', '76676'])])

    display(codgeo_bv2022[codgeo_bv2022['CODGEO'].isin(['76676', '76601'])])

    # 2. Doublons de CODGEO_2026 : communes qui ont été fusionnées
    dbl_2026 = passage_compo_communale2022[passage_compo_communale2022.duplicated("CODGEO_2026", keep=False)].sort_values(["CODGEO_2026", "CODGEO_INI"])
    dbl_2026_bv_idem = dbl_2026[dbl_2026.duplicated('BV2022', keep=False)]
    print(f"\n{len(dbl_2026_bv_idem)} lignes en doublon sur CODGEO_2026 avec BV2022 identique:")
    print(dbl_2026_bv_idem)
    display(dbl_2026.shape)

    display(dbl_2026_bv_idem.shape)

    # =========================================================================
    # 3. FILOSOFI
    # =========================================================================

    # --- Chargement ---
    afficher_section('3. Filosofi | Chargement')
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    filo = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet")

    pd.set_option("display.max_columns", None)
    print(filo.shape)

    # -------------------------------------------------------------------------
    # A. Identité
    # -------------------------------------------------------------------------

    # --- 2. Dimensions, unité d'observations ---
    afficher_section("3. Filosofi | 2. Dimensions, unité d'observations")
    print(filo.shape)
    print(filo.info())
    display(filo.head())

    # --- 3. Temporalité ---
    afficher_section('3. Filosofi | 3. Temporalité')
    # Bornes temporelles observées
    print("Première date :", filo["annee"].min())
    print("Dernière date :", filo["annee"].max())

    # Répartition des mutations par année
    print("\nMutations par année :")
    print(filo["annee"].value_counts().sort_index())

    # -------------------------------------------------------------------------
    # B. Variables
    # -------------------------------------------------------------------------

    # --- 4. Dictionnaire des variables ---

    # --- Comprendre les indicateurs de position et d'inégalité ---
    afficher_section("3. Filosofi | Comprendre les indicateurs de position et d'inégalité")
    # 1. Identifiant et variable temporelle
    df_ident = filo[['code_commune', 'annee']].copy()

    # 2. Variable quantitative discrète (dénombrement)
    df_num_discrete = filo[['NBMEN']].copy()

    # 3. Variables quantitatives continues
    df_num_quanti = filo[[
        # taille de la population
        'NBPERS', 'NBUC',
        # position dans la distribution des niveaux de vie
        'D1', 'D2', 'D3', 'D4', 'Q1', 'niveau_vie_median', 'Q3', 'D6', 'D7', 'D8', 'D9',
        # inégalités
        'Q3_Q1', 'RD', 'S80S20', 'GI',
        # composition du revenu disponible (%)
        'PACT', 'PBEN', 'PPEN', 'PPAT', 'PPSOC', 'PPFAM', 'PPMINI', 'PPLOGT', 'PIMPOT'
    ]].copy()


    print("Identifiant et temporelle :", df_ident.shape)
    print("Quantitative discrète     :", df_num_discrete.shape)
    print("Quantitatives continues   :", df_num_quanti.shape)
    display(df_num_discrete['NBMEN'].describe())

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Classes de plus en plus larges (logique logarithmique), en gardant les seuils Insee (50 et 1 000)
    bornes = [0, 50, 100, 250, 500, 1000, 2500, 10_000, 50_000, float("inf")]
    etiquettes = ["< 50", "50-99", "100-249", "250-499", "500-999",
                  "1 000-2 499", "2 500-9 999", "10 000-49 999", "≥ 50 000"]

    classes = pd.cut(df_num_discrete["NBMEN"], bins=bornes, labels=etiquettes, right=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.countplot(x=classes, ax=ax)
    ax.bar_label(ax.containers[0])
    ax.axvline(0.5, color="grey", ls="--")   # seuil de 50 ménages
    ax.axvline(4.5, color="grey", ls="--")   # seuil de 1 000 ménages
    ax.set_title("Communes × années selon le nombre de ménages fiscaux")
    ax.set_xlabel("Nombre de ménages fiscaux (pointillés : seuils de diffusion de l'Insee)")
    ax.set_ylabel("Nombre de lignes")
    ax.tick_params(axis="x", rotation=20)
    plt.show()

    display(df_num_quanti.describe().T)

    # Histogrammes des 26 variables continues (grille 7 × 4 = 28 cases)
    fig, axes = plt.subplots(7, 4, figsize=(20, 28))
    axes = axes.flatten()

    for i, col in enumerate(df_num_quanti.columns):
        sns.histplot(
            data=df_num_quanti,
            x=col,
            bins=30,
            ax=axes[i],
            log_scale=(col in ["NBPERS", "NBUC"]),   # très asymétriques : échelle logarithmique
        )
        axes[i].set_title(col)

    # Les 2 cases en trop restent vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # C. Qualité
    # -------------------------------------------------------------------------

    # --- 5.Doublons ---
    afficher_section('3. Filosofi | 5.Doublons')
    # 1. Doublons sur la clé commune × année
    print("Doublons commune × année :", filo.duplicated(subset=["code_commune", "annee"]).sum())

    # 2. Paris, Lyon, Marseille : commune entière vs somme des arrondissements
    plm = {
        "Paris":     ("75056", "751"),   # code commune, début des codes d'arrondissement
        "Lyon":      ("69123", "6938"),
        "Marseille": ("13055", "132"),
    }
    f21 = filo[filo["annee"] == 2021]
    for ville, (code, prefixe) in plm.items():
        commune = f21.loc[f21["code_commune"] == code, "NBMEN"].sum()
        arrond = f21[f21["code_commune"].str.startswith(prefixe)]
        print(f"{ville} 2021 | commune : {commune:,.0f} ménages | "
              f"{len(arrond)} arrondissements : {arrond['NBMEN'].sum():,.0f} ménages")
    # Codes utilisés par DVF+ pour Paris, Lyon et Marseille
    for dep in ["75", "69", "13"]:
        codes = pd.read_parquet(DATA_OUT / "dvf_plus_residentiel.parquet", columns=["l_codinsee"],
                                filters=[("coddep", "==", dep)])["l_codinsee"]
        ville = codes[codes.str.match(r"^(751|6938|132|75056|69123|13055)")]
        print(dep, sorted(ville.unique())[:5], "...", ville.nunique(), "codes")

    # --- 6. Valeurs manquantes ---
    afficher_section('3. Filosofi | 6. Valeurs manquantes')
    # Vérification des quantités de valeurs manquantes dans le dataset
    missing = (filo.isna().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)

    display(missing)
    grande = (filo["NBMEN"] >= 1000) | (filo["NBPERS"] >= 2000)

    display(pd.crosstab(
        grande.rename("≥ 1 000 ménages ou ≥ 2 000 personnes"),
        filo["D1"].isna().rename("D1 manquant"),
        margins=True
    )
    )
    print("Les indicateurs détaillés manquent pour toutes les communes sous le seuil de 1 000 ménages ou 2 000 personnes, "
          "et sont publiés pour toutes les autres, à une exception près.")

    # Nombre d'indicateurs détaillés vides par ligne : 0 ou 23 si le masquage se fait en bloc
    detail = [c for c in df_num_quanti.columns if c not in ["NBPERS", "NBUC", "niveau_vie_median"]]
    display(filo[detail].isna().sum(axis=1).value_counts())
    print("Les 23 indicateurs détaillés sont publiés ou masqués ensemble, jamais séparément.")

    # --- 7 Distributions et valeurs extrêmes ---
    afficher_section('3. Filosofi | 7 Distributions et valeurs extrêmes')
    fig, axes = plt.subplots(7, 4, figsize=(22, 28))
    axes = axes.flatten()

    for i, col in enumerate(df_num_quanti.columns):
        sns.boxplot(
            data=df_num_quanti,
            x=col,
            ax=axes[i],
            log_scale=(col in ["NBPERS", "NBUC"]),
        )
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Position des valeurs extrêmes : comparaison du 99e centile et du maximum
    quantiles = df_num_quanti.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).T.round(2)
    quantiles["max / p99"] = (quantiles["max"] / quantiles["99%"]).round(2)

    display(quantiles[["min", "1%", "5%", "50%", "95%", "99%", "max", "max / p99"]])
    # Les communes aux valeurs les plus isolées sont-elles les mêmes ?
    cols_vue = ["code_commune", "annee", "NBMEN", "niveau_vie_median", "D9", "GI", "S80S20", "PPAT", "PIMPOT"]
    display(filo.sort_values("S80S20", ascending=False)[cols_vue].head(30))
    # Série complète des deux communes suspectes
    for code in ["01160", "85284"]:
        print(f"=== {code} ===")
        display(filo[filo["code_commune"] == code]
                [["annee", "NBMEN", "D1", "niveau_vie_median", "D9", "GI", "S80S20", "PBEN", "PPAT"]])

    # =========================================================================
    # 4. MÉNAGES (RECENSEMENT 2022)
    # =========================================================================

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('4. Ménages (recensement 2022) | 1. Situer la source et ouvrir les données')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    menages = pd.read_csv(DATA_RAW / "base-ic-couples-familles-menages-2022_csv" / "base-ic-couples-familles-menages-2022.CSV",
                          sep=";", dtype={"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str})
    display(menages.head(10))

    # --- Ouverture de meta_menages qui contient le détails des variables ---
    afficher_section('4. Ménages (recensement 2022) | Ouverture de meta_menages qui contient le détails des variables')
    meta_menages = pd.read_csv(DATA_RAW / "base-ic-couples-familles-menages-2022_csv" / "meta_base-ic-couples-familles-menages-2022.CSV",
                               sep=";", dtype=str)

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('4. Ménages (recensement 2022) | 2. Comprendre la structure et une ligne')
    # ── 1. Dimensions, colonnes et quelques observations ───────────────────────
    print(menages.shape)
    menages.info()
    # ── 2. Une ligne en entier, transposée pour tout voir ──────────────────────
    display(menages.sample(1, random_state=0).T)
    # ── 4. Doublons exacts : lignes identiques sur toutes les colonnes ─────────
    print("Doublons exacts :", menages.duplicated().sum())
    print('codes IRIS en doubles :', menages['IRIS'].duplicated().sum())

    # --- 3. Comprendre les variables ---
    afficher_section('4. Ménages (recensement 2022) | 3. Comprendre les variables')
    from pandas.api.types import is_numeric_dtype

    # Liste qui va recevoir une ligne de résumé par colonne
    resume = []

    # On parcourt toutes les colonnes du DataFrame, une par une
    for col in menages.columns:

        # Si la colonne est entièrement vide : on le signale
        if menages[col].isna().all():
            valeurs = "100 % vide"

        # Si la colonne contient des nombres : on note le min et le max
        elif is_numeric_dtype(menages[col]):
            valeurs = "min = " + str(menages[col].min()) + ", max = " + str(menages[col].max())

        # Sinon, si elle a 6 modalités ou moins : on les liste (sans les valeurs manquantes)
        elif menages[col].nunique() <= 6:
            valeurs = str(list(menages[col].dropna().unique()))

        # Sinon : on note seulement le nombre de modalités
        else:
            valeurs = str(menages[col].nunique()) + " modalités"

        # Pourcentage de valeurs manquantes, arrondi à 1 décimale
        pct_na = round(menages[col].isna().mean() * 100, 1)

        # On ajoute la ligne de résumé à la liste
        resume.append({"variable": col,
                       "type": str(menages[col].dtype),
                       "valeurs": valeurs,
                       "% NaN": pct_na})

    # On transforme la liste en DataFrame, avec le nom de la variable comme index
    resume = pd.DataFrame(resume).set_index("variable")
    print(resume)

    # --- 5. Visualisations graphiques ---

    # --- Structure globale des ménages ---
    afficher_section('4. Ménages (recensement 2022) | Structure globale des ménages')
    menages_types = {
        'Personnes seules': menages['C22_MENPSEUL'].sum(),
        'Couples sans enfant': menages['C22_MENCOUPSENF'].sum(),
        'Couples avec enfant(s)': menages['C22_MENCOUPAENF'].sum(),
        'Familles monoparentales': menages['C22_MENFAMMONO'].sum(),
        'Ménages sans famille (coloc...)': menages['C22_MENSFAM'].sum()
    }
    s_men = pd.Series(menages_types)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1', '#c6dbef']
    wedges, texts, autotexts = ax.pie(s_men, labels=s_men.index, autopct='%1.1f%%', startangle=140, colors=colors)
    for autotext in autotexts:
        autotext.set_weight('bold')

    plt.title('1. Répartition des types de ménages en France (Insee 2022)', fontsize=13, fontweight='bold', pad=20)
    plt.show()

    # --- Structure par âge des personnes seules ---
    afficher_section('4. Ménages (recensement 2022) | Structure par âge des personnes seules')
    cols_age = {
        '15-24 ans': 'P22_POP1524_PSEUL',
        '25-54 ans': 'P22_POP2554_PSEUL',
        '55-79 ans': 'P22_POP5579_PSEUL',
        '80 ans et +': 'P22_POP80P_PSEUL'
    }

    values = {}
    for label, col in cols_age.items():
        s = pd.to_numeric(menages[col].astype(str).str.replace(',', '.'), errors='coerce')
        values[label] = s.sum() / 1e6 

    s_age = pd.Series(values)
    print("Series values (in M):")
    print(s_age)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.set_theme(style="whitegrid")

    bars = ax.bar(s_age.index, s_age.values, color='#2b5c8f', width=0.55)
    ax.set_ylabel('Millions de personnes', fontsize=11)
    ax.set_title('Répartition par tranche d\'âge des personnes vivant seules', fontsize=13, fontweight='bold', pad=15)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f'{yval:.2f} M', ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, max(s_age.values) * 1.18)
    plt.tight_layout()
    plt.show();

    # =========================================================================
    # 5. PARC DE LOGEMENT (RECENSEMENT 2022)
    # =========================================================================

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('5. Parc de logement (recensement 2022) | 1. Situer la source et ouvrir les données')
    import numpy as np
    import pandas as pd

    # ── 1. Chargement ────────────────────────────────────────────────────────────
    # sep=";" : séparateur des fichiers INSEE
    # dtype : les codes sont lus en texte pour garder les zéros du début ("01001") et la Corse ("2A004")
    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)

    # Dictionnaire des variables fourni par l'INSEE avec les données
    meta = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "meta_base-ic-logement-2022.CSV",
                       sep=";", dtype=str)

    # ── 2. La lecture s'est-elle bien passée ? ───────────────────────────────────
    # Si le séparateur était faux, on n'aurait qu'une seule colonne
    print("Dimensions :", logement.shape)
    display(logement.head(3))

    # Types : les codes en texte, les effectifs en nombres décimaux
    print(logement.dtypes.value_counts())

    # Les codes ont-ils gardé leur longueur ? (IRIS = 9 caractères, commune = 5)
    print("Longueur des codes IRIS :", logement["IRIS"].str.len().unique())
    print("Longueur des codes commune :", logement["COM"].str.len().unique())

    # Les accents du dictionnaire sont-ils bien lus ? (sinon : problème d'encodage)
    print("Exemple de libellé :", meta["LIB_VAR"].iloc[5])

    # ── 3. Toutes les colonnes sont-elles documentées dans le dictionnaire ? ─────
    variables_doc = list(meta["COD_VAR"].unique())
    non_documentees = []
    for c in logement.columns:
        if c not in variables_doc:
            non_documentees.append(c)
    print("Variables documentées :", len(variables_doc), "| Colonnes non documentées :", non_documentees)

    # ── 4. Population / objets : une ligne = un IRIS ? ───────────────────────────
    print("IRIS uniques :", logement["IRIS"].nunique(), "pour", len(logement), "lignes")
    non_irisee = logement["IRIS"].str[-4:] == "0000"            # commune non découpée en IRIS
    print("Communes non découpées (une seule ligne = la commune) :", non_irisee.sum())
    print("IRIS dans des communes découpées :", (~non_irisee).sum())

    # ── 5. Période : l'année figure dans le préfixe des noms de colonnes ─────────
    print("Préfixes des colonnes :", logement.columns[4:].str[:3].unique())

    # ── 6. Territoire ────────────────────────────────────────────────────────────
    # Département : 2 caractères en métropole (01, 2A, 68), 3 en outre-mer (971...)
    logement["dep"] = np.where(logement["COM"].str[:2] == "97",
                               logement["COM"].str[:3], logement["COM"].str[:2])
    print("Communes :", logement["COM"].nunique(), "| Départements :", logement["dep"].nunique())
    print("Outre-mer :", sorted(logement[logement["dep"].str[:2] == "97"]["dep"].unique()))
    print("Mayotte (976) présente ?", "976" in list(logement["dep"].unique()))
    print("IRIS en Alsace-Moselle (57, 67, 68) :", logement["dep"].isin(["57", "67", "68"]).sum())

    # ── 7. Cohérence avec les chiffres publiés par l'INSEE (fichier if388) ───────
    print("Total logements (millions) :", round(logement["P22_LOG"].sum() / 1e6, 2))

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('5. Parc de logement (recensement 2022) | 2. Comprendre la structure et une ligne')
    import pandas as pd

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)

    # ── 1. Dimensions, colonnes et quelques observations ─────────────────────────
    print("Lignes :", logement.shape[0], "| Colonnes :", logement.shape[1])
    print(list(logement.columns))                        # liste complète des colonnes

    # 5 lignes tirées au hasard (random_state : même tirage à chaque exécution)
    display(logement.sample(5, random_state=0).iloc[:, :10])

    # ── 2. Ce que représente une ligne : exemple d'une commune découpée ──────────
    # Colmar (68066) : plusieurs lignes, une par quartier (IRIS)
    display(logement[logement["COM"] == "68066"].iloc[:, :8])

    # ── 3. Clés possibles ────────────────────────────────────────────────────────
    # IRIS : une valeur différente pour chaque ligne ?
    print("IRIS uniques :", logement["IRIS"].nunique(), "/", len(logement))

    # COM : combien de lignes par commune ?
    lignes_par_commune = logement["COM"].value_counts()   # trié du plus grand au plus petit
    print("Communes :", len(lignes_par_commune))
    print("Communes avec une seule ligne :", (lignes_par_commune == 1).sum())
    print("Communes avec plusieurs lignes :", (lignes_par_commune > 1).sum())
    print(lignes_par_commune.head(5))                     # communes les plus découpées

    # Une commune peut-elle avoir à la fois une ligne "commune entière" (0000) et des IRIS ?
    logement["non_irisee"] = logement["IRIS"].str[-4:] == "0000"
    melange = logement.groupby("COM")["non_irisee"].nunique()   # 2 = les deux cas dans la même commune
    print("Communes mélangeant ligne entière et IRIS :", (melange > 1).sum())

    # Paris : codé par arrondissement (75101 à 75120) ou par commune (75056) ?
    print("Code 75056 (Paris commune) présent ?", (logement["COM"] == "75056").sum() > 0)
    print("Arrondissements parisiens :", logement[logement["COM"].str[:3] == "751"]["COM"].nunique())

    # ── 4. Doublons ──────────────────────────────────────────────────────────────
    # Doublons exacts : lignes identiques sur toutes les colonnes
    nb_doublons = logement.duplicated().sum()
    print("Doublons exacts :", nb_doublons, "(", round(nb_doublons / len(logement) * 100, 2), "% )")

    # Répétitions métier : deux IRIS différents avec exactement les mêmes chiffres ?
    # On compare les 97 colonnes de chiffres, sans les identifiants (colonnes 0 à 3)
    colonnes_chiffres = logement.columns[4:101]
    repetes = logement[logement.duplicated(subset=colonnes_chiffres, keep=False)]   # keep=False : toutes les lignes concernées
    print("Lignes aux chiffres identiques à une autre :", len(repetes))

    # Que sont ces lignes ? On regarde combien n'ont aucun logement
    print("dont IRIS avec 0 logement :", (repetes["P22_LOG"] == 0).sum())
    display(repetes[repetes["P22_LOG"] > 0].iloc[:, :10])          # les autres cas

    # --- 3. Comprendre les variables ---
    afficher_section('5. Parc de logement (recensement 2022) | 3. Comprendre les variables')
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)

    # ── 1. Sens des variables : on le prend dans le dictionnaire INSEE ───────────
    meta = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "meta_base-ic-logement-2022.CSV",
                       sep=";", dtype=str)
    meta = meta.drop_duplicates("COD_VAR").set_index("COD_VAR")    # une ligne par variable


    # ── 2. Nature statistique et unité : déduites du nom de la variable ─────────
    def nature_unite(col):
        if col in ["IRIS", "COM"]:
            return "qualitative nominale (identifiant)", "code"
        if col in ["TYP_IRIS", "LAB_IRIS"]:
            return "qualitative nominale", "modalités"
        if "NBPI" in col:                       # somme des pièces de tous les logements
            return "quantitative continue (total)", "pièces"
        if "NPER" in col or "PMEN" in col:      # somme des personnes
            return "quantitative continue (total)", "personnes"
        if "ANEM_RP" in col:                    # somme des années d'ancienneté d'emménagement
            return "quantitative continue (total)", "années"
        if "_MEN" in col:
            return "quantitative continue (effectif estimé)", "ménages"
        return "quantitative continue (effectif estimé)", "logements"


    # ── 3. Modalités ou plage de valeurs ────────────────────────────────────────
    def modalites(col):
        s = logement[col].dropna()                             # on ignore les valeurs manquantes
        if is_numeric_dtype(logement[col]):
            return "de " + str(round(s.min(), 1)) + " à " + str(round(s.max(), 1))
        if s.nunique() <= 6:
            return str(sorted(s.unique()))                     # peu de modalités : on les liste
        return str(s.nunique()) + " valeurs"                   # sinon : on les compte


    # ── 4. Type de stockage et cohérence avec la nature ─────────────────────────
    def coherence(col, nature):
        dtype = str(logement[col].dtype)
        manquants = round(logement[col].isna().mean() * 100, 1)
        texte = dtype + " : cohérent"
        if nature.startswith("quantitative") and not is_numeric_dtype(logement[col]):
            texte = dtype + " : incohérent"                    # un nombre stocké en texte
        if manquants > 0:
            texte = texte + " (" + str(manquants) + " % manquants)"
        return texte


    # ── 5. Assemblage du tableau (une ligne par variable) ───────────────────────
    lignes = []
    for col in logement.columns:
        nature, unite = nature_unite(col)
        lignes.append({"Variable": col,
                       "Signification": meta.loc[col, "LIB_VAR_LONG"],   # libellé long de l'INSEE
                       "Nature statistique": nature,
                       "Unité / modalités": unite + " ; " + modalites(col),
                       "Type de stockage et cohérence": coherence(col, nature)})
    tableau = pd.DataFrame(lignes).set_index("Variable")
    pd.set_option("display.max_colwidth", None)                # ne pas couper le texte
    display(tableau)


    # ── 6. Variables constantes, valeurs négatives, modalités de LAB_IRIS ───────
    constantes = []
    for col in logement.columns:
        if logement[col].nunique() <= 1:
            constantes.append(col)
    print("Variables constantes :", constantes)
    print("Valeurs négatives :", (logement.select_dtypes("number") < 0).sum().sum())
    print(logement["LAB_IRIS"].value_counts())


    # ── 7. Vérifier notre compréhension par des sommes ──────────────────────────
    # Si on a bien compris, ces sous-catégories doivent redonner le total des résidences principales
    rp = logement["P22_RP"].sum()
    pieces = logement[["P22_RP_1P", "P22_RP_2P", "P22_RP_3P", "P22_RP_4P", "P22_RP_5PP"]].sum().sum()
    statut = logement[["P22_RP_PROP", "P22_RP_LOC", "P22_RP_GRAT"]].sum().sum()
    occupation = logement[["C22_RP_NORME", "C22_RP_SOUSOCC_MOD", "C22_RP_SOUSOCC_ACC",
                           "C22_RP_SOUSOCC_TACC", "C22_RP_SUROCC_MOD", "C22_RP_SUROCC_ACC"]].sum().sum()
    print("Pièces (1 à 5+) / RP :", round(pieces / rp, 3))
    print("Propriétaires + locataires + gratuits / RP :", round(statut / rp, 3))
    print("Classes d'occupation C22 / RP :", round(occupation / rp, 3))
    print("Ménages = résidences principales ? écarts :", (abs(logement["P22_MEN"] - logement["P22_RP"]) > 0.5).sum())
    print("Construites avant 2020 (ACHTOT) / RP :", round(logement["P22_RP_ACHTOT"].sum() / rp, 3))

    # Les variables "NBPI" et "ANEM" sont des totaux : divisées par RP, elles donnent une moyenne
    print("Nombre moyen de pièces par résidence principale :", round(logement["P22_NBPI_RP"].sum() / rp, 1))
    print("Ancienneté moyenne d'emménagement (années) :", round(logement["P22_ANEM_RP"].sum() / rp, 1))

    # --- 4. Vérifier le périmètre réel ---
    afficher_section('5. Parc de logement (recensement 2022) | 4. Vérifier le périmètre réel')
    import numpy as np
    import pandas as pd

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)

    # Colonnes utiles pour décrire le périmètre
    # Département : 2 caractères en métropole (01, 2A, 68), 3 en outre-mer (971...)
    logement["dep"] = np.where(logement["COM"].str[:2] == "97",
                               logement["COM"].str[:3], logement["COM"].str[:2])
    logement["non_irisee"] = logement["IRIS"].str[-4:] == "0000"   # commune non découpée
    logement["outre_mer"] = logement["dep"].str[:2] == "97"

    # ── 1. Objets et groupes présents ────────────────────────────────────────────
    # Types d'IRIS : nombre de lignes ET nombre de logements dans chaque type
    groupes = logement.groupby("TYP_IRIS").agg(nb_lignes=("IRIS", "count"),
                                               logements=("P22_LOG", "sum"))
    groupes["logements"] = groupes["logements"].round(0)
    groupes["% logements"] = (groupes["logements"] / groupes["logements"].sum() * 100).round(1)
    display(groupes)

    # Communes découpées (villes) ou non : part des lignes et part des logements
    taille = logement.groupby("non_irisee").agg(nb_lignes=("IRIS", "count"),
                                                logements=("P22_LOG", "sum"))
    taille["% logements"] = (taille["logements"] / taille["logements"].sum() * 100).round(1)
    display(taille)

    # Lignes sans aucun logement
    print("IRIS sans logement :", (logement["P22_LOG"] == 0).sum())

    # ── 2. Temps : une seule année dans le nom des colonnes ──────────────────────
    # (colonnes 4 à 100 : les 97 effectifs, sans les colonnes ajoutées plus haut)
    print("Préfixes d'année :", logement.columns[4:101].str[:3].unique())

    # ── 3. Territoire ────────────────────────────────────────────────────────────
    print("Départements :", logement["dep"].nunique(), "| Communes :", logement["COM"].nunique())
    print("Mayotte (976) présente ?", "976" in list(logement["dep"].unique()))

    # Arrondissements de Paris (751..), Lyon (6938.) et Marseille (132..)
    arr_paris = logement[logement["COM"].str[:3] == "751"]["COM"].nunique()
    arr_lyon = logement[logement["COM"].str[:4] == "6938"]["COM"].nunique()
    arr_marseille = logement[logement["COM"].str[:3] == "132"]["COM"].nunique()
    print("Arrondissements Paris / Lyon / Marseille :", arr_paris, arr_lyon, arr_marseille)

    # Métropole / outre-mer : nombre de lignes et de logements
    print(logement.groupby("outre_mer")["P22_LOG"].agg(["count", "sum"]))

    # Alsace-Moselle (absente de DVF+)
    am = logement[logement["dep"].isin(["57", "67", "68"])]
    print("Alsace-Moselle :", len(am), "IRIS,", round(am["P22_LOG"].sum() / 1e6, 2), "million de logements,",
          round(am["P22_LOG"].sum() / logement["P22_LOG"].sum() * 100, 1), "% du total")

    # ── 4. Comparer les effectifs entre départements ────────────────────────────
    par_dep = logement.groupby("dep").agg(nb_lignes=("IRIS", "count"),
                                          logements=("P22_LOG", "sum"))
    par_dep = par_dep.sort_values("logements")
    print(par_dep.head(3))       # départements avec le moins de logements
    print(par_dep.tail(3))       # départements avec le plus de logements
    print(par_dep["logements"].describe().round(0))

    # ── 5. Point de comparaison externe : total publié par l'INSEE ──────────────
    # if388 (figure 2) : 24 023 milliers de logements en 1982, indice 156,2 en 2022
    publie_2022 = 24023 * 156.2 / 100 / 1000          # en millions
    print("Total du fichier :", round(logement["P22_LOG"].sum() / 1e6, 2), "millions")
    print("Total publié par l'INSEE pour 2022 :", round(publie_2022, 2), "millions")

    # --- 5. Repérer et investiguer les problèmes de qualité ---

    # --- 5.1 Valeurs manquantes ---
    afficher_section('5. Parc de logement (recensement 2022) | 5.1 Valeurs manquantes')
    import numpy as np
    import pandas as pd

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)
    # Département et territoire (métropole / outre-mer), pour étudier la concentration des absences
    logement["dep"] = np.where(logement["COM"].str[:2] == "97",
                               logement["COM"].str[:3], logement["COM"].str[:2])
    logement["outre_mer"] = logement["dep"].str[:2] == "97"

    # ── 1. Mesurer les absences par variable ─────────────────────────────────────
    manquants = pd.DataFrame({"nombre": logement.isna().sum(),
                              "%": (logement.isna().mean() * 100).round(1)})
    manquants = manquants[manquants["nombre"] > 0].sort_values("nombre", ascending=False)
    display(manquants)

    # ── Codes qui pourraient masquer une absence ─────────────────────────────────
    # Textes vides dans les colonnes de codes
    for col in ["IRIS", "COM", "TYP_IRIS", "LAB_IRIS"]:
        print(col, ": valeurs vides =", (logement[col].str.strip() == "").sum())

    # Valeurs "sentinelles" (ex. 9999 ou -1) souvent utilisées pour dire "inconnu"
    chiffres = logement.select_dtypes("number")
    print("Valeurs négatives :", (chiffres < 0).sum().sum())
    print("Valeurs égales à 9999, 99999 ou 999999 :", chiffres.isin([9999, 99999, 999999]).sum().sum())
    # Quelle ligne ? Est-ce plausible ? (on compare au nombre de résidences principales)
    suspect = logement[logement["P22_NBPI_RP"] == 9999]
    display(suspect[["IRIS", "COM", "P22_LOG", "P22_RP", "P22_NBPI_RP"]])
    print("Pièces par résidence principale :", round(9999 / suspect["P22_RP"].iloc[0], 1))

    # Le zéro : vrai zéro ou absence déguisée ?
    print("Part de zéros dans P22_LOGVAC :", round((logement["P22_LOGVAC"] == 0).mean() * 100, 1), "%")
    print("Part de zéros dans P22_RSECOCC :", round((logement["P22_RSECOCC"] == 0).mean() * 100, 1), "%")
    # Si ce sont de vrais zéros, ils doivent surtout concerner les petits IRIS
    zero_vac = logement[logement["P22_LOGVAC"] == 0]
    print("Logements médians des IRIS avec 0 vacant :", zero_vac["P22_LOG"].median())
    print("Logements médians de tous les IRIS :", logement["P22_LOG"].median())
    print("Part des IRIS à 0 vacant qui ont moins de 100 logements :", round((zero_vac["P22_LOG"] < 100).mean() * 100, 1), "%")

    # ── 2 et 3. Sens des absences : concentration métropole / outre-mer ─────────
    # Pour chaque variable incomplète : % de manquants en métropole et en outre-mer
    concentration = []
    for col in manquants.index:
        concentration.append({"variable": col,
                              "% manquants métropole": round(logement[logement["outre_mer"] == False][col].isna().mean() * 100, 1),
                              "% manquants outre-mer": round(logement[logement["outre_mer"] == True][col].isna().mean() * 100, 1)})
    display(pd.DataFrame(concentration).set_index("variable"))

    # Le dictionnaire le confirme-t-il ? (libellés des variables concernées)
    meta = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "meta_base-ic-logement-2022.CSV", sep=";", dtype=str)
    meta = meta.drop_duplicates("COD_VAR").set_index("COD_VAR")
    for col in manquants.index:
        print(col, ":", meta.loc[col, "LIB_VAR_LONG"])

    # --- 5.2 Formats, modalités et incohérences ---
    afficher_section('5. Parc de logement (recensement 2022) | 5.2 Formats, modalités et incohérences')
    import pandas as pd

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)
    L = logement                      # nom court pour alléger les formules
    tol = 0.5                         # tolérance : les effectifs sont des estimations avec décimales

    # ── 1. Formats des codes ─────────────────────────────────────────────────────
    # IRIS : 9 caractères ; COM : 5 caractères ; chiffres sauf en Corse (2A / 2B)
    print("IRIS de longueur ≠ 9 :", (L["IRIS"].str.len() != 9).sum())
    print("COM de longueur ≠ 5 :", (L["COM"].str.len() != 5).sum())
    print("COM non numériques :", (~L["COM"].str.isdigit()).sum(),
          "dont Corse (2A/2B) :", L["COM"].str[:2].isin(["2A", "2B"]).sum())
    print("COM ≠ 5 premiers caractères de IRIS :", (L["IRIS"].str[:5] != L["COM"]).sum())

    # Type d'IRIS et commune non découpée : Z doit correspondre exactement aux codes en 0000
    non_irisee = L["IRIS"].str[-4:] == "0000"
    print("Z sans code 0000 :", ((L["TYP_IRIS"] == "Z") & ~non_irisee).sum())
    print("Code 0000 sans type Z :", (non_irisee & (L["TYP_IRIS"] != "Z")).sum())
    print(pd.crosstab(L["LAB_IRIS"], L["TYP_IRIS"]))          # le label 5 correspond-il au type Z ?

    # Écriture des nombres : part d'entiers selon le type d'IRIS
    L["entier"] = L["P22_LOG"] % 1 == 0                        # True si pas de décimales
    print((L.groupby("TYP_IRIS")["entier"].mean() * 100).round(1))


    # ── 2 et 3. Règles de cohérence entre effectifs ──────────────────────────────
    def violations(nom, condition):
        """Affiche le nombre de lignes qui ne respectent pas la règle et les renvoie."""
        print(nom, ":", condition.sum())
        return L[condition]

    # Égalités : les sous-catégories doivent redonner le total
    v1 = violations("LOG ≠ RP + RSEC + VAC",
                    (L["P22_LOG"] - L["P22_RP"] - L["P22_RSECOCC"] - L["P22_LOGVAC"]).abs() > tol)
    v2 = violations("RP ≠ somme des pièces (1 à 5+)",
                    (L["P22_RP"] - L[["P22_RP_1P", "P22_RP_2P", "P22_RP_3P", "P22_RP_4P", "P22_RP_5PP"]].sum(axis=1)).abs() > tol)
    v3 = violations("RP ≠ somme des tranches de surface",
                    (L["P22_RP"] - L[["P22_RP_M30M2", "P22_RP_3040M2", "P22_RP_4060M2", "P22_RP_6080M2",
                                      "P22_RP_80100M2", "P22_RP_100120M2", "P22_RP_120M2P"]].sum(axis=1)).abs() > tol)
    v4 = violations("RP ≠ propriétaires + locataires + gratuits",
                    (L["P22_RP"] - L["P22_RP_PROP"] - L["P22_RP_LOC"] - L["P22_RP_GRAT"]).abs() > tol)
    v5 = violations("Ménages ≠ RP", (L["P22_MEN"] - L["P22_RP"]).abs() > tol)
    v6 = violations("Ménages ≠ somme des anciennetés d'emménagement",
                    (L["P22_MEN"] - L[["P22_MEN_ANEM0002", "P22_MEN_ANEM0204",
                                       "P22_MEN_ANEM0509", "P22_MEN_ANEM10P"]].sum(axis=1)).abs() > tol)
    v7 = violations("ACHTOT ≠ somme des époques",
                    (L["P22_RP_ACHTOT"] - L[["P22_RP_ACH1919", "P22_RP_ACH1945", "P22_RP_ACH1970",
                                             "P22_RP_ACH1990", "P22_RP_ACH2005", "P22_RP_ACH2019"]].sum(axis=1)).abs() > tol)
    v8 = violations("VOIT1P ≠ VOIT1 + VOIT2P",
                    (L["P22_RP_VOIT1P"] - L["P22_RP_VOIT1"] - L["P22_RP_VOIT2P"]).abs() > tol)

    # Inégalités : une partie ne peut pas dépasser le tout
    v9 = violations("Maisons + appartements > LOG", L["P22_MAISON"] + L["P22_APPART"] > L["P22_LOG"] + tol)
    v10 = violations("HLM > locataires", L["P22_RP_LOCHLMV"] > L["P22_RP_LOC"] + tol)
    v11 = violations("ACHTOT > RP", L["P22_RP_ACHTOT"] > L["P22_RP"] + tol)
    v12 = violations("Ménages avec voiture > ménages", L["P22_RP_VOIT1P"] > L["P22_MEN"] + tol)

    # Logements ni maison ni appartement (foyers, habitations mobiles...)
    autres = L["P22_LOG"] - L["P22_MAISON"] - L["P22_APPART"]
    print("Part des logements ni maison ni appartement :", round(autres.sum() / L["P22_LOG"].sum() * 100, 2), "%")
    display(L.loc[autres.sort_values().index[-3:], ["IRIS", "P22_LOG", "P22_MAISON", "P22_APPART", "P22_RSECOCC"]])

    # ── Plausibilité des moyennes (IRIS avec au moins 50 résidences principales) ─
    p = L[L["P22_RP"] >= 50]
    pieces = p["P22_NBPI_RP"] / p["P22_RP"]               # pièces par logement
    personnes = p["P22_NPER_RP"] / p["P22_RP"]            # personnes par ménage
    anciennete = p["P22_ANEM_RP"] / p["P22_RP"]           # années depuis l'emménagement
    print("Pièces par logement : de", round(pieces.min(), 1), "à", round(pieces.max(), 1))
    print("Personnes par ménage : de", round(personnes.min(), 1), "à", round(personnes.max(), 1))
    print("Ancienneté moyenne : de", round(anciennete.min(), 1), "à", round(anciennete.max(), 1), "ans")
    # Inspecter les cas extrêmes avant de conclure à une erreur
    display(p.loc[[personnes.idxmax(), pieces.idxmin()], ["IRIS", "TYP_IRIS", "P22_RP", "P22_NBPI_RP", "P22_NPER_RP"]])

    # --- 5.3 Distributions et valeurs extrêmes ---
    afficher_section('5. Parc de logement (recensement 2022) | 5.3 Distributions et valeurs extrêmes')
    import pandas as pd

    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    logement = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                           sep=";", dtype=codes_texte)

    # Noms des IRIS (dans le fichier meta) pour inspecter les cas extrêmes
    meta = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "meta_base-ic-logement-2022.CSV", sep=";", dtype=str)
    noms = meta[meta["COD_VAR"] == "IRIS"][["COD_MOD", "LIB_MOD"]]
    noms.columns = ["IRIS", "nom"]                       # mêmes noms de colonnes pour la jointure
    logement = logement.merge(noms, on="IRIS", how="left")

    # ── Taux calculés (IRIS d'au moins 50 logements et au moins 1 résidence principale) ─
    l = logement[(logement["P22_LOG"] >= 50) & (logement["P22_RP"] > 0)].copy()
    l["taux_vacance"] = l["P22_LOGVAC"] / l["P22_LOG"] * 100
    l["part_rsec"] = l["P22_RSECOCC"] / l["P22_LOG"] * 100
    l["part_maisons"] = l["P22_MAISON"] / l["P22_LOG"] * 100
    l["part_hlm"] = l["P22_RP_LOCHLMV"] / l["P22_RP"] * 100

    # ── 1. Distributions : quantiles des variables importantes ───────────────────
    # On ajoute le 1er et le 99e centile pour voir les extrémités de la distribution
    display(logement["P22_LOG"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(1))
    display(l[["taux_vacance", "part_rsec", "part_maisons", "part_hlm"]]
            .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(1))

    # ── 2. Valeurs éloignées : règle de l'écart interquartile (IQR) ──────────────
    # Une valeur est "extrême" si elle dépasse Q3 + 1,5 × (Q3 - Q1)
    for col in ["taux_vacance", "part_rsec", "part_hlm"]:
        q1 = l[col].quantile(0.25)
        q3 = l[col].quantile(0.75)
        seuil = q3 + 1.5 * (q3 - q1)
        nb = (l[col] > seuil).sum()
        print(col, ": seuil =", round(seuil, 1), "% |", nb, "IRIS au-dessus (", round(nb / len(l) * 100, 1), "% )")

    # Concentrations aux bornes (0 % et 100 %)
    print("IRIS à 0 % de HLM :", (l["part_hlm"] == 0).sum())
    print("IRIS à 100 % de maisons :", (l["part_maisons"] >= 99.9).sum())
    print("IRIS à plus de 50 % de résidences secondaires :", (l["part_rsec"] > 50).sum())

    # Catégories rares
    print(logement["TYP_IRIS"].value_counts())

    # ── 3. Examiner les cas extrêmes (avec leur nom) ─────────────────────────────
    colonnes = ["IRIS", "nom", "TYP_IRIS", "P22_LOG"]
    print("Plus grands IRIS :")
    display(logement.sort_values("P22_LOG").tail(5)[colonnes + ["P22_RSECOCC"]])
    print("Taux de vacance les plus élevés :")
    display(l.sort_values("taux_vacance").tail(5)[colonnes + ["taux_vacance"]].round(1))
    print("Parts de résidences secondaires les plus élevées :")
    display(l.sort_values("part_rsec").tail(5)[colonnes + ["part_rsec"]].round(1))

    # ── 4. Vacance très élevée : petits IRIS isolés ou vrais territoires ? ───────
    tres_vacant = l[l["taux_vacance"] > 25]
    print("IRIS > 25 % de vacance :", len(tres_vacant))
    print("dont communes non découpées (Z) :", round((tres_vacant["TYP_IRIS"] == "Z").mean() * 100, 1), "%")
    print("Part de Z parmi tous les IRIS :", round((l["TYP_IRIS"] == "Z").mean() * 100, 1), "%")   # point de comparaison
    print("Logements médians :", round(tres_vacant["P22_LOG"].median()), "contre", round(l["P22_LOG"].median()), "pour tous")
    print("Départements les plus représentés :")
    print(tres_vacant["COM"].str[:2].value_counts().head(5))

    # =========================================================================
    # 6. DÉLINQUANCE ENREGISTRÉE
    # =========================================================================

    # --- Chargement ---
    afficher_section('6. Délinquance enregistrée | Chargement')
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    delinq = pd.read_parquet(DATA_RAW / "dataset_delinquance_communale.parquet")

    pd.set_option("display.max_columns", None)
    print(delinq.shape)

    # -------------------------------------------------------------------------
    # A. Identité
    # -------------------------------------------------------------------------

    # --- 2. Dimensions, unité d'observations ---
    afficher_section("6. Délinquance enregistrée | 2. Dimensions, unité d'observations")
    print(delinq.shape)
    delinq.info(memory_usage="deep")
    display(delinq.head(15))

    # Une ligne = une commune × une année × un indicateur ?
    # Si la grille est complète, le nombre de lignes est le produit des trois
    n_com = delinq["CODGEO_2026"].nunique()
    n_an = delinq["annee"].nunique()
    n_ind = delinq["indicateur"].nunique()
    print(f"{n_com} communes × {n_an} années × {n_ind} indicateurs = {n_com * n_an * n_ind:,} | lignes : {len(delinq):,}")

    # --- 3. Temporalité ---
    afficher_section('6. Délinquance enregistrée | 3. Temporalité')
    # Bornes temporelles observées
    print("Première année :", delinq["annee"].min())
    print("Dernière année :", delinq["annee"].max())

    # Répartition des lignes par année
    print("\nLignes par année :")
    print(delinq["annee"].value_counts().sort_index())

    # -------------------------------------------------------------------------
    # B. Variables
    # -------------------------------------------------------------------------

    # --- 4. Dictionnaire des variables ---
    afficher_section('6. Délinquance enregistrée | 4. Dictionnaire des variables')
    # 1. Identifiant et variables temporelles
    df_ident = delinq[["CODGEO_2026", "annee", "insee_pop_millesime", "insee_log_millesime"]]

    # 2. Variables qualitatives
    df_quali = delinq[["indicateur", "unite_de_compte", "est_diffuse"]]

    # 3. Variables quantitatives discrètes (dénombrements)
    df_num_discrete = delinq[["nombre", "insee_pop", "insee_log"]]

    # 4. Variables quantitatives continues
    df_num_quanti = delinq[["taux_pour_mille", "complement_info_nombre", "complement_info_taux"]]

    print("Identifiant et temporelles :", df_ident.shape)
    print("Qualitatives               :", df_quali.shape)
    print("Quantitatives discrètes    :", df_num_discrete.shape)
    print("Quantitatives continues    :", df_num_quanti.shape)
    # Modalités et effectifs de chaque variable qualitative
    for col in df_quali.columns:
        print(df_quali[col].value_counts(), "\n")

    # Graphiques des deux variables qualitatives informatives
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["unite_de_compte", "est_diffuse"]):
        effectifs = df_quali[col].value_counts()
        (effectifs[effectifs > 0] / 1000).plot.barh(ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Nombre de lignes (milliers)")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
    # Paris, Lyon, Marseille : on retire les arrondissements pour ne pas compter ces villes deux fois
    sans_arr = delinq[~delinq["CODGEO_2026"].str.match(r"^(751|6938|132)")]

    # 2025 est l'année de test : on la laisse de côté
    sans_arr = sans_arr[sans_arr["annee"] <= 2024]

    # Total des faits par indicateur, en milliers, du plus petit au plus grand
    total = sans_arr.groupby("indicateur", observed=True)["nombre"].sum().sort_values() / 1000

    total.plot.barh(figsize=(9, 6))
    plt.title("Nombre de faits enregistrés par indicateur (2016-2024)")
    plt.xlabel("Nombre (milliers)")
    plt.ylabel("")
    plt.show()

    # -------------------------------------------------------------------------
    # C. Qualité
    # -------------------------------------------------------------------------

    # --- 5. Doublons exacts ---
    afficher_section('6. Délinquance enregistrée | 5. Doublons exacts')
    # 1. Doublons exacts et doublons sur la clé commune × année × indicateur
    print("Doublons exacts :", delinq.duplicated().sum())
    print("Doublons sur la clé :", delinq.duplicated(subset=["CODGEO_2026", "annee", "indicateur"]).sum())

    # 2. Paris, Lyon, Marseille : ville entière contre somme des arrondissements
    plm = {
        "Paris":     ("75056", "751"),   # code de la ville, début des codes d'arrondissement
        "Lyon":      ("69123", "6938"),
        "Marseille": ("13055", "132"),
    }
    d = delinq[(delinq["annee"] == 2024) & (delinq["indicateur"] == "Cambriolages de logement")]
    for ville, (code, prefixe) in plm.items():
        entiere = d[d["CODGEO_2026"] == code]
        arrond = d[d["CODGEO_2026"].str.startswith(prefixe)]
        print(f"{ville} 2024 | ville : {entiere['nombre'].sum():,.0f} cambriolages, {entiere['insee_pop'].sum():,.0f} hab. | "
              f"{len(arrond)} arrondissements : {arrond['nombre'].sum():,.0f} cambriolages, {arrond['insee_pop'].sum():,.0f} hab.")

    # --- 6. Valeurs manquantes ---
    afficher_section('6. Délinquance enregistrée | 6. Valeurs manquantes')
    manquants = (delinq.isna().mean() * 100).round(2)
    display(manquants[manquants > 0].sort_values(ascending=False))

    # Lignes avec un nombre mais sans taux
    ecart = delinq[delinq["nombre"].notna() & delinq["taux_pour_mille"].isna()]
    print("Lignes concernées :", len(ecart))
    display(ecart[["CODGEO_2026", "indicateur", "nombre", "insee_pop", "insee_log"]]
            .drop_duplicates("CODGEO_2026").head(10))
    # Le nombre est-il vide exactement quand la ligne est masquée ?
    display(pd.crosstab(delinq["est_diffuse"], delinq["nombre"].isna().rename("nombre vide")))

    # Le complément est-il rempli exactement quand la ligne est masquée ?
    display(pd.crosstab(delinq["est_diffuse"], delinq["complement_info_nombre"].isna().rename("complément vide")))
    # Pour chaque indicateur, part des lignes diffusées et masquées (en %)
    part = pd.crosstab(delinq["indicateur"], delinq["est_diffuse"], normalize="index") * 100

    part["ndiff"].sort_values().plot.barh(figsize=(9, 6))
    plt.title("Part des lignes masquées par le secret statistique, par indicateur")
    plt.xlabel("Lignes masquées (%)")
    plt.ylabel("")
    plt.show()

    # --- 7 Distributions et valeurs extrêmes ---
    afficher_section('6. Délinquance enregistrée | 7 Distributions et valeurs extrêmes')
    # Centiles du taux par indicateur (2024, communes diffusées, sans les arrondissements)
    d24 = delinq[(delinq["annee"] == 2024)
                 & ~delinq["CODGEO_2026"].str.match(r"^(751|6938|132)")
                 & (delinq["est_diffuse"] == "diff")]
    centiles = (d24.groupby("indicateur", observed=True)["taux_pour_mille"]
                   .describe(percentiles=[0.50, 0.90, 0.99]).round(2))
    centiles["max / p99"] = (centiles["max"] / centiles["99%"]).round(1)

    display(centiles[["count", "50%", "90%", "99%", "max", "max / p99"]].sort_values("max / p99"))
    # Les 5 communes aux taux les plus élevés, pour deux indicateurs
    cols = ["CODGEO_2026", "nombre", "taux_pour_mille", "insee_pop"]
    for ind in ["Trafic de stupéfiants", "Vols sans violence contre des personnes"]:
        print(ind)
        display(d24[d24["indicateur"] == ind].nlargest(5, "taux_pour_mille")[cols])

    # --- Test des bassins de vie ---
    afficher_section('6. Délinquance enregistrée | Test des bassins de vie')
    # Table de correspondance : un bassin de vie par code commune 2026
    liaison = pd.read_parquet(DATA_OUT / "liaison_commune_bv2022.parquet")
    # La table de liaison contient une ligne pour chaque code 2026 : on la lit sur CODGEO_INI
    liaison_2026 = liaison[["CODGEO_INI", "BV2022"]].rename(columns={"CODGEO_INI": "CODGEO_2026"})

    bv = d24.merge(liaison_2026, on="CODGEO_2026", how="left", validate="many_to_one")

    # Commune masquée : on remplace le nombre vide par la moyenne départementale des communes masquées
    bv["nombre_estime"] = bv["nombre"].fillna(bv["complement_info_nombre"])

    # Le taux des cambriolages se calcule pour 1 000 logements, les autres pour 1 000 habitants
    bv["denominateur"] = np.where(bv["indicateur"] == "Cambriolages de logement",
                                  bv["insee_log"], bv["insee_pop"])

    # Somme des faits et des habitants (ou logements) par bassin, puis taux
    taux_bv = bv.groupby(["BV2022", "indicateur"], observed=True)[["nombre_estime", "denominateur"]].sum()
    taux_bv["taux_pour_mille"] = taux_bv["nombre_estime"] / taux_bv["denominateur"] * 1000

    # Même tableau de centiles qu'au niveau communal
    centiles_bv = (taux_bv.groupby("indicateur", observed=True)["taux_pour_mille"]
                          .describe(percentiles=[0.50, 0.90, 0.99]).round(2))
    centiles_bv["max / p99"] = (centiles_bv["max"] / centiles_bv["99%"]).round(1)
    display(centiles_bv[["count", "50%", "90%", "99%", "max", "max / p99"]].sort_values("max / p99"))

    # =========================================================================
    # 7. LOVAC
    # =========================================================================
    afficher_section('7. LOVAC')
    import numpy as np
    import pandas as pd

    chemin_lovac = DATA_RAW / "lovac-open-data-2020-a-2026.xlsx"

    # ── Chargement : feuille communale, tout en texte (les "s" empêchent la lecture en nombres) ──
    lovac_brut = pd.read_excel(chemin_lovac, sheet_name="COM open data", dtype=str)
    print("Dimensions :", lovac_brut.shape)
    print(list(lovac_brut.columns))
    display(lovac_brut.head(3))

    # Les autres feuilles : métadonnées et totaux nationaux, régionaux, départementaux
    print(pd.ExcelFile(chemin_lovac).sheet_names)

    # ── Structure : une ligne = une commune ? ────────────────────────────────────
    print("Codes commune uniques :", lovac_brut["CODGEO_26"].nunique(), "/", len(lovac_brut))
    print("Longueur des codes :", lovac_brut["CODGEO_26"].str.len().unique())
    print("Doublons exacts :", lovac_brut.duplicated().sum())

    # Colonnes de chiffres : celles qui commencent par pp_ (parc privé) ou ff_ (fichiers fonciers)
    colonnes_chiffres = []
    for col in lovac_brut.columns:
        if col.startswith("pp_") or col.startswith("ff_"):
            colonnes_chiffres.append(col)
    print(len(colonnes_chiffres), "colonnes de chiffres")

    # ── Secret statistique : part de "s" par colonne ─────────────────────────────
    secret = pd.DataFrame({"nb_s": (lovac_brut[colonnes_chiffres] == "s").sum(),
                           "% s": ((lovac_brut[colonnes_chiffres] == "s").mean() * 100).round(1)})
    display(secret)

    # Y a-t-il d'autres textes que "s" dans les colonnes de chiffres ?
    for col in colonnes_chiffres:
        valeurs = lovac_brut[col].dropna()
        autres = valeurs[(valeurs != "s") & (~valeurs.str.isdigit())]
        if len(autres) > 0:
            print(col, ": valeurs inattendues", autres.unique()[:5])

    # Valeurs manquantes "vraies" (cases vides)
    manquants = lovac_brut.isna().sum()
    print(manquants[manquants > 0])

    # ── Territoire ───────────────────────────────────────────────────────────────
    print("Départements :", lovac_brut["DEP"].nunique())
    print("Mayotte (976) présente ?", (lovac_brut["DEP"] == "976").sum() > 0)
    print("Communes d'Alsace-Moselle :", lovac_brut["DEP"].isin(["57", "67", "68"]).sum())
    plm = (lovac_brut["CODGEO_26"].str[:3].isin(["751", "132"])) | (lovac_brut["CODGEO_26"].str[:4] == "6938")
    print("Arrondissements Paris / Lyon / Marseille :", plm.sum(),
          "| codes 75056/69123/13055 :", lovac_brut["CODGEO_26"].isin(["75056", "69123", "13055"]).sum())

    # ── Totaux nationaux par millésime (feuille FR) ──────────────────────────────
    fr = pd.read_excel(chemin_lovac, sheet_name="FR")
    display(fr)

    # ── Cohérence et distribution (exemple du millésime 2023) ────────────────────
    vac = pd.to_numeric(lovac_brut["pp_vacant_23"].replace("s", np.nan))
    vac2 = pd.to_numeric(lovac_brut["pp_vacant_plus_2ans_23"].replace("s", np.nan))
    parc = pd.to_numeric(lovac_brut["ff_pp_total_23"].replace("s", np.nan))
    print("Vacants 2 ans > vacants :", (vac2 > vac).sum(), "| vacants > parc :", (vac > parc).sum())

    taux = vac / parc * 100
    print(taux.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(1))
    # Les taux les plus élevés : quelles communes ?
    extremes = lovac_brut.assign(taux=taux).sort_values("taux").dropna(subset=["taux"]).tail(5)
    display(extremes[["CODGEO_26", "LIBGEO_26", "DEP", "pp_vacant_23", "ff_pp_total_23", "taux"]])

    # =========================================================================
    # 9. TAUX DES CRÉDITS IMMOBILIERS
    # =========================================================================

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('9. Taux des crédits immobiliers | 1. Situer la source et ouvrir les données')
    # 6 lignes de métadonnées en tête de fichier, pas de ligne d'en-tête
    taux = pd.read_csv(DATA_RAW / "Taux d'intérêt annuel des crédits immobiliers.csv",
                       sep=";", skiprows=6, header=None, names=["date", "taux"])

    # Taux en nombre (virgule décimale) et date en vraie date, pour le graphique
    taux["taux"] = pd.to_numeric(taux["taux"].str.replace(",", ".", regex=False))
    taux["date"] = pd.to_datetime(taux["date"])
    display(taux.head())

    # La colonne date est du texte ("2020-01") : on la convertit en vraie date
    date = pd.to_datetime(taux["date"])

    # Date de la plus ancienne et de la plus récente annonce
    print("Du", date.min(), "au", date.max())

    # Nombre d'annonces par année, triées dans l'ordre chronologique
    print(date.dt.year.value_counts().sort_index())

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('9. Taux des crédits immobiliers | 2. Comprendre la structure et une ligne')
    print(taux.shape) 
    taux.info() 

    # On regarde s'il y a des doublons exacts dans le dataset sans prendre en compte l'Index
    display(taux.duplicated().sum()) 

    import datetime
    annees_graphe=[datetime.date(1998,12,1), datetime.date(2000,1,1), datetime.date(2010,1,1), datetime.date(2020,1,1), datetime.date(2026,8,31)]
    libelles=['1998','2000', '2010', '2020', '2026']

    plt.plot(taux["date"], taux["taux"])
    plt.xlabel("Date")
    plt.xticks(annees_graphe, libelles)
    plt.ylabel("Taux d'intérêt")
    plt.locator_params(axis='y', nbins=10)
    plt.title("Évolution du taux d'intérêt annuel des crédits immobiliers")
    plt.show()

    # =========================================================================
    # 10. ANNONCES DE VENTE (HAUT-RHIN)
    # =========================================================================

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 1. Situer la source et ouvrir les données')
    import pandas as pd
    import numpy as np

    ventes = pd.read_csv(DATA_RAW / "ech_annonces_ventes_68.csv", sep=";", encoding="utf-8-sig", index_col='idannonce')
    ventes.head()
    # Nombre d'annonces par département 
    print(ventes["DEP"].value_counts())

    # Nombre de communes différentes du Haut-Rhin présentes dans le fichier
    # (on ne garde que les lignes du 68, puis on compte les codes communes distincts)
    print("Communes du 68 :", ventes[ventes["DEP"] == 68]["INSEE_COM"].nunique(), "sur 366")


    # La colonne date est du texte ("2020-01") : on la convertit en vraie date
    dates = pd.to_datetime(ventes["date"])

    # Date de la plus ancienne et de la plus récente annonce
    print("Du", dates.min(), "au", dates.max())

    # Nombre d'annonces par année, triées dans l'ordre chronologique
    print(dates.dt.year.value_counts().sort_index())

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 2. Comprendre la structure et une ligne')
    print(ventes.shape)  # idannonce est dans les index donc ne compte pas dans le nomhre de colonnes
    display(ventes.head())
    ventes.info() 


    # La clé semble être idannonce et on vérifie si elle est bien unique et jamais vide
    print("idannonce unique :", ventes.index.is_unique)
    print("idannonce vide   :", ventes.index.isna().sum())

    # On regarde s'il y a des doublons exacts dans le dataset sans prendre en compte l'Index
    ventes.duplicated().sum() 


    # Règle : même type de bien, même surface, même nombre de pièces, mêmes coordonnées GPS.

    cle_bien = ["typedebien", "surface", "nb_pieces",
                "mapCoordonneesLatitude", "mapCoordonneesLongitude"]

    # Annonces qui répètent un bien déjà présent
    nb_repet = ventes.duplicated(subset=cle_bien).sum()
    print("Répétitions :", nb_repet)
    print("Proportion (%) :", round(nb_repet / len(ventes) * 100, 1))

    # Règle plus stricte : mêmes caractéristiques et meme prix
    nb_strict = ventes.duplicated(subset=cle_bien + ["prix_bien"]).sum()
    print("Même bien et même prix :", nb_strict)
    print("Proportion (%) :", round(nb_strict / len(ventes) * 100, 1))

    # --- 3. Comprendre les variables ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 3. Comprendre les variables')
    from pandas.api.types import is_numeric_dtype
    # Liste qui va recevoir une ligne de résumé par colonne
    resume = []

    # On parcourt toutes les colonnes du DataFrame, une par une
    for col in ventes.columns:

        # Si la colonne contient des nombres : on note le min et le max
        if is_numeric_dtype(ventes[col]):
            valeurs = "min = " + str(ventes[col].min()) + ", max = " + str(ventes[col].max())

        # Sinon, si elle a 6 modalités ou moins : on les liste (sans les valeurs manquantes)
        elif ventes[col].nunique() <= 6:
            valeurs = str(list(ventes[col].dropna().unique()))

        # Sinon : on note seulement le nombre de modalités
        else:
            valeurs = str(ventes[col].nunique()) + " modalités"

        # On ajoute la ligne de résumé à la liste
        resume.append({"variable": col, "type": str(ventes[col].dtype), "valeurs": valeurs})

    # On transforme la liste en DataFrame, avec le nom de la variable comme index
    resume = pd.DataFrame(resume).set_index("variable")
    display(resume)

    # --- 4. Vérifier le périmètre réel ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 4. Vérifier le périmètre réel')
    # Nombre d'annonces pour chaque type de bien (a, m, an, mn, l)
    print(ventes["typedebien"].value_counts())

    # Nombre d'annonces neuves (o), anciennes (n) et non renseignées (NaN)
    # dropna=False : on compte aussi les valeurs manquantes
    print(ventes["logement_neuf"].value_counts(dropna=False))

    # Temps : quelle période, et combien d'annonces par an et par mois ? 

    # La colonne date est du texte : on la convertit en vraie date
    dates = pd.to_datetime(ventes["date"])

    # Date la plus ancienne et date la plus récente du jeu
    print("Première date :", dates.min())
    print("Dernière date :", dates.max())

    # .dt.year extrait l'année de chaque date ; on compte les annonces par année
    # sort_index() trie les années dans l'ordre chronologique
    print(dates.dt.year.value_counts().sort_index())

    # .dt.to_period("M") garde l'année et le mois ; on compte les annonces par mois
    # Utile pour repérer les mois vides (trous) ou anormalement chargés (pics)
    print(dates.dt.to_period("M").value_counts().sort_index())

    # Tableau croisé année × type de bien
    # normalize="index" : chaque ligne (année) est exprimée en proportions qui font 1
    # On vérifie si la part maisons / appartements change d'une année à l'autre
    print(pd.crosstab(dates.dt.year, ventes["typedebien_lite"], normalize="index").round(2))


    # Liste des départements présents (ici un seul : 68)
    print("Départements :", ventes["DEP"].unique())

    # Nombre de communes différentes dans le jeu
    print("Nombre de communes :", ventes["INSEE_COM"].nunique())

    # Les 5 communes qui ont le plus d'annonces, en proportion du total
    # normalize=True : donne des proportions au lieu de comptes
    print(ventes["INSEE_COM"].value_counts(normalize=True).head(5).round(3))

    # Nombre de communes qui ont 5 annonces ou moins
    # value_counts() compte les annonces par commune, "<= 5" donne True/False,
    # et sum() compte les True
    print("Communes avec 5 annonces ou moins :", (ventes["INSEE_COM"].value_counts() <= 5).sum())


    # Type d'annonceur : professionnel (pr) ou particulier (pa)
    print(ventes["type_annonceur"].value_counts())

    # Catégorie d'annonceur (codes non documentés : a, cm, m, ca, network, b)
    print(ventes["categorie_annonceur"].value_counts())

    # --- 5. Repérer et investiguer les problèmes de qualité ---

    # --- 5.1 Valeurs manquantes ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 5.1 Valeurs manquantes')
    # Nombre de NaN par colonne, puis proportion en %
    nb_na = ventes.isna().sum()
    pct_na = (ventes.isna().mean() * 100).round(1)

    # On regroupe les deux dans un tableau, on ne garde que les colonnes avec des NaN,
    # et on trie de la plus touchée à la moins touchée
    manquants = pd.DataFrame({"nb_NaN": nb_na, "pct_NaN": pct_na})
    manquants = manquants[manquants["nb_NaN"] > 0].sort_values("pct_NaN", ascending=False)
    display(manquants)



    # Dans les colonnes texte : combien de cases contiennent "0" au lieu d'une vraie valeur ?
    colonnes_texte = ventes.select_dtypes(exclude="number").columns
    nb_zeros_texte = (ventes[colonnes_texte] == "0").sum()
    print(nb_zeros_texte[nb_zeros_texte > 0])

    # Dans les colonnes numériques : valeurs impossibles ou de remplissage
    print("etage = 999 :", (ventes["etage"] == 999).sum())
    print("prix_terrain = 1 € :", (ventes["prix_terrain"] == 1).sum())
    print("mensualiteFinance = 0 :", (ventes["mensualiteFinance"] == 0).sum())
    print("dpeC = 0 :", (ventes["dpeC"] == 0).sum())

    # Modalités qui signifient « pas d'information » sans être des NaN
    print("dpeL = NS ou VI :", ventes["dpeL"].isin(["NS", "VI"]).sum())



    # Pour chaque type de bien (a = appartement, m = maison), % de NaN dans quelques colonnes
    # Si une colonne est presque toujours vide pour les maisons, c'est « non applicable »
    colonnes = ["surface_terrain", "ascenseur", "nb_logements_copro", "charges_copro",
                "nb_etages", "annee_construction", "dpeC", "cave"]
    print((ventes.groupby("typedebien_lite")[colonnes].apply(lambda g: g.isna().mean()) * 100).round(1).T)




    # On regarde si les valeurs manquantes sont concentrées dans un groupe : 
    annee = pd.to_datetime(ventes["date"]).dt.year

    # Par année
    print((ventes["annee_construction"].isna().groupby(annee).mean() * 100).round(1))
    print(((ventes["dpeL"] == "0").groupby(annee).mean() * 100).round(1))

    # Par catégorie d'annonceur
    print((ventes.groupby("categorie_annonceur")[["annee_construction", "dpeC", "cave"]]
           .apply(lambda g: g.isna().mean()) * 100).round(1))

    # Neuf / ancien
    print((ventes["dpeC"].isna().groupby(ventes["logement_neuf"]).mean() * 100).round(1))

    # Loyer médian local : selon que la commune est rurale (UU2010 = 68000) ou urbaine
    rurale = ventes["UU2010"] == 68000
    print((ventes["loyer_m2_median_n6"].isna().groupby(rurale).mean() * 100).round(1))

    # Lien avec le DPE : dpeC manque-t-il quand l'étiquette est inconnue ?
    print(pd.crosstab(ventes["dpeL"], ventes["dpeC"].isna()))


    # Les biens avec une valeur manquantes ont-ils un prix différent ?
    # Si oui, supprimer ces lignes ou imputer naïvement biaiserait l'analyse du prix

    print(ventes.groupby(ventes["annee_construction"].isna())["prix_m2_vente"].median())
    print(ventes.groupby(ventes["dpeC"].isna())["prix_m2_vente"].median())

    # --- 5.2 Formats, modalités et incohérences ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 5.2 Formats, modalités et incohérences')
    import numpy as np

    # Formats et écritures hétérogènes

    # Exposition : même orientation écrite de plusieurs façons ?
    ventes["exposition"].unique()
    print("Modalités brutes :", ventes["exposition"].nunique())
    print("Après mise en minuscules :", ventes["exposition"].str.lower().nunique())
    print(ventes["exposition"].unique())        # on regarde les écritures à la main

    # Chauffage : plusieurs valeurs dans une même case (séparées par une virgule) ?
    print("chauffage_energie multi-valeurs :", ventes["chauffage_energie"].str.contains(",").sum())
    print("chauffage_systeme multi-valeurs :", ventes["chauffage_systeme"].str.contains(",").sum())
    print("chauffage_mode multi-valeurs :", ventes["chauffage_mode"].str.contains(",").sum())
    print(ventes["chauffage_energie"].unique())  # repérer "Fioul, Bois" et "Bois, Fioul"

    # Date : toutes au format AAAA-MM ?
    print(ventes["date"].unique())
    print("Dates au bon format :", ventes["date"].str.match(r"^\d{4}-\d{2}$").mean())

    # Variables oui / non : codées de manières différentes selon les colonnes
    print(ventes["annonce_exclusive"].unique())  # Oui / Non / 0
    print(ventes["logement_neuf"].unique())      # o / n
    print(ventes["cave"].unique())               # True / False


    # Règles de cohérence et nombre de violations 
    appart = ventes["typedebien_lite"] == "a"
    maison = ventes["typedebien_lite"] == "m"

    # Règle : prix_m2_vente = prix_bien / surface
    print("Prix au m² incohérent :",
          ((ventes["prix_m2_vente"] - ventes["prix_bien"] / ventes["surface"]).abs() > 0.01).sum())

    # Règle : pour une offre terrain + maison, prix_maison + prix_terrain = prix_bien
    offre = ventes["prix_maison"].notna()
    ecart = (ventes["prix_maison"] + ventes["prix_terrain"] - ventes["prix_bien"]).abs()
    print("Somme maison + terrain différente du prix :", (offre & (ecart > 1)).sum(), "sur", offre.sum())

    # Règle : typedebien et typedebien_lite doivent concorder (première lettre)
    print("typedebien ≠ typedebien_lite :", (ventes["typedebien"].str[0] != ventes["typedebien_lite"]).sum())

    # Règle : les codes "neuf" (an, mn) et la colonne logement_neuf doivent concorder
    print(pd.crosstab(ventes["typedebien"], ventes["logement_neuf"]))

    # Règle : une maison est au rez-de-chaussée (étage 0) et n'a pas d'ascenseur
    print("Maison avec étage ≠ 0 :", (maison & (ventes["etage"] != 0)).sum())
    print("Maison avec ascenseur :", (maison & (ventes["ascenseur"] == True)).sum())

    # Règle : un appartement ne peut pas être plus haut que le nombre d'étages de l'immeuble
    print("Appartement : étage > nb_etages :",
          (appart & (ventes["etage"] > ventes["nb_etages"]) & (ventes["etage"] != 999)).sum())

    # Règle : un appartement n'a normalement pas de terrain privatif
    print("Appartement avec surface de terrain :", (appart & ventes["surface_terrain"].notna()).sum())

    # Règle : pas de surface de balcon si le nombre de balcons vaut 0
    print("0 balcon mais surface de balcon :",
          ((ventes["balcon"] == 0) & (ventes["surface_balcon"] > 0)).sum())

    # Règle : un logement ancien n'est pas construit après la date de l'annonce
    annee_annonce = pd.to_datetime(ventes["date"]).dt.year
    apres = ventes["annee_construction"] > annee_annonce
    print("Ancien construit après l'annonce :", (apres & (ventes["logement_neuf"] == "n")).sum())
    print("Neuf construit après l'annonce :", (apres & (ventes["logement_neuf"] == "o")).sum())

    # Règle : une durée de mise en ligne ne peut pas être négative
    print("duree_int négative :", (ventes["duree_int"] < 0).sum())

    # Règle : les deux colonnes TYP_IRIS devraient être identiques
    print("TYP_IRIS_x ≠ TYP_IRIS_y :", (ventes["TYP_IRIS_x"] != ventes["TYP_IRIS_y"]).sum())

    # Règle : l'étiquette DPE doit correspondre à la consommation (barème depuis juillet 2021)
    # pd.cut range chaque consommation dans sa classe théorique
    classe_calculee = pd.cut(ventes["dpeC"],
                             bins=[-1, 69, 109, 179, 249, 329, 419, np.inf],
                             labels=["A", "B", "C", "D", "E", "F", "G"]).astype(str)
    lettre_connue = ventes["dpeL"].isin(["A", "B", "C", "D", "E", "F", "G"]) & ventes["dpeC"].notna()
    # Lettre MEILLEURE que la consommation : impossible (A < B < … par ordre alphabétique)
    print("Étiquette meilleure que la consommation :", (lettre_connue & (ventes["dpeL"] < classe_calculee)).sum())
    # Lettre MOINS BONNE : possible, car la lettre dépend aussi des émissions de GES
    print("Étiquette moins bonne que la consommation :", (lettre_connue & (ventes["dpeL"] > classe_calculee)).sum())

    # --- 5.3 Distributions et valeurs extrêmes ---
    afficher_section('10. Annonces de vente (Haut-Rhin) | 5.3 Distributions et valeurs extrêmes')
    # Décrire les distributions des variables importantes
    variables = ["prix_bien", "prix_m2_vente", "surface", "nb_pieces", "surface_terrain"]

    # count, moyenne, écart-type, min, quartiles, max
    # Si la moyenne est nettement au-dessus de la médiane (50%), la distribution
    # a une longue queue vers les grandes valeurs
    display(ventes[variables].describe().round(0))

    # Répartition des catégories importantes, en %
    print((ventes["typedebien"].value_counts(normalize=True) * 100).round(1))
    print((ventes["typedetransaction"].value_counts(normalize=True) * 100).round(1))


    # Repérer les valeurs éloignées : règle de l'écart interquartile (IQR)
    # Une valeur est « éloignée » si elle sort de [Q1 - 1,5 × IQR ; Q3 + 1,5 × IQR]
    for col in variables:
        q1 = ventes[col].quantile(0.25)
        q3 = ventes[col].quantile(0.75)
        iqr = q3 - q1
        nb = ((ventes[col] < q1 - 1.5 * iqr) | (ventes[col] > q3 + 1.5 * iqr)).sum()
        print(col, ": valeurs éloignées =", nb)


    # Examiner les cas extrêmes et leur contexte 
    colonnes = ["typedebien", "typedetransaction", "surface", "nb_pieces", "prix_bien", "prix_m2_vente"]

    # On affiche les lignes qui vérifient une condition
    display(ventes[ventes["prix_bien"] < 20000][colonnes])         # prix très bas
    display(ventes[ventes["prix_bien"] > 2000000][colonnes])       # prix très hauts
    display(ventes[ventes["prix_m2_vente"] > 8000][colonnes])      # prix au m² très hauts
    display(ventes[ventes["surface"] < 9][colonnes])               # surfaces minuscules


    # Compter les cas selon des critères métier fixés à l'avance 
    print("Surface < 9 m² (seuil légal d'un logement) :", (ventes["surface"] < 9).sum())
    print("Prix < 20 000 € :", (ventes["prix_bien"] < 20000).sum())
    print("Prix au m² < 500 € :", (ventes["prix_m2_vente"] < 500).sum())
    print("Prix au m² > 8 000 € :", (ventes["prix_m2_vente"] > 8000).sum())
    print("Étage = 999 :", (ventes["etage"] == 999).sum())
    print("Année de construction < 1400 :", (ventes["annee_construction"] < 1400).sum())

    # Les extrêmes viennent-ils d'une autre population ? On compare les médianes par groupe
    print(ventes.groupby("typedetransaction")[["prix_bien", "surface", "prix_m2_vente"]].median())

    # =========================================================================
    # 11. ANNONCES DE LOCATION (HAUT-RHIN)
    # =========================================================================

    # --- 1. Situer la source et ouvrir les données ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 1. Situer la source et ouvrir les données')
    import pandas as pd
    import numpy as np

    location = pd.read_csv(DATA_RAW / "ech_annonces_locations_68.csv", sep=";", encoding="utf-8-sig",index_col="idannonce")
    location.head()


    # Nombre d'annonces par département 
    print(location["DEP"].value_counts())

    # Nombre de communes différentes du Haut-Rhin présentes dans le fichier
    # (on ne garde que les lignes du 68, puis on compte les codes communes distincts)
    print("Communes du 68 :", location[location["DEP"] == 68]["INSEE_COM"].nunique(), "sur 366")


    # La colonne date est du texte ("2020-01") : on la convertit en vraie date
    dates = pd.to_datetime(location["date"])

    # Date de la plus ancienne et de la plus récente annonce
    print("Du", dates.min(), "au", dates.max())

    # Nombre d'annonces par année, triées dans l'ordre chronologique
    print(dates.dt.year.value_counts().sort_index())

    # --- 2. Comprendre la structure et une ligne ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 2. Comprendre la structure et une ligne')
    # ── 1. Dimensions, colonnes et quelques observations ───────────────────────
    print(location.shape)            # (lignes, colonnes) ; idannonce est l'index
    display(location.head())
    location.info()

    # ── 2. Une ligne en entier, transposée pour tout voir ──────────────────────
    display(location.sample(1, random_state=0).T)

    # ── 3. Clé possible : idannonce est-il unique et toujours rempli ? ─────────
    print("idannonce unique :", location.index.is_unique)
    print("idannonce vide :", location.index.isna().sum())

    # ── 4. Doublons exacts : lignes identiques sur toutes les colonnes ─────────
    print("Doublons exacts :", location.duplicated().sum())

    # Répétitions métier : même bien = même type, surface, pièces et coordonnées GPS
    cle_bien = ["typedebien", "surface", "nb_pieces",
                "mapCoordonneesLatitude", "mapCoordonneesLongitude"]
    nb_repet = location.duplicated(subset=cle_bien).sum()
    print("Répétitions :", nb_repet)
    print("Proportion (%) :", round(nb_repet / len(location) * 100, 1))

    # Même bien ET même loyer
    nb_strict = location.duplicated(subset=cle_bien + ["prix_bien"]).sum()
    print("Même bien et même loyer :", nb_strict)
    print("Proportion (%) :", round(nb_strict / len(location) * 100, 1))

    # --- 3. Comprendre les variables ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 3. Comprendre les variables')
    from pandas.api.types import is_numeric_dtype

    # Liste qui va recevoir une ligne de résumé par colonne
    resume = []

    # On parcourt toutes les colonnes du DataFrame, une par une
    for col in location.columns:

        # Si la colonne est entièrement vide : on le signale
        if location[col].isna().all():
            valeurs = "100 % vide"

        # Si la colonne contient des nombres : on note le min et le max
        elif is_numeric_dtype(location[col]):
            valeurs = "min = " + str(location[col].min()) + ", max = " + str(location[col].max())

        # Sinon, si elle a 6 modalités ou moins : on les liste (sans les valeurs manquantes)
        elif location[col].nunique() <= 6:
            valeurs = str(list(location[col].dropna().unique()))

        # Sinon : on note seulement le nombre de modalités
        else:
            valeurs = str(location[col].nunique()) + " modalités"

        # Pourcentage de valeurs manquantes, arrondi à 1 décimale
        pct_na = round(location[col].isna().mean() * 100, 1)

        # On ajoute la ligne de résumé à la liste
        resume.append({"variable": col,
                       "type": str(location[col].dtype),
                       "valeurs": valeurs,
                       "% NaN": pct_na})

    # On transforme la liste en DataFrame, avec le nom de la variable comme index
    resume = pd.DataFrame(resume).set_index("variable")
    display(resume)

    # --- 4. Vérifier le périmètre réel ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 4. Vérifier le périmètre réel')
    # Quels types de biens et de transactions ? 
    print(location["typedebien"].value_counts())
    print(location["typedetransaction"].value_counts())
    print(location["logement_neuf"].value_counts(dropna=False))

    # Quelle période, combien d'annonces par an et par mois ? 
    dates = pd.to_datetime(location["date"])
    print("Première date :", dates.min())
    print("Dernière date :", dates.max())
    print(dates.dt.year.value_counts().sort_index())
    print(dates.dt.to_period("M").value_counts().sort_index())

    # Part maisons / appartements selon l'année (chaque ligne fait 1 avec le normalize)
    print(pd.crosstab(dates.dt.year, location["typedebien_lite"], normalize="index").round(2))

    # Sur quels territoires ?
    print(location["DEP"].value_counts())
    print("Nombre de communes :", location["INSEE_COM"].nunique())
    print(location["INSEE_COM"].value_counts(normalize=True).head(5).round(3))
    print("Communes avec 5 annonces ou moins :", (location["INSEE_COM"].value_counts() <= 5).sum())
    print("Part des annonces en commune rurale :", round((location["UU2010"] == 68000).mean(), 3))

    # Les annonces hors Haut-Rhin : quel type de transaction, quelle date ?
    hors_68 = location["DEP"] != 68
    print(pd.crosstab(hors_68, location["typedetransaction"]))
    print(location[hors_68]["date"].value_counts())

    # Autre dimension : qui publie ? 
    print(location["type_annonceur"].value_counts())
    print(location["categorie_annonceur"].value_counts())

    # --- 5. Repérer et investiguer les problèmes de qualité ---

    # --- 5.1 Valeurs manquantes ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 5.1 Valeurs manquantes')
    # Mesurer les NaN par variable 
    manquants = pd.DataFrame({"nb_NaN": location.isna().sum(),
                              "pct_NaN": (location.isna().mean() * 100).round(1)})
    manquants = manquants[manquants["nb_NaN"] > 0].sort_values("pct_NaN", ascending=False)
    display(manquants)

    # Codes qui masquent des absences 
    colonnes_texte = location.select_dtypes(exclude="number").columns
    nb_zeros_texte = (location[colonnes_texte] == "0").sum()
    print(nb_zeros_texte[nb_zeros_texte > 0])                    # "0" au lieu d'une vraie valeur
    print("etage = 999 :", (location["etage"] == 999).sum())
    print("dpeC = 0 :", (location["dpeC"] == 0).sum())
    print("dpeL = NS ou VI :", location["dpeL"].isin(["NS", "VI"]).sum())
    print("mensualiteFinance, valeurs :", location["mensualiteFinance"].unique())

    # Non applicable ou inconnu ? % de NaN par type de bien 
    colonnes = ["surface_terrain", "ascenseur", "nb_etages", "annee_construction",
                "dpeC", "cave", "places_parking", "nb_toilettes"]
    print((location.groupby("typedebien_lite")[colonnes].apply(lambda g: g.isna().mean()) * 100).round(1).T)

    # Concentration par année et par type d'annonceur 
    annee = pd.to_datetime(location["date"]).dt.year
    colonnes = ["annee_construction", "dpeC", "ges_class", "cave", "ascenseur"]
    print((location.groupby(annee)[colonnes].apply(lambda g: g.isna().mean()) * 100).round(1))
    print((location.groupby("type_annonceur")[["dpeC", "annee_construction", "cave"]]
           .apply(lambda g: g.isna().mean()) * 100).round(1))

    # annonce_exclusive : en quelles années est-elle remplie ?
    print(pd.crosstab(location["annonce_exclusive"], annee))

    # dpeC manque-t-il quand l'étiquette est inconnue ?
    print(pd.crosstab(location["dpeL"], location["dpeC"].isna()))

    # Les annonces avec une absence ont-elles un loyer différent ? 
    print(location.groupby(location["annee_construction"].isna())["prix_bien"].median())
    print(location.groupby(location["dpeC"].isna())["prix_bien"].median())

    # --- 5.2 Formats, modalités et incohérences ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 5.2 Formats, modalités et incohérences')
    # Masques réutilisés plus bas : True pour les appartements, True pour les maisons
    appart = location["typedebien_lite"] == "a"
    maison = location["typedebien_lite"] == "m"

    # Formats et écritures hétérogènes 
    # Nombre de façons différentes d'écrire l'exposition
    print("Exposition, modalités brutes :", location["exposition"].nunique())

    # Même calcul après passage en minuscules : si le nombre baisse, une même orientation
    # est écrite avec des majuscules différentes ("Sud" / "sud")
    print("Après mise en minuscules :", location["exposition"].str.lower().nunique())

    # Nombre de cases contenant plusieurs valeurs séparées par une virgule ("Gaz, Bois")
    print("chauffage_energie multi-valeurs :", location["chauffage_energie"].str.contains(",").sum())
    print("chauffage_systeme multi-valeurs :", location["chauffage_systeme"].str.contains(",").sum())
    print("chauffage_mode multi-valeurs :", location["chauffage_mode"].str.contains(",").sum())

    # Proportion de dates au format "AAAA-MM" (4 chiffres, un tiret, 2 chiffres) ; 1.0 = 100 %
    print("Dates au format AAAA-MM :", location["date"].str.match(r"^\d{4}-\d{2}$").mean())

    #  Règles de cohérence 
    # Principe : on écrit la condition qui viole la règle, puis .sum() compte les lignes concernées

    # La première lettre de typedebien (.str[0]) doit être égale à typedebien_lite
    print("typedebien ≠ typedebien_lite :", (location["typedebien"].str[0] != location["typedebien_lite"]).sum())

    # Une maison doit être à l'étage 0 (& = « et » : les deux conditions à la fois)
    print("Maison avec étage ≠ 0 :", (maison & (location["etage"] != 0)).sum())

    # Une maison n'a pas d'ascenseur
    print("Maison avec ascenseur :", (maison & (location["ascenseur"] == True)).sum())

    # Un appartement ne peut pas être plus haut que son immeuble (on exclut 999 = étage inconnu)
    print("Appartement : étage > nb_etages :",
          (appart & (location["etage"] > location["nb_etages"]) & (location["etage"] != 999)).sum())

    # Un appartement n'a normalement pas de terrain (.notna() = valeur renseignée)
    print("Appartement avec surface de terrain :", (appart & location["surface_terrain"].notna()).sum())

    # Pas de surface de balcon quand le nombre de balcons vaut 0
    print("0 balcon mais surface de balcon :",
          ((location["balcon"] == 0) & (location["surface_balcon"] > 0)).sum())

    # Une durée de mise en ligne ne peut pas être négative
    print("duree_int négative :", (location["duree_int"] < 0).sum())

    # Les deux colonnes TYP_IRIS (issues d'une jointure) devraient être identiques
    print("TYP_IRIS_x ≠ TYP_IRIS_y :", (location["TYP_IRIS_x"] != location["TYP_IRIS_y"]).sum())

    # Le département doit correspondre au code commune (les 2 premiers chiffres)
    # // 1000 supprime les 3 derniers chiffres : 68224 // 1000 = 68
    print("DEP incohérent avec INSEE_COM :", (location["INSEE_COM"] // 1000 != location["DEP"]).sum())

    # Les annonces hors 68 et les transactions lt : est-ce le même groupe ?
    # Tableau croisé : dans le 68 (True) ou non (False) × type de transaction (l / lt)
    print(pd.crosstab(location["DEP"] == 68, location["typedetransaction"]))

    # --- 5.3 Distributions et valeurs extrêmes ---
    afficher_section('11. Annonces de location (Haut-Rhin) | 5.3 Distributions et valeurs extrêmes')
    # Le loyer au m² n'existe pas dans le fichier : on le calcule
    location["loyer_m2"] = location["prix_bien"] / location["surface"]

    # Distributions des variables importantes 
    variables = ["prix_bien", "loyer_m2", "surface", "nb_pieces"]
    display(location[variables].describe().round(1))

    print((location["typedebien"].value_counts(normalize=True) * 100).round(1))
    print(location.groupby("typedebien_lite")[["prix_bien", "loyer_m2"]].median())

    #Valeurs éloignées selon la règle de l'IQR 
    for col in variables:
        q1 = location[col].quantile(0.25)
        q3 = location[col].quantile(0.75)
        iqr = q3 - q1
        nb = ((location[col] < q1 - 1.5 * iqr) | (location[col] > q3 + 1.5 * iqr)).sum()
        print(col, ": valeurs éloignées =", nb)

    # Examiner les cas extrêmes 
    colonnes = ["typedebien", "typedetransaction", "surface", "nb_pieces", "prix_bien", "DEP", "date"]
    display(location[location["prix_bien"] < 150][colonnes])       # loyers très bas
    display(location[location["prix_bien"] > 2500][colonnes])      # loyers très hauts
    display(location[location["surface"] < 10][colonnes])          # très petites surfaces
    display(location[location["bain"] > 10][colonnes + ["bain"]])
    display(location[location["nb_etages"] > 40][["nb_etages", "etage", "typedebien"]])
    display(location[location["places_parking"] > 20][["places_parking", "surface", "prix_bien"]])

    # Critères métier fixés à l'avance
    print("Loyer < 150 € :", (location["prix_bien"] < 150).sum())
    print("Surface < 9 m² :", (location["surface"] < 9).sum())
    print("Loyer au m² > 30 € :", (location["loyer_m2"] > 30).sum())
    print("Loyer au m² < 4 € :", (location["loyer_m2"] < 4).sum())
    print("Année de construction < 1400 :", (location["annee_construction"] < 1400).sum())

    # Les annonces hors 68 forment-elles une autre population ?
    print(location.groupby(location["DEP"] == 68)["loyer_m2"].median())
    enregistrer_figures()

    # Libère la mémoire des audits avant l'EDA : supprime les variables créées par les audits
    for _nom in set(globals()) - NOMS_INITIAUX:
        del globals()[_nom]
    gc.collect()


# #############################################################################
# ANALYSE EXPLORATOIRE (EDA)
# #############################################################################
if "eda" in PARTIES:
    _fig["partie"] = "eda"

    # =========================================================================
    # A. LA CIBLE
    # =========================================================================
    afficher_section('A. La cible')
    # Population DVF+ filtrée (bloc 2 de la préparation)
    cols = ["idmutation", "coddep", "l_codinsee", "anneemut", "datemut",
            "libtypbien", "valeurfonc", "sbati"]
    pop = pd.read_parquet(DATA_OUT / "dvf_plus_residentiel.parquet", columns=cols)

    # On ne garde que les années hors test
    pop = pop[pop["anneemut"] <= 2024]

    print(pop.shape)
    print(round(pop.memory_usage(deep=True).sum() / 1e9, 2), "Go en mémoire")
    display(pop.head())

    # --- A1. Construction de la cible ---

    # --- 1. Calcul du prix au m² de chaque vente ---
    afficher_section('1. Calcul du prix au m² de chaque vente')
    # Variables construites
    pop["prix_m2"] = pop["valeurfonc"] / pop["sbati"]
    pop["type_bien"] = pop["libtypbien"].map({"UNE MAISON": "maison",
                                              "UN APPARTEMENT": "appartement"})

    # Distribution du prix au m² par type, avec les centiles extrêmes
    display(pop.groupby("type_bien")["prix_m2"].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    ).round(0))

    # --- 2. Valeurs manquantes, nulles ou extrêmes ---
    afficher_section('2. Valeurs manquantes, nulles ou extrêmes')
    # Cas qui empêchent de calculer un prix au m² correct
    print("valeurfonc manquante :", pop["valeurfonc"].isna().sum())
    print("valeurfonc = 0       :", (pop["valeurfonc"] == 0).sum())
    print("sbati = 0            :", (pop["sbati"] == 0).sum())

    # Indicatrice : True si la valeur foncière est à 0
    pop["vf_nulle"] = pop["valeurfonc"] == 0

    # Nombre de ventes à 0 € par année
    display(pop.groupby("anneemut")["vf_nulle"].sum())
    # Valeurs foncières manquantes par année (même idée que pour les valeurs à 0)
    print(pop["valeurfonc"].isna().groupby(pop["anneemut"]).sum())

    # Prix au m² très bas (hors valeurs à 0) et très hauts
    bas = pop[(pop["prix_m2"] > 0) & (pop["prix_m2"] < 100)]
    haut = pop[pop["prix_m2"] > 20000]

    print("\nPrix au m² entre 0 et 100 € :", len(bas))
    print(bas["type_bien"].value_counts())
    print("\nPrix au m² au-dessus de 20 000 € :", len(haut))
    print(haut["type_bien"].value_counts())
    print(haut["coddep"].value_counts().head(5))

    display(bas.sample(8, random_state=42))
    display(haut.sample(8, random_state=42))
    # Valeurs foncières manquantes par année
    print(pop["valeurfonc"].isna().groupby(pop["anneemut"]).sum())

    # Nombre de ventes touchées selon plusieurs seuils possibles
    for s in [10, 50, 100]:
        print(f"prix au m² < {s} € :", ((pop["prix_m2"] > 0) & (pop["prix_m2"] < s)).sum())
    for s in [20000, 30000, 50000]:
        print(f"prix au m² > {s} € :", (pop["prix_m2"] > s).sum())

    # Exemples aux deux bouts
    tres_bas = pop[(pop["prix_m2"] > 0) & (pop["prix_m2"] < 50)]
    tres_haut = pop[pop["prix_m2"] > 30000]
    print(tres_haut["coddep"].value_counts().head(5))

    display(tres_bas.sample(8, random_state=42))
    display(tres_haut.sample(8, random_state=42))

    # --- Règles et Table cible ---
    afficher_section('Règles et Table cible')
    def regles_exclusion(d, afficher=False):
        """Applique les règles de A1 et ajoute le prix au m² et le type de bien.
        Garde les ventes avec une valeur foncière et un prix au m² entre 50 et 50 000 €."""
        prix_m2 = d["valeurfonc"] / d["sbati"]
        r1 = d["valeurfonc"].notna() & (d["valeurfonc"] > 0)   # valeur foncière présente
        r2 = prix_m2 >= 50                                      # seuil bas
        r3 = prix_m2 <= 50000                                   # seuil haut

        if afficher:
            print("Ventes au départ           :", len(d))
            print("Valeur foncière présente   :", r1.sum())
            print("Prix au m² >= 50 €         :", (r1 & r2).sum())
            print("Prix au m² <= 50 000 €     :", (r1 & r2 & r3).sum())

        garde = r1 & r2 & r3
        sortie = d[garde].copy()
        sortie["prix_m2"] = prix_m2[garde]
        sortie["type_bien"] = sortie["libtypbien"].map({"UNE MAISON": "maison",
                                                        "UN APPARTEMENT": "appartement"})
        return sortie

    # Application à la population, avec affichage de l'entonnoir
    ventes = regles_exclusion(pop, afficher=True)
    ventes["est_maison"] = ventes["type_bien"] == "maison"

    # Table cible : une ligne par commune et par année, calculée ici en mémoire pour A2
    # (la table cible du projet est produite par le bloc 4 de la préparation)
    cible = (ventes.groupby(["l_codinsee", "anneemut"])
                   .agg(coddep=("coddep", "first"),
                        nb_ventes=("prix_m2", "size"),
                        prix_m2_median=("prix_m2", "median"),
                        part_maisons=("est_maison", "mean"))
                   .reset_index()
                   .rename(columns={"l_codinsee": "code_commune", "anneemut": "annee"}))

    print(cible.shape)
    display(cible.head())

    # --- A2. Couverture de la cible et seuil de maille ---

    # --- 1. Dispersion de la variation annuelle selon le nombre de ventes ---
    afficher_section('1. Dispersion de la variation annuelle selon le nombre de ventes')
    c = cible.sort_values(["code_commune", "annee"]).copy()

    # Valeurs de l'année précédente, pour la même commune
    c["annee_prec"] = c.groupby("code_commune")["annee"].shift(1)
    c["prix_prec"] = c.groupby("code_commune")["prix_m2_median"].shift(1)
    c["nb_prec"] = c.groupby("code_commune")["nb_ventes"].shift(1)

    # On ne garde que deux années qui se suivent (pas de trou dans la série)
    c = c[c["annee"] - c["annee_prec"] == 1].copy()
    c["variation"] = (c["prix_m2_median"] / c["prix_prec"] - 1) * 100

    # Classe selon le plus petit nombre de ventes des deux années comparées
    c["nb_min"] = c[["nb_ventes", "nb_prec"]].min(axis=1)
    c["classe"] = pd.cut(c["nb_min"], bins=[0, 4, 9, 19, 29, 49, np.inf],
                         labels=["1-4", "5-9", "10-19", "20-29", "30-49", "50+"])

    # Dispersion de la variation dans chaque classe
    resume = c.groupby("classe", observed=True)["variation"].describe(percentiles=[0.25, 0.5, 0.75])
    resume["IQR"] = resume["75%"] - resume["25%"]
    display(resume[["count", "25%", "50%", "75%", "IQR"]].round(1))

    # Graphique
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=c, x="classe", y="variation", showfliers=False)
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("Nombre de ventes (le plus petit des deux années comparées)")
    plt.ylabel("Variation annuelle du prix médian au m² (%)")
    plt.title("Variation annuelle du prix médian au m² selon le nombre de ventes")
    plt.show()

    # --- 2. Ce que garde chaque seuil ---
    afficher_section('2. Ce que garde chaque seuil')
    seuils = [1, 5, 10, 20, 30, 50]
    total_ventes = cible["nb_ventes"].sum()
    n_annees = cible["annee"].nunique()   # nombre d'années étudiées (11 : de 2014 à 2024)
    lignes = []

    for s in seuils:
        ok = cible["nb_ventes"] >= s
        # Communes qui atteignent le seuil toutes les années
        annees_ok = ok.groupby(cible["code_commune"]).sum()
        lignes.append({
            "seuil": s,
            "lignes gardées (%)": round(ok.mean() * 100, 1),
            "ventes couvertes (%)": round(cible.loc[ok, "nb_ventes"].sum() / total_ventes * 100, 1),
            "communes, au moins 1 an": cible.loc[ok, "code_commune"].nunique(),
            f"communes, {n_annees} ans sur {n_annees}": (annees_ok == n_annees).sum(),
        })

    display(pd.DataFrame(lignes))

    # --- 3. Seuil par type de bien au bassin de vie ---
    afficher_section('3. Seuil par type de bien au bassin de vie')
    # Table de liaison commune -> bassin de vie (bloc 1 de la préparation)
    liaison = pd.read_parquet(DATA_OUT / "liaison_commune_bv2022.parquet")

    # Chaque vente reçoit son bassin de vie (seulement les colonnes utiles)
    # Les arrondissements de Paris, Lyon et Marseille sont rattachés au bassin de leur ville
    v = ventes[["l_codinsee", "anneemut", "type_bien", "prix_m2"]].merge(
        liaison[["CODGEO_INI", "BV2022"]],
        left_on="l_codinsee", right_on="CODGEO_INI", how="left", validate="many_to_one")
    v["bassin"] = v["BV2022"]

    # Prix médian et nombre de ventes par bassin x année x type de bien
    b = (v.groupby(["bassin", "anneemut", "type_bien"])["prix_m2"]
           .agg(nb_ventes="size", prix_m2_median="median")
           .reset_index()
           .sort_values(["bassin", "type_bien", "anneemut"]))

    # Valeurs de l'année précédente, pour le même bassin et le même type de bien
    groupe = b.groupby(["bassin", "type_bien"])
    b["annee_prec"] = groupe["anneemut"].shift(1)
    b["prix_prec"] = groupe["prix_m2_median"].shift(1)
    b["nb_prec"] = groupe["nb_ventes"].shift(1)

    # On ne garde que deux années qui se suivent (pas de trou dans la série)
    b = b[b["anneemut"] - b["annee_prec"] == 1].copy()
    b["variation"] = (b["prix_m2_median"] / b["prix_prec"] - 1) * 100

    # Mêmes classes qu'au point 1
    b["nb_min"] = b[["nb_ventes", "nb_prec"]].min(axis=1)
    b["classe"] = pd.cut(b["nb_min"], bins=[0, 4, 9, 19, 29, 49, np.inf],
                         labels=["1-4", "5-9", "10-19", "20-29", "30-49", "50+"])

    # Un graphique par type de bien, même échelle verticale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, t in zip(axes, ["maison", "appartement"]):
        sns.boxplot(data=b[b["type_bien"] == t], x="classe", y="variation", showfliers=False, ax=ax)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Bassins de vie : {t}s")
        ax.set_xlabel("Nombre de ventes (le plus petit des deux années comparées)")
    axes[0].set_ylabel("Variation annuelle du prix médian au m² (%)")
    plt.tight_layout()
    plt.show()

    # Part des bassins et des ventes gardés selon le seuil, par type de bien
    seuils = [1, 5, 10, 20, 30, 50]
    tableaux = []

    for t in ["maison", "appartement"]:
        # Nombre de ventes par bassin x année, pour ce type de bien
        n = v[v["type_bien"] == t].groupby(["bassin", "anneemut"]).size()
        total_bassins = n.index.get_level_values("bassin").nunique()   # bassins avec au moins une vente
        lignes = []
        for s in seuils:
            ok = n >= s
            # bassins qui atteignent le seuil au moins une année
            bassins_gardes = n[ok].index.get_level_values("bassin").nunique()
            lignes.append({
                "seuil": s,
                "bassins gardés (%)": round(bassins_gardes / total_bassins * 100, 1),
                "ventes gardées (%)": round(n[ok].sum() / n.sum() * 100, 1),
            })
        tableaux.append(pd.DataFrame(lignes).set_index("seuil"))

    # Maisons et appartements côte à côte
    display(pd.concat(tableaux, axis=1, keys=["maisons", "appartements"]))

    # =========================================================================
    # B. RELATIONS À L'INTÉRIEUR DE CHAQUE SOURCE
    # =========================================================================

    # -------------------------------------------------------------------------
    # B1. Revenus des habitants (Filosofi)
    # -------------------------------------------------------------------------
    afficher_section('B1. Revenus des habitants (Filosofi)')
    # Filosofi assemblé par le bloc 5 de la préparation (une ligne = commune × année)
    filo = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet")

    # Décision de l'audit : on garde les arrondissements, on retire Paris, Lyon et Marseille en commune entière
    filo = filo[~filo["code_commune"].isin(["75056", "69123", "13055"])]
    print(filo.shape)

    # --- B1.1 Redondances entre variables ---
    afficher_section('B1.1 Redondances entre variables')
    # Communes avec tous les indicateurs (indicateurs détaillés publiés en bloc)
    complet = filo.dropna(subset=["D1"])
    cols = [c for c in filo.columns if c not in ["code_commune", "annee"]]

    corr = complet[cols].corr(method="spearman")

    plt.figure(figsize=(16, 13))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".1f", annot_kws={"size": 7})
    plt.title(f"Corrélations de Spearman entre les variables de Filosofi (n = {len(complet):,} lignes)")
    plt.tight_layout()
    plt.show()

    # --- B1.2 Niveau de vie médian selon la taille de la commune ---
    afficher_section('B1.2 Niveau de vie médian selon la taille de la commune')
    # Une seule année : 2019, dernière année avant le changement de format des fichiers
    f19 = filo[filo["annee"] == 2019].dropna(subset=["niveau_vie_median"]).copy()

    bornes = [0, 50, 100, 250, 500, 1000, 2500, 10_000, float("inf")]
    etiquettes = ["< 50", "50-99", "100-249", "250-499", "500-999",
                  "1 000-2 499", "2 500-9 999", "≥ 10 000"]
    f19["taille"] = pd.cut(f19["NBMEN"], bins=bornes, labels=etiquettes, right=False)

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=f19, x="taille", y="niveau_vie_median", showfliers=False)
    plt.title("Niveau de vie médian selon le nombre de ménages fiscaux (2019, valeurs extrêmes masquées)")
    plt.xlabel("Nombre de ménages fiscaux")
    plt.ylabel("Niveau de vie médian (€ par an et par UC)")
    plt.show()

    print(f19.groupby("taille", observed=True)["niveau_vie_median"]
             .agg(nb_communes="count", mediane="median").round(0))
    groupes = [g["niveau_vie_median"].values for _, g in f19.groupby("taille", observed=True)]
    H, p, eta2 = eta2_kruskal(groupes)
    print(f"Kruskal-Wallis : p-value = {p:.1e} | eta² = {eta2:.3f}")

    rho, p = stats.spearmanr(f19["NBMEN"], f19["niveau_vie_median"])
    print(f"Spearman : rho = {rho:.2f} | p-value = {p:.1e} | n = {len(f19)}")

    # --- B1.3 Évolution du niveau de vie dans le temps ---
    afficher_section('B1.3 Évolution du niveau de vie dans le temps')
    # Une ligne par commune, une colonne par année ; on garde les communes complètes
    med = filo.pivot(index="code_commune", columns="annee", values="niveau_vie_median").dropna()
    print("Communes avec une médiane les 8 années :", len(med))

    # Médiane des communes par année, avec la bande Q1-Q3
    q = med.quantile([0.25, 0.5, 0.75])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(med.columns, q.loc[0.5], marker="o", label="médiane des communes")
    ax.fill_between(med.columns, q.loc[0.25], q.loc[0.75], alpha=0.3, label="de Q1 à Q3")
    ax.set_title("Niveau de vie médian des communes, 2014-2021")
    ax.set_ylabel("€ par an et par UC")
    ax.legend()
    plt.show()
    # 1. Les communes gardent-elles leur rang entre 2014 et 2021 ?
    rho, p = stats.spearmanr(med[2014], med[2021])
    print(f"Spearman 2014 / 2021 : rho = {rho:.2f} | p-value = {p:.1e} | n = {len(med)}")

    # 2. Part des écarts due aux différences entre communes (plutôt qu'au temps)
    total = med.stack().var()
    entre = med.mean(axis=1).var()
    print(f"Part des écarts entre communes : {entre / total * 100:.1f} %")

    # 3. Évolutions annuelles : ordre de grandeur
    evol = med.pct_change(axis=1).iloc[:, 1:] * 100   # évolution de chaque commune d'une année à l'autre (%)
    print("\nÉvolution annuelle (%) :")
    print(evol.stack().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1))

    # -------------------------------------------------------------------------
    # B2. Parc de logement et vacance (recensement 2022, LOVAC)
    # -------------------------------------------------------------------------
    afficher_section('B2. Parc de logement et vacance (recensement 2022, LOVAC)')
    sns.set_theme(style="whitegrid")
    bleu = "#2a78d6"

    # Base logement à la commune (bloc 7 : IRIS additionnés, 19 indicateurs en %)
    com = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet")

    # On garde les communes d'au moins 50 logements (taux trop instables en dessous)
    com = com[~com["petite_commune"]].copy()

    # Taille de la commune en 5 tranches (nombre de logements) : du village à la grande ville
    com["taille"] = pd.cut(com["P22_LOG"], bins=[0, 500, 2000, 10000, 50000, 2000000],
                           labels=["< 500", "500-2 000", "2 000-10 000", "10 000-50 000", "> 50 000"])
    print("Communes étudiées :", len(com))
    print(com["taille"].value_counts().sort_index())

    # --- B2.1 Distribution du taux de vacance ---
    afficher_section('B2.1 Distribution du taux de vacance')
    plt.figure(figsize=(9, 5))
    sns.histplot(com["taux_vacance"], bins=60, color=bleu)
    plt.axvline(com["taux_vacance"].median(), color="black", linestyle="--", label="médiane")
    plt.axvline(com["taux_vacance"].mean(), color="black", linestyle=":", label="moyenne")
    plt.title("Distribution du taux de vacance par commune (recensement 2022)")
    plt.xlabel("Taux de vacance (%)")
    plt.ylabel("Nombre de communes")
    plt.legend()
    plt.show()

    # Validation : moyenne > médiane = distribution étirée à droite ; quantiles pour situer les extrêmes
    print("Moyenne :", round(com["taux_vacance"].mean(), 1), "% | Médiane :", round(com["taux_vacance"].median(), 1), "%")
    print(com["taux_vacance"].quantile([0.25, 0.75, 0.9, 0.99]).round(1))
    # Taux national (somme des vacants / somme des logements) : point de comparaison
    print("Taux national :", round(com["P22_LOGVAC"].sum() / com["P22_LOG"].sum() * 100, 1), "%")

    # --- B2.2 Petits logements selon la taille de la commune ---
    afficher_section('B2.2 Petits logements selon la taille de la commune')
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=com, x="taille", y="part_moins40m2", color=bleu, showfliers=False)
    plt.title("Part de petits logements (< 40 m²) selon la taille de la commune")
    plt.xlabel("Nombre de logements de la commune")
    plt.ylabel("Part des résidences principales de moins de 40 m² (%)")
    plt.show()

    # Validation 1 : moyenne et médiane par tranche
    print(com.groupby("taille", observed=True)["part_moins40m2"].agg(["count", "mean", "median"]).round(1))

    # Validation 2 : ANOVA (5 groupes)
    tranches = ["< 500", "500-2 000", "2 000-10 000", "10 000-50 000", "> 50 000"]
    groupes = []
    for t in tranches:
        groupes.append(com[com["taille"] == t]["part_moins40m2"])
    print(stats.f_oneway(groupes[0], groupes[1], groupes[2], groupes[3], groupes[4]))

    # Taille d'effet : êta² = part de la variance expliquée par la taille de la commune
    moyenne = com["part_moins40m2"].mean()
    entre = 0
    for g in groupes:
        entre = entre + len(g) * (g.mean() - moyenne) ** 2
    total = ((com["part_moins40m2"] - moyenne) ** 2).sum()
    print("êta² :", round(entre / total, 3))

    # --- B2.3 Mobilité résidentielle et marché locatif ---
    afficher_section('B2.3 Mobilité résidentielle et marché locatif')
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=com, x="part_locataires", y="part_emmenages_2ans", s=5, alpha=0.15, color=bleu)
    plt.title("Ménages arrivés depuis moins de 2 ans selon la part de locataires")
    plt.xlabel("Part de locataires parmi les résidences principales (%)")
    plt.ylabel("Ménages emménagés depuis moins de 2 ans (%)")
    plt.show()

    # Validation : corrélations (le coefficient sert aussi de taille d'effet)
    print("Pearson :", stats.pearsonr(com["part_locataires"], com["part_emmenages_2ans"]))
    print("Spearman :", stats.spearmanr(com["part_locataires"], com["part_emmenages_2ans"]))

    # --- B2.4 Ménages sans voiture, communes denses ou non ---
    afficher_section('B2.4 Ménages sans voiture, communes denses ou non')
    # Commune "dense" : majorité d'appartements
    com["habitat"] = np.where(com["part_apparts"] > 50, "Majorité d'appartements", "Majorité de maisons")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=com, x="habitat", y="part_sans_voiture", color=bleu, showfliers=False)
    plt.title("Part des ménages sans voiture selon le type d'habitat")
    plt.xlabel("")
    plt.ylabel("Ménages sans voiture (%)")
    plt.show()

    # Validation : test t de Welch (2 groupes) + taille d'effet (d de Cohen)
    dense = com[com["habitat"] == "Majorité d'appartements"]["part_sans_voiture"]
    pavillon = com[com["habitat"] == "Majorité de maisons"]["part_sans_voiture"]
    print("Communes :", len(dense), "denses /", len(pavillon), "pavillonnaires")
    print("Moyennes :", round(dense.mean(), 1), "% /", round(pavillon.mean(), 1), "%")
    print(stats.ttest_ind(dense, pavillon, equal_var=False))
    ecart_type = np.sqrt((dense.var() + pavillon.var()) / 2)
    print("d de Cohen :", round((dense.mean() - pavillon.mean()) / ecart_type, 2))

    # --- B2.5 Corrélations entre indicateurs ---
    afficher_section('B2.5 Corrélations entre indicateurs')
    # Indicateurs calculés à l'IRIS (au moins 50 logements), sur la base brute du recensement
    # (nom `iris` : `l` sert plus loin aux annonces de location, en B5)
    codes_texte = {"IRIS": str, "COM": str, "TYP_IRIS": str, "LAB_IRIS": str}
    iris = pd.read_csv(DATA_RAW / "base-ic-logement-2022_csv" / "base-ic-logement-2022.CSV",
                       sep=";", dtype=codes_texte)
    iris = iris[iris["P22_LOG"] >= 50].copy()
    log_, rp = iris["P22_LOG"], iris["P22_RP"].replace(0, np.nan)
    iris["taux_vacance"] = iris["P22_LOGVAC"] / log_ * 100
    iris["part_rsec"] = iris["P22_RSECOCC"] / log_ * 100
    iris["part_maisons"] = iris["P22_MAISON"] / log_ * 100
    iris["part_proprio"] = iris["P22_RP_PROP"] / rp * 100
    iris["part_hlm"] = iris["P22_RP_LOCHLMV"] / rp * 100
    iris["part_avant1946"] = (iris["P22_RP_ACH1919"] + iris["P22_RP_ACH1945"]) / rp * 100
    print("IRIS étudiés :", len(iris))
    indicateurs = ["taux_vacance", "part_rsec", "part_maisons", "part_proprio", "part_hlm", "part_avant1946"]
    correlations = iris[indicateurs].corr(method="spearman")

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Corrélations de Spearman entre indicateurs de logement")
    plt.show()

    # Validation : test des corrélations les plus fortes
    # nan_policy="omit" : ignore les IRIS sans résidence principale (parts vides)
    print("Maisons / propriétaires :", stats.spearmanr(iris["part_maisons"], iris["part_proprio"], nan_policy="omit"))
    print("Propriétaires / HLM :", stats.spearmanr(iris["part_proprio"], iris["part_hlm"], nan_policy="omit"))
    print("Maisons / HLM :", stats.spearmanr(iris["part_maisons"], iris["part_hlm"], nan_policy="omit"))

    orange = "#eb6834"
    chemin_lovac = DATA_RAW / "lovac-open-data-2020-a-2026.xlsx"

    # LOVAC, feuille communale : "s" (secret) -> valeur manquante, puis conversion en nombres
    lovac = pd.read_excel(chemin_lovac, sheet_name="COM open data", dtype=str)
    for col in lovac.columns:
        if col.startswith("pp_") or col.startswith("ff_"):
            lovac[col] = pd.to_numeric(lovac[col].replace("s", np.nan))

    # Base logement 2022 (recensement), additionnée par commune (bloc 7) : sert de point de comparaison
    rp = (pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet",
                          columns=["code_commune", "P22_LOG", "P22_LOGVAC"])
            .rename(columns={"code_commune": "COM"}))

    # Jointure par code commune (LOVAC : géographie 2026 ; recensement : géographie 2024)
    compar = lovac.merge(rp, left_on="CODGEO_26", right_on="COM", how="inner")
    print("Communes LOVAC :", len(lovac), "| recensement :", len(rp), "| communes communes aux deux :", len(compar))

    # --- B2.6 LOVAC : évolution nationale et rupture de série ---
    afficher_section('B2.6 LOVAC : évolution nationale et rupture de série')
    # Feuille FR : totaux nationaux par millésime
    fr = pd.read_excel(chemin_lovac, sheet_name="FR")
    fr["annee_situation"] = fr["Millésime"] - 1          # millésime N = situation au 1er janvier N-1
    fr = fr.sort_values("annee_situation")

    plt.figure(figsize=(9, 5))
    plt.plot(fr["annee_situation"], fr["pp_vacant"] / 1e6, marker="o", color=bleu, label="Logements vacants")
    plt.plot(fr["annee_situation"], fr["pp_vacant_plus_2ans"] / 1e6, marker="o", color=orange, label="Vacants depuis plus de 2 ans")
    plt.axvspan(2022.5, 2024.5, color="grey", alpha=0.15, label="Après la fin de la taxe d'habitation")
    plt.title("Logements privés vacants en France (LOVAC)")
    plt.xlabel("Situation au 1er janvier de l'année")
    plt.ylabel("Nombre de logements (millions)")
    plt.ylim(0, 4.5)
    plt.legend()
    plt.show()

    # Validation 1 : les chiffres et le taux national (quand le parc est connu)
    fr["taux_%"] = (fr["pp_vacant"] / fr["ff_pp_total"] * 100).round(1)
    print(fr[["Millésime", "annee_situation", "pp_vacant", "pp_vacant_plus_2ans", "taux_%"]])

    # Validation 2 : le classement des communes est-il stable d'une année à l'autre ?
    # Corrélation de Spearman entre les taux de vacance de deux millésimes consécutifs
    for a, b in [("20", "21"), ("21", "22"), ("22", "23"), ("23", "24"), ("24", "25")]:
        ta = lovac["pp_vacant_" + a] / lovac["ff_pp_total_" + a]
        tb = lovac["pp_vacant_" + b] / lovac["ff_pp_total_" + b]
        ok = ta.notna() & tb.notna()
        print("Millésimes 20" + a, "-> 20" + b, ": Spearman =", round(stats.spearmanr(ta[ok], tb[ok])[0], 3))

    # --- B2.7 LOVAC : le secret statistique selon la taille de la commune ---
    afficher_section('B2.7 LOVAC : le secret statistique selon la taille de la commune')
    compar["taille"] = pd.cut(compar["P22_LOG"], bins=[0, 100, 500, 2000, 10000, 2000000],
                              labels=["< 100", "100-500", "500-2 000", "2 000-10 000", "> 10 000"])
    compar["parc_secret"] = compar["ff_pp_total_23"].isna()     # parc privé caché (millésime 2023)

    part_secret = compar.groupby("taille", observed=True)["parc_secret"].mean() * 100

    plt.figure(figsize=(9, 5))
    sns.barplot(x=part_secret.index, y=part_secret.values, color=bleu)
    plt.title("Part des communes dont le parc privé est caché (secret statistique, millésime 2023)")
    plt.xlabel("Nombre de logements de la commune (recensement 2022)")
    plt.ylabel("Communes sous secret (%)")
    plt.show()

    # Validation 1 : part de secret et part des logements couverts
    print(part_secret.round(1))
    couvert = compar[~compar["parc_secret"]]["P22_LOG"].sum() / compar["P22_LOG"].sum() * 100
    print("Part des logements situés dans une commune au parc connu :", round(couvert, 1), "%")

    # Validation 2 : khi-deux taille × secret + V de Cramér
    contingence = pd.crosstab(compar["taille"], compar["parc_secret"])
    khi2, p, ddl, attendu = stats.chi2_contingency(contingence)
    v_cramer = np.sqrt(khi2 / (contingence.values.sum() * (min(contingence.shape) - 1)))
    print("khi-deux :", round(khi2), "| p-value :", p, "| V de Cramér :", round(v_cramer, 2))

    # --- B2.8 Vacance LOVAC et vacance du recensement ---
    afficher_section('B2.8 Vacance LOVAC et vacance du recensement')
    # Millésime 2023 (situation au 1er janvier 2022), le plus proche du recensement 2022
    compar["taux_lovac"] = compar["pp_vacant_23"] / compar["ff_pp_total_23"] * 100
    compar["taux_recensement"] = compar["P22_LOGVAC"] / compar["P22_LOG"] * 100
    k = compar.dropna(subset=["taux_lovac", "taux_recensement"])
    k = k[k["P22_LOG"] >= 50]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=k, x="taux_recensement", y="taux_lovac", s=5, alpha=0.2, color=bleu)
    plt.plot([0, 40], [0, 40], color="black", linestyle="--", label="égalité")
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.title("Taux de vacance par commune : LOVAC (2022) et recensement (2022)")
    plt.xlabel("Taux de vacance, recensement (%)")
    plt.ylabel("Taux de vacance du parc privé, LOVAC (%)")
    plt.legend()
    plt.show()

    # Validation : niveaux, puis corrélations
    print("Communes comparées :", len(k))
    print("Médianes : recensement", round(k["taux_recensement"].median(), 1), "% | LOVAC", round(k["taux_lovac"].median(), 1), "%")
    print("Part des communes où LOVAC > recensement :", round((k["taux_lovac"] > k["taux_recensement"]).mean() * 100, 1), "%")
    print("Pearson :", stats.pearsonr(k["taux_recensement"], k["taux_lovac"]))
    print("Spearman :", stats.spearmanr(k["taux_recensement"], k["taux_lovac"]))

    # -------------------------------------------------------------------------
    # B3. Délinquance enregistrée (SSMSI)
    # -------------------------------------------------------------------------

    # --- Délinquance au bassin de vie ---
    afficher_section('Délinquance au bassin de vie')
    # Base communale brute
    delinq = pd.read_parquet(DATA_RAW / "dataset_delinquance_communale.parquet")

    # Table de liaison commune -> bassin de vie (bloc 1 de la préparation)
    liaison = pd.read_parquet(DATA_OUT / "liaison_commune_bv2022.parquet")
    liaison_ini = liaison[["CODGEO_INI", "BV2022"]]   # une ligne par code commune d'origine

    # Maille : bassin de vie (option A) ou arrondissements séparés pour Paris, Lyon, Marseille (option B)
    DECOUPER_GRANDES_VILLES = False
    # 1. Correspondance code 2026 -> bassin de vie
    liaison_2026 = liaison[["CODGEO_2026", "BV2022"]].drop_duplicates()

    # 2. Paris, Lyon, Marseille : un seul niveau, selon l'option choisie pour la cible
    arrondissement = delinq["CODGEO_2026"].str.match(r"^(751|6938|132)")
    ville_entiere = delinq["CODGEO_2026"].isin(["75056", "69123", "13055"])
    if DECOUPER_GRANDES_VILLES:
        d = delinq[~ville_entiere]    # option B : on garde les arrondissements
    else:
        d = delinq[~arrondissement]   # option A : on garde la ville entière

    # 3. Rattachement au bassin de vie, et maille identique à celle de la cible
    d = d.merge(liaison_2026, on="CODGEO_2026", how="left", validate="many_to_one")
    d["maille"] = d["BV2022"]
    if DECOUPER_GRANDES_VILLES:
        a = d["CODGEO_2026"].str.match(r"^(751|6938|132)")
        d.loc[a, "maille"] = d.loc[a, "CODGEO_2026"]

    # 4. Communes masquées : moyenne départementale ; dénominateur : logements pour les cambriolages
    d["nombre_estime"] = d["nombre"].fillna(d["complement_info_nombre"])
    d["denominateur"] = np.where(d["indicateur"] == "Cambriolages de logement", d["insee_log"], d["insee_pop"])

    # 5. Somme des faits et des habitants par maille × année × indicateur, puis taux
    delinq_bv = (d.groupby(["maille", "annee", "indicateur"], observed=True)[["nombre_estime", "denominateur"]]
                  .sum().reset_index())
    delinq_bv["taux"] = delinq_bv["nombre_estime"] / delinq_bv["denominateur"] * 1000

    # 6. Format large : une ligne par maille × année, une colonne par indicateur
    delinq_bv["indicateur"] = delinq_bv["indicateur"].astype(str)   # évite les colonnes vides des 3 indicateurs départementaux
    taux_bv = delinq_bv.pivot(index=["maille", "annee"], columns="indicateur", values="taux").reset_index()

    print(taux_bv.shape)
    print("Mailles :", taux_bv["maille"].nunique())
    # 2025 est l'année de test : on la retire dès maintenant de toute l'EDA
    taux_bv = taux_bv[taux_bv["annee"] <= 2024]
    print(taux_bv.shape)
    import geopandas as gpd

    # Contours des bassins de vie : contours des communes, fusionnés par bassin
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes-version-simplifiee.geojson"
    communes = gpd.read_file(url)
    communes = communes.merge(liaison_ini, left_on="code", right_on="CODGEO_INI", how="left")
    carte_bv = communes.dissolve(by="BV2022").reset_index()[["BV2022", "geometry"]]

    # Taux lissés d'une période : somme des faits / somme des habitants, au niveau demandé
    def taux_periode(df, debut, fin, cles):
        t = df[df["annee"].between(debut, fin)].groupby(cles)[["nombre_estime", "denominateur"]].sum()
        return t["nombre_estime"] / t["denominateur"] * 1000

    # Contours des communes : plus utiles une fois les bassins dessinés (libère la mémoire)
    del communes
    import gc; gc.collect()

    # --- B3.1 Redondances entre indicateurs ---
    afficher_section('B3.1 Redondances entre indicateurs')
    indicateurs = [c for c in taux_bv.columns if c not in ["maille", "annee"]]

    corr = taux_bv[indicateurs].corr(method="spearman")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".1f", annot_kws={"size": 8})
    plt.title(f"Corrélations de Spearman entre les taux de délinquance (n = {len(taux_bv):,} bassins × années)")
    plt.tight_layout()
    plt.show()

    # Correspondance indicateur -> famille (AFD exclu : déjà compris dans l'usage de stupéfiants)
    familles = {
        "Cambriolages de logement": "Cambriolages",
        "Vols de véhicule": "Vols liés aux véhicules",
        "Vols dans les véhicules": "Vols liés aux véhicules",
        "Vols d'accessoires sur véhicules": "Vols liés aux véhicules",
        "Vols avec armes": "Vols violents",
        "Vols violents sans arme": "Vols violents",
        "Vols sans violence contre des personnes": "Vols sans violence",
        "Violences physiques intrafamiliales": "Violences",
        "Violences physiques hors cadre familial": "Violences",
        "Violences sexuelles": "Violences",
        "Trafic de stupéfiants": "Stupéfiants",
        "Usage de stupéfiants": "Stupéfiants",
        "Escroqueries et fraudes aux moyens de paiement": "Escroqueries",
        "Destructions et dégradations volontaires": "Destructions et dégradations",
    }

    fam = delinq_bv[(delinq_bv["annee"] <= 2024) & delinq_bv["indicateur"].isin(familles)].copy()
    fam["famille"] = fam["indicateur"].map(familles)

    # Faits additionnés dans chaque famille ; population (ou logements) comptée une seule fois
    fam = (fam.groupby(["maille", "annee", "famille"])
              .agg(nombre_estime=("nombre_estime", "sum"), denominateur=("denominateur", "first"))
              .reset_index())
    fam["taux"] = fam["nombre_estime"] / fam["denominateur"] * 1000

    # Format large : une ligne par bassin × année, une colonne par famille
    taux_fam = fam.pivot(index=["maille", "annee"], columns="famille", values="taux").reset_index()
    liste_familles = ["Cambriolages", "Vols liés aux véhicules", "Vols violents", "Vols sans violence",
                      "Violences", "Stupéfiants", "Escroqueries", "Destructions et dégradations"]

    print(taux_fam.shape)
    # Une ligne par bassin : taux lissés sur 2021-2024, DOM exclus
    l21 = taux_periode(fam, 2021, 2024, ["maille", "famille"]).unstack("famille").reset_index()
    l21_metro = l21[l21["maille"].str[:2] != "97"]

    corr_fam = l21_metro[liste_familles].corr(method="spearman")

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_fam, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 9})
    plt.title(f"Corrélations de Spearman entre familles d'indicateurs (n = {len(l21_metro):,} bassins, 2021-2024)")
    plt.tight_layout()
    plt.show()

    # --- B3.2 Répartition sur le territoire ---
    afficher_section('B3.2 Répartition sur le territoire')
    # Taux lissés sur 2021-2024 : somme des faits sur 4 ans / somme des habitants sur 4 ans
    l21 = (fam[fam["annee"].between(2021, 2024)]
           .groupby(["maille", "famille"])[["nombre_estime", "denominateur"]].sum())
    l21["taux"] = l21["nombre_estime"] / l21["denominateur"] * 1000
    l21 = l21.reset_index().pivot(index="maille", columns="famille", values="taux").reset_index()

    carte_fam = carte_bv.merge(l21, left_on="BV2022", right_on="maille", how="left")

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))
    for ax, f in zip(axes.flatten(), liste_familles):
        carte_fam.plot(column=f, cmap="Reds", vmin=0, vmax=12, legend=True, linewidth=0,
                       missing_kwds={"color": "lightgrey"}, ax=ax,
                       legend_kwds={"label": "Taux pour 1 000", "shrink": 0.6})
        ax.set_title(f, fontsize=12)
        ax.set_axis_off()
    fig.suptitle("Taux par famille d'indicateurs et par bassin de vie, lissés sur 2021-2024 (même échelle)", fontsize=14)
    plt.tight_layout()
    plt.show()
    # Taux lissés sur 2021-2024 par famille
    lisse_fam = (fam[fam["annee"].between(2021, 2024)]
                 .groupby(["maille", "famille"])[["nombre_estime", "denominateur"]].sum())
    lisse_fam["taux"] = lisse_fam["nombre_estime"] / lisse_fam["denominateur"] * 1000
    lisse_fam = lisse_fam.reset_index().pivot(index="maille", columns="famille", values="taux").reset_index()

    lisse_fam["dep"] = lisse_fam["maille"].str[:2]   # département du bassin (celui de sa commune pôle)
    lisse_fam = lisse_fam[lisse_fam["dep"] != "97"]  # DOM hors périmètre

    lignes = []
    for f in liste_familles:
        x = lisse_fam[f].dropna()
        d1, med, d9 = x.quantile([0.1, 0.5, 0.9])
        groupes = [g[f].dropna().values for _, g in lisse_fam.groupby("dep")]
        H, p, eta2 = eta2_kruskal(groupes)
        lignes.append({"famille": f, "D1": round(d1, 2), "médiane": round(med, 2), "D9": round(d9, 2),
                       "D9 / D1": round(d9 / d1, 1) if d1 > 0 else np.nan,
                       "p-value": f"{p:.1e}", "η² département": round(eta2, 3)})

    display(pd.DataFrame(lignes).set_index("famille").sort_values("η² département", ascending=False))

    # --- B3.3 Évolution dans le temps ---
    afficher_section('B3.3 Évolution dans le temps')
    # Bassins de métropole (DOM hors périmètre) ; fam s'arrête déjà en 2024
    fam_metro = fam[fam["maille"].str[:2] != "97"]

    nat_fam = fam_metro.groupby(["famille", "annee"])[["nombre_estime", "denominateur"]].sum()
    nat_fam["taux"] = nat_fam["nombre_estime"] / nat_fam["denominateur"] * 1000
    nat_fam = nat_fam.reset_index()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    for ax, f in zip(axes.flatten(), liste_familles):
        g = nat_fam[nat_fam["famille"] == f]
        ax.plot(g["annee"], g["taux"], marker="o")
        ax.set_title(f, fontsize=11)
        ax.set_ylim(bottom=0)
    fig.suptitle("Taux national par famille d'indicateurs, pour 1 000 habitants (logements pour les cambriolages), 2016-2024")
    plt.tight_layout()
    plt.show()
    # Évolution de chaque bassin entre 2016-2018 et 2022-2024 (%)
    evol_fam = pd.DataFrame({"debut": taux_periode(fam_metro, 2016, 2018, ["maille", "famille"]),
                             "fin": taux_periode(fam_metro, 2022, 2024, ["maille", "famille"])}).reset_index()
    evol_fam["evol_bv"] = (evol_fam["fin"] / evol_fam["debut"] - 1) * 100
    evol_fam["evol_bv"] = evol_fam["evol_bv"].replace([np.inf, -np.inf], np.nan)   # aucun fait au départ : évolution impossible

    # Évolution nationale de chaque famille (%)
    evol_nat_fam = (taux_periode(fam_metro, 2022, 2024, "famille") / taux_periode(fam_metro, 2016, 2018, "famille") - 1) * 100
    print("Évolution nationale (%) :\n", evol_nat_fam.round(1))

    # Écart du bassin à la tendance nationale (points de %)
    evol_fam["ecart"] = evol_fam["evol_bv"] - evol_fam["famille"].map(evol_nat_fam)
    ecart_fam = evol_fam.pivot(index="maille", columns="famille", values="ecart").reset_index()
    carte_ecart_fam = carte_bv.merge(ecart_fam, left_on="BV2022", right_on="maille", how="left")

    fig, axes = plt.subplots(2, 4, figsize=(26, 13))
    for ax, f in zip(axes.flatten(), liste_familles):
        carte_ecart_fam.plot(column=f, cmap="RdBu_r", vmin=-50, vmax=50, legend=True, linewidth=0,
                             missing_kwds={"color": "lightgrey"}, ax=ax,
                             legend_kwds={"label": "Écart à la tendance nationale (points de %)", "shrink": 0.6})
        ax.set_title(f"{f}\n(France : {evol_nat_fam[f]:+.0f} %)", fontsize=12)
        ax.set_axis_off()
    fig.suptitle("Évolution 2016-2018 → 2022-2024 de chaque bassin, comparée à la France", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- B3.4 Stabilité des bassins dans le temps ---
    afficher_section('B3.4 Stabilité des bassins dans le temps')
    debut = taux_periode(fam_metro, 2016, 2018, ["maille", "famille"]).unstack("famille")
    fin = taux_periode(fam_metro, 2022, 2024, ["maille", "famille"]).unstack("famille")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, f in zip(axes.flatten(), liste_familles):
        d = pd.concat([debut[f], fin[f]], axis=1, keys=["debut", "fin"]).dropna()
        d = d[(d["debut"] > 0) & (d["fin"] > 0)]          # échelle logarithmique : pas de taux nul
        ax.scatter(d["debut"], d["fin"], s=4, alpha=0.3)
        lim = [d.min().min(), d.max().max()]
        ax.plot(lim, lim, color="grey", linestyle="--", label="même taux")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f, fontsize=11)
        ax.set_xlabel("Taux 2016-2018 (pour 1 000)")
        ax.set_ylabel("Taux 2022-2024 (pour 1 000)")
    axes[0, 0].legend()
    fig.suptitle("Taux de chaque bassin au début et à la fin de la période (échelles logarithmiques)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Taux de métropole par bassin et par année (DOM hors périmètre)
    tf = taux_fam[taux_fam["maille"].str[:2] != "97"]

    lignes = []
    for f in liste_familles:
        # 1. Les bassins gardent-ils leur rang ?
        d = pd.concat([debut[f], fin[f]], axis=1, keys=["debut", "fin"]).dropna()
        rho, p = stats.spearmanr(d["debut"], d["fin"])

        # 2. Part des écarts due aux différences entre bassins (plutôt qu'au temps)
        large = tf.pivot(index="maille", columns="annee", values=f).dropna()   # une ligne par bassin, une colonne par année
        total = large.stack().var()
        entre = large.mean(axis=1).var()

        lignes.append({"famille": f, "rho 2016-18 / 2022-24": round(rho, 2), "p-value": f"{p:.1e}",
                       "part entre bassins (%)": round(entre / total * 100, 1), "n bassins": len(d)})

    display(pd.DataFrame(lignes).set_index("famille"))

    # -------------------------------------------------------------------------
    # B5. Annonces immobilières du Haut-Rhin (axe 2)
    # -------------------------------------------------------------------------

    # --- Préparation des données suite à la première inspection ---
    afficher_section('Préparation des données suite à la première inspection')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    annonces_ventes = pd.read_csv(DATA_RAW / "ech_annonces_ventes_68.csv", sep=";", encoding="utf-8-sig", index_col='idannonce')
    location = pd.read_csv(DATA_RAW / "ech_annonces_locations_68.csv", sep=";", encoding="utf-8-sig",index_col="idannonce")

    sns.set_theme(style="whitegrid")          # fond blanc avec une grille légère
    couleurs = {"Appartement": "#2a78d6", "Maison": "#eb6834"}   # bleu / orange

    # ── Préparation : on retire les valeurs aberrantes repérées dans les premieres inspections ─────
    v = annonces_ventes[(annonces_ventes["surface"] >= 9)
               & (annonces_ventes["prix_bien"] >= 20000)
               & (annonces_ventes["prix_m2_vente"] >= 500)
               & (annonces_ventes["prix_m2_vente"] <= 8000)
               & (annonces_ventes["typedebien_lite"] != "l")
               & (annonces_ventes["typedetransaction"] == "v")].copy()

    # Libellés lisibles pour les graphiques
    v["type"] = v["typedebien_lite"].replace({"a": "Appartement", "m": "Maison"})
    v["neuf"] = v["logement_neuf"].replace({"o": "Neuf", "n": "Ancien"})
    v["annee"] = pd.to_datetime(v["date"]).dt.year

    # Location : Haut-Rhin uniquement, sans loyers ni surfaces aberrants
    l = location[(location["DEP"] == 68)
                 & (location["prix_bien"] >= 150)
                 & (location["surface"] >= 9)
                 & (location["typedebien_lite"] != "l")].copy()
    l["loyer_m2"] = l["prix_bien"] / l["surface"]      # loyer au m² (absent du fichier)

    print("Ventes conservées :", len(v), "| Locations conservées :", len(l))

    # --- B5.1 Distribution du prix des annonces ---
    afficher_section('B5.1 Distribution du prix des annonces')
    plt.figure(figsize=(6, 3))  

    sns.histplot(v["prix_bien"]/1000, bins=60) # on divise les valeurs par 1000 pour afficher en K€
    plt.title("Distribution du prix des annonces (K€)")

    plt.xlim(0, 1000)

    plt.xlabel("Prix (K€)")   

    plt.show()

    # Validation : si la moyenne dépasse nettement la médiane, la distribution est asymétrique
    print("Moyenne :", round(v["prix_bien"].mean()), "€ | Médiane :", v["prix_bien"].median(), "€")

    # --- B5.2 Prix au m² selon le type de bien et le neuf ---
    afficher_section('B5.2 Prix au m² selon le type de bien et le neuf')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=v, x="type", y="prix_m2_vente", hue="neuf",
                showfliers=False)                        # showfliers=False : on masque les points extrêmes
    plt.title("Prix au m² selon le type de bien et le neuf")
    plt.xlabel("")
    plt.ylabel("Prix au m² (€)")
    plt.show()

    # Validation : test t (compare deux moyennes), pour les appartements puis les maisons
    for t in ["Appartement", "Maison"]:
        groupe = v[v["type"] == t]
        neuf = groupe[groupe["neuf"] == "Neuf"]["prix_m2_vente"]
        ancien = groupe[groupe["neuf"] == "Ancien"]["prix_m2_vente"]
        print(t, ": neuf", round(neuf.mean()), "€/m² | ancien", round(ancien.mean()), "€/m²")
        print(stats.ttest_ind(neuf, ancien, equal_var=False))

    # --- B5.3 Prix en fonction de la surface ---
    afficher_section('B5.3 Prix en fonction de la surface')
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=v, x="surface", y="prix_bien", hue="type",
                    palette=couleurs, s=8, alpha=0.3)    # s = taille des points, alpha = transparence
    plt.xscale("log")                                    # échelles log : les points sont mieux répartis
    plt.yscale("log")
    plt.title("Prix selon la surface")
    plt.xlabel("Surface (m²)")
    plt.ylabel("Prix (€)")
    plt.show()

    # Validation : corrélation entre surface et prix, puis entre surface et prix au m²
    print("Pearson surface / prix :", stats.pearsonr(v["surface"], v["prix_bien"]))
    for t in ["Appartement", "Maison"]:
        groupe = v[v["type"] == t]
        print(t, "| Pearson surface / prix au m² :", stats.pearsonr(groupe["surface"], groupe["prix_m2_vente"]))
        print(t, "| Spearman surface / prix au m² :", stats.spearmanr(groupe["surface"], groupe["prix_m2_vente"]))

    # --- B5.4 Prix au m² selon l'étiquette DPE ---
    afficher_section("B5.4 Prix au m² selon l'étiquette DPE")
    ordre = ["A", "B", "C", "D", "E", "F", "G"]
    d = v[v["dpeL"].isin(ordre)]           # on retire 0, NS et VI

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=d, x="dpeL", y="prix_m2_vente", hue="type",
                order=ordre, palette=couleurs, showfliers=False)
    plt.title("Prix au m² selon l'étiquette DPE")
    plt.xlabel("Étiquette DPE")
    plt.ylabel("Prix au m² (€)")
    plt.show()

    # Validation 1 : ANOVA (compare les moyennes des 7 classes)
    print(stats.f_oneway(d[d["dpeL"] == "A"]["prix_m2_vente"], d[d["dpeL"] == "B"]["prix_m2_vente"],
                         d[d["dpeL"] == "C"]["prix_m2_vente"], d[d["dpeL"] == "D"]["prix_m2_vente"],
                         d[d["dpeL"] == "E"]["prix_m2_vente"], d[d["dpeL"] == "F"]["prix_m2_vente"],
                         d[d["dpeL"] == "G"]["prix_m2_vente"]))

    # Validation 2 : même ANOVA sur les seuls logements anciens (sans l'effet du neuf)
    anc = d[d["neuf"] == "Ancien"]
    print(anc.groupby("dpeL")["prix_m2_vente"].median()) # on affiche pour visualiser la decote de prix en fonction de l'étiquette pour les logements anciens
    print(stats.f_oneway(anc[anc["dpeL"] == "A"]["prix_m2_vente"], anc[anc["dpeL"] == "B"]["prix_m2_vente"],
                         anc[anc["dpeL"] == "C"]["prix_m2_vente"], anc[anc["dpeL"] == "D"]["prix_m2_vente"],
                         anc[anc["dpeL"] == "E"]["prix_m2_vente"], anc[anc["dpeL"] == "F"]["prix_m2_vente"],
                         anc[anc["dpeL"] == "G"]["prix_m2_vente"]))

    # --- B5.5 Évolution du prix au m² dans le temps ---
    afficher_section('B5.5 Évolution du prix au m² dans le temps')
    v2020 = v[v["annee"] >= 2020]                        # 2019 contient trop peu d'annonces

    # Prix au m² médian par année et par type de bien
    evolution = v2020.groupby(["annee", "type"])["prix_m2_vente"].median().reset_index()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=evolution, x="annee", y="prix_m2_vente", hue="type",
                 palette=couleurs, marker="o")
    plt.xticks([2020, 2021, 2022, 2023])                 # une graduation par année
    plt.title("Évolution du prix au m² médian (2023 : janvier à avril)")
    plt.xlabel("Année")
    plt.ylabel("Prix au m² médian (€)")
    plt.show()

    # Validation 1 : ANOVA entre les années, pour les maisons puis les appartements
    for t in ["Maison", "Appartement"]:
        groupe = v2020[v2020["type"] == t]
        print(t, groupe.groupby("annee")["prix_m2_vente"].mean().round())
        print(stats.f_oneway(groupe[groupe["annee"] == 2020]["prix_m2_vente"],
                             groupe[groupe["annee"] == 2021]["prix_m2_vente"],
                             groupe[groupe["annee"] == 2022]["prix_m2_vente"],
                             groupe[groupe["annee"] == 2023]["prix_m2_vente"]))

    # Validation 2 : khi-deux, la part maisons / appartements change-t-elle selon l'année ?
    print(stats.chi2_contingency(pd.crosstab(v["annee"], v["type"]))[:2])

    # =========================================================================
    # C. RELATIONS AVEC LA CIBLE
    # =========================================================================

    # -------------------------------------------------------------------------
    # C1. Caractéristiques des biens vendus (DVF+)
    # -------------------------------------------------------------------------
    afficher_section('C1. Caractéristiques des biens vendus (DVF+)')
    del pop   # population de A, plus utilisée : les ventes retenues sont dans `ventes`

    # Échantillon d'apprentissage (bloc 3 de la préparation)
    ech = pd.read_parquet(DATA_OUT / "dvf_plus_echantillon_apprentissage.parquet")
    ech = regles_exclusion(ech, afficher=True)

    # --- C1.1 Prix au m² selon la surface ---
    afficher_section('C1.1 Prix au m² selon la surface')
    # Tranches de surface
    tranches = [0, 30, 50, 70, 90, 120, np.inf]
    noms = ["moins de 30", "30-49", "50-69", "70-89", "90-119", "120 et plus"]
    ech["tranche_surface"] = pd.cut(ech["sbati"], bins=tranches, labels=noms, right=False)

    # Boxplot du prix au m² par tranche et par type de bien
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=ech, x="tranche_surface", y="prix_m2", hue="type_bien", showfliers=False)
    plt.xlabel("Surface bâtie (m²)")
    plt.ylabel("Prix au m² (€)")
    plt.title("Prix au m² selon la surface et le type de bien")
    plt.show()

    # Prix médian par tranche, pour les chiffres du constat
    display(ech.groupby(["tranche_surface", "type_bien"], observed=True)["prix_m2"].median().unstack().round(0))
    for t in ["appartement", "maison"]:
        d = ech[ech["type_bien"] == t]
        rho, p = stats.spearmanr(d["sbati"], d["prix_m2"])
        print(f"{t} : rho = {rho:.2f} | p-value = {p:.1e} | n = {len(d)}")

    # --- C1.2 Prix au m² selon le type de biens ---
    afficher_section('C1.2 Prix au m² selon le type de biens')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Échelle normale, limitée à 15 000 €/m² pour la lisibilité
    sns.histplot(data=ech, x="prix_m2", hue="type_bien", bins=100, stat="density",
                 common_norm=False, element="step", ax=axes[0])
    axes[0].set_xlim(0, 15000)
    axes[0].set_title("Prix au m² (échelle normale)")
    axes[0].set_xlabel("Prix au m² (€)")

    # Échelle log
    sns.histplot(data=ech, x="prix_m2", hue="type_bien", bins=100, stat="density",
                 common_norm=False, element="step", log_scale=True, ax=axes[1])
    axes[1].set_title("Prix au m² (échelle log)")
    axes[1].set_xlabel("Prix au m² (€, échelle log)")

    plt.tight_layout()
    plt.show()

    # Médianes par type
    display(ech.groupby("type_bien")["prix_m2"].median().round(0))# analyse / visualisation

    appart = ech.loc[ech["type_bien"] == "appartement", "prix_m2"]
    maison = ech.loc[ech["type_bien"] == "maison", "prix_m2"]

    U, p, r = r_mannwhitney(appart, maison)
    print(f"Mann-Whitney : p-value = {p:.1e} | r = {r:.2f}")
    print(f"Probabilité qu'un appartement soit plus cher au m² qu'une maison : {(1 + r) / 2:.0%}")
    print(f"Écart des médianes : {appart.median() - maison.median():.0f} €/m²")

    # --- C1.3 Prix au m² selon le département ---
    afficher_section('C1.3 Prix au m² selon le département')
    import geopandas as gpd

    # Contours des départements (dépôt GitHub france-geojson)
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    deps = gpd.read_file(url)

    # Prix médian au m² par département, sur la population
    med_dep = ventes.groupby("coddep")["prix_m2"].median().reset_index(name="prix_m2_median")

    # Jointure sur le code département (texte des deux côtés)
    deps = deps.merge(med_dep, left_on="code", right_on="coddep", how="left")
    print("Départements sans données :", deps.loc[deps["prix_m2_median"].isna(), "nom"].tolist())

    # Carte
    ax = deps.plot(column="prix_m2_median", cmap="viridis", legend=True, vmax=5000,
                   missing_kwds={"color": "lightgrey"}, figsize=(8, 8))
    ax.set_axis_off()
    ax.set_title("Prix médian au m² par département (2014-2024)")
    plt.show()

    # Les 5 départements les plus chers et les 5 moins chers
    print(med_dep.sort_values("prix_m2_median").iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]].round(0))

    groupes = [g["prix_m2"].values for _, g in ventes.groupby("coddep")]

    H, p, eta2 = eta2_kruskal(groupes)
    print(f"Kruskal-Wallis : p-value = {p:.1e} | eta² = {eta2:.3f} | nombre de départements = {len(groupes)}")

    # Part d'appartements et prix médian par département (population)
    dep = (ventes.groupby("coddep")
                 .agg(nb_ventes=("prix_m2", "size"),
                      part_maisons=("est_maison", "mean"),
                      prix_m2_median=("prix_m2", "median"))
                 .reset_index())
    dep["part_appart (%)"] = (1 - dep["part_maisons"]) * 100

    # Contours des départements (déjà chargés plus haut) + nouvelles données
    carte_part = deps[["code", "nom", "geometry"]].merge(dep, left_on="code", right_on="coddep", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Carte de la part d'appartements
    carte_part.plot(column="part_appart (%)", cmap="Oranges", legend=True,
                    missing_kwds={"color": "lightgrey"}, ax=axes[0])
    axes[0].set_axis_off()
    axes[0].set_title("Part d'appartements dans les ventes, par département (%)")

    # Un point par département : part d'appartements et prix médian
    sns.scatterplot(data=dep, x="part_appart (%)", y="prix_m2_median", ax=axes[1])
    for d in ["75", "92", "06", "23", "13"]:
        ligne = dep[dep["coddep"] == d]
        axes[1].annotate(d, (ligne["part_appart (%)"].iloc[0], ligne["prix_m2_median"].iloc[0]))
    axes[1].set_xlabel("Part d'appartements dans les ventes (%)")
    axes[1].set_ylabel("Prix médian au m² (€)")
    axes[1].set_title("Part d'appartements et prix médian, par département")

    plt.tight_layout()
    plt.show()
    for t in ["maison", "appartement"]:
        groupes = [g["prix_m2"].values for _, g in ventes[ventes["type_bien"] == t].groupby("coddep")]
        H, p, eta2 = eta2_kruskal(groupes)
        print(f"{t} : p-value = {p:.1e} | eta² = {eta2:.3f} | {len(groupes)} départements")

    # Lien entre la part d'appartements et le prix médian, un point par département
    rho, p = stats.spearmanr(dep["part_appart (%)"], dep["prix_m2_median"])
    print(f"Spearman part d'appartements / prix médian : rho = {rho:.2f} | p-value = {p:.1e} | n = {len(dep)}")

    # --- C1.4 Prix au m² selon le nombre de pièces ---
    afficher_section('C1.4 Prix au m² selon le nombre de pièces')
    # Nombre de pièces de chaque vente, à partir des colonnes de DVF+
    # nbapt1pp = nombre d'appartements d'1 pièce, ..., nbapt5pp = 5 pièces et plus (idem pour les maisons)
    # Chaque vente porte sur un seul logement : une seule de ces colonnes vaut 1
    ech["nb_pieces"] = 0
    for k in range(1, 6):
        ech["nb_pieces"] += k * (ech[f"nbapt{k}pp"] + ech[f"nbmai{k}pp"])

    print(ech["nb_pieces"].value_counts().sort_index())
    # Ventes sans nombre de pièces renseigné : on les compte, puis on les écarte du graphique
    print("Ventes sans nombre de pièces :", (ech["nb_pieces"] == 0).sum())
    p = ech[ech["nb_pieces"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Répartition des ventes
    sns.countplot(data=p, x="nb_pieces", hue="type_bien", hue_order=["maison", "appartement"], ax=axes[0])
    axes[0].set_title("Nombre de ventes selon le nombre de pièces")
    axes[0].set_xlabel("Nombre de pièces (5 = 5 et plus)")
    axes[0].set_ylabel("Nombre de ventes")

    # Prix au m² selon le nombre de pièces
    sns.boxplot(data=p, x="nb_pieces", y="prix_m2", hue="type_bien", hue_order=["maison", "appartement"],
                showfliers=False, ax=axes[1])
    axes[1].set_title("Prix au m² selon le nombre de pièces")
    axes[1].set_xlabel("Nombre de pièces (5 = 5 et plus)")
    axes[1].set_ylabel("Prix au m² (€)")

    plt.tight_layout()
    plt.show()

    # Prix médian par nombre de pièces et par type
    display(p.groupby(["nb_pieces", "type_bien"])["prix_m2"].median().unstack().round(0))
    apt = ech[(ech["type_bien"] == "appartement") & (ech["nb_pieces"] > 0)]

    # Part des studios vendus par département (10 premiers, en %)
    part_1p = (apt[apt["nb_pieces"] == 1]["coddep"].value_counts(normalize=True) * 100).head(10)

    # Prix médian au m² par département et nombre de pièces
    tab = apt.pivot_table(index="coddep", columns="nb_pieces", values="prix_m2", aggfunc="median")
    tab["nb_ventes_1p"] = apt[apt["nb_pieces"] == 1].groupby("coddep").size()
    tab = tab[tab["nb_ventes_1p"] >= 100]   # assez de studios pour une médiane fiable
    tab["ecart_1p_vs_3p (%)"] = (tab[1] / tab[3] - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1 : où sont vendus les studios
    sns.barplot(x=part_1p.values, y=part_1p.index, color="tab:orange", ax=axes[0])
    axes[0].set_title("Part des appartements d'une pièce vendus (10 premiers départements)")
    axes[0].set_xlabel("Part des studios vendus (%)")
    axes[0].set_ylabel("Département")

    # Graphique 2 : un point par département
    sns.scatterplot(data=tab, x=3, y=1, ax=axes[1])
    lim = tab[[1, 3]].max().max() * 1.05
    axes[1].plot([0, lim], [0, lim], color="grey", linestyle="--", label="même prix")
    for code in ["75", "92", "06", "69", "33"]:     # quelques départements repérés sur le graphique
        if code in tab.index:
            axes[1].annotate(code, (tab.loc[code, 3], tab.loc[code, 1]))
    axes[1].set_title("Prix médian au m² : 1 pièce contre 3 pièces, par département")
    axes[1].set_xlabel("Prix médian au m², 3 pièces (€)")
    axes[1].set_ylabel("Prix médian au m², 1 pièce (€)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("Départements gardés :", len(tab))
    print("Départements où le studio est plus cher au m² :", (tab[1] > tab[3]).sum())
    print("Écart médian entre 1 pièce et 3 pièces :", round(tab["ecart_1p_vs_3p (%)"].median(), 1), "%")

    for t in ["appartement", "maison"]:
        d = ech[(ech["type_bien"] == t) & (ech["nb_pieces"] > 0)]
        groupes = [g["prix_m2"].values for _, g in d.groupby("nb_pieces")]
        H, p, eta2 = eta2_kruskal(groupes)
        print(f"{t} : Kruskal-Wallis : p-value = {p:.1e} | eta² = {eta2:.3f}")

    for t in ["appartement", "maison"]:
        d = ech[(ech["type_bien"] == t) & (ech["nb_pieces"] > 0)]
        rho, p = stats.spearmanr(d["sbati"], d["nb_pieces"])
        print(f"{t} : rho surface / nombre de pièces = {rho:.2f}")

    # --- C1.5 Prix au m² des maisons selon la surface du terrain ---
    afficher_section('C1.5 Prix au m² des maisons selon la surface du terrain')
    mai = ech[ech["type_bien"] == "maison"].copy()
    print("Part des maisons sans terrain :", round((mai["sterr"] == 0).mean() * 100, 1), "%")

    # Tranches de surface de terrain
    tranches_t = [0, 1, 500, 1000, 2000, 5000, np.inf]
    noms_t = ["0", "1-499", "500-999", "1 000-1 999", "2 000-4 999", "5 000 et plus"]
    mai["tranche_terrain"] = pd.cut(mai["sterr"], bins=tranches_t, labels=noms_t, right=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=mai, x="tranche_terrain", color="tab:blue", ax=axes[0])
    axes[0].set_title("Nombre de ventes de maisons selon le terrain")
    axes[0].set_xlabel("Surface du terrain (m²)")
    axes[0].set_ylabel("Nombre de ventes")

    sns.boxplot(data=mai, x="tranche_terrain", y="prix_m2", color="tab:blue", showfliers=False, ax=axes[1])
    axes[1].set_title("Prix au m² des maisons selon le terrain")
    axes[1].set_xlabel("Surface du terrain (m²)")
    axes[1].set_ylabel("Prix au m² (€)")

    plt.tight_layout()
    plt.show()

    # Prix médian par tranche
    display(mai.groupby("tranche_terrain", observed=True)["prix_m2"].median().round(0))

    # 1. Maisons sans terrain : où sont-elles, et sont-elles en copropriété ?
    sans = mai[mai["sterr"] == 0]
    print((sans["coddep"].value_counts(normalize=True) * 100).head(10).round(1))
    print("En copropriété (nblot > 0) :", round((sans["nblot"] > 0).mean() * 100, 1), "% des maisons sans terrain,",
          "contre", round((mai["nblot"] > 0).mean() * 100, 1), "% de l'ensemble des maisons")

    # 2. Les grands terrains sont-ils dans des départements moins chers ?
    prix_dep = dep[["coddep", "prix_m2_median"]].rename(columns={"prix_m2_median": "prix_dep"})
    mai = mai.merge(prix_dep, on="coddep", how="left")
    display(mai.groupby("tranche_terrain", observed=True)["prix_dep"].median().round(0))

    # --- C1.6 Prix au m² selon le nombre de dépendances ---
    afficher_section('C1.6 Prix au m² selon le nombre de dépendances')
    # Classes de dépendances : 0, 1, 2, 3 et plus
    ech["dep_classe"] = pd.cut(ech["nblocdep"], bins=[-1, 0, 1, 2, np.inf],
                               labels=["0", "1", "2", "3 et plus"])

    # Répartition des ventes (en % de chaque type de bien)
    print((pd.crosstab(ech["dep_classe"], ech["type_bien"], normalize="columns") * 100).round(1))

    # Prix médian au m² selon le nombre de dépendances
    print(ech.groupby(["dep_classe", "type_bien"], observed=True)["prix_m2"].median().unstack().round(0))

    # Boxplot
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=ech, x="dep_classe", y="prix_m2", hue="type_bien", hue_order=["maison", "appartement"],
                showfliers=False)
    plt.xlabel("Nombre de dépendances (garage, cave, parking...)")
    plt.ylabel("Prix au m² (€)")
    plt.title("Prix au m² selon le nombre de dépendances")
    plt.show()
    # Prix médian du département de chaque vente (tableau dep calculé en C1.3)
    prix_dep = dep[["coddep", "prix_m2_median"]].rename(columns={"prix_m2_median": "prix_dep"})
    ech_dep = ech.merge(prix_dep, on="coddep", how="left")

    # Pour chaque type et nombre de dépendances : prix des biens et prix de leur département
    comp = (ech_dep.groupby(["type_bien", "dep_classe"], observed=True)
                   .agg(prix_bien=("prix_m2", "median"), prix_departement=("prix_dep", "median"))
                   .reset_index())
    comp["dep_classe"] = comp["dep_classe"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, t in zip(axes, ["appartement", "maison"]):
        d = comp[comp["type_bien"] == t]
        ax.plot(d["dep_classe"], d["prix_bien"], marker="o", label="Prix médian des biens vendus")
        ax.plot(d["dep_classe"], d["prix_departement"], marker="o", linestyle="--",
                label="Prix médian de leur département")
        ax.set_title(t.capitalize())
        ax.set_xlabel("Nombre de dépendances")
        ax.set_ylabel("Prix au m² (€)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    display(comp.round(0))
    # Échantillon d'apprentissage : plus utilisé après C1 (libère la mémoire)
    del ech
    import gc; gc.collect()

    # -------------------------------------------------------------------------
    # C2. Revenus des habitants (Filosofi)
    # -------------------------------------------------------------------------

    # --- C2.0 Rapprochement avec la table cible ---
    afficher_section('C2.0 Rapprochement avec la table cible')
    # Table cible, années communes aux deux sources
    cible = pd.read_parquet(DATA_OUT / "cible_commune_annee.parquet")
    cible = cible[cible["annee"] <= 2021].copy()
    cible["n_total"] = cible["n_appart"] + cible["n_maison"]

    # Jointure à gauche : on garde toutes les communes × années de la cible
    fusion = cible.merge(filo, on=["code_commune", "annee"], how="left", indicator=True)

    # Trois cas pour chaque ligne de la cible
    fusion["cas"] = np.select(
        [fusion["_merge"] == "left_only", fusion["niveau_vie_median"].isna()],
        ["code absent de Filosofi", "médiane masquée"],
        default="revenu trouvé",
    )

    # Part des lignes et des ventes dans chaque cas
    bilan = fusion.groupby("cas").agg(lignes=("code_commune", "size"), ventes=("n_total", "sum"))
    bilan["% lignes"] = (bilan["lignes"] / bilan["lignes"].sum() * 100).round(1)
    bilan["% ventes"] = (bilan["ventes"] / bilan["ventes"].sum() * 100).round(1)
    display(bilan)

    # Évolution par année de la part des ventes avec un revenu
    par_annee = fusion.groupby("annee").apply(
        lambda d: d.loc[d["cas"] == "revenu trouvé", "n_total"].sum() / d["n_total"].sum() * 100
    ).round(1)
    print("\n% des ventes avec un revenu, par année :")
    print(par_annee)

    # --- C2.1 Prix au m² selon le niveau de vie médian ---
    afficher_section('C2.1 Prix au m² selon le niveau de vie médian')
    annees = [2014, 2016, 2018, 2020]
    types = [("maison", "Maisons"), ("appart", "Appartements")]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    for i, (t, nom) in enumerate(types):
        for j, an in enumerate(annees):
            d = fusion[(fusion["annee"] == an) & (fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]
            ax = axes[i, j]
            ax.scatter(d["niveau_vie_median"], d[f"prix_m2_med_{t}"], s=4, alpha=0.3)
            ax.set_yscale("log")
            ax.set_title(f"{nom} {an} (n = {len(d):,})")
            if j == 0:
                ax.set_ylabel("Prix médian au m² (€, log)")
            if i == 1:
                ax.set_xlabel("Niveau de vie médian (€)")
    plt.tight_layout()
    plt.show()

    lignes = []
    for an in range(2014, 2022):
        for t, nom in types:
            d = fusion[(fusion["annee"] == an) & (fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]
            rho, p = stats.spearmanr(d["niveau_vie_median"], d[f"prix_m2_med_{t}"])
            lignes.append({"annee": an, "type": nom, "rho": round(rho, 2), "n": len(d)})

    rho_annees = pd.DataFrame(lignes).pivot(index="annee", columns="type", values=["rho", "n"])
    display(rho_annees)

    # --- C2.2 Évolution des prix et du niveau de vie ---
    afficher_section('C2.2 Évolution des prix et du niveau de vie')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (t, nom) in zip(axes, [("maison", "Maisons"), ("appart", "Appartements")]):
        d = fusion[(fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]

        # On suit les mêmes communes chaque année (au moins 10 ventes les 8 années),
        # sinon la courbe changerait aussi parce que les communes changent
        nb_annees = d.groupby("code_commune")["annee"].nunique()
        d = d[d["code_commune"].isin(nb_annees[nb_annees == 8].index)]

        # Médiane des communes chaque année, puis indice base 100 en 2014
        serie = d.groupby("annee")[[f"prix_m2_med_{t}", "niveau_vie_median"]].median()
        indice = serie / serie.loc[2014] * 100

        ax.plot(indice.index, indice[f"prix_m2_med_{t}"], marker="o", label="Prix médian au m²")
        ax.plot(indice.index, indice["niveau_vie_median"], marker="s", label="Niveau de vie médian")
        ax.axhline(100, color="grey", ls=":")
        ax.set_title(f"{nom} : {d['code_commune'].nunique():,} communes suivies de 2014 à 2021")
        ax.set_xlabel("Année")
        ax.set_ylabel("Indice (2014 = 100)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    def evolution(t, debut, fin):
        """Une ligne par commune : prix et niveau de vie au début et à la fin de la période, et leurs évolutions (%)."""
        cols = ["code_commune", f"n_{t}", f"prix_m2_med_{t}", "niveau_vie_median"]
        a = fusion.loc[fusion["annee"] == debut, cols]
        b = fusion.loc[fusion["annee"] == fin, cols]
        d = a.merge(b, on="code_commune", suffixes=("_debut", "_fin"))
        d = d[(d[f"n_{t}_debut"] >= 10) & (d[f"n_{t}_fin"] >= 10)].dropna()
        d["evol_prix"] = (d[f"prix_m2_med_{t}_fin"] / d[f"prix_m2_med_{t}_debut"] - 1) * 100
        d["evol_nv"] = (d["niveau_vie_median_fin"] / d["niveau_vie_median_debut"] - 1) * 100
        return d

    evo = {"Maisons": evolution("maison", 2014, 2021),
           "Appartements": evolution("appart", 2014, 2021)}

    for nom, d in evo.items():
        rho_niv, p_niv = stats.spearmanr(d["niveau_vie_median_debut"], d["evol_prix"])
        rho_evo, p_evo = stats.spearmanr(d["evol_nv"], d["evol_prix"])
        print(f"{nom} (n = {len(d)} communes)")
        print(f"  niveau de vie 2014         / évolution des prix : rho = {rho_niv:.2f} | p-value = {p_niv:.1e}")
        print(f"  évolution du niveau de vie / évolution des prix : rho = {rho_evo:.2f} | p-value = {p_evo:.1e}")

    for t, nom in [("maison", "Maisons"), ("appart", "Appartements")]:
        d = fusion[(fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]
        d = d[["code_commune", "annee", f"prix_m2_med_{t}", "niveau_vie_median"]].copy()
        d["log_prix"] = np.log(d[f"prix_m2_med_{t}"])

        prec = d[["code_commune", "annee", "log_prix"]].rename(columns={"log_prix": "log_prix_prec"})
        prec["annee"] += 1
        d = d.merge(prec, on=["code_commune", "annee"])

        train = d[d["annee"].between(2015, 2018)]
        test = d[d["annee"] == 2019]
        evol_reelle = test["log_prix"] - test["log_prix_prec"]

        print(f"{nom} (test : {len(test)} communes)")
        for nom_modele, X in [("sans revenu", ["log_prix_prec"]),
                              ("avec revenu", ["log_prix_prec", "niveau_vie_median"])]:
            pred = LinearRegression().fit(train[X], train["log_prix"]).predict(test[X])
            erreur = np.median(np.abs(np.exp(pred) / test[f"prix_m2_med_{t}"] - 1)) * 100
            print(f"  {nom_modele:12} R² évolution = {r2_score(evol_reelle, pred - test['log_prix_prec']):.3f}"
                  f" | erreur médiane = {erreur:.1f} %")

    # --- C2.3 Comparaison géographique : prix des maisons et niveau de vie ---
    afficher_section('C2.3 Comparaison géographique : prix des maisons et niveau de vie')
    import geopandas as gpd

    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    deps = gpd.read_file(url)

    groupes = ["1 - très bas", "2 - bas", "3 - moyen", "4 - élevé", "5 - très élevé"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 15))

    for i, (t, nom) in enumerate([("maison", "maisons"), ("appart", "appartements")]):
        # Communes de 2019 avec au moins 10 ventes du type de bien et un niveau de vie connu
        d = fusion[(fusion["annee"] == 2019) & (fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)].copy()
        d["dep"] = d["code_commune"].str[:2]

        # Une ligne par département : médiane des communes
        par_dep = (d.groupby("dep")
                     .agg(prix=(f"prix_m2_med_{t}", "median"),
                          niveau_vie=("niveau_vie_median", "median"),
                          nb_communes=("code_commune", "size"))
                     .reset_index())

        carte = deps.merge(par_dep, left_on="code", right_on="dep", how="left")
        carte["classe_prix"] = pd.qcut(carte["prix"], 5, labels=groupes)
        carte["classe_nv"] = pd.qcut(carte["niveau_vie"], 5, labels=groupes)

        for j, (col, titre) in enumerate([("classe_prix", f"Prix médian au m² des {nom} (2019)"),
                                          ("classe_nv", f"Niveau de vie médian (communes avec ventes de {nom})")]):
            ax = axes[i, j]
            carte.plot(column=col, cmap="YlOrRd", legend=True, ax=ax,
                       edgecolor="white", linewidth=0.3,
                       missing_kwds={"color": "lightgrey", "label": "pas de données"})
            ax.set_title(titre)
            ax.set_axis_off()

        print(f"{nom} : {par_dep['dep'].nunique()} départements, de {par_dep['nb_communes'].min()} "
              f"à {par_dep['nb_communes'].max()} communes par département")

    plt.tight_layout()
    plt.show()

    # Test 1 : lien entre prix et niveau de vie au niveau des départements, pour chaque type de bien
    for t, nom in [("maison", "Maisons"), ("appart", "Appartements")]:
        d = fusion[(fusion["annee"] == 2019) & (fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)].copy()
        d["dep"] = d["code_commune"].str[:2]

        # Une ligne par département : médiane des communes
        pdep = d.groupby("dep").agg(prix=(f"prix_m2_med_{t}", "median"),
                                    niveau_vie=("niveau_vie_median", "median"))

        rho, p = stats.spearmanr(pdep["prix"], pdep["niveau_vie"])
        print(f"{nom} : rho = {rho:.2f} | p-value = {p:.1e} | n = {len(pdep)} départements")

    # -------------------------------------------------------------------------
    # C3. Temps et financement (DVF+, taux des crédits)
    # -------------------------------------------------------------------------

    # --- C3.1 Évolution du prix au m² et du nombre de ventes dans le temps ---
    afficher_section('C3.1 Évolution du prix au m² et du nombre de ventes dans le temps')
    # Trimestre de chaque vente
    ventes["trimestre"] = pd.to_datetime(ventes["datemut"], format="%Y-%m-%d").dt.to_period("Q")

    # Prix médian par trimestre et par type
    trim = (ventes.groupby(["trimestre", "type_bien"])["prix_m2"]
                  .median().reset_index(name="prix_m2_median"))
    trim["trimestre"] = trim["trimestre"].dt.to_timestamp()   # format date, pour l'axe du graphique

    # Nombre de ventes par année et par type
    annuel = ventes.groupby(["anneemut", "type_bien"]).size().reset_index(name="nb_ventes")
    annuel["date"] = pd.to_datetime(annuel["anneemut"].astype(str) + "-07-01")   # point placé au milieu de l'année

    # Deux graphiques l'un au-dessus de l'autre, avec le même axe du temps
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    sns.lineplot(data=trim, x="trimestre", y="prix_m2_median", hue="type_bien", marker="o", ax=axes[0])
    axes[0].set_ylabel("Prix médian au m² (€)")
    axes[0].set_title("Prix médian au m² par trimestre")

    sns.lineplot(data=annuel, x="date", y="nb_ventes", hue="type_bien", marker="o", ax=axes[1])
    axes[1].set_ylabel("Nombre de ventes")
    axes[1].set_title("Nombre de ventes par année")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()
    # Ventes DVF+ de A : plus utilisées après C3.1, seul `trim` sert en C3.2 (libère la mémoire)
    del ventes
    import gc; gc.collect()

    # --- C3.2 Évolution des prix et taux des crédits immobiliers ---
    afficher_section('C3.2 Évolution des prix et taux des crédits immobiliers')
    # Taux mensuels -> taux moyen par trimestre (moyenne des 3 mois)
    taux = pd.read_csv(DATA_RAW / "Taux d'intérêt annuel des crédits immobiliers.csv",
                       sep=";", decimal=",", skiprows=6, header=None,
                       names=["date", "taux"], parse_dates=["date"], encoding="utf-8-sig")
    taux["trimestre"] = taux["date"].dt.to_period("Q").dt.to_timestamp()
    taux_trim = taux.groupby("trimestre")["taux"].mean()

    # Prix médian par trimestre (calculé en C3.1), une colonne par type de bien
    prix_trim = trim.pivot(index="trimestre", columns="type_bien", values="prix_m2_median")

    # Une ligne par trimestre, de 2014 à 2024
    tp = prix_trim.join(taux_trim).loc["2014":"2024"]
    print(len(tp), "trimestres")

    # Prix à gauche, taux à droite (second axe, car les unités sont différentes)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in prix_trim.columns:
        ax.plot(tp.index, tp[col], marker="o", label=f"prix médian au m² ({col})")
    ax.set_ylabel("Prix médian au m² (€)")
    ax2 = ax.twinx()
    ax2.plot(tp.index, tp["taux"], color="black", linestyle="--", label="taux moyen des crédits")
    ax2.set_ylabel("Taux (%)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.93))
    ax.set_title("Prix médian au m² et taux des crédits immobiliers, par trimestre")
    plt.tight_layout()
    plt.show()
    # Évolution du prix sur un an : on compare chaque trimestre au même trimestre de l'année
    # précédente, ce qui retire l'effet de saison
    for col in prix_trim.columns:
        tp[f"evol_{col}"] = (tp[col] / tp[col].shift(4) - 1) * 100

    # Lien entre le taux d'il y a k trimestres et l'évolution du prix sur un an
    for col in prix_trim.columns:
        print(col)
        for k in range(0, 9):
            x = tp["taux"].shift(k)            # taux k trimestres plus tôt
            ok = x.notna() & tp[f"evol_{col}"].notna()
            rho, p = stats.spearmanr(x[ok], tp.loc[ok, f"evol_{col}"])
            print(f"  retard de {k} trimestre(s) : rho = {rho:.2f} | p-value = {p:.1e} | n = {ok.sum()}")
    # Taux 2 trimestres plus tôt (le retard qui donne le lien le plus fort)
    tp["taux_ret2"] = tp["taux"].shift(2)
    d = tp.dropna(subset=["taux_ret2"] + [f"evol_{c}" for c in prix_trim.columns])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Dans le temps : évolution des prix et taux décalé, avec l'axe du taux inversé
    for col in prix_trim.columns:
        axes[0].plot(d.index, d[f"evol_{col}"], marker="o", label=f"évolution du prix sur un an ({col})")
    axes[0].axhline(0, color="grey", linestyle=":")
    axes[0].set_ylabel("Évolution du prix médian au m² sur un an (%)")
    ax2 = axes[0].twinx()
    ax2.plot(d.index, d["taux_ret2"], color="black", linestyle="--", label="taux 2 trimestres plus tôt")
    ax2.invert_yaxis()
    ax2.set_ylabel("Taux (%), axe inversé")
    axes[0].set_title("Évolution des prix et taux décalé de 2 trimestres")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, loc="lower center")

    # 2. Nuage de points : un point par trimestre, couleur = année
    for col, forme in zip(prix_trim.columns, ["o", "s"]):
        nuage = axes[1].scatter(d["taux_ret2"], d[f"evol_{col}"], c=d.index.year,
                                cmap="viridis", marker=forme, label=col)
    axes[1].axhline(0, color="grey", linestyle=":")
    axes[1].set_xlabel("Taux 2 trimestres plus tôt (%)")
    axes[1].set_ylabel("Évolution du prix sur un an (%)")
    axes[1].set_title("Un point par trimestre")
    axes[1].legend()
    plt.colorbar(nuage, ax=axes[1], label="Année")

    plt.tight_layout()
    plt.show()

    # --- C3.3 Prix au m², niveau de vie et taux des crédits ---
    afficher_section('C3.3 Prix au m², niveau de vie et taux des crédits')
    # C3.3 a besoin de la table cible jointe au niveau de vie (même construction qu'en C2.0).
    # On la reconstruit ici pour pouvoir lancer C3.3 sans exécuter B1 et C2.
    # Si la construction change en C2.0, la changer aussi ici.
    cible_2021 = pd.read_parquet(DATA_OUT / "cible_commune_annee.parquet")
    cible_2021 = cible_2021[cible_2021["annee"] <= 2021]          # Filosofi s'arrête en 2021

    filo_c33 = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet",
                               columns=["code_commune", "annee", "niveau_vie_median"])
    filo_c33 = filo_c33[~filo_c33["code_commune"].isin(["75056", "69123", "13055"])]   # comme en B1

    # Jointure à gauche : on garde toutes les communes x années de la cible
    fusion = cible_2021.merge(filo_c33, on=["code_commune", "annee"], how="left")

    # "revenu trouvé" : la commune a un niveau de vie médian publié cette année-là
    fusion["cas"] = np.where(fusion["niveau_vie_median"].notna(), "revenu trouvé", "pas de revenu")
    print(fusion.shape, "|", (fusion["cas"] == "revenu trouvé").mean().round(3), "des lignes avec un revenu")

    # Taux de l'année N : moyenne de juillet N-1 à juin N (retard de 2 trimestres, vu en C3.2)
    taux["annee"] = (taux["date"] + pd.DateOffset(months=6)).dt.year
    taux_an = taux.groupby("annee")["taux"].mean().rename("taux")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (t, nom) in zip(axes, [("maison", "Maisons"), ("appart", "Appartements")]):
        d = fusion[(fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]
        d = d[["code_commune", "annee", f"prix_m2_med_{t}", "niveau_vie_median"]].copy()

        # Évolution du prix de chaque commune par rapport à l'année précédente (%)
        prec = d[["code_commune", "annee", f"prix_m2_med_{t}"]].rename(columns={f"prix_m2_med_{t}": "prix_prec"})
        prec["annee"] += 1
        d = d.merge(prec, on=["code_commune", "annee"])
        d["evol_prix"] = (d[f"prix_m2_med_{t}"] / d["prix_prec"] - 1) * 100

        # Trois groupes de même taille selon le niveau de vie, recalculés chaque année
        rang = d.groupby("annee")["niveau_vie_median"].rank(pct=True)
        d["groupe_nv"] = pd.cut(rang, [0, 1/3, 2/3, 1], labels=["modestes", "intermédiaires", "aisées"])

        # Un point par année et par groupe : évolution médiane du prix
        g = (d.groupby(["annee", "groupe_nv"], observed=True)["evol_prix"].median()
              .reset_index().merge(taux_an, left_on="annee", right_index=True))

        sns.lineplot(data=g, x="taux", y="evol_prix", hue="groupe_nv", marker="o", sort=False, ax=ax)
        ax.axhline(0, color="grey", linestyle=":")
        ax.set_xlabel("Taux moyen de juillet de l'année précédente à juin (%)")
        ax.set_title(f"{nom} : évolution médiane du prix selon le taux")
    axes[0].set_ylabel("Évolution du prix sur un an (%)")
    plt.tight_layout()
    plt.show()
    for t, nom in [("maison", "Maisons"), ("appart", "Appartements")]:
        # Même préparation que pour le graphique : évolution du prix de chaque commune sur un an
        d = fusion[(fusion["cas"] == "revenu trouvé") & (fusion[f"n_{t}"] >= 10)]
        d = d[["code_commune", "annee", f"prix_m2_med_{t}", "niveau_vie_median"]].copy()
        prec = d[["code_commune", "annee", f"prix_m2_med_{t}"]].rename(columns={f"prix_m2_med_{t}": "prix_prec"})
        prec["annee"] += 1
        d = d.merge(prec, on=["code_commune", "annee"])
        d["evol_prix"] = (d[f"prix_m2_med_{t}"] / d["prix_prec"] - 1) * 100
        d = d.merge(taux_an, left_on="annee", right_index=True)
        rang = d.groupby("annee")["niveau_vie_median"].rank(pct=True)
        d["groupe_nv"] = pd.cut(rang, [0, 1/3, 2/3, 1], labels=["modestes", "intermédiaires", "aisées"])

        print(f"{nom} (n = {len(d)} lignes commune × année)")
        rho, p = stats.spearmanr(d["taux"], d["evol_prix"])
        print(f"  1. taux / évolution du prix          : rho = {rho:.2f} | p-value = {p:.1e}")
        rho, p = stats.spearmanr(d["niveau_vie_median"], d["evol_prix"])
        print(f"  2. niveau de vie / évolution du prix : rho = {rho:.2f} | p-value = {p:.1e}")
        for groupe, sous in d.groupby("groupe_nv", observed=True):
            rho, p = stats.spearmanr(sous["taux"], sous["evol_prix"])
            print(f"  3. taux / évolution, communes {groupe:14}: rho = {rho:.2f} | p-value = {p:.1e}")

    # -------------------------------------------------------------------------
    # C4. Parc de logement et vacance (recensement 2022, LOVAC)
    # -------------------------------------------------------------------------
    afficher_section('C4. Parc de logement et vacance (recensement 2022, LOVAC)')
    # Tables préparées : cible commune × année (bloc 4), logement (bloc 7), LOVAC (bloc 9)
    cible_com = pd.read_parquet(DATA_OUT / "cible_commune_annee.parquet")
    logement_com = pd.read_parquet(DATA_OUT / "logement_commune_2022.parquet")
    lovac_long = pd.read_parquet(DATA_OUT / "lovac_commune_annee.parquet")
    print("Cible :", cible_com.shape, "| Logement :", logement_com.shape, "| LOVAC :", lovac_long.shape)

    # Ajout de la base logement : une ligne par commune, même valeur pour toutes les années
    table = cible_com.merge(logement_com.drop(columns=["dep", "alsace_moselle"]),
                            on="code_commune", how="left")

    # Ajout de LOVAC : vente de l'année N -> millésime N
    colonnes_lovac = ["code_commune", "annee_vente_associee", "taux_vacance_complete",
                      "taux_vacance_2ans", "part_longue_parmi_vacants"]
    table = table.merge(lovac_long[colonnes_lovac],
                        left_on=["code_commune", "annee"], right_on=["code_commune", "annee_vente_associee"],
                        how="left").drop(columns=["annee_vente_associee"])
    print("Table fusionnée :", table.shape, "(doit avoir", len(cible_com), "lignes)")
    print("Lignes sans données logement :", table["P22_LOG"].isna().sum())

    # Séparation train / test : 2025 est réservée au test, on ne la regarde PAS pour choisir les variables
    train = table[table["annee"] <= 2024].copy()
    print("Train (2014-2024) :", train.shape)
    # Cibles : log du prix, seulement si au moins 10 ventes (médiane fiable)
    SEUIL_VENTES = 10
    train["log_prix_maison"] = np.where(train["n_maison"] >= SEUIL_VENTES, np.log(train["prix_m2_med_maison"]), np.nan)
    train["log_prix_appart"] = np.where(train["n_appart"] >= SEUIL_VENTES, np.log(train["prix_m2_med_appart"]), np.nan)

    recensement = ["taux_vacance", "part_rsec", "part_maisons", "part_apparts", "part_moins40m2",
                   "part_100m2_plus", "nb_pieces_moyen", "part_avant1946", "part_2006_2019",
                   "part_proprio", "part_locataires", "part_hlm", "part_emmenages_2ans",
                   "anciennete_moyenne", "taille_menage", "part_sans_voiture", "part_2voitures",
                   "part_parking", "part_suroccupes", "log_nb_logements"]
    variables_lovac = ["taux_vacance_complete", "taux_vacance_2ans", "part_longue_parmi_vacants"]
    candidates = recensement + variables_lovac

    # Lien avec le prix : Spearman (tout le train, puis 2024 seule)
    # 2024 seule = une ligne par commune : vérifie que la répétition des communes
    # sur plusieurs années ne change pas le résultat.
    train_2024 = train[train["annee"] == 2024]


    def spearman(df, x, y):
        ok = df[x].notna() & df[y].notna()
        return round(stats.spearmanr(df.loc[ok, x], df.loc[ok, y])[0], 2)


    lignes = []
    for col in candidates:
        ligne = {"variable": col}
        ligne["source"] = "recensement" if col in recensement else "LOVAC"
        ligne["rho_maisons"] = spearman(train, col, "log_prix_maison")
        ligne["rho_apparts"] = spearman(train, col, "log_prix_appart")
        ligne["rho_maisons_2024"] = spearman(train_2024, col, "log_prix_maison")
        ligne["rho_apparts_2024"] = spearman(train_2024, col, "log_prix_appart")
        ligne["pct_manquants"] = round(train[col].isna().mean() * 100, 1)
        lignes.append(ligne)

    tri = pd.DataFrame(lignes).set_index("variable")
    tri["force"] = tri[["rho_maisons", "rho_apparts"]].abs().max(axis=1)
    display(tri.sort_values("force", ascending=False))
    # Doublons : paires de variables corrélées à plus de 0,8
    matrice = train[candidates].corr(method="spearman")
    print("Paires en doublon (|rho| > 0,8) :")
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if abs(matrice.loc[a, b]) > 0.8:
                print("  ", a, "/", b, ":", round(matrice.loc[a, b], 2))

    # Choix (écrit à la main d'après les deux résultats ci-dessus)
    # Règle : |rho| > 0,2 pour les maisons OU les appartements, une seule variable par doublon
    indicateurs_gardes = [
        "part_avant1946", "taux_vacance", "anciennete_moyenne", "part_2006_2019",
        "part_apparts",          # à la place de part_maisons (doublon -0,96)
        "log_nb_logements", "part_parking", "taille_menage", "part_suroccupes",
        "part_rsec", "part_sans_voiture", "part_moins40m2",
    ]
    lovac_gardees = [
        "taux_vacance_complete", # complète de 2020 à 2025 grâce au taux de l'EPCI
        "taux_vacance_2ans",     # à tester : lien fort mais souvent cachée (secret statistique)
    ]                            # part_longue_parmi_vacants écartée (doublon 0,88)

    ecartees = []
    for col in candidates:
        if col not in indicateurs_gardes and col not in lovac_gardees:
            ecartees.append(col)
    print("Écartées :", ecartees)

    # -------------------------------------------------------------------------
    # C5. Délinquance enregistrée
    # -------------------------------------------------------------------------

    # --- C5.0 Cible au bassin de vie ---
    afficher_section('C5.0 Cible au bassin de vie')
    # 1. Ventes : mêmes colonnes et mêmes règles d'exclusion que la table cible (bloc 4)
    ventes_bv = pd.read_parquet(DATA_OUT / "dvf_plus_residentiel.parquet",
                                columns=["anneemut", "l_codinsee", "libtypbien", "valeurfonc", "sbati"])
    ventes_bv["prix_m2"] = ventes_bv["valeurfonc"] / ventes_bv["sbati"]
    ventes_bv = ventes_bv[ventes_bv["valeurfonc"].notna() & (ventes_bv["valeurfonc"] > 0)
                          & ventes_bv["prix_m2"].between(50, 50000)].copy()
    ventes_bv["type_bien"] = ventes_bv["libtypbien"].map({"UNE MAISON": "maison", "UN APPARTEMENT": "appart"})

    # 2. Rattachement de chaque vente à son bassin de vie
    ventes_bv = ventes_bv.merge(liaison_ini, left_on="l_codinsee", right_on="CODGEO_INI",
                                how="left", validate="many_to_one")

    # 3. Maille : même option qu'en B3 pour Paris, Lyon et Marseille
    ventes_bv["maille"] = ventes_bv["BV2022"]
    if DECOUPER_GRANDES_VILLES:
        arrondissement = ventes_bv["l_codinsee"].str.match(r"^(751|6938|132)")
        ventes_bv.loc[arrondissement, "maille"] = ventes_bv.loc[arrondissement, "l_codinsee"]

    # 4. Une ligne par maille × année, le type de bien en colonnes
    cible_bv = (ventes_bv.groupby(["maille", "anneemut", "type_bien"])["prix_m2"]
                         .agg(n="count", prix_m2_med="median")
                         .unstack("type_bien"))
    cible_bv.columns = [f"{stat}_{typ}" for stat, typ in cible_bv.columns]
    cible_bv = cible_bv.reset_index().rename(columns={"anneemut": "annee"})
    for col in ["n_maison", "n_appart"]:
        cible_bv[col] = cible_bv[col].fillna(0).astype(int)

    print(cible_bv.shape)
    print("Mailles :", cible_bv["maille"].nunique())
    print("Total des ventes :", cible_bv["n_appart"].sum() + cible_bv["n_maison"].sum())   # doit valoir 9 655 421

    # 5. 2025 est l'année de test : on la retire
    cible_bv = cible_bv[cible_bv["annee"] <= 2024]
    print("Lignes 2014-2024 :", cible_bv.shape)

    # --- C5.1 Rapprochement avec la table cible ---
    afficher_section('C5.1 Rapprochement avec la table cible')
    # Cible sur les années communes aux deux sources (pas de délinquance communale avant 2016)
    c = cible_bv[cible_bv["annee"].between(2016, 2024)].copy()
    c["n_total"] = c["n_appart"] + c["n_maison"]

    # Jointure à gauche : on garde tous les bassins × années de la cible
    fusion = c.merge(taux_fam, on=["maille", "annee"], how="left", indicator=True)

    bilan = fusion.groupby("_merge", observed=True).agg(lignes=("maille", "size"), ventes=("n_total", "sum"))
    bilan["% lignes"] = (bilan["lignes"] / bilan["lignes"].sum() * 100).round(1)
    bilan["% ventes"] = (bilan["ventes"] / bilan["ventes"].sum() * 100).round(1)
    display(bilan)

    # --- C5.2 Prix au m² selon le niveau de délinquance ---
    afficher_section('C5.2 Prix au m² selon le niveau de délinquance')
    # Prix médian au m² par bassin et par type de bien, sur les ventes de 2016-2024
    prix_p = (ventes_bv[ventes_bv["anneemut"].between(2016, 2024)]
              .groupby(["maille", "type_bien"])["prix_m2"]
              .agg(n="count", prix="median").reset_index())
    prix_p = prix_p[prix_p["n"] >= 30]   # assez de ventes pour une médiane fiable

    # Taux lissés sur la même période (2016-2024)
    taux_p = taux_periode(fam, 2016, 2024, ["maille", "famille"]).unstack("famille").reset_index()

    c1 = prix_p.merge(taux_p, on="maille", how="inner")

    # Nuages de points : maisons
    m = c1[c1["type_bien"] == "maison"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, f in zip(axes.flatten(), liste_familles):
        ax.scatter(m[f], m["prix"], s=5, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f, fontsize=11)
        ax.set_xlabel("Taux pour 1 000 (log)")
        ax.set_ylabel("Prix médian au m² (€, log)")
    fig.suptitle(f"Maisons : prix médian au m² et délinquance par bassin de vie, 2016-2024 (n = {len(m)} bassins)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Taux par indicateur lissés sur 2016-2024 (pour les sous-lignes du tableau)
    taux_ind = (delinq_bv[delinq_bv["annee"].between(2016, 2024)]
                .groupby(["maille", "indicateur"])[["nombre_estime", "denominateur"]].sum())
    taux_ind["taux"] = taux_ind["nombre_estime"] / taux_ind["denominateur"] * 1000
    taux_ind = taux_ind.reset_index().pivot(index="maille", columns="indicateur", values="taux").reset_index()
    c1_ind = prix_p.merge(taux_ind, on="maille", how="inner")

    def rho_prix(df, col):
        """rho de Spearman entre une colonne de taux et le prix, pour les maisons et les appartements."""
        res = {}
        for t, nom in [("maison", "rho maisons"), ("appart", "rho appartements")]:
            d = df[df["type_bien"] == t]
            res[nom] = round(stats.spearmanr(d[col], d["prix"], nan_policy="omit")[0], 2)
        return res

    # Indicateurs de chaque famille ; l'AFD est montré à part car il fait déjà partie de l'usage
    sous_lignes = {f: [i for i, fam_i in familles.items() if fam_i == f] for f in liste_familles}
    sous_lignes["Stupéfiants"].append("Usage de stupéfiants (AFD)")

    lignes = []
    for f in liste_familles:
        lignes.append({"": f.upper(), **rho_prix(c1, f)})                # ligne de la famille
        if len(sous_lignes[f]) > 1:                                      # sous-lignes si plusieurs indicateurs
            for ind in sous_lignes[f]:
                nom = "dont AFD (compris dans l'usage)" if "AFD" in ind else ind
                lignes.append({"": "   ↳ " + nom, **rho_prix(c1_ind, ind)})

    tableau = pd.DataFrame(lignes).set_index("")
    display(tableau.round(2))
    print("Bassins : maisons", (c1["type_bien"] == "maison").sum(), "| appartements", (c1["type_bien"] == "appart").sum())

    # --- C5.3 Évolution des prix selon le niveau de délinquance ---
    afficher_section('C5.3 Évolution des prix selon le niveau de délinquance')
    # 1. Prix médian au m² d'une période, par bassin et type de bien (recalculé à partir des ventes)
    def prix_periode(debut, fin):
        p = (ventes_bv[ventes_bv["anneemut"].between(debut, fin)]
             .groupby(["maille", "type_bien"])["prix_m2"].agg(n="count", prix="median"))
        return p[p["n"] >= 30]["prix"]

    c2 = pd.concat([prix_periode(2016, 2018), prix_periode(2021, 2023)],
                   axis=1, keys=["prix_debut", "prix_fin"]).dropna().reset_index()

    # 2. Évolution du prix de chaque bassin entre 2016-2018 et 2021-2023 (%)
    c2["evol_prix"] = (c2["prix_fin"] / c2["prix_debut"] - 1) * 100

    # 3. Évolution nationale par type de bien : médiane de toutes les ventes de chaque période
    nat = {}
    for t in ["maison", "appart"]:
        v = ventes_bv[ventes_bv["type_bien"] == t]
        nat[t] = (v.loc[v["anneemut"].between(2021, 2023), "prix_m2"].median()
                  / v.loc[v["anneemut"].between(2016, 2018), "prix_m2"].median() - 1) * 100
    print("Évolution nationale du prix au m² (%) :", {k: round(x, 1) for k, x in nat.items()})

    # 4. Évolution relative : écart du bassin à l'évolution nationale (points de %)
    c2["evol_relative"] = c2["evol_prix"] - c2["type_bien"].map(nat)

    # 5. Taux de délinquance de départ (2016-2018) par indicateur, bassins de métropole
    ind_depart = (delinq_bv[delinq_bv["annee"].between(2016, 2018) & (delinq_bv["maille"].str[:2] != "97")]
                  .groupby(["maille", "indicateur"])[["nombre_estime", "denominateur"]].sum())
    ind_depart["taux"] = ind_depart["nombre_estime"] / ind_depart["denominateur"] * 1000
    ind_depart = ind_depart.reset_index().pivot(index="maille", columns="indicateur", values="taux").reset_index()

    c2_ind = c2[["maille", "type_bien", "evol_relative"]].merge(ind_depart, on="maille", how="inner")

    # AFD exclu : les amendes forfaitaires n'existent pas en 2016-2018 (taux nul partout)
    liste_ind = [c for c in ind_depart.columns if c not in ["maille", "Usage de stupéfiants (AFD)"]]
    groupes_q = ["1 - moins\ntouchés", "2", "3", "4 - plus\ntouchés"]

    # 6. Graphiques : 4 groupes de bassins selon le taux de départ, écart médian à l'évolution nationale
    def barres_relatives_ind(type_bien, nom):
        d = c2_ind[c2_ind["type_bien"] == type_bien].copy()
        fig, axes = plt.subplots(3, 5, figsize=(26, 13), sharey=True)
        for ax, ind in zip(axes.flatten(), liste_ind):
            d["groupe"] = pd.qcut(d[ind].rank(method="first"), 4, labels=groupes_q)
            med = d.groupby("groupe", observed=True)["evol_relative"].median()
            med.plot.bar(ax=ax, color=["tab:blue" if v >= 0 else "tab:orange" for v in med])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(ind, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0, labelsize=8)
        axes.flatten()[-1].set_visible(False)   # 14 indicateurs pour 15 cases
        fig.suptitle(f"{nom} : écart à l'évolution nationale du prix (2016-18 → 2021-2023, France : {nat[type_bien]:+.1f} %) "
                     f"selon le niveau de délinquance de 2016-2018 (n = {len(d)} bassins)", fontsize=13)
        plt.tight_layout()
        plt.show()

    barres_relatives_ind("maison", "Maisons")
    # Corrélation de Spearman entre le taux de départ et l'évolution relative du prix, par indicateur
    lignes = []
    for ind in liste_ind:
        ligne = {"indicateur": ind}
        for t, nom in [("maison", "maisons"), ("appart", "appartements")]:
            d = c2_ind[c2_ind["type_bien"] == t]
            rho, p = stats.spearmanr(d[ind], d["evol_relative"], nan_policy="omit")
            ligne[f"rho {nom}"] = round(rho, 2)
            ligne[f"p-value {nom}"] = f"{p:.1e}"
        lignes.append(ligne)

    display(pd.DataFrame(lignes).set_index("indicateur").sort_values("rho maisons"))
    print("Bassins : maisons", (c2_ind["type_bien"] == "maison").sum(),
          "| appartements", (c2_ind["type_bien"] == "appart").sum())
    # Ventes rattachées aux bassins de vie : plus utilisées après C5 (libère la mémoire)
    del ventes_bv
    import gc; gc.collect()
    enregistrer_figures()

print("\nGraphiques enregistrés dans :", FIGURES)