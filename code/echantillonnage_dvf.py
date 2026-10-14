# # Echantillonnage de la base de données DVF+
# 
# **Projet :** Compagnon Immobilier · **Auteur :** Cécile NAUDET · **Relecteur :** _ · **Date :** 12/10/2026
# 

# ### Ce que l'audit a montré
# 
# L'audit a porté sur un échantillon de 100 000 mutations, tiré proportionnellement au poids de chaque département dans les 97 fichiers bruts (16,5 millions de mutations, 2014-2025, graine 42).
# 
# - **Une base hétérogène** : 92 % de ventes, 6,5 % de VEFA, le reste en échanges, adjudications et ventes de terrains à bâtir. Côté biens, « une maison » représente 34 % des mutations et « un appartement » 28 %. Le reste regroupe des terrains, des locaux d'activité et des mutations portant sur plusieurs biens.
# - **Une valeur foncière par mutation, pas par bien.** Le prix ne se rapporte à un bien identifiable que si la mutation porte sur un seul logement.
# - **Une qualité satisfaisante sur le résidentiel** (61 834 mutations) : aucun doublon, valeur foncière manquante dans 0,03 % des cas, surfaces bâties cohérentes.
# 
# L'audit a conduit à trois décisions :
# - **Le périmètre** : ventes d'une seule maison ou d'un seul appartement, avec d'éventuelles dépendances, sur une seule commune, VEFA exclues.
# - **La couverture** : 2014-2025, avec 2025 réservée au test ; France métropolitaine hors DOM (l'Alsace-Moselle est absente de DVF+).
# - **Les variables** : 11 variables vides, constantes, redondantes ou inutiles pour le modèle ne sont pas conservées dans l'échantillon de travail.
# 
# ### Pourquoi refaire un échantillon
# 
# - **L'échantillon d'audit a été tiré avant le filtrage.** Une fois réduit au résidentiel, il laisse trop peu de ventes par département et par année : la Lozère n'avait que 150 mutations au total, tous types confondus. C'est insuffisant pour étudier les prix par territoire et dans le temps.
# - **La part du résidentiel varie fortement selon les départements.** Dans les départements ruraux, une grande partie des mutations porte sur des terrains. Tirer avant de filtrer surreprésenterait donc ces départements dans l'échantillon. Il faut filtrer d'abord, puis tirer. 
# - **L'échantillon d'audit ne respecte pas le périmètre retenu** : il contient les VEFA, les DOM et les mutations portant sur plusieurs communes.

# ### Préparer les chemins et les liste de fichiers
# Imports
from pathlib import Path
import numpy as np
import pandas as pd

# Chemins : fichiers sources et dossier de sortie de la population filtrée
dossier = Path("/home/naude/formation/projet_compagnon/data/DVF+")
dossier_pop = dossier.parent / "dvf_plus_residentiel"

# Liste des fichiers départementaux, sans les DOM (hors périmètre)
DOM = {"dvf_plus_d971", "dvf_plus_d972", "dvf_plus_d973", "dvf_plus_d974"}
fichiers = [f for f in sorted(dossier.glob("dvf_plus_d*.csv")) if f.stem not in DOM]

# Contrôle : 93 fichiers attendus (97 - 4 DOM)
print("Nombre de fichiers retenus :", len(fichiers))
print("Premier :", fichiers[0].name, "| Dernier :", fichiers[-1].name)

# ### Ecrire les règles de filtrage

def filtre_perimetre(df):
    """Garde les ventes d'un seul logement ancien, sur une seule commune."""
    # Vente classique uniquement (la VEFA a son propre libellé)
    nature_ok = df["libnatmut"] == "Vente"
    # Une seule commune, pour que l_codinsee contienne un code unique
    une_commune = df["nbcomm"] == 1
    # Une seule maison, sans appartement ni local d'activité (dépendances acceptées)
    maison_ok = (
        (df["libtypbien"] == "UNE MAISON")
        & (df["nblocmai"] == 1)
        & (df["nblocapt"] == 0)
        & (df["nblocact"] == 0)
    )
    # Un seul appartement, sans maison ni local d'activité (dépendances acceptées)
    appartement_ok = (
        (df["libtypbien"] == "UN APPARTEMENT")
        & (df["nblocapt"] == 1)
        & (df["nblocmai"] == 0)
        & (df["nblocact"] == 0)
    )
    return df.loc[nature_ok & une_commune & (maison_ok | appartement_ok)]

# ### Tester la fonction sur l'audit d'échantillon

# Test de la fonction sur l'échantillon d'audit (100 000 lignes, déjà connu)
audit = pd.read_csv(dossier.parent / "dvf_plus_echantillon_100k.csv", sep="|",
                    dtype={"coddep": str, "l_codinsee": str}, low_memory=False)

test = filtre_perimetre(audit)

# Contrôles attendus : moins de 61 834 lignes, uniquement « Vente »,
# uniquement les deux types de logement, nbcomm maximum = 1
print("Lignes retenues :", len(test))
print(test["libnatmut"].value_counts())
print(test["libtypbien"].value_counts())
print("nbcomm max :", test["nbcomm"].max())

# ### Traiter un fichier test (departement 48)

# Essai complet sur un seul département, le plus petit (Lozère)
fichier = dossier / "dvf_plus_d48.csv"
types_texte = {"coddep": str, "l_codinsee": str}  # codes géographiques lus en texte

# Lecture par blocs, filtrage de chaque bloc, puis assemblage
morceaux = []
n_brut = 0
for bloc in pd.read_csv(fichier, sep="|", chunksize=200_000,
                        dtype=types_texte, low_memory=False):
    n_brut += len(bloc)
    morceaux.append(filtre_perimetre(bloc))
pop = pd.concat(morceaux, ignore_index=True)

# Écriture en Parquet
dossier_pop.mkdir(exist_ok=True)
sortie = dossier_pop / f"{fichier.stem}.parquet"
pop.to_parquet(sortie, index=False)

# Relecture sans préciser les types, pour vérifier qu'ils sont conservés
relu = pd.read_parquet(sortie)
print("Lignes brutes :", n_brut, "| retenues :", len(pop))
print("Taille CSV     :", round(fichier.stat().st_size / 1e6, 1), "Mo")
print("Taille Parquet :", round(sortie.stat().st_size / 1e6, 1), "Mo")
print(relu[["coddep", "l_codinsee"]].dtypes)
print(relu["l_codinsee"].head(3))

# ### Traiter l'ensemble des départements (hors DOM)

import time

types_texte = {"coddep": str, "l_codinsee": str}  # codes géographiques lus en texte
bilan = []            # une ligne par département : lignes brutes et retenues
debut = time.time()   # chronomètre

# Boucle sur les 93 départements
for fichier in fichiers:
    morceaux = []
    n_brut = 0
    # Lecture par blocs pour limiter la mémoire utilisée
    for bloc in pd.read_csv(fichier, sep="|", chunksize=200_000,
                            dtype=types_texte, low_memory=False):
        n_brut += len(bloc)
        morceaux.append(filtre_perimetre(bloc))
    pop = pd.concat(morceaux, ignore_index=True)
    # Un fichier Parquet par département
    pop.to_parquet(dossier_pop / f"{fichier.stem}.parquet", index=False)
    bilan.append({"fichier": fichier.stem, "brut": n_brut, "retenu": len(pop)})
    print(f"{fichier.stem} : {n_brut} -> {len(pop)}")

# Tableau récapitulatif, conservé pour la note de synthèse
bilan = pd.DataFrame(bilan)
bilan["part_retenue_%"] = (bilan["retenu"] / bilan["brut"] * 100).round(1)
bilan.to_csv(dossier.parent / "bilan_filtrage.csv", index=False)

print("\nDurée :", round((time.time() - debut) / 60, 1), "minutes")
print(bilan[["brut", "retenu"]].sum())


# ### Contrôle de la population obtenue

# Taille totale des fichiers Parquet produits
parquets = list(dossier_pop.glob("*.parquet"))
taille_go = sum(p.stat().st_size for p in parquets) / 1e9
print("Fichiers Parquet :", len(parquets))
print("Taille totale :", round(taille_go, 2), "Go")

# Zéro devant les codes : contrôle sur l'Ain (01), en ne lisant que 2 colonnes
d01 = pd.read_parquet(dossier_pop / "dvf_plus_d01.parquet",
                      columns=["coddep", "l_codinsee"])
print(d01.head(3))

# Départements où la part du résidentiel est la plus faible et la plus forte
print(bilan.sort_values("part_retenue_%").head(5))
print(bilan.sort_values("part_retenue_%").tail(5))

# ### Résultat du filtrage
# 
# - 93 départements traités en 2,6 minutes.
# - 16 375 738 mutations brutes → **9 680 495 ventes retenues (59,1 %)**, enregistrées en 93 fichiers Parquet (1,4 Go).
# - Les codes géographiques restent en texte : le zéro initial est conservé (`01`, `01378`).
# - La part du résidentiel varie du simple au double selon les départements : 35,8 % en Lozère, 71,9 % à Paris. C'est ce qui justifie de filtrer avant de tirer l'échantillon.

# ### Compter les ventes par strate

# Nombre de ventes par strate (département × année), sur toute la population
comptes = []
for p in sorted(parquets):
    # On ne lit que les 2 colonnes qui définissent la strate
    d = pd.read_parquet(p, columns=["coddep", "anneemut"])
    # value_counts compte chaque combinaison (département, année)
    comptes.append(d.value_counts().rename("N_h").reset_index())
strates = pd.concat(comptes, ignore_index=True)

print("Nombre de strates :", len(strates))
print(strates.groupby("anneemut")["N_h"].sum())
print(strates.sort_values("N_h").head(5))

# ### Calculer combien tirer dans chaque strate

# Taux de tirage et plancher par strate
taux, m = 0.10, 300

app = strates[strates["anneemut"] <= 2023].copy()
N = app["N_h"].sum()

# Part proportionnelle (10 %), relevée au plancher, plafonnée à la taille de la strate
app["n_h"] = np.minimum(
    app["N_h"],
    np.maximum(m, np.round(app["N_h"] * taux)),
).astype(int)

app["w_h"] = app["N_h"] / app["n_h"]

print("Population 2014-2023 :", N)
print("Taille de l'échantillon :", app["n_h"].sum())
print("Strates relevées par le plancher :", (app["n_h"] > np.round(app["N_h"] * taux)).sum())
print(app.sort_values("w_h").head(3))
print(app.sort_values("w_h").tail(3))

# ### Plan de tirage
# 
# - Population d'apprentissage (2014-2023) : 8 181 968 ventes, réparties en 930 strates (93 départements × 10 années).
# - Tirage d'**une vente sur dix** dans chaque strate (taux de 10 %), avec un plancher de 300 ventes par strate.
# - 188 strates (20 %) sont relevées par le plancher. Ce sont celles de moins de 3 000 ventes, dans les départements peu peuplés.
# - Taille de l'échantillon : **835 288 ventes**, dont 17 091 (2,0 %) ajoutées par le plancher.
# - Poids `w_h` compris entre 1,83 (Lozère 2014) et 10,02. **L'échantillon n'est pas proportionnel : toute statistique ou tout modèle portant sur plusieurs départements ou plusieurs années doit utiliser `w_h`.**

# ### Tirer l'échantillon


# Générateur de nombres aléatoires, graine fixée pour la reproductibilité
rng = np.random.default_rng(42)
morceaux = []

for p in sorted(dossier_pop.glob("*.parquet")):
    d = pd.read_parquet(p)
    # Années d'apprentissage uniquement
    d = d[d["anneemut"] <= 2023]
    # On ajoute à chaque vente le n_h et le w_h de sa strate
    d = d.merge(app[["coddep", "anneemut", "n_h", "w_h"]], on=["coddep", "anneemut"])
    # Ticket de loterie : un nombre au hasard par vente
    d["alea"] = rng.random(len(d))
    # Rang du ticket dans sa strate (1 = plus petit numéro)
    d["rang"] = d.groupby(["coddep", "anneemut"])["alea"].rank(method="first")
    # On garde les n_h premiers rangs de chaque strate
    morceaux.append(d[d["rang"] <= d["n_h"]].drop(columns=["alea", "rang", "n_h"]))

ech = pd.concat(morceaux, ignore_index=True)

# Contrôles : taille totale et respect de n_h dans chaque strate
print("Taille de l'échantillon :", len(ech))
verif = ech.groupby(["coddep", "anneemut"]).size().rename("tire").reset_index()
verif = verif.merge(app[["coddep", "anneemut", "n_h"]], on=["coddep", "anneemut"])
print("Strates où le nombre tiré diffère de n_h :", (verif["tire"] != verif["n_h"]).sum())
print("idmutation uniques :", ech["idmutation"].is_unique)


# ### Enregistrer l'échantillon et la liste des identifiants

# Chemins de sortie, dans data/
chemin_ech = dossier.parent / "dvf_plus_echantillon_apprentissage.parquet"
chemin_ids = dossier.parent / "ids_echantillon_apprentissage.csv"

# Échantillon complet (avec w_h) et liste des identifiants tirés
ech.to_parquet(chemin_ech, index=False)
ech[["idmutation"]].to_csv(chemin_ids, index=False)

# Taille des deux fichiers
print("Échantillon :", round(chemin_ech.stat().st_size / 1e6, 1), "Mo")
print("Identifiants :", round(chemin_ids.stat().st_size / 1e6, 1), "Mo")

# ### Choix du format d'enregistrement de l'échantillon
# 
# L'échantillon complet pèse 124 Mo, ce qui le rend lourd à partager et dépasse la limite de 100 Mo par fichier de GitHub. On teste deux leviers pour l'alléger, sans modifier le tirage :
# - retirer les 11 colonnes non conservées à l'issue de l'audit (vides, constantes, redondantes ou inutiles pour le modèle) ;
# - utiliser une compression plus forte (`zstd`) que celle par défaut (`snappy`).

# Colonnes vides, constantes, ou listes inutiles pour le modèle
a_retirer = ["idmutinvar", "idopendata", "codservch", "refdoc", "nbartcgi", "l_artcgi",
             "l_section", "l_idpar", "l_idparmut", "l_idlocmut", "l_dcnt"]
ech_leger = ech.drop(columns=a_retirer)

# Trois versions de test
essais = {
    "toutes colonnes, zstd": (ech, "zstd"),
    "colonnes allégées, snappy": (ech_leger, "snappy"),
    "colonnes allégées, zstd": (ech_leger, "zstd"),
}
for nom, (tableau, compression) in essais.items():
    chemin = dossier.parent / "test_taille.parquet"
    tableau.to_parquet(chemin, index=False, compression=compression)
    print(f"{nom} : {round(chemin.stat().st_size / 1e6, 1)} Mo")
    chemin.unlink()   # supprime le fichier de test

# **Résultat :**
# 
# Le retrait des colonnes est le levier principal (−73 %) : les listes de texte et les identifiants de 32 caractères représentaient l'essentiel du poids. 
# La version allégée avec la compression par défaut (snappy) est retenue : 33 Mo suffisent pour le partage, et zstd n'apporterait que 7 Mo de moins.

# Échantillon allégé : colonnes non conservées retirées (elles restent dans la population)
ech_leger = ech.drop(columns=a_retirer)

# On remplace le fichier de 124 Mo par la version allégée (compression par défaut)
ech_leger.to_parquet(chemin_ech, index=False)

print("Dimensions :", ech_leger.shape)
print("Taille :", round(chemin_ech.stat().st_size / 1e6, 1), "Mo")

# ### Échantillon enregistré
# 
# - `data/dvf_plus_echantillon_apprentissage.parquet` : 835 288 ventes, 53 colonnes, 33 Mo. À partager par le drive (hors GitHub).
# - `data/ids_echantillon_apprentissage.csv` : liste des `idmutation` tirés (6,9 Mo), versionnée sur GitHub.
# - Colonnes non conservées à l'issue de l'audit : `idmutinvar`, `idopendata`, `codservch`, `refdoc`, `nbartcgi`, `l_artcgi`, `l_section`, `l_idpar`, `l_idparmut`, `l_idlocmut`, `l_dcnt`. Elles restent disponibles dans la population (`data/dvf_plus_residentiel/`) et peuvent être rattachées par `idmutation`.
# - Clés de jointure conservées : `idmutation` (identifiant), `anneemut` et `moismut` (taux mensuels), `l_codinsee` (commune), `geompar_x` et `geompar_y` (rattachement éventuel à l'IRIS).
# - **L'échantillon n'est pas proportionnel** : toute statistique ou tout modèle portant sur plusieurs départements ou plusieurs années doit utiliser la colonne `w_h`.

# ### Vérifier que l'échantillon pondérée est représentatif


def mediane_ponderee(valeurs, poids):
    """Médiane où chaque valeur compte autant que son poids."""
    v = np.asarray(valeurs)
    w = np.asarray(poids)
    ordre = np.argsort(v)                 # rang des valeurs, de la plus petite à la plus grande
    v, w = v[ordre], w[ordre]
    cumul = np.cumsum(w)                  # poids additionnés au fur et à mesure
    return v[np.searchsorted(cumul, cumul[-1] / 2)]   # valeur où l'on atteint la moitié du poids total

cols = ["anneemut", "libtypbien", "valeurfonc", "sbati"]

# Population complète 2014-2023, en ne lisant que les colonnes utiles
popu = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(dossier_pop.glob("*.parquet"))],
                 ignore_index=True)
popu = popu[popu["anneemut"] <= 2023]

# Échantillon relu depuis le fichier enregistré (vérifie au passage le fichier lui-même)
e = pd.read_parquet(chemin_ech, columns=cols + ["w_h"])

# Prix au m², en écartant les valeurs foncières manquantes
popu = popu.dropna(subset=["valeurfonc"])
popu["prix_m2"] = popu["valeurfonc"] / popu["sbati"]
e = e.dropna(subset=["valeurfonc"])
e["prix_m2"] = e["valeurfonc"] / e["sbati"]

# Comparaison des trois versions
comparaison = pd.DataFrame({
    "population": [
        (popu["libtypbien"] == "UNE MAISON").mean() * 100,
        popu["prix_m2"].median(),
    ],
    "échantillon sans poids": [
        (e["libtypbien"] == "UNE MAISON").mean() * 100,
        e["prix_m2"].median(),
    ],
    "échantillon avec poids": [
        np.average(e["libtypbien"] == "UNE MAISON", weights=e["w_h"]) * 100,
        mediane_ponderee(e["prix_m2"], e["w_h"]),
    ],
}, index=["part de maisons (%)", "prix médian au m² (€)"]).round(1)

print(comparaison)

# L'échantillon pondéré reproduit la population : part de maisons identique, prix médian à 0,1 % près. Sans poids, l'échantillon contient plus de maisons et sous-estime le prix médian de 1,2 %, à cause de la surreprésentation des départements peu peuplés. 
# Le choix est fait d'adopter les poids `w_h` pour toute statistique nationale.

# ## Conclusion
# 
# ### Ce que nous avons fait
# 
# 1. **Filtrer les données.** Dans les 93 fichiers de France métropolitaine, nous avons gardé uniquement les ventes d'une seule maison ou d'un seul appartement. Résultat : 9 680 495 ventes sur 16 375 738 mutations (59 %).
# 2. **Tirer un échantillon.** Sur les années 2014 à 2023, nous avons tiré au hasard une vente sur dix dans chaque département et chaque année, avec au moins 300 ventes à chaque fois. Résultat : 835 288 ventes.
# 3. **Alléger le fichier.** Nous avons retiré les 11 colonnes inutiles repérées pendant l'audit. Le fichier passe de 124 Mo à 33 Mo.
# 
# ### Pourquoi
# 
# - L'échantillon de l'audit contenait trop peu de ventes par département et par année.
# - La part des ventes de logements varie beaucoup selon les départements : 36 % en Lozère, 72 % à Paris. Il fallait donc filtrer avant de tirer.
# - Le minimum de 300 ventes permet d'avoir assez de données dans les petits départements.
# 
# ### Résultat
# 
# Grâce aux poids (colonne `w_h`), l'échantillon donne les mêmes résultats que l'ensemble des ventes : 57,3 % de maisons et un prix médian d'environ 2 250 €/m².
# 
# ### À retenir pour la suite
# 
# - Toujours utiliser la colonne `w_h` pour les calculs sur toute la France.
# - Calculer les prix par commune sur l'ensemble des ventes, pas sur l'échantillon.
# - Les échantillons 2024 et 2025 restent à tirer.


