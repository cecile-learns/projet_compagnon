## code ayant permis les différentes analyses des fichiers

## FICHIER ANNONCES DE VENTES DE BIEN DANS LE 68

import pandas as pd
import numpy as np

ventes = pd.read_csv("data/ech_annonces_ventes_68.csv", sep=";", encoding="utf-8-sig", index_col='idannonce')
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


## ANALYSE FICHIER DES ANNONCES DE LOCATIONS DE BIEN DANS LE 68

location = pd.read_csv("data/ech_annonces_locations_68.csv", sep=";", encoding="utf-8-sig",index_col="idannonce")
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

## EDA FICHIERS DES ANNONCES DE VENTES ET LOCATIONS DE BIEN DANS LE 68

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")          # fond blanc avec une grille légère
couleurs = {"Appartement": "#2a78d6", "Maison": "#eb6834"}   # bleu / orange



# ── Préparation : on retire les valeurs aberrantes repérées dans les premieres inspections ─────
v = ventes[(ventes["surface"] >= 9)
           & (ventes["prix_bien"] >= 20000)
           & (ventes["prix_m2_vente"] >= 500)
           & (ventes["prix_m2_vente"] <= 8000)
           & (ventes["typedebien_lite"] != "l")
           & (ventes["typedetransaction"] == "v")].copy()

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

# prix au m2 selon le type de bien et le neuf / ancien
plt.figure(figsize=(6, 3))  
sns.histplot(v["prix_bien"]/1000, bins=60) # on divise les valeurs par 1000 pour afficher en K€
plt.title("Distribution du prix des annonces (K€)")

plt.xlim(0, 1000)


plt.xlabel("Prix (K€)")   


plt.show()

# Validation : si la moyenne dépasse nettement la médiane, la distribution est asymétrique
print("Moyenne :", round(v["prix_bien"].mean()), "€ | Médiane :", v["prix_bien"].median(), "€")

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

# prix en fonction de la surface
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

#prix selon l'étiquette DPE
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

# évolution du prix au M2 dans le temps 
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



## ANALYSE FICHIER DVF+
#Effectis et quotas proportionnels pour les fichiers DVF+ (sans charger les CSV en mémoire)
from pathlib import Path

dossier = Path("/home/naude/formation/projet_compagnon/data/DVF+")
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

sortie = dossier.parent / "dvf_plus_echantillon_100k.csv"
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

df=pd.read_csv('/home/naude/formation/projet_compagnon/data/dvf_plus_echantillon_100k.csv', sep="|", low_memory=False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(df.shape)
print(df.info())
display(df.head())

#Vérification de l'unicité des identifiants de mutation
df["idmutation"].nunique()

# Bornes temporelles observées
print("Première date :", df["datemut"].min())
print("Dernière date :", df["datemut"].max())

# Répartition des mutations par année
print("\nMutations par année :")
print(df["anneemut"].value_counts().sort_index())

#Vérification des variables de type str selon indications de la documentation DVF+
for col in ['coddep', 'libnatmut', 'libtypbien', 'departement_source']:
    print(f"\nVariable : {col}")
    print(df[col].value_counts())

df['vefa'].value_counts()

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

df_num_quanti.describe().T

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

# Nombre de lignes strictement identiques
df_residentiel.duplicated().sum()

# Nombre de répétitions de l'identifiant
df_residentiel.duplicated(subset=["idmutation"]).sum()

print("Doublons exacts :", df_residentiel.duplicated().sum())
print("Doublons idmutation :", df_residentiel.duplicated(subset=["idmutation"]).sum())

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

pd.crosstab(
df_residentiel["libtypbien"],
df_residentiel["sterr"] == 0,
normalize="index"
).mul(100).round(2)

df_residentiel.groupby("libtypbien")["sterr"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
).round(2)

