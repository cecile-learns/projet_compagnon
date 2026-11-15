# =============================================================================
# 04_outils/produire_ventes_comparables.py : fichier des ventes pour l'outil (comparables et fourchette)
# =============================================================================
# Produit 05_application/donnees/ventes_comparables.parquet : toutes les ventes de DVF+ de la période
# (pas l'échantillon du modèle), avec les règles du bloc 2 de preparation.py (un seul logement ancien,
# une seule commune, métropole) et les règles d'exclusion du bloc 4 (valeur renseignée, prix entre
# 50 et 50 000 €/m²). Rien qui désigne un bien précis : ni adresse, ni parcelle, ni coordonnées,
# ni identifiant de mutation, ni valeur foncière.
# Colonnes : code_commune, nom_commune, sous_bassin_de_vie, nom_sous_bassin, BV2022, nom_bassin,
#            type_bien (maison / appartement), annee, mois, sbati, nb_pieces, sterr, prix_m2.
# Prérequis (voir preparation.py) : bloc 1 (liaison commune -> bassin de vie) et bloc 2 (population DVF+).
# Pour ajouter janvier-juin 2026 : même script, avec le fichier DVF+ 2026 filtré par le bloc 2 (même format),
# en changeant ANNEE_FIN et FICHIER_POP.
# Lancement, depuis la racine du projet :  python 04_outils/produire_ventes_comparables.py

from pathlib import Path

import numpy as np
import pandas as pd

import commun

ANNEE_DEBUT, ANNEE_FIN = 2023, 2025            # période des ventes comparables (3 dernières années)
FICHIER_POP = commun.DATA_OUT / "dvf_plus_residentiel.parquet"
FICHIER_LIAISON = commun.DATA_OUT / "liaison_bv_scindes.parquet"          # sous-bassin par commune
FICHIER_NOMS = commun.DATA_OUT / "liaison_commune_bv2022.parquet"         # noms des communes et des bassins
SORTIE = Path(commun.DATA_OUT).parent.parent / "05_application" / "donnees" / "ventes_comparables.parquet"

COLONNES_LUES = (["anneemut", "moismut", "l_codinsee", "libtypbien", "valeurfonc", "sbati", "sterr"]
                 + [f"nbmai{i}pp" for i in range(1, 6)] + [f"nbapt{i}pp" for i in range(1, 6)])
ARRONDISSEMENT = r"^(751\d\d|6938\d|132\d\d)$"          # Paris, Lyon, Marseille : un arrondissement = un code
# Suffixe des sous-bassins des bassins de 100 000 habitants ou plus (grille de densité, bloc 1 de preparation.py)
SECTEUR = {"1": "centres", "2": "ceintures et petites villes", "3": "espaces ruraux"}


def table_des_noms(noms):
    """Nom de chaque code commune (2026 et arrondissements) et de chaque bassin de vie."""
    communes = noms.drop_duplicates("CODGEO_2026").set_index("CODGEO_2026")["LIBGEO_2026"].to_dict()
    arr = noms[noms["CODGEO_INI"].str.match(ARRONDISSEMENT)].drop_duplicates("CODGEO_INI")
    for code, ville in zip(arr["CODGEO_INI"], arr["LIBGEO_2026"]):
        numero = int(code[-2:])
        communes[code] = f"{ville} {numero}{'er' if numero == 1 else 'e'} arrondissement"
    bassins = noms.drop_duplicates("BV2022").set_index("BV2022")["LIBBV2022"].to_dict()
    return communes, bassins


def construire(pop, liaison, noms):
    """Ventes retenues, avec territoires et noms. Renvoie le tableau et un bilan des exclusions."""
    bilan = {"ventes_population": len(pop)}

    # --- Règles d'exclusion du bloc 4 : valeur renseignée, prix entre 50 et 50 000 €/m² ---
    pop = pop.copy()
    pop["prix_m2"] = pop["valeurfonc"] / pop["sbati"]
    garde = pop["valeurfonc"].notna() & (pop["valeurfonc"] > 0) & pop["prix_m2"].between(50, 50000)
    bilan["exclues_prix"] = int((~garde).sum())
    pop = pop[garde].copy()

    # --- Type de bien et nombre de pièces (une seule colonne nb..pp vaut 1 ; 5 = 5 pièces ou plus) ---
    pop["type_bien"] = pop["libtypbien"].map({"UNE MAISON": "maison", "UN APPARTEMENT": "appartement"})
    colonnes_pieces = [f"nbmai{i}pp" for i in range(1, 6)] + [f"nbapt{i}pp" for i in range(1, 6)]
    pieces = pop[colonnes_pieces[:5]].to_numpy() + pop[colonnes_pieces[5:]].to_numpy()
    nb_pieces = pd.array(pieces.argmax(axis=1) + 1, dtype="Int8")
    # Aucune colonne renseignée : argmax donnerait 1 par défaut ; on met "inconnu" pour ne pas fausser la comparaison
    nb_pieces[pieces.sum(axis=1) == 0] = pd.NA
    pop["nb_pieces"] = nb_pieces
    bilan["pieces_inconnues"] = int(pop["nb_pieces"].isna().sum())
    pop.loc[pop["type_bien"] == "appartement", "sterr"] = 0.0          # pas de terrain pour un appartement

    # --- Rattachement : commune 2026, bassin de vie, sous-bassin ---
    pop = pop.merge(liaison[["CODGEO_INI", "CODGEO_2026", "BV2022", "sous_bassin_de_vie"]],
                    left_on="l_codinsee", right_on="CODGEO_INI", how="left", validate="many_to_one")
    arrondissement = pop["l_codinsee"].str.match(ARRONDISSEMENT)
    pop["code_commune"] = pop["CODGEO_2026"].where(~arrondissement, pop["l_codinsee"])
    sans_territoire = pop["sous_bassin_de_vie"].isna()
    bilan["exclues_sans_sous_bassin"] = int(sans_territoire.sum())
    pop = pop[~sans_territoire].copy()

    # --- Noms (pour l'affichage) ---
    communes, bassins = table_des_noms(noms)
    pop["nom_commune"] = pop["code_commune"].map(communes)
    pop["nom_bassin"] = pop["BV2022"].map(bassins)
    # Bassins homonymes (même nom, codes différents) : on ajoute le département du bassin, puis le code si besoin
    b = pop[["BV2022", "nom_bassin"]].astype(str).drop_duplicates("BV2022")
    partage = b.duplicated("nom_bassin", keep=False)
    b["nom_final"] = b["nom_bassin"]
    b.loc[partage, "nom_final"] = b["nom_bassin"] + " (dép. " + b["BV2022"].str[:2] + ")"
    encore = partage & b.duplicated("nom_final", keep=False)
    b.loc[encore, "nom_final"] = b["nom_bassin"] + " (" + b["BV2022"] + ")"
    if partage.any():
        print("Bassins homonymes renommés :", ", ".join(sorted(b.loc[partage, "nom_final"])))
    pop["nom_bassin"] = pop["BV2022"].astype(str).map(dict(zip(b["BV2022"], b["nom_final"])))
    bilan["bassins_homonymes"] = int(partage.sum())
    # Sous-bassin : bassin non scindé = nom du bassin ; bassin scindé (code « BV_1/_2/_3 ») = nom du bassin
    # + secteur ; arrondissement de Paris, Lyon ou Marseille = nom de l'arrondissement
    code = pop["sous_bassin_de_vie"].astype(str)
    secteur = code.str.extract(r"_(\d)$")[0].map(SECTEUR)
    nom = pop["nom_bassin"].astype(str)
    pop["nom_sous_bassin"] = nom.where(secteur.isna(), nom + " (" + secteur + ")")
    pop["nom_sous_bassin"] = pop["nom_sous_bassin"].where(~code.str.match(ARRONDISSEMENT), code.map(communes))
    bilan["sans_nom_sous_bassin"] = int(pop["nom_sous_bassin"].isna().sum())

    # Sous-bassin sans suffixe dont le code n'est pas celui de son bassin (hors arrondissements) : une commune
    # rattachée à un bassin sous un autre code dans la table du bloc 1 (ex. 59481 dans le bassin 59053, Bavay).
    # À signaler, pas à corriger ici : ses ventes sont comptées à son propre niveau, puis à celui du bassin.
    codes = pop[["sous_bassin_de_vie", "BV2022"]].astype(str).drop_duplicates()
    ecart = (~codes["sous_bassin_de_vie"].str.contains("_") & ~codes["sous_bassin_de_vie"].str.match(ARRONDISSEMENT)
             & (codes["sous_bassin_de_vie"] != codes["BV2022"]))
    bilan["sous_bassins_a_code_different_du_bassin"] = int(ecart.sum())
    if ecart.any():
        print("Sous-bassins dont le code diffère de celui de leur bassin (code -> bassin) :",
              ", ".join(codes.loc[ecart, "sous_bassin_de_vie"] + " -> " + codes.loc[ecart, "BV2022"]))

    # Dernier recours : un nom encore partagé par plusieurs codes reçoit son code entre crochets
    paires = pop[["sous_bassin_de_vie", "nom_sous_bassin"]].astype(str).drop_duplicates("sous_bassin_de_vie")
    partage_sbv = paires.duplicated("nom_sous_bassin", keep=False)
    paires["final"] = paires["nom_sous_bassin"].where(
        ~partage_sbv, paires["nom_sous_bassin"] + " [" + paires["sous_bassin_de_vie"] + "]")
    if partage_sbv.any():
        print("Noms de sous-bassin encore partagés, code ajouté :", ", ".join(sorted(paires.loc[partage_sbv, "final"])))
    pop["nom_sous_bassin"] = pop["sous_bassin_de_vie"].astype(str).map(dict(zip(paires["sous_bassin_de_vie"], paires["final"])))
    bilan["sans_nom_commune"] = int(pop["nom_commune"].isna().sum())

    # --- Colonnes finales, types compacts ---
    pop["annee"] = pop["anneemut"].astype("int16")
    pop["mois"] = pop["moismut"].astype("int8")
    sortie = pop[["code_commune", "nom_commune", "sous_bassin_de_vie", "nom_sous_bassin", "BV2022", "nom_bassin",
                  "type_bien", "annee", "mois", "sbati", "nb_pieces", "sterr", "prix_m2"]].copy()
    for c in ["code_commune", "nom_commune", "sous_bassin_de_vie", "nom_sous_bassin", "BV2022", "nom_bassin", "type_bien"]:
        sortie[c] = sortie[c].astype("category")
    sortie["sbati"] = sortie["sbati"].round(0).astype("float32")
    sortie["sterr"] = sortie["sterr"].round(0).astype("float32")
    sortie["prix_m2"] = sortie["prix_m2"].round(0).astype("float32")
    sortie = sortie.sort_values(["sous_bassin_de_vie", "type_bien", "annee", "mois"]).reset_index(drop=True)
    bilan["ventes_retenues"] = len(sortie)
    return sortie, bilan


def controles(ventes):
    """Arrêt si le fichier ne respecte pas le contrat : pas de donnée sensible, bornes, période."""
    interdites = {"idmutation", "valeurfonc", "geompar_x", "geompar_y", "l_codinsee", "adresse", "idpar"}
    assert not interdites & set(ventes.columns), "colonne qui désigne un bien précis"
    assert ventes["prix_m2"].between(50, 50000).all(), "prix au m² hors de 50-50 000"
    assert ventes["annee"].between(ANNEE_DEBUT, ANNEE_FIN).all(), "vente hors période"
    assert set(ventes["type_bien"].unique()) <= {"maison", "appartement"}, "type de bien inconnu"
    assert (ventes.loc[ventes["type_bien"] == "appartement", "sterr"] == 0).all(), "terrain d'appartement non nul"
    assert ventes["code_commune"].astype(str).str.len().eq(5).all(), "code commune pas sur 5 caractères (texte)"
    assert ventes["sous_bassin_de_vie"].notna().all(), "vente sans sous-bassin"
    assert ventes["nom_sous_bassin"].notna().all(), "vente sans nom de sous-bassin"
    assert (ventes.groupby("nom_sous_bassin", observed=True)["sous_bassin_de_vie"].nunique() == 1).all(), \
        "un même nom de sous-bassin pour plusieurs codes"


if __name__ == "__main__":
    for f in (FICHIER_POP, FICHIER_LIAISON, FICHIER_NOMS):
        if not Path(f).exists():
            raise SystemExit(f"{Path(f).name} introuvable : lancer d'abord les blocs 1 et 2 de 01_preparation/preparation.py")
    if SORTIE.exists():
        raise SystemExit(f"{SORTIE} existe déjà : supprime-le ou renomme-le avant de relancer (pas d'écrasement).")

    pop = pd.read_parquet(FICHIER_POP, columns=COLONNES_LUES,
                          filters=[("anneemut", ">=", ANNEE_DEBUT), ("anneemut", "<=", ANNEE_FIN)])
    liaison = pd.read_parquet(FICHIER_LIAISON)
    noms = pd.read_parquet(FICHIER_NOMS, columns=["CODGEO_INI", "CODGEO_2026", "LIBGEO_2026", "BV2022", "LIBBV2022"])
    ventes, bilan = construire(pop, liaison, noms)
    controles(ventes)

    print(f"Ventes {ANNEE_DEBUT}-{ANNEE_FIN}")
    for cle, valeur in bilan.items():
        print(f"  {cle:<30} {valeur:>10,}".replace(",", " "))
    print("\nPar année et type de bien :")
    print(ventes.groupby(["annee", "type_bien"], observed=True).size().unstack().to_string())
    print("\nSous-bassins :", ventes["sous_bassin_de_vie"].nunique(),
          "| avec au moins 30 ventes (sur la période) :",
          int((ventes.groupby("sous_bassin_de_vie", observed=True).size() >= 30).sum()))

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    ventes.to_parquet(SORTIE, index=False, compression="zstd")
    print(f"\nÉcrit : {SORTIE} ({SORTIE.stat().st_size / 1e6:.1f} Mo)")