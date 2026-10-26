# =============================================================================
# 00 - TÉLÉCHARGEMENT DES DONNÉES BRUTES - projet Compagnon Immobilier
# =============================================================================
# Télécharge les sources brutes depuis le Google Drive du projet
# (dossier "compagnon_immo/raw", lisible par toute personne ayant le lien)
# et les range dans data/raw/.
#
# Utilisation, depuis la racine du projet :
#   python 00_telechargement/telechargement.py              -> affiche la liste des sources
#   python 00_telechargement/telechargement.py dvf taux     -> télécharge seulement ces sources
#   python 00_telechargement/telechargement.py tout         -> télécharge toutes les sources
#
# Un fichier déjà complet dans data/raw/ n'est pas re-téléchargé.
# Si un téléchargement échoue (Google Drive limite parfois les gros téléchargements),
# il suffit de relancer la même commande plus tard : seuls les fichiers manquants
# sont téléchargés.

import sys
from pathlib import Path

import gdown

# Permet d'importer config.py, qui se trouve dans le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_OUT


# -----------------------------------------------------------------------------
# 1. Catalogue des sources
# -----------------------------------------------------------------------------
# Pour chaque source : un nom court (utilisé en ligne de commande),
# le type d'objet sur le Drive (fichier ou dossier), son identifiant Drive
# (la suite de caractères dans le lien de partage) et le nom local.
# Par défaut, une source est rangée dans data/raw/ ; "dossier" permet de la
# ranger ailleurs (data/processed/ pour un fichier déjà préparé).

SOURCES = {
    "dvf": {
        "type": "dossier",
        "id": "11hNvTHWgIc6PTI0t3dgl8ResdTw7QQJK",
        "nom": "DVF+",
    },
    # DVF+ déjà filtré par le bloc 2 de 01_preparation (parcours rapide)
    "dvf_residentiel": {
        "type": "fichier",
        "id": "1zaayNM_ywVAswmePmTdWgOPxutFbf5Gt",
        "nom": "dvf_plus_residentiel.parquet",
        "dossier": DATA_OUT,
    },
    # Échantillon d'audit de 100 000 mutations tiré dans DVF+ brut (02_audit_eda)
    "dvf_audit": {
        "type": "fichier",
        "id": "17CP167oYjRA1rKT68PVF9S3RGqy3vcMC",
        "nom": "dvf_plus_echantillon_100k.csv",
        "dossier": DATA_OUT,
    },
    "filosofi": {
        "type": "dossier",
        "id": "1QVGH4D5z5WjQHPlYuNGehSm91W4DAPKS",
        "nom": "Filo_disp_com",
    },
    "menages_2019": {
        "type": "fichier",
        "id": "1wIBXzENX7x-eLiRuueumSszFA35DR7Fl",
        "nom": "base-ic-couples-familles-menages-2019.CSV",
    },
    "menages_2020": {
        "type": "fichier",
        "id": "1kKwAwzc_rhxFgtCpMRfBMVOoPPfF_TsH",
        "nom": "base-ic-couples-familles-menages-2020.CSV",
    },
    "menages_2021": {
        "type": "fichier",
        "id": "1C9RBwsNKQy4iQzepWbVlbTOL6z_Og5_k",
        "nom": "base-ic-couples-familles-menages-2021.CSV",
    },
    "menages_2022": {
        "type": "fichier",
        "id": "1ZO7fCg5S9WVtW7YMc-v5hPY7maBNRdHM",
        "nom": "base-ic-couples-familles-menages-2022.CSV",
    },
    
    "logement": {
        "type": "dossier",
        "id": "1iixzBgxre54KhwnzFuazFpRh2mJ7fCyc",
        "nom": "base-ic-logement-2022_csv",
    },
    "bpe": {
        "type": "fichier",
        "id": "1bYiuYTkfnWON8Nv9Gc1UAnNCgs09363E",
        "nom": "BPE25.parquet",
    },
    "bassins_vie": {
        "type": "fichier",
        "id": "106njCJVhPHHvHPVIXci3qFknIFOfccv6",
        "nom": "BV2022_2022_2026.xlsx",
    },
    # Bassins de vie : fichier des années 2019 à 2022 (fichier Excel sur le Drive)
    "bassins_vie_2019_2022": {
        "type": "fichier",
        "id": "1vY8O8tViuDknIhInExbKy-soHMcdyBvG",
        "nom": "bassins_vie_2019_2022.xlsx",
    },
    "passage_geo": {
        "type": "fichier",
        "id": "1o6_aOs4BBUNYjgeZGpglT27X3jJgVj7h",
        "nom": "table_passage_geo2003_geo2026.xlsx",
    },
    # Table construite par CB : ancienne table cible + indicateurs BPE 2015 et 2020
    "equipements": {
        "type": "fichier",
        "id": "1fykkr-91y6Wijw0mGQtmC7hATSS_OyYb",
        "nom": "cible_commune_annee_type_BPE_152_indicateurs.csv",
    },
    "lovac": {
        "type": "fichier",
        "id": "1xN2b_YVN741P9OfgXxvYjsy722sk90V3",
        "nom": "lovac-open-data-2020-a-2026.xlsx",
    },
    "delinquance": {
        "type": "fichier",
        "id": "1Mc5CodVRAbMhG5L3Kd8sDkhmGQQirr1p",
        "nom": "dataset_delinquance_communale.parquet",
    },
    "taux": {
        "type": "fichier",
        "id": "1YhJRedRswwSfTrhTJlR1abCjvHj2IonP",
        "nom": "Taux d'intérêt annuel des crédits immobiliers.csv",
    },
    # Annonces immobilières du Haut-Rhin (axe 2)
    "annonces_ventes": {
        "type": "fichier",
        "id": "15vSblxHZQCpMDj9Uxi1ZtSZX2U5Q__Tu",
        "nom": "ech_annonces_ventes_68.csv",
    },
    "annonces_locations": {
        "type": "fichier",
        "id": "11lEZw7wI3SaAvsIBKLLeRmTusPCs8MRj",
        "nom": "ech_annonces_locations_68.csv",
    },
}


# -----------------------------------------------------------------------------
# 2. Fonction de téléchargement d'une source
# -----------------------------------------------------------------------------

def telecharger(cle):
    """Télécharge une source. Renvoie True si tout est arrivé, False sinon."""
    source = SOURCES[cle]
    destination = source.get("dossier", DATA_RAW) / source["nom"]

    # Lien Drive pas encore renseigné dans le catalogue
    if source["id"] == "A_COMPLETER":
        print(f"[échec] {source['nom']} : lien Drive à renseigner dans le catalogue SOURCES")
        return False

    # Fichier déjà présent : on ne refait pas le téléchargement
    if source["type"] == "fichier" and destination.exists():
        print(f"[déjà présent] {source['nom']}")
        return True

    print(f"[téléchargement] {source['nom']}")
    try:
        if source["type"] == "fichier":
            url = f"https://drive.google.com/uc?id={source['id']}"
            gdown.download(url, str(destination), quiet=False, resume=True, retries=2)
        else:
            # Pour un dossier, on relance toujours : resume=True saute les fichiers
            # déjà complets et ne télécharge que ceux qui manquent
            url = f"https://drive.google.com/drive/folders/{source['id']}"
            gdown.download_folder(url, output=str(destination), quiet=False,
                                  resume=True, retries=2)
        return True
    except Exception as erreur:
        print(f"[échec] {source['nom']} : {erreur}")
        return False


# -----------------------------------------------------------------------------
# 3. Programme principal
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demandes = sys.argv[1:]

    # Aucun argument : on affiche les sources disponibles et on s'arrête
    if not demandes:
        print("Sources disponibles :")
        for cle, source in SOURCES.items():
            print(f"  {cle:<22} -> {source['nom']}")
        print("\nExemple : python 00_telechargement/telechargement.py dvf taux")
        sys.exit(0)

    if demandes == ["tout"]:
        demandes = list(SOURCES)

    # Vérifie les noms avant de lancer quoi que ce soit
    inconnues = [cle for cle in demandes if cle not in SOURCES]
    if inconnues:
        print(f"Source(s) inconnue(s) : {', '.join(inconnues)}")
        sys.exit(1)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    echecs = [cle for cle in demandes if not telecharger(cle)]

    # Bilan : les sources incomplètes sont à relancer plus tard
    # (les fichiers déjà complets ne seront pas retéléchargés)
    if echecs:
        print(f"\nSources incomplètes : {', '.join(echecs)}")
        print(f"Relancer plus tard : python 00_telechargement/telechargement.py {' '.join(echecs)}")
        sys.exit(1)
    print("\nToutes les sources demandées sont complètes.")