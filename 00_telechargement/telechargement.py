# =============================================================================
# 00 - TÉLÉCHARGEMENT DES DONNÉES BRUTES - projet Compagnon Immobilier
# =============================================================================
# Télécharge les sources brutes depuis le Google Drive du projet
# (dossier "compagnon_immo/raw", lisible par toute personne ayant le lien)
# et les range dans data/raw/. Une source vient directement d'une API publique
# (superficie des arrondissements) : sa réponse est enregistrée telle quelle.
#
# Utilisation, depuis la racine du projet :
#   python 00_telechargement/telechargement.py              -> affiche la liste des sources
#   python 00_telechargement/telechargement.py dvf taux     -> télécharge seulement ces sources
#   python 00_telechargement/telechargement.py tout         -> télécharge toutes les sources
#   python 00_telechargement/telechargement.py parcours_court -> télécharge le kit du parcours court
#                                                              (9 fichiers déjà préparés)
#
# Un fichier déjà complet dans data/raw/ n'est pas re-téléchargé.
# Si un téléchargement échoue (Google Drive limite parfois les gros téléchargements),
# il suffit de relancer la même commande plus tard : seuls les fichiers manquants
# sont téléchargés.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gdown
import requests

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
# Type "api" : pas d'identifiant Drive, mais l'adresse de l'API et ses paramètres.

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
    # Table de modélisation déjà construite (01_preparation puis 03_modelisation/feature_engineering.py) :
    # raccourci pour un examinateur ; l'équipe la reconstruit avec le pipeline
    "table_modelisation": {
        "type": "fichier",
        "id": "15w3jw5mqsMzK6fWvRnnNlBBN9ofVM40I",
        "nom": "table_modelisation_bv.parquet",
        "dossier": DATA_OUT,
    },
    # --- Kit du parcours court : fichiers déjà préparés, rangés dans data/processed/ ---
    # Avec la table de modélisation (ci-dessus), ils suffisent à lancer prepa_tour2 (P7, P9),
    # les hypothèses H3 à H7 et l'évaluation 2025. Liste : PARCOURS_COURT, plus bas.
    # Filosofi par commune (entrée de P7)
    "filosofi_commune": {
        "type": "fichier",
        "id": "1yaVTW81JLPbMW5j3dkxIeZulGx3vXzuC",
        "nom": "filosofi_commune_2014_2021.parquet",
        "dossier": DATA_OUT,
    },
    # Table de liaison commune -> sous-bassin de vie (entrée de P7)
    "liaison_bv_scindes": {
        "type": "fichier",
        "id": "12n1_4_4vGcedspUMvczF96LEBXRbhX-z",
        "nom": "liaison_bv_scindes.parquet",
        "dossier": DATA_OUT,
    },
    # Taux des crédits immobiliers par mois (entrée de P7)
    "taux_credit_mois": {
        "type": "fichier",
        "id": "1QK6fSNf0Bsqnr7l8m315N5DebsVc4RcJ",
        "nom": "taux_credit_mois.parquet",
        "dossier": DATA_OUT,
    },
    # Indicateurs de capacité d'achat (P7) et de volume et élan (P9) : lus par H5-H6, H7 et l'évaluation 2025
    "fe_capacite_achat": {
        "type": "fichier",
        "id": "1Y_HOSwKaSCuNe2SKI3Mt9U4ffFSmczTZ",
        "nom": "fe_capacite_achat_sbv.parquet",
        "dossier": DATA_OUT,
    },
    "fe_volume_elan": {
        "type": "fichier",
        "id": "1Hbypz34fvskGC5mRsV6xYh0h9Z5rgorh",
        "nom": "fe_volume_elan_sbv.parquet",
        "dossier": DATA_OUT,
    },
    # Ventes pour le modèle par vente, 2014-2024 (H7)
    "ventes_apprentissage": {
        "type": "fichier",
        "id": "1hLc8lCpAo4UOlL4HYjjMkga85ttXwv47",
        "nom": "ventes_apprentissage.parquet",
        "dossier": DATA_OUT,
    },
    # Ventes 2025 réservées au test final (évaluation 2025)
    "ventes_test_2025": {
        "type": "fichier",
        "id": "1ck8FkUyuMJIqSs6QvywidtEPeDFN4Rux",
        "nom": "ventes_test_2025.parquet",
        "dossier": DATA_OUT,
    },
    # Réglages optimisés du réseau Keras (résultat de l'optimisation de H7, déjà faite)
    "reglages_keras": {
        "type": "fichier",
        "id": "1RuuyLoebt0faZnt6jt4C0t2PqOE_wy9f",
        "nom": "meilleurs_reglages_keras.json",
        "dossier": DATA_OUT,
    },
    "filosofi": {
        "type": "dossier",
        "id": "1QVGH4D5z5WjQHPlYuNGehSm91W4DAPKS",
        "nom": "Filo_disp_com",
    },
    # Recensement, couples-familles-ménages : millésimes 2019 à 2022 et fichier de métadonnées
    "menages": {
        "type": "dossier",
        "id": "1kK54R_QGx4TG22LZbgWoJeJCSH-euKEK",
        "nom": "base-ic-couples-familles-menages-2022_csv",
    },
    "logement": {
        "type": "dossier",
        "id": "1iixzBgxre54KhwnzFuazFpRh2mJ7fCyc",
        "nom": "base-ic-logement-2022_csv",
    },
    # Recensement, bases IRIS Logement des millésimes 2016 à 2021 (et 2023, nouveau format) :
    # audit des millésimes (1_audits_sources, partie 5) et EDA (1_eda_sources, C4)
    "logement_millesimes": {
        "type": "dossier",
        "id": "1YK-Fz0PMYK2kk20wXUQVTT3vuJZWBlZJ",
        "nom": "logement_millesimes",
    },
    "bpe": {
        "type": "fichier",
        "id": "1bYiuYTkfnWON8Nv9Gc1UAnNCgs09363E",
        "nom": "BPE25.parquet",
    },
    # BPE en évolution (Insee, DS_BPE_EVOLUTION) : nombre d'équipements par type,
    # par territoire (dont bassin de vie BV2022), en 2015, 2020 et 2025
    "bpe_evolution": {
        "type": "fichier",
        "id": "1MEv5A8pYqQr9TqAtvQaX90qDm5t3WbTg",
        "nom": "DS_BPE_EVOLUTION_2025_data.csv",
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
    # Emploi salarié privé (Urssaf) : établissements et salariés au 31/12, par commune x activité (APE),
    # 2006 à 2025 ; lu directement par 03_modelisation/feature_engineering.py (section 6 bis)
    "emploi": {
        "type": "fichier",
        "id": "1BaFmodH9dCgrzKDm2K1gh9pq2uCClHyf",
        "nom": "etablissements_effectifs_commune.parquet",
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

    # Référentiel des communes françaises au 1er janvier 2026 (dont superficie en km²)
    "communes": {
        "type": "fichier",
        "id": "1w2WFp_Rw2MwUq0HZX7E1b7UmKhipmXSW",
        "nom": "communes-france-2026.csv",
    },
    # Superficie des 45 arrondissements de Paris, Lyon et Marseille, en hectares
    # (absents du référentiel des communes) : API Découpage administratif
    "superficie_arrondissements": {
        "type": "api",
        "url": "https://geo.api.gouv.fr/communes",
        "params": {"type": "arrondissement-municipal", "fields": "code,nom,surface", "format": "json"},
        "nom": "superficie_arrondissements_geoapi.json",
    },
}


# Groupes de sources : un mot-clé en ligne de commande qui en remplace plusieurs
PARCOURS_COURT = ["table_modelisation", "filosofi_commune", "liaison_bv_scindes", "taux_credit_mois",
                  "fe_capacite_achat", "fe_volume_elan", "ventes_apprentissage", "ventes_test_2025", "reglages_keras"]
GROUPES = {"parcours_court": PARCOURS_COURT}


# -----------------------------------------------------------------------------
# 2. Fonction de téléchargement d'une source
# -----------------------------------------------------------------------------

def telecharger(cle):
    """Télécharge une source. Renvoie True si tout est arrivé, False sinon."""
    source = SOURCES[cle]
    destination = source.get("dossier", DATA_RAW) / source["nom"]

    # Lien Drive pas encore renseigné dans le catalogue
    if source.get("id") == "A_COMPLETER":
        print(f"[échec] {source['nom']} : lien Drive à renseigner dans le catalogue SOURCES")
        return False

    # Fichier déjà présent : on ne refait pas le téléchargement
    if source["type"] in ("fichier", "api") and destination.exists():
        print(f"[déjà présent] {source['nom']}")
        return True

    print(f"[téléchargement] {source['nom']}")
    try:
        if source["type"] == "fichier":
            url = f"https://drive.google.com/uc?id={source['id']}"
            gdown.download(url, str(destination), quiet=False, resume=True, retries=2)
        elif source["type"] == "api":
            reponse = requests.get(source["url"], params=source["params"], timeout=60)
            reponse.raise_for_status()
            destination.write_bytes(reponse.content)      # réponse conservée telle quelle
            # La réponse d'une API peut changer : on garde l'adresse et la date de récupération
            meta = {"url": reponse.url,
                    "date_recuperation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            destination.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
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
        print("\nGroupes :")
        print(f"  {'tout':<22} -> toutes les sources ci-dessus")
        for nom, cles in GROUPES.items():
            print(f"  {nom:<22} -> {len(cles)} fichiers : {', '.join(cles)}")
        print("\nExemple : python 00_telechargement/telechargement.py dvf taux")
        sys.exit(0)

    if demandes == ["tout"]:
        demandes = list(SOURCES)
    # Un groupe (parcours_court) est remplacé par ses sources, sans doublon
    demandes = list(dict.fromkeys(cle for d in demandes for cle in GROUPES.get(d, [d])))

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