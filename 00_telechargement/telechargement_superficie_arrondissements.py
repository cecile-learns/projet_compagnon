"""Télécharge la superficie des 45 arrondissements municipaux (Paris, Lyon, Marseille).

Source : API Découpage Administratif (geo.api.gouv.fr), type "arrondissement-municipal", champ "surface" (hectares).

Fichiers déposés dans le dossier des sources brutes (par défaut data/raw) :
  - superficie_arrondissements_geoapi.json      : réponse de l'API conservée telle quelle (octets inchangés)
  - superficie_arrondissements_geoapi.meta.json : URL, paramètres, date de récupération

À lancer AVANT l'exécution de Pipeline_Referentiel_geographique.ipynb, pour que le fichier figure dans le
manifeste SHA-256 des sources brutes.

Usage : python telechargement_superficie_arrondissements.py [dossier_raw]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://geo.api.gouv.fr/communes"
PARAMS = {"type": "arrondissement-municipal", "fields": "code,nom,surface", "format": "json"}
NOM_BRUT = "superficie_arrondissements_geoapi.json"
NOM_META = "superficie_arrondissements_geoapi.meta.json"


def telecharger(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rep = requests.get(URL, params=PARAMS, timeout=60)
    rep.raise_for_status()

    # Contrôles minimaux avant d'écrire quoi que ce soit
    data = rep.json()
    if len(data) != 45:
        raise ValueError(f"45 arrondissements attendus, {len(data)} reçus : le fichier n'est pas écrit")
    if not all({"code", "surface"} <= set(d) and d["surface"] is not None for d in data):
        raise ValueError("Champs 'code' ou 'surface' absents ou vides : le fichier n'est pas écrit")

    (raw_dir / NOM_BRUT).write_bytes(rep.content)          # brut conservé tel quel
    meta = {
        "url": rep.url,
        "parametres": PARAMS,
        "date_recuperation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unite_surface": "hectares",
        "nombre_arrondissements": len(data),
    }
    (raw_dir / NOM_META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Écrit :", raw_dir / NOM_BRUT)
    print("Écrit :", raw_dir / NOM_META)


if __name__ == "__main__":
    telecharger(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw"))
