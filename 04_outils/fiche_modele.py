# =============================================================================
# fiche_modele.py : fiche et somme de contrôle du modèle enregistré dans models/
# =============================================================================
# Ce que fait ce fichier : à partir du réseau (.keras) et du fichier de paramètres (.json) rangés dans
# un dossier de models/, écrit à côté d'eux :
#   SHA256SUMS    : l'empreinte de chaque fichier (vérifiable avec `sha256sum -c SHA256SUMS`) ;
#   MODEL_CARD.md : la fiche du modèle, pour la personne qui le met en production.
# Tous les chiffres de la fiche sont lus dans le fichier de paramètres : rien n'est saisi à la main.
# Appelé par modele_final_h7v4.py ; peut aussi se lancer seul sur un dossier existant :
#   python 04_outils/fiche_modele.py models/h7v4

import hashlib
import json
import sys
from pathlib import Path

FICHIERS = ["h7v4.keras", "h7v4_parametres.json"]


def empreinte(fichier):
    h = hashlib.sha256()
    with open(fichier, "rb") as f:
        for morceau in iter(lambda: f.read(1 << 20), b""):
            h.update(morceau)
    return h.hexdigest()


def ecrire_fiche(dossier):
    dossier = Path(dossier)
    for nom in FICHIERS:
        assert (dossier / nom).exists(), f"{nom} absent de {dossier}"
    with open(dossier / "h7v4_parametres.json", encoding="utf-8") as f:
        p = json.load(f)

    sommes = {nom: empreinte(dossier / nom) for nom in FICHIERS}
    (dossier / "SHA256SUMS").write_text("".join(f"{h}  {nom}\n" for nom, h in sommes.items()), encoding="utf-8")

    a, r, v = p["apprentissage"], p["reglages"], p["versions"]
    fourchette = p["fourchette"]
    lignes_fourchette = "\n".join(
        f"| {t} | {d['ventes']} | {d['erreur_basse'] * 100:+.1f} % | {d['erreur_mediane'] * 100:+.1f} % "
        f"| {d['erreur_haute'] * 100:+.1f} % |" for t, d in fourchette["types"].items())

    fiche = f"""# Fiche du modèle {p['modele']}

Fichier généré par `04_outils/fiche_modele.py` à partir de `h7v4_parametres.json` : ne pas le modifier à la main.

## Ce que fait le modèle

Il estime le prix au m² d'un logement ancien (une maison ou un appartement, vendu seul) à partir des
caractéristiques du bien, de son sous-bassin de vie et des indicateurs de marché de la dernière année
connue. **Il estime un bien ; il ne prévoit pas l'évolution du marché.**

Périmètre : France métropolitaine hors Alsace-Moselle (DVF+ ne couvre pas les départements 57, 67 et 68) ;
départements d'outre-mer exclus ; ventes en l'état futur d'achèvement exclues.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `h7v4.keras` | le réseau de neurones (Keras) |
| `h7v4_parametres.json` | tout ce qu'il faut pour préparer un bien et lire la prévision : mise à l'échelle, numéro de chaque sous-bassin (embedding), liste et ordre des variables, transformations, réglages |
| `SHA256SUMS` | empreintes des deux fichiers (`sha256sum -c SHA256SUMS`) |

Le réseau et son fichier de paramètres vont ensemble : le premier sans le second donne des prévisions fausses.

## Apprentissage

- Date de production : {p['date']}
- Années d'apprentissage : {a['annees'][0]} à {a['annees'][1]} ({a['ventes']} ventes)
- Époques : {a['epoques']} (arrêt réglé sur {a['annee_arret']})
- Variables d'entrée : {len(p['preparation']['variables_h7v4'])} ; sous-bassins connus : {len(p['sous_bassins'])}
- Réglages : couches {r['couches']}, dropout {r['dropout']}, taux d'apprentissage {r['learning_rate']}, perte {r['perte']}, dimension de l'embedding {r['dim_embedding']}, cible apprise en logarithme : {r['cible_log']}, référence : {r['reference']}
- Versions : keras {v['keras']}, pandas {v['pandas']}, numpy {v['numpy']} (garder les mêmes pour recharger sans écart)

## Erreur attendue (fourchette)

Erreur d'une vente = prix réel / prévision − 1, mesurée sur les ventes de {fourchette['annee_test']} avec le modèle appris jusqu'en 2024
(les 10e et 90e centiles, pondérés : {fourchette['couverture'] * 100:.0f} % des ventes tombent entre les deux bornes).

| Type de bien | Ventes | Borne basse | Médiane | Borne haute |
| --- | --- | --- | --- | --- |
{lignes_fourchette}

L'outil n'affiche pas cette fourchette d'erreur : il affiche celle des prix de biens similaires vendus.

Le modèle de ce dossier a, lui, appris sur {a['annees'][1]} : il ne se juge donc pas sur {a['annees'][1]}. Les résultats du test
final (comparaison à la référence naïve) sont dans le rapport final, section 3.8.

## Utilisation

Le chargement, la préparation d'un bien et la prévision sont dans `05_application/estimation.py`
(`charger_modele`, `preparer_entrees`, `prevoir`). Cette fonction d'estimation est autonome : elle n'importe rien du socle.

## Régénération

```bash
python 04_outils/fourchette_h7v4.py     # fourchette d'erreur (test 2025)
python 04_outils/modele_final_h7v4.py   # réentraîne le modèle, écrit ce dossier et copie les deux fichiers dans 05_application/donnees/
```

Le script refuse d'écraser un modèle déjà présent : renommer ou déplacer l'ancien dossier d'abord.
"""
    (dossier / "MODEL_CARD.md").write_text(fiche, encoding="utf-8")
    return sommes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage : python 04_outils/fiche_modele.py models/h7v4")
    for nom, h in ecrire_fiche(sys.argv[1]).items():
        print(f"{h}  {nom}")
