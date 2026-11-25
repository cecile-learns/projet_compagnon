# Fiche du modèle H7v4

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

- Date de production : 2026-11-15
- Années d'apprentissage : 2017 à 2025 (773106 ventes)
- Époques : 20 (arrêt réglé sur 2025)
- Variables d'entrée : 35 ; sous-bassins connus : 1810
- Réglages : couches [128, 64, 32], dropout 0.1, taux d'apprentissage 0.0003, perte mae, dimension de l'embedding 4, cible apprise en logarithme : True, référence : log_prix_n1
- Versions : keras 3.15.1, pandas 3.0.6, numpy 2.5.3 (garder les mêmes pour recharger sans écart)

## Erreur attendue (fourchette)

Erreur d'une vente = prix réel / prévision − 1, mesurée sur les ventes de 2025 avec le modèle appris jusqu'en 2024
(les 10e et 90e centiles, pondérés : 80 % des ventes tombent entre les deux bornes).

| Type de bien | Ventes | Borne basse | Médiane | Borne haute |
| --- | --- | --- | --- | --- |
| appartement | 34545 | -34.3 % | +2.4 % | +49.8 % |
| maison | 45393 | -42.7 % | +2.0 % | +53.2 % |

L'outil n'affiche pas cette fourchette d'erreur : il affiche celle des prix de biens similaires vendus.

Le modèle de ce dossier a, lui, appris sur 2025 : il ne se juge donc pas sur 2025. Les résultats du test
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
