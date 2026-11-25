#!/usr/bin/env bash
# =============================================================================
# mettre_a_jour_dvf.sh : mise à jour de DVF+ à partir d'une nouvelle livraison du Cerema
# =============================================================================
# Trois étapes, lançables ensemble ou une par une, qui reprennent ce que fait le projet pour DVF+ :
#   extraire  : lit la livraison (zip de sauvegardes SQL) et garde les mutations d'une année, jusqu'à un mois ;
#               un fichier par région ; une région déjà faite est sautée (on peut interrompre et relancer) ;
#   assembler : réunit les régions en un seul fichier et affiche le contrôle mensuel (aucun mois ne doit manquer) ;
#   filtrer   : applique le filtre du bloc 2 de la préparation (un seul logement ancien, une seule commune,
#               DOM exclus) et écrit un Parquet dans data/processed/.
# Ce script met à jour les DONNÉES seulement : il ne réentraîne pas le modèle, ne régénère ni les lignes d'entrée
# ni les ventes comparables de l'application, et ne modifie aucun fichier du pipeline.
#
# Prérequis : l'environnement du projet activé, et  pip install py7zr  (nécessaire à l'étape extraire seulement).
#
# Utilisation, depuis n'importe quel dossier :
#   04_outils/mettre_a_jour_dvf.sh tout livraison.zip                       # les trois étapes, 2026 janvier à juin
#   04_outils/mettre_a_jour_dvf.sh tout livraison.zip --annee 2026 --mois-max 6
#   04_outils/mettre_a_jour_dvf.sh extraire livraison.zip --region R94_Corse   # essai sur la plus petite région
#   04_outils/mettre_a_jour_dvf.sh assembler
#   04_outils/mettre_a_jour_dvf.sh filtrer
# Les fichiers intermédiaires sont dans data/raw/dvf_plus_maj/ (hors Git, comme toutes les données).

set -euo pipefail

RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOSSIER="${DVF_MAJ_DOSSIER:-$RACINE/data/raw/dvf_plus_maj}"
PYTHON="${PYTHON:-python}"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

extraire()  { "$PYTHON" "$RACINE/04_outils/extraire_dvf_plus.py" extraire "$@"; }
assembler() { "$PYTHON" "$RACINE/04_outils/extraire_dvf_plus.py" assembler "$DOSSIER"; }
filtrer()   { "$PYTHON" "$RACINE/04_outils/filtrer_dvf_plus.py" "$DOSSIER/dvf_plus_assemble.csv"; }

commande="${1:-}"
[ -n "$commande" ] || usage 1
shift || true

case "$commande" in
    -h|--help|aide) usage 0 ;;
    extraire|tout)
        [ $# -ge 1 ] || { echo "Il manque le zip de la livraison (ou le dossier sql décompressé)." >&2; exit 1; }
        source_livraison="$1"; shift
        [ -e "$source_livraison" ] || { echo "Introuvable : $source_livraison" >&2; exit 1; }
        "$PYTHON" -c "import py7zr" 2>/dev/null || { echo "py7zr manque : pip install py7zr" >&2; exit 1; }
        mkdir -p "$DOSSIER"
        extraire "$source_livraison" "$DOSSIER" "$@"
        if [ "$commande" = "tout" ]; then
            assembler
            filtrer
        fi
        ;;
    assembler) assembler ;;
    filtrer)   filtrer ;;
    *) echo "Commande inconnue : $commande" >&2; usage 1 ;;
esac
