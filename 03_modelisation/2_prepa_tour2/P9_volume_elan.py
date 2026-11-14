# =============================================================================
# P9_volume_elan.py : indicateurs de volume et d'élan (prix 2017 à 2025)
# =============================================================================
# Contributrice : Gabrielle · Date : 09/11/2026
#
# Pourquoi : préparer H5 (« le volume et l'élan annoncent le sens de l'évolution »).
#   Ces indicateurs servent aussi à H7 (variables « quand »).
#
# Règle de la feuille de route : la préparation du tour 2 ne touche jamais la cible.
#   Ce script ne lit pas le prix de l'année N : il n'utilise que des informations de N-1
#   et de N-2 (contrôle : les indicateurs sont identiques quand la colonne prix est vidée).
#
# Constat : pas de nouvelle source. La table de modélisation contient déjà les ventes de
#   N-1 (nb_ventes_n1) et le prix de N-1 (prix_n1). Pour N-2, on lit les mêmes colonnes
#   sur la ligne de l'année précédente du même territoire.
#
#   Pour une ligne (sous-bassin, type de bien, année N) :
#     ventes N-1 : nb_ventes_n1 de la ligne N
#     ventes N-2 : nb_ventes_n1 de la ligne N-1 du même territoire
#     prix N-1   : prix_n1 de la ligne N
#     prix N-2   : prix_n1 de la ligne N-1 du même territoire
#
# Chargement : on part de la table brute (masquer=False). Le masque commun retire les prix
#   2015 et 2016 (délinquance vide, voir P6). Or la ligne 2016 est nécessaire pour calculer
#   l'élan des prix 2017, premier prix de la période longue. Le masque s'appliquera au
#   moment de modéliser, pas ici.
#
# Les quatre indicateurs :
#   evol_ventes_n2_n1   ventes N-1 / ventes N-2 - 1                          (%)
#   evol_prix_n2_n1     prix N-1 / prix N-2 - 1 (l'« élan »)                 (%)
#   rotation_parc       ventes N-1 / parc total de logements                 (% par an)
#   rotation_parc_type  ventes N-1 / parc du même type (appartements ou maisons)  (% par an)
#
# Règle : moins de 20 ventes en N-2 : case vide (à valider en équipe). Une ligne N-1 n'existe
#   dans la table que si le territoire avait au moins 20 ventes en N-1 et en N-2. Sans elle,
#   l'indicateur reste vide. C'est le même seuil que la cible : une médiane sur moins de
#   20 ventes est trop bruitée pour en tirer une évolution.
#
# Parc de logements : log_nb_logements est en log10 (recensement 2022), donc parc = 10
#   puissance log_nb_logements. Le parc du même type = parc × part_apparts pour les
#   appartements, parc × (100 - part_apparts) pour les maisons. Un parc du type nul (aucun
#   appartement) donne une case vide plutôt qu'une division par 0.
#
# Export : une ligne par année de prix × sous-bassin × type de bien, mêmes clés que la
#   table de modélisation. Les valeurs intermédiaires (ventes et prix N-2) ne sont pas
#   exportées. Le fichier n'est pas fusionné à la table de modélisation : H5 le joint à la
#   table au moment de modéliser, sur les clés.
#
# Lancement, depuis la racine du projet :  python 03_modelisation/2_prepa_tour2/P9_volume_elan.py
# (le socle doit être installé : pip install -e . depuis la racine, une seule fois)

import numpy as np

import commun
from commun import charger_table

CLES = ["sous_bassin_de_vie", "type_bien"]
COLONNES_P9 = ["evol_ventes_n2_n1", "evol_prix_n2_n1", "rotation_parc", "rotation_parc_type"]
FICHIER = "fe_volume_elan_sbv.parquet"


def a_plat(table):
    """Index (année, type de bien, sous-bassin) -> colonnes ; `annee` est déjà une colonne."""
    noms = [n for n in table.index.names if n is not None and n not in table.columns]
    return table.reset_index(level=noms).reset_index(drop=True) if noms else table.reset_index(drop=True)


def indicateurs_volume_elan(table):
    """Renvoie une ligne par (année, sous-bassin, type de bien) avec les 4 indicateurs P9.

    `table` : la table de modélisation (colonnes annee, nb_ventes_n1, prix_n1, log_nb_logements,
    part_apparts ; type_bien et sous_bassin_de_vie dans l'index). La colonne `prix` n'est jamais lue.
    """
    t = a_plat(table)
    base = t[["annee"] + CLES + ["nb_ventes_n1", "prix_n1", "log_nb_logements", "part_apparts"]].copy()

    # La ligne N-1 du même territoire, décalée d'un an : ses valeurs « N-1 » sont nos « N-2 »
    precedente = base[["annee"] + CLES + ["nb_ventes_n1", "prix_n1"]].rename(
        columns={"nb_ventes_n1": "ventes_n2", "prix_n1": "prix_n2"})
    precedente["annee"] += 1
    m = base.merge(precedente, on=["annee"] + CLES, how="left", validate="one_to_one")

    m["evol_ventes_n2_n1"] = (m["nb_ventes_n1"] / m["ventes_n2"] - 1) * 100
    m["evol_prix_n2_n1"] = (m["prix_n1"] / m["prix_n2"] - 1) * 100

    parc = 10 ** m["log_nb_logements"]
    part_type = np.where(m["type_bien"] == "appartement",
                         m["part_apparts"], 100 - m["part_apparts"]) / 100
    m["rotation_parc"] = m["nb_ventes_n1"] / parc * 100
    # parc du type nul (aucun appartement) -> case vide plutôt qu'une division par 0
    m["rotation_parc_type"] = m["nb_ventes_n1"] / (parc * np.where(part_type > 0, part_type, np.nan)) * 100

    return m[["annee"] + CLES + ["ventes_n2", "prix_n2"] + COLONNES_P9]


def verifier_a_la_main(table, p9, sous_bassin, type_bien, annee):
    """Relit les valeurs brutes N-1 / N-2 d'un territoire et refait le calcul."""
    t = a_plat(table)
    lignes = t[(t.sous_bassin_de_vie == sous_bassin) & (t.type_bien == type_bien)
               & t.annee.isin([annee - 1, annee])].set_index("annee")
    v_n1, v_n2 = lignes.loc[annee, "nb_ventes_n1"], lignes.loc[annee - 1, "nb_ventes_n1"]
    p_n1, p_n2 = lignes.loc[annee, "prix_n1"], lignes.loc[annee - 1, "prix_n1"]
    calc = p9[(p9.sous_bassin_de_vie == sous_bassin) & (p9.type_bien == type_bien)
              & (p9.annee == annee)].iloc[0]
    assert np.isclose(calc.evol_ventes_n2_n1, (v_n1 / v_n2 - 1) * 100), \
        "Évolution des ventes différente à la main : " + sous_bassin
    assert np.isclose(calc.evol_prix_n2_n1, (p_n1 / p_n2 - 1) * 100), \
        "Élan différent à la main : " + sous_bassin


brute = charger_table(masquer=False)   # toutes les lignes, avant le masque commun

p9 = indicateurs_volume_elan(brute)
assert len(p9) == len(brute), "Le calcul a changé le nombre de lignes"
assert not p9.duplicated(["annee"] + CLES).any(), "Doublons année x sous-bassin x type de bien"

# La cible n'est jamais lue : on vide la colonne prix (année N) et on recalcule, rien ne doit changer
assert indicateurs_volume_elan(brute.assign(prix=np.nan)).equals(p9), \
    "Un indicateur dépend du prix de l'année N : fuite"

# Vérification à la main : un appartement (prix 2024) et une maison (prix 2023)
exemples = p9.dropna(subset=COLONNES_P9)
ex_a = exemples[(exemples.type_bien == "appartement") & (exemples.annee == 2024)].iloc[0]
ex_m = exemples[(exemples.type_bien == "maison") & (exemples.annee == 2023)].iloc[100]
for ex in (ex_a, ex_m):
    verifier_a_la_main(brute, p9, ex.sous_bassin_de_vie, ex.type_bien, int(ex.annee))

export = p9.drop(columns=["ventes_n2", "prix_n2"])
export.to_parquet(commun.DATA_OUT / FICHIER, index=False)
print("Exporté :", FICHIER, export.shape)
