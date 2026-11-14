# =============================================================================
# P7_capacite_achat.py : indicateur de capacité d'achat (taux d'effort)
# =============================================================================
# Contributeur : Théophyle · Date : 09/11/2026
#
# Pourquoi : le taux des crédits est le même pour toute la France : seul, il ne peut pas
#   dire quel territoire va baisser plus qu'un autre. La mensualité d'un achat type,
#   rapportée au revenu local, change elle par territoire et par année. C'est la variable
#   de H6 (et une variable « quand » possible pour H7).
#
# Entrées : le prix N-1 (table de modélisation), le taux des crédits immobiliers
#   (Banque de France, taux_credit_mois.parquet), le niveau de vie médian des communes
#   (Filosofi 2014-2021, filosofi_commune_2014_2021.parquet).
#
# Produit : pour chaque ligne de la période longue (prix 2017 à 2025), le taux d'effort =
#   mensualité d'un prêt à taux fixe pour acheter un bien type au prix N-1, divisée par le
#   revenu mensuel d'un ménage type du territoire. Fichier fe_capacite_achat_sbv.parquet
#   (dans data/processed/, non suivi par Git), joint par H6 comme l'emploi l'est par H3c.
#
# Conventions (écrites avant toute mesure, puis figées) :
#   - Surface de référence : 60 m² pour un appartement, 100 m² pour une maison. Ordre de
#     grandeur des biens vendus ; pour un type de bien, la surface ne change que l'échelle
#     de l'indicateur, pas le classement des territoires.
#   - Financement : 100 % du prix emprunté, sans apport ni frais (formule de la feuille de
#     route). Un apport réduirait tous les taux d'effort dans la même proportion.
#   - Durée : 25 ans (300 mensualités), durée maximale fixée par le HCSF depuis le
#     1er janvier 2022.
#   - Taux : taux des crédits nouveaux à l'habitat de décembre N-1 (Banque de France),
#     dernier taux connu au moment de prévoir le prix N.
#   - Revenu : niveau de vie médian du sous-bassin × 1,5 unité de consommation, divisé
#     par 12. Filosofi donne un niveau de vie par unité de consommation ; 1,5 UC = un
#     couple, ménage type d'un achat.
#   - Millésime Filosofi : revenus de N-3 (2021 pour les prix 2024 et 2025). Filosofi est
#     publié environ deux ans après l'année des revenus : avec N-3, le revenu est connu
#     avant de prévoir N. Les millésimes après 2021 ne sont pas téléchargés.
#   - Agrégation au sous-bassin : moyenne des médianes communales pondérée par les
#     ménages fiscaux, comme dans le feature engineering (section 5).
#
# Formule de la mensualité (prêt à taux fixe) : avec K le montant emprunté, t le taux
#   mensuel (taux annuel / 12) et n le nombre de mensualités,
#       mensualité = K × t / (1 − (1 + t)^(−n))
#
# Seulement des informations connues en N-1 : prix N-1, taux de décembre N-1, revenus de N-3.
#
# Lancement, depuis la racine du projet :  python 03_modelisation/2_prepa_tour2/P7_capacite_achat.py
# (le socle doit être installé : pip install -e . depuis la racine, une seule fois)
#
# Utilisation dans H6 et H7 : ajouter_capacite_achat(table) ajoute la colonne taux_effort
# à la table de modélisation, sans toucher aux autres colonnes.

import numpy as np
import pandas as pd

import commun
from commun import charger_table

# Conventions ci-dessus : à ne plus changer
ANNEE_DEBUT_LONGUE = 2017       # période longue (P6)
SURFACE_APPARTEMENT = 60        # m²
SURFACE_MAISON = 100            # m²
DUREE_MOIS = 25 * 12            # 25 ans
MOIS_TAUX = 12                  # taux de décembre N-1
UC_MENAGE = 1.5                 # unités de consommation du ménage type (couple)
DECALAGE_FILOSOFI = 3           # revenus de N-3
VILLES_ENTIERES = ["75056", "69123", "13055"]   # Paris, Lyon, Marseille : arrondissements gardés
FICHIER = "fe_capacite_achat_sbv.parquet"
COLONNES = ["annee", "sous_bassin_de_vie", "type_bien", "taux_effort", "mensualite", "revenu_mensuel",
            "taux_credit", "millesime_filosofi"]


def taux_de_decembre():
    """Taux des crédits de décembre de chaque année (Banque de France), en % par an."""
    taux_mois = pd.read_parquet(commun.DATA_OUT / "taux_credit_mois.parquet")
    taux = taux_mois[taux_mois["mois"] == MOIS_TAUX][["annee", "taux"]]
    taux = taux.rename(columns={"annee": "annee_taux", "taux": "taux_credit"})

    # Un seul taux par année, et toutes les années utiles présentes (2016 à 2024)
    assert not taux["annee_taux"].duplicated().any(), "Plusieurs taux de décembre pour une même année"
    manquantes = sorted(set(range(ANNEE_DEBUT_LONGUE - 1, 2025)) - set(taux["annee_taux"]))
    assert not manquantes, "Taux de décembre manquant pour : " + str(manquantes)
    return taux


def revenu_par_sous_bassin():
    """Niveau de vie médian par sous-bassin et par millésime Filosofi (moyenne pondérée par les ménages fiscaux).

    Rattachement par la table de liaison (codes communes depuis 2003 : les anciens millésimes
    se rattachent aussi), sur les seules communes dont la médiane et le nombre de ménages sont connus.
    """
    filo = pd.read_parquet(commun.DATA_OUT / "filosofi_commune_2014_2021.parquet",
                           columns=["code_commune", "annee", "niveau_vie_median", "NBMEN"])
    filo = filo[~filo["code_commune"].isin(VILLES_ENTIERES)]
    liaison = pd.read_parquet(commun.DATA_OUT / "liaison_bv_scindes.parquet",
                              columns=["CODGEO_INI", "sous_bassin_de_vie"])

    com = filo.merge(liaison, left_on="code_commune", right_on="CODGEO_INI", how="inner")
    com = com[com["niveau_vie_median"].notna() & com["NBMEN"].notna()].copy()
    com["masse_niveau_vie"] = com["niveau_vie_median"] * com["NBMEN"]

    revenu = com.groupby(["sous_bassin_de_vie", "annee"])[["masse_niveau_vie", "NBMEN"]].sum().reset_index()
    revenu["niveau_vie_median"] = revenu["masse_niveau_vie"] / revenu["NBMEN"]
    return revenu.rename(columns={"annee": "millesime_filosofi"})[
        ["sous_bassin_de_vie", "millesime_filosofi", "niveau_vie_median"]]


def calculer_taux_effort(taux, revenu):
    """Taux d'effort de chaque ligne (sous-bassin x année x type de bien) de la période longue.

    Prix N-1 x surface -> montant emprunté -> mensualité au taux de décembre N-1 sur 25 ans
    -> divisée par le revenu mensuel du ménage type (revenus de N-3).
    """
    brute = charger_table(masquer=False)       # toutes les lignes : l'indicateur ne dépend pas du masque
    idx = brute.index
    lignes = pd.DataFrame({"annee": idx.get_level_values("annee"),
                           "sous_bassin_de_vie": idx.get_level_values("sous_bassin_de_vie"),
                           "type_bien": idx.get_level_values("type_bien"),
                           "prix_n1": brute["prix_n1"].to_numpy()})
    lignes = lignes[lignes["annee"] >= ANNEE_DEBUT_LONGUE].copy()

    # Correspondance des années : taux de décembre N-1, revenus de N-3 (2021 au plus : derniers téléchargés)
    lignes["annee_taux"] = lignes["annee"] - 1
    lignes["millesime_filosofi"] = (lignes["annee"] - DECALAGE_FILOSOFI).clip(upper=revenu["millesime_filosofi"].max())
    lignes = lignes.merge(taux, on="annee_taux", how="left", validate="many_to_one")
    lignes = lignes.merge(revenu, on=["sous_bassin_de_vie", "millesime_filosofi"], how="left", validate="many_to_one")

    # Montant emprunté : prix N-1 x surface de référence du type de bien
    lignes["surface_m2"] = np.where(lignes["type_bien"] == "maison", SURFACE_MAISON, SURFACE_APPARTEMENT)
    lignes["montant_emprunte"] = lignes["prix_n1"] * lignes["surface_m2"]

    # Mensualité d'un prêt à taux fixe : K x t / (1 - (1 + t)^(-n)), t = taux mensuel
    t = lignes["taux_credit"] / 100 / 12
    lignes["mensualite"] = lignes["montant_emprunte"] * t / (1 - (1 + t) ** (-DUREE_MOIS))

    # Revenu mensuel du ménage type et taux d'effort (en %)
    lignes["revenu_mensuel"] = lignes["niveau_vie_median"] * UC_MENAGE / 12
    lignes["taux_effort"] = lignes["mensualite"] / lignes["revenu_mensuel"] * 100

    # Contrôles : une ligne par sous-bassin x année x type ; seulement des informations connues avant N
    assert not lignes.duplicated(["annee", "sous_bassin_de_vie", "type_bien"]).any()
    assert (lignes["annee_taux"] == lignes["annee"] - 1).all()
    assert (lignes["millesime_filosofi"] <= lignes["annee"] - DECALAGE_FILOSOFI).all()
    return lignes


def controler(lignes):
    """Couverture, puis vérification à la main sur deux territoires (maisons, prix 2024)."""
    # Le taux d'effort existe dès que le prix N-1 et le revenu sont connus
    avec_prix_n1 = lignes[lignes["prix_n1"].notna()]
    assert avec_prix_n1["taux_effort"].notna().all(), "Des lignes ont un prix N-1 mais pas de taux d'effort (revenu manquant ?)"

    # On refait le calcul pas à pas pour un sous-bassin cher et un bon marché (environ 5e et 95e centiles du prix)
    maisons_2024 = avec_prix_n1[(avec_prix_n1["annee"] == 2024) & (avec_prix_n1["type_bien"] == "maison")
                                & avec_prix_n1["taux_effort"].notna()].sort_values("prix_n1")
    for ligne in [maisons_2024.iloc[len(maisons_2024) // 20], maisons_2024.iloc[-len(maisons_2024) // 20]]:
        k = ligne["prix_n1"] * SURFACE_MAISON
        t_mois = ligne["taux_credit"] / 100 / 12
        m = k * t_mois / (1 - (1 + t_mois) ** (-DUREE_MOIS))
        r = ligne["niveau_vie_median"] * UC_MENAGE / 12
        assert abs(m / r * 100 - ligne["taux_effort"]) < 1e-9, "Calcul à la main différent pour " + str(ligne["sous_bassin_de_vie"])


def ajouter_capacite_achat(table):
    """Ajoute le taux d'effort (P7) à la table de modélisation, sur l'année, le sous-bassin et le type de bien."""
    capacite = (pd.read_parquet(commun.DATA_OUT / FICHIER)
                .set_index(["annee", "sous_bassin_de_vie", "type_bien"])["taux_effort"])
    t = table.copy()
    t["taux_effort"] = capacite.reindex(t.index).to_numpy()
    return t


if __name__ == "__main__":
    lignes = calculer_taux_effort(taux_de_decembre(), revenu_par_sous_bassin())
    controler(lignes)
    lignes[COLONNES].to_parquet(commun.DATA_OUT / FICHIER, index=False)
    print("Exporté :", FICHIER, lignes[COLONNES].shape)
