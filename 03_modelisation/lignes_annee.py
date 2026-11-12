# =============================================================================
# 03_modelisation/lignes_annee.py : lignes d'entrée de H7v4 pour une année (bloc A de l'outil)
# =============================================================================
# construire_lignes(annee) fabrique, pour chaque sous-bassin x type de bien, les variables de
# TERRITOIRE / PRIX N-1 / VOLUME ET ÉLAN / TAUX D'EFFORT / TRIMESTRES que H7v4 lit pour prévoir
# l'année `annee`. Seules des informations connues en annee-1 (ou avant) sont utilisées.
#   - annee = 2025 : sert à CONTRÔLER la fonction contre la table, P7, P9 (controle_lignes_2025.py).
#   - annee = 2026 : produit lignes_2026.parquet pour l'outil (ventes 2025 complètes requises).
# Les variables du BIEN (surface, pièces, terrain, dépendances) viennent de la saisie de
# l'utilisateur : elles ne sont pas dans ces lignes.
#
# Sources (data/processed) :
#   fe_cible_toutes_sources_sbv.parquet : ventes et prix médians par sous-bassin x année, variables
#       de territoire (photos fixes : mêmes valeurs pour toutes les années)  [feature_engineering, §8]
#   taux_credit_mois.parquet, filosofi_commune_2014_2021.parquet, liaison_bv_scindes.parquet  [P7]
#   ventes_apprentissage.parquet (2014-2024), ventes_test_2025.parquet (2025)  [preparation, bloc 3 bis]
# Règles reprises à l'identique : seuil de 20 ventes AVANT prix_n1 (feature_engineering §8.2-8.3),
# indicateurs de P9 (volume, élan), taux d'effort de P7, glissements de trimestres de test2025_H7v4.py §5.

import numpy as np
import pandas as pd

from commun import DATA_OUT, charger_ventes_brutes, mediane_ponderee

SEUIL_VENTES = 20
TYPES = {"maison": {"n": "n_maison", "prix": "prix_m2_med_maison"},
         "appartement": {"n": "n_appart", "prix": "prix_m2_med_appart"}}
CLES = ["sous_bassin_de_vie", "type_bien"]

# Territoire de H7v4 (photos fixes : recensement 2022, LOVAC 2023, densité 2024)
PARC = ["taux_vacance", "part_rsec", "part_apparts", "log_nb_logements", "part_moins40m2",
        "part_avant1946", "part_2006_2019", "anciennete_moyenne", "part_sans_voiture"]
TERRITOIRE = PARC + ["taux_vacance_complete", "densite_population"]

# Taux d'effort (conventions de P7, figées)
SURFACE = {"maison": 100, "appartement": 60}   # m²
DUREE_MOIS = 25 * 12
MOIS_TAUX = 12                                 # taux de décembre N-1
UC_MENAGE = 1.5
DECALAGE_FILOSOFI = 3                          # revenus de N-3 (millésime 2021 au plus)
VILLES_ENTIERES = ["75056", "69123", "13055"]

# Trimestres précédents (test2025_H7v4.py, section 5)
SEUIL_TRIMESTRE = 30
TRIMESTRES = [3, 4]
BLOC_TRIMESTRES = [f"gliss_{x}_t{q}" for q in TRIMESTRES for x in ("prix", "volume")]


# -----------------------------------------------------------------------------
# 1. Prix, ventes et territoire (feature_engineering §8 et P9)
# -----------------------------------------------------------------------------
def charger_cible():
    """Format long (année, sous-bassin, type) avec le prix vidé sous 20 ventes, et photos du territoire."""
    df = pd.read_parquet(DATA_OUT / "fe_cible_toutes_sources_sbv.parquet")
    attendues = ["sous_bassin_de_vie", "annee"] + [c for t in TYPES.values() for c in t.values()] + TERRITOIRE
    absentes = sorted(set(attendues) - set(df.columns))
    assert not absentes, f"colonnes absentes de fe_cible_toutes_sources_sbv : {absentes}"
    blocs = []
    for type_bien, c in TYPES.items():
        b = df[["sous_bassin_de_vie", "annee", c["n"], c["prix"]]].rename(columns={c["n"]: "n_ventes", c["prix"]: "prix"})
        blocs.append(b.assign(type_bien=type_bien))
    longue = pd.concat(blocs, ignore_index=True)
    assert not longue.duplicated(["annee"] + CLES).any(), "doublons année x sous-bassin x type"
    longue.loc[~(longue["n_ventes"] >= SEUIL_VENTES), "prix"] = np.nan      # seuil AVANT prix_n1
    territoire = df.sort_values("annee").groupby("sous_bassin_de_vie")[TERRITOIRE].last()
    return longue, territoire


def lignes_base(longue, territoire, annee):
    """prix_n1, nb_ventes_n1, élan (N-2 -> N-1), territoire et rotation du parc : une ligne par sous-bassin x type."""
    n1 = (longue[longue["annee"] == annee - 1].set_index(CLES)[["n_ventes", "prix"]]
          .rename(columns={"n_ventes": "nb_ventes_n1", "prix": "prix_n1"}))
    n2 = longue[longue["annee"] == annee - 2].set_index(CLES)["prix"].rename("prix_n2")
    b = n1.join(n2, how="left")
    b = b[b["prix_n1"].notna()].reset_index()             # une ligne n'existe que si prix_n1 est connu (>= 20 ventes en N-1)
    b = b.join(territoire, on="sous_bassin_de_vie")
    b["annee"] = annee
    b["maison"] = (b["type_bien"] == "maison").astype(int)
    b["evol_prix_n2_n1"] = (b["prix_n1"] / b["prix_n2"] - 1) * 100       # vide si moins de 20 ventes en N-2
    parc = 10 ** b["log_nb_logements"]
    part_type = np.where(b["type_bien"] == "appartement", b["part_apparts"], 100 - b["part_apparts"]) / 100
    b["rotation_parc_type"] = b["nb_ventes_n1"] / (parc * np.where(part_type > 0, part_type, np.nan)) * 100
    return b.drop(columns="prix_n2")


# -----------------------------------------------------------------------------
# 2. Taux d'effort (P7)
# -----------------------------------------------------------------------------
def revenu_par_sous_bassin():
    """Niveau de vie médian par sous-bassin et millésime Filosofi, pondéré par les ménages fiscaux (P7, section 2)."""
    filo = pd.read_parquet(DATA_OUT / "filosofi_commune_2014_2021.parquet",
                           columns=["code_commune", "annee", "niveau_vie_median", "NBMEN"])
    filo = filo[~filo["code_commune"].isin(VILLES_ENTIERES)]
    liaison = pd.read_parquet(DATA_OUT / "liaison_bv_scindes.parquet", columns=["CODGEO_INI", "sous_bassin_de_vie"])
    com = filo.merge(liaison, left_on="code_commune", right_on="CODGEO_INI", how="inner")
    com = com[com["niveau_vie_median"].notna() & com["NBMEN"].notna()].copy()
    com["masse_niveau_vie"] = com["niveau_vie_median"] * com["NBMEN"]
    r = com.groupby(["sous_bassin_de_vie", "annee"])[["masse_niveau_vie", "NBMEN"]].sum().reset_index()
    r["niveau_vie_median"] = r["masse_niveau_vie"] / r["NBMEN"]
    return r.rename(columns={"annee": "millesime_filosofi"})[["sous_bassin_de_vie", "millesime_filosofi", "niveau_vie_median"]]


def ajouter_taux_effort(b, annee):
    """taux_effort = mensualité (prêt 25 ans, 100 % du prix N-1 x surface type, taux de décembre N-1)
    / revenu mensuel du ménage type (niveau de vie N-3 x 1,5 UC / 12), en %."""
    b = b.copy()
    tm = pd.read_parquet(DATA_OUT / "taux_credit_mois.parquet")
    taux = tm[(tm["mois"] == MOIS_TAUX) & (tm["annee"] == annee - 1)]["taux"]
    if len(taux) != 1:
        raise ValueError(f"taux de crédit de décembre {annee - 1} introuvable ou en double dans taux_credit_mois.parquet")
    revenu = revenu_par_sous_bassin()
    millesime = min(annee - DECALAGE_FILOSOFI, int(revenu["millesime_filosofi"].max()))
    r = revenu[revenu["millesime_filosofi"] == millesime].set_index("sous_bassin_de_vie")["niveau_vie_median"]
    t = float(taux.iloc[0]) / 100 / 12
    k = b["prix_n1"] * b["type_bien"].map(SURFACE)
    mensualite = k * t / (1 - (1 + t) ** (-DUREE_MOIS))
    b["taux_effort"] = mensualite / (b["sous_bassin_de_vie"].map(r) * UC_MENAGE / 12) * 100
    return b


# -----------------------------------------------------------------------------
# 3. Trimestres précédents (test2025_H7v4.py, section 5 : copie)
# -----------------------------------------------------------------------------
def charger_ventes_jusqua(annee_max):
    """Ventes pondérées (w_h) de 2014 à annee_max : apprentissage (2014-2024) puis, si besoin, tirage 2025."""
    parts = [charger_ventes_brutes()]
    if annee_max >= 2025:
        parts.append(pd.read_parquet(DATA_OUT / "ventes_test_2025.parquet"))
    v = pd.concat(parts, ignore_index=True)
    v = v[v["anneemut"] <= annee_max]
    assert v["anneemut"].max() == annee_max, f"ventes de {annee_max} absentes"
    return v


def stats_trimestre(v, lieu):
    g = v.groupby(lieu + ["anneemut", "trimestre"])
    return pd.DataFrame({"mediane": g.apply(lambda e: mediane_ponderee(e["prix_m2"], e["w_h"]), include_groups=False),
                         "volume": g["w_h"].sum(), "ventes": g.size()}).reset_index()


def glissements(s, lieu):
    sorties = []
    for q in TRIMESTRES:
        e = s[s["trimestre"] == q].drop(columns="trimestre")
        n1 = e.assign(annee=e["anneemut"] + 1).drop(columns="anneemut")
        n2 = e.assign(annee=e["anneemut"] + 2).drop(columns="anneemut")
        m = n1.merge(n2, on=lieu + ["annee"], suffixes=("_n1", "_n2"))
        sorties.append(pd.DataFrame({f"gliss_prix_t{q}": (m["mediane_n1"] / m["mediane_n2"] - 1).to_numpy() * 100,
                                     f"gliss_volume_t{q}": (m["volume_n1"] / m["volume_n2"] - 1).to_numpy() * 100,
                                     f"fiable_t{q}": (m[["ventes_n1", "ventes_n2"]].min(axis=1) >= SEUIL_TRIMESTRE).to_numpy()},
                                    index=pd.MultiIndex.from_frame(m[lieu + ["annee"]])))
    return pd.concat(sorties, axis=1).reset_index()


def ajouter_trimestres(b, annee):
    """Glissements des T3 et T4 de annee-1 contre annee-2, au département ; national si un trimestre est peu fiable."""
    vb = charger_ventes_jusqua(annee - 1)
    vb = vb.assign(departement=vb["sous_bassin_de_vie"].str[:2], trimestre=(vb["moismut"] - 1) // 3 + 1)
    g_dep = glissements(stats_trimestre(vb, ["departement", "type_bien"]), ["departement", "type_bien"])
    g_nat = glissements(stats_trimestre(vb, ["type_bien"]), ["type_bien"])
    k = b[["annee", "sous_bassin_de_vie", "type_bien"]].reset_index(drop=True).assign(
        departement=lambda d: d["sous_bassin_de_vie"].str[:2])
    k = k.merge(g_dep, on=["departement", "type_bien", "annee"], how="left", validate="many_to_one")
    k = k.merge(g_nat, on=["type_bien", "annee"], how="left", validate="many_to_one", suffixes=("", "_nat"))
    for q in TRIMESTRES:
        national = ~k[f"fiable_t{q}"].eq(True)
        for x in ("prix", "volume"):
            c = f"gliss_{x}_t{q}"
            k.loc[national, c] = k.loc[national, c + "_nat"]
    b = b.reset_index(drop=True)
    b[BLOC_TRIMESTRES] = k[BLOC_TRIMESTRES].to_numpy(dtype=float)
    b["trimestre_national"] = (~k[[f"fiable_t{q}" for q in TRIMESTRES]].eq(True).all(axis=1)).astype(int).to_numpy()
    return b


# -----------------------------------------------------------------------------
# 4. Assemblage
# -----------------------------------------------------------------------------
COLONNES = (["annee", "maison", "nb_ventes_n1", "prix_n1"] + TERRITOIRE
            + ["evol_prix_n2_n1", "rotation_parc_type", "taux_effort"] + BLOC_TRIMESTRES + ["trimestre_national"])


def construire_lignes(annee, exclure_incomplets=True):
    """Lignes d'entrée de H7v4 pour `annee`, index (annee, sous_bassin_de_vie, type_bien).

    Une ligne existe si le sous-bassin a au moins 20 ventes en annee-1 (prix_n1 connu). Une ligne dont une
    variable est vide est retirée (exclure_incomplets=True), sauf evol_prix_n2_n1, laissée vide : H7v4 la
    remplace par 0 et ajoute l'indicatrice evol_n2_inconnue. Les lignes retirées sont comptées à l'écran.
    """
    longue, territoire = charger_cible()
    b = lignes_base(longue, territoire, annee)
    b = ajouter_taux_effort(b, annee)
    b = ajouter_trimestres(b, annee)
    b = b.set_index(["annee", "sous_bassin_de_vie", "type_bien"], drop=False)[COLONNES]
    obligatoires = [c for c in COLONNES if c != "evol_prix_n2_n1"]
    vides = b[obligatoires].isna().any(axis=1)
    print(f"construire_lignes({annee}) : {len(b)} lignes | {int(vides.sum())} avec une variable vide"
          + (" (retirées)" if exclure_incomplets else " (gardées)"))
    return b[~vides] if exclure_incomplets else b
