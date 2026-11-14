# =============================================================================
# 04_outils/controle_lignes_2025.py : contrôle de construire_lignes sur 2025
# =============================================================================
# construire_lignes(2025) doit redonner les lignes 2025 déjà connues, sur les sous-bassins communs :
#   - la table de modélisation : prix_n1, nb_ventes_n1 et les 11 variables de territoire ;
#   - P9 (fe_volume_elan_sbv.parquet) : evol_prix_n2_n1, rotation_parc_type ;
#   - P7 (fe_capacite_achat_sbv.parquet) : taux_effort ;
#   - les trimestres : contre ref_trimestres_2025.parquet (facultatif, voir ci-dessous).
# Chaque écart est compté et listé dans controle_lignes_2025_ecarts.csv, pour être expliqué.
#
# Référence des trimestres (facultative) : lignes 2025 de `table` après la section 5 de modele_keras_test_final.py,
# enregistrées par :
#   t25 = table[table["annee"] == 2025]
#   t25[BLOC_TRIMESTRES + ["trimestre_national"]].to_parquet(commun.DATA_OUT / "ref_trimestres_2025.parquet")
# Lancement, depuis la racine du projet :  python 04_outils/controle_lignes_2025.py

import numpy as np
import pandas as pd

import commun
from commun import charger_table
from lignes_annee import construire_lignes, TERRITOIRE, BLOC_TRIMESTRES

TOLERANCE = 1e-6          # écart absolu admis (flottants) ; les variables sont des nombres décimaux
CLES = ["annee", "sous_bassin_de_vie", "type_bien"]

lignes = construire_lignes(2025, exclure_incomplets=False)


def a_plat(t):
    """Index (annee, sous-bassin, type) -> colonnes, quel que soit l'état de l'index."""
    noms = [n for n in t.index.names if n is not None and n not in t.columns]
    return (t.reset_index(level=noms) if noms else t).reset_index(drop=True)


def indexe(t):
    return a_plat(t).set_index(CLES)


table = indexe(charger_table(masquer=False))
table = table[table["annee"] == 2025] if "annee" in table.columns else table.xs(2025, level="annee", drop_level=False)
p9 = pd.read_parquet(commun.DATA_OUT / "fe_volume_elan_sbv.parquet").set_index(CLES)
p7 = pd.read_parquet(commun.DATA_OUT / "fe_capacite_achat_sbv.parquet").set_index(CLES)
references = [("table", table, ["prix_n1", "nb_ventes_n1"] + TERRITOIRE),
              ("P9", p9, ["evol_prix_n2_n1", "rotation_parc_type"]),
              ("P7", p7, ["taux_effort"])]
try:
    references.append(("trimestres", pd.read_parquet(commun.DATA_OUT / "ref_trimestres_2025.parquet"),
                       BLOC_TRIMESTRES + ["trimestre_national"]))
except FileNotFoundError:
    print("ref_trimestres_2025.parquet absent : trimestres non contrôlés (voir l'en-tête du fichier)")

lignes_i = lignes.copy()
lignes_i.index = lignes_i.index.set_names(CLES)
resume, ecarts = [], []
for nom, ref, colonnes in references:
    ref25 = ref[ref.index.get_level_values("annee") == 2025]
    communes = lignes_i.index.intersection(ref25.index)
    print(f"\n[{nom}] lignes 2025 de la référence : {len(ref25)} | reconstruites : {len(communes)}"
          f" | de la référence absentes de construire_lignes : {len(ref25.index.difference(lignes_i.index))}")
    for c in colonnes:
        a, b = lignes_i.loc[communes, c].astype(float), ref25.loc[communes, c].astype(float)
        diff = ~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (a.isna() & b.isna()))
        resume.append({"référence": nom, "variable": c, "lignes comparées": len(communes), "écarts": int(diff.sum())})
        if diff.any():
            d = pd.DataFrame({"référence": nom, "variable": c, "construire_lignes": a[diff], "valeur de la référence": b[diff]})
            ecarts.append(d.reset_index())

resume = pd.DataFrame(resume)
print("\n=== Résumé du contrôle sur 2025 ===")
print(resume.to_string(index=False))
sortie = commun.DATA_OUT / "controle_lignes_2025_ecarts.csv"
if ecarts:
    pd.concat(ecarts).to_csv(sortie, index=False)
    print(f"\n{int(resume['écarts'].sum())} écarts, listés dans {sortie.name} : à expliquer un par un.")
else:
    print("\nAucun écart : construire_lignes(2025) redonne les lignes 2025 de la table, de P9, de P7 et des trimestres contrôlés.")
