# 04_outils/produire_ref_trimestres_2025.py
# Référence des trimestres 2025 pour controle_lignes_2025.py : exécute les sections 1 à 5 de
# modele_keras_test_final.py (préparation seulement, ni réseau ni test), puis enregistre les lignes 2025.
# Lancement, depuis la racine du projet :  python 04_outils/produire_ref_trimestres_2025.py
from pathlib import Path

import commun

SCRIPT = Path("03_modelisation/4_evaluation_2025/modele_keras_test_final.py")
parties = SCRIPT.read_text(encoding="utf-8").split("\n# 6. Réglages optimisés")
assert len(parties) == 2, "repère de la section 6 introuvable"

espace = {"__name__": "__main__"}
exec(compile(parties[0], str(SCRIPT), "exec"), espace)       # sections 1 à 5 seulement

table = espace["table"]
colonnes = espace["BLOC_TRIMESTRES"] + ["trimestre_national"]
t25 = table[table.index.get_level_values("annee") == 2025]
sortie = commun.DATA_OUT / "ref_trimestres_2025.parquet"
assert not sortie.exists(), "ref_trimestres_2025.parquet existe déjà : le supprimer avant de relancer"
t25[colonnes].to_parquet(sortie)
print(len(t25), "lignes 2025 enregistrées :", sortie)