# =============================================================================
# 05_application/localisation.py : localiser(commune), bloc C (Théophyle)
# =============================================================================
# Pourquoi : chaque bloc de l'outil travaille au sous-bassin de vie (le territoire du modèle) ou au bassin
# de vie (recherche élargie des ventes comparables). localiser() traduit le code INSEE saisi en ces deux
# territoires.
#
# Contrat (contrats.py) : localiser(commune) -> dictionnaire
#   commune            : code de la commune utilisé dans les ventes (commune 2026, ou arrondissement)
#   sous_bassin_de_vie : code du sous-bassin (texte), None si la commune n'est pas localisée
#   bassin_de_vie      : code du bassin de vie (texte), None si la commune n'est pas localisée
#   message            : texte pour l'utilisateur, ou None si tout va bien
#
# Cas particuliers :
#   - ancien code (commune fusionnée depuis 2003) : rattaché à sa commune 2026 ;
#   - arrondissements de Paris, Lyon, Marseille : chacun est son propre sous-bassin ;
#   - Paris, Lyon ou Marseille sans arrondissement (75056, 69123, 13055) : pas de sous-bassin, on demande
#     l'arrondissement (les ventes et le modèle sont à l'arrondissement) ;
#   - code mal formé ou inconnu : pas de sous-bassin, message clair.
#
# liste_communes() : la liste déroulante de la page Outil (« Bordeaux (33000) » -> code INSEE).
#
# Données : donnees/communes_sous_bassins.parquet et donnees/communes_recherche.parquet, produits par
# 04_outils/preparer_communes_app.py.
# Vérification : python localisation.py (depuis 05_application).

import pandas as pd

from commun_app import charger_parquet

FICHIER = "communes_sous_bassins.parquet"
FICHIER_RECHERCHE = "communes_recherche.parquet"

# Villes suivies arrondissement par arrondissement : le code de la ville entière ne suffit pas
VILLES_A_ARRONDISSEMENTS = {"75056": "Paris", "69123": "Lyon", "13055": "Marseille"}


def _table_communes():
    """Table des communes, indexée par le code saisi (lue une seule fois grâce au cache de charger_parquet)."""
    return charger_parquet(FICHIER).set_index("CODGEO_INI")


def _non_localisee(commune, message):
    """Réponse quand la commune n'est pas localisée : codes vides et message."""
    return {"commune": commune, "sous_bassin_de_vie": None, "bassin_de_vie": None, "message": message}


def localiser(commune):
    """Code INSEE de la commune (texte de 5 caractères) -> sous-bassin et bassin de vie."""
    # 1. Mise en forme du code : texte sans espaces, zéro ajouté devant si besoin (« 1001 » -> « 01001 »).
    #    Les codes de Corse (2A..., 2B...) contiennent des lettres : on ne les convertit pas en nombre.
    if commune is None:
        return _non_localisee(None, "Aucune commune choisie.")
    code = str(commune).strip().upper().zfill(5)
    if len(code) != 5:
        return _non_localisee(code, "Code commune invalide : un code INSEE a 5 caractères.")

    # 2. Paris, Lyon, Marseille sans arrondissement
    if code in VILLES_A_ARRONDISSEMENTS:
        ville = VILLES_A_ARRONDISSEMENTS[code]
        return _non_localisee(code, "Pour " + ville + ", choisissez l'arrondissement : "
                                    "les prix y sont suivis arrondissement par arrondissement.")

    # 3. Recherche dans la table des communes
    table = _table_communes()
    if code not in table.index:
        return _non_localisee(code, "Commune inconnue : vérifiez la commune choisie.")
    ligne = table.loc[code]

    # 4. Commune sans territoire (cas très rare, signalé plutôt que de renvoyer une case vide)
    if pd.isna(ligne["sous_bassin_de_vie"]) or pd.isna(ligne["BV2022"]):
        return _non_localisee(code, "Cette commune n'est rattachée à aucun bassin de vie : "
                                    "pas d'estimation possible.")

    return {"commune": str(ligne["code_commune"]),
            "sous_bassin_de_vie": str(ligne["sous_bassin_de_vie"]),
            "bassin_de_vie": str(ligne["BV2022"]),
            "message": None}


def liste_communes():
    """Communes proposées dans la page Outil : colonnes libelle (« Bordeaux (33000) ») et code_insee.

    Paris, Lyon et Marseille y figurent par arrondissement. Dans la page :
        communes = liste_communes()
        choix = st.selectbox("Commune", communes["libelle"], index=None)
        code = communes.loc[communes["libelle"] == choix, "code_insee"].iloc[0]
    """
    return charger_parquet(FICHIER_RECHERCHE)


# -----------------------------------------------------------------------------
# Vérification : quelques communes tests, et le respect du contrat
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from contrats import verifier_format

    CAS = [("33063", "Bordeaux"), ("68224", "Mulhouse"), ("01001", "petite commune (L'Abergement-Clémenciat)"),
           ("2A004", "Ajaccio (code corse)"), ("1001", "code sans le zéro de tête"),
           ("75108", "Paris 8e"), ("69383", "Lyon 3e"), ("13201", "Marseille 1er"),
           ("75056", "Paris sans arrondissement"), ("99999", "code inconnu"), ("abc", "code mal formé"),
           (None, "rien choisi")]
    for code, libelle in CAS:
        resultat = verifier_format("localiser", localiser(code))
        print(libelle, ":", resultat)

    # Même saisie, même résultat
    assert localiser("33063") == localiser("33063")
    # Un arrondissement est son propre sous-bassin
    assert localiser("75108")["sous_bassin_de_vie"] == "75108"

    # La liste déroulante : chaque commune proposée doit être localisée sans message
    communes = liste_communes()
    print("Communes proposées :", len(communes))
    print(communes[communes["libelle"].str.startswith(("Bordeaux", "Paris 8e"))].to_string(index=False))
    sans_territoire = 0
    for code in communes["code_insee"]:
        if localiser(code)["sous_bassin_de_vie"] is None:
            sans_territoire = sans_territoire + 1
    print("Communes proposées sans territoire :", sans_territoire)
    assert sans_territoire == 0, "une commune de la liste n'est pas localisée"
    print("Contrôles réussis.")
