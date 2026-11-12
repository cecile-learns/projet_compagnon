# =============================================================================
# 03_modelisation/optimisation_keras.py : optimisation du réseau Keras de H7 (version H7v3)
# =============================================================================
# Constat de départ (H7, MLflow) : H7v3 bat la naïve pour les maisons les deux années jugées
# (évaluation B) ; pour les appartements, il réussit 2023 mais perd en 2024, dans les appartements
# les plus chers. Première optimisation : faire mieux sur les appartements sans dégrader les maisons.
#
# Plan, une étape à la fois ; chaque étape part de ce que la précédente a retenu :
#   Étape 1, les appartements :
#     H-a : le réseau ramène les prix élevés vers le milieu -> il apprend l'évolution par rapport
#           au prix N-1 du sous-bassin au lieu du prix ;
#     H-b : les maisons (deux fois plus de ventes) dominent l'apprentissage -> un réseau par type de bien.
#     Critère : MAE des appartements ; garde-fou : les maisons ne se dégradent pas.
#   Étape 2, la fonction de perte : MSE (actuelle), MAE, Huber. Critère : MAE de toutes les ventes ;
#     garde-fou : les appartements ne se dégradent pas.
#   Étape 3, les autres hyperparamètres (H-c) : grille complète sur le dropout, le taux d'apprentissage
#     et la taille de l'embedding (3 x 3 x 3 = 27 combinaisons), réseau fixe (couches inchangées).
#     Même critère et même garde-fou qu'à l'étape 2. Les couches ne sont testées que si les résultats
#     restent décevants (étape ultérieure).
#   Vérification (hors de ce script) : H7 relancé avec les réglages retenus, sur 2023 et 2024.
#
# Cadre commun à toutes les étapes :
# - variables de H7v3 ; apprentissage sur 20 % des ventes de 2017-2021 (tirées par année et type de
#   bien) ; validation sur toutes les ventes de 2022, qui n'est jamais une année jugée ;
# - mesure : MAE par vente pondérée par w_h ; affichées aussi par type de bien et pour le quart des
#   appartements au prix N-1 le plus élevé (Q4), avec la naïve (prix N-1) ;
# - 3 graines par candidat (le résultat d'un réseau varie un peu d'un entraînement à l'autre) ;
# - règle de choix : un candidat ne remplace la référence que s'il gagne plus que la variation due
#   à la graine, sans dégrader le garde-fou de plus que cette variation ; entre plusieurs gagnants
#   à égalité (écart inférieur à cette variation), le plus simple est gardé.
# Limite : 2022 est une année de hausse ; seule la vérification sur 2023 et 2024 teste le retournement.
#
# Le gabarit n'est pas modifié : la cible « évolution », le réseau par type de bien et la perte sont
# gérés ici. Ce qui est retenu sera ajouté au gabarit et à H7 seulement après la vérification.
#
# Entrées : DATA_OUT/ventes_h7.parquet et DATA_OUT/variables_h7v3.json (exportés par H7, section 4.7).
# Sorties : DATA_OUT/optimisation_keras_etat.json (configuration retenue et décisions, étape par étape),
#           DATA_OUT/meilleurs_reglages_keras.json (configuration retenue, lue plus tard par H7),
#           DATA_OUT/optimisation_keras_essais.csv (tous les essais) ;
#           option --mlflow : expérience MLflow « H7 optimisation », un run par essai (tag etape).
#
# Relancer une étape efface ses anciens résultats et ceux des étapes suivantes (état et tableau des essais),
# puis repart de ce que l'étape précédente avait retenu.
#
# Usage, depuis 03_modelisation, environnement activé, une étape après l'autre :
#     python optimisation_keras.py --etape 1
#     python optimisation_keras.py --etape 2
#     python optimisation_keras.py --etape 3
# Ajouter --mlflow pour enregistrer aussi chaque essai dans MLflow (suivi local, hors du dépôt) :
#     python optimisation_keras.py --etape 1 --mlflow

import json
import sys
import time

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import commun
from gabarit_keras import ReseauKeras

# =============================================================================
# PARAMÈTRES
# =============================================================================
ANNEE_DEBUT = 2017            # comme H7 : période longue
ANNEE_VALIDATION = 2022       # jamais une année jugée
PART_ECHANTILLON = 0.20       # part des ventes d'apprentissage utilisée
GRAINES = [commun.GRAINE, commun.GRAINE + 1, commun.GRAINE + 2]
TYPES = ["appartement", "maison"]
N_MEILLEURS = 2               # étape 3 : meilleures combinaisons contrôlées avec les autres graines

FICHIER_ETAT = commun.DATA_OUT / "optimisation_keras_etat.json"
FICHIER_REGLAGES = commun.DATA_OUT / "meilleurs_reglages_keras.json"
FICHIER_ESSAIS = commun.DATA_OUT / "optimisation_keras_essais.csv"
EXPERIENCE = "H7 optimisation"
AVEC_MLFLOW = "--mlflow" in sys.argv          # MLflow est facultatif : le script tourne sans lui

# Point de départ = H7v3 tel qu'il a tourné : réglages par défaut du gabarit, lus dans le gabarit
HYPERPARAMETRES = ["couches", "dropout", "learning_rate", "dim_embedding"]
_DEFAUT = ReseauKeras()
DEPART = {"cible": "prix", "organisation": "commun", "perte": "mse",
          **{k: getattr(_DEFAUT, k) for k in HYPERPARAMETRES}}

# Étape 3 : grille complète (le réseau, c'est-à-dire les couches, ne change pas)
GRILLE = {"dropout": [0.0, 0.1, 0.2],
          "learning_rate": [3e-4, 1e-3, 3e-3],
          "dim_embedding": [4, 8, 16]}


# =============================================================================
# 1. DONNÉES
# =============================================================================
def charger():
    """Échantillon d'apprentissage (2017-2021), validation 2022, variables de H7v3, seuil du Q4."""
    ventes = pd.read_parquet(commun.DATA_OUT / "ventes_h7.parquet")
    with open(commun.DATA_OUT / "variables_h7v3.json", encoding="utf-8") as f:
        variables = json.load(f)
    apprentissage = ventes[ventes["annee"].between(ANNEE_DEBUT, ANNEE_VALIDATION - 1)]
    validation = ventes[ventes["annee"] == ANNEE_VALIDATION]
    echantillon = (apprentissage.groupby(["annee", "type_bien"], group_keys=False)
                   .sample(frac=PART_ECHANTILLON, random_state=commun.GRAINE))
    appartements = validation["type_bien"] == "appartement"
    seuil_q4 = validation.loc[appartements, "prix_n1"].quantile(0.75)

    assert ANNEE_VALIDATION < min(commun.ANNEES_JUGEES), "la validation ne doit jamais être une année jugée"
    assert set(variables) <= set(ventes.columns), "variable de H7v3 absente de ventes_h7.parquet"
    assert echantillon[variables].notna().all().all() and validation[variables].notna().all().all()
    return {"echantillon": echantillon, "validation": validation, "variables": variables,
            "q4": appartements & (validation["prix_n1"] >= seuil_q4), "seuil_q4": seuil_q4}


# =============================================================================
# 2. UN RÉSEAU, UNE CONFIGURATION
# =============================================================================
class ReseauEssai(ReseauKeras):
    """Le réseau du gabarit, avec une perte au choix ('mse', 'mae' ou 'huber').

    Seule la compilation change : le réseau est construit par le gabarit, puis recompilé avec la perte
    demandée et le même optimiseur. La perte s'applique à la cible du gabarit (log puis standardisation).
    """
    perte = "mse"

    def _construire(self):
        modele = super()._construire()
        perte = keras.losses.Huber() if self.perte == "huber" else self.perte
        modele.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=perte)
        return modele


def _un_reseau(config, graine, apprentissage, validation, variables):
    """Entraîne un réseau et renvoie sa prévision en €/m² sur la validation.

    cible 'prix' : le réseau apprend le prix au m² (comme H7).
    cible 'evolution' : il apprend le rapport prix de la vente / prix médian N-1 du sous-bassin ;
    la prévision est prix N-1 x rapport prévu (le niveau du territoire vient du prix N-1).
    """
    colonnes = variables + ["annee"]
    evolution = config["cible"] == "evolution"
    y = apprentissage["prix_m2"] / apprentissage["prix_n1"] if evolution else apprentissage["prix_m2"]
    reglages = {k: config[k] for k in HYPERPARAMETRES}
    reglages["couches"] = tuple(reglages["couches"])
    reseau = ReseauEssai(graine=graine, **reglages)
    reseau.perte = config["perte"]
    reseau.fit(apprentissage[colonnes], y, sample_weight=apprentissage["w_h"])
    prevision = reseau.predict(validation[colonnes])
    return prevision * validation["prix_n1"] if evolution else prevision


def prevoir(config, graine, d):
    """Prévision de toutes les ventes de 2022 : un réseau commun, ou un réseau par type de bien."""
    if config["organisation"] == "commun":
        return _un_reseau(config, graine, d["echantillon"], d["validation"], d["variables"])
    morceaux = [_un_reseau(config, graine,
                           d["echantillon"][d["echantillon"]["type_bien"] == t],
                           d["validation"][d["validation"]["type_bien"] == t], d["variables"])
                for t in TYPES]
    return pd.concat(morceaux).reindex(d["validation"].index)


def mesurer(prevision, d):
    """MAE par vente pondérée par w_h : toutes les ventes, par type de bien, Q4 des appartements."""
    v = d["validation"]
    erreur, w = (v["prix_m2"] - prevision).abs(), v["w_h"]
    mesures = {"mae_tous": np.average(erreur, weights=w)}
    for t in TYPES:
        m = v["type_bien"] == t
        mesures[f"mae_{t}"] = np.average(erreur[m], weights=w[m])
    mesures["mae_q4_appartement"] = np.average(erreur[d["q4"]], weights=w[d["q4"]])
    return mesures


# =============================================================================
# 3. ESSAIS, TRACE ET CHOIX
# =============================================================================
def lancer(etape, candidats, graines, d, essais):
    """Entraîne chaque candidat avec chaque graine ; ajoute une ligne par essai et un run MLflow."""
    for nom, config in candidats:
        for graine in graines:
            debut = time.time()
            mesures = mesurer(prevoir(config, graine, d), d)
            minutes = (time.time() - debut) / 60
            essais.append({"etape": etape, "candidat": nom, "graine": graine,
                           **{k: str(v) for k, v in config.items()}, **mesures, "minutes": minutes})
            print(f"{nom:<34} graine {graine} | toutes : {mesures['mae_tous']:.1f} | appartements : "
                  f"{mesures['mae_appartement']:.1f} | maisons : {mesures['mae_maison']:.1f} | "
                  f"Q4 appartements : {mesures['mae_q4_appartement']:.1f} €/m² ({minutes:.1f} min)", flush=True)
            if AVEC_MLFLOW:
                tracer_mlflow(etape, nom, graine, config, {**mesures, "minutes": minutes})


def tracer_mlflow(etape, nom, graine, config, mesures):
    """Un run MLflow par essai : configuration en paramètres, mesures en métriques (option --mlflow)."""
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(commun.DATA_OUT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIENCE)
    with mlflow.start_run(run_name=f"étape {etape} | {nom} | graine {graine}"):
        mlflow.set_tags({"etape": str(etape), "candidat": nom})
        mlflow.log_params({**{k: str(v) for k, v in config.items()}, "graine": graine,
                           "annee_validation": ANNEE_VALIDATION, "part_echantillon": PART_ECHANTILLON})
        mlflow.log_metrics(mesures)


def choisir(essais_etape, candidats, reference, critere, garde_fou):
    """Règle de choix, sur la moyenne du critère sur les graines.

    Un candidat « gagne » si son gain sur la référence dépasse la variation due à la graine (plus grand
    écart-type du critère) et si le garde-fou n'est pas dégradé de plus que sa propre variation.
    Parmi les gagnants, on garde le plus simple (premier dans l'ordre des candidats) dont l'écart avec
    le meilleur gagnant ne dépasse pas la variation due à la graine. Sans gagnant, la référence est gardée.
    Seuls les candidats passés avec toutes les graines comptent.
    """
    complets = essais_etape.groupby("candidat").filter(lambda g: len(g) == len(GRAINES))
    bilan = complets.groupby("candidat")[[critere, garde_fou]].agg(["mean", "std"])
    hasard = bilan[(critere, "std")].max()
    hasard_garde = bilan[(garde_fou, "std")].max()
    moyenne, moyenne_garde = bilan[(critere, "mean")], bilan[(garde_fou, "mean")]

    admis = moyenne_garde <= moyenne_garde[reference] + hasard_garde
    gain = moyenne[reference] - moyenne
    gagnants = moyenne[admis & (gain > hasard)]
    if gagnants.empty:
        return reference, (f"aucun candidat ne gagne plus que la variation due à la graine "
                           f"({hasard:.1f} €/m²) sans dégrader le garde-fou : « {reference} » est gardé")
    ordre = [nom for nom, _ in candidats if nom in gagnants.index]
    meilleur = next(nom for nom in ordre if gagnants[nom] <= gagnants.min() + hasard)
    return meilleur, (f"« {meilleur} » gagne {gain[meilleur]:.1f} €/m² sur « {reference} » ({critere}), plus "
                      f"que la variation due à la graine ({hasard:.1f}) ; garde-fou {garde_fou} : "
                      f"{moyenne_garde[meilleur]:.1f} contre {moyenne_garde[reference]:.1f}")


def afficher(essais_etape, naive):
    """Moyenne des mesures sur les graines, par candidat, avec la naïve en première ligne."""
    colonnes = ["mae_tous", "mae_appartement", "mae_maison", "mae_q4_appartement"]
    bilan = essais_etape.groupby("candidat", sort=False)[colonnes].mean()
    bilan["variation selon la graine"] = essais_etape.groupby("candidat", sort=False)["mae_tous"].std()
    bilan["graines"] = essais_etape.groupby("candidat", sort=False).size()
    bilan = pd.concat([pd.DataFrame([naive], index=["naïve (prix N-1)"]), bilan])
    print("\nMAE par vente, validation 2022 (€/m², moyenne sur les graines) :")
    print(bilan.round(1).to_string())


# =============================================================================
# 4. LES ÉTAPES
# =============================================================================
def etape_1(depart, d, essais):
    """Appartements : H-a (évolution), H-b (un réseau par type de bien), puis les deux."""
    candidats = [("actuel (prix, commun)", depart),
                 ("H-a évolution", {**depart, "cible": "evolution"}),
                 ("H-b un réseau par type", {**depart, "organisation": "separe"}),
                 ("H-a + H-b", {**depart, "cible": "evolution", "organisation": "separe"})]
    lancer(1, candidats, GRAINES, d, essais)
    return candidats, "actuel (prix, commun)", "mae_appartement", "mae_maison"


def etape_2(depart, d, essais):
    """Fonction de perte : MSE (actuelle), MAE, Huber."""
    reference = f"perte {depart['perte']} (retenu)"
    candidats = [(reference, depart)] + [(f"perte {p}", {**depart, "perte": p})
                                         for p in ["mse", "mae", "huber"] if p != depart["perte"]]
    lancer(2, candidats, GRAINES, d, essais)
    return candidats, reference, "mae_tous", "mae_appartement"


def etape_3(depart, d, essais):
    """Hyperparamètres : grille complète avec une graine, puis contrôle des meilleures avec les autres graines.

    Les réglages actuels font partie de la grille : ils servent de référence et ne sont pas lancés deux fois.
    """
    reference = "réglages retenus"
    actuels = {k: depart[k] for k in GRILLE}
    combinaisons = [c for c in ParameterGrid(GRILLE) if c != actuels]
    candidats = [(reference, depart)] + [
        (f"dropout {c['dropout']} | lr {c['learning_rate']} | emb {c['dim_embedding']}", {**depart, **c})
        for c in combinaisons]

    debut_etape = len(essais)
    lancer(3, candidats, GRAINES[:1], d, essais)
    premiers = pd.DataFrame(essais[debut_etape:]).sort_values("mae_tous")
    meilleurs = [n for n in premiers["candidat"] if n != reference][:N_MEILLEURS]
    config_de = dict(candidats)
    lancer(3, [(n, config_de[n]) for n in [reference] + meilleurs], GRAINES[1:], d, essais)
    return candidats, reference, "mae_tous", "mae_appartement"


ETAPES = {1: etape_1, 2: etape_2, 3: etape_3}


def lire_etat():
    with open(FICHIER_ETAT, encoding="utf-8") as f:
        return json.load(f)


def ecrire(etat, essais):
    """Écrit l'état (configuration retenue + décisions), les réglages pour H7 et le tableau des essais."""
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in etat["config"].items()}
    etat["config"] = config
    for h in etat["historique"]:
        if "config" in h:                        # absente des états écrits par une version antérieure
            h["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in h["config"].items()}
    with open(FICHIER_ETAT, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=1)
    with open(FICHIER_REGLAGES, "w", encoding="utf-8") as f:
        json.dump({"reglages": config, "etapes_faites": [h["etape"] for h in etat["historique"]],
                   "annee_validation": ANNEE_VALIDATION, "version_reglee": "H7v3"},
                  f, ensure_ascii=False, indent=1)
    # Tableau des essais : on garde ceux des étapes précédentes, on remplace ceux de cette étape et des suivantes
    etape = etat["historique"][-1]["etape"]
    tableau = pd.DataFrame(essais)
    if FICHIER_ESSAIS.exists() and etape != 1:
        anciens = pd.read_csv(FICHIER_ESSAIS)
        tableau = pd.concat([anciens[anciens["etape"] < etape], tableau], ignore_index=True)
    tableau.to_csv(FICHIER_ESSAIS, index=False)


# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    assert "--etape" in sys.argv, "préciser l'étape : python optimisation_keras.py --etape 1"
    etape = int(sys.argv[sys.argv.index("--etape") + 1])
    assert etape in ETAPES, f"étape inconnue : {etape} (1, 2 ou 3)"

    # Étape 1 : on part de H7v3 tel quel ; étapes suivantes : de ce que l'étape précédente a retenu.
    # Relancer une étape efface ses anciens résultats et ceux des étapes suivantes.
    if etape == 1:
        etat = {"config": DEPART, "historique": []}
    else:
        assert FICHIER_ETAT.exists(), "lancer d'abord l'étape 1"
        etat = lire_etat()
        gardees = [h for h in etat["historique"] if h["etape"] < etape]
        assert [h["etape"] for h in gardees] == list(range(1, etape)), \
            f"étapes faites : {[h['etape'] for h in etat['historique']]} ; lancer l'étape {etape - 1} avant"
        effacees = [h["etape"] for h in etat["historique"] if h["etape"] >= etape]
        if effacees:
            print(f"Relance : les résultats des étapes {effacees} sont remplacés", flush=True)
        if "config" in gardees[-1]:
            config = gardees[-1]["config"]
        else:
            # état écrit par une version antérieure du script : la configuration n'est connue que pour la
            # dernière étape faite
            assert not effacees, "état d'une version antérieure du script : relancer depuis l'étape 1"
            config = etat["config"]
            gardees[-1]["config"] = config       # complété : une relance ultérieure repartira de là
        etat = {"config": config, "historique": gardees}

    debut = time.time()
    d = charger()
    print(f"Étape {etape} | {len(d['variables'])} variables | apprentissage : {len(d['echantillon'])} ventes "
          f"({ANNEE_DEBUT}-{ANNEE_VALIDATION - 1}) | validation {ANNEE_VALIDATION} : {len(d['validation'])} ventes "
          f"| Q4 des appartements : prix N-1 de {d['seuil_q4']:,.0f} €/m² ou plus", flush=True)
    print(f"Configuration de départ : {etat['config']}\n", flush=True)

    essais = []
    candidats, reference, critere, garde_fou = ETAPES[etape](etat["config"], d, essais)
    essais_etape = pd.DataFrame(essais)
    afficher(essais_etape, mesurer(d["validation"]["prix_n1"], d))

    retenu, decision = choisir(essais_etape, candidats, reference, critere, garde_fou)
    etat["config"] = dict(candidats)[retenu]
    etat["historique"].append({"etape": etape, "retenu": retenu, "critere": critere,
                               "garde_fou": garde_fou, "decision": decision, "config": etat["config"]})
    ecrire(etat, essais)
    print(f"\nDécision (critère {critere}, garde-fou {garde_fou}) : {decision}")
    print(f"Configuration retenue : {etat['config']}")
    print(f"Exporté : optimisation_keras_etat.json, meilleurs_reglages_keras.json, optimisation_keras_essais.csv "
          f"({(time.time() - debut) / 60:.1f} min)")