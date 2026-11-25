# =============================================================================
# extraire_dvf_plus.py : ventes d'une année et de quelques mois extraites d'une livraison DVF+ du Cerema (SQL)
# =============================================================================
# Étapes 1 et 2 de 04_outils/mettre_a_jour_dvf.sh (qui l'appelle ; on peut aussi le lancer seul).
# La livraison du Cerema (hors dépôt, plusieurs Go) est un zip de sauvegardes
# PostgreSQL en texte (pg_dump), une archive 7z par région, contenant un fichier .sql par département.
# Aucune base de données n'est nécessaire : on lit le bloc « COPY ... dvf_plus_mutation » de chaque
# fichier, on garde les mutations de l'année demandée (2026, janvier à juin par défaut) et les colonnes utiles.
#
# Prérequis :  pip install py7zr
# Lancement (un exemple d'abord, puis tout) :
#   python 04_outils/extraire_dvf_plus.py extraire livraison.zip dossier_sortie --region R94_Corse
#   python 04_outils/extraire_dvf_plus.py extraire livraison.zip dossier_sortie --annee 2026 --mois-max 6
#   python 04_outils/extraire_dvf_plus.py assembler dossier_sortie
# « extraire » écrit un fichier par région dans le dossier de sortie et saute les régions déjà faites
# (on peut l'interrompre et le relancer). « assembler » réunit les régions en un seul fichier
# dvf_plus_assemble.csv (séparateur |) et affiche le contrôle mensuel.
# Le fichier de contrôle donne, pour TOUTES les années de la livraison, le nombre de mutations par mois
# (utile pour vérifier qu'aucun mois n'est manquant, par exemple juin 2021 signalé par le correctif).

import argparse
import csv
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


# Colonnes non conservées : géométries et listes d'identifiants de parcelles ou de locaux (inutiles ici,
# et pas de parcelle dans ce que l'on garde)
EXCLURE = {"geomlocmut", "geomparmut", "geompar", "l_idpar", "l_idparmut", "l_idlocmut", "l_section"}
ENTETE_COPY = re.compile(rb"^COPY (\S+\.dvf_plus_mutation) \((.*)\) FROM stdin;")
FICHIER_DEP = re.compile(r"(^|/)dvf_plus_d\w+\.sql$")


def lire_mutations(chemin_sql, annees, mois_max, sortie_csv, ecrit_entete, compteur):
    """Parcourt un .sql, compte toutes les mutations par (année, mois), écrit celles à garder."""
    n_total = n_gardees = 0
    with open(chemin_sql, "rb") as f:
        colonnes = None
        for ligne in f:
            if colonnes is None:
                m = ENTETE_COPY.match(ligne)
                if m:
                    colonnes = [c.strip().decode() for c in m.group(2).split(b",")]
                    i_annee, i_mois = colonnes.index("anneemut"), colonnes.index("moismut")
                    gardees = [i for i, c in enumerate(colonnes) if c not in EXCLURE]
                    if ecrit_entete:
                        sortie_csv.writerow([colonnes[i] for i in gardees])
                continue
            if ligne.startswith(b"\\."):
                break                                   # fin du bloc de la table
            champs = ligne.rstrip(b"\n").split(b"\t")
            n_total += 1
            annee, mois = champs[i_annee], champs[i_mois]
            compteur[(annee.decode(), mois.decode())] += 1
            if annee.isdigit() and int(annee) in annees and mois.isdigit() and int(mois) <= mois_max:
                sortie_csv.writerow(["" if champs[i] == b"\\N" else champs[i].decode("utf-8") for i in gardees])
                n_gardees += 1
    if colonnes is None:
        raise SystemExit(f"{chemin_sql.name} : bloc COPY de la table dvf_plus_mutation introuvable")
    return n_total, n_gardees


def archives_par_region(source, filtre):
    """Renvoie {région: [(nom, ouvrir)]} : les volumes .7z.00N de chaque région, dans le zip ou un dossier."""
    regions = {}
    if Path(source).is_file():
        z = zipfile.ZipFile(source)
        for nom in sorted(z.namelist()):
            p = Path(nom)
            if "sql" in p.parts and ".7z." in p.name:
                regions.setdefault(p.parent.name, []).append((nom, lambda n=nom: z.open(n)))
    else:
        for p in sorted(Path(source).rglob("*")):
            if p.is_file() and ".7z." in p.name:
                regions.setdefault(p.parent.name, []).append((p.name, lambda q=p: open(q, "rb")))
    return {r: v for r, v in regions.items() if not filtre or filtre in r}


def extraire_region(region, volumes, annees, mois_max, dossier_sortie):
    import py7zr                       # nécessaire à cette étape seulement (assembler n'en a pas besoin)
    sortie = dossier_sortie / f"mutations_{region}.csv"
    controle = dossier_sortie / f"controle_{region}.csv"
    if sortie.exists() and controle.exists():
        print(f"[{region}] déjà fait, sauté")
        return
    compteur = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. Réunir les volumes (.001, .002 ...) en une archive
        externe = tmp / "externe.7z"
        with open(externe, "wb") as dst:
            for _, ouvrir in volumes:
                with ouvrir() as src:
                    shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
        with py7zr.SevenZipFile(externe) as a:
            a.extractall(tmp / "niveau1")
        externe.unlink()
        # 2. L'archive contient en général une seconde archive 7z, qui contient les .sql
        internes = list((tmp / "niveau1").rglob("*.7z"))
        archive = internes[0] if internes else None
        if archive is None:
            raise SystemExit(f"[{region}] pas d'archive interne trouvée dans {tmp / 'niveau1'}")
        with py7zr.SevenZipFile(archive) as a:
            noms = [n for n in a.getnames() if FICHIER_DEP.search(n)]
        n_tot = n_keep = 0
        with open(sortie.with_suffix(".tmp"), "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo, delimiter="|", lineterminator="\n")
            premier = True
            for nom in sorted(noms):
                with py7zr.SevenZipFile(archive) as a:      # une extraction à la fois, sur disque temporaire
                    a.extract(path=tmp / "sql", targets=[nom])
                chemin = tmp / "sql" / nom
                t, k = lire_mutations(chemin, annees, mois_max, w, premier, compteur)
                premier = False
                n_tot += t
                n_keep += k
                chemin.unlink()
                print(f"[{region}] {Path(nom).name} : {t} mutations, {k} gardées")
    sortie.with_suffix(".tmp").rename(sortie)
    with open(controle, "w", newline="", encoding="utf-8") as fc:
        w = csv.writer(fc)
        w.writerow(["anneemut", "moismut", "n_mutations"])
        for (a, m), n in sorted(compteur.items()):
            w.writerow([a, m, n])
    print(f"[{region}] terminé : {n_tot} mutations lues, {n_keep} gardées")


def assembler(dossier):
    dossier = Path(dossier)
    parts = sorted(dossier.glob("mutations_*.csv"))
    if not parts:
        raise SystemExit("aucun fichier mutations_*.csv à assembler")
    sortie = dossier / "dvf_plus_assemble.csv"
    n = 0
    ids = set()
    doublons = 0
    with open(sortie, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="|", lineterminator="\n")
        entete = None
        for p in parts:
            with open(p, newline="", encoding="utf-8") as fi:
                lecteur = csv.reader(fi, delimiter="|")
                e = next(lecteur)
                if entete is None:
                    entete = e
                    w.writerow(e)
                    i_id = e.index("idmutation")
                assert e == entete, f"{p.name} : colonnes différentes des autres régions"
                for ligne in lecteur:
                    if ligne[i_id] in ids:
                        doublons += 1
                    ids.add(ligne[i_id])
                    w.writerow(ligne)
                    n += 1
    print(f"{n} mutations écrites dans {sortie} ({len(parts)} régions) | identifiants en double : {doublons}")
    # Contrôle mensuel, toutes années de la livraison
    total = Counter()
    for p in dossier.glob("controle_*.csv"):
        with open(p, newline="", encoding="utf-8") as fi:
            for ligne in list(csv.reader(fi))[1:]:
                total[(int(ligne[0]), int(ligne[1]))] += int(ligne[2])
    with open(dossier / "controle_mensuel_toutes_annees.csv", "w", newline="", encoding="utf-8") as fc:
        w = csv.writer(fc)
        w.writerow(["anneemut", "moismut", "n_mutations"])
        for (a, m), v in sorted(total.items()):
            w.writerow([a, m, v])
    print("\nMutations par mois (toute la livraison, DOM compris), quatre dernières années :")
    for a in sorted({a for a, _ in total})[-4:]:      # les quatre dernières années de la livraison
        mois = {m: total.get((a, m), 0) for m in range(1, 13)}
        print(f"  {a} :", " ".join(f"{m:>2}:{mois[m]:>7}" for m in mois))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="commande", required=True)
    e = sp.add_parser("extraire")
    e.add_argument("source", help="le zip de la livraison, ou le dossier sql déjà décompressé")
    e.add_argument("sortie", help="dossier de sortie (créé si besoin)")
    e.add_argument("--region", help="une seule région, ex. R94_Corse")
    e.add_argument("--annee", type=int, default=2026)
    e.add_argument("--mois-max", type=int, default=6, help="dernier mois gardé (6 = janvier à juin)")
    a = sp.add_parser("assembler")
    a.add_argument("sortie")
    args = p.parse_args()
    if args.commande == "extraire":
        out = Path(args.sortie)
        out.mkdir(parents=True, exist_ok=True)
        regions = archives_par_region(args.source, args.region)
        if not regions:
            sys.exit("aucune archive .7z trouvée (région inconnue ?)")
        for region, volumes in regions.items():
            extraire_region(region, volumes, {args.annee}, args.mois_max, out)
    else:
        assembler(args.sortie)
