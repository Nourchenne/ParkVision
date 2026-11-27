"""Reconstruit le matériel de démonstration à partir du lot test de PKLot.

Tout vient de `data/processed/yolo/test.txt` (les 1 029 images jamais utilisées pour
l'entraînement ni la validation). Le script :

1. copie 8 photos variées dans `reports/demo/images/` et écrit `selection.csv`
   (vérité terrain lue dans les fichiers de labels) ;
2. assemble toutes les images d'une même journée, triées par heure, en une vidéo
   H.264 à 4 images par seconde (`demo_<date>.mp4`) ;
3. vérifie que la vidéo respecte les limites de l'API (`src/api/config.py`).

Prérequis : `dvc pull -r dagshub` (ou données déjà extraites) et un exécutable ffmpeg.

    python reports/demo/build_demo_assets.py --ffmpeg C:\\chemin\\vers\\ffmpeg.exe

Sans `--ffmpeg`, le script cherche la variable FFMPEG_BIN puis `ffmpeg` dans le PATH.

Les journées 2012-10-30 et 2012-11-07, bien qu'elles fassent partie du lot test, sont
volontairement écartées : leurs labels déclarent « libre » toutes les places alors que les
images montrent un parking plein.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
YOLO = ROOT / "data" / "processed" / "yolo"

VIDEO_DATE = "2013-03-15"
VIDEO_FPS = 4

# Limites de l'API (valeurs par défaut de src/api/config.py)
MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 420
MAX_FRAMES = 450
MAX_WIDTH, MAX_HEIGHT = 1280, 720

# (début du nom de fichier, ce que l'image montre)
PHOTOS = [
    ("2012-09-14_09_31_12", "Grand parking (100 places) presque plein, plein soleil"),
    ("2012-09-14_13_01_21", "Même parking à occupation moyenne en milieu de journée"),
    ("2012-09-14_13_16_21", "Lumière très forte : image surexposée, voitures délavées"),
    ("2012-09-14_18_06_32", "Fin de journée, lumière faible"),
    ("2012-10-13_06_58_36", "Grand parking quasi vide au petit matin"),
    ("2013-03-15_12_00_07", "Parking de 40 places presque plein (autre caméra)"),
    ("2013-04-13_13_30_08", "Parking de 40 places peu occupé, temps couvert"),
    ("2013-03-15_18_30_14", "Crépuscule, lampadaires allumés, parking de 40 places"),
]


def lire_liste_test():
    chemins = [ligne.strip() for ligne in (YOLO / "test.txt").read_text(encoding="utf-8").splitlines() if ligne.strip()]
    images = [(YOLO / chemin).resolve() for chemin in chemins]
    manquantes = [str(i) for i in images if not i.is_file()]
    if manquantes:
        sys.exit(f"Données absentes ({len(manquantes)} images, ex. {manquantes[0]}) : lancer dvc pull -r dagshub.")
    return images


def comptes_verite(image):
    """(libres, occupées) lues dans le fichier de labels YOLO de l'image."""
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    libres = occupees = 0
    for ligne in label.read_text(encoding="utf-8").splitlines():
        morceaux = ligne.split()
        if morceaux:
            if int(morceaux[0]) == 0:
                libres += 1
            else:
                occupees += 1
    return libres, occupees


def copier_photos(images_test):
    dossier = DEMO / "images"
    dossier.mkdir(parents=True, exist_ok=True)
    for ancien in dossier.glob("*.jpg"):
        ancien.unlink()
    lignes = []
    for debut, scene in PHOTOS:
        trouvees = [i for i in images_test if i.name.startswith(debut)]
        assert len(trouvees) == 1, f"{debut} : {len(trouvees)} image(s) dans le lot test"
        source = trouvees[0]
        shutil.copy2(source, dossier / source.name)
        libres, occupees = comptes_verite(source)
        lignes.append({
            "image": source.name, "date": debut[:10], "heure": debut[11:].replace("_", ":"),
            "scene": scene, "places": libres + occupees,
            "libres_annotees": libres, "occupees_annotees": occupees,
            "libres_predites": "", "occupees_predites": "",
        })
    with (dossier / "selection.csv").open("w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"{len(lignes)} photos copiées dans {dossier.relative_to(ROOT)}")


def trouver_ffmpeg(argument):
    candidat = argument or os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not candidat or not Path(candidat).is_file() and not shutil.which(candidat):
        sys.exit("ffmpeg introuvable : passer --ffmpeg <chemin> ou définir FFMPEG_BIN.")
    return candidat


def construire_video(images_test, ffmpeg):
    du_jour = sorted((i for i in images_test if i.name.startswith(VIDEO_DATE)), key=lambda i: i.name)
    assert du_jour, f"aucune image test pour {VIDEO_DATE}"
    sortie = DEMO / f"demo_{VIDEO_DATE}.mp4"
    with tempfile.TemporaryDirectory() as sequence:
        for numero, image in enumerate(du_jour):
            shutil.copy2(image, Path(sequence) / f"frame_{numero:04d}.jpg")
        commande = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(VIDEO_FPS),
                    "-i", str(Path(sequence) / "frame_%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(sortie)]
        subprocess.run(commande, check=True)
    print(f"{len(du_jour)} images de {VIDEO_DATE} ({du_jour[0].name[11:19]} à {du_jour[-1].name[11:19]}) -> {sortie.relative_to(ROOT)}")
    return sortie, len(du_jour)


def verifier_limites(video, images):
    capture = cv2.VideoCapture(str(video))
    largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lues = 0
    while capture.read()[0]:
        lues += 1
    capture.release()
    duree = lues / VIDEO_FPS
    taille = video.stat().st_size
    print(f"Vidéo : {largeur}x{hauteur}, {lues} images, {duree:.0f} s, {taille / 1e6:.1f} Mo")
    assert lues == images, f"{lues} images lues au lieu de {images}"
    assert largeur <= MAX_WIDTH and hauteur <= MAX_HEIGHT, "résolution au-dessus de la limite de l'API"
    assert taille <= MAX_BYTES, "fichier trop gros pour l'API"
    assert duree <= MAX_SECONDS and lues <= MAX_FRAMES, "durée ou nombre d'images au-dessus de la limite de l'API"
    print("Limites de l'API respectées.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffmpeg", help="chemin de l'exécutable ffmpeg")
    args = parser.parse_args()

    images_test = lire_liste_test()
    print(f"Lot test : {len(images_test)} images")
    copier_photos(images_test)
    video, nombre = construire_video(images_test, trouver_ffmpeg(args.ffmpeg))
    verifier_limites(video, nombre)


if __name__ == "__main__":
    main()
