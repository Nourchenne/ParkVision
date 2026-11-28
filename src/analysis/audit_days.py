"""Audit des journées : occupation annotée par jour, à partir des fichiers de labels.

Pour chaque journée du manifeste (train, valid et test confondus), calcule la part de
places annotées « occupées », puis signale les jours de semaine anormalement vides. Sert
à repérer des labels faux : un parking rempli à 59-96 % en semaine ne peut pas être vide
à 1 % un mardi ordinaire.

    python src/analysis/audit_days.py

Écrit reports/data_audit/journees.csv. Lecture seule : aucun label n'est modifié.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
OUTPUT = ROOT / "reports" / "data_audit" / "journees.csv"

OCCUPIED_CLASS = 1
# Critère : jour de semaine (lundi à vendredi) dont l'occupation annotée est sous ce seuil.
WEEKDAY_THRESHOLD = 0.10
# Vérification visuelle des jours signalés (boîtes de labels dessinées sur des images de
# chaque jour, voir reports/data_audit/verification_*.jpg). Le critère ci-dessus ne lit que
# les labels et ne peut pas, seul, distinguer un jour réellement vide d'un label faux.
VERIFICATION = {
    "2012-10-30": "labels_faux", "2012-11-06": "labels_faux", "2012-11-07": "labels_faux",
    "2012-10-12": "vide_reel", "2012-10-15": "vide_reel", "2012-12-24": "vide_reel",
    "2012-12-25": "vide_reel", "2012-12-26": "vide_reel", "2012-12-27": "vide_reel",
}

FIELDS = ["date", "jour", "split", "parking", "images", "places_par_image",
          "libres", "occupees", "occupation", "suspect", "verification", "raison"]


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / (image.stem + ".txt")


def count_label(path: Path) -> tuple[int, int]:
    """(libres, occupées) lues dans un fichier de labels YOLO."""
    free = busy = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if parts:
            if int(parts[0]) == OCCUPIED_CLASS:
                busy += 1
            else:
                free += 1
    return free, busy


def audit(manifest: Path = MANIFEST, root: Path = ROOT) -> list[dict]:
    days: dict[str, dict] = defaultdict(lambda: {"images": 0, "libres": 0, "occupees": 0, "splits": set()})
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            free, busy = count_label(label_path(root / row["image"]))
            day = days[row["date"]]
            day["images"] += 1
            day["libres"] += free
            day["occupees"] += busy
            day["splits"].add(row["split"])

    # Deux parkings : environ 100 places par image, ou environ 40. On sépare à mi-chemin.
    per_image = {d: (v["libres"] + v["occupees"]) / v["images"] for d, v in days.items()}
    middle = (max(per_image.values()) + min(per_image.values())) / 2

    result = []
    for day_name in sorted(days):
        v = days[day_name]
        places = v["libres"] + v["occupees"]
        occupation = v["occupees"] / places if places else 0.0
        weekday = date.fromisoformat(day_name).weekday()  # 0 = lundi
        suspect, reason = False, ""
        if weekday < 5 and occupation < WEEKDAY_THRESHOLD:
            suspect = True
            reason = f"jour de semaine à {occupation:.1%} d'occupation (seuil {WEEKDAY_THRESHOLD:.0%})"
        result.append({
            "date": day_name,
            "jour": ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")[weekday],
            "split": "+".join(sorted(v["splits"])),
            "parking": 100 if per_image[day_name] > middle else 40,
            "images": v["images"], "places_par_image": round(per_image[day_name], 1),
            "libres": v["libres"], "occupees": v["occupees"],
            "occupation": round(occupation, 4), "suspect": int(suspect),
            "verification": VERIFICATION.get(day_name, "non_verifie" if suspect else ""), "raison": reason,
        })
    return result


def write(rows: list[dict], output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = audit()
    write(rows, args.output)
    flagged = [r for r in rows if r["suspect"]]
    images = sum(r["images"] for r in flagged)
    print(f"{len(rows)} journées, {len(flagged)} suspectes ({images} images) -> {args.output}")
    for r in flagged:
        print(f"  {r['date']} {r['jour']:<9} {r['split']:<6} {r['images']:>4} images  {r['occupation']:.1%}  {r['verification']}")
    wrong = [r for r in rows if r["verification"] == "labels_faux"]
    print(f"Labels faux confirmés : {len(wrong)} journées, "
          + ", ".join(f"{s} {sum(r['images'] for r in wrong if r['split'] == s)} images" for s in sorted({r['split'] for r in wrong})))
    print("Médiane d'occupation des jours de semaine non suspects :",
          f"{statistics.median(r['occupation'] for r in rows if not r['suspect'] and r['jour'] not in ('samedi', 'dimanche')):.1%}")


if __name__ == "__main__":
    main()
