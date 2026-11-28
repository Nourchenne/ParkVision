"""Découpage train / valid / test par caméra (et non par jour).

`prepare_split.py` répartit les images par journée : toutes les caméras apparaissent alors
dans les trois lots, donc les scores mesurent la performance sur des points de vue déjà vus.
Ce script-ci tient à l'écart des caméras entières : un groupe pour le test, un pour la
validation, le reste pour l'entraînement.

Il s'appuie sur `data/processed/camera_map.csv` (produit par `cluster_cameras.py`) et écrit
un NOUVEAU manifeste, `data/processed/camera_split_manifest.csv`. Il ne touche ni
`split_manifest.csv`, ni `data/processed/yolo/`, ni la liste de test existante : tous les
résultats publiés restent reproductibles.

    python src/data/prepare_camera_split.py --train 0 --valid 2 --test 1
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
CAMERA_MAP = ROOT / "data" / "processed" / "camera_map.csv"
OUTPUT = ROOT / "data" / "processed" / "camera_split_manifest.csv"

FIELDS = ["image", "date", "cluster", "split"]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assign(manifest: list[dict], camera_map: list[dict], train: set[int], valid: set[int],
           test: set[int]) -> list[dict]:
    """Associe chaque image de `manifest` à un split selon son groupe-caméra."""
    if train & valid or train & test or valid & test:
        raise ValueError("un groupe ne peut appartenir qu'à un seul split")
    if not (train and valid and test):
        raise ValueError("train, valid et test doivent contenir au moins un groupe")
    cluster_of = {row["image"]: int(row["cluster"]) for row in camera_map}
    missing = [row["image"] for row in manifest if row["image"] not in cluster_of]
    if missing:
        raise ValueError(f"{len(missing)} images du manifeste absentes de camera_map.csv (ex. {missing[0]})")
    known = set(cluster_of.values())
    unused = known - train - valid - test
    if unused:
        raise ValueError(f"groupes non affectés : {sorted(unused)}")
    unknown = (train | valid | test) - known
    if unknown:
        raise ValueError(f"groupes inconnus dans camera_map.csv : {sorted(unknown)}")
    result = []
    for row in manifest:
        cluster = cluster_of[row["image"]]
        split = "train" if cluster in train else "valid" if cluster in valid else "test"
        result.append({"image": row["image"], "date": row["date"], "cluster": cluster, "split": split})
    return result


def write(rows: list[dict], output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_groups(text: str) -> set[int]:
    return {int(part) for part in text.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="0", help="groupe(s) d'entraînement, séparés par des virgules")
    parser.add_argument("--valid", default="2", help="groupe(s) de validation")
    parser.add_argument("--test", default="1", help="groupe(s) de test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows = assign(read_rows(MANIFEST), read_rows(CAMERA_MAP),
                  parse_groups(args.train), parse_groups(args.valid), parse_groups(args.test))
    write(rows, args.output)
    counts = Counter(row["split"] for row in rows)
    print(f"{len(rows)} images -> {args.output}")
    for split in ("train", "valid", "test"):
        clusters = sorted({row["cluster"] for row in rows if row["split"] == split})
        days = len({row["date"] for row in rows if row["split"] == split})
        print(f"  {split:<5} : {counts[split]:>5} images, {days:>2} jours, groupe(s) {clusters}")


if __name__ == "__main__":
    main()
