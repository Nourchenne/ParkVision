import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
OUTPUT = ROOT / "data" / "processed" / "yolo"

lists = {"train": [], "valid": [], "test": []}
seen_images = set()
split_by_day = {}

with MANIFEST.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        split = row["split"]
        day = row["date"]
        image = ROOT / row["image"]
        label = image.parent.parent / "labels" / (image.stem + ".txt")

        if split not in lists:
            raise ValueError(f"Lot inconnu : {split}")
        if image in seen_images:
            raise ValueError(f"Image dupliquée : {image}")
        if day in split_by_day and split_by_day[day] != split:
            raise ValueError(f"Journée présente dans plusieurs lots : {day}")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Image ou annotation absente : {image}")
        if not label.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Annotation vide : {label}")

        seen_images.add(image)
        split_by_day[day] = split

        # YOLO résout les chemins commençant par ./ depuis le fichier liste.
        relative_path = Path(os.path.relpath(image, OUTPUT)).as_posix()
        lists[split].append("./" + relative_path)

if any(not paths for paths in lists.values()):
    raise ValueError("Au moins un lot est vide.")

OUTPUT.mkdir(parents=True, exist_ok=True)

for split, paths in lists.items():
    content = "\n".join(sorted(paths)) + "\n"
    (OUTPUT / f"{split}.txt").write_text(content, encoding="utf-8")
    print(f"{split} : {len(paths)} images")

yaml_content = """train: train.txt
val: valid.txt
test: test.txt

nc: 2
names: ['space-empty', 'space-occupied']
"""
(OUTPUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

print("Fichiers YOLO créés dans data/processed/yolo")