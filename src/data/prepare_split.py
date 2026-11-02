import csv
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
SEED = 42

# Recenser les images annotées et conserver leur chemin d'origine.
images = []
excluded = 0

for original_split in ("train", "valid", "test"):
    folder = RAW / original_split
    for label in sorted((folder / "labels").glob("*.txt")):
        if not label.read_text(encoding="utf-8-sig").strip():
            excluded += 1
            continue

        image = folder / "images" / (label.stem + ".jpg")
        if not image.is_file():
            raise FileNotFoundError(image)

        day = date.fromisoformat(label.name[:10])
        images.append((image.relative_to(ROOT).as_posix(), day))

if not images:
    raise ValueError("Aucune image annotée trouvée dans data/raw.")

# Mélanger les journées une seule fois, avec une graine fixe.
days = sorted({day for _, day in images})
random.Random(SEED).shuffle(days)
end_train = round(0.70 * len(days))
end_valid = round(0.90 * len(days))

groups = {
    "train": days[:end_train],
    "valid": days[end_train:end_valid],
    "test": days[end_valid:],
}
split_by_day = {
    day: split for split, group in groups.items() for day in group
}

# Écrire le manifeste et afficher le bilan.
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
counts = dict.fromkeys(groups, 0)

with MANIFEST.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "date", "split"])
    for image, day in images:
        split = split_by_day[day]
        writer.writerow([image, day.isoformat(), split])
        counts[split] += 1

for split, group in groups.items():
    print(f"{split} : {len(group)} jours | {counts[split]} images")
print(f"Total retenu : {len(images)} images | Exclues : {excluded}")