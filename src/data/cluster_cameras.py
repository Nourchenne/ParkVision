"""Retrouve la caméra de chaque image par regroupement (k-means) sur l'arrière-plan.

PKLot a été filmé par quelques caméras fixes, mais l'export Roboflow ne garde aucun
identifiant de caméra : les fichiers sont nommés par horodatage seulement. Une caméra fixe
donne un arrière-plan quasi identique, donc on regroupe les images sur leur apparence :

1. chaque image de `data/processed/split_manifest.csv` est lue en niveaux de gris et réduite
   à 64x36 ;
2. chaque vecteur est centré et normalisé (retire la luminosité du jour, de la nuit, du
   soleil) ;
3. k-means (k-means++, plusieurs départs, graine fixe) répartit les images en k groupes ;
4. un même jour est filmé par une seule caméra : chaque image reçoit le groupe majoritaire de
   sa journée (vote par jour, désactivable avec --no-day-vote). Cela corrige quelques images
   difficiles que le k-means attribue au mauvais groupe.

Écrit `data/processed/camera_map.csv` (colonnes `image,cluster`, numéros triés du plus
grand groupe au plus petit) et, avec --samples, quelques images par groupe pour contrôle
visuel. Lecture seule sur les images ; ne modifie ni le manifeste ni les labels.

    python src/data/cluster_cameras.py --k 3 --samples reports/camera_split/samples
"""
import argparse
import csv
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
OUTPUT = ROOT / "data" / "processed" / "camera_map.csv"
SIZE = (64, 36)  # largeur, hauteur
SEED = 42


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_vector(path: Path, size: tuple[int, int] = SIZE, normalize: bool = True) -> np.ndarray:
    """Image en niveaux de gris, réduite, aplatie ; centrée et normalisée si demandé."""
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if image is None:
        raise FileNotFoundError(path)
    vector = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    if normalize:
        vector -= vector.mean()
        deviation = vector.std()
        if deviation > 1e-6:
            vector /= deviation
    return vector


def load_features(rows: list[dict], root: Path = ROOT, normalize: bool = True) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=8) as pool:
        vectors = list(pool.map(lambda row: load_vector(root / row["image"], normalize=normalize), rows))
    return np.stack(vectors)


def kmeans(data: np.ndarray, k: int, seed: int = SEED, restarts: int = 10,
           iterations: int = 100) -> tuple[np.ndarray, float]:
    """k-means++ avec plusieurs départs ; renvoie (étiquettes, inertie) du meilleur."""
    best_labels, best_inertia = None, np.inf
    squared_norms = (data ** 2).sum(axis=1)
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        centers = [data[rng.integers(len(data))]]
        for _ in range(1, k):
            distances = np.min([squared_norms - 2 * data @ c + (c ** 2).sum() for c in centers], axis=0)
            distances = np.clip(distances, 0, None)
            centers.append(data[rng.choice(len(data), p=distances / distances.sum())])
        centers = np.stack(centers)
        labels = np.full(len(data), -1)
        for _ in range(iterations):
            distances = squared_norms[:, None] - 2 * data @ centers.T + (centers ** 2).sum(axis=1)[None, :]
            updated = distances.argmin(axis=1)
            if np.array_equal(updated, labels):
                break
            labels = updated
            for cluster in range(k):
                members = data[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
        inertia = float(np.take_along_axis(distances, labels[:, None], axis=1).sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels.copy(), inertia
    return best_labels, best_inertia


def order_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumérote les groupes : 0 = le plus grand."""
    order = [cluster for cluster, _ in Counter(labels.tolist()).most_common()]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[label] for label in labels.tolist()])


def annotated_places(image: Path) -> int:
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    return sum(1 for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def summarize(rows: list[dict], labels: np.ndarray, root: Path = ROOT) -> list[dict]:
    clusters = defaultdict(list)
    for row, label in zip(rows, labels.tolist(), strict=True):
        clusters[label].append(row)
    summary = []
    for cluster in sorted(clusters):
        members = clusters[cluster]
        days = defaultdict(set)
        for row in members:
            days[row["date"]].add(row["image"])
        # Pureté par jour : un jour est filmé par une seule caméra, donc ses images doivent
        # presque toutes tomber dans le même groupe. On compte les jours dont c'est le cas.
        places = [annotated_places(root / row["image"]) for row in members]
        summary.append({
            "cluster": cluster, "images": len(members), "jours": len(days),
            "du": min(days), "au": max(days),
            "places_mediane": statistics.median(places),
            "places_min": min(places), "places_max": max(places),
        })
    return summary


def day_purity(rows: list[dict], labels: np.ndarray) -> float:
    """Part des images dont le groupe est le groupe majoritaire de leur journée."""
    by_day = defaultdict(list)
    for row, label in zip(rows, labels.tolist(), strict=True):
        by_day[row["date"]].append(label)
    agree = sum(Counter(values).most_common(1)[0][1] for values in by_day.values())
    return agree / len(rows)


def day_vote(rows: list[dict], labels: np.ndarray) -> np.ndarray:
    """Donne à chaque image le groupe majoritaire de sa journée."""
    by_day = defaultdict(list)
    for row, label in zip(rows, labels.tolist(), strict=True):
        by_day[row["date"]].append(label)
    winner = {day: Counter(values).most_common(1)[0][0] for day, values in by_day.items()}
    return np.array([winner[row["date"]] for row in rows])


def lot_of(places: int) -> int:
    """Parking déduit du nombre de places annotées (28, 40 ou 100) : contrôle, jamais utilisé pour regrouper."""
    return 28 if places <= 34 else 40 if places <= 70 else 100


def lot_agreement(rows: list[dict], labels: np.ndarray, root: Path = ROOT) -> float:
    """Part des images dont le parking (d'après le nombre de places annotées) est celui de leur groupe."""
    lots = defaultdict(list)
    for row, label in zip(rows, labels.tolist(), strict=True):
        lots[label].append(lot_of(annotated_places(root / row["image"])))
    agree = sum(Counter(values).most_common(1)[0][1] for values in lots.values())
    return agree / len(rows)


def write_map(rows: list[dict], labels: np.ndarray, output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "cluster"])
        for row, label in zip(rows, labels.tolist(), strict=True):
            writer.writerow([row["image"], label])


def write_samples(rows: list[dict], labels: np.ndarray, directory: Path, per_cluster: int, root: Path = ROOT,
                  seed: int = SEED) -> None:
    """Quelques images de jour par groupe, au hasard (graine fixe), réduites à 640x360."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for cluster in sorted(set(labels.tolist())):
        members = [row for row, label in zip(rows, labels.tolist(), strict=True) if label == cluster]
        for index, choice in enumerate(rng.choice(len(members), size=min(per_cluster, len(members)), replace=False)):
            image = cv2.imread(str(root / members[choice]["image"]))
            cv2.imwrite(str(directory / f"cluster{cluster}_{index + 1}.jpg"),
                        cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="nombre de groupes (défaut 3, les 3 parkings de PKLot)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--samples", type=Path, help="dossier où écrire des images d'exemple par groupe")
    parser.add_argument("--per-cluster", type=int, default=8)
    parser.add_argument("--no-day-vote", action="store_true", help="garder les groupes bruts du k-means")
    parser.add_argument("--no-normalize", action="store_true", help="ne pas centrer/normaliser chaque image")
    args = parser.parse_args()

    rows = read_manifest()
    print(f"{len(rows)} images à lire…")
    data = load_features(rows, normalize=not args.no_normalize)
    labels, inertia = kmeans(data, args.k)
    labels = order_by_size(labels)
    print(f"k-means brut : k={args.k}, inertie {inertia:.0f}, pureté par jour {day_purity(rows, labels):.1%}, "
          f"accord avec le parking d'après les places annotées {lot_agreement(rows, labels):.1%}")
    if not args.no_day_vote:
        changed = int((day_vote(rows, labels) != labels).sum())
        labels = day_vote(rows, labels)
        print(f"vote par jour : {changed} images changent de groupe, accord avec les places annotées "
              f"{lot_agreement(rows, labels):.1%}")
    write_map(rows, labels, args.output)
    print(f"-> {args.output}")
    for item in summarize(rows, labels):
        print(f"  groupe {item['cluster']} : {item['images']:>5} images, {item['jours']:>2} jours "
              f"({item['du']} à {item['au']}), places annotées {item['places_min']}-{item['places_max']} "
              f"(médiane {item['places_mediane']:g})")
    if args.samples:
        write_samples(rows, labels, args.samples, args.per_cluster)
        print(f"exemples -> {args.samples}")


if __name__ == "__main__":
    main()
