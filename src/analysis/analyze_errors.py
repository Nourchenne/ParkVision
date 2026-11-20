"""Analyse des erreurs d'un detecteur YOLO sur le lot valid de ParkVision.

Produit, pour un fichier de poids donne :
- comptage.csv : places libres/occupees annotees et predites, image par image ;
- comptage_resume.json : MAE par classe, seuils, effectifs, empreinte des poids ;
- erreurs_detail.csv : chaque erreur (oubli, fausse detection, double detection,
  confusion libre/occupee) avec sa position ;
- exemples/ : quelques images annotees (verite et predictions) par type d'erreur.
- evaluation_environment.json : environnement et empreintes pour comparer les runs.

Le lot valid est lu dans data/processed/split_manifest.csv (jamais data/raw/valid,
qui correspond a l'ancien decoupage). Le lot test n'est jamais ouvert.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("space-empty", "space-occupied")
MATCH_IOU = 0.5
ERROR_TYPES = ("oubli", "fausse_detection", "double_detection", "confusion")


def iou(a, b):
    """Intersection sur union de deux cadres (x1, y1, x2, y2)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def read_labels(path, width, height):
    """Lit une annotation YOLO normalisee et renvoie [(classe, (x1, y1, x2, y2))]."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Ligne d'annotation invalide dans {path} : {line!r}")
        cls = int(float(parts[0]))
        if cls not in (0, 1):
            raise ValueError(f"Classe inconnue dans {path} : {cls}")
        cx, cy, w, h = (float(value) for value in parts[1:])
        boxes.append((cls, ((cx - w / 2) * width, (cy - h / 2) * height,
                            (cx + w / 2) * width, (cy + h / 2) * height)))
    return boxes


def match_detections(truth, predictions, threshold=MATCH_IOU):
    """Apparier verite et predictions, puis classer chaque ecart.

    truth : [(classe, cadre)] ; predictions : [(classe, cadre, confiance)].
    Renvoie une liste d'evenements {type, classe_verite, classe_predite, cadre, confiance}.
    - oubli : place annotee sans prediction ;
    - confusion : bon emplacement, mauvaise classe ;
    - double_detection : prediction supplementaire sur une place deja detectee ;
    - fausse_detection : prediction sans place annotee correspondante.
    Les vraies detections ne produisent pas d'evenement.
    """
    pairs = sorted(
        ((iou(g[1], p[1]), gi, pi)
         for gi, g in enumerate(truth) for pi, p in enumerate(predictions)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    pairs = [item for item in pairs if item[0] >= threshold]
    used_truth, used_pred, events = set(), set(), []
    for _, gi, pi in pairs:  # 1) meme classe : detection correcte
        if gi not in used_truth and pi not in used_pred and truth[gi][0] == predictions[pi][0]:
            used_truth.add(gi)
            used_pred.add(pi)
    for _, gi, pi in pairs:  # 2) classe differente : confusion
        if gi not in used_truth and pi not in used_pred:
            used_truth.add(gi)
            used_pred.add(pi)
            events.append({"type": "confusion", "classe_verite": truth[gi][0],
                           "classe_predite": predictions[pi][0],
                           "cadre": predictions[pi][1], "confiance": predictions[pi][2]})
    for gi, (cls, box) in enumerate(truth):  # 3) oublis
        if gi not in used_truth:
            events.append({"type": "oubli", "classe_verite": cls, "classe_predite": None,
                           "cadre": box, "confiance": None})
    overlapped = {(pi, truth[gi][0]) for _, gi, pi in pairs}
    for pi, (cls, box, conf) in enumerate(predictions):  # 4) predictions en trop
        if pi in used_pred:
            continue
        kind = "double_detection" if (pi, cls) in overlapped else "fausse_detection"
        events.append({"type": kind, "classe_verite": None, "classe_predite": cls,
                       "cadre": box, "confiance": conf})
    return events


def count_row(image_id, truth, predictions):
    return {
        "image": image_id,
        "libres_annotees": sum(1 for cls, _ in truth if cls == 0),
        "libres_predites": sum(1 for p in predictions if p[0] == 0),
        "occupees_annotees": sum(1 for cls, _ in truth if cls == 1),
        "occupees_predites": sum(1 for p in predictions if p[0] == 1),
    }


def summarize_counts(rows):
    """MAE du comptage, calculee separement pour les libres et les occupees."""
    if not rows:
        raise ValueError("Aucune image evaluee.")
    n = len(rows)
    return {
        "images_evaluees": n,
        "mae_libres": sum(abs(r["libres_predites"] - r["libres_annotees"]) for r in rows) / n,
        "mae_occupees": sum(abs(r["occupees_predites"] - r["occupees_annotees"]) for r in rows) / n,
        "total_libres_annotees": sum(r["libres_annotees"] for r in rows),
        "total_libres_predites": sum(r["libres_predites"] for r in rows),
        "total_occupees_annotees": sum(r["occupees_annotees"] for r in rows),
        "total_occupees_predites": sum(r["occupees_predites"] for r in rows),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_valid_images(manifest):
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "valid"]
    if not rows:
        raise ValueError("Aucune image valid dans le manifeste.")
    return sorted(ROOT / row["image"] for row in rows)


def choose_examples(per_image_events, per_image_gap, per_type=2, largest=2):
    """Images a illustrer : les plus touchees par chaque type d'erreur + plus gros ecarts."""
    chosen, seen = [], set()
    for kind in ERROR_TYPES:
        ranked = sorted(((sum(1 for e in events if e["type"] == kind), image)
                         for image, events in per_image_events.items()),
                        key=lambda item: (-item[0], item[1]))
        added = 0
        for count, image in ranked:
            if added == per_type or not count:
                break
            if image not in seen:
                chosen.append((kind, image))
                seen.add(image)
                added += 1
    for gap, image in sorted(((g, i) for i, g in per_image_gap.items()),
                             key=lambda item: (-item[0], item[1]))[:largest]:
        if gap and image not in seen:
            chosen.append(("ecart_comptage", image))
            seen.add(image)
    return chosen


def draw_example(image_path, truth, predictions, events, destination):
    from PIL import Image, ImageDraw

    colors_truth = {0: (0, 200, 0), 1: (220, 0, 0)}
    colors_pred = {0: (0, 200, 255), 1: (255, 0, 255)}
    with Image.open(image_path) as source:
        picture = source.convert("RGB")
    draw = ImageDraw.Draw(picture)
    for cls, box in truth:
        draw.rectangle(box, outline=colors_truth[cls], width=1)
    for cls, box, _ in predictions:
        draw.rectangle(box, outline=colors_pred[cls], width=2)
    for event in events:
        draw.rectangle(event["cadre"], outline=(255, 255, 0), width=4)
    draw.rectangle((0, 0, 330, 70), fill=(0, 0, 0))
    draw.text((6, 4), "fin vert/rouge = verite (libre/occupee)", fill=colors_truth[0])
    draw.text((6, 20), "trait epais cyan/magenta = prediction (libre/occupee)", fill=colors_pred[0])
    draw.text((6, 36), "cadre jaune = erreur", fill=(255, 255, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    picture.save(destination, quality=92)


def predict(weights, images, conf, nms_iou, imgsz, device, batch):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, iou=nms_iou,
                                imgsz=imgsz, device=device, verbose=False)
        for path, result in zip(chunk, results):
            boxes = result.boxes
            yield path, [(int(c), tuple(float(v) for v in xyxy), float(s))
                         for c, xyxy, s in zip(boxes.cls.tolist(), boxes.xyxy.tolist(),
                                               boxes.conf.tolist())]


def evaluation_environment(rows, git_commit, ultralytics_version):
    names = sorted(row["image"] for row in rows)
    encoded = json.dumps(names, sort_keys=True, separators=(",", ":")).encode()
    return {
        "git_commit": git_commit,
        "analysis_code_sha256": sha256(__file__),
        "validation_image_names_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_dvc_sha256": sha256(ROOT / "data/raw.dvc"),
        "ultralytics_version": ultralytics_version,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Chemin de best.pt")
    parser.add_argument("--manifest", default="data/processed/split_manifest.csv")
    parser.add_argument("--output", default="reports/experience_b")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Seuil NMS")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, help="Essai : ne traiter que N images")
    args = parser.parse_args()

    import ultralytics
    from PIL import Image

    images = read_valid_images(ROOT / args.manifest)
    if args.limit:
        images = images[:args.limit]
    output = (ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    count_rows, details, per_events, per_gap, cache = [], [], {}, {}, {}
    for number, (path, predictions) in enumerate(
            predict(args.weights, images, args.conf, args.nms_iou, args.imgsz,
                    args.device, args.batch), start=1):
        with Image.open(path) as picture:
            width, height = picture.size
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        truth = read_labels(label, width, height)
        events = match_detections(truth, predictions)
        row = count_row(path.name, truth, predictions)
        count_rows.append(row)
        per_events[path.name] = events
        per_gap[path.name] = (abs(row["libres_predites"] - row["libres_annotees"])
                              + abs(row["occupees_predites"] - row["occupees_annotees"]))
        cache[path.name] = (path, truth, predictions)
        for event in events:
            details.append({"image": path.name, "type": event["type"],
                            "classe_verite": CLASS_NAMES[event["classe_verite"]]
                            if event["classe_verite"] is not None else "",
                            "classe_predite": CLASS_NAMES[event["classe_predite"]]
                            if event["classe_predite"] is not None else "",
                            "confiance": "" if event["confiance"] is None
                            else f"{event['confiance']:.3f}",
                            "x1": round(event["cadre"][0], 1), "y1": round(event["cadre"][1], 1),
                            "x2": round(event["cadre"][2], 1), "y2": round(event["cadre"][3], 1)})
        if number % 200 == 0:
            print(f"{number}/{len(images)} images analysees", flush=True)

    with (output / "comptage.csv").open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(count_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(count_rows)
    with (output / "erreurs_detail.csv").open("w", encoding="utf-8", newline="\n") as stream:
        fields = ["image", "type", "classe_verite", "classe_predite", "confiance",
                  "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)

    summary = summarize_counts(count_rows)
    summary["erreurs_par_type"] = {kind: sum(1 for d in details if d["type"] == kind)
                                   for kind in ERROR_TYPES}
    summary.update({
        "seuil_confiance": args.conf, "seuil_nms_iou": args.nms_iou,
        "seuil_appariement_iou": MATCH_IOU, "imgsz": args.imgsz,
        "poids": Path(args.weights).as_posix(), "poids_sha256": sha256(args.weights),
        "commit_git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
    })
    (output / "comptage_resume.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (output / "evaluation_environment.json").write_text(
        json.dumps(evaluation_environment(count_rows, summary["commit_git"],
                                          ultralytics.__version__), indent=2) + "\n",
        encoding="utf-8", newline="\n")

    for kind, name in choose_examples(per_events, per_gap):
        path, truth, predictions = cache[name]
        draw_example(path, truth, predictions, per_events[name],
                     output / "exemples" / f"{kind}_{Path(name).stem[:40]}.jpg")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Resultats ecrits dans", output)


if __name__ == "__main__":
    sys.exit(main())
