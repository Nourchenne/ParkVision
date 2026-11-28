"""Audit des journées : calcul de l'occupation et critère de suspicion, sur des données synthétiques."""
import csv

from src.analysis.audit_days import audit, count_label, write


def make_day(root, day, classes_per_image, split="train", images=2):
    folder = root / "data" / "raw" / split
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for number in range(images):
        stem = f"{day}_12_00_{number:02d}"
        (folder / "images" / f"{stem}.jpg").write_bytes(b"")
        (folder / "labels" / f"{stem}.txt").write_text(
            "\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in classes_per_image), encoding="utf-8")
        rows.append({"image": f"data/raw/{split}/images/{stem}.jpg", "date": day, "split": split})
    return rows


def run(tmp_path, days):
    rows = []
    for day, classes, split in days:
        rows += make_day(tmp_path, day, classes, split)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "date", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return {r["date"]: r for r in audit(manifest, tmp_path)}


def test_count_label_counts_free_and_occupied(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n1 0.3 0.3 0.1 0.1\n\n", encoding="utf-8")
    assert count_label(label) == (1, 2)


def test_occupation_is_share_of_occupied_places(tmp_path):
    result = run(tmp_path, [("2012-09-14", [0, 1, 1, 1], "train")])  # un vendredi
    row = result["2012-09-14"]
    assert row["libres"] == 2 and row["occupees"] == 6 and row["occupation"] == 0.75
    assert row["jour"] == "vendredi" and row["suspect"] == 0


def test_empty_weekday_is_flagged_but_empty_weekend_is_not(tmp_path):
    empty = [0] * 10
    result = run(tmp_path, [("2012-10-30", empty, "test"),   # mardi
                            ("2012-10-27", empty, "train")])  # samedi
    assert result["2012-10-30"]["suspect"] == 1
    assert "mardi" in result["2012-10-30"]["jour"] and "occupation" in result["2012-10-30"]["raison"]
    assert result["2012-10-27"]["suspect"] == 0


def test_threshold_boundary_and_split_are_reported(tmp_path):
    result = run(tmp_path, [("2012-10-30", [1] + [0] * 8, "test"),   # 11,1 % : au-dessus du seuil
                            ("2012-11-06", [1] + [0] * 19, "valid")])  # 5 % : sous le seuil
    assert result["2012-10-30"]["suspect"] == 0 and result["2012-10-30"]["split"] == "test"
    assert result["2012-11-06"]["suspect"] == 1 and result["2012-11-06"]["split"] == "valid"


def test_known_days_get_their_verification_status(tmp_path):
    result = run(tmp_path, [("2012-10-30", [0] * 10, "test"), ("2012-12-25", [0] * 10, "valid")])
    assert result["2012-10-30"]["verification"] == "labels_faux"
    assert result["2012-12-25"]["verification"] == "vide_reel"


def test_write_produces_a_readable_csv(tmp_path):
    rows = run(tmp_path, [("2012-09-14", [0, 1], "train")])
    out = tmp_path / "out" / "journees.csv"
    write(list(rows.values()), out)
    with out.open(encoding="utf-8", newline="") as handle:
        read = list(csv.DictReader(handle))
    assert read[0]["date"] == "2012-09-14" and read[0]["suspect"] == "0"
