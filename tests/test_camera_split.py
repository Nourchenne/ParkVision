"""Regroupement par caméra et découpage camera-aware, sur des images synthétiques."""
import csv

import cv2
import numpy as np
import pytest

from src.data import cluster_cameras as cc
from src.data import prepare_camera_split as pcs


def pattern(camera, noise_seed):
    """Image 'caméra fixe' : un motif de fond propre à chaque caméra, plus un peu de bruit."""
    image = np.full((90, 160), 100, np.uint8)
    if camera == 0:
        image[:, ::16] = 220
    elif camera == 1:
        image[::12, :] = 30
    else:
        image[30:60, 40:120] = 240
    noise = np.random.default_rng(noise_seed).integers(0, 12, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def make_dataset(root, per_camera=6, places=(100, 40, 28)):
    rows = []
    for camera in range(3):
        for number in range(per_camera):
            day = f"2012-10-{10 + camera:02d}"
            folder = root / "data" / "raw" / "train"
            (folder / "images").mkdir(parents=True, exist_ok=True)
            (folder / "labels").mkdir(parents=True, exist_ok=True)
            stem = f"{day}_12_00_{number:02d}"
            cv2.imwrite(str(folder / "images" / f"{stem}.jpg"), pattern(camera, camera * 100 + number))
            (folder / "labels" / f"{stem}.txt").write_text(
                "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(places[camera])), encoding="utf-8")
            rows.append({"image": f"data/raw/train/images/{stem}.jpg", "date": day, "split": "train"})
    return rows


def test_vector_is_normalized_and_fixed_size(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), pattern(0, 1))
    vector = cc.load_vector(tmp_path / "a.jpg")
    assert vector.shape == (64 * 36,)
    assert abs(vector.mean()) < 1e-4 and abs(vector.std() - 1) < 1e-3


def test_blank_image_does_not_divide_by_zero(tmp_path):
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((90, 160), 128, np.uint8))
    assert np.isfinite(cc.load_vector(tmp_path / "blank.png")).all()


def test_missing_image_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        cc.load_vector(tmp_path / "absent.jpg")


def test_kmeans_separates_distinct_backgrounds_and_is_deterministic(tmp_path):
    rows = make_dataset(tmp_path)
    data = cc.load_features(rows, root=tmp_path)
    labels, _ = cc.kmeans(data, 3, restarts=3)
    expected = np.repeat([0, 1, 2], 6)
    # chaque caméra tombe dans un seul groupe, et les trois groupes sont distincts
    groups = [set(labels[expected == camera].tolist()) for camera in range(3)]
    assert all(len(group) == 1 for group in groups) and len({next(iter(g)) for g in groups}) == 3
    again, _ = cc.kmeans(data, 3, restarts=3)
    assert np.array_equal(labels, again)


def test_order_by_size_puts_the_largest_group_first():
    assert cc.order_by_size(np.array([5, 5, 5, 2, 2, 9])).tolist() == [0, 0, 0, 1, 1, 2]


def test_day_vote_corrects_isolated_misassignments():
    rows = [{"date": "d1"}] * 5 + [{"date": "d2"}] * 3
    labels = np.array([0, 0, 0, 2, 0, 1, 1, 1])
    assert cc.day_vote(rows, labels).tolist() == [0] * 5 + [1] * 3
    assert cc.day_purity(rows, labels) == 7 / 8


def test_lot_is_inferred_from_the_annotated_place_count():
    assert [cc.lot_of(n) for n in (28, 33, 40, 70, 100)] == [28, 28, 40, 40, 100]


def test_summary_and_agreement_use_the_label_files(tmp_path):
    rows = make_dataset(tmp_path)
    labels = np.repeat([0, 1, 2], 6)
    assert cc.lot_agreement(rows, labels, root=tmp_path) == 1.0
    summary = cc.summarize(rows, labels, root=tmp_path)
    assert [item["images"] for item in summary] == [6, 6, 6]
    assert [item["places_mediane"] for item in summary] == [100, 40, 28]
    shifted = np.array([0] * 12 + [1] * 6)  # les parkings à 40 et 28 places mélangés dans un même groupe
    assert cc.lot_agreement(rows, shifted, root=tmp_path) < 1.0


def test_write_map_and_samples(tmp_path):
    rows = make_dataset(tmp_path)
    labels = np.repeat([0, 1, 2], 6)
    output = tmp_path / "out" / "camera_map.csv"
    cc.write_map(rows, labels, output)
    read = list(csv.DictReader(output.open(encoding="utf-8")))
    assert list(read[0]) == ["image", "cluster"] and len(read) == 18 and read[-1]["cluster"] == "2"
    cc.write_samples(rows, labels, tmp_path / "samples", per_cluster=2, root=tmp_path)
    assert len(list((tmp_path / "samples").glob("*.jpg"))) == 6


# --- découpage par caméra --------------------------------------------------------------------------

MANIFEST = [{"image": f"i{n}.jpg", "date": f"2012-10-{n % 3 + 10}", "split": "train"} for n in range(9)]
CAMERA_MAP = [{"image": f"i{n}.jpg", "cluster": str(n % 3)} for n in range(9)]


def test_each_camera_goes_to_exactly_one_split():
    rows = pcs.assign(MANIFEST, CAMERA_MAP, {0}, {2}, {1})
    by_cluster = {}
    for row in rows:
        by_cluster.setdefault(row["cluster"], set()).add(row["split"])
    assert by_cluster == {0: {"train"}, 1: {"test"}, 2: {"valid"}}
    assert len(rows) == 9 and [r["image"] for r in rows] == [m["image"] for m in MANIFEST]


@pytest.mark.parametrize(("train", "valid", "test", "message"), [
    ({0}, {0}, {1}, "un seul split"),
    ({0}, {1}, set(), "au moins un groupe"),
    ({0}, {1}, {2, 3}, "inconnus"),
])
def test_invalid_assignments_are_rejected(train, valid, test, message):
    with pytest.raises(ValueError, match=message):
        pcs.assign(MANIFEST, CAMERA_MAP, train, valid, test)


def test_unassigned_group_is_rejected():
    with_extra = [*CAMERA_MAP, {"image": "extra.jpg", "cluster": "3"}]
    with pytest.raises(ValueError, match="non affectés"):
        pcs.assign(MANIFEST, with_extra, {0}, {1}, {2})


def test_images_missing_from_the_camera_map_are_rejected():
    manifest = [*MANIFEST, {"image": "x.jpg", "date": "2012-10-10", "split": "train"}]
    with pytest.raises(ValueError, match="absentes"):
        pcs.assign(manifest, CAMERA_MAP, {0}, {1}, {2})


def test_parse_groups_and_write_roundtrip(tmp_path):
    assert pcs.parse_groups("0, 2") == {0, 2}
    out = tmp_path / "camera_split_manifest.csv"
    pcs.write(pcs.assign(MANIFEST, CAMERA_MAP, {0}, {2}, {1}), out)
    read = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(read[0]) == ["image", "date", "cluster", "split"] and {r["split"] for r in read} == {"train", "valid", "test"}
