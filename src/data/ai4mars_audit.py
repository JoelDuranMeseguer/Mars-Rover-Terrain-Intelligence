"""Utilities to audit an AI4Mars raw dataset and export manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".png", ".jpg", ".jpeg"}

MER_TEST_SUFFIX_RE = re.compile(r"_\d+_t0_merged$", re.IGNORECASE)
MER_TRAIN_SUFFIX_RE = re.compile(r"_merged\d+$", re.IGNORECASE)


@dataclass
class Record:
    kind: str
    path: str
    relpath: str
    mission: str
    sensor: str
    split_hint: str
    stem_raw: str
    stem_norm: str
    ext: str


def path_parts_lower(path: Path) -> list[str]:
    return [part.lower() for part in path.parts]


def infer_mission(rel: Path) -> str:
    parts = path_parts_lower(rel)
    for mission in ("m2020", "mer", "msl"):
        if mission in parts:
            return mission
    return "unknown"


def infer_sensor(rel: Path) -> str:
    parts = path_parts_lower(rel)
    for sensor in ("ncam", "mcam", "hafiq", "nav"):
        if sensor in parts:
            return sensor
    if "eff" in parts:
        return "eff"
    return "unknown"


def infer_split_hint(rel: Path) -> str:
    parts = path_parts_lower(rel)
    hints = [
        "train",
        "val",
        "valid",
        "validation",
        "test",
        "raw_unmerged",
        "merged_unmasked",
        "masked-gold-min1-100agree",
        "masked-gold-min2-100agree",
        "masked-gold-min3-100agree",
        "m2020_geo",
        "nav",
        "eff",
        "images",
        "labels",
    ]
    for hint in hints:
        if hint in parts:
            return hint
    return "unknown"


def normalize_stem(stem: str) -> str:
    """Normalize filename stem for image/label matching."""
    normalized = stem.strip().lower()
    normalized = MER_TEST_SUFFIX_RE.sub("", normalized)
    normalized = MER_TRAIN_SUFFIX_RE.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def scan_records(root: Path) -> tuple[list[Record], list[Record]]:
    image_records: list[Record] = []
    label_records: list[Record] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        rel = path.relative_to(root)
        rel_lower_parts = path_parts_lower(rel)

        in_images_tree = "images" in rel_lower_parts
        in_labels_tree = "labels" in rel_lower_parts

        if in_images_tree and ext in IMAGE_EXTS:
            image_records.append(
                Record(
                    kind="image",
                    path=str(path.resolve()),
                    relpath=str(rel),
                    mission=infer_mission(rel),
                    sensor=infer_sensor(rel),
                    split_hint=infer_split_hint(rel),
                    stem_raw=path.stem,
                    stem_norm=normalize_stem(path.stem),
                    ext=ext,
                )
            )
        elif in_labels_tree and ext in LABEL_EXTS:
            label_records.append(
                Record(
                    kind="label",
                    path=str(path.resolve()),
                    relpath=str(rel),
                    mission=infer_mission(rel),
                    sensor=infer_sensor(rel),
                    split_hint=infer_split_hint(rel),
                    stem_raw=path.stem,
                    stem_norm=normalize_stem(path.stem),
                    ext=ext,
                )
            )

    return image_records, label_records


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_groups(records: list[Record]) -> dict[tuple[str, str, str], list[Record]]:
    groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for rec in records:
        groups[(rec.mission, rec.sensor, rec.stem_norm)].append(rec)
    return groups


def build_groups_mission_stem(records: list[Record]) -> dict[tuple[str, str], list[Record]]:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for rec in records:
        groups[(rec.mission, rec.stem_norm)].append(rec)
    return groups


def rows_from_records(records: list[Record]) -> list[dict]:
    return [asdict(record) for record in records]


def contains_path_fragment(rec: Record, fragment: str) -> bool:
    return fragment.lower() in rec.relpath.lower()


def make_subset_manifest(name: str, image_records: list[Record], label_records: list[Record]) -> list[dict]:
    img_groups = build_groups(image_records)
    lbl_groups = build_groups(label_records)

    rows: list[dict] = []

    if name == "msl_ncam_v1":
        target = ("msl", "ncam")
        for (mission, sensor, stem_norm), imgs in img_groups.items():
            if (mission, sensor) != target:
                continue
            lbls = lbl_groups.get((mission, sensor, stem_norm), [])
            if len(imgs) == 1 and len(lbls) == 1:
                rows.append(
                    {
                        "subset": name,
                        "id": stem_norm,
                        "image_relpath": imgs[0].relpath,
                        "label_relpath": lbls[0].relpath,
                        "mission": mission,
                        "sensor": sensor,
                    }
                )

    elif name == "mer_test_gold_min3":
        mer_test_images = [
            rec
            for rec in image_records
            if rec.mission == "mer" and contains_path_fragment(rec, "/images/test/")
        ]
        mer_gold_labels = [
            rec
            for rec in label_records
            if rec.mission == "mer" and contains_path_fragment(rec, "masked-gold-min3-100agree")
        ]

        img_groups_mer = build_groups_mission_stem(mer_test_images)
        lbl_groups_mer = build_groups_mission_stem(mer_gold_labels)

        for mission, stem_norm in sorted(set(img_groups_mer) | set(lbl_groups_mer)):
            imgs = img_groups_mer.get((mission, stem_norm), [])
            lbls = lbl_groups_mer.get((mission, stem_norm), [])
            if len(imgs) == 1 and len(lbls) == 1:
                rows.append(
                    {
                        "subset": name,
                        "id": stem_norm,
                        "image_relpath": imgs[0].relpath,
                        "label_relpath": lbls[0].relpath,
                        "mission": mission,
                        "sensor": imgs[0].sensor,
                    }
                )

    elif name == "mer_train_candidates":
        mer_eff_images = [
            rec
            for rec in image_records
            if rec.mission == "mer" and contains_path_fragment(rec, "/images/eff/")
        ]
        mer_train_labels = [
            rec
            for rec in label_records
            if rec.mission == "mer" and contains_path_fragment(rec, "merged_unmasked")
        ]

        img_groups_mer = build_groups_mission_stem(mer_eff_images)
        lbl_groups_mer = build_groups_mission_stem(mer_train_labels)

        for mission, stem_norm in sorted(set(img_groups_mer) | set(lbl_groups_mer)):
            imgs = img_groups_mer.get((mission, stem_norm), [])
            lbls = lbl_groups_mer.get((mission, stem_norm), [])
            if len(imgs) == 1 and len(lbls) == 1:
                rows.append(
                    {
                        "subset": name,
                        "id": stem_norm,
                        "image_relpath": imgs[0].relpath,
                        "label_relpath": lbls[0].relpath,
                        "mission": mission,
                        "sensor": imgs[0].sensor,
                    }
                )
    else:
        raise ValueError(f"Unknown subset: {name}")

    rows.sort(key=lambda row: row["id"])
    return rows


def run_audit(root: Path, out: Path) -> dict:
    root = root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    image_records, label_records = scan_records(root)
    img_groups = build_groups(image_records)
    lbl_groups = build_groups(label_records)

    matched_rows = []
    unmatched_image_rows = []
    unmatched_label_rows = []
    duplicate_image_rows = []
    duplicate_label_rows = []

    for (mission, sensor, stem_norm), recs in img_groups.items():
        if len(recs) > 1:
            for rec in recs:
                duplicate_image_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "relpath": rec.relpath,
                        "ext": rec.ext,
                    }
                )

    for (mission, sensor, stem_norm), recs in lbl_groups.items():
        if len(recs) > 1:
            for rec in recs:
                duplicate_label_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "relpath": rec.relpath,
                        "ext": rec.ext,
                    }
                )

    for mission, sensor, stem_norm in sorted(set(img_groups) | set(lbl_groups)):
        imgs = img_groups.get((mission, sensor, stem_norm), [])
        lbls = lbl_groups.get((mission, sensor, stem_norm), [])

        if len(imgs) == 1 and len(lbls) == 1:
            matched_rows.append(
                {
                    "mission": mission,
                    "sensor": sensor,
                    "stem_norm": stem_norm,
                    "image_relpath": imgs[0].relpath,
                    "label_relpath": lbls[0].relpath,
                }
            )
            continue

        if not imgs:
            for lbl in lbls:
                unmatched_label_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "label_relpath": lbl.relpath,
                        "reason": "label_without_image",
                    }
                )
        elif not lbls:
            for img in imgs:
                unmatched_image_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "image_relpath": img.relpath,
                        "reason": "image_without_label",
                    }
                )
        else:
            reason = f"ambiguous_match_{len(imgs)}imgs_{len(lbls)}labels"
            for img in imgs:
                unmatched_image_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "image_relpath": img.relpath,
                        "reason": reason,
                    }
                )
            for lbl in lbls:
                unmatched_label_rows.append(
                    {
                        "mission": mission,
                        "sensor": sensor,
                        "stem_norm": stem_norm,
                        "label_relpath": lbl.relpath,
                        "reason": reason,
                    }
                )

    subset_names = ["msl_ncam_v1", "mer_test_gold_min3", "mer_train_candidates"]
    subset_sizes = {}
    for subset_name in subset_names:
        subset_rows = make_subset_manifest(subset_name, image_records, label_records)
        subset_sizes[subset_name] = len(subset_rows)
        write_csv(out / f"{subset_name}.csv", subset_rows)

    write_csv(out / "images_inventory.csv", rows_from_records(image_records))
    write_csv(out / "labels_inventory.csv", rows_from_records(label_records))
    write_csv(out / "matched_pairs.csv", matched_rows)
    write_csv(out / "unmatched_images.csv", unmatched_image_rows)
    write_csv(out / "unmatched_labels.csv", unmatched_label_rows)
    write_csv(out / "duplicate_images.csv", duplicate_image_rows)
    write_csv(out / "duplicate_labels.csv", duplicate_label_rows)

    summary = {
        "root": str(root),
        "num_images": len(image_records),
        "num_labels": len(label_records),
        "num_matched_pairs": len(matched_rows),
        "num_unmatched_images": len(unmatched_image_rows),
        "num_unmatched_labels": len(unmatched_label_rows),
        "num_duplicate_image_entries": len(duplicate_image_rows),
        "num_duplicate_label_entries": len(duplicate_label_rows),
        "images_by_mission": dict(Counter(rec.mission for rec in image_records)),
        "labels_by_mission": dict(Counter(rec.mission for rec in label_records)),
        "images_by_sensor": dict(Counter(rec.sensor for rec in image_records)),
        "labels_by_sensor": dict(Counter(rec.sensor for rec in label_records)),
        "subset_sizes": subset_sizes,
    }

    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw AI4Mars tree and export manifests")
    parser.add_argument("--root", type=Path, required=True, help="Path to data/raw/AI4Mars")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifests"),
        help="Output directory for CSV/JSON reports",
    )
    args = parser.parse_args()

    summary = run_audit(args.root, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
