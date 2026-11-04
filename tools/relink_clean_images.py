#!/usr/bin/env python3
"""Relink clean original scans into the YOLO dataset.

The `yashveer` and `yuvraj` labels were derived from CVAT preview renders whose
pixels contain drawn annotations. This tool swaps those images for clean
originals while leaving every label untouched (matching is by the annotation's
original filename stem, e.g. ``IMG_20261015_0005`` or ``1``).

Images are accepted from a directory, a zip archive, or one file. The image must
have the same aspect ratio as the annotation (labels are normalised, so a
different resolution is fine), and must not contain baked-in annotation colours.

Usage
-----
    # check what would happen
    python tools/relink_clean_images.py --images path/to/originals --dry-run

    # replace the files, update manifest/stats/clean configs
    python tools/relink_clean_images.py --images path/to/originals --apply
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.build_yolo_dataset import (
    CLASSES,
    FLAT_SOURCES,
    OVERLAY_PURE_PCT,
    SPLITS,
    write_readme,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ASPECT_TOL = 0.005


def load_manifest(root: Path) -> tuple[list[dict], list[str]]:
    with open(root / "manifest.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def image_stats(data: bytes) -> tuple[int, int, float, float]:
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        rgb = np.asarray(
            im.convert("RGB").resize((700, max(1, round(h * 700 / w)))),
            dtype=np.uint16,
        )
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    total = mx.size
    pure = float(((mx >= 250) & (mn <= 5)).sum()) / total * 100
    sat = float(
        ((mx > 90) & (((mx - mn) / np.maximum(mx, 1)) > 0.55)).sum()
    ) / total * 100
    return w, h, pure, sat


def dhash_file(path: Path) -> int:
    with Image.open(path) as im:
        grey = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.uint8)
    bits = (grey[:, 1:] > grey[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def find_duplicates(root: Path) -> list[list]:
    files = sorted((root / "detect" / "images").glob("*/*"))
    print(f"  hashing {len(files)} images for duplicate check ...")
    hashes = {p.name: dhash_file(p) for p in files}
    names = list(hashes)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = (hashes[names[i]] ^ hashes[names[j]]).bit_count()
            if d <= 5:
                pairs.append([names[i], names[j], d])
    return pairs


def read_missing(root: Path) -> list[dict]:
    path = root / "missing_images.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"name": r["original_name"], "source": r["source"], "shapes": int(r["shapes"])}
            for r in csv.DictReader(f)
        ]


def build_stats(root: Path, rows: list[dict], dup_pairs: list[list]) -> dict:
    order = list(FLAT_SOURCES) + [
        "amanAnnotations", "cvat_sidd", "sajalAnnotations", "Tanishka",
    ]
    present = {r["source"] for r in rows}
    ordered = [s for s in order if s in present] + sorted(present - set(order))

    stats = {
        "classes": list(CLASSES),
        "sources": {},
        "splits": {s: {"images": 0, "shapes": 0, "classes": {}} for s in SPLITS},
        "totals": {"images": 0, "shapes": 0, "classes": {}},
        "image_quality": {"clean": 0, "preview_with_baked_annotations": 0},
        "near_duplicate_pairs": dup_pairs,
        "missing_images": read_missing(root),
        "relinked_images": 0,
    }

    split_classes = {s: Counter() for s in SPLITS}
    source_classes = {s: Counter() for s in ordered}
    source_images = Counter()
    source_shapes = Counter()
    total_classes = Counter()
    total_shapes = 0

    for r in rows:
        shapes = int(r["boxes"]) + int(r["polygons"])
        cls = Counter()
        for part in r["classes"].split(";"):
            if part:
                k, v = part.split(":")
                cls[k] = int(v)
        total_shapes += shapes
        total_classes.update(cls)
        source_images[r["source"]] += 1
        source_shapes[r["source"]] += shapes
        source_classes[r["source"]].update(cls)
        split_classes[r["split"]].update(cls)
        stats["splits"][r["split"]]["images"] += 1
        stats["splits"][r["split"]]["shapes"] += shapes
        if r["baked_in_annotations"] == "true":
            stats["image_quality"]["preview_with_baked_annotations"] += 1
        else:
            stats["image_quality"]["clean"] += 1
            if r["source"] in FLAT_SOURCES:
                stats["relinked_images"] += 1

    for s in SPLITS:
        stats["splits"][s]["classes"] = dict(split_classes[s].most_common())
    for s in ordered:
        sel = [r for r in rows if r["source"] == s]
        previews = sum(1 for r in sel if r["baked_in_annotations"] == "true")
        if previews == 0:
            image_type = "clean photo"
        elif previews == len(sel):
            image_type = "CVAT preview render"
        else:
            image_type = f"mixed ({len(sel) - previews} clean / {previews} preview)"
        stats["sources"][s] = {
            "images": source_images[s],
            "shapes": source_shapes[s],
            "classes": dict(source_classes[s].most_common()),
            "image_type": image_type,
        }
    stats["totals"] = {
        "images": len(rows),
        "shapes": total_shapes,
        "classes": dict(total_classes.most_common()),
    }
    return stats


def write_clean_lists(root: Path, rows: list[dict]) -> None:
    for tree_name in ("detect", "segment"):
        tree = root / tree_name
        for split in SPLITS:
            files = [
                f"./images/{split}/{r['file']}"
                for r in rows
                if r["split"] == split and r["baked_in_annotations"] == "false"
            ]
            (tree / f"clean_{split}.txt").write_text(
                "\n".join(files) + "\n", encoding="utf-8"
            )


def collect_images(path: Path) -> list[tuple[str, bytes]]:
    items: list[tuple[str, bytes]] = []
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                items.append((p.name, p.read_bytes()))
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            by_stem: dict[str, tuple[str, bytes]] = {}
            for name in sorted(z.namelist()):
                if name.endswith("/") or "__MACOSX" in name:
                    continue
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    stem = Path(name).stem.lower()
                    if (
                        stem not in by_stem
                        or ("raw" in name.lower() and "raw" not in by_stem[stem][0].lower())
                        or (name.lower().endswith((".jpg", ".jpeg")) and by_stem[stem][0].lower().endswith(".png"))
                    ):
                        by_stem[stem] = (Path(name).name, z.read(name))
            items.extend(by_stem.values())
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        items.append((path.name, path.read_bytes()))
    else:
        raise SystemExit(f"unsupported --images path: {path}")
    if not items:
        raise SystemExit(f"no images found in {path}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="directory, zip, or image file with clean originals")
    ap.add_argument("--dataset", default=str(ROOT / "dataset"), help="dataset root")
    ap.add_argument("--source", choices=list(FLAT_SOURCES), help="restrict to one flat batch")
    ap.add_argument("--apply", action="store_true", help="write changes (default is dry run)")
    ap.add_argument("--dry-run", action="store_true", help="validate only (default)")
    ap.add_argument("--allow-baked", action="store_true", help="accept images that still look annotated")
    ap.add_argument("--skip-duplicate-check", action="store_true")
    args = ap.parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("--apply and --dry-run are mutually exclusive")

    root = Path(args.dataset).resolve()
    if not (root / "manifest.csv").exists():
        raise SystemExit(f"no manifest.csv under {root}")

    rows, fieldnames = load_manifest(root)
    targets: dict[str, dict] = {}
    for row in rows:
        if row["source"] not in FLAT_SOURCES:
            continue
        if args.source and row["source"] != args.source:
            continue
        for key in {
            Path(row["original_name"]).stem.lower(),
            Path(row["file"]).stem.lower(),
        }:
            if key in targets and targets[key] is not row:
                raise SystemExit(f"ambiguous filename stem {key!r}")
            targets[key] = row

    plans, errors, unmatched = [], [], []
    seen = set()
    for name, data in collect_images(Path(args.images)):
        stem = Path(name).stem.lower()
        if stem in seen:
            errors.append(f"{name}: duplicate input stem")
            continue
        seen.add(stem)
        row = targets.get(stem)
        if row is None:
            unmatched.append(name)
            continue
        try:
            w, h, pure, sat = image_stats(data)
        except Exception as exc:
            errors.append(f"{name}: cannot read image ({exc})")
            continue
        mw, mh = int(row["width"]), int(row["height"])
        if abs((w / h) / (mw / mh) - 1) > ASPECT_TOL:
            errors.append(
                f"{name}: aspect {w}x{h} does not match annotation {mw}x{mh}"
            )
            continue
        if pure > OVERLAY_PURE_PCT and not args.allow_baked:
            errors.append(
                f"{name}: {pure:.2f}% baked-in overlay pixels (still an annotated preview)"
            )
            continue
        ext = ".jpg" if Path(name).suffix.lower() in (".jpg", ".jpeg") else ".png"
        plans.append(
            {
                "row": row,
                "name": name,
                "new_file": Path(row["file"]).stem + ext,
                "w": w,
                "h": h,
                "pure": pure,
                "sat": sat,
                "data": data,
            }
        )

    print(
        f"inputs={len(seen)} matched={len(plans)} "
        f"unmatched={len(unmatched)} errors={len(errors)}"
    )
    for p in plans:
        print(
            f"  {p['row']['source']:9s} {p['row']['file']} -> {p['new_file']} "
            f"({p['w']}x{p['h']}, overlay {p['pure']:.3f}%)"
        )
    for name in unmatched[:20]:
        print(f"  unmatched: {name}")
    for e in errors[:20]:
        print(f"  ERROR: {e}")
    if errors:
        sys.exit(1)
    if not args.apply:
        print("dry run - pass --apply to replace the files")
        return

    for p in plans:
        row = p["row"]
        split = row["split"]
        old_file, new_file = row["file"], p["new_file"]
        label_stem = Path(new_file).stem
        for tree in ("detect", "segment"):
            label = root / tree / "labels" / split / f"{label_stem}.txt"
            if not label.exists():
                raise SystemExit(f"missing label {label}; aborting")
            for fname in {old_file, new_file}:
                img = root / tree / "images" / split / fname
                if img.exists():
                    img.unlink()
        dest = root / "detect" / "images" / split / new_file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p["data"])
        seg = root / "segment" / "images" / split / new_file
        seg.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(dest, seg)
        except OSError:
            shutil.copy2(dest, seg)
        with Image.open(dest) as im:
            if im.size != (p["w"], p["h"]):
                raise SystemExit(f"written image has wrong size: {dest}")
        row.update(
            {
                "file": new_file,
                "width": str(p["w"]),
                "height": str(p["h"]),
                "image_format": Path(new_file).suffix.lstrip("."),
                "baked_in_annotations": "false",
                "overlay_pct": f"{p['pure']:.3f}",
            }
        )

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.skip_duplicate_check:
        old = json.loads((root / "stats.json").read_text(encoding="utf-8"))
        dup_pairs = old.get("near_duplicate_pairs", [])
    else:
        dup_pairs = find_duplicates(root)

    stats = build_stats(root, rows, dup_pairs)
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_clean_lists(root, rows)
    write_readme(stats, stats["missing_images"], out_root=root)

    bad = 0
    for row in rows:
        split = row["split"]
        stem = Path(row["file"]).stem
        for tree in ("detect", "segment"):
            if not (root / tree / "images" / split / row["file"]).exists():
                bad += 1
                print(f"missing image: {tree}/{split}/{row['file']}")
            if not (root / tree / "labels" / split / f"{stem}.txt").exists():
                bad += 1
                print(f"missing label: {tree}/{split}/{stem}.txt")
    print(
        f"relinked {len(plans)} images | clean={stats['image_quality']['clean']} "
        f"preview={stats['image_quality']['preview_with_baked_annotations']} | "
        f"verification problems={bad}"
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
