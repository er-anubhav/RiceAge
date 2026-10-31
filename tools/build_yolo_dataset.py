#!/usr/bin/env python3
"""Build a unified YOLO dataset from every rice annotation batch in the project.

Sources
-------
flat CVAT exports (single label + ``Condition`` attribute):
    yashveer rice annotation.zip
    yuvraj.zip
nested batch archive (class carried by the CVAT ``label``):
    Sakku.zip -> Sakku/{amanAnnotations,cvat_sidd,sajalAnnotations,Tanishka}.zip

All spellings are normalised to eight classes. Only the Sakku images are clean
photographs; the two flat exports contain CVAT preview renders with annotations
burned into the pixels, so they are flagged per image.

Output tree
-----------
    dataset/
      detect/{data.yaml, images/{train,val,test}, labels/{train,val,test}}
      segment/{data.yaml, images/{train,val,test}, labels/{train,val,test}}
      manifest.csv, missing_images.csv, stats.json, README.md
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset"
SAKKU_ZIP = ROOT / "Sakku.zip"
ACTUAL_ZIP = ROOT / "Actual Images with .XML file.zip"

YASHVEER_CLEAN_ZIP = ROOT / "yashveer rice annotation" / "yashveer rice annotation (2).zip"
YUVRAJ_CLEAN_ZIP = ROOT / "yuvraj" / "yuvraj rice.zip"
FLAT_SOURCES = {
    "yashveer": {
        "zip": YASHVEER_CLEAN_ZIP if YASHVEER_CLEAN_ZIP.exists() else ROOT / "yashveer rice annotation.zip",
        "xml": "yashveer rice annotation/annotations.xml",
    },
    "yuvraj": {
        "zip": YUVRAJ_CLEAN_ZIP if YUVRAJ_CLEAN_ZIP.exists() else ROOT / "yuvraj.zip",
        "xml": "annotations.xml" if YUVRAJ_CLEAN_ZIP.exists() else "yuvraj/annotations.xml",
    },
    "yuvraj_02": {
        "zip": ROOT / "yuvraj" / "rice 02.zip",
        "xml": "annotations.xml",
    },
    "yuvraj_03": {
        "zip": ROOT / "yuvraj" / "rice 03.zip",
        "xml": "annotations.xml",
    },
    "yuvraj_04": {
        "zip": ROOT / "yuvraj" / "rice 04.zip",
        "xml": "annotations.xml",
    },
}

SAKKU_BATCHES = [
    "Sakku/amanAnnotations.zip",
    "Sakku/cvat_sidd.zip",
    "Sakku/sajalAnnotations.zip",
    "Sakku/Tanishka.zip",
]

ACTUAL_BATCHES = [
    ("Chalky.zip", "actual_chalky"),
    ("Damaged.zip", "actual_damaged"),
    ("Foreign Matter.zip", "actual_foreign_matter"),
    ("broken.zip", "actual_broken"),
    ("mixed.zip", "actual_mixed"),
    ("whole.zip", "actual_whole"),
]

# Canonical class order (also the YOLO class ids).
CLASSES = [
    "whole",
    "broken",
    "chalky",
    "damaged",
    "discoloured",
    "dehusked",
    "foreign_matter",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# Every spelling seen across the annotators -> canonical class.
CONDITION_MAP = {
    "whole grain": "whole",
    "whole": "whole",
    "brokens": "broken",
    "broken": "broken",
    "chalky grains": "chalky",
    "chalky": "chalky",
    "damaged": "damaged",
    "damage": "damaged",
    "discoloured grains": "discoloured",
    "discoloured": "discoloured",
    "discolored": "discoloured",
    "discolour": "discoloured",
    "dehusked grain": "dehusked",
    "dehusked": "dehusked",
    "foreign matter": "foreign_matter",
    "admixture of lower class": "foreign_matter",
    "admixture": "foreign_matter",
}

SPLITS = ("train", "val", "test")
SEED = 42
RARE_CLASS_MAX = 50          # classes with fewer instances get priority placement
DUPLICATE_HAMMING = 5        # dHash distance for "same scan" grouping
OVERLAY_PURE_PCT = 0.02      # > this fraction of pure palette pixels => preview render


@dataclass
class Shape:
    kind: str                             # "box" | "polygon"
    cls: str
    points: list[tuple[float, float]]
    raw_label: str
    condition: str


@dataclass
class ImageRec:
    source: str
    archive: Path                         # outer zip on disk
    inner_member: str | None              # inner zip member, None for flat exports
    image_member: str | None              # member holding the image
    out_name: str
    original_name: str
    width: int
    height: int
    image_format: str
    shapes: list[Shape] = field(default_factory=list)
    split: str = ""
    dhash: int = 0
    overlay_pct: float = 0.0
    saturated_pct: float = 0.0

    @property
    def classes(self) -> Counter:
        return Counter(s.cls for s in self.shapes)

    @property
    def is_preview(self) -> bool:
        return self.overlay_pct > OVERLAY_PURE_PCT


def norm_condition(text: str) -> str:
    key = re.sub(r"\s+", " ", text.strip().lower())
    if key not in CONDITION_MAP:
        raise ValueError(f"unknown condition/label: {text!r}")
    return CONDITION_MAP[key]


def safe_name(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return s or "image"


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clamp(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _clean_stem(name: str) -> str:
    s = Path(name).name.lower()
    for _ in range(2):
        for ext in (".jpeg", ".jpg", ".png"):
            if s.endswith(ext):
                s = s[:-len(ext)]
    return s


def _image_members(z: zipfile.ZipFile) -> dict[str, str]:
    """stem(lowercase) -> member for image files."""
    members: dict[str, str] = {}
    for n in sorted(z.namelist()):
        if n.lower().endswith((".png", ".jpg", ".jpeg")):
            stem = _clean_stem(n)
            if (
                stem not in members
                or ("raw" in n.lower() and "raw" not in members[stem].lower())
                or (n.lower().endswith((".jpg", ".jpeg")) and members[stem].lower().endswith(".png"))
            ):
                members[stem] = n
    return members


def _parse_shapes(xml_root: ET.Element) -> list[ImageRec]:
    recs: list[ImageRec] = []
    for im in xml_root.findall("image"):
        xml_name = im.get("name")
        rec = ImageRec(
            source="",
            archive=Path(),
            inner_member=None,
            image_member=None,
            out_name="",
            original_name=xml_name,
            width=int(im.get("width")),
            height=int(im.get("height")),
            image_format="",
        )
        for sh in im:
            if sh.tag not in ("box", "polygon"):
                continue
            condition = next(
                (
                    a.text
                    for a in sh.findall("attribute")
                    if a.get("name", "").lower() in ("condition", "condititon", "grian", "grain")
                    and a.text
                ),
                None,
            )
            raw_label = sh.get("label", "")
            cls = norm_condition(condition if condition else raw_label)
            if sh.tag == "box":
                x1, y1 = float(sh.get("xtl")), float(sh.get("ytl"))
                x2, y2 = float(sh.get("xbr")), float(sh.get("ybr"))
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                points = [
                    tuple(map(float, p.split(",")))
                    for p in sh.get("points", "").split(";")
                    if p.strip()
                ]
                if polygon_area(points) < 1.0:
                    continue
            rec.shapes.append(Shape(sh.tag, cls, points, raw_label, condition or ""))
        recs.append(rec)
    return recs


def parse_flat(source: str, spec: dict) -> list[ImageRec]:
    with zipfile.ZipFile(spec["zip"]) as z:
        root = ET.fromstring(z.read(spec["xml"]))
        members = _image_members(z)
    recs = _parse_shapes(root)
    for rec in recs:
        rec.source = source
        rec.archive = spec["zip"]
        stem = _clean_stem(rec.original_name)
        member = members.get(stem)
        rec.image_member = member
        ext = Path(member).suffix.lower().lstrip(".") if member else "jpg"
        if ext == "jpeg":
            ext = "jpg"
        rec.image_format = ext
        rec.out_name = f"{source}__{safe_name(stem)}.{ext}"
    return recs


def parse_nested(archive: Path, inner_member: str, source: str) -> list[ImageRec]:
    with zipfile.ZipFile(archive) as root:
        data = root.read(inner_member)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xmls = [n for n in z.namelist() if n.lower().endswith("annotations.xml")]
        if not xmls:
            raise FileNotFoundError(f"no annotations.xml in {inner_member}")
        root_elem = ET.fromstring(z.read(xmls[0]))
        members = _image_members(z)
    recs = _parse_shapes(root_elem)
    for rec in recs:
        rec.source = source
        rec.archive = archive
        rec.inner_member = inner_member
        stem = _clean_stem(rec.original_name)
        member = members.get(stem)
        rec.image_member = member
        ext = Path(member).suffix.lower().lstrip(".") if member else "jpg"
        if ext == "jpeg":
            ext = "jpg"
        rec.image_format = ext
        rec.out_name = f"{source}__{safe_name(stem)}.{ext}"
    return recs


def parse_all() -> tuple[list[ImageRec], list[ImageRec]]:
    recs: list[ImageRec] = []
    for source, spec in FLAT_SOURCES.items():
        recs.extend(parse_flat(source, spec))
    for inner in SAKKU_BATCHES:
        source = Path(inner).stem
        recs.extend(parse_nested(SAKKU_ZIP, inner, source))
    if ACTUAL_ZIP.exists():
        for inner, source in ACTUAL_BATCHES:
            recs.extend(parse_nested(ACTUAL_ZIP, inner, source))

    # keep only annotated images; report annotated images without a file
    missing = [r for r in recs if r.shapes and not r.image_member]
    usable = [r for r in recs if r.shapes and r.image_member]
    return usable, missing


def extract_images(recs: list[ImageRec], staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    flat = [r for r in recs if r.inner_member is None]

    by_zip: dict[Path, list[ImageRec]] = defaultdict(list)
    for r in flat:
        by_zip[r.archive].append(r)
    for zip_path, group in by_zip.items():
        with zipfile.ZipFile(zip_path) as z:
            for r in group:
                dest = staging / r.out_name
                with z.open(r.image_member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)

    by_nested: dict[Path, dict[str, list[ImageRec]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.inner_member:
            by_nested[r.archive][r.inner_member].append(r)

    for outer_archive, inner_dict in by_nested.items():
        with zipfile.ZipFile(outer_archive) as root:
            for inner, group in inner_dict.items():
                data = root.read(inner)
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for r in group:
                        dest = staging / r.out_name
                        with z.open(r.image_member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)

    # verify dimensions against the annotation XML
    for r in recs:
        with Image.open(staging / r.out_name) as im:
            if im.size != (r.width, r.height):
                raise RuntimeError(
                    f"dimension mismatch for {r.out_name}: xml {r.width}x{r.height}, "
                    f"file {im.size[0]}x{im.size[1]}"
                )


def analyse_images(recs: list[ImageRec], staging: Path) -> None:
    for r in recs:
        path = staging / r.out_name
        with Image.open(path) as im:
            grey = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.uint8)
            bits = (grey[:, 1:] > grey[:, :-1]).flatten()
            r.dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
            rgb = np.asarray(
                im.convert("RGB").resize((700, max(1, round(im.height * 700 / im.width)))),
                dtype=np.uint16,
            )
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        total = mx.size
        r.overlay_pct = float(((mx >= 250) & (mn <= 5)).sum()) / total * 100
        r.saturated_pct = float(
            ((mx > 90) & (((mx - mn) / np.maximum(mx, 1)) > 0.55)).sum()
        ) / total * 100


def duplicate_groups(recs: list[ImageRec]) -> dict[int, list[ImageRec]]:
    parent = list(range(len(recs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if (recs[i].dhash ^ recs[j].dhash).bit_count() <= DUPLICATE_HAMMING:
                union(i, j)

    groups: dict[int, list[ImageRec]] = defaultdict(list)
    for i, r in enumerate(recs):
        groups[find(i)].append(r)
    return groups


def plan_split(recs: list[ImageRec], groups: dict[int, list[ImageRec]]) -> None:
    """70/15/15 per batch, keeping near-duplicate groups inside one split."""
    rng = random.Random(SEED)
    class_totals = Counter()
    for r in recs:
        class_totals.update(r.classes)
    rare = {c for c, n in class_totals.items() if n < RARE_CLASS_MAX}

    group_of = {}
    for gid, members in groups.items():
        for r in members:
            group_of[id(r)] = gid

    sources = sorted(list({r.source for r in recs}))
    capacities: dict[str, dict[str, int]] = {}
    for src in sources:
        n = sum(1 for r in recs if r.source == src)
        train = int(round(n * 0.70))
        val = int(round(n * 0.15))
        capacities[src] = {"train": train, "val": val, "test": n - train - val}

    assigned: set[int] = set()

    def assign(unit: list[ImageRec], split: str) -> None:
        for m in unit:
            m.split = split
            assigned.add(id(m))
        for src, count in Counter(m.source for m in unit).items():
            capacities[src][split] -= count

    # duplicate groups that span batches must stay together in one split
    for members in groups.values():
        if len({m.source for m in members}) > 1:
            need = Counter(m.source for m in members)
            for split in SPLITS:
                if all(capacities[s][split] >= k for s, k in need.items()):
                    assign(members, split)
                    break
            else:
                assign(members, "train")

    def solve(units: list[list[ImageRec]], caps: list[int]) -> list[str] | None:
        place: list[str | None] = [None] * len(units)

        def rec(i: int) -> bool:
            if i == len(units):
                return True
            for s, split in enumerate(SPLITS):
                if caps[s] >= len(units[i]):
                    caps[s] -= len(units[i])
                    place[i] = split
                    if rec(i + 1):
                        return True
                    caps[s] += len(units[i])
            return False

        return place if rec(0) else None

    for src in sources:
        pool = [r for r in recs if r.source == src and id(r) not in assigned]
        if not pool:
            continue

        unit_map: dict[int, list[ImageRec]] = {}
        for r in pool:
            unit_map.setdefault(group_of[id(r)], []).append(r)
        units = list(unit_map.values())
        rng.shuffle(units)

        # keep at least one rare-class image in train when possible
        for u in [u for u in units if any(set(m.classes) & rare for m in u)]:
            if len(u) <= capacities[src]["train"]:
                assign(u, "train")
                units = [x for x in units if x is not u]
                break

        units.sort(key=len, reverse=True)
        caps = [capacities[src][s] for s in SPLITS]
        placement = solve(units, caps)
        if placement is None:
            for u in units:
                split = max(SPLITS, key=lambda s: capacities[src][s])
                assign(u, split)
        else:
            for u, split in zip(units, placement):
                assign(u, split)

        for r in pool:
            if not r.split:
                raise RuntimeError(f"unassigned {r.out_name}")


def write_labels(rec: ImageRec, tree: Path) -> None:
    lines = []
    W, H = rec.width, rec.height
    for s in rec.shapes:
        if tree.name == "detect":
            xs = [p[0] for p in s.points]
            ys = [p[1] for p in s.points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            lines.append(
                f"{CLASS_ID[s.cls]} {clamp(cx):.6f} {clamp(cy):.6f} "
                f"{clamp(bw):.6f} {clamp(bh):.6f}"
            )
        else:
            coords = " ".join(
                f"{clamp(x / W):.6f} {clamp(y / H):.6f}" for x, y in s.points
            )
            lines.append(f"{CLASS_ID[s.cls]} {coords}")
    path = tree / "labels" / rec.split / (Path(rec.out_name).stem + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(tree: Path) -> None:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    task = "object detection" if tree.name == "detect" else "instance segmentation"
    abs_path = str(tree.resolve()).replace("\\", "/")
    text = (
        f"# RiceAge grain dataset - {task}\n"
        f"path: {abs_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n"
    )
    (tree / "data.yaml").write_text(text, encoding="utf-8")


def write_clean_config(tree: Path, recs: list[ImageRec]) -> None:
    """Extra config that selects only the clean (non-preview) images."""
    abs_path = str(tree.resolve()).replace("\\", "/")
    for split in SPLITS:
        files = [
            f"{abs_path}/images/{split}/{r.out_name}"
            for r in recs
            if r.split == split and not r.is_preview
        ]
        (tree / f"clean_{split}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    text = (
        "# RiceAge grain dataset - clean subset only\n"
        f"path: {abs_path}\n"
        "train: clean_train.txt\n"
        "val: clean_val.txt\n"
        "test: clean_test.txt\n\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n"
    )
    (tree / "data_clean.yaml").write_text(text, encoding="utf-8")


def write_readme(stats: dict, missing, out_root: Path = OUT) -> None:
    class_rows = "\n".join(
        f"| {c} | {stats['totals']['classes'].get(c, 0)} |" for c in CLASSES
    )
    split_rows = "\n".join(
        f"| {s} | {stats['splits'][s]['images']} | {stats['splits'][s]['shapes']} |"
        for s in SPLITS
    )
    source_rows = "\n".join(
        f"| {name} | {s['images']} | {s['shapes']} | {s['image_type']} |"
        for name, s in stats["sources"].items()
    )
    miss = ", ".join(
        f"`{m['name']}` ({m['source']})" if isinstance(m, dict)
        else f"`{m.original_name}` ({m.source})"
        for m in missing
    ) or "none"
    dup = stats["near_duplicate_pairs"]
    dup_text = (
        f"{len(dup)} near-duplicate pair(s) kept inside the same split"
        if dup else "none"
    )
    preview_count = stats["image_quality"]["preview_with_baked_annotations"]
    clean_count = stats["image_quality"]["clean"]
    preview_sources = [
        name for name, s in stats["sources"].items() if "preview" in s["image_type"]
    ]
    if preview_count:
        quality_text = (
            f"{clean_count} images are clean photographs. **{preview_count} images are\n"
            f"CVAT preview renders** with shape outlines, polygon fills and `Condition`\n"
            f"text burned into the pixels ({', '.join(preview_sources)}); they are useful\n"
            f"for annotation review but **must not be used for training**. The\n"
            f"`baked_in_annotations` column in `manifest.csv` marks every affected image.\n"
            f"\n"
            f"`data_clean.yaml` (+ `clean_{{train,val,test}}.txt`) selects only the\n"
            f"images without baked overlays."
        )
    else:
        quality_text = (
            f"All {clean_count} images are clean photographs and can be trained on\n"
            f"directly; `data_clean.yaml` and `data.yaml` select the same images."
        )
    relink_note = (
        f"\n- {stats['relinked_images']} flat-export images were relinked to clean\n"
        f"  originals; any remaining flat images stay flagged as previews."
        if stats.get("relinked_images") else ""
    )
    text = f"""# RiceAge grain dataset

Unified YOLO dataset built from **all annotation batches in the project**:

- Flat CVAT exports (`yashveer`, `yuvraj`, `yuvraj_02`, `yuvraj_03`, `yuvraj_04`)
- `Sakku.zip` — four nested CVAT batches (`amanAnnotations`, `cvat_sidd`, `sajalAnnotations`, `Tanishka`)

Both annotation styles are normalised into the same eight classes.

## Layout

```
dataset/
├── detect/                  # YOLO object detection
│   ├── data.yaml
│   ├── images/{{train,val,test}}/*.{{png,jpg}}
│   └── labels/{{train,val,test}}/*.txt   # class cx cy w h
├── segment/                 # YOLO instance segmentation
│   ├── data.yaml
│   ├── images/{{train,val,test}}/*.{{png,jpg}}   # hard links to detect/images
│   └── labels/{{train,val,test}}/*.txt           # class x1 y1 x2 y2 ...
├── manifest.csv             # per-image provenance, counts, image quality
├── missing_images.csv       # annotated images absent from the archives
├── stats.json
└── README.md
```

Total: **{stats['totals']['images']} images**, **{stats['totals']['shapes']} grain annotations**.

| Batch | Images | Annotations | Image type |
|---|---|---|---|
{source_rows}

| Split | Images | Annotations |
|---|---|---|
{split_rows}

## Classes

| Class | Annotations |
|---|---|
{class_rows}

Detection labels take each shape's bounding box; segmentation labels use the
polygon as-is and convert boxes to 4-corner rectangles.

## Training

```bash
pip install ultralytics

# recommended: clean subset (every image without baked overlays)
yolo detect train data=dataset/detect/data_clean.yaml model=yolo11n.pt imgsz=1280
yolo segment train data=dataset/segment/data_clean.yaml model=yolo11n-seg.pt imgsz=1280

# full set (includes any remaining preview renders)
yolo detect train data=dataset/detect/data.yaml model=yolo11n.pt imgsz=1280
yolo segment train data=dataset/segment/data.yaml model=yolo11n-seg.pt imgsz=1280
```

## Image quality — read this before training

{quality_text}

## Relinking clean originals

If you re-export the flat batches from CVAT without annotations, replace the
preview files (labels stay untouched; matching is by filename stem):

```bash
python tools/relink_clean_images.py --images path/to/originals --dry-run
python tools/relink_clean_images.py --images path/to/originals --apply
```

Do **not** rerun `build_yolo_dataset.py --force` afterwards; it re-extracts the
previews from the archives.

## Notes

- Split is 70/15/15 per batch, seed {SEED}, keeping duplicates together
  (near-duplicate pairs: {dup_text}).
- `segment/images` are hard links to `detect/images` (same inode); copying the
  folder to another drive expands them to real copies.
- Annotated images without a file in the archives: {miss}.
  Their annotations are listed in `missing_images.csv`.
- Class balance is imbalanced (whole dominates); `admixture` has only a handful
  of instances and appears only in the flat exports.
- Near-duplicate scans (including exact copies with different label sets) are
  kept in the same split and listed in `stats.json`; deduplicate before
  treating them as independent samples.
{relink_note}"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="replace an existing dataset/ directory")
    args = ap.parse_args()

    if OUT.exists():
        if not args.force:
            sys.exit(f"{OUT} already exists; pass --force to rebuild")
        shutil.rmtree(OUT)

    print("parsing annotations ...")
    recs, missing = parse_all()
    by_source = Counter(r.source for r in recs)
    for source, n in by_source.items():
        shapes = sum(len(r.shapes) for r in recs if r.source == source)
        print(f"  {source:20s} images={n:3d} shapes={shapes:6d}")
    print(f"  annotated but missing image: {len(missing)}")

    staging = OUT / ".staging"
    print("extracting images ...")
    extract_images(recs, staging)

    print("hashing and checking image quality ...")
    analyse_images(recs, staging)
    groups = duplicate_groups(recs)
    dup_pairs = []
    for members in groups.values():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    dup_pairs.append([
                        members[i].out_name,
                        members[j].out_name,
                        (members[i].dhash ^ members[j].dhash).bit_count(),
                    ])
    print(f"  near-duplicate pairs: {len(dup_pairs)}")

    plan_split(recs, groups)
    for split in SPLITS:
        sel = [r for r in recs if r.split == split]
        print(f"  {split}: {len(sel)} images, {sum(len(r.shapes) for r in sel)} shapes")

    det, seg = OUT / "detect", OUT / "segment"
    print("building YOLO trees ...")
    for r in recs:
        dest = det / "images" / r.split / r.out_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / r.out_name, dest)
        link = seg / "images" / r.split / r.out_name
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(dest, link)
        except OSError:
            shutil.copy2(dest, link)
        write_labels(r, det)
        write_labels(r, seg)
    write_yaml(det)
    write_yaml(seg)
    write_clean_config(det, recs)
    write_clean_config(seg, recs)
    shutil.rmtree(staging)

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "source", "original_name", "split", "width", "height",
            "boxes", "polygons", "classes", "image_format",
            "baked_in_annotations", "overlay_pct",
        ])
        for r in sorted(recs, key=lambda r: (r.split, r.out_name)):
            w.writerow([
                r.out_name, r.source, r.original_name, r.split, r.width, r.height,
                sum(1 for s in r.shapes if s.kind == "box"),
                sum(1 for s in r.shapes if s.kind == "polygon"),
                ";".join(f"{k}:{v}" for k, v in sorted(r.classes.items())),
                r.image_format, str(r.is_preview).lower(), f"{r.overlay_pct:.3f}",
            ])

    with open(OUT / "missing_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_name", "source", "width", "height", "shapes", "classes"])
        for r in missing:
            w.writerow([
                r.original_name, r.source, r.width, r.height, len(r.shapes),
                ";".join(f"{k}:{v}" for k, v in sorted(r.classes.items())),
            ])

    stats = {
        "classes": CLASSES,
        "sources": {},
        "splits": {},
        "totals": {},
        "image_quality": {
            "clean": sum(1 for r in recs if not r.is_preview),
            "preview_with_baked_annotations": sum(1 for r in recs if r.is_preview),
        },
        "near_duplicate_pairs": dup_pairs,
        "missing_images": [
            {"name": r.original_name, "source": r.source, "shapes": len(r.shapes)}
            for r in missing
        ],
    }
    for source in by_source:
        sel = [r for r in recs if r.source == source]
        stats["sources"][source] = {
            "images": len(sel),
            "shapes": sum(len(r.shapes) for r in sel),
            "classes": dict(sum((r.classes for r in sel), Counter()).most_common()),
            "image_type": (
                "clean photo" if not any(r.is_preview for r in sel)
                else "CVAT preview render"
            ),
        }
    for split in SPLITS:
        sel = [r for r in recs if r.split == split]
        stats["splits"][split] = {
            "images": len(sel),
            "shapes": sum(len(r.shapes) for r in sel),
            "classes": dict(sum((r.classes for r in sel), Counter()).most_common()),
        }
    stats["totals"] = {
        "images": len(recs),
        "shapes": sum(len(r.shapes) for r in recs),
        "classes": dict(sum((r.classes for r in recs), Counter()).most_common()),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    write_readme(stats, missing)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
