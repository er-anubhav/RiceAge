#!/usr/bin/env python3
"""Build reviewable YOLO instance-segmentation candidates from RiceAge M0 outputs.

This exports pseudo-labels from the classical M0 instance mask. They are NOT
human-verified ground truth and should not be used for training until reviewed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRUSTED_CANDIDATE_STATUSES = {"isolated_single", "separated_from_cluster"}
MANIFEST_FIELDS = [
    "image_id", "source_image", "output_image", "label_file", "instance_mask",
    "overlay", "width", "height", "pipeline_grains", "candidate_instances",
    "excluded_instances", "label_source", "review_status", "training_ready",
]


def make_image_id(relative_path: str | Path) -> str:
    """Create a stable, collision-resistant identifier from a relative path."""
    normalized = Path(relative_path).as_posix()
    stem = str(Path(normalized).with_suffix(""))
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:64] or "scan"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}"


def discover_images(
    input_path: str | Path,
    *,
    recursive: bool = True,
    excluded_root: str | Path | None = None,
) -> list[Path]:
    """Find supported source images, excluding generated output folders."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    excluded = Path(excluded_root).expanduser().resolve() if excluded_root else None

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]

    iterator = source.rglob("*") if recursive else source.iterdir()
    images: list[Path] = []
    for candidate in iterator:
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = candidate.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            continue
        images.append(resolved)
    return sorted(images, key=lambda item: item.as_posix().lower())


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def polygon_for_instance(
    instance_mask: np.ndarray,
    grain_id: int,
    *,
    minimum_mask_area_px: int = 20,
    epsilon_px: float = 0.8,
) -> dict[str, Any] | None:
    """Return a normalized full-image YOLO polygon for one instance ID."""
    if instance_mask.ndim != 2:
        raise ValueError("instance_mask must be a 2D integer label map")
    height, width = instance_mask.shape[:2]
    if width < 2 or height < 2:
        return None

    binary = (instance_mask == int(grain_id)).astype(np.uint8)
    pixel_area = int(cv2.countNonZero(binary))
    if pixel_area < minimum_mask_area_px:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    epsilon = max(0.0, float(epsilon_px))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3 or len(np.unique(polygon, axis=0)) < 3:
        return None

    normalized: list[tuple[float, float]] = []
    for x, y in polygon:
        nx = min(1.0, max(0.0, float(x) / float(width - 1)))
        ny = min(1.0, max(0.0, float(y) / float(height - 1)))
        normalized.append((nx, ny))

    coordinates = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
    return {
        "line": f"0 {coordinates}",
        "point_count": len(normalized),
        "pixel_area": pixel_area,
    }


def build_yolo_annotations(
    instance_mask: np.ndarray,
    measurements: Iterable[dict[str, Any]],
    *,
    minimum_mask_area_px: int = 20,
    epsilon_px: float = 0.8,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build single-class YOLO labels and an audit record for every measured grain.

    Only M0 instances with an isolated/separated status and no boundary flag are
    exported as candidate YOLO polygons. All other instances are retained in
    metadata for review, but not written into the candidate training labels.
    """
    label_lines: list[str] = []
    audit: list[dict[str, Any]] = []

    for row in measurements:
        try:
            grain_id = int(row["grain_id"])
        except (KeyError, TypeError, ValueError):
            audit.append({
                "grain_id": None,
                "included_in_candidate_labels": False,
                "exclusion_reason": "missing_or_invalid_grain_id",
            })
            continue

        status = str(row.get("separation_status", "") or "").strip().lower()
        is_boundary = _as_bool(row.get("is_boundary", False))
        quality_class = str(row.get("quality_class", "") or "")
        reason: str | None = None

        if is_boundary:
            reason = "boundary_truncated"
        elif status not in TRUSTED_CANDIDATE_STATUSES:
            reason = f"untrusted_m0_separation_status:{status or 'missing'}"
        elif quality_class == "UNRESOLVED_CLUSTER":
            reason = "unresolved_cluster"

        polygon = None
        if reason is None:
            polygon = polygon_for_instance(
                instance_mask,
                grain_id,
                minimum_mask_area_px=minimum_mask_area_px,
                epsilon_px=epsilon_px,
            )
            if polygon is None:
                reason = "missing_or_too_small_instance_mask"

        included = polygon is not None and reason is None
        if included:
            label_lines.append(polygon["line"])

        bbox = {
            key: row.get(key)
            for key in ("x", "y", "width", "height")
            if key in row
        }
        audit.append({
            "grain_id": grain_id,
            "quality_class_m0": quality_class,
            "separation_status": status,
            "is_boundary": is_boundary,
            "bbox": bbox,
            "included_in_candidate_labels": included,
            "exclusion_reason": reason,
            "polygon_points": polygon["point_count"] if polygon else 0,
            "mask_pixel_area": polygon["pixel_area"] if polygon else 0,
            "label_source": "m0_pseudolabel",
            "review_status": "unreviewed",
        })

    return label_lines, audit


def remap_instance_ids(instance_mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Create a compact unsigned instance map and mapping from saved ID to M0 ID."""
    if instance_mask.ndim != 2:
        raise ValueError("instance_mask must be a 2D integer label map")
    unique_values, inverse = np.unique(instance_mask, return_inverse=True)
    lookup = np.zeros(len(unique_values), dtype=np.uint32)
    exported_to_original: dict[str, int] = {}
    next_id = 1
    for index, value in enumerate(unique_values):
        original_id = int(value)
        if original_id <= 0:
            continue
        lookup[index] = next_id
        exported_to_original[str(next_id)] = original_id
        next_id += 1
    remapped = lookup[inverse].reshape(instance_mask.shape)
    return remapped, exported_to_original


def save_instance_map(instance_mask: np.ndarray, output_dir: Path, image_id: str) -> tuple[Path, dict[str, int]]:
    """Save a lossless, compact instance-ID map as uint16 PNG or uint32 NPY."""
    remapped, id_map = remap_instance_ids(instance_mask)
    if len(id_map) <= np.iinfo(np.uint16).max:
        output_path = output_dir / f"{image_id}.png"
        ok = cv2.imwrite(str(output_path), remapped.astype(np.uint16))
        if not ok:
            raise OSError(f"Could not write instance map: {output_path}")
    else:
        output_path = output_dir / f"{image_id}.npy"
        np.save(output_path, remapped.astype(np.uint32))
    return output_path, id_map


def render_review_overlay(
    image_bgr: np.ndarray,
    instance_mask: np.ndarray,
    audit: list[dict[str, Any]],
) -> np.ndarray:
    """Draw green outlines for candidate instances and red outlines for excluded ones.

    Restrict contour extraction to each measured bounding box. Searching the full
    high-resolution scan once per grain is unnecessarily expensive.
    """
    overlay = image_bgr.copy()
    image_height, image_width = instance_mask.shape[:2]
    for item in audit:
        if item.get("grain_id") is None:
            continue
        bbox = item.get("bbox") or {}
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            width = int(bbox["width"])
            height = int(bbox["height"])
            grain_id = int(item["grain_id"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            continue

        region = (instance_mask[y1:y2, x1:x2] == grain_id).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 220, 0) if item.get("included_in_candidate_labels") else (0, 0, 230)
        if contours:
            cv2.drawContours(
                overlay, contours, -1, color, 1, cv2.LINE_AA, offset=(x1, y1)
            )
    return overlay


def write_dataset_readme(output_root: Path) -> None:
    content = """# RiceAge segmentation candidates

These files are post-processed M0 pseudo-label candidates, not verified ground truth.

- images/: copies of full source images, at the original dimensions.
- labels/: YOLO instance-segmentation polygon labels for candidate instances only. Class 0 means rice_grain.
- instance_masks/: lossless instance-ID maps (PNG or NPY); use the per-image JSON mapping to map saved IDs to M0 grain IDs.
- overlays/: green outlines are candidate label instances; red outlines are excluded/suspect M0 instances.
- metadata/: per-image measurements, inclusion/exclusion reasons, and ID mapping.
- manifest.csv: inventory and counts.

The current M0 pipeline uses classical image processing and watershed. Its masks can be wrong, especially for touching grains, fragments, and boundary-truncated grains. All candidate labels have label_source=m0_pseudolabel and review_status=unreviewed. This directory intentionally has no data.yaml or train/validation/test split: do not train from it until the masks have been visually reviewed and corrected. Split reviewed images by original scan or physical sample batch to avoid leakage.
"""
    (output_root / "README.md").write_text(content, encoding="utf-8")


def process_images(args: argparse.Namespace) -> int:
    from configs.default_config import ExportConfig, PipelineConfig
    from src.pipeline import RiceQualityPipeline

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subdirs = {
        name: output_root / name
        for name in ("images", "labels", "instance_masks", "overlays", "metadata")
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    images = discover_images(
        input_path,
        recursive=not args.non_recursive,
        excluded_root=output_root,
    )
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    if not images:
        logging.error("No supported images found under %s", input_path)
        return 2

    source_root = input_path if input_path.is_dir() else input_path.parent
    config = PipelineConfig(
        export=ExportConfig(
            save_crops=False,
            save_masks=False,
            persist_feedback=False,
            output_dir=str(output_root / "_unused_pipeline_output"),
            visualizations_dir=str(output_root / "_unused_pipeline_output" / "visualizations"),
            crops_dir=str(output_root / "_unused_pipeline_output" / "crops"),
            masks_dir=str(output_root / "_unused_pipeline_output" / "masks"),
            reports_dir=str(output_root / "_unused_pipeline_output" / "reports"),
        )
    )
    pipeline = RiceQualityPipeline(config)
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, source_image in enumerate(images, start=1):
        try:
            relative = source_image.relative_to(source_root).as_posix()
        except ValueError:
            relative = source_image.name
        image_id = make_image_id(relative)
        logging.info("[%d/%d] Processing %s", index, len(images), source_image)

        try:
            image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("OpenCV could not decode the image")
            result = pipeline.run(
                str(source_image),
                raw_reference_path=str(args.raw_reference) if args.raw_reference else None,
                generate_outputs=False,
            )
            instance_mask = np.asarray(result["instance_mask"])
            measurements_df = result["measurements_df"]
            measurements = measurements_df.to_dict(orient="records")

            label_lines, audit = build_yolo_annotations(
                instance_mask,
                measurements,
                minimum_mask_area_px=args.min_mask_area_px,
                epsilon_px=args.polygon_epsilon_px,
            )

            suffix = source_image.suffix.lower()
            image_output = subdirs["images"] / f"{image_id}{suffix}"
            label_output = subdirs["labels"] / f"{image_id}.txt"
            overlay_output = subdirs["overlays"] / f"{image_id}.jpg"
            metadata_output = subdirs["metadata"] / f"{image_id}.json"

            shutil.copy2(source_image, image_output)
            label_output.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            instance_map_path, id_map = save_instance_map(
                instance_mask, subdirs["instance_masks"], image_id
            )
            overlay = render_review_overlay(image_bgr, instance_mask, audit)
            if not cv2.imwrite(str(overlay_output), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"Could not write review overlay: {overlay_output}")

            summary = result.get("summary", {})
            width = int(image_bgr.shape[1])
            height = int(image_bgr.shape[0])
            included_count = sum(bool(row.get("included_in_candidate_labels")) for row in audit)
            metadata = {
                "schema_version": 1,
                "image_id": image_id,
                "source_image": str(source_image),
                "source_relative_path": relative,
                "image_dimensions": {"width": width, "height": height},
                "class_names": {"0": "rice_grain"},
                "label_format": "YOLO instance segmentation polygon",
                "label_source": "m0_pseudolabel",
                "review_status": "unreviewed",
                "training_ready": False,
                "warning": (
                    "M0-generated pseudo-labels require human review; they are not verified ground truth."
                ),
                "m0_scan_id": result.get("scan_id"),
                "m0_summary": summary,
                "candidate_instances": included_count,
                "excluded_instances": len(audit) - included_count,
                "instance_mask_id_map": id_map,
                "instances": audit,
            }
            metadata_output.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )

            manifest_rows.append({
                "image_id": image_id,
                "source_image": str(source_image),
                "output_image": str(image_output.relative_to(output_root)),
                "label_file": str(label_output.relative_to(output_root)),
                "instance_mask": str(instance_map_path.relative_to(output_root)),
                "overlay": str(overlay_output.relative_to(output_root)),
                "width": width,
                "height": height,
                "pipeline_grains": len(measurements),
                "candidate_instances": included_count,
                "excluded_instances": len(audit) - included_count,
                "label_source": "m0_pseudolabel",
                "review_status": "unreviewed",
                "training_ready": False,
            })
            logging.info(
                "  -> %d candidate polygons, %d excluded/suspect instances",
                included_count, len(audit) - included_count,
            )
        except Exception as exc:
            logging.exception("Failed processing %s", source_image)
            failures.append({"source_image": str(source_image), "error": f"{type(exc).__name__}: {exc}"})

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    errors_path = output_root / "errors.json"
    errors_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    write_dataset_readme(output_root)

    print(f"Processed images: {len(manifest_rows)} / {len(images)}")
    print(f"Candidate polygons: {sum(int(row['candidate_instances']) for row in manifest_rows)}")
    print(f"Excluded/suspect instances: {sum(int(row['excluded_instances']) for row in manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    print("Status: UNREVIEWED M0 PSEUDO-LABELS — NOT READY FOR TRAINING")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create reviewable YOLO segmentation candidates from RiceAge M0 instance masks."
    )
    parser.add_argument("--input", required=True, help="Source image or directory containing source images")
    parser.add_argument(
        "--output",
        default="data/processed/segmentation_candidates",
        help="Output root for image copies, YOLO labels, masks, overlays, and manifest",
    )
    parser.add_argument(
        "--raw-reference",
        default=None,
        help="Optional raw scanner reference image used by RiceAge calibration",
    )
    parser.add_argument("--non-recursive", action="store_true", help="Do not search subdirectories")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for a small pilot run")
    parser.add_argument("--min-mask-area-px", type=int, default=20, help="Minimum pixels per candidate mask")
    parser.add_argument("--polygon-epsilon-px", type=float, default=0.8, help="Contour simplification tolerance")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_mask_area_px < 1:
        parser.error("--min-mask-area-px must be >= 1")
    if args.polygon_epsilon_px < 0:
        parser.error("--polygon-epsilon-px must be >= 0")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >= 1")
    if args.raw_reference and not Path(args.raw_reference).expanduser().exists():
        parser.error(f"--raw-reference does not exist: {args.raw_reference}")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return process_images(args)


if __name__ == "__main__":
    raise SystemExit(main())
