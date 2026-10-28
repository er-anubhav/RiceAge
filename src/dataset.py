# Dataset parsing, validation, and polygon rasterization for RiceAge dataset.

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os
import glob
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2


@dataclass
class AnnotatedGrain:
    grain_id: str
    image_id: int
    image_name: str
    category: str
    condition: str
    points: List[Tuple[float, float]]
    bbox_xyxy: Tuple[float, float, float, float]
    polygon_area_px: float

    @property
    def num_vertices(self) -> int:
        return len(self.points)

    def to_contour(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))


@dataclass
class DatasetImageRecord:
    image_id: int
    filename: str
    category: str
    width: int
    height: int
    total_grains: int
    counts_by_condition: Dict[str, int]
    image_path: str
    annotated_image_path: str
    annotation_xml_path: str
    grains: List[AnnotatedGrain] = field(default_factory=list)


class DatasetCatalog:
    def __init__(self, images: List[DatasetImageRecord], dataset_dir: str):
        self.images = images
        self.dataset_dir = dataset_dir
        self._by_id: Dict[int, DatasetImageRecord] = {img.image_id: img for img in images}
        self._by_filename: Dict[str, DatasetImageRecord] = {img.filename: img for img in images}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> DatasetImageRecord:
        return self.images[idx]

    def get_by_id(self, image_id: int) -> Optional[DatasetImageRecord]:
        return self._by_id.get(image_id)

    def get_by_filename(self, filename: str) -> Optional[DatasetImageRecord]:
        return self._by_filename.get(filename)

    def filter_by_category(self, category: str) -> List[DatasetImageRecord]:
        return [img for img in self.images if img.category.lower() == category.lower()]

    def filter_by_condition(self, condition: str) -> List[DatasetImageRecord]:
        return [img for img in self.images if any(g.condition.lower() == condition.lower() for g in img.grains)]

    @property
    def total_grains(self) -> int:
        return sum(img.total_grains for img in self.images)

    @property
    def condition_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for img in self.images:
            for g in img.grains:
                dist[g.condition] = dist.get(g.condition, 0) + 1
        return dist

    def split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[DatasetImageRecord], List[DatasetImageRecord], List[DatasetImageRecord]]:
        # Splits dataset by image while keeping duplicate image pairs in the same partition.
        np.random.seed(seed)
        categories = sorted(list({img.category for img in self.images}))
        train_set, val_set, test_set = [], [], []

        for cat in categories:
            cat_images = [img for img in self.images if img.category == cat]
            
            # Group identical images (e.g., whole 4.jpg and whole 5.jpg) into the same unit
            groups: List[List[DatasetImageRecord]] = []
            seen = set()
            for img in cat_images:
                if img.filename in seen:
                    continue
                if img.filename in ["whole 4.jpg", "whole 5.jpg"]:
                    dups = [i for i in cat_images if i.filename in ["whole 4.jpg", "whole 5.jpg"]]
                    groups.append(dups)
                    for d in dups:
                        seen.add(d.filename)
                else:
                    groups.append([img])
                    seen.add(img.filename)

            np.random.shuffle(groups)
            n_total = len(groups)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))

            train_groups = groups[:n_train]
            val_groups = groups[n_train:n_train + n_val]
            test_groups = groups[n_train + n_val:]

            for g in train_groups:
                train_set.extend(g)
            for g in val_groups:
                val_set.extend(g)
            for g in test_groups:
                test_set.extend(g)

        return train_set, val_set, test_set


def parse_cvat_xml(xml_path: str) -> List[Dict[str, Any]]:
    # Parse CVAT XML 1.1 format annotations into structured dictionary.
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Annotation XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    parsed_images = []

    for img_elem in root.findall("image"):
        im_id = int(img_elem.get("id", "0"))
        im_name = img_elem.get("name", "")
        im_w = int(img_elem.get("width", "0"))
        im_h = int(img_elem.get("height", "0"))

        polygons = []
        for p_elem in img_elem.findall("polygon"):
            label = p_elem.get("label", "grain")
            pts_str = p_elem.get("points", "")
            
            attrs = {}
            for a_elem in p_elem.findall("attribute"):
                attrs[a_elem.get("name")] = a_elem.text
            condition = attrs.get("Condition", "Unknown")

            pt_pairs = []
            for token in pts_str.split(";"):
                if not token.strip():
                    continue
                coords = [float(v.strip()) for v in token.split(",")]
                if len(coords) >= 2:
                    pt_pairs.append((coords[0], coords[1]))

            polygons.append({
                "label": label,
                "condition": condition,
                "points": pt_pairs,
                "pts_str": pts_str
            })

        parsed_images.append({
            "image_id": im_id,
            "filename": im_name,
            "width": im_w,
            "height": im_h,
            "polygons": polygons
        })

    return parsed_images


def rasterize_instance_mask(grains: List[AnnotatedGrain], shape: Tuple[int, int]) -> np.ndarray:
    # Convert list of AnnotatedGrain polygons into a 2D int32 instance mask (1-indexed).
    h, w = shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)

    for idx, grain in enumerate(grains, start=1):
        if len(grain.points) >= 3:
            pts = np.array(grain.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(instance_mask, [pts], idx)

    return instance_mask


def rasterize_semantic_mask(
    grains: List[AnnotatedGrain],
    shape: Tuple[int, int],
    class_map: Optional[Dict[str, int]] = None
) -> np.ndarray:
    # Convert list of AnnotatedGrain polygons into an integer semantic mask.
    default_map = {
        "whole": 1,
        "broken": 2,
        "chalky": 3,
        "damaged": 4,
        "foreign matter": 5,
        "discoloured": 6,
    }
    mapping = class_map or default_map
    h, w = shape[:2]
    semantic_mask = np.zeros((h, w), dtype=np.uint8)

    for grain in grains:
        cls_idx = mapping.get(grain.condition.lower(), 1)
        if len(grain.points) >= 3:
            pts = np.array(grain.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(semantic_mask, [pts], int(cls_idx))

    return semantic_mask


def load_dataset(dataset_dir: str = "data/dataset", repo_root: Optional[str] = None) -> DatasetCatalog:
    # Load and index complete RiceAge dataset from CSV and XML annotations.
    root = repo_root or os.getcwd()
    abs_dataset_dir = os.path.join(root, dataset_dir) if not os.path.isabs(dataset_dir) else dataset_dir
    metadata_csv = os.path.join(abs_dataset_dir, "metadata.csv")
    grains_csv = os.path.join(abs_dataset_dir, "grains_annotations.csv")

    if not os.path.exists(metadata_csv) or not os.path.exists(grains_csv):
        raise FileNotFoundError(f"Dataset CSV tables missing in {abs_dataset_dir}")

    meta_df = pd.read_csv(metadata_csv)
    grains_df = pd.read_csv(grains_csv)

    grains_by_image: Dict[str, List[AnnotatedGrain]] = {}
    for _, row in grains_df.iterrows():
        pts = []
        for pair in str(row["points"]).split(";"):
            if not pair.strip():
                continue
            coords = [float(v.strip()) for v in pair.split(",")]
            if len(coords) >= 2:
                pts.append((coords[0], coords[1]))

        grain = AnnotatedGrain(
            grain_id=str(row["grain_id"]),
            image_id=int(row["image_id"]),
            image_name=str(row["image_name"]),
            category=str(row["category"]),
            condition=str(row["condition"]),
            points=pts,
            bbox_xyxy=(float(row["bbox_xmin"]), float(row["bbox_ymin"]), float(row["bbox_xmax"]), float(row["bbox_ymax"])),
            polygon_area_px=float(row["polygon_area_px"])
        )
        grains_by_image.setdefault(grain.image_name, []).append(grain)

    images: List[DatasetImageRecord] = []
    for _, row in meta_df.iterrows():
        fname = str(row["filename"])
        im_grains = grains_by_image.get(fname, [])
        cond_counts = {
            "Whole": int(row.get("count_whole", 0)),
            "Broken": int(row.get("count_broken", 0)),
            "Chalky": int(row.get("count_chalky", 0)),
            "Discoloured": int(row.get("count_discoloured", 0)),
            "Damaged": int(row.get("count_damaged", 0)),
            "Foreign Matter": int(row.get("count_foreign_matter", 0)),
        }

        # Resolve paths relative to repository root
        im_path = os.path.join(root, str(row["image_path"]))
        ann_path = os.path.join(root, str(row["annotated_image_path"]))
        xml_path = os.path.join(root, str(row["annotation_xml_path"]))

        record = DatasetImageRecord(
            image_id=int(row["image_id"]),
            filename=fname,
            category=str(row["category"]),
            width=int(row["width"]),
            height=int(row["height"]),
            total_grains=int(row["total_grains"]),
            counts_by_condition=cond_counts,
            image_path=im_path,
            annotated_image_path=ann_path,
            annotation_xml_path=xml_path,
            grains=im_grains
        )
        images.append(record)

    return DatasetCatalog(images=images, dataset_dir=abs_dataset_dir)


def validate_dataset(dataset_dir: str = "data/dataset") -> Dict[str, Any]:
    # Run full programmatic verification of dataset and return integrity dictionary.
    catalog = load_dataset(dataset_dir)
    total_imgs = len(catalog)
    total_grains = catalog.total_grains

    missing_raw = [img.filename for img in catalog.images if not os.path.exists(img.image_path)]
    missing_ann = [img.filename for img in catalog.images if not os.path.exists(img.annotated_image_path)]

    malformed_polygons = 0
    out_of_bounds_pts = 0
    for img in catalog.images:
        for g in img.grains:
            if len(g.points) < 3:
                malformed_polygons += 1
            for x, y in g.points:
                if x < 0 or x > img.width or y < 0 or y > img.height:
                    out_of_bounds_pts += 1

    return {
        "total_images": total_imgs,
        "total_grains": total_grains,
        "condition_distribution": catalog.condition_distribution,
        "missing_raw_images": len(missing_raw),
        "missing_annotated_images": len(missing_ann),
        "malformed_polygons": malformed_polygons,
        "out_of_bounds_points": out_of_bounds_pts,
        "is_valid": len(missing_raw) == 0 and len(missing_ann) == 0 and malformed_polygons == 0 and out_of_bounds_pts == 0
    }
