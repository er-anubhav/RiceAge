"""Rule-based morphological quality classifier based on international grain standards."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .taxonomy import RealDataTaxonomy


def classify_grains_rule_based(
    measurements_df: pd.DataFrame,
    broken_length_ratio_thresh: float = 0.75,
    broken_aspect_ratio_thresh: float = 1.6
) -> pd.DataFrame:
    """Classify grains into WHOLE_RICE, BROKEN_RICE, or UNRESOLVED_CLUSTER based on morphology.
    
    Standard definition (Codex Alimentarius CXS 198-1995 / USDA / IRRI):
    - Broken kernel: A piece of kernel that is smaller than three-quarters (75%)
      of the average length of the unbroken kernel.
    
    Returns:
        DataFrame augmented with 'quality_class' and 'classification_reason'.
    """
    df = measurements_df.copy()
    if df.empty:
        df["quality_class"] = []
        df["classification_reason"] = []
        return df

    # Filter isolated single or separated grains to estimate median unbroken length
    unbroken_candidates = df[
        (df["separation_status"].isin(["isolated_single", "separated_from_cluster"])) &
        (~df["is_boundary"]) &
        (df["solidity"] >= 0.90) &
        (df["aspect_ratio"] >= 1.8)
    ]

    if not unbroken_candidates.empty:
        median_whole_length = float(unbroken_candidates["length_px"].median())
    else:
        median_whole_length = float(df["length_px"].median())

    quality_classes = []
    reasons = []

    for _, row in df.iterrows():
        status = row.get("separation_status", "")
        length = row["length_px"]
        ar = row["aspect_ratio"]
        area = row["area_px"]

        if status == "under_segmented_cluster":
            quality_classes.append("UNRESOLVED_CLUSTER")
            reasons.append("Multi-grain cluster unresolved by watershed")
        elif row["is_boundary"]:
            quality_classes.append("BOUNDARY_TRUNCATED")
            reasons.append("Grain touches image boundary (truncated measurements)")
        elif length < (broken_length_ratio_thresh * median_whole_length) or ar < broken_aspect_ratio_thresh:
            quality_classes.append(RealDataTaxonomy.BROKEN_RICE.value)
            reasons.append(f"Length {length:.1f}px < 75% of median unbroken ({median_whole_length:.1f}px)")
        else:
            quality_classes.append(RealDataTaxonomy.WHOLE_RICE.value)
            reasons.append("Length >= 75% median unbroken length and aspect ratio >= 1.6")

    df["quality_class"] = quality_classes
    df["classification_reason"] = reasons
    return df
