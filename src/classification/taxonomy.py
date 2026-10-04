# Standard grain quality taxonomy definition.

from enum import Enum


class RealDataTaxonomy(str, Enum):
    # Taxonomy verified and present in current controlled scanner clean rice data.
    WHOLE_RICE = "WHOLE_RICE"
    BROKEN_RICE = "BROKEN_RICE"  # Morphological broken pieces (<75% whole length)


class FutureDefectTaxonomy(str, Enum):
    # Full target production taxonomy to be populated by public and curated defect datasets. WARNING: Do NOT annotate or predict these defect classes from clean rice samples without real defect ground truth.
    WHOLE_RICE = "WHOLE_RICE"
    BROKEN_RICE = "BROKEN_RICE"
    CHALKY_RICE = "CHALKY_RICE"
    DISCOLORED_RICE = "DISCOLORED_RICE"
    DAMAGED_RICE = "DAMAGED_RICE"
    PEST_DAMAGED = "PEST_DAMAGED"
    UNRIPE_RICE = "UNRIPE_RICE"
    FOREIGN_MATERIAL = "FOREIGN_MATERIAL"
    OTHER_GRAIN = "OTHER_GRAIN"
