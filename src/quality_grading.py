# Quality grading engine for RiceAge Milestone M3.
# Computes sample-level counts and percentages, enforces explicit denominator rules,
# prevents double-counting physical grains with coexisting defects, and applies
# configurable standards-based grading profiles (Codex, USDA, Indian Standard IS 4333).

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set
import json


@dataclass
class GrainClassification:
    """Individual grain classification output from multi-head models."""
    object_id: str
    is_rice: bool
    is_foreign_matter: bool
    morphology: str  # "Whole", "Broken", or "N/A" (for foreign matter)
    is_chalky: bool = False
    is_damaged: bool = False
    is_discoloured: bool = False
    confidences: Dict[str, float] = field(default_factory=dict)


@dataclass
class SampleQualityMetrics:
    """Aggregated sample-level counts, percentages, and explicit denominator rules."""
    sample_id: str

    # Object counts
    total_detected_objects: int
    total_rice_grains: int
    total_foreign_matter: int

    # Grain condition counts (on rice grains)
    whole_count: int
    broken_count: int
    chalky_count: int
    damaged_count: int
    discoloured_count: int

    # Coexisting defect count (grains with >= 2 defect flags)
    multi_defect_rice_count: int

    # Percentages
    whole_percentage: float
    broken_percentage: float
    chalky_percentage: float
    damaged_percentage: float
    discoloured_percentage: float
    foreign_matter_percentage: float

    # Explicit denominator definitions
    denominator_definitions: Dict[str, str] = field(default_factory=lambda: {
        "whole_percentage": "count(Whole Rice) / total_rice_grains * 100",
        "broken_percentage": "count(Broken Rice) / total_rice_grains * 100",
        "chalky_percentage": "count(Chalky Rice) / total_rice_grains * 100",
        "damaged_percentage": "count(Damaged Rice) / total_rice_grains * 100",
        "discoloured_percentage": "count(Discoloured Rice) / total_rice_grains * 100",
        "foreign_matter_percentage": "total_foreign_matter / total_detected_objects * 100"
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GradeThreshold:
    """Maximum allowable defect limits for a specific grade tier."""
    max_broken_pct: Optional[float] = None
    max_chalky_pct: Optional[float] = None
    max_damaged_pct: Optional[float] = None
    max_discoloured_pct: Optional[float] = None
    max_foreign_matter_pct: Optional[float] = None
    min_whole_pct: Optional[float] = None


@dataclass
class GradingProfile:
    """Formal quality standard profile containing grade tiers and citations."""
    profile_id: str
    standard_name: str
    issuing_authority: str
    citation: str
    grades: Dict[str, GradeThreshold]
    unsupported_criteria: List[str] = field(default_factory=list)


@dataclass
class GradingDecision:
    """Formal grading evaluation outcome."""
    profile_id: str
    standard_name: str
    assigned_grade: Optional[str]
    status: str  # "DECIDED", "INDETERMINATE", "INCOMPLETE"
    is_compliant: bool
    passed_criteria: Dict[str, bool]
    failed_criteria: Dict[str, str]
    unsupported_criteria: List[str]
    audit_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------------
# STANDARD PROFILES DEFINITION
# -------------------------------------------------------------------------

CODEX_CXS_198_1995 = GradingProfile(
    profile_id="CODEX_CXS_198_1995",
    standard_name="Codex Standard for Rice (CXS 198-1995, Revised 2019)",
    issuing_authority="Codex Alimentarius Commission (FAO / WHO)",
    citation="Codex Alimentarius Standard CXS 198-1995: Section 3 Quality Factors",
    grades={
        "Extra": GradeThreshold(
            max_broken_pct=5.0,
            max_chalky_pct=2.0,
            max_damaged_pct=1.5,
            max_discoloured_pct=0.5,
            max_foreign_matter_pct=0.1
        ),
        "Grade_1": GradeThreshold(
            max_broken_pct=10.0,
            max_chalky_pct=3.0,
            max_damaged_pct=2.0,
            max_discoloured_pct=1.0,
            max_foreign_matter_pct=0.2
        ),
        "Grade_2": GradeThreshold(
            max_broken_pct=15.0,
            max_chalky_pct=5.0,
            max_damaged_pct=3.0,
            max_discoloured_pct=2.0,
            max_foreign_matter_pct=0.5
        ),
        "Grade_3": GradeThreshold(
            max_broken_pct=25.0,
            max_chalky_pct=8.0,
            max_damaged_pct=4.0,
            max_discoloured_pct=3.0,
            max_foreign_matter_pct=1.0
        ),
    },
    unsupported_criteria=[
        "moisture_content (Max 15% m/m - Requires oven/moisture meter)",
        "odor (Must be free from abnormal flavors and odors)",
        "live_insects (Free from live insects)"
    ]
)

USDA_MILLED_RICE = GradingProfile(
    profile_id="USDA_MILLED_RICE",
    standard_name="United States Standards for Milled Rice (7 CFR Part 868)",
    issuing_authority="USDA Federal Grain Inspection Service (FGIS)",
    citation="USDA FGIS Milled Rice Standards (Effective July 1, 2020)",
    grades={
        "US_No_1": GradeThreshold(
            max_broken_pct=4.0,
            max_chalky_pct=2.0,
            max_damaged_pct=0.5,
            max_discoloured_pct=0.1,
            max_foreign_matter_pct=0.1
        ),
        "US_No_2": GradeThreshold(
            max_broken_pct=7.0,
            max_chalky_pct=4.0,
            max_damaged_pct=1.0,
            max_discoloured_pct=0.2,
            max_foreign_matter_pct=0.2
        ),
        "US_No_3": GradeThreshold(
            max_broken_pct=15.0,
            max_chalky_pct=6.0,
            max_damaged_pct=1.5,
            max_discoloured_pct=0.5,
            max_foreign_matter_pct=0.5
        ),
        "US_No_4": GradeThreshold(
            max_broken_pct=25.0,
            max_chalky_pct=8.0,
            max_damaged_pct=2.0,
            max_discoloured_pct=1.0,
            max_foreign_matter_pct=1.0
        ),
        "US_No_5": GradeThreshold(
            max_broken_pct=35.0,
            max_chalky_pct=10.0,
            max_damaged_pct=3.0,
            max_discoloured_pct=2.0,
            max_foreign_matter_pct=1.5
        ),
        "US_No_6": GradeThreshold(
            max_broken_pct=50.0,
            max_chalky_pct=15.0,
            max_damaged_pct=4.0,
            max_discoloured_pct=3.0,
            max_foreign_matter_pct=2.0
        ),
    },
    unsupported_criteria=[
        "moisture_content (Max 14.0% - Requires moisture meter)",
        "red_rice_and_damaged_kernels_by_mass (Requires gravimetric scale)",
        "odor (Must be free from commercially objectionable odors)"
    ]
)

IS_4333_AGMARK = GradingProfile(
    profile_id="IS_4333_AGMARK",
    standard_name="Indian Standard IS 4333 / AGMARK Rules for Milled Rice",
    issuing_authority="Bureau of Indian Standards / Directorate of Marketing & Inspection",
    citation="IS 4333 (Part 1 to 5): Methods of Analysis for Foodgrains / AGMARK Rice Grading Rules",
    grades={
        "Grade_I": GradeThreshold(
            max_broken_pct=5.0,
            max_chalky_pct=3.0,
            max_damaged_pct=1.0,
            max_discoloured_pct=1.0,
            max_foreign_matter_pct=0.5
        ),
        "Grade_II": GradeThreshold(
            max_broken_pct=10.0,
            max_chalky_pct=5.0,
            max_damaged_pct=2.0,
            max_discoloured_pct=2.0,
            max_foreign_matter_pct=1.0
        ),
        "Grade_III": GradeThreshold(
            max_broken_pct=20.0,
            max_chalky_pct=8.0,
            max_damaged_pct=3.0,
            max_discoloured_pct=3.0,
            max_foreign_matter_pct=1.5
        ),
    },
    unsupported_criteria=[
        "moisture_content (Max 14% - Requires IS 4333 Part 2 test)",
        "inorganic_vs_organic_foreign_matter_gravimetric (Requires chemical balance)",
        "uric_acid_content (Requires enzymatic test for rodent contamination)"
    ]
)

STANDARD_PROFILES: Dict[str, GradingProfile] = {
    "CODEX_CXS_198_1995": CODEX_CXS_198_1995,
    "USDA_MILLED_RICE": USDA_MILLED_RICE,
    "IS_4333_AGMARK": IS_4333_AGMARK
}


# -------------------------------------------------------------------------
# COMPUTATION & GRADING LOGIC
# -------------------------------------------------------------------------

def compute_sample_quality_metrics(
    classifications: List[GrainClassification],
    sample_id: str = "SAMPLE_001"
) -> SampleQualityMetrics:
    """
    Compute rigorous sample-level counts and percentages from grain classifications.
    Guarantees:
    1. Total rice grains = Whole count + Broken count.
    2. Grains with multiple coexisting defects (e.g. Broken + Chalky) are never double-counted.
    3. Percentages are bounded in [0.0, 100.0] with explicit denominator definitions.
    """
    total_objects = len(classifications)
    if total_objects == 0:
        return SampleQualityMetrics(
            sample_id=sample_id,
            total_detected_objects=0,
            total_rice_grains=0,
            total_foreign_matter=0,
            whole_count=0,
            broken_count=0,
            chalky_count=0,
            damaged_count=0,
            discoloured_count=0,
            multi_defect_rice_count=0,
            whole_percentage=0.0,
            broken_percentage=0.0,
            chalky_percentage=0.0,
            damaged_percentage=0.0,
            discoloured_percentage=0.0,
            foreign_matter_percentage=0.0
        )

    fm_count = 0
    whole_count = 0
    broken_count = 0
    chalky_count = 0
    damaged_count = 0
    disc_count = 0
    multi_defect_count = 0

    for g in classifications:
        if g.is_foreign_matter or not g.is_rice:
            fm_count += 1
        else:
            # Grain is verified rice
            if g.morphology == "Whole":
                whole_count += 1
            else:
                broken_count += 1

            defect_flags = 0
            if g.is_chalky:
                chalky_count += 1
                defect_flags += 1
            if g.is_damaged:
                damaged_count += 1
                defect_flags += 1
            if g.is_discoloured:
                disc_count += 1
                defect_flags += 1

            if defect_flags >= 2:
                multi_defect_count += 1

    total_rice = whole_count + broken_count

    # Denominators:
    # Defect percentages are evaluated on total rice grains
    # Foreign matter percentage is evaluated on total sample objects
    if total_rice > 0:
        whole_pct = round((whole_count / total_rice) * 100.0, 2)
        broken_pct = round((broken_count / total_rice) * 100.0, 2)
        chalky_pct = round((chalky_count / total_rice) * 100.0, 2)
        damaged_pct = round((damaged_count / total_rice) * 100.0, 2)
        disc_pct = round((disc_count / total_rice) * 100.0, 2)
    else:
        whole_pct = broken_pct = chalky_pct = damaged_pct = disc_pct = 0.0

    fm_pct = round((fm_count / total_objects) * 100.0, 2) if total_objects > 0 else 0.0

    return SampleQualityMetrics(
        sample_id=sample_id,
        total_detected_objects=total_objects,
        total_rice_grains=total_rice,
        total_foreign_matter=fm_count,
        whole_count=whole_count,
        broken_count=broken_count,
        chalky_count=chalky_count,
        damaged_count=damaged_count,
        discoloured_count=disc_count,
        multi_defect_rice_count=multi_defect_count,
        whole_percentage=whole_pct,
        broken_percentage=broken_pct,
        chalky_percentage=chalky_pct,
        damaged_percentage=damaged_pct,
        discoloured_percentage=disc_pct,
        foreign_matter_percentage=fm_pct
    )


def evaluate_quality_grade(
    metrics: SampleQualityMetrics,
    profile: Optional[GradingProfile] = None,
    profile_id: Optional[str] = "CODEX_CXS_198_1995",
    require_full_criteria: bool = False
) -> GradingDecision:
    """
    Evaluate sample metrics against a formal grading profile.
    
    If require_full_criteria is True, and the profile specifies physical criteria
    unsupported by optical scan imagery (e.g., moisture, odor), the grading status
    is marked 'INCOMPLETE'.
    Otherwise, if evaluated purely on optical criteria, the optical grade is decided
    and audit notes explicitly list the unverified criteria.
    """
    if profile is None:
        if profile_id not in STANDARD_PROFILES:
            raise ValueError(f"Unknown profile_id: {profile_id}. Available: {list(STANDARD_PROFILES.keys())}")
        profile = STANDARD_PROFILES[profile_id]

    if metrics.total_rice_grains == 0:
        return GradingDecision(
            profile_id=profile.profile_id,
            standard_name=profile.standard_name,
            assigned_grade=None,
            status="INDETERMINATE",
            is_compliant=False,
            passed_criteria={},
            failed_criteria={"total_rice_grains": "Sample contains 0 rice grains."},
            unsupported_criteria=profile.unsupported_criteria,
            audit_notes=["Cannot grade sample: 0 valid rice grains detected."]
        )

    # Check each grade tier from highest quality to lowest
    assigned_grade = None
    passed_for_assigned = {}
    failed_reasons = {}

    for grade_name, limits in profile.grades.items():
        grade_passed = True
        current_passed = {}
        current_failed = {}

        # 1. Broken Rice Limit
        if limits.max_broken_pct is not None:
            if metrics.broken_percentage <= limits.max_broken_pct:
                current_passed["broken_rice"] = True
            else:
                grade_passed = False
                current_passed["broken_rice"] = False
                current_failed["broken_rice"] = f"Broken {metrics.broken_percentage}% exceeds max {limits.max_broken_pct}%"

        # 2. Chalky Rice Limit
        if limits.max_chalky_pct is not None:
            if metrics.chalky_percentage <= limits.max_chalky_pct:
                current_passed["chalky_rice"] = True
            else:
                grade_passed = False
                current_passed["chalky_rice"] = False
                current_failed["chalky_rice"] = f"Chalky {metrics.chalky_percentage}% exceeds max {limits.max_chalky_pct}%"

        # 3. Damaged Rice Limit
        if limits.max_damaged_pct is not None:
            if metrics.damaged_percentage <= limits.max_damaged_pct:
                current_passed["damaged_rice"] = True
            else:
                grade_passed = False
                current_passed["damaged_rice"] = False
                current_failed["damaged_rice"] = f"Damaged {metrics.damaged_percentage}% exceeds max {limits.max_damaged_pct}%"

        # 4. Discoloured Rice Limit
        if limits.max_discoloured_pct is not None:
            if metrics.discoloured_percentage <= limits.max_discoloured_pct:
                current_passed["discoloured_rice"] = True
            else:
                grade_passed = False
                current_passed["discoloured_rice"] = False
                current_failed["discoloured_rice"] = f"Discoloured {metrics.discoloured_percentage}% exceeds max {limits.max_discoloured_pct}%"

        # 5. Foreign Matter Limit
        if limits.max_foreign_matter_pct is not None:
            if metrics.foreign_matter_percentage <= limits.max_foreign_matter_pct:
                current_passed["foreign_matter"] = True
            else:
                grade_passed = False
                current_passed["foreign_matter"] = False
                current_failed["foreign_matter"] = f"Foreign Matter {metrics.foreign_matter_percentage}% exceeds max {limits.max_foreign_matter_pct}%"

        if grade_passed:
            assigned_grade = grade_name
            passed_for_assigned = current_passed
            break
        else:
            failed_reasons[grade_name] = current_failed

    notes = [
        f"Optical screening estimate evaluated using optical criteria under {profile.standard_name}.",
        f"Authority: {profile.issuing_authority}.",
        "Screening Disclaimer: Count-based optical screening does not substitute for certified gravimetric mass-fraction grading. Non-optical parameters (moisture, odor, chemical analysis) are unmeasured."
    ]

    if require_full_criteria and len(profile.unsupported_criteria) > 0:
        status = "INCOMPLETE"
        assigned_grade = None
        notes.append("Status is marked INCOMPLETE because required non-optical regulatory criteria (e.g. moisture content) cannot be verified from scanner images.")
    elif assigned_grade is not None:
        status = "DECIDED"
        notes.append(f"Sample meets optical requirements for {assigned_grade}.")
    else:
        status = "DECIDED"
        assigned_grade = "Substandard / Below FAQ"
        notes.append("Sample exceeds maximum allowed defect limits for all defined grade tiers.")

    return GradingDecision(
        profile_id=profile.profile_id,
        standard_name=profile.standard_name,
        assigned_grade=assigned_grade,
        status=status,
        is_compliant=(assigned_grade is not None and assigned_grade != "Substandard / Below FAQ"),
        passed_criteria=passed_for_assigned,
        failed_criteria=failed_reasons if (assigned_grade == "Substandard / Below FAQ" or assigned_grade is None) else failed_reasons.get(assigned_grade, {}),
        unsupported_criteria=profile.unsupported_criteria,
        audit_notes=notes
    )
