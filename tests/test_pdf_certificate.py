import os
import pytest
from pypdf import PdfReader

from src.reporting_pdf import generate_apscscl_pdf_certificate


def test_generate_apscscl_pdf_certificate(tmp_path):
    out_pdf = str(tmp_path / "test_certificate.pdf")

    sample_data = {
        "sample_id": "APSCSCL-2026-TEST",
        "commodity": "Raw Rice (Grade A)",
        "timestamp_str": "04/11/2026 12:30:00",
        "total_grains": 420,
        "total_sample_mass_g": 5.12,
        "avg_length_mm": 6.55,
        "avg_breadth_mm": 2.10,
        "length_breadth_ratio": 3.12,
        "commercial_grade": "Grade 1 (Conforming to APSCSCL Specifications)",
        "mass_basis": {
            "whole_pct": 82.5,
            "broken_pct": 11.2,
            "small_broken_pct": 0.5,
            "total_broken_pct": 11.7,
            "foreign_matter_pct": 0.1,
            "damaged_pct": 0.6,
            "discoloured_pct": 0.4,
            "chalky_pct": 1.2,
            "whole_mass_g": 4.22,
            "broken_mass_g": 0.57,
            "small_broken_mass_g": 0.03,
            "total_broken_mass_g": 0.60,
            "foreign_matter_mass_g": 0.005,
            "damaged_mass_g": 0.03,
            "discoloured_mass_g": 0.02,
            "chalky_mass_g": 0.06,
        },
        "count_basis": {
            "whole_pct": 80.0,
            "broken_pct": 12.5,
            "small_broken_pct": 1.0,
            "total_broken_pct": 13.5,
            "foreign_matter_pct": 0.1,
            "damaged_pct": 0.8,
            "discoloured_pct": 0.5,
            "chalky_pct": 1.5,
            "whole_count": 336,
            "broken_count": 52,
            "small_broken_count": 4,
            "total_broken_count": 56,
            "foreign_matter_count": 1,
            "damaged_count": 3,
            "discoloured_count": 2,
            "chalky_count": 6,
        }
    }

    pdf_path = generate_apscscl_pdf_certificate(sample_data, out_pdf)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

    # Verify PDF structure using pypdf
    reader = PdfReader(pdf_path)
    # Must fit exactly on 1 single page as per specification
    assert len(reader.pages) == 1

    text = reader.pages[0].extract_text()
    assert "ANDHRA PRADESH STATE CIVIL SUPPLIES CORPORATION LIMITED" in text
    assert "APSCSCL-2026-TEST" in text
    assert "Head Rice" in text
    assert "Small Broken Grains" in text
    assert "Mass Basis" in text
    assert "OVERALL DISPOSITION: GRADE 1" in text
