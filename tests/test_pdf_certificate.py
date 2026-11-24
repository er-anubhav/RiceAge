import os
import pytest
from pypdf import PdfReader
from PIL import Image as PILImage

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

    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 1

    text = reader.pages[0].extract_text()
    assert "ANDHRA PRADESH STATE CIVIL SUPPLIES CORPORATION LIMITED" in text
    assert "APSCSCL-2026-TEST" in text
    assert "Head Rice" in text
    assert "Small Broken Grains" in text
    assert "Mass Basis" in text
    assert "OVERALL DISPOSITION: GRADE 1" in text


def test_generate_apscscl_pdf_certificate_with_chemical_data(tmp_path):
    out_pdf = str(tmp_path / "test_certificate_with_chemical.pdf")

    # Create dummy images for platen and chemical thumbnail
    platen_img = str(tmp_path / "dummy_platen.jpg")
    chem_img = str(tmp_path / "dummy_chem.jpg")
    img_a = PILImage.new("RGB", (400, 200), color=(50, 100, 150))
    img_a.save(platen_img)
    img_b = PILImage.new("RGB", (300, 200), color=(200, 150, 50))
    img_b.save(chem_img)

    sample_data = {
        "sample_id": "Sample-0004",
        "commodity": "Raw Rice (Grade A)",
        "timestamp_str": "19/11/2026 18:08:31",
        "total_grains": 650,
        "total_sample_mass_g": 5.00,
        "avg_length_mm": 6.45,
        "avg_breadth_mm": 2.11,
        "length_breadth_ratio": 3.05,
        "commercial_grade": "Grade 1 (Conforming to APSCSCL Specifications)",
        "mass_basis": {
            "whole_pct": 84.0,
            "broken_pct": 10.5,
            "small_broken_pct": 0.4,
            "total_broken_pct": 10.9,
            "foreign_matter_pct": 0.0,
            "damaged_pct": 0.5,
            "discoloured_pct": 0.3,
            "chalky_pct": 1.1,
            "whole_mass_g": 4.20,
            "broken_mass_g": 0.52,
            "small_broken_mass_g": 0.02,
            "total_broken_mass_g": 0.54,
            "foreign_matter_mass_g": 0.0,
            "damaged_mass_g": 0.025,
            "discoloured_mass_g": 0.015,
            "chalky_mass_g": 0.055,
        },
        "count_basis": {
            "whole_pct": 82.0,
            "broken_pct": 12.0,
            "small_broken_pct": 0.8,
            "total_broken_pct": 12.8,
            "foreign_matter_pct": 0.0,
            "damaged_pct": 0.7,
            "discoloured_pct": 0.4,
            "chalky_pct": 1.3,
            "whole_count": 533,
            "broken_count": 78,
            "small_broken_count": 5,
            "total_broken_count": 83,
            "foreign_matter_count": 0,
            "damaged_count": 4,
            "discoloured_count": 3,
            "chalky_count": 8,
        },
        "rice_age": {
            "connected": True,
            "host": "rice.local:5000",
            "serial": "Sample-0004",
            "status": "Mixed Ages Detected",
            "primary_age": "6-8 Months (98%)",
            "dominant_age_bracket": "6-8 Months",
            "dominant_color": "Yellow",
            "ages": ["98% 6-8 Months", "2% 8-10 Months"],
            "colors": ["Yellow", "Yellow Orange"],
            "freshness_rating": "Sound Commercial Grain (Medium Storage Age)",
            "image_filename": "dummy_chem.jpg"
        }
    }

    pdf_path = generate_apscscl_pdf_certificate(
        sample_data,
        out_pdf,
        overlay_image_path=platen_img,
        chemical_image_path=chem_img
    )
    assert os.path.exists(pdf_path)

    reader = PdfReader(pdf_path)
    # Strict requirement: single page!
    assert len(reader.pages) == 1

    text = reader.pages[0].extract_text()
    norm_text = " ".join(text.split())
    assert "Sample-0004" in text
    assert "BIOCHEMICAL RICE AGE & MILLING FRESHNESS EVALUATION" in text
    assert "6-8 Months (98%)" in text
    assert "Sound Commercial Grain (Medium Storage Age)" in norm_text
    assert "Yellow" in text
    assert "COLOR REFERENCE GUIDE" in norm_text
    assert "Avocado Green" in norm_text
    assert "Above 1 Year" in norm_text
