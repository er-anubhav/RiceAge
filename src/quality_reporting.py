# Comprehensive quality reporting module for RiceAge Milestone M3.
# Exports machine-readable JSON and CSV alongside human-readable HTML reports.

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import cv2

from src.quality_grading import SampleQualityMetrics, GradingDecision, GrainClassification


def generate_html_report(
    sample_id: str,
    metrics: SampleQualityMetrics,
    grading: GradingDecision,
    grains: List[GrainClassification],
    analysis_date: Optional[str] = None,
    image_filename: Optional[str] = None,
    annotated_rel_path: Optional[str] = None
) -> str:
    """Generate self-contained human-readable HTML quality report."""
    if analysis_date is None:
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Grade styling
    grade_color = "#10b981" if grading.is_compliant else "#ef4444"
    if grading.status == "INCOMPLETE":
        grade_color = "#f59e0b"
    elif grading.status == "INDETERMINATE":
        grade_color = "#6b7280"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RiceAge Quality Inspection Report — {sample_id}</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #3b82f6;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text-main);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1080px;
    margin: 0 auto;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ margin: 0; font-size: 24px; font-weight: 700; }}
  .meta {{ color: var(--text-muted); font-size: 14px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }}
  .badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 16px;
    color: #fff;
    background: {grade_color};
  }}
  .metric-val {{
    font-size: 28px;
    font-weight: 700;
    margin: 4px 0;
  }}
  .metric-sub {{
    font-size: 13px;
    color: var(--text-muted);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
  }}
  th, td {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--text-muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .alert {{
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 24px;
    font-size: 14px;
  }}
  .alert-title {{ font-weight: 700; color: #f59e0b; margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="container">

<header>
  <div>
    <h1>RiceAge Optical Screening Estimate (Provisional)</h1>
    <div class="meta">Sample ID: <strong>{sample_id}</strong> | Inspection Date: {analysis_date} | <strong>Optical Screening Estimate</strong></div>
  </div>
  <div>
    <span class="badge">{grading.assigned_grade or "INDETERMINATE"}</span>
  </div>
</header>

<div class="grid">
  <div class="card">
    <div class="meta">Total Rice Kernels</div>
    <div class="metric-val">{metrics.total_rice_grains}</div>
    <div class="metric-sub">Whole: {metrics.whole_count} | Broken: {metrics.broken_count}</div>
  </div>
  <div class="card">
    <div class="meta">Foreign Matter Objects</div>
    <div class="metric-val">{metrics.total_foreign_matter}</div>
    <div class="metric-sub">{metrics.foreign_matter_percentage}% of total detected objects</div>
  </div>
  <div class="card">
    <div class="meta">Head Rice Ratio (Whole %)</div>
    <div class="metric-val">{metrics.whole_percentage}%</div>
    <div class="metric-sub">Broken: {metrics.broken_percentage}%</div>
  </div>
  <div class="card">
    <div class="meta">Overall Defect Rate</div>
    <div class="metric-val">{metrics.chalky_count + metrics.damaged_count + metrics.discoloured_count}</div>
    <div class="metric-sub">{metrics.multi_defect_rice_count} grains with multi-defects</div>
  </div>
</div>

<div class="card" style="margin-bottom: 24px;">
  <h2>Quality Factor Distribution & Standards Compliance</h2>
  <div class="meta">Evaluated under standard: <strong>{grading.standard_name}</strong></div>
  <table>
    <thead>
      <tr>
        <th>Quality Criterion</th>
        <th>Measured Count</th>
        <th>Measured %</th>
        <th>Mathematical Denominator</th>
        <th>Grade Limit</th>
        <th>Compliance Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Whole Rice (Head Rice)</strong></td>
        <td>{metrics.whole_count}</td>
        <td>{metrics.whole_percentage}%</td>
        <td>Total Rice Grains ({metrics.total_rice_grains})</td>
        <td>Baseline Standard</td>
        <td>PASS</td>
      </tr>
      <tr>
        <td><strong>Broken Rice</strong></td>
        <td>{metrics.broken_count}</td>
        <td>{metrics.broken_percentage}%</td>
        <td>Total Rice Grains ({metrics.total_rice_grains})</td>
        <td>Max allowable for tier</td>
        <td>{"PASS" if grading.passed_criteria.get("broken_rice", True) else "EXCEEDED"}</td>
      </tr>
      <tr>
        <td><strong>Chalky Kernels</strong></td>
        <td>{metrics.chalky_count}</td>
        <td>{metrics.chalky_percentage}%</td>
        <td>Total Rice Grains ({metrics.total_rice_grains})</td>
        <td>Max allowable for tier</td>
        <td>{"PASS" if grading.passed_criteria.get("chalky_rice", True) else "EXCEEDED"}</td>
      </tr>
      <tr>
        <td><strong>Damaged Kernels</strong></td>
        <td>{metrics.damaged_count}</td>
        <td>{metrics.damaged_percentage}%</td>
        <td>Total Rice Grains ({metrics.total_rice_grains})</td>
        <td>Max allowable for tier</td>
        <td>{"PASS" if grading.passed_criteria.get("damaged_rice", True) else "EXCEEDED"}</td>
      </tr>
      <tr>
        <td><strong>Discoloured Kernels</strong></td>
        <td>{metrics.discoloured_count}</td>
        <td>{metrics.discoloured_percentage}%</td>
        <td>Total Rice Grains ({metrics.total_rice_grains})</td>
        <td>Max allowable for tier</td>
        <td>{"PASS" if grading.passed_criteria.get("discoloured_rice", True) else "EXCEEDED"}</td>
      </tr>
      <tr>
        <td><strong>Foreign Matter</strong></td>
        <td>{metrics.total_foreign_matter}</td>
        <td>{metrics.foreign_matter_percentage}%</td>
        <td>Total Objects ({metrics.total_detected_objects})</td>
        <td>Max allowable for tier</td>
        <td>{"PASS" if grading.passed_criteria.get("foreign_matter", True) else "EXCEEDED"}</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="alert">
  <div class="alert-title">Regulatory Certification & Laboratory Limitations Notice</div>
  <div>This report provides an <strong>optical inspection assessment</strong> based on flatbed scan imagery. The following criteria required by {grading.standard_name} cannot be evaluated optically and require dedicated physical laboratory assays:</div>
  <ul>
    {"".join(f"<li>{c}</li>" for c in grading.unsupported_criteria)}
  </ul>
  <div>This report does not replace certified laboratory grain certificates where legal trade compliance requires moisture or gravimetric weighing.</div>
</div>

</div>
</body>
</html>
"""
    return html


def export_sample_quality_report(
    sample_id: str,
    metrics: SampleQualityMetrics,
    grading: GradingDecision,
    grains: List[GrainClassification],
    output_dir: str = "outputs/reports",
    image_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Export sample report in JSON, CSV, and HTML formats.
    Returns dictionary with paths of generated report files.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Machine-Readable JSON Export
    json_path = os.path.join(output_dir, f"{sample_id}_report.json")
    json_payload = {
        "report_metadata": {
            "sample_id": sample_id,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "RiceAge M4.0-Integrated-Quality",
            "screening_type": "Optical Screening Estimate (Provisional)",
            "statutory_certification_claim": "NONE (Optical Count-Based Screening Only)",
            "source_image": image_path
        },
        "quality_metrics": metrics.to_dict(),
        "grading_decision": grading.to_dict(),
        "grain_classifications": [
            {
                "object_id": g.object_id,
                "is_rice": g.is_rice,
                "is_foreign_matter": g.is_foreign_matter,
                "morphology": g.morphology,
                "is_chalky": g.is_chalky,
                "is_damaged": g.is_damaged,
                "is_discoloured": g.is_discoloured,
                "confidences": g.confidences
            }
            for g in grains
        ]
    }
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    # 2. Granular CSV Export
    csv_path = os.path.join(output_dir, f"{sample_id}_grains.csv")
    grain_rows = []
    for g in grains:
        grain_rows.append({
            "sample_id": sample_id,
            "object_id": g.object_id,
            "is_rice": int(g.is_rice),
            "is_foreign_matter": int(g.is_foreign_matter),
            "morphology": g.morphology,
            "is_chalky": int(g.is_chalky),
            "is_damaged": int(g.is_damaged),
            "is_discoloured": int(g.is_discoloured)
        })
    pd.DataFrame(grain_rows).to_csv(csv_path, index=False)

    # 3. Human-Readable HTML Export
    html_path = os.path.join(output_dir, f"{sample_id}_report.html")
    html_content = generate_html_report(
        sample_id=sample_id,
        metrics=metrics,
        grading=grading,
        grains=grains,
        image_filename=os.path.basename(image_path) if image_path else None
    )
    with open(html_path, "w") as f:
        f.write(html_content)

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "html_path": html_path
    }
