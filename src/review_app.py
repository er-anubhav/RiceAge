"""Optional local Flask audit application for the existing M0 CV pipeline.

Run from the repository root with:
    python -m src.review_app

This audit server intentionally supports only labels produced by the current
rule-based M0 classifier. It does not expose nonexistent defect probabilities
or pretend that model retraining is available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, send_file

from src.feedback_store import FeedbackStore

M0_LABELS = {
    "WHOLE_RICE": "Whole rice",
    "BROKEN_RICE": "Broken rice",
    "BOUNDARY_TRUNCATED": "Boundary truncated",
    "UNRESOLVED_CLUSTER": "Unresolved cluster",
}


def create_app(db_path: str | Path | None = None) -> Flask:
    repo_root = Path(__file__).resolve().parent.parent
    template_dir = repo_root / "templates"
    configured_db = db_path or os.environ.get(
        "RICEAGE_FEEDBACK_DB", str(repo_root / "outputs" / "feedback.db")
    )
    store = FeedbackStore(configured_db)

    app = Flask(__name__, template_folder=str(template_dir))
    app.config["FEEDBACK_STORE"] = store
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def home():
        return redirect("/review")

    @app.get("/review")
    def review():
        return render_template("review.html", labels=M0_LABELS)

    @app.get("/api/scans")
    def scans():
        rows = store.list_scans(limit=100)
        return jsonify({
            "scans": [
                {
                    "scan_id": row["scan_id"],
                    "created_at": row["created_at"],
                    "grain_count": row["grain_count"],
                    "image_url": f"/api/scans/{row['scan_id']}/image",
                }
                for row in rows
            ],
            "pending_corrections": store.get_pending_count(),
            "total_corrections": store.get_total_corrections(),
            "label_scope": "M0 rule-based classes",
        })

    @app.get("/api/scan_grains/<scan_id>")
    def scan_grains(scan_id: str):
        rows = store.get_scan_grains(scan_id)
        if not rows:
            return jsonify({"error": "Scan not found or contains no grains."}), 404

        # Features remain server-side. Only UI-safe fields are returned.
        result = []
        for row in rows:
            measurements = row.get("measurements") or {}
            result.append({
                "grain_id": row["grain_id"],
                "original_pred": row["original_pred"],
                "corrected_label": row.get("corrected_label"),
                "display_label": row.get("corrected_label") or row["original_pred"],
                "bbox": row.get("bbox"),
                "probabilities": row.get("probabilities") or {},
                "measurements": measurements,
                "review_flags": measurements.get("review_flags", []),
            })
        return jsonify({
            "scan_id": scan_id,
            "image_url": f"/api/scans/{scan_id}/image",
            "grains": result,
            "model_type": "rule_based_m0",
            "confidence_available": False,
        })

    @app.get("/api/scans/<scan_id>/image")
    def scan_image(scan_id: str):
        image_path = store.get_scan_image_path(scan_id)
        if not image_path:
            return jsonify({"error": "No image is associated with this scan."}), 404
        path = Path(image_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
        }:
            return jsonify({"error": "The source scan image is unavailable."}), 404
        return send_file(path, conditional=True)

    @app.get("/api/feedback/status")
    def feedback_status():
        return jsonify({
            "pending_corrections": store.get_pending_count(),
            "total_corrections": store.get_total_corrections(),
            "threshold_adaptation_available": False,
            "retraining_available": False,
            "last_training_run": store.latest_training_run(),
            "note": (
                "The current M0 classifier is rule-based. Threshold adaptation and "
                "GBDT retraining are unavailable until a probabilistic defect model exists."
            ),
        })

    @app.post("/api/feedback/correct")
    def correct_grain():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        scan_id = str(payload.get("scan_id", "")).strip()
        grain_id = str(payload.get("grain_id", "")).strip()
        corrected_label = str(payload.get("corrected_label", "")).strip()

        if not scan_id or not grain_id or not corrected_label:
            return jsonify({
                "error": "scan_id, grain_id, and corrected_label are required."
            }), 400
        if corrected_label not in M0_LABELS:
            return jsonify({
                "error": "Unsupported label for the current M0 classifier.",
                "allowed_labels": list(M0_LABELS),
            }), 400

        grain = next(
            (item for item in store.get_scan_grains(scan_id)
             if item["grain_id"] == grain_id),
            None,
        )
        if grain is None:
            return jsonify({"error": "Unknown scan/grain pair."}), 404

        effective_label = grain.get("corrected_label") or grain["original_pred"]
        if effective_label == corrected_label:
            return jsonify({
                "message": "No change; this grain already has that label.",
                "pending_corrections": store.get_pending_count(),
                "grain_id": grain_id,
                "display_label": effective_label,
            }), 200

        # M0 has no probabilistic defect heads, so this is an audited label
        # correction only; a FALSE_POSITIVE/FALSE_NEGATIVE threshold update
        # would falsely imply model confidence that does not exist.
        try:
            record = store.record_correction(
                scan_id,
                grain_id,
                corrected_label,
                error_type="LABEL_CORRECTION",
                affected_head=None,
                operator_id="inspector",
            )
        except LookupError:
            return jsonify({"error": "Unknown scan/grain pair."}), 404
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({
            "message": "Correction saved to the audit log.",
            "grain_id": grain_id,
            "corrected_label": record["corrected_label"],
            "pending_corrections": store.get_pending_count(),
            "threshold_update": None,
            "retraining_available": False,
        }), 200

    @app.post("/api/feedback/retrain")
    def retrain():
        return jsonify({
            "error": "Retraining is not available in the current M0 repository.",
            "detail": (
                "The production GBDT trainer and validated defect-labelled dataset "
                "are not present. No model or weights were changed."
            ),
        }), 501

    return app


app = create_app()


if __name__ == "__main__":
    # Localhost only by default: this dashboard has no authentication layer.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
