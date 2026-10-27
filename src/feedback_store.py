"""SQLite persistence for human-reviewed RiceAge scan corrections.

This module intentionally stores only measurements produced by the current
classical-CV pipeline. It does not fabricate classifier probabilities or claim
to retrain a GBDT model that is not present in this repository.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_LABELS = {"WHOLE_RICE", "BROKEN_RICE"}
VISIBLE_CLASSES = SUPPORTED_LABELS | {"UNRESOLVED_CLUSTER", "BOUNDARY_TRUNCATED"}


def _json_default(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values into JSON-safe values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False, separators=(",", ":"))


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if value is None or isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        else:
            result[str(key)] = _json_default(value)
    return result


class FeedbackStore:
    """Transactional scan cache and idempotent correction store."""

    def __init__(self, db_path: str | os.PathLike[str] = "data/feedback.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scan_runs (
                    scan_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    grain_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS scan_grains (
                    scan_id TEXT NOT NULL,
                    grain_id TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    review_recommended INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (scan_id, grain_id),
                    FOREIGN KEY (scan_id) REFERENCES scan_runs(scan_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    grain_id TEXT NOT NULL,
                    original_pred TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    operator_id TEXT NOT NULL DEFAULT 'inspector',
                    timestamp INTEGER NOT NULL,
                    trained INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (scan_id, grain_id),
                    FOREIGN KEY (scan_id, grain_id)
                        REFERENCES scan_grains(scan_id, grain_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at INTEGER NOT NULL,
                    finished_at INTEGER,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_pending
                    ON feedback_audit(trained, timestamp);
                CREATE INDEX IF NOT EXISTS idx_scan_created
                    ON scan_runs(created_at DESC);
                """
            )

    def cache_scan(
        self,
        scan_id: str,
        image_path: str,
        summary: dict[str, Any],
        grains: Iterable[dict[str, Any]],
    ) -> None:
        """Cache canonical server-side features from a completed pipeline run."""
        if not scan_id or not image_path:
            raise ValueError("scan_id and image_path are required")
        safe_grains: list[dict[str, Any]] = []
        for idx, source in enumerate(grains, start=1):
            row = _safe_record(dict(source))
            grain_id = str(row.get("grain_id", row.get("id", f"grain_{idx:04d}")))
            bbox = [
                int(row.get("bbox_x", 0) or 0),
                int(row.get("bbox_y", 0) or 0),
                int(row.get("bbox_w", 0) or 0),
                int(row.get("bbox_h", 0) or 0),
            ]
            label = str(row.get("quality_class", "UNRESOLVED_CLUSTER"))
            if label not in VISIBLE_CLASSES:
                label = "UNRESOLVED_CLUSTER"
            # The baseline has no probabilistic defect heads. Mark review items
            # using existing segmentation/classification metadata, not fake scores.
            review_recommended = (
                label in {"UNRESOLVED_CLUSTER", "BOUNDARY_TRUNCATED"}
                or row.get("separation_status") == "under_segmented_cluster"
            )
            safe_grains.append(
                {
                    "grain_id": grain_id,
                    "bbox": bbox,
                    "features": row,
                    "predicted_label": label,
                    "review_recommended": bool(review_recommended),
                }
            )

        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_runs(scan_id, image_path, created_at, summary_json, grain_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scan_id) DO UPDATE SET
                    image_path=excluded.image_path,
                    created_at=excluded.created_at,
                    summary_json=excluded.summary_json,
                    grain_count=excluded.grain_count
                """,
                (scan_id, str(Path(image_path).resolve()), now, _json_dumps(summary), len(safe_grains)),
            )
            # A scan ID is immutable after creation in normal use. Replacing its
            # grains is useful for deterministic retries; cascading corrections
            # is avoided, so delete child audit rows first only on a true re-run.
            conn.execute("DELETE FROM feedback_audit WHERE scan_id = ?", (scan_id,))
            conn.execute("DELETE FROM scan_grains WHERE scan_id = ?", (scan_id,))
            conn.executemany(
                """
                INSERT INTO scan_grains(
                    scan_id, grain_id, bbox_json, features_json,
                    predicted_label, review_recommended
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_id,
                        g["grain_id"],
                        _json_dumps(g["bbox"]),
                        _json_dumps(g["features"]),
                        g["predicted_label"],
                        int(g["review_recommended"]),
                    )
                    for g in safe_grains
                ],
            )

    def list_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scan_id, image_path, created_at, summary_json, grain_count
                FROM scan_runs ORDER BY created_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [
            {
                "scan_id": row["scan_id"],
                "image_url": f"/api/scans/{row['scan_id']}/image",
                "created_at": row["created_at"],
                "grain_count": row["grain_count"],
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scan_runs WHERE scan_id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        return {
            "scan_id": row["scan_id"],
            "image_path": row["image_path"],
            "created_at": row["created_at"],
            "grain_count": row["grain_count"],
            "summary": json.loads(row["summary_json"]),
        }

    def get_scan_grains(self, scan_id: str) -> list[dict[str, Any]] | None:
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM scan_runs WHERE scan_id = ?", (scan_id,)).fetchone()
            if exists is None:
                return None
            rows = conn.execute(
                "SELECT * FROM scan_grains WHERE scan_id = ? ORDER BY grain_id", (scan_id,)
            ).fetchall()
        return [
            {
                "scan_id": row["scan_id"],
                "grain_id": row["grain_id"],
                "bbox": json.loads(row["bbox_json"]),
                "features": json.loads(row["features_json"]),
                "predicted_label": row["predicted_label"],
                "review_recommended": bool(row["review_recommended"]),
            }
            for row in rows
        ]

    def get_grain(self, scan_id: str, grain_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scan_grains WHERE scan_id = ? AND grain_id = ?",
                (scan_id, grain_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "scan_id": row["scan_id"],
            "grain_id": row["grain_id"],
            "bbox": json.loads(row["bbox_json"]),
            "features": json.loads(row["features_json"]),
            "predicted_label": row["predicted_label"],
            "review_recommended": bool(row["review_recommended"]),
        }

    def add_correction(
        self,
        scan_id: str,
        grain_id: str,
        corrected_label: str,
        operator_id: str = "inspector",
    ) -> dict[str, Any]:
        """Record a correction using the server-cached prediction/features only."""
        if corrected_label not in SUPPORTED_LABELS:
            raise ValueError(f"Unsupported correction label: {corrected_label}")
        if not operator_id or len(operator_id) > 128:
            raise ValueError("operator_id must contain 1-128 characters")

        now = int(time.time())
        with self._connect() as conn:
            grain = conn.execute(
                "SELECT * FROM scan_grains WHERE scan_id = ? AND grain_id = ?",
                (scan_id, grain_id),
            ).fetchone()
            if grain is None:
                raise KeyError("Unknown scan/grain. Submit a correction for a cached scan grain.")
            original = str(grain["predicted_label"])
            if original == corrected_label:
                raise ValueError("Correction matches the current prediction; no update is needed.")

            if original == "BROKEN_RICE" and corrected_label == "WHOLE_RICE":
                error_type = "FALSE_POSITIVE"
            elif original == "WHOLE_RICE" and corrected_label == "BROKEN_RICE":
                error_type = "FALSE_NEGATIVE"
            else:
                error_type = "MANUAL_REVIEW"

            # Only take features from the SQLite scan cache. The request body
            # cannot override the feature vector used for future model training.
            features_json = grain["features_json"]
            conn.execute(
                """
                INSERT INTO feedback_audit(
                    scan_id, grain_id, original_pred, corrected_label, error_type,
                    features_json, operator_id, timestamp, trained
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(scan_id, grain_id) DO UPDATE SET
                    original_pred=excluded.original_pred,
                    corrected_label=excluded.corrected_label,
                    error_type=excluded.error_type,
                    features_json=excluded.features_json,
                    operator_id=excluded.operator_id,
                    timestamp=excluded.timestamp,
                    trained=0
                """,
                (
                    scan_id, grain_id, original, corrected_label, error_type,
                    features_json, operator_id, now,
                ),
            )
            result = conn.execute(
                """
                SELECT id, scan_id, grain_id, original_pred, corrected_label, error_type,
                       operator_id, timestamp, trained
                FROM feedback_audit WHERE scan_id = ? AND grain_id = ?
                """,
                (scan_id, grain_id),
            ).fetchone()
        return dict(result)

    def correction_status(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN trained = 0 THEN 1 ELSE 0 END), 0) AS pending
                FROM feedback_audit
                """
            ).fetchone()
        return {"total_corrections": int(row["total"]), "pending_corrections": int(row["pending"])}

    def record_training_run(self, status: str, message: str, metrics: dict[str, Any] | None = None) -> int:
        now = int(time.time())
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_runs(started_at, finished_at, status, message, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, now, status, message, _json_dumps(metrics or {})),
            )
            return int(cur.lastrowid)

    def latest_training_run(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "message": row["message"],
            "metrics": json.loads(row["metrics_json"]),
        }
