"""SQLite persistence for server-verified grain audit feedback.

This module is intentionally independent of the current M0 CV pipeline. Call
cache_scan_grains() from an inference adapter once stable scan IDs and feature
records are available; never accept features from the browser.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


class FeedbackStore:
    def __init__(self, db_path: str | Path = "outputs/feedback.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scan_grains (
                    scan_id TEXT NOT NULL,
                    grain_id TEXT NOT NULL,
                    original_pred TEXT NOT NULL,
                    bbox_json TEXT NOT NULL DEFAULT 'null',
                    probabilities_json TEXT NOT NULL DEFAULT '{}',
                    measurements_json TEXT NOT NULL DEFAULT '{}',
                    features_json TEXT NOT NULL DEFAULT '{}',
                    image_path TEXT,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (scan_id, grain_id)
                );

                CREATE TABLE IF NOT EXISTS feedback_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    grain_id TEXT NOT NULL,
                    original_pred TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    affected_head TEXT,
                    error_type TEXT NOT NULL CHECK (
                        error_type IN ('FALSE_POSITIVE', 'FALSE_NEGATIVE', 'DEFECT_TO_DEFECT', 'LABEL_CORRECTION')
                    ),
                    threshold_before REAL,
                    threshold_after REAL,
                    features_json TEXT NOT NULL,
                    operator_id TEXT NOT NULL DEFAULT 'inspector',
                    timestamp INTEGER NOT NULL,
                    trained INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (scan_id, grain_id),
                    FOREIGN KEY (scan_id, grain_id)
                        REFERENCES scan_grains(scan_id, grain_id)
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at INTEGER NOT NULL,
                    finished_at INTEGER,
                    status TEXT NOT NULL,
                    pending_count INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_pending
                    ON feedback_audit(trained, timestamp);
                CREATE INDEX IF NOT EXISTS idx_scans_created
                    ON scan_grains(created_at);
            """)

    @staticmethod
    def _sanitize_json(value: Any) -> Any:
        """Convert NumPy/Pandas scalars and non-finite floats to JSON-safe values."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(key): FeedbackStore._sanitize_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FeedbackStore._sanitize_json(item) for item in value]
        # NumPy arrays expose tolist(); NumPy scalar values expose item().
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return FeedbackStore._sanitize_json(value.tolist())
            except (TypeError, ValueError):
                pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return FeedbackStore._sanitize_json(value.item())
            except (TypeError, ValueError):
                pass
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported value for feedback JSON: {type(value).__name__}")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            FeedbackStore._sanitize_json(value),
            separators=(",", ":"),
            allow_nan=False,
        )

    def cache_scan_grains(
        self,
        scan_id: str,
        grains: Iterable[dict[str, Any]],
        image_path: str | None = None,
    ) -> None:
        """Cache inference results; feature vectors originate on the server."""
        if not scan_id or not scan_id.strip():
            raise ValueError("scan_id must be a non-empty string")
        now = int(time.time())
        rows = []
        for grain in grains:
            grain_id = str(grain.get("grain_id", grain.get("id", ""))).strip()
            pred = str(grain.get("predicted_label", grain.get("quality_class", ""))).strip()
            if not grain_id or not pred:
                raise ValueError("each grain requires an id and predicted_label")
            rows.append((
                scan_id, grain_id, pred,
                self._json(grain.get("bbox")),
                self._json(grain.get("probabilities")),
                self._json(grain.get("measurements", {
                    k: v for k, v in grain.items()
                    if k not in {"features", "bbox", "probabilities"}
                })),
                self._json(grain.get("features", {
                    k: v for k, v in grain.items()
                    if k not in {"features", "bbox", "probabilities"}
                })),
                image_path, now,
            ))
        with self._connect() as conn:
            conn.executemany("""
                INSERT INTO scan_grains
                    (scan_id, grain_id, original_pred, bbox_json, probabilities_json,
                     measurements_json, features_json, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scan_id, grain_id) DO UPDATE SET
                    original_pred=excluded.original_pred,
                    bbox_json=excluded.bbox_json,
                    probabilities_json=excluded.probabilities_json,
                    measurements_json=excluded.measurements_json,
                    features_json=excluded.features_json,
                    image_path=excluded.image_path,
                    created_at=excluded.created_at
            """, rows)

    def list_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT scan_id, MAX(created_at) AS created_at, COUNT(*) AS grain_count,
                       MAX(image_path) AS image_path
                FROM scan_grains GROUP BY scan_id
                ORDER BY created_at DESC LIMIT ?
            """, (max(1, min(int(limit), 500)),)).fetchall()
        return [dict(row) for row in rows]

    def get_scan_image_path(self, scan_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM scan_grains WHERE scan_id = ? LIMIT 1",
                (scan_id,),
            ).fetchone()
        return str(row["image_path"]) if row and row["image_path"] else None

    def get_total_corrections(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM feedback_audit").fetchone()[0])

    def get_scan_grains(self, scan_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM scan_grains WHERE scan_id = ? ORDER BY grain_id
            """, (scan_id,)).fetchall()
            corrections = conn.execute("""
                SELECT grain_id, corrected_label FROM feedback_audit WHERE scan_id = ?
            """, (scan_id,)).fetchall()
        corrected = {row["grain_id"]: row["corrected_label"] for row in corrections}
        result = []
        for row in rows:
            item = dict(row)
            for column, key in (
                ("bbox_json", "bbox"), ("probabilities_json", "probabilities"),
                ("measurements_json", "measurements"), ("features_json", "features")
            ):
                item[key] = json.loads(item.pop(column))
            item["corrected_label"] = corrected.get(item["grain_id"])
            result.append(item)
        return result

    def record_correction(
        self,
        scan_id: str,
        grain_id: str,
        corrected_label: str,
        *,
        error_type: str,
        affected_head: str | None = None,
        threshold_before: float | None = None,
        threshold_after: float | None = None,
        operator_id: str = "inspector",
    ) -> dict[str, Any]:
        """Record a correction using features fetched from scan_grains, never client data."""
        allowed = {"FALSE_POSITIVE", "FALSE_NEGATIVE", "DEFECT_TO_DEFECT", "LABEL_CORRECTION"}
        if error_type not in allowed:
            raise ValueError(f"error_type must be one of {sorted(allowed)}")
        if not corrected_label or not corrected_label.strip():
            raise ValueError("corrected_label must be non-empty")
        with self._connect() as conn:
            grain = conn.execute("""
                SELECT original_pred, features_json FROM scan_grains
                WHERE scan_id = ? AND grain_id = ?
            """, (scan_id, grain_id)).fetchone()
            if grain is None:
                raise LookupError("Unknown scan_id/grain_id; grain must exist in server cache")
            now = int(time.time())
            conn.execute("""
                INSERT INTO feedback_audit
                    (scan_id, grain_id, original_pred, corrected_label, affected_head,
                     error_type, threshold_before, threshold_after, features_json,
                     operator_id, timestamp, trained)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(scan_id, grain_id) DO UPDATE SET
                    corrected_label=excluded.corrected_label,
                    affected_head=excluded.affected_head,
                    error_type=excluded.error_type,
                    threshold_before=excluded.threshold_before,
                    threshold_after=excluded.threshold_after,
                    features_json=excluded.features_json,
                    operator_id=excluded.operator_id,
                    timestamp=excluded.timestamp,
                    trained=0
            """, (
                scan_id, grain_id, grain["original_pred"], corrected_label.strip(),
                affected_head, error_type, threshold_before, threshold_after,
                grain["features_json"], operator_id or "inspector", now,
            ))
            row = conn.execute("""
                SELECT * FROM feedback_audit WHERE scan_id = ? AND grain_id = ?
            """, (scan_id, grain_id)).fetchone()
        return dict(row)

    def get_pending_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM feedback_audit WHERE trained = 0"
            ).fetchone()[0])

    def get_pending_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM feedback_audit WHERE trained = 0 ORDER BY timestamp, id"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(0, int(limit)),)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        records = [dict(row) for row in rows]
        for record in records:
            record["features"] = json.loads(record.pop("features_json"))
        return records

    def mark_as_trained(self, record_ids: Iterable[int]) -> int:
        ids = list(dict.fromkeys(int(value) for value in record_ids))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE feedback_audit SET trained = 1 WHERE trained = 0 AND id IN ({placeholders})",
                ids,
            )
            return int(cursor.rowcount)

    def record_training_run(
        self, status: str, pending_count: int = 0,
        metrics: dict[str, Any] | None = None, message: str | None = None,
        run_id: int | None = None,
    ) -> int:
        now = int(time.time())
        with self._connect() as conn:
            if run_id is None:
                cursor = conn.execute("""
                    INSERT INTO training_runs(started_at, status, pending_count, metrics_json, message)
                    VALUES (?, ?, ?, ?, ?)
                """, (now, status, pending_count, self._json(metrics), message))
                return int(cursor.lastrowid)
            conn.execute("""
                UPDATE training_runs SET finished_at=?, status=?, pending_count=?,
                    metrics_json=?, message=? WHERE id=?
            """, (now, status, pending_count, self._json(metrics), message, run_id))
            return run_id

    def latest_training_run(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item
