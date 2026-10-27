"""Bounded, directional online threshold calibration.

This is a calibration primitive, not a reinforcement-learning policy. It does not
claim that a threshold nudge improves model accuracy; validation is still required.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

MIN_THRESHOLD = 0.10
MAX_THRESHOLD = 0.90
DEFAULT_STEP = 0.015
_VALID_ERRORS = {"FALSE_POSITIVE", "FALSE_NEGATIVE"}
_FILE_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.RLock:
    normalized = str(Path(path).resolve())
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(normalized, threading.RLock())


def _get_threshold(data: dict[str, Any], head: str) -> float:
    if isinstance(data.get("thresholds"), dict) and head in data["thresholds"]:
        return float(data["thresholds"][head])
    heads = data.get("heads")
    if isinstance(heads, dict) and isinstance(heads.get(head), dict):
        if "threshold" in heads[head]:
            return float(heads[head]["threshold"])
    return 0.5


def _set_threshold(data: dict[str, Any], head: str, value: float) -> None:
    if isinstance(data.get("thresholds"), dict):
        data["thresholds"][head] = value
        return
    heads = data.setdefault("heads", {})
    if not isinstance(heads, dict):
        raise ValueError("weights JSON 'heads' must be an object")
    config = heads.setdefault(head, {})
    if not isinstance(config, dict):
        raise ValueError(f"weights JSON heads[{head!r}] must be an object")
    config["threshold"] = value


class ThresholdBandit:
    def __init__(
        self,
        weights_path: str | Path = "quality_classifier_weights.json",
        *,
        step: float = DEFAULT_STEP,
        min_threshold: float = MIN_THRESHOLD,
        max_threshold: float = MAX_THRESHOLD,
    ) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        if not 0 <= min_threshold < max_threshold <= 1:
            raise ValueError("threshold bounds must satisfy 0 <= min < max <= 1")
        self.weights_path = Path(weights_path)
        self.step = float(step)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)

    def _read(self) -> dict[str, Any]:
        if not self.weights_path.exists():
            return {"thresholds": {}}
        try:
            data = json.loads(self.weights_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read threshold weights: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("weights JSON root must be an object")
        return data

    def get_threshold(self, head: str) -> float:
        if not head or not head.strip():
            raise ValueError("head must be non-empty")
        with _lock_for(str(self.weights_path)):
            return _get_threshold(self._read(), head)

    def record_feedback(self, head: str, error_type: str) -> dict[str, float | str]:
        """Apply one explicit error signal and atomically persist the new threshold.

        FALSE_POSITIVE raises the defect threshold. FALSE_NEGATIVE lowers it.
        """
        if not head or not head.strip():
            raise ValueError("head must be non-empty")
        if error_type not in _VALID_ERRORS:
            raise ValueError(f"error_type must be one of {sorted(_VALID_ERRORS)}")

        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(str(self.weights_path)):
            data = self._read()
            before = _get_threshold(data, head)
            if not self.min_threshold <= before <= self.max_threshold:
                raise ValueError(
                    f"Existing threshold {before} for {head!r} is outside "
                    f"[{self.min_threshold}, {self.max_threshold}]"
                )
            direction = 1 if error_type == "FALSE_POSITIVE" else -1
            after = min(self.max_threshold, max(self.min_threshold, before + direction * self.step))
            _set_threshold(data, head, round(after, 6))

            temp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.weights_path.parent,
                    prefix=f".{self.weights_path.name}.", suffix=".tmp", delete=False
                ) as handle:
                    temp_path = handle.name
                    json.dump(data, handle, indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.weights_path)
            except Exception:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            return {
                "head": head,
                "error_type": error_type,
                "threshold_before": before,
                "threshold_after": round(after, 6),
            }
