# Human-in-the-Loop Active Learning Foundation

## What this branch adds

This branch introduces two reusable primitives for the proposed RiceAge audit loop:

- `src/feedback_store.py`: SQLite persistence for scan-grain snapshots, idempotent corrections, pending feedback, and training-run audit records. Corrections retrieve features from the server-side `scan_grains` table; the browser must not be treated as the source of feature vectors.
- `src/threshold_calibration.py`: bounded, atomic threshold persistence with explicit error semantics. `FALSE_POSITIVE` increases the affected defect threshold by 0.015; `FALSE_NEGATIVE` decreases it by 0.015. Values are clamped to [0.10, 0.90].
- `tests/test_feedback_loop_foundation.py`: tests for server-side feature lookup, idempotency, directional updates, bounds, and both supported threshold JSON layouts.

The SQLite database defaults to `outputs/feedback.db`; pass a different path in tests or deployments as appropriate.

## Important integration boundary

The current repository is the M0 classical-CV pipeline. Its existing `main.py` invokes `src.pipeline.RiceQualityPipeline`, and `src.classification.py` currently produces rule-based `WHOLE_RICE`, `BROKEN_RICE`, and unresolved/boundary states. This repository does **not** currently contain the planned `final/app.py`, a Flask dashboard, the five-head GBDT inference engine, `quality_classifier_weights.json`, or `scripts/train_master_models.py`.

For that reason, this branch deliberately does not claim to deliver the full `/review` UI, six-class defect inference, model retraining, or candidate promotion. Those pieces must be integrated only after the production inference/training implementation and its real feature schema are present. In particular, do not fabricate defect probabilities or train a GBDT on the current rule-based labels as though they were verified ground truth.

## Integration contract

1. Generate a stable unique `scan_id` for each inference run.
2. From the server-side inference path, call `FeedbackStore.cache_scan_grains(scan_id, grains, image_path=...)`. Each grain needs `id` (or `grain_id`) and `predicted_label`; include its actual bbox, probabilities, measurements, and feature vector when available.
3. Validate each requested correction against the scan and grain IDs in the database. Determine the error type on the server; never accept a browser-provided feature vector or error type as authoritative.
4. Use `ThresholdBandit.record_feedback(head, error_type)` only after the correction has been validated and the affected head/error direction is known.
5. Train candidate models in an isolated artifact path. Compare against a held-out scan-level benchmark before promotion; preserve the active model if evaluation fails. Mark feedback as trained only after successful promotion.
6. Add the Flask review UI and API as a separate optional application or integrate them into the actual web server once that server exists. Keep the current CLI pipeline behavior unchanged.

## Tests

Run:

```bash
python -m pytest tests/test_feedback_loop_foundation.py -v
python -m pytest -v
```
