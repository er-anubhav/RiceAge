# Human-in-the-Loop Active Learning Foundation

## What this branch adds

This branch introduces two reusable primitives for the proposed RiceAge audit loop:

- `src/feedback_store.py`: SQLite persistence for scan-grain snapshots, idempotent corrections, pending feedback, and training-run audit records. Corrections retrieve features from the server-side `scan_grains` table; the browser must not be treated as the source of feature vectors.
- `src/threshold_calibration.py`: bounded, atomic threshold persistence with explicit error semantics. `FALSE_POSITIVE` increases the affected defect threshold by 0.015; `FALSE_NEGATIVE` decreases it by 0.015. Values are clamped to [0.10, 0.90].
- `src/pipeline.py`: assigns each run a unique scan ID, produces grain-level review records from the actual M0 measurements, and attempts to persist them to SQLite. Persistence failure is logged and does not abort the CV pipeline. Existing return fields remain and `scan_id`, `grains_data`, and a feedback status are additive.
- `configs/default_config.py`: adds `persist_feedback` and optional `feedback_db_path` settings.
- `tests/test_feedback_loop_foundation.py` and `tests/test_pipeline_feedback_integration.py`: cover storage, threshold primitives, and M0 scan caching.

The SQLite database defaults to `outputs/feedback.db` (or the configured `output_dir`); configure `feedback_db_path` to override it or set `persist_feedback=False` to disable caching.

## Important integration boundary

The current repository is the M0 classical-CV pipeline. Its existing `main.py` invokes `src.pipeline.RiceQualityPipeline`, and `src.classification.py` currently produces rule-based `WHOLE_RICE`, `BROKEN_RICE`, and unresolved/boundary states. This repository does **not** currently contain the planned `final/app.py`, a Flask dashboard, the five-head GBDT inference engine, `quality_classifier_weights.json`, or `scripts/train_master_models.py`.

The M0 pipeline is now connected to SQLite scan caching, but the full `/review` UI/API, six-class defect inference, threshold-to-inference hot reload, GBDT retraining, validation gate, and candidate promotion are still not implemented. The current labels and morphology features are real M0 outputs; defect probabilities intentionally remain empty. Do not fabricate defect probabilities or train a GBDT on rule-based labels as though they were verified ground truth. Also note that M0 has no probability threshold to calibrate, so directional threshold updates should only be connected to a genuine probabilistic defect head.

## Integration contract

1. Each M0 run now receives a unique `scan_id`; the pipeline caches grain labels, bbox, center, measurements, and the available feature vector.
2. The current M0 pipeline returns its existing response fields and adds `scan_id`, `grains_data`, and feedback persistence status. Database errors are logged without aborting analysis.
3. Build a review API that accepts only scan ID, grain ID, and corrected class. Read labels/features from SQLite; derive and validate the correction direction server-side.
4. Connect `ThresholdBandit.record_feedback(head, error_type)` only to a real probabilistic model head. Do not change thresholds for the present rule-based M0 classifier.
5. Train candidate models in an isolated artifact path only after a real labeled defect training set and feature schema exist. Compare against a held-out scan-level benchmark; preserve the active model if evaluation fails.
6. Add the Flask review UI/API as a separate optional application first, keeping the current CLI pipeline behavior unchanged.

## Tests

Run:

```bash
python -m pytest tests/test_feedback_loop_foundation.py tests/test_pipeline_feedback_integration.py tests/test_review_app.py -v
python -m pytest -v

# Launch the optional local review server after running a pipeline scan
python -m src.review_app
```
