# RiceAge segmentation dataset post-processing

## Why this step exists

RiceAge's current M0 pipeline creates a per-pixel instance-ID map at runtime. This utility turns that map into full-image YOLO polygon annotations and review artifacts. It does not train a model and does not claim the M0 masks are ground truth.

## Run a small pilot

From the repository root, after installing requirements:

    python scripts/build_segmentation_candidates.py \
      --input data/samples/sample_1489x2048.jpg \
      --output data/processed/segmentation_candidates_pilot \
      --max-images 1 \
      --verbose

Then inspect the generated overlay and metadata. If the image looks reasonable, process the image directory:

    python scripts/build_segmentation_candidates.py \
      --input data/samples \
      --output data/processed/segmentation_candidates

To preserve scanner calibration for a standardized sample, supply a raw reference only when it applies to the image being processed:

    python scripts/build_segmentation_candidates.py \
      --input data/samples/sample_1489x2048.jpg \
      --raw-reference data/raw/laserjet_1200dpi_sample.jpg \
      --output data/processed/segmentation_candidates_pilot

## Output structure

    segmentation_candidates/
    ├── images/          # full original images, same dimensions as labels
    ├── labels/          # YOLO segmentation polygons, class 0 = rice_grain
    ├── instance_masks/  # uint16 PNG or uint32 NPY ID maps
    ├── overlays/        # green = candidate, red = excluded/suspect
    ├── metadata/        # per-grain reasons, M0 measurements, mask-ID mapping
    ├── manifest.csv
    ├── errors.json
    └── README.md

The candidate label policy is intentionally conservative: only M0 instances with isolated_single or separated_from_cluster status and no boundary-truncation flag are written to YOLO label files. Excluded grains still appear in the metadata and instance-map review artifacts.

## Mandatory quality gate before training

All results are marked label_source=m0_pseudolabel, review_status=unreviewed, and training_ready=false. Do not create train/validation/test splits or train from these labels until a reviewer has inspected and corrected the overlays. M0 watershed can merge or split grains incorrectly; exported polygons are candidate labels only.

After review, split by original scan or physical sample batch, never by random crops from the same scan. Keep a record of accepted/rejected instances and corrections. Defect classes are not inferred here: this dataset uses one class (rice_grain) for instance segmentation only.
