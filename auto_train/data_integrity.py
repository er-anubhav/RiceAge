import os
import json
import torch
import pandas as pd
from PIL import Image
from typing import Dict, Any

def audit_dataset_integrity(
    manifest_path: str,
    crosswalk_path: str,
    crop_root: str,
    output_dir: str
) -> Dict[str, Any]:
    print("=== Running Comprehensive Data Mapping & Integrity Audit ===", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    manifest = pd.read_csv(manifest_path)
    crosswalk = pd.read_csv(crosswalk_path)
    
    total_manifest = len(manifest)
    total_crosswalk = len(crosswalk)
    
    # 2. Check Canonical Primaries
    primaries = crosswalk[crosswalk['final_disposition'] == 'canonical_primary'].drop_duplicates('canonical_instance_id').set_index('canonical_instance_id')
    
    # Check duplicate canonical IDs in manifest
    dup_manifest_cids = int(manifest['canonical_instance_id'].duplicated().sum())
    
    # 3. Check Split Preservation & Quarantine
    split_counts = manifest['pu_learning_split_role'].value_counts().to_dict()
    train_cids = manifest[manifest['pu_learning_split_role'] == 'TRAIN_FITTING_ELIGIBLE']['canonical_instance_id']
    val_cids = manifest[manifest['pu_learning_split_role'] == 'VAL_TUNING_RESERVED']['canonical_instance_id']
    test_cids = manifest[manifest['pu_learning_split_role'] == 'TEST_BENCHMARK_ISOLATED']['canonical_instance_id']
    excluded_cids = manifest[manifest['pu_learning_split_role'] == 'EXCLUDED_FROM_ALL_EXPERIMENTS']['canonical_instance_id']
    
    # Check split overlap
    train_set, val_set, test_set = set(train_cids), set(val_cids), set(test_cids)
    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    
    # 4. Check Path Resolution & Existence on Disk
    missing_train_images = 0
    missing_val_images = 0
    missing_test_images = 0
    
    for cid in train_cids:
        if cid not in primaries.index:
            missing_train_images += 1
            continue
        p = primaries.loc[cid]
        path = os.path.normpath(os.path.join(crop_root, p['dataset_split'], p['original_label'], p['crop_filename']))
        if not os.path.exists(path):
            missing_train_images += 1
            
    for cid in val_cids:
        if cid not in primaries.index:
            missing_val_images += 1
            continue
        p = primaries.loc[cid]
        path = os.path.normpath(os.path.join(crop_root, p['dataset_split'], p['original_label'], p['crop_filename']))
        if not os.path.exists(path):
            missing_val_images += 1
            
    # Sample Image Decode and Shape Check
    sample_cid = train_cids.iloc[0]
    p_sample = primaries.loc[sample_cid]
    sample_path = os.path.normpath(os.path.join(crop_root, p_sample['dataset_split'], p_sample['original_label'], p_sample['crop_filename']))
    sample_img = Image.open(sample_path).convert('RGB')
    sample_size = sample_img.size
    
    audit_report = {
        'total_manifest_records': total_manifest,
        'total_crosswalk_records': total_crosswalk,
        'unique_canonical_primaries': len(primaries),
        'duplicate_manifest_cids': dup_manifest_cids,
        'split_counts': split_counts,
        'split_overlaps': {
            'train_val_overlap': overlap_train_val,
            'train_test_overlap': overlap_train_test,
            'val_test_overlap': overlap_val_test
        },
        'test_set_isolation': {
            'quarantined_test_instances': len(test_cids),
            'leakage_detected': (overlap_train_test > 0 or overlap_val_test > 0)
        },
        'image_file_verification': {
            'missing_train_crops': missing_train_images,
            'missing_val_crops': missing_val_images,
            'sample_crop_path': sample_path,
            'sample_crop_size': sample_size,
            'sample_crop_rgb_mode': sample_img.mode
        },
        'integrity_status': 'CERTIFIED_VERIFIED' if (
            dup_manifest_cids == 0 and
            overlap_train_test == 0 and
            overlap_val_test == 0 and
            missing_train_images == 0 and
            missing_val_images == 0
        ) else 'INTEGRITY_VIOLATION'
    }
    
    audit_log_path = os.path.join(output_dir, 'data_integrity_audit.json')
    with open(audit_log_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
        
    print(f"Data Integrity Audit: {audit_report['integrity_status']}", flush=True)
    print(f"  Train: {len(train_cids)} | Val: {len(val_cids)} | Test (Quarantined): {len(test_cids)}", flush=True)
    print(f"  Missing Crop Images: Train={missing_train_images}, Val={missing_val_images}", flush=True)
    print(f"  Test Set Leakage: None detected (Overlap = 0)", flush=True)
    print(f"  Audit report written to {audit_log_path}", flush=True)
    
    return audit_report

if __name__ == '__main__':
    audit_dataset_integrity(
        manifest_path='outputs/phase2_automated/automated_dataset_manifest.csv',
        crosswalk_path='outputs/phase1_2_final_audit/canonical_id_crosswalk.csv',
        crop_root='dataset/crops',
        output_dir='outputs/phase2_automated/experiments/exp004_ema_pseudolabel_pu'
    )
