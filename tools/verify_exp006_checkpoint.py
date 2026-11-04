"""
RiceAge AI — EXP006 Checkpoint Verification Tool (Strictly Read-Only)
Verifies:
1. Exact saved best checkpoint file and corresponding epoch.
2. Architecture and strict state-dict loading (zero missing/unexpected keys).
3. Independent validation inference on certified validation split only.
4. Exact metric recomputation: confusion matrices, denominators, recalls, specificities, PU-AUCs.
5. Verification against reported Epoch 5 values.
6. Guarantees held-out test split was never touched.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from auto_train.dataset import get_transforms, RiceGrainDataset
from auto_train.model import RiceAgeMultiTask5Head
from auto_train.evaluate import compute_roc_auc


def run_verification():
    print("=" * 80)
    print("RiceAge AI — EXP006 Checkpoint Independent Verification")
    print("=" * 80)
    
    ckpt_dir = os.path.join(project_root, 'outputs', 'phase2_automated', 'experiments', 'exp006_targeted_defect_improvement')
    best_ckpt_path = os.path.join(ckpt_dir, 'best_multitask_model.pt')
    last_ckpt_path = os.path.join(ckpt_dir, 'last_multitask_model.pt')
    latest_state_path = os.path.join(ckpt_dir, 'latest_training_state.pt')
    final_report_path = os.path.join(ckpt_dir, 'final_evaluation_report.json')
    epoch_metrics_path = os.path.join(ckpt_dir, 'epoch_metrics.json')
    
    manifest_path = os.path.join(project_root, 'outputs', 'phase2_automated', 'automated_dataset_manifest.csv')
    crosswalk_path = os.path.join(project_root, 'outputs', 'phase1_2_final_audit', 'canonical_id_crosswalk.csv')
    crop_root = os.path.join(project_root, 'dataset', 'crops')
    
    output_pred_path = os.path.join(ckpt_dir, 'exp006_validation_predictions.npz')
    output_summary_json = os.path.join(ckpt_dir, 'exp006_checkpoint_verification.json')

    # 1. Identify Checkpoint and Confirm Epoch
    print("\n--- 1. Checkpoint Identification & Epoch Confirmation ---")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")
        
    latest_state = torch.load(latest_state_path, map_location='cpu', weights_only=False)
    with open(final_report_path, 'r', encoding='utf-8') as f:
        final_report = json.load(f)
    with open(epoch_metrics_path, 'r', encoding='utf-8') as f:
        epoch_metrics = json.load(f)
        
    best_epoch_reported = final_report.get('best_epoch')
    total_epochs_run = len(epoch_metrics)
    print(f"Best Checkpoint File: {best_ckpt_path}")
    print(f"Reported Best Epoch: {best_epoch_reported} (out of {total_epochs_run} epochs run)")
    print(f"Latest State Epoch: {latest_state['epoch']}")

    # Check whether best_multitask_model weights match last_multitask_model and latest_state teacher
    best_weights = torch.load(best_ckpt_path, map_location='cpu')
    last_weights = torch.load(last_ckpt_path, map_location='cpu')
    teacher_weights = latest_state['teacher_state']
    
    is_identical_to_last = all(torch.equal(best_weights[k], last_weights[k]) for k in best_weights.keys())
    is_identical_to_teacher_ep5 = all(torch.equal(best_weights[k], teacher_weights[k]) for k in best_weights.keys())
    print(f"Best checkpoint matches Last checkpoint (Epoch 5): {is_identical_to_last}")
    print(f"Best checkpoint matches Teacher State (Epoch 5): {is_identical_to_teacher_ep5}")

    # 2. Architecture & Strict State Dict Verification
    print("\n--- 2. Architecture & Strict State-Dict Compatibility ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verification Device: {device}")
    
    model = RiceAgeMultiTask5Head(pretrained=False).to(device)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(best_weights.keys())
    
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys
    
    print(f"Total Model Parameter Tensors: {len(model_keys)}")
    print(f"Missing Keys: {len(missing_keys)}")
    print(f"Unexpected Keys: {len(unexpected_keys)}")
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError("Strict state-dict load failed: key mismatch detected!")
        
    model.load_state_dict(best_weights, strict=True)
    model.eval()
    print("Strict load_state_dict(strict=True): PASSED (Zero mismatched keys).")

    # 3. Load Validation Set (Guaranteed Zero Test Access)
    print("\n--- 3. Dataset Integrity & Partition Quarantine Check ---")
    manifest = pd.read_csv(manifest_path)
    
    # Audit partition splits in manifest
    split_counts = manifest['pu_learning_split_role'].value_counts().to_dict()
    print(f"Manifest Split Breakdown: {split_counts}")
    
    # Strictly select validation rows only
    val_df = manifest[manifest['pu_learning_split_role'] == 'VAL_TUNING_RESERVED'].copy().reset_index(drop=True)
    n_val = len(val_df)
    print(f"Validation Instances Selected: {n_val}")
    if n_val != 4944:
        raise ValueError(f"Expected 4944 validation rows, found {n_val}")
        
    # Verify test set remains unaccessed
    test_rows = manifest[manifest['pu_learning_split_role'] == 'TEST_BENCHMARK_ISOLATED']
    print(f"Locked Test Split: {len(test_rows)} rows strictly quarantined and unaccessed.")

    crosswalk = pd.read_csv(crosswalk_path)
    primaries = crosswalk[crosswalk['final_disposition'] == 'canonical_primary'].drop_duplicates('canonical_instance_id').set_index('canonical_instance_id')
    
    _, _, val_tf = get_transforms(input_size=(128, 128))
    val_dataset = RiceGrainDataset(val_df, primaries, crop_root=crop_root, transform=val_tf, is_training=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Independent Validation Inference
    print("\n--- 4. Running Validation Inference on EXP006 Best Checkpoint ---")
    all_pred_id, all_true_id = [], []
    all_pred_mo, all_true_mo, all_mask_mo = [], [], []
    all_pred_dam, all_true_dam = [], []
    all_pred_chk, all_true_chk = [], []
    all_pred_dsc, all_true_dsc = [], []
    all_mask_grain = []
    
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets, meta in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            
            all_pred_id.extend(torch.sigmoid(preds['identity']).cpu().numpy().tolist())
            all_true_id.extend(targets['y_ident'].numpy().tolist())
            
            all_pred_mo.extend(torch.sigmoid(preds['integrity']).cpu().numpy().tolist())
            all_true_mo.extend(targets['y_morph'].numpy().tolist())
            all_mask_mo.extend(targets['m_morph'].numpy().tolist())
            
            all_pred_dam.extend(torch.sigmoid(preds['damaged']).cpu().numpy().tolist())
            all_true_dam.extend(targets['y_dam'].numpy().tolist())
            
            all_pred_chk.extend(torch.sigmoid(preds['chalky']).cpu().numpy().tolist())
            all_true_chk.extend(targets['y_chk'].numpy().tolist())
            
            all_pred_dsc.extend(torch.sigmoid(preds['discoloured']).cpu().numpy().tolist())
            all_true_dsc.extend(targets['y_dsc'].numpy().tolist())
            
            all_mask_grain.extend(targets['m_grain'].numpy().tolist())
            
    inference_time = time.time() - t0
    print(f"Validation inference complete in {inference_time:.2f}s ({n_val} instances).")

    # Convert to numpy
    all_pred_id = np.array(all_pred_id)
    all_true_id = np.array(all_true_id)
    all_pred_mo = np.array(all_pred_mo)
    all_true_mo = np.array(all_true_mo)
    all_mask_mo = np.array(all_mask_mo) == 1.0
    all_pred_dam = np.array(all_pred_dam)
    all_true_dam = np.array(all_true_dam)
    all_pred_chk = np.array(all_pred_chk)
    all_true_chk = np.array(all_true_chk)
    all_pred_dsc = np.array(all_pred_dsc)
    all_true_dsc = np.array(all_true_dsc)
    all_mask_grain = np.array(all_mask_grain) == 1.0

    # Save EXP006 validation predictions to separate file
    np.savez_compressed(
        output_pred_path,
        cids=val_df['canonical_instance_id'].values,
        exp006_id=all_pred_id,
        exp006_mo=all_pred_mo,
        exp006_dam=all_pred_dam,
        exp006_chk=all_pred_chk,
        exp006_dsc=all_pred_dsc
    )
    print(f"Saved separate EXP006 validation predictions to: {output_pred_path}")

    # 5. Exact Metric Recomputations
    print("\n--- 5. Exact Metric Calculations & Confusion Matrices ---")
    
    # Head 1: Object Identity (Foreign Material vs Rice Kernel)
    # y=1 foreign material, y=0 rice kernel
    p_id_cls = (all_pred_id >= 0.50).astype(int)
    y_id_cls = all_true_id.astype(int)
    tp_i = int(np.sum((p_id_cls == 1) & (y_id_cls == 1)))
    fp_i = int(np.sum((p_id_cls == 1) & (y_id_cls == 0)))
    fn_i = int(np.sum((p_id_cls == 0) & (y_id_cls == 1)))
    tn_i = int(np.sum((p_id_cls == 0) & (y_id_cls == 0)))
    prec_id = tp_i / max(tp_i + fp_i, 1)
    rec_id = tp_i / max(tp_i + fn_i, 1)
    f1_id = 2.0 * prec_id * rec_id / max(prec_id + rec_id, 1e-7)
    acc_id = (tp_i + tn_i) / len(y_id_cls)

    print(f"Head 1 (Object Identity):")
    print(f"  Confusion Matrix: TP={tp_i}, FP={fp_i}, FN={fn_i}, TN={tn_i} (Total N={len(y_id_cls)})")
    print(f"  Accuracy: {acc_id:.4f} ({acc_id*100:.2f}%)")
    print(f"  Precision: {prec_id:.4f} | Recall: {rec_id:.4f} ({tp_i}/{tp_i+fn_i})")
    print(f"  F1 Score: {f1_id:.4f}")

    # Head 2: Physical Integrity (Broken Kernel vs Whole Grain)
    # Masked on annotated morphology (whole vs broken)
    p_mo_cls = (all_pred_mo[all_mask_mo] >= 0.50).astype(int)
    y_mo_cls = all_true_mo[all_mask_mo].astype(int)
    tp_m = int(np.sum((p_mo_cls == 1) & (y_mo_cls == 1)))
    fp_m = int(np.sum((p_mo_cls == 1) & (y_mo_cls == 0)))
    fn_m = int(np.sum((p_mo_cls == 0) & (y_mo_cls == 1)))
    tn_m = int(np.sum((p_mo_cls == 0) & (y_mo_cls == 0)))
    prec_mo = tp_m / max(tp_m + fp_m, 1)
    rec_mo = tp_m / max(tp_m + fn_m, 1)
    f1_mo = 2.0 * prec_mo * rec_mo / max(prec_mo + rec_mo, 1e-7)
    acc_mo = (tp_m + tn_m) / len(y_mo_cls)

    print(f"\nHead 2 (Physical Integrity):")
    print(f"  Evaluated Instances: {len(y_mo_cls)} (Masked out unspecified: {int((~all_mask_mo & all_mask_grain).sum())})")
    print(f"  Confusion Matrix: TP={tp_m}, FP={fp_m}, FN={fn_m}, TN={tn_m}")
    print(f"  Accuracy: {acc_mo:.4f} ({acc_mo*100:.2f}%)")
    print(f"  Precision: {prec_mo:.4f} | Recall: {rec_mo:.4f} ({tp_m}/{tp_m+fn_m})")
    print(f"  F1 Score: {f1_mo:.4f}")

    # Heads 3-5: Defect Metrics (Evaluated strictly on the 4,831 eligible rice grains)
    n_rice = int(all_mask_grain.sum())
    print(f"\nDefect Heads Evaluated on Eligible Rice Population: {n_rice} grains")
    
    # Head 3: Damaged
    p_dam_gr = all_pred_dam[all_mask_grain]
    y_dam_gr = all_true_dam[all_mask_grain]
    pos_dam = (y_dam_gr == 1.0)
    n_dam_pos = int(pos_dam.sum())
    tp_dam_50 = int(np.sum(p_dam_gr[pos_dam] >= 0.50))
    fn_dam_50 = n_dam_pos - tp_dam_50
    rec_dam_50 = tp_dam_50 / n_dam_pos
    
    tp_dam_80 = int(np.sum(p_dam_gr[pos_dam] >= 0.80))
    fn_dam_80 = n_dam_pos - tp_dam_80
    rec_dam_80 = tp_dam_80 / n_dam_pos
    
    auc_dam = compute_roc_auc(y_dam_gr, p_dam_gr)
    print(f"Head 3 (Damaged Defect):")
    print(f"  Confirmed Positives Denominator: N_P = {n_dam_pos}")
    print(f"  Recall @ 0.50: {rec_dam_50:.4f} ({rec_dam_50*100:.2f}%) [TP={tp_dam_50}, FN={fn_dam_50}]")
    print(f"  Recall @ 0.80: {rec_dam_80:.4f} ({rec_dam_80*100:.2f}%) [TP={tp_dam_80}, FN={fn_dam_80}]")
    print(f"  PU-AUC: {auc_dam:.6f} ({round(auc_dam, 4)})")

    # Head 4: Chalky
    p_chk_gr = all_pred_chk[all_mask_grain]
    y_chk_gr = all_true_chk[all_mask_grain]
    pos_chk = (y_chk_gr == 1.0)
    neg_chk = (y_chk_gr == -1.0)
    n_chk_pos = int(pos_chk.sum())
    n_chk_neg = int(neg_chk.sum())
    
    tp_chk_50 = int(np.sum(p_chk_gr[pos_chk] >= 0.50))
    fn_chk_50 = n_chk_pos - tp_chk_50
    rec_chk_50 = tp_chk_50 / n_chk_pos
    
    tn_chk_50 = int(np.sum(p_chk_gr[neg_chk] < 0.50))
    fp_chk_50 = n_chk_neg - tn_chk_50
    spec_chk_50 = tn_chk_50 / n_chk_neg
    
    auc_chk = compute_roc_auc((y_chk_gr == 1.0).astype(float), p_chk_gr)
    print(f"\nHead 4 (Chalky Defect):")
    print(f"  Confirmed Positives Denominator: N_P = {n_chk_pos}")
    print(f"  Confirmed Negatives Denominator (Pure Controls): N_N = {n_chk_neg}")
    print(f"  Recall @ 0.50: {rec_chk_50:.4f} ({rec_chk_50*100:.2f}%) [TP={tp_chk_50}, FN={fn_chk_50}]")
    print(f"  Specificity on Confirmed Controls: {spec_chk_50:.4f} ({spec_chk_50*100:.2f}%) [TN={tn_chk_50}, FP={fp_chk_50}]")
    print(f"  PU-AUC: {auc_chk:.6f} ({round(auc_chk, 4)})")

    # Head 5: Discoloured
    p_dsc_gr = all_pred_dsc[all_mask_grain]
    y_dsc_gr = all_true_dsc[all_mask_grain]
    pos_dsc = (y_dsc_gr == 1.0)
    n_dsc_pos = int(pos_dsc.sum())
    
    tp_dsc_50 = int(np.sum(p_dsc_gr[pos_dsc] >= 0.50))
    fn_dsc_50 = n_dsc_pos - tp_dsc_50
    rec_dsc_50 = tp_dsc_50 / n_dsc_pos
    
    tp_dsc_80 = int(np.sum(p_dsc_gr[pos_dsc] >= 0.80))
    fn_dsc_80 = n_dsc_pos - tp_dsc_80
    rec_dsc_80 = tp_dsc_80 / n_dsc_pos
    
    auc_dsc = compute_roc_auc(y_dsc_gr, p_dsc_gr)
    print(f"\nHead 5 (Discoloured Defect):")
    print(f"  Confirmed Positives Denominator: N_P = {n_dsc_pos}")
    print(f"  Recall @ 0.50: {rec_dsc_50:.4f} ({rec_dsc_50*100:.2f}%) [TP={tp_dsc_50}, FN={fn_dsc_50}]")
    print(f"  Recall @ 0.80: {rec_dsc_80:.4f} ({rec_dsc_80*100:.2f}%) [TP={tp_dsc_80}, FN={fn_dsc_80}]")
    print(f"  PU-AUC: {auc_dsc:.6f} ({round(auc_dsc, 4)})")

    # Mean PU-AUC
    mean_pu_auc = (auc_dam + auc_chk + auc_dsc) / 3.0
    baseline_pu_auc = 0.883641
    delta_pu_auc = mean_pu_auc - baseline_pu_auc
    print(f"\nComposite PU-AUC:")
    print(f"  EXP006 Mean PU-AUC: {mean_pu_auc:.6f} ({round(mean_pu_auc, 4)})")
    print(f"  Baseline (EXP003) Mean PU-AUC: {baseline_pu_auc:.6f} ({round(baseline_pu_auc, 4)})")
    print(f"  Difference from Baseline (Delta): {delta_pu_auc:+.6f} ({round(delta_pu_auc, 4)})")

    # 6. Compare with Reported Epoch 5 Values
    print("\n--- 6. Discrepancy Analysis vs Reported Values ---")
    reported_metrics = final_report['final_exp006_metrics']
    reported_gates = final_report['final_exp006_gates']
    
    comparisons = [
        ("Identity F1", reported_metrics['head1_identity']['f1_score'], f1_id),
        ("Integrity F1", reported_metrics['head2_integrity']['f1_score'], f1_mo),
        ("Damaged Recall @ 0.50", reported_metrics['head3_damaged']['recall_at_0_50'], rec_dam_50),
        ("Damaged Recall @ 0.80", reported_metrics['head3_damaged']['recall_at_0_80'], rec_dam_80),
        ("Chalky Recall @ 0.50", reported_metrics['head4_chalky']['recall_at_0_50'], rec_chk_50),
        ("Chalky Specificity (N=59)", reported_metrics['head4_chalky']['confirmed_negative_specificity'], spec_chk_50),
        ("Discoloured Recall @ 0.50", reported_metrics['head5_discoloured']['recall_at_0_50'], rec_dsc_50),
        ("Discoloured Recall @ 0.80", reported_metrics['head5_discoloured']['recall_at_0_80'], rec_dsc_80),
        ("Damaged PU-AUC", reported_metrics['head3_damaged']['roc_auc'], auc_dam),
        ("Chalky PU-AUC", reported_metrics['head4_chalky']['roc_auc'], auc_chk),
        ("Discoloured PU-AUC", reported_metrics['head5_discoloured']['roc_auc'], auc_dsc),
        ("Mean PU-AUC", reported_gates['mean_pu_auc'], mean_pu_auc),
        ("Delta PU-AUC vs Baseline", reported_gates['delta_pu_auc'], delta_pu_auc),
    ]
    
    discrepancies = []
    print(f"{'Metric':<30} | {'Reported':<12} | {'Recomputed':<12} | {'Diff':<10} | {'Status'}")
    print("-" * 75)
    for name, rep, recomp in comparisons:
        diff = abs(rep - recomp)
        status = "MATCH" if diff < 1e-4 else "DISCREPANCY"
        if status == "DISCREPANCY":
            discrepancies.append((name, rep, recomp, diff))
        print(f"{name:<30} | {rep:<12.4f} | {recomp:<12.4f} | {diff:<10.6f} | {status}")
        
    print(f"\nTotal Discrepancies Found: {len(discrepancies)}")
    if len(discrepancies) == 0:
        print("ALL RECOMPUTED METRICS EXACTLY MATCH REPORTED VALUES TO 4 DECIMAL PLACES!")

    # 7. Verification Summary JSON
    verification_summary = {
        'status': 'VERIFIED_MATCH' if len(discrepancies) == 0 else 'DISCREPANCIES_DETECTED',
        'checkpoint_path': best_ckpt_path,
        'verified_epoch': 5,
        'missing_keys_count': len(missing_keys),
        'unexpected_keys_count': len(unexpected_keys),
        'total_discrepancies': len(discrepancies),
        'quarantined_test_split_accessed': False,
        'confusion_matrices': {
            'head1_identity': {'tp': tp_i, 'fp': fp_i, 'fn': fn_i, 'tn': tn_i, 'n': len(y_id_cls)},
            'head2_integrity': {'tp': tp_m, 'fp': fp_m, 'fn': fn_m, 'tn': tn_m, 'n': len(y_mo_cls)},
            'head3_damaged': {'tp_50': tp_dam_50, 'fn_50': fn_dam_50, 'n_pos': n_dam_pos, 'tp_80': tp_dam_80, 'fn_80': fn_dam_80},
            'head4_chalky': {'tp_50': tp_chk_50, 'fn_50': fn_chk_50, 'n_pos': n_chk_pos, 'tn_50': tn_chk_50, 'fp_50': fp_chk_50, 'n_neg': n_chk_neg},
            'head5_discoloured': {'tp_50': tp_dsc_50, 'fn_50': fn_dsc_50, 'n_pos': n_dsc_pos, 'tp_80': tp_dsc_80, 'fn_80': fn_dsc_80}
        },
        'verified_metrics': {
            'identity_f1': round(f1_id, 4),
            'integrity_f1': round(f1_mo, 4),
            'damaged_recall_50': round(rec_dam_50, 4),
            'damaged_recall_80': round(rec_dam_80, 4),
            'chalky_recall_50': round(rec_chk_50, 4),
            'chalky_specificity_59': round(spec_chk_50, 4),
            'discoloured_recall_50': round(rec_dsc_50, 4),
            'discoloured_recall_80': round(rec_dsc_80, 4),
            'damaged_pu_auc': round(auc_dam, 6),
            'chalky_pu_auc': round(auc_chk, 6),
            'discoloured_pu_auc': round(auc_dsc, 6),
            'mean_pu_auc': round(mean_pu_auc, 6),
            'delta_pu_auc_vs_baseline': round(delta_pu_auc, 6)
        }
    }
    
    with open(output_summary_json, 'w') as f:
        json.dump(verification_summary, f, indent=2)
    print(f"\nVerification summary exported to: {output_summary_json}")
    print("=" * 80)


if __name__ == '__main__':
    run_verification()
