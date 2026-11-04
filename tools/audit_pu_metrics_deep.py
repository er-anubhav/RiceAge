"""
Deep PU Metric Audit & Sensitivity Analysis Script for RiceAge AI
Executes full audit of PU metric definitions, scan-level robustness,
prior sensitivity sweeps, and model checkpoint comparisons.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from auto_train.dataset import RiceGrainDataset, get_transforms
from auto_train.model import RiceAgeMultiTask5Head
from auto_train.evaluate import compute_roc_auc, compute_auprc

def calculate_pu_adjusted_precision_f1(r_p, p_total, prior):
    """
    Computes PU-adjusted Precision and F1 with numerical safeguards.
    Handles zero division, negative clipping, and extreme edge cases.
    """
    if p_total <= 0.0 or r_p <= 0.0 or prior <= 0.0:
        return 0.0, 0.0, False # prec, f1, is_clipped
    
    raw_prec = (prior * r_p) / p_total
    is_clipped = raw_prec > 1.0
    prec = min(1.0, raw_prec)
    
    if (prec + r_p) <= 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * prec * r_p) / (prec + r_p)
    return float(prec), float(f1), bool(is_clipped)

def run_deep_audit():
    print("=" * 80)
    print("  RICEAGE AI — DEEP PU METRICS & SENSITIVITY AUDIT")
    print("=" * 80)
    
    output_dir = os.path.join(project_root, 'outputs', 'phase2_automated')
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_path = os.path.join(output_dir, 'automated_dataset_manifest.csv')
    crosswalk_path = os.path.join(project_root, 'outputs', 'phase1_2_final_audit', 'canonical_id_crosswalk.csv')
    crop_root = os.path.join(project_root, 'dataset', 'crops')
    
    df_manifest = pd.read_csv(manifest_path)
    df_crosswalk = pd.read_csv(crosswalk_path)
    primaries = df_crosswalk[df_crosswalk['final_disposition'] == 'canonical_primary'].drop_duplicates('canonical_instance_id').set_index('canonical_instance_id')
    
    val_df = df_manifest[df_manifest['pu_learning_split_role'] == 'VAL_TUNING_RESERVED'].copy()
    
    n_total_val = len(val_df)
    is_rice = (val_df['canonical_object_type'] != 'foreign_material').to_numpy()
    is_foreign = (val_df['canonical_object_type'] == 'foreign_material').to_numpy()
    n_rice = int(is_rice.sum())
    n_foreign = int(is_foreign.sum())
    
    print(f"Validation Split: Total={n_total_val} | Rice Kernels={n_rice} | Foreign Material={n_foreign}")
    
    # Per-head targets
    y_id = is_foreign.astype(float) # 1.0 = foreign, 0.0 = rice
    
    # Head 2 Integrity: evaluate only whole vs broken rice grains
    annotated_morph = val_df['annotated_morphology'].to_numpy()
    m_morph = is_rice & np.isin(annotated_morph, ['whole', 'broken'])
    y_morph = (annotated_morph == 'broken').astype(float)
    
    # Defect labels on rice grains
    y_dam_pos = (val_df['is_damaged_supervision_state'] == 'POSITIVE').to_numpy() & is_rice
    y_dam_unl = (val_df['is_damaged_supervision_state'] == 'UNLABELLED').to_numpy() & is_rice
    
    y_chk_pos = (val_df['is_chalky_supervision_state'] == 'POSITIVE').to_numpy() & is_rice
    y_chk_neg = (val_df['is_chalky_supervision_state'] == 'NEGATIVE').to_numpy() & is_rice
    y_chk_unl = (val_df['is_chalky_supervision_state'] == 'UNLABELLED').to_numpy() & is_rice
    
    y_dsc_pos = (val_df['is_discoloured_supervision_state'] == 'POSITIVE').to_numpy() & is_rice
    y_dsc_unl = (val_df['is_discoloured_supervision_state'] == 'UNLABELLED').to_numpy() & is_rice
    
    scans = val_df['source_scan'].to_numpy()
    unique_scans = np.unique(scans)
    
    # Model checkpoints to evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Models on Device: {device}...")
    
    _, _, val_tf = get_transforms()
    val_ds = RiceGrainDataset(val_df, primaries, crop_root=crop_root, transform=val_tf, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    models = {
        'EXP003_Baseline': {
            'path': os.path.join(output_dir, 'experiments', 'exp003_multitask_integrated', 'best_multitask_model.pt'),
            'is_state': False
        },
        'EXP004_SelfTrain': {
            'path': os.path.join(output_dir, 'experiments', 'exp004_ema_pseudolabel_pu', 'latest_training_state.pt'),
            'is_state': True
        },
        'EXP005_Targeted': {
            'path': os.path.join(output_dir, 'experiments', 'exp005_targeted_improvement', 'latest_training_state.pt'),
            'is_state': True
        }
    }
    
    predictions = {}
    for exp_name, m_info in models.items():
        if not os.path.exists(m_info['path']):
            print(f"Warning: Checkpoint not found: {m_info['path']}")
            continue
        model = RiceAgeMultiTask5Head(False).to(device)
        ckpt = torch.load(m_info['path'], map_location=device)
        if m_info['is_state']:
            state = ckpt.get('teacher_state', ckpt.get('student_state', ckpt))
        else:
            state = ckpt
        model.load_state_dict(state)
        model.eval()
        
        preds = {'id': [], 'mo': [], 'dam': [], 'chk': [], 'dsc': []}
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                out = model(imgs.to(device))
                preds['id'].extend(torch.sigmoid(out['identity']).cpu().numpy())
                preds['mo'].extend(torch.sigmoid(out['integrity']).cpu().numpy())
                preds['dam'].extend(torch.sigmoid(out['damaged']).cpu().numpy())
                preds['chk'].extend(torch.sigmoid(out['chalky']).cpu().numpy())
                preds['dsc'].extend(torch.sigmoid(out['discoloured']).cpu().numpy())
        predictions[exp_name] = {k: np.array(v) for k, v in preds.items()}
    
    print(f"Successfully loaded and evaluated predictions for: {list(predictions.keys())}")
    
    # ----------------------------------------------------
    # SECTION 1: PER-HEAD SAMPLE ACCOUNTING
    # ----------------------------------------------------
    population_audit = {
        'total_validation_instances': n_total_val,
        'eligible_rice_kernels': n_rice,
        'foreign_material_instances': n_foreign,
        'head1_object_identity': {
            'eligible_population': n_total_val,
            'confirmed_positives_foreign': int(y_id.sum()),
            'confirmed_negatives_rice': int((1.0 - y_id).sum()),
            'unlabelled_count': 0,
            'supervision_status': 'FULLY_SUPERVISED_100_PERCENT'
        },
        'head2_physical_integrity': {
            'eligible_population_rice_only': n_rice,
            'evaluated_population_with_known_morphology': int(m_morph.sum()),
            'confirmed_positives_broken': int((y_morph[m_morph] == 1.0).sum()),
            'confirmed_negatives_whole': int((y_morph[m_morph] == 0.0).sum()),
            'masked_out_unspecified_morphology': int((is_rice & ~m_morph).sum()),
            'supervision_status': 'SELECTIVELY_SUPERVISED_MASKED'
        },
        'head3_damaged_defect': {
            'eligible_population_rice_only': n_rice,
            'confirmed_positives': int(y_dam_pos.sum()),
            'confirmed_negatives': 0,
            'unlabelled_grains': int(y_dam_unl.sum()),
            'observed_positive_rate_val': round(float(y_dam_pos.sum() / n_rice), 6),
            'observed_positive_rate_train': 0.206373,
            'assumed_class_prior_nominal': 0.2650,
            'supervision_status': 'STRICTLY_POSITIVE_UNLABELLED'
        },
        'head4_chalky_defect': {
            'eligible_population_rice_only': n_rice,
            'confirmed_positives': int(y_chk_pos.sum()),
            'confirmed_negatives_pure_reference': int(y_chk_neg.sum()),
            'unlabelled_grains': int(y_chk_unl.sum()),
            'observed_positive_rate_val': round(float(y_chk_pos.sum() / n_rice), 6),
            'observed_positive_rate_train': 0.071368,
            'assumed_class_prior_nominal': 0.1250,
            'supervision_status': 'POSITIVE_NEGATIVE_UNLABELLED_PNU'
        },
        'head5_discoloured_defect': {
            'eligible_population_rice_only': n_rice,
            'confirmed_positives': int(y_dsc_pos.sum()),
            'confirmed_negatives': 0,
            'unlabelled_grains': int(y_dsc_unl.sum()),
            'observed_positive_rate_val': round(float(y_dsc_pos.sum() / n_rice), 6),
            'observed_positive_rate_train': 0.019909,
            'assumed_class_prior_nominal': 0.0420,
            'supervision_status': 'STRICTLY_POSITIVE_UNLABELLED'
        }
    }
    
    # ----------------------------------------------------
    # SECTION 2: PRIOR SENSITIVITY SWEEPS
    # ----------------------------------------------------
    prior_ranges = {
        'damaged': [0.150, 0.2089, 0.235, 0.265, 0.300, 0.350],
        'chalky': [0.0677, 0.090, 0.110, 0.125, 0.150, 0.180],
        'discoloured': [0.0108, 0.020, 0.030, 0.042, 0.060, 0.080]
    }
    
    sensitivity_records = []
    thresholds = [0.50, 0.80]
    
    for exp_name, p_dict in predictions.items():
        # Damaged
        p_dam = p_dict['dam'][is_rice]
        y_dam = y_dam_pos[is_rice].astype(float)
        rec_dam_50 = float(np.mean(p_dam[y_dam == 1.0] >= 0.50))
        rec_dam_80 = float(np.mean(p_dam[y_dam == 1.0] >= 0.80))
        p_tot_dam_50 = float(np.mean(p_dam >= 0.50))
        p_tot_dam_80 = float(np.mean(p_dam >= 0.80))
        
        for pi in prior_ranges['damaged']:
            prec_50, f1_50, clip_50 = calculate_pu_adjusted_precision_f1(rec_dam_50, p_tot_dam_50, pi)
            prec_80, f1_80, clip_80 = calculate_pu_adjusted_precision_f1(rec_dam_80, p_tot_dam_80, pi)
            sensitivity_records.append({
                'experiment': exp_name,
                'defect': 'damaged',
                'prior_pi': pi,
                'is_nominal_prior': (pi == 0.265),
                'recall_50': round(rec_dam_50, 4),
                'p_total_50': round(p_tot_dam_50, 4),
                'pu_adj_prec_50': round(prec_50, 4),
                'pu_adj_f1_50': round(f1_50, 4),
                'is_clipped_50': clip_50,
                'recall_80': round(rec_dam_80, 4),
                'p_total_80': round(p_tot_dam_80, 4),
                'pu_adj_prec_80': round(prec_80, 4),
                'pu_adj_f1_80': round(f1_80, 4),
                'is_clipped_80': clip_80
            })
            
        # Chalky
        p_chk = p_dict['chk'][is_rice]
        y_chk = y_chk_pos[is_rice].astype(float)
        rec_chk_50 = float(np.mean(p_chk[y_chk == 1.0] >= 0.50))
        rec_chk_80 = float(np.mean(p_chk[y_chk == 1.0] >= 0.80))
        p_tot_chk_50 = float(np.mean(p_chk >= 0.50))
        p_tot_chk_80 = float(np.mean(p_chk >= 0.80))
        
        for pi in prior_ranges['chalky']:
            prec_50, f1_50, clip_50 = calculate_pu_adjusted_precision_f1(rec_chk_50, p_tot_chk_50, pi)
            prec_80, f1_80, clip_80 = calculate_pu_adjusted_precision_f1(rec_chk_80, p_tot_chk_80, pi)
            sensitivity_records.append({
                'experiment': exp_name,
                'defect': 'chalky',
                'prior_pi': pi,
                'is_nominal_prior': (pi == 0.125),
                'recall_50': round(rec_chk_50, 4),
                'p_total_50': round(p_tot_chk_50, 4),
                'pu_adj_prec_50': round(prec_50, 4),
                'pu_adj_f1_50': round(f1_50, 4),
                'is_clipped_50': clip_50,
                'recall_80': round(rec_chk_80, 4),
                'p_total_80': round(p_tot_chk_80, 4),
                'pu_adj_prec_80': round(prec_80, 4),
                'pu_adj_f1_80': round(f1_80, 4),
                'is_clipped_80': clip_80
            })
            
        # Discoloured
        p_dsc = p_dict['dsc'][is_rice]
        y_dsc = y_dsc_pos[is_rice].astype(float)
        rec_dsc_50 = float(np.mean(p_dsc[y_dsc == 1.0] >= 0.50))
        rec_dsc_80 = float(np.mean(p_dsc[y_dsc == 1.0] >= 0.80))
        p_tot_dsc_50 = float(np.mean(p_dsc >= 0.50))
        p_tot_dsc_80 = float(np.mean(p_dsc >= 0.80))
        
        for pi in prior_ranges['discoloured']:
            prec_50, f1_50, clip_50 = calculate_pu_adjusted_precision_f1(rec_dsc_50, p_tot_dsc_50, pi)
            prec_80, f1_80, clip_80 = calculate_pu_adjusted_precision_f1(rec_dsc_80, p_tot_dsc_80, pi)
            sensitivity_records.append({
                'experiment': exp_name,
                'defect': 'discoloured',
                'prior_pi': pi,
                'is_nominal_prior': (pi == 0.042),
                'recall_50': round(rec_dsc_50, 4),
                'p_total_50': round(p_tot_dsc_50, 4),
                'pu_adj_prec_50': round(prec_50, 4),
                'pu_adj_f1_50': round(f1_50, 4),
                'is_clipped_50': clip_50,
                'recall_80': round(rec_dsc_80, 4),
                'p_total_80': round(p_tot_dsc_80, 4),
                'pu_adj_prec_80': round(prec_80, 4),
                'pu_adj_f1_80': round(f1_80, 4),
                'is_clipped_80': clip_80
            })
            
    df_sensitivity = pd.DataFrame(sensitivity_records)
    csv_sens_path = os.path.join(output_dir, 'pu_prior_sensitivity_audit.csv')
    df_sensitivity.to_csv(csv_sens_path, index=False)
    print(f"Saved prior sensitivity audit to: {csv_sens_path}")
    
    # ----------------------------------------------------
    # SECTION 3: SCAN-LEVEL ROBUSTNESS AUDIT
    # ----------------------------------------------------
    scan_records = []
    
    for sc in unique_scans:
        sc_rice_mask = (scans == sc) & is_rice
        n_sc_rice = int(sc_rice_mask.sum())
        
        if n_sc_rice == 0:
            continue
            
        n_dam_p = int(y_dam_pos[sc_rice_mask].sum())
        n_chk_p = int(y_chk_pos[sc_rice_mask].sum())
        n_chk_n = int(y_chk_neg[sc_rice_mask].sum())
        n_dsc_p = int(y_dsc_pos[sc_rice_mask].sum())
        
        row = {
            'source_scan': sc,
            'total_rice_grains': n_sc_rice,
            'damaged_confirmed_positives': n_dam_p,
            'chalky_confirmed_positives': n_chk_p,
            'chalky_confirmed_negatives': n_chk_n,
            'discoloured_confirmed_positives': n_dsc_p
        }
        
        for exp_name, p_dict in predictions.items():
            # Damaged recall & predicted rate
            p_dam_sc = p_dict['dam'][sc_rice_mask]
            row[f'{exp_name}_dam_pred_rate_50'] = round(float(np.mean(p_dam_sc >= 0.50)), 4)
            if n_dam_p > 0:
                sc_dam_pos_mask = y_dam_pos[sc_rice_mask]
                row[f'{exp_name}_dam_recall_50'] = round(float(np.mean(p_dam_sc[sc_dam_pos_mask] >= 0.50)), 4)
            else:
                row[f'{exp_name}_dam_recall_50'] = None # Undefined
                
            # Chalky recall & predicted rate
            p_chk_sc = p_dict['chk'][sc_rice_mask]
            row[f'{exp_name}_chk_pred_rate_50'] = round(float(np.mean(p_chk_sc >= 0.50)), 4)
            if n_chk_p > 0:
                sc_chk_pos_mask = y_chk_pos[sc_rice_mask]
                row[f'{exp_name}_chk_recall_50'] = round(float(np.mean(p_chk_sc[sc_chk_pos_mask] >= 0.50)), 4)
            else:
                row[f'{exp_name}_chk_recall_50'] = None # Undefined
                
            # Discoloured recall & predicted rate
            p_dsc_sc = p_dict['dsc'][sc_rice_mask]
            row[f'{exp_name}_dsc_pred_rate_50'] = round(float(np.mean(p_dsc_sc >= 0.50)), 4)
            if n_dsc_p > 0:
                sc_dsc_pos_mask = y_dsc_pos[sc_rice_mask]
                row[f'{exp_name}_dsc_recall_50'] = round(float(np.mean(p_dsc_sc[sc_dsc_pos_mask] >= 0.50)), 4)
            else:
                row[f'{exp_name}_dsc_recall_50'] = None # Undefined
                
        scan_records.append(row)
        
    df_scans = pd.DataFrame(scan_records)
    csv_scan_path = os.path.join(output_dir, 'pu_per_scan_validation_audit.csv')
    df_scans.to_csv(csv_scan_path, index=False)
    print(f"Saved scan-level audit to: {csv_scan_path}")
    
    # Calculate Scan-CV under both definitions
    scan_cv_analysis = {}
    for exp_name in predictions.keys():
        # Definition A: Current evaluate.py implementation (CV of predicted positive rates across all 29 rice scans)
        dam_pred_rates = df_scans[f'{exp_name}_dam_pred_rate_50'].values
        chk_pred_rates = df_scans[f'{exp_name}_chk_pred_rate_50'].values
        dsc_pred_rates = df_scans[f'{exp_name}_dsc_pred_rate_50'].values
        
        cv_a_dam = float(np.std(dam_pred_rates) / (np.mean(dam_pred_rates) + 1e-7))
        cv_a_chk = float(np.std(chk_pred_rates) / (np.mean(chk_pred_rates) + 1e-7))
        cv_a_dsc = float(np.std(dsc_pred_rates) / (np.mean(dsc_pred_rates) + 1e-7))
        mean_cv_a = float((cv_a_dam + cv_a_chk + cv_a_dsc) / 3.0)
        
        # Definition B: Protocol specification (CV of recall strictly across scans with positive samples)
        dam_recalls = df_scans[f'{exp_name}_dam_recall_50'].dropna().values
        chk_recalls = df_scans[f'{exp_name}_chk_recall_50'].dropna().values
        dsc_recalls = df_scans[f'{exp_name}_dsc_recall_50'].dropna().values
        
        cv_b_dam = float(np.std(dam_recalls) / (np.mean(dam_recalls) + 1e-7)) if len(dam_recalls) > 0 else None
        cv_b_chk = float(np.std(chk_recalls) / (np.mean(chk_recalls) + 1e-7)) if len(chk_recalls) > 0 else None
        cv_b_dsc = float(np.std(dsc_recalls) / (np.mean(dsc_recalls) + 1e-7)) if len(dsc_recalls) > 0 else None
        
        # Valid scans count
        n_scans_dam = len(dam_recalls)
        n_scans_chk = len(chk_recalls)
        n_scans_dsc = len(dsc_recalls)
        
        scan_cv_analysis[exp_name] = {
            'definition_A_current_predicted_rate_cv': {
                'damaged_cv': round(cv_a_dam, 4),
                'chalky_cv': round(cv_a_chk, 4),
                'discoloured_cv': round(cv_a_dsc, 4),
                'mean_scan_cv': round(mean_cv_a, 4)
            },
            'definition_B_protocol_recall_cv': {
                'damaged_eligible_scans': n_scans_dam,
                'damaged_recall_cv': round(cv_b_dam, 4) if cv_b_dam is not None else None,
                'chalky_eligible_scans': n_scans_chk,
                'chalky_recall_cv': round(cv_b_chk, 4) if cv_b_chk is not None else None,
                'discoloured_eligible_scans': n_scans_dsc,
                'discoloured_recall_cv': round(cv_b_dsc, 4) if cv_b_dsc is not None else None,
                'discoloured_zero_positive_scans_dropped': int(len(df_scans) - n_scans_dsc)
            }
        }
        
    # ----------------------------------------------------
    # SECTION 4: FULL MODEL CHECKPOINT COMPARISON
    # ----------------------------------------------------
    checkpoint_comparison = {}
    
    for exp_name, p_dict in predictions.items():
        # Head 1
        p_id = (p_dict['id'] >= 0.5).astype(int)
        y_id_i = y_id.astype(int)
        tp_i = int(np.sum((p_id == 1) & (y_id_i == 1)))
        fp_i = int(np.sum((p_id == 1) & (y_id_i == 0)))
        fn_i = int(np.sum((p_id == 0) & (y_id_i == 1)))
        tn_i = int(np.sum((p_id == 0) & (y_id_i == 0)))
        f1_id = 2.0 * tp_i / max(2*tp_i + fp_i + fn_i, 1)
        acc_id = (tp_i + tn_i) / len(y_id_i)
        
        # Head 2
        p_mo = (p_dict['mo'][m_morph] >= 0.5).astype(int)
        y_mo_i = y_morph[m_morph].astype(int)
        tp_m = int(np.sum((p_mo == 1) & (y_mo_i == 1)))
        fp_m = int(np.sum((p_mo == 1) & (y_mo_i == 0)))
        fn_m = int(np.sum((p_mo == 0) & (y_mo_i == 1)))
        tn_m = int(np.sum((p_mo == 0) & (y_mo_i == 0)))
        f1_mo = 2.0 * tp_m / max(2*tp_m + fp_m + fn_m, 1)
        acc_mo = (tp_m + tn_m) / max(len(y_mo_i), 1)
        
        # Defect heads
        p_dam = p_dict['dam'][is_rice]
        y_dam = y_dam_pos[is_rice].astype(float)
        rec_dam_50 = float(np.mean(p_dam[y_dam == 1.0] >= 0.50))
        rec_dam_80 = float(np.mean(p_dam[y_dam == 1.0] >= 0.80))
        auc_dam = compute_roc_auc(y_dam, p_dam)
        auprc_dam_legacy = compute_auprc(y_dam, p_dam)
        p_tot_dam_50 = float(np.mean(p_dam >= 0.50))
        prec_dam_pu, f1_dam_pu, clip_dam = calculate_pu_adjusted_precision_f1(rec_dam_50, p_tot_dam_50, 0.2650)
        
        p_chk = p_dict['chk'][is_rice]
        y_chk = y_chk_pos[is_rice].astype(float)
        rec_chk_50 = float(np.mean(p_chk[y_chk == 1.0] >= 0.50))
        rec_chk_80 = float(np.mean(p_chk[y_chk == 1.0] >= 0.80))
        spec_chk_neg = float(np.mean(p_chk[y_chk_neg[is_rice]] < 0.50))
        auc_chk = compute_roc_auc((y_chk == 1.0).astype(float), p_chk)
        auprc_chk_legacy = compute_auprc((y_chk == 1.0).astype(float), p_chk)
        p_tot_chk_50 = float(np.mean(p_chk >= 0.50))
        prec_chk_pu, f1_chk_pu, clip_chk = calculate_pu_adjusted_precision_f1(rec_chk_50, p_tot_chk_50, 0.1250)
        
        p_dsc = p_dict['dsc'][is_rice]
        y_dsc = y_dsc_pos[is_rice].astype(float)
        rec_dsc_50 = float(np.mean(p_dsc[y_dsc == 1.0] >= 0.50))
        rec_dsc_80 = float(np.mean(p_dsc[y_dsc == 1.0] >= 0.80))
        auc_dsc = compute_roc_auc(y_dsc, p_dsc)
        auprc_dsc_legacy = compute_auprc(y_dsc, p_dsc)
        p_tot_dsc_50 = float(np.mean(p_dsc >= 0.50))
        prec_dsc_pu, f1_dsc_pu, clip_dsc = calculate_pu_adjusted_precision_f1(rec_dsc_50, p_tot_dsc_50, 0.0420)
        
        # Composite scores
        comp_legacy = (f1_id + f1_mo + auprc_dam_legacy + auprc_chk_legacy + auprc_dsc_legacy) / 5.0
        comp_recall = (f1_id + f1_mo + rec_dam_50 + rec_chk_50 + rec_dsc_50) / 5.0
        comp_pu_auc = (f1_id + f1_mo + auc_dam + auc_chk + auc_dsc) / 5.0
        comp_pu_f1 = (f1_id + f1_mo + f1_dam_pu + f1_chk_pu + f1_dsc_pu) / 5.0
        
        checkpoint_comparison[exp_name] = {
            'head1_identity': {
                'accuracy': round(acc_id, 4),
                'f1_score': round(f1_id, 4),
                'false_positives': fp_i,
                'false_negatives': fn_i
            },
            'head2_integrity': {
                'accuracy': round(acc_mo, 4),
                'f1_score': round(f1_mo, 4),
                'false_positives': fp_m,
                'false_negatives': fn_m
            },
            'head3_damaged': {
                'recall_50': round(rec_dam_50, 4),
                'recall_80': round(rec_dam_80, 4),
                'pu_auc': round(auc_dam, 4),
                'legacy_auprc': round(auprc_dam_legacy, 4),
                'predicted_positive_fraction_50': round(p_tot_dam_50, 4),
                'pu_adjusted_prec_50_nominal': round(prec_dam_pu, 4),
                'pu_adjusted_f1_50_nominal': round(f1_dam_pu, 4),
                'precision_clipped_at_1': clip_dam
            },
            'head4_chalky': {
                'recall_50': round(rec_chk_50, 4),
                'recall_80': round(rec_chk_80, 4),
                'confirmed_negative_specificity': round(spec_chk_neg, 4),
                'pu_auc': round(auc_chk, 4),
                'legacy_auprc': round(auprc_chk_legacy, 4),
                'predicted_positive_fraction_50': round(p_tot_chk_50, 4),
                'pu_adjusted_prec_50_nominal': round(prec_chk_pu, 4),
                'pu_adjusted_f1_50_nominal': round(f1_chk_pu, 4),
                'precision_clipped_at_1': clip_chk
            },
            'head5_discoloured': {
                'recall_50': round(rec_dsc_50, 4),
                'recall_80': round(rec_dsc_80, 4),
                'pu_auc': round(auc_dsc, 4),
                'legacy_auprc': round(auprc_dsc_legacy, 4),
                'predicted_positive_fraction_50': round(p_tot_dsc_50, 4),
                'pu_adjusted_prec_50_nominal': round(prec_dsc_pu, 4),
                'pu_adjusted_f1_50_nominal': round(f1_dsc_pu, 4),
                'precision_clipped_at_1': clip_dsc
            },
            'composite_scores': {
                'legacy_composite_invalid_auprc': round(comp_legacy, 4),
                'exp003_recall_joint_score': round(comp_recall, 4),
                'pu_auc_rank_joint_score': round(comp_pu_auc, 4),
                'pu_adjusted_f1_joint_score_nominal': round(comp_pu_f1, 4)
            },
            'scan_cv': scan_cv_analysis[exp_name]
        }
        
    full_audit_summary = {
        'audit_phase': 'Pre-EXP006 Validation Semantics & PU Metric Audit',
        'eligible_populations': population_audit,
        'checkpoint_comparison': checkpoint_comparison
    }
    
    json_summary_path = os.path.join(output_dir, 'pu_metric_audit_summary.json')
    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(full_audit_summary, f, indent=2)
    print(f"Saved machine-readable audit summary to: {json_summary_path}")
    
    print("\n" + "=" * 80)
    print("AUDIT EXECUTION COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    run_deep_audit()
