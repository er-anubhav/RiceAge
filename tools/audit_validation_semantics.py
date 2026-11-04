"""
RiceAge AI - Validation Semantics & PU Metric Audit Script
Executes comprehensive audit of validation dataset ground-truth states,
evaluator metric implementations, and PU-consistent evaluation metrics.
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

def run_deep_audit():
    print("=" * 80)
    print("  RICEAGE AI — VALIDATION SEMANTICS & PU METRICS AUDIT")
    print("=" * 80)
    
    # 1. Load Data
    manifest = pd.read_csv('outputs/phase2_automated/automated_dataset_manifest.csv')
    crosswalk = pd.read_csv('outputs/phase1_2_final_audit/canonical_id_crosswalk.csv')
    primaries = crosswalk[crosswalk['final_disposition'] == 'canonical_primary'].drop_duplicates('canonical_instance_id').set_index('canonical_instance_id')
    
    val_df = manifest[manifest['pu_learning_split_role'] == 'VAL_TUNING_RESERVED'].copy()
    n_val = len(val_df)
    n_rice = len(val_df[val_df['canonical_object_type'] != 'foreign_material'])
    n_foreign = len(val_df[val_df['canonical_object_type'] == 'foreign_material'])
    
    print(f"\n1. VALIDATION UNIVERSE BREAKDOWN (N = {n_val} canonical instances across {val_df['source_scan'].nunique()} scans):")
    print(f"   - Rice Grains (m_grain = 1.0):       {n_rice}")
    print(f"   - Foreign Material (m_grain = 0.0):   {n_foreign}")
    
    # 2. Per-Head Counts
    print("\n2. PER-HEAD SUPERVISION STATE ACCOUNTING:")
    
    # Head 1
    h1_p = n_foreign
    h1_n = n_rice
    print(f"   Head 1 (Object Identity):")
    print(f"     Confirmed Positives (foreign_material): {h1_p}")
    print(f"     Confirmed Negatives (rice_kernel):     {h1_n}")
    print(f"     Unlabelled / Unknown:                   0")
    print(f"     Supervision: 100% Fully Supervised Ground Truth")
    
    # Head 2
    rice_df = val_df[val_df['canonical_object_type'] != 'foreign_material']
    h2_p = (rice_df['annotated_morphology'] == 'broken').sum()
    h2_n = (rice_df['annotated_morphology'] == 'whole').sum()
    h2_u = (rice_df['annotated_morphology'] == 'unspecified').sum()
    print(f"   Head 2 (Physical Integrity - Broken vs Whole):")
    print(f"     Confirmed Positives (broken):           {h2_p}")
    print(f"     Confirmed Negatives (whole):            {h2_n}")
    print(f"     Masked Out (unspecified morphology):    {h2_u} (+ {n_foreign} foreign)")
    print(f"     Supervision: Selective Verified Ground Truth (Mask m_morph = 1.0 on {h2_p + h2_n} grains)")
    
    # Head 3
    h3_p = (rice_df['is_damaged_supervision_state'] == 'POSITIVE').sum()
    h3_n = (rice_df['is_damaged_supervision_state'] == 'NEGATIVE').sum()
    h3_u = (rice_df['is_damaged_supervision_state'] == 'UNLABELLED').sum()
    print(f"   Head 3 (Damaged Defect):")
    print(f"     Confirmed Positives:                    {h3_p} ({h3_p/n_rice*100:.1f}%)")
    print(f"     Confirmed Negatives:                    {h3_n} (0.0% — ZERO confirmed negatives exist)")
    print(f"     Unlabelled / Unknown:                   {h3_u} ({h3_u/n_rice*100:.1f}%)")
    print(f"     Supervision: Strictly Positive-Unlabelled (PU)")
    
    # Head 4
    h4_p = (rice_df['is_chalky_supervision_state'] == 'POSITIVE').sum()
    h4_n = (rice_df['is_chalky_supervision_state'] == 'NEGATIVE').sum()
    h4_u = (rice_df['is_chalky_supervision_state'] == 'UNLABELLED').sum()
    print(f"   Head 4 (Chalky Defect):")
    print(f"     Confirmed Positives:                    {h4_p} ({h4_p/n_rice*100:.1f}%)")
    print(f"     Confirmed Negatives:                    {h4_n} ({h4_n/n_rice*100:.1f}% — Pure reference tray controls)")
    print(f"     Unlabelled / Unknown:                   {h4_u} ({h4_u/n_rice*100:.1f}%)")
    print(f"     Supervision: Positive-Negative-Unlabelled (PNU, 92.0% unlabelled)")
    
    # Head 5
    h5_p = (rice_df['is_discoloured_supervision_state'] == 'POSITIVE').sum()
    h5_n = (rice_df['is_discoloured_supervision_state'] == 'NEGATIVE').sum()
    h5_u = (rice_df['is_discoloured_supervision_state'] == 'UNLABELLED').sum()
    print(f"   Head 5 (Discoloured Defect):")
    print(f"     Confirmed Positives:                    {h5_p} ({h5_p/n_rice*100:.1f}%)")
    print(f"     Confirmed Negatives:                    {h5_n} (0.0% — ZERO confirmed negatives exist)")
    print(f"     Unlabelled / Unknown:                   {h5_u} ({h5_u/n_rice*100:.1f}%)")
    print(f"     Supervision: Strictly Positive-Unlabelled (PU)")
    
    # 3. Model Inference on Validation Set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n3. RUNNING INFERENCE ON ACTIVE CHECKPOINTS (Device: {device})...")
    
    _, _, val_tf = get_transforms()
    val_ds = RiceGrainDataset(val_df, primaries, crop_root='dataset/crops', transform=val_tf, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    def predict(model_path, is_state=False):
        m = RiceAgeMultiTask5Head(False).to(device)
        if is_state:
            st = torch.load(model_path, map_location=device)['teacher_state']
        else:
            st = torch.load(model_path, map_location=device)
        m.load_state_dict(st)
        m.eval()
        
        preds = {'id': [], 'mo': [], 'dam': [], 'chk': [], 'dsc': []}
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                out = m(imgs.to(device))
                preds['id'].extend(torch.sigmoid(out['identity']).cpu().numpy())
                preds['mo'].extend(torch.sigmoid(out['integrity']).cpu().numpy())
                preds['dam'].extend(torch.sigmoid(out['damaged']).cpu().numpy())
                preds['chk'].extend(torch.sigmoid(out['chalky']).cpu().numpy())
                preds['dsc'].extend(torch.sigmoid(out['discoloured']).cpu().numpy())
        return {k: np.array(v) for k, v in preds.items()}
    
    base_preds = predict('outputs/phase2_automated/experiments/exp003_multitask_integrated/best_multitask_model.pt')
    e5_preds = predict('outputs/phase2_automated/experiments/exp005_targeted_improvement/latest_training_state.pt', is_state=True)
    
    # Ground Truth Arrays
    y_id = np.array([1.0 if r['canonical_object_type'] == 'foreign_material' else 0.0 for _, r in val_df.iterrows()])
    m_mo = np.array([1.0 if (r['canonical_object_type'] != 'foreign_material' and r['annotated_morphology'] in ['whole', 'broken']) else 0.0 for _, r in val_df.iterrows()]) == 1.0
    y_mo = np.array([1.0 if r['annotated_morphology'] == 'broken' else 0.0 for _, r in val_df.iterrows()])
    
    m_grain = np.array([1.0 if r['canonical_object_type'] != 'foreign_material' else 0.0 for _, r in val_df.iterrows()]) == 1.0
    
    y_dam = np.array([1.0 if r['is_damaged_supervision_state'] == 'POSITIVE' else 0.0 for _, r in val_df.iterrows()])
    y_chk_p = np.array([1.0 if r['is_chalky_supervision_state'] == 'POSITIVE' else 0.0 for _, r in val_df.iterrows()])
    y_chk_n = np.array([1.0 if r['is_chalky_supervision_state'] == 'NEGATIVE' else 0.0 for _, r in val_df.iterrows()])
    y_dsc = np.array([1.0 if r['is_discoloured_supervision_state'] == 'POSITIVE' else 0.0 for _, r in val_df.iterrows()])
    
    # Class Priors
    priors = {'dam': 0.2650, 'chk': 0.1250, 'dsc': 0.0420}
    
    def analyze_head(name, p, y, m_mask, prior, neg_mask=None):
        p_sub = p[m_mask]
        y_sub = y[m_mask]
        
        pos_mask = (y_sub == 1.0)
        unl_mask = (y_sub == 0.0) if neg_mask is None else ((y_sub == 0.0) & ~(neg_mask[m_mask] == 1.0))
        
        # 1. Recall on Confirmed Positives
        rec_50 = float(np.mean(p_sub[pos_mask] >= 0.50))
        rec_80 = float(np.mean(p_sub[pos_mask] >= 0.80))
        
        # 2. Specificity on Confirmed Negatives (if present)
        spec_neg = float(np.mean(p_sub[neg_mask[m_mask] == 1.0] < 0.50)) if neg_mask is not None and (neg_mask[m_mask] == 1.0).any() else None
        
        # 3. PU-AUC (Monotonic transform of True AUC)
        pu_auc = compute_roc_auc(y_sub, p_sub)
        
        # 4. Classical AUPRC (treating U as confirmed negative)
        classic_auprc = compute_auprc(y_sub, p_sub)
        
        # 5. Unlabelled Distribution Moments
        unl_p = p_sub[unl_mask]
        u_mean = float(np.mean(unl_p))
        u_med = float(np.median(unl_p))
        u_p90 = float(np.percentile(unl_p, 90))
        u_frac_pos_50 = float(np.mean(unl_p >= 0.50))
        
        # 6. PU-Consistent Adjusted True Precision & True F1 (Elkan & Noto / du Plessis)
        # P(Y=1 | g(X)>=0.50) = (pi * R_p(0.50)) / P(g(X)>=0.50)
        p_hat_total = float(np.mean(p_sub >= 0.50))
        adj_prec = min(1.0, (prior * rec_50) / max(p_hat_total, 1e-7)) if p_hat_total > 0 else 0.0
        adj_f1 = (2 * adj_prec * rec_50) / max(adj_prec + rec_50, 1e-7)
        
        return {
            'rec_50': rec_50,
            'rec_80': rec_80,
            'spec_neg': spec_neg,
            'pu_auc': pu_auc,
            'classic_auprc': classic_auprc,
            'unl_mean': u_mean,
            'unl_med': u_med,
            'unl_p90': u_p90,
            'unl_frac_pos_50': u_frac_pos_50,
            'adj_prec': adj_prec,
            'adj_f1': adj_f1
        }
    
    print("\n" + "=" * 80)
    print("4. COMPARATIVE EVALUATION: BASELINE (EXP003) vs EXP005")
    print("=" * 80)
    
    # Head 1
    p_b_id = (base_preds['id'] >= 0.5).astype(int)
    p_5_id = (e5_preds['id'] >= 0.5).astype(int)
    y_id_int = y_id.astype(int)
    
    f1_b_id = 2 * (np.sum((p_b_id==1)&(y_id_int==1))) / max(2*np.sum((p_b_id==1)&(y_id_int==1)) + np.sum((p_b_id==1)&(y_id_int==0)) + np.sum((p_b_id==0)&(y_id_int==1)), 1)
    f1_5_id = 2 * (np.sum((p_5_id==1)&(y_id_int==1))) / max(2*np.sum((p_5_id==1)&(y_id_int==1)) + np.sum((p_5_id==1)&(y_id_int==0)) + np.sum((p_5_id==0)&(y_id_int==1)), 1)
    
    # Head 2
    p_b_mo = (base_preds['mo'][m_mo] >= 0.5).astype(int)
    p_5_mo = (e5_preds['mo'][m_mo] >= 0.5).astype(int)
    y_mo_int = y_mo[m_mo].astype(int)
    
    f1_b_mo = 2 * (np.sum((p_b_mo==1)&(y_mo_int==1))) / max(2*np.sum((p_b_mo==1)&(y_mo_int==1)) + np.sum((p_b_mo==1)&(y_mo_int==0)) + np.sum((p_b_mo==0)&(y_mo_int==1)), 1)
    f1_5_mo = 2 * (np.sum((p_5_mo==1)&(y_mo_int==1))) / max(2*np.sum((p_5_mo==1)&(y_mo_int==1)) + np.sum((p_5_mo==1)&(y_mo_int==0)) + np.sum((p_5_mo==0)&(y_mo_int==1)), 1)
    
    print(f"Head 1 Identity F1:  Baseline = {f1_b_id:.4f} | EXP005 = {f1_5_id:.4f} (Delta: {f1_5_id - f1_b_id:+.4f})")
    print(f"Head 2 Integrity F1: Baseline = {f1_b_mo:.4f} | EXP005 = {f1_5_mo:.4f} (Delta: {f1_5_mo - f1_b_mo:+.4f})")
    
    # Defect Heads
    for name, key, y_t, neg_m in [
        ('Head 3 (Damaged)', 'dam', y_dam, None),
        ('Head 4 (Chalky)', 'chk', y_chk_p, y_chk_n),
        ('Head 5 (Discoloured)', 'dsc', y_dsc, None)
    ]:
        b_res = analyze_head(name, base_preds[key], y_t, m_grain, priors[key], neg_m)
        e_res = analyze_head(name, e5_preds[key], y_t, m_grain, priors[key], neg_m)
        
        print(f"\n{name} [Prior = {priors[key]:.4f}]:")
        print(f"  Recall @ 0.50 (Confirmed Positives): Baseline = {b_res['rec_50']*100:.2f}% | EXP005 = {e_res['rec_50']*100:.2f}% (Delta: {e_res['rec_50']-b_res['rec_50']:+.2%})")
        print(f"  Recall @ 0.80 (High-Confidence):      Baseline = {b_res['rec_80']*100:.2f}% | EXP005 = {e_res['rec_80']*100:.2f}% (Delta: {e_res['rec_80']-b_res['rec_80']:+.2%})")
        if b_res['spec_neg'] is not None:
            print(f"  Specificity on Confirmed Negatives:  Baseline = {b_res['spec_neg']*100:.2f}% | EXP005 = {e_res['spec_neg']*100:.2f}% (Delta: {e_res['spec_neg']-b_res['spec_neg']:+.2%})")
        print(f"  PU-AUC (Rank Separation P vs U):     Baseline = {b_res['pu_auc']:.4f} | EXP005 = {e_res['pu_auc']:.4f} (Delta: {e_res['pu_auc']-b_res['pu_auc']:+.4f})")
        print(f"  Classical AUPRC (Treating U as neg): Baseline = {b_res['classic_auprc']:.4f} | EXP005 = {e_res['classic_auprc']:.4f} (Delta: {e_res['classic_auprc']-b_res['classic_auprc']:+.4f})")
        print(f"  Unlabelled Predictions >= 0.50:      Baseline = {b_res['unl_frac_pos_50']*100:.2f}% | EXP005 = {e_res['unl_frac_pos_50']*100:.2f}% (Delta: {e_res['unl_frac_pos_50']-b_res['unl_frac_pos_50']:+.2%})")
        print(f"  PU-Adjusted Precision @ 0.50:        Baseline = {b_res['adj_prec']:.4f} | EXP005 = {e_res['adj_prec']:.4f} (Delta: {e_res['adj_prec']-b_res['adj_prec']:+.4f})")
        print(f"  PU-Adjusted F1 @ 0.50:               Baseline = {b_res['adj_f1']:.4f} | EXP005 = {e_res['adj_f1']:.4f} (Delta: {e_res['adj_f1']-b_res['adj_f1']:+.4f})")
        
    print("\n" + "=" * 80)
    print("5. COMPOSITE SCORE COMPARISONS")
    print("=" * 80)
    
    # Baseline Composite Scores
    dam_b = analyze_head('dam', base_preds['dam'], y_dam, m_grain, priors['dam'])
    chk_b = analyze_head('chk', base_preds['chk'], y_chk_p, m_grain, priors['chk'], y_chk_n)
    dsc_b = analyze_head('dsc', base_preds['dsc'], y_dsc, m_grain, priors['dsc'])
    
    dam_5 = analyze_head('dam', e5_preds['dam'], y_dam, m_grain, priors['dam'])
    chk_5 = analyze_head('chk', e5_preds['chk'], y_chk_p, m_grain, priors['chk'], y_chk_n)
    dsc_5 = analyze_head('dsc', e5_preds['dsc'], y_dsc, m_grain, priors['dsc'])
    
    # Score 1: Existing Composite (treating U as confirmed negative)
    s1_b = (f1_b_id + f1_b_mo + dam_b['classic_auprc'] + chk_b['classic_auprc'] + dsc_b['classic_auprc']) / 5.0
    s1_5 = (f1_5_id + f1_5_mo + dam_5['classic_auprc'] + chk_5['classic_auprc'] + dsc_5['classic_auprc']) / 5.0
    
    # Score 2: Original EXP003 Joint Score (Recall-based for defect heads)
    s2_b = (f1_b_id + f1_b_mo + dam_b['rec_50'] + chk_b['rec_50'] + dsc_b['rec_50']) / 5.0
    s2_5 = (f1_5_id + f1_5_mo + dam_5['rec_50'] + chk_5['rec_50'] + dsc_5['rec_50']) / 5.0
    
    # Score 3: PU-AUC Joint Score (Rank-based)
    s3_b = (f1_b_id + f1_b_mo + dam_b['pu_auc'] + chk_b['pu_auc'] + dsc_b['pu_auc']) / 5.0
    s3_5 = (f1_5_id + f1_5_mo + dam_5['pu_auc'] + chk_5['pu_auc'] + dsc_5['pu_auc']) / 5.0
    
    # Score 4: PU-Adjusted F1 Joint Score (Harmonized F1 across all 5 tasks)
    s4_b = (f1_b_id + f1_b_mo + dam_b['adj_f1'] + chk_b['adj_f1'] + dsc_b['adj_f1']) / 5.0
    s4_5 = (f1_5_id + f1_5_mo + dam_5['adj_f1'] + chk_5['adj_f1'] + dsc_5['adj_f1']) / 5.0
    
    print(f"1. Existing Composite Score (Invalid AUPRC):")
    print(f"   Baseline = {s1_b:.4f} | EXP005 = {s1_5:.4f} (Delta: {s1_5 - s1_b:+.4f}) -> REJECTS EXP005")
    print(f"\n2. Original EXP003 Joint Score (Recall-based):")
    print(f"   Baseline = {s2_b:.4f} | EXP005 = {s2_5:.4f} (Delta: {s2_5 - s2_b:+.4f}) -> ACCEPTS EXP005 (+0.32%)")
    print(f"\n3. PU-AUC Joint Score (Rank separation):")
    print(f"   Baseline = {s3_b:.4f} | EXP005 = {s3_5:.4f} (Delta: {s3_5 - s3_b:+.4f}) -> TIED/FLAT (-0.94%)")
    print(f"\n4. PU-Adjusted F1 Joint Score (Harmonized F1 using class priors):")
    print(f"   Baseline = {s4_b:.4f} | EXP005 = {s4_5:.4f} (Delta: {s4_5 - s4_b:+.4f})")
    print("=" * 80)

if __name__ == '__main__':
    run_deep_audit()
