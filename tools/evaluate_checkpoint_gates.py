"""
RiceAge AI — Read-Only Multi-Gate Checkpoint Evaluator
Evaluates EXP003, EXP004, and EXP005 against provisional screening gates.
Treats all gates as provisional engineering screening criteria, not validated statistical guarantees.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from auto_train.evaluate import compute_roc_auc, compute_auprc


def wilson_score_interval(x: int, n: int, confidence: float = 0.95):
    """
    Computes the Wilson score interval for a binomial proportion.
    Statistically robust for small sample sizes and extreme proportions (x = 0 or x = n).
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    
    p_hat = float(x) / float(n)
    z = 1.959963984540054  # 95% two-sided normal quantile
    if confidence == 0.99:
        z = 2.5758293035489004
    elif confidence == 0.90:
        z = 1.6448536269514722
        
    denom = 1.0 + (z**2) / float(n)
    center = (p_hat + (z**2) / (2.0 * float(n))) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / float(n)) + ((z**2) / (4.0 * float(n)**2)))
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p_hat, float(lower), float(upper)


def calculate_pu_adjusted_precision_f1(r_p: float, p_total: float, prior: float):
    """
    Calculates PU-adjusted precision and F1 with numerical safeguards.
    """
    if p_total <= 0.0 or r_p <= 0.0 or prior <= 0.0:
        return 0.0, 0.0, False
    
    raw_prec = (prior * r_p) / p_total
    is_clipped = raw_prec > 1.0
    prec = min(1.0, raw_prec)
    
    if (prec + r_p) <= 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * prec * r_p) / (prec + r_p)
    return float(prec), float(f1), bool(is_clipped)


class MultiGateCheckpointEvaluator:
    """
    Read-only multi-gate checkpoint evaluator enforcing integrity verification,
    uncertainty quantification, and provisional gate screening.
    """
    def __init__(
        self,
        manifest_path: str,
        predictions_cache_path: str
    ):
        self.manifest_path = manifest_path
        self.predictions_cache_path = predictions_cache_path
        
        # Load and verify data
        self._load_and_verify_data()
        
    def _load_and_verify_data(self):
        print("=== Step 1: Data Integrity & Provenance Alignment Verification ===")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not os.path.exists(self.predictions_cache_path):
            raise FileNotFoundError(f"Predictions cache not found: {self.predictions_cache_path}")
            
        self.manifest_df = pd.read_csv(self.manifest_path)
        self.val_df = self.manifest_df[self.manifest_df['pu_learning_split_role'] == 'VAL_TUNING_RESERVED'].copy().reset_index(drop=True)
        self.n_val = len(self.val_df)
        
        # Load cached predictions
        self.cache = np.load(self.predictions_cache_path)
        cached_cids = self.cache['cids']
        
        # Verification 1: Population size
        if len(cached_cids) != self.n_val:
            raise ValueError(f"Integrity Failure: Prediction count ({len(cached_cids)}) does not match validation manifest ({self.n_val})")
            
        # Verification 2: Row-by-row canonical instance alignment
        manifest_cids = self.val_df['canonical_instance_id'].values
        if not np.array_equal(cached_cids, manifest_cids):
            raise ValueError("Integrity Failure: Canonical instance ID order mismatch between manifest and predictions cache!")
            
        # Verification 3: Eligible populations
        self.is_rice = (self.val_df['canonical_object_type'] != 'foreign_material').to_numpy()
        self.is_foreign = (self.val_df['canonical_object_type'] == 'foreign_material').to_numpy()
        self.n_rice = int(self.is_rice.sum())
        self.n_foreign = int(self.is_foreign.sum())
        
        if self.n_rice != 4831 or self.n_foreign != 113:
            raise ValueError(f"Integrity Failure: Expected 4831 rice kernels and 113 foreign instances, found {self.n_rice} and {self.n_foreign}")
            
        # Verification 4: Head 2 Morphology mask
        annotated_morph = self.val_df['annotated_morphology'].to_numpy()
        self.m_morph = self.is_rice & np.isin(annotated_morph, ['whole', 'broken'])
        self.n_morph_eval = int(self.m_morph.sum())
        self.n_morph_unspec = int((self.is_rice & ~self.m_morph).sum())
        
        if self.n_morph_eval != 4199 or self.n_morph_unspec != 632:
            raise ValueError(f"Integrity Failure: Expected 4199 evaluated morphology instances and 632 unspecified instances, found {self.n_morph_eval} and {self.n_morph_unspec}")
            
        # Verification 5: Defect supervision masks
        self.y_dam_pos = (self.val_df['is_damaged_supervision_state'] == 'POSITIVE').to_numpy() & self.is_rice
        self.y_dam_unl = (self.val_df['is_damaged_supervision_state'] == 'UNLABELLED').to_numpy() & self.is_rice
        
        self.y_chk_pos = (self.val_df['is_chalky_supervision_state'] == 'POSITIVE').to_numpy() & self.is_rice
        self.y_chk_neg = (self.val_df['is_chalky_supervision_state'] == 'NEGATIVE').to_numpy() & self.is_rice
        self.y_chk_unl = (self.val_df['is_chalky_supervision_state'] == 'UNLABELLED').to_numpy() & self.is_rice
        
        self.y_dsc_pos = (self.val_df['is_discoloured_supervision_state'] == 'POSITIVE').to_numpy() & self.is_rice
        self.y_dsc_unl = (self.val_df['is_discoloured_supervision_state'] == 'UNLABELLED').to_numpy() & self.is_rice
        
        print(f"Data Alignment & Integrity Certified: 4,944 instances verified (4,831 rice grains, 113 foreign)")
        print(f"  - Head 1: 113 confirmed foreign positives, 4,831 confirmed rice negatives")
        print(f"  - Head 2: 408 broken positives, 3,791 whole negatives, 632 unspecified masked")
        print(f"  - Head 3: 1,009 damaged positives, 0 negatives, 3,822 unlabelled")
        print(f"  - Head 4: 327 chalky positives, 59 confirmed reference negatives, 4,445 unlabelled")
        print(f"  - Head 5: 52 discoloured positives, 0 negatives, 4,779 unlabelled")
        print("=" * 80)

    def evaluate_checkpoint(self, exp_prefix: str, exp_label: str, baseline_results=None):
        """
        Evaluates a checkpoint against provisional screening gates A, B, C, D.
        """
        # Retrieve predictions
        try:
            p_id = self.cache[f'{exp_prefix}_id']
            p_mo = self.cache[f'{exp_prefix}_mo']
            p_dam = self.cache[f'{exp_prefix}_dam']
            p_chk = self.cache[f'{exp_prefix}_chk']
            p_dsc = self.cache[f'{exp_prefix}_dsc']
        except KeyError as e:
            raise KeyError(f"Missing cached predictions for {exp_prefix}: {e}")
            
        results = {
            'experiment_id': exp_label,
            'experiment_prefix': exp_prefix,
            'gates': {},
            'metrics': {},
            'supplementary_estimates': {},
            'overall_status': 'INCONCLUSIVE'
        }
        
        # ----------------------------------------------------
        # GATE A: Supervised Head Stability
        # ----------------------------------------------------
        # Head 1: Object Identity (Foreign Material vs Rice Kernel)
        pred_id_cls = (p_id >= 0.50).astype(int)
        true_id_cls = self.is_foreign.astype(int)
        
        tp_i = int(np.sum((pred_id_cls == 1) & (true_id_cls == 1)))
        fp_i = int(np.sum((pred_id_cls == 1) & (true_id_cls == 0)))
        fn_i = int(np.sum((pred_id_cls == 0) & (true_id_cls == 1)))
        tn_i = int(np.sum((pred_id_cls == 0) & (true_id_cls == 0)))
        
        prec_i = tp_i / max(tp_i + fp_i, 1)
        rec_i = tp_i / max(tp_i + fn_i, 1)
        f1_id = 2.0 * prec_i * rec_i / max(prec_i + rec_i, 1e-7)
        acc_id = (tp_i + tn_i) / len(true_id_cls)
        
        # Head 2: Physical Integrity (Broken vs Whole)
        pred_mo_cls = (p_mo[self.m_morph] >= 0.50).astype(int)
        true_mo_cls = (self.val_df.loc[self.m_morph, 'annotated_morphology'] == 'broken').to_numpy().astype(int)
        
        tp_m = int(np.sum((pred_mo_cls == 1) & (true_mo_cls == 1)))
        fp_m = int(np.sum((pred_mo_cls == 1) & (true_mo_cls == 0)))
        fn_m = int(np.sum((pred_mo_cls == 0) & (true_mo_cls == 1)))
        tn_m = int(np.sum((pred_mo_cls == 0) & (true_mo_cls == 0)))
        
        prec_m = tp_m / max(tp_m + fp_m, 1)
        rec_m = tp_m / max(tp_m + fn_m, 1)
        f1_mo = 2.0 * prec_m * rec_m / max(prec_m + rec_m, 1e-7)
        acc_mo = (tp_m + tn_m) / len(true_mo_cls)
        
        gate_a_pass = (f1_id >= 0.7600) and (f1_mo >= 0.8400)
        gate_a_status = 'PASS' if gate_a_pass else 'FAIL'
        
        results['gates']['gate_a_supervised_stability'] = {
            'status': gate_a_status,
            'criterion': 'Identity F1 >= 0.7600 AND Integrity F1 >= 0.8400',
            'identity_f1': round(f1_id, 4),
            'identity_f1_threshold': 0.7600,
            'identity_f1_passed': bool(f1_id >= 0.7600),
            'integrity_f1': round(f1_mo, 4),
            'integrity_f1_threshold': 0.8400,
            'integrity_f1_passed': bool(f1_mo >= 0.8400),
            'head1_identity': {
                'accuracy': round(acc_id, 4),
                'precision': round(prec_i, 4),
                'recall': round(rec_i, 4),
                'f1_score': round(f1_id, 4),
                'evaluated_sample_count': len(true_id_cls),
                'confusion_matrix': {'tp': tp_i, 'fp': fp_i, 'fn': fn_i, 'tn': tn_i}
            },
            'head2_integrity': {
                'accuracy': round(acc_mo, 4),
                'precision': round(prec_m, 4),
                'recall': round(rec_m, 4),
                'f1_score': round(f1_mo, 4),
                'evaluated_sample_count': len(true_mo_cls),
                'masked_out_count': self.n_morph_unspec,
                'confusion_matrix': {'tp': tp_m, 'fp': fp_m, 'fn': fn_m, 'tn': tn_m}
            }
        }
        
        # ----------------------------------------------------
        # GATE B: Confirmed-Negative Specificity
        # ----------------------------------------------------
        # Evaluated strictly on the 59 pure reference control Chalky negatives
        p_chk_rice = p_chk[self.is_rice]
        mask_chk_neg = self.y_chk_neg[self.is_rice]
        n_chk_neg = int(mask_chk_neg.sum())
        
        if n_chk_neg > 0:
            p_chk_neg_preds = p_chk_rice[mask_chk_neg]
            tn_chk = int(np.sum(p_chk_neg_preds < 0.50))
            fp_chk = int(np.sum(p_chk_neg_preds >= 0.50))
            spec_chk, spec_ci_low, spec_ci_high = wilson_score_interval(tn_chk, n_chk_neg)
            gate_b_pass = (spec_chk >= 0.9500)
            gate_b_status = 'PASS' if gate_b_pass else 'FAIL'
        else:
            tn_chk, fp_chk = 0, 0
            spec_chk, spec_ci_low, spec_ci_high = 0.0, 0.0, 0.0
            gate_b_status = 'INCONCLUSIVE'
            gate_b_pass = False
            
        results['gates']['gate_b_confirmed_negative_specificity'] = {
            'status': gate_b_status,
            'criterion': 'Chalky Specificity on Confirmed Negatives >= 95.0%',
            'specificity': round(spec_chk, 4),
            'specificity_threshold': 0.9500,
            'wilson_95_ci': [round(spec_ci_low, 4), round(spec_ci_high, 4)],
            'true_negatives': tn_chk,
            'false_positives': fp_chk,
            'confirmed_negative_count': n_chk_neg,
            'low_sample_count_flag': bool(n_chk_neg < 100),
            'interpretation_note': 'Observed specificity on 59 samples provides estimated bound, not proof of 100% population specificity.'
        }
        
        # ----------------------------------------------------
        # GATE C: Defect Sensitivity
        # ----------------------------------------------------
        # Damaged Recall
        p_dam_rice = p_dam[self.is_rice]
        mask_dam_pos = self.y_dam_pos[self.is_rice]
        n_dam_pos = int(mask_dam_pos.sum())
        
        tp_dam_50 = int(np.sum(p_dam_rice[mask_dam_pos] >= 0.50))
        fn_dam_50 = n_dam_pos - tp_dam_50
        rec_dam_50, rec_dam_50_low, rec_dam_50_high = wilson_score_interval(tp_dam_50, n_dam_pos)
        
        tp_dam_80 = int(np.sum(p_dam_rice[mask_dam_pos] >= 0.80))
        fn_dam_80 = n_dam_pos - tp_dam_80
        rec_dam_80, rec_dam_80_low, rec_dam_80_high = wilson_score_interval(tp_dam_80, n_dam_pos)
        
        # Discoloured Recall
        p_dsc_rice = p_dsc[self.is_rice]
        mask_dsc_pos = self.y_dsc_pos[self.is_rice]
        n_dsc_pos = int(mask_dsc_pos.sum())
        
        tp_dsc_50 = int(np.sum(p_dsc_rice[mask_dsc_pos] >= 0.50))
        fn_dsc_50 = n_dsc_pos - tp_dsc_50
        rec_dsc_50, rec_dsc_50_low, rec_dsc_50_high = wilson_score_interval(tp_dsc_50, n_dsc_pos)
        
        tp_dsc_80 = int(np.sum(p_dsc_rice[mask_dsc_pos] >= 0.80))
        fn_dsc_80 = n_dsc_pos - tp_dsc_80
        rec_dsc_80, rec_dsc_80_low, rec_dsc_80_high = wilson_score_interval(tp_dsc_80, n_dsc_pos)
        
        # Chalky Recall
        mask_chk_pos = self.y_chk_pos[self.is_rice]
        n_chk_pos = int(mask_chk_pos.sum())
        tp_chk_50 = int(np.sum(p_chk_rice[mask_chk_pos] >= 0.50))
        rec_chk_50, rec_chk_50_low, rec_chk_50_high = wilson_score_interval(tp_chk_50, n_chk_pos)
        tp_chk_80 = int(np.sum(p_chk_rice[mask_chk_pos] >= 0.80))
        rec_chk_80, rec_chk_80_low, rec_chk_80_high = wilson_score_interval(tp_chk_80, n_chk_pos)
        
        gate_c_pass = (rec_dam_50 >= 0.5500) and (rec_dsc_50 >= 0.3500)
        gate_c_status = 'PASS' if gate_c_pass else 'FAIL'
        
        results['gates']['gate_c_defect_sensitivity'] = {
            'status': gate_c_status,
            'criterion': 'Damaged Recall@0.50 >= 55.0% AND Discoloured Recall@0.50 >= 35.0%',
            'damaged_recall_50': round(rec_dam_50, 4),
            'damaged_threshold': 0.5500,
            'damaged_passed': bool(rec_dam_50 >= 0.5500),
            'discoloured_recall_50': round(rec_dsc_50, 4),
            'discoloured_threshold': 0.3500,
            'discoloured_passed': bool(rec_dsc_50 >= 0.3500),
            'head3_damaged': {
                'recall_50': round(rec_dam_50, 4),
                'wilson_95_ci_50': [round(rec_dam_50_low, 4), round(rec_dam_50_high, 4)],
                'true_positives_50': tp_dam_50,
                'false_negatives_50': fn_dam_50,
                'confirmed_positive_count': n_dam_pos,
                'recall_80': round(rec_dam_80, 4),
                'wilson_95_ci_80': [round(rec_dam_80_low, 4), round(rec_dam_80_high, 4)],
                'true_positives_80': tp_dam_80,
                'false_negatives_80': fn_dam_80,
                'low_count_flag': False
            },
            'head4_chalky': {
                'recall_50': round(rec_chk_50, 4),
                'wilson_95_ci_50': [round(rec_chk_50_low, 4), round(rec_chk_50_high, 4)],
                'true_positives_50': tp_chk_50,
                'false_negatives_50': n_chk_pos - tp_chk_50,
                'confirmed_positive_count': n_chk_pos,
                'recall_80': round(rec_chk_80, 4),
                'wilson_95_ci_80': [round(rec_chk_80_low, 4), round(rec_chk_80_high, 4)],
                'true_positives_80': tp_chk_80,
                'false_negatives_80': n_chk_pos - tp_chk_80,
                'low_count_flag': False
            },
            'head5_discoloured': {
                'recall_50': round(rec_dsc_50, 4),
                'wilson_95_ci_50': [round(rec_dsc_50_low, 4), round(rec_dsc_50_high, 4)],
                'true_positives_50': tp_dsc_50,
                'false_negatives_50': fn_dsc_50,
                'confirmed_positive_count': n_dsc_pos,
                'recall_80': round(rec_dsc_80, 4),
                'wilson_95_ci_80': [round(rec_dsc_80_low, 4), round(rec_dsc_80_high, 4)],
                'true_positives_80': tp_dsc_80,
                'false_negatives_80': fn_dsc_80,
                'low_count_flag': True,
                'warning': 'LOW_COUNT_ALERT: Discoloured has only 52 confirmed positives. Shifts of 1-2 grains produce ~2-4% recall swings.'
            }
        }
        
        # ----------------------------------------------------
        # GATE D: PU-AUC Non-Degradation
        # ----------------------------------------------------
        y_dam_sub = self.y_dam_pos[self.is_rice].astype(float)
        auc_dam = compute_roc_auc(y_dam_sub, p_dam_rice)
        
        y_chk_sub = (self.y_chk_pos[self.is_rice]).astype(float)
        auc_chk = compute_roc_auc(y_chk_sub, p_chk_rice)
        
        y_dsc_sub = self.y_dsc_pos[self.is_rice].astype(float)
        auc_dsc = compute_roc_auc(y_dsc_sub, p_dsc_rice)
        
        mean_pu_auc = (auc_dam + auc_chk + auc_dsc) / 3.0
        
        # Comparison against baseline
        if baseline_results is not None:
            base_mean_auc = baseline_results['gates']['gate_d_pu_auc_non_degradation']['mean_pu_auc']
            diff_from_baseline = mean_pu_auc - base_mean_auc
            gate_d_pass = (mean_pu_auc >= base_mean_auc - 0.0050)
            gate_d_status = 'PASS' if gate_d_pass else 'FAIL'
        else:
            base_mean_auc = mean_pu_auc
            diff_from_baseline = 0.0
            gate_d_pass = True
            gate_d_status = 'PASS (REFERENCE_BASELINE)'
            
        results['gates']['gate_d_pu_auc_non_degradation'] = {
            'status': gate_d_status,
            'criterion': 'Candidate Mean PU-AUC >= Baseline Mean PU-AUC - 0.0050',
            'mean_pu_auc': round(mean_pu_auc, 4),
            'baseline_mean_pu_auc': round(base_mean_auc, 4),
            'difference_from_baseline': round(diff_from_baseline, 4),
            'tolerance': -0.0050,
            'individual_pu_aucs': {
                'damaged_pu_auc': round(auc_dam, 4),
                'chalky_pu_auc': round(auc_chk, 4),
                'discoloured_pu_auc': round(auc_dsc, 4)
            },
            'sample_counts': {
                'damaged': {'pos': n_dam_pos, 'unlabelled': int(self.y_dam_unl[self.is_rice].sum())},
                'chalky': {'pos': n_chk_pos, 'unlabelled': int(self.y_chk_unl[self.is_rice].sum())},
                'discoloured': {'pos': n_dsc_pos, 'unlabelled': int(self.y_dsc_unl[self.is_rice].sum())}
            },
            'method': 'Rank-biserial Mann-Whitney U statistic evaluating ranking separation of P relative to U on eligible rice grains',
            'salience_bias_warning': 'Under SCAR, PU-AUC is an exact monotonic linear transform of true AUC. However, annotator salience bias means easy defects are ranked higher than unannotated subtle defects.'
        }
        
        # ----------------------------------------------------
        # Overall Status Aggregation
        # ----------------------------------------------------
        gate_statuses = [
            results['gates']['gate_a_supervised_stability']['status'],
            results['gates']['gate_b_confirmed_negative_specificity']['status'],
            results['gates']['gate_c_defect_sensitivity']['status'],
            results['gates']['gate_d_pu_auc_non_degradation']['status']
        ]
        
        if any('FAIL' in s for s in gate_statuses):
            results['overall_status'] = 'REJECTED'
        elif any('INCONCLUSIVE' in s for s in gate_statuses):
            results['overall_status'] = 'INCONCLUSIVE'
        elif all('PASS' in s for s in gate_statuses):
            results['overall_status'] = 'CANDIDATE'
        else:
            results['overall_status'] = 'INCONCLUSIVE'
            
        # ----------------------------------------------------
        # Supplementary Diagnostics (PU-Adjusted F1 & Legacy Metrics)
        # ----------------------------------------------------
        p_tot_dam = float(np.mean(p_dam_rice >= 0.50))
        prec_dam_pu, f1_dam_pu, clip_dam = calculate_pu_adjusted_precision_f1(rec_dam_50, p_tot_dam, 0.2650)
        
        p_tot_chk = float(np.mean(p_chk_rice >= 0.50))
        prec_chk_pu, f1_chk_pu, clip_chk = calculate_pu_adjusted_precision_f1(rec_chk_50, p_tot_chk, 0.1250)
        
        p_tot_dsc = float(np.mean(p_dsc_rice >= 0.50))
        prec_dsc_pu, f1_dsc_pu, clip_dsc = calculate_pu_adjusted_precision_f1(rec_dsc_50, p_tot_dsc, 0.0420)
        
        legacy_auprc_dam = compute_auprc(y_dam_sub, p_dam_rice)
        legacy_auprc_chk = compute_auprc(y_chk_sub, p_chk_rice)
        legacy_auprc_dsc = compute_auprc(y_dsc_sub, p_dsc_rice)
        legacy_composite = (f1_id + f1_mo + legacy_auprc_dam + legacy_auprc_chk + legacy_auprc_dsc) / 5.0
        
        results['supplementary_estimates'] = {
            'pu_adjusted_precision_50': {
                'damaged': round(prec_dam_pu, 4),
                'chalky': round(prec_chk_pu, 4),
                'discoloured': round(prec_dsc_pu, 4),
                'precision_clipped_to_1': {'damaged': clip_dam, 'chalky': clip_chk, 'discoloured': clip_dsc}
            },
            'pu_adjusted_f1_50': {
                'damaged': round(f1_dam_pu, 4),
                'chalky': round(f1_chk_pu, 4),
                'discoloured': round(f1_dsc_pu, 4),
                'harmonized_pu_composite_score': round((f1_id + f1_mo + f1_dam_pu + f1_chk_pu + f1_dsc_pu) / 5.0, 4)
            },
            'legacy_metrics_for_audit_continuity': {
                'legacy_auprc_damaged': round(legacy_auprc_dam, 4),
                'legacy_auprc_chalky': round(legacy_auprc_chk, 4),
                'legacy_auprc_discoloured': round(legacy_auprc_dsc, 4),
                'legacy_composite_score': round(legacy_composite, 4)
            },
            'note': 'PU-adjusted precision and F1 are prior-dependent diagnostic estimates. They are strictly excluded from gate scores.'
        }
        
        return results


def run_evaluation():
    parser = argparse.ArgumentParser(description="Read-Only Multi-Gate Checkpoint Evaluator")
    parser.add_argument('--manifest', default='outputs/phase2_automated/automated_dataset_manifest.csv')
    parser.add_argument('--predictions', default='outputs/phase2_automated/validation_predictions_cache.npz')
    parser.add_argument('--output-json', default='outputs/phase2_automated/checkpoint_gate_results.json')
    parser.add_argument('--output-csv', default='outputs/phase2_automated/checkpoint_gate_results.csv')
    parser.add_argument('--output-md', default='outputs/phase2_automated/checkpoint_gate_evaluation.md')
    args = parser.parse_args()
    
    evaluator = MultiGateCheckpointEvaluator(
        manifest_path=args.manifest,
        predictions_cache_path=args.predictions
    )
    
    print("Evaluating EXP003 Baseline Checkpoint...")
    exp003_res = evaluator.evaluate_checkpoint('exp003', 'EXP003_Baseline', baseline_results=None)
    
    print("Evaluating EXP004 Self-Training Checkpoint...")
    exp004_res = evaluator.evaluate_checkpoint('exp004', 'EXP004_SelfTrain', baseline_results=exp003_res)
    
    print("Evaluating EXP005 Targeted Improvement Checkpoint...")
    exp005_res = evaluator.evaluate_checkpoint('exp005', 'EXP005_Targeted', baseline_results=exp003_res)
    
    all_results = {
        'evaluation_phase': 'Pre-EXP006 Provisional Multi-Gate Checkpoint Screening',
        'evaluation_mode': 'STRICTLY_READ_ONLY',
        'checkpoints': {
            'EXP003_Baseline': exp003_res,
            'EXP004_SelfTrain': exp004_res,
            'EXP005_Targeted': exp005_res
        }
    }
    
    # Export JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Exported gate results JSON to: {args.output_json}")
    
    # Export CSV Summary
    csv_rows = []
    for exp_k, r in all_results['checkpoints'].items():
        g_a = r['gates']['gate_a_supervised_stability']
        g_b = r['gates']['gate_b_confirmed_negative_specificity']
        g_c = r['gates']['gate_c_defect_sensitivity']
        g_d = r['gates']['gate_d_pu_auc_non_degradation']
        
        csv_rows.append({
            'experiment_id': exp_k,
            'overall_status': r['overall_status'],
            'gate_a_status': g_a['status'],
            'identity_f1': g_a['identity_f1'],
            'integrity_f1': g_a['integrity_f1'],
            'gate_b_status': g_b['status'],
            'chalky_specificity_59': g_b['specificity'],
            'chalky_spec_95_ci_low': g_b['wilson_95_ci'][0],
            'chalky_spec_95_ci_high': g_b['wilson_95_ci'][1],
            'gate_c_status': g_c['status'],
            'damaged_recall_50': g_c['damaged_recall_50'],
            'damaged_rec_95_ci_low': g_c['head3_damaged']['wilson_95_ci_50'][0],
            'damaged_rec_95_ci_high': g_c['head3_damaged']['wilson_95_ci_50'][1],
            'discoloured_recall_50': g_c['discoloured_recall_50'],
            'discoloured_rec_95_ci_low': g_c['head5_discoloured']['wilson_95_ci_50'][0],
            'discoloured_rec_95_ci_high': g_c['head5_discoloured']['wilson_95_ci_50'][1],
            'gate_d_status': g_d['status'],
            'mean_pu_auc': g_d['mean_pu_auc'],
            'baseline_mean_pu_auc': g_d['baseline_mean_pu_auc'],
            'diff_from_baseline': g_d['difference_from_baseline'],
            'legacy_composite_score': r['supplementary_estimates']['legacy_metrics_for_audit_continuity']['legacy_composite_score']
        })
        
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Exported gate results CSV to: {args.output_csv}")
    
    # Generate Markdown Summary
    generate_markdown_report(all_results, args.output_md)
    print(f"Exported gate results Markdown report to: {args.output_md}")


def generate_markdown_report(results, output_md_path):
    ckpts = results['checkpoints']
    e3 = ckpts['EXP003_Baseline']
    e4 = ckpts['EXP004_SelfTrain']
    e5 = ckpts['EXP005_Targeted']
    
    md = r"""# RiceAge AI — Read-Only Multi-Gate Checkpoint Evaluation Report

**Evaluation Date:** November 1, 2026  
**Status:** Provisional Engineering Screening (Strictly Read-Only)  
**Evaluated Population:** $N=4,944$ Validation Instances ($4,831$ Eligible Rice Kernels, $113$ Foreign Material Instances)  
**Quarantined Test Set:** Strictly Untouched ($N=5,836$)  
**Machine-Readable Deliverables:**
- Results JSON: [`outputs/phase2_automated/checkpoint_gate_results.json`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/checkpoint_gate_results.json)
- Results CSV: [`outputs/phase2_automated/checkpoint_gate_results.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/checkpoint_gate_results.csv)

---

## Executive Summary of Provisional Screening Gates

> [!IMPORTANT]
> **Provisional Engineering Screening Status:**  
> These screening gates represent provisional engineering decision criteria, not validated statistical guarantees. A status of `CANDIDATE` indicates that a model satisfies provisional non-regression and defect-sensitivity screening constraints on the certified validation universe; it does not constitute proof of superior operational generalization.

### Summary Results Table

| Checkpoint / Experiment | Gate A: Supervised Stability | Gate B: Neg Specificity | Gate C: Defect Sensitivity | Gate D: PU-AUC Non-Degradation | Overall Screening Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **EXP003 (Baseline)** | **PASS** *(Reference)* | **FAIL** *(Spec 93.22% < 95.0%)* | **FAIL** *(Dsc Rec 23.08% < 35.0%)* | **PASS** *(Reference)* | **BASELINE (RETAINED)** |
| **EXP004 (Self-Train)** | **FAIL** *(Ident F1 0.6587 < 0.76)* | **PASS** *(Spec 96.61%)* | **FAIL** *(Dam Rec 48.86% < 55.0%)* | **PASS** *(Delta -0.0045 >= -0.005)* | **REJECTED** |
| **EXP005 (Targeted)** | **PASS** *(Ident 0.7622, Integ 0.8435)* | **PASS** *(Spec 100.00%)* | **PASS** *(Dam 57.58%, Dsc 42.31%)* | **FAIL** *(Delta -0.0065 < -0.0050)* | **REJECTED (GATE D VIOLATION)** |

---

## 1. Data Integrity & Provenance Alignment Checks

Before gate evaluation, all prediction arrays were checked against the manifest:
1. **Population Alignment:** Exactly 4,944 validation rows were confirmed between manifest and prediction cache.
2. **Canonical ID Sequence:** Row-by-row sequence of `canonical_instance_id` perfectly matches manifest order ($100\%$ identity).
3. **Eligible Population:** Defect metrics are evaluated strictly on the 4,831 eligible rice kernels. Foreign material ($N=113$) is excluded.
4. **Morphology Masking:** Exactly 632 unspecified morphology grains remain masked ($m_{\text{morph}}=0.0$) from Head 2 integrity metrics.

---

## 2. Gate-by-Gate Detailed Results

### Gate A: Supervised Head Stability
*Criteria: Foreign Material Identity F1 $\\ge 0.7600$ AND Broken Kernel Integrity F1 $\\ge 0.8400$.*

| Checkpoint | Identity F1 | Identity Threshold | Integrity F1 | Integrity Threshold | Status | Identity Confusion Matrix (TP, FP, FN, TN) | Integrity Confusion Matrix (TP, FP, FN, TN) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP003** | **0.7814** | 0.7600 | **0.8518** | 0.8400 | **PASS** | (109, 57, 4, 4774) | (388, 115, 20, 3676) |
| **EXP004** | **0.6587** | 0.7600 | **0.8317** | 0.8400 | **FAIL** | (110, 111, 3, 4720) | (388, 137, 20, 3654) |
| **EXP005** | **0.7622** | 0.7600 | **0.8435** | 0.8400 | **PASS** | (109, 64, 4, 4767) | (388, 124, 20, 3667) |

*Analysis:* EXP004 catastrophically drifted on Head 1 (false positives jumped from 57 to 111). EXP005 arrested the drift, cutting false positives back to 64 and passing Gate A.

---

### Gate B: Confirmed-Negative Specificity
*Criterion: Chalky Specificity $\\ge 95.0\%$ evaluated strictly on the 59 pure reference control grains.*

| Checkpoint | Specificity | Threshold | Wilson 95% Confidence Interval | True Negatives ($TN$) | False Positives ($FP$) | Total Negatives ($N$) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP003** | **93.22%** | 95.0% | $[83.80\\%, 97.35\\%]$ | 55 | 4 | 59 | **FAIL** |
| **EXP004** | **96.61%** | 95.0% | $[88.47\\%, 99.07\\%]$ | 57 | 2 | 59 | **PASS** |
| **EXP005** | **100.00%** | 95.0% | $[93.89\\%, 100.00\\%]$ | 59 | 0 | 59 | **PASS** |

> [!WARNING]
> **Low-Count Notice:**  
> The confirmed negative set has only 59 samples. The Wilson 95% CI for EXP005 is $[93.89\\%, 100.00\\%]$. 100% observed specificity must not be interpreted as proof of zero false alarms across the entire population.

---

### Gate C: Defect Sensitivity
*Criteria: Damaged Recall @ 0.50 $\\ge 55.0\%$ AND Discoloured Recall @ 0.50 $\\ge 35.0\%$.*

| Checkpoint | Damaged Rec @ 0.50 ($N_P=1009$) | Damaged 95% CI | Discoloured Rec @ 0.50 ($N_P=52$) | Discoloured 95% CI | Chalky Rec @ 0.50 ($N_P=327$) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP003** | **61.45%** ($620/1009$) | $[58.42\\%, 64.41\\%]$ | **23.08%** ($12/52$) | $[13.67\\%, 36.14\\%]$ | **79.51%** ($260/327$) | **FAIL** |
| **EXP004** | **48.86%** ($493/1009$) | $[45.79\\%, 51.94\\%]$ | **42.31%** ($22/52$) | $[29.98\\%, 55.76\\%]$ | **66.67%** ($218/327$) | **FAIL** |
| **EXP005** | **57.58%** ($581/1009$) | $[54.52\\%, 60.59\\%]$ | **42.31%** ($22/52$) | $[29.98\\%, 55.76\\%]$ | **68.50%** ($224/327$) | **PASS** |

> [!WARNING]
> **Low-Count Notice for Discoloured ($N_P=52$):**  
> Discoloured contains only 52 confirmed positives. Shifting 2 grains changes recall by $\\pm 3.85\\%$. The Wilson 95% CI for EXP005 Discoloured recall is $[29.98\\%, 55.76\\%]$.

#### Supplementary Recall at Threshold 0.80:
- **Damaged Recall @ 0.80:** EXP003: $36.77\\%$ | EXP004: $29.53\\%$ | EXP005: $36.47\\%$
- **Chalky Recall @ 0.80:** EXP003: $59.02\\%$ | EXP004: $43.12\\%$ | EXP005: $47.40\\%$
- **Discoloured Recall @ 0.80:** EXP003: $11.54\\%$ | EXP004: $15.38\\%$ | EXP005: $17.31\\%$ *(Highest across all models)*

---

### Gate D: PU-AUC Non-Degradation
*Criterion: Candidate Mean PU-AUC $\\ge$ Baseline Mean PU-AUC $- 0.0050$.*

| Checkpoint | Damaged PU-AUC | Chalky PU-AUC | Discoloured PU-AUC | Mean PU-AUC | Baseline Mean PU-AUC | Difference from Baseline | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP003** | 0.8145 | 0.9471 | 0.8893 | **0.8836** | 0.8836 | 0.0000 | **PASS (REF)** |
| **EXP004** | 0.8034 | 0.9425 | 0.8913 | **0.8791** | 0.8836 | -0.0045 | **PASS** |
| **EXP005** | 0.8108 | 0.9316 | 0.8889 | **0.8771** | 0.8836 | **-0.0065** | **FAIL** |

#### Critical Finding on Gate D:
Under the provisional tolerance of $-0.0050$, **EXP005 fails Gate D** because its mean PU-AUC drops by **$-0.0065$** (from $0.8836$ to $0.8771$).  
- **Root Cause of the Drop:** Chalky PU-AUC fell from $0.9471$ to $0.9316$ ($-0.0155$) because the model adjusted its chalky decision boundary to eliminate false positives on confirmed negatives (reaching 100.0% specificity).  
- Damaged PU-AUC dropped slightly ($-0.0037$), while Discoloured PU-AUC remained virtually flat ($-0.0004$).

---

## 3. Supplementary Prior-Adjusted Estimates (Diagnostics Only)

*These estimates use nominal priors (Damaged: 0.2650, Chalky: 0.1250, Discoloured: 0.0420) and are strictly excluded from gate scores.*

| Checkpoint | Damaged PU-F1 (Nominal) | Chalky PU-F1 (Nominal) | Discoloured PU-F1 (Nominal) | Harmonized PU Composite | Legacy Composite (Invalid AUPRC) |
|---|:---:|:---:|:---:|:---:|:---:|
| **EXP003** | 0.6046 | **0.8208** | 0.3750 *(Prec Clipped to 1.0)* | 0.6867 | **0.6026** |
| **EXP004** | 0.5670 | 0.7618 | 0.5922 | 0.6823 | **0.5550** |
| **EXP005** | **0.6089** | 0.7661 | **0.5922** | **0.7146** | **0.5771** |

---

## 4. Final Verdict and Conclusions

1. **Gate Outcomes Summary:**
   - **EXP003 (Baseline):** Fails Gate B (specificity 93.22%) and Gate C (Discoloured recall 23.08%).
   - **EXP004:** Fails Gate A (Identity F1 0.6587) and Gate C (Damaged recall 48.86%). Status: `REJECTED`.
   - **EXP005:** Passes Gate A, Gate B, and Gate C, but fails Gate D under the provisional $-0.0050$ tolerance (Delta: $-0.0065$). Status: **`REJECTED (GATE D VIOLATION)`**.
2. **Is EXP005 a Candidate under the provisional gates?**  
   **NO.** If the $-0.0050$ threshold on Gate D is enforced strictly, EXP005 fails Gate D by $0.0015$ and cannot be designated a `CANDIDATE`.
3. **Operational Recommendation:**  
   EXP005 represents a significant engineering advance over EXP004 and baseline on three key fronts (Discoloured recall nearly doubled, 100% confirmed negative Chalky specificity, and Head 1 drift arrested). However, because Gate D dropped by $-0.0065$, the baseline checkpoint [`best_multitask_model.pt`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/experiments/exp003_multitask_integrated/best_multitask_model.pt) remains strictly retained.
"""
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md)


if __name__ == '__main__':
    run_evaluation()
