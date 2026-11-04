import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos_mask = (y_true == 1.0)
    n_pos = np.sum(pos_mask)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1
    u_stat = np.sum(ranks[pos_mask]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = np.sum(y_true == 1.0)
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y == 1.0)
    ranks = np.arange(1, len(sorted_y) + 1)
    precisions = cum_pos / ranks
    return float(np.sum(precisions[sorted_y == 1.0]) / n_pos)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model strictly against the certified validation protocol.
    Never treats unlabelled validation instances as confirmed negatives for binary metrics.
    """
    model.eval()
    
    all_pred_id, all_true_id = [], []
    all_pred_mo, all_true_mo, all_mask_mo = [], [], []
    all_pred_dam, all_true_dam = [], []
    all_pred_chk, all_true_chk = [], []
    all_pred_dsc, all_true_dsc = [], []
    all_mask_grain = []
    all_scans = []
    
    with torch.no_grad():
        for imgs, targets, meta in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            
            all_pred_id.extend(torch.sigmoid(preds['identity']).cpu().numpy())
            all_true_id.extend(targets['y_ident'].numpy())
            
            all_pred_mo.extend(torch.sigmoid(preds['integrity']).cpu().numpy())
            all_true_mo.extend(targets['y_morph'].numpy())
            all_mask_mo.extend(targets['m_morph'].numpy())
            
            all_pred_dam.extend(torch.sigmoid(preds['damaged']).cpu().numpy())
            all_true_dam.extend(targets['y_dam'].numpy())
            
            all_pred_chk.extend(torch.sigmoid(preds['chalky']).cpu().numpy())
            all_true_chk.extend(targets['y_chk'].numpy())
            
            all_pred_dsc.extend(torch.sigmoid(preds['discoloured']).cpu().numpy())
            all_true_dsc.extend(targets['y_dsc'].numpy())
            
            all_mask_grain.extend(targets['m_grain'].numpy())
            all_scans.extend(meta['source_scan'])
            
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
    all_scans = np.array(all_scans)
    
    # ---------------- Head 1: Object Identity ----------------
    p_id_cls = (all_pred_id >= 0.50).astype(int)
    y_id_cls = all_true_id.astype(int)
    tp_i = int(np.sum((p_id_cls == 1) & (y_id_cls == 1)))
    fp_i = int(np.sum((p_id_cls == 1) & (y_id_cls == 0)))
    fn_i = int(np.sum((p_id_cls == 0) & (y_id_cls == 1)))
    tn_i = int(np.sum((p_id_cls == 0) & (y_id_cls == 0)))
    
    prec_id = tp_i / max(tp_i + fp_i, 1)
    rec_id = tp_i / max(tp_i + fn_i, 1)
    f1_id = 2 * prec_id * rec_id / max(prec_id + rec_id, 1e-7)
    acc_id = (tp_i + tn_i) / len(y_id_cls)
    
    # ---------------- Head 2: Physical Integrity ----------------
    p_mo_cls = (all_pred_mo[all_mask_mo] >= 0.50).astype(int)
    y_mo_cls = all_true_mo[all_mask_mo].astype(int)
    tp_m = int(np.sum((p_mo_cls == 1) & (y_mo_cls == 1)))
    fp_m = int(np.sum((p_mo_cls == 1) & (y_mo_cls == 0)))
    fn_m = int(np.sum((p_mo_cls == 0) & (y_mo_cls == 1)))
    tn_m = int(np.sum((p_mo_cls == 0) & (y_mo_cls == 0)))
    
    prec_mo = tp_m / max(tp_m + fp_m, 1)
    rec_mo = tp_m / max(tp_m + fn_m, 1)
    f1_mo = 2 * prec_mo * rec_mo / max(prec_mo + rec_mo, 1e-7)
    acc_mo = (tp_m + tn_m) / max(len(y_mo_cls), 1)
    
    # ---------------- Heads 3-5: Defect Heads (Rice Grains Only) ----------------
    # Damaged
    p_dam_gr = all_pred_dam[all_mask_grain]
    y_dam_gr = all_true_dam[all_mask_grain]
    pos_dam = (y_dam_gr == 1.0)
    unl_dam = (y_dam_gr == 0.0)
    
    rec_dam_50 = float(np.mean(p_dam_gr[pos_dam] >= 0.50)) if pos_dam.any() else 0.0
    rec_dam_80 = float(np.mean(p_dam_gr[pos_dam] >= 0.80)) if pos_dam.any() else 0.0
    auprc_dam = compute_auprc(y_dam_gr, p_dam_gr)
    auc_dam = compute_roc_auc(y_dam_gr, p_dam_gr)
    
    unl_p_dam = p_dam_gr[unl_dam]
    moments_dam = {
        'mean': float(np.mean(unl_p_dam)),
        'std': float(np.std(unl_p_dam)),
        'median': float(np.median(unl_p_dam)),
        'p90': float(np.percentile(unl_p_dam, 90))
    }
    
    # Chalky
    p_chk_gr = all_pred_chk[all_mask_grain]
    y_chk_gr = all_true_chk[all_mask_grain]
    pos_chk = (y_chk_gr == 1.0)
    neg_chk = (y_chk_gr == -1.0)
    unl_chk = (y_chk_gr == 0.0)
    
    rec_chk_50 = float(np.mean(p_chk_gr[pos_chk] >= 0.50)) if pos_chk.any() else 0.0
    rec_chk_80 = float(np.mean(p_chk_gr[pos_chk] >= 0.80)) if pos_chk.any() else 0.0
    spec_chk_neg = float(np.mean(p_chk_gr[neg_chk] < 0.50)) if neg_chk.any() else None
    auprc_chk = compute_auprc((y_chk_gr == 1.0).astype(float), p_chk_gr)
    auc_chk = compute_roc_auc((y_chk_gr == 1.0).astype(float), p_chk_gr)
    
    unl_p_chk = p_chk_gr[unl_chk]
    moments_chk = {
        'mean': float(np.mean(unl_p_chk)),
        'std': float(np.std(unl_p_chk)),
        'median': float(np.median(unl_p_chk)),
        'p90': float(np.percentile(unl_p_chk, 90))
    }
    
    # Discoloured
    p_dsc_gr = all_pred_dsc[all_mask_grain]
    y_dsc_gr = all_true_dsc[all_mask_grain]
    pos_dsc = (y_dsc_gr == 1.0)
    unl_dsc = (y_dsc_gr == 0.0)
    
    rec_dsc_50 = float(np.mean(p_dsc_gr[pos_dsc] >= 0.50)) if pos_dsc.any() else 0.0
    rec_dsc_80 = float(np.mean(p_dsc_gr[pos_dsc] >= 0.80)) if pos_dsc.any() else 0.0
    auprc_dsc = compute_auprc(y_dsc_gr, p_dsc_gr)
    auc_dsc = compute_roc_auc(y_dsc_gr, p_dsc_gr)
    
    unl_p_dsc = p_dsc_gr[unl_dsc]
    moments_dsc = {
        'mean': float(np.mean(unl_p_dsc)),
        'std': float(np.std(unl_p_dsc)),
        'median': float(np.median(unl_p_dsc)),
        'p90': float(np.percentile(unl_p_dsc, 90))
    }
    
    # ---------------- Scan-Level Variability (Across 32 Val Scans) ----------------
    grain_scans = all_scans[all_mask_grain]
    unique_scans = np.unique(grain_scans)
    
    scan_dam_rates, scan_chk_rates, scan_dsc_rates = [], [], []
    for sc in unique_scans:
        sc_m = (grain_scans == sc)
        scan_dam_rates.append(np.mean(p_dam_gr[sc_m] >= 0.50))
        scan_chk_rates.append(np.mean(p_chk_gr[sc_m] >= 0.50))
        scan_dsc_rates.append(np.mean(p_dsc_gr[sc_m] >= 0.50))
        
    def get_cv(arr):
        arr = np.array(arr)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        return float(s / (m + 1e-7))
        
    cv_dam = get_cv(scan_dam_rates)
    cv_chk = get_cv(scan_chk_rates)
    cv_dsc = get_cv(scan_dsc_rates)
    mean_cv = (cv_dam + cv_chk + cv_dsc) / 3.0
    
    # Composite Score
    composite_score = (f1_id + f1_mo + auprc_dam + auprc_chk + auprc_dsc) / 5.0
    
    return {
        'composite_score': round(float(composite_score), 4),
        'head1_identity': {
            'accuracy': round(float(acc_id), 4),
            'precision': round(float(prec_id), 4),
            'recall': round(float(rec_id), 4),
            'f1_score': round(float(f1_id), 4),
            'cm': {'tp': tp_i, 'fp': fp_i, 'fn': fn_i, 'tn': tn_i}
        },
        'head2_integrity': {
            'accuracy': round(float(acc_mo), 4),
            'precision': round(float(prec_mo), 4),
            'recall': round(float(rec_mo), 4),
            'f1_score': round(float(f1_mo), 4),
            'cm': {'tp': tp_m, 'fp': fp_m, 'fn': fn_m, 'tn': tn_m}
        },
        'head3_damaged': {
            'recall_at_0_50': round(float(rec_dam_50), 4),
            'recall_at_0_80': round(float(rec_dam_80), 4),
            'auprc': round(float(auprc_dam), 4),
            'roc_auc': round(float(auc_dam), 4),
            'unlabelled_moments': {k: round(v, 4) for k, v in moments_dam.items()},
            'scan_cv': round(float(cv_dam), 4)
        },
        'head4_chalky': {
            'recall_at_0_50': round(float(rec_chk_50), 4),
            'recall_at_0_80': round(float(rec_chk_80), 4),
            'confirmed_negative_specificity': round(float(spec_chk_neg), 4) if spec_chk_neg is not None else None,
            'auprc': round(float(auprc_chk), 4),
            'roc_auc': round(float(auc_chk), 4),
            'unlabelled_moments': {k: round(v, 4) for k, v in moments_chk.items()},
            'scan_cv': round(float(cv_chk), 4)
        },
        'head5_discoloured': {
            'recall_at_0_50': round(float(rec_dsc_50), 4),
            'recall_at_0_80': round(float(rec_dsc_80), 4),
            'auprc': round(float(auprc_dsc), 4),
            'roc_auc': round(float(auc_dsc), 4),
            'unlabelled_moments': {k: round(v, 4) for k, v in moments_dsc.items()},
            'scan_cv': round(float(cv_dsc), 4)
        },
        'scan_variability': {
            'num_scans': len(unique_scans),
            'damaged_scan_cv': round(float(cv_dam), 4),
            'chalky_scan_cv': round(float(cv_chk), 4),
            'discoloured_scan_cv': round(float(cv_dsc), 4),
            'mean_scan_cv': round(float(mean_cv), 4)
        }
    }
