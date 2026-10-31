import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class PseudoLabeler:
    """
    Automated Multi-Signal Triangulation Pseudo-Labeling Engine.
    Filters by EMA Teacher prediction confidence and TTA prediction stability.
    Implements quota caps to prevent pseudo-label collapse and confirmation bias.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        priors: Dict[str, float],
        output_dir: str
    ):
        self.config = config
        self.priors = priors
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.thresholds = config['thresholds']
        self.variance_threshold = float(config['variance_threshold'])
        self.max_fraction = float(config['max_pseudo_label_fraction'])
        self.tta_views = int(config.get('tta_views', 3))
        self.allow_pseudo_negatives = bool(config.get('allow_pseudo_negatives', False))

    def generate_pseudo_labels(
        self,
        teacher: nn.Module,
        train_dataset,
        device: torch.device,
        round_idx: int = 1
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        teacher.eval()
        
        # 1. Collect all eligible unlabelled training instances (rice grains only)
        unlabelled_indices = []
        for idx, s in enumerate(train_dataset.samples):
            if not s['is_foreign']:
                # At least one defect is unlabelled
                if s['base_y_dam'] == 0.0 or s['base_y_chk'] == 0.0 or s['base_y_dsc'] == 0.0:
                    unlabelled_indices.append(idx)
                    
        print(f"Generating pseudo-labels for {len(unlabelled_indices)} eligible unlabelled grains (Round {round_idx})...", flush=True)
        
        # TTA view transforms
        tta_transforms = [
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ][:self.tta_views]

        # Evaluate unlabelled instances
        candidate_records = []
        
        # Batch evaluation
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(unlabelled_indices), batch_size):
                b_indices = unlabelled_indices[i:i + batch_size]
                
                # Load images for batch
                b_imgs_pil = []
                b_samples = [train_dataset.samples[idx] for idx in b_indices]
                for s in b_samples:
                    try:
                        img = Image.open(s['path']).convert('RGB')
                    except Exception:
                        img = Image.new('RGB', (128, 128), (0, 0, 0))
                    b_imgs_pil.append(img)
                    
                # Collect TTA predictions
                tta_preds_dam, tta_preds_chk, tta_preds_dsc = [], [], []
                for t_fn in tta_transforms:
                    b_tensors = torch.stack([t_fn(img) for img in b_imgs_pil]).to(device)
                    out = teacher(b_tensors)
                    tta_preds_dam.append(torch.sigmoid(out['damaged']).cpu().numpy())
                    tta_preds_chk.append(torch.sigmoid(out['chalky']).cpu().numpy())
                    tta_preds_dsc.append(torch.sigmoid(out['discoloured']).cpu().numpy())
                    
                # Stack: (V, B)
                p_dam_stack = np.stack(tta_preds_dam, axis=0)
                p_chk_stack = np.stack(tta_preds_chk, axis=0)
                p_dsc_stack = np.stack(tta_preds_dsc, axis=0)
                
                # Means and variances
                mean_dam = np.mean(p_dam_stack, axis=0)
                var_dam = np.var(p_dam_stack, axis=0)
                
                mean_chk = np.mean(p_chk_stack, axis=0)
                var_chk = np.var(p_chk_stack, axis=0)
                
                mean_dsc = np.mean(p_dsc_stack, axis=0)
                var_dsc = np.var(p_dsc_stack, axis=0)
                
                for b_i, s in enumerate(b_samples):
                    candidate_records.append({
                        'cid': s['cid'],
                        'base_damaged': s['base_y_dam'],
                        'base_chalky': s['base_y_chk'],
                        'base_discoloured': s['base_y_dsc'],
                        'p_damaged': mean_dam[b_i],
                        'var_damaged': var_dam[b_i],
                        'p_chalky': mean_chk[b_i],
                        'var_chalky': var_chk[b_i],
                        'p_discoloured': mean_dsc[b_i],
                        'var_discoloured': var_dsc[b_i]
                    })
                    
        cand_df = pd.DataFrame(candidate_records)
        
        # 2. Confidence & Uncertainty Filtering with Quota Caps
        pseudo_label_map: Dict[str, Dict[str, float]] = {}
        summary_stats = {}
        
        for defect in ['damaged', 'chalky', 'discoloured']:
            col_p = f'p_{defect}'
            col_v = f'var_{defect}'
            col_base = f'base_{defect}'
            
            cfg_th = self.thresholds[defect]
            pos_th = float(cfg_th['pos'])
            neg_th = float(cfg_th['neg']) if (cfg_th.get('neg') is not None) else None
            
            # Eligible mask: strictly unlabelled in ground truth
            unl_mask = (cand_df[col_base] == 0.0)
            stable_mask = (cand_df[col_v] <= self.variance_threshold)
            
            # Potential Positives
            pot_pos_mask = unl_mask & stable_mask & (cand_df[col_p] >= pos_th)
            pos_candidates = cand_df[pot_pos_mask].copy()
            
            # Quota Cap: max_fraction * (prior * N_unlabelled)
            quota_cap = int(self.max_fraction * (self.priors[defect] * len(cand_df)))
            
            # Sort by highest confidence and take top up to quota_cap
            if len(pos_candidates) > quota_cap:
                pos_candidates = pos_candidates.sort_values(by=col_p, ascending=False).iloc[:quota_cap]
                
            accepted_pos_cids = set(pos_candidates['cid'])
            
            accepted_neg_cids = set()
            if self.allow_pseudo_negatives and (neg_th is not None):
                # Potential Negatives (conservative negative threshold)
                pot_neg_mask = unl_mask & stable_mask & (cand_df[col_p] <= neg_th)
                neg_candidates = cand_df[pot_neg_mask].copy()
                
                # Negative quota cap (to prevent flooding)
                neg_quota_cap = int(self.max_fraction * (1.0 - self.priors[defect]) * len(cand_df))
                if len(neg_candidates) > neg_quota_cap:
                    neg_candidates = neg_candidates.sort_values(by=col_p, ascending=True).iloc[:neg_quota_cap]
                    
                accepted_neg_cids = set(neg_candidates['cid'])
            
            # Populate map
            for cid in accepted_pos_cids:
                if cid not in pseudo_label_map:
                    pseudo_label_map[cid] = {}
                pseudo_label_map[cid][defect] = 1.0 # Pseudo-positive
                
            for cid in accepted_neg_cids:
                if cid not in pseudo_label_map:
                    pseudo_label_map[cid] = {}
                pseudo_label_map[cid][defect] = -1.0 # Pseudo-negative
                
            summary_stats[defect] = {
                'unlabelled_evaluated': int(unl_mask.sum()),
                'accepted_pseudo_positives': len(accepted_pos_cids),
                'accepted_pseudo_negatives': len(accepted_neg_cids),
                'rejected_high_variance': int((unl_mask & ~stable_mask).sum()),
                'threshold_positive': pos_th,
                'threshold_negative': neg_th
            }
            
        # 3. Export Versioned Pseudo-Label CSV
        rows_to_export = []
        for cid, d_labels in pseudo_label_map.items():
            row = {'canonical_instance_id': cid, 'round': round_idx}
            for k, v in d_labels.items():
                row[f'pseudo_{k}'] = 'POSITIVE' if v == 1.0 else 'NEGATIVE'
            rows_to_export.append(row)
            
        export_csv_path = os.path.join(self.output_dir, f'pseudo_labels_round_{round_idx}.csv')
        pd.DataFrame(rows_to_export).to_csv(export_csv_path, index=False)
        print(f"Exported {len(rows_to_export)} pseudo-labeled grains to {export_csv_path}", flush=True)
        
        return pseudo_label_map, summary_stats
