import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

def get_transforms(input_size=(128, 128), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    weak_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    strong_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return weak_transform, strong_transform, val_transform


class RiceGrainDataset(Dataset):
    """
    Unified Rice Grain Dataset mapping canonical instances to physical crop files.
    Supports dynamic overlay of EMA Teacher pseudo-labels while preserving ground truth.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        primaries: pd.DataFrame,
        crop_root: str,
        transform=None,
        is_training: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.primaries = primaries
        self.crop_root = crop_root
        self.transform = transform
        self.is_training = is_training
        
        # In-memory pseudo-label overlay map: cid -> {defect: label_val}
        self.pseudo_labels: Dict[str, Dict[str, float]] = {}
        
        # Precompute sample metadata and base ground-truth targets
        self.samples = []
        for idx, r in self.df.iterrows():
            cid = r['canonical_instance_id']
            p_row = self.primaries.loc[cid]
            path = os.path.normpath(os.path.join(
                self.crop_root,
                p_row['dataset_split'],
                p_row['original_label'],
                p_row['crop_filename']
            ))
            
            is_foreign = (r['canonical_object_type'] == 'foreign_material')
            
            # Head 1: Object Identity (1 = foreign_material, 0 = rice_kernel)
            y_ident = 1.0 if is_foreign else 0.0
            m_ident = 1.0 # Always unmasked
            
            # Head 2: Physical Integrity (1 = broken, 0 = whole)
            morph = r['annotated_morphology']
            if is_foreign or morph not in ['whole', 'broken']:
                y_morph = 0.0
                m_morph = 0.0 # Masked out
            elif morph == 'broken':
                y_morph = 1.0
                m_morph = 1.0
            else:
                y_morph = 0.0
                m_morph = 1.0
                
            # Heads 3-5: Defect supervision base states
            # m_grain = 0.0 for foreign material, 1.0 for valid rice kernels
            m_grain = 0.0 if is_foreign else 1.0
            
            # Damaged base label
            y_dam = 1.0 if (not is_foreign and r['is_damaged_supervision_state'] == 'POSITIVE') else 0.0
            
            # Chalky base label (Chalky has confirmed negatives in train/val!)
            if is_foreign:
                y_chk = 0.0
            elif r['is_chalky_supervision_state'] == 'POSITIVE':
                y_chk = 1.0
            elif r['is_chalky_supervision_state'] == 'NEGATIVE':
                y_chk = -1.0 # Explicit confirmed negative
            else:
                y_chk = 0.0
                
            # Discoloured base label
            y_dsc = 1.0 if (not is_foreign and r['is_discoloured_supervision_state'] == 'POSITIVE') else 0.0
            
            self.samples.append({
                'cid': cid,
                'path': path,
                'source_scan': r.get('source_scan', 'unknown'),
                'is_foreign': is_foreign,
                'y_ident': y_ident,
                'm_ident': m_ident,
                'y_morph': y_morph,
                'm_morph': m_morph,
                'm_grain': m_grain,
                'base_y_dam': y_dam,
                'base_y_chk': y_chk,
                'base_y_dsc': y_dsc
            })

    def update_pseudo_labels(self, pseudo_label_map: Dict[str, Dict[str, float]]):
        """
        Dynamically updates pseudo-labels for eligible unlabelled instances.
        Confirmed ground truth labels are strictly preserved and never overwritten.
        """
        self.pseudo_labels = pseudo_label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cid = s['cid']
        
        # Load image safely
        try:
            img = Image.open(s['path']).convert('RGB')
        except Exception as e:
            # Fallback to zero tensor image if corrupted
            img = Image.new('RGB', (128, 128), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        # Retrieve target labels, overlaying pseudo-labels if unlabelled
        y_dam = s['base_y_dam']
        y_chk = s['base_y_chk']
        y_dsc = s['base_y_dsc']
        
        if cid in self.pseudo_labels:
            pl = self.pseudo_labels[cid]
            # Only overlay if base ground truth is unlabelled (0.0)
            if y_dam == 0.0 and 'damaged' in pl:
                y_dam = pl['damaged']
            if y_chk == 0.0 and 'chalky' in pl:
                y_chk = pl['chalky']
            if y_dsc == 0.0 and 'discoloured' in pl:
                y_dsc = pl['discoloured']
                
        targets = {
            'y_ident': torch.tensor(s['y_ident'], dtype=torch.float32),
            'm_ident': torch.tensor(s['m_ident'], dtype=torch.float32),
            'y_morph': torch.tensor(s['y_morph'], dtype=torch.float32),
            'm_morph': torch.tensor(s['m_morph'], dtype=torch.float32),
            'm_grain': torch.tensor(s['m_grain'], dtype=torch.float32),
            'y_dam': torch.tensor(y_dam, dtype=torch.float32),
            'y_chk': torch.tensor(y_chk, dtype=torch.float32),
            'y_dsc': torch.tensor(y_dsc, dtype=torch.float32)
        }
        
        meta = {
            'canonical_instance_id': cid,
            'source_scan': s['source_scan'],
            'is_foreign': s['is_foreign']
        }
        
        return img, targets, meta
