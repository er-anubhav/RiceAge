import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class NonNegativePULoss(nn.Module):
    """
    Non-Negative Positive-Unlabelled (nnPU) Loss (du Plessis et al., 2015).
    Guarantees that an unlabelled-only batch is NEVER treated as confirmed negative evidence.
    
    Labels encoding:
      1.0 : Confirmed Positive (or high-confidence pseudo-positive)
     -1.0 : Confirmed Negative (e.g. Chalky confirmed negatives in train)
      0.0 : Unlabelled / Unknown
    """
    def __init__(self, prior: float, beta: float = 0.0):
        super().__init__()
        self.prior = float(prior)
        self.beta = float(beta)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (B,), labels: (B,)
        probs = torch.sigmoid(logits)
        
        pos_mask = (labels == 1.0)
        neg_mask = (labels == -1.0)
        unl_mask = (labels == 0.0)
        
        loss_components = []
        
        # 1. Confirmed Negatives (if present, direct binary cross-entropy on negative state)
        if neg_mask.any():
            loss_confirmed_neg = -torch.log(1.0 - probs[neg_mask] + 1e-7).mean()
            loss_components.append(loss_confirmed_neg)
            
        # 2. Non-Negative PU Risk on Positive and Unlabelled sets
        if not pos_mask.any():
            # CRITICAL SAFEGUARD:
            # An unlabelled-only batch MUST NOT be treated as confirmed negative evidence!
            # If no positive anchor exists in the batch, do not penalize unlabelled instances as negative.
            if len(loss_components) > 0:
                return sum(loss_components)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        loss_p_pos = -torch.log(probs[pos_mask] + 1e-7).mean()
        loss_p_neg = -torch.log(1.0 - probs[pos_mask] + 1e-7).mean()
        
        if not unl_mask.any():
            # Batch has positives but no unlabelled instances
            pu_loss = self.prior * loss_p_pos
        else:
            loss_u_neg = -torch.log(1.0 - probs[unl_mask] + 1e-7).mean()
            neg_risk = loss_u_neg - self.prior * loss_p_neg
            
            if neg_risk >= 0:
                pu_loss = self.prior * loss_p_pos + neg_risk
            else:
                # Corrective slack penalty when negative risk becomes sub-zero
                pu_loss = -self.beta * neg_risk
                
        loss_components.append(pu_loss)
        return sum(loss_components)


class MultiTaskLossContainer(nn.Module):
    """
    Container managing joint multi-task loss with dynamic independent supervision masking.
    """
    def __init__(
        self,
        priors: Dict[str, float],
        identity_pos_weight: float = 46.6,
        integrity_pos_weight: float = 8.85,
        beta_slack: float = 0.0
    ):
        super().__init__()
        self.register_buffer('pos_weight_ident', torch.tensor([identity_pos_weight]))
        self.register_buffer('pos_weight_morph', torch.tensor([integrity_pos_weight]))
        
        self.loss_pu_dam = NonNegativePULoss(prior=priors['damaged'], beta=beta_slack)
        self.loss_pu_chk = NonNegativePULoss(prior=priors['chalky'], beta=beta_slack)
        self.loss_pu_dsc = NonNegativePULoss(prior=priors['discoloured'], beta=beta_slack)
        
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Head 1: Object Identity (Always unmasked)
        loss_id_raw = F.binary_cross_entropy_with_logits(
            preds['identity'], targets['y_ident'],
            pos_weight=self.pos_weight_ident, reduction='none'
        )
        m_id = targets['m_ident']
        loss_id = (loss_id_raw * m_id).sum() / (m_id.sum() + 1e-7)
        
        # Head 2: Physical Integrity (Masked for foreign matter and unspecified morphology)
        loss_mo_raw = F.binary_cross_entropy_with_logits(
            preds['integrity'], targets['y_morph'],
            pos_weight=self.pos_weight_morph, reduction='none'
        )
        m_mo = targets['m_morph']
        loss_mo = (loss_mo_raw * m_mo).sum() / (m_mo.sum() + 1e-7) if m_mo.any() else torch.tensor(0.0, device=m_mo.device)
        
        # Heads 3-5: Defect heads on valid grain mask
        m_gr = targets['m_grain']
        grain_mask = (m_gr == 1.0)
        
        if grain_mask.any():
            loss_dam = self.loss_pu_dam(preds['damaged'][grain_mask], targets['y_dam'][grain_mask])
            loss_chk = self.loss_pu_chk(preds['chalky'][grain_mask], targets['y_chk'][grain_mask])
            loss_dsc = self.loss_pu_dsc(preds['discoloured'][grain_mask], targets['y_dsc'][grain_mask])
        else:
            loss_dam = torch.tensor(0.0, device=m_gr.device)
            loss_chk = torch.tensor(0.0, device=m_gr.device)
            loss_dsc = torch.tensor(0.0, device=m_gr.device)
            
        total_loss = loss_id + loss_mo + loss_dam + loss_chk + loss_dsc
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_identity': loss_id.item(),
            'loss_integrity': loss_mo.item(),
            'loss_damaged': loss_dam.item(),
            'loss_chalky': loss_chk.item(),
            'loss_discoloured': loss_dsc.item()
        }
        
        return total_loss, loss_dict
