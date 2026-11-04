import copy
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Optional, Any

class RiceAgeMultiTask5Head(nn.Module):
    """
    Decoupled 5-head multi-task architecture for RiceAge AI grain analyzer.
    Fully reconciled with the uploaded checkpoint state dictionary:
      - backbone: ResNet-50 truncated before final pooling (2048-dim output)
      - head_identity: Linear(2048 -> 1)
      - head_integrity: Linear(2048 -> 1)
      - head_damaged: MLP(2048 -> 256 -> 1)
      - head_chalky: MLP(2048 -> 256 -> 1)
      - head_discoloured: MLP(2048 -> 128 -> 1)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # Output: (B, 2048, 1, 1)
        
        # Head 1: Object Identity (Foreign Material vs Rice Kernel)
        self.head_identity = nn.Linear(2048, 1)
        
        # Head 2: Physical Integrity (Broken vs Whole)
        self.head_integrity = nn.Linear(2048, 1)
        
        # Head 3: Damaged Defect Head
        self.head_damaged = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Head 4: Chalky Defect Head
        self.head_chalky = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Head 5: Discoloured Defect Head
        self.head_discoloured = nn.Sequential(
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        return {
            'identity': self.head_identity(feat).squeeze(1),
            'integrity': self.head_integrity(feat).squeeze(1),
            'damaged': self.head_damaged(feat).squeeze(1),
            'chalky': self.head_chalky(feat).squeeze(1),
            'discoloured': self.head_discoloured(feat).squeeze(1),
            'features': feat
        }


class EMATeacher:
    """
    Exponential Moving Average (EMA) Teacher Model wrapper.
    Maintains a shadow copy of the model parameters:
      theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
    def update(self, student: nn.Module):
        with torch.no_grad():
            for p_t, p_s in zip(self.model.parameters(), student.parameters()):
                p_t.data.mul_(self.decay).add_(p_s.data, alpha=1.0 - self.decay)
            for b_t, b_s in zip(self.model.buffers(), student.buffers()):
                b_t.data.copy_(b_s.data)
                
    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.model(x)

    def to(self, device: torch.device):
        self.model.to(device)
        return self


def load_reconciled_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device, strict: bool = True) -> Dict[str, Any]:
    """
    Explicitly reconciles and loads the model checkpoint.
    Verifies state dict keys against model architecture.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict):
        if 'teacher_state' in state_dict:
            state_dict = state_dict['teacher_state']
        elif 'student_state' in state_dict:
            state_dict = state_dict['student_state']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Check key reconciliation
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys
    
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        if strict:
            raise RuntimeError(
                f"Architecture reconciliation failure!\n"
                f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}\n"
                f"Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}"
            )
            
    model.load_state_dict(state_dict, strict=strict)
    return {
        'status': 'RECONCILED_SUCCESS',
        'checkpoint_path': checkpoint_path,
        'missing_keys_count': len(missing_keys),
        'unexpected_keys_count': len(unexpected_keys)
    }
