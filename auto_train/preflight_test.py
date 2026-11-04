import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from auto_train.config import load_config
from auto_train.model import RiceAgeMultiTask5Head, EMATeacher, load_reconciled_checkpoint
from auto_train.losses import NonNegativePULoss, MultiTaskLossContainer
from auto_train.dataset import RiceGrainDataset, get_transforms
from auto_train.evaluate import compute_roc_auc, compute_auprc
from auto_train.data_integrity import audit_dataset_integrity
import pandas as pd

def test_pipeline():
    print("=== Running AutoTrain Pre-Flight Test Suite ===", flush=True)
    cfg = load_config("auto_train/config.json")
    print("1. Config loaded successfully:", cfg.experiment_name)
    
    # 2. Test Model Reconciled Loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RiceAgeMultiTask5Head(pretrained=False).to(device)
    ckpt_path = cfg.resolve_path(cfg.paths.base_checkpoint_path)
    res = load_reconciled_checkpoint(model, ckpt_path, device=device, strict=True)
    print("2. Model Reconciled Checkpoint Loaded:", res['status'])
    
    # 3. Test EMA Teacher
    ema = EMATeacher(model, decay=0.999).to(device)
    x = torch.randn(2, 3, 128, 128).to(device)
    out_s = model(x)
    out_t = ema(x)
    assert 'identity' in out_t and 'damaged' in out_t
    print("3. EMA Teacher forward pass verified: out shapes =", {k: v.shape for k, v in out_t.items() if k != 'features'})
    
    # 4. Test NonNegativePULoss Guardrail (Unlabelled-only batch must NOT be penalized as confirmed negative)
    pu_loss = NonNegativePULoss(prior=0.265, beta=0.0)
    dummy_logits = torch.tensor([1.2, -0.5, 0.3]).to(device)
    unlabelled_labels = torch.tensor([0.0, 0.0, 0.0]).to(device) # No positives!
    loss_unl = pu_loss(dummy_logits, unlabelled_labels)
    assert loss_unl.item() == 0.0, f"Expected 0.0 loss for unlabelled-only batch, got {loss_unl.item()}"
    print("4. nnPU Loss unlabelled-only safeguard verified: loss =", loss_unl.item())
    
    # 5. Test Metric Functions
    auc = compute_roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    auprc = compute_auprc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert auc == 1.0
    assert auprc == 1.0
    print("5. Metrics computation verified: AUC =", auc, "AUPRC =", auprc)
    
    print("\n>>> ALL PRE-FLIGHT TESTS PASSED! System is 100% ready for training! <<<")

if __name__ == '__main__':
    test_pipeline()
