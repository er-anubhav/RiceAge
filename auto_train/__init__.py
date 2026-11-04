"""
RiceAge AI Automated Model-Training and Self-Improvement Package.
"""
from auto_train.config import load_config, AutoTrainConfig
from auto_train.model import RiceAgeMultiTask5Head, EMATeacher, load_reconciled_checkpoint
from auto_train.losses import NonNegativePULoss, MultiTaskLossContainer
from auto_train.dataset import RiceGrainDataset, get_transforms
from auto_train.pseudo_labeler import PseudoLabeler
from auto_train.evaluate import evaluate_model
from auto_train.data_integrity import audit_dataset_integrity

__all__ = [
    'load_config',
    'AutoTrainConfig',
    'RiceAgeMultiTask5Head',
    'EMATeacher',
    'load_reconciled_checkpoint',
    'NonNegativePULoss',
    'MultiTaskLossContainer',
    'RiceGrainDataset',
    'get_transforms',
    'PseudoLabeler',
    'evaluate_model',
    'audit_dataset_integrity'
]
