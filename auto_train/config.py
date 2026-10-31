import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class PathConfig:
    project_root: str
    crop_root: str
    manifest_path: str
    crosswalk_path: str
    base_checkpoint_path: str
    output_dir: str

@dataclass
class ModelConfig:
    backbone: str
    num_heads: int
    input_size: List[int]
    mean: List[float]
    std: List[float]

@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    ema_decay: float
    num_workers: int
    pin_memory: bool
    pseudo_label_refresh_epochs: int
    freeze_identity_integrity: bool = True
    backbone_lr: Optional[float] = None
    heads_lr: Optional[float] = None

@dataclass
class AutoTrainConfig:
    experiment_name: str
    description: str
    paths: PathConfig
    model: ModelConfig
    training: TrainingConfig
    priors: Dict[str, float]
    beta_slack: float
    pseudo_labeling: Dict[str, Any]
    supervised_weights: Dict[str, float]
    validation: Dict[str, Any]
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.normpath(os.path.join(self.paths.project_root, path_str))

def load_config(config_path: str = "auto_train/config.json") -> AutoTrainConfig:
    if not os.path.exists(config_path):
        # Fallback to local file relative to this script
        alt_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    p_cfg = PathConfig(
        project_root=data["paths"]["project_root"],
        crop_root=data["paths"]["crop_root"],
        manifest_path=data["paths"]["manifest_path"],
        crosswalk_path=data["paths"]["crosswalk_path"],
        base_checkpoint_path=data["paths"]["base_checkpoint_path"],
        output_dir=data["paths"]["output_dir"]
    )

    m_cfg = ModelConfig(
        backbone=data["model"]["backbone"],
        num_heads=data["model"]["num_heads"],
        input_size=data["model"]["input_size"],
        mean=data["model"]["normalization"]["mean"],
        std=data["model"]["normalization"]["std"]
    )

    t_cfg = TrainingConfig(
        seed=data["training"]["seed"],
        batch_size=data["training"]["batch_size"],
        learning_rate=data["training"]["learning_rate"],
        weight_decay=data["training"]["weight_decay"],
        epochs=data["training"]["epochs"],
        ema_decay=data["training"]["ema_decay"],
        num_workers=data["training"]["num_workers"],
        pin_memory=data["training"]["pin_memory"],
        pseudo_label_refresh_epochs=data["training"]["pseudo_label_refresh_epochs"],
        freeze_identity_integrity=data["training"].get("freeze_identity_integrity", True),
        backbone_lr=data["training"].get("backbone_lr", None),
        heads_lr=data["training"].get("heads_lr", None)
    )

    return AutoTrainConfig(
        experiment_name=data["experiment_name"],
        description=data["description"],
        paths=p_cfg,
        model=m_cfg,
        training=t_cfg,
        priors=data["pu_learning"]["priors"],
        beta_slack=data["pu_learning"]["beta_slack"],
        pseudo_labeling=data["pseudo_labeling"],
        supervised_weights=data["supervised_weights"],
        validation=data["validation"],
        raw_dict=data
    )
