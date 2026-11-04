import os
import sys
import time
import json
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Ensure auto_train package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from auto_train.config import load_config
from auto_train.data_integrity import audit_dataset_integrity
from auto_train.dataset import get_transforms, RiceGrainDataset
from auto_train.model import RiceAgeMultiTask5Head, EMATeacher, load_reconciled_checkpoint
from auto_train.losses import MultiTaskLossContainer
from auto_train.pseudo_labeler import PseudoLabeler
from auto_train.evaluate import evaluate_model

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_auto_train(config_path: str = "auto_train/config.json", resume_checkpoint: str = None):
    print("=" * 70, flush=True)
    print("  RICEAGE AI — AUTOMATED MULTI-TASK SELF-TRAINING PIPELINE", flush=True)
    print("  EMA Teacher-Student Pseudo-Labeling & Non-Negative PU Learning", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load Configuration
    cfg = load_config(config_path)
    set_seed(cfg.training.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    
    output_dir = cfg.resolve_path(cfg.paths.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Comprehensive Data Mapping & Integrity Audit
    audit_report = audit_dataset_integrity(
        manifest_path=cfg.resolve_path(cfg.paths.manifest_path),
        crosswalk_path=cfg.resolve_path(cfg.paths.crosswalk_path),
        crop_root=cfg.resolve_path(cfg.paths.crop_root),
        output_dir=output_dir
    )
    if audit_report['integrity_status'] != 'CERTIFIED_VERIFIED':
        raise RuntimeError(f"Data integrity audit failed: {audit_report}")
        
    # 3. Load DataFrames
    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_path))
    crosswalk = pd.read_csv(cfg.resolve_path(cfg.paths.crosswalk_path))
    primaries = crosswalk[crosswalk['final_disposition'] == 'canonical_primary'].drop_duplicates('canonical_instance_id').set_index('canonical_instance_id')
    
    train_df = manifest[manifest['pu_learning_split_role'] == 'TRAIN_FITTING_ELIGIBLE'].copy()
    val_df = manifest[manifest['pu_learning_split_role'] == 'VAL_TUNING_RESERVED'].copy()
    
    print(f"Active Datasets: Train={len(train_df)} canonical instances | Val={len(val_df)} canonical instances", flush=True)
    
    # 4. Transforms and Datasets
    weak_tf, strong_tf, val_tf = get_transforms(
        input_size=tuple(cfg.model.input_size),
        mean=cfg.model.mean,
        std=cfg.model.std
    )
    
    crop_root = cfg.resolve_path(cfg.paths.crop_root)
    train_dataset = RiceGrainDataset(train_df, primaries, crop_root=crop_root, transform=strong_tf, is_training=True)
    val_dataset = RiceGrainDataset(val_df, primaries, crop_root=crop_root, transform=val_tf, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )
    
    # 5. Initialize Model Architecture & Load Reconciled Base Checkpoint
    student_model = RiceAgeMultiTask5Head(pretrained=False).to(device)
    
    base_ckpt_path = resume_checkpoint if resume_checkpoint else cfg.resolve_path(cfg.paths.base_checkpoint_path)
    print(f"\nExplicitly loading & reconciling checkpoint: {base_ckpt_path}...", flush=True)
    reconcile_info = load_reconciled_checkpoint(student_model, base_ckpt_path, device=device, strict=True)
    print("Checkpoint architecture successfully reconciled with strict=True! Zero mismatched keys.", flush=True)
    
    if cfg.training.freeze_identity_integrity:
        print("\n[Safe Feature Protection] Freezing Head 1 (Identity) and Head 2 (Integrity) parameters.", flush=True)
        for param in student_model.head_identity.parameters():
            param.requires_grad = False
        for param in student_model.head_integrity.parameters():
            param.requires_grad = False

    # 6. Initialize EMA Teacher
    ema_teacher = EMATeacher(student_model, decay=cfg.training.ema_decay).to(device)
    
    # 7. Establish Baseline Performance on Validation Set
    print("\n--- Evaluating Baseline Performance (Before Self-Training) ---", flush=True)
    t0_eval = time.time()
    baseline_metrics = evaluate_model(student_model, val_loader, device=device)
    print(f"Baseline Composite Validation Score: {baseline_metrics['composite_score']:.4f} (Evaluated in {time.time()-t0_eval:.1f}s)", flush=True)
    print(f"  Head 1 (Identity): Acc={baseline_metrics['head1_identity']['accuracy']*100:.2f}%, F1={baseline_metrics['head1_identity']['f1_score']:.4f}", flush=True)
    print(f"  Head 2 (Integrity): Acc={baseline_metrics['head2_integrity']['accuracy']*100:.2f}%, F1={baseline_metrics['head2_integrity']['f1_score']:.4f}", flush=True)
    print(f"  Head 3 (Damaged PU): Rec@0.50={baseline_metrics['head3_damaged']['recall_at_0_50']*100:.2f}%, AU-PRC={baseline_metrics['head3_damaged']['auprc']:.4f}", flush=True)
    print(f"  Head 4 (Chalky PU): Rec@0.50={baseline_metrics['head4_chalky']['recall_at_0_50']*100:.2f}%, AU-PRC={baseline_metrics['head4_chalky']['auprc']:.4f}", flush=True)
    print(f"  Head 5 (Discoloured PU): Rec@0.50={baseline_metrics['head5_discoloured']['recall_at_0_50']*100:.2f}%, AU-PRC={baseline_metrics['head5_discoloured']['auprc']:.4f}", flush=True)
    print(f"  Scan-CV Consistency: Mean CV = {baseline_metrics['scan_variability']['mean_scan_cv']:.4f}", flush=True)
    
    with open(os.path.join(output_dir, 'baseline_evaluation.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
        
    # 8. Setup Losses & Optimizer
    loss_container = MultiTaskLossContainer(
        priors=cfg.priors,
        identity_pos_weight=cfg.supervised_weights['identity_pos_weight'],
        integrity_pos_weight=cfg.supervised_weights['integrity_pos_weight'],
        beta_slack=cfg.beta_slack
    ).to(device)
    
    backbone_lr = cfg.training.backbone_lr if cfg.training.backbone_lr is not None else cfg.training.learning_rate
    heads_lr = cfg.training.heads_lr if cfg.training.heads_lr is not None else cfg.training.learning_rate
    
    param_groups = [
        {'params': [p for p in student_model.backbone.parameters() if p.requires_grad], 'lr': backbone_lr},
        {'params': [p for p in student_model.head_damaged.parameters() if p.requires_grad], 'lr': heads_lr},
        {'params': [p for p in student_model.head_chalky.parameters() if p.requires_grad], 'lr': heads_lr},
        {'params': [p for p in student_model.head_discoloured.parameters() if p.requires_grad], 'lr': heads_lr},
    ]
    if not cfg.training.freeze_identity_integrity:
        param_groups.extend([
            {'params': [p for p in student_model.head_identity.parameters() if p.requires_grad], 'lr': heads_lr},
            {'params': [p for p in student_model.head_integrity.parameters() if p.requires_grad], 'lr': heads_lr},
        ])
        
    print(f"Optimizer Parameter Groups: Backbone LR = {backbone_lr}, Heads LR = {heads_lr}", flush=True)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    
    pseudo_labeler = PseudoLabeler(
        config=cfg.pseudo_labeling,
        priors=cfg.priors,
        output_dir=output_dir
    )
    
    # 9. Iterative Self-Training Loop
    best_score = baseline_metrics['composite_score']
    best_checkpoint_path = os.path.join(output_dir, 'best_improved_multitask_model.pt')
    
    epochs = cfg.training.epochs
    refresh_interval = cfg.training.pseudo_label_refresh_epochs
    training_history = []
    round_idx = 1
    
    print("\n" + "=" * 70, flush=True)
    print(f"  STARTING AUTOMATED ITERATIVE SELF-TRAINING ({epochs} EPOCHS)", flush=True)
    print("=" * 70, flush=True)
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        
        # A. Pseudo-Label Refresh Schedule
        if cfg.pseudo_labeling['enabled'] and ((epoch - 1) % refresh_interval == 0):
            print(f"\n[Epoch {epoch}] Generating EMA Teacher Triangulated Pseudo-Labels...", flush=True)
            t_pl_start = time.time()
            pseudo_map, pl_stats = pseudo_labeler.generate_pseudo_labels(
                teacher=ema_teacher,
                train_dataset=train_dataset,
                device=device,
                round_idx=round_idx
            )
            train_dataset.update_pseudo_labels(pseudo_map)
            print(f"Pseudo-label generation completed in {time.time()-t_pl_start:.1f}s:", flush=True)
            for d_name, d_stats in pl_stats.items():
                print(f"  {d_name.capitalize()}: +{d_stats['accepted_pseudo_positives']} pseudo-positives, +{d_stats['accepted_pseudo_negatives']} pseudo-negatives (rejected {d_stats['rejected_high_variance']} high-variance)", flush=True)
            round_idx += 1
            
        # B. Student Model Training on Ground Truth + Filtered Pseudo-Labels
        student_model.train()
        running_loss_total = 0.0
        running_loss_id = 0.0
        running_loss_mo = 0.0
        running_loss_dam = 0.0
        running_loss_chk = 0.0
        running_loss_dsc = 0.0
        
        for batch_idx, (imgs, targets, meta) in enumerate(train_loader):
            imgs = imgs.to(device)
            gpu_targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            preds = student_model(imgs)
            
            loss, loss_breakdown = loss_container(preds, gpu_targets)
            loss.backward()
            optimizer.step()
            
            # Update EMA Teacher continuously after every optimization step
            ema_teacher.update(student_model)
            
            bs = len(imgs)
            running_loss_total += loss_breakdown['loss_total'] * bs
            running_loss_id += loss_breakdown['loss_identity'] * bs
            running_loss_mo += loss_breakdown['loss_integrity'] * bs
            running_loss_dam += loss_breakdown['loss_damaged'] * bs
            running_loss_chk += loss_breakdown['loss_chalky'] * bs
            running_loss_dsc += loss_breakdown['loss_discoloured'] * bs
            
        scheduler.step()
        
        n_train = len(train_dataset)
        epoch_loss = running_loss_total / n_train
        epoch_elapsed = time.time() - t_epoch_start
        
        # C. Validation Evaluation (Evaluates EMA Teacher on Validation Set)
        t_val_start = time.time()
        val_metrics = evaluate_model(ema_teacher.model, val_loader, device=device)
        val_score = val_metrics['composite_score']
        val_elapsed = time.time() - t_val_start
        
        # Scan-CV comparative stability check
        baseline_cv = baseline_metrics['scan_variability']['mean_scan_cv']
        current_cv = val_metrics['scan_variability']['mean_scan_cv']
        cv_guardrail = baseline_cv * cfg.validation.get('scan_cv_guardrail_multiplier', 1.25)
        cv_passed = (current_cv <= cv_guardrail)
        
        print(f"\nEpoch {epoch}/{epochs} [{epoch_elapsed:.1f}s train, {val_elapsed:.1f}s val] - Loss: {epoch_loss:.4f} | Val Score: {val_score:.4f} (Baseline: {best_score:.4f})", flush=True)
        print(f"  Head 1 (Identity): Acc={val_metrics['head1_identity']['accuracy']*100:.2f}%, F1={val_metrics['head1_identity']['f1_score']:.4f}", flush=True)
        print(f"  Head 2 (Integrity): Acc={val_metrics['head2_integrity']['accuracy']*100:.2f}%, F1={val_metrics['head2_integrity']['f1_score']:.4f}", flush=True)
        print(f"  Head 3 (Damaged PU): Rec@0.50={val_metrics['head3_damaged']['recall_at_0_50']*100:.2f}%, AU-PRC={val_metrics['head3_damaged']['auprc']:.4f}", flush=True)
        print(f"  Head 4 (Chalky PU): Rec@0.50={val_metrics['head4_chalky']['recall_at_0_50']*100:.2f}%, AU-PRC={val_metrics['head4_chalky']['auprc']:.4f}", flush=True)
        print(f"  Head 5 (Discoloured PU): Rec@0.50={val_metrics['head5_discoloured']['recall_at_0_50']*100:.2f}%, AU-PRC={val_metrics['head5_discoloured']['auprc']:.4f}", flush=True)
        print(f"  Scan-CV Guardrail: {current_cv:.4f} <= {cv_guardrail:.4f} ({'PASSED' if cv_passed else 'EXCEEDED'})", flush=True)
        
        epoch_record = {
            'epoch': epoch,
            'train_loss': round(epoch_loss, 4),
            'composite_score': val_score,
            'validation_metrics': val_metrics,
            'scan_cv_passed': cv_passed,
            'elapsed_sec': round(epoch_elapsed + val_elapsed, 1)
        }
        training_history.append(epoch_record)
        
        # D. Automated Validation Gate & Checkpointing
        min_improvement = cfg.validation.get('min_val_improvement', 0.001)
        if val_score >= (best_score + min_improvement) and cv_passed:
            best_score = val_score
            torch.save(ema_teacher.model.state_dict(), best_checkpoint_path)
            print(f"  >>> SUCCESS: New Best Validated Checkpoint Saved! (Composite Score = {best_score:.4f})", flush=True)
            
        # Save running progress checkpoint
        torch.save({
            'epoch': epoch,
            'student_state': student_model.state_dict(),
            'teacher_state': ema_teacher.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_score': best_score
        }, os.path.join(output_dir, 'latest_training_state.pt'))
        
    # 10. Final Evaluation & Report Generation
    print("\n" + "=" * 70, flush=True)
    print("  TRAINING COMPLETE — GENERATING FINAL AUDIT REPORT", flush=True)
    print("=" * 70, flush=True)
    
    # Load best checkpoint for final certified evaluation
    if os.path.exists(best_checkpoint_path):
        final_model = RiceAgeMultiTask5Head(pretrained=False).to(device)
        final_model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        final_eval = evaluate_model(final_model, val_loader, device=device)
    else:
        final_eval = val_metrics
        best_checkpoint_path = base_ckpt_path
        
    final_report = {
        'experiment_name': cfg.experiment_name,
        'description': cfg.description,
        'hardware': {
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        },
        'baseline_checkpoint': base_ckpt_path,
        'best_improved_checkpoint': best_checkpoint_path,
        'baseline_composite_score': baseline_metrics['composite_score'],
        'final_improved_composite_score': final_eval['composite_score'],
        'composite_score_improvement': round(final_eval['composite_score'] - baseline_metrics['composite_score'], 4),
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_eval,
        'epoch_training_history': training_history
    }
    
    final_report_path = os.path.join(output_dir, 'final_evaluation_report.json')
    with open(final_report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
        
    print(f"\nFinal Evaluation Report saved to: {final_report_path}", flush=True)
    print(f"Best Improved Checkpoint saved to: {best_checkpoint_path}", flush=True)
    print(f"Baseline Score: {baseline_metrics['composite_score']:.4f} -> Final Score: {final_eval['composite_score']:.4f} (Delta: {final_report['composite_score_improvement']:+.4f})", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RiceAge AI Automated Training Pipeline")
    parser.add_argument('--config', type=str, default='auto_train/config.json', help='Path to config JSON')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    run_auto_train(config_path=args.config, resume_checkpoint=args.resume)
