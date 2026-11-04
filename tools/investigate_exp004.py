import os
import json
import pandas as pd
import numpy as np

def run_investigation():
    print("=" * 70)
    print("  EXP004 REGRESSION ROOT CAUSE INVESTIGATION")
    print("=" * 70)
    
    exp_dir = "outputs/phase2_automated/experiments/exp004_ema_pseudolabel_pu"
    
    # 1. Inspect Pseudo-Labels
    r1 = pd.read_csv(os.path.join(exp_dir, "pseudo_labels_round_1.csv"))
    r2 = pd.read_csv(os.path.join(exp_dir, "pseudo_labels_round_2.csv"))
    r3 = pd.read_csv(os.path.join(exp_dir, "pseudo_labels_round_3.csv"))
    
    print(f"\n1. Pseudo-Label Counts per Round:")
    print(f"   Round 1: {len(r1)} grains")
    print(f"   Round 2: {len(r2)} grains")
    print(f"   Round 3: {len(r3)} grains")
    
    for r_idx, df in enumerate([r1, r2, r3], 1):
        print(f"\n   --- Round {r_idx} Breakdown ---")
        for col in ['pseudo_damaged', 'pseudo_chalky', 'pseudo_discoloured']:
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                pos = counts.get('POSITIVE', 0)
                neg = counts.get('NEGATIVE', 0)
                ratio = neg / max(pos, 1)
                print(f"   {col:20s}: Pos={pos:5d}, Neg={neg:5d} (Neg/Pos Ratio = {ratio:.1f}:1)")

    # 2. Check Stability & Flip Rate between rounds
    print(f"\n2. Inter-Round Stability & Thrashing Analysis:")
    cids_r1 = set(r1['canonical_instance_id'])
    cids_r2 = set(r2['canonical_instance_id'])
    cids_r3 = set(r3['canonical_instance_id'])
    
    jaccard_1_2 = len(cids_r1 & cids_r2) / len(cids_r1 | cids_r2)
    jaccard_2_3 = len(cids_r2 & cids_r3) / len(cids_r2 | cids_r3)
    print(f"   Jaccard Similarity R1 <-> R2: {jaccard_1_2:.4f} ({len(cids_r1 & cids_r2)} shared)")
    print(f"   Jaccard Similarity R2 <-> R3: {jaccard_2_3:.4f} ({len(cids_r2 & cids_r3)} shared)")
    
    # Check flip rate on shared grains
    shared_1_2 = pd.merge(r1, r2, on='canonical_instance_id', suffixes=('_r1', '_r2'))
    for defect in ['damaged', 'chalky', 'discoloured']:
        col1 = f'pseudo_{defect}_r1'
        col2 = f'pseudo_{defect}_r2'
        if col1 in shared_1_2.columns and col2 in shared_1_2.columns:
            valid_mask = shared_1_2[col1].notna() & shared_1_2[col2].notna()
            flips = (shared_1_2.loc[valid_mask, col1] != shared_1_2.loc[valid_mask, col2]).sum()
            total = valid_mask.sum()
            flip_rate = flips / max(total, 1)
            print(f"   Flip Rate R1->R2 ({defect:12s}): {flips}/{total} ({flip_rate*100:.2f}%)")

    # 3. Check for ground truth conflicts
    print(f"\n3. Ground Truth Supervision Integrity Check:")
    manifest = pd.read_csv("outputs/phase2_automated/automated_dataset_manifest.csv")
    train_m = manifest[manifest['pu_learning_split_role'] == 'TRAIN_FITTING_ELIGIBLE'].set_index('canonical_instance_id')
    
    conflicts_dam = 0
    conflicts_chk = 0
    conflicts_dsc = 0
    
    for _, row in r1.iterrows():
        cid = row['canonical_instance_id']
        if cid in train_m.index:
            gt = train_m.loc[cid]
            if pd.notna(row.get('pseudo_damaged')):
                if gt['is_damaged_supervision_state'] == 'POSITIVE' and row['pseudo_damaged'] == 'NEGATIVE':
                    conflicts_dam += 1
            if pd.notna(row.get('pseudo_chalky')):
                if gt['is_chalky_supervision_state'] == 'POSITIVE' and row['pseudo_chalky'] == 'NEGATIVE':
                    conflicts_chk += 1
                if gt['is_chalky_supervision_state'] == 'NEGATIVE' and row['pseudo_chalky'] == 'POSITIVE':
                    conflicts_chk += 1
            if pd.notna(row.get('pseudo_discoloured')):
                if gt['is_discoloured_supervision_state'] == 'POSITIVE' and row['pseudo_discoloured'] == 'NEGATIVE':
                    conflicts_dsc += 1
                    
    print(f"   Conflicts with Ground Truth in R1: Damaged={conflicts_dam}, Chalky={conflicts_chk}, Discoloured={conflicts_dsc}")

    # 4. Analyze Training Dynamics from final_evaluation_report.json
    print(f"\n4. Training Progression Analysis (From final_evaluation_report.json):")
    with open(os.path.join(exp_dir, "final_evaluation_report.json"), "r") as f:
        rep = json.load(f)
        
    print(f"   Baseline Composite Score: {rep['baseline_composite_score']:.4f}")
    print(f"   Candidate Final Score:   {rep['final_improved_composite_score']:.4f}")
    
    print("\n   Epoch-by-Epoch Metric Trajectory:")
    print("   Epoch | Train Loss | Identity F1 | Integrity F1 | Damaged AUPRC | Chalky AUPRC | Discoloured Rec | Score")
    print("   " + "-" * 85)
    
    # Baseline
    bm = rep['baseline_metrics']
    print(f"   Base  |    N/A     |   {bm['head1_identity']['f1_score']:.4f}    |    {bm['head2_integrity']['f1_score']:.4f}    |    {bm['head3_damaged']['auprc']:.4f}     |    {bm['head4_chalky']['auprc']:.4f}    |     {bm['head5_discoloured']['recall_at_0_50']*100:5.2f}%     | {bm['composite_score']:.4f}")
    
    for ep in rep['epoch_training_history']:
        e_num = ep['epoch']
        tl = ep['train_loss']
        vm = ep['validation_metrics']
        cs = ep['composite_score']
        id_f1 = vm['head1_identity']['f1_score']
        int_f1 = vm['head2_integrity']['f1_score']
        dam_au = vm['head3_damaged']['auprc']
        chk_au = vm['head4_chalky']['auprc']
        dsc_rec = vm['head5_discoloured']['recall_at_0_50'] * 100
        print(f"   Ep {e_num}  |   {tl:.4f}   |   {id_f1:.4f}    |    {int_f1:.4f}    |    {dam_au:.4f}     |    {chk_au:.4f}    |     {dsc_rec:5.2f}%     | {cs:.4f}")

    # 5. Analyze Identity / Integrity Head Dynamics
    print(f"\n5. Identity and Integrity Head Degradation Analysis:")
    print(f"   Head 1 (Identity):")
    print(f"     Baseline Confusion Matrix: TP={bm['head1_identity']['cm']['tp']}, FP={bm['head1_identity']['cm']['fp']}, FN={bm['head1_identity']['cm']['fn']}, TN={bm['head1_identity']['cm']['tn']}")
    print(f"     Candidate Confusion Matrix: TP={rep['final_metrics']['head1_identity']['cm']['tp']}, FP={rep['final_metrics']['head1_identity']['cm']['fp']}, FN={rep['final_metrics']['head1_identity']['cm']['fn']}, TN={rep['final_metrics']['head1_identity']['cm']['tn']}")
    print(f"     FP jumped from 57 -> 111! Precision collapsed from 65.7% -> 49.8%!")
    
    print(f"\n   Head 2 (Integrity):")
    print(f"     Baseline Confusion Matrix: TP={bm['head2_integrity']['cm']['tp']}, FP={bm['head2_integrity']['cm']['fp']}, FN={bm['head2_integrity']['cm']['fn']}, TN={bm['head2_integrity']['cm']['tn']}")
    print(f"     Candidate Confusion Matrix: TP={rep['final_metrics']['head2_integrity']['cm']['tp']}, FP={rep['final_metrics']['head2_integrity']['cm']['fp']}, FN={rep['final_metrics']['head2_integrity']['cm']['fn']}, TN={rep['final_metrics']['head2_integrity']['cm']['tn']}")

if __name__ == '__main__':
    run_investigation()
