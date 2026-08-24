"""
ATAD5AI — Stage 3: Scaffold-Stratified 80/10/10 Split
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 3: Scaffold-Stratified Data Split (80% Train / 10% Val / 10% Test)
# =====================================================================
# INPUT:
#   outputs/stage2_features/ecfp6_counts.npz    (sparse CSR, 241,861 × 4096)
#   outputs/stage2_features/labels.npy         (binary labels)
#   outputs/stage2_features/smiles.npy         (canonical SMILES for Chemprop)
#   outputs/stage2_features/metadata.csv       (compound metadata + bm_scaffold)
#
# OUTPUT:
#   outputs/stage3_split/
#     - X_train.npz, X_val.npz, X_test.npz       (sparse CSR matrices)
#     - y_train.npy, y_val.npy, y_test.npy       (labels)
#     - smiles_train.csv, smiles_val.csv, smiles_test.csv  (for Chemprop)
#     - metadata_train.csv, metadata_val.csv, metadata_test.csv
#     - scaffold_folds.csv                       (5-fold CV on TRAIN set)
#     - split_report.txt
#
# MODULE 3 SPEC:
#   - Bemis-Murcko Scaffold Split (80% Train, 10% Validation, 10% Test)
#   - NO scaffold in test set may appear in train set
#   - Round-robin assignment for balanced active distribution
#   - 5-fold scaffold-stratified CV on TRAIN set (for hyperparameter tuning)
# =====================================================================

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

# =====================================================================
# CONFIG
# =====================================================================
STAGE2_DIR = 'outputs/stage2_features'
OUTPUT_DIR = 'outputs/stage3_split'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 78)
print("ATAD5AI — STAGE 3: SCAFFOLD-STRATIFIED 80/10/10 SPLIT")
print("=" * 78)

# =====================================================================
# 1. Load Stage 2 outputs
# =====================================================================
print(f"\n[1/4] Loading Stage 2 outputs...")
X = sp.load_npz(os.path.join(STAGE2_DIR, 'ecfp6_counts.npz'))
y = np.load(os.path.join(STAGE2_DIR, 'labels.npy'))
smiles = np.load(os.path.join(STAGE2_DIR, 'smiles.npy'), allow_pickle=True)
metadata = pd.read_csv(os.path.join(STAGE2_DIR, 'metadata.csv'))

print(f"  X shape: {X.shape} (sparse CSR, {X.dtype})")
print(f"  y shape: {y.shape} (actives: {y.sum():,})")
print(f"  smiles shape: {smiles.shape}")
print(f"  Metadata rows: {len(metadata):,}")

# Sanity check: all lengths match
assert X.shape[0] == len(y) == len(smiles) == len(metadata), \
    f"Length mismatch: X={X.shape[0]}, y={len(y)}, smiles={len(smiles)}, metadata={len(metadata)}"

# Get scaffolds (already extracted in Stage 1, stored in metadata)
scaffold_col = 'bm_scaffold' if 'bm_scaffold' in metadata.columns else None
if scaffold_col is None:
    raise ValueError("bm_scaffold column not found in metadata")
scaffolds = metadata[scaffold_col].values
print(f"  Unique scaffolds: {len(set(scaffolds)):,}")

# =====================================================================
# 2. Build scaffold groups and separate active-containing vs inactive-only
# =====================================================================
print(f"\n[2/4] Building scaffold groups...")

# Group compound indices by scaffold
scaffold_to_indices = defaultdict(list)
for idx, scaffold in enumerate(scaffolds):
    scaffold_to_indices[scaffold].append(idx)

# Build scaffold summary: (scaffold, n_total, n_actives)
scaffold_summary = []
for scaffold, indices in scaffold_to_indices.items():
    n_total_scaf = len(indices)
    n_actives_scaf = int(y[indices].sum())
    scaffold_summary.append((scaffold, n_total_scaf, n_actives_scaf))

# Separate active-containing scaffolds from inactive-only scaffolds
active_scaffolds = [(s, n_t, n_a) for s, n_t, n_a in scaffold_summary if n_a > 0]
inactive_scaffolds = [(s, n_t, n_a) for s, n_t, n_a in scaffold_summary if n_a == 0]

# Sort: active scaffolds by n_actives DESC (high-active first → even distribution)
active_scaffolds.sort(key=lambda x: (-x[2], -x[1]))
# Inactive scaffolds by size DESC (large first for stability)
inactive_scaffolds.sort(key=lambda x: (-x[1],))

total_actives = int(y.sum())
total_compounds = len(y)
target_test_actives = int(total_actives * 0.10)    # ~761 actives in test
target_val_actives = int(total_actives * 0.10)     # ~761 actives in val
target_test_total = int(total_compounds * 0.10)    # ~24,186 in test
target_val_total = int(total_compounds * 0.10)     # ~24,186 in val

print(f"  Total actives: {total_actives:,} → target val actives: ~{target_val_actives}")
print(f"                              → target test actives: ~{target_test_actives}")
print(f"  Total compounds: {total_compounds:,} → target val size: ~{target_val_total}")
print(f"                                  → target test size: ~{target_test_total}")
print(f"  Active-containing scaffolds: {len(active_scaffolds):,}")
print(f"  Inactive-only scaffolds: {len(inactive_scaffolds):,}")

# =====================================================================
# 3. Round-robin scaffold assignment for 80/10/10 split
# =====================================================================
print(f"\n[3/4] Round-robin scaffold assignment (80/10/10)...")
print(f"  Method: every 10th active scaffold → test, every 10th offset by 5 → val")

train_indices = []
val_indices = []
test_indices = []

train_actives = 0
val_actives = 0
test_actives = 0
train_total = 0
val_total = 0
test_total = 0

# Distribute active-containing scaffolds: every 10th → test, every 10th+5 → val
print(f"\n  Distributing {len(active_scaffolds):,} active-containing scaffolds...")
for i, (scaffold, n_total_scaf, n_actives_scaf) in enumerate(active_scaffolds):
    indices = scaffold_to_indices[scaffold]
    # Pattern: 0-4 → train, 5 → val, 6-9 → train, 10 → test, repeat
    # Simpler: every 10th → test, every 10th+5 → val, rest → train
    mod = i % 10
    if mod == 9:   # 10% → test
        test_indices.extend(indices)
        test_actives += n_actives_scaf
        test_total += n_total_scaf
    elif mod == 4:  # 10% → val (offset by 5 from test)
        val_indices.extend(indices)
        val_actives += n_actives_scaf
        val_total += n_total_scaf
    else:           # 80% → train
        train_indices.extend(indices)
        train_actives += n_actives_scaf
        train_total += n_total_scaf

print(f"  After active scaffolds:")
print(f"    Train: {train_total:,} compounds, {train_actives} actives")
print(f"    Val:   {val_total:,} compounds, {val_actives} actives")
print(f"    Test:  {test_total:,} compounds, {test_actives} actives")

# Distribute inactive-only scaffolds to reach target sizes for val and test
print(f"\n  Distributing {len(inactive_scaffolds):,} inactive-only scaffolds...")
for i, (scaffold, n_total_scaf, n_actives_scaf) in enumerate(inactive_scaffolds):
    indices = scaffold_to_indices[scaffold]
    mod = i % 10
    if mod == 9 and test_total < target_test_total:  # 10% → test
        test_indices.extend(indices)
        test_total += n_total_scaf
    elif mod == 4 and val_total < target_val_total:  # 10% → val
        val_indices.extend(indices)
        val_total += n_total_scaf
    else:                                              # rest → train
        train_indices.extend(indices)
        train_total += n_total_scaf

# Convert to sorted arrays
train_indices = np.array(sorted(train_indices))
val_indices = np.array(sorted(val_indices))
test_indices = np.array(sorted(test_indices))

# =====================================================================
# 4. Report split stats
# =====================================================================
n_train_act = int(y[train_indices].sum())
n_train_inact = len(train_indices) - n_train_act
n_val_act = int(y[val_indices].sum())
n_val_inact = len(val_indices) - n_val_act
n_test_act = int(y[test_indices].sum())
n_test_inact = len(test_indices) - n_test_act

print(f"\n  FINAL 80/10/10 SPLIT:")
print(f"  {'Split':<8}{'Total':>10}{'Actives':>10}{'Inactives':>12}{'Imbalance':>12}{'Prevalence':>12}")
print(f"  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*11}")
print(f"  {'TRAIN':<8}{len(train_indices):>10,}{n_train_act:>10,}{n_train_inact:>12,}{n_train_inact/max(n_train_act,1):>11.2f}:{n_train_act/len(train_indices)*100:>10.3f}%")
print(f"  {'VAL':<8}{len(val_indices):>10,}{n_val_act:>10,}{n_val_inact:>12,}{n_val_inact/max(n_val_act,1):>11.2f}:{n_val_act/len(val_indices)*100:>10.3f}%")
print(f"  {'TEST':<8}{len(test_indices):>10,}{n_test_act:>10,}{n_test_inact:>12,}{n_test_inact/max(n_test_act,1):>11.2f}:{n_test_act/len(test_indices)*100:>10.3f}%")

# Sanity check: scaffold overlap
train_scaffolds = set(scaffolds[train_indices])
val_scaffolds = set(scaffolds[val_indices])
test_scaffolds = set(scaffolds[test_indices])
print(f"\n  Scaffolds in train: {len(train_scaffolds):,}")
print(f"  Scaffolds in val:   {len(val_scaffolds):,}")
print(f"  Scaffolds in test:  {len(test_scaffolds):,}")
print(f"  Train ∩ Val overlap:  {len(train_scaffolds & val_scaffolds)}")
print(f"  Train ∩ Test overlap: {len(train_scaffolds & test_scaffolds)}")
print(f"  Val ∩ Test overlap:   {len(val_scaffolds & test_scaffolds)}")
print(f"  (All overlaps should be 0)")

# =====================================================================
# 5. Build 5-fold scaffold-stratified CV on TRAIN set (for hyperparameter tuning)
# =====================================================================
print(f"\n[4/4] Building 5-fold scaffold-stratified CV on TRAIN set...")

# Group train compounds by scaffold
train_scaffolds_arr = scaffolds[train_indices]
train_scaffold_to_indices = defaultdict(list)
for i, scaffold in enumerate(train_scaffolds_arr):
    train_scaffold_to_indices[scaffold].append(i)

# Build train scaffold summary
train_scaffold_summary = []
for scaffold, indices in train_scaffold_to_indices.items():
    train_scaffold_summary.append((scaffold, len(indices), int(y[train_indices][indices].sum())))

train_active_scaffolds = [(s, n_t, n_a) for s, n_t, n_a in train_scaffold_summary if n_a > 0]
train_inactive_scaffolds = [(s, n_t, n_a) for s, n_t, n_a in train_scaffold_summary if n_a == 0]
train_active_scaffolds.sort(key=lambda x: (-x[2], -x[1]))
train_inactive_scaffolds.sort(key=lambda x: (-x[1],))

n_train_actives_total = int(y[train_indices].sum())
target_fold_actives = n_train_actives_total // 5
print(f"  Train actives: {n_train_actives_total:,} → ~{target_fold_actives} per fold")
print(f"  Train active-containing scaffolds: {len(train_active_scaffolds):,}")
print(f"  Train inactive-only scaffolds: {len(train_inactive_scaffolds):,}")

folds = [[] for _ in range(5)]
fold_actives = [0] * 5
fold_totals = [0] * 5

# Round-robin assign active scaffolds to folds 0..4
for i, (scaffold, n_total_scaf, n_actives_scaf) in enumerate(train_active_scaffolds):
    indices = train_scaffold_to_indices[scaffold]
    target_fold = i % 5
    folds[target_fold].extend(indices)
    fold_actives[target_fold] += n_actives_scaf
    fold_totals[target_fold] += n_total_scaf

# Distribute inactive scaffolds round-robin
for i, (scaffold, n_total_scaf, n_actives_scaf) in enumerate(train_inactive_scaffolds):
    indices = train_scaffold_to_indices[scaffold]
    target_fold = i % 5
    folds[target_fold].extend(indices)
    fold_totals[target_fold] += n_total_scaf

# Build fold assignment array (one value 0-4 per train compound)
fold_assignment = np.full(len(train_indices), -1, dtype=int)
for fold_idx, indices in enumerate(folds):
    for i in indices:
        fold_assignment[i] = fold_idx

print(f"\n  Fold |  Total  | Actives | Inactives | Imbalance")
print(f"  -----|---------|---------|----------|----------")
for k in range(5):
    n_a = fold_actives[k]
    n_i = fold_totals[k] - n_a
    imb = n_i / max(n_a, 1)
    print(f"   {k}   | {fold_totals[k]:>7,} |  {n_a:>5}  |  {n_i:>7,} | {imb:.2f}:1")
print(f"\n  Fold actives: {fold_actives}")
print(f"  Sum: {sum(fold_actives)} (should = train actives: {n_train_actives_total})")

# Sanity check: no scaffold overlap across folds
fold_scaffold_sets = []
for k in range(5):
    fold_scaffolds_k = set(scaffolds[train_indices][folds[k]])
    fold_scaffold_sets.append(fold_scaffolds_k)
fold_overlap_count = 0
for i in range(5):
    for j in range(i+1, 5):
        fold_overlap_count += len(fold_scaffold_sets[i] & fold_scaffold_sets[j])
print(f"  Total scaffold overlaps across folds: {fold_overlap_count}")

# =====================================================================
# 6. Save outputs
# =====================================================================
print(f"\n=== SAVING OUTPUTS ===")

# Sparse feature matrices
sp.save_npz(os.path.join(OUTPUT_DIR, 'X_train.npz'), X[train_indices])
sp.save_npz(os.path.join(OUTPUT_DIR, 'X_val.npz'),   X[val_indices])
sp.save_npz(os.path.join(OUTPUT_DIR, 'X_test.npz'),  X[test_indices])
print(f"  Saved X_train.npz: {X[train_indices].shape}")
print(f"  Saved X_val.npz:   {X[val_indices].shape}")
print(f"  Saved X_test.npz:  {X[test_indices].shape}")

# Labels
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y[train_indices])
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   y[val_indices])
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y[test_indices])
print(f"  Saved y_train.npy (actives: {y[train_indices].sum()})")
print(f"  Saved y_val.npy   (actives: {y[val_indices].sum()})")
print(f"  Saved y_test.npy  (actives: {y[test_indices].sum()})")

# SMILES for Chemprop (CSV with smiles + label)
for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
    chemprop_df = pd.DataFrame({
        'smiles': smiles[indices],
        'label': y[indices]
    })
    chemprop_df.to_csv(os.path.join(OUTPUT_DIR, f'smiles_{split_name}.csv'), index=False)
    print(f"  Saved smiles_{split_name}.csv ({len(chemprop_df):,} rows)")

# Metadata for each split
for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
    metadata_split = metadata.iloc[indices].reset_index(drop=True)
    metadata_split.to_csv(os.path.join(OUTPUT_DIR, f'metadata_{split_name}.csv'), index=False)
    print(f"  Saved metadata_{split_name}.csv ({len(metadata_split):,} rows)")

# 5-fold CV assignment (for Stage 4 model training)
fold_df = pd.DataFrame({
    'pubchem_sid': metadata.iloc[train_indices]['pubchem_sid'].values if 'pubchem_sid' in metadata.columns else None,
    'canonical_smiles': smiles[train_indices],
    'bm_scaffold': scaffolds[train_indices],
    'label': y[train_indices],
    'fold': fold_assignment
})
fold_df.to_csv(os.path.join(OUTPUT_DIR, 'scaffold_folds.csv'), index=False)
print(f"  Saved scaffold_folds.csv (5-fold CV assignment)")

# =====================================================================
# 7. Save split report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'split_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 3 SPLIT REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("SPLIT STRATEGY\n")
    f.write("  Bemis-Murcko scaffold-stratified 80/10/10\n")
    f.write("  Round-robin assignment (every 10th → test, every 10th+5 → val)\n")
    f.write("  Zero scaffold overlap between splits\n\n")
    f.write("TRAIN/VAL/TEST SIZES\n")
    f.write(f"  Train: {len(train_indices):,} ({n_train_act:,} actives, {n_train_inact:,} inactives)\n")
    f.write(f"  Val:   {len(val_indices):,} ({n_val_act:,} actives, {n_val_inact:,} inactives)\n")
    f.write(f"  Test:  {len(test_indices):,} ({n_test_act:,} actives, {n_test_inact:,} inactives)\n\n")
    f.write("IMBALANCE RATIOS\n")
    f.write(f"  Train: {n_train_inact/max(n_train_act,1):.2f}:1 (prevalence {n_train_act/len(train_indices)*100:.3f}%)\n")
    f.write(f"  Val:   {n_val_inact/max(n_val_act,1):.2f}:1 (prevalence {n_val_act/len(val_indices)*100:.3f}%)\n")
    f.write(f"  Test:  {n_test_inact/max(n_test_act,1):.2f}:1 (prevalence {n_test_act/len(test_indices)*100:.3f}%)\n\n")
    f.write("SCAFFOLD OVERLAP (should all be 0)\n")
    f.write(f"  Train ∩ Val:  {len(train_scaffolds & val_scaffolds)}\n")
    f.write(f"  Train ∩ Test: {len(train_scaffolds & test_scaffolds)}\n")
    f.write(f"  Val ∩ Test:   {len(val_scaffolds & test_scaffolds)}\n\n")
    f.write("5-FOLD CV ON TRAIN SET\n")
    for k in range(5):
        n_a = fold_actives[k]
        n_i = fold_totals[k] - n_a
        f.write(f"  Fold {k}: {fold_totals[k]:,} ({n_a} actives, {n_i:,} inactives, imbalance {n_i/max(n_a,1):.2f}:1)\n")
    f.write(f"  Fold scaffold overlaps: {fold_overlap_count}\n\n")
    f.write("NEXT STEPS (Stage 4 — three models, parallel)\n")
    f.write("  4a. Chemprop D-MPNN on Stream A (SMILES → graph → D-MPNN)\n")
    f.write("      - 3 message-passing steps, hidden=300, 2-layer FFN\n")
    f.write("      - Adam optimizer lr=1e-4, early stopping on val PR-AUC (patience=15)\n")
    f.write("      - Expects smiles_train.csv, smiles_val.csv from this stage\n")
    f.write("  4b. CatBoost on Stream B (ECFP6 count vectors)\n")
    f.write("      - Logloss, auto_class_weights='Balanced', depth=6, 1000 iter, lr=0.03\n")
    f.write("      - L2=3.0, subsample=0.8, early stopping 50 rounds on val PR-AUC\n")
    f.write("      - Expects X_train.npz, X_val.npz from this stage\n")
    f.write("  4c. Balanced Random Forest on Stream B (ECFP6 count vectors)\n")
    f.write("      - 600 trees, max_depth=18, max_features='sqrt'\n")
    f.write("      - class_weight='balanced_subsample', n_jobs=-1\n")
    f.write("      - Expects X_train.npz, X_val.npz from this stage\n")
print(f"  Saved split_report.txt")

print(f"\n" + "=" * 78)
print("STAGE 3 COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - X_train.npz, X_val.npz, X_test.npz       (sparse features)")
print(f"  - y_train.npy, y_val.npy, y_test.npy        (labels)")
print(f"  - smiles_train.csv, smiles_val.csv, smiles_test.csv  (for Chemprop)")
print(f"  - metadata_train.csv, metadata_val.csv, metadata_test.csv")
print(f"  - scaffold_folds.csv                       (5-fold CV on train)")
print(f"  - split_report.txt")
print(f"\nReady for Stage 4 (three-model training).")
print(f"  Stage 4a: Chemprop D-MPNN       (~6-12 hours on RTX 2080)")
print(f"  Stage 4b: CatBoost on ECFP6     (~30-60 min on CPU)")
print(f"  Stage 4c: Balanced RF on ECFP6  (~30-60 min on CPU)")
print(f"\nRecommended: run 4b and 4c in parallel with 4a (they don't need GPU).")
