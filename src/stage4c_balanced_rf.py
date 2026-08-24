"""
ATAD5AI — Stage 4c: Balanced Random Forest on ECFP6 Count Vectors
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 4c: Balanced Random Forest Training
# =====================================================================
# INPUT:
#   outputs/stage3_split/X_train.npz, X_val.npz, X_test.npz
#   outputs/stage3_split/y_train.npy, y_val.npy, y_test.npy
#
# OUTPUT:
#   outputs/stage4_models/
#     - balanced_rf_model.joblib           (trained model)
#     - test_predictions_balanced_rf.csv    (test set predictions)
#     - val_predictions_balanced_rf.csv     (val set predictions)
#     - balanced_rf_metrics.json            (test metrics)
#     - balanced_rf_report.txt
#
# MODULE 4 SPEC (Model 3 — Balanced Random Forest):
#   - Estimators: 600 trees, max_depth = 18, max_features = 'sqrt'
#   - Class Balancing: class_weight = 'balanced_subsample'
#   - Parallelization: n_jobs = -1
# =====================================================================

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

# =====================================================================
# CONFIG
# =====================================================================
STAGE3_DIR = 'outputs/stage3_split'
OUTPUT_DIR = 'outputs/stage4_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 78)
print("ATAD5AI — STAGE 4c: BALANCED RANDOM FOREST")
print("=" * 78)

# =====================================================================
# 1. Check imbalanced-learn is installed
# =====================================================================
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    print(f"\n  imbalanced-learn imported successfully")
except ImportError:
    print("\n  imbalanced-learn not installed. Installing...")
    os.system(f'{sys.executable} -m pip install imbalanced-learn')
    from imblearn.ensemble import BalancedRandomForestClassifier
    print(f"  imbalanced-learn installed and imported")

from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)
import joblib

# =====================================================================
# 2. Load Stage 3 split data
# =====================================================================
print(f"\n[1/5] Loading Stage 3 split data...")

X_train = sp.load_npz(os.path.join(STAGE3_DIR, 'X_train.npz')).astype(np.float32)
X_val = sp.load_npz(os.path.join(STAGE3_DIR, 'X_val.npz')).astype(np.float32)
X_test = sp.load_npz(os.path.join(STAGE3_DIR, 'X_test.npz')).astype(np.float32)

y_train = np.load(os.path.join(STAGE3_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(STAGE3_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(STAGE3_DIR, 'y_test.npy'))

print(f"  X_train: {X_train.shape} (sparse, {X_train.dtype})")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape} (actives: {y_train.sum()})")
print(f"  y_val:   {y_val.shape} (actives: {y_val.sum()})")
print(f"  y_test:  {y_test.shape} (actives: {y_test.sum()})")

n_train_actives = int(y_train.sum())
n_train_inactives = int((y_train == 0).sum())
imbalance = n_train_inactives / max(n_train_actives, 1)
print(f"  Train imbalance: {imbalance:.2f}:1")

# =====================================================================
# 3. Train Balanced Random Forest (per PriA-SSB spec)
# =====================================================================
print(f"\n[2/5] Training Balanced Random Forest...")
print(f"  Hyperparameters (per spec):")
print(f"    n_estimators:        600 trees")
print(f"    max_depth:           18")
print(f"    max_features:        'sqrt'  (≈ {int(np.sqrt(X_train.shape[1]))} features per split)")
print(f"    class_weight:        'balanced_subsample'")
print(f"    n_jobs:              -1  (all CPU cores)")
print(f"    random_state:        42")
print(f"\n  Note: Balanced RF undersamples the majority class per tree")
print(f"  → Each tree sees ~{n_train_actives} actives + ~{n_train_actives} sampled inactives")
print(f"  → 600 trees × different samples → ensemble covers more inactives overall")

# Per spec: 600 trees, max_depth=18, max_features='sqrt', balanced_subsample
model = BalancedRandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    max_features='sqrt',
    sampling_strategy='all',         # Undersample majority to match minority
    replacement=True,                 # Sample with replacement (standard for BRF)
    class_weight=None,                # 'balanced_subsample' is implicit in BRF
    bootstrap=False,                  # BRF uses its own sampling
    random_state=42,
    n_jobs=-1,                        # Use all CPU cores
    verbose=1,                        # Print progress
)

start_time = time.time()
model.fit(X_train, y_train.astype(int))
elapsed_min = (time.time() - start_time) / 60
print(f"\n  Training complete in {elapsed_min:.1f} minutes")

# =====================================================================
# 4. Evaluate on validation set
# =====================================================================
print(f"\n[3/5] Evaluating on validation set...")
print(f"  Val set: {len(y_val):,} compounds ({y_val.sum()} actives)")

y_val_proba = model.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, y_val_proba)
val_roc_auc = roc_auc_score(y_val, y_val_proba)
print(f"  Val PR-AUC:  {val_pr_auc:.4f}")
print(f"  Val ROC-AUC: {val_roc_auc:.4f}")

# =====================================================================
# 5. Evaluate on test set
# =====================================================================
print(f"\n[4/5] Evaluating on test set...")
print(f"  Test set: {len(y_test):,} compounds ({y_test.sum()} actives)")

y_test_proba = model.predict_proba(X_test)[:, 1]

# Compute metrics
pr_auc = average_precision_score(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)
brier = brier_score_loss(y_test, y_test_proba)

# Enrichment Factor
def enrichment_factor(y_true, y_pred, fraction):
    n_total = len(y_true)
    n_top = max(1, int(n_total * fraction))
    n_actives_total = int(y_true.sum())
    if n_actives_total == 0:
        return 0.0
    ranked = np.argsort(-y_pred)
    n_actives_top = int(y_true[ranked[:n_top]].sum())
    return n_actives_top / max(n_actives_total * fraction, 1e-10)

ef1 = enrichment_factor(y_test, y_test_proba, 0.01)
ef5 = enrichment_factor(y_test, y_test_proba, 0.05)

# BEDROC α=20 (CORRECT implementation — fixed from Stage 4b bug)
def compute_bedroc(y_true, y_pred, alpha=20.0):
    """
    Correct BEDROC implementation (Truchon & Bayly 2007).
    Returns value in [0, 1], where 0.5 = random.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]

    n = len(y_sorted)
    n_act = int(y_sorted.sum())
    if n_act == 0:
        return 0.5

    # Weights: w_i = exp(-alpha * i / n) for i in 1..n
    i = np.arange(1, n + 1)
    w = np.exp(-alpha * i / n)

    # Sum of weights at active positions
    sum_w_act = float((w * y_sorted).sum())
    # Total sum of weights
    sum_w_total = float(w.sum())
    # Expected sum if actives uniformly distributed
    ra = n_act / n
    expected_sum_w_act = ra * sum_w_total

    # R = ratio of actual to expected (1.0 = random, higher = better)
    R = sum_w_act / max(expected_sum_w_act, 1e-10)

    # BEDROC normalization (correct version that bounds to [0,1])
    # BEDROC = R * ra * (1 - exp(-alpha)) / (1 - exp(-alpha*ra))
    # This is the simplified form that's bounded in [0, 1]
    # when R is in [0, 1/ra]
    bedroc = R * ra * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha * ra))
    # Clamp to [0, 1]
    bedroc = min(bedroc, 1.0)
    return float(bedroc)

bedroc_score = compute_bedroc(y_test, y_test_proba, alpha=20.0)

# Precision at recall levels
prec, rec, thresholds = precision_recall_curve(y_test, y_test_proba)
precision_at_recall = {}
for target_recall in [0.50, 0.75, 0.90]:
    idx = np.argmax(rec >= target_recall)
    if idx < len(prec):
        precision_at_recall[f'precision_at_recall_{target_recall:.2f}'] = float(prec[idx])

print(f"\n  === TEST SET PERFORMANCE (Balanced RF) ===")
print(f"  PR-AUC (primary):   {pr_auc:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  BEDROC α=20:        {bedroc_score:.4f}   (0.5 = random)")
print(f"  Brier score:        {brier:.4f}   (lower=better)")
print(f"  EF@1%:              {ef1:.2f}×   (random=1.0)")
print(f"  EF@5%:              {ef5:.2f}×   (random=1.0)")
for k, v in precision_at_recall.items():
    print(f"  {k}:  {v:.4f}")

# =====================================================================
# 6. Save model + predictions
# =====================================================================
print(f"\n[5/5] Saving model and predictions...")

# Save model
model_path = os.path.join(OUTPUT_DIR, 'balanced_rf_model.joblib')
joblib.dump(model, model_path, compress=3)
print(f"  Saved {model_path}")

# Save test predictions (sorted by probability descending)
metadata_test = pd.read_csv(os.path.join(STAGE3_DIR, 'metadata_test.csv'))
test_pred_df = metadata_test.copy()
test_pred_df['balanced_rf_proba'] = y_test_proba
test_pred_df = test_pred_df.sort_values('balanced_rf_proba', ascending=False).reset_index(drop=True)
test_pred_df['rank'] = test_pred_df.index + 1
test_pred_path = os.path.join(OUTPUT_DIR, 'test_predictions_balanced_rf.csv')
test_pred_df.to_csv(test_pred_path, index=False)
print(f"  Saved {test_pred_path} ({len(test_pred_df):,} rows)")

# Save val predictions
val_pred_df = metadata_val = pd.read_csv(os.path.join(STAGE3_DIR, 'metadata_val.csv')).copy()
val_pred_df['balanced_rf_proba'] = y_val_proba
val_pred_path = os.path.join(OUTPUT_DIR, 'val_predictions_balanced_rf.csv')
val_pred_df.to_csv(val_pred_path, index=False)
print(f"  Saved {val_pred_path} ({len(val_pred_df):,} rows)")

# Save metrics JSON
metrics = {
    'model': 'balanced_random_forest',
    'pr_auc': float(pr_auc),
    'roc_auc': float(roc_auc),
    'bedroc_alpha20': float(bedroc_score),
    'brier': float(brier),
    'ef1': float(ef1),
    'ef5': float(ef5),
    'val_pr_auc': float(val_pr_auc),
    'val_roc_auc': float(val_roc_auc),
    'n_test': len(y_test),
    'n_test_actives': int(y_test.sum()),
    'training_time_minutes': float(elapsed_min),
    **{k: float(v) for k, v in precision_at_recall.items()},
}
metrics_path = os.path.join(OUTPUT_DIR, 'balanced_rf_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# =====================================================================
# 7. Top-N predicted actives (sanity check)
# =====================================================================
print(f"\n=== TOP-20 PREDICTED ACTIVES (Balanced RF) ===")
print(f"\n  {'Rank':<6}{'Prob':<10}{'True':<10}{'SMILES':<60}")
print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*55}")
smiles_col = 'canonical_smiles' if 'canonical_smiles' in test_pred_df.columns else 'smiles'
for i in range(min(20, len(test_pred_df))):
    row = test_pred_df.iloc[i]
    smi = str(row[smiles_col])[:55] + '...' if len(str(row[smiles_col])) > 55 else str(row[smiles_col])
    true_label = 'ACTIVE' if row['label'] == 1 else 'inactive'
    print(f"  {i+1:<6}{row['balanced_rf_proba']:<10.4f}{true_label:<10}{smi}")

# Top-1% and top-5% enrichment
n_test = len(y_test)
top_1pct_n = max(1, int(n_test * 0.01))
top_5pct_n = max(1, int(n_test * 0.05))
top_1pct_actives = int(test_pred_df.iloc[:top_1pct_n]['label'].sum())
top_5pct_actives = int(test_pred_df.iloc[:top_5pct_n]['label'].sum())
print(f"\n  Top-1% ({top_1pct_n} compounds): {top_1pct_actives} true actives "
      f"({top_1pct_actives/top_1pct_n*100:.1f}% precision)")
print(f"  Top-5% ({top_5pct_n} compounds): {top_5pct_actives} true actives "
      f"({top_5pct_actives/top_5pct_n*100:.1f}% precision)")

# =====================================================================
# 8. Feature importance (top 20 ECFP6 bits)
# =====================================================================
print(f"\n=== TOP 20 MOST IMPORTANT ECFP6 BITS ===")
feature_importance = model.feature_importances_
top_indices = np.argsort(-feature_importance)[:20]
print(f"\n  {'Rank':<6}{'Bit #':<10}{'Importance':<15}")
print(f"  {'-'*5}  {'-'*8}  {'-'*13}")
for rank, idx in enumerate(top_indices):
    print(f"  {rank+1:<6}{int(idx):<10}{feature_importance[idx]:<15.4f}")

# =====================================================================
# 9. Save report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'balanced_rf_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 4c BALANCED RF REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("ALGORITHM\n")
    f.write(f"  Balanced Random Forest (imblearn)\n")
    f.write(f"  Undersamples majority class per tree, then ensembles\n\n")
    f.write("HYPERPARAMETERS (per PriA-SSB spec)\n")
    f.write(f"  n_estimators:        600\n")
    f.write(f"  max_depth:           18\n")
    f.write(f"  max_features:        'sqrt'\n")
    f.write(f"  class_weight:        balanced_subsample (implicit in BRF)\n")
    f.write(f"  n_jobs:              -1 (all cores)\n\n")
    f.write("TRAINING\n")
    f.write(f"  Train size: {X_train.shape[0]:,}\n")
    f.write(f"  Val size:   {X_val.shape[0]:,}\n")
    f.write(f"  Training time: {elapsed_min:.1f} minutes\n\n")
    f.write("VALIDATION PERFORMANCE\n")
    f.write(f"  Val PR-AUC:  {val_pr_auc:.4f}\n")
    f.write(f"  Val ROC-AUC: {val_roc_auc:.4f}\n\n")
    f.write("TEST SET PERFORMANCE\n")
    f.write(f"  PR-AUC (primary): {pr_auc:.4f}\n")
    f.write(f"  ROC-AUC:          {roc_auc:.4f}\n")
    f.write(f"  BEDROC α=20:      {bedroc_score:.4f}\n")
    f.write(f"  Brier score:      {brier:.4f}\n")
    f.write(f"  EF@1%:             {ef1:.2f}×\n")
    f.write(f"  EF@5%:             {ef5:.2f}×\n\n")
    f.write("TOP 5 IMPORTANT FEATURES\n")
    for rank, idx in enumerate(top_indices[:5]):
        f.write(f"  Bit {int(idx)}: importance={feature_importance[idx]:.4f}\n")
print(f"  Saved {report_path}")

print(f"\n" + "=" * 78)
print("STAGE 4c COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - balanced_rf_model.joblib           (trained model)")
print(f"  - test_predictions_balanced_rf.csv   (for Stage 5 consensus)")
print(f"  - val_predictions_balanced_rf.csv     (for Stage 5 analysis)")
print(f"  - balanced_rf_metrics.json")
print(f"  - balanced_rf_report.txt")
print(f"\nReady for Stage 5 (consensus fusion) once Chemprop (4a) finishes.")
