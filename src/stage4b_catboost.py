"""
ATAD5AI — Stage 4b: CatBoost Classifier on ECFP6 Count Vectors
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 4b: CatBoost Training
# =====================================================================
# INPUT:
#   outputs/stage3_split/X_train.npz, X_val.npz, X_test.npz
#   outputs/stage3_split/y_train.npy, y_val.npy, y_test.npy
#
# OUTPUT:
#   outputs/stage4_models/
#     - catboost_model.cbm               (trained model)
#     - test_predictions_catboost.csv     (test set predictions)
#     - val_predictions_catboost.csv      (val set predictions)
#     - catboost_metrics.json             (test metrics)
#     - catboost_report.txt
#
# MODULE 4 SPEC (Model 2 — CatBoost):
#   - Objective: 'Logloss' with auto_class_weights='Balanced'
#   - Tree Structure: Symmetric/oblivious trees (depth=6, iterations=1000, lr=0.03)
#   - Regularization: L2 leaf regularization = 3.0, subsample = 0.8
#   - Early stopping: 50 rounds monitoring Validation PR-AUC
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
print("ATAD5AI — STAGE 4b: CATBOOST CLASSIFIER")
print("=" * 78)

# =====================================================================
# 1. Check CatBoost is installed
# =====================================================================
try:
    from catboost import CatBoostClassifier, Pool
    print(f"\n  CatBoost imported successfully")
except ImportError:
    print("\n  CatBoost not installed. Installing...")
    os.system(f'{sys.executable} -m pip install catboost')
    from catboost import CatBoostClassifier, Pool
    print(f"  CatBoost installed and imported")

from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)

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
# 3. Build CatBoost Pool objects
# =====================================================================
print(f"\n[2/5] Building CatBoost Pools...")

# CatBoost Pool supports scipy.sparse CSR matrices natively
# Feature names go in the Pool, not in the Classifier
feature_names = [f'ecfp6_bit_{i}' for i in range(X_train.shape[1])]

# Convert labels to int for binary classification
train_pool = Pool(X_train, label=y_train.astype(int), feature_names=feature_names)
val_pool = Pool(X_val, label=y_val.astype(int), feature_names=feature_names)
test_pool = Pool(X_test, label=y_test.astype(int), feature_names=feature_names)

print(f"  Train pool: {train_pool.num_row()} rows, {train_pool.num_col()} features")
print(f"  Val pool:   {val_pool.num_row()} rows")
print(f"  Test pool:  {test_pool.num_row()} rows")

# =====================================================================
# 4. Train CatBoost (per PriA-SSB spec)
# =====================================================================
print(f"\n[3/5] Training CatBoost...")
print(f"  Hyperparameters (per spec):")
print(f"    Objective:           Logloss")
print(f"    auto_class_weights:  Balanced")
print(f"    depth:               6 (symmetric/oblivious trees)")
print(f"    iterations:          1000")
print(f"    learning_rate:       0.03")
print(f"    L2 leaf reg:         3.0")
print(f"    subsample:           0.8")
print(f"    early_stopping:      50 rounds on val PRAUC")
print(f"    eval_metric:         PRAUC")

# Per spec: depth=6, iterations=1000, lr=0.03, L2=3.0, subsample=0.8
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    subsample=0.8,
    loss_function='Logloss',
    eval_metric='PRAUC',         # Monitor PR-AUC on validation
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=50,                   # Print progress every 50 iterations
    early_stopping_rounds=50,     # Per spec: 50 rounds
    task_type='CPU',              # macOS doesn't have CUDA; CatBoost CPU is fine
)

start_time = time.time()
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,
)
elapsed_min = (time.time() - start_time) / 60
print(f"\n  Training complete in {elapsed_min:.1f} minutes")
print(f"  Best iteration: {model.get_best_iteration()}")
print(f"  Best PRAUC (val): {model.get_best_score()['validation']['PRAUC']:.4f}")

# =====================================================================
# 5. Evaluate on test set
# =====================================================================
print(f"\n[4/5] Evaluating on test set...")
print(f"  Test set: {len(y_test):,} compounds ({y_test.sum()} actives)")

# Predict probabilities on test set
y_test_proba = model.predict_proba(test_pool)[:, 1]

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

# BEDROC α=20
def compute_bedroc(y_true, y_pred, alpha=20.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    n_act = int(y_sorted.sum())
    if n_act == 0:
        return 0.5
    weights = np.exp(-alpha * np.arange(n) / n)
    R = (weights * y_sorted).sum() / (weights.sum() * n_act / n)
    ra = n_act / n
    return float(R * ra * np.sinh(alpha / n) / (np.cosh(alpha / n) - 1) +
                  (1 - ra) / (np.cosh(alpha / n) - 1) * (np.sinh(alpha / n) - alpha / n * np.cosh(alpha / n)))

bedroc_score = compute_bedroc(y_test, y_test_proba, alpha=20.0)

# Precision at recall levels
prec, rec, thresholds = precision_recall_curve(y_test, y_test_proba)
precision_at_recall = {}
for target_recall in [0.50, 0.75, 0.90]:
    idx = np.argmax(rec >= target_recall)
    if idx < len(prec):
        precision_at_recall[f'precision_at_recall_{target_recall:.2f}'] = float(prec[idx])

print(f"\n  === TEST SET PERFORMANCE (CatBoost) ===")
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
model_path = os.path.join(OUTPUT_DIR, 'catboost_model.cbm')
model.save_model(model_path)
print(f"  Saved {model_path}")

# Save test predictions (sorted by probability descending)
metadata_test = pd.read_csv(os.path.join(STAGE3_DIR, 'metadata_test.csv'))
test_pred_df = metadata_test.copy()
test_pred_df['catboost_proba'] = y_test_proba
test_pred_df = test_pred_df.sort_values('catboost_proba', ascending=False).reset_index(drop=True)
test_pred_df['rank'] = test_pred_df.index + 1
test_pred_path = os.path.join(OUTPUT_DIR, 'test_predictions_catboost.csv')
test_pred_df.to_csv(test_pred_path, index=False)
print(f"  Saved {test_pred_path} ({len(test_pred_df):,} rows)")

# Also save val predictions (for Stage 5 analysis)
y_val_proba = model.predict_proba(val_pool)[:, 1]
metadata_val = pd.read_csv(os.path.join(STAGE3_DIR, 'metadata_val.csv'))
val_pred_df = metadata_val.copy()
val_pred_df['catboost_proba'] = y_val_proba
val_pred_path = os.path.join(OUTPUT_DIR, 'val_predictions_catboost.csv')
val_pred_df.to_csv(val_pred_path, index=False)
print(f"  Saved {val_pred_path} ({len(val_pred_df):,} rows)")

# Save metrics JSON
metrics = {
    'model': 'catboost',
    'pr_auc': float(pr_auc),
    'roc_auc': float(roc_auc),
    'bedroc_alpha20': float(bedroc_score),
    'brier': float(brier),
    'ef1': float(ef1),
    'ef5': float(ef5),
    'n_test': len(y_test),
    'n_test_actives': int(y_test.sum()),
    'training_time_minutes': float(elapsed_min),
    'best_iteration': int(model.get_best_iteration()) if model.get_best_iteration() is not None else 999,
    'best_val_prauc': float(model.get_best_score()['validation']['PRAUC']),
    **{k: float(v) for k, v in precision_at_recall.items()},
}
metrics_path = os.path.join(OUTPUT_DIR, 'catboost_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# =====================================================================
# 7. Top-N predicted actives (sanity check)
# =====================================================================
print(f"\n=== TOP-20 PREDICTED ACTIVES (CatBoost) ===")
print(f"\n  {'Rank':<6}{'Prob':<10}{'True':<10}{'SMILES':<60}")
print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*55}")
smiles_col = 'canonical_smiles' if 'canonical_smiles' in test_pred_df.columns else 'smiles'
for i in range(min(20, len(test_pred_df))):
    row = test_pred_df.iloc[i]
    smi = str(row[smiles_col])[:55] + '...' if len(str(row[smiles_col])) > 55 else str(row[smiles_col])
    true_label = 'ACTIVE' if row['label'] == 1 else 'inactive'
    print(f"  {i+1:<6}{row['catboost_proba']:<10.4f}{true_label:<10}{smi}")

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
feature_importance = model.get_feature_importance()
top_indices = np.argsort(-feature_importance)[:20]
print(f"\n  {'Rank':<6}{'Bit #':<10}{'Importance':<15}")
print(f"  {'-'*5}  {'-'*8}  {'-'*13}")
for rank, idx in enumerate(top_indices):
    print(f"  {rank+1:<6}{int(idx):<10}{feature_importance[idx]:<15.4f}")

# =====================================================================
# 9. Save report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'catboost_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 4b CATBOOST REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("ALGORITHM\n")
    f.write(f"  CatBoost (Categorical Boosting)\n")
    f.write(f"  Tree type: Symmetric / oblivious\n")
    f.write(f"  task_type: CPU (macOS)\n\n")
    f.write("HYPERPARAMETERS (per PriA-SSB spec)\n")
    f.write(f"  iterations:         1000\n")
    f.write(f"  depth:               6\n")
    f.write(f"  learning_rate:       0.03\n")
    f.write(f"  l2_leaf_reg:         3.0\n")
    f.write(f"  subsample:           0.8\n")
    f.write(f"  loss_function:       Logloss\n")
    f.write(f"  eval_metric:         PRAUC\n")
    f.write(f"  auto_class_weights:  Balanced\n")
    f.write(f"  early_stopping:      50 rounds\n\n")
    f.write("TRAINING\n")
    f.write(f"  Train size: {X_train.shape[0]:,}\n")
    f.write(f"  Val size:   {X_val.shape[0]:,}\n")
    f.write(f"  Training time: {elapsed_min:.1f} minutes\n")
    f.write(f"  Best iteration: {model.get_best_iteration()}\n")
    f.write(f"  Best val PRAUC: {model.get_best_score()['validation']['PRAUC']:.4f}\n\n")
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
print("STAGE 4b COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - catboost_model.cbm               (trained model)")
print(f"  - test_predictions_catboost.csv     (for Stage 5 consensus)")
print(f"  - val_predictions_catboost.csv      (for Stage 5 analysis)")
print(f"  - catboost_metrics.json")
print(f"  - catboost_report.txt")
print(f"\nReady for Stage 5 (consensus fusion) once Chemprop (4a) finishes.")
