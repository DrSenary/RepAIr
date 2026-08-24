"""
ATAD5AI — Stage 5: Hybrid Consensus Fusion
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 5: Hybrid Consensus Fusion Engine
# =====================================================================
# INPUT:
#   outputs/stage4_models/test_predictions_chemprop.csv    (D-MPNN)
#   outputs/stage4_models/test_predictions_catboost.csv     (CatBoost)
#   outputs/stage4_models/test_predictions_balanced_rf.csv (Balanced RF)
#   outputs/stage3_split/y_test.npy, metadata_test.csv
#
# OUTPUT:
#   outputs/stage5_consensus/
#     - test_predictions_consensus.csv   (final ranked predictions)
#     - consensus_metrics.json           (full metric suite)
#     - per_model_comparison.csv         (single-model vs consensus)
#     - consensus_report.txt
#
# MODULE 5 SPEC:
#   For every test compound, extract individual predicted probabilities:
#     P_chemprop, P_catboost, P_rf in range [0.0, 1.0]
#   Compute Consensus Metrics:
#     1. Soft-Voting Probability: P_consensus = (P_chemprop + P_catboost + P_rf) / 3.0
#     2. Model Disagreement Variance: Var = σ²(P_chemprop, P_catboost, P_rf)
#     3. Percentile Rank Score: Geometric mean of individual model percentile ranks
#   Confidence Filter:
#     Retain hits with P_consensus >= threshold (top 1%) AND Var < 0.04
#     (discard high-variance false positives where models diverge)
# =====================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, gmean
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)

# =====================================================================
# CONFIG
# =====================================================================
STAGE4_DIR = 'outputs/stage4_models'
STAGE3_DIR = 'outputs/stage3_split'
OUTPUT_DIR = 'outputs/stage5_consensus'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Confidence filter thresholds (per spec)
TOP_PERCENTILE_THRESHOLD = 0.01   # Top 1% by P_consensus
VARIANCE_THRESHOLD = 0.04         # Discard high-variance predictions

print("=" * 78)
print("ATAD5AI — STAGE 5: HYBRID CONSENSUS FUSION")
print("=" * 78)

# =====================================================================
# 1. Load all three model predictions
# =====================================================================
print(f"\n[1/5] Loading predictions from three models...")

# Load Chemprop predictions
chemprop_path = os.path.join(STAGE4_DIR, 'test_predictions_chemprop.csv')
if not os.path.exists(chemprop_path):
    print(f"  ✗ {chemprop_path} not found!")
    print(f"  Make sure you've downloaded test_predictions_chemprop.csv from Colab")
    raise FileNotFoundError(chemprop_path)
df_chemprop = pd.read_csv(chemprop_path)
print(f"  ✓ Chemprop:  {len(df_chemprop):,} predictions")
# Find the probability column (might be 'chemprop_proba' or 'predicted_proba')
prob_col_cp = [c for c in df_chemprop.columns if 'chemprop' in c.lower() and 'prob' in c.lower()]
if not prob_col_cp:
    prob_col_cp = [c for c in df_chemprop.columns if 'prob' in c.lower()]
prob_col_cp = prob_col_cp[0]
print(f"    Probability column: '{prob_col_cp}'")

# Load CatBoost predictions
catboost_path = os.path.join(STAGE4_DIR, 'test_predictions_catboost.csv')
df_catboost = pd.read_csv(catboost_path)
print(f"  ✓ CatBoost:  {len(df_catboost):,} predictions")
prob_col_cb = [c for c in df_catboost.columns if 'catboost' in c.lower() and 'prob' in c.lower()][0]
print(f"    Probability column: '{prob_col_cb}'")

# Load Balanced RF predictions
rf_path = os.path.join(STAGE4_DIR, 'test_predictions_balanced_rf.csv')
df_rf = pd.read_csv(rf_path)
print(f"  ✓ Balanced RF: {len(df_rf):,} predictions")
prob_col_rf = [c for c in df_rf.columns if 'balanced_rf' in c.lower() and 'prob' in c.lower()][0]
print(f"    Probability column: '{prob_col_rf}'")

# Load ground truth
y_test = np.load(os.path.join(STAGE3_DIR, 'y_test.npy'))
print(f"\n  Ground truth: {len(y_test):,} compounds ({y_test.sum()} actives)")

# =====================================================================
# 2. Align predictions by SMILES (handles potential row-order mismatches)
# =====================================================================
print(f"\n[2/5] Aligning predictions by SMILES...")

# Determine which column to merge on — prefer canonical_smiles if available
smiles_col = 'canonical_smiles' if 'canonical_smiles' in df_chemprop.columns else 'smiles'
print(f"  Merging on column: '{smiles_col}'")

# Build a unified DataFrame indexed by SMILES
merged = df_chemprop[[smiles_col, prob_col_cp, 'label']].rename(columns={prob_col_cp: 'p_chemprop'}).copy()
merged = merged.merge(
    df_catboost[[smiles_col, prob_col_cb]].rename(columns={prob_col_cb: 'p_catboost'}),
    on=smiles_col, how='inner'
)
merged = merged.merge(
    df_rf[[smiles_col, prob_col_rf]].rename(columns={prob_col_rf: 'p_rf'}),
    on=smiles_col, how='inner'
)

print(f"  Merged dataset: {len(merged):,} compounds with all 3 predictions")
print(f"  Active count:   {merged['label'].sum()}")

# Verify labels match ground truth
assert len(merged) == len(y_test), f"Length mismatch: merged={len(merged)}, y_test={len(y_test)}"
assert (merged['label'].values == y_test).all(), "Label mismatch between predictions and ground truth!"
print(f"  ✓ Labels match ground truth")

# =====================================================================
# 3. Compute consensus metrics (per Module 5 spec)
# =====================================================================
print(f"\n[3/5] Computing consensus metrics...")

p_chemprop = merged['p_chemprop'].values
p_catboost = merged['p_catboost'].values
p_rf = merged['p_rf'].values

# 1. Soft-Voting Probability (arithmetic mean)
merged['p_consensus'] = (p_chemprop + p_catboost + p_rf) / 3.0

# 2. Model Disagreement Variance (population variance, σ²)
# Var = (1/3) * Σ(p_i - mean)² for i ∈ {chemprop, catboost, rf}
mean_p = (p_chemprop + p_catboost + p_rf) / 3.0
merged['model_variance'] = ((p_chemprop - mean_p)**2 +
                              (p_catboost - mean_p)**2 +
                              (p_rf - mean_p)**2) / 3.0

# 3. Percentile Rank Score (geometric mean of individual model percentile ranks)
# Each model's percentile rank is computed independently, then geomean
rank_chemprop = rankdata(p_chemprop, method='average') / len(p_chemprop)
rank_catboost = rankdata(p_catboost, method='average') / len(p_catboost)
rank_rf = rankdata(p_rf, method='average') / len(p_rf)

# Geometric mean of ranks (add small epsilon to avoid log(0))
eps = 1e-10
merged['percentile_rank_geomean'] = gmean(np.clip(
    np.stack([rank_chemprop, rank_catboost, rank_rf], axis=1) + eps,
    1e-10, 1.0
))

print(f"  Computed 3 consensus metrics:")
print(f"    • P_consensus (soft-voting mean)")
print(f"    • Model disagreement variance (σ²)")
print(f"    • Percentile rank geometric mean")

# Stats
print(f"\n  P_consensus distribution:")
print(f"    Min:    {merged['p_consensus'].min():.4f}")
print(f"    Median: {merged['p_consensus'].median():.4f}")
print(f"    Max:    {merged['p_consensus'].max():.4f}")
print(f"    Mean:   {merged['p_consensus'].mean():.4f}")
print(f"\n  Model variance distribution:")
print(f"    Min:    {merged['model_variance'].min():.6f}")
print(f"    Median: {merged['model_variance'].median():.6f}")
print(f"    Max:    {merged['model_variance'].max():.6f}")
print(f"    Mean:   {merged['model_variance'].mean():.6f}")

# How many compounds have high variance (model disagreement)?
n_high_var = (merged['model_variance'] > VARIANCE_THRESHOLD).sum()
print(f"\n  High-variance predictions (Var > {VARIANCE_THRESHOLD}): {n_high_var:,} ({n_high_var/len(merged)*100:.1f}%)")
print(f"  Low-variance predictions (Var ≤ {VARIANCE_THRESHOLD}): {len(merged) - n_high_var:,} ({(len(merged) - n_high_var)/len(merged)*100:.1f}%)")

# =====================================================================
# 4. Helper functions
# =====================================================================
def enrichment_factor(y_true, y_pred, fraction):
    n_total = len(y_true)
    n_top = max(1, int(n_total * fraction))
    n_actives_total = int(y_true.sum())
    if n_actives_total == 0:
        return 0.0
    ranked = np.argsort(-y_pred)
    n_actives_top = int(y_true[ranked[:n_top]].sum())
    return n_actives_top / max(n_actives_total * fraction, 1e-10)

def compute_bedroc(y_true, y_pred, alpha=20.0):
    """Correct BEDROC implementation (bounded [0, 1], 0.5 = random)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    n = len(y_sorted)
    n_act = int(y_sorted.sum())
    if n_act == 0:
        return 0.5
    i = np.arange(1, n + 1)
    w = np.exp(-alpha * i / n)
    sum_w_act = float((w * y_sorted).sum())
    sum_w_total = float(w.sum())
    ra = n_act / n
    expected_sum_w_act = ra * sum_w_total
    R = sum_w_act / max(expected_sum_w_act, 1e-10)
    bedroc = R * ra * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha * ra))
    return float(min(bedroc, 1.0))

def compute_all_metrics(y_true, y_pred, prefix=''):
    metrics = {}
    metrics[f'{prefix}pr_auc'] = float(average_precision_score(y_true, y_pred))
    metrics[f'{prefix}roc_auc'] = float(roc_auc_score(y_true, y_pred))
    metrics[f'{prefix}bedroc_alpha20'] = compute_bedroc(y_true, y_pred, 20.0)
    metrics[f'{prefix}brier'] = float(brier_score_loss(y_true, y_pred))
    metrics[f'{prefix}ef1'] = float(enrichment_factor(y_true, y_pred, 0.01))
    metrics[f'{prefix}ef5'] = float(enrichment_factor(y_true, y_pred, 0.05))
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    for target_recall in [0.50, 0.75, 0.90]:
        idx = np.argmax(rec >= target_recall)
        if idx < len(prec):
            metrics[f'{prefix}precision_at_recall_{target_recall:.2f}'] = float(prec[idx])
    return metrics

# =====================================================================
# 5. Evaluate per-model AND consensus on test set
# =====================================================================
print(f"\n[4/5] Evaluating per-model and consensus metrics...")

y_true = merged['label'].values.astype(int)

# Per-model metrics
metrics_chemprop = compute_all_metrics(y_true, p_chemprop, prefix='chemprop_')
metrics_catboost = compute_all_metrics(y_true, p_catboost, prefix='catboost_')
metrics_rf = compute_all_metrics(y_true, p_rf, prefix='rf_')
metrics_consensus = compute_all_metrics(y_true, merged['p_consensus'].values, prefix='consensus_')

# Also evaluate the percentile rank geomean as an alternative consensus
metrics_geomean = compute_all_metrics(y_true, merged['percentile_rank_geomean'].values, prefix='geomean_')

# Print comparison table
print(f"\n  {'Model':<20}{'PR-AUC':<10}{'ROC-AUC':<10}{'BEDROC':<10}{'EF@1%':<10}{'EF@5%':<10}{'Brier':<10}")
print(f"  {'-'*19}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for name, m in [
    ('Chemprop D-MPNN', metrics_chemprop),
    ('CatBoost', metrics_catboost),
    ('Balanced RF', metrics_rf),
    ('CONSENSUS (mean)', metrics_consensus),
    ('Geomean (rank)', metrics_geomean),
]:
    print(f"  {name:<20}{m[list(m.keys())[0]]:<10.4f}{m[list(m.keys())[1]]:<10.4f}"
          f"{m[list(m.keys())[2]]:<10.4f}{m[list(m.keys())[4]]:<10.2f}"
          f"{m[list(m.keys())[5]]:<10.2f}{m[list(m.keys())[3]]:<10.4f}")

# =====================================================================
# 6. Apply confidence filter (per spec)
# =====================================================================
print(f"\n=== CONFIDENCE FILTER (per Module 5 spec) ===")
print(f"  Filter: P_consensus in top {TOP_PERCENTILE_THRESHOLD*100:.0f}% AND Var < {VARIANCE_THRESHOLD}")

# Top 1% by consensus probability
n_total = len(merged)
top_1pct_n = max(1, int(n_total * TOP_PERCENTILE_THRESHOLD))
print(f"\n  Top 1% of test set: {top_1pct_n} compounds")

# Sort by consensus probability
merged_sorted = merged.sort_values('p_consensus', ascending=False).reset_index(drop=True)
top_1pct = merged_sorted.head(top_1pct_n).copy()

# Among top 1%, apply variance filter
n_top1_before = len(top_1pct)
n_actives_top1_before = int(top_1pct['label'].sum())
top_1pct_filtered = top_1pct[top_1pct['model_variance'] < VARIANCE_THRESHOLD].copy()
n_top1_after = len(top_1pct_filtered)
n_actives_top1_after = int(top_1pct_filtered['label'].sum())

print(f"\n  BEFORE variance filter (top 1% by P_consensus):")
print(f"    Compounds: {n_top1_before}")
print(f"    Actives:   {n_actives_top1_before}")
print(f"    Precision: {n_actives_top1_before/n_top1_before*100:.1f}%")
print(f"\n  AFTER variance filter (Var < {VARIANCE_THRESHOLD}):")
print(f"    Compounds: {n_top1_after} (filtered out {n_top1_before - n_top1_after})")
print(f"    Actives:   {n_actives_top1_after}")
if n_top1_after > 0:
    print(f"    Precision: {n_actives_top1_after/n_top1_after*100:.1f}%")
    print(f"    → Variance filter {'IMPROVED' if n_actives_top1_after/n_top1_after > n_actives_top1_before/n_top1_before else 'did not improve'} precision")
else:
    print(f"    (no compounds passed the variance filter)")

# =====================================================================
# 7. Save outputs
# =====================================================================
print(f"\n[5/5] Saving outputs...")

# Save full consensus predictions (sorted by P_consensus)
output_predictions = merged_sorted.copy()
output_predictions['rank'] = output_predictions.index + 1
output_predictions = output_predictions[[
    smiles_col, 'label',
    'p_chemprop', 'p_catboost', 'p_rf',
    'p_consensus', 'model_variance', 'percentile_rank_geomean',
    'rank'
]]
pred_path = os.path.join(OUTPUT_DIR, 'test_predictions_consensus.csv')
output_predictions.to_csv(pred_path, index=False)
print(f"  Saved {pred_path} ({len(output_predictions):,} rows)")

# Per-model comparison CSV
comparison_data = []
for name, m in [
    ('chemprop_dmpnn', metrics_chemprop),
    ('catboost', metrics_catboost),
    ('balanced_rf', metrics_rf),
    ('consensus_mean', metrics_consensus),
    ('consensus_geomean', metrics_geomean),
]:
    row = {'model': name}
    row.update(m)
    comparison_data.append(row)
comparison_df = pd.DataFrame(comparison_data)
comparison_path = os.path.join(OUTPUT_DIR, 'per_model_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"  Saved {comparison_path}")

# Save metrics JSON (full consensus metrics)
metrics_path = os.path.join(OUTPUT_DIR, 'consensus_metrics.json')
all_metrics = {
    'chemprop': metrics_chemprop,
    'catboost': metrics_catboost,
    'balanced_rf': metrics_rf,
    'consensus_mean': metrics_consensus,
    'consensus_geomean': metrics_geomean,
    'confidence_filter': {
        'top_percentile': TOP_PERCENTILE_THRESHOLD,
        'variance_threshold': VARIANCE_THRESHOLD,
        'n_top1_before': int(n_top1_before),
        'n_actives_top1_before': int(n_actives_top1_before),
        'precision_top1_before': float(n_actives_top1_before/n_top1_before),
        'n_top1_after': int(n_top1_after),
        'n_actives_top1_after': int(n_actives_top1_after),
        'precision_top1_after': float(n_actives_top1_after/n_top1_after) if n_top1_after > 0 else 0.0,
    }
}
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# =====================================================================
# 8. Top-20 consensus predictions (sanity check)
# =====================================================================
print(f"\n=== TOP-20 CONSENSUS PREDICTIONS ===")
print(f"\n  {'Rk':<4}{'P_cons':<8}{'Var':<8}{'P_chem':<8}{'P_cat':<8}{'P_rf':<8}{'True':<8}{'SMILES':<55}")
print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*50}")
for i in range(min(20, len(output_predictions))):
    row = output_predictions.iloc[i]
    smi = str(row[smiles_col])[:50] + '...' if len(str(row[smiles_col])) > 50 else str(row[smiles_col])
    true_label = 'ACTIVE' if row['label'] == 1 else 'inactive'
    print(f"  {i+1:<4}{row['p_consensus']:<8.4f}{row['model_variance']:<8.4f}"
          f"{row['p_chemprop']:<8.4f}{row['p_catboost']:<8.4f}{row['p_rf']:<8.4f}"
          f"{true_label:<8}{smi}")

# =====================================================================
# 9. Save report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'consensus_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 5 CONSENSUS REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("CONSENSUS STRATEGY (per PriA-SSB spec)\n")
    f.write(f"  1. Soft-Voting Probability: P_consensus = (P_chemprop + P_catboost + P_rf) / 3.0\n")
    f.write(f"  2. Model Disagreement Variance: Var = σ²(3 predictions)\n")
    f.write(f"  3. Percentile Rank Geometric Mean\n\n")
    f.write("CONFIDENCE FILTER\n")
    f.write(f"  Threshold: top {TOP_PERCENTILE_THRESHOLD*100:.0f}% by P_consensus AND Var < {VARIANCE_THRESHOLD}\n\n")
    f.write("PER-MODEL vs CONSENSUS PERFORMANCE (test set)\n")
    f.write(f"  {'Model':<20}{'PR-AUC':<10}{'ROC-AUC':<10}{'BEDROC':<10}{'EF@1%':<10}{'EF@5%':<10}\n")
    for name, m in [
        ('Chemprop D-MPNN', metrics_chemprop),
        ('CatBoost', metrics_catboost),
        ('Balanced RF', metrics_rf),
        ('CONSENSUS (mean)', metrics_consensus),
        ('Geomean (rank)', metrics_geomean),
    ]:
        f.write(f"  {name:<20}{m['pr_auc']:<10.4f}{m['roc_auc']:<10.4f}"
                f"{m['bedroc_alpha20']:<10.4f}{m['ef1']:<10.2f}{m['ef5']:<10.2f}\n")
    f.write(f"\nCONFIDENCE FILTER RESULTS (top-1%)\n")
    f.write(f"  Before: {n_top1_before} compounds, {n_actives_top1_before} actives, precision={n_actives_top1_before/n_top1_before*100:.1f}%\n")
    if n_top1_after > 0:
        f.write(f"  After:  {n_top1_after} compounds, {n_actives_top1_after} actives, precision={n_actives_top1_after/n_top1_after*100:.1f}%\n")
    f.write(f"\nNEXT STEPS\n")
    f.write(f"  Stage 6: FDA drug repurposing screen using consensus model\n")
    f.write(f"  Stage 7: External validation on AID 493107 + counter-screens\n")
    f.write(f"  Stage 8: Safety triage (PAINS, luciferase inhibitor, Lipinski/Veber)\n")
print(f"  Saved {report_path}")

print(f"\n" + "=" * 78)
print("STAGE 5 COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - test_predictions_consensus.csv   (final ranked predictions)")
print(f"  - consensus_metrics.json            (full metrics)")
print(f"  - per_model_comparison.csv          (single-model vs consensus)")
print(f"  - consensus_report.txt")
print(f"\nReady for Stage 6 (FDA repurposing screen).")
