"""
ATAD5AI — Stage 6: Hybrid Consensus FDA Drug Repurposing Screen (Google Colab)

Hybrid Consensus Screening Pipeline — Final Stage of Repurposing

Screens FDA-approved drugs (from CLUE Repurposing Hub) with the trained
3-model consensus (Chemprop D-MPNN + CatBoost + Balanced RF) to identify
ELG1/ATAD5 DNA repair inhibition repurposing candidates.

==============================================================================
HOW TO USE (on Google Colab with T4 GPU)
==============================================================================

PREREQUISITES (4 files to upload):
  1. fda_library/fda_approved.csv                (~680 KB, from filter_fda_library.py)
  2. outputs/stage4_models/catboost_model.cbm     (~5 MB, from Stage 4b)
  3. outputs/stage4_models/balanced_rf_model.joblib  (~50 MB, from Stage 4c)
  4. outputs/stage4_models/chemprop_best_model.ckpt  (~50 MB, from Stage 4a)
     (Only needed if you're on a new Colab session — if Stage 4a session
      is still active, the checkpoint is already there)

STEPS:
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. In a cell, run: !pip install catboost imbalanced-learn chemprop lightning rdkit
  4. Paste this entire script into a new cell
  5. Press Shift+Enter
  6. When prompted, upload the 4 files (script will guide you)
  7. Wait ~6 minutes for screening to complete
  8. Download the outputs (instructions printed at end)

EXPECTED RUNTIME: ~6 minutes total (after uploads)

==============================================================================
MODULE 6 SPEC (Confidence Filter for Top FDA Candidates)
==============================================================================
  For every screening compound, extract individual predicted probabilities:
    P_chemprop, P_catboost, P_rf in range [0.0, 1.0]
  Compute Consensus Metrics:
    1. Soft-Voting Probability: P_consensus = (P_chemprop + P_catboost + P_rf) / 3.0
    2. Model Disagreement Variance: Var = σ²(P_chemprop, P_catboost, P_rf)
    3. Percentile Rank Score: Geometric mean of individual model percentile ranks
  Confidence Filter:
    Retain hits with P_consensus >= threshold (top 5%) AND Var < 0.04
    (discard high-variance false positives where models diverge)

==============================================================================
OUTPUTS
==============================================================================
  /content/data/fda_predictions/
    - fda_consensus_predictions.csv   (all FDA drugs, ranked by P_consensus)
    - fda_top_candidates.csv          (top 5% with Var < 0.04 — confidence filtered)
"""

# =====================================================================
# STEP 1: Install dependencies (ACTIVE — runs automatically)
# =====================================================================
!pip install catboost imbalanced-learn chemprop lightning rdkit 2>&1 | tail -3

# =====================================================================
# STEP 2: Upload files (or skip if already uploaded)
# =====================================================================
import os
import glob
import json
import time
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import rankdata, gmean
import joblib

DATA_DIR = '/content/data'
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 78)
print("ATAD5AI — STAGE 6: FDA REPURPOSING SCREEN (3-MODEL CONSENSUS)")
print("=" * 78)

# Upload required files (skips if already uploaded)
from google.colab import files

required_files = [
    ('fda_approved.csv', 'FDA library (2,680 drugs with SMILES)'),
    ('catboost_model.cbm', 'Trained CatBoost model from Stage 4b'),
    ('balanced_rf_model.joblib', 'Trained Balanced RF model from Stage 4c'),
]
for fname, description in required_files:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {fname} ({size_mb:.2f} MB) — already uploaded")
    else:
        print(f"\n  ⚠ Upload {fname}")
        print(f"    Description: {description}")
        uploaded = files.upload()
        for uploaded_name in uploaded.keys():
            os.rename(uploaded_name, os.path.join(DATA_DIR, uploaded_name))
            size_mb = os.path.getsize(os.path.join(DATA_DIR, uploaded_name)) / (1024 * 1024)
            print(f"  ✓ Saved {uploaded_name} ({size_mb:.2f} MB)")

# Find Chemprop checkpoint — upload if missing
print(f"\n  Checking for Chemprop checkpoint...")
chemprop_ckpt_candidates = [
    '/content/data/chemprop_outputs/chemprop_best_model.ckpt',
    '/content/data/chemprop_best_model.ckpt',
] + sorted(glob.glob('/content/data/chemprop_outputs/checkpoints/*.ckpt'))
chemprop_ckpt_path = next((p for p in chemprop_ckpt_candidates if os.path.exists(p)), None)

if chemprop_ckpt_path is None:
    print(f"  ⚠ Chemprop checkpoint not found. Upload chemprop_best_model.ckpt")
    ckpt_target = os.path.join(DATA_DIR, 'chemprop_best_model.ckpt')
    if not os.path.exists(ckpt_target):
        uploaded = files.upload()
        for uploaded_name in uploaded.keys():
            os.rename(uploaded_name, ckpt_target)
            size_mb = os.path.getsize(ckpt_target) / (1024 * 1024)
            print(f"  ✓ Saved chemprop checkpoint ({size_mb:.2f} MB)")
    chemprop_ckpt_path = ckpt_target
else:
    print(f"  ✓ Found Chemprop checkpoint at: {chemprop_ckpt_path}")

# Verify GPU
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =====================================================================
# STEP 3: Load FDA library
# =====================================================================
print(f"\n[1/6] Loading FDA library...")

df_fda = pd.read_csv(os.path.join(DATA_DIR, 'fda_approved.csv'))
print(f"  Loaded {len(df_fda):,} FDA-approved drugs")

# Find columns
smiles_col = 'SMILES' if 'SMILES' in df_fda.columns else 'canonical_smiles'
drug_name_col = 'Drug_Name' if 'Drug_Name' in df_fda.columns else df_fda.columns[0]
print(f"  SMILES column: '{smiles_col}'")
print(f"  Drug_Name column: '{drug_name_col}'")

# Drop rows with missing SMILES
df_fda = df_fda.dropna(subset=[smiles_col]).reset_index(drop=True)
print(f"  After dropping missing SMILES: {len(df_fda):,}")

# =====================================================================
# STEP 4: Generate ECFP6 features for CatBoost + RF
# =====================================================================
print(f"\n[2/6] Generating ECFP6 count vectors for {len(df_fda):,} drugs...")

N_BITS = 4096
RADIUS = 3  # ECFP6 = radius 3
DTYPE = np.uint16

def smiles_to_ecfp6_counts(smiles):
    """Convert SMILES to ECFP6 count vector (4096-bit, radius=3)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=DTYPE)
    for bit, count in fp.GetNonzeroElements().items():
        arr[bit] = min(count, 65535)  # uint16 max
    return arr

# Build sparse matrix
rows, cols, vals = [], [], []
valid_indices = []
n_failed = 0

for i, smi in enumerate(df_fda[smiles_col]):
    fp = smiles_to_ecfp6_counts(smi)
    if fp is None:
        n_failed += 1
        continue
    valid_indices.append(i)
    nz_indices = np.nonzero(fp)[0]
    nz_values = fp[nz_indices]
    rows.extend([len(valid_indices) - 1] * len(nz_indices))
    cols.extend(nz_indices.tolist())
    vals.extend(nz_values.tolist())

print(f"  Failed: {n_failed}")

X_fda = sp.coo_matrix(
    (np.array(vals, dtype=DTYPE),
     (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
    shape=(len(valid_indices), N_BITS),
    dtype=DTYPE
).tocsr().astype(np.float32)  # CatBoost/RF expect float

# Filter FDA DataFrame to valid drugs
df_fda_valid = df_fda.iloc[valid_indices].reset_index(drop=True)
print(f"  Valid drugs for screening: {len(df_fda_valid):,}")
print(f"  Sparse matrix shape: {X_fda.shape}")
print(f"  Average bits set per drug: {X_fda.getnnz() / X_fda.shape[0]:.1f}")

# =====================================================================
# STEP 5: Load trained models
# =====================================================================
print(f"\n[3/6] Loading trained models...")

# CatBoost
print(f"  Loading CatBoost model...")
from catboost import CatBoostClassifier, Pool
catboost_model = CatBoostClassifier()
catboost_model.load_model(os.path.join(DATA_DIR, 'catboost_model.cbm'))
print(f"    ✓ CatBoost loaded")

# Balanced RF
print(f"  Loading Balanced RF model...")
rf_model = joblib.load(os.path.join(DATA_DIR, 'balanced_rf_model.joblib'))
print(f"    ✓ Balanced RF loaded")

# Chemprop D-MPNN
print(f"  Loading Chemprop model...")
from chemprop.models import MPNN
import lightning
chemprop_model = MPNN.load_from_checkpoint(chemprop_ckpt_path)
chemprop_model = chemprop_model.to('cuda')
chemprop_model.eval()
print(f"    ✓ Chemprop loaded from {chemprop_ckpt_path}")

# =====================================================================
# STEP 6: Run predictions on FDA drugs
# =====================================================================
print(f"\n[4/6] Running predictions on {len(df_fda_valid):,} FDA drugs...")

# --- CatBoost ---
print(f"  CatBoost...")
t0 = time.time()
fda_pool = Pool(X_fda, feature_names=[f'ecfp6_bit_{i}' for i in range(X_fda.shape[1])])
p_catboost = catboost_model.predict_proba(fda_pool)[:, 1]
print(f"    ✓ {len(p_catboost)} predictions in {time.time()-t0:.1f}s")
print(f"      Range: [{p_catboost.min():.4f}, {p_catboost.max():.4f}], mean={p_catboost.mean():.4f}")

# --- Balanced RF ---
print(f"  Balanced RF...")
t0 = time.time()
p_rf = rf_model.predict_proba(X_fda)[:, 1]
print(f"    ✓ {len(p_rf)} predictions in {time.time()-t0:.1f}s")
print(f"      Range: [{p_rf.min():.4f}, {p_rf.max():.4f}], mean={p_rf.mean():.4f}")

# --- Chemprop D-MPNN ---
print(f"  Chemprop D-MPNN...")
t0 = time.time()

from chemprop import data as cdata
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

# Convert SMILES to RDKit Mols (filter invalid)
fda_smiles_list = df_fda_valid[smiles_col].tolist()
fda_mols = []
fda_smiles_valid = []
for s in fda_smiles_list:
    mol = Chem.MolFromSmiles(str(s))
    if mol is not None:
        fda_mols.append(mol)
        fda_smiles_valid.append(s)
n_chemprop_valid = len(fda_mols)
print(f"    Valid mols for Chemprop: {n_chemprop_valid} / {len(fda_smiles_list)}")

# Build datapoints (dummy target=0 — we just want predictions)
fda_datapoints = [cdata.MoleculeDatapoint(mol=m, y=np.array([0.0], dtype=np.float32))
                  for m in fda_mols]
featurizer = SimpleMoleculeMolGraphFeaturizer()
fda_dataset = cdata.MoleculeDataset(fda_datapoints, featurizer)
fda_loader = cdata.build_dataloader(fda_dataset, batch_size=64, shuffle=False)

# Predict using Lightning trainer.predict()
predict_trainer = lightning.Trainer(
    accelerator='gpu',
    devices=1,
    enable_progress_bar=True,
    logger=False,
)
predictions_list = predict_trainer.predict(chemprop_model, fda_loader)
p_chemprop_valid = torch.cat([p.flatten() for p in predictions_list]).cpu().numpy()

# Map Chemprop predictions back to FDA DataFrame
p_chemprop = np.full(len(df_fda_valid), np.nan)
valid_smiles_to_pred = {smi: pred for smi, pred in zip(fda_smiles_valid, p_chemprop_valid)}
for i, smi in enumerate(df_fda_valid[smiles_col]):
    pred = valid_smiles_to_pred.get(str(smi))
    if pred is not None:
        p_chemprop[i] = pred

print(f"    ✓ {n_chemprop_valid} predictions in {time.time()-t0:.1f}s")
print(f"      Range: [{np.nanmin(p_chemprop):.4f}, {np.nanmax(p_chemprop):.4f}], mean={np.nanmean(p_chemprop):.4f}")

# =====================================================================
# STEP 7: Compute consensus + variance + percentile rank geomean
# =====================================================================
print(f"\n[5/6] Computing consensus metrics...")

pred_df = df_fda_valid.copy()
pred_df['p_catboost'] = p_catboost
pred_df['p_rf'] = p_rf
pred_df['p_chemprop'] = p_chemprop

# Drop drugs where Chemprop failed
n_before = len(pred_df)
pred_df = pred_df.dropna(subset=['p_chemprop']).reset_index(drop=True)
print(f"  Dropped {n_before - len(pred_df)} drugs with failed Chemprop predictions")
print(f"  Final screening set: {len(pred_df):,} FDA drugs with all 3 predictions")

# 1. Soft-voting consensus (arithmetic mean of 3 probabilities)
pred_df['p_consensus'] = (
    pred_df['p_chemprop'] + pred_df['p_catboost'] + pred_df['p_rf']
) / 3.0

# 2. Model disagreement variance (population variance, σ²)
mean_p = pred_df['p_consensus'].values
pred_df['model_variance'] = (
    (pred_df['p_chemprop'].values - mean_p)**2 +
    (pred_df['p_catboost'].values - mean_p)**2 +
    (pred_df['p_rf'].values - mean_p)**2
) / 3.0

# 3. Percentile rank geometric mean
# (rank of each compound within each model's predictions, then geomean across models)
rank_chemprop = rankdata(pred_df['p_chemprop'].values, method='average') / len(pred_df)
rank_catboost = rankdata(pred_df['p_catboost'].values, method='average') / len(pred_df)
rank_rf = rankdata(pred_df['p_rf'].values, method='average') / len(pred_df)
eps = 1e-10
rank_matrix = np.stack([rank_chemprop, rank_catboost, rank_rf], axis=1) + eps
pred_df['percentile_rank_geomean'] = gmean(rank_matrix, axis=1)

# Sort by P_consensus (descending)
pred_df = pred_df.sort_values('p_consensus', ascending=False).reset_index(drop=True)
pred_df['rank'] = pred_df.index + 1

print(f"\n  P_consensus distribution:")
print(f"    Min:    {pred_df['p_consensus'].min():.4f}")
print(f"    Median: {pred_df['p_consensus'].median():.4f}")
print(f"    Max:    {pred_df['p_consensus'].max():.4f}")
print(f"    Mean:   {pred_df['p_consensus'].mean():.4f}")

print(f"\n  Model variance distribution:")
print(f"    Min:    {pred_df['model_variance'].min():.6f}")
print(f"    Median: {pred_df['model_variance'].median():.6f}")
print(f"    Max:    {pred_df['model_variance'].max():.6f}")

# =====================================================================
# STEP 8: Apply confidence filter (top 5% AND Var < 0.04)
# =====================================================================
print(f"\n[6/6] Applying confidence filter...")
print(f"  Filter: top 5% by P_consensus AND Var < 0.04")

TOP_PERCENTILE = 0.05
VARIANCE_THRESHOLD = 0.04

n_total = len(pred_df)
top_5pct_n = max(1, int(n_total * TOP_PERCENTILE))
print(f"  Top 5% of FDA library: {top_5pct_n} drugs")

top_5pct = pred_df.head(top_5pct_n).copy()
n_before_filter = len(top_5pct)

top_filtered = top_5pct[top_5pct['model_variance'] < VARIANCE_THRESHOLD].copy()
n_after_filter = len(top_filtered)

print(f"\n  BEFORE variance filter (top 5%):")
print(f"    Drugs: {n_before_filter}")
print(f"    Mean P_consensus: {top_5pct['p_consensus'].mean():.4f}")
print(f"    Mean variance: {top_5pct['model_variance'].mean():.4f}")

print(f"\n  AFTER variance filter (Var < {VARIANCE_THRESHOLD}):")
print(f"    Drugs: {n_after_filter} (filtered out {n_before_filter - n_after_filter})")
print(f"    Mean P_consensus: {top_filtered['p_consensus'].mean():.4f}")
print(f"    Mean variance: {top_filtered['model_variance'].mean():.4f}")

# =====================================================================
# STEP 9: Save outputs
# =====================================================================
print(f"\n=== SAVING OUTPUTS ===")
output_dir = '/content/data/fda_predictions'
os.makedirs(output_dir, exist_ok=True)

output_columns = [
    'rank', drug_name_col, smiles_col, 'clinical_phase', 'moa', 'target',
    'disease_area', 'indication',
    'p_chemprop', 'p_catboost', 'p_rf',
    'p_consensus', 'model_variance', 'percentile_rank_geomean'
]
output_columns = [c for c in output_columns if c in pred_df.columns]

pred_df[output_columns].to_csv(
    os.path.join(output_dir, 'fda_consensus_predictions.csv'),
    index=False
)
print(f"  Saved fda_consensus_predictions.csv ({len(pred_df):,} drugs)")

top_filtered[output_columns].to_csv(
    os.path.join(output_dir, 'fda_top_candidates.csv'),
    index=False
)
print(f"  Saved fda_top_candidates.csv ({len(top_filtered)} drugs)")

metrics = {
    'n_fda_drugs_screened': int(len(pred_df)),
    'n_failed_chemprop': int(n_before - len(pred_df)),
    'n_failed_ecfp6': int(n_failed),
    'p_consensus_stats': {
        'min': float(pred_df['p_consensus'].min()),
        'median': float(pred_df['p_consensus'].median()),
        'mean': float(pred_df['p_consensus'].mean()),
        'max': float(pred_df['p_consensus'].max()),
    },
    'confidence_filter': {
        'top_percentile': TOP_PERCENTILE,
        'variance_threshold': VARIANCE_THRESHOLD,
        'n_top_5pct': int(n_before_filter),
        'n_after_variance_filter': int(n_after_filter),
        'mean_p_consensus_top5pct': float(top_5pct['p_consensus'].mean()),
        'mean_p_consensus_filtered': float(top_filtered['p_consensus'].mean()),
    },
}
with open(os.path.join(output_dir, 'fda_screening_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved fda_screening_metrics.json")

# =====================================================================
# STEP 10: Top-30 prioritized FDA candidates (sanity check)
# =====================================================================
print(f"\n=== TOP-30 PRIORITIZED FDA REPURPOSING CANDIDATES ===")
print(f"\n  {'Rk':<4}{'P_cons':<8}{'Var':<8}{'Drug Name':<40}{'MoA':<30}{'Target':<15}")
print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*38}  {'-'*28}  {'-'*13}")
for i in range(min(30, len(top_filtered))):
    row = top_filtered.iloc[i]
    drug_name = str(row[drug_name_col])[:38]
    moa = str(row['moa'])[:28] if pd.notna(row['moa']) else '—'
    target = str(row['target'])[:13] if pd.notna(row['target']) else '—'
    print(f"  {i+1:<4}{row['p_consensus']:<8.4f}{row['model_variance']:<8.4f}"
          f"{drug_name:<40}{moa:<30}{target:<15}")

# =====================================================================
# STEP 11: Save report
# =====================================================================
report_path = os.path.join(output_dir, 'fda_screening_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 6 FDA REPURPOSING REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("SCREENING LIBRARY\n")
    f.write(f"  Source: CLUE Repurposing Hub (Broad Institute)\n")
    f.write(f"  Filter: clinical_phase == 'Launched' (FDA-approved)\n")
    f.write(f"  Total FDA drugs screened: {len(pred_df):,}\n\n")
    f.write("MODELS USED\n")
    f.write(f"  1. Chemprop D-MPNN (graph-based)\n")
    f.write(f"  2. CatBoost on ECFP6 count vectors\n")
    f.write(f"  3. Balanced Random Forest on ECFP6 count vectors\n\n")
    f.write("CONSENSUS STRATEGY\n")
    f.write(f"  Soft-voting: P_consensus = (P_chemprop + P_catboost + P_rf) / 3.0\n")
    f.write(f"  Model variance: σ² of the 3 predictions\n")
    f.write(f"  Percentile rank geometric mean\n\n")
    f.write("CONFIDENCE FILTER\n")
    f.write(f"  Top 5% by P_consensus AND Var < {VARIANCE_THRESHOLD}\n")
    f.write(f"  Before filter: {n_before_filter} drugs\n")
    f.write(f"  After filter:  {n_after_filter} drugs (high-confidence candidates)\n\n")
    f.write("TOP 10 FDA REPURPOSING CANDIDATES\n")
    for i in range(min(10, len(top_filtered))):
        row = top_filtered.iloc[i]
        f.write(f"  {i+1}. {row[drug_name_col]}\n")
        f.write(f"     P_consensus={row['p_consensus']:.4f}, Var={row['model_variance']:.4f}\n")
        if pd.notna(row['moa']):
            f.write(f"     MoA: {row['moa']}\n")
        if pd.notna(row['target']):
            f.write(f"     Target: {row['target']}\n")
        f.write(f"     Indication: {row.get('indication', '—')}\n\n")
print(f"  Saved fda_screening_report.txt")

print(f"\n{'=' * 78}")
print("STAGE 6 COMPLETE")
print("=" * 78)
print(f"\nFiles saved to Colab: {output_dir}/")
print(f"  - fda_consensus_predictions.csv   (all {len(pred_df):,} drugs, ranked)")
print(f"  - fda_top_candidates.csv          (top {len(top_filtered)} high-confidence)")
print(f"  - fda_screening_metrics.json")
print(f"  - fda_screening_report.txt")
print(f"\nNEXT STEPS:")
print(f"  1. Download fda_top_candidates.csv and fda_consensus_predictions.csv to your Mac")
print(f"  2. Stage 7: External validation on AID 493107")
print(f"  3. Stage 8: Safety triage (PAINS, luciferase inhibitor, Lipinski/Veber)")
