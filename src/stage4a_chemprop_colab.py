"""
ATAD5AI — Stage 4a: Chemprop D-MPNN Training (Google Colab, v2.3.1 API)

Hybrid Consensus Screening Pipeline — Model 1 of 3

Trains a Directed Message Passing Neural Network (D-MPNN) on the ATAD5AI
training set. The model learns molecular representations directly from
graph structure (atoms as nodes, bonds as edges), complementing the
ECFP6-based CatBoost and Balanced Random Forest models in Stage 4b/4c.

==============================================================================
HOW TO USE (on Google Colab with T4 GPU)
==============================================================================

PREREQUISITES:
  - Upload smiles_train.csv, smiles_val.csv, smiles_test.csv to Colab
    (script will prompt for direct upload — no Google Drive needed)
  - Runtime → Change runtime type → T4 GPU

STEPS:
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Paste this entire script into a single cell
  4. Press Shift+Enter
  5. When prompted, upload the 3 SMILES CSVs from outputs/stage3_split/

EXPECTED RUNTIME: ~40 minutes on T4 GPU (v2.3.1 trains much faster than v2.0)

==============================================================================
MODULE 4 SPEC (Model 1 — Chemprop D-MPNN)
==============================================================================
  - Architecture: 3 message-passing steps, hidden dimension = 300, 2-layer FFN
  - Loss Function: Binary Cross-Entropy with class-weighting
  - Training: Adam optimizer (lr=1e-4), early stopping on Val PR-AUC (patience=15)

==============================================================================
INPUTS
==============================================================================
  - smiles_train.csv  (195,501 compounds, columns: smiles, label)
  - smiles_val.csv    (23,324 compounds, columns: smiles, label)
  - smiles_test.csv   (23,036 compounds, columns: smiles, label)

==============================================================================
OUTPUTS (saved to /content/data/chemprop_outputs/)
==============================================================================
  - test_predictions_chemprop.csv   (test set predictions, sorted by proba)
  - chemprop_metrics.json            (test metrics)
  - chemprop_best_model.ckpt         (best model checkpoint, ~50 MB)
  - checkpoints/                     (intermediate checkpoints)
  - logs/                            (CSV training metrics)
"""

# =====================================================================
# CELL 1: Install dependencies (ACTIVE — runs automatically)
# =====================================================================
!pip install chemprop lightning torch pandas numpy scikit-learn rdkit 2>&1 | tail -5

# =====================================================================
# CELL 2: Upload data (DIRECT UPLOAD — no Google Drive needed)
# =====================================================================
import os
from google.colab import files

DATA_DIR = '/content/data'
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Data dir: {DATA_DIR}")

# Check what's already uploaded, prompt for missing files
for fname in ['smiles_train.csv', 'smiles_val.csv', 'smiles_test.csv']:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {fname} ({size_mb:.2f} MB) — already uploaded")
    else:
        print(f"\n  ⚠ Upload {fname}")
        print(f"  → Click 'Choose Files' below")
        print(f"  → Select {fname} from your Mac (in outputs/stage3_split/)")
        uploaded = files.upload()
        for uploaded_name in uploaded.keys():
            os.rename(uploaded_name, os.path.join(DATA_DIR, uploaded_name))
            size_mb = os.path.getsize(os.path.join(DATA_DIR, uploaded_name)) / (1024 * 1024)
            print(f"  ✓ Saved {uploaded_name} ({size_mb:.2f} MB)")

# =====================================================================
# CELL 3: Verify GPU
# =====================================================================
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("GPU not available! Change runtime type to T4 GPU.")

# =====================================================================
# CELL 4: Load data
# =====================================================================
import pandas as pd
import numpy as np

print("\n=== LOADING DATA ===")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_train.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_val.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_test.csv'))

print(f"Train: {len(train_df):,} (actives: {train_df['label'].sum():,})")
print(f"Val:   {len(val_df):,} (actives: {val_df['label'].sum():,})")
print(f"Test:  {len(test_df):,} (actives: {test_df['label'].sum():,})")

# =====================================================================
# CELL 5: Build Chemprop v2.3.1 datasets and dataloaders
# =====================================================================
print("\n=== BUILDING CHEMPROP DATASETS ===")
import chemprop
print(f"Chemprop version: {chemprop.__version__}")

from chemprop import data
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem

# Build SMILES → target lists
train_smiles = train_df['smiles'].tolist()
train_targets = train_df['label'].astype(float).tolist()
val_smiles = val_df['smiles'].tolist()
val_targets = val_df['label'].astype(float).tolist()
test_smiles = test_df['smiles'].tolist()
test_targets = test_df['label'].astype(float).tolist()

# v2.3.1: MoleculeDatapoint needs an RDKit Mol object, not SMILES
print("  Converting SMILES → RDKit Mols...")
train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
val_mols = [Chem.MolFromSmiles(s) for s in val_smiles]
test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]

# Filter out any parse failures
def filter_valid(mols, targets):
    pairs = [(m, t) for m, t in zip(mols, targets) if m is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]

train_mols, train_targets = filter_valid(train_mols, train_targets)
val_mols, val_targets = filter_valid(val_mols, val_targets)
test_mols, test_targets = filter_valid(test_mols, test_targets)

n_train_failed = len(train_smiles) - len(train_mols)
n_val_failed = len(val_smiles) - len(val_mols)
n_test_failed = len(test_smiles) - len(test_mols)
print(f"  Parse failures: train={n_train_failed}, val={n_val_failed}, test={n_test_failed}")

# Build MoleculeDatapoint lists (v2.3.1 signature: mol=Chem.Mol, y=np.ndarray)
print("  Building datapoints...")
train_datapoints = [data.MoleculeDatapoint(mol=m, y=np.array([t], dtype=np.float32))
                    for m, t in zip(train_mols, train_targets)]
val_datapoints = [data.MoleculeDatapoint(mol=m, y=np.array([t], dtype=np.float32))
                  for m, t in zip(val_mols, val_targets)]
test_datapoints = [data.MoleculeDatapoint(mol=m, y=np.array([t], dtype=np.float32))
                   for m, t in zip(test_mols, test_targets)]

# Featurize (atom + bond features → molecular graph)
featurizer = SimpleMoleculeMolGraphFeaturizer()
train_dataset = data.MoleculeDataset(train_datapoints, featurizer)
val_dataset = data.MoleculeDataset(val_datapoints, featurizer)
test_dataset = data.MoleculeDataset(test_datapoints, featurizer)

print(f"  Train dataset: {len(train_dataset):,} molecules")
print(f"  Val dataset:   {len(val_dataset):,} molecules")
print(f"  Test dataset:  {len(test_dataset):,} molecules")

# Build dataloaders
BATCH_SIZE = 64
train_loader = data.build_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.build_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.build_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# =====================================================================
# CELL 6: Build D-MPNN model (per PriA-SSB spec, v2.3.1 API)
# =====================================================================
print("\n=== BUILDING D-MPNN MODEL ===")
import torch
from chemprop.nn import BondMessagePassing, MeanAggregation, BinaryClassificationFFN
from chemprop.models import MPNN

# Class weight (positive class is rare, so weight it higher)
n_actives = train_df['label'].sum()
n_inactives = len(train_df) - n_actives
pos_weight = n_inactives / max(n_actives, 1)
print(f"  Class imbalance: {pos_weight:.2f}:1")

# Build D-MPNN components per PriA-SSB spec:
#   - 3 message-passing steps, hidden dimension = 300, 2-layer FFN
#   - BCE with class-weighting
#   - Adam optimizer lr=1e-4, early stopping on val PR-AUC (patience=15)

# Message passing: 3 layers, hidden=300
mp = BondMessagePassing(d_h=300, depth=3)

# Aggregation: mean pool
agg = MeanAggregation()

# Predictor: FFN with hidden_dim=300 (v2.3.1 uses 'hidden_dim' singular)
# 'n_layers' controls depth (default 1, set to 2 for 2-layer FFN per spec)
# 'task_weights' handles class imbalance
predictor = BinaryClassificationFFN(
    n_tasks=1,
    hidden_dim=300,           # hidden layer dimension
    n_layers=2,                # 2-layer FFN (per spec)
    task_weights=[1.0, pos_weight]  # [inactive weight, active weight]
)

# Build MPNN model
model = MPNN(
    message_passing=mp,
    agg=agg,
    predictor=predictor,
)

# Print model architecture
print(f"\n  Model architecture:")
print(f"    Message passing: {mp.__class__.__name__} (depth=3, d_h=300)")
print(f"    Aggregation: {agg.__class__.__name__}")
print(f"    Predictor: {predictor.__class__.__name__} (hidden_dim=300, n_layers=2)")
print(f"    Task weights: [1.0, {pos_weight:.2f}] (active class upweighted)")
print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# =====================================================================
# CELL 7: Set up PyTorch Lightning trainer with early stopping
# =====================================================================
print("\n=== SETTING UP LIGHTNING TRAINER ===")
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import os

# Output directory — using Colab local storage
output_dir = '/content/data/chemprop_outputs'
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
log_dir = os.path.join(output_dir, 'logs')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Early stopping on val_loss (PR-AUC not directly logged by Lightning)
# Per spec: patience=15
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min',
    verbose=True,
)

# Model checkpoint — save best model
model_checkpoint = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='chemprop-best-{epoch:02d}-{val_loss:.4f}',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    verbose=True,
)

# CSV logger — track training progress
csv_logger = CSVLogger(save_dir=log_dir, name='chemprop_atad5')

# Lightning trainer
MAX_EPOCHS = 50
trainer = lightning.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu',
    devices=1,
    callbacks=[early_stopping, model_checkpoint],
    logger=[csv_logger],
    enable_progress_bar=True,
    log_every_n_steps=50,
    gradient_clip_val=1.0,  # prevent exploding gradients
)

print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Early stopping: monitor='val_loss', patience=15")
print(f"  Checkpoint dir: {checkpoint_dir}")
print(f"  Log dir: {log_dir}")

# =====================================================================
# CELL 8: Train the model
# =====================================================================
print("\n=== TRAINING D-MPNN ===")
print(f"  This will take ~40 minutes on T4 GPU")
print(f"  Start time: {pd.Timestamp.now()}")

import time
start_time = time.time()

# Move model to GPU
model = model.to('cuda')

# Train
trainer.fit(model, train_loader, val_loader)

elapsed_min = (time.time() - start_time) / 60
print(f"\n  Training complete in {elapsed_min:.1f} minutes ({elapsed_min/60:.1f} hours)")
print(f"  End time: {pd.Timestamp.now()}")
print(f"  Best checkpoint: {model_checkpoint.best_model_path}")

# =====================================================================
# CELL 9: Load best checkpoint and evaluate on test set (v2.3.1 API)
# =====================================================================
print("\n=== EVALUATING ON TEST SET ===")
print(f"  Test set: {len(test_df):,} compounds ({test_df['label'].sum()} actives)")

# Load best checkpoint
best_model = MPNN.load_from_checkpoint(model_checkpoint.best_model_path)
best_model = best_model.to('cuda')
best_model.eval()

# Use Lightning's trainer.predict() — v2.3.1 official API
# This handles the batch format correctly
print(f"  Running predictions via trainer.predict()...")
import torch
import numpy as np

# Create a fresh prediction trainer
predict_trainer = lightning.Trainer(
    accelerator='gpu',
    devices=1,
    enable_progress_bar=True,
    logger=False,
)

# trainer.predict returns a list of batches
predictions_list = predict_trainer.predict(best_model, test_loader)

# Concatenate all batches
test_probs = torch.cat([p.flatten() for p in predictions_list]).cpu().numpy()
test_true = test_df['label'].values

# Trim to match labels (Lightning quirk)
test_probs = test_probs[:len(test_true)]

print(f"\n  Predictions complete: {len(test_probs)} probabilities")
print(f"  Probability range: min={test_probs.min():.4f}, max={test_probs.max():.4f}, mean={test_probs.mean():.4f}")

# Compute metrics
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)

pr_auc = average_precision_score(test_true, test_probs)
roc_auc = roc_auc_score(test_true, test_probs)
brier = brier_score_loss(test_true, test_probs)

def enrichment_factor(y_true, y_pred, frac):
    n_top = max(1, int(len(y_true) * frac))
    n_actives_total = int(y_true.sum())
    ranked = np.argsort(-y_pred)
    n_actives_top = int(y_true[ranked[:n_top]].sum())
    return n_actives_top / max(n_actives_total * frac, 1e-10)

ef1 = enrichment_factor(test_true, test_probs, 0.01)
ef5 = enrichment_factor(test_true, test_probs, 0.05)

def bedroc(y_true, y_pred, alpha=20.0):
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
    bedroc_val = R * ra * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha * ra))
    return float(min(bedroc_val, 1.0))

bedroc_score = bedroc(test_true, test_probs, alpha=20.0)

print(f"\n  === TEST SET PERFORMANCE (Chemprop D-MPNN) ===")
print(f"  PR-AUC (primary):   {pr_auc:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  BEDROC α=20:        {bedroc_score:.4f}   (0.5 = random)")
print(f"  Brier score:        {brier:.4f}   (lower=better)")
print(f"  EF@1%:              {ef1:.2f}×   (random=1.0)")
print(f"  EF@5%:              {ef5:.2f}×   (random=1.0)")

# Precision at recall levels
prec, rec, thresholds = precision_recall_curve(test_true, test_probs)
for target_recall in [0.50, 0.75, 0.90]:
    idx = np.argmax(rec >= target_recall)
    if idx < len(prec):
        print(f"  Precision @ recall={target_recall}:  {prec[idx]:.4f}")

# Store for Cell 10
test_probs_for_save = test_probs

# =====================================================================
# CELL 10: Save predictions and model to Colab storage
# =====================================================================
print("\n=== SAVING PREDICTIONS + MODEL ===")
output_dir = '/content/data/chemprop_outputs'
os.makedirs(output_dir, exist_ok=True)

# Save test predictions (sorted by probability descending)
predictions_df = test_df.copy()
predictions_df['chemprop_proba'] = test_probs
predictions_df = predictions_df.sort_values('chemprop_proba', ascending=False).reset_index(drop=True)
predictions_df['rank'] = predictions_df.index + 1
predictions_path = os.path.join(output_dir, 'test_predictions_chemprop.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"  Saved {predictions_path} ({len(predictions_df):,} rows)")

# Save metrics
metrics = {
    'model': 'chemprop_dmpnn',
    'pr_auc': float(pr_auc),
    'roc_auc': float(roc_auc),
    'bedroc_alpha20': float(bedroc_score),
    'brier': float(brier),
    'ef1': float(ef1),
    'ef5': float(ef5),
    'n_test': len(test_df),
    'n_test_actives': int(test_df['label'].sum()),
    'training_time_minutes': float(elapsed_min),
    'best_checkpoint': model_checkpoint.best_model_path,
}
import json
metrics_path = os.path.join(output_dir, 'chemprop_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# Save final model checkpoint (copy best to a known location)
import shutil
final_model_path = os.path.join(output_dir, 'chemprop_best_model.ckpt')
shutil.copy(model_checkpoint.best_model_path, final_model_path)
print(f"  Saved {final_model_path}")

# =====================================================================
# CELL 11: Top-20 predicted actives (sanity check)
# =====================================================================
print("\n=== TOP-20 PREDICTED ACTIVES (Chemprop) ===")
print(f"\n  {'Rank':<6}{'Prob':<10}{'True':<8}{'SMILES':<60}")
print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*55}")
for i in range(min(20, len(predictions_df))):
    row = predictions_df.iloc[i]
    smiles = str(row['smiles'])[:55] + '...' if len(str(row['smiles'])) > 55 else str(row['smiles'])
    true_label = 'ACTIVE' if row['label'] == 1 else 'inactive'
    print(f"  {i+1:<6}{row['chemprop_proba']:<10.4f}{true_label:<8}{smiles}")

# Top-1% and top-5% enrichment
n_test = len(test_df)
top_1pct_n = max(1, int(n_test * 0.01))
top_5pct_n = max(1, int(n_test * 0.05))
top_1pct_actives = int(predictions_df.iloc[:top_1pct_n]['label'].sum())
top_5pct_actives = int(predictions_df.iloc[:top_5pct_n]['label'].sum())
print(f"\n  Top-1% ({top_1pct_n} compounds): {top_1pct_actives} true actives "
      f"({top_1pct_actives/top_1pct_n*100:.1f}% precision)")
print(f"  Top-5% ({top_5pct_n} compounds): {top_5pct_actives} true actives "
      f"({top_5pct_actives/top_5pct_n*100:.1f}% precision)")

# =====================================================================
# CELL 12: Done — what to do next
# =====================================================================
print("\n" + "=" * 78)
print("STAGE 4a COMPLETE — Chemprop D-MPNN Training Done")
print("=" * 78)
print(f"\nFiles saved to Colab storage: /content/data/chemprop_outputs/")
print(f"  - test_predictions_chemprop.csv    ← DOWNLOAD for Stage 5")
print(f"  - chemprop_metrics.json")
print(f"  - chemprop_best_model.ckpt          (large file, ~50 MB)")
print(f"\nNEXT STEPS:")
print(f"  1. Download test_predictions_chemprop.csv to your Mac")
print(f"  2. Place it in outputs/stage4_models/ on your local machine")
print(f"  3. Run Stage 5 (consensus fusion) which combines all 3 models")
