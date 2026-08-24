"""
ATAD5AI — Stage 2: Dual Feature Representation (ECFP6 Counts + SMILES for D-MPNN)
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 2: Dual Feature Representation
# =====================================================================
# INPUT:
#   outputs/stage1_clean_data/train_pool.csv  (241,861 compounds, standardized)
#
# OUTPUT:
#   outputs/stage2_features/
#     - ecfp6_counts.npz             — Sparse CSR matrix (241,861 × 4096, uint16)
#     - smiles_for_chemprop.csv      — Canonical SMILES for D-MPNN (Stream A)
#     - feature_report.txt
#
# MODULE 2 SPEC:
#   Stream A (Graph Stream for Chemprop):
#     * Pass raw canonical SMILES directly into D-MPNN
#     * Node Features: Atom identity, degree, formal charge, hybridization, aromaticity
#     * Edge Features: Bond type, conjugation, stereochemistry
#
#   Stream B (High-Resolution Substructure Counts for GBDT & RF):
#     * Generate pure 4096-bit ECFP6 (Morgan Fingerprint, radius=3) COUNT vectors
#     * Use INTEGER COUNTS (not binary bits) to capture functional group multiplicity
#     * Matrix format: Compressed sparse CSR matrix (scipy.sparse.csr_matrix, uint16)
#       to prevent memory overflow
#
# Why ECFP6 (radius=3) instead of ECFP4 (radius=2)?
#   - ECFP4 captures substructures up to diameter 4 bonds
#   - ECFP6 captures up to diameter 6 bonds — better at recognizing
#     larger pharmacophores (e.g., tricyclic cores, fused heterocycles)
#   - Standard choice for GBDT/RF in modern virtual screening
#
# Why COUNT vectors instead of binary?
#   - Binary: only "present/absent" — two molecules with one vs three
#     primary amines look identical
#   - Count: distinguishes them — captures functional group multiplicity
#     which is critical for ADME/PK and potency
#
# Memory math:
#   241,861 × 4096 × 2 bytes (uint16) = 1.98 GB DENSE
#   Sparsity ~97.7% → CSR matrix: ~50 MB (40× smaller than dense)
#   uint16 allows counts up to 65,535 — plenty for 4096-bit fingerprints
# =====================================================================

import os
import sys
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import scipy.sparse as sp

# =====================================================================
# CONFIG
# =====================================================================
INPUT_FILE = 'outputs/stage1_clean_data/train_pool.csv'
OUTPUT_DIR = 'outputs/stage2_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BITS = 4096       # Per spec: 4096-bit ECFP6
RADIUS = 3          # ECFP6 = Morgan radius 3 (diameter 6)
DTYPE = np.uint16   # Per spec: uint16 (counts up to 65,535)

print("=" * 78)
print("ATAD5AI — STAGE 2: DUAL FEATURE REPRESENTATION")
print("=" * 78)
print(f"\nStream A: Canonical SMILES → Chemprop D-MPNN (saved as CSV)")
print(f"Stream B: ECFP6 count vectors, {N_BITS}-bit, radius={RADIUS}")
print(f"          Storage: scipy.sparse.csr_matrix, dtype={DTYPE.__name__}")

# =====================================================================
# 1. Load Stage 1 output
# =====================================================================
print(f"\n[1/4] Loading training pool from Stage 1...")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(df):,} compounds")
print(f"  Actives: {(df['label'] == 1).sum():,}")
print(f"  Inactives: {(df['label'] == 0).sum():,}")
print(f"  Unique scaffolds: {df['bm_scaffold'].nunique():,}")
print(f"  Columns: {list(df.columns)}")

# Use canonical_smiles (already standardized in Stage 1)
smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'smiles'
print(f"\n  Using SMILES column: '{smiles_col}'")
print(f"  Sample SMILES:")
for i in range(3):
    print(f"    {i+1}. {df.iloc[i][smiles_col][:80]}...")

# =====================================================================
# 2. STREAM A: Save canonical SMILES for Chemprop D-MPNN
# =====================================================================
print(f"\n[2/4] Stream A: Saving canonical SMILES for Chemprop...")

# Chemprop expects a CSV with 'smiles' and 'label' columns
chemprop_df = pd.DataFrame({
    'smiles': df[smiles_col],
    'label': df['label']
})

chemprop_path = os.path.join(OUTPUT_DIR, 'smiles_for_chemprop.csv')
chemprop_df.to_csv(chemprop_path, index=False)
print(f"  Saved {chemprop_path} ({len(chemprop_df):,} compounds)")
print(f"  Chemprop will featurize SMILES internally (atom + bond features → D-MPNN graph)")

# =====================================================================
# 3. STREAM B: Generate ECFP6 count vectors (the slow step)
# =====================================================================
print(f"\n[3/4] Stream B: Generating ECFP6 count vectors...")
print(f"  Configuration: {N_BITS}-bit Morgan FP, radius={RADIUS}, INTEGER COUNTS")
print(f"  Storage: scipy.sparse.csr_matrix, dtype={DTYPE.__name__}")
print(f"  Expected sparsity: ~97-98% (most compounds set ~50-100 bits)")
print(f"  This will take ~10-20 minutes for {len(df):,} compounds")

def smiles_to_ecfp6_counts(smiles, n_bits=N_BITS, radius=RADIUS):
    """
    Convert SMILES to ECFP6 count vector.
    Uses GetHashedMorganFingerprint which returns INTEGER COUNTS
    (not binary bits). Captures functional group multiplicity.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    # GetHashedMorganFingerprint returns SparseBitVect by default,
    # but we need COUNTS — use GetMorganFingerprintAsBitVect's count variant.
    # Actually, the proper way to get counts is to use a different function:
    #
    # Option 1: AllChem.GetHashedMorganFingerprint(mol, radius, nBits) → UIntSparseIntVect
    # Option 2: Use the underlying generator with count version
    #
    # Let's use the explicit count version via numpy:
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
    # fp is a UIntSparseIntVect — convert to numpy array
    arr = np.zeros(n_bits, dtype=DTYPE)
    # Get nonzero positions and their counts
    nonzero = fp.GetNonzeroElements()
    for bit, count in nonzero.items():
        arr[bit] = min(count, 65535)  # cap at uint16 max
    return arr

# Featurize in batches
n_total = len(df)
batch_size = 5000
start_time = time.time()

# Pre-allocate list of (row, col, val) for sparse construction
rows = []
cols = []
vals = []
n_failed = 0

for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    for i in range(start, end):
        fp = smiles_to_ecfp6_counts(df.iloc[i][smiles_col])
        if fp is None:
            n_failed += 1
            continue
        # Find nonzero positions
        nz_indices = np.nonzero(fp)[0]
        nz_values = fp[nz_indices]
        # Append to lists
        rows.extend([i] * len(nz_indices))
        cols.extend(nz_indices.tolist())
        vals.extend(nz_values.tolist())
    if end % 20000 == 0 or end == n_total:
        elapsed = time.time() - start_time
        rate = end / max(elapsed, 1)
        eta_min = (n_total - end) / max(rate, 1) / 60
        print(f"  Processed {end:,} / {n_total:,} ({end/n_total*100:.1f}%) — {rate:.0f}/s, ETA {eta_min:.1f} min")

print(f"\n  Featurization complete.")
print(f"  Failed molecules: {n_failed}")
print(f"  Total nonzero entries: {len(vals):,}")

# Build CSR matrix
print(f"\n  Building sparse CSR matrix...")
# COO format → CSR (CSR is more efficient for row slicing in training)
X_sparse = sp.coo_matrix(
    (np.array(vals, dtype=DTYPE),
     (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
    shape=(n_total, N_BITS),
    dtype=DTYPE
).tocsr()

# Compute sparsity stats
nnz = X_sparse.nnz
total_elements = X_sparse.shape[0] * X_sparse.shape[1]
sparsity_pct = (1 - nnz / total_elements) * 100
avg_bits_set = nnz / n_total
avg_total_count = X_sparse.sum(axis=1).mean()

print(f"  Matrix shape: {X_sparse.shape}")
print(f"  dtype: {X_sparse.dtype}")
print(f"  NNZ (nonzero entries): {nnz:,}")
print(f"  Sparsity: {sparsity_pct:.2f}%")
print(f"  Average bits set per compound: {avg_bits_set:.1f} (of {N_BITS})")
print(f"  Average total count per compound: {float(avg_total_count):.1f}")
print(f"  Max count in any cell: {X_sparse.max()}")

# Save sparse matrix in NPZ format (efficient for sparse storage)
sparse_path = os.path.join(OUTPUT_DIR, 'ecfp6_counts.npz')
sp.save_npz(sparse_path, X_sparse)
sparse_size_mb = os.path.getsize(sparse_path) / (1024 * 1024)
print(f"\n  Saved {sparse_path}")
print(f"  File size: {sparse_size_mb:.1f} MB (vs ~{total_elements * 2 / (1024*1024):.0f} MB if dense)")

# =====================================================================
# 4. Save labels + metadata for downstream stages
# =====================================================================
print(f"\n[4/4] Saving labels and metadata for downstream stages...")

# Save labels (aligned with sparse matrix row order)
labels_path = os.path.join(OUTPUT_DIR, 'labels.npy')
np.save(labels_path, df['label'].values.astype(np.uint8))
print(f"  Saved {labels_path} ({len(df):,} labels)")

# Save SMILES (aligned with sparse matrix row order) — Chemprop Stream A
smiles_path = os.path.join(OUTPUT_DIR, 'smiles.npy')
np.save(smiles_path, df[smiles_col].values.astype(str))
print(f"  Saved {smiles_path}")

# Save metadata (for SHAP analysis + result tables later)
metadata_cols = ['pubchem_sid', 'pubchem_cid', 'canonical_smiles', 'bm_scaffold',
                 'label', 'activity_score', 'ac50_umol', 'efficacy_pct',
                 'fit_r2', 'curve_description']
metadata_cols_present = [c for c in metadata_cols if c in df.columns]
metadata = df[metadata_cols_present].copy()

# Map any remaining original columns
metadata_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
metadata.to_csv(metadata_path, index=False)
print(f"  Saved {metadata_path} ({len(metadata):,} rows)")

# =====================================================================
# 5. Feature distribution analysis (sanity check)
# =====================================================================
print(f"\n=== FEATURE DISTRIBUTION ANALYSIS ===")

# Bit frequency: how many compounds set each bit?
bit_frequency = np.array((X_sparse > 0).sum(axis=0)).flatten()
print(f"\n  Bit frequency (across {n_total:,} compounds):")
print(f"    Min:    {bit_frequency.min():,} (rare substructure)")
print(f"    5%:     {np.percentile(bit_frequency, 5):,.0f}")
print(f"    Median: {np.median(bit_frequency):,.0f}")
print(f"    95%:    {np.percentile(bit_frequency, 95):,.0f}")
print(f"    Max:    {bit_frequency.max():,} (very common substructure)")

# How many bits are NEVER set?
n_dead_bits = (bit_frequency == 0).sum()
print(f"    Dead bits (never set): {n_dead_bits:,} / {N_BITS} ({n_dead_bits/N_BITS*100:.1f}%)")
print(f"      → Normal for ECFP6 on small-to-medium libraries; high-radius fingerprints explore a sparse substructure space")

# Top 10 most frequent bits
top_bits = np.argsort(-bit_frequency)[:10]
print(f"\n  Top 10 most frequent bits:")
for rank, bit in enumerate(top_bits):
    n_active = X_sparse[df['label'].values == 1, bit].nnz
    n_total_bit = bit_frequency[bit]
    print(f"    {rank+1}. Bit {bit:>4}: set in {n_total_bit:>6,} compounds "
          f"({n_total_bit/n_total*100:.2f}%) | active ratio: {n_active}/{(df['label']==1).sum()}")

# Per-compound bit count distribution
per_compound_bits = np.array((X_sparse > 0).sum(axis=1)).flatten()
per_compound_counts = np.array(X_sparse.sum(axis=1)).flatten()
print(f"\n  Per-compound bit count (how many bits set per molecule):")
print(f"    Min:    {per_compound_bits.min()}")
print(f"    Median: {np.median(per_compound_bits):.0f}")
print(f"    Max:    {per_compound_bits.max()}")
print(f"    Mean:   {per_compound_bits.mean():.1f}")
print(f"\n  Per-compound total count (sum of all bit counts):")
print(f"    Min:    {per_compound_counts.min()}")
print(f"    Median: {np.median(per_compound_counts):.0f}")
print(f"    Max:    {per_compound_counts.max()}")
print(f"    Mean:   {per_compound_counts.mean():.1f}")

# =====================================================================
# 6. Active vs Inactive comparison (sanity check)
# =====================================================================
print(f"\n=== ACTIVE vs INACTIVE FEATURE COMPARISON ===")

actives_mask = (df['label'] == 1).values
inactives_mask = (df['label'] == 0).values

# Average bit count for each class
avg_bits_active = per_compound_bits[actives_mask].mean()
avg_bits_inactive = per_compound_bits[inactives_mask].mean()
print(f"\n  Average bits set:")
print(f"    Actives:   {avg_bits_active:.1f}")
print(f"    Inactives: {avg_bits_inactive:.1f}")

# Most discriminative bits (high in actives, low in inactives)
active_freq = np.array((X_sparse[actives_mask] > 0).sum(axis=0)).flatten() / max(actives_mask.sum(), 1)
inactive_freq = np.array((X_sparse[inactives_mask] > 0).sum(axis=0)).flatten() / max(inactives_mask.sum(), 1)
bit_enrichment = active_freq - inactive_freq  # positive = enriched in actives

# Top 10 most enriched bits in actives
top_enriched = np.argsort(-bit_enrichment)[:10]
print(f"\n  Top 10 bits most enriched in actives (likely ELG1 pharmacophore):")
print(f"    {'Bit':<8} {'Active %':<12} {'Inactive %':<12} {'Enrichment':<12}")
for bit in top_enriched:
    print(f"    {bit:<8} {active_freq[bit]*100:<12.2f} {inactive_freq[bit]*100:<12.2f} {bit_enrichment[bit]:.3f}")

# =====================================================================
# 7. Save feature report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'feature_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 2 FEATURE REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("INPUT\n")
    f.write(f"  File: {INPUT_FILE}\n")
    f.write(f"  Compounds: {len(df):,}\n")
    f.write(f"  Actives: {(df['label']==1).sum():,}\n")
    f.write(f"  Inactives: {(df['label']==0).sum():,}\n\n")
    f.write("STREAM A (for Chemprop D-MPNN)\n")
    f.write(f"  Output: smiles_for_chemprop.csv\n")
    f.write(f"  Format: Canonical SMILES strings (Chemprop featurizes internally)\n")
    f.write(f"  Graph features: atom identity, degree, formal charge, hybridization, aromaticity\n")
    f.write(f"  Bond features: bond type, conjugation, stereochemistry\n\n")
    f.write("STREAM B (for CatBoost + Balanced RF)\n")
    f.write(f"  Output: ecfp6_counts.npz (scipy.sparse.csr_matrix)\n")
    f.write(f"  Fingerprint: ECFP6 (Morgan radius=3, diameter 6)\n")
    f.write(f"  n_bits: {N_BITS}\n")
    f.write(f"  Vector type: INTEGER COUNTS (not binary)\n")
    f.write(f"  dtype: {DTYPE.__name__} (max count per bit: 65,535)\n")
    f.write(f"  Matrix shape: {X_sparse.shape}\n")
    f.write(f"  NNZ: {nnz:,}\n")
    f.write(f"  Sparsity: {sparsity_pct:.2f}%\n")
    f.write(f"  File size: {sparse_size_mb:.1f} MB (vs {total_elements * 2 / (1024*1024):.0f} MB if dense)\n")
    f.write(f"  Avg bits set per compound: {avg_bits_set:.1f}\n")
    f.write(f"  Avg total count per compound: {float(avg_total_count):.1f}\n")
    f.write(f"  Failed molecules: {n_failed}\n\n")
    f.write("FEATURE DISTRIBUTION\n")
    f.write(f"  Bit frequency: min={bit_frequency.min():,}, median={np.median(bit_frequency):,.0f}, max={bit_frequency.max():,}\n")
    f.write(f"  Dead bits (never set): {n_dead_bits:,} / {N_BITS} ({n_dead_bits/N_BITS*100:.1f}%)\n")
    f.write(f"  Per-compound bits: min={per_compound_bits.min()}, median={np.median(per_compound_bits):.0f}, max={per_compound_bits.max()}\n\n")
    f.write("ACTIVE vs INACTIVE COMPARISON\n")
    f.write(f"  Avg bits set in actives:   {avg_bits_active:.1f}\n")
    f.write(f"  Avg bits set in inactives: {avg_bits_inactive:.1f}\n\n")
    f.write("NEXT STEPS (Stage 3)\n")
    f.write(f"  - Scaffold-stratified 80/10/10 split (round-robin)\n")
    f.write(f"  - Train ~193K, Validation ~24K, Test ~24K\n")
    f.write(f"  - Compounds grouped by Bemis-Murcko scaffold (from Stage 1)\n")
    f.write(f"  - No scaffold appears in more than one split\n\n")
    f.write("NEXT STEPS (Stage 4 — three models, parallel)\n")
    f.write(f"  4a. Chemprop D-MPNN on Stream A (SMILES → graph → D-MPNN)\n")
    f.write(f"  4b. CatBoost on Stream B (ECFP6 count vectors)\n")
    f.write(f"  4c. Balanced Random Forest on Stream B (ECFP6 count vectors)\n")
print(f"  Saved {report_path}")

print(f"\n" + "=" * 78)
print("STAGE 2 COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - smiles_for_chemprop.csv    ({len(df):,} SMILES for D-MPNN)")
print(f"  - ecfp6_counts.npz           ({X_sparse.shape[0]:,} × {X_sparse.shape[1]} sparse, {sparse_size_mb:.1f} MB)")
print(f"  - labels.npy                 ({len(df):,} binary labels)")
print(f"  - smiles.npy                 (SMILES array for Stream A)")
print(f"  - metadata.csv               (compound metadata)")
print(f"  - feature_report.txt")
print(f"\nReady for Stage 3 (scaffold-stratified 80/10/10 split).")
