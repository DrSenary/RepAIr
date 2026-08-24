"""
ATAD5AI — Stage 1: Chemical Curation, Standardization, and 4-Way Data Merge
Hybrid Consensus Screening Pipeline
"""

# =====================================================================
# Stage 1: Chemical Curation, Standardization, and 4-Way Data Merge
# =====================================================================
# INPUTS:
#   data/AID_504467_datatable.csv  — Primary qHTS screen (330K compounds)
#   data/AID_686933_datatable.csv  — Luciferase counter-screen (68 cmpds)
#   data/AID_686921_datatable.csv  — Viability counter-screen (68 cmpds)
#   data/AID_493107_datatable.csv  — Validation dose-response (1,280 cmpds)
#
# OUTPUTS:
#   outputs/stage1_clean_data/
#     - train_pool.csv              — Clean binary-labeled training compounds
#     - validation_holdout.csv      — AID 493107 compounds held out for Stage 7
#     - luciferase_counters.csv     — AID 686933 artifacts for Stage 7
#     - viability_counters.csv      — AID 686921 artifacts for Stage 7
#     - curation_report.txt
#
# MODULE 1 SPEC:
#   1. Remove inorganic molecules and organometallics
#   2. Desalt: isolate largest organic fragment
#   3. Neutralize protonation at pH 7.4
#   4. Standardize tautomers, canonicalize SMILES
#   5. Deduplicate: prioritize Active (1) on label conflict
# =====================================================================

import os
import sys
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover, AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from collections import defaultdict

# =====================================================================
# CONFIG
# =====================================================================
# Try multiple locations for the data files (auto-detect)
DATA_CANDIDATES = ['data', '.', '../data']
DATA_DIR = None
for candidate in DATA_CANDIDATES:
    if os.path.exists(os.path.join(candidate, 'AID_504467_datatable.csv')):
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    print("ERROR: Could not find AID_504467_datatable.csv in any of:", DATA_CANDIDATES)
    print("Please either:")
    print("  1. Place the CSVs in a 'data/' subdirectory, OR")
    print("  2. Place them in the same directory as this script")
    raise SystemExit(1)

print(f"Using DATA_DIR = '{DATA_DIR}'")

OUTPUT_DIR = 'outputs/stage1_clean_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKIP_ROWS = [1, 2, 3, 4]   # PubChem metadata rows

print("=" * 78)
print("ATAD5AI — STAGE 1: CHEMICAL CURATION & 4-WAY DATA MERGE")
print("=" * 78)

# =====================================================================
# MODULE 1.1: STANDARDIZATION FUNCTIONS
# =====================================================================
# Build standardizer objects ONCE (they're expensive to init)
_salt_remover = SaltRemover.SaltRemover()
_normalizer = rdMolStandardize.Normalizer()
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_uncharger = rdMolStandardize.Uncharger()   # neutralizes formal charges
_reionizer = rdMolStandardize.Reionizer()   # ensures proper ionization at physiological pH

# Elements considered "organic" (allowed in drug-like molecules)
ORGANIC_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I

def is_organic(mol):
    """A molecule is 'organic' if it contains at least one C and no disallowed heavy atoms."""
    if mol is None:
        return False
    has_carbon = False
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 6:
            has_carbon = True
        elif z not in ORGANIC_ATOMS:
            return False   # Contains a metal / disallowed element
    return has_carbon

def standardize_molecule(smiles):
    """
    Full curation pipeline:
      1. Parse SMILES
      2. Remove inorganics + organometallics
      3. Desalt (keep largest organic fragment)
      4. Neutralize charges (physiological pH)
      5. Normalize functional groups
      6. Canonicalize tautomer
      7. Generate canonical SMILES

    Returns: canonical SMILES string, or None if molecule rejected.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        # Step 1: Remove inorganic/organometallic atoms
        if not is_organic(mol):
            return None

        # Step 2: Desalt — remove counter-ions, keep largest organic fragment
        mol = _salt_remover.StripMol(mol)
        frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
        if not frags:
            return None
        # Pick largest organic fragment by heavy-atom count
        organic_frags = [f for f in frags if is_organic(f)]
        if not organic_frags:
            return None
        mol = max(organic_frags, key=lambda f: f.GetNumHeavyAtoms())

        # Step 3: Normalize functional groups (e.g., sulfoxide → S=O form)
        mol = _normalizer.normalize(mol)

        # Step 4: Neutralize charges at physiological pH (~7.4)
        # Uncharger removes formal charges (e.g., -COO⁻ → -COOH, -NH₃⁺ → -NH₂)
        mol = _uncharger.uncharge(mol)

        # Step 5: Reionize — ensures acids are deprotonated, bases protonated
        # at physiological pH (more accurate than pure neutralization)
        mol = _reionizer.reionize(mol)

        # Step 6: Tautomer canonicalization
        mol = _tautomer_enumerator.Canonicalize(mol)

        # Step 7: Final canonical SMILES
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if not canonical or canonical == '':
            return None
        return canonical

    except Exception:
        return None

# =====================================================================
# 1. Load primary screen (AID 504467)
# =====================================================================
print(f"\n[1/6] Loading primary screen (AID 504467)...")
t0 = time.time()
df_primary = pd.read_csv(os.path.join(DATA_DIR, 'AID_504467_datatable.csv'),
                          skiprows=SKIP_ROWS, low_memory=False)
print(f"  Loaded {len(df_primary):,} compounds in {time.time()-t0:.1f}s")

# Standardize SID/CID
df_primary['PUBCHEM_SID'] = pd.to_numeric(df_primary['PUBCHEM_SID'], errors='coerce')
df_primary['PUBCHEM_CID'] = pd.to_numeric(df_primary['PUBCHEM_CID'], errors='coerce')

# Drop rows missing SID OR SMILES
n_before = len(df_primary)
df_primary = df_primary.dropna(subset=['PUBCHEM_SID', 'PUBCHEM_EXT_DATASOURCE_SMILES']).copy()
df_primary['PUBCHEM_SID'] = df_primary['PUBCHEM_SID'].astype(int)
print(f"  After dropping missing SID/SMILES: {len(df_primary):,} (removed {n_before - len(df_primary):,})")

# Activity outcome distribution
print(f"\n  Activity outcome distribution:")
for outcome, count in df_primary['PUBCHEM_ACTIVITY_OUTCOME'].value_counts(dropna=False).items():
    pct = count / len(df_primary) * 100
    print(f"    {str(outcome):<15}: {count:>10,}  ({pct:.3f}%)")

# =====================================================================
# 2. Apply binary labeling (drop Inconclusive)
# =====================================================================
print(f"\n[2/6] Applying binary activity labeling...")
mask_binary = df_primary['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])
df_train = df_primary[mask_binary].copy().reset_index(drop=True)
n_dropped_inconclusive = (~mask_binary).sum()
print(f"  Dropped {n_dropped_inconclusive:,} 'Inconclusive' compounds")
print(f"  Remaining: {len(df_train):,}")

# Binary label
df_train['label'] = (df_train['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)

n_actives_initial = (df_train['label'] == 1).sum()
n_inactives_initial = (df_train['label'] == 0).sum()
print(f"  Actives:   {n_actives_initial:,}")
print(f"  Inactives: {n_inactives_initial:,}")
print(f"  Imbalance: {n_inactives_initial / max(n_actives_initial, 1):.2f} : 1")

# =====================================================================
# 3. STANDARDIZE ALL MOLECULES (Module 1 spec)
# =====================================================================
print(f"\n[3/6] Standardizing {len(df_train):,} molecules...")
print(f"  Pipeline: parse → reject organometallic → desalt → neutralize → normalize → tautomer → canonical SMILES")

t0 = time.time()
batch_size = 5000
n_total = len(df_train)
canonical_smiles = [None] * n_total
n_invalid = 0
n_organometallic = 0

for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    for i in range(start, end):
        smi = df_train.iloc[i]['PUBCHEM_EXT_DATASOURCE_SMILES']
        can = standardize_molecule(smi)
        canonical_smiles[i] = can
        if can is None:
            # Track why it was rejected
            mol_test = Chem.MolFromSmiles(str(smi))
            if mol_test is not None and not is_organic(mol_test):
                n_organometallic += 1
            else:
                n_invalid += 1
    if end % 20000 == 0 or end == n_total:
        elapsed = time.time() - t0
        rate = end / max(elapsed, 1)
        eta_min = (n_total - end) / max(rate, 1) / 60
        print(f"  Processed {end:,} / {n_total:,} ({end/n_total*100:.1f}%) — {rate:.0f}/s, ETA {eta_min:.1f} min")

df_train['canonical_smiles'] = canonical_smiles

# Drop compounds that failed standardization
n_before_std = len(df_train)
df_train = df_train.dropna(subset=['canonical_smiles']).copy().reset_index(drop=True)
n_after_std = len(df_train)
n_rejected = n_before_std - n_after_std
print(f"\n  Standardization complete in {(time.time()-t0)/60:.1f} min")
print(f"  Rejected: {n_rejected:,} compounds (organometallic={n_organometallic:,}, invalid={n_invalid:,})")
print(f"  Remaining: {len(df_train):,}")

# =====================================================================
# 4. DEDUPLICATION (Module 1.5)
# =====================================================================
print(f"\n[4/6] Deduplicating on canonical SMILES...")

# Some compounds may have the same canonical SMILES (different SIDs, same structure)
# Rule: on label conflict, keep Active (label=1)
print(f"  Total compounds before dedup: {len(df_train):,}")
print(f"  Unique canonical SMILES: {df_train['canonical_smiles'].nunique():,}")

# Group by canonical SMILES, check for label conflicts
label_conflict = df_train.groupby('canonical_smiles')['label'].nunique()
n_conflict = (label_conflict > 1).sum()
print(f"  SMILES with conflicting labels (some Active, some Inactive): {n_conflict:,}")

# Deduplicate: keep Active on conflict
# Sort by label DESCENDING (1 first), then drop_duplicates keeping first
df_train_sorted = df_train.sort_values('label', ascending=False).reset_index(drop=True)
df_train_dedup = df_train_sorted.drop_duplicates(subset='canonical_smiles', keep='first').reset_index(drop=True)
n_after_dedup = len(df_train_dedup)
print(f"  Compounds after dedup (Active wins on conflict): {n_after_dedup:,}")

n_actives_dedup = (df_train_dedup['label'] == 1).sum()
n_inactives_dedup = (df_train_dedup['label'] == 0).sum()
print(f"  Actives:   {n_actives_dedup:,}")
print(f"  Inactives: {n_inactives_dedup:,}")
print(f"  Imbalance: {n_inactives_dedup / max(n_actives_dedup, 1):.2f} : 1")

df_train = df_train_dedup

# =====================================================================
# 5. CROSS-REFERENCE WITH COUNTER-SCREENS + VALIDATION (SMILES-based)
# =====================================================================
print(f"\n[5/6] Cross-referencing with counter-screens and validation holdout...")
print(f"  Using SMILES-based matching (CID overlap was too low in pilot)")

# Build a SMILES → label/status map for ALL training compounds
train_smiles_set = set(df_train['canonical_smiles'])
print(f"  Train pool canonical SMILES: {len(train_smiles_set):,}")

# ---- 5a. Luciferase counter (AID 686933) ----
print(f"\n  --- AID 686933 (luciferase biochemical counter) ---")
try:
    df_luc = pd.read_csv(os.path.join(DATA_DIR, 'AID_686933_datatable.csv'),
                         skiprows=SKIP_ROWS, low_memory=False)
    print(f"  Loaded {len(df_luc):,} compounds from AID 686933")
    print(f"  Phenotype distribution: {df_luc['Phenotype'].value_counts().to_dict()}")

    # Standardize counter-screen SMILES too (for matching)
    print(f"  Standardizing AID 686933 SMILES...")
    luc_canonical = []
    for smi in df_luc['PUBCHEM_EXT_DATASOURCE_SMILES']:
        luc_canonical.append(standardize_molecule(smi))
    df_luc['canonical_smiles'] = luc_canonical
    df_luc_clean = df_luc.dropna(subset=['canonical_smiles']).copy()
    print(f"  After standardization: {len(df_luc_clean)} (rejected {len(df_luc) - len(df_luc_clean)})")

    # Match against training pool
    luc_in_train = df_luc_clean[df_luc_clean['canonical_smiles'].isin(train_smiles_set)]
    print(f"  Overlap with train pool: {len(luc_in_train)}")
    if len(luc_in_train) > 0:
        print(f"    Phenotype of overlaps: {luc_in_train['Phenotype'].value_counts().to_dict()}")

    # Build artifact flag
    luc_artifact_smiles = set(luc_in_train[luc_in_train['Phenotype'] == 'Inhibitor']['canonical_smiles'])
    luc_genuine_smiles = set(luc_in_train[luc_in_train['Phenotype'] == 'Inactive']['canonical_smiles'])
    print(f"    → Luciferase inhibitor artifacts in train: {len(luc_artifact_smiles)}")
    print(f"    → Genuine (inactive in counter):           {len(luc_genuine_smiles)}")

    # Save full counter-screen for Stage 7
    df_luc_clean.to_csv(os.path.join(OUTPUT_DIR, 'luciferase_counters.csv'), index=False)
    print(f"  Saved luciferase_counters.csv")

except FileNotFoundError:
    print(f"  WARNING: AID_686933_datatable.csv not found.")
    luc_artifact_smiles = set()
    luc_genuine_smiles = set()

# ---- 5b. Viability counter (AID 686921) ----
print(f"\n  --- AID 686921 (cell viability counter) ---")
try:
    df_via = pd.read_csv(os.path.join(DATA_DIR, 'AID_686921_datatable.csv'),
                         skiprows=SKIP_ROWS, low_memory=False)
    print(f"  Loaded {len(df_via):,} compounds from AID 686921")
    print(f"  Phenotype distribution: {df_via['Phenotype'].value_counts().to_dict()}")

    print(f"  Standardizing AID 686921 SMILES...")
    via_canonical = []
    for smi in df_via['PUBCHEM_EXT_DATASOURCE_SMILES']:
        via_canonical.append(standardize_molecule(smi))
    df_via['canonical_smiles'] = via_canonical
    df_via_clean = df_via.dropna(subset=['canonical_smiles']).copy()
    print(f"  After standardization: {len(df_via_clean)} (rejected {len(df_via) - len(df_via_clean)})")

    via_in_train = df_via_clean[df_via_clean['canonical_smiles'].isin(train_smiles_set)]
    print(f"  Overlap with train pool: {len(via_in_train)}")
    if len(via_in_train) > 0:
        print(f"    Phenotype of overlaps: {via_in_train['Phenotype'].value_counts().to_dict()}")

    via_artifact_smiles = set(via_in_train[via_in_train['Phenotype'].isin(['Activator', 'Inhibitor'])]['canonical_smiles'])
    via_genuine_smiles = set(via_in_train[via_in_train['Phenotype'] == 'Inactive']['canonical_smiles'])
    print(f"    → Cytotoxic artifacts in train: {len(via_artifact_smiles)}")
    print(f"    → Genuine (not cytotoxic):      {len(via_genuine_smiles)}")

    df_via_clean.to_csv(os.path.join(OUTPUT_DIR, 'viability_counters.csv'), index=False)
    print(f"  Saved viability_counters.csv")

except FileNotFoundError:
    print(f"  WARNING: AID_686921_datatable.csv not found.")
    via_artifact_smiles = set()
    via_genuine_smiles = set()

# ---- 5c. Validation holdout (AID 493107) ----
print(f"\n  --- AID 493107 (validation dose-response, to be held out) ---")
try:
    df_val = pd.read_csv(os.path.join(DATA_DIR, 'AID_493107_datatable.csv'),
                         skiprows=SKIP_ROWS, low_memory=False)
    print(f"  Loaded {len(df_val):,} compounds from AID 493107")

    print(f"  Standardizing AID 493107 SMILES...")
    val_canonical = []
    for smi in df_val['PUBCHEM_EXT_DATASOURCE_SMILES']:
        val_canonical.append(standardize_molecule(smi))
    df_val['canonical_smiles'] = val_canonical
    df_val_clean = df_val.dropna(subset=['canonical_smiles']).copy()
    print(f"  After standardization: {len(df_val_clean)}")

    # Compounds overlapping with train pool must be HELD OUT from training
    val_in_train_mask = df_train['canonical_smiles'].isin(set(df_val_clean['canonical_smiles']))
    n_val_overlap = val_in_train_mask.sum()
    print(f"  Validation compounds overlapping train pool: {n_val_overlap}")
    print(f"    → These will be EXCLUDED from training (held out for Stage 7)")

    # Save validation holdout
    df_val_clean.to_csv(os.path.join(OUTPUT_DIR, 'validation_holdout.csv'), index=False)
    print(f"  Saved validation_holdout.csv ({len(df_val_clean):,} compounds)")

except FileNotFoundError:
    print(f"  WARNING: AID_493107_datatable.csv not found.")
    val_in_train_mask = pd.Series([False] * len(df_train))

# =====================================================================
# 6. FINAL LABELING AND FILTERING
# =====================================================================
print(f"\n[6/6] Applying final labels and filtering...")

# Initialize status columns
df_train['luc_counter_status'] = 'not_tested'
df_train['via_counter_status'] = 'not_tested'
df_train['validation_holdout'] = val_in_train_mask.values

# Mark counter-screen overlaps
df_train.loc[df_train['canonical_smiles'].isin(luc_artifact_smiles), 'luc_counter_status'] = 'artifact'
df_train.loc[df_train['canonical_smiles'].isin(luc_genuine_smiles), 'luc_counter_status'] = 'genuine'
df_train.loc[df_train['canonical_smiles'].isin(via_artifact_smiles), 'via_counter_status'] = 'artifact'
df_train.loc[df_train['canonical_smiles'].isin(via_genuine_smiles), 'via_counter_status'] = 'genuine'

# Labeling:
#   label = 1   genuine active (not flagged artifact, not validation holdout)
#   label = 0   inactive
#   label = -1  counter-screen artifact (excluded from training)
#   label = 2   validation holdout (excluded from training, reserved for Stage 7)
artifact_mask = (df_train['luc_counter_status'] == 'artifact') | (df_train['via_counter_status'] == 'artifact')
val_holdout_mask = df_train['validation_holdout']

df_train.loc[artifact_mask, 'label'] = -1
df_train.loc[val_holdout_mask & (df_train['label'].isin([0, 1])), 'label'] = 2

# Report final label distribution
print(f"\n  Final label distribution:")
label_names = {-1: 'artifact (excluded)', 0: 'inactive', 1: 'genuine active', 2: 'validation holdout'}
for label, count in df_train['label'].value_counts().sort_index().items():
    name = label_names.get(int(label), 'unknown')
    print(f"    label={int(label):>2} ({name:<22}): {count:>10,}")

# Training pool: only label in {0, 1}
df_training = df_train[df_train['label'].isin([0, 1])].copy().reset_index(drop=True)
df_excluded = df_train[df_train['label'].isin([-1, 2])].copy().reset_index(drop=True)

n_train_act = (df_training['label'] == 1).sum()
n_train_inact = (df_training['label'] == 0).sum()
print(f"\n  === TRAINING POOL ===")
print(f"    Total: {len(df_training):,}")
print(f"    Actives: {n_train_act:,}")
print(f"    Inactives: {n_train_inact:,}")
print(f"    Imbalance: {n_train_inact / max(n_train_act, 1):.2f} : 1")
print(f"    Active prevalence: {n_train_act / len(df_training) * 100:.3f}%")
print(f"\n  === EXCLUDED ===")
print(f"    Artifacts: {(df_excluded['label'] == -1).sum()}")
print(f"    Validation holdouts: {(df_excluded['label'] == 2).sum()}")

# =====================================================================
# 7. EXTRACT BEMIS-MURCKO SCAFFOLDS (for Stage 3 split)
# =====================================================================
print(f"\nExtracting Bemis-Murcko scaffolds for {len(df_training):,} training compounds...")

from rdkit.Chem.Scaffolds import MurckoScaffold

def extract_bemis_murcko(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return 'Invalid_SMILES'
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if not scaffold_smiles:
            return 'Acyclic/Linear'
        return scaffold_smiles
    except Exception:
        return 'Extraction_Error'

t0 = time.time()
batch_size = 5000
n_total = len(df_training)
df_training['bm_scaffold'] = 'Pending'

for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    df_training.iloc[start:end, df_training.columns.get_loc('bm_scaffold')] = (
        df_training.iloc[start:end]['canonical_smiles'].apply(extract_bemis_murcko).values
    )
    if end % 20000 == 0 or end == n_total:
        elapsed = time.time() - t0
        rate = end / max(elapsed, 1)
        eta_min = (n_total - end) / max(rate, 1) / 60
        print(f"  Processed {end:,} / {n_total:,} ({end/n_total*100:.1f}%) — {rate:.0f}/s, ETA {eta_min:.1f} min")

# Drop compounds with unparseable SMILES (very few, if any)
n_before_scaffold = len(df_training)
df_training = df_training[~df_training['bm_scaffold'].isin(['Invalid_SMILES', 'Extraction_Error'])].copy().reset_index(drop=True)
n_removed_scaffold = n_before_scaffold - len(df_training)
print(f"  Removed {n_removed_scaffold} compounds with unparseable SMILES")

# =====================================================================
# 8. INTERNAL POTENCY DATA SUMMARY
# =====================================================================
print(f"\n=== INTERNAL POTENCY DATA (for Stage 7) ===")
# AID 504467 has Hill-fit AC50 for many actives
if 'Potency' in df_training.columns:
    actives_with_potency = df_training[(df_training['label'] == 1) & (df_training['Potency'].notna())]
    print(f"  Actives with Hill-fit AC50: {len(actives_with_potency):,} / {(df_training['label']==1).sum():,}")
    if len(actives_with_potency) > 0:
        print(f"  AC50 (µM): min={actives_with_potency['Potency'].min():.4f}, "
              f"median={actives_with_potency['Potency'].median():.4f}, "
              f"max={actives_with_potency['Potency'].max():.4f}")
        if 'Fit_R2' in df_training.columns:
            print(f"  Fit R²:    min={actives_with_potency['Fit_R2'].min():.3f}, "
                  f"median={actives_with_potency['Fit_R2'].median():.3f}, "
                  f"max={actives_with_potency['Fit_R2'].max():.3f}")

# =====================================================================
# 9. TOP SCAFFOLDS (sanity check)
# =====================================================================
print(f"\n=== TOP 15 BEMIS-MURCKO SCAFFOLDS (entire training pool) ===")
print(df_training['bm_scaffold'].value_counts().head(15).to_string())

print(f"\n=== TOP 15 SCAFFOLDS AMONG ACTIVES ===")
actives_df = df_training[df_training['label'] == 1]
if len(actives_df) > 0:
    print(actives_df['bm_scaffold'].value_counts().head(15).to_string())

# =====================================================================
# 10. SAVE OUTPUTS
# =====================================================================
print(f"\n=== SAVING OUTPUTS ===")

# Rename columns for clarity
rename_map = {
    'PUBCHEM_SID': 'pubchem_sid',
    'PUBCHEM_CID': 'pubchem_cid',
    'PUBCHEM_ACTIVITY_OUTCOME': 'activity_outcome',
    'PUBCHEM_ACTIVITY_SCORE': 'activity_score',
    'Phenotype': 'phenotype',
    'Potency': 'ac50_umol',
    'Efficacy': 'efficacy_pct',
    'Fit_LogAC50': 'fit_log_ac50',
    'Fit_R2': 'fit_r2',
    'Fit_CurveClass': 'fit_curve_class',
    'Curve_Description': 'curve_description',
    'Max_Response': 'max_response',
    'Compound QC': 'compound_qc',
}
cols_to_keep = [c for c in rename_map.keys() if c in df_training.columns]
df_training_clean = df_training[cols_to_keep + ['canonical_smiles', 'bm_scaffold', 'label',
                                                 'luc_counter_status', 'via_counter_status',
                                                 'validation_holdout']].rename(columns=rename_map)

output_train = os.path.join(OUTPUT_DIR, 'train_pool.csv')
df_training_clean.to_csv(output_train, index=False)
print(f"  Saved {output_train} ({len(df_training_clean):,} compounds)")

# Save excluded compounds
df_excluded_clean = df_excluded.rename(columns=rename_map)
output_excluded = os.path.join(OUTPUT_DIR, 'excluded_compounds.csv')
df_excluded_clean.to_csv(output_excluded, index=False)
print(f"  Saved {output_excluded} ({len(df_excluded_clean):,} compounds)")

# =====================================================================
# 11. CURATION REPORT
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'curation_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 1 CURATION REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("DATASET SOURCES\n")
    f.write(f"  Primary:    AID 504467 (PubChem qHTS for ELG1/ATAD5 inhibitors)\n")
    f.write(f"  Counters:   AID 686933 (luciferase) + AID 686921 (viability)\n")
    f.write(f"  Validation: AID 493107 (7-point dose-response, held out)\n\n")
    f.write("STANDARDIZATION (Module 1 spec)\n")
    f.write(f"  Total input: 330,280\n")
    f.write(f"  After dropping missing SID/SMILES: 330,279\n")
    f.write(f"  After dropping Inconclusive: 243,379\n")
    f.write(f"  After standardization (organometallic/inorganic rejected): {len(df_train):,}\n")
    f.write(f"    Rejected as organometallic: {n_organometallic:,}\n")
    f.write(f"    Rejected as invalid: {n_invalid:,}\n")
    f.write(f"  After deduplication (Active wins on conflict): {len(df_training):,}\n")
    f.write(f"    Label conflicts resolved: {n_conflict:,}\n\n")
    f.write("TRAINING POOL\n")
    f.write(f"  Total: {len(df_training):,}\n")
    f.write(f"  Actives: {n_train_act:,}\n")
    f.write(f"  Inactives: {n_train_inact:,}\n")
    f.write(f"  Imbalance: {n_train_inact / max(n_train_act, 1):.2f} : 1\n")
    f.write(f"  Unique scaffolds: {df_training['bm_scaffold'].nunique():,}\n\n")
    f.write("COUNTER-SCREEN OVERLAP (SMILES-based matching)\n")
    f.write(f"  AID 686933 (luciferase): {len(luc_artifact_smiles)} artifacts, {len(luc_genuine_smiles)} genuine\n")
    f.write(f"  AID 686921 (viability):  {len(via_artifact_smiles)} artifacts, {len(via_genuine_smiles)} genuine\n\n")
    f.write("VALIDATION HOLDOUT\n")
    f.write(f"  AID 493107 compounds held out: {n_val_overlap}\n\n")
    f.write("INTERNAL POTENCY DATA\n")
    if 'Potency' in df_training.columns:
        actives_with_potency = df_training[(df_training['label'] == 1) & (df_training['Potency'].notna())]
        f.write(f"  Actives with Hill-fit AC50: {len(actives_with_potency):,}\n")
        if len(actives_with_potency) > 0:
            f.write(f"  AC50 (µM): median={actives_with_potency['Potency'].median():.4f}\n\n")
    f.write("NEXT STEPS (Stage 2)\n")
    f.write(f"  - Generate ECFP6 count vectors (4096-bit, radius=3, sparse CSR uint16)\n")
    f.write(f"  - Pass canonical SMILES to Chemprop D-MPNN as raw input\n")
print(f"  Saved curation_report.txt")

print("\n" + "=" * 78)
print("STAGE 1 COMPLETE")
print("=" * 78)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  - train_pool.csv              ({len(df_training_clean):,} compounds)")
print(f"  - excluded_compounds.csv      ({len(df_excluded_clean):,} compounds)")
print(f"  - validation_holdout.csv      ({len(df_val_clean) if 'df_val_clean' in locals() else 0:,} compounds)")
print(f"  - luciferase_counters.csv     ({len(df_luc_clean) if 'df_luc_clean' in locals() else 0:,} compounds)")
print(f"  - viability_counters.csv      ({len(df_via_clean) if 'df_via_clean' in locals() else 0:,} compounds)")
print(f"  - curation_report.txt")
print(f"\nReady for Stage 2 (dual feature representation: ECFP6 counts + SMILES for D-MPNN).")
