"""
RepAIr — FDA Library Filter
Merges CLUE drug info + sample info, filters to FDA-approved (Launched).
Run before Stage 6.
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = 'data'
OUTPUT_FILE = 'fda_library/fda_approved.csv'

print("=" * 60)
print("RepAIr — Filter CLUE to FDA-Approved Drugs")
print("=" * 60)

# Load drug information
print("\nLoading drug information...")
df_drug = pd.read_csv(os.path.join(DATA_DIR, 'clue_drug_info.txt'),
                      sep='\t', skiprows=9, low_memory=False)
print(f"  Loaded {len(df_drug):,} compounds")
print(f"\n  Clinical phase distribution:")
print(df_drug['clinical_phase'].value_counts().to_string())

# Load sample information
print(f"\nLoading sample information...")
df_sample = pd.read_csv(os.path.join(DATA_DIR, 'clue_sample_info.txt'),
                        sep='\t', skiprows=9, low_memory=False)
df_sample_unique = df_sample.drop_duplicates(subset='pert_iname', keep='first').copy()
print(f"  Unique compounds: {len(df_sample_unique):,}")

# Merge
print(f"\nMerging drug + sample info...")
merged = df_drug.merge(
    df_sample_unique[['pert_iname', 'smiles', 'InChIKey', 'pubchem_cid']],
    on='pert_iname', how='inner'
)
print(f"  Merged: {len(merged):,} compounds with both annotation + SMILES")

# Filter to FDA-approved
print(f"\nFiltering to FDA-approved (clinical_phase='Launched')...")
fda_df = merged[merged['clinical_phase'] == 'Launched'].copy()
print(f"  FDA-approved drugs: {len(fda_df):,}")

# Clean SMILES (remove CLUE coordinate annotations)
def clean_smiles(smi):
    if pd.isna(smi): return None
    smi = str(smi).strip()
    idx = smi.find(' |')
    if idx > 0: smi = smi[:idx]
    return smi.strip()

fda_df['smiles_clean'] = fda_df['smiles'].apply(clean_smiles)

# Validate with RDKit
try:
    from rdkit import Chem
    print(f"\nValidating SMILES with RDKit...")
    valid_mask = []
    canonical_smiles = []
    for smi in fda_df['smiles_clean']:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mask.append(True)
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            valid_mask.append(False)
            canonical_smiles.append(None)
    fda_df = fda_df[valid_mask].copy()
    fda_df['SMILES'] = canonical_smiles
    print(f"  Valid SMILES: {len(fda_df):,}")
except ImportError:
    print(f"  (rdkit not installed — skipping validation)")
    fda_df['SMILES'] = fda_df['smiles_clean']

# Deduplicate by canonical SMILES
fda_df = fda_df.drop_duplicates(subset='SMILES', keep='first').reset_index(drop=True)
print(f"  After deduplication: {len(fda_df):,}")

# Build final CSV
final_df = pd.DataFrame({
    'Drug_Name': fda_df['pert_iname'],
    'SMILES': fda_df['SMILES'],
    'clinical_phase': fda_df['clinical_phase'],
    'moa': fda_df['moa'],
    'target': fda_df['target'],
    'disease_area': fda_df['disease_area'],
    'indication': fda_df['indication'],
    'InChIKey': fda_df['InChIKey'],
    'pubchem_cid': fda_df['pubchem_cid'],
})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {OUTPUT_FILE} ({len(final_df):,} FDA-approved drugs)")
