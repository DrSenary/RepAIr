"""
RepAIr — FDA Library Filter & Salt-Strip Standardizer
======================================================
Merges CLUE drug info + sample info, filters to FDA-approved (Launched),
then applies the 5-step RDKit parent-structure standardization pipeline so
duplicate salt formulations (e.g., doxazosin vs. doxazosin mesylate,
omeprazole vs. omeprazole-magnesium) collapse into single canonical parent
drug entries.

Run before Stage 6:
    python3 src/filter_fda_library.py

Outputs:
    data/fda_approved.csv  — standardized FDA-approved parent drugs

5-Step Standardization Pipeline (matches Stage 1 curation order):
    1. SaltRemover + LargestFragmentChooser(preferOrganic=True)  → strip counter-ions
    2. Filter to allowed organic elements {H, C, N, O, F, P, S, Cl, Br, I}
    3. Uncharger (neutralize formal charges) + Reionizer (pH 7.4)
    4. TautomerEnumerator.Canonicalize
    5. InChIKey connectivity-block dedup + parent-name normalization
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = 'data'
OUTPUT_FILE = 'data/fda_approved.csv'

print("=" * 78)
print("RepAIr — Filter CLUE to FDA-Approved Drugs & Salt-Strip Parent Structures")
print("=" * 78)

# ---------------------------------------------------------------------
# 1. Load CLUE drug + sample info
# ---------------------------------------------------------------------
print("\n[1/5] Loading CLUE drug + sample information...")
df_drug = pd.read_csv(os.path.join(DATA_DIR, 'clue_drug_info.txt'),
                      sep='\t', skiprows=9, low_memory=False)
print(f"  Loaded {len(df_drug):,} compounds from clue_drug_info.txt")
print(f"\n  Clinical phase distribution:")
print(df_drug['clinical_phase'].value_counts().to_string())

df_sample = pd.read_csv(os.path.join(DATA_DIR, 'clue_sample_info.txt'),
                        sep='\t', skiprows=9, low_memory=False)
df_sample_unique = df_sample.drop_duplicates(subset='pert_iname', keep='first').copy()
print(f"  Unique compounds in clue_sample_info.txt: {len(df_sample_unique):,}")

# Merge
print(f"\n[2/5] Merging drug + sample info on 'pert_iname'...")
merged = df_drug.merge(
    df_sample_unique[['pert_iname', 'smiles', 'InChIKey', 'pubchem_cid']],
    on='pert_iname', how='inner'
)
print(f"  Merged: {len(merged):,} compounds with both annotation + SMILES")

# Filter to FDA-approved
print(f"\n[3/5] Filtering to FDA-approved (clinical_phase='Launched')...")
fda_df = merged[merged['clinical_phase'] == 'Launched'].copy()
print(f"  FDA-approved drugs: {len(fda_df):,}")

# Clean SMILES (remove CLUE coordinate annotations)
def clean_smiles(smi):
    if pd.isna(smi):
        return None
    smi = str(smi).strip()
    idx = smi.find(' |')
    if idx > 0:
        smi = smi[:idx]
    return smi.strip()

fda_df['smiles_clean'] = fda_df['smiles'].apply(clean_smiles)

# ---------------------------------------------------------------------
# 2. RDKit 5-step parent standardization (matches Stage 1 curation)
# ---------------------------------------------------------------------
print(f"\n[4/5] Running 5-step RDKit parent standardization on FDA library...")
print(f"  Pipeline: SaltRemover → LargestFragmentChooser → element filter →")
print(f"           Uncharger → Reionizer → TautomerEnumerator → canonical SMILES")

try:
    from rdkit import Chem
    from rdkit.Chem import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize

    ALLOWED_ORGANIC_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I

    _salt_remover = SaltRemover.SaltRemover()
    _largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    _uncharger = rdMolStandardize.Uncharger()
    _reionizer = rdMolStandardize.Reionizer()
    _tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def standardize_parent(smiles):
        """Apply the 5-step RDKit parent-structure standardization.

        Returns (canonical_smiles, inchikey_first_block) or (None, None) on rejection.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return None, None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # Step 1: Desalt FIRST — strips counter-ions (Na+, K+, Mg2+, Ca2+,
        # mesylate, HCl, etc.) and keeps the largest organic parent fragment.
        try:
            mol = _salt_remover.StripMol(mol)
            mol = _largest_fragment_chooser.choose(mol)
        except Exception:
            return None, None
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None, None

        # Step 2: Element filter — reject genuine organometallics that survive
        # LargestFragmentChooser (e.g., ferrocene, cisplatin analogs).
        atom_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if not atom_nums.issubset(ALLOWED_ORGANIC_ATOMS):
            return None, None

        # Step 3: Neutralize + reionize at physiological pH (~7.4)
        try:
            mol = _uncharger.uncharge(mol)
            mol = _reionizer.reionize(mol)
        except Exception:
            return None, None

        # Step 4: Tautomer canonicalization
        try:
            mol = _tautomer_enumerator.Canonicalize(mol)
        except Exception:
            return None, None

        # Step 5: Canonical SMILES + InChIKey connectivity block
        try:
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            ik_first = inchikey.split("-")[0] if inchikey else can_smiles
            if not can_smiles or can_smiles == '':
                return None, None
            return can_smiles, ik_first
        except Exception:
            return None, None

    # Apply standardization
    print(f"  Standardizing {len(fda_df):,} FDA drugs...")
    std_results = fda_df['smiles_clean'].apply(standardize_parent)
    fda_df['SMILES'] = [r[0] for r in std_results]
    fda_df['inchikey_first'] = [r[1] for r in std_results]
    n_before = len(fda_df)
    fda_df = fda_df.dropna(subset=['SMILES']).copy()
    print(f"  Valid parent SMILES after standardization: {len(fda_df):,} "
          f"(rejected {n_before - len(fda_df):,} inorganic/invalid)")

except ImportError:
    print(f"  (rdkit not installed — skipping standardization)")
    fda_df['SMILES'] = fda_df['smiles_clean']
    fda_df['inchikey_first'] = None

# ---------------------------------------------------------------------
# 3. Parent-name normalization (strip salt suffixes)
# ---------------------------------------------------------------------
SALT_SUFFIX_PATTERN = re.compile(
    r"[\s\-]+(mesylate|hydrochloride|hcl|magnesium|sodium|potassium|calcium|"
    r"besylate|maleate|fumarate|tartrate|citrate|sulfate|sulphate|acetate|"
    r"phosphate|succinate|bromide|chloride|iodide|nitrate|lactate|tosylate|"
    r"dihydrate|monohydrate|trihydrate|hemihydrate|anhydrous)\b",
    flags=re.IGNORECASE,
)

def normalize_drug_name(name):
    """Strip common pharmaceutical salt/hydrate suffixes to get the parent INN name."""
    if pd.isna(name):
        return name
    cleaned = SALT_SUFFIX_PATTERN.sub("", str(name).strip())
    return cleaned.strip(" -")

fda_df['parent_name'] = fda_df['pert_iname'].apply(normalize_drug_name)

# ---------------------------------------------------------------------
# 4. Deduplicate by InChIKey connectivity block (preferred) or canonical SMILES
# ---------------------------------------------------------------------
print(f"\n[5/5] Deduplicating salt forms by InChIKey connectivity block...")
n_before_dedup = len(fda_df)

# Sort by SMILES (descending — keeps first occurrence after sorting by pert_iname)
fda_df = fda_df.sort_values(['parent_name', 'pert_iname']).reset_index(drop=True)

# Drop duplicates: prefer InChIKey first-block if available, else canonical SMILES
if 'inchikey_first' in fda_df.columns and fda_df['inchikey_first'].notna().any():
    dedup_subset = ['inchikey_first']
    print(f"  Using InChIKey connectivity block for deduplication")
else:
    dedup_subset = ['SMILES']
    print(f"  Using canonical SMILES for deduplication (InChIKey not available)")

# Keep the first occurrence per parent structure (alphabetical by parent_name)
fda_df = fda_df.drop_duplicates(subset=dedup_subset, keep='first').reset_index(drop=True)
print(f"  Deduplication removed {n_before_dedup - len(fda_df):,} duplicate salt forms")
print(f"  Final unique FDA-approved parent drugs: {len(fda_df):,}")

# ---------------------------------------------------------------------
# 5. Build final CSV
# ---------------------------------------------------------------------
final_df = pd.DataFrame({
    'Drug_Name': fda_df['parent_name'],            # normalized parent INN name
    'original_pert_iname': fda_df['pert_iname'],   # original CLUE name (for traceability)
    'SMILES': fda_df['SMILES'],                    # canonical parent SMILES
    'InChIKey': fda_df['inchikey_first'],          # connectivity block (first 14 chars)
    'clinical_phase': fda_df['clinical_phase'],
    'moa': fda_df['moa'],
    'target': fda_df['target'],
    'disease_area': fda_df['disease_area'],
    'indication': fda_df['indication'],
    'pubchem_cid': fda_df['pubchem_cid'],
})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

# Summary
n_duplex_drugs = final_df['Drug_Name'].nunique()
print(f"\n{'=' * 78}")
print(f"✓ Saved {OUTPUT_FILE}")
print(f"  Total rows:           {len(final_df):,}")
print(f"  Unique parent drugs:  {n_duplex_drugs:,}")
print(f"  Salt duplicates merged: {n_before_dedup - len(final_df):,}")
print(f"{'=' * 78}")

# Show a few examples of merged salts (if any)
if n_before_dedup > len(final_df):
    print(f"\nSample of merged salt forms:")
    merged_examples = final_df[final_df['original_pert_iname'] != final_df['Drug_Name']][
        ['Drug_Name', 'original_pert_iname', 'SMILES']
    ].head(10)
    if len(merged_examples) > 0:
        print(merged_examples.to_string(index=False))
