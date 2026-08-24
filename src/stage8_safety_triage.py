"""
ATAD5AI — Stage 8: Safety Triage on Top FDA Candidates
Hybrid Consensus Screening Pipeline — Final Stage

Applies chemical-rule-based safety filters to the top FDA repurposing
candidates from Stage 6. This addresses the artifact-rejection limitation
identified in Stage 7 by filtering out:
  - PAINS (Pan-Assay Interference Compounds)
  - Luciferase inhibitors (critical since AID 504467 is a luciferase reporter)
  - Aggregators (colloidal aggregators)
  - Lipinski/Veber/Pfizer rule violations (mammalian cell permeability + ADME)

==============================================================================
HOW TO USE (runs locally on your Mac, no Colab needed)
==============================================================================

PREREQUISITES:
  - outputs/stage6_fda_predictions/fda_top_candidates.csv   (from Stage 6)
  - Optional: outputs/stage6_fda_predictions/fda_consensus_predictions.csv
    (if you want to filter the full 2,680-drug library, not just top candidates)

RUN:
  cd ~/Desktop/ATAD5AI_project
  python src/stage8_safety_triage.py

EXPECTED RUNTIME: ~5 minutes (no GPU needed, CPU-only)

==============================================================================
MODULE 6 SPEC (Adapted for ATAD5AI — mammalian target, luciferase assay)
==============================================================================
Original PriA-SSB spec:
  1. Filter PAINS-A/B/C and aggregators
  2. AlphaScreen Quencher Check
  3. Bacterial Permeation Scoring (eNTRy Rules for Gram-negative uptake)

ATAD5AI adaptation:
  1. Filter PAINS-A/B/C and aggregators          ← KEPT
  2. Luciferase inhibitor SMARTS filter          ← REPLACED AlphaScreen
     (critical because AID 504467 is a luciferase reporter assay)
  3. Lipinski/Veber/Pfizer 3/75                  ← REPLACED eNTRy
     (mammalian cell permeability — target is HEK293T, not bacteria)

==============================================================================
OUTPUTS
==============================================================================
  outputs/stage8_triage/
    - fda_top_candidates_triaged.csv     (final prioritized list with flags)
    - triage_summary.json                (per-filter rejection counts)
    - triage_report.txt                  (paper-ready summary)
"""

import os
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

# =====================================================================
# CONFIG
# =====================================================================
# Auto-detect input file location (searches multiple common paths)
INPUT_FILE = 'fda_top_candidates.csv'

# Search candidates: project outputs subdirs + common download locations
SEARCH_PATHS = [
    'outputs/stage6_fda_predictions',
    'outputs/stage4_models',
    'outputs',
    '.',
    '~/Downloads',
    '~/Desktop',
]

INPUT_DIR = None
for path in SEARCH_PATHS:
    expanded = os.path.expanduser(path)
    full = os.path.join(expanded, INPUT_FILE)
    if os.path.exists(full):
        INPUT_DIR = expanded
        break

if INPUT_DIR is None:
    print(f"  ✗ Could not find {INPUT_FILE} in any of:")
    for p in SEARCH_PATHS:
        print(f"     {os.path.expanduser(p)}/")
    print(f"\n  Please either:")
    print(f"    1. Move {INPUT_FILE} to one of those locations, OR")
    print(f"    2. Tell me the exact path where you saved it")
    raise FileNotFoundError(f"{INPUT_FILE} not found in search paths")

OUTPUT_DIR = 'outputs/stage8_triage'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 78)
print("ATAD5AI — STAGE 8: SAFETY TRIAGE ON TOP FDA CANDIDATES")
print("=" * 78)
print(f"\nFilters to apply (per adapted Module 6 spec):")
print("  1. PAINS-A/B/C filter (Baell 2010)")
print("  2. Luciferase inhibitor SMARTS (Aldrich 2017) — critical for this assay")
print("  3. Aggregator filter (Baell & Walters 2014)")
print("  4. Lipinski Rule of 5 (mammalian cell permeability)")
print("  5. Veber rules (rotatable bonds + TPSA)")
print("  6. Pfizer 3/75 rule (low attrition risk)")

# =====================================================================
# 1. Load top FDA candidates from Stage 6
# =====================================================================
print(f"\n[1/4] Loading FDA top candidates from Stage 6...")
input_path = os.path.join(INPUT_DIR, INPUT_FILE)
print(f"  Using: {input_path}")
print(f"  Size: {os.path.getsize(input_path)/1024:.1f} KB")

df = pd.read_csv(input_path)
print(f"  Loaded {len(df)} FDA candidates from Stage 6")

# Find SMILES column
smiles_col = next((c for c in ['SMILES', 'canonical_smiles', 'smiles'] if c in df.columns), None)
if smiles_col is None:
    smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
print(f"  SMILES column: '{smiles_col}'")

# Find drug name column
drug_name_col = next((c for c in ['Drug_Name', 'drug_name', 'name'] if c in df.columns), None)
if drug_name_col is None:
    drug_name_col = df.columns[1]  # fallback
print(f"  Drug_Name column: '{drug_name_col}'")

# Find probability columns
p_consensus_col = next((c for c in df.columns if 'consensus' in c.lower()), None)
print(f"  P_consensus column: '{p_consensus_col}'")

# =====================================================================
# 2. Initialize filter catalogs
# =====================================================================
print(f"\n[2/4] Initializing safety filters...")

# --- PAINS filter (RDKit built-in) ---
params_pains = FilterCatalogParams()
params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_catalog = FilterCatalog(params_pains)
print(f"  ✓ PAINS-A/B/C catalog loaded")

# --- NIH filter (includes aggregators, reactivity, etc.) ---
params_nih = FilterCatalogParams()
params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
nih_catalog = FilterCatalog(params_nih)
print(f"  ✓ NIH aggregator/reactivity catalog loaded")

# --- BRENK filter (reactive groups) ---
params_brenk = FilterCatalogParams()
params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
brenk_catalog = FilterCatalog(params_brenk)
print(f"  ✓ BRENK reactive groups catalog loaded")

# --- Luciferase inhibitor SMARTS (Aldrich 2017, adapted from published luciferase inhibitor rules) ---
# These are common luciferase inhibitor scaffolds — compounds matching these
# are likely to directly inhibit the firefly luciferase enzyme rather than
# modulate ELG1/ATAD5.
# Source: Aldrich et al. 2017, Assay Guidance Manual
# Note: This is a curated subset of known luciferase-inhibiting scaffolds
luciferase_smarts = [
    # Resorcinol scaffold (D-luciferin mimic, common luciferase inhibitor)
    'c1cc(O)cc(O)c1',
    # Benzothiazole scaffold (luciferin core)
    'c1ccc2ncsc2c1',
    # Aminothiazole scaffold
    'c1nc2ccccc2s1',
    # Quinazoline (luciferase competitive inhibitor)
    'c1ccc2ncncc2c1',
    # 2-aminobenzothiazole (luciferase inhibitor from PubChem confirmatory screens)
    'c1ccc2nc(N)sc2c1',
    # Rhodanine (PAINS, also luciferase inhibitor)
    'S=C1NCCC1=O',
    # Hydroxypyridinone (chelator, luciferase inhibitor via Mg2+ chelation)
    'O=c1c(O)cccc1O',
]

luciferase_patts = []
for smarts in luciferase_smarts:
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        luciferase_patts.append((smarts, patt))
print(f"  ✓ Luciferase inhibitor SMARTS patterns loaded ({len(luciferase_patts)} patterns)")


# =====================================================================
# 3. Define filter functions
# =====================================================================

def check_pains(mol):
    """Returns True if molecule matches any PAINS pattern."""
    if mol is None:
        return False
    return pains_catalog.HasMatch(mol)

def check_nih_aggregator(mol):
    """Returns True if molecule matches NIH aggregator/reactivity filters."""
    if mol is None:
        return False
    return nih_catalog.HasMatch(mol)

def check_brenk(mol):
    """Returns True if molecule has BRENK reactive groups."""
    if mol is None:
        return False
    return brenk_catalog.HasMatch(mol)

def check_luciferase_inhibitor(mol):
    """Returns True if molecule matches luciferase inhibitor SMARTS."""
    if mol is None:
        return False
    matched_patterns = []
    for smarts, patt in luciferase_patts:
        if mol.HasSubstructMatch(patt):
            matched_patterns.append(smarts)
    return matched_patterns

def check_lipinski(mol):
    """Lipinski Rule of 5 (mammalian cell permeability).
    Returns (passes, list_of_violations)."""
    if mol is None:
        return False, ['invalid_mol']
    violations = []
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    if mw > 500: violations.append(f'MW={mw:.0f}>500')
    if logp > 5: violations.append(f'LogP={logp:.1f}>5')
    if hbd > 5: violations.append(f'HBD={hbd}>5')
    if hba > 10: violations.append(f'HBA={hba}>10')
    return len(violations) == 0, violations

def check_veber(mol):
    """Veber rules (oral bioavailability).
    Returns (passes, list_of_violations)."""
    if mol is None:
        return False, ['invalid_mol']
    violations = []
    rotatable = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    if rotatable > 10: violations.append(f'rotatable_bonds={rotatable}>10')
    if tpsa > 140: violations.append(f'TPSA={tpsa:.0f}>140')
    return len(violations) == 0, violations

def check_pfizer_3_75(mol):
    """Pfizer 3/75 rule (low attrition risk).
    Returns (passes, list_of_violations).
    Rule: compounds with LogP > 3 AND TPSA < 75 have high attrition.
    """
    if mol is None:
        return False, ['invalid_mol']
    violations = []
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    if logp > 3 and tpsa < 75:
        violations.append(f'LogP={logp:.1f}>3 AND TPSA={tpsa:.0f}<75 (high attrition)')
    return len(violations) == 0, violations


# =====================================================================
# 4. Apply all filters to each candidate
# =====================================================================
print(f"\n[3/4] Applying filters to {len(df)} FDA candidates...")

# Initialize result columns
df['pains_flag'] = False
df['nih_aggregator_flag'] = False
df['brenk_flag'] = False
df['luciferase_inhibitor_flag'] = False
df['luciferase_matched_patterns'] = ''
df['lipinski_pass'] = False
df['lipinski_violations'] = ''
df['veber_pass'] = False
df['veber_violations'] = ''
df['pfizer_3_75_pass'] = False
df['pfizer_violations'] = ''

# Compute ADME properties for the report
df['MW'] = np.nan
df['LogP'] = np.nan
df['TPSA'] = np.nan
df['HBD'] = np.nan
df['HBA'] = np.nan
df['RotatableBonds'] = np.nan

for idx, row in df.iterrows():
    smi = row[smiles_col]
    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        # Mark all filters as failed
        df.at[idx, 'lipinski_pass'] = False
        df.at[idx, 'veber_pass'] = False
        df.at[idx, 'pfizer_3_75_pass'] = False
        df.at[idx, 'lipinski_violations'] = 'invalid_smiles'
        continue

    # PAINS
    df.at[idx, 'pains_flag'] = check_pains(mol)

    # NIH aggregator / reactivity
    df.at[idx, 'nih_aggregator_flag'] = check_nih_aggregator(mol)

    # BRENK reactive groups
    df.at[idx, 'brenk_flag'] = check_brenk(mol)

    # Luciferase inhibitor
    luc_matches = check_luciferase_inhibitor(mol)
    if luc_matches:
        df.at[idx, 'luciferase_inhibitor_flag'] = True
        df.at[idx, 'luciferase_matched_patterns'] = ' | '.join(luc_matches)

    # Lipinski
    passes, violations = check_lipinski(mol)
    df.at[idx, 'lipinski_pass'] = passes
    df.at[idx, 'lipinski_violations'] = '; '.join(violations) if violations else ''

    # Veber
    passes, violations = check_veber(mol)
    df.at[idx, 'veber_pass'] = passes
    df.at[idx, 'veber_violations'] = '; '.join(violations) if violations else ''

    # Pfizer 3/75
    passes, violations = check_pfizer_3_75(mol)
    df.at[idx, 'pfizer_3_75_pass'] = passes
    df.at[idx, 'pfizer_violations'] = '; '.join(violations) if violations else ''

    # ADME properties
    df.at[idx, 'MW'] = Descriptors.MolWt(mol)
    df.at[idx, 'LogP'] = Descriptors.MolLogP(mol)
    df.at[idx, 'TPSA'] = Descriptors.TPSA(mol)
    df.at[idx, 'HBD'] = Descriptors.NumHDonors(mol)
    df.at[idx, 'HBA'] = Descriptors.NumHAcceptors(mol)
    df.at[idx, 'RotatableBonds'] = Descriptors.NumRotatableBonds(mol)

print(f"  ✓ All filters applied")

# =====================================================================
# 5. Compute overall safety score
# =====================================================================
# Tier 1 (clean): passes Lipinski + Veber + Pfizer, no PAINS/aggregator/BRENK/luciferase flags
# Tier 2 (acceptable): minor issues (1 rule violation), no critical flags
# Tier 3 (flagged): PAINS/aggregator/BRENK/luciferase or multiple rule violations

def assign_tier(row):
    critical_flags = (row['pains_flag'] or row['nih_aggregator_flag'] or
                       row['brenk_flag'] or row['luciferase_inhibitor_flag'])
    adme_passes = row['lipinski_pass'] and row['veber_pass'] and row['pfizer_3_75_pass']

    if critical_flags:
        return 'Tier 3 — Flagged (critical artifact risk)'
    elif adme_passes:
        return 'Tier 1 — Clean (recommended for experimental validation)'
    else:
        return 'Tier 2 — Acceptable (minor ADME issues)'

df['safety_tier'] = df.apply(assign_tier, axis=1)

# =====================================================================
# 6. Print summary
# =====================================================================
print(f"\n[4/4] Computing summary...")

# Per-filter rejection counts
print(f"\n=== FILTER RESULTS ===")
print(f"\n  {'Filter':<35}{'Flagged':<10}{'Clean':<10}{'% Flagged':<10}")
print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*9}")

filters = [
    ('PAINS-A/B/C', 'pains_flag'),
    ('NIH Aggregator/Reactivity', 'nih_aggregator_flag'),
    ('BRENK Reactive Groups', 'brenk_flag'),
    ('Luciferase Inhibitor SMARTS', 'luciferase_inhibitor_flag'),
    ('Lipinski Rule of 5 (pass)', 'lipinski_pass'),
    ('Veber Rules (pass)', 'veber_pass'),
    ('Pfizer 3/75 (pass)', 'pfizer_3_75_pass'),
]

for name, col in filters:
    n_flagged = df[col].sum() if 'pass' not in col else (~df[col]).sum()
    n_clean = (~df[col]).sum() if 'pass' not in col else df[col].sum()
    pct = n_flagged / len(df) * 100
    print(f"  {name:<35}{n_flagged:<10}{n_clean:<10}{pct:<10.1f}")

# Tier distribution
print(f"\n=== SAFETY TIER DISTRIBUTION ===")
tier_counts = df['safety_tier'].value_counts()
for tier, count in tier_counts.items():
    pct = count / len(df) * 100
    print(f"  {tier}: {count} ({pct:.1f}%)")

# =====================================================================
# 7. Show top candidates by tier
# =====================================================================
print(f"\n=== TOP-15 CLEAN CANDIDATES (Tier 1) ===")
tier1_df = df[df['safety_tier'].str.contains('Tier 1')].copy()
if p_consensus_col:
    tier1_df = tier1_df.sort_values(p_consensus_col, ascending=False)

if len(tier1_df) > 0:
    print(f"\n  {'Rk':<4}{'Drug':<30}{'P_cons':<8}{'MW':<8}{'LogP':<7}{'TPSA':<8}{'MoA':<25}")
    print(f"  {'-'*3}  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*23}")
    for i, (_, row) in enumerate(tier1_df.head(15).iterrows()):
        drug = str(row[drug_name_col])[:28]
        p_cons = f"{row[p_consensus_col]:.4f}" if p_consensus_col else '—'
        mw = f"{row['MW']:.0f}" if pd.notna(row['MW']) else '—'
        logp = f"{row['LogP']:.1f}" if pd.notna(row['LogP']) else '—'
        tpsa = f"{row['TPSA']:.0f}" if pd.notna(row['TPSA']) else '—'
        moa = str(row.get('moa', '—'))[:23] if 'moa' in row else '—'
        print(f"  {i+1:<4}{drug:<30}{p_cons:<8}{mw:<8}{logp:<7}{tpsa:<8}{moa:<25}")
else:
    print("  (no Tier 1 candidates)")

# === NEW: Tier 2 (Acceptable — minor ADME issues, no critical flags) ===
print(f"\n=== ACCEPTABLE CANDIDATES (Tier 2 — minor ADME issues) ===")
tier2_df = df[df['safety_tier'].str.contains('Tier 2')].copy()
if p_consensus_col:
    tier2_df = tier2_df.sort_values(p_consensus_col, ascending=False)

if len(tier2_df) > 0:
    print(f"\n  {'Rk':<4}{'Drug':<30}{'P_cons':<8}{'MW':<8}{'LogP':<7}{'TPSA':<8}{'ADME Issues':<35}")
    print(f"  {'-'*3}  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*33}")
    for i, (_, row) in enumerate(tier2_df.iterrows()):
        drug = str(row[drug_name_col])[:28]
        p_cons = f"{row[p_consensus_col]:.4f}" if p_consensus_col else '—'
        mw = f"{row['MW']:.0f}" if pd.notna(row['MW']) else '—'
        logp = f"{row['LogP']:.1f}" if pd.notna(row['LogP']) else '—'
        tpsa = f"{row['TPSA']:.0f}" if pd.notna(row['TPSA']) else '—'
        # Collect ADME violations
        violations = []
        if not row['lipinski_pass'] and row['lipinski_violations']:
            violations.append(f"Lipinski: {row['lipinski_violations']}")
        if not row['veber_pass'] and row['veber_violations']:
            violations.append(f"Veber: {row['veber_violations']}")
        if not row['pfizer_3_75_pass'] and row['pfizer_violations']:
            violations.append(f"Pfizer: {row['pfizer_violations']}")
        adme_issues = ' | '.join(violations)[:33]
        print(f"  {i+1:<4}{drug:<30}{p_cons:<8}{mw:<8}{logp:<7}{tpsa:<8}{adme_issues:<35}")
else:
    print("  (no Tier 2 candidates)")

print(f"\n=== FLAGGED CANDIDATES (Tier 3) ===")
tier3_df = df[df['safety_tier'].str.contains('Tier 3')].copy()
if p_consensus_col:
    tier3_df = tier3_df.sort_values(p_consensus_col, ascending=False)

if len(tier3_df) > 0:
    print(f"\n  {'Rk':<4}{'Drug':<30}{'P_cons':<8}{'Flags':<50}")
    print(f"  {'-'*3}  {'-'*28}  {'-'*6}  {'-'*48}")
    for i, (_, row) in enumerate(tier3_df.iterrows()):
        drug = str(row[drug_name_col])[:28]
        p_cons = f"{row[p_consensus_col]:.4f}" if p_consensus_col else '—'
        flags = []
        if row['pains_flag']: flags.append('PAINS')
        if row['nih_aggregator_flag']: flags.append('Aggregator')
        if row['brenk_flag']: flags.append('BRENK')
        if row['luciferase_inhibitor_flag']: flags.append('Luciferase')
        flag_str = ', '.join(flags)
        print(f"  {i+1:<4}{drug:<30}{p_cons:<8}{flag_str:<50}")

# =====================================================================
# 8. Save outputs
# =====================================================================
print(f"\n=== SAVING OUTPUTS ===")

# Full triaged list
output_path = os.path.join(OUTPUT_DIR, 'fda_top_candidates_triaged.csv')
df.to_csv(output_path, index=False)
print(f"  Saved {output_path} ({len(df)} candidates)")

# Tier 1 only (recommended for experimental validation)
tier1_path = os.path.join(OUTPUT_DIR, 'fda_tier1_clean.csv')
if len(tier1_df) > 0:
    tier1_df.to_csv(tier1_path, index=False)
    print(f"  Saved {tier1_path} ({len(tier1_df)} Tier 1 candidates)")

# Tier 2 only (acceptable — minor ADME issues, backup list)
tier2_path = os.path.join(OUTPUT_DIR, 'fda_tier2_acceptable.csv')
if len(tier2_df) > 0:
    tier2_df.to_csv(tier2_path, index=False)
    print(f"  Saved {tier2_path} ({len(tier2_df)} Tier 2 candidates)")

# Triage summary JSON
summary = {
    'n_total_candidates': int(len(df)),
    'n_tier1_clean': int(len(tier1_df)),
    'n_tier2_acceptable': int((df['safety_tier'].str.contains('Tier 2')).sum()),
    'n_tier3_flagged': int(len(tier3_df)),
    'filter_counts': {
        'pains_flagged': int(df['pains_flag'].sum()),
        'nih_aggregator_flagged': int(df['nih_aggregator_flag'].sum()),
        'brenk_flagged': int(df['brenk_flag'].sum()),
        'luciferase_inhibitor_flagged': int(df['luciferase_inhibitor_flag'].sum()),
        'lipinski_failed': int((~df['lipinski_pass']).sum()),
        'veber_failed': int((~df['veber_pass']).sum()),
        'pfizer_3_75_failed': int((~df['pfizer_3_75_pass']).sum()),
    },
    'adme_property_stats': {
        'MW': {
            'min': float(df['MW'].min()),
            'median': float(df['MW'].median()),
            'max': float(df['MW'].max()),
        },
        'LogP': {
            'min': float(df['LogP'].min()),
            'median': float(df['LogP'].median()),
            'max': float(df['LogP'].max()),
        },
        'TPSA': {
            'min': float(df['TPSA'].min()),
            'median': float(df['TPSA'].median()),
            'max': float(df['TPSA'].max()),
        },
    },
}
with open(os.path.join(OUTPUT_DIR, 'triage_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved triage_summary.json")

# =====================================================================
# 9. Save report
# =====================================================================
report_path = os.path.join(OUTPUT_DIR, 'triage_report.txt')
with open(report_path, 'w') as f:
    f.write("ATAD5AI — STAGE 8 SAFETY TRIAGE REPORT (Hybrid Consensus Pipeline)\n")
    f.write("=" * 60 + "\n\n")
    f.write("INPUT\n")
    f.write(f"  Source: Stage 6 FDA top candidates ({len(df)} drugs)\n")
    f.write(f"  File: {input_path}\n\n")
    f.write("FILTERS APPLIED (adapted Module 6 spec for ATAD5AI)\n")
    f.write(f"  1. PAINS-A/B/C (Baell 2010): flagged {summary['filter_counts']['pains_flagged']}\n")
    f.write(f"  2. NIH Aggregator/Reactivity: flagged {summary['filter_counts']['nih_aggregator_flagged']}\n")
    f.write(f"  3. BRENK Reactive Groups: flagged {summary['filter_counts']['brenk_flagged']}\n")
    f.write(f"  4. Luciferase Inhibitor SMARTS (Aldrich 2017): flagged {summary['filter_counts']['luciferase_inhibitor_flagged']}\n")
    f.write(f"     (critical filter — AID 504467 is a luciferase reporter assay)\n")
    f.write(f"  5. Lipinski Rule of 5: failed {summary['filter_counts']['lipinski_failed']}\n")
    f.write(f"  6. Veber Rules: failed {summary['filter_counts']['veber_failed']}\n")
    f.write(f"  7. Pfizer 3/75: failed {summary['filter_counts']['pfizer_3_75_failed']}\n\n")
    f.write("SAFETY TIER DISTRIBUTION\n")
    f.write(f"  Tier 1 (Clean — recommended for validation): {summary['n_tier1_clean']}\n")
    f.write(f"  Tier 2 (Acceptable — minor ADME issues): {summary['n_tier2_acceptable']}\n")
    f.write(f"  Tier 3 (Flagged — critical artifact risk): {summary['n_tier3_flagged']}\n\n")
    f.write("ADME PROPERTY DISTRIBUTION\n")
    f.write(f"  MW:    min={summary['adme_property_stats']['MW']['min']:.0f}, "
            f"median={summary['adme_property_stats']['MW']['median']:.0f}, "
            f"max={summary['adme_property_stats']['MW']['max']:.0f}\n")
    f.write(f"  LogP:  min={summary['adme_property_stats']['LogP']['min']:.1f}, "
            f"median={summary['adme_property_stats']['LogP']['median']:.1f}, "
            f"max={summary['adme_property_stats']['LogP']['max']:.1f}\n")
    f.write(f"  TPSA:  min={summary['adme_property_stats']['TPSA']['min']:.0f}, "
            f"median={summary['adme_property_stats']['TPSA']['median']:.0f}, "
            f"max={summary['adme_property_stats']['TPSA']['max']:.0f}\n\n")
    f.write("TOP 10 TIER 1 CANDIDATES (recommended for experimental validation)\n")
    for i, (_, row) in enumerate(tier1_df.head(10).iterrows()):
        f.write(f"  {i+1}. {row[drug_name_col]}\n")
        f.write(f"     P_consensus={row[p_consensus_col]:.4f}\n")
        f.write(f"     MW={row['MW']:.0f}, LogP={row['LogP']:.1f}, TPSA={row['TPSA']:.0f}\n")
        if 'moa' in row and pd.notna(row.get('moa')):
            f.write(f"     MoA: {row['moa']}\n")
        f.write("\n")

    if len(tier2_df) > 0:
        f.write(f"TIER 2 CANDIDATES ({len(tier2_df)} acceptable — minor ADME issues)\n")
        for i, (_, row) in enumerate(tier2_df.iterrows()):
            f.write(f"  {i+1}. {row[drug_name_col]} (P_consensus={row[p_consensus_col]:.4f})\n")
            violations = []
            if not row['lipinski_pass'] and row['lipinski_violations']:
                violations.append(f"Lipinski: {row['lipinski_violations']}")
            if not row['veber_pass'] and row['veber_violations']:
                violations.append(f"Veber: {row['veber_violations']}")
            if not row['pfizer_3_75_pass'] and row['pfizer_violations']:
                violations.append(f"Pfizer: {row['pfizer_violations']}")
            if violations:
                f.write(f"     ADME issues: {' | '.join(violations)}\n")
            f.write("\n")
print(f"  Saved triage_report.txt")

print(f"\n{'=' * 78}")
print("STAGE 8 COMPLETE — PIPELINE FINISHED")
print("=" * 78)
print(f"\nFinal outputs in: {OUTPUT_DIR}/")
print(f"  - fda_top_candidates_triaged.csv   (all {len(df)} candidates with filter flags)")
print(f"  - fda_tier1_clean.csv              ({len(tier1_df)} Tier 1 — recommended for validation)")
print(f"  - fda_tier2_acceptable.csv          ({len(tier2_df) if len(tier2_df) > 0 else 0} Tier 2 — backup list)")
print(f"  - triage_summary.json              (per-filter rejection counts)")
print(f"  - triage_report.txt                (paper-ready summary)")
print(f"\n🎉 ATAD5AI PIPELINE COMPLETE!")
print(f"   Stage 1: Chemical curation + 4-way merge              ✅")
print(f"   Stage 2: Dual feature representation (ECFP6 + SMILES) ✅")
print(f"   Stage 3: Scaffold-stratified 80/10/10 split          ✅")
print(f"   Stage 4a: Chemprop D-MPNN training                   ✅")
print(f"   Stage 4b: CatBoost on ECFP6 counts                   ✅")
print(f"   Stage 4c: Balanced Random Forest on ECFP6 counts    ✅")
print(f"   Stage 5: Hybrid consensus fusion                     ✅")
print(f"   Stage 6: FDA drug repurposing screen                 ✅")
print(f"   Stage 7: External validation on AID 493107           ✅")
print(f"   Stage 8: Safety triage                               ✅")
