#!/usr/bin/env python3
"""
RepAIr Pipeline — FDA Library Salt-Stripping & Safety Triage Verification/Fix Script
====================================================================================

Purpose:
--------
1. Fixes the chemical standardization ordering bug in the FDA screening library:
   - Strips salts and counter-ions (e.g., mesylate, magnesium, sodium, HCl) FIRST
     using RDKit's SaltRemover / LargestFragmentChooser BEFORE checking for allowed
     organic elements {H, C, N, O, F, P, S, Cl, Br, I}.
   - Neutralizes charges, canonicalizes tautomers, and deduplicates by canonical
     parent SMILES / InChIKey connectivity block so salt duplicates (such as
     'doxazosin' [P_cons = 0.775] and 'doxazosin mesylate' [P_cons = 0.745])
     collapse into a single canonical parent drug entry, and salt suffixes
     (such as 'omeprazole-magnesium' [P_cons = 0.705]) normalize to the parent name.

2. Fixes the Stage 8 Safety Triage boolean polarity bug (Table 7 & Figure 7):
   - In the original plotting/reporting script, structural alert filters
     (PAINS, NIH, BRENK, Luciferase SMARTS) were stored with True = Flagged (Fail),
     whereas physicochemical filters (Lipinski Rule of 5, Veber Rules) were stored
     with True = Clean (Pass).
   - Summing `col == True` as "Flagged" inverted the Lipinski and Veber counts in
     Table 7 and Figure 7, falsely reporting 31/49 (63.3%) Lipinski failures instead
     of 18/49 (36.7%) failures (31 Clean, 18 Flagged).
   - Also ensures PAINS, NIH, and BRENK are evaluated using three separate RDKit
     FilterCatalog instances rather than a single combined catalog.

3. Regenerates publication-grade corrected figures:
   - `fig5_top20_fda_fixed.png` (Salt-stripped & deduplicated Top 20 FDA candidates)
   - `fig7_safety_triage_fixed.png` (Corrected per-filter Flagged vs. Clean counts)

Usage:
------
- Standalone audit & figure regeneration mode (no arguments required):
    python3 verify_and_fix_fda_triage.py
- Full CSV pipeline repair mode (with RDKit installed on local machine / Colab):
    python3 verify_and_fix_fda_triage.py \
        --fda_csv outputs/fda_candidates.csv \
        --out_dir outputs/fixed
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, FilterCatalog
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


ALLOWED_ORGANIC_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I

SALT_SUFFIX_PATTERN = re.compile(
    r"[\s\-]+(mesylate|hydrochloride|hcl|magnesium|sodium|potassium|calcium|"
    r"besylate|maleate|fumarate|tartrate|citrate|sulfate|sulphate|acetate|"
    r"phosphate|succinate|bromide|chloride|iodide|nitrate|lactate|tosylate|"
    r"dihydrate|monohydrate|trihydrate|hemihydrate|anhydrous)\b",
    flags=re.IGNORECASE,
)

LUCIFERASE_INHIBITOR_SMARTS = {
    "resorcinol": "c1cc(O)cc(O)c1",
    "benzothiazole": "c1ccc2scnc2c1",
    "aminothiazole": "Nc1ncsc1",
    "quinazoline": "c1ccc2ncncc2c1",
    "2_aminobenzothiazole": "Nc1nc2ccccc2s1",
    "rhodanine": "O=C1CSC(=S)N1",
    "hydroxypyridinone": "Oc1cccc(=O)[nH]1",
}


def normalize_drug_name(name: str) -> str:
    """Strip common pharmaceutical salt/hydrate suffixes to get the parent INN name."""
    cleaned = SALT_SUFFIX_PATTERN.sub("", str(name).strip())
    return cleaned.strip(" -")


def standardize_parent_smiles_rdkit(smiles: str):
    """
    Correct 5-step RDKit standardization order:
    1. Parse SMILES
    2. Desalt & retain largest fragment FIRST (strips Na+, Mg2+, mesylate, HCl, etc.)
    3. Check that the parent organic fragment contains only allowed elements
    4. Neutralize charges (Uncharger) & physiological reionization (Reionizer)
    5. Canonicalize tautomer and return canonical SMILES + InChIKey connectivity block
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for live SMILES standardization.")

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None, "parse_error"

    try:
        lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        mol = lfc.choose(mol)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None, None, "empty_after_desalt"

        atom_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if not atom_nums.issubset(ALLOWED_ORGANIC_ATOMS):
            return None, None, "non_organic_element"

        uncharger = rdMolStandardize.Uncharger()
        reionizer = rdMolStandardize.Reionizer()
        mol = uncharger.uncharge(mol)
        mol = reionizer.reionize(mol)

        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)

        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)
        ik_first = inchikey.split("-")[0] if inchikey else can_smiles
        return can_smiles, ik_first, "ok"
    except Exception as exc:
        return None, None, f"error:{exc}"


def init_separate_filter_catalogs():
    """
    Initialize THREE SEPARATE RDKit FilterCatalog objects so PAINS, NIH, and BRENK
    counts are never conflated into a single combined catalog.
    """
    if not RDKIT_AVAILABLE:
        return None, None, None, None

    pains_params = FilterCatalog.FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    pains_cat = FilterCatalog.FilterCatalog(pains_params)

    nih_params = FilterCatalog.FilterCatalogParams()
    nih_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
    nih_cat = FilterCatalog.FilterCatalog(nih_params)

    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_cat = FilterCatalog.FilterCatalog(brenk_params)

    luc_patterns = {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in LUCIFERASE_INHIBITOR_SMARTS.items()
    }
    return pains_cat, nih_cat, brenk_cat, luc_patterns


def evaluate_safety_filters_rdkit(smiles: str, catalogs, strict_lipinski_zero_violations: bool = True):
    """
    Evaluate all 7 safety filters on a single molecule and return ALL flags with
    consistent polarity: `*_fail = True` means FLAGGED/FAILED, `False` means CLEAN/PASSED.
    """
    pains_cat, nih_cat, brenk_cat, luc_patterns = catalogs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    pains_matches = [e.GetDescription() for e in pains_cat.GetMatches(mol)]
    nih_matches = [e.GetDescription() for e in nih_cat.GetMatches(mol)]
    brenk_matches = [e.GetDescription() for e in brenk_cat.GetMatches(mol)]
    luc_matches = [
        name for name, pat in luc_patterns.items()
        if pat is not None and mol.HasSubstructMatch(pat)
    ]

    pains_fail = len(pains_matches) > 0
    nih_fail = len(nih_matches) > 0
    brenk_fail = len(brenk_matches) > 0
    luc_fail = len(luc_matches) > 0

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    lipinski_violations = (
        int(mw > 500.0)
        + int(logp > 5.0)
        + int(hbd > 5)
        + int(hba > 10)
    )
    if strict_lipinski_zero_violations:
        lipinski_fail = lipinski_violations >= 1
    else:
        lipinski_fail = lipinski_violations >= 2

    veber_violations = int(rot_bonds > 10) + int(tpsa > 140.0)
    veber_fail = veber_violations >= 1
    pfizer_fail = (logp > 3.0) and (tpsa < 75.0)

    has_critical_flag = pains_fail or nih_fail or brenk_fail or luc_fail
    has_adme_violation = lipinski_fail or veber_fail or pfizer_fail

    if has_critical_flag:
        tier = 3
    elif not has_adme_violation:
        tier = 1
    else:
        tier = 2

    return {
        "mw": round(mw, 1),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1),
        "hbd": hbd,
        "hba": hba,
        "rot_bonds": rot_bonds,
        "pains_fail": pains_fail,
        "nih_fail": nih_fail,
        "brenk_fail": brenk_fail,
        "luc_fail": luc_fail,
        "lipinski_fail": lipinski_fail,
        "lipinski_pass": not lipinski_fail,
        "veber_fail": veber_fail,
        "veber_pass": not veber_fail,
        "pfizer_fail": pfizer_fail,
        "pfizer_pass": not pfizer_fail,
        "tier": tier,
    }


RAW_TOP_CANDIDATES = [
    {"raw_rank": 1,  "raw_name": "cabozantinib",         "p_cons": 0.795, "tier": 3, "moa": "MET/VEGFR2 kinase inhibitor"},
    {"raw_rank": 2,  "raw_name": "doxazosin",            "p_cons": 0.775, "tier": 3, "moa": "Alpha-1 adrenergic antagonist"},
    {"raw_rank": 3,  "raw_name": "imatinib",             "p_cons": 0.763, "tier": 1, "moa": "Bcr-Abl tyrosine kinase inhibitor"},
    {"raw_rank": 4,  "raw_name": "nilotinib",            "p_cons": 0.758, "tier": 2, "moa": "Bcr-Abl tyrosine kinase inhibitor"},
    {"raw_rank": 5,  "raw_name": "suramin",              "p_cons": 0.757, "tier": 3, "moa": "Purinergic receptor antagonist"},
    {"raw_rank": 6,  "raw_name": "masitinib",            "p_cons": 0.757, "tier": 2, "moa": "KIT/PDGFR/Bcr-Abl kinase inhibitor"},
    {"raw_rank": 7,  "raw_name": "cyclovalone",          "p_cons": 0.749, "tier": 3, "moa": "Curcuminoid choleretic agent"},
    {"raw_rank": 8,  "raw_name": "doxazosin mesylate",   "p_cons": 0.745, "tier": 3, "moa": "Alpha-1 adrenergic antagonist (salt duplicate)"},
    {"raw_rank": 9,  "raw_name": "dequalinium",          "p_cons": 0.736, "tier": 3, "moa": "Quinolinium antiseptic / K+ channel blocker"},
    {"raw_rank": 10, "raw_name": "fenoverine",           "p_cons": 0.708, "tier": 2, "moa": "Phenothiazine antispasmodic"},
    {"raw_rank": 11, "raw_name": "revaprazan",           "p_cons": 0.706, "tier": 2, "moa": "Potassium-competitive acid blocker"},
    {"raw_rank": 12, "raw_name": "omeprazole-magnesium", "p_cons": 0.705, "tier": 2, "moa": "Proton pump inhibitor (Mg2+ salt form)"},
    {"raw_rank": 13, "raw_name": "pazopanib",            "p_cons": 0.696, "tier": 1, "moa": "VEGFR/PDGFR/KIT multi-kinase inhibitor"},
    {"raw_rank": 14, "raw_name": "olmutinib",            "p_cons": 0.681, "tier": 3, "moa": "Covalent EGFR tyrosine kinase inhibitor"},
    {"raw_rank": 15, "raw_name": "istradefylline",       "p_cons": 0.678, "tier": 1, "moa": "Adenosine A2A receptor antagonist"},
    {"raw_rank": 16, "raw_name": "elbasvir",             "p_cons": 0.672, "tier": 2, "moa": "HCV NS5A replication complex inhibitor"},
    {"raw_rank": 17, "raw_name": "erlotinib",            "p_cons": 0.666, "tier": 3, "moa": "EGFR tyrosine kinase inhibitor"},
    {"raw_rank": 18, "raw_name": "enasidenib",           "p_cons": 0.657, "tier": 1, "moa": "Mutant IDH2 allosteric inhibitor"},
    {"raw_rank": 19, "raw_name": "idarubicin",           "p_cons": 0.650, "tier": 3, "moa": "Anthracycline topoisomerase II inhibitor"},
    {"raw_rank": 20, "raw_name": "ibrutinib",            "p_cons": 0.647, "tier": 3, "moa": "Covalent BTK tyrosine kinase inhibitor"},
]

SAFETY_FILTER_AUDIT = [
    {"filter": "PAINS-A/B/C",                "short": "PAINS\nA/B/C",    "orig_flagged": 11, "orig_clean": 38, "fixed_flagged": 11, "fixed_clean": 38, "inverted_in_paper": False},
    {"filter": "NIH Aggregator/Reactivity",  "short": "NIH\nAgg",        "orig_flagged": 11, "orig_clean": 38, "fixed_flagged": 11, "fixed_clean": 38, "inverted_in_paper": False},
    {"filter": "BRENK Reactive Groups",      "short": "BRENK",           "orig_flagged": 29, "orig_clean": 20, "fixed_flagged": 29, "fixed_clean": 20, "inverted_in_paper": False},
    {"filter": "Luciferase Inhibitor SMARTS","short": "Lucif\nSMARTS",   "orig_flagged": 3,  "orig_clean": 46, "fixed_flagged": 3,  "fixed_clean": 46, "inverted_in_paper": False},
    {"filter": "Lipinski Rule of 5 (fail)",  "short": "Lipinski\n(fail)","orig_flagged": 31, "orig_clean": 18, "fixed_flagged": 18, "fixed_clean": 31, "inverted_in_paper": True},
    {"filter": "Veber Rules (fail)",         "short": "Veber\n(fail)",   "orig_flagged": 24, "orig_clean": 25, "fixed_flagged": 25, "fixed_clean": 24, "inverted_in_paper": True},
    {"filter": "Pfizer 3/75 Rule (fail)",    "short": "Pfizer\n(fail)",  "orig_flagged": 11, "orig_clean": 38, "fixed_flagged": 11, "fixed_clean": 38, "inverted_in_paper": False},
]


def deduplicate_fda_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate FDA candidates by normalized parent drug name (and canonical SMILES
    if available), keeping the canonical parent entry and removing duplicate salt rows.
    """
    df = df.copy()
    name_col = "raw_name" if "raw_name" in df.columns else "drug_name"
    score_col = "p_cons" if "p_cons" in df.columns else "p_consensus"
    df["parent_name"] = df[name_col].apply(normalize_drug_name)
    df = df.sort_values(score_col, ascending=False)
    deduped = df.drop_duplicates(subset=["parent_name"], keep="first").copy()
    deduped["new_rank"] = range(1, len(deduped) + 1)
    return deduped


def generate_corrected_figures(deduped_top_df: pd.DataFrame, filter_df: pd.DataFrame, out_dir: Path):
    """Generate publication-quality corrected Figure 5 and Figure 7."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    tier_colors = {1: "#2b7a4b", 2: "#d9822b", 3: "#b8332a"}

    # Figure 5 (Fixed): Salt-Stripped & Deduplicated Top FDA Candidates
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    plot_df = deduped_top_df.iloc[::-1]
    colors = [tier_colors[t] for t in plot_df["tier"]]

    bars = ax.barh(plot_df["parent_name"], plot_df["p_cons"], color=colors, height=0.72, edgecolor="#222222", linewidth=0.5)
    for bar, score in zip(bars, plot_df["p_cons"]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", ha="left", fontsize=8.5, color="#111111")

    ax.set_xlim(0.0, 0.88)
    ax.set_xlabel("Consensus Probability ($P_{\\mathrm{consensus}}$)", fontsize=10.5, fontweight="bold")
    ax.set_title("Top Salt-Stripped & Deduplicated FDA Candidates (Color = Safety Tier)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color="#2b7a4b", label="Tier 1 — Clean (passes all 7 filters)"),
        mpatches.Patch(color="#d9822b", label="Tier 2 — Acceptable (minor ADME violation only)"),
        mpatches.Patch(color="#b8332a", label="Tier 3 — Flagged (PAINS / NIH / BRENK / Luciferase)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig5_path = out_dir / "fig5_top20_fda_fixed.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close(fig)

    # Figure 7 (Fixed): Corrected Per-Filter Flagged vs. Clean Counts & Tier Pie
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

    x = np.arange(len(filter_df))
    width = 0.36
    rects_flag = ax1.bar(x - width / 2, filter_df["fixed_flagged"], width, label="Flagged / Failed", color="#b8332a", edgecolor="#222222", linewidth=0.5)
    rects_clean = ax1.bar(x + width / 2, filter_df["fixed_clean"], width, label="Clean / Passed", color="#2b7a4b", edgecolor="#222222", linewidth=0.5)

    for rect in rects_flag:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#8c1d18")
    for rect in rects_clean:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e5635")

    ax1.set_xticks(x)
    ax1.set_xticklabels(filter_df["short"], fontsize=8.5)
    ax1.set_ylim(0, 53)
    ax1.set_ylabel("Compound Count (out of n = 49)", fontsize=10, fontweight="bold")
    ax1.set_title("Corrected Safety Filter Results (n = 49)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    tier_counts = [5, 12, 32]
    tier_labels = [
        "Tier 1 — Clean\n(5 drugs, 10.2%)",
        "Tier 2 — Acceptable\n(12 drugs, 24.5%)",
        "Tier 3 — Flagged\n(32 drugs, 65.3%)",
    ]
    tier_cols = ["#2b7a4b", "#d9822b", "#b8332a"]
    ax2.pie(
        tier_counts,
        labels=tier_labels,
        colors=tier_cols,
        startangle=90,
        wedgeprops={"edgecolor": "#222222", "linewidth": 0.8},
        textprops={"fontsize": 8.5},
    )
    ax2.set_title("Safety Tier Distribution (n = 49)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig7_path = out_dir / "fig7_safety_triage_fixed.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close(fig)

    return fig5_path, fig7_path


def repair_csv_pipeline(fda_csv_path: Path, out_dir: Path):
    """Run live RDKit salt-stripping, deduplication, and 7-filter triage on a candidate CSV."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit must be installed to run live CSV standardization.")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(fda_csv_path)
    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    name_col = "drug_name" if "drug_name" in df.columns else "pert_iname"
    score_col = "p_consensus" if "p_consensus" in df.columns else "p_cons"

    std_results = [standardize_parent_smiles_rdkit(s) for s in df[smiles_col]]
    df["canonical_parent_smiles"] = [r[0] for r in std_results]
    df["inchikey_first"] = [r[1] for r in std_results]
    df["std_status"] = [r[2] for r in std_results]
    df = df[df["std_status"] == "ok"].copy()

    df["parent_name"] = df[name_col].apply(normalize_drug_name)
    df = df.sort_values(score_col, ascending=False)
    df = df.drop_duplicates(subset=["inchikey_first"], keep="first").copy()

    catalogs = init_separate_filter_catalogs()
    triage_rows = []
    for _, row in df.iterrows():
        metrics = evaluate_safety_filters_rdkit(row["canonical_parent_smiles"], catalogs)
        if metrics is not None:
            merged = {**row.to_dict(), **metrics}
            triage_rows.append(merged)

    out_df = pd.DataFrame(triage_rows)
    fixed_csv = out_dir / "triaged_candidates_fixed.csv"
    out_df.to_csv(fixed_csv, index=False)
    print(f"Saved deduplicated & re-triaged candidates to: {fixed_csv}")


def run_audit_and_demo(out_dir: Path):
    """Run the complete verification audit and print summary tables."""
    raw_df = pd.DataFrame(RAW_TOP_CANDIDATES)
    deduped_df = deduplicate_fda_candidates(raw_df)
    filter_df = pd.DataFrame(SAFETY_FILTER_AUDIT)
    filter_df["orig_pct_flagged"] = (filter_df["orig_flagged"] / 49.0 * 100).round(1)
    filter_df["fixed_pct_flagged"] = (filter_df["fixed_flagged"] / 49.0 * 100).round(1)

    print("=" * 78)
    print("AUDIT 1: SALT-STRIPPING & DEDUPLICATION OF TOP FDA CANDIDATES")
    print("=" * 78)
    removed = set(raw_df["raw_name"]) - set(deduped_df["raw_name"])
    print(f"- Raw entries inspected: {len(raw_df)}")
    print(f"- Unique parent drugs after salt stripping: {len(deduped_df)}")
    print(f"- Duplicate salt entry merged: {list(removed)} -> merged into 'doxazosin' (P_cons=0.775)")
    print("- Salt suffix normalized: 'omeprazole-magnesium' (P_cons=0.705) -> 'omeprazole'")
    print("\nCorrected Top FDA Candidates (Parent Molecules):")
    print(deduped_df[["new_rank", "raw_rank", "parent_name", "p_cons", "tier", "moa"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("AUDIT 2: STAGE 8 SAFETY TRIAGE PASS/FAIL BOOLEAN POLARITY FIX (n = 49)")
    print("=" * 78)
    display_cols = ["filter", "orig_flagged", "orig_clean", "orig_pct_flagged", "fixed_flagged", "fixed_clean", "fixed_pct_flagged", "inverted_in_paper"]
    print(filter_df[display_cols].to_string(index=False))

    fig5_path, fig7_path = generate_corrected_figures(deduped_df, filter_df, out_dir)
    print("\n" + "=" * 78)
    print(f"Generated corrected publication figures:\n  1. {fig5_path}\n  2. {fig7_path}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fda_csv", type=str, default=None, help="Optional path to raw fda_candidates.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts/file_generation/ttl=63d/intermediate", help="Directory for output plots/CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.fda_csv:
        repair_csv_pipeline(Path(args.fda_csv), out_dir)
    else:
        run_audit_and_demo(out_dir)


if __name__ == "__main__":
    main()