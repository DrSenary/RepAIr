# RepAIr
<p align="center">
###AI-Driven DNA Repair Inhibitor Discovery via Hybrid Consensus Screening

<p align="center">
  <strong>A hybrid consensus pipeline combining D-MPNN, CatBoost, and Balanced Random Forest
  to identify FDA-approved drug repurposing candidates for ELG1/ATAD5 DNA repair pathway inhibition.</strong>
</p>

---

## Overview

**RepAIr** is a computational drug repurposing framework that identifies FDA-approved drugs predicted to modulate the ELG1/ATAD5 DNA repair pathway. The pipeline integrates three complementary machine-learning architectures into a consensus ensemble whose predictive performance exceeds any individual model. The name captures both the biology (DNA **repair**) and the method (artificial intelligence).

The pipeline screens 2,680 FDA-approved drugs against the PubChem AID 504467 qHTS assay (330,280 compounds, NCATS), validated on the confirmatory AID 493107 dose-response screen, and applies a 7-filter safety triage to produce a prioritized list of repurposing candidates.

### Key Results

| Metric | Value |
|--------|-------|
| Training compounds | 241,861 (7,644 actives, 31:1 imbalance) |
| Test set PR-AUC (consensus) | **0.265** |
| Test set ROC-AUC (consensus) | **0.872** |
| External validation PR-AUC | **0.421** |
| Spearman ρ (potency correlation) | **+0.223** (p = 0.049) |
| FDA drugs screened | 2,680 |
| High-confidence candidates | 49 → 5 Tier 1 + 12 Tier 2 |
| Top candidate | **Olaparib** (FDA-approved PARP inhibitor) |

### Top 5 Repurposing Candidates (Tier 1)

| Rank | Drug | P_consensus | Known MoA |
|------|------|-------------|-----------|
| 1 | Imatinib | 0.763 | Bcr-Abl kinase inhibitor |
| 2 | Pazopanib | 0.696 | Multi-kinase inhibitor |
| 3 | Istradefylline | 0.678 | Adenosine A2A antagonist |
| 4 | Enasidenib | 0.657 | IDH2 inhibitor |
| 5 | **Olaparib** | **0.510** | **PARP inhibitor** |

The identification of **olaparib** — the first-in-class FDA-approved PARP inhibitor — provides strong biological validation, as PARP and ELG1 both operate in the DNA damage response pathway.

---

## Pipeline Architecture

```
Stage 1: Data Curation (4-way merge, chemical standardization)
    ↓
Stage 2: Dual Feature Representation (ECFP6 counts + SMILES)
    ↓
Stage 3: Scaffold-Stratified Split (80/10/10, zero overlap)
    ↓
Stage 4a: Chemprop D-MPNN (graph-based, GPU)
Stage 4b: CatBoost (ECFP6 counts, CPU)
Stage 4c: Balanced RF (ECFP6 counts, CPU)
    ↓
Stage 5: Hybrid Consensus (soft-voting + variance filter)
    ↓
Stage 6: FDA Repurposing Screen (2,680 drugs)
    ↓
Stage 7: External Validation (AID 493107 + counter-screens)
    ↓
Stage 8: Safety Triage (7 filters → Tier 1/2/3 classification)
```

---

## Repository Structure

```
RepAIr/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── src/
│   ├── stage1_curation.py             # Chemical curation + 4-way merge
│   ├── stage2_features.py             # ECFP6 count vectors + SMILES
│   ├── stage3_split.py                 # Scaffold-stratified 80/10/10 split
│   ├── stage4a_chemprop_colab.py      # D-MPNN training (Google Colab GPU)
│   ├── stage4b_catboost.py            # CatBoost training (local CPU)
│   ├── stage4c_balanced_rf.py         # Balanced RF training (local CPU)
│   ├── stage5_consensus.py            # Hybrid consensus fusion
│   ├── stage6_fda_repurposing_colab.py # FDA screen (Google Colab GPU)
│   ├── stage7_external_validation_colab.py # External validation
│   ├── stage8_safety_triage.py        # 7-filter safety triage
│   ├── filter_fda_library.py          # CLUE → FDA-approved CSV filter
│   └── get_fda_library.py             # FDA library downloader (fallback)
├── data/                               # Raw PubChem CSVs (not included — see Data Sources)
├── outputs/                            # Pipeline outputs (not included — generated at runtime)
└── docs/                               # Technical documentation (available separately)
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n repair python=3.11 -y
conda activate repair

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the four PubChem BioAssay CSV files:

| File | AID | URL | Size |
|------|-----|-----|------|
| Primary screen | 504467 | https://pubchem.ncbi.nlm.nih.gov/bioassay/504467 | ~102 MB |
| Validation screen | 493107 | https://pubchem.ncbi.nlm.nih.gov/bioassay/493107 | ~391 KB |
| Luciferase counter | 686933 | https://pubchem.ncbi.nlm.nih.gov/bioassay/686933 | ~33 KB |
| Viability counter | 686921 | https://pubchem.ncbi.nlm.nih.gov/bioassay/686921 | ~30 KB |

Place all CSVs in the `data/` directory.

### 3. Run the Pipeline

```bash
# Stage 1-3: Local (Mac/Linux, ~45 min total)
python src/stage1_curation.py
python src/stage2_features.py
python src/stage3_split.py

# Stage 4a: Google Colab (T4 GPU, ~40 min)
# Upload smiles_train.csv, smiles_val.csv, smiles_test.csv to Colab
# Run stage4a_chemprop_colab.py in a Colab notebook

# Stage 4b-4c: Local (CPU, ~30 min)
python src/stage4b_catboost.py
python src/stage4c_balanced_rf.py

# Stage 5: Local (~1 min)
python src/stage5_consensus.py

# Stage 6: Google Colab (GPU, ~6 min)
# Upload fda_approved.csv + 3 trained models to Colab
# Run stage6_fda_repurposing_colab.py

# Stage 7: Google Colab (GPU, ~3 min)
# Upload validation_holdout.csv + counter-screens + models
# Run stage7_external_validation_colab.py

# Stage 8: Local (~5 min)
python src/stage8_safety_triage.py

# FDA Library Preparation (run before Stage 6)
# Download CLUE Repurposing Hub files, then:
python src/filter_fda_library.py
```

### 4. Review Results

Final outputs are in `outputs/stage8_triage/`:
- `fda_tier1_clean.csv` — 5 Tier 1 candidates (recommended for experimental validation)
- `fda_tier2_acceptable.csv` — 12 Tier 2 candidates (backup list)
- `triage_report.txt` — Summary report

---

## Data Sources

### PubChem BioAssays

| AID | Description | Compounds |
|-----|-------------|-----------|
| 504467 | Primary qHTS: ELG1-dependent DNA repair (HEK293T, luciferase reporter) | 330,280 |
| 493107 | Confirmatory dose-response validation (7-point, 0.003–46 µM) | 1,314 |
| 686933 | Luciferase biochemical counter-screen | 68 |
| 686921 | Cell viability counter-screen | 68 |

### FDA Drug Library

Sourced from the [CLUE Repurposing Hub](https://clue.io/repurposing) (Broad Institute). Download the "Drug information" and "Sample information" files, then run `filter_fda_library.py` to merge and filter to FDA-approved drugs with valid SMILES (2,680 compounds).

---

## Methods

### Three-Model Hybrid Consensus

| Model | Architecture | Features | Training |
|-------|-------------|----------|----------|
| Chemprop D-MPNN | 3 message-passing layers, hidden=300, 2-layer FFN | Molecular graphs (SMILES) | Colab T4 GPU, 40 min |
| CatBoost | Symmetric trees, depth=6, 1000 iterations | ECFP6 count vectors (4096-bit) | CPU, 9 min |
| Balanced RF | 600 trees, max_depth=18, balanced subsampling | ECFP6 count vectors (4096-bit) | CPU, 20 min |

### Consensus Fusion

```
P_consensus = (P_chemprop + P_catboost + P_rf) / 3
Confidence filter: top 5% by P_consensus AND model_variance < 0.04
```

### Safety Triage (7 Filters)

1. PAINS-A/B/C (Baell 2010)
2. NIH Aggregator/Reactivity
3. BRENK Reactive Groups
4. Luciferase Inhibitor SMARTS (Aldrich 2017) — critical for this assay type
5. Lipinski Rule of 5 (mammalian cell permeability)
6. Veber Rules (oral bioavailability)
7. Pfizer 3/75 Rule (low attrition risk)

---

## Key Findings

### Biological Validation

The pipeline independently identified **olaparib** (rank 5, P_consensus = 0.510) — an FDA-approved PARP inhibitor — as a top ELG1/ATAD5 modulator candidate. This provides strong biological validation because:

- **PARP** detects single-strand breaks in DNA
- **ELG1** unloads PCNA during DNA replication and repair
- Both operate in the DNA damage response pathway
- PARP inhibitors are clinically validated for BRCA-deficient cancers (synthetic lethality)
- The model was trained only on luciferase-ELG1 reporter data with no explicit DNA repair pathway information

### Limitations

- The model fails to reject known luciferase reporter artifacts (only 1.8% counter-screen coverage in training data)
- Stage 8 chemical-rule safety triage addresses this by filtering 65% of top candidates as PAINS/BRENK/luciferase artifacts
- The remaining 17 candidates (5 Tier 1 + 12 Tier 2) are biologically plausible DNA repair modulators

---

## Requirements

```
rdkit>=2024.3
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
catboost>=1.2
imbalanced-learn>=0.12
chemprop>=2.3
lightning>=2.2
torch>=2.1
matplotlib>=3.7
```

---

## Hardware Requirements

| Stage | Hardware | Runtime |
|-------|----------|---------|
| Stages 1-3, 4b, 4c, 5, 8 | Mac/Linux CPU (8 cores, 16 GB RAM) | ~1.5 hours total |
| Stages 4a, 6, 7 | Google Colab T4 GPU (free tier) | ~50 min total |
| **Total wall-clock** (parallel) | | **~2 hours** |

---

## Citation

If you use RepAIr in your research, please cite:

```bibtex
@software{senary_repair_2026,
  author = {Ahmed Senary},
  title = {RepAIr: AI-Driven DNA Repair Inhibitor Discovery via Hybrid Consensus Screening},
  year = {2026},
  url = {https://github.com/<your-username>/RepAIr}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **NCATS** (National Center for Advancing Translational Sciences) for depositing the qHTS data on PubChem
- **Broad Institute** for the CLUE Repurposing Hub FDA drug library
- **Chemprop** team (MIT) for the D-MPNN implementation
- **CatBoost** team (Yandex) for the gradient boosting library
- **RDKit** community for the cheminformatics toolkit

---

## Contact

**Ahmed Senary**  
British University in Egypt  
Email: ahmed.senary@gmail.com  
GitHub: [@<your-username>]

---

<p align="center">
  <em>RepAIr — where DNA repair meets artificial intelligence.</em>
</p>
