# RepAIr — Source Code

This directory contains all 12 core pipeline scripts alongside the standalone verification and triage audit suite.

## Execution Order

| # | Script | Runs On | Runtime | Description |
|---|--------|---------|---------|-------------|
| 1 | `stage1_curation.py` | Local (Mac/Linux) | ~30 min | 5-step RDKit standardization, desalting, tautomer canonicalization, deduplication |
| 2 | `stage2_features.py` | Local | ~15 min | Dual featurization: SMILES graphs for D-MPNN and 4,096-bit ECFP6 count CSR matrices |
| 3 | `stage3_split.py` | Local | <1 min | Bemis–Murcko scaffold-stratified 80/10/10 split with zero scaffold overlap |
| 4a | `stage4a_chemprop_colab.py` | Google Colab (T4 GPU) | ~40 min | Directed Message Passing Neural Network (Chemprop v2.3.1) with imbalance loss weighting |
| 4b | `stage4b_catboost.py` | Local (CPU) | ~9 min | Symmetric gradient-boosted decision trees with auto-balanced class weighting |
| 4c | `stage4c_balanced_rf.py` | Local (CPU) | ~20 min | Balanced Random Forest (600 trees) with per-tree random majority undersampling |
| 5 | `stage5_consensus.py` | Local | <1 min | Soft-voting probability fusion and model-disagreement variance calculation |
| — | `filter_fda_library.py` | Local | <1 min | Preprocessing of CLUE Repurposing Hub: 5-step desalting & parent drug deduplication |
| 6 | `stage6_fda_repurposing_colab.py` | Google Colab (T4 GPU) | ~6 min | Inference on 2,680 FDA drugs; top-5% thresholding & variance filtering (Var < 0.04) |
| 7 | `stage7_external_validation_colab.py` | Google Colab (T4 GPU) | ~3 min | External validation on PubChem AID 493107 & counter-screen artifact evaluation |
| 8 | `stage8_safety_triage.py` | Local | ~5 min | 7-filter safety triage (PAINS, NIH, BRENK, Luciferase SMARTS, Lipinski, Veber, Pfizer) |
| **Audit** | `verify_and_fix_fda_triage.py` | Local / Standalone | <10 sec | Standalone audit & verification script for candidate deduplication and figure generation |

## Verification, Reproducibility, and Audit Script (`verify_and_fix_fda_triage.py`)

### Purpose and Significance
The `verify_and_fix_fda_triage.py` script was authored as an independent, deterministic audit gate to reconcile raw screening outputs with the finalized peer-reviewed manuscript:
1. **Parent-Structure Salt Consolidation:** Raw pharmaceutical annotations in public repositories (such as the Broad Institute CLUE Repurposing Hub) frequently index different salt formulations of the exact same active pharmaceutical ingredient as separate records (e.g., free-base `doxazosin` vs. `doxazosin mesylate`, or `omeprazole-magnesium`). This script verifies that all salt forms collapse into unique neutral parent chemical entities before candidate ranking.
2. **Standardized Flag Polarity:** It unifies the boolean reporting polarity across structural alert catalogs (`PAINS`, `NIH`, `BRENK`, `Luciferase SMARTS`) and physicochemical rules (`Lipinski`, `Veber`, `Pfizer 3/75`), ensuring that `True` consistently denotes a flagged rule violation and preventing inverted pass/fail counts in summary tables.
3. **Automated Figure Regeneration:** It programmatically regenerates publication-quality, 300 DPI figures (`fig5_top20_fda_fixed.png` and `fig7_safety_triage_fixed.png`) directly from the deduplicated data tables.

### Note for Reviewers on End-to-End Retraining
If you choose to re-execute the entire pipeline from scratch (Stages 1 through 8):
* **Stochastic Hardware Variations:** Chemprop D-MPNN training uses CUDA non-deterministic operations across GPU kernel calls. While all random seeds are pinned (`seed = 42`), training deep neural networks on different hardware architectures (e.g., NVIDIA T4 vs. A100 vs. V100) or differing PyTorch/CUDA driver versions may introduce minor decimal fluctuations in raw probability outputs.
* **Ranking Invariance:** These minor numerical shifts do not affect the qualitative outcome of the study: the top Tier 1 candidates (**imatinib, pazopanib, istradefylline, enasidenib, olaparib**) consistently emerge at the top of the actionable pool across random seeds and hyperparameter perturbations.
* **Instant Verification:** To audit the manuscript's reported tables and figures without spending ~90 minutes retraining on GPU, run `python src/verify_and_fix_fda_triage.py` to immediately verify the triage rules and generate the exact candidate breakdowns reported in the paper.

## Notes

- **Colab scripts (4a, 6, 7)**: Paste the entire script into a single Colab cell. The script will prompt you to upload required files. See each script's docstring for details.
- **Local scripts**: Run from the project root directory: `python src/stage1_curation.py`
- **Auto-detection**: Stage 1 auto-detects the `data/` directory location. Stage 8 auto-detects the input file location.
- **Chemprop v2.3.1**: Stage 4a is written for the Chemprop v2.3.1 API. The `!pip install` line at the top of each Colab script installs the correct version automatically.
