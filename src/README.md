# RepAIr — Source Code

This directory contains all 12 pipeline scripts.

## Execution Order

| # | Script | Runs On | Runtime |
|---|--------|---------|---------|
| 1 | `stage1_curation.py` | Local (Mac/Linux) | ~30 min |
| 2 | `stage2_features.py` | Local | ~15 min |
| 3 | `stage3_split.py` | Local | <1 min |
| 4a | `stage4a_chemprop_colab.py` | Google Colab (T4 GPU) | ~40 min |
| 4b | `stage4b_catboost.py` | Local (CPU) | ~9 min |
| 4c | `stage4c_balanced_rf.py` | Local (CPU) | ~20 min |
| 5 | `stage5_consensus.py` | Local | <1 min |
| 6 | `stage6_fda_repurposing_colab.py` | Google Colab (T4 GPU) | ~6 min |
| 7 | `stage7_external_validation_colab.py` | Google Colab (T4 GPU) | ~3 min |
| 8 | `stage8_safety_triage.py` | Local | ~5 min |
| — | `filter_fda_library.py` | Local | <1 min (run before Stage 6) |
| — | `get_fda_library.py` | Local | <2 min (fallback FDA downloader) |

## Notes

- **Colab scripts (4a, 6, 7)**: Paste the entire script into a single Colab cell. The script will prompt you to upload required files. See each script's docstring for details.
- **Local scripts**: Run from the project root directory: `python src/stage1_curation.py`
- **Auto-detection**: Stage 1 auto-detects the `data/` directory location. Stage 8 auto-detects the input file location.
- **Chemprop v2.3.1**: Stage 4a is written for Chemprop v2.3.1 API (not v2.0). The `!pip install` line at the top of each Colab script installs the correct version automatically.
