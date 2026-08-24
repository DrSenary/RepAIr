# Outputs

This directory contains pipeline outputs generated at runtime. Each stage creates a subdirectory:

```
outputs/
├── stage1_clean_data/        # Curated training pool + validation holdout
├── stage2_features/          # ECFP6 sparse matrix + SMILES CSVs
├── stage3_split/             # Train/val/test splits (80/10/10)
├── stage4_models/            # Trained models + test predictions
├── stage5_consensus/         # Consensus predictions + metrics
├── stage6_fda_predictions/   # FDA screening results
├── stage7_validation/        # External validation results
└── stage8_triage/            # Safety triage results (final output)
```

All output files are gitignored. The pipeline recreates them on each run.

## Key Output Files

| File | Stage | Description |
|------|-------|-------------|
| `train_pool.csv` | 1 | 241,861 compounds with binary labels + scaffolds |
| `ecfp6_counts.npz` | 2 | Sparse ECFP6 count matrix (241,861 × 4,096) |
| `chemprop_best_model.ckpt` | 4a | Trained D-MPNN model (~50 MB) |
| `catboost_model.cbm` | 4b | Trained CatBoost model (~5 MB) |
| `balanced_rf_model.joblib` | 4c | Trained Balanced RF model (~50 MB) |
| `test_predictions_consensus.csv` | 5 | 23,036 test predictions with consensus scores |
| `fda_top_candidates.csv` | 6 | 49 high-confidence FDA candidates |
| `validation_metrics.json` | 7 | External validation metrics (Spearman ρ, PR-AUC) |
| `fda_tier1_clean.csv` | 8 | **5 Tier 1 candidates (final deliverable)** |
| `fda_tier2_acceptable.csv` | 8 | 12 Tier 2 candidates (backup list) |
