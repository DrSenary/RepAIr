# Data

Download the four PubChem BioAssay CSV files and place them in this directory.

## Required Files

| File | AID | Source URL | Approx. Size |
|------|-----|-----------|-------------|
| `AID_504467_datatable.csv` | 504467 | https://pubchem.ncbi.nlm.nih.gov/bioassay/504467 | ~102 MB |
| `AID_493107_datatable.csv` | 493107 | https://pubchem.ncbi.nlm.nih.gov/bioassay/493107 | ~391 KB |
| `AID_686933_datatable.csv` | 686933 | https://pubchem.ncbi.nlm.nih.gov/bioassay/686933 | ~33 KB |
| `AID_686921_datatable.csv` | 686921 | https://pubchem.ncbi.nlm.nih.gov/bioassay/686921 | ~30 KB |

## How to Download

1. Visit each PubChem BioAssay URL above
2. Click the "Download" button (top-right of the table)
3. Select "CSV" format
4. Save with the exact filename shown above (case-sensitive)
5. Place all files in this `data/` directory

## CLUE Repurposing Hub (for FDA Library)

For Stage 6 (FDA repurposing screen), also download:
1. Go to https://clue.io/repurposing
2. Click "Download Data"
3. Download both "Drug information" and "Sample information" (latest version)
4. Save as `clue_drug_info.txt` and `clue_sample_info.txt` in this directory
5. Run `python src/filter_fda_library.py` to merge and filter to FDA-approved drugs

> **Note:** Raw data files are NOT included in this repository due to size and licensing. The pipeline auto-detects these files when present.
