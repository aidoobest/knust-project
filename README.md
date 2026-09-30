# Fine-Tuning DistilBERT for ICT Misconception Detection

See CHANGELOG.md for the fix-to-finding mapping against Testbed feedback round 1.

## Setup
pip install -r requirements.txt

## Run
python main.py

## Artifacts
Per-seed, per-model predictions are saved to `artifacts/` as JSON files,
enabling regeneration of all downstream statistics without retraining.

## Data
Not included due to participant privacy constraints under KNUST CHRPE
governance. See data/README.md.
