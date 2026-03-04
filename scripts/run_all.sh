#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

python -m src.dpm_data --data-root data
python scripts/train_and_predict.py --data-root data --outputs-dir outputs --models-dir models --bestmodel-dir BestModel
python scripts/run_eda.py --data-root data --figures-dir figures --reports-dir reports
python scripts/error_analysis.py --dev-details outputs/dev_predictions_detailed.csv --reports-dir reports --figures-dir figures

echo "Done. Key files:"
echo "- outputs/dev.txt"
echo "- outputs/test.txt"
echo "- BestModel/dev.txt"
echo "- BestModel/test.txt"
echo "- reports/EDA_NOTES.md"
echo "- reports/LOCAL_EVALUATION.md"
