# NLP Coursework (PCL Detection) - Complete Starter Project

This repository provides a full pipeline for SemEval 2022 Task 4 Subtask 1 (PCL binary classification):

- Data download and parsing from official sources
- Model training and dev evaluation
- Submission file generation (`dev.txt`, `test.txt`)
- EDA figures/tables
- Error analysis outputs
- `BestModel/` folder with required deliverables

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run Everything

```bash
bash scripts/run_all.sh
```

If all succeeds, your key submission artifacts are:

- `BestModel/dev.txt`
- `BestModel/test.txt`
- `BestModel/best_model.joblib`
- `BestModel/MODEL_INFO.txt`

You also get:

- `outputs/dev_predictions_detailed.csv`
- `reports/EDA_NOTES.md`
- `reports/LOCAL_EVALUATION.md`
- figures under `figures/`

## 3) Important Coursework Notes

- `dev.txt` and `test.txt` must contain one prediction (`0` or `1`) per line.
- Keep the same row order as official dev/test files.
- Add a working public GitHub/GitLab link on the report front page.
- Keep a `BestModel/` folder in the repository.

## 4) Project Structure

```text
.
├── BestModel/
├── data/
├── figures/
├── models/
├── outputs/
├── reports/
├── scripts/
│   ├── run_all.sh
│   ├── train_and_predict.py
│   ├── run_eda.py
│   └── error_analysis.py
├── src/
│   └── dpm_data.py
├── requirements.txt
└── README.md
```

## 5) Commands (Manual)

```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp
python -m src.dpm_data --data-root data
python scripts/train_and_predict.py --data-root data --outputs-dir outputs --models-dir models --bestmodel-dir BestModel
python scripts/run_eda.py --data-root data --figures-dir figures --reports-dir reports
python scripts/error_analysis.py --dev-details outputs/dev_predictions_detailed.csv --reports-dir reports --figures-dir figures
```

## 6) Push to GitHub and Create Link

### Option A: already have an empty GitHub repo

```bash
git init
git add .
git commit -m "Initial coursework pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

### Option B: use GitHub CLI (`gh`)

```bash
git init
git add .
git commit -m "Initial coursework pipeline"
gh repo create pcl-coursework-2026 --public --source=. --remote=origin --push
```

Then use the resulting repo URL on the report front page.

## 7) Troubleshooting

- If data download fails, manually place these files in `data/`:
  - `dontpatronizeme_pcl.tsv`
  - `train_semeval_parids-labels.csv`
  - `dev_semeval_parids-labels.csv`
  - `task4_test.tsv`
- Re-run the pipeline after placing files.
