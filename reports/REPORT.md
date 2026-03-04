# NLP Coursework Report (SemEval 2022 Task 4 Subtask 1)

## Front Page
- Name: `Molan Yang`
- Student ID: `my1322`
- Leaderboard Name (<=20 chars): `my1322_pcl2026`
- GitHub/GitLab Repository Link (clickable): [PeterYang-niubidehen/nlp_cw](https://github.com/PeterYang-niubidehen/nlp_cw)

---

## Exercise 1: Critical Paper Review
**Paper reviewed:** Pérez-Almendros et al. (2020), *Don't Patronize Me! An Annotated Dataset with Explanations for Detecting Patronizing and Condescending Language towards Vulnerable Communities*.

### Q1. Primary contributions of the work
1. The paper introduces a dedicated PCL dataset for English news paragraphs, targeting a difficult and socially important phenomenon that is more subtle than standard toxicity/hate tasks.
2. It defines both binary labels (PCL vs No-PCL) and fine-grained PCL categories, giving the community a shared benchmark for both coarse and nuanced analysis.
3. It documents annotation guidelines and releases data/resources that later enabled shared-task style evaluation (including the SemEval setup).

### Q2. Technical strengths that justify publication
1. **Task novelty and relevance:** The paper operationalizes a linguistically subtle phenomenon (patronising/condescending language) that is under-studied compared with sentiment or abuse detection.
2. **Annotation design:** The labeling setup captures both binary and category-level signals, which is valuable for analysis and future multi-task modeling.
3. **Research utility:** The resource is practically useful: it supports reproducible experiments, benchmark comparisons, and downstream shared tasks.

### Q3. Weaknesses / insufficient evidence
1. **Subjectivity challenge:** PCL boundaries are inherently ambiguous, and some examples are context-dependent; this can introduce annotation noise and uncertain gold labels.
2. **Sampling/domain bias risk:** Data is drawn from specific sources/topic framing, which may limit generalization to other domains (social media, forums, conversational text).
3. **Limited causal evidence on model behavior:** The work provides benchmark framing, but deeper evidence (e.g., stronger ablations or robust cross-domain stress tests) is limited.

---

## Exercise 2: Exploratory Data Analysis (EDA)

### Technique 1: Class distribution (imbalance analysis)
**Evidence**
- Table: `reports/eda_class_distribution.csv`
- Figure: `figures/eda_class_distribution.png`

Key values from `reports/eda_class_distribution.csv`:
- Train split: 7581 negatives, 794 positives, positive ratio = **9.48%**
- Dev split: 1895 negatives, 199 positives, positive ratio = **9.50%**

**Analysis**
- The dataset is strongly imbalanced (roughly 1:9.5 positive-to-negative).
- A naive classifier can get high accuracy by predicting mostly 0, so accuracy is not an informative objective.

**Impact on modeling choices**
- Use class-imbalance handling in training (class-weighted loss).
- Optimize threshold for positive-class F1 on dev, instead of using a fixed 0.5 threshold.
- Focus reporting on F1 for label 1 (PCL), matching task evaluation.

### Technique 2: Text length distribution
**Evidence**
- Table: `reports/eda_token_count_stats.csv`
- Figure: `figures/eda_token_count_distribution.png`

Key values from `reports/eda_token_count_stats.csv`:
- Negative class mean token count: **47.88** (median 42)
- Positive class mean token count: **53.62** (median 47)

**Analysis**
- PCL examples are slightly longer on average, indicating that useful cues may require broader context rather than isolated keywords.

**Impact on modeling choices**
- Use a transformer with contextual encoding rather than only sparse lexical features.
- Set maximum input length to 256 tokens for a context/efficiency trade-off.
- Preserve metadata (`keyword`, `country`) as additional context tokens in the input string.

---

## Exercise 3: Proposed Approach
### Proposed approach
I propose a **class-weighted DistilRoBERTa fine-tuning pipeline** with **metadata augmentation** and **dev-threshold optimization**:
1. Base model: `distilroberta-base`.
2. Input construction: prepend metadata tokens (`kw_<keyword> c_<country>`) before text.
3. Training loss: weighted cross-entropy using class weights from training label frequencies.
4. Model selection and inference: tune decision threshold on dev probabilities to maximize F1(PCL=1), then apply that threshold for dev/test predictions.

### Rationale and expected outcome
- DistilRoBERTa provides stronger contextual representation than tf-idf baselines, which is important for subtle condescension cues.
- Class-weighted loss directly targets minority-class under-detection.
- Threshold tuning aligns prediction behavior with the official metric and class imbalance.
- Expected outcome: higher positive-class recall and improved F1 above the RoBERTa-base baseline score (0.48 on dev).

---

## Exercise 4: Model Training (brief)
Training implementation is in:
- `scripts/train_roberta.py`

Best run command:
```bash
python scripts/train_roberta.py \
  --model-name distilroberta-base \
  --epochs 2 \
  --lr 1.5e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256 \
  --use-meta-tokens \
  --output-dir outputs/roberta_distil_e2 \
  --models-dir models/roberta_distil_e2 \
  --save-bestmodel
```

Selected best model details are stored in:
- `BestModel/MODEL_INFO.txt`
- `BestModel/TRANSFORMER_MODEL_PATH.txt`

---

## Exercise 5.1: Global Evaluation
### Required files
- `BestModel/dev.txt` (2094 lines)
- `BestModel/test.txt` (3832 lines)

### Dev performance (official dev labels available)
From `outputs/roberta_distil_e2/run_metadata.json`:
- Dev F1 (PCL=1): **0.5687**
- Tuned threshold: **0.3268**

### Comparison against baseline
- Shared-task baseline (RoBERTa-base): **0.48 (dev)**
- My submitted model: **0.5687 (dev)**
- Absolute improvement over baseline: **+0.0887**

Note: official test labels are hidden, so test-set F1 will be known only after leaderboard evaluation.

---

## Exercise 5.2: Local Evaluation
### A. Error analysis
**Evidence**
- `reports/roberta_distil_e2/false_positives_top50.csv`
- `reports/roberta_distil_e2/false_negatives_top50.csv`

**Observed error patterns**
1. **False positives** frequently occur in compassionate/supportive news language containing vulnerable-community keywords (e.g., `hopeless`, `in-need`, `homeless`) without actual patronising intent.
2. **False negatives** often appear in implicit or institutionally framed condescension where tone is subtle and less lexically obvious.
3. Category-level miss analysis on positive examples shows lower recall for nuanced categories like **presupposition** and **the poorer, the merrier**, indicating remaining weakness on implicit framing patterns.

### B. Other local evaluation
**Evidence**
- `figures/roberta_distil_e2/confusion_matrix_dev.png`
- `figures/roberta_distil_e2/score_distribution_by_outcome.png`
- `reports/roberta_distil_e2/LOCAL_EVALUATION.md`

Key metrics from `reports/roberta_distil_e2/LOCAL_EVALUATION.md`:
- F1 (PCL=1): **0.5687**
- Precision (PCL=1): **0.5463**
- Recall (PCL=1): **0.5930**
- False Positives: **98**
- False Negatives: **81**

Confusion-matrix counts on dev:
- TN=1797, FP=98, FN=81, TP=118

Interpretation:
- The current configuration achieves a stronger precision/recall balance than the earlier run while improving overall F1.
- Score-distribution overlap suggests many borderline cases; future gains likely require better representation of nuanced pragmatic cues rather than only threshold changes.

---

## Exercise 6: Communication and Reporting
### Repository organization summary
The repository is structured for reproducibility and marking convenience:
- `src/`: data loading and split alignment logic.
- `scripts/`: end-to-end pipeline scripts (EDA, training, evaluation, error analysis).
- `outputs/`: model outputs and detailed prediction files.
- `reports/`: report tables/analysis artifacts.
- `figures/`: generated figures.
- `BestModel/`: required deliverables for submission.

### Reproducibility notes
Main end-to-end baseline pipeline:
```bash
bash scripts/run_all.sh
```

Transformer best-model pipeline:
```bash
python scripts/train_roberta.py --use-meta-tokens --save-bestmodel --epochs 2 --lr 1.5e-5
python scripts/error_analysis.py \
  --dev-details outputs/roberta_distil_e2/dev_predictions_detailed.csv \
  --reports-dir reports/roberta_distil_e2 \
  --figures-dir figures/roberta_distil_e2
```

All key artifacts required by the spec are generated in deterministic file locations.

---

## Conclusion
### What worked
- Moving from a sparse-feature baseline to class-weighted DistilRoBERTa plus threshold tuning improved dev F1 to **0.5687**, clearly above the 0.48 baseline.
- Error analysis and local evaluation identified concrete failure modes, not just aggregate score changes.

### What failed and why
- The model still confuses empathetic/supportive language with patronising intent in some contexts.
- Implicit condescension categories remain difficult due to subtle semantics and annotation ambiguity.

### Next improvement
1. Add category-aware multi-task learning (binary + 7-category auxiliary loss).
2. Add hard-negative mining for supportive-but-non-PCL examples.
3. Test ensembling (DistilRoBERTa + lexical model) to reduce false positives without large recall loss.
