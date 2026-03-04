# Local Evaluation Report

- F1 (PCL=1): 0.4340
- Precision (PCL=1): 0.3911
- Recall (PCL=1): 0.4874
- False Positives: 151
- False Negatives: 102

## Error Analysis
- Inspect false_positives_top50.csv for cases where the model over-detects PCL.
- Inspect false_negatives_top50.csv for missed PCL cases.

## Additional Local Evaluation
- confusion_matrix_dev.png shows class-level trade-offs.
- score_distribution_by_outcome.png shows score separability and overlap.

## Suggested Discussion Points
- Are false negatives concentrated in subtle/implicit patronising language?
- Are false positives caused by keywords that are emotionally loaded but not condescending?
- What preprocessing or training changes can reduce these specific errors?
