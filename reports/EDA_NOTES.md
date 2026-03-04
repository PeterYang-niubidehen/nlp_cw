# EDA Notes

## Technique 1: Class Distribution
Evidence: eda_class_distribution.csv + eda_class_distribution.png
Insight: PCL is minority class, so class imbalance handling is required.
Impact: Use class_weight='balanced' and optimize threshold for F1(positive class).

## Technique 2: Text Length Distribution
Evidence: eda_token_count_distribution.png + eda_token_count_stats.csv
Insight: Label-specific length differences can affect model behavior.
Impact: Keep tokenization robust and include n-grams to capture short and long cues.
