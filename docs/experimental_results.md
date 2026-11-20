# Experimental Results and Analysis

## 1. Objective

This experiment compares four classification model families under two training-data treatments: original imbalanced training data and SMOTE-resampled training data. The goal is to examine how the imbalance treatment changes fraud detection, especially the precision/recall trade-off.

## 2. Experimental setup

- Cleaned dataset: 283,726 rows and 30 input features.
- Target: `Class` (0 = legitimate transaction, 1 = fraudulent transaction).
- Stratified split, `random_state=42`: 181,584 training rows (64%), 45,396 validation rows (16%), 56,746 test rows (20%).
- Test composition: 56,651 legitimate transactions and 95 fraudulent transactions.
- `StandardScaler` was fitted on the training subset only and used to transform validation and test data.
- SMOTE was applied only to the training subset for the SMOTE experiments. The SMOTE training subset contained 181,282 records in each class after resampling.
- All reported test classifications use a decision threshold of 0.50.
- ROC-AUC and Average Precision were computed from model scores/probabilities, not from binary labels.

## 3. Full results

| Model | Imbalance treatment | Accuracy | Precision | Recall | F1-score | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | SMOTE | 97.10% | 4.84% | 87.37% | 9.17% | 95.41% | 68.77% |
| Logistic Regression | No SMOTE | 99.91% | 84.85% | 58.95% | 69.57% | 94.70% | 69.39% |
| Decision Tree | SMOTE | 98.64% | 8.86% | 76.84% | 15.89% | 82.60% | 36.11% |
| Decision Tree | No SMOTE | 99.93% | 86.67% | 68.42% | 76.47% | 85.38% | 63.96% |
| Random Forest | SMOTE | 99.86% | 54.68% | 80.00% | 64.96% | 97.40% | 76.05% |
| Random Forest | No SMOTE | 99.95% | 95.83% | 72.63% | 82.63% | 95.74% | 78.19% |
| Artificial Neural Network | SMOTE | 99.92% | 73.79% | 80.00% | 76.77% | 93.62% | 79.83% |
| Artificial Neural Network | No SMOTE | 99.93% | 96.67% | 61.05% | 74.84% | 95.09% | 81.09% |

The CSV file `results/model_comparison.csv` contains the recorded metric values, confusion-matrix counts, and the decision threshold. ANN metrics were transcribed at the four-decimal precision printed by the training script; confusion-matrix counts are exact.

## 4. Confusion-matrix counts

The matrix layout is `[[TN, FP], [FN, TP]]`.

| Model | Treatment | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|
| Logistic Regression | SMOTE | 55,018 | 1,633 | 12 | 83 |
| Logistic Regression | No SMOTE | 56,641 | 10 | 39 | 56 |
| Decision Tree | SMOTE | 55,900 | 751 | 22 | 73 |
| Decision Tree | No SMOTE | 56,641 | 10 | 30 | 65 |
| Random Forest | SMOTE | 56,588 | 63 | 19 | 76 |
| Random Forest | No SMOTE | 56,648 | 3 | 26 | 69 |
| Artificial Neural Network | SMOTE | 56,624 | 27 | 19 | 76 |
| Artificial Neural Network | No SMOTE | 56,649 | 2 | 37 | 58 |

## 5. Findings

### 5.1 Recall and false-positive trade-off

At the 0.50 threshold, SMOTE increased recall for every model family:

- Logistic Regression: 58.95% → 87.37% (+28.42 percentage points).
- Decision Tree: 68.42% → 76.84% (+8.42 percentage points).
- Random Forest: 72.63% → 80.00% (+7.37 percentage points).
- ANN: 61.05% → 80.00% (+18.95 percentage points).

The recall improvements came with more false positives and lower precision. For example, Random Forest produced 63 false positives with SMOTE compared with 3 without SMOTE, while detecting 76 versus 69 fraud cases. Logistic Regression showed a particularly large precision cost under SMOTE.

### 5.2 Best F1-score and precision

Random Forest without SMOTE achieved the highest F1-score (82.63%) and precision (95.83%) among the eight runs. At the reported threshold it identified 69 of 95 fraud cases and generated 3 false positives.

### 5.3 Ranking metrics

Random Forest with SMOTE achieved the highest ROC-AUC (97.40%). ANN without SMOTE achieved the highest Average Precision (81.09%). These metrics summarize score ranking across thresholds and should not be conflated with the classification results at the fixed 0.50 threshold.

### 5.4 Why accuracy is not enough

The test set contains 95 fraud cases out of 56,746 transactions. Because legitimate transactions dominate, accuracy can remain very high even when a model misses a substantial share of fraud cases. Precision, recall, F1-score, ROC-AUC, Average Precision, and confusion matrices are therefore needed to understand model behaviour.

## 6. Interpretation

The experiments show that SMOTE can be useful when recall is prioritized, but it does not automatically improve precision or F1-score. Its effect differs by model family. The best configuration depends on the operational cost assigned to missed fraud versus false alerts.

No single model should be described as universally best. Based on these runs:

- Choose Random Forest without SMOTE when prioritizing the best observed precision/F1 balance at threshold 0.50.
- Choose Logistic Regression with SMOTE when maximizing recall at threshold 0.50 is the immediate objective, while accepting a large false-positive burden.
- Note Random Forest with SMOTE for the best ROC-AUC.
- Note ANN without SMOTE for the best Average Precision.

These choices are conditional on this dataset, split, parameter settings, and threshold.

## 7. Limitations and next experiments

- One stratified split was used; repeated seeds or cross-validation would help estimate variability.
- The test threshold was fixed at 0.50. Threshold selection using validation data could be studied as a separate extension, with the test set kept untouched until final evaluation.
- No confidence intervals, calibration analysis, or business-cost evaluation were performed.
- The reported AP is scikit-learn's `average_precision_score` output (Average Precision), not necessarily trapezoidal area under a precision-recall curve.
- The dataset files are excluded from the repository, so readers must supply the source CSV locally to reproduce the run.

## 8. Reproducibility notes

- Split random state: 42.
- SMOTE random state: 42.
- Logistic Regression: `random_state=42`, `max_iter=1000`.
- Decision Tree: `random_state=42`, `max_depth=10`.
- Random Forest: `random_state=42`, `n_estimators=30`, `max_depth=10`, `n_jobs=-1`.
- ANN: Dense layer with 67 ReLU units followed by a one-unit Sigmoid output, Adam optimizer, binary cross-entropy loss, 10 epochs, batch size 32. The reported ANN runs used the 64/16/20 training/validation/test allocation.


## 9. Resume-ready project description

**Credit Card Fraud Detection using Machine Learning and Deep Learning** — Compared Logistic Regression, Decision Tree, Random Forest, and a TensorFlow/Keras ANN with and without SMOTE using a stratified 64:16:20 train/validation/test split. Evaluated models using precision, recall, F1-score, ROC-AUC, Average Precision, and confusion matrices; observed recall gains with SMOTE and the highest F1-score (82.63%) from Random Forest without SMOTE at the 0.50 threshold.
