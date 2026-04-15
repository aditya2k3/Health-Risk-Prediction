# Multi-Disease Risk Prediction Project

This project predicts patient risk for three diseases using machine learning:
- Diabetes
- Heart disease
- Kidney disease

It includes dataset download, model training, saved model files, and final performance metrics.

## Objective

Build a screening-oriented ML system that identifies whether a patient is at risk for major diseases from clinical attributes.  
This is a **decision-support prototype** and not a medical diagnosis system.

## What Has Been Implemented

- `diabetes_risk_project.ipynb`: detailed diabetes workflow (EDA, preprocessing, training, evaluation, explainability)
- `multi_disease_risk_project.ipynb`: multi-disease analysis and comparison
- `download_train_export.py`: downloads datasets, trains models, saves artifacts, exports performance index
- `diabetes_risk_report.md`: deep technical report
- `project_objective_and_work_done.pdf`: concise project summary PDF

## Project Structure

- `datasets/` -> downloaded CSV files
- `models/` -> trained `.joblib` models
- `performance/` -> exported performance metrics and model index
- `results_package/` -> reusable package for reading and summarizing outcomes

## Datasets Used

- Diabetes: `datasets/diabetes.csv`
- Heart: `datasets/heart.csv`
- Kidney: `datasets/kidney.csv`

Source URLs are defined in `download_train_export.py`.

## Training and Evaluation Procedure

1. Load disease dataset
2. Clean and preprocess features (imputation + scaling)
3. Train candidate models (Logistic Regression, Random Forest)
4. Select best model with cross-validation ROC-AUC
5. Calibrate probabilities using `CalibratedClassifierCV`
6. Evaluate on holdout test split
7. Save model and export metrics

## Outcome Metrics

The following metrics are generated in `performance/performance_index.csv`:
- ROC-AUC
- PR-AUC
- F1-score
- Precision
- Recall
- Confusion values (`tp`, `tn`, `fp`, `fn`)

## Model Performance (Current Run)

| Disease | Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Diabetes | Logistic Regression | 0.8117 | 0.6716 | 0.5361 | 0.6047 | 0.4815 |
| Heart | Random Forest | 0.9026 | 0.9202 | 0.8649 | 0.7805 | 0.9697 |
| Kidney | Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and generate outputs:

```bash
python download_train_export.py
```

Generate a compact summary report from metrics package:

```bash
python -m results_package.metrics_report
```

## Output Files After Run

- `models/diabetes_model.joblib`
- `models/heart_model.joblib`
- `models/kidney_model.joblib`
- `performance/performance_index.csv`
- `performance/performance_index.json`
- `performance/model_index.json`
- `performance/results_summary.md` (from package)

## Important Note

This project is for risk screening and educational/research purposes.  
Any clinical decision must be reviewed by qualified medical professionals.
