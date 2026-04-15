# Diabetes Risk Prediction - Deep Technical Report

## 1) Abstract
This project develops a machine learning-based diabetes risk screening prototype using the public Pima Indians Diabetes dataset. The goal is to estimate patient-level risk probability and classify patients into risk groups for early screening support. The workflow includes data quality checks, preprocessing with leakage-safe pipelines, model comparison, probability calibration, threshold optimization with recall priority, and explainability analysis.

This system is designed for decision support and triage assistance only. It is not intended to replace clinician judgment or diagnostic testing.

## 2) Objective and Problem Definition
- **Task type:** Binary classification (`Outcome` in {0, 1})
- **Goal:** Predict whether a patient has elevated diabetes risk
- **Operational priority:** Higher recall for positive-risk cases to reduce missed high-risk patients
- **Output artifacts:**
  - Class prediction at selected threshold
  - Probability score in [0, 1]
  - Risk band (`Low`, `Medium`, `High`)

## 3) Dataset Description
- **Dataset:** Pima Indians Diabetes dataset (public CSV mirror)
- **Source URL used in notebook:**  
  `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`
- **Target:** `Outcome` (1 = elevated diabetes risk, 0 = lower risk)
- **Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### 3.1 Data quality concerns
Certain physiological variables contain zeros that are clinically implausible in many cases (for example, `Glucose`, `BMI`, `BloodPressure`). In this project, these zeros are treated as missing values and handled through median imputation in a preprocessing pipeline.

### 3.2 Class distribution
Class balance is inspected in notebook EDA to understand skew and support metric interpretation beyond raw accuracy.

## 4) Methodology

### 4.1 Train/Validation/Test protocol
- Stratified split into:
  - **Train:** 60%
  - **Validation:** 20%
  - **Test:** 20%
- The test split remains untouched for final performance reporting.

### 4.2 Preprocessing
- Replace invalid zeros with missing values in selected columns.
- Median imputation for missing numeric values.
- Standard scaling for numeric features.
- All transformations wrapped in scikit-learn `Pipeline`/`ColumnTransformer` to prevent data leakage.

### 4.3 Candidate models
- Logistic Regression (class-weight balanced)
- Random Forest (class-weight balanced)
- Gradient Boosting

### 4.4 Model selection
5-fold stratified cross-validation is applied on training data using:
- ROC-AUC
- PR-AUC (Average Precision)
- F1
- Precision
- Recall

Best model is selected by strongest discriminatory performance (primarily ROC-AUC) with recall considered for screening suitability.

### 4.5 Probability calibration
The selected model is calibrated using `CalibratedClassifierCV` (`sigmoid`) so probability outputs better reflect empirical risk likelihood, supporting threshold policy decisions.

### 4.6 Threshold optimization
Threshold is selected on validation data with a recall-prioritized policy:
- Target minimum recall = 0.80
- Among eligible thresholds, pick the threshold with highest F1
- If no threshold reaches minimum recall, fallback to global best F1

## 5) Evaluation Framework
Final evaluation on the untouched test split includes:
- ROC-AUC
- PR-AUC
- F1
- Precision
- Recall
- Brier score (probability quality)
- Confusion matrix
- Classification report

### 5.1 Metrics table (fill from notebook output)
| Metric | Value |
|---|---|
| ROC-AUC | _fill after run_ |
| PR-AUC | _fill after run_ |
| F1 | _fill after run_ |
| Precision | _fill after run_ |
| Recall | _fill after run_ |
| Brier Score | _fill after run_ |
| Selected Threshold | _fill after run_ |

## 6) Explainability and Clinical Interpretability

### 6.1 Global explainability
Permutation importance is computed on the test set (ROC-AUC scoring) to estimate each feature's impact on predictive performance.

### 6.2 Local case interpretation
Sample patient rows from low, medium, and high predicted risk groups are shown to support clinician-style interpretation of probability-based triage.

## 7) Risk Banding for Triage
This project defines practical risk groups from calibrated probabilities:
- **Low risk:** p < 0.33
- **Medium risk:** 0.33 <= p < 0.66
- **High risk:** p >= 0.66

These bands are configurable and should be refined based on clinical workflow, prevalence, and acceptable false-negative trade-offs.

## 8) Ethical, Bias, and Safety Considerations
- The dataset represents a limited population and may not generalize to all demographics.
- Model outputs can reflect historical data biases.
- False negatives may delay follow-up screening; false positives may increase unnecessary testing burden.
- The tool must be used with clinician oversight and should never serve as a standalone diagnosis.

## 9) Limitations
- Single public dataset; no external validation cohort.
- No temporal validation for drift.
- Limited feature space (lifestyle, family history depth, and lab panel breadth are not fully represented).
- Calibration and threshold policy may shift in different populations.

## 10) Deployment Outlook (Prototype to Practice)
To transition toward real-world deployment:
1. Validate on external multi-center cohorts.
2. Perform subgroup performance audits for fairness.
3. Integrate with EHR intake flow as a screening assistant.
4. Monitor performance drift and recalibrate periodically.
5. Introduce human-in-the-loop review for all high-risk flags.

## 11) Reproducibility Notes
- Random seed is fixed (`RANDOM_STATE = 42`).
- All training steps are notebook-contained.
- Dependency versions should be pinned (see optional `requirements.txt`).

## 12) Conclusion
This project demonstrates a robust, reproducible machine learning workflow for diabetes risk screening with emphasis on recall-sensitive thresholding, calibrated probabilities, and explainable outputs. It is suitable as a real-world prototype baseline, with clear constraints and next steps for responsible clinical adaptation.

## 13) Multi-Disease Extension (Heart and Kidney)
The project is extended with a second notebook:
- `multi_disease_risk_project.ipynb`

This notebook performs parallel risk modeling for:
- Diabetes
- Heart disease
- Chronic kidney disease

### 13.1 What is included
- Disease-wise dataset loading (with fallback URLs)
- Unified preprocessing and evaluation pipeline
- Disease-wise model selection (Logistic Regression vs Random Forest)
- Calibrated probability outputs
- Comparative metrics table across diseases
- Disease-specific top feature importance

### 13.2 Why this matters
- Supports integrated early-risk screening instead of one-disease-only analysis.
- Enables side-by-side model reliability checks across disease domains.
- Helps prioritize which screening model is ready first for pilot deployment.

### 13.3 Additional cautions for multi-disease use
- Do not reuse one threshold policy across all diseases.
- External validation must be done separately per disease population.
- Feature definitions and data quality assumptions differ by dataset source.
