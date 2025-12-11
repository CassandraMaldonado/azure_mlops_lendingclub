# Azure Lending Club Loan Default Prediction

An end-to-end MLOps workflow on Azure Machine Learning using the Lending Club dataset to predict loan defaults, with comprehensive drift detection and model monitoring.

---

## 1. Overview

**Goal:** Predict whether a loan will default and operationalize the model using Azure Machine Learning with production-grade monitoring.

### What We Built

| Component | Description |
|-----------|-------------|
| **Model Training** | Azure AutoML to train and select the best model |
| **Model Registry** | MLflow model registered in Azure ML |
| **Model Evaluation** | Test job on held-out data with comprehensive metrics |
| **Drift Detection** | Evidently AI monitoring for feature, target, and performance drift |
| **Local Inference** | Python script for batch scoring and drift simulation |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AZURE MACHINE LEARNING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Data       │    │   AutoML     │    │   Model      │                   │
│  │   Assets     │───▶│   Training   │───▶│   Registry   │                   │
│  │              │    │              │    │              │                   │
│  │ • train data │    │ • experiment │    │ • best model │                   │
│  │ • test data  │    │ • serverless │    │ • MLflow     │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                 │                           │
└─────────────────────────────────────────────────┼───────────────────────────┘
                                                  │
                        ┌─────────────────────────┼───────────────────────────┐
                        │                         ▼                           │
                        │  ┌──────────────┐  ┌──────────────┐                 │
                        │  │  Test Model  │  │   Local      │                 │
                        │  │  (Azure)     │  │   Inference  │                 │
                        │  └──────────────┘  └──────┬───────┘                 │
                        │                           │                         │
                        │                           ▼                         │
                        │         ┌─────────────────────────────┐             │
                        │         │     DRIFT DETECTION         │             │
                        │         │     (Evidently AI)          │             │
                        │         │                             │             │
                        │         │  • Feature Drift            │             │
                        │         │  • Target Drift             │             │
                        │         │  • Performance Shift        │             │
                        │         └─────────────────────────────┘             │
                        │                    LOCAL ENVIRONMENT                │
                        └─────────────────────────────────────────────────────┘
```
### Components

| Layer | Resource | Purpose |
|-------|----------|---------|
| **Data** | `lending_club_final` | Training/validation data asset |
| | `lending_club_test` | Held-out test data asset |
| **Training** | `loan_default_experiment` | AutoML experiment |
| | `loan_default_automl_run_1` | AutoML job (Serverless CPU) |
| **Registry** | `loan_default_best_model` | Best model (MLflow format) |
| **Evaluation** | Test Model (Preview) | Azure ML evaluation on test set |
| **Monitoring** | Evidently AI | Drift detection and model monitoring |

> **Note:** Real-time deployment to an online endpoint was attempted but blocked by quota limits in the student subscription. The repo focuses on **training, registration, testing, and drift monitoring**, covering the core MLOps lifecycle.

---

## 3. Model Evaluation Summary

After registering the best AutoML model, evaluation was performed using Azure ML's **Test Model (Preview)** tool.

### Overall Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 0.912 | 91.2% of predictions correct |
| **AUC-ROC** | 0.957–0.975 | Excellent class separation |
| **F1 Score** | 0.944 | Strong balance of precision/recall |
| **Log Loss** | 0.204 | Well-calibrated probabilities |


### Evaluation Insights

| Analysis | Finding |
|----------|---------|
| **ROC Curve** | AUC > 0.95 consistently; strong sensitivity and specificity across thresholds |
| **Precision-Recall** | Precision > 0.85 at most recall levels; reliable high-risk identification |
| **Calibration** | Predicted probabilities align with actual default frequencies |
| **Lift Chart** | Top 10% highest-risk borrowers show 2×–3× lift over random selection |
| **Cumulative Gains** | Nearly all defaults captured within top 40% of risk-ranked borrowers |

### Conclusion

The model effectively distinguishes high-risk from low-risk borrowers, produces trustworthy probability estimates and delivers actionable ranking for credit-risk decisions.

---

## 4. Drift Detection & Model Monitoring

Production models degrade over time as real-world data changes. We implemented comprehensive drift monitoring using **Evidently AI**.

### Why Monitor for Drift?

```
Training Data (Reference)         Production Data (Drifted)
─────────────────────────         ─────────────────────────────
Model AUC:    0.954         ->     Model AUC:    0.529            <- -44.5%
F1 Score:     0.776         ->     F1 Score:     0.172            <- -77.8%
Precision:    0.769         ->     Precision:    0.217            <- -71.8%
Recall:       0.783         ->     Recall:       0.143            <- -81.7%
Avg Precision: 0.840        ->     Avg Precision:0.221            <- -73.7%
```

**Without drift monitoring, we'd be using a model that's essentially flipping a coin.**

### Three Types of Drift Monitored

| Drift Type | What It Detects | Why It Matters |
|------------|-----------------|----------------|
| **Covariate Shift** | Feature distributions changed. | Model may not generalize to new input patterns. |
| **Prior Probability Shift** | Target distribution changed. | Model's baseline assumptions are off. |
| **Performance Shift** | Model accuracy degraded. | Direct signal that retraining is needed. |

### Features Monitored

**Numerical Features (12 primary):**
- `loan_amnt`, `funded_amnt`, `funded_amnt_inv`, `int_rate`
- `installment`, `annual_inc`, `dti`, `open_acc`
- `pub_rec`, `revol_bal`, `revol_util`, `total_acc`

**Categorical Features (7):**
- `term`, `grade`, `sub_grade`, `home_ownership`
- `verification_status`, `purpose`, `application_type`

**Extended Analysis:** 86 features analyzed for comprehensive drift detection.

### Drift Detection Pipeline

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset

# 1. Feature Drift Detection
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=train_data, current_data=production_data)

# 2. Target Drift Detection  
target_drift_report = Report(metrics=[TargetDriftPreset()])
target_drift_report.run(reference_data=train_data, current_data=production_data)

# 3. Performance Monitoring
performance_report = Report(metrics=[ClassificationPreset()])
performance_report.run(reference_data=train_data, current_data=production_data)
```

### Alert Thresholds

| AUC Drift | Status | Recommended Action |
|-----------|--------|-------------------|
| < 2% | Normal | Continue monitoring. |
| 2–5% | Warning | Investigate root cause. |
| 5–15% | Critical | Prioritize model retraining. |
| > 15% | Emergency* | Stop predictions, immediate retraining required.|

> **Our Result:** 44.5% AUC drop -> Emergency level, this validates that our drift detection system correctly identifies catastrophic model failure.

### Generated Reports

| Report | Contents |
|--------|----------|
| `data_drift_report.html` | Per-feature drift analysis with statistical tests |
| `prediction_drift_report.html` | Prediction distribution comparison |
| `classification_quality_report.html` | Side-by-side performance metrics |
| `model_drift_dashboard.html` | Interactive summary dashboard |

### Visualizations

| File | Description |
|------|-------------|
| `roc_comparison.png` | Side-by-side ROC curves showing AUC collapse |
| `roc_curves.png` | Individual ROC curves for reference and production |
| `pr_curves.png` | Precision-Recall curves (AP: 0.84 → 0.22) |
| `confusion_matrices.png` | Confusion matrices showing prediction shifts |
| `metrics_comparison.png` | Bar chart comparing all metrics |

---

## 5. Results: Drifted vs Regular Test Data

We compared model performance on drifted data vs clean test data, revealing **catastrophic performance degradation**:

### ROC Curve Comparison

The production model (red) performs barely better than random guessing (dashed line), while the reference model (blue) shows excellent discrimination.

### Precision-Recall Analysis

| Dataset | Average Precision | Interpretation |
|---------|-------------------|----------------|
| Reference | 0.8403 | Strong performance across all thresholds. |
| Production | 0.2208 | Collapses immediately — nearly unusable. |

The PR curves tell a dramatic story: Reference maintains precision >0.75 up to 80% recall, while Production drops to ~0.20 precision almost immediately.

### Confusion Matrix Breakdown

| | Reference | Production | Change |
|--|-----------|------------|--------|
| **True Negatives** (Correct "Fully Paid") | 607,795 | 188,177 | -419,618 |
| **False Positives** (Wrong "Default") | 38,027 | 27,509 | -10,518 |
| **False Negatives** (Missed Defaults) | 34,991 | 45,748 | +10,757 |
| **True Positives** (Correct "Default") | 126,373 | 7,628 | -118,745 |

**Critical Issue:** The model went from catching **126,373 defaults** to only **7,628** — missing 94% of actual defaults on drifted data.

### What the Dashboard Shows

| Panel | Observation |
|-------|-------------|
| **ROC Curves** | Blue (reference) hugs top-left corner; Red (production) nearly follows the diagonal random line |
| **Prediction Distribution** | Both datasets show predictions clustered near 0, but production data has different underlying patterns |
| **Metrics Comparison** | All blue bars (reference) tower over red bars (production) — dramatic collapse across every metric |
| **Confusion Matrix Diff** | Massive shifts: -419,618 in True Negatives, -118,745 in True Positives |

### Key Finding

> ⚠️ **Critical Alert:** The model's AUC dropped from **0.954 to 0.529** — a **44.5% degradation** that renders the model essentially useless on drifted production data. The model now misses **94% of actual loan defaults**. This demonstrates why drift monitoring is essential: without it, we'd be approving high-risk loans thinking they're safe.

---

## 6. Project Structure

```
├── data/
│   ├── train_data_cleaned.csv
│   ├── test_data_cleaned.csv
│   └── test_dataset_drifted_v2.csv
├── src/
│   ├── inference_local.py          # Local model scoring
│   └── drift_detection.py          # Evidently drift analysis
├── reports/
│   ├── data_drift_report.html
│   ├── prediction_drift_report.html
│   ├── classification_quality_report.html
│   └── model_drift_dashboard.html
├── visualizations/
│   ├── roc_comparison.png          # ROC curves overlay
│   ├── roc_curves.png              # Individual ROC curves
│   ├── pr_curves.png               # Precision-Recall curves
│   ├── confusion_matrices.png      # Side-by-side confusion matrices
│   └── metrics_comparison.png      # Metrics bar chart
├── notebooks/
│   └── drift_detection.ipynb       # Full drift analysis notebook
└── README.md
```

---

## 7. Quick Start

### Prerequisites
```bash
pip install evidently pandas scikit-learn xgboost matplotlib seaborn plotly
```

### Run Drift Detection
```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load data
reference = pd.read_csv("data/train_data_cleaned.csv")
production = pd.read_csv("data/test_dataset_drifted_v2.csv")

# Run drift detection
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=production)
report.save_html("drift_report.html")
```

---

## 8. Key Takeaways

1. **Model Performance:** Azure AutoML achieved **0.954 AUC** and **0.840 Average Precision** on training data

2. **Drift Impact Demonstrated:** Production (drifted) data caused AUC to collapse from 0.954 → **0.529**, and the model now misses **94% of loan defaults**

3. **Monitoring Works:** Our Evidently AI pipeline successfully detected catastrophic drift before it could cause business damage

4. **Business Implication:** Without monitoring, we'd approve high-risk loans as safe — the model catches only 7,628 of 53,376 actual defaults

---

## 9. Future Improvements

- [ ] Deploy to Azure ML online endpoint (when quota available)
- [ ] Add `DataQualityPreset()` for input data validation
- [ ] Implement automated retraining triggers based on drift alerts
- [ ] Set up scheduled drift monitoring jobs in Azure ML

---

## 10. References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Azure ML AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

*MLOps Project - Azure Machine Learning + Evidently AI Drift Detection*
