# Azure Lending Club Loan Default Prediction

An end-to-end MLOps workflow on Azure Machine Learning using the Lending Club dataset to predict loan defaults, with comprehensive drift detection and model monitoring.

By: Cassandra Maldonado, Aida Aida Sarinzhipova, Mahima Masetty, Aarav Dewangan

---

## 1. Overview

**Goal:** Predict whether a loan will default and operationalize the model using Azure Machine Learning with production-grade monitoring.

### What We Built

| Component | Description |
|-----------|-------------|
| **Model Training** | Azure AutoML to train and select the best model |
| **Model Registry** | MLflow model registered in Azure ML. |
| **Model Evaluation** | Test job on held-out data with comprehensive metrics. |
| **Drift Detection** | Evidently AI monitoring for feature, target and performance drift. |
| **Local Inference** | Python script for batch scoring and drift simulation. |

---

## 2. Architecture

<div align="center">
  <img src="visualizations/architecture.png">
</div>

## 3. Model Evaluation Summary

After registering the best AutoML model, evaluation was performed using Azure ML's **Test Model (Preview)** tool.

### Overall Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 0.912 | 91.2% of predictions correct. |
| **AUC-ROC** | 0.957–0.975 | Excellent class separation. |
| **F1 Score** | 0.944 | Strong balance of precision/recall. |
| **Log Loss** | 0.204 | Well-calibrated probabilities. |


### Evaluation Insights

| Analysis | Finding |
|----------|---------|
| **ROC Curve** | AUC > 0.95 consistently, there's a strong sensitivity and specificity across thresholds. |
| **Precision-Recall** | Precision over 0.85 at most recall levels, this means a reliable high-risk identification. |
| **Calibration** | Predicted probabilities align with actual default frequencies. |
| **Lift Chart** | Top 10% highest-risk borrowers show a 2× to 3× lift over random selection. |
| **Cumulative Gains** | Nearly all defaults captured within top 40% of risk-ranked borrowers. |


The model effectively distinguishes high-risk from low-risk borrowers, produces trustworthy probability estimates and delivers actionable ranking for credit-risk decisions.

---

## 4. Drift Detection & Model Monitoring

Production models degrade over time as real-world data changes. We implemented comprehensive drift monitoring using **Evidently AI**.

### Why Monitor for Drift?

```
Training Data (Reference)         Production Data (Drifted)
─────────────────────────         ─────────────────────────────
Model AUC:    0.964         ->     Model AUC:    0.529            <- -44.5%
F1 Score:     0.921         ->     F1 Score:     0.172            <- -77.8%
Precision:    0.921         ->     Precision:    0.217            <- -71.8%
Recall:       0.867         ->     Recall:       0.143            <- -81.7%
Avg Precision: 0.969        ->     Avg Precision:0.221            <- -73.7%
```

**Without drift monitoring, we'd be using a model that's essentially like flipping a coin.**

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
| `roc_comparison.png` | Side-by-side ROC curves showing AUC collapse. |
| `roc_curves.png` | Individual ROC curves for reference and production. |
| `pr_curves.png` | Precision-Recall curves (AP: 0.84 -> 0.22) |
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

The PR curves tell a dramatic story: Reference maintains precision over 0.75 up to 80% recall, while Production drops to ~0.20 precision almost immediately.

### Confusion Matrix Breakdown

| | Reference | Production | Change |
|--|-----------|------------|--------|
| **True Negatives** (Correct "Fully Paid") | 607,795 | 188,177 | -419,618 |
| **False Positives** (Wrong "Default") | 38,027 | 27,509 | -10,518 |
| **False Negatives** (Missed Defaults) | 34,991 | 45,748 | +10,757 |
| **True Positives** (Correct "Default") | 126,373 | 7,628 | -118,745 |

**Critical Issue:** The model went from catching **126,373 defaults** to only **7,628**, missing 94% of actual defaults on drifted data.

### Key Finding

The model's AUC dropped from **0.964 to 0.529**, a **45.12% degradation** that renders the model essentially useless on drifted production data. The model now misses **94% of actual loan defaults**. This demonstrates why drift monitoring is essential: without it, we'd be approving high-risk loans thinking they're safe.

---

## 6. Project Structure

```
├── data/                           # Link to google drive with data provided due to file size issues
│   ├── raw_data.csv
│   ├── train.csv
│   └── test.csv
│   └── val.csv
│   └── val_drifted.csv
├── evidently reports/
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
│   └── architecture.png      # Overall pipeline architecture
├── notebooks/
│   └── EDA.ipynb
│   ├── test_drifted.ipynb      
│   └── model_monitoring_with_drift.ipynb
└── README.md
```

---


## 8. Key Takeaways

1. **Model Performance:** Azure AutoML achieved **0.964 AUC** and **0.969 Average Precision** on training data.

2. **Drift Impact Demonstrated:** Production (drifted) data caused AUC to collapse from 0.964 → **0.529**, and the model now misses **94% of loan defaults**.

3. **Monitoring Works:** Our Evidently AI pipeline successfully detected catastrophic drift before it could cause business damage.

4. **Business Implication:** Without monitoring, we'd approve high-risk loans as safe which causes high financial impact. 

---

## 10. References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Azure ML AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

