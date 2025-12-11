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
