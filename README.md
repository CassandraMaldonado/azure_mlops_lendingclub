# Azure Lending Club Loan Default (AutoML + Drift)

This repo contains an end-to-end MLOps workflow on Azure Machine Learning using the public **Lending Club** dataset to predict loan default.

---

## 1. Overview

**Goal:**  To predict whether a loan will default and operationalize the model using Azure Machine Learning:

- Used **Azure AutoML** to train many models.
- Registered the best MLflow model in the Azure ML model registry.
- Ran a test model job to evaluate the model on a held-out test set.
- Use a **Python testing script (`model_test.py`)** to:
  - Load the registered MLflow model locally.
  - Score a test set from 2015.
  - Compute metrics (AUC, accuracy, F1, confusion matrix).
  - Simulate **data drift** (e.g., by shifting feature distributions) and compare.

---

## 2. Architecture

High-level components:

1. **Data**  
   - `lending_club_final:1` – training/validation data asset.  
   - `lending_club_test:1` – 2015-10 test data asset.

2. **Training / AutoML**
   - Experiment: `loan_default_experiment`
   - AutoML job: `loan_default_automl_run_1`
   - Compute: **Serverless** CPU (configured in the UI).

3. **Model Registry**
   - Best model registered as  
     **`loan_default_best_model:1`** (type: MLflow).

4. **Model Testing**
   - **UI Test model job**: evaluates the registered model on `lending_club_test:1`.
   - **Scripted test (`src/model_test.py`)**: downloads the MLflow model and runs:
     - Metrics on the test set.
     - A simple **data drift check** vs training data.

5. **Local Inference**
   - `src/inference_local.py` loads the MLflow model and scores a few example loans.

> **Note:** Real-time deployment to an online endpoint was attempted, but blocked by quota limits in the student subscription. The repo therefore focuses on **training, registration, testing, and drift**, which already covers the core MLOps lifecycle.

---

## 3. Repository Structure

```text
azure-mlops-lendingclub/
├── data/               # (paths only, data not committed)
├── notebooks/          # EDA, AutoML setup, drift simulation
├── src/                # scripts for Azure ML + testing
├── model/              # downloaded MLflow model (ignored in git)
└── docs/               # diagrams / screenshots (optional)
