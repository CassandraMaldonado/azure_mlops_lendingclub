# Azure Lending Club Loan Default Prediction

An end-to-end MLOps workflow on Azure Machine Learning using the Lending Club dataset to predict loan defaults, with comprehensive drift detection and model monitoring.

---

### What I built

| Component | Description |
|-----------|-------------|
| **Model Training** | Azure AutoML to train and select the best model. |
| **Model Registry** | MLflow model registered in Azure ML. |
| **Model Evaluation** | Test job on held-out data with comprehensive metrics. |
| **Drift Detection** | Evidently AI monitoring for feature, target and performance drift. |
| **Local Inference** | Python script for batch scoring and drift simulation. |


## 2. Architecture

High-level components:

1. **Data**  
   - `lending_club_final:`: training/validation data asset.  
   - `lending_club_test:`: test data asset.

2. **Training / AutoML**
   - Experiment: `loan_default_experiment`.
   - AutoML job: `loan_default_automl_run_1`.
   - Compute: **Serverless** CPU.

3. **Model Registry**
   - Best model registered as `loan_default_best_model:`.

4. **Model Testing**
   - **UI Test model job**: evaluates the registered model on `lending_club_test:`.
   - **Scripted test**: downloads the MLflow model and runs:
     - Metrics on the test set.
     - A simple **data drift check** vs training data.

5. **Local Inference**
   - `src/inference_local.py` loads the MLflow model and scores a few example loans.

**Note:** Real-time deployment to an online endpoint was attempted, but blocked by quota limits in the student subscription. The repo therefore focuses on **training, registration, testing, and drift**, which already covers the core MLOps lifecycle.

## 3. Model evaluation summary

After registering the best AutoML model, I evaluated it using Azure ML’s Test Model (Preview) tool. This allowed me to validate the model on a fully held-out dataset using standardized metrics and diagnostic plots.

1. **Overall performance**

Metrics:
- **Accuracy:** 0.912

- **AUC:** 0.957–0.975

- **F1 (Binary)**: 0.944

- **Log Loss:** 0.204

The model achieves excellent separation between Charged Off vs. Fully Paid loans and produces reliable probability estimates, both essential in credit-risk modeling.

2. **ROC curve**

The ROC curves rise sharply toward the top-left corner with AUC consistently above 0.95. This meand the model maintains strong sensitivity and specificity across various thresholds and rarely confuses risky borrowers with low-risk ones.

3. **Precision–Recall curve**

Precision remains above 0.85 across most recall levels, so when the model identifies someone as high-risk, it is usually correct. This is especially important in imbalanced datasets, where defaults are rare.

4. **Calibration curve**

Predicted probabilities align closely with actual default frequencies. The model is not only accurate but also well-calibrated, which is crucial for decisions involving pricing, expected losses or credit limits.

5. **Lift chart**

The top 10% highest-risk borrowers show 2×–3× lift over random selection. Focusing on this top segment dramatically increases the number of true defaults captured, enabling more efficient manual reviews and proactive interventions.

6. **Cumulative gains curve**

Nearly all default cases are captured within the top 40% of borrowers ranked by risk. Operational teams can concentrate effort on a smaller portion of the portfolio while still identifying most high-risk customers—significantly reducing operational cost.

✔ Overall Conclusion

Across all evaluation tools, the model performs consistently well. It:

Distinguishes high-risk from low-risk borrowers effectively

Produces trustworthy probability estimates

Delivers meaningful lift and actionable ranking

Supports real-world credit-risk decisions

These characteristics make the model suitable for deployment in lending workflows and further integration within an MLOps pipeline.
