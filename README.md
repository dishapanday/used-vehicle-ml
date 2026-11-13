# Used Vehicle ML: Price Prediction & Sales-Timeliness Classification

Two end-to-end machine learning projects on a dataset of 1,815 used-vehicle listings, built in Python with pandas and scikit-learn for MAI101 Machine Learning (Master of AI Design & Development, Seneca Polytechnic).

## 1. Selling-price regression (`Assignment1.ipynb`)

- **Goal:** predict a vehicle's selling price from attributes such as make, model, year, kilometres, engine size, condition and accident history.
- **Data cleaning:** removed invalid entries (negative kilometres, 0.0 engine sizes, out-of-range years), imputed missing values (median / mode), one-hot encoded categorical features, and dropped `LISTING_DAYS` to avoid target leakage.
- **Models:** Linear Regression baseline vs. Ridge Regression on a log-transformed target.
- **Results (20% hold-out test set):** MAE $3,143, RMSE $4,696, R² 0.871. The log-target Ridge model did not improve dollar-scale error (MAE $3,765), so the linear baseline was kept.

## 2. Will it sell on schedule? (`Assignment_2_ML.ipynb`)

- **Goal:** classify whether a vehicle sells within its body-type SLA window (binary target `ON_SCHEDULE`, 83% / 17% class imbalance).
- **Leakage-safe preprocessing:** scikit-learn `Pipeline` + `ColumnTransformer` (imputation, scaling, one-hot encoding) fitted on the training split only, with a stratified 80/20 split.
- **Models:** Decision Tree, Logistic Regression and kNN, tuned with 5-fold `GridSearchCV` on macro-F1, with balanced class weights where supported.
- **Results:** the Decision Tree was selected (CV macro-F1 0.558, test ROC-AUC 0.618). It flags 57% of late listings, which the always-"on schedule" baseline never catches.
- Includes coefficient interpretation, ROC curve comparison, a confusion matrix and a discussion of the business trade-off between missed late listings and false alarms.

## What I'd improve next

- Move all cleaning and encoding in the regression project inside a pipeline fitted after the train/test split, and add cross-validation and a naive baseline.
- Standardize inconsistent categorical labels (e.g. "Toyota" vs "TOYOTA") and reduce the high-cardinality `MODEL` feature.
- Try gradient-boosted trees and decision-threshold tuning for the classifier.

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · Google Colab

> The dataset was provided by the course and is not included in this repository.
