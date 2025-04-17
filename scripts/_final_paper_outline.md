Final Paper Outline
================
Will Doyle
2025-04-17

# Introduction

## Problem Statement

- What is the prediction or classification problem?
- Why is it important? Who cares?

## Motivation & Use Cases

- Practical applications (e.g., policy, decision support, automation).
- Stakeholders who benefit from accurate predictions.
- Specific scenarios where model output could inform decisions.

# Data

## Data Source

- Description of dataset (e.g., origin, structure, time frame, units of
  analysis).

## Data Access and Cleaning

- Steps taken to access, preprocess, and prepare data.
- Handling of missing values, filtering, or transformations.

# Exploratory Analysis

## Univariate Analysis of the Outcome Variable

- Distribution, skew, presence of outliers, imbalance (for
  classification).

## Key Feature Relationships with Outcome

- Bivariate plots or summaries (scatterplots, boxplots, conditional
  means).
- Correlation or association patterns between predictors and the target.
- Domain-specific interpretation of patterns.

# Model Development

## Model Types Considered

- Justification for model choice (e.g., interpretability, performance).

## Hyperparameter Tuning

- Cross-validation setup (e.g., k-fold, stratified sampling).
- Optimization technique (e.g., grid search, Bayesian tuning).
- Evaluation metric used during tuning.

# Model Performance

## Evaluation on Holdout/Test Set

- Summary of performance metrics (e.g., RMSE, AUC, accuracy).
- Calibration/diagnostic plots if relevant.

## Comparison Across Models (Optional)

- If multiple models were considered, report relative performance.

# Final Model Interpretation

## Characteristics of Final Model

- Variable importance plot or table (e.g., SHAP values, permutation
  importance).
- Selected hyperparameters and rationale.
- Parsimony vs. complexity tradeoff (e.g., how many features matter?).

## Prediction Intervals

- Method used for constructing intervals (e.g., conformal prediction,
  quantile regression).
- Width and calibration of intervals on test data.
- Use case relevance: when are intervals more useful than point
  predictions?

## Robustness Checks

- Sensitivity to changes in training/test split, performance by
  subgroup, etc. (if applicable)

# Implications and Use Cases Revisited

## Application in Practice

- How the model might be used operationally or in policy settings.
- Caveats or conditions for deployment.

## Limitations and Future Work

- Known weaknesses (e.g., limited generalizability, temporal drift).
- Suggestions for additional data or extensions (e.g., adding time
  series, causal analysis).

# Appendix

- Code snippets, extended tables, alternative specifications, etc.
