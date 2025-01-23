Assignment 2
================

### Assignment: Analyzing Postsecondary Credits Earned Using Elastic Net and Validation Techniques

1.  **Data Cleaning**

Open up `hsls_extract2.csv` and apply corrections for missing data.

2.  **Exploring Relationships**
    - Create scatterplots to visualize the relationship between
      `x5postern` and two other variables of your choice (e.g.,
      `x1txmtscor` or `x1schooleng`).
    - Add linear regression lines to the plots. Describe any visible
      trends.
3.  **Elastic Net Regularization**
    - Define a recipe with `x5postern` as the outcome and all other
      variables as predictors.
    - Perform hyperparameter tuning for an elastic net model, varying
      `penalty` and `mixture` parameters. Use a grid search and a Monte
      Carlo cross-validation approach with 25 splits.
    - Identify the best combination of `penalty` and `mixture` based on
      RMSE.
4.  **Model Comparison**
    - Fit and evaluate three models on the training data: Ridge
      regression (`mixture=0`), Lasso regression (`mixture=1`), and
      Elastic Net (best parameters identified earlier).
    - Compare the RMSE and R-squared values for each model on the
      validation dataset. Which model performs best and why?
5.  **Variable Importance**
    - Using the final elastic net model, extract and rank the variables
      by their importance (coefficient magnitude).
    - Which predictors are the most influential for explaining the
      variance in `x5postern`? Provide an interpretation of the top
      three variables.
