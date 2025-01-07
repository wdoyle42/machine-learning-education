Introduction to Tidyverse/Tidymodels
================
Will Doyle
2025-01-07

## Datset

The dataset for this session comes from HSLS, which is nationally
representative sample of ninth graders as of 2009. The data combines
questionnaire information and transcript information.

## Outcome of interest

We want to predict total HS GPA from a set of characteristics of
incoming ninth graders. The use case here is quickly identifying those
students that may end up with low GPAs so they can be targeted for early
interventions.

## Measure of accuracy

We’ll root mean sqaure error (rmse) as our measure of accuracy

## Load libraries

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
    ## ✔ broom        1.0.6     ✔ rsample      1.2.1
    ## ✔ dials        1.3.0     ✔ tune         1.2.1
    ## ✔ infer        1.0.7     ✔ workflows    1.1.4
    ## ✔ modeldata    1.4.0     ✔ workflowsets 1.1.0
    ## ✔ parsnip      1.2.1     ✔ yardstick    1.3.1
    ## ✔ recipes      1.1.0     
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard() masks purrr::discard()
    ## ✖ dplyr::filter()   masks stats::filter()
    ## ✖ recipes::fixed()  masks stringr::fixed()
    ## ✖ dplyr::lag()      masks stats::lag()
    ## ✖ yardstick::spec() masks readr::spec()
    ## ✖ recipes::step()   masks stats::step()
    ## • Learn how to get started at https://www.tidymodels.org/start/

``` r
library(janitor)
```

    ## 
    ## Attaching package: 'janitor'
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     chisq.test, fisher.test

## Load dataset

``` r
hs<-read_csv("hsls_extract.csv")%>%clean_names()
```

    ## Rows: 23503 Columns: 12
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (9): X1PAR1EDU, X1PAR1EMP, X1HHNUMBER, X1FAMINCOME, X1STUEDEXPCT, X1IEPF...
    ## dbl (3): X1TXMTSCOR, X1SCHOOLENG, X3TGPATOT
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

## Codebook

Categorical/factor variables marked with (f)

x3gpatot: Final HS GPA x1region: Region of the country (f) x1locale:
Locale (f) x1control: Control (f) x1iepflag: IEP Flag (f) x1stuedexpct:
Student expectations for educational attainment (f) x1schooleng: Student
engagement x1famincome: Family income (f) x1hhnumber: Number in

## Data Cleaning

``` r
hs <- hs %>%
  mutate(across(-x1txmtscor, ~ ifelse(. < 0, NA, .)))%>%
  drop_na()
```

`tidymodels` is a collection of R packages that provides a consistent
and tidy approach to building, evaluating, and tuning statistical
models. It integrates various stages of the modeling process, from data
preprocessing to model evaluation, under a unified framework that’s easy
to use and aligns with the principles of the `tidyverse`.

**Philosophy**: - **Tidy Data Principles**: `tidymodels` is built on the
same principles as the `tidyverse`, emphasizing clear, readable, and
consistent code. It’s designed to work seamlessly with other `tidyverse`
packages. - **Unified Interface**: It offers a consistent interface for
various modeling tasks, regardless of the underlying model type.

**Components**: - **`recipes`**: Provides tools for feature engineering
and data preprocessing. It allows you to specify a series of steps to
prepare your data for modeling. - **`parsnip`**: A unified interface for
model specification. It allows you to define a model without committing
to a specific computational engine. - **`workflows`**: Combines
pre-processing steps from `recipes` and model specifications from
`parsnip` into a single object to streamline the modeling process. -
**`tune`**: Tools for model tuning, such as grid search and
cross-validation. - **`yardstick`**: For model evaluation and metric
calculation. - **`dials`**: Helps in creating tuning grids for different
model parameters. - **`rsample`**: Provides infrastructure for data
splitting and resampling, which is essential for model validation and
testing.

**Workflow**: - **Data Splitting**: Using `rsample`, you can create
training and testing datasets. - **Preprocessing**: With `recipes`, you
define a series of preprocessing steps like normalization, encoding
categorical variables, and more. - **Model Specification**: Using
`parsnip`, you specify the type of model you want (e.g., linear
regression, decision tree) without choosing the computational engine
(e.g., lm, xgboost). - **Combining Preprocessing and Modeling**:
`workflows` lets you combine your preprocessing steps and model
specification into a single object. - **Model Training**: Train your
model on the training data. - **Model Evaluation**: Once trained, you
can evaluate your model’s performance on the testing data using
`yardstick`. - **Tuning (if needed)**: If your model has
hyperparameters, you can use `tune` and `dials` to find the best
parameters.

**Advantages**: - **Flexibility**: Easily switch between different model
types or computational engines without drastically changing your code. -
**Consistency**: Regardless of the model type, the code structure
remains consistent. - **Integration**: Designed to work seamlessly with
other `tidyverse` packages, making it easier to integrate modeling into
a broader data analysis pipeline.

Below I’ll use tidymodels to run a regresion predicting overall HS GPA
from the incoming student characteristics.

## Splitting into training and testing

The code is dividing the `cr` dataset into two parts: a training set
(`train`) to teach a machine learning model and a testing set (`test`)
to evaluate how well the model has learned. This split ensures that the
model is evaluated on data it hasn’t seen before, providing a more
realistic assessment of its performance.

``` r
hs_split<-initial_split(hs)

train<-training(hs_split)

test<-testing(hs_split)
```

**`hs_split <- initial_split(hs)`**: - This line uses the
`initial_split` function to divide the `cr` dataset into two parts. By
default, `initial_split` typically allocates 75% of the data to the
training set and the remaining 25% to the testing set, though this ratio
can be adjusted. - The result, `cr_split`, is a special split object
that contains information about which rows of the original `cr` dataset
belong to the training set and which belong to the testing set.

**`train <- training(hs_split)`**: - Here, the `training` function
extracts the training portion of the data from the `cr_split` object and
assigns it to the `train` variable. This dataset will be used to train a
machine learning model.

**`test <- testing(hs_split)`**: - Similarly, the `testing` function
extracts the testing portion of the data from the `cr_split` object and
assigns it to the `test` variable. This dataset will be used to evaluate
the performance of the trained model on unseen data.

The code is setting up a linear regression model using the standard
linear modeling engine in R.

``` r
hs_model<-linear_reg(mode="regression",engine="lm")
```

**Detailed Explanation**:

- **`linear_reg(mode="regression", engine="lm")`**:
  - **`linear_reg()`**: This function is from the `parsnip` package,
    which is part of the `tidymodels` framework. It’s used to specify a
    linear regression model.
  - **`mode="regression"`**: This argument specifies the type of
    modeling. In this case, it’s regression, which is used to predict a
    continuous outcome variable based on one or more predictor
    variables.
  - **`engine="lm"`**: This argument specifies the computational engine
    to use for the linear regression. The “lm” engine refers to R’s
    built-in `lm()` function for linear modeling.
- **`hs_model`**: The specified linear regression model is then stored
  in the `hs_model` variable. This doesn’t train the model yet; it
  merely sets up the type of model and the engine to be used. Training
  the model would require additional steps using the training data.

## Set Recipe

In the tidymodels framework, recipes is a package that provides a
streamlined way to define and preprocess data for modeling. It allows
users to specify a series of steps to transform and preprocess data,
such as normalization, encoding categorical variables, and handling
missing values. These steps are defined in a consistent and reproducible
manner, creating a “recipe” that can be applied to both training and new
datasets. This ensures that data transformations are consistent across
different stages of the modeling process, facilitating more reliable
model training and prediction. Essentially, recipes offers a tidy and
systematic approach to data preparation in the modeling workflow.

The code below is preparing the data for modeling. It sets up a series
of steps to process the data, such as handling missing values,
converting categorical variables into a format suitable for modeling,
and normalizing the data.

``` r
hs_formula<-as.formula("x3tgpatot~.")

hs_rec<-recipe(hs_formula,data=train)%>%
  update_role(x3tgpatot,new_role = "outcome")%>%
  step_other(all_nominal_predictors(),threshold = .01)%>%
  step_dummy(all_nominal_predictors())%>%
  step_filter_missing(all_predictors(),threshold = .1)%>%
  step_naomit(all_outcomes(),all_predictors())%>%
  step_corr(all_predictors(),threshold = .95)%>%
  step_zv(all_predictors())%>%
  step_normalize(all_predictors())
```

The code is preparing the data for modeling. It sets up a series of
steps to process the data, such as handling missing values, converting
categorical variables into a format suitable for modeling, and
normalizing the data.

**Detailed Explanation**:

1.  **`hs_formula <- as.formula("x3gpatot~.")`**:
    - This creates a formula indicating that the variable `x3gpatot` is
      the outcome (or dependent variable) we want to predict, and the
      `.` means we want to use all other variables in the dataset as
      predictors (or independent variables).
2.  **`recipe(hs_formula, data=train)`**:
    - The `recipe` function starts the specification of preprocessing
      steps. It uses the formula and the training data (`train`) as
      inputs.
3.  **`update_role(x3gpatot, new_role = "outcome")`**:
    - This explicitly sets the role of the `x3gpatot` variable as the
      “outcome” or the variable we’re trying to predict.
4.  **`step_other(all_nominal_predictors(), threshold = .01)`**:
    - For categorical (nominal) predictors, any categories that
      constitute less than 1% of the data will be lumped together into a
      new category, typically called “other”.
5.  **`step_dummy(all_nominal_predictors())`**:
    - Converts categorical variables into dummy variables (also known as
      one-hot encoding). This is necessary because many modeling
      algorithms require numerical input.
6.  **`step_filter_missing(all_predictors(), threshold = .1)`**:
    - This step removes any predictor variables that have more than 10%
      missing values.
7.  **`step_naomit(all_outcomes(), all_predictors())`**:
    - Removes rows (observations) from the data where either the outcome
      or any of the predictor variables have missing values.
8.  **`step_corr(all_predictors(), threshold = .95)`**:
    - Identifies and removes predictor variables that have a correlation
      higher than 0.95 with any other predictor. This helps in
      addressing multicollinearity.
9.  **`step_zv(all_predictors())`**:
    - Removes predictor variables that have a zero variance, meaning
      they have the same value for all observations.
10. **`step_normalize(all_predictors())`**:

- Normalizes all predictor variables so they have a mean of 0 and a
  standard deviation of 1. We’ll do this for everything, it’s a fairly
  standard step.

The result of all these steps is stored in `hs_rec`, which can then be
used to preprocess the training data and any future data in a consistent
manner before modeling.

## Viewing the results of feature engineering

``` r
hs_rec%>%prep()
```

    ## 

    ## ── Recipe ──────────────────────────────────────────────────────────────────────

    ## 

    ## ── Inputs

    ## Number of variables by role

    ## outcome:    1
    ## predictor: 11

    ## 

    ## ── Training information

    ## Training data contained 8331 data points and no incomplete rows.

    ## 

    ## ── Operations

    ## • Collapsing factor levels for: x1hhnumber and x1famincome, ... | Trained

    ## • Dummy variables from: x1par1edu, x1par1emp, x1hhnumber, ... | Trained

    ## • Missing value column filter removed: <none> | Trained

    ## • Removing rows with NA values in: x3tgpatot and x1txmtscor, ... | Trained

    ## • Correlation filter on: x1par1edu_Unit.non.response, ... | Trained

    ## • Zero variance filter removed: <none> | Trained

    ## • Centering and scaling for: x1txmtscor and x1schooleng, ... | Trained

``` r
hs_rec%>%prep()%>%bake(train)
```

    ## # A tibble: 8,331 × 47
    ##    x1txmtscor x1schooleng x3tgpatot x1par1edu_Bachelor.s.degree
    ##         <dbl>       <dbl>     <dbl>                       <dbl>
    ##  1     -0.769      -0.107       1                        -0.507
    ##  2     -0.405      -1.54        3                         1.97 
    ##  3     -1.31       -1.19        2.5                      -0.507
    ##  4      0.607      -0.554       3                         1.97 
    ##  5      1.18       -0.389       3.5                      -0.507
    ##  6      0.462       0.597       3                        -0.507
    ##  7     -0.516       1.49        2.5                      -0.507
    ##  8     -1.06       -0.389       2                        -0.507
    ##  9      0.939      -1.54        4                         1.97 
    ## 10      0.914       1.49        3                        -0.507
    ## # ℹ 8,321 more rows
    ## # ℹ 43 more variables: x1par1edu_High.school.diploma.or.GED <dbl>,
    ## #   x1par1edu_Less.than.high.school <dbl>, x1par1edu_Master.s.degree <dbl>,
    ## #   x1par1edu_Ph.D.M.D.Law.other.high.lvl.prof.degree <dbl>,
    ## #   x1par1emp_P1.currently.working.PT...35.hrs.wk. <dbl>,
    ## #   x1par1emp_P1.has.never.worked.for.pay <dbl>,
    ## #   x1par1emp_P1.not.currently.working.for.pay <dbl>, …

1.  **`hs_rec %>% prep()`**:
    - **`prep()`**: This function prepares the steps defined in the
      `cr_rec` recipe. It computes any required statistics or parameters
      needed for the transformations (e.g., mean and standard deviation
      for normalization) but doesn’t apply the transformations to the
      data yet. Think of it as getting the recipe ready for cooking but
      not actually cooking.
2.  **`hs_rec %>% prep() %>% bake(train)`**:
    - **`bake()`**: Once the recipe is prepared with `prep()`, the
      `bake()` function is used to apply the transformations to a
      dataset. In this case, the transformations are applied to the
      `train` dataset.
    - Essentially, this sequence of functions means “prepare the
      transformations and then apply them to the `train` dataset.”

By separating the `prep()` and `bake()` steps, `tidymodels` allows for a
consistent set of transformations to be easily applied to different
datasets, ensuring that both training and testing data undergo the same
preprocessing.

## Creating a workflow and fitting the model

``` r
hs_wf<-workflow()%>%
  add_model(hs_model)%>%
  add_recipe(hs_rec)%>%
  fit(train)
```

The code is setting up a workflow that combines both the data
preprocessing steps (from `hs_rec`) and the modeling specification (from
`hs_model`). It then trains the model using the `train` dataset.

1.  **`workflow()`**:
    - This function initializes a workflow. In the `tidymodels`
      framework, a workflow is a way to bundle together preprocessing
      steps (like those defined in a recipe) and a model specification.
2.  **`add_model(cr_model)`**:
    - This adds the model specification from `hs_model` (which was
      defined earlier) to the workflow. It tells the workflow what kind
      of model we’re planning to train.
3.  **`add_recipe(cr_rec)`**:
    - This adds the data preprocessing steps from `hs_rec` (which were
      defined earlier) to the workflow. It tells the workflow how to
      preprocess the data before training the model.
4.  **`fit(train)`**:
    - Once the workflow has both the model and the recipe, the `fit()`
      function is used to train the model using the `train` dataset.
      This involves applying the preprocessing steps to the training
      data and then training the model on the processed data.

The result is stored in `hs_wf`. This object now contains the trained
model along with all the preprocessing steps, making it ready for
predictions on new data.

## Testing predictions in the testing split

``` r
test<-
  hs_wf%>%
  predict(new_data=test)%>%
  bind_cols(test)
```

``` r
test%>%
  rmse(truth="x3tgpatot",estimate=.pred)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.594

The code calculates the Root Mean Squared Error (RMSE) for the
predictions in the `test` dataset by comparing the predicted values
(`.pred`) with the actual values (`x3tgpatot`).

**Detailed Explanation**:

1.  **`test %>%`**:
    - This takes the `test` dataset, which now contains both the actual
      values (`x3tgpatot`) and the predicted values (`.pred`), as the
      starting point for the subsequent operations.
2.  **`rmse(truth="iss", estimate=.pred)`**:
    - The `rmse()` function from the `yardstick` package (part of
      `tidymodels`) is used to calculate the RMSE.
    - **`truth="x3tgpatot"`**: Specifies that the actual values are in
      the `x3tgpatot` column of the `test` dataset.
    - **`estimate=.pred`**: Specifies that the predicted values are in
      the `.pred` column of the `test` dataset.

The result will be the RMSE value, which provides a measure of the
differences between the predicted and actual values. A lower RMSE
indicates a better fit of the model to the data, while a higher RMSE
suggests a poorer fit.

Note: Ensure that the `yardstick` package is loaded to use the `rmse()`
function.

## Examining Coefficients

Coefficients? Who cares about coefficients? We’ve already got an rmse!

The code fits the model above to the full dataset, extracts the
coefficients from the trained model from the workflow, organizes these
results in a tidy format, sorts them based on the magnitude of the
estimates (from highest to lowest), and then displays the top 100
results.

``` r
hs_wf_full<-
  hs_wf%>%
  fit(hs)

hs_wf_full%>%
  extract_fit_parsnip()%>%
  tidy()%>%
  arrange(-abs(estimate))%>%
  print(n=100)
```

    ## # A tibble: 46 × 5
    ##    term                                    estimate std.error statistic  p.value
    ##    <chr>                                      <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)                              2.93e+0   0.00573  512.     0       
    ##  2 x1txmtscor                               3.39e-1   0.00657   51.6    0       
    ##  3 x1schooleng                              1.23e-1   0.00581   21.2    7.05e-98
    ##  4 x1stuedexpct_High.school.diploma.or.GED -7.45e-2   0.00706  -10.6    6.02e-26
    ##  5 x1control_Public                        -6.96e-2   0.00628  -11.1    1.97e-28
    ##  6 x1famincome_Family.income....15.000.an… -5.98e-2   0.0104    -5.72   1.08e- 8
    ##  7 x1hhnumber_X4.Household.members          5.74e-2   0.0148     3.88   1.07e- 4
    ##  8 x1famincome_Family.income.less.than.or… -5.65e-2   0.00900   -6.28   3.52e-10
    ##  9 x1locale_Rural                           5.55e-2   0.00719    7.71   1.36e-14
    ## 10 x1famincome_Unit.non.response           -4.88e-2   0.0180    -2.71   6.84e- 3
    ## 11 x1hhnumber_X5.Household.members          3.97e-2   0.0130     3.05   2.26e- 3
    ## 12 x1region_South                          -3.52e-2   0.00715   -4.92   8.78e- 7
    ## 13 x1locale_Town                            3.47e-2   0.00652    5.31   1.09e- 7
    ## 14 x1par1edu_Bachelor.s.degree              3.35e-2   0.00855    3.92   8.94e- 5
    ## 15 x1hhnumber_X6.Household.members          3.16e-2   0.00995    3.17   1.51e- 3
    ## 16 x1region_Northeast                      -3.08e-2   0.00670   -4.59   4.39e- 6
    ## 17 x1par1edu_Master.s.degree                2.92e-2   0.00743    3.93   8.65e- 5
    ## 18 x1hhnumber_X3.Household.members          2.87e-2   0.0121     2.36   1.81e- 2
    ## 19 x1stuedexpct_Complete.an.Associate.s.d… -2.87e-2   0.00640   -4.48   7.57e- 6
    ## 20 x1famincome_Family.income....35.000.an… -2.70e-2   0.0102    -2.65   7.98e- 3
    ## 21 x1stuedexpct_Don.t.know                 -2.49e-2   0.00764   -3.26   1.11e- 3
    ## 22 x1stuedexpct_other                      -2.42e-2   0.00616   -3.93   8.40e- 5
    ## 23 x1hhnumber_X8.Household.members          2.24e-2   0.00660    3.38   7.14e- 4
    ## 24 x1stuedexpct_Complete.a.Master.s.degree  2.16e-2   0.00783    2.77   5.69e- 3
    ## 25 x1par1edu_Less.than.high.school         -1.83e-2   0.00712   -2.57   1.01e- 2
    ## 26 x1hhnumber_X7.Household.members          1.79e-2   0.00765    2.35   1.90e- 2
    ## 27 x1par1emp_P1.currently.working.PT...35…  1.71e-2   0.00608    2.81   4.96e- 3
    ## 28 x1par1emp_P1.not.currently.working.for… -1.33e-2   0.00635   -2.10   3.59e- 2
    ## 29 x1iepflag_Student.has.no.IEP             1.30e-2   0.00595    2.18   2.93e- 2
    ## 30 x1region_West                           -1.20e-2   0.00673   -1.78   7.50e- 2
    ## 31 x1par1edu_Ph.D.M.D.Law.other.high.lvl.…  1.07e-2   0.00669    1.60   1.10e- 1
    ## 32 x1stuedexpct_Complete.Ph.D.M.D.Law.oth…  1.03e-2   0.00799    1.29   1.98e- 1
    ## 33 x1par1edu_High.school.diploma.or.GED    -9.90e-3   0.00897   -1.10   2.70e- 1
    ## 34 x1famincome_Family.income....55.000.an… -7.60e-3   0.00995   -0.764  4.45e- 1
    ## 35 x1hhnumber_other                        -4.84e-3   0.00642   -0.755  4.50e- 1
    ## 36 x1famincome_Family.income....195.000.a…  4.58e-3   0.00658    0.696  4.86e- 1
    ## 37 x1locale_Suburb                          4.45e-3   0.00715    0.623  5.34e- 1
    ## 38 x1famincome_Family.income....235.000    -2.97e-3   0.00763   -0.388  6.98e- 1
    ## 39 x1iepflag_Student.has.an.IEP             2.36e-3   0.00607    0.389  6.97e- 1
    ## 40 x1famincome_Family.income....135.000.a… -2.31e-3   0.00757   -0.305  7.60e- 1
    ## 41 x1famincome_Family.income....75.000.an… -1.59e-3   0.00920   -0.173  8.63e- 1
    ## 42 x1famincome_Family.income....175.000.a… -1.33e-3   0.00641   -0.207  8.36e- 1
    ## 43 x1famincome_other                       -1.30e-3   0.00609   -0.213  8.32e- 1
    ## 44 x1par1emp_P1.has.never.worked.for.pay   -8.84e-4   0.00609   -0.145  8.85e- 1
    ## 45 x1famincome_Family.income....155.000.a… -7.95e-4   0.00668   -0.119  9.05e- 1
    ## 46 x1famincome_Family.income....95.000.an… -1.22e-4   0.00876   -0.0139 9.89e- 1

**Detailed Explanation**:

1.  **`hs_wf %>%`**:
    - This takes the trained workflow `cr_wf` as the starting point for
      the subsequent operations.
2.  **`extract_fit_parsnip()`**:
    - This function extracts the results of the trained model from the
      workflow. In the context of a linear regression model (as
      indicated by previous interactions), this would typically include
      coefficients for each predictor variable.
3.  **`tidy()`**:
    - This function, from the `broom` package, tidies the results of the
      model, converting them into a clean and standardized data frame
      format. For a linear regression model, this would typically result
      in a data frame with columns like `term` (the predictor variable
      name), `estimate` (the coefficient value), and others like
      `std.error`, `statistic`, and `p.value`.
4.  **`arrange(-estimate)`**:
    - This arranges (or sorts) the results based on the `estimate`
      column in descending order (from highest to lowest coefficient
      value).
5.  **`print(n=100)`**:
    - This displays the top 100 results after sorting. It’s especially
      useful if there are many predictor variables in the model and you
      want to see the ones with the highest coefficient values.

Overall, this code is useful for understanding the impact and
significance of different predictor variables in the trained model. The
sorted list can give insights into which variables have the most
substantial positive or negative effects on the outcome.

## Moving forward

We’ll next use lasso, ridge and elastic net to select the features that
contribute the most to the model, resulting in a model that should be
able to better predict the outcome in the testing data.
