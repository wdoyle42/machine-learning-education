# keras_els_pipeline.R
# Tune, fit, and evaluate a Keras MLP for multinomial CIP-2 major prediction
# using ELS:2002. Stripped-down script version of Lecture 18.

library(tidyverse)
library(haven)
library(here)
library(janitor)
library(tidymodels)
library(keras)
library(vip)

fit_model <- FALSE   # set TRUE to re-run tuning and final fit

# ── 1. Variable lists ─────────────────────────────────────────────────────────

els_vars <- c(
  "stu_id", "f3tzbch1cip2",
  "bygnstat", "bystlang", "byfcomp", "bysibhom",
  "bypared", "bymothed", "byfathed", "bygpared",
  "byoccum", "byoccuf", "byincome", "byses2",
  "bygrdrpt", "byriskfc",
  "bytxmstd", "bytxrstd",
  "bymathse", "byenglse", "byconexp", "byinstmo",
  "byactctl", "bystprep", "byacclim",
  "bystexp", "byparasp", "byocc30",
  "byhmwrk", "bynsport", "byxtracu", "byworksy", "bywrkhrs",
  "bysctrl", "byurban", "byregion",
  "f1stexp", "f1byedex", "f1mathse",
  "f1occ30", "f1psepln",
  "f1ses2",  "f1xtracu", "f1wrkhrs",
  "f1colinf", "f1himath", "f1pared",
  "f2evratt",  "f2ps1lvl", "f2ps1sec",
  "f2ps1ftp",  "f2ps1rem", "f2ps1aid",
  "f2psstrt",  "f2enrgap", "f2switch",
  "f2stexp",   "f2f1edex", "f2pseexm", "f2psepln",
  "f2psapsl",  "f2ptn3ps",
  "f3tzps1sec", "f3tzps1slc",
  "f3tzhs2ps1",
  "f3tzyr1ern", "f3tzyr2ern", "f3tzyr12ern",
  "f3tzyr1gpa", "f3tzyr2gpa",
  "f3tzremtot", "f3tzremmttot", "f3tzrementot",
  "f3tznumcrswd",
  "bysex", "byrace"
)

nominal_vars <- c(
  "bygnstat", "bystlang", "byfcomp",
  "byoccum",  "byoccuf",  "byocc30",
  "bysctrl",  "byurban",  "byregion",
  "byworksy",
  "f1psepln", "f1colinf", "f1himath", "f1occ30",
  "f2ps1lvl", "f2ps1sec",
  "f2ps1ftp", "f2ps1rem", "f2ps1aid",
  "f2enrgap", "f2switch", "f2pseexm",
  "f2ptn3ps",
  "f3tzps1sec", "f3tzps1slc"
)

# ── 2. Data loading and wrangling ─────────────────────────────────────────────

if (fit_model) {
  els_raw <- read_dta(here("scripts", "els_02_12_byf3pststu_v1_0.dta")) %>%
    clean_names() %>%
    select(any_of(els_vars)) %>%
    mutate(across(everything(), haven::zap_labels)) %>%
    mutate(across(where(is.numeric), ~ replace(.x, .x %in% -9:-1, NA)))

  cip_labels <- read_csv(here("scripts", "cip2_labels.csv"),
                         show_col_types = FALSE)

  cip_collapse <- cip_labels %>%
    select(cip2_code, cip2_collapsed_code, cip2_collapsed_label) %>%
    distinct()

  analytic_sample <- els_raw %>%
    mutate(across(any_of(nominal_vars), as.factor)) %>%
    left_join(cip_collapse, by = join_by(f3tzbch1cip2 == cip2_code)) %>%
    mutate(
      cip2 = factor(
        cip2_collapsed_code,
        levels = sort(unique(cip_collapse$cip2_collapsed_code))
      )
    ) %>%
    filter(f2evratt == 1, !is.na(cip2))

  saveRDS(analytic_sample, here("scripts", "analytic_sample.rds"))
} else {
  analytic_sample <- readRDS(here("scripts", "analytic_sample.rds"))
}

# ── 3. Train / test split and CV folds ───────────────────────────────────────

els_split <- initial_split(analytic_sample, prop = 0.80, strata = cip2)
els_train <- training(els_split)
els_test  <- testing(els_split)

if (fit_model) {
  els_folds <- vfold_cv(els_train, v = 5, strata = cip2)
}

# ── 4. Recipe ─────────────────────────────────────────────────────────────────

els_recipe <- recipe(cip2 ~ ., data = els_train) %>%
  update_role(stu_id,               new_role = "id") %>%
  update_role(bysex,                new_role = "subgroup") %>%
  update_role(byrace,               new_role = "subgroup") %>%
  update_role(f3tzbch1cip2,         new_role = "original_outcome") %>%
  update_role(cip2_collapsed_code,  new_role = "id") %>%
  update_role(cip2_collapsed_label, new_role = "id") %>%
  step_indicate_na(f3tzyr1ern, f3tzyr2ern, f3tzyr1gpa, f3tzyr2gpa) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ── 5. Model specification ────────────────────────────────────────────────────

keras_spec <- mlp(
  hidden_units = tune(),
  epochs       = tune(),
  penalty      = 0,
  dropout      = tune()
) %>%
  set_engine(
    "keras",
    validation_split = 0.2,
    callbacks = list(
      keras::callback_early_stopping(
        monitor              = "val_loss",
        patience             = 15,
        restore_best_weights = TRUE
      )
    )
  ) %>%
  set_mode("classification")

# ── 6. Workflow ───────────────────────────────────────────────────────────────

els_workflow <- workflow() %>%
  add_recipe(els_recipe) %>%
  add_model(keras_spec)

# ── 7. Tuning ─────────────────────────────────────────────────────────────────

tuning_grid <- grid_space_filling(
  hidden_units(range = c(32L, 256L)),
  epochs(range       = c(50L, 300L)),
  dropout(range      = c(0, 0.5)),
  size = 20
)

if (fit_model) {
  tuning_results <- tune_grid(
    els_workflow,
    resamples = els_folds,
    grid      = tuning_grid,
    metrics   = metric_set(roc_auc, accuracy, mn_log_loss),
    control   = control_grid(
      save_pred = TRUE,
      verbose   = TRUE,
      allow_par = FALSE   # Keras/TF manages its own threads; parallel R workers conflict
    )
  )

  saveRDS(tuning_results, here("scripts", "tuning_results.rds"))
} else {
  tuning_results <- readRDS(here("scripts", "tuning_results.rds"))
}

# Tuning overview
collect_metrics(tuning_results, summarize = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  select(hidden_units, epochs, dropout, mean, std_err) %>%
  print(n = 5)

collect_metrics(tuning_results) %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  mutate(rank = row_number()) %>%
  ggplot(aes(x = rank, y = mean, color = dropout)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                width = 0.3, alpha = 0.4) +
  scale_color_viridis_c(option = "plasma") +
  labs(
    title = "Tuning results: ROC AUC by rank",
    x     = "Rank (best to worst)",
    y     = "Mean ROC AUC",
    color = "Dropout"
  ) +
  theme_minimal()

# ── 8. Final fit ──────────────────────────────────────────────────────────────

best_params <- select_best(tuning_results, metric = "roc_auc")

final_workflow <- finalize_workflow(els_workflow, best_params)

final_fit <- last_fit(
  final_workflow,
  split   = els_split,
  metrics = metric_set(roc_auc, accuracy, mn_log_loss)
)

# Extract the fitted workflow for prediction (Keras models don't survive RDS serialization)
fitted_wf <- extract_workflow(final_fit)

# ── 9. Evaluation ─────────────────────────────────────────────────────────────

collect_metrics(final_fit) %>%
  select(.metric, .estimate) %>%
  print()

test_preds <- collect_predictions(final_fit)

# Confusion matrix
test_preds %>%
  count(truth = cip2, predicted = .pred_class, .drop = FALSE) %>%
  group_by(truth) %>%
  mutate(row_pct = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = predicted, y = fct_rev(truth), fill = row_pct)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = ifelse(n > 0, as.character(n), "")),
            size = 2.5, color = "grey20") +
  scale_fill_gradient(
    low = "white", high = "#2C7BB6",
    limits = c(0, 1), labels = scales::percent_format(accuracy = 1),
    name = "Row %\n(recall)"
  ) +
  scale_x_discrete(position = "top") +
  labs(
    title    = "Confusion matrix — test set",
    subtitle = "Rows: true class  |  Columns: predicted class",
    x = NULL, y = "True major"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = -40, hjust = 1, size = 7),
    axis.text.y = element_text(size = 7)
  )

# Macro-averaged per-class metrics
bind_rows(
  precision(test_preds, truth = cip2, estimate = .pred_class, estimator = "macro"),
  recall(   test_preds, truth = cip2, estimate = .pred_class, estimator = "macro"),
  f_meas(   test_preds, truth = cip2, estimate = .pred_class, estimator = "macro")
) %>%
  select(.metric, .estimate) %>%
  print()

# ── 10. Permutation importance ────────────────────────────────────────────────

# vip::vi_permute() cannot handle multiclass probability metrics directly,
# so we implement the loop ourselves.

vip_sample <- els_train %>% slice_sample(prop = 0.75)

score_auc <- function(df) {
  df %>%
    roc_auc(
      truth     = cip2,
      matches("^\\.pred_(?!class)", perl = TRUE),
      estimator = "macro_weighted"
    ) %>%
    pull(.estimate)
}

baseline_auc <- score_auc(augment(fitted_wf, vip_sample))
cat("Baseline AUC:", round(baseline_auc, 4), "\n")

non_pred_cols <- c(
  "cip2", "stu_id", "bysex", "byrace",
  "f3tzbch1cip2", "cip2_collapsed_code", "cip2_collapsed_label"
)
pred_features <- setdiff(names(vip_sample), non_pred_cols)

permute_one <- function(col, nsim = 5) {
  map_dbl(seq_len(nsim), function(i) {
    permuted_data  <- vip_sample %>% mutate(across(all_of(col), sample))
    permuted_probs <- augment(fitted_wf, permuted_data)
    baseline_auc - score_auc(permuted_probs)
  }) %>% mean()
}

if (fit_model) {
  vip_scores <- tibble(Variable = pred_features) %>%
    mutate(Importance = map_dbl(Variable, permute_one)) %>%
    arrange(desc(Importance))

  write_rds(vip_scores, here("scripts", "vip_scores.rds"))
} else {
  vip_scores <- read_rds(here("scripts", "vip_scores.rds"))
}

vip_scores %>%
  slice_max(Importance, n = 30) %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#2C7BB6", alpha = 0.85) +
  labs(
    title    = "Permutation importance (top 30)",
    subtitle = "Mean decrease in macro-weighted ROC AUC | 5 sims, 75% sample",
    x        = "Mean AUC drop",
    y        = NULL
  ) +
  theme_minimal()
