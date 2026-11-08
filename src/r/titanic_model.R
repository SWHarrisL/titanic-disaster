# Titanic survival model: logistic regression (R).
#
# Loads the Kaggle Titanic data from src/data/, explores it, engineers a few
# features, trains a logistic regression model and reports its accuracy on the
# training and test sets. Every step prints what it did, so the full analysis
# can be followed from the terminal output alone.
#
# Run with Docker from the repository root:
#   docker build -t titanic-r -f src/r/Dockerfile .
#   docker run --rm titanic-r

suppressPackageStartupMessages(library(dplyr))

DATA_DIR <- "src/data" # relative to the repository root, the image's WORKDIR
KAGGLE_URL <- "https://www.kaggle.com/competitions/titanic/data"
TARGET <- "Survived"
RANDOM_SEED <- 42
CV_FOLDS <- 5

# Model inputs, grouped by how each one is prepared. Unlike the Python model,
# Sex is left out: Title already encodes it, and an unregularized glm becomes
# unstable when both are included.
NUMERIC_FEATURES <- c("Age", "Fare", "FamilySize")
BINARY_FEATURES <- c("IsAlone", "HasCabin")
CATEGORICAL_FEATURES <- c("Pclass", "Embarked", "Title")
FEATURES <- c(NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES)
MODEL_FORMULA <- reformulate(FEATURES, response = TARGET)

# Raw columns replaced by the engineered features or without usable signal
DROPPED_COLUMNS <- c("Name", "Ticket", "Cabin", "SibSp", "Parch")

COMMON_TITLES <- c("Mr", "Mrs", "Miss", "Master")
CLASS_LABELS <- c("1" = "1st", "2" = "2nd", "3" = "3rd")
FAMILY_LABELS <- c("0" = "with family", "1" = "alone")
CABIN_LABELS <- c("0" = "not recorded", "1" = "recorded")

LINE_WIDTH <- 72


# Print a numbered section header.
section <- function(number, title) {
  cat(sprintf("\n[%d] %s\n", number, title))
}

# Print text wrapped to the output width.
paragraph <- function(text, indent = 0) {
  wrapped <- strwrap(text, width = LINE_WIDTH, indent = indent, exdent = indent)
  writeLines(wrapped)
}

# Format a proportion as a percentage, e.g. 0.3838 -> "38.4%".
pct <- function(value) {
  sprintf("%.1f%%", 100 * value)
}

# Survival rate for each value of a column, e.g. "female 74.2% | male 18.9%".
survival_by <- function(df, column, labels = NULL) {
  rates <- tapply(df[[TARGET]], df[[column]], mean)
  groups <- if (is.null(labels)) names(rates) else labels[names(rates)]
  paste(groups, pct(rates), collapse = " | ")
}

# Print a data frame without row names, indented like the rest of the output.
print_indented <- function(df, indent = "  ") {
  writeLines(paste0(indent, capture.output(print(df, row.names = FALSE))))
}

# Most frequent non-missing value, as text.
most_common <- function(values) {
  counts <- table(values)
  names(counts)[which.max(counts)]
}


# Read a CSV from src/data/, stopping with a clear message if it is missing.
load_csv <- function(filename, required = TRUE) {
  path <- file.path(DATA_DIR, filename)
  if (file.exists(path)) {
    return(read.csv(path, na.strings = c("", "NA")))
  }
  if (required) {
    message(
      "ERROR: ", path, " was not found.\n",
      "Download the Titanic data from ", KAGGLE_URL, ",\n",
      "put the CSV files in src/data/, then rebuild the Docker image."
    )
    quit(save = "no", status = 1)
  }
  NULL
}

# Add Title, FamilySize, IsAlone and HasCabin, then drop the raw columns.
add_features <- function(df) {
  df |>
    mutate(
      Title = sub("^[^,]*,\\s*([^.]*)\\..*$", "\\1", Name, perl = TRUE),
      Title = case_when(
        Title %in% c("Mlle", "Ms") ~ "Miss",
        Title == "Mme" ~ "Mrs",
        Title %in% COMMON_TITLES ~ Title,
        .default = "Rare"
      ),
      FamilySize = SibSp + Parch + 1L,
      IsAlone = as.integer(FamilySize == 1L),
      HasCabin = as.integer(!is.na(Cabin))
    ) |>
    select(-all_of(DROPPED_COLUMNS))
}

# Learn the values used to fill, scale and encode the data. This is only ever
# called on training rows, so nothing is learned from the passengers scored.
learn_preparation <- function(df) {
  medians <- vapply(df[NUMERIC_FEATURES], median, numeric(1), na.rm = TRUE)
  filled <- Map(
    function(values, fill) replace(values, is.na(values), fill),
    df[NUMERIC_FEATURES], medians
  )
  list(
    medians = medians,
    means = vapply(filled, mean, numeric(1)),
    sds = vapply(filled, sd, numeric(1)),
    modes = vapply(df[CATEGORICAL_FEATURES], most_common, character(1)),
    levels = lapply(
      df[CATEGORICAL_FEATURES],
      function(values) as.character(sort(unique(values[!is.na(values)])))
    )
  )
}

# Apply learned values: fill missing data, scale the numeric columns and turn
# the categorical columns into factors with the training levels.
prepare <- function(df, prep) {
  for (column in NUMERIC_FEATURES) {
    values <- df[[column]]
    values[is.na(values)] <- prep$medians[[column]]
    df[[column]] <- (values - prep$means[[column]]) / prep$sds[[column]]
  }
  for (column in CATEGORICAL_FEATURES) {
    values <- as.character(df[[column]])
    values[is.na(values)] <- prep$modes[[column]]
    df[[column]] <- factor(values, levels = prep$levels[[column]])
  }
  df
}

fit_model <- function(df) {
  glm(MODEL_FORMULA, family = binomial, data = df)
}

accuracy <- function(probability, actual) {
  mean((probability > 0.5) == (actual == 1))
}

# Stratified k-fold cross-validation. The preparation is relearned inside each
# fold, so every held-out fold is scored like genuinely new passengers.
cross_validate <- function(df) {
  set.seed(RANDOM_SEED)
  folds <- integer(nrow(df))
  for (outcome in unique(df[[TARGET]])) {
    rows <- which(df[[TARGET]] == outcome)
    folds[rows] <- sample(rep_len(seq_len(CV_FOLDS), length(rows)))
  }
  vapply(seq_len(CV_FOLDS), function(fold) {
    fit_rows <- df[folds != fold, ]
    holdout <- df[folds == fold, ]
    prep <- learn_preparation(fit_rows)
    model <- fit_model(prepare(fit_rows, prep))
    probability <- predict(
      model, newdata = prepare(holdout, prep), type = "response"
    )
    accuracy(probability, holdout[[TARGET]])
  }, numeric(1))
}


# Print a short profile of the training data.
explore <- function(train, test) {
  survived <- sum(train[[TARGET]])
  cat(sprintf(
    "Survival rate: %s (%d of %d passengers survived)\n",
    pct(survived / nrow(train)), survived, nrow(train)
  ))

  cat("Missing values (train / test):\n")
  for (column in names(test)) {
    n_train <- sum(is.na(train[[column]]))
    n_test <- sum(is.na(test[[column]]))
    if (n_train > 0 || n_test > 0) {
      cat(sprintf("  %-10s%4d / %d\n", column, n_train, n_test))
    }
  }

  cat("Survival rate by group:\n")
  cat("  Sex       ", survival_by(train, "Sex"), "\n", sep = "")
  cat("  Class     ", survival_by(train, "Pclass", CLASS_LABELS), "\n",
      sep = "")
  cat("  Port      ", survival_by(train, "Embarked"), "\n", sep = "")
  cat("            (C = Cherbourg, Q = Queenstown, S = Southampton)\n")

  age <- train$Age
  cat(sprintf(
    "Age:  median %.0f years, youngest %.2f, oldest %.0f\n",
    median(age, na.rm = TRUE), min(age, na.rm = TRUE), max(age, na.rm = TRUE)
  ))
  cat(sprintf(
    "Fare: median %.2f, highest %.2f\n",
    median(train$Fare, na.rm = TRUE), max(train$Fare, na.rm = TRUE)
  ))
}

# Print the engineered features and what they reveal about survival.
report_features <- function(train_fe, test_fe) {
  title_counts <- sort(table(train_fe$Title), decreasing = TRUE)
  cat("Added Title: taken from Name, with uncommon titles grouped as Rare\n")
  cat("  ", paste(names(title_counts), title_counts, collapse = ", "), "\n",
      sep = "")
  cat(sprintf(
    "Added FamilySize: SibSp + Parch + 1 (1 to %d people)\n",
    max(train_fe$FamilySize)
  ))
  cat(sprintf(
    "Added IsAlone: 1 if travelling without family (%d passengers)\n",
    sum(train_fe$IsAlone)
  ))
  cat(sprintf(
    "Added HasCabin: 1 if a cabin number is recorded (%d passengers)\n",
    sum(train_fe$HasCabin)
  ))
  paragraph(paste0(
    "Dropped ", paste(DROPPED_COLUMNS, collapse = ", "),
    ": captured by the new features, and Ticket has no usable pattern"
  ))
  cat("Columns now (train and test):\n")
  paragraph(paste(names(test_fe), collapse = ", "), indent = 2)

  cat("What the new features show:\n")
  cat("  Title     ", survival_by(train_fe, "Title"), "\n", sep = "")
  cat("  Family    ", survival_by(train_fe, "IsAlone", FAMILY_LABELS), "\n",
      sep = "")
  cat("  Cabin     ", survival_by(train_fe, "HasCabin", CABIN_LABELS), "\n",
      sep = "")
}

# Print the values learned from train to fill, scale and encode the data.
report_preparation <- function(prep, model, train_fe, test_fe) {
  report_fill <- function(column, fill_text) {
    n_train <- sum(is.na(train_fe[[column]]))
    n_test <- sum(is.na(test_fe[[column]]))
    if (n_train > 0 || n_test > 0) {
      cat(sprintf(
        "Filled %s with %s: %d missing in train, %d in test\n",
        column, fill_text, n_train, n_test
      ))
    }
  }
  for (column in NUMERIC_FEATURES) {
    report_fill(column, sprintf("the median (%.2f)", prep$medians[[column]]))
  }
  for (column in CATEGORICAL_FEATURES) {
    common_value <- prep$modes[[column]]
    report_fill(column, sprintf("the most common value (%s)", common_value))
  }

  cat(sprintf(
    "Scaled %s to mean 0 and standard deviation 1\n",
    paste(NUMERIC_FEATURES, collapse = ", ")
  ))
  baselines <- vapply(prep$levels, `[`, character(1), 1)
  paragraph(sprintf(
    "Encoded %s into %d dummy columns, each compared with a baseline (%s)",
    paste(CATEGORICAL_FEATURES, collapse = ", "),
    sum(lengths(prep$levels) - 1L),
    paste(names(baselines), baselines, collapse = ", ")
  ))
  paragraph(paste(
    "Left out Sex: Title already encodes it (Mr and Master are male, Miss",
    "and Mrs female), and an unregularized model becomes unstable with both"
  ))

  features <- names(coef(model))[-1]
  cat(sprintf(
    "Model input: %d rows x %d features\n", nrow(train_fe), length(features)
  ))
  paragraph(paste(features, collapse = ", "), indent = 2)
}

# Print training and cross-validated accuracy plus the strongest effects.
report_training <- function(model, train_fe) {
  train_accuracy <- accuracy(fitted(model), train_fe[[TARGET]])
  cv_scores <- cross_validate(train_fe)

  cat(sprintf("Training accuracy:        %s\n", pct(train_accuracy)))
  cat(sprintf(
    "Cross-validated accuracy: %s (+/- %s across %d folds)\n",
    pct(mean(cv_scores)), pct(sd(cv_scores)), CV_FOLDS
  ))
  paragraph(paste(
    "Cross-validation scores the model on passengers held out from training,",
    "so it is the best estimate of accuracy on passengers it has not seen."
  ), indent = 2)

  effects <- sort(coef(model)[-1], decreasing = TRUE)
  shown <- c(head(effects, 3), tail(effects, 3))
  cat("Strongest effects (coefficients; positive = more likely to survive):\n")
  writeLines(sprintf("  %-14s%+.2f", names(shown), shown))

  c(train = train_accuracy, cv = mean(cv_scores))
}

# Predict survival for every test passenger and print a summary.
predict_test <- function(model, test_fe, prep) {
  probability <- unname(predict(
    model, newdata = prepare(test_fe, prep), type = "response"
  ))
  predictions <- data.frame(
    PassengerId = test_fe$PassengerId,
    Pclass = test_fe$Pclass,
    Sex = test_fe$Sex,
    Age = test_fe$Age,
    Probability = round(probability, 2),
    Survived = as.integer(probability > 0.5)
  )

  n_survived <- sum(predictions$Survived)
  cat(sprintf(
    "Predicted survivors: %d of %d passengers (%s)\n",
    n_survived, nrow(predictions), pct(n_survived / nrow(predictions))
  ))
  cat("Predicted survival rate by sex: ", survival_by(predictions, "Sex"), "\n",
      sep = "")
  cat("First 5 predictions:\n")
  print_indented(head(predictions, 5))
  predictions
}

# Compare test predictions with gender_submission.csv, the only reference.
report_test_accuracy <- function(predictions, reference) {
  if (is.null(reference)) {
    paragraph(paste(
      "Kaggle keeps the true outcomes for test.csv private, and",
      "gender_submission.csv was not found in src/data/, so this step is",
      "skipped."
    ))
    return(NULL)
  }

  paragraph(paste(
    "Kaggle keeps the true outcomes for test.csv private, so the file has no",
    TARGET, "column. The only reference it provides is gender_submission.csv,",
    "an example submission that predicts every woman survived and every man",
    "died."
  ))
  merged <- merge(
    predictions, reference, by = "PassengerId", suffixes = c("", "_ref")
  )
  matches <- sum(merged[[TARGET]] == merged[[paste0(TARGET, "_ref")]])
  test_accuracy <- matches / nrow(merged)
  cat(sprintf(
    "Test accuracy against gender_submission.csv: %s (%d of %d match)\n",
    pct(test_accuracy), matches, nrow(merged)
  ))
  paragraph(paste(
    "This measures agreement with that baseline, not true accuracy. The",
    "cross-validated accuracy above is the better estimate of real",
    "performance."
  ), indent = 2)
  test_accuracy
}


main <- function() {
  cat("Titanic survival model: logistic regression (R)\n")
  cat(sprintf(
    "R %s | dplyr %s\n",
    as.character(getRversion()), as.character(packageVersion("dplyr"))
  ))

  section(1, "Loading the data")
  train <- load_csv("train.csv")
  test <- load_csv("test.csv")
  reference <- load_csv("gender_submission.csv", required = FALSE)
  cat(sprintf("train.csv: %d rows x %d columns\n", nrow(train), ncol(train)))
  cat(sprintf(
    "test.csv:  %d rows x %d columns (no %s)\n", nrow(test), ncol(test), TARGET
  ))
  cat("Columns:\n")
  paragraph(paste(names(train), collapse = ", "), indent = 2)

  section(2, "Exploring the training data")
  explore(train, test)

  section(3, "Adding and adjusting features (same steps on train and test)")
  train_fe <- add_features(train)
  test_fe <- add_features(test)
  report_features(train_fe, test_fe)

  section(4, "Preparing the features (learned from train, applied to test)")
  prep <- learn_preparation(train_fe)
  model <- fit_model(prepare(train_fe, prep))
  report_preparation(prep, model, train_fe, test_fe)

  section(5, "Training the logistic regression model")
  accuracies <- report_training(model, train_fe)

  section(6, "Predicting survival for test.csv")
  predictions <- predict_test(model, test_fe, prep)

  section(7, "Measuring accuracy on the test set")
  test_accuracy <- report_test_accuracy(predictions, reference)

  section(8, "Summary")
  metric_names <- c(
    "Training accuracy",
    sprintf("Cross-validated accuracy (%d folds)", CV_FOLDS)
  )
  metric_values <- c(accuracies[["train"]], accuracies[["cv"]])
  if (!is.null(test_accuracy)) {
    metric_names <- c(metric_names, "Test accuracy vs gender_submission.csv")
    metric_values <- c(metric_values, test_accuracy)
  }
  writeLines(sprintf("%-44s%7s", metric_names, pct(metric_values)))
}

main()
