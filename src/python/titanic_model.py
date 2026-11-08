"""Titanic survival model: logistic regression.

Loads the Kaggle Titanic data from src/data/, explores it, engineers a few
features, trains a logistic regression model and reports its accuracy on the
training and test sets. Every step prints what it did, so the full analysis
can be followed from the terminal output alone.

Run with Docker from the repository root:
    docker build -t titanic-python -f src/python/Dockerfile .
    docker run --rm titanic-python
"""

import platform
import sys
import textwrap
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_URL = "https://www.kaggle.com/competitions/titanic/data"
TARGET = "Survived"
RANDOM_STATE = 42
CV_FOLDS = 5

# Model inputs, grouped by how each one is prepared
NUMERIC_FEATURES = ["Age", "Fare", "FamilySize"]
BINARY_FEATURES = ["IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Pclass", "Embarked", "Title"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Raw columns that the engineered features replace or that carry no usable signal
DROPPED_COLUMNS = ["Name", "Ticket", "Cabin", "SibSp", "Parch"]

COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
CLASS_LABELS = {1: "1st", 2: "2nd", 3: "3rd"}
FAMILY_LABELS = {0: "with family", 1: "alone"}
CABIN_LABELS = {0: "not recorded", 1: "recorded"}

LINE_WIDTH = 72


def section(number: int, title: str) -> None:
    """Print a numbered section header."""
    print(f"\n[{number}] {title}")


def paragraph(text: str, indent: str = "") -> None:
    """Print text wrapped to the output width."""
    print(
        textwrap.fill(
            text, LINE_WIDTH, initial_indent=indent, subsequent_indent=indent
        )
    )


def pct(value: float) -> str:
    """Format a proportion as a percentage, e.g. 0.3838 -> '38.4%'."""
    return f"{value:.1%}"


def survival_by(df: pd.DataFrame, column: str, labels: dict | None = None) -> str:
    """Survival rate for each value of a column, e.g. 'female 74.2% | male 18.9%'."""
    rates = df.groupby(column)[TARGET].mean()
    if labels:
        rates = rates.rename(index=labels)
    return " | ".join(f"{group} {pct(rate)}" for group, rate in rates.items())


def load_csv(filename: str, required: bool = True) -> pd.DataFrame | None:
    """Read a CSV from src/data/, stopping with a clear message if it is missing."""
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    if required:
        sys.exit(
            f"ERROR: src/data/{filename} was not found.\n"
            f"Download the Titanic data from {KAGGLE_URL},\n"
            "put the CSV files in src/data/, then rebuild the Docker image."
        )
    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Title, FamilySize, IsAlone and HasCabin, then drop the raw columns."""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    title = title.replace(TITLE_ALIASES)
    df["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df.drop(columns=DROPPED_COLUMNS)


def build_model() -> Pipeline:
    """Preprocessing (imputation, scaling, one-hot encoding) + logistic regression.

    Every value used to fill, scale or encode the data is learned from the
    training set only and then applied unchanged to the test set.
    """
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def explore(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print a short profile of the training data."""
    survived = int(train[TARGET].sum())
    print(
        f"Survival rate: {pct(survived / len(train))} "
        f"({survived} of {len(train)} passengers survived)"
    )

    print("Missing values (train / test):")
    for column in test.columns:
        n_train, n_test = train[column].isna().sum(), test[column].isna().sum()
        if n_train or n_test:
            print(f"  {column:<10}{n_train:>4} / {n_test}")

    print("Survival rate by group:")
    print(f"  Sex       {survival_by(train, 'Sex')}")
    print(f"  Class     {survival_by(train, 'Pclass', CLASS_LABELS)}")
    print(f"  Port      {survival_by(train, 'Embarked')}")
    print("            (C = Cherbourg, Q = Queenstown, S = Southampton)")

    age, fare = train["Age"], train["Fare"]
    print(
        f"Age:  median {age.median():.0f} years, "
        f"youngest {age.min():.2f}, oldest {age.max():.0f}"
    )
    print(f"Fare: median {fare.median():.2f}, highest {fare.max():.2f}")


def report_features(train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> None:
    """Print the engineered features and what they reveal about survival."""
    title_counts = train_fe["Title"].value_counts()
    print("Added Title: taken from Name, with uncommon titles grouped as Rare")
    print("  " + ", ".join(f"{t} {n}" for t, n in title_counts.items()))
    print(
        "Added FamilySize: SibSp + Parch + 1 "
        f"(1 to {train_fe['FamilySize'].max()} people)"
    )
    print(
        "Added IsAlone: 1 if travelling without family "
        f"({train_fe['IsAlone'].sum()} passengers)"
    )
    print(
        "Added HasCabin: 1 if a cabin number is recorded "
        f"({train_fe['HasCabin'].sum()} passengers)"
    )
    paragraph(
        f"Dropped {', '.join(DROPPED_COLUMNS)}: captured by the new features, "
        "and Ticket has no usable pattern"
    )
    print("Columns now (train and test):")
    paragraph(", ".join(test_fe.columns), indent="  ")

    print("What the new features show:")
    print(f"  Title     {survival_by(train_fe, 'Title')}")
    print(f"  Family    {survival_by(train_fe, 'IsAlone', FAMILY_LABELS)}")
    print(f"  Cabin     {survival_by(train_fe, 'HasCabin', CABIN_LABELS)}")


def report_preprocessing(
    model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Print the values the fitted pipeline used to fill, scale and encode data."""
    preprocess = model.named_steps["preprocess"]
    transformers = preprocess.named_transformers_
    numeric_imputer = transformers["numeric"].named_steps["simpleimputer"]
    categorical_imputer = transformers["categorical"].named_steps["simpleimputer"]

    def report_fill(column: str, fill_value: str) -> None:
        n_train, n_test = X_train[column].isna().sum(), X_test[column].isna().sum()
        if n_train or n_test:
            print(
                f"Filled {column} with {fill_value}: "
                f"{n_train} missing in train, {n_test} in test"
            )

    for column, median in zip(NUMERIC_FEATURES, numeric_imputer.statistics_):
        report_fill(column, f"the median ({median:.2f})")
    for column, mode in zip(CATEGORICAL_FEATURES, categorical_imputer.statistics_):
        report_fill(column, f"the most common value ({mode})")

    feature_names = preprocess.get_feature_names_out()
    n_encoded = len(feature_names) - len(NUMERIC_FEATURES) - len(BINARY_FEATURES)
    print(
        f"Scaled {', '.join(NUMERIC_FEATURES)} "
        "to mean 0 and standard deviation 1"
    )
    print(
        f"Encoded {', '.join(CATEGORICAL_FEATURES)} "
        f"into {n_encoded} one-hot columns"
    )
    print(f"Model input: {len(X_train)} rows x {len(feature_names)} features")
    paragraph(", ".join(feature_names), indent="  ")


def report_training(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, float]:
    """Print training and cross-validated accuracy plus the strongest effects."""
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="accuracy")

    print(f"Training accuracy:        {pct(train_accuracy)}")
    print(
        f"Cross-validated accuracy: {pct(cv_scores.mean())} "
        f"(+/- {pct(cv_scores.std())} across {CV_FOLDS} folds)"
    )
    paragraph(
        "Cross-validation scores the model on passengers held out from training, "
        "so it is the best estimate of accuracy on passengers it has not seen.",
        indent="  ",
    )

    names = model.named_steps["preprocess"].get_feature_names_out()
    effects = pd.Series(model.named_steps["classifier"].coef_[0], index=names)
    effects = effects.sort_values(ascending=False)
    print("Strongest effects (coefficients; positive = more likely to survive):")
    for name, value in pd.concat([effects.head(3), effects.tail(3)]).items():
        print(f"  {name:<14}{value:+.2f}")

    return train_accuracy, cv_scores.mean()


def predict_test(model: Pipeline, test_fe: pd.DataFrame) -> pd.DataFrame:
    """Predict survival for every test passenger and print a summary."""
    X_test = test_fe[FEATURES]
    predictions = test_fe[["PassengerId", "Pclass", "Sex", "Age"]].copy()
    predictions["Probability"] = model.predict_proba(X_test)[:, 1].round(2)
    predictions[TARGET] = model.predict(X_test)

    n_survived = int(predictions[TARGET].sum())
    print(
        f"Predicted survivors: {n_survived} of {len(predictions)} passengers "
        f"({pct(n_survived / len(predictions))})"
    )
    print(f"Predicted survival rate by sex: {survival_by(predictions, 'Sex')}")
    print("First 5 predictions:")
    print(textwrap.indent(predictions.head().to_string(index=False), "  "))
    return predictions


def report_test_accuracy(
    predictions: pd.DataFrame, reference: pd.DataFrame | None
) -> float | None:
    """Compare test predictions with gender_submission.csv, the only reference."""
    if reference is None:
        paragraph(
            "Kaggle keeps the true outcomes for test.csv private, and "
            "gender_submission.csv was not found in src/data/, so this step is skipped."
        )
        return None

    paragraph(
        "Kaggle keeps the true outcomes for test.csv private, so the file has no "
        f"{TARGET} column. The only reference it provides is gender_submission.csv, "
        "an example submission that predicts every woman survived and every man died."
    )
    merged = predictions.merge(reference, on="PassengerId", suffixes=("", "_ref"))
    accuracy = accuracy_score(merged[f"{TARGET}_ref"], merged[TARGET])
    matches = int((merged[TARGET] == merged[f"{TARGET}_ref"]).sum())
    print(
        f"Test accuracy against gender_submission.csv: {pct(accuracy)} "
        f"({matches} of {len(merged)} match)"
    )
    paragraph(
        "This measures agreement with that baseline, not true accuracy. The "
        "cross-validated accuracy above is the better estimate of real performance.",
        indent="  ",
    )
    return accuracy


def main() -> None:
    print("Titanic survival model: logistic regression (Python)")
    print(
        f"Python {platform.python_version()} | pandas {pd.__version__} | "
        f"scikit-learn {sklearn.__version__}"
    )

    section(1, "Loading the data")
    train = load_csv("train.csv")
    test = load_csv("test.csv")
    reference = load_csv("gender_submission.csv", required=False)
    print(f"train.csv: {train.shape[0]} rows x {train.shape[1]} columns")
    print(f"test.csv:  {test.shape[0]} rows x {test.shape[1]} columns (no {TARGET})")
    print("Columns:")
    paragraph(", ".join(train.columns), indent="  ")

    section(2, "Exploring the training data")
    explore(train, test)

    section(3, "Adding and adjusting features (same steps on train and test)")
    train_fe, test_fe = add_features(train), add_features(test)
    report_features(train_fe, test_fe)

    section(4, "Preparing the features (learned from train, applied to test)")
    X_train, y_train = train_fe[FEATURES], train_fe[TARGET]
    model = build_model().fit(X_train, y_train)  # learns everything from train only
    report_preprocessing(model, X_train, test_fe[FEATURES])

    section(5, "Training the logistic regression model")
    train_accuracy, cv_accuracy = report_training(model, X_train, y_train)

    section(6, "Predicting survival for test.csv")
    predictions = predict_test(model, test_fe)

    section(7, "Measuring accuracy on the test set")
    test_accuracy = report_test_accuracy(predictions, reference)

    section(8, "Summary")
    results = [
        ("Training accuracy", train_accuracy),
        (f"Cross-validated accuracy ({CV_FOLDS} folds)", cv_accuracy),
    ]
    if test_accuracy is not None:
        results.append(("Test accuracy vs gender_submission.csv", test_accuracy))
    for label, value in results:
        print(f"{label:<44}{pct(value):>7}")


if __name__ == "__main__":
    main()
