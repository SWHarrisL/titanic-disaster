# Titanic Survival Prediction

Predicts which passengers survived the Titanic disaster of April 15, 1912, using logistic regression. The same analysis is implemented twice, in Python and in R, and each version runs in its own Docker container, so nothing needs to be installed except Docker.

Each run loads the Kaggle data, explores it, engineers features, trains the model, predicts the test set and prints every step to the terminal.

## Contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Step 1: Clone the repository](#step-1-clone-the-repository)
- [Step 2: Download the data](#step-2-download-the-data)
- [Step 3: Run the Python model](#step-3-run-the-python-model)
- [Step 4: Run the R model](#step-4-run-the-r-model)
- [What the models do](#what-the-models-do)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Troubleshooting](#troubleshooting)

## Quick start

If Docker is already running and you have the three Kaggle CSV files:

```bash
git clone https://github.com/SWHarrisL/titanic-disaster.git
cd titanic-disaster
# copy train.csv, test.csv and gender_submission.csv into src/data/
docker build -t titanic-python -f src/python/Dockerfile .
docker run --rm titanic-python
docker build -t titanic-r -f src/r/Dockerfile .
docker run --rm titanic-r
```

The rest of this README walks through each step in detail.

## Prerequisites

| Tool | Used for | Download |
|---|---|---|
| Docker Desktop | Building and running both models | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Git | Cloning this repository | [git-scm.com](https://git-scm.com/downloads) |
| Kaggle account (free) | Downloading the data | [kaggle.com](https://www.kaggle.com) |

Python and R do not need to be installed: each Docker image includes its own.

Before you start, open Docker Desktop and wait until it says it is running. You can confirm with:

```bash
docker --version
```

The commands in this README work the same in a macOS or Linux terminal and in Windows PowerShell.

## Step 1: Clone the repository

```bash
git clone https://github.com/SWHarrisL/titanic-disaster.git
cd titanic-disaster
```

Run every remaining command from this `titanic-disaster` folder, the repository root.

## Step 2: Download the data

The data is not included in this repository, by design. Download it from Kaggle:

1. Sign in to Kaggle and open the competition's data page: <https://www.kaggle.com/competitions/titanic/data>
2. If Kaggle asks you to join the competition and accept its rules, do so. It requires this before allowing downloads.
3. Click **Download All**. This saves `titanic.zip`.
4. Unzip it (some browsers do this automatically) and move these three files into the repository's `src/data/` folder:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv`

The files must sit directly inside `src/data/`, not in a subfolder such as `src/data/titanic/`.

If you use the Kaggle command-line tool with your API credentials set up, you can download and unzip from the terminal instead:

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

On Windows, unzip the file with File Explorer or `Expand-Archive` instead of `unzip`.

Check that the files are in place:

```bash
ls src/data
```

You should see `gender_submission.csv`, `test.csv` and `train.csv`. The folder also contains a hidden `.gitkeep` placeholder, which you can ignore.

**Download the data before building.** The data is copied into each Docker image when the image is built. If you add or change the data afterwards, build the image again.

## Step 3: Run the Python model

```bash
docker build -t titanic-python -f src/python/Dockerfile .
docker run --rm titanic-python
```

- The `.` at the end of the build command is required. It tells Docker to build from the repository root, where both the code and the data are.
- The first build takes about a minute while Docker downloads Python 3.12 and installs the pinned packages. Later builds take seconds.
- `--rm` removes the container once it finishes.

The output starts with the Python and package versions, prints each step and ends with this summary:

```
[8] Summary
Training accuracy                             83.8%
Cross-validated accuracy (5 folds)            83.3%
Test accuracy vs gender_submission.csv        92.8%
```

## Step 4: Run the R model

```bash
docker build -t titanic-r -f src/r/Dockerfile .
docker run --rm titanic-r
```

The first build takes a minute or two while Docker downloads R 4.5.1 and installs the R packages. The output ends with:

```
[8] Summary
Training accuracy                             84.0%
Cross-validated accuracy (5 folds)            82.7%
Test accuracy vs gender_submission.csv        92.1%
```

## What the models do

Both scripts follow the same eight steps and print what each one does:

1. **Load** `train.csv`, `test.csv` and `gender_submission.csv`.
2. **Explore** the training data: survival rate, missing values, and survival by sex, class and port of embarkation.
3. **Add and adjust features**, applying the same steps to the training and test data:
   - `Title`, taken from the passenger's name (Mr, Mrs, Miss, Master or Rare)
   - `FamilySize`, the number of siblings, spouses, parents and children aboard, plus the passenger
   - `IsAlone`, for passengers travelling without family
   - `HasCabin`, for passengers with a recorded cabin number
   - `Name`, `Ticket`, `Cabin`, `SibSp` and `Parch` are then dropped
4. **Prepare the features**: fill missing `Age`, `Fare` and `Embarked` values, scale the numeric columns and encode the categories. Every value used here is learned from the training data only and then applied unchanged to the test data.
5. **Train** a logistic regression model and report its training accuracy and 5-fold cross-validated accuracy.
6. **Predict** survival for every passenger in `test.csv`.
7. **Measure test accuracy** against `gender_submission.csv` (see the note under [Results](#results)).
8. **Summarize** the results.

The two versions differ in a few deliberate ways:

| | Python | R |
|---|---|---|
| Script | `src/python/titanic_model.py` | `src/r/titanic_model.R` |
| Model | scikit-learn `LogisticRegression` (regularized) | base R `glm` with a binomial family (unregularized) |
| Features | Age, Fare, FamilySize, IsAlone, HasCabin, Sex, Pclass, Embarked, Title | The same, except Sex |
| Package versions | Pinned in `src/python/requirements.txt` | Pinned by the `rocker/r-ver:4.5.1` base image, which installs packages from a CRAN snapshot frozen on a fixed date |

The R model leaves out `Sex` because `Title` already encodes it: Mr and Master are male, Miss and Mrs are female. With both included, an unregularized model becomes unstable. scikit-learn regularizes by default, which handles the overlap, so the Python model keeps both.

## Results

| Metric | Python | R |
|---|---|---|
| Training accuracy | 83.8% | 84.0% |
| Cross-validated accuracy (5 folds) | 83.3% | 82.7% |
| Test accuracy vs `gender_submission.csv` | 92.8% | 92.1% |

The two independent implementations reach almost identical results.

Notable findings from the data:

- Women survived at 74.2%, men at 18.9%.
- First-class passengers survived at 63.0%, third-class passengers at 24.2%.
- Boys (title Master) survived at 57.5%, compared with 15.7% for adult men (title Mr).
- Passengers with a recorded cabin survived at 66.7%, compared with 30.0% for those without.
- Passengers travelling with family survived at 50.6%, compared with 30.4% for those travelling alone.

**A note on test accuracy.** Kaggle keeps the true outcomes for `test.csv` private, so the file has no `Survived` column. The only reference Kaggle provides is `gender_submission.csv`, an example submission that predicts every woman survived and every man died. The test figure therefore measures agreement with that simple baseline, not true accuracy. The cross-validated accuracy is the better estimate of how the models perform on passengers they have not seen.

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS               makes @SWHarrisL the owner of all files
├── src/
│   ├── data/                    the Kaggle CSV files go here (not tracked by Git)
│   ├── python/
│   │   ├── Dockerfile           Python 3.12 image
│   │   ├── requirements.txt     pinned Python packages
│   │   └── titanic_model.py     Python analysis and model
│   └── r/
│       ├── Dockerfile           R 4.5.1 image
│       ├── install_packages.R   installs the R packages
│       └── titanic_model.R      R analysis and model
├── .gitignore                   keeps data, OS files and environments out of Git
└── README.md
```

Development follows a simple branch flow: work happens on a feature branch, is merged into `develop` through a pull request, and `develop` is merged into `main` once it is complete.

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` or `Cannot connect to the Docker daemon` | Install Docker Desktop, open it and wait until it says it is running. Then open a new terminal. |
| `'docker buildx build' requires 1 argument` | The build command is missing the `.` at the end. |
| `ERROR: src/data/train.csv was not found` | Download the data into `src/data/` (Step 2), then build the image again. |
| An error saying a Dockerfile or file was `not found` during the build | Run the commands from the repository root (`titanic-disaster`), not from inside `src/` or another folder. |

To remove the Docker images when you are finished:

```bash
docker rmi titanic-python titanic-r
```
