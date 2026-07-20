# Mental Health and Student Dropout Prediction

This project predicts student dropout risk at a university level using socioeconomic, demographic, and academic performance features. Two supervised classification models are developed and compared: a Logistic Regression baseline and an XGBoost classifier.

---

## Project Structure

```
├── data/
│   ├── student_dropout_dataset.csv   # Raw dataset
│   └── cleaned_df.csv                # Preprocessed dataset used for modelling
├── models/
│   ├── baseline_logistic_regression.pkl
│   ├── xgboost_model.pkl
│   ├── confusion_matrix_xgboost.png
│   ├── roc_curve_xgboost.png
│   ├── feature_importance_xgboost.png
│   └── precision_recall_xgboost.png
├── basline_model.ipynb               # Model 1 — Logistic Regression
├── xgboost_model.ipynb               # Model 2 — XGBoost
├── eda.ipynb                         # Exploratory Data Analysis
├── working.ipynb                     # Scratch / working notebook
├── requirements.txt
└── README.md
```

---

## Dataset

**File:** `data/cleaned_df.csv`  
**Target variable:** `Dropout` (binary — 0 = retained, 1 = dropped out)

### Input Features

| Feature                 | Type              | Description                                           |
| ----------------------- | ----------------- | ----------------------------------------------------- |
| `Age`                   | Numeric           | Student age in years                                  |
| `Gender`                | Numeric (encoded) | Student gender                                        |
| `Family_Income`         | Numeric           | Annual family income                                  |
| `Internet_Access`       | Numeric (binary)  | Whether the student has internet access at home       |
| `Study_Hours_per_Day`   | Numeric           | Average daily study hours                             |
| `Attendance_Rate`       | Numeric           | Percentage class attendance                           |
| `Assignment_Delay_Days` | Numeric           | Average number of days assignments are submitted late |
| `Travel_Time_Minutes`   | Numeric           | Daily commute time in minutes                         |
| `Part_Time_Job`         | Numeric (binary)  | Whether the student holds a part-time job             |
| `Scholarship`           | Numeric (binary)  | Whether the student is on a scholarship               |
| `Stress_Index`          | Numeric           | Self-reported stress score                            |
| `GPA`                   | Numeric           | Cumulative grade point average                        |
| `Semester_GPA`          | Numeric           | Current semester GPA                                  |
| `CGPA`                  | Numeric           | Cumulative GPA across all semesters                   |
| `Year_of_study`         | Numeric           | Current year of enrolment                             |
| `Department`            | Categorical       | Academic department (e.g., Engineering, Arts)         |
| `Parental_Education`    | Categorical       | Highest education level of parents                    |

**Train / Test split:** 80% training, 20% test, stratified by target, `random_state=42`.

---

## Preprocessing Pipeline

Both models share the same `sklearn` preprocessing pipeline applied inside a `ColumnTransformer`:

| Column type | Step 1                                    | Step 2                                   |
| ----------- | ----------------------------------------- | ---------------------------------------- |
| Numeric     | `SimpleImputer(strategy='median')`        | `StandardScaler()`                       |
| Categorical | `SimpleImputer(strategy='most_frequent')` | `OneHotEncoder(handle_unknown='ignore')` |

---

## Model 1 — Logistic Regression Baseline

**Notebook:** `basline_model.ipynb`  
**Saved model:** `models/baseline_logistic_regression.pkl`

### Development Steps

1. Install dependencies (`joblib`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`).
2. Import libraries and set `random_state = 42`.
3. Load `data/cleaned_df.csv` and inspect shape and target distribution.
4. Separate features (`X`) from target (`y = Dropout`).
5. Apply `train_test_split` (80/20, stratified).
6. Detect numeric and categorical feature columns automatically via `select_dtypes`.
7. Build the preprocessing `ColumnTransformer` (see table above).
8. Wrap preprocessor and classifier in a single `sklearn` `Pipeline`.
9. Fit the pipeline on the training set.
10. Evaluate on the test set and save the fitted pipeline with `joblib`.

### Model Parameters

| Parameter      | Value                |
| -------------- | -------------------- |
| Classifier     | `LogisticRegression` |
| `solver`       | `liblinear`          |
| `max_iter`     | `1000`               |
| `random_state` | `42`                 |

### Evaluation Metrics

| Metric           | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| Accuracy         | Overall proportion of correct predictions                     |
| Precision        | True positives / (true positives + false positives) per class |
| Recall           | True positives / (true positives + false negatives) per class |
| F1-score         | Harmonic mean of precision and recall per class               |
| Confusion Matrix | Heatmap of predicted vs actual labels                         |

### Plots

**Confusion Matrix** — seaborn heatmap visualising true positives, true negatives, false positives, and false negatives for both classes.

---

## Model 2 — XGBoost Classifier

**Notebook:** `xgboost_model.ipynb`  
**Saved model:** `models/xgboost_model.pkl`

### Development Steps

1. Install dependencies (adds `xgboost` to those used in Model 1).
2. Import libraries including `xgboost`, `LabelEncoder`, and extended `sklearn.metrics`.
3. Load `data/cleaned_df.csv` and inspect shape and target distribution.
4. Encode target with `LabelEncoder` if dtype is non-numeric.
5. Apply `train_test_split` (80/20, stratified, `random_state=42`).
6. Detect numeric and categorical feature columns automatically.
7. Build the same preprocessing `ColumnTransformer` as Model 1 (with `sparse_output=False` for compatibility).
8. Wrap preprocessor and `XGBClassifier` in a `Pipeline`.
9. Fit the pipeline on the training set.
10. Save the fitted pipeline to `models/xgboost_model.pkl` using `joblib`.
11. Generate predictions and predicted probabilities on the test set.
12. Compute and print all evaluation metrics.
13. Produce and save all evaluation plots.

### Model Parameters

| Parameter          | Value     | Description                                       |
| ------------------ | --------- | ------------------------------------------------- |
| `n_estimators`     | `300`     | Number of boosting rounds                         |
| `max_depth`        | `6`       | Maximum depth of each tree                        |
| `learning_rate`    | `0.05`    | Step size shrinkage to prevent overfitting        |
| `subsample`        | `0.8`     | Fraction of training samples used per tree        |
| `colsample_bytree` | `0.8`     | Fraction of features used per tree                |
| `eval_metric`      | `logloss` | Evaluation metric used internally during training |
| `n_jobs`           | `-1`      | Use all available CPU cores                       |
| `random_state`     | `42`      | Reproducibility seed                              |

### Evaluation Metrics

| Metric                 | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| Accuracy               | Overall proportion of correct predictions                        |
| Precision              | True positives / (true positives + false positives) per class    |
| Recall                 | True positives / (true positives + false negatives) per class    |
| F1-score               | Harmonic mean of precision and recall per class                  |
| AUC-ROC                | Area under the Receiver Operating Characteristic curve           |
| Average Precision (AP) | Area under the Precision-Recall curve; robust to class imbalance |

### Plots

| Plot                   | File                             | Description                                                          |
| ---------------------- | -------------------------------- | -------------------------------------------------------------------- |
| Confusion Matrix       | `confusion_matrix_xgboost.png`   | Heatmap of predicted vs actual labels                                |
| ROC / AUC Curve        | `roc_curve_xgboost.png`          | True positive rate vs false positive rate with AUC score annotated   |
| Feature Importance     | `feature_importance_xgboost.png` | Top 20 features ranked by XGBoost `feature_importances_`             |
| Precision-Recall Curve | `precision_recall_xgboost.png`   | Precision vs recall trade-off with average precision score annotated |

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`.

---

## How to Run

1. Run `eda.ipynb` to explore the raw data.
2. Run `basline_model.ipynb` top to bottom to train and save the Logistic Regression model.
3. Run `xgboost_model.ipynb` top to bottom to train, evaluate, and save the XGBoost model with all plots.
