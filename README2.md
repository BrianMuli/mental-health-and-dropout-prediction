# Data Cleaning & Feature Engineering

## Overview

This notebook is responsible for transforming the raw student dropout dataset into a machine learning-ready dataset. The preprocessing pipeline improves data quality, creates meaningful features, and prepares the data for predictive modeling.

The output of this notebook is a cleaned and feature-engineered dataset (`cleaned_df.csv`) that will be used by all subsequent modeling notebooks.

---

## Objectives

The main objectives of this notebook are to:

- Clean and preprocess the raw dataset.
- Handle missing values.
- Remove unnecessary variables.
- Engineer informative features that improve predictive performance.
- Encode categorical variables into numerical representations.
- Reduce multicollinearity by removing highly correlated features.
- Export the processed dataset for model training.

---

## Data Cleaning

The following preprocessing steps are performed:

### 1. Remove Non-Predictive Features

- Drop the `Student_ID` column since it uniquely identifies students and does not contribute to prediction.

### 2. Handle Missing Values

Numerical variables are imputed using the median to reduce the influence of outliers.

Categorical variables are imputed using the mode (most frequent value).

Examples include:

- Family Income
- Study Hours per Day
- Parental Education
- Stress Index (when applicable)

---

## Feature Engineering

Feature engineering creates new variables that better represent student behavior and academic performance.

The engineered features include:

| Feature | Description |
|----------|-------------|
| Academic_Load | Combines study hours and assignment delays to estimate workload. |
| GPA_Trend | Difference between Semester GPA and GPA, indicating academic improvement or decline. |
| Academic_Consistency | Absolute difference between Semester GPA and CGPA. Lower values indicate more consistent performance. |
| Delay_per_Hour | Assignment delays relative to study time. |
| Financial_Risk | Binary indicator derived from family income. |
| Long_Commute | Indicates whether a student has a long travel time to campus. |
| Engagement | Combines attendance and study hours to estimate academic engagement. |
| Working_Student | Indicates students balancing employment and academics. |

These engineered variables provide additional information that may not be directly captured by the original dataset.

---

## Encoding

Machine learning algorithms require numerical input.

Categorical variables are converted into numerical representations using One-Hot Encoding.

The following variables are encoded:

- Gender
- Internet Access
- Part-Time Job
- Scholarship
- Department
- Parental Education

The first category is dropped to avoid multicollinearity (Dummy Variable Trap).

---

## Feature Selection

A correlation analysis is performed to identify highly correlated variables.

Features with an absolute correlation greater than **0.90** are reviewed, and redundant variables are removed to reduce multicollinearity while preserving meaningful information.

Domain knowledge is also considered before removing features to ensure important academic indicators are retained.

---

## Output

The final processed dataset is exported as:

```python
data/cleaned_df.csv
```

This dataset will be used in:

- Baseline Logistic Regression
- XGBoost
- Catboost
- Model Evaluation

---

## Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
      │
      ▼
Correlation Analysis
      │
      ▼
Feature Selection
      │
      ▼
cleaned_df.csv
      │
      ▼
Machine Learning Models
```

---

## Expected Outcome

At the end of this notebook, a clean, consistent, and machine learning-ready dataset is produced. This ensures that all predictive models are trained using the same standardized preprocessing pipeline, improving reproducibility and model reliability.