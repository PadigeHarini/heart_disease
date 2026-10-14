# Heart Disease Prediction Using Machine Learning

## Project Overview

This project applies machine learning classification algorithms to predict the presence of heart disease using clinical and demographic features.

The project demonstrates an end-to-end machine learning workflow, including data cleaning, feature preprocessing, model comparison, stratified cross-validation, hyperparameter tuning, ensemble learning, and performance evaluation.

Multiple classification algorithms are compared to understand their performance on a held-out test dataset.

**Disclaimer:** This project is intended for educational purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.

## Objectives

- Clean the dataset and remove duplicate records.
- Preprocess numerical and categorical features.
- Build and compare multiple machine learning classifiers.
- Apply stratified train-test splitting and cross-validation.
- Optimize model hyperparameters using GridSearchCV.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Compare individual classifiers with voting and stacking ensembles.
- Identify the best-performing model in the recorded experiment.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook
- **Optional Library:** XGBoost

## Dataset

The project uses a heart disease dataset stored in `heart.csv`. Its 14 columns correspond to 13 predictor features and one target variable.

### Dataset Summary

| Property | Details |
|---|---|
| Original records | 303 |
| Duplicate records removed | 1 |
| Records after cleaning | 302 |
| Predictor features | 13 |
| Target variable | `target` |
| Problem type | Binary classification |

### Features

The dataset contains the following clinical and demographic attributes:

- `age` — Age in years
- `sex` — Sex
- `chest_pain_type` — Chest pain category
- `resting_blood_pressure` — Resting blood pressure
- `cholesterol` — Serum cholesterol level
- `fasting_blood_sugar` — Fasting blood sugar indicator
- `rest_ecg` — Resting electrocardiographic results
- `max_heart_rate_achieved` — Maximum heart rate achieved
- `exercise_induced_angina` — Exercise-induced angina indicator
- `st_depression` — ST depression induced by exercise relative to rest
- `st_slope` — Slope of the peak exercise ST segment
- `num_major_vessels` — Number of major vessels visualized by fluoroscopy
- `thalassemia` — Thalassemia category

The `target` variable represents the binary outcome:

- `1` — Heart disease present
- `0` — Heart disease absent

These labels follow the coding used in the dataset supplied to the notebook.

### Target Class Distribution

After duplicate removal, the target distribution is:

| Class | Records | Percentage |
|---|---:|---:|
| Heart disease present (`1`) | 164 | 54.30% |
| Heart disease absent (`0`) | 138 | 45.70% |
| **Total** | **302** | **100%** |

The dataset is reasonably balanced, with no severe imbalance between the two classes. Therefore, oversampling or undersampling was not required solely to balance the target classes.

### Dataset Source

The dataset structure is consistent with the commonly used processed Cleveland Heart Disease dataset. The official UCI Machine Learning Repository provides documentation and access to the original Heart Disease dataset:

[UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

The exact source of the local `heart.csv` file should be verified before publication. Dataset redistribution and licensing conditions should be checked before including the CSV in a public repository.

## Methodology

### 1. Data Cleaning

- Loaded the dataset using Pandas.
- Identified and removed one duplicate record.
- Renamed the columns for readability.
- Separated the predictor variables from the target variable.

### 2. Data Preprocessing

The notebook uses Scikit-learn preprocessing pipelines to prepare the data for model training.

- **StandardScaler:** Scales numerical features.
- **OneHotEncoder:** Encodes categorical features.
- **ColumnTransformer:** Applies the appropriate transformations to different feature groups.
- **Pipeline:** Combines preprocessing and model training into a single workflow.

Numerical features include age, resting blood pressure, cholesterol, maximum heart rate, and ST depression.

Categorical features include sex, chest pain type, fasting blood sugar, resting ECG results, exercise-induced angina, ST slope, number of major vessels, and thalassemia.

Placing preprocessing inside the pipeline helps prevent data leakage during cross-validation because transformations are fitted within the relevant training folds.

### 3. Train-Test Split

The dataset was divided using a stratified train-test split.

| Subset | Records | Percentage |
|---|---:|---:|
| Training set | 241 | 80% |
| Test set | 61 | 20% |
| Total | 302 | 100% |

The split uses `random_state=42` and `stratify=y` to preserve the approximate class proportions in the training and test sets.

### 4. Machine Learning Models

The recorded notebook execution evaluated seven individual classifiers:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. Support Vector Machine (SVM)

It also evaluated two ensemble classifiers:

- Voting Classifier
- Stacking Classifier

**XGBoost was not included in the recorded execution** because the package was not installed. The notebook contains conditional code to include XGBoost when the required library is available.

### 5. Cross-Validation and Hyperparameter Tuning

The following techniques were used:

- Stratified 10-fold cross-validation on the training set.
- GridSearchCV for hyperparameter optimization.
- Accuracy as the model-selection scoring metric during grid search.
- Selection of the best estimator for each individual model.
- Evaluation of the tuned estimators on the held-out test set.

Hyperparameter tuning was performed on the training data using cross-validation.

### 6. Model Evaluation

Model performance was evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that were actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

The notebook also generates a model accuracy comparison plot, a confusion matrix for the selected best-performing model, and a classification report.

## Results

### Cross-Validation Results

The following table reports the best cross-validation accuracy recorded after hyperparameter tuning.

| Model | Best CV Accuracy |
|---|---:|
| Gaussian Naive Bayes | 84.65% |
| Support Vector Machine | 83.80% |
| Logistic Regression | 82.98% |
| Decision Tree | 82.57% |
| Random Forest | 82.15% |
| K-Nearest Neighbors | 81.73% |
| Gradient Boosting | 81.72% |

Gaussian Naive Bayes achieved the highest recorded cross-validation accuracy of **84.65%**.

### Held-Out Test Results

The following metrics were recorded for the tuned individual classifiers and ensemble models.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 83.61% | 82.86% | 87.88% | 85.29% |
| Gaussian Naive Bayes | 80.33% | 81.82% | 81.82% | 81.82% |
| SVM | 78.69% | 76.32% | 87.88% | 81.69% |
| Voting Ensemble | 78.69% | 77.78% | 84.85% | 81.16% |
| Stacking Ensemble | 78.69% | 77.78% | 84.85% | 81.16% |
| KNN | 75.41% | 76.47% | 78.79% | 77.61% |
| Gradient Boosting | 75.41% | 75.00% | 81.82% | 78.26% |
| Random Forest | 73.77% | 72.97% | 81.82% | 77.14% |
| Decision Tree | 65.57% | 65.00% | 78.79% | 71.23% |

*Note: Results are from the notebook's saved execution and may vary with different data splits or software versions.*

### Best-Performing Model

Logistic Regression achieved the highest recorded test accuracy among the evaluated models.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 83.61% |
| Precision | 82.86% |
| Recall | 87.88% |
| F1-score | 85.29% |

Gaussian Naive Bayes achieved the highest cross-validation accuracy, while Logistic Regression achieved the highest accuracy on the test split.

These findings illustrate that the model with the best cross-validation score does not necessarily achieve the best score on a particular test split.

### Interpretation

Logistic Regression performed well on this dataset, achieving an accuracy of 83.61% and a recall of 87.88% for the positive class.

The ensemble models did not outperform Logistic Regression on the recorded test split. This highlights the importance of comparing models empirically instead of assuming that more complex algorithms will always perform better.

The test set contains only 61 records, so the reported metrics are subject to sampling variability.

**Methodological limitation:** Although hyperparameter tuning was performed using cross-validation on the training set, the notebook compares multiple models using the same test set and identifies the winner from those results. Consequently, the test set is not a completely untouched final evaluation for the selected winner. An independent test set or nested cross-validation would provide a more rigorous estimate of generalization performance.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook, or use an environment such as Anaconda.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your actual repository URL and directory name.

### 2. Install Dependencies

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

XGBoost is optional. To enable the corresponding model in the notebook, install it separately:

```bash
pip install xgboost
```

### 3. Add the Dataset

Place `heart.csv` in the same directory as `heart_disease_final.ipynb`.

Make sure the dataset is obtained from a legitimate source and that its license permits any intended redistribution.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `heart_disease_final.ipynb` and run the cells in order.

## Project Structure

```text
heart-disease-prediction/
│
├── heart_disease_final.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── heart.csv  # Include only if redistribution is permitted
```

## Key Learnings

- Data cleaning and duplicate detection.
- Numerical scaling and categorical feature encoding.
- Building Scikit-learn preprocessing pipelines.
- Stratified train-test splitting.
- Cross-validation and hyperparameter optimization.
- Classification metrics and confusion matrix interpretation.
- Comparison of individual classifiers and ensemble learning.
- Understanding the difference between cross-validation and test-set performance.
- Recognizing limitations when working with small datasets.

## Limitations

- The dataset contains only 302 records after duplicate removal.
- Results are based on a single train-test split.
- Comparing several models on the same test set can introduce selection bias.
- Performance may vary across different datasets and patient populations.
- High accuracy on this dataset does not establish clinical validity.
- The model has not been validated for medical diagnosis or clinical use.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for a heart disease classification task. Seven individual classifiers and two ensemble classifiers were compared using preprocessing pipelines, stratified cross-validation, hyperparameter tuning, and held-out test evaluation.

Gaussian Naive Bayes achieved the highest recorded cross-validation accuracy of 84.65%, while Logistic Regression achieved the highest recorded test accuracy of 83.61%.

The project highlights the importance of data preprocessing, systematic model comparison, appropriate evaluation metrics, and careful interpretation of results. Further validation on independent data would be required before drawing conclusions about performance beyond this dataset.

## License

The project code may be distributed under the MIT License, provided you have the necessary rights to license the material. Third-party datasets and other included materials remain subject to their own licensing terms.
