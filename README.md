# Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict the presence of heart disease based on clinical and demographic features.

The project focuses on data preprocessing, feature encoding, model comparison, cross-validation, hyperparameter tuning, and performance evaluation. Multiple classification algorithms are trained and evaluated to identify the model that performs best on the held-out test dataset.

This project demonstrates an end-to-end machine learning workflow using Python and Scikit-learn.

**Note:** This project is intended for educational purposes only and is not a medical diagnostic tool.

## Objectives

- Clean the dataset and remove duplicate records.
- Preprocess numerical and categorical features.
- Build and compare multiple classification models.
- Apply stratified train-test splitting and cross-validation.
- Optimize model hyperparameters using GridSearchCV.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Compare individual classifiers with voting and stacking ensembles.
- Identify the best-performing model based on test-set performance.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook
- **Optional:** XGBoost

## Dataset

The project uses a heart disease dataset stored in `heart.csv`.

- Original records: 303
- Duplicate records removed: 1
- Records after duplicate removal: 302
- Number of features: 13
- Target variable: `target`

The dataset contains features related to patient demographics and clinical measurements, including:

- Age and sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression and ST slope
- Number of major vessels
- Thalassemia

The target variable indicates the presence (`1`) or absence (`0`) of heart disease according to the dataset's coding.

**Dataset source:** Add the original dataset URL here before publishing. Check the dataset's license and redistribution conditions before uploading `heart.csv` to a public repository.

## Methodology

### 1. Data Cleaning

- Loaded the dataset using Pandas.
- Identified and removed duplicate records.
- Renamed columns for readability and consistency.
- Separated predictor variables from the target variable.

### 2. Data Preprocessing

- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.
- Used `ColumnTransformer` to handle numerical and categorical columns separately.
- Combined preprocessing and classification models using Scikit-learn pipelines.

Keeping preprocessing inside the pipeline helps prevent information leakage from validation and test data during model training and selection.

### 3. Train-Test Split

The dataset was divided using a stratified train-test split.

- Training set: 241 records (80%)
- Test set: 61 records (20%)
- Random state: 42

Stratification preserves approximately the same target-class proportions in both sets.

### 4. Machine Learning Models

The saved notebook executed and evaluated the following seven individual classifiers:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. Support Vector Machine (SVM)

Two ensemble classifiers were also evaluated:

- Voting Classifier
- Stacking Classifier

XGBoost is included conditionally in the code but was not available in the recorded execution because the package was not installed.

### 5. Cross-Validation and Hyperparameter Tuning

- Used stratified 10-fold cross-validation on the training set.
- Applied `GridSearchCV` to search for suitable model hyperparameters.
- Compared models using cross-validation accuracy.
- Selected the best estimator for each model based on cross-validation.
- Evaluated the tuned models on the held-out test set.

The test set was not used to select the random seed or tune the individual models.

### 6. Model Evaluation

The following evaluation metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score

The notebook also generates a model accuracy comparison plot, a confusion matrix for the selected best model, and a classification report.

## Results

### Cross-Validation Results

The following are the recorded best cross-validation accuracies after hyperparameter tuning.

| Model | Best CV Accuracy |
|---|---:|
| Gaussian Naive Bayes | 84.65% |
| Support Vector Machine | 83.80% |
| Logistic Regression | 82.98% |
| Decision Tree | 82.57% |
| Random Forest | 82.15% |
| K-Nearest Neighbors | 81.73% |
| Gradient Boosting | 81.72% |

**Gaussian Naive Bayes** achieved the highest recorded cross-validation accuracy of 84.65%.

### Held-Out Test Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 83.61% | 82.86% | 87.88% | 85.29% |
| Gaussian Naive Bayes | 80.33% | 81.82% | 81.82% | 81.82% |
| SVM | 78.69% | 76.32% | 87.88% | 81.69% |
| KNN | 75.41% | 76.47% | 78.79% | 77.61% |
| Gradient Boosting | 75.41% | 75.00% | 81.82% | 78.26% |
| Random Forest | 73.77% | 72.97% | 81.82% | 77.14% |
| Decision Tree | 65.57% | 65.00% | 78.79% | 71.23% |
| Voting Ensemble | 78.69% | 77.78% | 84.85% | 81.16% |
| Stacking Ensemble | 78.69% | 77.78% | 84.85% | 81.16% |

### Best Performing Model

**Logistic Regression** achieved the highest test accuracy in the recorded experiment.

- Accuracy: 83.61%
- Precision: 82.86%
- Recall: 87.88%
- F1-score: 85.29%

Although Gaussian Naive Bayes performed best during cross-validation, Logistic Regression achieved the highest accuracy on the held-out test set. This illustrates why cross-validation performance and final test performance should be reported separately.

The test set contained 61 records, so the reported metrics may vary with different data samples. These results should not be interpreted as evidence of clinical reliability.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook, or use Anaconda.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

XGBoost is optional. To enable it in the notebook, install it separately:

```bash
pip install xgboost
```

### 3. Add the dataset

Place `heart.csv` in the same directory as the notebook. Make sure you have permission to use the dataset and redistribute it if you intend to include it in the public repository.

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
└── heart.csv
```

Include `heart.csv` only if its license permits redistribution. Otherwise, provide the dataset source and instructions for obtaining it.

## Key Learnings

- Data cleaning and duplicate detection
- Numerical scaling and categorical feature encoding
- Building Scikit-learn preprocessing pipelines
- Stratified train-test splitting
- Cross-validation and hyperparameter tuning
- Classification metrics and confusion matrix interpretation
- Comparing individual models and ensemble methods
- Evaluating model generalization using a held-out test set

## Limitations

- The dataset contains only 302 records after duplicate removal.
- Results are based on a single train-test split.
- Performance may vary on different datasets and patient populations.
- High accuracy on this dataset does not establish clinical validity.
- The model has not been validated for medical diagnosis.

## Conclusion

This project demonstrates a complete machine learning classification workflow for a heart disease prediction task. Seven individual classifiers and two ensemble classifiers were compared using cross-validation, hyperparameter tuning, and held-out test evaluation.

Gaussian Naive Bayes achieved the highest cross-validation accuracy, while Logistic Regression achieved the highest test accuracy of 83.61% in the recorded experiment. The results highlight the importance of comparing multiple algorithms and evaluating their performance on data that was not used for model selection.

## License

The project code is intended to be shared under the MIT License, provided you have the necessary rights to license it. Third-party datasets and other included materials remain subject to their own licensing terms.

