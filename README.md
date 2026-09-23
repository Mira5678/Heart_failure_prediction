# Heart Failure Prediction Using Machine Learning

A machine learning project exploring the prediction of heart disease using patient clinical data. The project follows an end-to-end machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, and final model evaluation.

## Project Overview

Cardiovascular diseases (CVDs) are a major global health concern. Early identification of patients at higher risk of heart disease can support further medical assessment and intervention.

The aim of this project is to investigate whether machine learning classification algorithms can predict the presence of heart disease based on a patient's demographic, clinical, and exercise-related characteristics.

The project focuses on:

- Understanding and cleaning the dataset
- Exploring relationships between patient characteristics and heart disease
- Preparing numerical and categorical features for machine learning
- Comparing several classification algorithms
- Using cross-validation to compare model performance
- Performing hyperparameter tuning
- Evaluating the final model using multiple classification metrics
- Developing a simple Streamlit interface for interacting with the trained model

> **Note:** This project is intended for educational and portfolio purposes. The model is not a medical diagnostic tool and should not be used to make clinical decisions.

## Dataset

The dataset contains **918 patient observations** and **11 input features**, with `HeartDisease` as the target variable.

You can find the reference to the original dataset here: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction 

## Technologies & Libraries

The project was developed using Python and the following libraries:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

## Running the Project

### Install dependencies
```
pip install -r requirements.txt
```
### Run the notebook
Open the Jupyter Notebook and run all cells to reproduce the data analysis, preprocessing, model training, and evaluation.

### Run the Streamlit application
```
streamlit run app.py
```

## Machine Learning Models used in this project

Several classification algorithms were compared:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Gradient Boosting

These models were selected to compare different approaches to binary classification, including linear, tree-based, ensemble, distance-based, and kernel-based methods.

## Cross-Validation

Five-fold cross-validation was used to obtain a more reliable estimate of model performance on the training data.

The models were compared using their mean cross-validation accuracy and standard deviation.

| Model               | Mean CV Accuracy | Standard Deviation |
| ------------------- | ---------------: | -----------------: |
| Logistic Regression |           84.72% |              1.77% |
| Decision Tree       |           76.54% |              1.89% |
| SVM                 |           85.27% |              2.62% |
| KNN                 |           84.17% |              0.81% |
| Gradient Boosting   |           85.95% |              2.04% |


Gradient Boosting achieved the highest mean cross-validation accuracy, while KNN showed the lowest variation between folds.

The results were used to identify promising candidate models for further evaluation.

## Hyperparameter Tuning

Hyperparameter tuning was performed on the selected candidate model(s) using cross-validation.

Grid Search / Randomized Search was used to explore different combinations of model hyperparameters and identify configurations that improved or maintained predictive performance.

The tuned model was subsequently evaluated on the held-out test set.

## Model Evaluation

Model performance was evaluated using multiple metrics rather than accuracy alone.

### Evaluation metrics

#### Accuracy

The proportion of all predictions that were classified correctly.

#### Precision

The proportion of patients predicted to have heart disease who actually had heart disease.

#### Recall

The proportion of patients with heart disease that were correctly identified by the model.

#### F1-score

The harmonic mean of precision and recall.

#### Confusion Matrix

Used to examine:
- True positives
- True negatives
- False positives
- False negatives

Note:
Because this is a healthcare-related prediction problem, particular attention was given to **recall** for the heart disease class, since false-negative predictions represent patients with heart disease who were not identified by the model.

## Final Results

The final selected model was **Logistic Regression**.

| Metric                    | Test Result |
| ------------------------- | ----------: |
| Accuracy                  |         88% |
| Precision                 |         91% |
| Recall                    |         87% |
| F1-score                  |         89% |


The final model was selected based on its performance on the held-out test set as well as considerations such as recall and model interpretability.

## Limitations

There are several limitations to this project:
- A substantial number of cholesterol measurements were recorded as 0 and had to be treated as missing.
- The model has only been evaluated using this dataset and does not establish how well it would generalise to other populations.
- Model performance may differ when applied to patients from different demographic or clinical populations.
- The model should not be used as a substitute for professional medical assessment.

Further work could investigate external validation using an independent dataset and explore additional methods for handling missing data and model interpretability.

## Future Improvements

Potential extensions to this project include:

- Testing additional machine learning algorithms
- Comparing different missing-value imputation strategies
- Investigating class imbalance and alternative evaluation metrics
- Applying more extensive hyperparameter optimisation
- Adding model explainability using SHAP
- Improving the Streamlit interface
- Deploying the application
- Testing the model on an independent external dataset


