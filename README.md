# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit. The assignment also includes deploying the trained models using a Streamlit web application for interactive evaluation.

---

## b. Dataset Description

The Bank Marketing dataset is used for this assignment. The dataset contains information related to bank customers and their responses to previous marketing campaigns. The goal is to predict whether a customer will subscribe to a term deposit.

- **Dataset Source:** Public Bank Marketing Dataset  
- **Total number of instances:** 11,162  
- **Total number of features:** 16 input features and 1 target variable  
- **Type of problem:** Binary Classification  
- **Target Variable:** `deposit` (yes/no)

The dataset consists of both numerical and categorical attributes, including age, job, marital status, education, account balance, housing loan, personal loan, contact type, campaign details, and previous campaign outcomes.

---

## c. Data Preprocessing

The following preprocessing steps were performed before model training:

- Categorical features were converted into a numerical format using **Label Encoding**
- The target variable `deposit` was converted to binary values (yes → 1, no → 0)
- The dataset was split into **training (80%)** and **testing (20%)** sets
- Feature scaling was applied using **StandardScaler** to improve the performance of distance-based models such as KNN and Logistic Regression

---

## d. Models Implemented

The following six machine learning classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest Classifier (Ensemble Model)  
6. XGBoost Classifier (Ensemble Model)  

All trained models were saved using **Joblib** for reuse in the Streamlit web application.

---

## e. Evaluation Metrics

Each model was evaluated using the following performance metrics:

- Accuracy  
- Area Under ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

These metrics provide a comprehensive understanding of model performance, especially for binary classification problems.

---

## f. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.7900 | 0.7886 | 0.7931 | 0.7582 | 0.7753 | 0.5788 |
| Decision Tree | 0.7631 | 0.7621 | 0.7582 | 0.7404 | 0.7492 | 0.5249 |
| KNN | 0.7734 | 0.7711 | 0.7877 | 0.7198 | 0.7522 | 0.5461 |
| Naive Bayes | 0.7465 | 0.7492 | 0.7042 | 0.8097 | 0.7533 | 0.5004 |
| Random Forest (Ensemble) | 0.8334 | 0.8343 | 0.8083 | 0.8538 | 0.8304 | 0.6679 |
| XGBoost (Ensemble) | 0.8424 | 0.8432 | 0.8178 | 0.8622 | 0.8394 | 0.6858 |

---

## g. Observations

| Model | Observation |
|------|-------------|
| Logistic Regression | Provided a strong baseline but could not capture complex non-linear patterns. |
| Decision Tree | Captured non-linear relationships but showed signs of overfitting. |
| KNN | Performed reasonably well but was sensitive to feature scaling and choice of neighbors. |
| Naive Bayes | Achieved good recall but lower precision due to independence assumptions. |
| Random Forest | Improved performance by reducing overfitting through ensemble learning. |
| XGBoost | Achieved the best overall performance by iteratively correcting model errors. |




---

## h. Streamlit Web Application

A Streamlit-based web application was developed and deployed to demonstrate the trained models.

### Features of the Application:
- Upload test dataset in CSV format  
- Select a trained machine learning model  
- Display evaluation metrics  
- View confusion matrix and classification report  

### Streamlit App URL: https://mlassignment2-kwrwgr9j57n9wvpftwrq4n.streamlit.app/



---

## i Repository Structure

ML_Assignment_2/
│
├── app.py
├── model_training.ipynb
├── README.md
├── requirements.txt
├── bank.csv
├── test_data.csv
├── bits_virtual_lab_execution.png
│
└── model/
    ├── model_lr.pkl
    ├── model_dt.pkl
    ├── model_knn.pkl
    ├── model_nb.pkl
    ├── model_rf.pkl
    └── model_xgb.pkl


## j. Execution Proof


---
<img width="959" height="445" alt="bits_virtual_lab_execution" src="https://github.com/user-attachments/assets/f2cee151-ca44-48f6-9430-b5450b18fb21" />
<img width="1342" height="950" alt="image" src="https://github.com/user-attachments/assets/e93e4d12-b318-4530-b7f3-06b515edf632" />
<img width="1227" height="701" alt="image" src="https://github.com/user-attachments/assets/3159accf-5db9-466b-b662-593bd2d53c58" />




---

## k. Conclusion

This assignment demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, comparison, and deployment. Ensemble models such as Random Forest and XGBoost outperformed individual classifiers, highlighting the effectiveness of ensemble learning techniques in improving predictive performance for real-world banking datasets.
