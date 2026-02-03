# Machine Learning Assignment 2


---

## a. Problem Statement

The objective of this assignment is to build, compare, and evaluate multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit. The project also involves deploying the trained models using a Streamlit web application to demonstrate practical model usage.

---

## b. Dataset Description

The Bank Marketing dataset is used for this assignment. It contains information related to bank customers and their responses to previous marketing campaigns. The goal is to predict whether a customer will subscribe to a term deposit based on demographic and campaign-related attributes.

- **Dataset Source:** Public Bank Marketing Dataset  
- **Total number of instances:** 11,162  
- **Total number of features:** 16 input features and 1 target variable  
- **Type of problem:** Binary Classification  
- **Target Variable:** `deposit` (yes / no)  

The dataset consists of both numerical and categorical features such as age, job, marital status, education, account balance, loan information, contact type, campaign duration, and previous campaign outcomes.

---

## c. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented and evaluated using the same dataset and train–test split:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following performance metrics:

- Accuracy  
- ROC-AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

These metrics provide a comprehensive evaluation of classification performance, especially for datasets with class imbalance.

---

## d. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.7900 | 0.7886 | 0.7931 | 0.7582 | 0.7753 | 0.5788 |
| Decision Tree | 0.7631 | 0.7621 | 0.7582 | 0.7404 | 0.7492 | 0.5249 |
| KNN | 0.7734 | 0.7711 | 0.7877 | 0.7198 | 0.7522 | 0.5461 |
| Naive Bayes | 0.7465 | 0.7492 | 0.7042 | 0.8097 | 0.7533 | 0.5004 |
| Random Forest (Ensemble) | 0.8334 | 0.8343 | 0.8083 | 0.8538 | 0.8304 | 0.6679 |
| XGBoost (Ensemble) | 0.8424 | 0.8432 | 0.8178 | 0.8622 | 0.8394 | 0.6858 |

---

## e. Observations on Model Performance

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Provided a strong baseline performance but was limited in capturing complex, non-linear patterns. |
| Decision Tree | Captured non-linear relationships but showed signs of overfitting, leading to slightly reduced performance. |
| KNN | Performed reasonably well; however, results were sensitive to feature scaling and the choice of neighbors. |
| Naive Bayes | Achieved high recall, indicating good identification of positive cases, but precision was affected due to strong independence assumptions. |
| Random Forest (Ensemble) | Delivered significantly improved performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieved the best overall results by sequentially correcting errors using boosting techniques. |

---

## f. Streamlit Web Application

A Streamlit-based interactive web application was developed and deployed using Streamlit Community Cloud. The application provides the following features:

- Uploading a test dataset in CSV format  
- Selecting a trained machine learning model  
- Viewing evaluation metrics  
- Visualizing confusion matrices and classification reports  

This application demonstrates real-world deployment of machine learning models in an interactive environment.

---

## g. Repository Structure

## g. Repository Structure

ML_Assignment_2/
├── model_training.ipynb
├── app.py
├── requirements.txt
├── README.md
├── bank.csv
├── test_data.csv
└── model/
    ├── model_lr.pkl
    ├── model_dt.pkl
    ├── model_knn.pkl
    ├── model_nb.pkl
    ├── model_rf.pkl
    └── model_xgb.pkl



---

## h. Conclusion

This assignment demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, comparison, and deployment. Ensemble models such as Random Forest and XGBoost outperformed individual classifiers, highlighting the effectiveness of ensemble learning techniques in improving predictive performance for real-world banking datasets.
