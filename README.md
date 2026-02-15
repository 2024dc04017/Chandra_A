# Chandra_A
Machine Learning - Assignment 2
Name: Chandra A (BITS ID: 2024dc04017)

Step 1: Dataset choice
Customer Churn Prediction – IBM Telco Dataset

Dataset: [Telco Customer Churn (Kaggle)]
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) is chosen for this assignment.

Dataset Description
- Source: Kaggle – Telco Customer Churn  
- Rows: 7,043  
- Features: 19 (mixed numeric & categorical)  
- Target: Churn (Yes/No)  
- Notable columns: gender, SeniorCitizen, tenure, Contract, InternetService, PaymentMethod, MonthlyCharges, TotalCharges, etc.  
- Preprocessing: Missing values imputed; categorical features one-hot encoded; numeric features scaled; customerID dropped.

Problem Statement
The model predicts whether a telecom customer will churn (leave the service) using demographic, account, and service-usage features. The goal is to compare multiple ML algorithms and deploy an interactive Streamlit app for evaluation.

Step 2: Machine Learning Classification models and Evaluation metrics

Models Used & Evaluation Criteria

The following 6 models are trained on the selected dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbour (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Metrics captured for each model: Accuracy, AUC Score, Precision, Recall, F1 score, MCC score.

After running train_models.py, a consolidated table is exported to folder metrics/comparison_metrics.csv.

Comparison Table is also given below after training the model:

<img width="461" height="149" alt="image" src="https://github.com/user-attachments/assets/c52f5ae9-b358-460c-a8d2-23d4d9946a4c" />


Observations on Model Performance

-	Gradient Boosting and Logistic Regression emerged as the best-performing models, providing strong accuracy and AUC values, indicating excellent ability to distinguish churn vs. non churn customers. 
-	Gradient Boosting had the highest precision, making it ideal for minimizing false churn predictions. 
-	Naïve Bayes achieved the highest recall, meaning it identifies churners most effectively, but at the cost of many false alarms. 
-	KNN and Random Forest offered moderate, balanced performance but did not outperform the top models. 
-	Decision Tree performed the worst, showing classic overfitting behavior with low AUC and unstable metrics.


Step 3: GitHub Repository
<img width="611" height="223" alt="image" src="https://github.com/user-attachments/assets/e50adaba-93f6-4443-8bff-9d1d28466f04" />


Step 4: Create requirements.txt

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
