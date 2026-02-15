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


ML Model Name	      Accuracy	    AUC	    Precision	    Recall	    F1	    MCC
LogisticRegression	0.806	        0.842	  0.657	        0.559	      0.604	  0.479
DecisionTree	      0.729	        0.657	  0.490	        0.505	      0.497	  0.312
KNN	                0.778	        0.822	  0.584	        0.567	      0.575	  0.425
GaussianNB	        0.695	        0.807	  0.459	        0.837	      0.593	  0.424
RandomForest	      0.779	        0.817	  0.606	        0.473	      0.532	  0.395
GradientBoosting	  0.806	        0.843	  0.674	        0.524	      0.589	  0.472


Observations on Model Performance

-	Gradient Boosting and Logistic Regression emerged as the best-performing models, providing strong accuracy and AUC values, indicating excellent ability to distinguish churn vs. non churn customers. 
-	Gradient Boosting had the highest precision, making it ideal for minimizing false churn predictions. 
-	Naïve Bayes achieved the highest recall, meaning it identifies churners most effectively, but at the cost of many false alarms. 
-	KNN and Random Forest offered moderate, balanced performance but did not outperform the top models. 
-	Decision Tree performed the worst, showing classic overfitting behavior with low AUC and unstable metrics.


Step 3: GitHub Repository

project-folder/
│-- app.py               # Streamlit app (uploading test CSV, pick model, see metrics & confusion matrix)
│-- train_models.py      # Trained 6 models, save pipelines and metrics
│-- requirements.txt     # Dependencies for local run and Streamlit Cloud
│-- README.md            # This file
│-- model/               # Saved pipelines (*.joblib) – created after training
│-- metrics/             # comparison_metrics.csv + per-model reports
│-- artifacts/
│   └─ confusion_matrices/ # PNGs saved during training
└-- data/
    └─ test_sample.csv     # Generated test split for quick app demo


Step 4: Create requirements.txt

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
