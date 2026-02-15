"""
Train 5 classification models on the IBM Telco Customer Churn dataset and save:
- Trained pipelines (*.joblib) into ./model/
- A comparison metrics CSV into ./metrics/comparison_metrics.csv
- A test sample into ./data/test_sample.csv
- python train_models.py --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import joblib
import seaborn as sns
import matplotlib.pyplot as plt

ART_DIR = Path('artifacts/confusion_matrices')
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path('model')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path('metrics')
METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['Churn'])
    return df


def prepare_features(df: pd.DataFrame):
    y = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    X = df.drop(columns=['Churn'])
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])

    categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return X, y, preprocessor


def evaluate(y_true, y_prob, y_pred):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': auc,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }


def plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
    ax.set_title(f'Confusion Matrix – {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    out = ART_DIR / f'{model_name}_confusion.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(args):
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found at {data_path}. Place the Telco CSV there or pass --data_path.")

    df = load_data(data_path)
    X, y, preprocessor = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    test_out = X_test.copy(); test_out['Churn'] = y_test.values
    test_out.to_csv(DATA_DIR / 'test_sample.csv', index=False)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=15),
        'GaussianNB': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    rows = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
        pipe.fit(X_train, y_train)

        if hasattr(pipe.named_steps['model'], 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['model'], 'decision_function'):
            scores = pipe.decision_function(X_test)
            s_min, s_max = scores.min(), scores.max()
            y_prob = (scores - s_min) / (s_max - s_min + 1e-9)
        else:
            y_prob = np.zeros_like(y_test, dtype=float)
        y_pred = pipe.predict(X_test)

        m = evaluate(y_test, y_prob, y_pred)
        m_row = {'ML Model Name': name}; m_row.update(m); rows.append(m_row)
        plot_confusion(y_test, y_pred, name)
        joblib.dump(pipe, MODEL_DIR / f'{name}.joblib')
        rpt = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
        (METRICS_DIR / f'{name}_classification_report.txt').write_text(rpt)

    import pandas as pd
    metrics_df = pd.DataFrame(rows, columns=['ML Model Name','Accuracy','AUC','Precision','Recall','F1','MCC'])
    metrics_df.to_csv(METRICS_DIR / 'comparison_metrics.csv', index=False)
    print('\nTraining complete. Files saved to ./model and ./metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    args = parser.parse_args()
    main(args)
