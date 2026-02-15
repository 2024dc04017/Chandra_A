import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Customer Churn – ML Models', page_icon='📊', layout='wide')
st.title('📊 Customer Churn – Interactive ML Demo (Telco)')
st.caption('Upload a **test CSV** (with the original Telco schema). Choose a model to evaluate or predict.')

MODEL_DIR = Path('model')
DEFAULT_TEST = Path('data/test_sample.csv')

@st.cache_resource
def list_models():
    return sorted([p for p in MODEL_DIR.glob('*.joblib')])

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def _calc_metrics(y_true, y_prob, y_pred):
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


left, right = st.columns([1,1])

with left:
    st.subheader('1) Choose a Model')
    model_files = list_models()
    if not model_files:
        st.warning('No models found in ./model. Please run **train_models.py** first.')
    selected = st.selectbox('Available models', options=model_files, format_func=lambda p: p.stem)

with right:
    st.subheader('2) Upload Test CSV (or use sample)')
    up = st.file_uploader('CSV file', type=['csv'])
    use_sample = st.checkbox('Use saved sample test set (data/test_sample.csv)', value=True if DEFAULT_TEST.exists() and not up else False)


def _read_csv(file_like_or_path):
    if isinstance(file_like_or_path, (str, Path)):
        return pd.read_csv(file_like_or_path)
    return pd.read_csv(file_like_or_path)

if selected:
    pipe = load_model(selected)

    # Data
    df = None
    if up is not None:
        df = _read_csv(up)
    elif use_sample and DEFAULT_TEST.exists():
        df = _read_csv(DEFAULT_TEST)
        st.info('Using sample test set saved during training (data/test_sample.csv).')

    if df is not None:
        st.write('**Preview**', df.head())
        has_label = 'Churn' in df.columns
        if has_label:
            y_true = df['Churn'].map({'Yes':1, 'No':0}).astype(int)
            X = df.drop(columns=['Churn'])
        else:
            y_true = None
            X = df.copy()

        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])

        # Predict
        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X)[:,1]
        else:
            scores = pipe.decision_function(X)
            s_min, s_max = scores.min(), scores.max()
            y_prob = (scores - s_min) / (s_max - s_min + 1e-9)
        y_pred = pipe.predict(X)

        # Show outputs
        st.subheader('3) Results')
        pred_df = pd.DataFrame({
            'Predicted_Prob(Churn)': y_prob,
            'Predicted_Label': np.where(y_pred==1, 'Yes', 'No')
        })
        st.dataframe(pred_df.head(20))

        if has_label:
            metrics = _calc_metrics(y_true, y_prob, y_pred)
            mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
            for (k, v), col in zip(metrics.items(), [mcol1, mcol2, mcol3, mcol4, mcol5, mcol6]):
                col.metric(k, f"{v:.4f}" if v==v else 'NA')

            # Confusion matrix
            st.markdown('**Confusion Matrix**')
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No','Yes'], yticklabels=['No','Yes'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix – {selected.stem}')
            st.pyplot(fig)

            # Classification report
            st.markdown('**Classification Report**')
            rpt = classification_report(y_true, y_pred, target_names=['No','Yes'])
            st.code(rpt, language='text')
        else:
            st.info('No ground-truth **Churn** column detected. Showing predictions only. To view metrics, upload a file that includes the **Churn** column.')
    else:
        st.warning('Please upload a CSV or tick "Use sample test set".')
