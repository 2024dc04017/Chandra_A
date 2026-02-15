
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


# ------------------------ Page setup ------------------------
st.set_page_config(page_title='Customer Churn – ML Models', page_icon='📊', layout='wide')
st.title('📊 Customer Churn – Interactive ML Demo (Telco)')
st.caption('Upload a **test CSV** (Telco schema) or use the saved sample. Choose a model to evaluate or predict.')

MODEL_DIR = Path('model')
DEFAULT_TEST = Path('data/test_sample.csv')


# ------------------------ Utilities ------------------------
@st.cache_resource
def list_models():
    """Return available trained pipelines (*.joblib) from ./model."""
    return sorted([p for p in MODEL_DIR.glob('*.joblib')])

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

def read_csv(file_like_or_path):
    """Read CSV from upload or local path."""
    if isinstance(file_like_or_path, (str, Path)):
        return pd.read_csv(file_like_or_path)
    return pd.read_csv(file_like_or_path)

def normalize_churn_series(s: pd.Series) -> pd.Series:
    """
    Normalize a 'Churn' column to Yes/No -> 1/0, handling variants:
    - 'yes', 'y', 'true', '1'  -> 1
    - 'no', 'n', 'false', '0'  -> 0
    - trims whitespace, case-insensitive
    Returns a Float series (0/1/NaN). Caller can dropna() before .astype(int).
    """
    # Cast to string, strip, lowercase
    s = s.astype(str).str.strip().str.lower()

    # Empty-like strings -> NaN
    s = s.replace({'': np.nan, 'nan': np.nan, 'none': np.nan})

    mapping = {
        'yes': 1, 'y': 1, 'true': 1, '1': 1,
        'no': 0,  'n': 0, 'false': 0, '0': 0
    }
    return s.map(mapping)

def safe_predict_proba_or_score(pipe, X):
    """Return probability scores in [0,1] for AUC/logit displays."""
    if hasattr(pipe, 'predict_proba'):
        return pipe.predict_proba(X)[:, 1]
    elif hasattr(pipe, 'decision_function'):
        scores = pipe.decision_function(X).astype(float)
        # min-max to 0..1 for AUC compatibility
        s_min, s_max = np.min(scores), np.max(scores)
        return (scores - s_min) / (s_max - s_min + 1e-12)
    else:
        # No probabilistic score available
        return np.zeros(len(X), dtype=float)

def calc_metrics(y_true, y_prob, y_pred):
    """Compute all required metrics; be robust if AUC cannot be computed."""
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

def plot_confusion(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    st.pyplot(fig)


# ------------------------ UI Layout ------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader('1) Choose a Model')
    model_files = list_models()
    if not model_files:
        st.warning('No models found in ./model. Please run **train_models.py** locally and upload the .joblib files.')
    selected = st.selectbox('Available models', options=model_files, format_func=lambda p: p.stem)

with right:
    st.subheader('2) Upload Test CSV (or use sample)')
    up = st.file_uploader('Upload CSV', type=['csv'])
    use_sample_default = True if (DEFAULT_TEST.exists() and up is None) else False
    use_sample = st.checkbox('Use saved sample (data/test_sample.csv)', value=use_sample_default)


# ------------------------ Inference & Metrics ------------------------
if selected:
    pipe = load_model(selected)

    # Obtain a dataframe either from upload or from the saved sample
    df = None
    if up is not None:
        try:
            df = read_csv(up)
            st.success('Uploaded CSV loaded.')
        except Exception as e:
            st.error(f'Could not read the uploaded CSV: {e}')
    elif use_sample and DEFAULT_TEST.exists():
        try:
            df = read_csv(DEFAULT_TEST)
            st.info('Using sample test set saved during training (data/test_sample.csv).')
        except Exception as e:
            st.error(f'Could not read sample test set: {e}')

    if df is not None and not df.empty:
        st.write('**Preview**')
        st.dataframe(df.head())

        # Prepare features and (optional) labels
        df = df.copy()
        has_label = 'Churn' in df.columns

        # X for prediction
        X = df.drop(columns=['Churn'], errors='ignore')
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])

        # Predict always (for UX)
        try:
            y_prob_all = safe_predict_proba_or_score(pipe, X)
            y_pred_all = pipe.predict(X)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            st.stop()

        # Display predictions (top rows)
        st.subheader('3) Results')
        pred_df = pd.DataFrame({
            'Predicted_Prob(Churn)': y_prob_all,
            'Predicted_Label': np.where(y_pred_all == 1, 'Yes', 'No')
        })
        st.dataframe(pred_df.head(20))

        # Compute evaluation metrics only if label exists
        if has_label:
            # Normalize the 'Churn' column robustly
            y_map = normalize_churn_series(df['Churn'])

            # Indices where mapping succeeded (0 or 1)
            valid_idx = y_map.dropna().index

            # Inform about any dropped rows due to invalid/blank labels
            dropped = len(df) - len(valid_idx)
            if dropped > 0:
                st.warning(
                    f"{dropped} row(s) had invalid or missing values in 'Churn' and were excluded from metrics. "
                    "Accepted values include: Yes/No, 1/0, True/False (any case, spaces ignored)."
                )

            if len(valid_idx) == 0:
                st.info("No valid ground-truth labels found after cleaning. Showing predictions only.")
            else:
                # Align predictions to valid rows for fair evaluation
                y_true = y_map.loc[valid_idx].astype(int).values
                y_prob = y_prob_all[valid_idx]
                y_pred = y_pred_all[valid_idx]

                metrics = calc_metrics(y_true, y_prob, y_pred)

                # KPIs
                mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
                for (k, v), col in zip(metrics.items(), [mcol1, mcol2, mcol3, mcol4, mcol5, mcol6]):
                    if pd.isna(v):
                        col.metric(k, "NA")
                    else:
                        col.metric(k, f"{v:.4f}")

                # Confusion matrix
                st.markdown('**Confusion Matrix**')
                plot_confusion(y_true, y_pred, f'Confusion Matrix – {selected.stem}')

                # Classification report
                st.markdown('**Classification Report**')
                rpt = classification_report(y_true, y_pred, target_names=['No', 'Yes'])
                st.code(rpt, language='text')
        else:
            st.info("No **Churn** column detected. Showing predictions only. "
                    "To view evaluation metrics, include a **Churn** column with Yes/No (or 1/0, True/False).")

    elif up is None and not use_sample:
        st.warning('Please upload a CSV or tick **Use saved sample**.')
