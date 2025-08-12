"""
Train predictive models to forecast failure within next 24 hours.
Saves model to models/ and evaluation plots/data to reports/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLUMNS = [
    "temperature", "vibration", "pressure", "humidity",
    "runtime_hours", "temp_roll_mean_3", "vib_roll_mean_3", "temp_roll_std_3", "time_idx"
]

def load_processed(processed_dir="data/processed"):
    path = os.path.join(processed_dir, "machines_processed_hourly.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=['timestamp'])

def prepare_xy(df):
    df = df.copy()
    # drop rows where target is NaN
    df = df.dropna(subset=['failure_next_24h'])
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df['failure_next_24h'].astype(int)
    return X, y, df

def train_and_evaluate(processed_dir="data/processed", models_dir="models", reports_dir="reports"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    df = load_processed(processed_dir)
    X, y, df_full = prepare_xy(df)

    # simple train-test split (time-aware: split by timestamp)
    df_full = df_full.sort_values('timestamp')
    split_idx = int(0.8 * len(df_full))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # class imbalance handling: use class_weight in RF
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    print("Classification report (test):\n", report_text)

    # ROC AUC
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else float("nan")
    print("ROC AUC:", auc)

    # Save model
    model_path = os.path.join(models_dir, "rf_failure_predictor.joblib")
    joblib.dump(clf, model_path)

    # Save evaluation metrics CSV
    metrics = {
        "auc": auc,
        "n_test": len(y_test),
        "n_positives_test": int(y_test.sum())
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(reports_dir, "evaluation_metrics.csv"), index=False)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix (test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # Feature importance
    importances = clf.feature_importances_
    fi = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
    plt.figure(figsize=(8,4))
    sns.barplot(data=fi, x="importance", y="feature")
    plt.title("Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close()

    # Save sample predictions
    preds_df = X_test.copy()
    preds_df['y_true'] = y_test.values
    preds_df['y_pred'] = y_pred
    preds_df['y_proba'] = y_proba
    preds_df.to_csv(os.path.join(reports_dir, "predictions_sample.csv"), index=False)

    print(f"Model saved to {model_path}, reports saved under {reports_dir}")
