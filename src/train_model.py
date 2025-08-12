# src/train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,  recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter




def train_and_evaluate(processed_dir, models_dir, reports_dir):
    # Load processed data
    file_path = os.path.join(processed_dir, "machines_processed_hourly.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data not found at {file_path}")

    df = pd.read_csv(file_path)

    # Ensure target column exists
    target_col = "failure"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Drop rows where target is NaN
    before_rows = len(df)
    df = df.dropna(subset=[target_col])
    after_rows = len(df)
    if before_rows != after_rows:
        print(f"⚠️ Dropped {before_rows - after_rows} rows due to NaN in target column '{target_col}'.")

    # Drop rows where any feature is NaN
    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)
    if before_rows != after_rows:
        print(f"⚠️ Dropped {before_rows - after_rows} rows due to NaN in features.")

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    

    # Drop timestamp (or any unwanted column) from both train & test
    drop_cols = ['timestamp']
    X_train = X_train.drop(columns=drop_cols, errors='ignore')
    X_test = X_test.drop(columns=drop_cols, errors='ignore')
    
    # Clean training data before SMOTE
    
    X_train = X_train.fillna(0)  # Fill NaN values with 0
    y_train = y_train.fillna(0)

# Keep only numeric columns for SMOTE
    X_train = X_train.select_dtypes(include=['number'])

# Reset index to avoid mismatch
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    print("Before SMOTE:", Counter(y_train))
    
    # Apply SMOTE for imbalance handling
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", Counter(y_train_res))
    
    # Clean test data (same steps, but no SMOTE)
    X_test = X_test.fillna(0)
    X_test = X_test.select_dtypes(include=['number'])
    X_test = X_test.reset_index(drop=True)
    
    


    # Train Random Forest model
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_res, y_train_res)

# Predict probabilities instead of direct class labels
    y_prob = model.predict_proba(X_test)[:, 1]

# Custom threshold for recall boost
    THRESHOLD = 0.3  # Lower than default 0.5
    y_pred = (y_prob >= THRESHOLD).astype(int)

# Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.2f}")
    print(f"📈 Recall: {recall:.2f}")
    print(report)

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "predictive_model.pkl")
    joblib.dump(model, model_path)

    # Save report
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "model_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n\n")
        f.write(report)

    print(f"💾 Model saved to {model_path}")
    print(f"📝 Report saved to {report_path}")
