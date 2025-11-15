# ml_train.py
"""
Train classic ML models using spatial + frequency features.
Models included:
 - Logistic Regression
 - SVM (RBF)
 - Random Forest
 - Gradient Boosting / XGBoost (optional)
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def train_classic_ml_models(df: pd.DataFrame, output_dir="ML_Models"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n===== PREPARING DATA =====")

    # Encode labels: fighting=1, not_fighting=0
    df["label_encoded"] = df["label"].map({"fighting": 1, "not_fighting": 0})

    # Drop any non-numeric column
    X = df.drop(columns=["label", "label_encoded"]).select_dtypes(include="number")
    y = df["label_encoded"]

    print(f"Using {X.shape[1]} numeric features.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Scaling done.")

    # Dictionary of models
    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "gradient_boosting": GradientBoostingClassifier(),
    }

    # Try loading XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8
        )
        print("XGBoost AVAILABLE!")
    except ImportError:
        print("XGBoost not installed — skipping.")

    results = {}

    print("\n===== TRAINING MODELS =====")
    for name, model in models.items():
        print(f"\n▶ Training: {name}")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save results
        results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Save model
        import joblib
        joblib.dump(model, f"{output_dir}/{name}.joblib")

    # Save scaler
    import joblib
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")

    print("\n===== DONE TRAINING =====")
    print(f"Saved all models + scaler in: {output_dir}")

    return results
