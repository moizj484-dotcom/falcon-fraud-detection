"""
FALCON - Dual Model Training & Evaluation Engine
Trains and compares:
  1. Logistic Regression (Baseline Model)
  2. Random Forest (Tree-based Ensemble Model)
Evaluates on Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
Automatically selects and deploys the best-performing model based on F1-Score.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "falcon_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("       FALCON DUAL MODEL TRAINING & COMPARISON")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError("Processed dataset not found at data/processed/falcon_features.csv")

print("Loading processed features...")
df = pd.read_csv(DATA_PATH)

if "isFraud" not in df.columns:
    raise ValueError("Target column 'isFraud' missing from features dataset.")

# Separate target and features
drop_cols = ["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].select_dtypes(include=[np.number])
y = df["isFraud"].astype(int)

# Clean any residual infinite or NaN values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

total_records = len(X)
fraud_records = int(y.sum())
legit_records = total_records - fraud_records

print(f"Total Working Dataset:   {total_records:,}")
print(f"Legitimate Transactions: {legit_records:,}")
print(f"Fraudulent Transactions: {fraud_records:,}")
print(f"Engineered Features:     {len(feature_cols)}")

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Set: {len(X_train):,} samples ({int(y_train.sum()):,} fraud)")
print(f"Testing Set:  {len(X_test):,} samples ({int(y_test.sum()):,} fraud)")

# -------------------------------------------------------------
# 1. Train Baseline: Logistic Regression (Scaled + Balanced)
# -------------------------------------------------------------
print("\n" + "-" * 50)
print("1/2 Training Model 1: Logistic Regression (Baseline)...")
print("-" * 50)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

cm_lr = confusion_matrix(y_test, y_pred_lr)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

metrics_lr = {
    "accuracy": float(accuracy_score(y_test, y_pred_lr)),
    "precision": float(precision_score(y_test, y_pred_lr, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_lr, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred_lr, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob_lr)),
    "true_negatives": int(tn_lr),
    "false_positives": int(fp_lr),
    "false_negatives": int(fn_lr),
    "true_positives": int(tp_lr)
}

print(f"Logistic Regression Results:")
print(f"  Accuracy:  {metrics_lr['accuracy'] * 100:.2f}%")
print(f"  Precision: {metrics_lr['precision'] * 100:.2f}%")
print(f"  Recall:    {metrics_lr['recall'] * 100:.2f}%")
print(f"  F1-Score:  {metrics_lr['f1_score'] * 100:.2f}%")
print(f"  ROC-AUC:   {metrics_lr['roc_auc']:.4f}")
print(f"  False Positives: {fp_lr:,} | False Negatives: {fn_lr:,}")

# -------------------------------------------------------------
# 2. Train Ensemble: Random Forest (Balanced)
# -------------------------------------------------------------
print("\n" + "-" * 50)
print("2/2 Training Model 2: Random Forest (Tree Ensemble)...")
print("-" * 50)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)
y_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]

cm_rf = confusion_matrix(y_test, y_pred_rf)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

metrics_rf = {
    "accuracy": float(accuracy_score(y_test, y_pred_rf)),
    "precision": float(precision_score(y_test, y_pred_rf, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_rf, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred_rf, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob_rf)),
    "true_negatives": int(tn_rf),
    "false_positives": int(fp_rf),
    "false_negatives": int(fn_rf),
    "true_positives": int(tp_rf)
}

print(f"Random Forest Results:")
print(f"  Accuracy:  {metrics_rf['accuracy'] * 100:.2f}%")
print(f"  Precision: {metrics_rf['precision'] * 100:.2f}%")
print(f"  Recall:    {metrics_rf['recall'] * 100:.2f}%")
print(f"  F1-Score:  {metrics_rf['f1_score'] * 100:.2f}%")
print(f"  ROC-AUC:   {metrics_rf['roc_auc']:.4f}")
print(f"  False Positives: {fp_rf:,} | False Negatives: {fn_rf:,}")

# -------------------------------------------------------------
# 3. Model Comparison & Best Selection (Primary: F1-Score)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("       MODEL COMPARISON & SELECTION")
print("=" * 60)

comparison = {
    "Logistic Regression": metrics_lr,
    "Random Forest": metrics_rf
}

if metrics_rf["f1_score"] >= metrics_lr["f1_score"]:
    best_model_name = "Random Forest"
    best_model = rf_classifier
    best_metrics = metrics_rf
    best_cm = cm_rf
else:
    best_model_name = "Logistic Regression"
    best_model = lr_pipeline
    best_metrics = metrics_lr
    best_cm = cm_lr

print(f"Winner: {best_model_name} (Highest F1-Score: {best_metrics['f1_score'] * 100:.2f}%)")

# Extract Feature Importances from Random Forest
rf_importances = dict(zip(X.columns, [float(v) for v in rf_classifier.feature_importances_]))
sorted_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))

# -------------------------------------------------------------
# 4. Serialize Models & Metrics for Dashboard & Git Deployment
# -------------------------------------------------------------
print("\nSaving deployment artifacts...")

# Save both individual models
joblib.dump(lr_pipeline, MODEL_DIR / "logistic_regression.pkl")
joblib.dump(rf_classifier, MODEL_DIR / "random_forest.pkl")

# Save primary deployment model (best model)
joblib.dump(best_model, MODEL_DIR / "falcon_model.pkl")

# Save feature names list
feature_names_list = list(X.columns)
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names_list, f, indent=4)

# Save comparison summary
comparison_meta = {
    "models": comparison,
    "best_model": best_model_name,
    "selection_metric": "f1_score",
    "feature_names": feature_names_list,
    "test_size": len(y_test)
}

with open(MODEL_DIR / "model_comparison.json", "w") as f:
    json.dump(comparison_meta, f, indent=4)

# Save best model metrics
with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=4)

# Save confusion matrix
with open(MODEL_DIR / "confusion_matrix.json", "w") as f:
    json.dump(best_cm.tolist(), f, indent=4)

# Save feature importances
with open(MODEL_DIR / "feature_importance.json", "w") as f:
    json.dump(sorted_importances, f, indent=4)

# Save a comparison CSV table in experiments
comp_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Positives", "False Negatives"],
    "Logistic Regression (Baseline)": [
        f"{metrics_lr['accuracy'] * 100:.2f}%",
        f"{metrics_lr['precision'] * 100:.2f}%",
        f"{metrics_lr['recall'] * 100:.2f}%",
        f"{metrics_lr['f1_score'] * 100:.2f}%",
        f"{metrics_lr['roc_auc']:.4f}",
        f"{metrics_lr['false_positives']:,}",
        f"{metrics_lr['false_negatives']:,}"
    ],
    "Random Forest (Ensemble)": [
        f"{metrics_rf['accuracy'] * 100:.2f}%",
        f"{metrics_rf['precision'] * 100:.2f}%",
        f"{metrics_rf['recall'] * 100:.2f}%",
        f"{metrics_rf['f1_score'] * 100:.2f}%",
        f"{metrics_rf['roc_auc']:.4f}",
        f"{metrics_rf['false_positives']:,}",
        f"{metrics_rf['false_negatives']:,}"
    ]
})
comp_df.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=False)

print("\n" + "=" * 60)
print("FALCON MODEL TRAINING COMPLETED SUCCESSFULLY")
print(f"Active Model:    {best_model_name}")
print(f"Saved artifacts:")
print(f"  - {MODEL_DIR / 'falcon_model.pkl'}")
print(f"  - {MODEL_DIR / 'logistic_regression.pkl'}")
print(f"  - {MODEL_DIR / 'random_forest.pkl'}")
print(f"  - {MODEL_DIR / 'model_comparison.json'}")
print(f"  - {MODEL_DIR / 'feature_importance.json'}")
print(f"  - {OUTPUT_DIR / 'model_comparison_table.csv'}")
print("=" * 60)
