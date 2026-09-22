# src/evaluate.py

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "IMDB_Dataset_clean.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sentiment_model.pkl"
)

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=[
        "clean_review",
        "sentiment"
    ]
)

X = df["clean_review"]
y = df["sentiment"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Test samples: {len(X_test):,}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTIONS
# ============================================================

print("Generating predictions...")

y_pred = model.predict(X_test)

# Probability for the positive class
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# CONVERT LABELS TO BINARY
# ============================================================

# Your dataset uses:
#
# negative -> 0
# positive -> 1

y_test_binary = (
    y_test == "positive"
).astype(int)


y_pred_binary = (
    y_pred == "positive"
).astype(int)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test_binary,
    y_probability
)

average_precision = average_precision_score(
    y_test_binary,
    y_probability
)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

report_path = (
    REPORT_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_path
)

print(
    f"\nClassification report saved:"
    f"\n{report_path}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "negative",
        "positive"
    ]
)

print("\nConfusion Matrix:")

print(cm)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Positive"
    ]
)

display.plot(
    ax=ax
)

ax.set_title(
    "Sentiment Analysis - Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    REPORT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_path = (
    REPORT_DIR
    / "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = precision_recall_curve(
    y_test_binary,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

pr_path = (
    REPORT_DIR
    / "precision_recall_curve.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "roc_auc",
            "average_precision"
        ],
        "value": [
            accuracy,
            roc_auc,
            average_precision
        ]
    }
)

metrics_path = (
    REPORT_DIR
    / "metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print(f"- {cm_path}")
print(f"- {roc_path}")
print(f"- {pr_path}")
print(f"- {report_path}")
print(f"- {metrics_path}")
