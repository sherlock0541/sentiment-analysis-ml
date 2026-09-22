# src/compare_models.py

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SENTIMENT ANALYSIS - MODEL COMPARISON")
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print(
    f"TF-IDF features: {X_train_tfidf.shape[1]:,}"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(
            random_state=42
        )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    REPORT_DIR
    / "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nResults saved to:\n{results_path}"
)


# ============================================================
# DETAILED REPORTS
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)

for name, model in models.items():

    y_pred = model.predict(
        X_test_tfidf
    )

    print(f"\n{name}")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


# ============================================================
# FINISHED
# ============================================================

print("=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)