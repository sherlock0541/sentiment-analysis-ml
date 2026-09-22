
# src/train.py

from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SENTIMENT ANALYSIS - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset size: {len(df):,} reviews")


# ============================================================
# CHECK DATA
# ============================================================

required_columns = {
    "clean_review",
    "sentiment"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# Remove missing values just in case

df = df.dropna(
    subset=[
        "clean_review",
        "sentiment"
    ]
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["clean_review"]

y = df["sentiment"]


print("\nClasses:")

print(
    y.value_counts()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

print("\nCreating TF-IDF + Logistic Regression pipeline...")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Model saved to:\n{MODEL_PATH}"
)


# ============================================================
# TEST THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("QUICK MODEL TEST")
print("=" * 60)

test_sentences = [
    "This movie was absolutely fantastic and I loved every minute of it.",
    "This was a terrible movie and I completely hated it.",
    "The movie was okay but nothing special."
]

for sentence in test_sentences:

    prediction = model.predict(
        [sentence]
    )[0]

    probabilities = model.predict_proba(
        [sentence]
    )[0]

    confidence = max(
        probabilities
    )

    print("\nText:")
    print(sentence)

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.2%}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "\nYour trained model is ready for deployment."
)

print(
    f"Saved model: {MODEL_PATH}"
)