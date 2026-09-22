# src/preprocessing.py

import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "IMDB_Dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "IMDB_Dataset_clean.csv"


# ============================================================
# STOPWORDS
# ============================================================

STOP_WORDS = set(stopwords.words("english"))

# Keep negation words because they are important for sentiment.
NEGATION_WORDS = {
    "no",
    "nor",
    "not",
    "never",
    "neither",
    "hardly",
    "barely",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
}

STOP_WORDS -= NEGATION_WORDS


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean a single text.

    Steps:
    - Convert to string
    - Lowercase
    - Remove HTML
    - Remove URLs
    - Remove emails
    - Remove punctuation
    - Remove numbers
    - Remove extra spaces
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# STOPWORD REMOVAL
# ============================================================

def remove_stopwords(text):
    """
    Remove English stopwords while preserving
    sentiment-important negation words.
    """

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_words)


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Apply the complete preprocessing pipeline.
    """

    text = clean_text(text)
    text = remove_stopwords(text)

    return text


# ============================================================
# DATASET PREPROCESSING
# ============================================================

def preprocess_dataset(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH
):
    """
    Load, preprocess and save the complete dataset.
    """

    print("=" * 60)
    print("SENTIMENT DATASET PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{input_path}"
        )

    print(f"\nLoading dataset:")
    print(input_path)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    print(f"\nOriginal dataset:")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {list(df.columns)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {"review", "sentiment"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["review", "sentiment"]
    )

    print(
        f"\nRemoved missing rows: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Remove duplicate reviews
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["review"]
    )

    print(
        f"Removed duplicate reviews: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Clean reviews
    # --------------------------------------------------------

    print("\nCleaning reviews...")

    df["clean_review"] = df["review"].apply(
        preprocess_text
    )

    # --------------------------------------------------------
    # Remove empty reviews
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["clean_review"].str.strip() != ""
    ]

    print(
        f"Removed empty reviews: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Normalize sentiment labels
    # --------------------------------------------------------

    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Display sentiment distribution
    # --------------------------------------------------------

    print("\nSentiment distribution:")

    print(
        df["sentiment"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nFinal rows    : {len(df)}")
    print(f"Final columns : {list(df.columns)}")
    print(f"\nSaved to:")
    print(output_path)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_dataset()

