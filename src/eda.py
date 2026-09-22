from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "IMDB_Dataset_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SENTIMENT ANALYSIS - EDA")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SENTIMENT DISTRIBUTION")
print("=" * 60)

sentiment_counts = df["sentiment"].value_counts()

print(sentiment_counts)

print("\nPercentages:")
print(
    (df["sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# REVIEW LENGTH
# ============================================================

df["review_length"] = (
    df["clean_review"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\n" + "=" * 60)
print("REVIEW LENGTH")
print("=" * 60)

print(df["review_length"].describe())


# ============================================================
# POSITIVE / NEGATIVE REVIEW LENGTH
# ============================================================

print("\nAverage review length by sentiment:")

print(
    df.groupby("sentiment")["review_length"]
    .mean()
    .round(2)
)


# ============================================================
# MOST COMMON WORDS
# ============================================================

print("\n" + "=" * 60)
print("MOST COMMON WORDS")
print("=" * 60)


def get_most_common_words(data, sentiment, n=20):

    text = " ".join(
        data[data["sentiment"] == sentiment]["clean_review"]
        .fillna("")
    )

    words = text.split()

    counter = Counter(words)

    return counter.most_common(n)


print("\nPositive reviews:")

positive_words = get_most_common_words(
    df,
    "positive",
    20
)

for word, count in positive_words:
    print(f"{word:20} {count:,}")


print("\nNegative reviews:")

negative_words = get_most_common_words(
    df,
    "negative",
    20
)

for word, count in negative_words:
    print(f"{word:20} {count:,}")


# ============================================================
# VISUALIZATION 1
# SENTIMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

sentiment_plot = (
    OUTPUT_DIR / "sentiment_distribution.png"
)

plt.savefig(
    sentiment_plot,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# VISUALIZATION 2
# REVIEW LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="review_length",
    hue="sentiment",
    bins=50,
    kde=True
)

plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")

plt.tight_layout()

length_plot = (
    OUTPUT_DIR / "review_length_distribution.png"
)

plt.savefig(
    length_plot,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# VISUALIZATION 3
# BOXPLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="sentiment",
    y="review_length"
)

plt.title("Review Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Words")

plt.tight_layout()

boxplot_path = (
    OUTPUT_DIR / "review_length_boxplot.png"
)

plt.savefig(
    boxplot_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# SAVE EDA SUMMARY
# ============================================================

summary = {
    "total_reviews": len(df),
    "positive_reviews": int(
        (df["sentiment"] == "positive").sum()
    ),
    "negative_reviews": int(
        (df["sentiment"] == "negative").sum()
    ),
    "missing_values": int(
        df.isnull().sum().sum()
    ),
    "duplicate_rows": int(
        df.duplicated().sum()
    ),
    "average_review_length": round(
        df["review_length"].mean(),
        2
    ),
    "median_review_length": round(
        df["review_length"].median(),
        2
    ),
}

summary_df = pd.DataFrame(
    [summary]
)

summary_path = OUTPUT_DIR / "eda_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print(f"- {sentiment_plot}")
print(f"- {length_plot}")
print(f"- {boxplot_path}")
print(f"- {summary_path}")

print("\nNext step:")
print("Build src/train.py")