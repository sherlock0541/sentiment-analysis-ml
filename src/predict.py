# src/predict.py

from pathlib import Path
import joblib

from preprocessing import preprocess_text


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading sentiment model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):
    """
    Predict the sentiment of a new review.

    Returns:
        sentiment
        confidence
        probabilities
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty.")

    # Apply the same preprocessing used during training
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        raise ValueError(
            "The text contains no usable words after preprocessing."
        )

    # Prediction
    prediction = model.predict([cleaned_text])[0]

    # Probability for each class
    probabilities_array = model.predict_proba([cleaned_text])[0]
    probabilities = dict(
        zip(model.classes_, probabilities_array)
    )

    confidence = float(max(probabilities_array))

    return prediction, confidence, probabilities


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    print("\nEnter a movie review to analyze.")
    print("Type 'exit' to quit.")

    while True:

        text = input("\nReview: ")

        if text.strip().lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            sentiment, confidence, probabilities = predict_sentiment(text)

            print("\n" + "-" * 40)
            print(f"Sentiment : {sentiment.upper()}")
            print(f"Confidence: {confidence:.2%}")

            print("\nProbabilities:")

            for label, probability in probabilities.items():
                print(f"  {label.capitalize():<10}: {probability:.2%}")

        except ValueError as error:

            print(f"\nError: {error}")