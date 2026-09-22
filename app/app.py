# app/app.py

from pathlib import Path
import sys

from flask import Flask, request, jsonify, render_template


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import predict_sentiment


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route("/api/model-info")
def model_info():

    return jsonify({
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "ngram_range": "1-2",
        "max_features": 50000,
        "accuracy": 0.9017,
        "roc_auc": 0.9658,
        "average_precision": 0.9657,
        "dataset": "IMDb Movie Reviews"
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body must contain JSON."
            }), 400

        text = data.get("text")

        if not text or not isinstance(text, str):
            return jsonify({
                "error": "The 'text' field is required."
            }), 400

        sentiment, confidence, probabilities = predict_sentiment(text)

        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                label: round(float(probability), 4)
                for label, probability in probabilities.items()
            }
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "error": "Internal server error."
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SENTIMENT ANALYSIS WEB APP")
    print("=" * 60)
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )