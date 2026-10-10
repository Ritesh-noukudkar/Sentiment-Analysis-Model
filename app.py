import flask 
from flask import Flask, request, jsonify, render_template
import pickle
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = str(data.get("review", "")).strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    features = vectorizer.transform([review])
    prediction = model.predict(features)[0]

    # Confirm this mapping against your notebook's label encoding.
    sentiment = str(prediction)

    return jsonify({
        "review": review,
        "prediction": sentiment
    })


if __name__ == "__main__":
    app.run(debug=True)
