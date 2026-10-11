
from flask import Flask, request, jsonify, render_template
import pickle
import csv
import io
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def get_sentiment(review):
    features = vectorizer.transform([review])
    prediction = model.predict(features)[0]

    # Notebook encoding: 1 = Positive, 0 = Negative
    if str(prediction) == "1":
        return "Positive"
    elif str(prediction) == "0":
        return "Negative"
    else:
        return str(prediction)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = str(data.get("review", "")).strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    sentiment = get_sentiment(review)

    return jsonify({
        "review": review,
        "sentiment": sentiment
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Please select a CSV or TXT file."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    extension = Path(file.filename).suffix.lower()

    if extension not in [".csv", ".txt"]:
        return jsonify({"error": "Upload only CSV or TXT files."}), 400

    try:
        content = file.read().decode("utf-8-sig")

        if extension == ".txt":
            reviews = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

        else:
            reader = csv.DictReader(io.StringIO(content))

            if not reader.fieldnames:
                return jsonify({"error": "The CSV file is empty or has no header."}), 400

            # Prefer a column named review, text, comment, or content.
            column_lookup = {
                name.strip().lower(): name
                for name in reader.fieldnames
                if name
            }

            review_column = None
            for candidate in ["review", "reviews", "text", "comment", "content"]:
                if candidate in column_lookup:
                    review_column = column_lookup[candidate]
                    break

            if review_column is None:
                return jsonify({
                    "error": "Could not find a review column. Use a column named Review, Text, Comment, or Content."
                }), 400

            reviews = [
                str(row.get(review_column, "")).strip()
                for row in reader
                if str(row.get(review_column, "")).strip()
            ]

        if not reviews:
            return jsonify({"error": "No reviews found in the file."}), 400

        results = []
        for review in reviews:
            results.append({
                "review": review,
                "sentiment": get_sentiment(review)
            })

        positive_count = sum(
            1 for item in results if item["sentiment"] == "Positive"
        )
        negative_count = sum(
            1 for item in results if item["sentiment"] == "Negative"
        )

        return jsonify({
            "total": len(results),
            "positive": positive_count,
            "negative": negative_count,
            "results": results
        })

    except UnicodeDecodeError:
        return jsonify({
            "error": "Could not read this file. Please save it as UTF-8 and try again."
        }), 400
    except Exception as e:
        app.logger.exception("File analysis failed")
        return jsonify({"error": "Could not analyze the file. Please check its format."}), 500


if __name__ == "__main__":
    app.run(debug=True)
