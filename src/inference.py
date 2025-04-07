from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)

