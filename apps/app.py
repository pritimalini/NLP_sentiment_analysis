from flask import Flask, request, jsonify, render_template_string
import pickle

app = Flask(__name__)

# Load the saved TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load the trained sentiment analysis model
with open("models/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# Inline HTML code for the front-end
HTML_CODE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NLP Sentiment Analysis</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <style>
    body { background-color: #f8f9fa; }
    .container { max-width: 600px; margin-top: 50px; }
    .card { border-radius: 10px; }
    #result { margin-top: 20px; padding: 10px; border-radius: 5px; }
  </style>
</head>
<body>
  <div class="container">
    <h2 class="text-center">📝 Sentiment Analysis</h2>
    <div class="card p-4 shadow">
      <label for="text-input" class="form-label"><b>Enter your text:</b></label>
      <textarea id="text-input" class="form-control" rows="4" placeholder="Type your text here..."></textarea>
      <button class="btn btn-primary mt-3 w-100" onclick="processText()">Analyze Sentiment</button>
      <div id="result" class="alert alert-info mt-3" style="display:none;"></div>
    </div>
  </div>
  
  <script>
    function processText() {
      var text = document.getElementById("text-input").value;
      if (!text) {
        alert("Please enter some text!");
        return;
      }
  
      fetch("/predict", {
        method: "POST",
        body: new URLSearchParams({ "text": text }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
      })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          alert(data.error);
        } else {
          document.getElementById("result").style.display = "block";
          document.getElementById("result").innerHTML = "<b>Sentiment:</b> " + data.sentiment;
        }
      })
      .catch(error => console.error("Error:", error));
    }
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_CODE)

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve text from the form data
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided!"}), 400

    # Transform the text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    # Get sentiment prediction from the model
    prediction = model.predict(text_vectorized)
    # Assume the model returns 1 for positive sentiment and 0 for negative
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
