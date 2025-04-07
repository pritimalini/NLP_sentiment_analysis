import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load vectorized features
def load_data(features_filepath, labels_filepath):
    X = pd.read_csv(features_filepath)
    y = pd.read_csv(labels_filepath)["sentiment"]  # Assuming 'label' column exists
    return X, y

# Train a Logistic Regression Model
def train_model(features_filepath, labels_filepath, model_filepath):
    X, y = load_data(features_filepath, labels_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy is here : {accuracy_score(y_test, y_pred):.2f}")

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_filepath}")

# Example Usage
if __name__ == "__main__":
    train_model("data/vectorized/features.csv", "data/processed/cleaned_data.csv", "models/sentiment_model.pkl")
