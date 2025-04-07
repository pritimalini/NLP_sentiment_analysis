import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load preprocessed data
def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)  # Fix dtype warning

# Convert text into numerical representation
def vectorize_text(input_filepath, output_filepath):
    df = load_data(input_filepath)

    # Handle NaN values
    df["cleaned_text"] = df["cleaned_text"].fillna("")

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_text"])
    
    # Save the vectorizer
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save the transformed features
    pd.DataFrame(X.toarray()).to_csv(output_filepath, index=False)
    print(f"Vectorized data saved to {output_filepath}")

# Example Usage
if __name__ == "__main__":
    vectorize_text("data/processed/cleaned_data.csv", "data/vectorized/features.csv")
