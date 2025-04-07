import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load dataset
def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)  # Fix dtype warning

# Text Cleaning Function
def clean_text(text):
    if pd.isna(text):  # If NaN, return an empty string
        return ""

    text = str(text)  # Convert to string
    text = re.sub(r'\d+\.\d+', '', text)  # Remove floating-point numbers
    text = re.sub(r'\d+', '', text)  # Remove integers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text

# Tokenization, Stopword Removal, and Lemmatization
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")  # Remove 'not' from the stopwords list
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply Lemmatization
    return " ".join(words)

# Apply preprocessing
def process_dataset(filepath, output_filepath):
    df = load_data(filepath)
    df['sentiment']=df['Star Rating'].apply(lambda x: 1 if x > 3 else 0)
    
    # Ensure "Review Body" column exists
    if "Review Body" not in df.columns:
        raise KeyError("Column 'Review Body' not found in dataset!")

    df["cleaned_text"] = df["Review Body"].apply(clean_text).apply(preprocess_text)
    df.to_csv(output_filepath, index=False)
    print(f"Processed data saved to this file : {output_filepath}")

# Example Usage
if __name__ == "__main__":
    process_dataset("https://raw.githubusercontent.com/joshivaibhav/AmazonCustomerReview/master/amazondata.csv", 
                    "data/processed/cleaned_data.csv")
