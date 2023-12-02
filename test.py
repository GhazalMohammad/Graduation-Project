import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

# Load dataset
data = pd.read_csv("sentiment_data.csv")

# Create a TfidfVectorizer to convert raw text into a numerical representation
vectorizer = TfidfVectorizer(stop_words="english")

# Convert raw text into a numerical matrix
X = vectorizer.fit_transform(data["text"])

# Set the target variable
y = data["text"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LinearSVC model
model = LinearSVC()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "sentiment_model.joblib")
