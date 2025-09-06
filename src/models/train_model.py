from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os  # <-- ADD THIS IMPORT AT THE TOP OF THE FILE

def train_model(X, y):
    """Train a Logistic Regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:\n", report)
    
    # Return the model and vectorizer for future predictions
    return model, vectorizer

# Function to save the model (you'll add this later)
def save_model(model, vectorizer, model_path='models/sentiment_model.joblib', vectorizer_path='models/tfidf_vectorizer.joblib'):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")