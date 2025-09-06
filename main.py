from src.data.preprocess import load_and_preprocess_data
from src.models.train_model import train_model, save_model

def main():
    print("Starting Sentiment Analysis Project...")
    
    # 1. Load and preprocess data
    # You need to download a dataset (e.g., from Kaggle) and place it in data/raw/
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data/raw/movie_reviews.csv')
    
    # 2. Train the model
    print("Training model...")
    X = df['cleaned_review']
    y = df['sentiment'] # This column should have labels like 'positive'/'negative'
    model, vectorizer = train_model(X, y)
    
    # 3. Save the model for later use
    print("Saving model...")
    save_model(model, vectorizer, 'models/sentiment_model.joblib', 'models/tfidf_vectorizer.joblib')
    print("Done! Model saved.")

if __name__ == "__main__":
    main()