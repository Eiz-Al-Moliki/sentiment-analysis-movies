import re
import pandas as pd
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import os

# 1. First, define the clean_text function
def clean_text(text):
    """Clean and preprocess a text string."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

# 2. THEN, define the load_and_preprocess_data function
# Now it can "see" the clean_text function defined above it
def load_and_preprocess_data(filepath='data/raw/movie_reviews.csv'):
    """Load a CSV dataset and preprocess the text column. Falls back to NLTK dataset."""
    
    # Check if the CSV file exists first
    if os.path.exists(filepath):
        print("Loading data from CSV file...")
        df = pd.read_csv(filepath)
        # Assuming the CSV has 'review' and 'sentiment' columns
    else:
        print("CSV file not found. Loading built-in NLTK movie_reviews dataset...")
        # Download the dataset if you haven't already
        nltk.download('movie_reviews', quiet=True)
        
        # Create a list of documents and their labels
        documents = []
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                text = movie_reviews.raw(fileid)
                documents.append((text, category))
        
        # Convert the list to a DataFrame
        df = pd.DataFrame(documents, columns=['review', 'sentiment'])
        # The NLTK dataset uses 'pos'/'neg', let's map them for clarity
        df['sentiment'] = df['sentiment'].map({'pos': 'positive', 'neg': 'negative'})
    
    # Now this line will work because clean_text is defined above!
    df['cleaned_review'] = df['review'].apply(clean_text)
    return df