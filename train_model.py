# train_model.py
import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# Preprocess
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned'] = df['message'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
