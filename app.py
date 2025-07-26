# 📦 Install necessary libraries
!pip install streamlit pyngrok scikit-learn pandas numpy nltk --quiet

# 📁 Import Libraries
import pandas as pd
import numpy as np
import string
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pyngrok import ngrok

# 📊 Load Dataset
df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/sms-spam-collection-dataset.csv')
df.columns = ['label', 'message']

# 🧹 Clean Text Function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# 🧹 Apply Cleaning
df['cleaned'] = df['message'].apply(clean_text)

# 🔡 Vectorize Text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})

# 🧠 Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 📈 Print Evaluation
y_pred = model.predict(X_test)
print("📊 Model Evaluation:\n", classification_report(y_test, y_pred))

# 💾 Save Model and Vectorizer
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 📝 Create Streamlit App
app_code = '''
import streamlit as st
import pickle

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Spam Classifier", page_icon="📱")
st.title("📱 Spam Message Classifier")
st.write("Enter an SMS message to check if it's spam or not:")

user_input = st.text_area("✉️ Message:")

if st.button("🔍 Predict"):
    if user_input:
        cleaned = user_input.lower()
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success("🚨 Spam!" if prediction == 1 else "✅ Not Spam.")
    else:
        st.warning("⚠️ Please enter a message.")
'''

# 💾 Write Streamlit app to file
with open("app.py", "w") as f:
    f.write(app_code)

# 🚀 Run Streamlit App
!streamlit run app.py &> /dev/null &

# 🌐 Setup Ngrok Tunnel
public_url = ngrok.connect(8501)
print(f"🔗 Click here to access the Spam Detector Web App:\n{public_url}")
