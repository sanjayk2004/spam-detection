from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import os

# Load the trained model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None, accuracy=None)

# Route for predicting spam
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    if not message.strip():
        return render_template('index.html', prediction="⚠️ Please enter a message.", accuracy=None, message=message)

    # Minimal preprocessing
    cleaned_message = message.lower()
    transformed = vectorizer.transform([cleaned_message])
    prediction = model.predict(transformed)[0]

    result = "🚨 This message is Spam!" if prediction == 1 else "✅ This message is Not Spam."
    return render_template('index.html', prediction=result, accuracy=None, message=message)

# Route to evaluate model accuracy on the dataset
@app.route('/accuracy')
def accuracy():
    if not os.path.exists("SMSSpamCollection"):
        return render_template("index.html", prediction=None, accuracy="Dataset not found.")

    # Load dataset
    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].str.lower()

    X = vectorizer.transform(df['message'])
    y_true = df['label_num']
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    acc_percent = round(acc * 100, 2)

    return render_template('index.html', prediction=None, accuracy=f"📊 Model Accuracy: {acc_percent}%", message=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
