import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
try:
    with open("spam_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

# App UI
st.set_page_config(page_title="Spam Message Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a valid message.")
    else:
        # Transform and predict
        transformed = vectorizer.transform([message.lower()])
        prediction = model.predict(transformed)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
