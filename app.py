import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Load the saved model and vectorizer
model = joblib.load('spam_detector_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Create the Streamlit interface
st.title("📧 Spam Message Detector")

st.write("""
This app detects whether a message is spam or not. 
Enter your message below and click 'Analyze' to check!
""")

# Create text input
message = st.text_area("Enter your message:", height=100)

if st.button("Analyze"):
    if message:
        # Preprocess the input
        processed_text = preprocess_text(message)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Display result
        st.markdown("### Analysis Result")
        
        if prediction == 1:
            st.error("🚨 This message is likely SPAM!")
            st.write(f"Confidence: {probability[1]:.2%}")
        else:
            st.success("✅ This message appears to be legitimate.")
            st.write(f"Confidence: {probability[0]:.2%}")
        
        # Show preprocessing details
        with st.expander("See preprocessing steps"):
            st.write("Original message:", message)
            st.write("Processed message:", processed_text)
    else:
        st.warning("Please enter a message to analyze.")

# Add sidebar information
with st.sidebar:
    st.header("About the Model")
    st.write("""
    This spam detector uses an XGBoost classifier trained on a dataset of spam and legitimate messages.
    
    Model Performance:
    - Training Accuracy: 99.7%
    - Testing Accuracy: 98.9%
    """)
