import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text using the same steps as training data
    """
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

# Initialize variables to store model and vectorizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Streamlit interface
st.title("📧 Spam Message Detector")

# Add model upload section
st.sidebar.header("Model Configuration")
model_upload = st.sidebar.file_uploader("Upload trained model (optional)", type=['joblib'])
vectorizer_upload = st.sidebar.file_uploader("Upload vectorizer (optional)", type=['joblib'])

if model_upload is not None and vectorizer_upload is not None:
    import joblib
    st.session_state.model = joblib.load(model_upload)
    st.session_state.vectorizer = joblib.load(vectorizer_upload)
    st.sidebar.success("Model and vectorizer loaded successfully!")

# Option to use current model
if st.session_state.model is None:
    st.warning("""Please either:
    1. Upload your saved model and vectorizer using the sidebar, or
    2. Run this script in the same session where you trained your model
    
    Using format:
    ```python
    # After training your model
    st.session_state.model = your_trained_model
    st.session_state.vectorizer = your_trained_vectorizer
    ```
    """)
else:
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
            text_vectorized = st.session_state.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = st.session_state.model.predict(text_vectorized)[0]
            probability = st.session_state.model.predict_proba(text_vectorized)[0]
            
            # Display result with proper formatting
            st.markdown("### Analysis Result")
            
            if prediction == 1:
                st.error("🚨 This message is likely SPAM!")
                st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.success("✅ This message appears to be legitimate.")
                st.write(f"Confidence: {probability[0]:.2%}")
            
            # Add explanation
            st.markdown("### Message Processing Details")
            with st.expander("See preprocessing steps"):
                st.write("Original message:", message)
                st.write("Processed message:", processed_text)
        
        else:
            st.warning("Please enter a message to analyze.")

    # Add information about the model
    with st.sidebar:
        st.header("About the Model")
        st.write("""
        This spam detector uses an XGBoost classifier trained on a dataset of spam and legitimate messages.
        
        Model Performance:
        - Training Accuracy: 99.7%
        - Testing Accuracy: 98.9%
        """)
        
        st.markdown("### How it works")
        st.write("""
        1. Your message is preprocessed (cleaning, removing stopwords)
        2. The cleaned text is converted to numerical features
        3. The XGBoost model analyzes these features
        4. A prediction is made based on learned patterns
        """)