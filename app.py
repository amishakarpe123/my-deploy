import streamlit as st
import pandas as pd
import pickle
import time
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(page_title="Performance Classifier", page_icon="📊", layout="centered")

# Custom CSS for styling and animations
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        text-align: center;
        animation: slideUp 0.8s ease-out;
    }
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation asset
lottie_data = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")

# Header Section
st.title("📊 Level Predictor AI")
st.write("Predict performance levels based on study habits and metrics.")

if lottie_data:
    st_lottie(lottie_data, height=200, key="data_anim")

st.markdown("---")

# Input Section
st.subheader("📝 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    student_id = st.number_input("Reference ID", min_value=0, value=101)
    study_hours = st.number_input("Study Hours/Day", min_value=0.0, max_value=24.0, value=6.5)
    sleep_hours = st.number_input("Sleep Hours/Day", min_value=0.0, max_value=24.0, value=7.5)

with col2:
    social_media = st.number_input("Social Media Hours/Day", min_value=0.0, max_value=24.0, value=1.5)
    exam_score = st.number_input("Current Exam Score (%)", min_value=0.0, max_value=100.0, value=80.0)

# Prediction Button
if st.button("🚀 Analyze & Predict"):
    # Prepare input for the DecisionTreeClassifier
    # Note: Model expects features in this exact order
    input_df = pd.DataFrame([[student_id, study_hours, sleep_hours, social_media, exam_score]], 
                            columns=['id', 'study_hours', 'sleep_hours', 'social_media_hours', 'exam_score'])
    
    with st.spinner('Running AI patterns...'):
        time.sleep(1.2) # Aesthetic delay
        prediction = model.predict(input_df)[0]
        
    st.markdown("### Model Insight:")
    
    # Visual feedback based on category
    if prediction == "High":
        color, icon = "#2e7d32", "🏆"
        st.balloons()
    elif prediction == "Medium":
        color, icon = "#f9a825", "⚖️"
    else:
        color, icon = "#c62828", "⚠️"
    
    st.markdown(f"""
        <div class="result-card" style="border-left: 10px solid {color};">
            <h1 style="color: {color}; margin-bottom: 0;">{icon} {prediction}</h1>
            <p style="font-size: 1.2em; color: #555;">The student is classified into the <b>{prediction}</b> category.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Deployment Template | Model: DecisionTreeClassifier")
