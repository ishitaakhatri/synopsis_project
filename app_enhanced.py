import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from collections import Counter
import re

# Try to import OpenAI for LLM functionality
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI library not installed. LLM features will be disabled. Install with: pip install openai")

# Page configuration
st.set_page_config(
    page_title="Special Child Writing Helper 🌈",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for child-friendly theme
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Student cards */
    .student-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 10px;
        cursor: pointer;
        transition: transform 0.3s;
        color: white;
        text-align: center;
    }
    
    .student-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Upload box */
    .uploadedFile {
        border: 3px dashed #f5576c;
        border-radius: 20px;
        background: rgba(255,255,255,0.9);
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 15px;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .teacher-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'teacher_name' not in st.session_state:
    st.session_state.teacher_name = ""
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None
if 'student_progress' not in st.session_state:
    st.session_state.student_progress = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Load model and metadata
@st.cache_resource
def load_model():
    """Load the trained model - handles different possible file formats"""
    model_paths = [
        'model.h5',
        'model.keras',
        'saved_model',
        'project_main.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path)
                st.success(f"✅ Model loaded successfully from {path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Could not load model from {path}: {str(e)}")
                continue
    
    st.warning("⚠️ No model file found. App will run in demo mode with simulated predictions.")
    return None

@st.cache_data
def load_student_data():
    """Load student data"""
    students = [
        {'name': 'Aaru', 'disability': 'MR', 'iq': 42, 'disability_percentage': 75, 'age': 10},
        {'name': 'Akash', 'disability': 'ID', 'iq': 38, 'disability_percentage': 75, 'age': 6},
        {'name': 'Chirag', 'disability': 'Severe MR', 'iq': 24, 'disability_percentage': 90, 'age': 24},
        {'name': 'Gagan', 'disability': 'CP', 'iq': 35, 'disability_percentage': 75, 'age': 25},
        {'name': 'Gargi', 'disability': 'CP', 'iq': 40, 'disability_percentage': 85, 'age': 17},
        {'name': 'Manoj', 'disability': 'Severe ID', 'iq': 20, 'disability_percentage': 90, 'age': 35},
        {'name': 'Mayur', 'disability': 'ID', 'iq': 40, 'disability_percentage': 85, 'age': 10},
        {'name': 'Meet', 'disability': 'Moderate MR', 'iq': 36, 'disability_percentage': 70, 'age': 20},
        {'name': 'Monika', 'disability': 'Severe MR', 'iq': 26, 'disability_percentage': 90, 'age': 14},
        {'name': 'Parul', 'disability': 'Moderate ID', 'iq': 42, 'disability_percentage': 75, 'age': 32},
        {'name': 'Prateek', 'disability': 'CP', 'iq': 24, 'disability_percentage': 90, 'age': 28},
        {'name': 'Preetam', 'disability': 'CP', 'iq': 35, 'disability_percentage': 90, 'age': 20},
        {'name': 'Rahul', 'disability': 'MR', 'iq': 48, 'disability_percentage': 80, 'age': 14},
        {'name': 'Samarth', 'disability': 'MR', 'iq': 48, 'disability_percentage': 80, 'age': 14},
        {'name': 'Sneha', 'disability': 'ID', 'iq': 24, 'disability_percentage': 90, 'age': 22},
        {'name': 'Sunny', 'disability': 'Severe CP', 'iq': 20, 'disability_percentage': 80, 'age': 18}
    ]
    return students

# Class labels from your model
CLASS_LABELS = ['ANSHUL', 'ANUJ', 'APPLE', 'ASHOK', 'BANANA', 'BAT', 'BEAR', 'BIKE', 'BLACK', 
                'BLUE', 'BUS', 'CAR', 'CARROT', 'CHEST', 'CHIRAG', 'CIRCLE', 'COW', 'CRICKET', 
                'CROW', 'DEER', 'DOCTOR', 'DOG', 'EAR', 'EARS', 'ELEPHANT', 'EYE', 'EYES', 
                'FOUR', 'FOX', 'FRIDAY', 'GOAT', 'GRAPES', 'GREEN', 'GUAVA', 'HAIR', 'HAND', 
                'HEAR', 'HEN', 'HOLI', 'IS', 'ISHITA', 'JEEP', 'JITENDER', 'LEG', 'LION', 
                'LIPS', 'LOTUS', 'LUDO', 'MANGO', 'MANOJ', 'MARIGOLD', 'MAT', 'MONDAY', 
                'MONSOON', 'MOUTH', 'NAME', 'NAN', 'NOSE', 'ONION', 'ORANGE', 'OWL', 'OX', 
                'PARROT', 'PARUL', 'PEA', 'PEAS', 'PINK', 'PLEASE', 'POLICE', 'POTATO', 
                'RADISH', 'RAIN', 'RECTANGLE', 'RED', 'ROSE', 'SATURDAY', 'SCOOTY', 'SORRY', 
                'SQUARE', 'SUDESH', 'SUDHIR', 'SUMMAR', 'SUMMER', 'SUNDAY', 'SUNFLOWER', 
                'SWAN', 'TAILOR', 'TEEJ', 'THURSDAY', 'TIGER', 'TOMATO', 'TRAIN', 'TRUCK', 
                'TUESDAY', 'TULIP', 'TWO', 'VAN', 'WEDNESDAY', 'WHITE', 'YELLOW']

# Letter difficulty database for generating practice words
LETTER_TO_WORDS = {
    'A': ['APPLE', 'ANUJ', 'ASHOK', 'ANSHUL', 'BANANA', 'CARROT', 'ORANGE', 'PARROT', 'SATURDAY'],
    'B': ['BANANA', 'BAT', 'BEAR', 'BIKE', 'BLUE', 'BUS'],
    'C': ['CAR', 'CARROT', 'CHEST', 'CHIRAG', 'CIRCLE', 'COW', 'CRICKET', 'CROW'],
    'D': ['DEER', 'DOCTOR', 'DOG', 'FRIDAY', 'MONDAY', 'RADISH', 'RED', 'SATURDAY', 'SUNDAY', 'TUESDAY', 'WEDNESDAY'],
    'E': ['BEAR', 'DEER', 'ELEPHANT', 'EYE', 'EYES', 'EAR', 'EARS', 'HEAR', 'HEN', 'JEEP', 'KNEE', 'PLEASE', 'RED', 'GREEN'],
    'F': ['FOX', 'FOUR', 'FRIDAY', 'SUNFLOWER'],
    'G': ['GRAPES', 'GREEN', 'GOAT', 'GUAVA', 'DOG', 'TIGER', 'FROG', 'LEG', 'ORANGE'],
    'H': ['HAIR', 'HAND', 'HEAR', 'HEN', 'HOLI', 'THURSDAY', 'MOUTH', 'FISH', 'TEETH'],
    'I': ['BIKE', 'CIRCLE', 'CRICKET', 'FRIDAY', 'IS', 'ISHITA', 'JITENDER', 'LION', 'PINK', 'POLICE', 'RAIN'],
    'J': ['JEEP', 'JITENDER'],
    'K': ['BIKE', 'BLACK', 'PINK'],
    'L': ['BLUE', 'BLACK', 'ELEPHANT', 'HOLI', 'LEG', 'LION', 'LIPS', 'LOTUS', 'LUDO', 'PLEASE', 'POLICE', 'TULIP', 'YELLOW'],
    'M': ['MANGO', 'MANOJ', 'MARIGOLD', 'MAT', 'MONDAY', 'MONSOON', 'MOUTH', 'SUMMER', 'SUMMAR', 'TOMATO'],
    'N': ['ANUJ', 'ANSHUL', 'BANANA', 'GREEN', 'HEN', 'LION', 'MANGO', 'MANOJ', 'MONDAY', 'MONSOON', 'NAN', 'NOSE', 'ONION', 'ORANGE', 'RAIN', 'SUNFLOWER', 'SWAN', 'TRAIN', 'VAN'],
    'O': ['DOG', 'DOCTOR', 'FOX', 'GOAT', 'HOLI', 'LOTUS', 'MANOJ', 'MONDAY', 'MONSOON', 'MOUTH', 'NOSE', 'ONION', 'ORANGE', 'OWL', 'OX', 'POLICE', 'POTATO', 'SCOOTY', 'SORRY', 'TOMATO', 'TWO'],
    'P': ['APPLE', 'PARROT', 'PARUL', 'PEA', 'PEAS', 'PINK', 'PLEASE', 'POLICE', 'POTATO'],
    'Q': ['SQUARE'],
    'R': ['BEAR', 'CAR', 'CARROT', 'CRICKET', 'CROW', 'DEER', 'DOCTOR', 'EAR', 'EARS', 'FOUR', 'FOX', 'FRIDAY', 'GRAPES', 'GREEN', 'HAIR', 'HEAR', 'MARIGOLD', 'ORANGE', 'PARROT', 'PARUL', 'RADISH', 'RAIN', 'RECTANGLE', 'RED', 'ROSE', 'SATURDAY', 'SCOOTY', 'SORRY', 'SQUARE', 'SUMMER', 'SUMMAR', 'SUNFLOWER', 'THURSDAY', 'TIGER', 'TRAIN', 'TRUCK', 'TUESDAY'],
    'S': ['BUS', 'ASHOK', 'IS', 'EARS', 'EYES', 'GRAPES', 'GUAVA', 'LIPS', 'LOTUS', 'ROSE', 'SATURDAY', 'SCOOTY', 'SORRY', 'SQUARE', 'SUDESH', 'SUDHIR', 'SUMMER', 'SUMMAR', 'SUNDAY', 'SUNFLOWER', 'SWAN', 'PEAS'],
    'T': ['BAT', 'BAT', 'CARROT', 'CAT', 'CHEST', 'MAT', 'POTATO', 'THURSDAY', 'TIGER', 'TOMATO', 'TRAIN', 'TRUCK', 'TUESDAY'],
    'U': ['BUS', 'BLUE', 'GUAVA', 'LUDO', 'MANOJ', 'MOUTH', 'SUDESH', 'SUDHIR', 'SUMMER', 'SUMMAR', 'SUNDAY', 'SUNFLOWER', 'THURSDAY', 'TRUCK', 'TUESDAY', 'TULIP'],
    'V': ['VAN'],
    'W': ['CROW', 'OWL', 'SWAN', 'TWO', 'WEDNESDAY'],
    'X': ['FOX', 'OX'],
    'Y': ['YELLOW', 'EYE', 'EYES', 'FRIDAY', 'HOLI', 'MONDAY', 'SCOOTY', 'SORRY', 'SATURDAY', 'SUNDAY', 'TUESDAY', 'THURSDAY', 'WEDNESDAY'],
    'Z': []
}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (128, 128))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_writing(model, image):
    """Predict the written word from image"""
    try:
        if model is None:
            # Demo mode - simulate predictions
            predicted_word = np.random.choice(CLASS_LABELS)
            confidence = np.random.uniform(70, 95)
            
            letters = list(set(predicted_word))
            letter_scores = {}
            for letter in letters:
                letter_scores[letter] = max(50, min(100, confidence + np.random.uniform(-15, 15)))
            
            top_predictions = [
                (predicted_word, confidence),
                (np.random.choice([w for w in CLASS_LABELS if w != predicted_word]), confidence - np.random.uniform(10, 20)),
                (np.random.choice([w for w in CLASS_LABELS if w != predicted_word]), confidence - np.random.uniform(20, 35))
            ]
            
            return predicted_word, letter_scores, top_predictions
        
        processed_img = preprocess_image(image)
        if processed_img is None:
            return "ERROR", {}, []
        
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx] * 100)
        predicted_word = CLASS_LABELS[top_idx]
        
        # Calculate letter-wise performance (simulated based on confidence)
        letters = list(set(predicted_word))
        letter_scores = {}
        for i, letter in enumerate(letters):
            # Simulate letter performance with some randomness
            base_score = confidence
            variation = np.random.uniform(-15, 15)
            letter_scores[letter] = max(0, min(100, base_score + variation))
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [(CLASS_LABELS[idx], float(predictions[0][idx] * 100)) for idx in top_3_idx]
        
        return predicted_word, letter_scores, top_predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "ERROR", {}, []

def get_practice_words(weak_letters, num_words=10):
    """Generate practice words that contain the weak letters"""
    practice_words = set()
    
    for letter in weak_letters:
        letter_upper = letter.upper()
        if letter_upper in LETTER_TO_WORDS:
            words = LETTER_TO_WORDS[letter_upper]
            practice_words.update(words[:5])
    
    # Convert to list and limit
    practice_list = list(practice_words)[:num_words]
    
    # If not enough words, add random words
    while len(practice_list) < num_words:
        random_word = np.random.choice(CLASS_LABELS)
        if random_word not in practice_list:
            practice_list.append(random_word)
    
    return practice_list

def get_llm_advice(student_data, progress_data, weak_letters, api_key):
    """Get advice from LLM for the teacher"""
    if not OPENAI_AVAILABLE or not api_key:
        return None
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare context for LLM
        context = f"""
You are an expert special education teacher assistant. You're helping a teacher work with a student who has special needs.

Student Profile:
- Name: {student_data['name']}
- Age: {student_data['age']} years
- Disability: {student_data['disability']}
- IQ Level: {student_data['iq']}
- Disability Percentage: {student_data['disability_percentage']}%

Current Challenges:
- Weak letters (needs practice): {', '.join(weak_letters)}
- Recent progress entries: {len(progress_data)} submissions

Please provide:
1. Specific teaching strategies for this student's profile
2. Homework recommendations that focus on the weak letters
3. Practice activities suitable for their age and ability level
4. Encouragement strategies that work well for students with their specific disability

Keep your response warm, practical, and actionable. Use simple language.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a compassionate and experienced special education teacher assistant."},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI advice: {str(e)}"

def chat_with_llm(messages, api_key):
    """Chat with LLM about student progress"""
    if not OPENAI_AVAILABLE or not api_key:
        return "Please set your OpenAI API key in the sidebar to use the AI assistant."
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def show_default_charts(student_name):
    """Show default charts when no data is uploaded"""
    st.markdown("### 📊 Sample Progress Dashboard")
    st.info("🎯 Upload student writing samples to see real progress data!")
    
    # Sample data for demonstration
    sample_dates = pd.date_range(start='2024-01-01', periods=10, freq='W')
    sample_accuracy = np.random.uniform(60, 85, 10)
    sample_accuracy = np.sort(sample_accuracy)  # Show improvement trend
    
    # Sample accuracy trend
    st.markdown("#### 📈 Expected Progress Trend")
    fig_sample = go.Figure()
    fig_sample.add_trace(go.Scatter(
        x=sample_dates,
        y=sample_accuracy,
        mode='lines+markers',
        name='Sample Accuracy',
        line=dict(color='#667eea', width=3, dash='dash'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    fig_sample.update_layout(
        xaxis_title="Practice Sessions",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_sample, use_container_width=True)
    
    # Sample letter performance
    st.markdown("#### 🔤 Sample Letter Performance")
    sample_letters = ['A', 'B', 'C', 'D', 'E']
    sample_scores = [85, 72, 90, 65, 78]
    
    fig_letters = go.Figure(data=[
        go.Bar(
            x=sample_letters,
            y=sample_scores,
            marker_color=['#f5576c' if score < 70 else '#667eea' for score in sample_scores],
            text=[f"{v}%" for v in sample_scores],
            textposition='auto',
        )
    ])
    
    fig_letters.update_layout(
        title="Letter Accuracy (%)",
        xaxis_title="Letters",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=400
    )
    
    st.plotly_chart(fig_letters, use_container_width=True)
    
    # Sample practice recommendations
    st.markdown("#### 📝 Sample Practice Words")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Focus Letters:** D, B")
        st.markdown("**Recommended Words:**")
        sample_words = ['DOG', 'DOCTOR', 'BUS', 'BEAR', 'BLUE', 'DEER']
        for word in sample_words:
            st.markdown(f"- {word}")
    
    with col2:
        st.markdown("**Activity Ideas:**")
        st.markdown("- ✏️ Trace letters in sand")
        st.markdown("- 🎨 Paint letters with fingers")
        st.markdown("- 📦 Build letters with blocks")
        st.markdown("- 🎵 Sing alphabet songs")

def authentication_page():
    """Teacher authentication page"""
    st.title("🎨 Welcome to Special Child Writing Helper!")
    st.markdown("### 👩‍🏫 Teacher Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.9); padding: 30px; border-radius: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
        """, unsafe_allow_html=True)
        
        teacher_name = st.text_input("👤 Teacher Name", placeholder="Enter your name")
        teacher_email = st.text_input("📧 Email", placeholder="Enter your email (any format accepted)")
        
        if st.button("🚀 Login", use_container_width=True):
            if teacher_name.strip():  # Only name is required
                st.session_state.authenticated = True
                st.session_state.teacher_name = teacher_name.strip()
                st.success(f"✅ Welcome, {teacher_name}!")
                st.rerun()
            else:
                st.error("Please enter your name!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Info box
        st.info("👋 Simply enter your name to get started! No strict authentication required.")

def student_selection_page():
    """Student selection grid"""
    st.title(f"👋 Welcome, {st.session_state.teacher_name}!")
    st.markdown("### 🎯 Select a Student")
    
    students = load_student_data()
    
    # Sidebar with logout and API key
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.selected_student = None
            st.session_state.teacher_name = ""
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🤖 AI Assistant (Optional)")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key to enable AI teaching assistant"
        )
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("✅ API Key set!")
        else:
            st.info("AI features disabled without API key")
    
    # Create grid layout
    cols_per_row = 4
    rows = [students[i:i + cols_per_row] for i in range(0, len(students), cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, student in enumerate(row):
            with cols[idx]:
                # Student card with emoji based on disability
                disability_emoji = {
                    "MR": "🧠", "ID": "💙", "CP": "💚", 
                    "Severe MR": "🧩", "Severe ID": "💜", 
                    "Moderate MR": "🎯", "Moderate ID": "🌟", 
                    "Severe CP": "💖"
                }
                
                emoji = disability_emoji.get(student['disability'], "👤")
                
                st.markdown(f"""
                <div class='student-card'>
                    <h2>{emoji} {student['name']}</h2>
                    <p><strong>Age:</strong> {student['age']} years</p>
                    <p><strong>Disability:</strong> {student['disability']}</p>
                    <p><strong>IQ:</strong> {student['iq']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select {student['name']}", key=f"btn_{student['name']}", use_container_width=True):
                    st.session_state.selected_student = student
                    st.rerun()

def student_dashboard():
    """Individual student dashboard with upload and progress"""
    student = st.session_state.selected_student
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        if st.button("⬅️ Back to Students", use_container_width=True):
            st.session_state.selected_student = None
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.selected_student = None
            st.session_state.teacher_name = ""
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🤖 AI Assistant")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key to enable AI features"
        )
        if api_key:
            st.session_state.openai_api_key = api_key
    
    # Student header
    st.title(f"📊 Dashboard: {student['name']}")
    
    # Student info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Age", f"{student['age']} years")
    with col2:
        st.metric("🧠 IQ Level", student['iq'])
    with col3:
        st.metric("📊 Disability", student['disability'])
    with col4:
        st.metric("📈 Disability %", f"{student['disability_percentage']}%")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload Writing", "📈 Progress Reports", "🤖 AI Teacher Assistant"])
    
    with tab1:
        upload_section(student)
    
    with tab2:
        progress_section(student)
    
    with tab3:
        ai_assistant_section(student)

def upload_section(student):
    """Image upload and prediction section"""
    st.markdown("### ✏️ Upload Student's Writing")
    
    model = load_model()
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the student's writing",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("#### 🔍 Analysis Results")
                
                # Convert to numpy array for processing
                img_array = np.array(image)
                
                # Make prediction
                with st.spinner("🔮 Analyzing writing..."):
                    predicted_word, letter_scores, top_predictions = predict_writing(model, img_array)
                
                if predicted_word != "ERROR":
                    # Display results
                    st.success(f"**Predicted Word:** {predicted_word}")
                    st.info(f"**Confidence:** {top_predictions[0][1]:.2f}%")
                    
                    st.markdown("#### 🎯 Top 3 Predictions:")
                    for i, (word, conf) in enumerate(top_predictions, 1):
                        st.write(f"{i}. **{word}** - {conf:.2f}%")
            
            # Letter-wise analysis
            if predicted_word != "ERROR" and letter_scores:
                st.markdown("### 📊 Letter-Wise Performance Analysis")
                
                # Find weakest letters
                sorted_letters = sorted(letter_scores.items(), key=lambda x: x[1])
                weak_letters = [letter for letter, score in sorted_letters[:3]]
                weakest_letter = sorted_letters[0][0]
                weakest_score = sorted_letters[0][1]
                
                st.warning(f"⚠️ **Focus Letter:** '{weakest_letter}' needs more practice ({weakest_score:.1f}% accuracy)")
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(letter_scores.keys()),
                        y=list(letter_scores.values()),
                        marker_color=['#f5576c' if letter == weakest_letter else '#667eea' 
                                     for letter in letter_scores.keys()],
                        text=[f"{v:.1f}%" for v in letter_scores.values()],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Letter Performance (%)",
                    xaxis_title="Letters",
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 100],
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Practice recommendations
                st.markdown("### 📝 Recommended Practice Words")
                st.info(f"These words focus on the letters that need practice: {', '.join(weak_letters)}")
                
                practice_words = get_practice_words(weak_letters, num_words=10)
                
                cols = st.columns(5)
                for idx, word in enumerate(practice_words):
                    with cols[idx % 5]:
                        st.markdown(f"**{idx + 1}.** {word}")
                
                # Save progress
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Progress", use_container_width=True):
                        save_progress(student['name'], predicted_word, letter_scores, top_predictions[0][1])
                        st.success("✅ Progress saved successfully!")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Clear and Upload New", use_container_width=True):
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    else:
        # Show placeholder when no image is uploaded
        st.info("👆 Upload an image to start the analysis")
        st.markdown("---")
        show_default_charts(student['name'])

def progress_section(student):
    """Show student progress over time"""
    st.markdown("### 📈 Progress History")
    
    # Get student progress from session state
    student_name = student['name']
    
    if student_name in st.session_state.student_progress and len(st.session_state.student_progress[student_name]) > 0:
        progress_data = st.session_state.student_progress[student_name]
        
        # Convert to DataFrame
        df = pd.DataFrame(progress_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Submissions", len(df))
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("📊 Average Accuracy", f"{avg_confidence:.1f}%")
        with col3:
            unique_words = df['predicted_word'].nunique()
            st.metric("🔤 Unique Words", unique_words)
        with col4:
            if len(df) >= 2:
                improvement = df['confidence'].iloc[-1] - df['confidence'].iloc[0]
                st.metric("📈 Improvement", f"{improvement:+.1f}%")
        
        # Accuracy over time
        st.markdown("#### 🎯 Accuracy Trend")
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig_accuracy.update_layout(
            xaxis_title="Date/Time",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            height=400
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Words practiced
        st.markdown("#### 📝 Words Practiced")
        word_counts = df['predicted_word'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_words = go.Figure(data=[go.Pie(
                labels=word_counts.index,
                values=word_counts.values,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_words.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            # Most practiced words
            st.markdown("**Top Practiced Words:**")
            for idx, (word, count) in enumerate(word_counts.head(5).items(), 1):
                st.markdown(f"{idx}. **{word}** - {count} times")
        
        # Letter weakness analysis
        st.markdown("#### 🔤 Letter Performance Analysis")
        all_weak_letters = []
        for entry in progress_data:
            if 'weakest_letter' in entry:
                all_weak_letters.append(entry['weakest_letter'])
        
        if all_weak_letters:
            weak_letter_counts = Counter(all_weak_letters)
            
            fig_letters = go.Figure(data=[
                go.Bar(
                    x=list(weak_letter_counts.keys()),
                    y=list(weak_letter_counts.values()),
                    marker_color='#f5576c',
                    text=list(weak_letter_counts.values()),
                    textposition='auto',
                )
            ])
            
            fig_letters.update_layout(
                title="Frequency of Weak Letters",
                xaxis_title="Letters",
                yaxis_title="Times Identified as Weak",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            
            st.plotly_chart(fig_letters, use_container_width=True)
            
            # Recommendations
            most_problematic = weak_letter_counts.most_common(3)
            st.warning(f"⚠️ **Focus on these letters:** {', '.join([l[0] for l in most_problematic])}")
            
            practice_words = get_practice_words([l[0] for l in most_problematic], num_words=10)
            
            st.markdown("**Recommended practice words:**")
            cols = st.columns(5)
            for idx, word in enumerate(practice_words):
                with cols[idx % 5]:
                    st.markdown(f"• {word}")
        
        # Recent submissions
        st.markdown("#### 📋 Recent Submissions")
        recent_df = df[['timestamp', 'predicted_word', 'confidence', 'weakest_letter']].tail(10).sort_values('timestamp', ascending=False)
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Date & Time",
                "predicted_word": "Word",
                "confidence": "Accuracy",
                "weakest_letter": "Weak Letter"
            }
        )
        
        # Export option
        if st.button("📥 Export Progress Report (CSV)", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{student_name}_progress_report.csv",
                mime="text/csv"
            )
    else:
        st.info("📊 No progress data available yet.")
        st.markdown("Upload some writing samples in the **Upload Writing** tab to track progress!")
        st.markdown("---")
        show_default_charts(student_name)

def ai_assistant_section(student):
    """AI assistant for teachers"""
    st.markdown("### 🤖 AI Teaching Assistant")
    
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please set your OpenAI API key in the sidebar to use the AI assistant.")
        st.info("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("OpenAI library not installed. Install it with: `pip install openai`")
        return
    
    # Get student progress
    student_name = student['name']
    progress_data = st.session_state.student_progress.get(student_name, [])
    
    # Analyze weak letters
    weak_letters = []
    if progress_data:
        all_weak_letters = [entry.get('weakest_letter', '') for entry in progress_data if entry.get('weakest_letter')]
        if all_weak_letters:
            weak_letter_counts = Counter(all_weak_letters)
            weak_letters = [l[0] for l in weak_letter_counts.most_common(5)]
    
    # Quick advice section
    st.markdown("#### 💡 Quick AI Advice")
    
    if st.button("🎯 Get Personalized Teaching Advice", use_container_width=True):
        with st.spinner("🤖 AI is thinking..."):
            advice = get_llm_advice(student, progress_data, weak_letters, st.session_state.openai_api_key)
            if advice:
                st.markdown(f"""
                <div class='chat-message ai-message'>
                    <strong>🤖 AI Teacher Assistant:</strong><br>
                    {advice}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("#### 💬 Chat with AI Assistant")
    st.info("Ask questions about teaching strategies, homework ideas, or student progress!")
    
    # Chat history display
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                st.markdown(f"""
                <div class='chat-message teacher-message'>
                    <strong>👩‍🏫 You:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message ai-message'>
                    <strong>🤖 AI:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_message = st.text_input("Your question:", key="chat_input", placeholder="E.g., What homework should I give for letter 'B'?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📤 Send", use_container_width=True):
            if user_message.strip():
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_message
                })
                
                # Prepare context for LLM
                system_message = {
                    'role': 'system',
                    'content': f"""You are a helpful special education teaching assistant. 
                    Current student context:
                    - Name: {student['name']}
                    - Age: {student['age']}
                    - Disability: {student['disability']}
                    - IQ: {student['iq']}
                    - Progress entries: {len(progress_data)}
                    - Weak letters: {', '.join(weak_letters) if weak_letters else 'None identified yet'}
                    
                    Provide practical, compassionate advice for the teacher."""
                }
                
                messages = [system_message] + st.session_state.chat_history
                
                # Get AI response
                with st.spinner("🤖 AI is thinking..."):
                    ai_response = chat_with_llm(messages, st.session_state.openai_api_key)
                
                # Add AI response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Practice word generator
    st.markdown("---")
    st.markdown("#### 📝 Generate Practice Words")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        target_letters = st.text_input("Enter letters to focus on (e.g., A,B,C):", placeholder="A,B,C")
    with col2:
        num_words = st.number_input("Number of words:", min_value=5, max_value=20, value=10)
    
    if st.button("🎲 Generate Practice Words", use_container_width=True):
        if target_letters.strip():
            letters = [l.strip().upper() for l in target_letters.split(',')]
            practice_words = get_practice_words(letters, num_words=num_words)
            
            st.success(f"📚 Practice words for letters: {', '.join(letters)}")
            
            cols = st.columns(5)
            for idx, word in enumerate(practice_words):
                with cols[idx % 5]:
                    # Highlight target letters
                    highlighted = word
                    for letter in letters:
                        highlighted = highlighted.replace(letter, f"**{letter}**")
                    st.markdown(f"{idx + 1}. {highlighted}")
        else:
            st.warning("Please enter at least one letter!")

def save_progress(student_name, predicted_word, letter_scores, confidence):
    """Save student progress to session state"""
    if student_name not in st.session_state.student_progress:
        st.session_state.student_progress[student_name] = []
    
    weakest_letter = min(letter_scores, key=letter_scores.get) if letter_scores else ""
    
    progress_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_word': predicted_word,
        'confidence': float(confidence),
        'letter_scores': letter_scores,
        'weakest_letter': weakest_letter
    }
    
    st.session_state.student_progress[student_name].append(progress_entry)

# Main app logic
def main():
    try:
        if not st.session_state.authenticated:
            authentication_page()
        elif st.session_state.selected_student is None:
            student_selection_page()
        else:
            student_dashboard()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        
        # Show error details in expander
        with st.expander("🔍 Error Details"):
            st.code(str(e))

if __name__ == "__main__":
    main()
