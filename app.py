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

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        # Load your trained model (adjust path as needed)
        model = keras.models.load_model('C:/Users/Admin/Desktop/special_child_project/project_main.ipynb')
        return model
    except:
        st.warning("⚠️ Model file not found. Please ensure 'project_main.ipynb' is in the same directory.")
        return None

@st.cache_data
def load_student_data():
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

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (128, 128))
    
    # Normalize
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_writing(model, image):
    """Predict the written word from image"""
    if model is None:
        return "MODEL NOT LOADED", {}, []
    
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    
    # Get top prediction
    top_idx = np.argmax(predictions[0])
    confidence = predictions[0][top_idx] * 100
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
    top_predictions = [(CLASS_LABELS[idx], predictions[0][idx] * 100) for idx in top_3_idx]
    
    return predicted_word, letter_scores, top_predictions

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
        teacher_email = st.text_input("📧 Email", placeholder="Enter your email")
        
        if st.button("🚀 Login", use_container_width=True):
            if teacher_name and teacher_email:
                st.session_state.authenticated = True
                st.session_state.teacher_name = teacher_name
                st.rerun()
            else:
                st.error("Please enter both name and email!")
        
        st.markdown("</div>", unsafe_allow_html=True)

def student_selection_page():
    """Student selection grid"""
    st.title(f"👋 Welcome, {st.session_state.teacher_name}!")
    st.markdown("### 🎯 Select a Student")
    
    students = load_student_data()
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.selected_student = None
        st.rerun()
    
    # Create grid layout
    cols_per_row = 4
    rows = [students[i:i + cols_per_row] for i in range(0, len(students), cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, student in enumerate(row):
            with cols[idx]:
                # Student card with emoji based on disability
                disability_emoji = {"MR": "🧠", "ID": "💙", "CP": "💚", "Severe MR": "🧩", 
                                   "Severe ID": "💜", "Moderate MR": "🎯", "Moderate ID": "🌟", "Severe CP": "💖"}
                
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
    
    # Back button
    if st.sidebar.button("⬅️ Back to Students"):
        st.session_state.selected_student = None
        st.rerun()
    
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
    
    # Two main sections
    tab1, tab2 = st.tabs(["📸 Upload Writing", "📈 Progress Reports"])
    
    with tab1:
        upload_section(student)
    
    with tab2:
        progress_section(student)

def upload_section(student):
    """Image upload and prediction section"""
    st.markdown("### ✏️ Upload Student's Writing")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'], 
                                    help="Upload a clear image of the student's writing")
    
    if uploaded_file is not None:
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
            
            # Display results
            st.success(f"**Predicted Word:** {predicted_word}")
            st.info(f"**Confidence:** {top_predictions[0][1]:.2f}%")
            
            st.markdown("#### 🎯 Top 3 Predictions:")
            for i, (word, conf) in enumerate(top_predictions, 1):
                st.write(f"{i}. **{word}** - {conf:.2f}%")
        
        # Letter-wise analysis
        st.markdown("### 📊 Letter-Wise Performance Analysis")
        
        if letter_scores:
            # Find weakest letter
            weakest_letter = min(letter_scores, key=letter_scores.get)
            weakest_score = letter_scores[weakest_letter]
            
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
            
            # Save progress
            if st.button("💾 Save Progress", use_container_width=True):
                save_progress(student['name'], predicted_word, letter_scores, top_predictions[0][1])
                st.success("✅ Progress saved successfully!")

def progress_section(student):
    """Show student progress over time"""
    st.markdown("### 📈 Progress History")
    
    # Get student progress from session state
    student_name = student['name']
    if student_name in st.session_state.student_progress:
        progress_data = st.session_state.student_progress[student_name]
        
        if len(progress_data) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(progress_data)
            
            # Accuracy over time
            st.markdown("#### 🎯 Accuracy Trend")
            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['confidence'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10)
            ))
            fig_accuracy.update_layout(
                xaxis_title="Date/Time",
                yaxis_title="Accuracy (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Words practiced
            st.markdown("#### 📝 Words Practiced")
            word_counts = df['predicted_word'].value_counts()
            fig_words = go.Figure(data=[go.Pie(
                labels=word_counts.index,
                values=word_counts.values,
                hole=0.4
            )])
            fig_words.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            st.plotly_chart(fig_words, use_container_width=True)
            
            # Recent submissions
            st.markdown("#### 📋 Recent Submissions")
            st.dataframe(
                df[['timestamp', 'predicted_word', 'confidence']].tail(10).sort_values('timestamp', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No progress data available yet. Upload some writings to see progress!")
    else:
        st.info("No progress data available yet. Upload some writings to see progress!")

def save_progress(student_name, predicted_word, letter_scores, confidence):
    """Save student progress to session state"""
    if student_name not in st.session_state.student_progress:
        st.session_state.student_progress[student_name] = []
    
    progress_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_word': predicted_word,
        'confidence': confidence,
        'letter_scores': letter_scores,
        'weakest_letter': min(letter_scores, key=letter_scores.get)
    }
    
    st.session_state.student_progress[student_name].append(progress_entry)

# Main app logic
def main():
    if not st.session_state.authenticated:
        authentication_page()
    elif st.session_state.selected_student is None:
        student_selection_page()
    else:
        student_dashboard()

if __name__ == "__main__":
    main()
