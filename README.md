# 🌈 Special Child Writing Helper

An AI-powered Streamlit application designed to help teachers track and improve writing skills for children with special needs. This app analyzes handwriting samples, provides progress tracking, and offers personalized teaching recommendations through an integrated AI assistant.

## ✨ Features

### 📸 Core Features
- **Handwriting Analysis**: Upload images of student writing for AI-powered analysis
- **Letter-wise Performance Tracking**: Identify which letters need more practice
- **Progress Dashboard**: Visualize student improvement over time with interactive charts
- **Default Charts**: See sample progress visualizations even without uploaded data
- **Multi-Student Management**: Easy switching between different students
- **Child-Friendly UI**: Colorful, engaging interface designed for educational settings

### 🤖 AI Teaching Assistant
- **Personalized Advice**: Get AI-generated teaching strategies tailored to each student's profile
- **Interactive Chat**: Ask questions about homework ideas, teaching methods, and student progress
- **Practice Word Generator**: Automatically generate practice words focusing on weak letters
- **Homework Recommendations**: Receive customized homework suggestions based on student performance

### 📊 Analytics & Reporting
- **Accuracy Trends**: Track improvement over time with line charts
- **Word Practice Distribution**: See which words have been practiced most
- **Weak Letter Analysis**: Identify patterns in letter difficulties
- **CSV Export**: Download progress reports for record-keeping

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/ishitaakhatri/synopsis_project.git
cd synopsis_project
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Basic Usage (Without AI Features)
```bash
streamlit run app_enhanced.py
```

The app will open in your default browser at `http://localhost:8501`

### Using AI Features (Optional)
To enable the AI Teaching Assistant:

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Run the app and enter your API key in the sidebar
3. The AI assistant will be activated for personalized teaching advice

## 📚 Usage Guide

### 1. Teacher Login
- Simply enter your name (and optionally email) to get started
- No strict authentication required - designed for ease of use

### 2. Select a Student
- Choose from 16 pre-loaded students with different profiles
- Each student card shows:
  - Name and age
  - Disability type
  - IQ level
  - Disability percentage

### 3. Upload Writing Samples
- Navigate to the "Upload Writing" tab
- Upload clear images of student handwriting (PNG, JPG, JPEG)
- The AI analyzes the writing and provides:
  - Predicted word
  - Confidence score
  - Top 3 predictions
  - Letter-wise performance scores
  - Recommended practice words

### 4. Track Progress
- View the "Progress Reports" tab to see:
  - Accuracy trends over time
  - Most practiced words
  - Weak letter patterns
  - Recent submission history
  - Export progress as CSV

### 5. Use AI Assistant (Optional)
- Navigate to "AI Teacher Assistant" tab
- Get instant personalized teaching advice
- Chat with AI about specific challenges
- Generate practice word lists for target letters

## 📝 Model Setup

### Using Your Own Model
If you have a trained model, place it in the root directory with one of these names:
- `model.h5`
- `model.keras`
- `saved_model/`
- `project_main.h5`

The app will automatically detect and load your model.

### Demo Mode
If no model is found, the app runs in demo mode with simulated predictions. This is perfect for testing the UI and features.

## 🧩 Key Improvements in Enhanced Version

### ✅ Bug Fixes
1. **Robust Error Handling**: Graceful handling of missing models, corrupt images, and API errors
2. **Input Validation**: Proper validation of uploaded files and user inputs
3. **Session State Management**: Fixed state persistence across page interactions
4. **Model Loading**: Multiple fallback paths for model loading
5. **Image Processing**: Better handling of different image formats and sizes

### 🌟 New Features
1. **Default Charts**: Sample visualizations shown when no data is uploaded
2. **AI Teaching Assistant**: Complete LLM integration with OpenAI
3. **Practice Word Generator**: Intelligent word selection based on weak letters
4. **Chat Interface**: Interactive conversation with AI about student progress
5. **Export Functionality**: Download progress reports as CSV
6. **Enhanced Analytics**: More detailed progress tracking and insights

### 🎨 UI/UX Improvements
1. **Better Navigation**: Clearer buttons and flow between sections
2. **Loading States**: Spinners and progress indicators for async operations
3. **Error Messages**: User-friendly error messages with helpful suggestions
4. **Responsive Design**: Better layout on different screen sizes
5. **Visual Feedback**: Success/error notifications for all actions

## 💻 Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **ML/DL**: TensorFlow/Keras for handwriting recognition
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Plotly for interactive charts
- **AI Integration**: OpenAI GPT-3.5-turbo for teaching assistance

## 📁 File Structure

```
synopsis_project/
├── app_enhanced.py              # Enhanced Streamlit application
├── app.py                      # Original application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── project_main.ipynb          # Jupyter notebook (model training)
├── children_writing_data_final.xlsx  # Training data
└── model.h5                    # Trained model (add your own)
```

## 🔒 Privacy & Security

- **Local Data Storage**: All student progress is stored in session state (not persisted)
- **API Key Security**: OpenAI keys are stored only in session (not logged)
- **No External Database**: No data is sent to external servers except OpenAI API calls
- **Optional AI**: App works fully without AI features if privacy is a concern

## 👥 Student Profiles

The app includes 16 pre-configured student profiles with various special needs:
- Mental Retardation (MR)
- Intellectual Disability (ID)
- Cerebral Palsy (CP)
- Various severity levels (Mild, Moderate, Severe)

## 🎯 Recommended Workflow

1. **Login** as teacher
2. **Select** a student from the grid
3. **Upload** writing sample
4. **Review** analysis and weak letters
5. **Save** progress for tracking
6. **Consult** AI assistant for teaching strategies
7. **Generate** practice word list
8. **Repeat** regularly to track improvement

## ❓ Troubleshooting

### App won't start
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt --upgrade

# Check Python version (should be 3.8+)
python --version
```

### Model not loading
- Ensure your model file is in the root directory
- Check that the model is in a supported format (.h5, .keras, or SavedModel)
- The app will work in demo mode if no model is found

### AI features not working
- Verify your OpenAI API key is correct
- Check your internet connection
- Ensure you have API credits available
- Install openai package: `pip install openai`

### Image upload fails
- Ensure image is in PNG, JPG, or JPEG format
- Try reducing image size if it's very large
- Check image isn't corrupted

## 📈 Future Enhancements

- [ ] Persistent database for progress tracking
- [ ] Multi-teacher authentication
- [ ] Parent portal for progress viewing
- [ ] Voice recording for pronunciation practice
- [ ] Gamification elements for student engagement
- [ ] Detailed report generation with recommendations
- [ ] Integration with more LLM providers
- [ ] Mobile app version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is created for educational purposes.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for special education teachers and students**
