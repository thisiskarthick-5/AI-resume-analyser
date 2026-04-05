# AI-Based Resume Screening System 🚀

A professional final-year project that utilizes Deep Learning (PyTorch) to automate the recruitment process by predicting job roles and extracting key technical skills from resumes.

## ✨ Live Demo
**[Visit Live on Render](https://ai-resume-analyser-z2qg.onrender.com)**

## ✨ Features
- **AI-Powered Prediction**: Uses a PyTorch Neural Network with an Embedding layer to categorize resumes into roles (Data Scientist, Web Developer, Java Developer, HR).
- **Skill Matrix Extraction**: Automatically identifies technical keywords and calculates a match percentage against job requirements.
- **Modern Dashboard**: A premium, dark-themed UI built with Glassmorphism principles and Inter typography.
- **PDF Processing**: Seamless text extraction from PDF files using PyPDF2.
- **Robust Backend**: Flask API with global error handling and secure file management.

## 🛠️ Technology Stack
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (ES6+)
- **Backend**: Flask, Flask-CORS
- **AI/ML**: PyTorch, Pandas, Scikit-learn
- **PDF Utils**: PyPDF2

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Training (Optional)
The project comes with a pre-trained model. To retrain it on the sample dataset:
```bash
python train_model.py
```

### 4. Running the Application
Launch the Flask server:
```bash
python app.py
```

### 5. Access the Dashboard
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## 📂 Project Structure
- `app.py`: Main Flask API & Frontend Server.
- `train_model.py`: Training script for the AI model.
- `index.html`: Modern frontend dashboard.
- `model/`: Saved weights, vocabulary, and label mappings.
- `utils/`: Text processing and PDF extraction logic.
- `data/`: Sample resume dataset for training.

## 📝 License
This project was built as a complete final-year project demonstration.

---
*Built with ❤️ by AI Engineer*
