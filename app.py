from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import torch
import torch.nn.functional as F
from model.neural_network import ResumeClassifier
from utils.pdf_utils import extract_text_from_pdf
from utils.text_processor import clean_text, tokenize, text_to_sequence, extract_skills, calculate_match_score, SKILLS_DB

app = Flask(__name__)
# Explicitly allowing common headers for JSON POSTs
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/')
def index():
    return send_file('index.html')

# Load Model Artifacts
MODEL_PATH = 'model/resume_model.pth'
VOCAB_PATH = 'model/vocab.json'
LABEL_PATH = 'model/label_encoder.json'

# Global variables for model/vocab
model = None
word_to_idx = None
idx_to_label = None

def load_resources():
    global model, word_to_idx, idx_to_label
    try:
        with open(VOCAB_PATH, 'r') as f:
            word_to_idx = json.load(f)

        with open(LABEL_PATH, 'r') as f:
            label_to_idx = json.load(f)
            idx_to_label = {int(v): k for k, v in label_to_idx.items()}

        vocab_size = len(word_to_idx)
        num_classes = len(label_to_idx)
        model = ResumeClassifier(vocab_size=vocab_size, num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Model and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

# Temporary upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if hasattr(e, 'code'):
        return jsonify({"error": str(e)}), e.code
    # Handle non-HTTP exceptions
    print(f"CRITICAL SYSTEM ERROR: {e}")
    return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Received upload request...")
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and file.filename.endswith('.pdf'):
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(file_path)
            print(f"File saved to: {file_path}")
            return jsonify({"message": "File uploaded!", "path": file_path}), 200
        
        return jsonify({"error": "Only PDFs allowed"}), 400
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
        
    try:
        print("Received predict request...")
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        file_path = data.get('path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": f"File not found on server"}), 404
        
        # 1. Extract Text
        try:
            raw_text = extract_text_from_pdf(file_path)
        except Exception as e:
            return jsonify({"error": f"PDF extraction failed: {str(e)}"}), 500

        if not raw_text or len(raw_text.strip()) < 10:
            return jsonify({"error": "Insufficient text found in PDF. Please use a text-based resume."}), 400
        
        # 2. Prediction
        cleaned_txt = clean_text(raw_text)
        tokens = tokenize(cleaned_txt)
        seq = text_to_sequence(tokens, word_to_idx, max_len=100)
        
        input_tensor = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            role_idx = torch.argmax(probabilities).item()
            predicted_role = idx_to_label[role_idx]
        
        # 3. Skills
        extracted_skills = extract_skills(raw_text)
        role_required_skills = SKILLS_DB.get(predicted_role, [])
        match_score = calculate_match_score(extracted_skills, role_required_skills)
        
        return jsonify({
            "role": predicted_role,
            "skills": extracted_skills,
            "match_score": f"{match_score}%"
        })
    except Exception as e:
        print(f"Prediction logic error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Use reloader=False to avoid port conflicts during rapid restarts in dev
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
