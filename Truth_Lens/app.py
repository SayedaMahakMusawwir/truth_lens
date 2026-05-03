from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import pandas as pd
import pickle
import os
import sqlite3
from datetime import datetime

# Deep Learning libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = "truthlens_academic_secret"

# --- 1. CONFIGURATION & MODEL LOADING ---
# Ensure these files are inside your 'model_files' folder
MODEL_PATH = 'model_files/truthlens_model.h5'
TOKENIZER_PATH = 'model_files/tokenizer.pkl'
CONFIG_PATH = 'model_files/config.pkl'

# Load the AI assets
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('truthlens_users.db')
    cursor = conn.cursor()
    # Stores login history for the Admin spreadsheet
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_logs 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       username TEXT, 
                       role TEXT, 
                       timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 3. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        
        # Admin Logic: Any username containing 'admin' is granted privileges
        role = "Administrator" if "admin" in username.lower() else "Standard User"
        
        session['user'] = username
        session['role'] = role
        
        # Log to Database
        conn = sqlite3.connect('truthlens_users.db')
        conn.execute("INSERT INTO user_logs (username, role, timestamp) VALUES (?, ?, ?)",
                     (username, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = ""
    
    # Check if a document was uploaded
    if 'file' in request.files and request.files['file'].filename != '':
        news_text = request.files['file'].read().decode('utf-8')
    else:
        # Get pasted text from the textarea
        news_text = request.form.get('news_text')

    if not news_text or len(news_text) < 10:
        return jsonify({'error': 'Insufficient content for analysis.'})

    # --- NLP PREPROCESSING ---
    # Tokenizing the text based on the training vocabulary
    sequences = tokenizer.texts_to_sequences([news_text])
    # Padding ensures the input length matches the notebook's max_length
    padded = pad_sequences(sequences, maxlen=config['max_length'])
    
    # --- PREDICTION ---
    prediction_score = model.predict(padded)[0][0]
    
    # Logical mapping based on your notebook: 
    # Usually > 0.5 is FAKE and <= 0.5 is REAL in binary sigmoid models
    verdict = "FAKE" if prediction_score > 0.5 else "REAL"
    confidence = round(float(prediction_score * 100) if verdict == "FAKE" else float((1 - prediction_score) * 100), 2)

    return jsonify({
        'verdict': verdict,
        'confidence': confidence
    })

@app.route('/admin/download')
def download_data():
    # Security Check: Only 'Administrator' role can access this
    if session.get('role') != "Administrator":
        return "Access Denied: You do not have permission to view this data.", 403
    
    # Query the database
    conn = sqlite3.connect('truthlens_users.db')
    df = pd.read_sql_query("SELECT * FROM user_logs", conn)
    conn.close()
    
    # Save as Excel
    file_path = "TruthLens_User_Report.xlsx"
    df.to_excel(file_path, index=False)
    
    return send_file(file_path, as_attachment=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Running the server
    app.run(debug=True)