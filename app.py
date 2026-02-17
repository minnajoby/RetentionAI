from flask import Flask, render_template, request
from datetime import datetime
import joblib
import numpy as np
import sqlite3
import os
from pytorch_tabnet.tab_model import TabNetClassifier

app = Flask(__name__)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    # Schema: id, time_stamp, model_used, probability, result
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  time_stamp TEXT,
                  model_used TEXT, 
                  probability TEXT, 
                  result TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- GLOBAL MODELS DICTIONARY ---
models = {}
scaler = None

def load_all_models():
    global scaler
    try:
        scaler = joblib.load('models/scaler.pkl')
        
        if os.path.exists('models/catboost_model.pkl'):
            models["CatBoost"] = joblib.load('models/catboost_model.pkl')
            
        if os.path.exists('models/lightgbm_model.pkl'):
            models["LightGBM"] = joblib.load('models/lightgbm_model.pkl')
            
        if os.path.exists('models/tabnet_model.zip'):
            tn = TabNetClassifier()
            tn.load_model('models/tabnet_model.zip')
            models["TabNet"] = tn
            
        if os.path.exists('models/tabpfn_model.pkl'):
            models["TabPFN"] = joblib.load('models/tabpfn_model.pkl')
            
        print(f"--- SUCCESS: {len(models)} models loaded ---")
    except Exception as e:
        print(f"--- ERROR LOADING MODELS: {e} ---")

load_all_models()

@app.route('/')
def home():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    # FIX: Changed 'age' to 'time_stamp' to match the init_db schema
    c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
    history = c.fetchall()
    conn.close()
    return render_template('index.html', inputs=None, history=history, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        choice = request.form.get('ModelChoice')
        if choice not in models:
            return f"Error: Model '{choice}' not loaded."

        current_model = models[choice]

        feature_list = [
            float(request.form.get('CreditScore')),
            float(request.form.get('Geography')),
            float(request.form.get('Gender')),
            float(request.form.get('Age')),
            float(request.form.get('Tenure')),
            float(request.form.get('Balance')),
            float(request.form.get('NumOfProducts')),
            float(1 if request.form.get('HasCrCard') else 0),
            float(1 if request.form.get('IsActiveMember') else 0),
            float(request.form.get('EstimatedSalary'))
        ]

        final_features = np.array(feature_list).reshape(1, -1)
        
        if choice in ["TabNet", "TabPFN"]:
            final_features = final_features.astype(np.float32)

        scaled_features = scaler.transform(final_features)
        prediction = current_model.predict(scaled_features)
        prob = current_model.predict_proba(scaled_features)[0][1]
        
        result_text = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
        prob_display = f"{prob * 100:.2f}%"

        # Save to Database
        current_time = datetime.now().strftime("%H:%M:%S")
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("INSERT INTO predictions (time_stamp, model_used, probability, result) VALUES (?, ?, ?, ?)",
                (current_time, choice, prob_display, result_text))
        conn.commit()

        # Fetch History
        c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
        history = c.fetchall()
        conn.close()

        return render_template('index.html', 
                               prediction_text=result_text, 
                               prob_text=prob_display, 
                               inputs=request.form, 
                               history=history)

    except Exception as e:
        return f"Error: {str(e)}. Please ensure all fields are filled."

if __name__ == "__main__":
    app.run(debug=True)