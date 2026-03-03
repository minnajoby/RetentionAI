from flask import Flask, render_template, request
from datetime import datetime
import joblib
import numpy as np
import sqlite3
import os
import pandas as pd 
import shap 
import matplotlib.pyplot as plt 
from pytorch_tabnet.tab_model import TabNetClassifier

# --- Set Matplotlib to non-interactive mode for web safety ---
plt.switch_backend('Agg')

app = Flask(__name__)

# --- CONFIGURATION ---
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
os.environ["HF_TOKEN"] = "your_huggingface_token_here" 

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  time_stamp TEXT, model_used TEXT, probability TEXT, result TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- GLOBAL MODELS & SCALER ---
models = {}
scaler_7 = None 

def load_models():
    global scaler_7
    try:
        scaler_7 = joblib.load('models/scaler_7.pkl')
        models["CatBoost"] = joblib.load('models/catboost_7.pkl')
        models["LightGBM"] = joblib.load('models/lightgbm_7.pkl')
        models["TabPFN"] = joblib.load('models/tabpfn_7.pkl')
        
        tn = TabNetClassifier()
        tn.load_model('models/tabnet_7.zip')
        models["TabNet"] = tn
        print("--- ALL 7-FEATURE MODELS AND SCALER READY ---")
    except Exception as e:
        print(f"--- MODEL LOADING ERROR: {e} ---")

load_models()

@app.route('/')
def home():
    conn = sqlite3.connect('history.db'); c = conn.cursor()
    c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
    history = c.fetchall(); conn.close()
    return render_template('index.html', inputs=None, history=history)

@app.route('/predict', methods=['POST'])
def predict():
    conn = sqlite3.connect('history.db'); c = conn.cursor()
    c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
    history = c.fetchall(); conn.close()

    try:
        choice = request.form.get('ModelChoice', 'CatBoost')
        current_model = models.get(choice, models['CatBoost'])

        feature_names = ['Geography','Gender','Age','Tenure','Balance','NumOfProducts','IsActiveMember']
        display_names = {'Geography': 'Location', 'Gender': 'Gender', 'Age': 'Customer Age', 
                         'Tenure': 'Tenure', 'Balance': 'Account Balance', 
                         'NumOfProducts': 'Product Count', 'IsActiveMember': 'Membership Activity'}

        # 1. Validation Loop (No Defaults)
        feature_list = []
        for f in feature_names:
            val = request.form.get(f)
            if val is None or val.strip() == "":
                return render_template('index.html', error_msg="All fields are required.", history=history, inputs=request.form)
            feature_list.append(float(val))

        # 2. Process Data
        raw_features = np.array(feature_list).reshape(1, -1)
        scaled_features = scaler_7.transform(raw_features)
        if choice in ["TabNet", "TabPFN"]: scaled_features = scaled_features.astype(np.float32)

        # 3. Predict
        prediction = current_model.predict(scaled_features)
        prob_value = float(current_model.predict_proba(scaled_features)[0][1])
        res_text = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
        prob_display = f"{prob_value * 100:.2f}%"

        # 4. Local XAI Waterfall Plot
        explainer = shap.TreeExplainer(models["CatBoost"])
        X_explain = pd.DataFrame(scaled_features, columns=feature_names)
        shap_obj = explainer(X_explain)
        
        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap_obj[0], show=False)
        plt.title(f"Risk Factor Decomposition", pad=20, fontsize=10, fontweight='bold')
        plt.savefig('static/local_explanation.png', bbox_inches='tight', dpi=100)
        plt.close()

        # 5. AI Narrative Generation
        vals = shap_obj.values[0]
        impact_dict = dict(zip(feature_names, vals))
        top_risk = max(impact_dict, key=impact_dict.get)
        top_stable = min(impact_dict, key=impact_dict.get)

        risk_reason = f"The primary driver for churn risk is {display_names[top_risk]}."
        stable_reason = f"The strongest factor for retention is {display_names[top_stable]}."

        # 6. Database Log (Including Probability)
        current_time = datetime.now().strftime("%H:%M:%S")
        conn = sqlite3.connect('history.db'); c = conn.cursor()
        c.execute("INSERT INTO predictions (time_stamp, model_used, probability, result) VALUES (?, ?, ?, ?)",
                  (current_time, choice, prob_display, res_text))
        conn.commit()
        c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
        history = c.fetchall(); conn.close()

        return render_template('index.html', prediction_text=res_text, prob_text=prob_display, 
                               inputs=request.form, history=history, risk_reason=risk_reason, stable_reason=stable_reason)

    except Exception as e:
        return render_template('index.html', error_msg=f"System Error: {str(e)}", history=history)

if __name__ == "__main__":
    app.run(debug=True)