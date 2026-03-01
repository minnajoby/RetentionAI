import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
X_sample_fast = X.sample(100, random_state=42) # For slow models
X_sample_med = X.sample(500, random_state=42)  # For fast models

# 2. Load all 4 Models
print("Loading models...")
cb = joblib.load('models/catboost_model.pkl')
lgb = joblib.load('models/lightgbm_model.pkl')
# For TabNet, we load it slightly differently
from pytorch_tabnet.tab_model import TabNetClassifier
tn = TabNetClassifier()
tn.load_model('models/tabnet_model.zip')

if not os.path.exists('static'): os.makedirs('static')

# --- FUNCTION TO SAVE PLOT ---
def save_shap_plot(explainer, sample_data, filename, title):
    print(f"Generating {title}...")
    shap_values = explainer.shap_values(sample_data)
    
    # Handle multi-class output format differences
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False)
    plt.title(title)
    plt.savefig(f'static/{filename}', bbox_inches='tight')
    plt.close()

# --- 3. EXPLAIN MODELS ---

# A. CatBoost (Fast)
explainer_cb = shap.TreeExplainer(cb)
save_shap_plot(explainer_cb, X_sample_med, 'shap_catboost.png', "CatBoost XAI (SHAP)")

# B. LightGBM (Fast)
explainer_lgb = shap.TreeExplainer(lgb)
save_shap_plot(explainer_lgb, X_sample_med, 'shap_lightgbm.png', "LightGBM XAI (SHAP)")

# --- C. TabNet (Fixed Logic) ---
print("Explaining TabNet (Deep Learning)...")
# Use 200 samples for a better 'swarm' look
X_sample_tabnet = X.sample(200, random_state=42)
# Use a representative background (medoid of data) for speed + accuracy
background = shap.sample(X, 50) 
explainer_tn = shap.KernelExplainer(tn.predict_proba, background)

# Calculate standard SHAP values (NOT interactions)
shap_values_tn = explainer_tn.shap_values(X_sample_tabnet)

# Handle output format
if isinstance(shap_values_tn, list):
    shap_values_tn = shap_values_tn[1]

plt.figure(figsize=(10, 6))
# 'plot_type="dot"' ensures we get the swarm look
shap.summary_plot(shap_values_tn, X_sample_tabnet, show=False, plot_type="dot")
plt.title("TabNet Feature Importance (SHAP)", pad=20)
plt.savefig('static/shap_tabnet.png', bbox_inches='tight')
plt.close()
print("--- SUCCESS: SHAP plots saved to static/ ---")