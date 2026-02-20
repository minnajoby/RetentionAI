import pandas as pd
import numpy as np
import joblib
import shap
import os
import time
from pytorch_tabnet.tab_model import TabNetClassifier

# 1. Setup
print("--- Initializing Consensus Feature Selection ---")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
feature_names = X.columns.tolist()

# 2. Load Models
print("Loading models...")
cb = joblib.load('models/catboost_model.pkl')
lgb = joblib.load('models/lightgbm_model.pkl')
pfn = joblib.load('models/tabpfn_model.pkl')
tn = TabNetClassifier()
tn.load_model('models/tabnet_model.zip')

models = {"CatBoost": cb, "LightGBM": lgb, "TabNet": tn, "TabPFN": pfn}

# 3. Calculate Importance
importance_results = {}

# Use small samples for speed
X_sample_tree = X.sample(100, random_state=42)
X_sample_kernel = X.sample(5, random_state=42) 
background = X.median().values.reshape(1, -1)

for name, model in models.items():
    print(f"Processing {name}...")
    
    if name in ["CatBoost", "LightGBM"]:
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_sample_tree)
    else:
        # Wrapper for TabNet
        def predict_fn(data): return model.predict_proba(data.astype(np.float32))
        explainer = shap.KernelExplainer(predict_fn if name == "TabNet" else model.predict_proba, background)
        raw_shap = explainer.shap_values(X_sample_kernel, nsamples=100)

    # --- THE FIX FOR THE 20 COLUMN ERROR ---
    # If it's a list, it's [Class 0, Class 1]. We take index 1 (Churn).
    if isinstance(raw_shap, list):
        s_vals = np.array(raw_shap[1])
    # If it's a 3D array (N, Features, Classes), we take the second class slice
    elif len(np.array(raw_shap).shape) == 3:
        s_vals = np.array(raw_shap)[:, :, 1]
    else:
        s_vals = np.array(raw_shap)

    # Calculate mean absolute importance for exactly 10 features
    avg_imp = np.abs(s_vals).mean(0)
    
    # Ensure it's a flat 1D array of length 10
    importance_results[name] = avg_imp.flatten()

# 4. Create the Table
# We build the DataFrame using the dictionary keys as columns and feature_names as index
importance_df = pd.DataFrame(importance_results, index=feature_names)

# Calculate the final Consensus Score
importance_df['Consensus_Score'] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values(by='Consensus_Score', ascending=False)

print("\n" + "="*50)
print("SOTA CONSENSUS FEATURE RANKING")
print("="*50)
print(importance_df)
print("="*50)

# Save for your report
if not os.path.exists('static'): os.makedirs('static')
importance_df.to_csv('static/feature_importance_consensus.csv')
print("Table saved to static/feature_importance_consensus.csv")