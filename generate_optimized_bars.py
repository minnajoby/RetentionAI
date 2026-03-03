import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import torch
from dotenv import load_dotenv
from pytorch_tabnet.tab_model import TabNetClassifier

load_dotenv()
# --- 1. RESEARCH SETUP ---
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
# Ensure your token is here
hf_token = os.getenv("HF_TOKEN")

print("--- Generating 7-Feature Optimized Bar Charts ---")
df = pd.read_csv('data/Processed_Churn_Data.csv')
# The 7 features currently in your optimized system
X_7 = df.drop(['Exited', 'CreditScore', 'EstimatedSalary', 'HasCrCard'], axis=1)
feature_names = X_7.columns.tolist()

# 2. Load the 7-Feature Models
print("Loading models...")
cb = joblib.load('models/catboost_7.pkl')
lgb = joblib.load('models/lightgbm_7.pkl')
pfn = joblib.load('models/tabpfn_7.pkl')
tn = TabNetClassifier(); tn.load_model('models/tabnet_7.zip')

def save_optimized_bar(model_func, name, filename, is_tree=False, tree_model=None):
    print(f"Analyzing {name}...")
    try:
        if is_tree:
            # TreeExplainer is exact and fast
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_7.sample(100, random_state=42))
        else:
            # KernelExplainer with Median Background for speed
            background = X_7.median().values.reshape(1, -1)
            explainer = shap.KernelExplainer(model_func, background)
            # Small sample to ensure it finishes in minutes
            shap_values = explainer.shap_values(X_7.sample(10 if "PFN" in name else 30, random_state=42), nsamples=100)

        # Handle output formats
        if isinstance(shap_values, list): shap_values = shap_values[1]
        # Fix for TabPFN 3D array (samples, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: 
            shap_values = shap_values[:, :, 1]

        # Calculate Mean Absolute Importance
        importances = np.abs(shap_values).mean(0)
        indices = np.argsort(importances)

        # --- MANUAL MATPLOTLIB PLOT (Guarantees 7 features) ---
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], color='#d4a017', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12, fontweight='bold')
        
        plt.xlabel("Mean Absolute SHAP Value (Global Impact)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        plt.savefig(f'static/{filename}', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Successfully saved {filename}")
        
    except Exception as e:
        print(f"Error for {name}: {e}")

# --- 3. EXECUTION ---
# Note the new filenames with '_7'
save_optimized_bar(None, "CatBoost", "shap_catboost_7.png", is_tree=True, tree_model=cb)
save_optimized_bar(None, "LightGBM", "shap_lightgbm_7.png", is_tree=True, tree_model=lgb)
save_optimized_bar(lambda d: tn.predict_proba(d.astype(np.float32)), "TabNet", "shap_tabnet_7.png")
save_optimized_bar(lambda d: pfn.predict_proba(d), "TabPFN", "shap_tabpfn_7.png")

print("\n--- ALL OPTIMIZED PLOTS SAVED ---")