import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 1. Load Data and Model
print("Loading TabPFN model...")
pfn = joblib.load('models/tabpfn_model.pkl')
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)

# 2. EMERGENCY SAMPLING (Very small for speed)
# 10 samples will take about 5-10 minutes on a standard CPU
X_sample_pfn = X.sample(10, random_state=42) 
background = shap.sample(X, 5) # Tiny background for speed

# 3. Initialize Explainer
print("Initializing KernelExplainer for TabPFN...")
explainer_pfn = shap.KernelExplainer(pfn.predict_proba, background)

# 4. Calculate SHAP
print("Generating TabPFN Plot (10 samples)... Please wait.")
shap_values_pfn = explainer_pfn.shap_values(X_sample_pfn)

# Handle output format (Select Class 1: Churn)
if isinstance(shap_values_pfn, list):
    shap_values_pfn = shap_values_pfn[1]

# 5. Save the Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_pfn, X_sample_pfn, show=False, plot_type="dot")
plt.title("TabPFN Feature Importance (SHAP - Sampled)", pad=20)
plt.savefig('static/shap_tabpfn.png', bbox_inches='tight')
plt.close()

print("--- SUCCESS: TabPFN XAI Plot saved to static/shap_tabpfn.png ---")