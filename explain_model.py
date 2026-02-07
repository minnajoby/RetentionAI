import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 1. Load both models and the data
print("Loading models and data...")
model_cb = joblib.load('models/catboost_model.pkl')
model_lgb = joblib.load('models/lightgbm_model.pkl')
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)

# Use a sample for speed
X_sample = X.sample(1000, random_state=42)

# 2. Explain CatBoost
print("Explaining CatBoost...")
explainer_cb = shap.TreeExplainer(model_cb)
shap_cb = explainer_cb.shap_values(X_sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_cb, X_sample, show=False)
plt.title("CatBoost Feature Importance (SHAP)")
plt.savefig('static/shap_catboost.png', bbox_inches='tight')
plt.close()

# 3. Explain LightGBM
print("Explaining LightGBM...")
explainer_lgb = shap.TreeExplainer(model_lgb)
shap_lgb = explainer_lgb.shap_values(X_sample)

# Note: LightGBM SHAP output is sometimes a list for multiclass, 
# for binary we take the values for class 1
if isinstance(shap_lgb, list):
    shap_lgb = shap_lgb[1]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_lgb, X_sample, show=False)
plt.title("LightGBM Feature Importance (SHAP)")
plt.savefig('static/shap_lightgbm.png', bbox_inches='tight')
plt.close()

print("--- SUCCESS: Comparative SHAP plots saved to static/ folder ---")

# 4. Print Comparison Table
def get_importance(shap_vals, features):
    vals = np.abs(shap_vals).mean(0)
    return pd.DataFrame(list(zip(features, vals)), columns=['Feature','Importance']).sort_values(by='Importance', ascending=False)

imp_cb = get_importance(shap_cb, X.columns)
imp_lgb = get_importance(shap_lgb, X.columns)

print("\nCatBoost Top Features:\n", imp_cb.head(5))
print("\nLightGBM Top Features:\n", imp_lgb.head(5))