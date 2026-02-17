import pandas as pd
import numpy as np
import joblib
import shap
import time
from pytorch_tabnet.tab_model import TabNetClassifier

# 1. Load Data and Models
print("Loading data...")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
feature_names = X.columns.tolist()

# Load all 4 models
models = {
    "CatBoost": joblib.load('models/catboost_model.pkl'),
    "LightGBM": joblib.load('models/lightgbm_model.pkl'),
    "TabPFN": joblib.load('models/tabpfn_model.pkl')
}
tn = TabNetClassifier(); tn.load_model('models/tabnet_model.zip')
models["TabNet"] = tn

print("--- Initializing Fast Consensus Audit ---")
all_importances = []

# We use the Median as a single background point to speed up KernelExplainer
background_median = X.median().values.reshape(1, -1)

for name, model in models.items():
    start = time.time()
    print(f"Auditing {name}...", end=" ", flush=True)
    
    if name in ["CatBoost", "LightGBM"]:
        # TreeExplainer is already fast
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.sample(100, random_state=42))
    else:
        # KernelExplainer for TabNet/TabPFN using the Median Trick
        # We only need 10 samples to get a clear ranking consensus
        explainer = shap.KernelExplainer(model.predict_proba, background_median)
        sv = explainer.shap_values(X.sample(10, random_state=42), nsamples=100)
    
    if isinstance(sv, list): sv = sv[1]
    
    # Calculate Importance (Mean Absolute SHAP)
    importance = np.abs(sv).mean(0)
    # Normalize (0 to 1) so we can average different models fairly
    importance = importance / np.sum(importance)
    all_importances.append(importance)
    
    print(f"Done ({time.time() - start:.2f}s)")

# 2. Calculate the Consensus Score
consensus_matrix = np.array(all_importances)
mean_importance = np.mean(consensus_matrix, axis=0)

# 3. Create Results Table
results = pd.DataFrame({
    'Feature': feature_names,
    'Consensus_Score': mean_importance
}).sort_values(by='Consensus_Score', ascending=False)

print("\n" + "="*50)
print("SOTA CONSENSUS FEATURE RANKING")
print("="*50)
print(results)
print("="*50)

# 4. Save the Top 7
top_7 = results.head(7)['Feature'].tolist()
joblib.dump(top_7, 'models/selected_features.pkl')
print(f"\n✅ TOP 7 FEATURES SAVED: {top_7}")