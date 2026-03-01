import pandas as pd
import numpy as np
import joblib
import torch
import os
import time
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tabpfn import TabPFNClassifier

# --- 1. SETUP & ENVIRONMENT ---
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
# Ensure your HF_TOKEN is set in your system environment or paste it here:
# os.environ["HF_TOKEN"] = "your_token_here"

print("--- Initializing Final 7-Feature Optimized Training ---")

# 2. LOAD DATA
df = pd.read_csv('data/Processed_Churn_Data.csv')

# 3. FEATURE PRUNING (The XAI Feedback Loop)
# We drop the 3 features SHAP identified as noise: CreditScore, EstimatedSalary, HasCrCard
df_optimized = df.drop(['CreditScore', 'EstimatedSalary', 'HasCrCard'], axis=1)
X_full = df_optimized.drop('Exited', axis=1)
y_full = df_optimized['Exited']

print(f"Target Features: {X_full.columns.tolist()}")

# --- 4. MODEL 1: CATBOOST (The Champion) ---
print("\nTraining Optimized CatBoost...")
cb_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)
cb_model.fit(X_full, y_full)
joblib.dump(cb_model, 'models/catboost_7.pkl')
print("Saved: models/catboost_7.pkl")

# --- 5. MODEL 2: LIGHTGBM (The Speed King) ---
print("Training Optimized LightGBM...")
lgb_model = LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, verbose=-1)
lgb_model.fit(X_full, y_full)
joblib.dump(lgb_model, 'models/lightgbm_7.pkl')
print("Saved: models/lightgbm_7.pkl")

# --- 6. MODEL 3: TABNET (Deep Learning) ---
print("Training Optimized TabNet...")
# TabNet requires float32 and int64
X_tn = X_full.values.astype(np.float32)
y_tn = y_full.values.astype(np.int64)

tn_model = TabNetClassifier(verbose=0)
tn_model.fit(X_tn, y_tn, max_epochs=15, batch_size=1024, virtual_batch_size=64)
tn_model.save_model('models/tabnet_7') # Saves as .zip
print("Saved: models/tabnet_7.zip")

# --- 7. MODEL 4: TABPFN (Transformer Foundation Model) ---
print("Running Optimized TabPFN...")
# THE FIX: Aligned Sampling (Sample from DF first, then split)
# We take 2000 rows for the In-Context Learning context
df_sample = df_optimized.sample(2000, random_state=42)
X_pfn_sample = df_sample.drop('Exited', axis=1)
y_pfn_sample = df_sample['Exited']

pfn_model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
pfn_model.fit(X_pfn_sample, y_pfn_sample) # Rows are now perfectly matched
joblib.dump(pfn_model, 'models/tabpfn_7.pkl')
print("Saved: models/tabpfn_7.pkl")

print("\n" + "="*40)
print("SUCCESS: ALL 4 SOTA MODELS RETRAINED & SAVED")
print("="*40)