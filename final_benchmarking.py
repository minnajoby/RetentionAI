import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tabpfn import TabPFNClassifier
import torch

# 1. Setup
print("--- Initializing 4-Model SOTA Benchmarking (5-Fold CV) ---")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
y = df['Exited']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(model_name, model_obj, data_x, data_y, is_expensive=False):
    print(f"\nEvaluating {model_name}...")
    fold_scores = []
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(data_x, data_y), 1):
        X_train, X_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]
        
        # --- TABPFN / TABNET RESEARCH SHORTCUT ---
        # If model is expensive, we use a representative sample of 2000 rows per fold
        if is_expensive:
            sample_idx = np.random.choice(len(X_train), 2000, replace=False)
            X_train, y_train = X_train.iloc[sample_idx], y_train.iloc[sample_idx]

        # Data Type Fix for Deep Learning
        if model_name in ["TabNet", "TabPFN"]:
            X_train = X_train.values.astype(np.float32)
            X_test = X_test.values.astype(np.float32)
            y_train = y_train.values.astype(np.int64)
            y_test = y_test.values.astype(np.int64)

        # Training
        if model_name == "TabNet":
            model_obj.fit(X_train, y_train, max_epochs=10, patience=2, batch_size=1024, virtual_batch_size=64)
        else:
            model_obj.fit(X_train, y_train)
            
        preds = model_obj.predict(X_test)
        score = f1_score(y_test, preds)
        fold_scores.append(score)
        print(f"  Fold {fold}: F1 = {score:.4f}")
        
    avg_score = np.mean(fold_scores)
    std_dev = np.std(fold_scores)
    total_time = time.time() - start_time
    return avg_score, std_dev, total_time

# 2. Define Models
models_to_test = [
    ("CatBoost", CatBoostClassifier(iterations=500, verbose=0), False),
    ("LightGBM", LGBMClassifier(n_estimators=500, verbose=-1), False),
    ("TabNet", TabNetClassifier(verbose=0), True), # Marked as expensive
    ("TabPFN", TabPFNClassifier(device='cpu', ignore_pretraining_limits=True), True) # Marked as expensive
]

# 3. Execute Benchmarking
results = []
for name, obj, expensive in models_to_test:
    avg, std, duration = run_cv(name, obj, X, y, is_expensive=expensive)
    results.append({"Model": name, "Mean F1": avg, "Std Dev": std, "Time (s)": duration})

# 4. Final Leaderboard
report = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL SOTA BENCHMARKING REPORT (10-FEATURE BASELINE)")
print("="*60)
print(report.to_string(index=False))
print("="*60)

report.to_csv('static/final_benchmarking_results.csv', index=False)