import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time

# 1. Load data
print("Loading data...")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
y = df['Exited']

# 2. Define the models to compare
models = {
    "CatBoost": CatBoostClassifier(iterations=500, verbose=0),
    "LightGBM": LGBMClassifier(n_estimators=500, verbose=-1)
}

# 3. Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"--- Starting Comparative 5-Fold Cross-Validation ---")

results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    f1_scores = []
    start_time = time.time()
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        score = f1_score(y_test, preds)
        f1_scores.append(score)
        print(f"  Fold {fold}: F1 = {score:.4f}")
        fold += 1
    
    end_time = time.time()
    results[name] = {
        "Mean F1": np.mean(f1_scores),
        "Std Dev": np.std(f1_scores),
        "Total Time": end_time - start_time
    }

# 4. Final Comparison Table
print("\n" + "="*40)
print("FINAL COMPARATIVE VALIDATION RESULTS")
print("="*40)
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Mean F1-Score: {metrics['Mean F1']:.4f}")
    print(f"  Stability (Std Dev): {metrics['Std Dev']:.4f}")
    print(f"  Total Val Time: {metrics['Total Time']:.2f}s")
    print("-" * 20)