import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tabpfn import TabPFNClassifier

# 1. Load Data and Prune (Drop the bottom 3 from your consensus table)
df = pd.read_csv('data/Processed_Churn_Data.csv')
X_7 = df.drop(['Exited', 'CreditScore', 'EstimatedSalary', 'HasCrCard'], axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X_7, y, test_size=0.2, random_state=42)

print("--- Calculating 'After XAI' Scores (7 Features) ---")

# A. CatBoost
cb = CatBoostClassifier(iterations=500, verbose=0).fit(X_train, y_train)
print(f"CatBoost (7-Feat) F1: {f1_score(y_test, cb.predict(X_test)):.4f}")

# B. LightGBM
lgb = LGBMClassifier(n_estimators=500, verbose=-1).fit(X_train, y_train)
print(f"LightGBM (7-Feat) F1: {f1_score(y_test, lgb.predict(X_test)):.4f}")

# C. TabNet
tn = TabNetClassifier(verbose=0)
tn.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int64), max_epochs=5)
print(f"TabNet   (7-Feat) F1: {f1_score(y_test, tn.predict(X_test.values.astype(np.float32))):.4f}")

# D. TabPFN (Sampled)
pfn = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
pfn.fit(X_train.sample(2000), y_train.sample(2000))
print(f"TabPFN   (7-Feat) F1: {f1_score(y_test, pfn.predict(X_test)):.4f}")