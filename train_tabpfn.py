import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib
import time
import os

# 1. Load Data
print("Loading data for TabPFN...")
df = pd.read_csv('data/Processed_Churn_Data.csv')

# TabPFN is optimized for N <= 10,000. 
df_sample = df.sample(10000, random_state=42)
X = df_sample.drop('Exited', axis=1)
y = df_sample['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize TabPFN with CPU Override
print("Initializing TabPFN Transformer (Foundation Model)...")
# ignore_pretraining_limits=True allows us to run more than 1000 rows on CPU
model_pfn = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)

# 3. Run Inference
print(f"Running TabPFN Inference on {len(X_train)} samples...")
print("Note: This may take 1-2 minutes on CPU...")
start_time = time.time()
model_pfn.fit(X_train, y_train)
end_time = time.time()

# 4. Evaluate
preds = model_pfn.predict(X_test)
score = f1_score(y_test, preds)

print("\n" + "="*30)
print(f"TabPFN F1-Score (10 Features): {score:.4f}")
print(f"Inference Time: {end_time - start_time:.2f}s")
print("="*30)
print(classification_report(y_test, preds))

# 5. Save
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model_pfn, 'models/tabpfn_model.pkl')
print("Model saved to models/tabpfn_model.pkl")