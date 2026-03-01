import pandas as pd
import joblib
import numpy as np

# 1. Check the Training Order
df = pd.read_csv('data/Processed_Churn_Data.csv')
# This is the order we used in train_final_7_feature.py
train_cols = df.drop(['Exited', 'CreditScore', 'EstimatedSalary', 'HasCrCard'], axis=1).columns.tolist()

# 2. Check the App Order
# This is the order we used in app.py
app_cols = ['Geography','Gender','Age','Tenure','Balance','NumOfProducts','IsActiveMember']

print("--- FEATURE INTEGRITY AUDIT ---")
print(f"Training Order: {train_cols}")
print(f"App Order:      {app_cols}")

if train_cols == app_cols:
    print("\n✅ SUCCESS: Feature order is perfectly aligned.")
else:
    print("\n❌ WARNING: Feature order mismatch detected!")
    print("You must fix the order in app.py to match the training order.")

# 3. Check Scaler
scaler = joblib.load('models/scaler_7.pkl')
print(f"\nScaler expects {scaler.n_features_in_} features.")
if scaler.n_features_in_ == 7:
    print("✅ SUCCESS: Scaler is correct for 7 features.")
else:
    print("❌ ERROR: Scaler is still expecting 10 features!")