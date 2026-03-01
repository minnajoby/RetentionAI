import pandas as pd
import numpy as np

# Load the processed data
df = pd.read_csv('data/Processed_Churn_Data.csv')

print("--- PREPROCESSING VERIFICATION ---")

# 1. Check Class Balance (SMOTE-Tomek result)
print(f"\n1. Class Distribution (Should be 50/50):")
print(df['Exited'].value_counts())

# 2. Check Feature Scaling (StandardScaler result)
# Scaled data should have a mean near 0 and standard deviation of 1
print(f"\n2. Scaling Check (Mean should be ~0, Std should be ~1):")
print(f"Mean: {df['CreditScore'].mean():.4f}")
print(f"Std:  {df['CreditScore'].std():.4f}")

# 3. Check for Categorical columns
# These should no longer contain text like 'France' or 'Male'
print(f"\n3. Data Types (Should all be float or int):")
print(df.dtypes.head(5))

# 4. Check for dropped columns
forbidden = ['id', 'CustomerId', 'Surname']
found = [c for c in forbidden if c in df.columns]
print(f"\n4. Privacy Check:")
if not found:
    print("Success: ID and Surnames were correctly removed.")
else:
    print(f"Warning: {found} still exist in data.")