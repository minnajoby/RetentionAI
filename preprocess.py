import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek
import joblib
import os

def run_advanced_preprocessing():
    # 1. DATA INGESTION
    print("Step 1: Loading Dataset (165,034 records)...")
    if not os.path.exists('data/Churn_Modelling.csv'):
        print("Error: CSV file not found in data/ folder!")
        return
    df = pd.read_csv('data/Churn_Modelling.csv')

    # 2. FEATURE SELECTION (Dropping Identifiers)
    # Removing columns that have no predictive power to avoid 'Noise'
    print("Step 2: Dropping non-predictive columns (id, CustomerId, Surname)...")
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    # 3. CATEGORICAL ENCODING (Label Encoding)
    # Converting text to numbers so the mathematical models can process them
    print("Step 3: Encoding Geography and Gender...")
    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography']) 
    df['Gender'] = le_gen.fit_transform(df['Gender'])

    # 4. FEATURE SCALING (Standardization)
    # Scaling numerical data to Mean=0 and StdDev=1 using StandardScaler
    print("Step 4: Standardizing numerical features...")
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Convert back to DataFrame to keep column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 5. CLASS IMBALANCE HANDLING (SMOTE-Tomek)
    # Using a hybrid approach to balance the 80/20 ratio of the target classes
    print("Step 5: Applying SMOTE-Tomek (This takes about 60 seconds)...")
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_scaled_df, y)

    # 6. MODEL PERSISTENCE (Saving for Flask Deployment)
    # Saving the 'tools' so the website can use the same logic later
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(le_geo, 'models/le_geo.pkl')
    joblib.dump(le_gen, 'models/le_gen.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save the processed data for the training script
    X_resampled['Exited'] = y_resampled
    X_resampled.to_csv('data/Processed_Churn_Data.csv', index=False)

    print("-" * 40)
    print("PREPROCESSING COMPLETE")
    print(f"Original Row Count: {len(df)}")
    print(f"Balanced Row Count: {len(X_resampled)}")
    print("Saved: le_geo.pkl, le_gen.pkl, scaler.pkl, Processed_Churn_Data.csv")
    print("-" * 40)

if __name__ == "__main__":
    run_advanced_preprocessing()