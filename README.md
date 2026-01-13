# RetentionAI - Bank Churn Project 

## Current Progress (Jan 10, 2026)
[x] Initial Research & Synopsis
[x] Environment Setup (Python 3.11, venv)
[x] Data Ingestion (165,034 records)
[x] Data Preprocessing Pipeline
    - Feature Selection (Dropped non-predictive columns)
    - Label Encoding (Geography, Gender)
    - Feature Scaling (StandardScaler)
    - Class Balancing (SMOTE-Tomek Hybrid Resampling)
[x] Model Training (CatBoost SOTA Architecture)
    - Achieved F1-Score: 0.9109
    - Achieved Accuracy: 91%
    - Training Time: ~11 seconds for 2.5 lakh balanced records.

## Results
The model shows high precision (0.93) for churn detection, proving the effectiveness of the hybrid resampling strategy.

## How to Run Preprocessing
1. Activate venv: `venv\Scripts\activate`
2. Run: `python preprocess.py`
3. Run Training: `python train.py`