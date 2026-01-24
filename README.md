# RetentionAI - Bank Churn Project 

## Current Progress (Updated Jan 21, 2026)
- [x] **Milestone 1 Completion:** Data Engineering & Baseline CatBoost Model (91% F1).
- [x] **Methodology Standardization:** Transitioned to **Agile/Scrum framework** with iterative Sprints.
- [x] **Academic Alignment:** Standardized all project references to **IEEE Format**.
- [x] **Research Framework:** Defined comparative benchmarking metrics for SOTA models (CatBoost, LightGBM, TabNet, TabPFN).

## Technical Pipeline
- **Data Ingestion:** 165,034 records (Kaggle S4E1 Synthetic Dataset).
- **Preprocessing:** 
    - Feature Selection & Noise Reduction.
    - **SMOTE-Tomek Hybrid Resampling** (Balanced training set to ~2.5 Lakh records).
    - Z-score Normalization (StandardScaler).

## System Architecture
- **Modular Design:** Refined the pipeline into four distinct layers (Ingestion, Modeling, XAI, Deployment).
- **Feature Selection Loop:** Implemented a dashed feedback loop logic where XAI insights are used to iteratively refine the Preprocessing stage.
- **Redundancy Removal:** Streamlined the architectural flow to ensure a unified data path.

## Research Methodology (Sprint 2 Planning)
The project now follows an iterative Scrum approach:
  **Sprint 1:** Baseline MVP (CatBoost + Flask Integration) - **Completed**.
  **Sprint 2:** Comparative Analysis (LightGBM & TabNet implementation) - **In Progress**.
  **Sprint 3:** Foundation Modeling (TabPFN) & XAI (SHAP) - **Planned**.

## How to Run
1. Activate venv: `venv\Scripts\activate`
2. Install requirements: `pip install -r requirements.txt`
3. Run Preprocessing: `python preprocess.py`
4. Run Baseline Training: `python train.py`