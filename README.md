# 🏦 RetentionAI: Enterprise Bank Churn Analytics

RetentionAI is a high-performance, end-to-end machine learning pipeline designed to predict customer churn in the banking sector. It leverages State-of-the-Art (SOTA) algorithms—including Gradient Boosting and Foundation Models—coupled with eXplainable AI (XAI) to provide actionable insights for customer retention.

---

## 🚀 Key Features

- **Multi-Model Prediction:** Utilizes CatBoost, LightGBM, TabNet, and TabPFN for robust churn assessment.
- **eXplainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) for local and global model transparency.
- **Personalized Retention Offers:** Dynamic recommendation engine that suggests targeted retention strategies based on individual risk factors.
- **Automated Reporting:** Generates professional, executive-ready PDF risk reports with visual SHAP breakdowns.
- **Real-time Dashboard:** A responsive Flask-based web interface for single-customer churn scoring and history tracking.
- **Agile Methodology:** Developed using an iterative Sprint-based Scrum framework.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, SQLite3 |
| **Machine Learning** | CatBoost, LightGBM, PyTorch-TabNet, TabPFN |
| **Explainability** | SHAP, Matplotlib |
| **Data Engineering** | Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE-Tomek) |
| **Reporting** | ReportLab |
| **Frontend** | HTML5 (Templates), CSS3 (Modern/Glassmorphism) |

---

## 🏆 Model Benchmarking (SOTA Leaderboard)

*Updated as of Jan 30, 2026*

| Rank | Architecture | F1-Score | Latency | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **CatBoost** | **91.09%** | 10.7s | **Champion** |
| 2 | LightGBM | 91.05% | 4.3s | Optimized |
| 3 | TabNet | *Evaluating* | - | Research |
| 4 | TabPFN | *Evaluating* | - | Research |

---

## 📂 Project Structure

```text
RetentionAI/
├── app.py                # Main Flask application & PDF generation logic
├── preprocess.py         # Data engineering & SMOTE-Tomek resampling pipeline
├── train.py              # Baseline model training scripts
├── models/               # Serialized model (.pkl, .zip) and scalers
├── static/               # CSS, JavaScript, and generated XAI plots
├── templates/            # Flask HTML templates (index.html, etc.)
├── data/                 # Raw and processed datasets
└── requirements.txt      # Project dependencies
```

---

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/yourusername/RetentionAI.git
cd RetentionAI

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pipeline Execution
Before running the web app, ensure the data is processed and models are trained:

```bash
# 1. Engineering & Resampling
python preprocess.py

# 2. Model Training (CatBoost/LightGBM)
python train.py
```

### 4. Running the Application
```bash
python app.py
```
Access the dashboard at `http://127.0.0.1:5000/`.

---

## 📖 Research Framework
This project follows standardized **Academic/Industry Alignment** utilizing **IEEE formatting** for documentation and **Agile/Scrum** for development lifecycles. The architecture features a dashed feedback loop where **XAI insights** inform the **Preprocessing layer** to iteratively reduce feature noise and improve predictive precision.

---

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.