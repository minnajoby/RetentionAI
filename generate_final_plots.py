import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import os

# 1. Setup and Data Loading
print("Loading data and models...")
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop(['Exited', 'CreditScore', 'EstimatedSalary', 'HasCrCard'], axis=1)
y = df['Exited']

# Use the same split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- THE SPEED FIX: SAMPLE THE TEST SET ---
# 2000 samples is plenty for a smooth ROC curve and accurate Confusion Matrix
sample_size = 2000
X_test_sample = X_test.sample(sample_size, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

# Load Scaler and Scale the sample
scaler = joblib.load('models/scaler_7.pkl')
X_test_scaled = scaler.transform(X_test_sample)
# Convert back to DataFrame to keep feature names (fixes the warnings)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Load Models
models = {
    "CatBoost": joblib.load('models/catboost_7.pkl'),
    "LightGBM": joblib.load('models/lightgbm_7.pkl'),
    "TabPFN": joblib.load('models/tabpfn_7.pkl')
}

tn = TabNetClassifier()
tn.load_model('models/tabnet_7.zip')
models["TabNet"] = tn

if not os.path.exists('static'): os.makedirs('static')

# 2. Generate Multi-Model ROC Curve
print(f"Generating ROC Curve using {sample_size} samples...")
plt.figure(figsize=(10, 8))
colors = ['#003366', '#d4a017', '#22c55e', '#7c3aed']

for (name, model), color in zip(models.items(), colors):
    print(f"Processing {name}...")
    # Use DataFrame for Boosting/PFN, cast to float32 for TabNet
    if name == "TabNet":
        probs = model.predict_proba(X_test_df.values.astype(np.float32))[:, 1]
    else:
        probs = model.predict_proba(X_test_df)[:, 1]
        
    fpr, tpr, _ = roc_curve(y_test_sample, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC Analysis (Optimized Suite)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('static/roc_curve.png', dpi=150, bbox_inches='tight')

# 3. Generate Confusion Matrix for Champion (CatBoost)
print("Generating Confusion Matrix for CatBoost...")
plt.figure(figsize=(8, 6))
y_pred = models["CatBoost"].predict(X_test_df)
cm = confusion_matrix(y_test_sample, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix: CatBoost Champion')
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')

print("\n--- SUCCESS: Research Figures Generated in seconds ---")