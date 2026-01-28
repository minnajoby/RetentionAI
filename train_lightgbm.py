import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib
import time

# 1. Load data
df = pd.read_csv('data/Processed_Churn_Data.csv')
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train LightGBM
print("Training LightGBM...")
start_time = time.time()

model_lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

end_time = time.time()

# 3. Evaluate
preds = model_lgbm.predict(X_test)
print(f"LightGBM F1-Score: {f1_score(y_test, preds):.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")
print(classification_report(y_test, preds))

# 4. Save
joblib.dump(model_lgbm, 'models/lightgbm_model.pkl')
print("Model saved as models/lightgbm_model.pkl")