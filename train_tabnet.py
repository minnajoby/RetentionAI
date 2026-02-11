import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric # <--- Added for custom F1
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import time
import os

# --- CUSTOM F1 METRIC CLASS ---
# This teaches TabNet how to calculate F1-Score
class F1(Metric):
    def __init__(self):
        self._name = "f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        # y_score contains probabilities, we convert them to 0 or 1
        y_pred = np.argmax(y_score, axis=1)
        return f1_score(y_true, y_pred)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/Processed_Churn_Data.csv')

# Data Type Conversion for PyTorch
X = df.drop('Exited', axis=1).values.astype(np.float32)
y = df['Exited'].values.astype(np.int64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize TabNet
model_tabnet = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name='cpu'
)

# 3. Train the Model
print("Training TabNet (Deep Learning)... This will take a few minutes.")
start_time = time.time()

model_tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=[F1], # <--- Using the custom F1 class we made above
    max_epochs=20, 
    patience=5,
    batch_size=1024, 
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

end_time = time.time()

# 4. Evaluate
preds = model_tabnet.predict(X_test)
score = f1_score(y_test, preds)

print("\n" + "="*30)
print(f"TabNet F1-Score (10 Features): {score:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")
print("="*30)
print(classification_report(y_test, preds))

# 5. Save the model
if not os.path.exists('models'):
    os.makedirs('models')
model_tabnet.save_model('models/tabnet_model') 
print("Model saved to models/tabnet_model.zip")