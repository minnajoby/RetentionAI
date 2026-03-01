import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Load the ORIGINAL raw data (not the processed one)
df = pd.read_csv('data/Churn_Modelling.csv')

# 2. Keep only the 7 features we use now
# Order MUST match your app.py: Geography, Gender, Age, Tenure, Balance, NumOfProducts, IsActiveMember
X_raw = df[['Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember']].copy()

# 3. Manually encode the text so the scaler can work
X_raw['Gender'] = X_raw['Gender'].map({'Female': 0, 'Male': 1})
X_raw['Geography'] = X_raw['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})

# 4. Fit the new 7-feature scaler
scaler_7 = StandardScaler()
scaler_7.fit(X_raw)

# 5. Save it
joblib.dump(scaler_7, 'models/scaler_7.pkl')
print("--- SUCCESS: models/scaler_7.pkl created! ---")