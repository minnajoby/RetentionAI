from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Load Model and Scaler ---
# These must be in your 'models/' folder
try:
    model = joblib.load('models/catboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("--- SUCCESS: Model and Scaler loaded ---")
except Exception as e:
    print(f"--- ERROR LOADING MODELS: {e} ---")

@app.route('/')
def home():
    # Pass None for inputs and results on first load
    return render_template('index.html', inputs=None, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Fetch values for the AI model in the correct training order
        feature_list = [
            float(request.form.get('CreditScore')),
            float(request.form.get('Geography')),
            float(request.form.get('Gender')),
            float(request.form.get('Age')),
            float(request.form.get('Tenure')),
            float(request.form.get('Balance')),
            float(request.form.get('NumOfProducts')),
            float(request.form.get('HasCrCard', 0)),
            float(request.form.get('IsActiveMember', 0)),
            float(request.form.get('EstimatedSalary'))
        ]

        # 2. Reshape and Scale
        final_features = np.array(feature_list).reshape(1, -1)
        scaled_features = scaler.transform(final_features)

        # 3. Predict Class and Probability
        prediction = model.predict(scaled_features)
        prob = model.predict_proba(scaled_features)[0][1]
        
        # 4. Format Output
        result_text = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
        prob_display = f"{prob * 100:.2f}%"

        # 5. Send results AND the form data back to the UI
        return render_template('index.html', 
                               prediction_text=result_text, 
                               prob_text=prob_display,
                               inputs=request.form) # Keeps the form filled

    except Exception as e:
        return f"Error: {str(e)}. Please ensure all fields are filled."

if __name__ == "__main__":
    app.run(debug=True)