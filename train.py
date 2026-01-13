import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib

def run_training():
    print("Step 1: Loading the Processed Data...")
    # This is the file created by your preprocess.py
    df = pd.read_csv('data/Processed_Churn_Data.csv')

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Step 2: Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 2: Training the SOTA CatBoost Model...")
    # We use iterations=500 for a good balance of speed and accuracy
    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=100)
    
    model.fit(X_train, y_train)

    # Step 3: Evaluation
    preds = model.predict(X_test)
    print("\nTraining Complete!")
    print(f"F1-Score: {f1_score(y_test, preds):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, preds))

    # Step 4: Save the "Brain"
    joblib.dump(model, 'models/catboost_model.pkl')
    print("Model saved to models/catboost_model.pkl")

if __name__ == "__main__":
    run_training()