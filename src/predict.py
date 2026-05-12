import pandas as pd
import joblib

from preprocess import preprocess_transaction


# Load saved artifacts

model = joblib.load("../models/fraud_model.pkl")
scaler = joblib.load("../models/amount_scaler.pkl")
threshold = joblib.load("../models/best_threshold.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

def predict_transaction(df):
    """
    Predict fraud probability and final class.
    """

    # Preprocess data
    X = predict_transaction(df, scaler, features_names)

    # Predict probabilities
    probabilities = model.predict_proba(X)[:,1]

    # Apply business threshold
    predictions = (probabilities>=threshold).astype(int)

    return pd.DataFrame({
        "Fraud_Probability": probabilities,
        "Prediction": predictions
    })

print("SUCCESS")

    