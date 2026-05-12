import pandas as pd
from src.predict import predict_transaction

# Example: Load sample transaction
sample = pd.read_csv("data/CreditCardData.csv")

# Make prediction
results = predict_transaction(sample)

print(results)

print("SUCCESS")