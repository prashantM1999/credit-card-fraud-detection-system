import pandas as pd


def preprocess_transaction(df, scaler, feature_names):
    """
    Preprocess incoming transaction data.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw input data.
    scaler : fitted scaler
        Scaler used during training.
    feature_names : list
        Final feature order used by model.

    Returns:
    --------
    pd.DataFrame
        Processed data ready for prediction.
    """

    df = df.copy()

    # Create Hour from Time
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24

    # Scale Amount
    if "Amount" in df.columns:
        df["Amount_scaled"] = scaler.transform(df[["Amount"]])

    # Drop unused columns
    columns_to_drop = []

    if "Amount" in df.columns:
        columns_to_drop.append("Amount")

    if "Class" in df.columns:
        columns_to_drop.append("Class")

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    # Keep correct feature order
    df = df[feature_names]

    return df

print("RUN Successfully")