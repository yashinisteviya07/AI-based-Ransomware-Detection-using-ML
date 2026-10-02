import joblib

model = joblib.load("../models/malware_detector.pkl")

def predict(df):

    # Remove columns not used during training
    df = df.drop(columns=["Category", "Class"], errors="ignore")

    predictions = model.predict(df)

    return predictions