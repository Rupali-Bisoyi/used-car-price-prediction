import argparse
import joblib
import pandas as pd

# List of features in the same order used for training
FEATURES = ["year", "brand", "fuel", "km_driven", "engine_cc", "mileage", "hp"]

# Default values if not provided
DEFAULTS = {
    "year": 2018,
    "brand": "Maruti",
    "fuel": "Petrol",
    "km_driven": 50000,
    "engine_cc": 1200,
    "mileage": 18.0,
    "hp": 80
}

def parse_features(features_str):
    features = DEFAULTS.copy()
    for item in features_str.split(","):
        key, value = item.split("=")
        key = key.strip()
        value = value.strip()
        try:
            value = float(value) if "." in value or key not in ["brand", "fuel"] else int(value)
        except ValueError:
            pass
        features[key] = value
    return features

def main(args):
    # Load model
    model = joblib.load(args.model)
    print("Model loaded from", args.model)

    # Prepare feature dict
    features_dict = parse_features(args.features)

    # Create DataFrame with correct columns order
    df = pd.DataFrame([[features_dict[col] for col in FEATURES]], columns=FEATURES)

    # Predict
    prediction = model.predict(df)[0]
    print(f"Predicted price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model.pkl")
    parser.add_argument("--features", required=True, help="Comma-separated values, e.g., year=2018,brand=Maruti,fuel=Petrol,km_driven=45000,engine_cc=1197,mileage=21.0,hp=85")
    args = parser.parse_args()
    main(args)
