from flask import Flask, request, jsonify
import joblib
import argparse
import pandas as pd

app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    try:
        pred = model.predict(df)[0]
        return jsonify({'predicted_price': float(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    print('Model loaded. Starting server on http://127.0.0.1:5000')
    app.run(debug=True)
