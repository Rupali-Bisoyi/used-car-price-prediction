# Used Car Price Prediction

A machine learning model to predict used car prices based on features like brand, year, mileage, engine capacity and horsepower.

## Features
- Uses Random Forest Regressor
- Simple preprocessing pipeline with StandardScaler and OneHotEncoder
- ~94% R-squared score on test data
- Command line interface for predictions
- Flask API server for integration

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/used-car-price-prediction.git
cd used-car-price-prediction
```

2. Create virtual environment and install requirements:
```bash 
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Generate sample training data (optional):
```bash
python data_generate.py
# Output: wrote used_cars_sample.csv with 1200 rows
```

2. Train the model:
```bash
python train.py --data used_cars_sample.csv --outdir out
# Output:
# training...
# done 
# MAE: 26544.86, RMSE: 33813.67, R2: 0.9445
# saved model to out/model.pkl
```

3. Make predictions:
```bash
python predict.py --model out/model.pkl --features "year=2018,brand=Maruti,fuel=Petrol,km_driven=45000"
# Output:
# Model loaded from out/model.pkl
# Predicted price: ₹490,323.31
```

4. Run API server (optional):
```bash
python app.py --model out/model.pkl
# Access at http://127.0.0.1:5000
```

## API Usage

Send POST request to `/predict` endpoint:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"year":2018,"brand":"Maruti","fuel":"Petrol","km_driven":45000}'
```

## Features for Prediction
- year: Car manufacturing year (e.g. 2018)
- brand: Car manufacturer (e.g. Maruti, Honda)
- fuel: Fuel type (e.g. Petrol, Diesel) 
- km_driven: Total kilometers driven
- engine_cc: Engine capacity in cc
- mileage: Fuel efficiency (km/l)
- hp: Horsepower

Default values will be used for any missing features.
