# Used Car Price Prediction

Final-year mini-project using Python, NumPy, Pandas, SciPy, scikit-learn and Matplotlib.

## Quick summary
- Problem: Predict resale price of used cars based on features.
- Model: RandomForestRegressor (baseline) with simple preprocessing.

## How to run (quick)
1. Create virtualenv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Generate sample data (optional):
   ```bash
   python data_generate.py
   ```
3. Train model:
   ```bash
   python train.py --data used_cars_sample.csv --outdir out
   ```
4. Run demo server:
   ```bash
   python app.py --model out/model.pkl
   ```

## Git commands
```bash
git init
git add .
git commit -m "Initial commit: Used Car Price Prediction final year project"
git remote add origin https://github.com/Rupali-Bisoyi/used-car-price-prediction.git
git branch -M main
git push -u origin main
```
