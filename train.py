import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import math  # Added for RMSE calculation

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    return Pipeline([('pre', preprocessor), ('model', model)])

def main(args):
    df = load_data(args.data).dropna()
    X = df.drop(columns=['price'])
    y = df['price']
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
    pipe = build_pipeline(cat_cols, num_cols)
    print('training...')
    pipe.fit(X_train, y_train)
    print('done')

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))  # Manual RMSE
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, 'model.pkl'))
    print('saved model to', os.path.join(args.outdir, 'model.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--outdir', default='out')
    args = parser.parse_args()
    main(args)
