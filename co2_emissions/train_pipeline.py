import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from custom_encoder import CustomEncoder
import kagglehub

def main():
    path = kagglehub.dataset_download("debajyotipodder/co2-emission-by-vehicles")
    print("Downloaded files:", os.listdir(path))

    df = pd.read_csv(os.path.join(path, "CO2 Emissions_Canada.csv"))

    X = df.drop(columns=['CO2 Emissions(g/km)', 'Make', 'Model', 'Fuel Consumption Comb (mpg)'])
    y = df['CO2 Emissions(g/km)']

    fuel_counts = X['Fuel Type'].value_counts()
    X = X[X['Fuel Type'].isin(fuel_counts[fuel_counts > 1].index)]
    y = y.loc[X.index]

    X = X.drop(columns=[
        'Fuel Consumption Comb (L/100 km)',
        'Fuel Consumption City (L/100 km)',
        'Fuel Consumption Hwy (L/100 km)'
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=X['Fuel Type'], random_state=42
    )

    log_transformer = FunctionTransformer(np.log1p, validate=True)
    preprocessor = ColumnTransformer(
        transformers=[('log', log_transformer, ['Cylinders'])],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('custom_encoder', CustomEncoder()),
        ('log_transform', preprocessor),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    BASE = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgb_pipeline.pkl")
    saved = joblib.dump(pipeline, model_path)

if __name__ == "__main__":
    main()