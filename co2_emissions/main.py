from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from custom_encoder import CustomEncoder
import joblib

pipeline = joblib.load('model/xgb_pipeline.pkl')

app = FastAPI()

class UserInput(BaseModel):
    Vehicle_Class: str = Field(..., alias='Vehicle Class')
    Engine_Size: float = Field(..., alias='Engine Size(L)')
    Cylinders: int
    Transmission: str
    Fuel_Type: str = Field(..., alias='Fuel Type')

@app.post("/predict")
def predict(input_data: UserInput):
    input_df = pd.DataFrame([{
        'Vehicle Class': input_data.Vehicle_Class,
        'Engine Size(L)': input_data.Engine_Size,
        'Cylinders': input_data.Cylinders,
        'Transmission': input_data.Transmission,
        'Fuel Type': input_data.Fuel_Type
    }])

    prediction = pipeline.predict(input_df)

    return {"prediction": float(prediction[0])}
