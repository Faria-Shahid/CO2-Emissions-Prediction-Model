This is a FastAPI-based backend for predicting CO₂ emissions using a machine learning model.  
The API accepts vehicle input features and returns the predicted CO₂ emissions in grams per kilometer.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Faria-Shahid/CO2-Emissions-Prediction-Model.git
cd CO2-Emissions-Prediction-Model
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the FASTAPI Server
```bash
uvicorn main:app --reload
```

## API Usage

### Endpoint: POST /predict
Request Example
```bash
{
  "engine_size": 2.5,
  "fuel_type": "Gasoline",
  "cylinders": 4,
  "transmission": "Automatic"
}
```

Response Example
```bash
{
  "predicted_co2_emission": 190.5
}
```

