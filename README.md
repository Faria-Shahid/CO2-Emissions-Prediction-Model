This is a FastAPI-based backend for predicting CO₂ emissions using a machine learning model.  
The API accepts vehicle input features and returns the predicted CO₂ emissions in grams per kilometer.

## Project Structure
├── app/
│ ├── main.py
│ ├── model.pkl
│ └── utils.py
├── requirements.txt
└── README.md

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Faria-Shahid/CO2-Emissions-Prediction-Model.git
cd CO2-Emissions-Prediction-Model

### 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Requirements
pip install -r requirements.txt

### 4. Run the FASTAPI Server
uvicorn app.main:app --reload

##API Usage

###Endpoint: POST /predict
Request Example
{
  "engine_size": 2.5,
  "fuel_type": "Gasoline",
  "cylinders": 4,
  "transmission": "Automatic"
}

Response Example
{
  "predicted_co2_emission": 190.5
}


