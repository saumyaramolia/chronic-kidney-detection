import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load the serialized model
try:
    with open('kidney_prediction_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# Prediction labels
prediction_labels = {0: "Not a patient of CKD", 1: "Patient has CKD"}


# Define request body
class PredictionInput(BaseModel):
    age: int
    blood_pressure: int
    specific_gravity: float
    albumin: int
    sugar: int
    red_blood_cells: str
    pus_cell: str
    pus_cell_clumps: str
    bacteria: str
    blood_glucose_random: int
    blood_urea: int
    serum_creatinine: float
    sodium: int
    potassium: float
    haemoglobin: float
    packed_cell_volume: int
    white_blood_cell_count: int
    red_blood_cell_count: float
    hypertension: str
    diabetes_mellitus: str
    coronary_artery_disease: str
    appetite: str
    peda_edema: str
    anemia: str


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Preprocess input data
        le = LabelEncoder()
        input_df['red_blood_cells'] = le.fit_transform(input_df['red_blood_cells'])
        input_df['pus_cell'] = le.fit_transform(input_df['pus_cell'])
        input_df['pus_cell_clumps'] = le.fit_transform(input_df['pus_cell_clumps'])
        input_df['bacteria'] = le.fit_transform(input_df['bacteria'])
        input_df['hypertension'] = le.fit_transform(input_df['hypertension'])
        input_df['diabetes_mellitus'] = le.fit_transform(input_df['diabetes_mellitus'])
        input_df['coronary_artery_disease'] = le.fit_transform(input_df['coronary_artery_disease'])
        input_df['appetite'] = le.fit_transform(input_df['appetite'])
        input_df['peda_edema'] = le.fit_transform(input_df['peda_edema'])
        input_df['anemia'] = le.fit_transform(input_df['anemia'])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_df)

        # Convert prediction to human-readable output
        prediction_output = prediction_labels.get(prediction[0], "Unknown")

        # Return prediction
        return {"prediction": prediction_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting: {e}")
