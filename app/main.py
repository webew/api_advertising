from fastapi import FastAPI
import joblib
import numpy as np
from schemas import Inputdata

# Chargement des modèles
scaler_cv = joblib.load("scaler_cv.joblib")
model_cv = joblib.load("model_cv.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advertising API"}

@app.post("/predict")
def predict(input_data: Inputdata):
    print("Données reçues : ", input_data.tv)
    data = np.array([[input_data.tv, input_data.radio, input_data.newspaper]])
    data_scaled = scaler_cv.transform(data)
    prediction = model_cv.predict(data_scaled)
    
    return {"Prédiction : ": prediction[0]}