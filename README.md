# Créer le projet

> Linux, macos
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Windows
```bash
.venv\Scripts\Activate.ps1
```

# Dépendances

```bash
pip install fastapi uvicorn scikit-learn numpy joblib pydantic requests
```

# Application

> schemas.py

```python
from pydantic import BaseModel

class Inputdata(BaseModel):
    tv: float
    radio: float
    newspaper: float
```

> main.py

```python
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
```

# Lancer le serveur

```bash
uvicorn main:app --reload
```

# Swagger Api

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)