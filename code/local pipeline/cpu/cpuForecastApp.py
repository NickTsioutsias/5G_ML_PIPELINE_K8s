from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle

app = FastAPI()

# Load model once when app starts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class LaggedInput(BaseModel):
    lagged_values: List[float]

class ForecastInput(BaseModel):
    lagged_values: List[float]
    steps: int = 20  # default forecast steps

@app.post("/predict")
def predict(input_data: LaggedInput):
    # Convert input list to numpy array for model
    x_input = np.array([input_data.lagged_values])
    pred = model.predict(x_input)[0]
    return {"prediction": float(pred)}

@app.post("/forecast")
def forecast(input_data: ForecastInput):
    last_known = input_data.lagged_values
    steps = input_data.steps
    
    preds = []
    # recursive multi-step prediction
    for _ in range(steps):
        x_input = np.array([last_known])
        next_pred = model.predict(x_input)[0]
        preds.append(float(next_pred))
        # update last_known lags by inserting new pred at front, removing oldest
        last_known = [next_pred] + last_known[:-1]
    
    return {"forecast": preds}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
