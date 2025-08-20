from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os

# Import your model class
from backend.models.model import ANN
from backend.src.utils import *
from training.preprocess import preprocess_input

app = FastAPI(title="Model Prediction API")

# --------------------------------------------------------------------
# Fix: Always resolve model path relative to project root
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# Load model
model = ANN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Request schema
class PredictionRequest(BaseModel):
    features: list

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    input_tensor = preprocess_input(request.features)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return {"prediction": predicted_class}
