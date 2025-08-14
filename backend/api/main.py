from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import YourModelClass
from backend.src.utils import *
from training.preprocess import preprocess_input

app = FastAPI(title="Model Prediction API")

# Load model at startup
MODEL_PATH = "models/best_model.pth"
model = YourModelClass()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    input_tensor = preprocess_input(request.features)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return {"prediction": predicted_class}
