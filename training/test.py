import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import pandas as pd
from backend.models.model import *
from training.preprocess import le, scaler

def predict_quality(
    model_class,            # Your ANN class
    model_path,             # Path to saved model weights, e.g., "best_model.pth"
    scaler,                 # Fitted StandardScaler used during training
    label_encoder,          # Fitted LabelEncoder used during training
    new_data: pd.DataFrame  # New samples with same features as training data
):
    """
    Predicts class labels and probabilities for new samples using a trained ANN.

    Returns:
        predicted_labels: List of predicted class names
        probabilities: Tensor of shape (n_samples, n_classes)
    """
    # 1. Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # 2. Load model
    model = model_class()  # Initialize the architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    # 3. Make predictions
    with torch.no_grad():
        outputs = model(new_data_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Class probabilities
        _, predicted_indices = torch.max(outputs, 1)   # Predicted class indices

    # 4. Convert indices to labels
    predicted_labels = label_encoder.inverse_transform(predicted_indices.numpy())

    return predicted_labels, probabilities


# Example usage
new_samples = pd.DataFrame([
    {"Moisture": 12.19, "Ash": 7.356, "Volatile_Oil": 0.626, "Acid_Insoluble_Ash": 0.581, "Chromium": 0.002, "Coumarin": 0.015},    
])

# Ensure the folder exists
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/models'))
os.makedirs(model_dir, exist_ok=True)

# Set the model path
model_path = os.path.join(model_dir, "best_model.pth")

predicted_labels, probabilities = predict_quality(
    model_class=ANN,
    model_path=model_path,
    scaler=scaler,
    label_encoder=le,
    new_data=new_samples
)

print("Predicted labels:", predicted_labels)
print("Predicted probabilities:\n", probabilities)