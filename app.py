from flask import Flask, request, jsonify, render_template
import torch
from model_loader import load_model
from preprocessing import preprocess_input, le

app = Flask(__name__)

# Load model
model, device = load_model("best_model.pth")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data['features']
        input_tensor = preprocess_input(features).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = le.inverse_transform([predicted_class])[0]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
