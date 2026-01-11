from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil
import os
from model import RetinalNet
from predict_utils import predict_image, load_class_names, transform

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute model path (fixes your FileNotFound problem)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "resnet50_retinal.pth")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Load model
model = RetinalNet(num_classes=5)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle key mismatch safely
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("backbone."):
        new_state_dict[k] = v
    elif any(k.startswith(x) for x in ["conv1", "layer", "bn1", "fc"]):
        new_state_dict[f"backbone.{k}"] = v
    else:
        new_state_dict[k] = v

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print("✅ Model loaded successfully.")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.to(DEVICE)
model.eval()
class_names = load_class_names()

# ---- PREDICT ROUTE ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded image temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        pred, probs = predict_image(temp_path, model, class_names)

        # Clean up
        os.remove(temp_path)

        # Format JSON response
        confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        predicted_class = class_names[pred]

        print({
            "predicted_class": predicted_class,
            "confidences": confidences
        })

        return {
            "predicted_class": predicted_class,
            "confidences": confidences
        }

    except Exception as e:
        import traceback
        print("❌ Prediction Error:", traceback.format_exc())
        return {"error": str(e)}
