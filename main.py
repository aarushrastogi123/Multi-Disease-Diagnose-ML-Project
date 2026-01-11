# main.py (project root)
import torch
from src.dataloader_setup import get_dataloaders
from src.model import RetinalNet
from src.train import train_model
from src.evaluate import evaluate_model
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths (edit if necessary)
TRAIN_CSV = "data/train.csv"
TRAIN_IMG_DIR = "data/train"
VAL_CSV = "data/val_labels.csv"    # may be empty; dataloader handles splitting
VAL_IMG_DIR = "data/val"
MODEL_SAVE_PATH = "models/resnet50_retinal.pth"

# class names file - create models/class_names.txt with names (one per line)
CLASS_NAMES_PATH = "models/class_names.txt"
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = [x.strip() for x in f if x.strip()]
else:
    CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

# ----------------- dataloaders -----------------
train_loader, val_loader = get_dataloaders(TRAIN_CSV, TRAIN_IMG_DIR, VAL_CSV, VAL_IMG_DIR, batch_size=32)

# ----------------- model -----------------
model = RetinalNet(num_classes=len(CLASS_NAMES))
print("Model created. Device:", DEVICE)

# ----------------- train -----------------
model = train_model(model, train_loader, val_loader, device=DEVICE, epochs=5, lr=1e-4, save_path=MODEL_SAVE_PATH)

# ----------------- evaluate -----------------
print("Evaluating saved best model...")
# load best saved
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
evaluate_model(model, val_loader, CLASS_NAMES, DEVICE)
