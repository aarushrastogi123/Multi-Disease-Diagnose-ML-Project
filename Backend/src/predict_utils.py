import torch
from torchvision import transforms
from PIL import Image
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_class_names(path="models/class_names.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [x.strip() for x in f.readlines() if x.strip()]
    return ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

def predict_image(img_path, model, class_names):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = out.argmax(1).item()
    return pred, probs