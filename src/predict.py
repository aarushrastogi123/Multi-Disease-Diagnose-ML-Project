import torch
from torchvision import transforms
from PIL import Image
import os
from model import RetinalNet

MODEL_PATH = "models/resnet50_retinal.pth"  # adjust if different
CLASS_NAMES_PATH = "models/class_names.txt"  # one per line (optional)
DEFAULT_IMG = "data/test/0aebb1b2aef1.png"  # or .jpg
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_class_names(path=CLASS_NAMES_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            names = [x.strip() for x in f.readlines() if x.strip()]
        return names
    # default mapping (edit if your labels differ)
    return ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(img_path, model, class_names):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = out.argmax(1).item()
    return pred, probs

if __name__ == "__main__":
    model = RetinalNet(num_classes=5)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        # Handle "backbone." prefix mismatch automatically
    new_state_dict = {}
    for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_state_dict[k] = v
            elif any(k.startswith(x) for x in ["conv1", "layer", "bn1", "fc"]):
                new_state_dict[f"backbone.{k}"] = v
            else:
                new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


    model.to(DEVICE)
    model.eval()

    class_names = load_class_names()
    # single default image (change path or pass via CLI)
    img = DEFAULT_IMG
    if not os.path.exists(img):
        # if default missing, try first image in test_images
        test_dir = "../test_images"
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
            if files:
                img = os.path.join(test_dir, files[0])
            else:
                raise FileNotFoundError("No test image found in test_images/ and default image missing")
        else:
            raise FileNotFoundError("No test_images folder and default image missing")

    pred, probs = predict_image(img, model, class_names)
    print(f"Image: {img}")
    print(f"Predicted: {class_names[pred]}  (confidence: {probs[pred]*100:.2f}%)")
    # also print full probs
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i]*100:.2f}%")
