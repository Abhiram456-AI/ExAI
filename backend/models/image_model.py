"""
Image Analysis Module using ResNet-18

Predicts microbial growth type from Petri dish images.

Classes:
0 -> Bacterial
1 -> Fungal
2 -> Mixed
3 -> Low Growth
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "backend/models/resnet18_microbe.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["Bacterial", "Fungal", "Mixed", "Low Growth"]

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------

def _load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model

# -----------------------------
# Main Inference Function
# -----------------------------

def analyze_image(image_path: str) -> dict:
    """
    Analyze petri dish image and return microbial profile.
    """

    model = _get_model()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    microbial_type = CLASSES[pred.item()]

    return {
        "microbial_dominance": microbial_type,
        "confidence": round(conf.item(), 3)
    }


# -----------------------------
# Simple Test
# -----------------------------
if __name__ == "__main__":
    result = analyze_image("sample.jpg")
    print(result)