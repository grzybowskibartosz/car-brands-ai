import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1.1 Odzyskaj liczbę klas --------------------------------------
# Najbezpieczniej jest mieć ją zapisaną w checkpoint-cie,
# ale jeśli została tylko w dataset-cie, możesz policzyć katalogi:
from pathlib import Path
DATA_DIR = './dataset/DATA'           # ten sam co wcześniej
num_classes = len([d for d in Path(DATA_DIR).iterdir() if d.is_dir()])

# --- 1.2 Zbuduj identyczną architekturę  ---------------------------
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# --- 1.3 Załaduj wagi ----------------------------------------------
state_dict = torch.load("cars_brands.pth",
                        map_location=DEVICE)    # CPU lub GPU
model.load_state_dict(state_dict)
model.to(DEVICE).eval()

print("✅ Model gotowy do inference")


from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.utils import save_image


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict_image(path, model, id2label):
    img = Image.open(path).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(DEVICE)
    save_image(x, "image.png")

    with torch.no_grad():
        logits = model(x)
        pred_id = torch.argmax(logits, 1).item()
    return id2label[pred_id]

# --- mapa id -> etykieta -------------------------------------------
# Jeśli masz ją w pliku / checkpoint-cie:
# id2label = np.load("id2label.npy", allow_pickle=True).tolist()
# Dla demo zbudujmy z nazw katalogów:
id2label = sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])

print("Predykcja:",
      predict_image("golf8888.jpg", model, id2label))
