# ✅ Install necessary packages (if not already in Colab)

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# 🔧 Configurations
DATA_DIR = './dataset/DATA'  # Update with your dataset path
IMG_SIZE = 224
BATCH_SIZE = 32
INIT_LR = 1e-3
FINE_LR = 1e-5
INIT_EPOCHS = 10
FINE_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 Step 1: Collect image paths and labels
def collect_image_paths(data_dir):
    image_paths = []
    labels = []
    class_to_idx = {}
    for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, fname))
                    labels.append(idx)
    return np.array(image_paths), np.array(labels), class_to_idx

# 🧪 Step 2: Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# 🧹 Step 3: Prepare Data Loaders
def prepare_dataloaders(data_dir):
    image_paths, labels, class_to_idx = collect_image_paths(data_dir)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, labels))

    temp_paths, temp_labels = image_paths[temp_idx], labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_paths, temp_labels))

    train_paths, train_labels = image_paths[train_idx], labels[train_idx]
    val_paths, val_labels = temp_paths[val_idx], temp_labels[val_idx]
    test_paths, test_labels = temp_paths[test_idx], temp_labels[test_idx]

    # 📦 Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    train_ds = CustomImageDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_ds = CustomImageDataset(val_paths, val_labels, transform=data_transforms['val'])
    test_ds = CustomImageDataset(test_paths, test_labels, transform=data_transforms['test'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(class_to_idx)

# 🧠 Step 4: Build model using Transfer Learning
def create_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze base
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(DEVICE)

# 🎓 Step 5: Training function
def train_model(model, train_loader, val_loader, epochs, lr, unfreeze=False):
    if unfreeze:
        for param in model.features.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ✅ Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                val_corrects += torch.sum(preds == labels)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Acc: {val_acc:.4f}")

# 🧪 Step 6: Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            correct += torch.sum(preds == labels)
    acc = correct.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # 🚀 Run the full pipeline
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(DATA_DIR)
    print("Classes:", num_classes)

    model = create_model(num_classes)

    # 🧠 Train head only
    train_model(model, train_loader, val_loader, INIT_EPOCHS, INIT_LR, unfreeze=False)

    # 🔓 Fine-tune full model
    train_model(model, train_loader, val_loader, FINE_EPOCHS, FINE_LR, unfreeze=True)

    # 🧪 Evaluate on test set
    evaluate_model(model, test_loader)


