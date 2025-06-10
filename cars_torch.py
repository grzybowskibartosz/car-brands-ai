# ✅ Install necessary packages (if not already in Colab)
# pip install torch torchvision tqdm scikit-learn matplotlib seaborn pillow

"""
Enhanced Car Recognition Neural Network
- Handles pre-sized 224x224 training images
- Includes augmentation to simulate real-world resizing
- Windows-compatible (handles multiprocessing issues)
- Multiple model architectures supported
- Test-time augmentation for robust predictions
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 🔧 Configurations
DATA_DIR = './dataset/DATA'  # Update with your dataset path
IMG_SIZE = 224
BATCH_SIZE = 32
INIT_LR = 1e-3
FINE_LR = 1e-5
INIT_EPOCHS = 20  # Increased from 10
FINE_EPOCHS = 30  # Increased from 10
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set num_workers based on OS
# Windows has issues with multiprocessing in DataLoader, so we set it to 0
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2

# 📁 Step 1: Collect image paths and labels
def collect_image_paths(data_dir):
    image_paths = []
    labels = []
    class_to_idx = {}
    idx_to_class = {}
    for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, fname))
                    labels.append(idx)
    return np.array(image_paths), np.array(labels), class_to_idx, idx_to_class

# Custom transform class to replace lambda (for Windows compatibility)
class IdentityTransform:
    """Identity transform that returns input unchanged"""
    def __call__(self, x):
        return x

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

# 📊 Calculate class weights for imbalanced dataset
def calculate_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(class_counts)):
        weights.append(total / (len(class_counts) * class_counts[i]))
    return torch.FloatTensor(weights).to(DEVICE)

# 🧹 Step 3: Prepare Data Loaders with enhanced augmentation
def prepare_dataloaders(data_dir):
    image_paths, labels, class_to_idx, idx_to_class = collect_image_paths(data_dir)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, labels))

    temp_paths, temp_labels = image_paths[temp_idx], labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_paths, temp_labels))

    train_paths, train_labels = image_paths[train_idx], labels[train_idx]
    val_paths, val_labels = temp_paths[val_idx], temp_labels[val_idx]
    test_paths, test_labels = temp_paths[test_idx], temp_labels[test_idx]

    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_labels)

    # 📦 Enhanced Transforms - Special handling for pre-sized 224x224 images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Since images are already 224x224, we need to simulate real-world resizing artifacts
    data_transforms = {
        'train': transforms.Compose([
            # Simulate various input resolutions by scaling up/down
            transforms.RandomChoice([
                transforms.Compose([  # Simulate high-res input
                    transforms.Resize(int(IMG_SIZE * 1.5)),  # Scale up
                    transforms.RandomCrop(IMG_SIZE),  # Then crop
                ]),
                transforms.Compose([  # Simulate low-res input
                    transforms.Resize(int(IMG_SIZE * 0.8)),  # Scale down
                    transforms.Resize(IMG_SIZE),  # Then scale back up (introduces blur)
                ]),
                transforms.Compose([  # Keep original
                    transforms.Resize(IMG_SIZE),
                ]),
            ]),
            # Add padding variations to simulate different aspect ratios
            transforms.RandomChoice([
                transforms.Pad(padding=10, fill=0, padding_mode='constant'),
                transforms.Pad(padding=(20, 10), fill=0, padding_mode='constant'),
                transforms.Pad(padding=(10, 20), fill=0, padding_mode='constant'),
                IdentityTransform(),  # No padding - replaced lambda
            ]),
            transforms.CenterCrop(IMG_SIZE),  # Crop back to size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # Add Gaussian blur to simulate different image qualities
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3)
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, len(class_to_idx), class_weights, idx_to_class

# 🧠 Step 4: Build enhanced model with dropout
def create_model(num_classes, dropout_rate=0.5, model_name='mobilenet_v2'):
    """
    Create model with option to use different architectures.
    EfficientNet models are particularly good at handling different input scales.
    """
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        
        # Freeze only 70% of layers initially
        total_params = len(list(model.features.parameters()))
        for i, param in enumerate(model.features.parameters()):
            if i < total_params * 0.7:
                param.requires_grad = False
        
        # Add more robust classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.last_channel, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in model.features[:-2].parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        
        # Freeze all layers except the last few
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    return model.to(DEVICE)

# 🎓 Step 5: Enhanced training function with scheduler
def train_model(model, train_loader, val_loader, epochs, lr, class_weights, unfreeze=False, patience=5):
    if unfreeze:
        for param in model.features.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    
    best_val_acc = 0
    patience_counter = 0

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
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Load best model
                model.load_state_dict(torch.load('best_model.pth'))
                break

    return model

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
    return acc

# 🖼️ Helper function to resize image while preserving aspect ratio
def resize_with_padding(img, target_size=IMG_SIZE, fill_color=(0, 0, 0)):
    """Resize image to target size while preserving aspect ratio with padding"""
    # Calculate aspect ratio
    w, h = img.size
    aspect = w / h
    
    if aspect > 1:  # Width > Height
        new_w = target_size
        new_h = int(target_size / aspect)
    else:  # Height >= Width
        new_h = target_size
        new_w = int(target_size * aspect)
    
    # Resize image
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_img = Image.new('RGB', (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img

# 🔍 Predict single custom image with multiple preprocessing strategies
def predict_custom_image(model, image_path, idx_to_class):
    """Predict using multiple preprocessing strategies and show all results"""
    
    # Strategy 1: Direct resize (may distort aspect ratio)
    transform_direct = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Strategy 2: Center crop after resize
    transform_crop = transforms.Compose([
        transforms.Resize(256),  # Resize shorter edge to 256
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Strategy 3: Preserve aspect ratio with padding (manual)
    transform_pad = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model.eval()
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    print(f"\nProcessing image: {image_path}")
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    
    predictions = {}
    
    # Try different preprocessing strategies
    strategies = {
        'Direct Resize': lambda: transform_direct(img),
        'Center Crop': lambda: transform_crop(img),
        'Aspect Preserve (Pad)': lambda: transform_pad(resize_with_padding(img))
    }
    
    for strategy_name, transform_func in strategies.items():
        img_tensor = transform_func().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        predictions[strategy_name] = (top5_prob, top5_idx)
        
        print(f"\n{strategy_name} predictions:")
        for i in range(3):  # Show top 3 for each strategy
            class_idx = top5_idx[0][i].item()
            class_name = idx_to_class[class_idx]
            confidence = top5_prob[0][i].item()
            print(f"  {i+1}. {class_name}: {confidence:.2%}")
    
    # Find consensus prediction
    all_top_predictions = []
    for _, (_, indices) in predictions.items():
        all_top_predictions.append(indices[0][0].item())
    
    from collections import Counter
    consensus = Counter(all_top_predictions).most_common(1)[0][0]
    
    print(f"\nConsensus prediction: {idx_to_class[consensus]}")
    
    return idx_to_class[consensus], predictions

# 🔄 Enhanced Test-Time Augmentation with multiple preprocessing strategies
def predict_with_tta(model, image_path, idx_to_class, n_augmentations=10):
    """TTA with multiple preprocessing strategies for robustness"""
    
    # Various augmentation strategies
    tta_transforms = [
        # Standard resize
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Crop-based
        transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.2)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # With rotation
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # With color jitter
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]
    
    model.eval()
    img = Image.open(image_path).convert('RGB')
    
    all_predictions = []
    
    # Apply each transform multiple times
    for _ in range(n_augmentations):
        # Randomly choose a transform
        transform = np.random.choice(tta_transforms)
        
        # Also randomly decide whether to flip
        if np.random.random() > 0.5:
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_tensor = transform(img_flipped).unsqueeze(0).to(DEVICE)
        else:
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.nn.functional.softmax(outputs, dim=1)
            all_predictions.append(predictions)
    
    # Also add aspect-preserved version
    img_padded = resize_with_padding(img)
    transform_standard = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform_standard(img_padded).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        all_predictions.append(predictions)
    
    # Average all predictions
    avg_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    top5_prob, top5_idx = torch.topk(avg_predictions, 5)
    
    print(f"\nEnhanced TTA Predictions for {image_path} ({len(all_predictions)} variants):")
    for i in range(5):
        class_idx = top5_idx[0][i].item()
        class_name = idx_to_class[class_idx]
        confidence = top5_prob[0][i].item()
        print(f"{i+1}. {class_name}: {confidence:.2%}")
    
    # Also show prediction variance (uncertainty)
    std_predictions = torch.std(torch.stack(all_predictions), dim=0)
    top_class_std = std_predictions[0, top5_idx[0][0]].item()
    print(f"\nPrediction uncertainty (std): {top_class_std:.4f}")
    
    return idx_to_class[top5_idx[0][0].item()], avg_predictions

# 📊 Plot confusion matrix
def plot_confusion_matrix(model, test_loader, idx_to_class):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Get class names in order
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print most confused classes
    print("\nMost confused class pairs:")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_normalized[i, j] > 0.1:  # More than 10% confusion
                print(f"{class_names[i]} -> {class_names[j]}: {cm_normalized[i, j]:.2%} ({cm[i, j]} samples)")

if __name__ == "__main__":
    # 🚀 Run the full pipeline
    train_loader, val_loader, test_loader, num_classes, class_weights, idx_to_class = prepare_dataloaders(DATA_DIR)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model with dropout
    model = create_model(num_classes, dropout_rate=DROPOUT_RATE)

    # 🧠 Train head only with early stopping
    print("\n--- Phase 1: Training classifier head ---")
    model = train_model(model, train_loader, val_loader, INIT_EPOCHS, INIT_LR, 
                       class_weights, unfreeze=False, patience=5)

    # 🔓 Fine-tune full model with early stopping
    print("\n--- Phase 2: Fine-tuning entire model ---")
    model = train_model(model, train_loader, val_loader, FINE_EPOCHS, FINE_LR, 
                       class_weights, unfreeze=True, patience=7)

    # 🧪 Evaluate on test set
    print("\n--- Final Evaluation ---")
    test_accuracy = evaluate_model(model, test_loader)

    # 📊 Generate confusion matrix
    print("\n--- Generating Confusion Matrix ---")
    plot_confusion_matrix(model, test_loader, idx_to_class)

    # 💾 Save final model
    torch.save(model.state_dict(), 'final_car_classifier.pth')
    print("\nModel saved as 'final_car_classifier.pth'")

    # 🔍 Example: How to use for custom image prediction
    print("\n--- Custom Image Prediction Example ---")
    print("\nIMPORTANT: Since training images are pre-sized to 224x224,")
    print("the model may struggle with real-world images of different sizes.")
    print("Use the enhanced prediction functions for better results:")
    print("\n1. predict_custom_image() - tries multiple preprocessing strategies")
    print("2. predict_with_tta() - uses test-time augmentation for robustness")
    
    # Uncomment and modify the path below to test on your custom image
    # custom_image_path = "path/to/your/custom/car/image.jpg"
    # 
    # # Multi-strategy prediction (recommended for images of different sizes)
    # predicted_class, _ = predict_custom_image(model, custom_image_path, idx_to_class)
    # 
    # # Enhanced TTA prediction (most robust)
    # tta_predicted_class, _ = predict_with_tta(model, custom_image_path, idx_to_class, n_augmentations=15)
    # 
    # # For best results with custom images:
    # # 1. Try both prediction methods
    # # 2. If predictions differ significantly, the image might be out-of-distribution
    # # 3. High uncertainty (std) in TTA indicates the model is unsure
    
    print("\n--- How to Load and Use Saved Model ---")
    print("""
# To load and use the saved model later:
checkpoint = torch.load('final_car_classifier_complete.pth')
model = create_model(
    checkpoint['num_classes'], 
    model_name=checkpoint['model_architecture']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
idx_to_class = checkpoint['idx_to_class']
""")
    
    print("\n--- RECOMMENDATIONS FOR BETTER REAL-WORLD PERFORMANCE ---")
    print("""
1. The main issue is that training data is pre-sized to 224x224
2. Consider collecting a small set of real-world images (various sizes/aspects)
3. Add these to your training set to improve generalization
4. Try EfficientNet models - they handle scale variations better
5. Use predict_with_tta() for production - it's much more robust
6. Monitor prediction uncertainty - high std indicates low confidence
""")

# Ensure proper execution on Windows with multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows executable
    main()