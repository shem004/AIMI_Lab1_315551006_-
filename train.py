# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

def build_model(model_type='resnet18', num_classes=2):
    if model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError("Unsupported model type")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1): tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0): tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0): fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1): fn += 1
    return tp, tn, fp, fn

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    history = {'train_acc': [], 'train_f1': [], 'val_acc': [], 'val_f1': []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        train_loss, tp, tn, fp, fn = 0.0, 0, 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = torch.max(outputs, 1).indices
            sub_tp, sub_tn, sub_fp, sub_fn = measurement(preds, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

        epoch_train_acc = (tp + tn) / (tp + tn + fp + fn) * 100
        epoch_train_f1 = (2 * tp) / (2 * tp + fp + fn)
        history['train_acc'].append(epoch_train_acc)
        history['train_f1'].append(epoch_train_f1)

        model.eval()
        val_loss, tp, tn, fp, fn = 0.0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(preds, labels)
                tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

        epoch_val_acc = (tp + tn) / (tp + tn + fp + fn) * 100
        epoch_val_f1 = (2 * tp) / (2 * tp + fp + fn)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        print(f'↳ Val Acc: {epoch_val_acc:.2f}% | Val F1: {epoch_val_f1:.4f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pt')

    return history

if __name__ == '__main__':
    # Environment Setup
    os.environ['KAGGLE_USERNAME'] = "shemlawalata"
    os.environ['KAGGLE_KEY'] = "KGAT_dbea1a814ed4fad294d54827199bfa2d"
    os.system("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data --unzip")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data Augmentation[cite: 31]
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dir = "./data/chest_xray/train"
    raw_dataset = ChestXRayDataset(train_dir, transform=None)
    labels = raw_dataset.labels

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=42
    )

    train_loader = DataLoader(TransformSubset(Subset(raw_dataset, train_indices), transform=train_transform), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(TransformSubset(Subset(raw_dataset, val_indices), transform=eval_transform), batch_size=32, shuffle=False, num_workers=2)

    # Initialize and Train Model
    model_18 = build_model('resnet18').to(device)
    class_weights = torch.FloatTensor([3.8896346, 1.346]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_18 = optim.Adam(model_18.parameters(), lr=1e-5, weight_decay=0.9)

    print("Starting Training...")
    history = train_model(model_18, train_loader, val_loader, criterion, optimizer_18, device, num_epochs=10)
    print("Training Complete. Best weights saved locally.")