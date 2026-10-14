# inference.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import seaborn as sns
import matplotlib.pyplot as plt

# Import dataset class from train to prevent code duplication
from train import ChestXRayDataset, build_model, measurement

def test_model(model, test_loader, device):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(preds, labels)
            tp += sub_tp; tn += sub_tn; fp += sub_fp; fn += sub_fn

    test_acc = (tp + tn) / (tp + tn + fp + fn) * 100
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    print(f'Test Acc: {test_acc:.2f}% | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1_score:.4f}')

    c_matrix = [[int(tn), int(fp)],
                [int(fn), int(tp)]]

    plt.figure(figsize=(6, 5))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='rocket',
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    plt.title('Confusion Matrix - Test Set')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_dir = "./data/chest_xray/test"
    if not os.path.exists(test_dir):
        print("Test data not found. Please run train.py first to download the dataset.")
        exit()
        
    test_dataset = ChestXRayDataset(test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = build_model('resnet18').to(device)
    
    weights_path = 'best_ResNet.pt'
    if not os.path.exists(weights_path):
        print(f"Weights file '{weights_path}' not found. Please train the model first.")
        exit()
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    print("Starting Evaluation on Test Set...")
    test_model(model, test_loader, device)