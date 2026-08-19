"""
==============================================================================
Evaluation & Benchmark Script for EyeStateNet
Calculates: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
==============================================================================
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from train_eye_state import EyeStateNet, SyntheticMRLDataset

def evaluate_model(model_path=None):
    if model_path is None:
        if os.path.exists('model_training/models_out/eyestatenet_best.pth'):
            model_path = 'model_training/models_out/eyestatenet_best.pth'
        else:
            model_path = 'models_out/eyestatenet_best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating model: {model_path} on {device}")

    val_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = SyntheticMRLDataset(num_samples=600, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = EyeStateNet(num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Evaluating uninitialized model.")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics computation
    tp = np.sum((all_preds == 1) & (all_labels == 1))  # Open correct
    tn = np.sum((all_preds == 0) & (all_labels == 0))  # Closed correct
    fp = np.sum((all_preds == 1) & (all_labels == 0))  # Closed predicted as Open
    fn = np.sum((all_preds == 0) & (all_labels == 1))  # Open predicted as Closed

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=======================================================")
    print("           EYESTATENET EVALUATION RESULTS              ")
    print("=======================================================")
    print(f"Total Test Samples: {len(all_labels)}")
    print(f"Test Accuracy:      {accuracy * 100:.2f}%")
    print(f"Precision (Open):   {precision * 100:.2f}%")
    print(f"Recall (Open):      {recall * 100:.2f}%")
    print(f"F1-Score:           {f1:.4f}")
    print("-------------------------------------------------------")
    print("CONFUSION MATRIX:")
    print(f"               Predicted Closed  |  Predicted Open")
    print(f"Actual Closed:       {tn:4d}        |       {fp:4d}")
    print(f"Actual Open:         {fn:4d}        |       {tp:4d}")
    print("=======================================================\n")

if __name__ == '__main__':
    evaluate_model()
