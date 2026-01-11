# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from collections import Counter

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
def train_model(model, train_loader, val_loader, device,
                epochs=10, lr=1e-4, save_path="models/resnet50_retinal.pth"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        tk = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in tk:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tk.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} ValAcc={val_acc:.2f}%")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (ValAcc={best_val_acc:.2f}%)")

    print("Training finished.")
    return model
