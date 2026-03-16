import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np


class ECGNet(nn.Module):
    """
    1D CNN for binary ECG heartbeat classification.
    Input:  (Batch, 1, 360)
    Output: (Batch, 2)  →  0: Normal, 1: Arrhythmia
    """
    def __init__(self):
        super(ECGNet, self).__init__()

        # --- Block 1: (B, 1, 360) → (B, 16, 180) ---
        self.conv1 = nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)   
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # --- Block 2: (B, 16, 180) → (B, 32, 90) ---
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # --- Block 3: (B, 32, 90) → (B, 64, 45) ---
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # --- Classifier: 64 * 45 = 2880 ---
        self.fc1     = nn.Linear(64 * 45, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)         
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)                


def train_model(net, train_loader, epochs, device):
    """
    Performs local training on an FL client.

    Returns:
        avg_loss (float): average loss of the last epoch
        accuracy (float): accuracy of the last epoch
    """
    criterion = nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train()
    avg_loss, accuracy = 0.0, 0.0     

    for epoch in range(epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(signals)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"  [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    return avg_loss, accuracy     
       

def test_model(net, test_loader, device):
    """
    Evaluates the model on the global test set.

    Returns:
        avg_loss (float)
        accuracy (float)
        f1       (float): weighted F1 — more reliable than accuracy for imbalanced classes
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = net(signals)
            total_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(test_loader)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1