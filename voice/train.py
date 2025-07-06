import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import AudioDataset
from model import AudioCNN
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載資料集
dataset = AudioDataset()
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=32, shuffle=False)

# 初始化模型
num_classes = len(dataset.cat_strs)
model = AudioCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 訓練與驗證函數
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(dim=1)).sum().item()

        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs) 
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets.argmax(dim=1)).sum().item()

        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)



def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies, label='Verification accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Verification loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()



train_model(model, train_loader, val_loader, criterion, optimizer, epochs=80)
torch.save(model.state_dict(), 'voice/model/audio_cnn_model6.pth')
