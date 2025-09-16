import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import time

HEADLESS = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet5Improved(nn.Module):
    def __init__(self):
        super(LeNet5Improved, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.dropout = nn.Dropout(0.3)  # Giảm xuống 0.3

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDigitsDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = []
        self.labels = []

        if not os.path.exists(directory):
            raise Exception(f"Directory does not exist: {directory}")
        if not os.access(directory, os.R_OK):
            raise Exception(f"Cannot access the directory: {directory}")

        for subdir, dirs, files in os.walk(directory):
            if subdir != directory:
                try:
                    label = int(subdir.split("/")[-1])
                    if label < 0 or label > 9:
                        print(f"Skipping folder: {subdir} (Not a valid label)")
                        continue
                except ValueError:
                    print(f"Skipping folder: {subdir} (Invalid label)")
                    continue
                
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_files.append(os.path.join(subdir, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise Exception(f"Không thể đọc ảnh: {img_path}")
        
        image = cv2.resize(image, (28, 28))

        # (Tuỳ chọn) Nếu ảnh có nền trắng thì đảo ngược lại
        if np.mean(image) > 127:
            image = 255 - image

        # (Tuỳ chọn) Bỏ threshold nếu muốn giữ lại mức xám
        # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def train_model():
    num_epochs = 30  # Tăng lên 30
    batch_size = 64
    learning_rate = 0.001

    transform_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # Bỏ RandomHorizontalFlip(), RandomResizedCrop
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    custom_dir = 'data/handwritten_digits_train'
    train_dataset_full = CustomDigitsDataset(directory=custom_dir, transform=transform_train)
    
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet5Improved().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Điều chỉnh StepLR: giảm lr sớm hơn (mỗi 3 epoch)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    model_path = "models/cnn_model.pth"
    best_model_path = "models/best_cnn_model.pth"
    if os.path.exists(model_path):
        print("Tải mô hình đã lưu từ trước...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Không có mô hình đã lưu, bắt đầu huấn luyện từ đầu.")

    best_val_acc = 0.0

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    patience = 10
    trigger_times = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss = running_loss_val / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()  # Update learning rate

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Lưu mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Mô hình tốt nhất đã được lưu tại {best_model_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Mô hình đã được lưu tại {model_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Thời gian huấn luyện: {elapsed_time:.2f} giây")

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.tight_layout()
    if HEADLESS:
        plt.savefig("training_curves.png")
        print("Biểu đồ tiến độ huấn luyện đã được lưu tại training_curves.png")
    else:
        plt.show()

if __name__ == "__main__":
    train_model()
