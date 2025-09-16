import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

HEADLESS = True  # Đặt True nếu chạy môi trường không có GUI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Kiến trúc LeNet5Improved (đồng bộ với file train)
# -------------------------------
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

def test_model():
    # Định nghĩa transform giống lúc huấn luyện test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Khởi tạo mô hình và tải trọng số đã huấn luyện
    model = LeNet5Improved().to(device)
    
    best_model_path = "models/best_cnn_model.pth"  # Đường dẫn đến mô hình tốt nhất
    if not os.path.exists(best_model_path):
        print(f"Không tìm thấy mô hình tốt nhất tại '{best_model_path}'. Vui lòng huấn luyện mô hình trước.")
        return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    custom_dir = "data/handwritten_digits_train"  # Đường dẫn tới thư mục ảnh thô
    if not os.path.exists(custom_dir):
        print(f"Không tìm thấy thư mục ảnh {custom_dir}.")
        return
    
    # Lấy danh sách các tệp ảnh trong thư mục (bao gồm cả nhãn từ cấu trúc thư mục)
    image_files = []
    for subdir, dirs, files in os.walk(custom_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(subdir, file))

    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục:", custom_dir)
        return

    # Các biến đếm để tính độ chính xác
    total_images = 0
    correct_predictions = 0

    # Tạo thư mục kết quả chính
    os.makedirs("results", exist_ok=True)

    for img_file in image_files:
        # Đọc ảnh từ tệp
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Không thể đọc ảnh: {img_file}")
            continue

        # Nếu ảnh có nền trắng, đảo ngược lại
        if np.mean(image) > 127:
            image = 255 - image

        # Không áp dụng threshold để giữ lại mức xám (đồng bộ với file train)
        # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        # Thay đổi kích thước ảnh về 28x28 (kích thước mong muốn)
        image = cv2.resize(image, (28, 28))

        # Biến đổi ảnh thành tensor và chuẩn hóa
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

        # Lấy nhãn thật từ tên thư mục (giả sử thư mục cha của ảnh là nhãn)
        true_label = None
        parent_dir = os.path.basename(os.path.dirname(img_file))
        try:
            true_label = int(parent_dir)
        except ValueError:
            pass

        # Nếu có nhãn thật, cập nhật bộ đếm
        if true_label is not None:
            total_images += 1
            if predicted_label == true_label:
                correct_predictions += 1

        # Chọn nhãn để tạo thư mục con: dùng nhãn thật nếu có, ngược lại dùng nhãn dự đoán
        label_to_save = true_label if true_label is not None else predicted_label

        # Tạo thư mục con trong "results" tương ứng với nhãn
        save_dir = os.path.join("results", str(label_to_save))
        os.makedirs(save_dir, exist_ok=True)

        # Hiển thị kết quả trên ảnh
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        if true_label is not None:
            plt.title(f"Dự đoán: {predicted_label} - Thực: {true_label}")
        else:
            plt.title(f"Dự đoán: {predicted_label}")
        plt.axis('off')

        # Lưu ảnh kết quả vào thư mục con tương ứng
        result_path = os.path.join(save_dir, f"result_{os.path.basename(img_file)}")
        if HEADLESS:
            plt.savefig(result_path)
            print(f"Kết quả nhận diện của {img_file} được lưu tại: {result_path}")
            plt.close()
        else:
            plt.show()

    # Tính và in độ chính xác của mô hình
    if total_images > 0:
        accuracy = correct_predictions / total_images * 100
        print(f"Kết quả train của mô hình: {accuracy:.2f}% trên {total_images} ảnh.")
    else:
        print("Không có ảnh nào có nhãn thực để tính toán độ chính xác.")

if __name__ == "__main__":
    test_model()

