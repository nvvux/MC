import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Tạo transform cho dữ liệu MNIST
transform = transforms.ToTensor()

# Tải bộ dữ liệu MNIST
mnist_train = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tính toán chính xác số lượng đặc trưng đầu vào cho lớp fully connected
        # 28x28 -> 14x14 (sau max pool)
        # 14x14 -> 7x7 (sau max pool)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # Cập nhật đúng số đặc trưng
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool1(x)  # Max Pooling 1
        x = F.relu(self.conv3(x))  # Conv3 + ReLU
        x = F.relu(self.conv4(x))  # Conv4 + ReLU
        x = self.pool2(x)  # Max Pooling 2
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected 1 + ReLU
        x = self.fc2(x)  # Fully connected 2 (output)
        return x  # Return logits thô

# Kiểm tra nếu có GPU và chuyển mô hình vào GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Khởi tạo mô hình CNN
model = CNN().to(device)

# Đặt các hyperparameters
epochs = 3
learning_rate = 0.001

# Chọn optimizer và loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Huấn luyện mô hình
for epoch in trange(epochs, desc="Training Epochs"):
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu vào GPU

        optimizer.zero_grad()  # Làm sạch gradients
        outputs = model(images)  # Tiến hành forward pass
        loss = criterion(outputs, labels)  # Tính loss
        loss.backward()  # Tính toán gradients
        optimizer.step()  # Cập nhật weights

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Đánh giá mô hình trên tập kiểm tra
model.eval()  # Đặt mô hình ở chế độ đánh giá
correct = 0
total = 0

with torch.no_grad():  # Tắt gradient khi đánh giá
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu vào GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Lấy chỉ số lớp có xác suất cao nhất
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
