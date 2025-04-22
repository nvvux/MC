import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# 2. Định nghĩa mô hình MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # Hidden layer với 128 neuron
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 lớp)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten ảnh
        x = F.relu(self.fc1(x))  # Hidden + ReLU
        x = self.fc2(x)  # Output logits
        return x


# 3. Khởi tạo mô hình, loss, và optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_values = []  # Trước vòng lặp huấn luyện
# 4. Huấn luyện mô hình
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_values.append(total_loss)  # Ghi lại loss
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

plt.plot(range(1, epochs + 1), loss_values, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()