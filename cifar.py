import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Загрузка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация данных
])

dataset = datasets.CIFAR10(
    root='C:/Users/MSI/Desktop/data',
    train=True,
    download=True,
    transform=transform
)

# Исправлено имя переменной с data на dataloader, чтобы избежать конфликта имен
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True
)

print(f"Размер датасета: {len(dataset)}")
print(f"Форма одного элемента: {dataset[0][0].shape}")

# Исправленная архитектура модели
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),  # Преобразуем 3D тензор в 1D
            nn.Linear(32*32*3, 128),  # CIFAR10 изображения 32x32 с 3 каналами
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 10)  # 10 классов в CIFAR10
        )
    
    def forward(self, x):
        return self.layers(x)
    
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(params=model.parameters(), lr=0.001)

# Цикл обучения
num_epochs = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 99:  # Печатаем каждые 100 мини-батчей
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], '
                  f'Loss: {running_loss/100:.3f}, Accuracy: {100*correct/total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

print('Обучение завершено')

torch.save(model.state_dict(), 'cifar10_model_state_dict.pth')
print("Модель сохранена как cifar10_model_state_dict.pth (только веса)")