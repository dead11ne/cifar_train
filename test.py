import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Определим те же преобразования, что и при обучении
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Загрузим тестовые данные
testset = torchvision.datasets.CIFAR10(
    root='C:/Users/MSI/Desktop/cifar_train/data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=16, 
    shuffle=True
)

# Классы CIFAR-10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Определим модель (такая же архитектура как при обучении)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Создадим модель и загрузим сохраненные веса
model = Net()
model.load_state_dict(torch.load('cifar10_model_state_dict.pth'))
model.eval()  # Переводим модель в режим оценки

# Функция для отображения изображения
def imshow(img):
    img = img / 2 + 0.5  # Денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Получим несколько изображений
dataiter = iter(testloader)
images, labels = next(dataiter)

# Покажем изображения
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))

# Сделаем предсказания
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(16)))

# Вычислим точность на тестовом наборе
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')