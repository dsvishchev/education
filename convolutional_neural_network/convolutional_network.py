import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# Создание трансформаций изображений для обучения и тестирования
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Задание пути к данным
train_path = "../data/train"
test_path = "../data/val"

# Создание датасета для обучения и тестирования
train_dataset = ImageFolder(train_path, transform=transform_train)
test_dataset = ImageFolder(test_path, transform=transform_test)

# Создание загрузчика данных
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Определение сверточной нейронной сети
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)

# Описание функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
model.train()
epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}")
            running_loss = 0.0
    print(f"\nEpoch {epoch + 1} loss: {running_loss / len(train_loader)}")

# Оценка модели на тестовом наборе данных
model.eval()
correct = 0
total = 0
# отключим вычисление градиентов, т.к. будем делать только прямой проход
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images: %d %%' % (100 * correct / total))


def get_class_name(predicted):
    match predicted:
        case 0:
            return "Cat"
        case 1:
            return "Dog"
        case 2:
            return "Wild"


def test_one_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    trans_image = transform(Image.open(image_path)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(trans_image)
        _, predicted = torch.max(output.data, 1)
    pred = predicted.item()

    class_name = get_class_name(pred)

    # Нанесение текста на изображение
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = class_name
    position = (50, 50)
    font_scale = 1
    color = (255, 0, 0)  # Синий цвет
    thickness = 2
    image = cv2.imread(image_path)
    cv2.putText(image, text, position, font, font_scale, color, thickness)

    # Отображение изображения
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_one_image("../data/val/cat/flickr_cat_000011.jpg")
test_one_image("../data/val/dog/flickr_dog_000045.jpg")
test_one_image("../data/val/wild/flickr_wild_000040.jpg")