import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

batch_size = 64
lr = 0.1
epoch = 30
criterion = nn.NLLLoss()

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Model()
optimizer = optim.SGD(model.parameters(), lr=lr)

train_losses, test_losses = [], []
for e in range(epoch):
    train_loss = 0

    for images, labels in trainloader:
        optimizer.zero_grad()
        op = model(images)
        loss = criterion(op, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        print("Epoch: ", e + 1, "/", epoch, "Training Loss: ", train_loss / len(trainloader))
        train_losses.append(train_loss / len(trainloader))

hit = 0

with torch.no_grad():
    model.eval()
    for images, labels in testloader:
        log_ps = model(images)
        prob = torch.exp(log_ps)
        top_probs, top_classes = prob.topk(1, dim=1)
        equals = labels == top_classes.view(labels.shape)
        hit += equals.type(torch.FloatTensor).sum()

print("accuracy: ", hit.item() * 100 / len(testset), "%")

plt.plot(train_losses, label="Train losses")
plt.legend()
plt.show()