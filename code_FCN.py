import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 64
lr = 0.1
epoch = 30
criterion = nn.NLLLoss()

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train = True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', train = False, transform=transform, download = True)
testloader = torch.utils.data.DataLoader(testset, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 10)
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.log_softmax(self.l4(x), dim=1)
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