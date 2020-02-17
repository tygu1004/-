'''
Resnet50 for project4
'''
import time
import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

plt.ion()

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")  #using GPU
EPOCHS = 30  #set epoch

trainset = datasets.ImageFolder(root="/content/gdrive/My Drive/trainset/train", 
                                transform=transforms.Compose([
                                transforms.Scale(128),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
]))

validset = datasets.ImageFolder(root="/content/gdrive/My Drive/trainset/test", 
                                transform=transforms.Compose([
                                transforms.Scale(128),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
]))

train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=4,
)

test_loader = torch.utils.data.DataLoader(validset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=4,
)

inputs, classes = next(iter(train_loader))

out = torchvision.utils.make_grid(inputs)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean   
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

imshow(out)

net = models.resnet50(pretrained=True) 
for param in net.parameters():
    param.requires_grad = True
net.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 1000)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(1000, 512)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(512, 2)),
]))
if os.path.exists("/content/gdrive/My Drive/params.ckpt"):
    net.load_state_dict(torch.load('/content/gdrive/My Drive/params.ckpt'))
model = net.to(DEVICE)

lossdata = []

def train(model, train_loader, optimizer, epoch): #train model
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        lossdata.append(loss)
        optimizer.step()

        if batch_idx % 150 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

def test(model, test_loader): #test model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

start = time.time() #check start train time

for epoch in range(1, EPOCHS + 1): #train and test
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, train_loader, optimizer, epoch)
    #test_loss, test_accuracy = test(model, test_loader)
  
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch,
        test_loss,
        test_accuracy
    ))
    
end = time.time() #check end train time
print('Total Time: {:.4f}'.format(end-start))
torch.save(net.state_dict(), '/content/gdrive/My Drive/params.ckpt')
plt.plot(lossdata)