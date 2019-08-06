# !/usr/bin/python

from functools import reduce

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np

from sparkle.utils import train, test
from sparkle.utils.metrics import accuracy
from sparkle.train import App, Trainer, Checkpoint


class MNet(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(MNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden)
        # self.layer1.weight = torch.nn.Parameter(torch.zeros(self.layer1.weight.size()))
        self.layer2 = nn.Linear(n_hidden, out_dim)
        # self.layer2.weight = torch.nn.Parameter(torch.zeros(self.layer2.weight.size()))

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        # x = F.softmax(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential( # 1, 28, 28
            nn.Conv2d(1, 16, 5),    # 16, 24, 24
            nn.ReLU(),
            nn.MaxPool2d(2),        # 16, 12, 12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),   # 32, 8, 8
            nn.ReLU(),
            nn.MaxPool2d(2)         # 32, 4, 4
        )
        self.out = nn.Sequential(
            nn.Linear(32 * 4 * 4, 10),
            # nn.Softmax()
        )
    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        # import pdb; pdb.set_trace()
        x_dim = reduce(lambda a,b: a*b, x.shape[1:], 1)
        x = x.view(batch_size, x_dim)
        x = self.out(x)
        return x


if __name__ == "__main__":
    batch_size = 128

    train_set = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set)


    app = Checkpoint(Trainer(App(model=LeNet5(), criterion=nn.CrossEntropyLoss())))

    @app.on("iter_started")
    def mnist_train(e):
        # import pdb; pdb.set_trace()
        inputs, labels = e.batch
        e.model.zero_grad()
        y_predict = e.model(inputs)
        loss = e.criterion(y_predict, labels)
        loss.backward()
        e.optimizer.step()
        print(loss)

    app.fastforward()   \
       .save_every(iters=500)   \
       .set_optimizer(optim.Adam, lr=0.0015, eps=1e-4)  \
       .to("cpu")  \
       .half()  \
       .run(train_loader, max_iters=4000)

    yp, y = app.eval(test_loader)
    yp = [torch.argmax(p, dim=-1) for p in yp]
    acc = accuracy(yp, y)
    print(acc)