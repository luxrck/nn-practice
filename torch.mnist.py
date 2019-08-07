# !/usr/bin/python
import pdb
from functools import reduce

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np

from sparkle.utils.metrics import accuracy
from sparkle.train import App, Trainer, Checkpoint

from sparkle.models import LeNet5


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



if __name__ == "__main__":
    batch_size = 128

    train_set = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set)


    app = Checkpoint(Trainer(App(model=LeNet5(), criterion=nn.CrossEntropyLoss())))

    @app.on("train")
    def mnist_train(e):
        inputs, labels = e.batch
        e.model.zero_grad()
        y_predict = e.model(inputs)
        loss = e.criterion(y_predict, labels)
        loss.backward()
        e.optimizer.step()
        e.a.loss = loss.item()

    @app.on("iter_completed")
    def logging(e):
        if e.current_iter % 200 == 0:
            print(e.a.loss)

    @app.on("evaluate")
    def mnist_eval(e):
        inputs, targets = e.batch
        y_predict = e.model(inputs)
        return y_predict, targets

    app.fastforward()   \
       .save_every(iters=2000)   \
       .set_optimizer(optim.Adam, lr=0.0015, eps=1e-4)  \
       .to("cpu")  \
       .run(train_loader, max_iters=20)    \
    
    pdb.set_trace()
    yp, y = app.eval(test_loader)
    acc = accuracy(yp, y)
    print(acc)
