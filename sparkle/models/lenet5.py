import torch
from torch import nn, optim
import torch.nn.functional as F



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
            nn.Softmax()
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