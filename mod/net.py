import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, d_in, d_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.fc2 = nn.Linear(d_in, 8)
        self.fc3 = nn.Linear(8, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Trainer:

    def __init__(self, dataset):
        assert len(dataset), "[x] Require 1 test case at least"
        label_size = len(dataset[0][1])
        input_size = len(dataset[0][0])
        self.dataset = dataset
        self.net = Net(input_size, label_size).float()
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

    def is_match(self, y, y_pred):
        temp = (y > 0.5) == (y_pred > 0.5)
        temp = temp.type(torch.float)
        return 1 if temp.sum() == len(y) else 0

    def topk(self, x, y, k=5):
        x.requires_grad = True
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            top_k = np.array(x.grad).argsort()[-k:][::-1]
            print(top_k)

    def train(self):

        for epoch in range(100):
            accuracy = 0
            for (x, y) in self.dataset:
                y_pred = self.net(x)
                loss = self.loss_fn(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    accuracy += self.is_match(y, y_pred) / len(self.dataset) * 100

