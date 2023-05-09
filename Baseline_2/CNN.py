
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

class CNN(nn.Module):
    '''
    Convolutional Neural Network specifically designed for 100 X 150 mel spectograms.
    Has two convolutional layers, 2 max pooling layers, 3 hidden layers and one output layer.
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 23 * 35, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 80)
        self.fc4 = nn.Linear(80, 10)

    def forward(self, x):
        '''
            Propogates the input 'x' to find out the output.
        '''
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # x = x.reshape(nxt.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
