# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        # KMNIST dataset with 28x28 image = 784 pixels
        self.main = nn.Sequential(
            #First use a linear function to transform input size to output size
            nn.Linear(784, 10), # We have input 784 pixel and a output of 10 characters, defining input and output size
            nn.LogSoftmax(dim=1) # Followed by log_softmax
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) #Reshape
        x = self.main(x) #linear function
        return x # CHANGE CODE HERE


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.main = nn.Sequential(
            nn.Linear(784, 140), # Linear Transform to 140 nodes
            nn.Tanh(), # Tanh Layer
            nn.Linear(140,10), # Linear Transform to 10 nodes
            nn.LogSoftmax(dim=1) # followed by log_softmax
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.main(x)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        # 2 Convolution Layer
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5), # First layer, attempt extracting more layer with larger kernel size
            nn.ReLU(),
            nn.MaxPool2d(5, stride=1), # Max Pooling to reduce size of previous, further reduce total operation
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3), # Second layer, abstract from previous, and reduce to326 features
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1), # Max Pooling to reduce size, note this affects the upcoming linear layer size
        )

        #Follow by a lineary layer then output layer
        self.linear_out = nn.Sequential(
            nn.Linear(4096, 126), #Fully Connected Layer
            nn.ReLU(),
            nn.Linear(126, 10), #Output to 10
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x)
        #Now we need to flatten before inputting to fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.linear_out(x)
        return x
