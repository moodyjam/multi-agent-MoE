from torch import nn
import torch.nn.functional as F
import torch

# Taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x
    
    
# From the code
class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters=3, kernel_size=5, linear_width=64, num_nodes=10, num_labels=10):
        super().__init__()
        conv_out_width = 28 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)
        self.linear_width = linear_width
        self.fc1_indim = fc1_indim
        self.num_labels = num_labels

        # Our f network
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.fc1_indim, self.linear_width))
            
        print(sum(p.numel() for p in self.encoder.parameters()))
        
        self.prototype = nn.Parameter(torch.randn(self.linear_width))
        # self.prototype = nn.Parameter(torch.randn(self.linear_width))

        self.specialist = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.fc1_indim, self.linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_width, self.num_labels),
            nn.LogSoftmax(dim=1),
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def get_scores(self, x):
        print()

    def forward(self, x):
        return self.seq(x)