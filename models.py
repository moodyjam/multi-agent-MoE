from torch import nn
import torch.nn.functional as F
import torch

# Taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class SimpleImageEncoder(nn.Module):
    def __init__(self, prototype_dim=512):
        super(SimpleImageEncoder, self).__init__()

        # Reduce the number of filters in each layer to decrease the model size
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)   # Output: 8x32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # Output: 16x16x16
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: 32x8x8

        # Flatten and pass through a linear layer to get the desired output dimension
        self.fc = nn.Linear(32 * 8 * 8, prototype_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the features into a vector
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, num_classes, prototype_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(prototype_dim, 128)  # First layer: 128 to 64 units
        self.fc2 = nn.Linear(128, 128)  # First layer: 128 to 64 units
        self.fc3 = nn.Linear(128, num_classes)  # Second layer: 64 to num_classes units

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))  # Output layer, no activation here to allow for flexibility (e.g., softmax for classification)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x
    
class SimpleImageClassifier(nn.Module):
    def __init__(self, prototype_dim=512, num_classes=50):
        super(SimpleImageClassifier, self).__init__()

        # Reduce the number of filters in each layer to decrease the model size
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)   # Output: 8x32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # Output: 16x16x16
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: 32x8x8

        # Flatten and pass through a linear layer to get the desired output dimension
        self.fc = nn.Linear(32 * 8 * 8, prototype_dim)
        self.fc1 = nn.Linear(prototype_dim, 128)  # First layer: 128 to 64 units
        self.fc2 = nn.Linear(128, 128)  # First layer: 128 to 64 units
        self.fc3 = nn.Linear(128, num_classes)  # Second layer: 64 to num_classes units

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the features into a vector
        x = self.fc(x)
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))  # Output layer, no activation here to allow for flexibility (e.g., softmax for classification)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
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