import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten: 784
        x = x.view(x.size(0), 784)
        # linear: 784 -> 512 + relu
        x = F.relu(self.linear1(x))
        # linear: 512 -> 128 + relu
        x = F.relu(self.linear2(x))
        # linear: 128 -> num_classes
        x = self.linear3(x)
        return x


class Convolutional(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape: 1@28x28
        x = x.view(x.size(0), 1, 28, 28)
        # convolution: 1@28x28 -> 16@24x24 + relu
        x = F.relu(self.conv1(x))
        # max pooling: 16@24x24 -> 32@12x12
        x = F.max_pool2d(x, 2)
        # convolution: 16@12x12 -> 32@8x8
        x = F.relu(self.conv2(x))
        # max pooling: 32@8x8 -> 32@4x4
        x = F.max_pool2d(x, 2)
        # flatten: 32@12x12 -> 512
        x = x.view(x.size(0), 512)
        # linear: 512 -> 128 + relu
        x = F.relu(self.fc1(x))
        # linear: 128 -> 64 + relu
        x = F.relu(self.fc2(x))
        # linear: 64 -> num_classes
        x = self.fc3(x)
        return x


class Recurrent(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(28, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # init hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # reshape: 28x28
        x = x.view(x.size(0), 28, 28)
        # gru: 28x28 -> 64x28 (x2)
        x, h = self.gru(x, h0)
        # linear: 64 -> num_classes
        x = self.linear(x[:, -1, :])
        return x


def get_model(model: str) -> nn.Module:
    if model in {'feedforward', 'ffnn'}:
        return FeedForward()
    elif model in {'convolutional', 'cnn'}:
        return Convolutional()
    elif model in {'recurrent', 'rnn'}:
        return Recurrent()
    else:
        raise ValueError(f'Unknown model: {model}')
