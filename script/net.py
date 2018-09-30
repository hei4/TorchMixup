# -*- coding: utf-8 -*-
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_layers = nn.ModuleDict({
            'conv_layer0': nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),  # (N, 3, 32, 32) -> (N, 32, 16, 16)
            'conv_layer1': nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),  # (N, 32, 16, 16) -> (N, 64, 8, 8)
            'conv_layer2': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )   # (N, 64, 8, 8) -> (N, 128, 4, 4)
        })

        self.fc_layers = nn.ModuleDict({
            'fc_layer0': nn.Sequential(
                nn.Linear(128 * 4 * 4, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
            ),  # (N, 2048) -> (N, 1024)
            'fc_layer1': nn.Linear(1024, 10)    # (N, 1024) -> (N, 10)
        })

    def forward(self, x):
        for conv_layer in self.conv_layers.values():
            x = conv_layer(x)

        x = x.view(-1, 128 * 4 * 4)

        for fc_layer in self.fc_layers.values():
            x = fc_layer(x)

        return x




