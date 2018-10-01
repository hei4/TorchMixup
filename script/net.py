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


class Generator(nn.Module):
    def __init__(self, nz=100, ng_ch=64, nch=3):
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, ng_ch * 8, 4, 1, 0),
                nn.BatchNorm2d(ng_ch * 8),
                nn.ReLU()
            ),  # (N, 100, 1, 1) -> (N, ng_ch*8, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(ng_ch * 8, ng_ch * 4, 4, 2, 1),
                nn.BatchNorm2d(ng_ch * 4),
                nn.ReLU()
            ),  # (N, ng_ch*8, 4, 4) -> (N, ng_ch*4, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(ng_ch * 4, ng_ch * 2, 4, 2, 1),
                nn.BatchNorm2d(ng_ch * 2),
                nn.ReLU()
            ),  # (N, ng_ch*4, 8, 8) -> (N, ng_ch*2, 16, 16)

            'layer3': nn.Sequential(
                nn.ConvTranspose2d(ng_ch * 2, ng_ch, 4, 2, 1),
                nn.BatchNorm2d(ng_ch),
                nn.ReLU()
            ),  # (N, ng_ch*2, 16, 16) -> (N, ng_ch, 32, 32)
            'layer4': nn.Sequential(
                nn.ConvTranspose2d(ng_ch, nch, 4, 2, 1),
                nn.Tanh()
            )   # (N, ng_ch, 32, 32) -> (N, nch, 64, 64)
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, nch=3, nd_ch=64):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nch, nd_ch, 4, 2, 1),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (N, nch, 64, 64) -> (N, nd_ch, 32, 32)
            'layer1': nn.Sequential(
                nn.Conv2d(nd_ch, nd_ch * 2, 4, 2, 1),
                nn.BatchNorm2d(nd_ch * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (N, nd_ch, 32, 32) -> (N, nd_ch*2, 16, 16)
            'layer2': nn.Sequential(
                nn.Conv2d(nd_ch * 2, nd_ch * 4, 4, 2, 1),
                nn.BatchNorm2d(nd_ch * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (N, nd_ch*2, 16, 16) -> (N, nd_ch*4, 8, 8)
            'layer3': nn.Sequential(
                nn.Conv2d(nd_ch * 4, nd_ch * 8, 4, 2, 1),
                nn.BatchNorm2d(nd_ch * 8),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (N, nd_ch*4, 8, 8) -> (N, ng_ch*8, 4, 4)
            'layer4': nn.Conv2d(nd_ch * 8, 1, 4, 1, 0)
            # (N, nd_ch*8, 4, 4) -> (N, ng_ch*8, 1, 1)
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x.view(-1, 1).squeeze(1)
