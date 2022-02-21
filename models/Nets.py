#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNCifarStd5(nn.Module):
    """659,818: """
    def __init__(self, args):
        super(CNNCifarStd5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMNIST(nn.Module):
    """7,106"""
    def __init__(self, args):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(len(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    # def __init__(self, args):
    #     super(CNNMNIST, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
    #     self.pool1 = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(8, 8, 3, bias=False)
    #     self.pool2 = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(7 * 7 * 8, 16)
    #     self.fc2 = nn.Linear(16, 10)

    # def forward(self, x):
    #     x = self.pool1(F.relu(self.conv1(x)))
    #     x = self.pool2(F.relu(self.conv2(x)))
    #     # x = x.view(-1, 7 * 7 * 8)
    #     x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    #     x_out = F.relu(self.fc1(x))
    #     x = self.fc2(x_out)
    #     return x, x_out


class CNNEmnistStd5(nn.Module):
    """438,074: """
    # def __init__(self, args):
    #     super(CNNEmnistStd5, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
    #     self.pool1 = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
    #     self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    #     self.pool2 = nn.MaxPool2d(2, 2)
    #     self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    #     self.conv5 = nn.Conv2d(64, 64, 3, padding=0, bias=False)
    #     self.pool3 = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(64 * 3 * 3, 512)
    #     self.fc2 = nn.Linear(512, 26)

    # def forward(self, x):
    #     x = self.pool1(F.relu(self.conv1(x)))
    #     x = F.relu(self.conv2(x))
    #     x = self.pool2(F.relu(self.conv3(x)))
    #     x = F.relu(self.conv4(x))
    #     x = self.pool3(F.relu(self.conv5(x)))
    #     x = x.view(-1, 64 * 3 * 3)
    #     x_out = F.relu(self.fc1(x))
    #     x = self.fc2(x_out)
    #     return F.log_softmax(x, dim=1)
    def __init__(self, args):
        super(CNNEmnistStd5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 64 * 3 * 3)
        out = self.fc(out)
        return out
