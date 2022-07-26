# -*- coding: utf-8 -*-
# @Time    : 2020/2/14 21:42
# @Author  : LiLing
import torch
from matplotlib import pyplot as plt
import numpy as np
import random

print(torch.__version__)


# ======================================================
# -------------------从零开始实现-----------------------
# ======================================================
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield features.index_select(0, j), labels.index_select(0, j)  # 0 表示按行索引


def linreg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # ues .data to operate param without gradient track


# set input feature number
num_inputs = 2
# set example number
num_examples = 1000
# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2
# 标准正态分布
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# numpy类型的数据转成torch的tensor
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# another method
labels2 = labels+torch.empty(labels.size()).normal_(0, 0.01)

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 当前量是否需要在计算中保留对应的梯度信息
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# super parameters init
lr = 0.03
num_epochs = 5
batch_size = 10

net = linreg
loss = squared_loss

# training
for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once

    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        # calculate the gradient of batch sample loss
        l.backward()
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(w, true_w, b, true_b)


# ======================================================
# ----------使用pytorch的简洁实现-----------------------
# ======================================================
import torch
from torch import nn
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()  # call father function to init
        self.linear = nn.Linear(n_feature,
                                1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y


import torch.utils.data as Data

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)

net = LinearNet(num_inputs)
print(net)

# ways to init a multilayer network
# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly

for param in net.parameters():
    print(param)

loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# result comparision
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
