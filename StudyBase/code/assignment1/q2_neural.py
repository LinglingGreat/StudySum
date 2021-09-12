#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- N x Dx matrix, where each row is a training example.
    labels -- N x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    N = data.shape[0]
    #样本 = 20， 特征=10，节点 = 5， 类别 = 10
    #forward propagation                   
    A0 = data                             # shape = (样本，特征) (20, 10)
    Z1 = np.dot(A0, W1) + b1              # shape = (样本, 节点) (20, 5)
    A1 = sigmoid(Z1)                      # shape = (样本, 节点) (20, 5)
    Z2 = np.dot(A1, W2) + b2              # shape = (样本，类别) (20, 10)
    A2 = softmax(Z2)                      # shape = (样本，类别) (20, 10)

    #compute the cost
    target = np.argmax(labels, axis=1)    # 得到所有样本正确类别的索引，target是[2,3,1,6,7]这样的向量
    cost_each = -np.log(A2[range(N), target]).reshape(-1, 1)   #shape = (样本, 1), 计算li
    cost = np.mean(cost_each, axis=0)        #shape =  (1,), 计算L

    #backward propagation
    dZ2 = (A2 - labels) / N                     # shape = (样本，类别) (20, 10)
    dW2 = np.dot(A1.T, dZ2)                     # shape = (节点， 特征) (5, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True)    # shape = (1, 类别)  (1, 10)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_grad(A1)  # shape = (1, 类别)  (1, 10)
    dW1 = np.dot(A0.T, dZ1)                     # shape = (样本, 节点)(20, 5)
    db1 = np.sum(dZ1, axis=0, keepdims=True)    # shape = (1, 节点)   (1, 5)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
