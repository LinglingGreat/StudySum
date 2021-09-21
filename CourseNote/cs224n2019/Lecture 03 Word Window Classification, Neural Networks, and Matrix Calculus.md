### 分类问题

情感分类，命名实体识别，买卖决策等

softmax分类器，cross-entropy损失函数(线性分类器)

神经网络分类器，词向量分类的不同(同时学习权重矩阵和词向量，因此参数也更多)，神经网络简介



### 命名实体识别(NER)

找到文本中的"名字"并且进行分类

- 可能的用途
  - 跟踪文档中提到的特定实体（组织、个人、地点、歌曲名、电影名等）
  - 对于问题回答，答案通常是命名实体
  - 许多需要的信息实际上是命名实体之间的关联
  - 同样的技术可以扩展到其他 slot-filling 槽填充 分类
- 通常后面是命名实体链接/规范化到知识库

难点

- 很难计算出实体的边界
  - 第一个实体是 “First National Bank” 还是 “National Bank”
- 很难知道某物是否是一个实体
  - 是一所名为“Future School” 的学校，还是这是一所未来的学校？
- 很难知道未知/新奇实体的类别
  - “Zig Ziglar” ? 一个人
- 实体类是模糊的，依赖于上下文
  - 这里的“Charles Schwab” 是 PER 不是 ORG

**在上下文语境中给单词分类，怎么用上下文？**

将词及其上下文词的向量连接起来

假设我们要对中心词是否为一个地点，进行分类。与word2vec类似，我们将遍历语料库中的所有位置。如果这个词在上下文中是表示位置，给高分，否则给低分。

与word2vec类似，可以采样一些“损坏窗口”作为负样本。



### 梯度



神经网络，最大边缘目标函数，反向传播



### 技巧

梯度检验

```
def eval_numerical_gradient(f, x):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient  
    at
    """

    f(x) = f(x) # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index',
                     op_flags=['readwrite'])

    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxh_left = f(x) # evaluate f(x + h)
        x[ix] = old_value - h # decrement by h
        fxh_right = f(x) # evaluate f(x - h)
        # restore to previous value (very important!)
        x[ix] = old_value 

        # compute the partial derivative
        # the slope
        grad[ix] = (fxh_left - fxh_right) / (2 * h)
        it.iternext() # step to next dimension
    return grad
```



正则

Dropout

激活函数(sigmoid, tanh, relu)

数据预处理

- 减去均值——注意在训练集、验证集和测试集都是减去同一平均值
- 标准化——将每个输入特征维度缩小，使得具有相似的幅度范围没有减去均值有用
- 白化Whitening——不常用，首先对数据进行减去均值处理得到X'，然后进行奇异值分解得到U, S, V，计算UX'将X'投影到由U的列定义的基上。最后将结果的每个维度除以 SS 中的相应奇异值，从而适当地缩放我们的数据（如果其中有奇异值为 0 ，我们就除以一个很小的值代替）。

参数初始化——一般将权重初始化为通常分布在0附近的很小的随机数，xavier初始化

学习策略

优化策略(momentum, adaptive)

```
# Computes a standard momentum update
# on parameters x
v = mu * v - alpha * grad_x
x += v
```

AdaGrad：每个参数的学习率取决于每个参数梯度更新的历史,参数的历史更新越小，就使用更大的学习率加快更新。换句话说，过去没有更新太大的参数现在更有可能有更高的学习率。

如果梯度的历史 **RMS** 很低，那么学习率会非常高。

```
# Assume the gradient dx and parameter vector x
cache += dx ** 2
x += -learning_rate * dx / np.sqrt(cache + 1e-8)
```

**RMSProp** 是利用平方梯度的移动平局值，是 **AdaGrad** 的一个变体——实际上，和 **AdaGrad** 不一样，**它的更新不会单调变小**。

**Adam** 更新规则又是 **RMSProp** 的一个变体，但是加上了动量更新。

```
# Update rule for RMS prop
cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
x += -learning_rate * dx / (np.sqrt(cache) + eps)

# Update rule for Adam
m = beta * m + (1 - beta1) * dx
v = beta * v + (1 - beta2) * (dx ** 2)
x += -learning_rate * m / (np.sqrt(v) + eps)
```

梯度优化算法的具体细节：

 [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)





### 参考资料

[https://looperxx.github.io/CS224n-2019-03-Word%20Window%20Classification,Neural%20Networks,%20and%20Matrix%20Calculus/](https://looperxx.github.io/CS224n-2019-03-Word Window Classification,Neural Networks, and Matrix Calculus/)