```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```



### one-hot向量

用到了scatter_函数

```python
def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result

x = torch.tensor([0, 2])
x_one_hot = one_hot(x, vocab_size)
print(x_one_hot)
print(x_one_hot.shape)
print(x_one_hot.sum(axis=1))

tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.]])
torch.Size([2, 1027])
tensor([1., 1.])
```

scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。

参考：https://blog.csdn.net/qq_16234613/article/details/79827006

```python
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)   # 输入（批量大小, 时间步数）
inputs = to_onehot(X, vocab_size) # 得到（批量大小, 词典大小）的矩阵，矩阵个数等于时间步数
print(len(inputs), inputs[0].shape)
# 5 torch.Size([2, 1027])
```





### 初始化模型参数

```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数
# num_outputs: q

def get_params():
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)
```



### 裁剪梯度

```python
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
```



### 预测函数

给定输入prefix，预测接下来的num_chars个字符。

```python
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
```

首先初始化隐藏状态state为一个(1, num_hiddens)的零向量，假设输入的prefix是“分开”两个字符，num_chars是10，即预测接下来的10个字符。首先output列表的初始值设为“分开”的第一个字符“分”对应的index，将这个index转化为one_hot向量，作为输入X，通过rnn，得到Y和隐藏状态。这里的X是前一个字符，Y是预测X的后一个字符。所以当t小于输入字符的长度时，Y就是X的真实下一个字符。即“分”的下一个字符是“开”，但是“开”的下一个字符呢？不知道，需要通过rnn预测得到。

这个过程无法并行。因为只有通过X得到预测的Y，才能有下一步的输入X。

最后将所有的Y转化为字符输出即可。



当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播

detach():

返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad .这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播

detach_()

将一个Variable从创建它的图中分离，并把它设置成叶子variable .其实就相当于变量之间的关系本来是x -> m -> y,这里的叶子variable是x，但是这个时候对m进行了.detach_()操作,其实就是进行了两个操作： 

 1.将m的grad_fn的值设置为None,这样m就不会再与前一个节点x关联，这里的关系就会变成x, m -> y,此时的m就变成了叶子结点。

2.然后会将m的requires_grad设置为False，这样对y进行backward()时就不会求m的梯度。

﻿

其实detach()和detach_()很像，两个的区别就是detach_()是对本身的更改，detach()则是生成了一个新的variable 。比如x -> m -> y中如果对m进行detach()，后面如果反悔想还是对原来的计算图进行操作还是可以的 。但是如果是进行了detach_()，那么原来的计算图也发生了变化，就不能反悔了。

﻿

参考https://www.cnblogs.com/wanghui-garcia/p/10677071.html；参考：https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-autograd/#detachsource。



### 困惑度

我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下[“softmax回归”](https://www.kesci.com/api/notebooks/chapter_deep-learning-basics/softmax-regression.ipynb)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

- 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
- 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
- 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。

显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小`vocab_size`。



