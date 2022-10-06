论文：《[Convolutional Neural Networks for Sentence Classification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1408.5882)》

![img](img/1182656-20180919171920103-1233770993.png)

可以做多标签分类和多分类

## 输入

输入是一个用预训练好的词向量（Word2Vector或者glove）方法得到的一个Embedding layer。

每一个词向量都是通过无监督的方法训练得到的。

两个维度，0轴是单词、1轴是词向量的维度（固定的）。得到了一张二维的图（矩阵）。

## 模型结构

textCNN 其实只有一层卷积,一层max-pooling, 最后将输出外接softmax 来n分类。

[CNN](../../BasicKnow/CNN/CNN.md)

## 卷积(convolution)

相比于一般CNN中的卷积核，这里的卷积核的宽度一般需要跟词向量的维度一样

卷积核的高度则是一个超参数可以设置，比如设置为2、3等

然后剩下的就是正常的卷积过程了。

**文本是一维数据，因此在TextCNN卷积用的是一维卷积**（在**word-level**上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要**通过设计不同 kernel_size 的 filter 获取不同宽度的视野**。

## 池化(pooling)

这里的池化操作是max-overtime-pooling，其实就是在对应的feature map求一个最大值。最后把得到的值做concate。

## 优化、正则化

池化层后面加上全连接层和SoftMax层做分类任务，同时防止过拟合，一般会添加L2和Dropout正则化方法。最后整体使用梯度法进行参数的更新模型的优化。

## **多种模型**

### CNN-rand

作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。

### CNN-static

模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。

### CNN-non-static

同（2），只是训练的时候，Embedding layer跟随整个网络一起训练。

### CNN-multichannel

Embedding layer有两个channel，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。



实验证明，除了随机初始化Embedding layer的外，使用预训练的word2vec初始化的效果都更加好。

非静态的比静态的效果好一些。

总的来看，使用预训练的word2vec初始化的TextCNN，效果更好。

多channels**并没有明显提升模型的分类能力**

## 代码实现

### Pytorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
```





## 实践trick

- textCNN 的优势：模型简单, 训练速度快，效果不错。
- textCNN的缺点：模型可解释型不强，在调优模型的时候，很难根据训练的结果去针对性的调整具体的特征，因为在textCNN中没有类似gbdt模型中特征重要度(feature importance)的概念, 所以很难去评估每个特征的重要度。 

默认的TextCNN模型超参数一般都是这种配置。如下表：

![img](https://pic1.zhimg.com/v2-a0de86fee7c073e95ee325fea3ba21f8_b.jpg)

这里将一下调参的问题，主要方法来自论文[1]。在最简单的仅一层卷积的TextCNN结构中，下面的超参数都对模型表现有影响：

1. 输入词向量表征：词向量表征的选取(如选word2vec还是GloVe)
2. 卷积核大小：一个合理的值范围在1~10。若语料中的句子较长，可以考虑使用更大的卷积核。另外，可以在寻找到了最佳的单个filter的大小后，尝试在该filter的尺寸值附近寻找其他合适值来进行组合。实践证明这样的组合效果往往比单个最佳filter表现更出色
3. feature map特征图个数：主要考虑的是当增加特征图个数时，训练时间也会加长，因此需要权衡好。当特征图数量增加到将性能降低时，可以加强正则化效果，如将dropout率提高过0.5
4. 激活函数：ReLU和tanh是最佳候选者
5. 池化策略：1-max pooling表现最佳
6. 正则化项(dropout/L2)：相对于其他超参数来说，影响较小点



## 参考资料

论文

- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1510.03820)

博客等

[自然语言中的CNN--TextCNN（基础篇）](https://zhuanlan.zhihu.com/p/40276005)

[深入TextCNN（一）详述CNN及TextCNN原理](https://zhuanlan.zhihu.com/p/77634533)

[TextCNN进行文本分类多标签分类](https://blog.csdn.net/weixin_42813521/article/details/104991490)

