# 循环序列模型

## 为什么选择序列模型

![sequence](img/sequence.png)

## notation

![seqnotation](img/seqnotation.png)

## 循环神经网络模型

Recurrent Neural Networks

Why not a standard network?

Problems:

- Inputs, outputs can be different lengths in different examples.比如每个句子长度不一样
- Doesn't share features learned across different positions of text.比如每个名字在每个句子中出现的位置不一样

![rnn1](img/rnn1.png)

缺点：没有用到序列后面的信息

![rnn2](img/rnn2.png)

![rnn3](img/rnn3.png)

## Backpropagation through time

![rnn4](img/rnn4.png)

## 不同类型的RNN

![rnn5](img/rnn5.png)

举例

One to many:  Music generation

Many to one: Sentiment classification

Many to many: 命名实体识别

Many to many: Machine translation，前半部分是encoder,后半部分是decoder

## 语言模型和序列生成

![rnn6](img/rnn6.png)

![rnn7](img/rnn7.png)

corpus:很长的或数量众多的英文句子组成的文本

Tokenize标记化:比如给定字典，把每个单词变成one-hot向量

EOS: end of sentence

UNK：unknown word，不在字典中

![rnn8](img/rnn8.png)

## 对新序列采样

![rnnsample](img/rnnsample.png)

![rnnchar](img/rnnchar.png)

![rnnseq](img/rnnseq.png)

## Vanishing gradient with RNNs

![rnnvanish](img/rnnvanish.png)

梯度爆炸问题可以通过梯度修剪gradient clipping解决

## GRU单元

![rnngru1](img/rnngru1.png)

![rnngru2](img/rnngru2.png)

![rnngru3](img/rnngru3.png)

## LSTM

![rnnlstm](img/rnnlstm.png)

![rnnlstm2](img/rnnlstm2.png)

## LSTM和GRU的区别？

-   GRU和LSTM的性能在很多任务上不分伯仲。
-   GRU 参数更少因此更容易收敛，但是数据集很大的情况下，LSTM表达性能更好。
-   从结构上来说，GRU只有两个门（update和reset），LSTM有三个门（forget，input，output），GRU直接将hidden state 传给下一个单元，而LSTM则用memory cell 把hidden state 包装起来。

## 双向RNN

Bidirectional RNN

![rnnb1](img/rnnb1.png)

![rnnb2](img/rnnb2.png)

## Deep RNNs

![rnndeep](img/rnndeep.png)

## Dropout在RNN中的应用综述

Dropout核心概念是“在加入了dropout的神经网络在训练时，每个隐藏单元必须学会随机选择其他单元样本。这理应使每个隐藏的单元更加健壮，并驱使它自己学到有用的特征，而不依赖于其他隐藏的单元来纠正它的错误“。在标准神经网络中，每个参数通过梯度下降逐渐优化达到全局最小值。因此，隐藏单元可能会为了修正其他单元的错误而更新参数。这可能导致“共适应”，反过来会导致过拟合。而假设通过使其他隐藏单元的存在不可靠，dropout阻止了每个隐藏单元的共适应。

dropout的示例代码如下：

```python
class Dropout():
    def __init__(self, prob=0.5):
        self.prob = prob
        self.params = []
    def forward(self,X):
        self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
        out = X * self.mask
    	return out.reshape(X.shape)
    def backward(self,dout):
        dX = dout * self.mask
        return dX,[]
```

在Dropout之后，Wan等人的进一步提出了DropConnect，它“通过随机丢弃权重而不是激活来扩展Dropout”。 “使用Drop Connect, drop的是每个连接，而不是每个输出单元。”与Dropout一样，该技术仅适用于全连接层。

上面提到的两种dropout方法被应用于前馈卷积神经网络。 RNN与仅前馈神经网络的不同之处在于先前的状态反馈到网络中，允许网络保留先前状态。 因此，将标准dropout应用于RNN会限制网络保留先前状态的能力，从而阻碍其性能。 Bayer等人指出了将dropout应用于递归神经网络（RNN）的问题。如果将完整的输出权重向量设置为零，则“在每次前向传递期间导致RNN动态变化是非常显著的。”

将Dropout应用于RNN的代码

```python
class RNN:
# ...
def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
rnn = RNN()
y = rnn.step(x) # x is an input ve
```

Dropout用于RNN

作为克服应用于RNN的dropout性能问题的一种方法，Zaremba和Pham等仅将dropout应用于非循环连接（Dropout未应用于隐藏状态）。 “通过不对循环连接使用dropout，LSTM可以从dropout正则化中受益，而不会牺牲其宝贵的记忆能力。”

变分Dropout

Gal和Ghahramani（2015）分析了将Dropout应用于仅前馈RNN的部分，发现这种方法仍然导致过拟合。 他们提出了“变分Dropout”，通过重复“输入，输出和循环层的每个时间步长相同的dropout掩码（在每个时间步骤丢弃相同的网络单元）”，使用贝叶斯解释，他们看到了语言的改进 建模和情感分析任务超过'纯dropout'。

循环Dropout

与Moon和Gal和Ghahramani一样，Semeniuta等人提出将dropout应用于RNN的循环连接，以便可以对循环权重进行正则化以提高性能。 Gal和Ghahramani使用网络的隐藏状态作为计算门值和小区更新以及使用dropout的子网络的输入来正则化子网络（图9b）。 Semeniuta等人的不同之处在于他们认为“整个架构以隐藏状态为关键部分并使整个网络正则化”。 这类似于Moon等人的概念（如图9a所示），但Semeniuta等人发现根据Moon等人直接丢弃先前的状态产生了混合结果，而将dropout应用于隐藏状态更新向量是一种更有原则的方法。

![img](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw2xANM9wMPz3dzossTh6hUpovbOo430jlBZ9bbyHVRUGEI8ibXozBYVOFqyicYFvjQDspCGx5HibIZcw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Zoneout

作为dropout的一个变种，Krueger等人提出了Zoneout，它“不是将某些单位的激活设置为0，Zoneout随机替换某些单位的激活与他们从前一个时间步的激活。”这“使网络更容易保留以前时间步的信息向前发展，促进而不是阻碍梯度信息向后流动“。

用TensorFlow实现Zoneout的核心思想

```
if self.is_training:
    new_state = (1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
        new_state_part - state_part, (1 - state_part_zoneout_prob),
         seed=self._seed) + state_part
else:
    new_state = state_part_zoneout_prob * state_part 
        + (1 - state_part_zoneout_prob) * new_state_part
```

AWD-LSTM

在关于语言建模的RNN正规化的开创性工作中，Merity等人提出了一种他们称为AWD-LSTM的方法。在这种方法中，Merity等人使用对隐藏权重矩阵使用DropConnect，而所有其他dropout操作使用变分dropout，以及包括随机长度在内的其他几种正则化策略反向传播到时间（BPTT），激活正则化（AR）和时间激活正则化（TAR）。

Merity等人使用的代码：

```
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
```

因此在RNNModel（nn.Module）中应用前向方法dropout的位置（注意self.lockdrop = LockedDropout（mask = mask））：

```
def forward(self, input, hidden, return_h=False):
    emb = embedded_dropout(self.encoder, input, dropout=self.dropoute 
    if self.training else 0)
    emb = self.lockdrop(emb, self.dropouti)
    raw_output = emb
    new_hidden = []
    raw_outputs = []
    outputs = []
    for l, rnn in enumerate(self.rnns):
        current_input = raw_output
        raw_output, new_h = rnn(raw_output, hidden[l])
        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        if l != self.nlayers - 1:
            raw_output = self.lockdrop(raw_output, self.dropouth)
            outputs.append(raw_output)
    hidden = new_hidden
    output = self.lockdrop(raw_output, self.dropout)
    outputs.append(output)
    result = output.view(output.size(0)*output.size(1), output.size(2))
    if return_h:
        return result, hidden, raw_outputs, outputs
    return result, hidden
```

DropConnect应用于上述相同RNNModel的**init**方法中：

```
if rnn_type == 'LSTM':
    self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid,
     nhid if l != nlayers - 1 
        else (ninp if tie_weights else nhid),
      1, dropout=0) for l in range(nlayers)]
    if wdrop:
        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                for rnn in self.rnns]
```

WeightDrop类的关键部分是以下方法：

```
def _setweights(self):
    for name_w in self.weights:
        raw_w = getattr(self.module, name_w + '_raw')
        w = None
        if self.variational:
            mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
            if raw_w.is_cuda: mask = mask.cuda()
            mask = torch.nn.functional.dropout(mask, p=self.dropout,
             training=True)
            w = mask.expand_as(raw_w) * raw_w
        else:
            w = torch.nn.functional.dropout(raw_w, p=self.dropout,
         training=self.training)
        setattr(self.module, name_w, w)
```

# 自然语言处理与词嵌入

## 词嵌入简介

### 词汇表征

1-hot representation：把每个词孤立起来，很难泛化两个单词之间的关系

featurized representation：

![5rep](img/5rep.png)

visualizing word embeddings:  t-SNE

### 使用词嵌入

会使用大量文本，迁移学习到命名实体识别

Transfer learning and word embeddings:

1. Learn word embeddings from large text corpus.(1-100B words)

   (Or download pre-trained embedding online.)

2. Transfer embedding to new task with smaller training set.(say, 100k words)

3. Optional: Continue to finetune the word embeddings with new data.

Relation to face encoding:

encoding和embedding类似

但是encoding可以对一个未知图片学习encoding, embedding是对已有的词汇学习出每个词的embedding

### 词嵌入的特性

Analogies

e(man) - e(woman) ≈ e(king) - e(queen)

Mikolov et. al., 2013, Linguistic regularities in continuous space word representations

For word w： argmax sim(e(w), e(king) - e(man)+e(woman))

Cosine similarity

![5cosine](img/5cosine.png)

### 嵌入矩阵

embedding matrix

每一列代表词的向量，行代表向量，维度是：向量维度*词数量

embedding matrix(E) × word(j)'s one-hot = word(j)'s embedding

In practice, use specialized function to look up an embedding.

## 学习词嵌入：Word2vec & Glove

### 学习词嵌入

Neural language model

![5nerual](img/5nerual.png)

![5context](img/5context.png)

可以看所有单词，也可以只看前面几个单词，固定的窗口

Bengio et. al., 2003, A neural probabilistic language model

### Word2vec

Skip-grams

在句子中随机选择一个词作为context word,再在这个词的前后比如5个间距的单词里随机选择一个作为Target word.

比如句子：I want a glass of orange juice to go along with my cereal.

context :    orange	orange	orange

Target   :    juice		glass	my

这样就形成三组词，context word是输入x, target word是输出y

首先根据词嵌入矩阵×单词的one-hot向量得到输入词c的词嵌入$e_c$, 经过softmax层，得到输出$\hat y$

$softmax:  p(t|c) = e^(\theta_t^T e_c) / \sum_{j=1}^{10000} e^(\theta_j^T e_c), 其中 \theta_t 是与输出t有关的参数$

$Loss(\hat y, y)= -\sum_{i=1}^{10000}y_ilog\hat y_i， y是一个one-hot 向量$

Problems with softmax classification

一、计算softmax的分母，计算速度慢

solutions:

1. Hierarchical softmax，多个二分类器组合成一棵树，比如第一次判断这个词是在词汇表的前5000个还是后5000个，第二次判断是在前5000个词的前2500个，还是后2500个，以此类推。

常用词放在前面，不常用的词放在底部，树最深的地方，复杂度大约是log|V|

2. 负采样

二、How to sample the context c？

如果使用均匀采样，常见词比如the of 等会常常被采样到，重要词很少被采样到。实际中会使用一些启发式的策略进行采样。

Mikolov rt. al., 2013, Efficient estimation of word representations in vector space.

### 负采样

抽取context word和target word形成正样本，标记为1；在词典中随机选择一个词和context word组成负样本，标记为0。k个负样本

输入x变成一对词(c,t)，输出y是对应的0或1

$P(y=1 | c, t) = \sigma (\theta_t^T e_c)$ 使用多个二分类器如logistic，每次只训练选择的正样本和负样本。

selecting negative examples

$p(w_i) = f(w_i)^{\frac34} / \sum_{j=1}^{10000} f(w_j)^{\frac34}，其中f(w_i)是单词w_i的词频$

Mikolov rt. al., 2013, Distributed representation of words and phrases and their compositionality

### Glove词向量

global vectors for word representation

(c, t)

$X_{ij}$ = times i appears in context of j

minimize$  \sum_{i=1}^{10000}\sum_{j=1}^{10000}f(X_{ij})(\theta_i^T e_j + b_i + b_j - logX_{ij})^2$

weighting term $f(X_{ij}) = 0 如果 x_{ij}=0，且定义0log0=0, 且对于停用词给一个不过分的权重，以及不常用的词一个不小的有意义的权重$

$\theta_i和e_j是对称的，所以可以有e_w^{(final)} = \frac{e_w+\theta_w}2$

注意word embeddings的每个轴不一定是有解释性的。

Pennington et. al., 2014. Golve: Global vectors for word representation

## 词嵌入的应用

### 情绪分类

经常没有足够的标记样本

比如评论对应打分(1-5分)

1. Simple sentiment classification model

把所有单词的词嵌入加起来或者平均，输入给softmax，输出1-5的分类

词嵌入的学习来自其它大量样本

问题：没有考虑词序，比如某个词good出现了很多次，最后得到的结果会是高分，但实际上是个差评：

Completely lacking in good taste, good service, and good ambience.

2. RNN for sentiment classification

many-to-one， 输入是单词的embedding, 输出是1-5的分类

### 词嵌入除偏

Debiasing word embeddings

![5debias](img/5debias.png)

![5debias2](img/5debias2.png)

not definitional比如doctor,babysitter等

Equalize pairs，使得babysitter和grandmother的距离, babysitter和grandfather的距离两者相近，没有显著区别

Bolukbasi rt. al. , 2016, Man is to computer as woman is to homemaker? Debiasing word embeddings

# 序列模型和注意力机制

## 多种sequence to sequence architectures

### 基础模型

![5seq2seq](img/5seq2seq.png)

![5image](img/5image.png)

### 选择最可能的句子

![5trans](img/5trans.png)

![5trans2](img/5trans2.png)

![5trans3](img/5trans3.png)

贪心搜索：先找到第一个最好的，然后找第二个单词最好的，以此类推

### 定向搜索

Beam search集束搜索

第一步：选择翻译的第一个单词，通过RNN，softmax输出概率，选择前B个单词保留下来，并且记录概率

第二步：将第一步得到的单词作为输入，分别训练，得到第二个单词的概率，选择前B个单词，并记录概率

以此类推， 直到出现句子结尾标志。

![5beam1](img/5beam1.png)

![5beam2](img/5beam2.png)

![5beam3](img/5beam3.png)

### 改进定向搜索

概率相乘，容易产生数值下溢underflow，所以取log

另外，由于概率小于1，取log后小于0，翻译会倾向于选择简短的句子，这样概率值会大一些

所以要除以句子长度

![5beam4](img/5beam4.png)

Beam search discussion

large B: better result, slower

small B: worse result, faster

Beam width B: B越大，性能提升越不明显。B的选择取决于应用。一般可以取10，100(蛮大了),工业中也可能取1000，3000等。

Unlike exact 精确search algorithms like BFS (Breadth First Search) or DFS(Depth First Search), Beam Search runs faster but is not guaranteed to find exact maximum for $argmax_y P(y|x)$

### 定向搜索的误差分析

![5error1](img/5error1.png)

![5error2](img/5error2.png)

![5error3](img/5error3.png)

### Bleu得分

![5bleu1](img/5bleu1.png)

![5bleu2](img/5bleu2.png)

![5bleu3](img/5bleu3.png)

![5bleu4](img/5bleu4.png)

### 注意力模型直观理解

神经网络不擅于处理长句子翻译

生成第一个词的时候，对第一个词、第二个词...应该放多少注意力。

![5attention1](img/5attention1.png)

![5attention2](img/5attention2.png)

### 注意力模型

![5attention3](img/5attention3.png)

![5attention4](img/5attention4.png)

![5attention5](img/5attention5.png)

## Speech recognition-Audio data

### 语音辨识

![5speech1](img/5speech1.png)

![5speech2](img/5speech2.png)

![5speech3](img/5speech3.png)

### 触发字检测

![5trigger1](img/5trigger1.png)

![5trigger2](img/5trigger2.png)

音频中发现trigger word的地方输出1，其它地方输出0.但是这样会导致0和1的比例很不均衡，可以采用一种粗暴的做法：将出现1的地方的后面若干个输出也变成1

## Conclusion

