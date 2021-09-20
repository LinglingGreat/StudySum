### 重点笔记

#### 单词的表示

**WordNet**, 一个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms** 的列表的辞典

同义词：

```
from nltk.corpus import wordnet as wn
poses = { 'n':'noun', 'v':'verb', 's':'adj (s)', 'a':'adj', 'r':'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()],
                    ", ".join([l.name() for l in synset.lemmas()])))
```

![](https://looperxx.github.io/imgs/1560068762906.png)

上位词：

```
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
```

![](https://looperxx.github.io/imgs/1560068729196.png)

存在的问题：

- 作为一个资源很好，但忽略了细微差别，比如"proficient"只在某些上下文和"good"是同义词
- 难以持续更新单词的新含义
- 主观的
- 需要人类劳动来创造和调整
- 无法计算单词相似度



在传统的自然语言处理中，我们把词语看作离散的符号，单词通过one-hot向量表示。所有向量是正交的，没有相似性概念，向量维度过大。

在Distributional semantics中，一个单词的意思是由经常出现在该单词附近的词(上下文)给出的，单词通过一个向量表示，称为word embeddings或者word representations，它们是分布式表示(distributed representation)



#### Word2vec

中心思想：给定大量文本数据，训练每个单词的向量，使得给定中心词c时，上下文词o的概率最大，而这个概率的衡量方式是c和o两个词向量的相似性。

c和o相似性的计算方法是：

$P(o|c)=\frac{exp(u^T_ov_c)}{\sum_{w\in V}exp(u^T_wv_c)}$

每个词有两个向量：作为中心词的向量u和作为上下文词的v

这里用了$softmax(x_i)$函数，放大了最大的概率(max)，仍然为较小的xi赋予了一定概率(soft)

**两个算法**：

CBOW——根据中心词周围的上下文单词来预测该词的词向量

skip-gram——根据中心词预测周围上下文词的概率分布

**两个训练方法**：

negative sampling——通过抽取负样本来定义目标

hierarchical softmax——通过使用一个有效的树结构来计算所有词的概率来定义目标

##### CBOW

对于模型中的每个单词，需要学习两个向量：

- v (输入向量) 当词在上下文中
- u (输出向量) 当词是中心词

对应地有两个矩阵：

$V\in R^{n\times |V|}$是输入词矩阵，第i列是词$w_i$的n维嵌入向量，定义为$v_i$

$U\in R^{|V|\times n}$是输出词矩阵，第i行是词$w_i$的n维嵌入向量，定义为$u_i$

模型步骤：

- 对于大小为m的输入上下文，每个词生成一个one-hot词向量$x^i, i=c-m, c-m+1,...,c-1,c+1,...,c+m$
- 从上下文矩阵$V$中得到每个词的嵌入词向量$v_i=Vx^i$
- 对上述向量求平均值得到$\hat v$
- 生成一个分数向量$z=U\hat v \in R^{|V|}$，相似向量的点积越高，就会令相似的词更为靠近，从而获得更高的分数。将分数转换为概率$\hat y=softmax(z)$
- 希望生成的概率$\hat y$和实际概率$y$相匹配，使得实际的词刚好就是这个one-hot向量

损失函数：

$H(\hat y, y)=-\sum_{j=1}^{|V|}y_j log(\hat y_j)$

上述公式中的$y_j$是one-hot向量，因此损失函数可以简化为

$H(\hat y, y)=-y_c log(\hat y_c)$

c是正确词的one-hot向量的索引。

优化目标函数的公式为：

$$
\begin{align}
\text {minimize} J &=-\log P\left(w_{c} | w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m}\right) \\
&=-\log P\left(u_{c} | \hat{v}\right) \\
&=-\log \frac{\exp \left(u_{c}^{T} \hat{v}\right)}{\sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)} \\
&=-u_{c}^{T} \hat{v}+\log \sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)
\end{align}
$$


##### Skip-Gram

和CBOW类似，只是交换了x和y

- 生成中心词的one-hot向量x
- 得到中心词的词嵌入向量$v_c=Vx$
- 生成分数向量$z=Uv_c$
- 将分数向量转化为概率$\hat y=softmax(z)$，$\hat y_{c-m},...,\hat y_{c-1}, \hat y_{c+1},...,\hat y_{c+m}$是每个上下文词观察到的概率
- 希望生成的概率向量匹配真实概率$y_{c-m},...,y_{c-1},y_{c+1},...,y_{c+m}$，one-hot向量是实际的输出。

与CBOW模型不同的是，引用了朴素贝叶斯假设来拆分概率，即给定中心词，所有输出的词是完全独立的。
$$
\begin{aligned} 
\text { minimize } J&=-\log P\left(w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m} | w_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(w_{c-m+j} | w_{c}\right) \\ 
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(u_{c-m+j} | v_{c}\right) \\ 
&=-\log \prod_{j=0, j \neq m}^{2 m} \frac{\exp \left(u_{c-m+j}^{T} v_{c}\right)}{\sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right)} \\ 
&=-\sum_{j=0, j \neq m}^{2 m} u_{c-m+j}^{T} v_{c}+2 m \log \sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right) 
\end{aligned}
$$

##### Negative Sampling

论文：《Distributed Representations of Words and Phrases and their Compositionality.》

目标函数中对|V|的求和计算量是非常大的，任何的更新或者对目标函数的评估都要花费O(|V|)的时间复杂度。一个简单的想法是不去直接计算，而是求近似值。

在每一个训练步骤中，不去遍历整个词汇表，而仅仅抽取一些负样例，对噪声分布$P_n(w)$抽样，这个概率是和词频的排序相匹配的。

考虑一对中心词和上下文词$(w,c)$，通过$P(D=1|w,c)$表示该词对来自语料库，$P(D=0|w,c)$表示不是来自语料库。对$P(D=1|w,c)$用sigmoid函数建模。

建立一个新的目标函数，如果词对确实在语料库中，就最大化概率$P(D=1|w,c)$，否则最大化概率$P(D=0|w,c)$，对这两个概率采用简单的极大似然估计方法
$$
\begin{aligned} \theta &=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 | w, c, \theta) \prod_{(w, c) \in \widetilde{D}} P(D=0 | w, c, \theta) \\ 
&=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 | w, c, \theta) \prod_{(w, c) \in \widetilde{D}}(1-P(D=1 | w, c, \theta)) \\ 
&=\underset{\theta}{\operatorname{argmax}} \sum_{(w, c) \in D} \log P(D=1 | w, c, \theta)+\sum_{(w, c) \in \widetilde{D}} \log (1-P(D=1 | w, c, \theta)) \\ 
&=\arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \widetilde{D}} \log \left(1-\frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}\right) \\ 
&=\arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \widetilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right) 
\end{aligned}
$$
最大化似然函数等价于最小化负对数似然。

抽样分布从实际效果来看最好的是指数为3/4的Unigram模型。举例：
$$
\begin{eqnarray}  is: 0.9^{3/4} = 0.92 \nonumber \\ Constitution: 0.09^{3/4}= 0.16 \nonumber \\ bombastic:0.01^{3/4}= 0.032 \nonumber \end{eqnarray}
$$
“Bombastic”现在被抽样的概率是之前的三倍，而“is”只比之前的才提高了一点点。

##### Hierarchical Softmax

论文：《Distributed Representations of Words and Phrases and their Compositionality.》

**在实际中，hierarchical softmax 对低频词往往表现得更好，负采样对高频词和较低维度向量表现得更好**。

Hierarchical softmax 使用一个二叉树来表示词表中的所有词。树中的每个叶结点都是一个单词，而且只有一条路径从根结点到叶结点。在这个模型中，没有词的输出表示。相反，图的每个节点（根节点和叶结点除外）与模型要学习的向量相关联。单词作为输出单词的概率定义为从根随机游走到单词所对应的叶的概率。计算成本变为 O(log(|V|))而不是 O(|V|) 。





#### 基于SVD的词嵌入方法

对共现矩阵X应用SVD分解方法得到$X=USV^T$，选择U前k行得到k维的词向量。

方法存在的问题：维度经常发生改变，矩阵稀疏，矩阵维度高，计算复杂度高等



#### 参考资料

[https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/](https://looperxx.github.io/CS224n-2019-01-Introduction and Word Vectors/)





