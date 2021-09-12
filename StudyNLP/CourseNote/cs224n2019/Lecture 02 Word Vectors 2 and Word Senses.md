### 复习word2vec

一个单词的向量是一行；得到的概率分布不区分上下文的相对位置；每个词和and, of等词共同出现的概率都很高

为什么需要两个向量？——数学上更简单(中心词和上下文词分开考虑),最终是把2个向量平均。也可以每个词只用一个向量。

word2vec的两个模型：Skip-grams(SG), Continuous Bag of Words(CBOW), 还有negative sampling技巧，抽样分布技巧(unigram分布的3/4次方)



### optimization

梯度下降，随机梯度下降SGD，mini-batch(32或64,减少噪声，提高计算速度)，每次只更新出现的词的向量(特定行)



### 为什么不直接用共现计数矩阵？

随着词语的变多会变得很大；维度很高，需要大量空间存储；后续的分类问题会遇到稀疏问题。

解决方法：降维，只存储一些重要信息，固定维度。即做SVD。很少起作用，但在某些领域内被用的比较多，举例：Hacks to X(several used in Rohde et al. 2005)



### Glove

Count based vs. direct prediction

比较SVD这种count based模型与Word2Vec这种direct prediction模型，它们各有优缺点：

Count based模型优点是训练快速，并且有效的利用了统计信息，缺点是对于高频词汇较为偏向，并且仅能概括词组的相关性，而且有的时候产生的word vector对于解释词的含义如word analogy等任务效果不好；

Direct Prediction优点是可以概括比相关性更为复杂的信息，进行word analogy等任务时效果较好，缺点是对统计信息利用的不够充分。

Glove结合两个流派的想法，在神经网络中使用计数矩阵，共现概率的比值可以编码成meaning component

![](https://looperxx.github.io/imgs/1560266202421.png)

相较于单纯的co-occurrence probability，实际上co-occurrence probability的相对比值更有意义

log-bilinear模型：$w_{i} \cdot w_{j}=\log P(i | j)$

向量差异：$w_{x} \cdot\left(w_{a}-w_{b}\right)=\log \frac{P(x | a)}{P(x | b)}$

如果使向量点积等于共现概率的对数，那么向量差异变成了共现概率的比率

损失函数：

$J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}$

详细解释见Glove论文或参考资料



### 评估词向量的方法

内在—同义词、类比等，计算速度快，有助于理解这个系统，但是不清楚是否真的有用，除非与实际任务建立了相关性

外在—在真实任务中测试，eg命名实体识别；计算精确度可能需要很长时间；不清楚子系统是问题所在，是交互问题，还是其他子系统；如果用另一个子系统替换一个子系统可以提高精确度



### 词语多义性问题

1.聚类该词的所有上下文，得到不同的簇，将该词分解为不同的场景下的词。

2.直接加权平均各个场景下的向量，奇迹般地有很好的效果



### 参考资料

[CS224N笔记(二)：GloVe](https://zhuanlan.zhihu.com/p/60208480)

[https://looperxx.github.io/CS224n-2019-02-Word%20Vectors%202%20and%20Word%20Senses/](https://looperxx.github.io/CS224n-2019-02-Word Vectors 2 and Word Senses/)