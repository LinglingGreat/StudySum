# 基本概念

## ground truth

在机器学习中ground truth表示有监督学习的训练集的分类准确性，用于证明或者推翻某个假设。有监督的机器学习会对训练数据打标记，试想一下如果训练标记错误，那么将会对测试数据的预测产生影响，因此这里将那些**正确打标记的数据称为ground truth**。

## part of speech(PoS or POS)

**词类**（part of speech、PoS、POS）是一个[语言学](https://zh.wikipedia.org/wiki/%E8%AF%AD%E8%A8%80%E5%AD%A6)术语，是一种[语言](https://zh.wikipedia.org/wiki/%E8%AF%AD%E8%A8%80)中[词](https://zh.wikipedia.org/wiki/%E5%8D%95%E8%AF%8D)的[语法](https://zh.wikipedia.org/wiki/%E8%AF%AD%E6%B3%95)分类，是以语法特征（包括[句法](https://zh.wikipedia.org/wiki/%E5%8F%A5%E6%B3%95)功能和[形态变化](https://zh.wikipedia.org/w/index.php?title=%E5%BD%A2%E6%80%81%E5%8F%98%E5%8C%96&action=edit&redlink=1)）为主要依据、兼顾词汇意义对词进行划分的结果。

从[组合](https://zh.wikipedia.org/w/index.php?title=%E7%BB%84%E5%90%88_(%E8%AF%AD%E8%A8%80%E5%AD%A6)&action=edit&redlink=1)和[聚合](https://zh.wikipedia.org/w/index.php?title=%E8%81%9A%E5%90%88_(%E8%AF%AD%E8%A8%80%E5%AD%A6)&action=edit&redlink=1)关系来看，一个词类是指：在一个语言中，众多具有相同句法功能、能在同样的组合位置中出现的词，聚合在一起形成的[范畴](https://zh.wikipedia.org/wiki/%E8%8C%83%E7%95%B4)。词类是最普遍的语法的聚合.

举例来说，[汉语中](https://zh.wikipedia.org/w/index.php?title=%E6%B1%89%E8%AF%AD%E4%B8%AD&action=edit&redlink=1)有名词、[动词](https://zh.wikipedia.org/wiki/%E5%8A%A8%E8%AF%8D)、[形容词](https://zh.wikipedia.org/wiki/%E5%BD%A2%E5%AE%B9%E8%AF%8D)等词类

## 模块性(modularity)

假设节点表示的每一维对应着该节点从属于一个社区的强度。模块性是**衡量网络分离程度**的指标，高的模块性值意味着在同一模块内节点连接紧密，而不同模块间节点连接稀疏。

# 基于网络结构的网络表示学习

## 基于矩阵特征向量计算

### 谱聚类

spectral clustering

通过计算关系矩阵的前k 个特征向量或奇异向量来得到k 维的节点表示。关系矩阵一般就是网络的邻接矩阵或者Laplace矩阵。

强烈的依赖于关系矩阵的构建；时间复杂度较高；空间复杂度也较高

#### 局部线性表示(LLE)

locally linear embedding(LLE)，适用于无向图

假设节点的表示是从同一个流形中采样得到的. 局部线性表示假设一个节点和它邻居的表示都位于该流形的一个局部线性的区域. 也就是说, **一个节点的表示可以通过它的邻居节点的表示的线性组合来近似得到**. 局部线性表示使用邻居节点表示的加权和与中心节点表示的距离作为损失函数. 最小化损失函数的优化问题最终转化成某个关系矩阵特征向量计算问题求解.

Roweis S T, Saul L K. Nonlinear dimensionality reduction by locally linear embedding. Science, 2000, 290: 2323–2326

Saul L K, Roweis S T. An introduction to locally linear embedding. 2000. http://www.cs.toronto.edu/roweis/lle/
publications.html

#### Laplace特征表

Laplace eigenmap，适用于无向图、加权图

简单的假设两个相连的节点的表示应该相近. 特别地, 这里表示相近是由向量表示的欧氏距离的平方来定义. 该优化问题可以类似地转化为Laplace 矩阵的特征向量计算问题.

Belkin M, Niyogi P. Laplacian eigenmaps and spectral techniques for embedding and clustering. In: Proceedings of the 14th International Conference on Neural Information Processing Systems: Natural and Synthetic, Vancouver, 2001.
585–591

Tang L, Liu H. Leveraging social media networks for classification. Data Min Knowl Discov, 2011, 23: 447–478

#### 有向图表示(DGE)

directed graph embedding，适用于无向图、加权图、有向图

进一步扩展了Laplace 特征表方法, 给不同点的损失函数以不同的权重. 其中点的权重是由基于随机游走的排序方法来决定, 如PageRank.

Chen M, Yang Q, Tang X. Directed graph embedding. In: Proceedings of the 20th International Joint Conference on
Artifical Intelligence, Hyderabad, 2007. 2707–2712

## 基于简单神经网络的算法

### DeepWalk

Target:Nodes;      Input:Node sequences;    Output:Node embeddings

DeepWalk 首先在网络上采样生成大量的随机游走序列, 然后用Skip-gram 和Hierarchical Softmax 模型对随机游走序列中每个局部窗口内的节点对进行概率建模, 最大化随机游走序列的似然概率, 并最终使用随机梯度下降学习参数.

DeepWalk将随机游走生成的节点序列看作句子, 将序列中的节点看作文本中的词, 直接用训练词向量的Skip-Gram 模型来训练节点向量.

Perozzi B, Al-Rfou R, Skiena S. Deepwalk: online learning of social representations. In: Proceedings of the 20th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, 2014. 701–710

### Line算法

可以适用于大规模的有向带权图

为了对节点间的关系进行建模, LINE 算法用观察到的节点间连接刻画了第一级相似度关系, 用不直接相连的两个节点的共同邻居刻画了这两个点之间的第二级相似度关系.

LINE 算法对所有的第一级相似度和第二级相似度节点对进行了概率建模, 并最小化该概率分布和经验分布之间的KL 距离. 参数学习由随机梯度下降算法决定.

Tang J, Qu M, Wang M, et al. Line: large-scale information network embedding. In: Proceedings of the 24th
International Conference on World Wide Web, Florence, 2015. 1067–1077

## 基于矩阵分解的方法

给定关系矩阵, 对关系矩阵进行矩阵分解达到降维的效果, 从而得到节点的网络表示.

DeepWalk 算法实际上等价于某个特定关系矩阵的矩阵分解.

Yang C, Liu Z, Zhao D, et al. Network representation learning with rich text information. In: Proceedings of the 24th International Conference on Artificial Intelligence, Buenos Aires, 2015. 2111–2117

### GraRep算法

考虑了一种特别的关系矩阵. GraRep 通过SVD 分解对该关系矩阵进行降维从而得到k 步网络表示.

形式化地, 假设首先对邻接矩阵A 进行行归一化处理, 使得矩阵A 中每行的加和等于1. 则GraRep 算法在计算k 步网络表示时分解了矩阵Ak, 该关系矩阵中的每个单元对应着两个节点间通过k 步的随机游走抵达的概率. 更进一步, GraRep 尝试了不同的k 值, 并将不同k 值对应的k 步网络表示拼接起来, 组成维度更高、表达能力也更强的节点表示. 

但GraRep 的主要缺点在于其在计算关系矩阵Ak 的时候计算效率很低.

Cao S, Lu W, Xu Q. Grarep: learning graph representations with global structural information. In: Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, Melbourne, 2015. 891–900

## 基于深层神经网络的方法

### SDNE

使用深层神经网络对节点表示间的非线性进行建模. 整个模型可以被分为两个部分: 一个是由Laplace 矩阵监督的建模第一级相似度的模块, 另一个是由无监督的深层自编码器对第二级相似度关系进行建模. 最终SDNE 算法将深层自编码器的中间层作为节点的网络表示.

Wang D, Cui P, Zhu W. Structural deep network embedding. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, 2016. 1225–1234

## 基于社区发现的算法

让节点表示的每一维对应该节点从属于一个社区的强度, 然后设计最优化目标进行求解. 这类算法会学习得到上述的社区强度关系表示, 然后转化为社区发现的结果. 而学习社区强度关系表示的过程可以看作是无监督的非负节点表示学习的过程.

### BIGCLAM

作为一个可覆盖社区发现算法, 为每个网络中的节点学习了一个上述的k 维非负向量表示. 

BIGCLAM 算法对网络中每条边的生成概率进行建模: 两个点的向量表示内积越大, 那么这两个点之间形成边的概率也就越高. 算法的最大化目标是整个网络结构的最大似然概率. 最优化求解参数的过程由随机梯度下降算法实现.

Yang J, Leskovec J. Overlapping community detection at scale: a nonnegative matrix factorization approach. In: Proceedings of the 6th ACM International Conference on Web Search and Data Mining, Rome, 2013. 587–596

## 保存特殊性质的网络表示

### HOPE算法

为每个节点刻画了两种不同的表示,并着眼于保存原始网络中的非对称性信息.HOPE 构建了不同的非对称的关系矩阵, 然后使用JDGSVD 算法进行矩阵降维得到节点的网络表示.

Ou M, Cui P, Pei J, et al. Asymmetric transitivity preserving graph embedding. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, 2016. 1105–1114

### CNRL算法

考虑了在节点表示中嵌入网络隐藏的社区信息.

CNRL 假设每个节点属于多个社区, 也就是每个节点在所有的社区上有一个概率分布.

CNRL 将网络中的社区看作文本中的主题, 也就是说, 网络中相关的节点倾向于形成社区, 而文本中相关的词则会构成主题. 因此, CNRL 算法在生成的随机游走序列上, 将每个节点序列看成一篇文档, 通过基于Gibbs 采样的LDA来学习每个节点的社区分布, 并通过随机采样的方式, 来给序列中的节点分配其对应的社区标签. 随后, 在Skip-Gram 模型的基础上, 用中心节点的节点表示和对应的社区表示同时去预测随机游走序列中的邻近节点, 从而将社区结构信息保存在节点表示中.

CNRL 能够有效检测出不同规模的有重叠的社区, 以及有效的识别出社区边界.

Tu C, Wang H, Zeng X, et al. Community-enhanced network representation learning for network analysis. arXiv:1611.06645

Thomas L G, Mark S. Finding scientific topics. Proc National Acad Sci, 2004, 101: 5228–5235

# 结合外部信息的网络表示学习

## 结合文本信息的方法

### TADW

在矩阵分解框架下, 将节点的文本特征引入网络表示学习. TADW 算法基于矩阵分解形式的DeepWalk 算法进一步加强得到: 将关系矩阵M 分解成3 个小的矩阵乘积M=W^T * H * T, 其中矩阵T 是固定的文本特征向量, 另外两个矩阵是参数矩阵. TADW 算法使用共轭梯度下降法迭代更新W 矩阵和H 矩阵求解参数.

Yang C, Liu Z, Zhao D, et al. Network representation learning with rich text information. In: Proceedings of the 24th International Conference on Artificial Intelligence, Buenos Aires, 2015. 2111–2117

### CANE

利用网络节点的文本信息来对节点之间的关系进行解释, 来为网络节点
根据不同的邻居学习上下文相关的网络表示.

CANE 假设每个节点的表示向量由文本表示向量及结构表示向量构成, 其中, 文本表示向量的生成过程与边上的邻居相关, 所以生成的节点表示也是上下文相关的. CANE 利用卷积神经网络对一条边上两个节点的文本信息进行编码. 在文本表示生成的过程中, 利用相互注意力机制,选取两个节点彼此最相关的卷积结果构成最后的文本表示向量.

Tu C C, Liu H, Liu Z Y, et al. CANE: context-aware network embedding for relation modeling. In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, Vancouve, 2017. 1722–1731

## 半监督的网络表示学习

把已经标注的节点的节点类别或者标签利用起来, 加入到网络表示学习的过程中,从而针对性的提升节点网络表示在后续分类任务中的效果.

### MMDW

MMDW 同时学习矩阵分解形式的网络表示模型和最大间隔分类器. 为了
增大网络表示的区分性, MMDW 会针对分类边界上的支持向量计算其偏置向量, 使其在学习过程中向正确的类别方向进行偏置, 从而增大表示向量的区分能力.

受最大间距的分类器影响, 该模型学习得到的节点向量不仅包含网络结构的特征, 也会拥有分类标签带来的区分性.

Tu C C, Zhang W C, Liu Z Y, et al. Max-Margin DeepWalk: discriminative learning of network representation. In:Proceedings of International Joint Conference on Artificial Intelligence (IJCAI), New York, 2016

### DDRW

也采用了类似的方式, 同时训练DeepWalk 模型和最大间隔分类器, 来提高网络节点分类的效果.

Li J Z, Zhu J, Zhang B. Discriminative deep random walk for network classification. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, Berlin, 2016. 1004–1013

### Node2vec

通过改变随机游走序列生成的方式进一步扩展了DeepWalk 算法. DeepWalk 选取随机游走序列中下一个节点的方式是均匀随机分布的. 而node2vec 通过引入两个参数p 和q, 将宽度优先搜索和深度优先搜索引入了随机游走序列的生成过程. 宽度优先搜索注重邻近的节点并刻画了相对局部的一种网络表示, 宽度优先中的节点一般会出现很多次, 从而降低刻画中心节点的邻居节点的方差; 深度优先搜索反应了更高层面上的节点间的同质性.

node2vec 中的两个参数p 和q 控制随机游走序列的跳转概率,p 控制跳向上一个节点的邻居的概率, q 控制跳向上一个节点的非邻居的概率.

为了获得最优的超参数p 和q 的取值, node2vec 通过半监督形式, 利用网格搜索最合适的参数学习节点表示.

Grover A, Leskovec J. Node2vec: scalable feature learning for networks. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, 2016. 855–864

### GCN

设计了一种作用于网络结构上的卷积神经网络, 并使用一种基于边的标签传播规则实现半监督的网络表示学习.

Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks. In: Proceedings of the 5th International Conference on Learning Representations, Toulon, 2017

### Planetoid

联合地预测一个节点的邻居节点和类别标签, 类别标签同时取决于节点表示和已知节点标签, 从而进行半监督表示学习.

Yang Z, Cohen W, Salakhutdinov R. Revisiting semi-supervised learning with graph embeddings. In: Proceedings of the 33rd International Conference on International Conference on Machine Learning, New York, 2016

## word2vec

Target:Words;      Input:Sentences;    Output:Word embeddings

Word2Vec其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。 

Word2vec需要做到两个目标： 
（1）word embedding：将每一个单词映射为低维向量，可以自己设置。例如在Python的gensim包中封装的Word2Vec接口默认的词向量大小为100，一般取100-300之间。 
（2）让（1）中生成的词向量包含上下文信息，两种方法：

  法一：输入中间单词，输出其周围词（context）出现的可能性，称为Skip Gram 模型

  法二：与上相反，输入周围词，输出为中间次出现的可能性，称为Continuous Bag Of Words(CBOW)。

Skip Gram

模型的输出概率代表着到我们词典中每个词有多大可能性跟input word同时出现。 

gram的理解：A gram is a group of n （continuous ）words, where n is the gram window size. 例如：这个句子“The cat sat on the mat”的 3-gram representation 是 “The cat sat”, “cat sat on”, “sat on the”, “on the mat”. 

https://blog.csdn.net/u013527419/article/details/74129996

https://zhuanlan.zhihu.com/p/27234078

Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality.In: Proceedings of Advances in Neural Information Processing Systems, Lake Tahoe, 2013. 3111–3119

Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space. arXiv:1301.3781

Mikolov T, Karafi´at M, Burget L, et al. Recurrent neural network based language model. In: Proceedings of International Speech Communication Association, Makuhari, 2010. 1045–1048



