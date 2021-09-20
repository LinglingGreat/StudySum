# Attention

attention有很多种形态，大家通常会说5种：hard attention，soft attention，Global Attention，local attention，self attention。


**hard attention** ：
Hard Attention不会选择整个encoder的隐层输出做为其输入，它在t时刻会依多元伯努利分布来采样输入端的一部分隐状态来进行计算，而不是整个encoder的隐状态。在 hard attention 里面，每个时刻 t 模型的序列 [ St1,…,StL ] 只有一个取 1，其余全部为 0，是 one-hot 形式。也就是说每次只focus 一个位置。


**soft attention** ：
对于hard attention，soft attention 每次会照顾到全部的位置，只是不同位置的权重不同罢了。


**global attention** ：
global attention 顾名思义，forcus 全部的 position。它 和 local attention 的主要区别在于 attention 所 forcus 的 source positions 数目的不同。


**Local Attention** :
Local Attention只 focus 一部分 position，则为 local attention
global attention 的缺点在于每次都要扫描全部的 source hidden state，计算开销较大，为了提升效率，提出 local attention，每次只 focus 一小部分的 source position。


**self attention** ：
因为transformer现在已经在nlp领域大放异彩，所以，我这里打算详细的说一下transformer的核心组件self-attention，虽然网上关于self-attention的描述已经很多了。
说起self-attention的提出，是因为rnn存在非法并行计算的问题，而cnn存在无法捕获长距离特征的问题，因为既要又要的需求，当看到attention的巨大优势，《Attention is all you need》的作者决定用attention机制代替rnn搭建整个模型，于是Multi-headed attention，横空出世。


Self-attention的意思就是自己和自己做 attention，即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。


而Multi-head Attention其实就是多个Self-Attention结构的结合，每个head学习到在不同表示空间中的特征。


multi-head attention 由多个 scaled dot-product attention 这样的基础单元经过 stack 而成。那什么是scaled dot-product attention，字面意思：放缩点积注意力。使用点积进行相似度计算的attention，只是多除了一个（为K的维度）起到调节作用，使得内积不至于太大。scaled dot-product attention跟使用点积进行相似度计算的attention差不多，只是多除了一个（为K的维度）起到调节作用，使得内积不至于太大，有利于训练的收敛。
Query，Key，Value 做embedding，获得相应的word embedding，然后输入到放缩点积attention，注意这里要做h次，其实也就是所谓的多头，每一次算一个头。而且每次Q，K，V进行线性变换的参数W是不一样的。然后将h次的放缩点积attention结果进行拼接，再进行一次线性变换得到的值作为多头attention的结果。可以看到，google提出来的多头attention的不同之处在于进行了h次计算而不仅仅算一次，论文中说到这样的好处是可以允许模型在不同的表示子空间里学习到相关的信息。论文里有attention可视化的图，可以去看看。


self-attention的特点在于无视词之间的距离直接计算依赖关系，能够学习一个句子的内部结构，实现也较为简单并行可以并行计算。


如果你还想了解更多的attention，请查看：[https://zhuanlan.zhihu.com/p/73](https://zhuanlan.zhihu.com/p/73357761)


## 参考资料

[https://zhuanlan.zhihu.com/p/73357761](https://zhuanlan.zhihu.com/p/73357761)

