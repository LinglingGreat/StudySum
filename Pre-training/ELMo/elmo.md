之前我们一般比较常用的词嵌入的方法是诸如Word2Vec和GloVe这种，但这些词嵌入的训练方式一般都是上下文无关的，并且对于同一个词，不管它处于什么样的语境，它的词向量都是一样的，这样对于那些有歧义的词非常不友好。因此，论文就考虑到了要根据输入的句子作为上下文，来具体计算每个词的表征，提出了ELMo（Embeddings from Language Model）。它的基本思想，用大白话来说就是，还是用训练语言模型的套路，然后把语言模型中间隐含层的输出提取出来，作为这个词在当前上下文情境下的表征，简单但很有用！



ELMo在模型层上就是一个stacked bi-lstm（严格来说是训练了两个单向的stacked lstm）

ELMo的亮点当然不在于模型层，而是其通过实验间接说明了在多层的RNN中，不同层学到的特征其实是有差异的，因此ELMo提出在预训练完成并迁移到下游NLP任务中时，要为原始词向量层和每一层RNN的隐层都设置一个可训练参数，这些参数通过softmax层归一化后乘到其相应的层上并求和便起到了weighting的作用，然后对“加权和”得到的词向量再通过一个参数来进行词向量整体的scaling以更好的适应下游任务。

## 参考资料

[NLP的游戏规则从此改写？从word2vec, ELMo到BERT](https://zhuanlan.zhihu.com/p/47488095)

[ELMo解读（论文 + PyTorch源码）](https://blog.csdn.net/Magical_Bubble/article/details/89160032)

