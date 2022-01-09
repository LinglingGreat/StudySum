论文地址：[paper](https://arxiv.org/abs/1909.00204)

代码地址：[code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA)

NEZHA论文是也是基于[Transformer](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)的预训练模型，从文章来看，它对[BERT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.04805)模型进行了四点改进，具体如下：

1.  增加相对位置编码函数（Functional Relative Positional Encoding）
2.  全词掩码（Whole Word Masking）
3.  混合精度训练（Mixed Precision Training）
4.  优化器改进（LAMB Optimizer）

**在BERT模型预训练时，很多数据的真实数据长度达不到最大长度，因此靠后位置的位置向量训练的次数要比靠前位置的位置向量的次数少，造成靠后的参数位置编码学习的不够。在计算当前位置的向量的时候，应该考虑与它相互依赖的token之间相对位置关系，可以更好地学习到信息之间的交互传递。**


## 参考资料

[NEZHA（哪吒）论文阅读笔记](https://zhuanlan.zhihu.com/p/100044919)