论文：Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation

代码：https://github.com/jungokasai/deep-shallow

关键词： #机器翻译 ， #自回归 ， #浅层decoder 



传统的机器翻译方法：autoregressively，一个一个预测的（需要基于之前的词），不能并行。

新的non-autoregressive的机器翻译的并行化decoder的方法(NAR)：质量不好，因为假设了输出token之间的条件独立性。

近来大量工作致力于非自回归翻译技术，意在达到翻译质量和推理速度的平衡。

在本文工作中，我们重新探究了这种平衡，并认为transformer-based的自回归模型可以在不损失翻译质量的情况下加速推理。具体地，研究了不同深度encoder和decoder的自回归模型。**大量实验表明，给定一个足够深度的encoder,一层的decoder可以产生最先进的精度，并且延迟可与强大的非自回归模型相媲美。**

但对非回归模型而言，更深的解码器则是更好效果的前提。



从3个方面评估：speed measurement，layer allocation, knowledge distillation



## 参考资料

[[论文阅读]Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation](https://blog.csdn.net/ZY_miao/article/details/110424367)

[香侬读 | 更深的编码器+更浅的解码器=更快的自回归模型](https://zhuanlan.zhihu.com/p/150065091)













