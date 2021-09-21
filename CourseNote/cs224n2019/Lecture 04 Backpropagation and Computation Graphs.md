梯度计算分解，一些tips



使用预训练的词向量的问题：

- 预训练的单词向量TV，telly, television是相似的
- 训练数据中只有TV和telly，更新向量时这两个单词的向量会发生移动，而television保持原样
- 但即便如此，在数据量不够时还是应该使用预训练向量，因为他们接受了大量的数据训练
- 如果只有一个小的训练数据集，不要fine tune词向量
- 如果有大型数据集，那么train=update=fine-tune词向量到任务可能会更好



计算图表示前向传播和反向传播，用上游的梯度和链式法则来得到下游的梯度



正则，矢量化，非线性，初始化，优化器，学习率



### 参考资料

[https://looperxx.github.io/CS224n-2019-04-Backpropagation%20and%20Computation%20Graphs/](https://looperxx.github.io/CS224n-2019-04-Backpropagation and Computation Graphs/)