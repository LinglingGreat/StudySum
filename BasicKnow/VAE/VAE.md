## 自动编码器(AutoEncoder)

自动编码器(AutoEncoder)最开始作为一种数据的压缩方法，其特点有:

1)跟数据相关程度很高，这意味着自动编码器只能压缩与训练数据相似的数据

2)压缩后数据是有损的，这是因为在降维的过程中不可避免的要丢失掉信息；

自动编码器主要应用有：数据去噪，进行可视化降维，生成数据。

输入的数据经过神经网络降维到一个编码(code)，接着又通过另外一个神经网络去解码得到一个与输入原数据一模一样的生成数据，然后通过去比较这两个数据，最小化他们之间的差异来训练这个网络中编码器和解码器的参数。当这个过程训练完之后，我们可以拿出这个解码器，随机传入一个编码(code)，希望通过解码器能够生成一个和原数据差不多的数据

## 变分自动编码器(Variational Autoencoder)

在AE中，我们其实并不能任意生成图片，因为我们没有办法自己去构造隐藏向量，我们需要通过一张图片输入编码我们才知道得到的隐含向量是什么，可以通过变分自动编码器来解决这个问题。

只需要在编码过程给它增加一些限制，迫使其生成的隐含向量能够粗略的遵循一个标准正态分布，这就是其与一般的自动编码器最大的不同。

在实际情况中，我们需要在模型的准确率上与隐含向量服从标准正态分布之间做一个权衡，所谓模型的准确率就是指解码器生成的图片与原图片的相似程度。我们可以让网络自己来做这个决定，非常简单，我们只需要将这两者都做一个loss，然后在将他们求和作为总的loss，这样网络就能够自己选择如何才能够使得这个总的loss下降。

两个损失函数：MSE损失和KL散度损失

变分编码器使用了一个技巧“重新参数化”来解决KL divergence的计算问题。

不再是每次产生一个隐含向量，而是生成两个向量，一个表示均值，一个表示标准差，然后通过这两个统计量来合成隐含向量

## 参考资料

[花式解释AutoEncoder与VAE](https://zhuanlan.zhihu.com/p/27549418)
