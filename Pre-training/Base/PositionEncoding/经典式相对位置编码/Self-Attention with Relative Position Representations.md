---
title: Self-Attention with Relative Position Representations
created: 2023-02-11
tags: selfattention 位置编码
type: 论文
papername: Self-Attention with Relative Position Representations
conference: NAACL
year: 2018
institution: 谷歌
---

## 论文基本信息

标题：Self-Attention with Relative Position Representations

作者：

链接：

代码：

框架图：

本文是最早提出Relation-aware Self-attention的工作。本文探索了一种将Self-attention中加入relation模块来编码relative position信息方法，使用该结构后，transformer将可以不使用额外的positional encoding模块也能获取位置信息。

实验表明，单独使用positional encoding模块和单独使用relation-aware Self-attention模块都能大幅度提升机器翻译任务上的表现，而且使用relation-aware Self-attention模块比使用positional encoding模块的效果要好。不过，作者也做实验发现，将relation-aware Self-attention模块和positional encoding模块结合使用，并不会带来进一步的提升。

最后，作者也说到使用relation-aware Self-attention编码相对位置信息可以被看作是一个特殊的例子，实际上，我们可以考虑输入的任意两个元素之间的任意关系，这也是后续Relation-aware Self-attention被广泛应用的原因。

将输入建模为标记的(labeled)，有向的( directed)，完全连接的图( fully-connected graph)。



Relation-aware Self-attention实际上就是在计算任意两个token i和j的Self-attention时，考虑他们的先验关系rij​。

![](img/Pasted%20image%2020230211102206.png)

在定义了Relation-aware Self-attention后，作者将它用来编码token i和j之间的相对位置信息来应用到机器翻译中。这里，作者假设超过一定距离后的相对位置信息将不那么重要（比如，第一个token和第500个token之间的相对位置信息肯定没有第一个token和第二token之间的相对位置信息重要），所以这里作者设置了一个阈值k，即只考虑相对位置偏差在k范围内的位置信息。（这里的aij就是rij）

![](img/Pasted%20image%2020230211102301.png)

与当前token i距离超过k的token j之间的相对位置信息rij​将被截断，不会考虑。

![](img/Pasted%20image%2020230211102537.png)

这样将一共会有2k+1中相对位置编码关系。

为了尽量不增加计算的复杂度和时间，作者使用了多种方式来进行高效的代码实现。

首先，作者在实现时，将token i和token j之间的relation信息rij在H个头之间共享。 其次，在相应的计算eij​和zi​时，可以展开以方便并行计算。

![](img/Pasted%20image%2020230211102635.png)

### 实验

作者在机器翻译任务进行实验，baseline就是使用额外的positional encoding的transformer模型。结果如下：

![](img/Pasted%20image%2020230211102927.png)

可以看到，将相对位置信息编码到Self-attention后，效果比直接在输入层添加positional encoding要好。

接着，作者对比了截取阈值k对于最终结果的影响。

![](img/Pasted%20image%2020230211102952.png)

可以看到，K=0时，由于完全没有相对位置编码信息，效果很差。在K>2之后，提升K对实现结果影响也不大了。这表明在当前token附近的几个token之间的相对位置信息对于翻译任务很重要。

最后，作者还对比了rK​和rV​带来的提升影响。

![](img/Pasted%20image%2020230211103018.png)



## 核心亮点

## 主要收获


## 参考资料

https://juejin.cn/post/7105633443072983053

https://zhuanlan.zhihu.com/p/374091445

