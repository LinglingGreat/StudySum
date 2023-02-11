---
title: Flamingo
created: 2022-11-08
tags: 对话 多模态 fewshot
type: 论文
papername: Flamingo-a Visual Language Model for Few-Shot Learning
conference: NeurIPS
year: 2022
institution: DeepMind
---

## 论文基本信息 

题目：Flamingo: a Visual Language Model for Few-Shot Learning

网站： https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model

模型结构图：

![](img/Pasted%20image%2020221108225406.png)


要把图像和文本合并到一起训练，**第一个挑战就是怎样把2D的图像压缩到1D，好跟文本一起输入Transformer**

作者首先用了一个预训练的ResNet对图像进行编码，之后为了节省后续Transformer的计算量，把图像编码压缩到R个token中，毕竟图像本来的信息密度就小一些，压一压也损失不了多少。

压缩的方法就是引入一个Perceiver Resampler模块，核心是做attention，如下图所示，用R个可学习的token作为Query，图像编码Xf和R个token的拼接作为K和V（拼接后效果略好），这样attention下来输出也是R个token。

![](img/Pasted%20image%2020221108225146.png)

把2D图像压缩到1D后，**第二个挑战是如何让图像和文本做交互**。

作者为了提升效率，直接从一个预训练好的70B语言模型Chinchilla[1]开始加入图像信息，考虑到保留之前LM学到的知识，训练时需要把LM的参数冻住，因此提出了**Gated Cross attention**机制来融合图像和文本。

具体来说，还是利用attention机制，如下图所示，把文本编码当作Q，图像编码当作K和V，这样输出的维度还是跟之前文本的一样。之后还是正常过FFW。同时为了提升稳定性和效果，在attention和FFW之后都增加了tanh门控，先开始是0，之后逐步增加。

![](img/Pasted%20image%2020221108225256.png)

但为了减少参数量，这个机制不一定每层都有，具体增加的策略如下：

![](img/Pasted%20image%2020221108225315.png)

同时为了减少不相关的图像噪声，也增加了一些mask策略。在Cross-attention时，每段文字只能关注到它之前的一个图像，如下图：

![](img/Pasted%20image%2020221108225340.png)

不过在最终预测的时候，还是保持LM的causal attention，也就是Cross-attention之后的正常attention可以保证在预测输出时会关注到以前所有的文字和图像。

有了上述两个机制后，图像和文本就成功融合到一起了，模型总结构如下（**注意每个图像都是单独编码的**）：

![](img/Pasted%20image%2020221108225406.png)

模型部分解决了，接下来**第三个挑战是训练数据构造**。

之前很多数据集要不是某个任务的，要不是处理好的文本和图像pair，而作者的目的是做一个泛化性更强的模型，因此不加任何的下游任务数据，直接从网页自己挖一些带有文本和图像的数据数据，构成M3W数据集。同时也用了别人的文本和图像pair的数据，如下所示：

![](img/Pasted%20image%2020221108225428.png)

最终效果还是很惊艳的，Flamingo只用4-shot就可以碾压其他的Zero/Few-shot SOTA，并且32-shot在6个任务上超越了精调SOTA

另外，作者提供了很详细的消融实验，可以给后续多模态模型参考，可以得到的一些关键结论是：

1.  数据集真的很重要，其中M3W有作者们从网页收集的185M图像和182G的文本，LTIP包含312M的图像和文本，VTP包含27M的短视频和文本，M3W是其中占比很大的一个数据集
    
2.  Cross-attention也起到了很大的作用，一方面是门控机制，另一方面是添加的频率（每层都加比偶尔加好）



## 核心亮点

它的创新点主要在模型上面：

1.  设计了一个很优雅地把图片从3D压缩到2D的机制
    
2.  让图片特征和文本特征做交叉注意力
    

在预训练阶段，它直接从互联网挖掘大量语料，并让图片和其之后跟随的文本做交互，是个很方便的自监督任务。

## 主要收获



