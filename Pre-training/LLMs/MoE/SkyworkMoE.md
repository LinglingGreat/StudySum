---
title: SkyworkMoE
created: 2024-07-05
tags:
  - 专家模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - skywork
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## MoE结构

Switch Transformer: 用transformer架构实现了MoE思想。

MoE 架构通过用混合专家替换部分或全部 FFN 来修改典型的transformer，其中每个专家本身就是一个小型 FFN，并且 MoE 层容纳多个此类专家。

MoE 层通过为每个输入令牌有选择地激活一些专家网络来增加transformer模型的容量，同时保持计算效率。专家的选择是通过门控机制执行的，允许模型动态地将代币路由给最相关的专家。

门控机制由一个 softmax 层组成，该层计算每个令牌的可用专家的概率分布。嵌入 xi 的第 i 个令牌的门输出 g 由下式给出

![](img/Pasted%20image%2020240629114906.png)

![](img/Pasted%20image%2020240629115016.png)


## MoE损失

### Auxiliary Loss

为了确保专家之间的负载平衡并防止单个专家占据主导地位，Switch Transformer 采用了辅助损失函数，鼓励在专家之间均匀分配token。

![](img/Pasted%20image%2020240704203551.png)

这是switch transformer训练中常用的实际辅助loss。通过最大限度地减少这种损失，模型可以有效地学习平衡专家之间的负载，防止任何单个专家过载或未充分利用。

total loss function $L_{total}$ for training the Switch Transformer is a combination of the cross entropy loss $L_{ce}$ for the next token prediction task and the auxiliary loss $L_{aux}$, weighted by a hyperparameter α


## Upcycling vs. From Scratch

在当前的情况下，要训练一个MoE模型有两条路线可以选择：
- upcycling：用一个dense模型做MoE模型的初始化，进行一定的继续预训练。这样的好处是MoE模型能在一个比较好的初始化点开始训练，直觉上这样的模型应该收敛得相对比较快，成本也比较低。存在的问题是dense模型的选择可能存在一些权衡取舍，且从dense进行初始化可能对最终效果存在负面影响。
- from scratch：直接随机初始化一个MoE模型，从零开始训练。这样成本相比upcycling就比较高，但是效果可能比upcycling更好。

当然还有一种方法是，先从零训一个dense模型，再从这个dense模型训练一个MoE模型。但是后面的实验告诉我们，如果这个dense模型纯粹是为最终的MoE模型服务的话，那这种方法是费力不讨好的。

要决定是upcycling还是from scratch，需要看现有的dense模型的水平，以及MoE模型的训练预算。首先如果预算根本支持不了MoE模型这个规模的训练，那我们当然只能选择upcycling。只有当预算充足，我们才有机会选择from scratch这条路。而如果没有可用的dense模型，那就只能选择from scratch。

前面我们从直觉上认为from scratch效果会更好，下面就从实验上来验证这个想法。

首先，在300B token的数据上训练一个0.3B的dense模型，并分别取100B和300B时的checkpoint作为后续实验的起始点。这两个checkpoint起个名字叫"checkpoint-100B"和"checkpoint-300B"。

然后在相同结构下，把dense模型扩成有8个专家的MoE模型，并使用3种不同的初始化策略：from-scratch / checkpoint-100B / checkpoint-300B。

假设我们现在有两种MoE模型的训练预算，100B和300B（token）。

对于100B训练预算，对比以下几个模型

![](img/Pasted%20image%2020240705144817.png)

同样地，对于300B预算的情况，训练了init_scratch-decay_300b和init_100b-decay_300b。另外还训练了一个init_300b-3xLR，相比init_300b-const提升了3倍的学习率，用于验证学习率的影响。

各个模型的训练结果如下图所示

![](img/Pasted%20image%2020240705141738.png)

左图：在100B的训练预算下，from scratch已经可以和从dense初始化的MoE模型loss持平，甚至比init_300b-const好。报告认为init_300b-const效果不好有一部分原因是学习率太小了。

中图：在300B的训练预算下，from scratch模型已经超越所有其他模型。另外学习率最小的模型表现最差。

右图：把中图几个模型的expert similarity画出来，发现expert similarity越低的模型，表现越好，并且对于upcycling的模型，expert similarity在训练过程中越来越低，对应着模型效果越来越好。而from scratch的模型的expert similarity基本上一直保持为0，这也说明从dense模型初始化会使得专家多样性比较弱，从而使得模型收敛到suboptimal的点。

据此，报告给出路线选择的经验法则。假设 是dense模型的训练成本， 是MoE模型的训练预算，那么：
- 如果$C_{moe}<< C_{dense}$，选择upcycling，upcycling能更好利用上dense模型已投入的成本。
- 如果$C_{moe}>= 2C_{dense}$ ，选择from scratch，能获得更好的效果。

另外，学习率的影响很大，这个要仔细设置。



## Training Techniques



## Skywork-MoE



## 参考资料


[社区供稿 | Mixtral-8x7B Pytorch 实现](https://mp.weixin.qq.com/s/HProBDSA9WxyD-JuKpJ9ew)

[全网最细致大模型MoE原理+代码手撕版](https://mp.weixin.qq.com/s/76a-7fDJumv6iB08L2BUKg)

skywork-MoE

