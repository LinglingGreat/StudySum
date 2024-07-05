---
title: MoE
created: 2024-07-04
tags:
  - 专家模型
---



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




## Training Techniques



## Skywork-MoE



## 参考资料


[社区供稿 | Mixtral-8x7B Pytorch 实现](https://mp.weixin.qq.com/s/HProBDSA9WxyD-JuKpJ9ew)

[全网最细致大模型MoE原理+代码手撕版](https://mp.weixin.qq.com/s/76a-7fDJumv6iB08L2BUKg)

skywork-MoE

