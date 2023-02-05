---
title: VLMO
created: 2023-02-05
tags: 多模态
type: 论文
papername: VLMo Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
---

## 论文基本信息

标题：VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts

链接： https://arxiv.org/abs/2111.02358

代码：

框架图：



## 核心亮点
为什么要做mixture of experts？
- dual-encoder的结构（比如CLIP，ALIGN）不适合复杂的多模态任务
- fusion encoder结构（比如ALBEF）做检索又非常慢

能不能融合两者？

VLMo提出分阶段训练，因为当时多模态数据没有那么多，单模态的数据集有很多。

![](img/Pasted%20image%2020230205162644.png)

损失函数跟ALBEF的一样 [ALBEF](../ALBEF/ALBEF.md)

![](img/Pasted%20image%2020230205165130.png)

完全拿一个在视觉上训练好的自注意力模型，用在文本上也能做的很好（图2中冻住了多头自注意力），但是反过来不行。



![](img/Pasted%20image%2020230205165531.png)

未来的工作（都做了）

![](img/Pasted%20image%2020230205165741.png)


## 主要收获

## 个人评价
