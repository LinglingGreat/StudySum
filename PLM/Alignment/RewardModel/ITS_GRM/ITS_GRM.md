---
title: ITS_GRM
created: 2025-04-06
tags:
  - reward模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - DeepSeek
  - 清华
---

## 论文基本信息

标题：Inference-Time Scaling for Generalist Reward Modeling

作者：

链接：

代码：

框架图：


## 背景

自从DeepSeek-R1发布之后，数学、代码等可进行结果验证领域的RL训练路径似乎比较清晰了，就是采取ORM（outcome reward model）的方式。虽然这不代表PRM（Process Reward Model）方法就不生效了，但ORM的简单性使得结果可验证领域的研究多了许多，效果也提升了一大截。

但随之而来的另一个问题是，结果不可验证（或者说没有标准答案）的领域的RL又该何去何从？这些领域目前还是需要依靠高质量的数据标注（需要制定标准，进行人工标注或者大模型标注），且每个领域都需要单独收集数据，并不具有泛化能力。有什么办法能够解决数据获取、泛化能力这2个难点呢？这篇DeepSeek和清华合作的论文也许能给我们提供一些思路。

现有的研究主要集中在特定领域的奖励生成上，而在一般领域的奖励生成更具挑战性。

该问题的研究内容包括如何通过增加推理计算来提高通用奖励建模的质量和可扩展性，并提出了一种新的学习方法Self-Principled Critique Tuning（SPCT），以促进GRMs的推理时间可扩展性。

该问题的相关工作包括标量、半标量和生成式奖励建模方法。现有研究主要集中在如何通过增加训练计算来提高奖励质量，但很少关注推理时间的可扩展性。


## 相关研究




## 核心亮点

不同RM方法的比较：

![](img/Pasted%20image%2020250406151247.png)

生成式方法有scalar，semi-scalar, and generative
- scalar：输入query和response，输出标量值作为reward
- semi-scalar：除了输出标量值之外，还会输出文本描述（critique）
- generative：只生成 critiques 作为文本 reward，可以从中提取 reward 值。
打分方法有pointwise and pairwise
- pointwise：给每个response一个分数
- pairwise：从给定候选中选出最好的response



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
