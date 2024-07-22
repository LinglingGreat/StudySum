---
title: LORA
created: 2023-04-11
tags: 微调
type: 论文
papername: LoRA Low-Rank Adaptation of Large Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2021
institution: Microsoft
---

## 论文基本信息

标题：LoRA: Low-Rank Adaptation of Large Language Models

作者：

链接： [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

代码： https://github.com/microsoft/LoRA 

框架图：

![](img/Pasted%20image%2020230411150734.png)

### 具体做法

-   在原模型旁边增加一个旁路，**通过低秩分解（先降维再升维）来模拟参数的更新量**；
-   训练时，原模型固定，只训练降维矩阵A和升维矩阵B；
-   推理时，可将BA加到原参数上，不引入额外的推理延迟；
-   初始化，A采用高斯分布初始化，B初始化为全0，保证训练开始时旁路为0矩阵；
-   可插拔式的切换任务，当前任务W0+B1A1，将lora部分减掉，换成B2A2，即可实现任务切换；


## 背景
自然语言处理目前存在一个重要范式：一般领域数据的大规模预训练，对特定任务或领域的适应（finetune）。

但是随着预训练语言模型越来越大，这个范式存在以下问题：

-   当我们finetune大模型时，由于训练成本太高，不太可能重新训练所有模型参数
-   以前的方法（论文发表于2021年）都或多或少有其它性能问题，如adapter增加了模型层数，引入了额外的推理延迟；prefix-tuning比较难训练，效果不如直接finetune。

基于上述背景，论文作者得益于前人的一些关于内在维度（intrinsic dimension）的发现：模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配。在任务适配过程中，即使随机投影到较小的子空间，仍然可以有效地学习。假设模型在任务适配过程中权重的改变量是低秩（low rank）的，由此提出低秩自适应（LoRA）方法，LoRA允许我们通过优化适应过程中密集层变化的秩分解矩阵来间接训练神经网络中的一些密集层，同时保持预先训练的权重不变。

Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning


## 贡献


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？


## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向


## 核心亮点


## 主要收获


