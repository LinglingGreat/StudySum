---
title: S2A
created: 2023-11-26
tags:
  - attention
type: 论文
papername: System 2 Attention (is something you might need too)
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution: Meta
---

## 论文基本信息

标题：System 2 Attention (is something you might need too)

作者：

链接： https://arxiv.org/abs/2311.11829

代码：

框架图：


## 背景

Problem: LLM 可能会因不相关的上下文或者输入提示中固有的偏好或意见做出错误的判断。后一种情况表现出的问题被叫做「阿谀奉承」，即模型与输入保持一致。

![](img/Pasted%20image%2020231126153355.png)

Idea: 软注意力既倾向于将概率分配给大部分上下文（包括不相关的部分），也倾向于过度关注重复的 token。因此，研究者提出了一种完全不同的注意力机制方法，即通过将 LLM 用作一个自然语言推理器来执行注意力。具体来讲，他们利用 LLM 遵循指令的能力，提示它们生成应该注意的上下文，从而使它们只包含不会扭曲自身推理的相关资料。研究者将这一过程称为 System 2 Attention（S2A），他们将底层 transformer 及其注意力机制视为类似于人类 System 1 推理的自动操作。

当人们需要特意关注一项任务并且 System 1 可能出错时，System 2 就会分配费力的脑力活动，并接管人类的工作。因此，这一子系统与研究者提出的 S2A 具有类似目标，后者希望通过额外的推理引擎工作来减轻上述 transformer 软注意力的失败。

## 相关研究

添加更多监督训练数据或通过强化学习策略来解决


## 核心亮点

S2A 包含两个过程：

- 给定上下文 x，S2A 首先重新生成上下文 x '，从而删除会对输出产生不利影响的上下文的不相关部分。本文将其表示为 x ′ ∼ S2A (x)。
    
- 给定 x ′ ，然后使用重新生成的上下文而不是原始上下文生成 LLM 的最终响应：y ∼ LLM (x ′ )。

![](img/Pasted%20image%2020231126153521.png)

本文考虑了 S2A 方法的几种变体。

无上下文和问题分离。在图 2 的实现中，本文选择重新生成分解为两部分（上下文和问题）的上下文。图 12 给出了该提示变体。

![](img/Pasted%20image%2020231126153751.png)

保留原始上下文在 S2A 中，在重新生成上下文之后，应该包含所有应该注意的必要元素，然后模型仅在重新生成的上下文上进行响应，原始上下文被丢弃。图 14 给出了该提示变体。

![](img/Pasted%20image%2020231126153826.png)

指令式提示。图 2 中给出的 S2A 提示鼓励从上下文中删除固执己见的文本，并使用步骤 2（图 13）中的说明要求响应不固执己见。

![](img/Pasted%20image%2020231126153910.png)

强调相关性与不相关性。以上 S2A 的实现都强调重新生成上下文以提高客观性并减少阿谀奉承。然而，本文认为还有其他需要强调的点， 例如，人们可以强调相关性与不相关性。图 15 中的提示变体给出了这种方法的一个实例：

![](img/Pasted%20image%2020231126153932.png)


## 实验

他们证实与基于标准注意力的 LLM 相比，S2A 可以产生更讲事实、更少固执己见或阿谀奉承的 LLM。

特别是在问题中包含干扰性观点的修正后 TriviQA 数据集上，与 LLaMA-2-70B-chat 相比，S2A 将事实性从 62.8% 提高到 80.3%；在包含干扰性输入情绪的长格式参数生成任务重，S2A 的客观性提高了 57.4%，并且基本上不受插入观点的影响。此外对于 GSM-IC 中带有与主题不相关语句的数学应用题，S2A 将准确率从 51.7% 提高到了 61.3%。

本文在三种设置下进行了实验：事实问答、长论点生成以及对数学应用题的解决。此外，本文还使用 LLaMA-2-70B-chat 作为基础模型，在两种设置下进行评估：

- 基线：数据集中提供的输入提示被馈送到模型，并以零样本方式回答。模型生成可能会受到输入中提供的虚假相关性的影响。
    
- Oracle Prompt：没有附加意见或不相关句子的提示被输入到模型中，并以零样本的方式回答。 
    
图 5 (左) 展示了在事实问答上的评估结果。System 2 Attention 比原来的输入提示有了很大的改进，准确率达到 80.3%—— 接近 Oracle Prompt 性能。

![](img/Pasted%20image%2020231126154021.png)

图 6（左）显示了长论点生成的总体结果，基线、Oracle Prompt 以及 System 2 Attention 都被评估为可以提供类似的高质量评估。图 6（右）为细分结果：

![](img/Pasted%20image%2020231126154029.png)

图 7 显示了不同方法在 GSM-IC 任务上的结果。与 Shi 等人的研究结果一致，本文发现基线准确率远低于 oracle。当不相关的句子与问题属于同一主题时，这种影响甚至更大，如图 7（右）所示。

![](img/Pasted%20image%2020231126154106.png)



## 未来方向



## 主要收获


## 参考资料

[Yann LeCun点赞！Meta对Transformer架构下手了：新注意力机制更懂推理](https://mp.weixin.qq.com/s/Y2WkPnUhbwQJOEp7w4sB-Q)
