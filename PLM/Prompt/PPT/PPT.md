---
title: PPT
created: 2023-02-11
tags: prompt fewshot
type: 论文
papername: PPT Pre-trained prompt tuning for few-shot learning
conference: ACL
year: 2022
institution: 清华
---

## 论文基本信息

标题：PPT: Pre-trained prompt tuning for few-shot learning

作者：Yuxian Gu, Xu Han, Zhiyuan Liu, Minlie Huang

链接：

代码： https://github.com/thu-coai/PPT

框架图：

之前的工作都是在finetune阶段去使用prompt，这篇文章第一次提出了prompt pretraining的过程。一开始是因为观察了prompt tuning中的大模型尽管在全量数据下能够媲美finetune，但是在少样本情况下并不好，作者认为是因为在大模型上soft prompt对初始化很敏感，所以设计了一系列预训练的prompt task来给soft prompt提供一个很好的初始化。这就是 Pre-trained Prompt Tuning (PPT)。

![](img/Pasted%20image%2020230211151344.png)

### 试点试验

Hybrid Prompt Tuning和Verbalizer Selection.

作者将 soft prompt 和 3 个人工设计的 hard prompt、2 个自动生成的 hard prompt 相结合。_P_ 是 soft prompt，s 是输入语句。该方法有益于 prompt tuning，但是效果依然不如 fine-tuning。

作者对比了同一个 prompt 模板下不同 verbalizer 的效果，发现verbalizer 的选择影响很大。一般来说，解释对应标签含义的词效果更好。


![](img/Pasted%20image%2020230211151743.png)

Real Word Initialization

在fewshot场景下，作者尝试了四种初始化策略，这些策略在以前的工作中得到了验证，被证明在小型模型中是有效的。但是作者尝试了在具有 11B 参数的模型中使用具体词的嵌入来初始化 soft prompt 标记，作用很小甚至为负。

![](img/Pasted%20image%2020230211152141.png)

此外，上面三种方法都不能很好地解决 few-shot 的情况下的 prompt tuning 问题。

### 方法

![](img/Pasted%20image%2020230211161509.png)

### 实验

作者在每个数据集上使用32个训练样本和32个验证样本进行实验。分类任务结果如下：

![](img/Pasted%20image%2020230211163106.png)

主要有以下几个结论：

-   **fine-tuning 之间的对比**：模型越大，fine-tuning 的效果越好。这说明 few-shot 情况下大模型还是更有优势的。
-   **prompt-tuning 之间的对比**：大部分数据集下，PPT 明显优于 Vanilla PT 和 LM Adaption，而在简单地将 PPT 和 hard prompt 结合之后（即 Hybrid PPT），几乎在所有数据集中都取得了最好的效果。这说明预训练 prompt 和混合 prompt 可能是互补的。
-   **PPT 与 fine-tuning 的对比**：PPT 在大多数英文数据集和所有中文数据集下都由于 fine-tuning，这说明 PPT 比 fine-tuning 更能够弥合 MLM 与下游任务之间的差异。
-   **prompt-tuning 效果的方差对比**：few-shot 情况下，各家 prompt-tuning 在不同数据集上的表现非常不稳定，而 PPT 在所有数据集上的表现方差显著减小。

  

## 核心亮点

## 主要收获

## 参考资料

https://juejin.cn/post/7063853199307833381  

https://www.jiqizhixin.com/articles/2021-09-11-2