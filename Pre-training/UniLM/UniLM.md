---
title: UniLM
created: 2023-02-11
tags: 预训练 语言模型
type: 论文
papername: Unified language model pre-training for natural language understanding and generation
conference: NeurIPS
year: 2019
institution: 微软
---

## 论文基本信息

标题：Unified language model pre-training for natural language understanding and generation

作者：Li Dong,  Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon

链接： 

代码： https://github.com/microsoft/unilm

框架图：

中文版：https://github.com/YunwenTechnology/Unilm


UniLM是微软研究院在Bert的基础上，最新产出的预训练语言模型，被称为统一预训练语言模型。它可以完成单向、序列到序列和双向预测任务，可以说是结合了AR和AE两种语言模型的优点，Unilm在**抽象摘要、生成式问题回答**和**语言生成数据集的抽样领域**取得了最优秀的成绩。

-   单向训练语言模型，mask词的语境就是其单侧的words，左边或者右边。
-   双向训练语言模型，mask词的语境就是左右两侧的words。
-   Seq-to-Seq语言模型，左边的seq我们称source sequence，右边的seq我们称为target sequence，我们要预测的就是target sequence，所以其语境就是所有的source sequence和其左侧已经预测出来的target sequence。

**混合训练方式**：对于一个batch，1/3时间采用双向(bidirectional)语言模型的目标，1/3的时间采用seq-to-seq语言模型目标，最后1/3平均分配给两种单向学习的语言模型，也就是left-to-right和right-to-left方式各占1/6时间。

**masking 方式**：总体比例15%，其中80%的情况下直接用[MASK]替代，10%的情况下随机选择一个词替代，最后10%的情况用真实值。还有就是80%的情况是每次只mask一个词，另外20%的情况是mask掉bigram或者trigram。

模型输入X是一串word序列，该序列要么是用于单向语言模型的一段文本片段，要么是一对文本片段，主要用于双向或者seq-to-seq语言模型 。在输入的起始处会添加一个[SOS]标记，结尾处添加[EOS]标记。[EOS]一方面可以作为NLU任务中的边界标识，另一方面还能在NLG任务中让模型学到何时终止解码过程。其输入表征方式与 BERT 的一样，包括token embedding，position embedding，segment embedding，同时segment embedding还可以作为模型采取何种训练方式(单向，双向，序列到序列)的一种标识。

![](img/Pasted%20image%2020220109212739.png)

## 核心亮点

## 主要收获

## 参考资料

[【论文解读】UniLM:一种既能阅读又能自动生成的预训练模型](https://cloud.tencent.com/developer/article/1573393)

