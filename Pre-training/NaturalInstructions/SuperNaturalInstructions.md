---
title: NaturalInstructionsv2
created: 2023-03-07
tags: Benchmark
type: 论文
papername: SUPER-NATURALINSTRUCTIONS- Generalization via Declarative Instructions on 1600+ NLP Tasks
conference: EMNLP
year: 2022
institution: AllenAI
---

## 论文基本信息

标题：SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks

**Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks 2022.4.16**

作者：

链接：

代码： https://instructions.apps.allenai.org/

框架图：

![](img/Pasted%20image%2020230307174940.png)

![](img/Pasted%20image%2020230307174956.png)


## 背景

Motivation：

提出NATURAL-INSTRUCTIONSv2新的评估模型泛化能力Benchmark，涵盖了1600+个任务、70+个不同任务类型、50+种不同语言，和FLAN不同的是使用了in-context learning。
- 包括但不限于分类、提取、填充、序列标记、文本重写和文本组合

Method：

a.训练模型：3B T5

b.数据集：1616个task，76个task类型，16种推理类型， 非英文任务有576个，每个任务平均有3k+个样本。这些任务是由88位人员从GitHub社区收集来的，覆盖了业界已公布的数据以及新构造的任务，并且添加了详尽的任务介绍，已确保满足读者的需求。其中包括机器翻译、QA，文本分类等任务类型，英语、西班牙语、日语等语种，涉及新闻、对话、数学等多个领域学科，可见其丰富程度。分成了验证集合和训练集合两个子集，一个用于评估，一个用于监督训练。其中人工挑选的验证集合包括12个类别154个任务，其中119个英文任务，35个跨语种任务。每个任务选出100个随机实例用来评估，增加了Explanation。

c.评估方法：在评估指标上，选用文本生成中广泛使用的ROUGE-L，除此之外，对于输出较短且有限的任务，还引入了Exact Match,衡量模型推理出的字符串和标准输出完全匹配的比率，其中主要评估english和x-lingual两大类数据集。

d:训练方法：使用in-context learning，Tk-INSTRUCT其中k代表示例的数量，k=2代表2个示例。训练2个epoch，输入最大长度设置为1024，输出设置为128，学习率采用1e-5。

## 贡献

在新的benchmark上比T5、GPT3、T0和GPT3-Instruct效果好，其中实验表明更多任务和模型规模增大有利于模型泛化提升，而每个任务数量和更多的examples并没有帮助模型泛化提升，负样例对效果有一点提升。

## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？


## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向


## 核心亮点


## 主要收获

## 参考资料

https://zhuanlan.zhihu.com/p/558286175   