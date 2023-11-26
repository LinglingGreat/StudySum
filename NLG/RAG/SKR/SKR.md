---
title: SKR
created: 2023-11-26
tags:
  - 检索增强
type: 论文
papername: Self-Knowledge Guided Retrieval Augmentation for Large Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
  - 清华
  - 上海人工智能实验室
---

## 论文基本信息

标题：Self-Knowledge Guided Retrieval Augmentation for Large Language Models

作者：

链接： https://arxiv.org/pdf/2310.05002.pdf

代码：

框架图：


## 背景


## 相关研究


## 核心亮点

发现 RAG 召回的辅助信息不总是有用，甚至可能起负作用，因此设计了名为 SKR （Self-Knowledge Guided Retrieval Augmentation）的框架，对模型本身已知的问题直接生成答案，对未知的问题才调用 RAG 模块。

![](img/Pasted%20image%2020231126155851.png)

**1. 自我知识收集：**首先要知道自己知道什么，不知道什么（开始绕口令），因此收集一批有标注的训练集，模型可以直接答对的视为 known，检索增强后才能答对的视为 unknown； 

**2. 识别是否已知：**对输入的测试问题，利用在训练集上构建的分类器识别其是否已知。分类器构建的方式作者试了好几种，可以用大模型本身上下文学习，可以用 RoBERTa 小模型训个分类器，也可以用 SimCSE的 embedding 为嵌入直接 KNN 分类（实验中 KNN 的性能最好）； 

**3. 自适应式检索增强：**只对第二步中识别为 unknown 的输入进行检索增强，其余输入视为 known，直接回答。



## 实验

实验是在一些 QA 数据集上做的，LM 是 InstructGPT 和 ChatGPT，似乎没有详细说明预训练 retriever 是什么模型，结果显示 KNN 版本的 SKR 与不带检索增强的 CoT 以及非自适应的 RAG+CoT 类型的基线相比，能取得 3%-4% 的显著提升。

![](img/Pasted%20image%2020231126155957.png)

上述方法的前二步中，识别 known/unknown 的分类器是在和测试样本同分布的训练集上构建的，而且实验中似乎设定是用了完整的训练集（这样一来实际上有信息泄露，与其他的 zero-shot 和 few-shot 方法比较并不公平）。作者也讨论训练集大小的影响，但是有一点避重就轻的感觉，只表示训练集减小到 10% 会导致 2-3 个点的下降，该方法在训练集和测试集不同分布/可用的样本数很少的情况下的有效性还有待确认。



## 未来方向



## 主要收获


## 参考资料
