---
title: SuperGLUE
created: 2023-03-01
tags: Benchmark
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2019
institution: Facebook 纽约大学 华盛顿大学 DeepMind
---

## 论文基本信息

标题：

作者：

链接：

代码： https://super.gluebenchmark.com/leaderboard/

框架图：


8 个语言理解任务

-   **布尔问题**（Boolean Questions，**BoolQ**）：QA任务，每个例子由一段短文和一个关于这段话的是/否问题组成。用准确性来评估。
    
-   **CommitmentBank**（**CB**）：要求模型识别 文本中包含的假设，包括《华尔街日报》的信息来源，并确定该假设是否成立。
    
-   **合理选择**（Choice of plausible alternatives，**COPA**）： 提供了一个关于博客主题的前提语句，以及一本与摄影相关的百科全书，模型必须从中确定两种可能选择的因果关系。
    
-   **多句阅读理解**（Multi-Sentence Reading Comprehension，**MultiRC**）：这是一项问答式的任务，其中每个样本都包含一段上下文段落、一个关于该段落的问题，以及一系列可能的答案。一种模型必须预测哪些答案是真的，哪些答案是假的。
    
-   **基于常识推理数据集的阅读理解**（Reading Comprehension with Commonsense Reasoning Dataset，**ReCoRD**）：模型根据 CNN 和《每日邮报》的选文列表中预测被掩盖的单词和短语，在这些选文中，同一单词或短语可能以多种不同的形式表达，所有这些都被认为是正确的。
    
-   **识别文本内容**（Recognizing Textual Entailment，**RTE**）：挑战自然语言模型，以确定一个文本摘录的真实性是否来自另一个文本摘录。

-   **Word-in-Context**（**WiC**）：为两个文本片段和一个多义词（即具有多重含义的单词）提供模型，并要求它们判定这个单词是否在两个句子中有相同的含义。
    
-   **Winograd 模式挑战**（Winograd Schema Challenge，**WSC**）：是一项任务，在这项任务中，模型给定小说书中的段落，必须回答关于歧义代词先行词的多项选择题。它被设计为图灵测试的改进。

![](img/Pasted%20image%2020230301180747.png)

![](img/Pasted%20image%2020230301180809.png)

