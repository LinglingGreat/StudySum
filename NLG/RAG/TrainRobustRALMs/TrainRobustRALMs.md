---
title: TrainRobustRALMs
created: 2023-11-26
tags: 
type: 论文
papername: Making Retrieval-Augmented Language Models Robust to Irrelevant Context
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
  - AllenAI
  - TAU
---

## 论文基本信息

标题：Making Retrieval-Augmented Language Models Robust to Irrelevant Context

作者：

链接： https://arxiv.org/pdf/2310.01558.pdf

代码：

框架图：


## 背景

本文是篇偏实验分析性质的文章，测试了 NLI 过滤召回结果和直接模拟带噪声的召回内容进行训练两类方法。 

首先，直接用 NLI 预训练模型判断召回的文档和问题是否相关来进行过滤，结论是 NLI 模型的过滤虽然能提升召回信息质量低时模型的鲁棒性，但也会伤及无辜（过滤掉有用的信息），在以 Google 搜索 top-1 为召回内容时总体上是掉点的。 

接下来的方法非常直接暴力，既然 RAG 范式中检索来的内容 可能有噪音，大模型预训练的时候又没见过这种鱼龙混杂的上文，干脆发扬 end-to-end 的精神直接训练。坐着构建了一个 1.5k 样本的训练集，其中包含干净的 context 和扰动的 context，希望模型学习到“不论如何都能输出正确答案”的能力。

结果确实显示该数据上微调后的 LLaMa2-13B 模型在各种 QA 任务上，无论是正常的 Google 搜索召回、故意召回排名低的文档（low-rank retrieval），还是随机召回，都能比普通的 RAG 显著提升准确率，在 low-rank 和 random 的设定下基本和不带 RAG 的原模型相当。

有一点缺憾是，本文没有讨论这种微调是否影响了模型在其他领域的通用能力，未来或许可以考虑将这种为 RAG 鲁棒性设计的数据集加到模型的预训练或者 SFT 阶段中。


## 相关研究




## 核心亮点



## 实验




## 未来方向



## 主要收获


## 参考资料
