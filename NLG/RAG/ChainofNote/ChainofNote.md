---
title: ChainofNote
created: 2023-11-26
tags: 
type: 论文
papername: "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models"
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
  - 腾讯
---

## 论文基本信息

标题：Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models

作者：

链接：https://arxiv.org/pdf/2311.09210.pdf

代码：

框架图：


## 背景



## 相关研究




## 核心亮点

将思维链（CoT）方法用于增强 RAG 的鲁棒性，在中间推理过程中输出每一篇召回文档与输入问题的相关性（即对召回内容的 note）和自身对问题的认知，最后总结输出答案。作者用 ChatGPT 构造了一个这种格式的 CoT 训练集，将此能力蒸馏到了 LLaMa2 上，显著提升了 LLaMa2 带 RAG 时的鲁棒性。

本文考虑了OOD detection 的问题，即当模型本身和召回文档都不掌握回答问题需要的知识时，应该回答 unknown 而不是胡编乱造

![](img/Pasted%20image%2020231126161041.png)



## 实验




## 未来方向



## 主要收获


## 参考资料
