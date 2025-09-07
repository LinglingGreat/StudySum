---
title: InstructionPreTraining
created: 2024-06-26
tags:
  - 预训练
type: 论文
papername: Instruction Pre-Training- Language Models are Supervised Multitask Learners
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 微软
  - 清华
---

## 论文基本信息

标题：Instruction Pre-Training: Language Models are Supervised Multitask Learners

作者：

链接：

代码： https://huggingface.co/instruction-pretrain

框架图：

![](img/Pasted%20image%2020240626203827.png)


## 背景



## 相关研究




## 核心亮点

收集数据：我们从各种基于上下文的任务完成数据集中进行采样。如图所示。

![](img/Pasted%20image%2020240626205754.png)

训练框架

![](img/Pasted%20image%2020240626204153.png)

![](img/Pasted%20image%2020240626204443.png)

训练目标依然是Next token prediction，计算所有文本的loss

## 实验

General Pre-Training From Scratch：一部分数据去生成指令数据，剩下的保持不变。还加入微调instruction synthesizer的数据，增强任务多样性。

Domain-Adaptive Continual Pre-Training：所有数据都去生成指令数据，加入general instructions增强prompting ability，不加入微调instruction synthesizer的数据（因为general instructions中已经有了）

Instruction Synthesizer：基于Mistral-7B-v0.1训练得到。推理的时候每个原始文本会构造大约5个instruction-response对。

General Pre-Training From Scratch：从RefinedWeb中采样了200M pieces of text containing about 100B tokens，去生成指令数据。由于微调数据量（0.2B tokens）与原始语料相比太小，因此我们增加其样本比例，使其在整个预训练过程中重复 4 次。

Domain-Adaptive Continual Pre-Training：两个领域的数据PubMed Abstracts for biomedicine and financial news for finance.


训练参数：

![](img/Pasted%20image%2020240626210735.png)


![](img/Pasted%20image%2020240626211412.png)

我们推断，在指令预训练和指令调整阶段训练任务的更紧密结合有利于预训练和微调之间的平滑过渡。这种对齐使模型能够更快地学习下游任务。因此，指令预训练提供了一个有前途的解决方案，可以显着减少进一步微调步骤的数量。

![](img/Pasted%20image%2020240626211612.png)

![](img/Pasted%20image%2020240626211910.png)



## 未来方向



## 主要收获


## 参考资料
