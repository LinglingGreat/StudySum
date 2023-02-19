---
title: FLAN-PALM_T5
created: 2023-02-19
tags: instruction-tuning 
type: 论文
papername: Scaling Instruction-Finetuned Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: 谷歌
---

## 论文基本信息

标题：Scaling Instruction-Finetuned Language Models

作者：

链接：

代码：

框架图：


在吸收FLAN的精华的基础上，加入了CoT的数据来做finetune

![](img/Pasted%20image%2020230219165016.png)

这么finetune过后的模型，其实不论在CoT任务和非CoT任务上其实都表现得最好，而且在BBH上做zeroshot优势更是巨大。这也进一步证明了CoT是可以和当前流行的instruction tuning无缝衔接的。

![](img/Pasted%20image%2020230219170727.png)

![](img/Pasted%20image%2020230219170809.png)



## 核心亮点

## 主要收获

