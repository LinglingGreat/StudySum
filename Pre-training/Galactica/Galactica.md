---
title: Galactica
created: 2023-02-24
tags: Science 大模型
type: 论文
papername: Galactica A Large Language Model for Science
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: MetaAI
---

## 论文基本信息

标题：Galactica: A Large Language Model for Science

作者：

链接：

代码：

框架图：

### 数据集

![](img/Pasted%20image%2020230224150558.png)

在不同的数据类型上用了special token
- citation: [START_REF] and [END_REF]
- Step-by-Step Reasoning: `<work>`
- Mathematics: 把ASCII分割成单个字符
- Numbers：737612.62 -> 7,3,7,6,1,2,.,6,2.
- SMILES formula：[START_SMILES] and [END_SMILES]
- Amino acid sequences：[START_AMINO] and [END_AMINO]，MIRLGAPQTL -> M,I,R,L,G,A,P,Q,T,L.
- DNA sequences: [START_DNA] and [END_DNA]. For example, CGGTACCCTC -> C, G, G, T, A, C, C, C, T, C.

![](img/Pasted%20image%2020230227134535.png)

![](img/Pasted%20image%2020230227134607.png)

![](img/Pasted%20image%2020230227135810.png)

### 模型

decoder模型

2048长度

![](img/Pasted%20image%2020230227140137.png)

![](img/Pasted%20image%2020230227140232.png)



## 核心亮点

## 主要收获

