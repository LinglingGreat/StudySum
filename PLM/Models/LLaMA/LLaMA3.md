---
title: LLaMA3
created: 2024-07-24
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - MetaAI
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：

![](img/Pasted%20image%2020240724144456.png)

![](img/Pasted%20image%2020240724144934.png)

LLaMA-3
- supports multilinguality, coding, reasoning, and tool usage
- 最大的模型是405B的dense模型，支持128K

训练高质量基座模型的三个关键点：data, scale, and managing complexity
- pre-training and post-training阶段的数据质量和数量都提升了。Llama3用了15T tokens训练，而Llama2用了1.8T tokens
- 训练了一个405B的dense模型，用了15.6T tokens。we also train our smaller models for much longer than is compute-optimal. 结果是更好
- 选择dense架构而不是MOE，训练更稳定；采用SFT，RS(拒绝采样), DPO而不是难以训练和扩展的PPO



## 参考资料
