---
title: Qwen2.5
created: 2024-12-22
tags:
  - 大模型
  - LLM
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 阿里
---

## 论文基本信息

标题：Qwen2.5 Technical Report

作者：

链接：

https://huggingface.co/Qwen 
https://modelscope.cn/organization/qwen 
https://github.com/QwenLM/Qwen2.5

代码：

框架图：


## 背景
论文试图解决什么问题？这是否是一个新的问题？

这篇文章要验证一个什么科学假设？

论文中提到的解决方案之关键是什么？


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

在预训练方面，将高质量的预训练数据集从之前的 7 万亿个 token 扩展到了 18 万亿个 token。这为常识、专业知识和推理能力提供了坚实的基础。

在后训练方面，实现了超过100万个样本的复杂监督微调，以及多阶段强化学习，包括离线学习DPO和在线学习GRPO。训练后技术显着增强了人类的偏好，并显着改善了长文本生成、结构数据分析和指令遵循。

推出了丰富配置的Qwen2.5 LLM系列。包括基座模型和指令调整模型，参数大小为 0.5B、1.5B、3B、7B、14B、32B 和 72B。还提供了指令调整模型的量化版本。可以从 Hugging Face Hub、ModelScope 和 Kaggle 访问 100 多个模型。此外，对于托管解决方案，专有模型目前包括两种专家混合 (MoE) 变体：Qwen2.5Turbo 和 Qwen2.5-Plus，均可以从阿里云模型工作室获取。

Qwen2.5-72B-Instruct 表现出与最先进的开放式重量模型 Llama-3-405B-Instruct 相比的竞争性能。Qwen2.5-Turbo 和 Qwen2.5- Plus，其性能分别与 GPT-4o-mini 和 GPT-4o 竞争。 

## 模型结构

![](img/Pasted%20image%2020241222124849.png)

基于Transformer的decoder结构，模型结构的关键要素是：有助于高效的KV缓存利用的GQA（分组查询注意力机制），SwiGLU非线性激活函数，用于编码位置信息的RoPE，注意力机制中的QKV偏差，具有预归一化的RMSNorm。后两者是为了保证稳定的训练。



![](img/Pasted%20image%2020241222122023.png)




## 未来方向



## 主要收获


## 参考资料
