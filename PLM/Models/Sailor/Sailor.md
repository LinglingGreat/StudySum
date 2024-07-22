---
title: Sailor
created: 2024-07-13
tags:
  - 多语言
  - 增量预训练
type: 论文
papername: Sailor-Open Language Models for South-East Asia
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - seaAI
  - SUTD
---

## 论文基本信息

标题：Sailor: Open Language Models for South-East Asia

作者：

链接：

代码：https://sailorllm.github.io/

框架图：

![](img/Pasted%20image%2020240713155413.png)

Sailor 是一套专为东南亚 (SEA) 量身定制的开放语言模型，专注于 🇮🇩 印度尼西亚语、🇹🇭 泰语、🇻🇳 越南语、🇲🇾 马来语和 🇱🇦 老挝语等语言。Sailor 模型是在精心的数据管理下开发的，旨在理解和生成东南亚地区不同语言环境中的文本。Sailor 基于 Qwen 1.5 构建，包含不同大小的模型，从 0.5B 到 14B 版本，可满足不同需求。基准测试结果表明，Sailor 在东南亚语言的问答、常识推理、阅读理解等任务中表现出色。

- **对 7 种语言的2000 亿到 4000 亿个**标记 进行持续预训练，包括印尼语、泰语、越南语、马来语、老挝语、英语和中文。
- 各种型号尺寸（**0.5B**、**1.8B**、**4B**、**7B**、**14B**）可支持不同要求。
- 在 XQuAD、TydiQA、XCOPA、Belebele 和 M3Exam 等 SEA 基准测试中表现出色。

Sailor 的存在归功于开源社区。它通过语言模型（例如卓越的[Qwen 1.5](https://qwenlm.github.io/blog/qwen1.5/)模型）的持续预训练而精心打造，该模型在东南亚语言上已经表现出色。预训练语料库大量利用了公开可用的语料库，包括[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B)、[SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B)、[CC100](https://huggingface.co/datasets/cc100)和[MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400)。

通过对收集到的语料库进行积极的数据去重和仔细的数据清理，我们获得了涵盖各种语言的高质量数据集。通过系统性实验确定不同语言的权重，Sailor 模型从 200B 到 400B 个 token 进行训练，并根据不同的模型大小进行量身定制。这种方法提高了它们在东南亚语言上的表现，同时保持了对英语和中文的熟练程度，没有显著的妥协。最后，我们不断用 4000 亿个 token 对 Qwen1.5-0.5B 模型进行预训练，用 2000 亿个 token 对其他模型进行预训练，以获得 Sailor 模型。

## 背景



## 相关研究




## 核心亮点



## 实验




## 未来方向



## 主要收获


## 参考资料
