---
title: LLaMA
created: 2023-02-27
tags: 大模型 语言模型
type: 论文
papername: LLaMA Open and Efficient Foundation Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution: MetaAI
---

## 论文基本信息

标题：LLaMA: Open and Efficient Foundation Language Models

作者：

链接：

代码： https://github.com/facebookresearch/llama

框架图：

LLaMA-13B的性能优于GPT-3，而体积却小了10倍以上，LLaMA-65B与Chinchilla-70B和PaLM-540B具有竞争性。

Meta表示，该模型在数以万亿计的token上进行训练，并表明有可能完全使用公开的数据集来训练最先进的模型，而不需要求助于专有的和不可获取的数据集。

Hoffmann等人（2022）最近的工作表明，在给定的计算预算下，最好的性能不是由最大的模型实现的，而是由在更多数据上训练的较小的模型实现的。

Hoff-mann等人（2022）的缩放定律的目标是确定如何在特定的训练计算预算下最佳地扩展数据集和模型大小。然而，这个目标忽略了推理预算，而推理预算在大规模服务语言模型时变得至关重要。

在这种情况下，给定一个目标性能水平，首选的模型不是训练速度最快的，而是推理速度最快的，尽管训练一个大的模型以达到一定的性能水平可能更便宜，但训练时间较长的小模型最终会在推理中更便宜。

例如，Hoffmann等人（2022年）曾建议在200B的token上训练一个10B的模型，但研究发现7B的模型的性能甚至在1T的token之后还能继续提高。

因此，该工作的重点是训练一系列语言模型，通过对比通常使用的更多的token进行训练，在不同的推理预算下达到最佳的性能。

### 数据



## 核心亮点

## 主要收获

