---
title: DeepSeek-R1
created: 2025-01-21
tags:
  - cot
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - DeepSeek
---

## 论文基本信息

标题：

作者：

链接：https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

代码：https://huggingface.co/deepseek-ai

框架图：


## 背景

我们介绍我们的第一代推理模型 DeepSeek-R1-Zero 和 DeepSeek-R1。 

我们迈出了**使用纯强化学习（RL）（无需监督学习）提高语言模型推理能力**的第一步。我们的目标是探索法学硕士在没有任何监督数据的情况下发展推理能力的潜力，重点关注它们通过纯强化学习过程的自我进化。

我们使用 DeepSeek-V3-Base 作为基础模型，并采用 GRPO (Shao et al., 2024) 作为 RL 框架来提高模型的推理性能。在训练过程中，DeepSeek-R1-Zero自然而然地表现出了许多强大而有趣的推理行为。经过数千个 RL 步骤后，DeepSeek-R1-Zero 在推理基准测试中展现出超强的性能。例如，AIME 2024 上的 pass@1 分数从 15.6% 提高到 71.0%，通过多数投票，分数进一步提高到 86.7%，与 OpenAI-o1-0912 的性能相当。

然而，DeepSeek-R1-Zero 遇到了可读性差、语言混合等挑战。为了解决这些问题并进一步提高推理性能，我们推出了 DeepSeek-R1，**它结合了少量的冷启动数据和多阶段训练管道**。具体来说，我们首先收集数千个冷启动数据来微调 DeepSeek-V3-Base 模型。接下来，我们像 DeepSeek-R1Zero 一样执行面向推理的强化学习。当 RL 过程接近收敛时，我们通过 RL 检查点上的拒绝采样，结合来自 DeepSeek-V3 在写作、事实 QA 和自我认知等领域的监督数据来创建新的 SFT 数据，然后重新训练 DeepSeek-V3 -基础模型。在使用新数据进行微调后，检查点会经历额外的强化学习过程，同时考虑所有场景的提示。经过这些步骤，我们获得了一个名为 DeepSeek-R1 的检查点，其性能与 OpenAI-o1-1217 相当。

我们进一步探索从 DeepSeek-R1 到更小的密集模型的蒸馏。使用 Qwen-2.5-32B（Qwen，2024b）作为基础模型，**从 DeepSeek-R1 直接蒸馏的性能优于对其应用 RL**。这表明较大的基础模型发现的推理模式对于提高推理能力至关重要。我们的蒸馏 14B 模型大幅优于最先进的开源 QwQ-32B-Preview（Qwen，2024a），并且蒸馏 32B 和 70B 模型在密集模型的推理基准上创下了新记录。

为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1 以及基于 Qwen 2.5和 Llama3 从 DeepSeek-R1 中蒸馏出来的六个密集模型（1.5B、7B、8B、14B、32B、70B）。

![](img/Pasted%20image%2020250121103056.png)


## 相关研究

process-based reward models (Lightman et al., 2023; Uesato et al., 2022; Wang et al., 2023)
reinforcement learning (Kumar et al., 2024)
search algorithms such as Monte Carlo Tree Search and Beam Search (Feng et al., 2024; Trinh et al., 2024; Xin et al., 2024).


## DeepSeek-R1-Zero: Reinforcement Learning on the Base Model



## DeepSeek-R1: Reinforcement Learning with Cold Start



## Distillation: Empower Small Models with Reasoning Capability



## 实验




## 讨论



## 主要收获


## 参考资料
