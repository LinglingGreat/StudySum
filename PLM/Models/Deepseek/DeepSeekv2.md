---
title: DeepSeekv2
created: 2025-01-22
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - DeepSeek
---

## 论文基本信息

标题：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

作者：

链接：

代码：

框架图：


## 背景
我们提出了 DeepSeek-V2，一种强大的专家混合 (MoE) 语言模型，其特点是经济的训练和高效的推理。它总共包括236B个参数，其中每个令牌激活21B个参数，并支持128K令牌的上下文长度。 DeepSeek-V2采用多头潜在注意力（MLA）和DeepSeekMoE等创新架构。 MLA 通过将键值 (KV) 缓存显着压缩为潜在向量来保证高效推理，而 DeepSeekMoE 则可以通过稀疏计算以经济的成本训练强大的模型。与 DeepSeek 67B 相比，DeepSeek-V2 性能显着增强，同时节省了 42.5% 的训练成本，减少了 93.3% 的 KV 缓存，最大生成吞吐量提升至 5.76 倍。我们在由 8.1T 代币组成的高质量多源语料库上对 DeepSeek-V2 进行预训练，并进一步进行监督微调（SFT）和强化学习（RL）以充分释放其潜力。评估结果显示，即使只有21B个激活参数，DeepSeek-V2及其聊天版本仍然达到了开源模型中顶级的性能

![](img/Pasted%20image%2020250122112029.png)

我们构建了由 8.1T 令牌组成的高质量、多源预训练语料库。与DeepSeek 67B（我们之前版本）（DeepSeek-AI，2024）使用的语料库相比，该语料库的数据量特别是中文数据量更大，数据质量更高。我们首先在完整的预训练语料库上预训练 DeepSeek-V2。然后，我们收集 150 万个对话会话，其中涵盖数学、代码、写作、推理、安全等各个领域，为 DeepSeek-V2 聊天 (SFT) 执行监督微调 (SFT)。最后，我们遵循 DeepSeekMath (Shao et al., 2024) 采用组相对策略优化 (GRPO) 进一步使模型与人类偏好保持一致，并生成 DeepSeek-V2 Chat (RL)。


## 架构

![](img/Pasted%20image%2020250122113329.png)

总的来说，DeepSeek-V2 仍然采用 Transformer 架构（Vaswani 等人，2017），其中每个 Transformer 块由一个注意力模块和一个前馈网络（FFN）组成。然而，对于注意力模块和 FFN，我们设计并采用了创新的架构。为了引起注意，我们设计了MLA，它利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效的推理。对于 FFN，我们采用 DeepSeekMoE 架构（Dai et al., 2024），这是一种高性能 MoE 架构，能够以经济的成本训练强大的模型。 DeepSeek-V2 的架构如图 2 所示。

### Multi-Head Latent Attention

![](img/Pasted%20image%2020250122114057.png)

设 d 为embedding维度，$n_h$ 为注意力头的数量，$d_h$ 为每个头的维度，$h_t$ ∈ Rd 为注意力层第 t 个标记的注意力输入。$q_t, k_t, v_t ∈ R^{d_hn_h}$

MHA 需要为每个 token 缓存 $2n_hd_hl$ 元素。（l是层数）

![](img/Pasted%20image%2020250122114929.png)

![](img/Pasted%20image%2020250122114552.png)

MLA的核心是对key和value进行低秩联合压缩，以减少KV缓存

![](img/Pasted%20image%2020250122115535.png)

为了减少训练期间的激活记忆，也对查询进行低秩压缩，即使它不能减少 KV 缓存

![](img/Pasted%20image%2020250122115743.png)

ROPE

![](img/Pasted%20image%2020250122120013.png)

我们在表 1 中展示了不同注意力机制中每个 token 的 KV 缓存的比较。MLA 仅需要少量的 KV 缓存，相当于只有 2.25 个组的 GQA，但可以实现比 MHA 更强的性能。

![](img/Pasted%20image%2020250122120132.png)




## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
