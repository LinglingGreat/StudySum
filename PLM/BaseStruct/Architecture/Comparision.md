---
title: Comparision
created: 2025-07-22
tags:
  - 模型架构
---

[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)  

# DeepSeek

- DeepSeek-MLA，在DeepSeek-V2的论文中的消融实验表明，GQA 的性能似乎不如 MHA，而 MLA 的建模性能优于 MHA，这可能就是 DeepSeek 团队选择 MLA 而非 GQA 的原因。
- DeepSeek-MOE，MOE是将Transformer 模块中的每个单个前馈模块替换为多个前馈模块。DeepSeek-V3 每个 MoE 模块有 256 位专家，总共 6710 亿个参数。在推理过程中，每次只有 9 位专家处于活动状态（1 位共享专家加上 8 位由路由器选择的专家）。这意味着每个推理步骤仅使用 370 亿个参数。
- DeepSeek-V3 的 MoE 设计的一个显著特点是使用了一个共享专家。这是一个始终对每个 token 保持活跃的专家。在 [DeepSeek 2024 MoE](https://arxiv.org/abs/2401.06066) 和 [2022 DeepSpeedMoE 论文](https://arxiv.org/abs/2201.05596)中就已提出 。他们发现与没有共享专家相比，共享专家可以提升整体建模性能。

# OLMo

在训练数据和代码方面非常透明，并且技术报告也相对详细。

![](img/Comparision-20250722161847.png)

OLMo 2 仍然采用传统的多头注意力（MHA），而不是 MLA 或 GQA。

## RMNSorm

与 Llama、Gemma 和大多数其他 LLM 类似，OLMo 2 从 LayerNorm 转换为 RMSNorm。RMSNorm 已经过时（它本质上是 LayerNorm 的简化版本，可训练参数更少）。

然而，RMNSorm 层的放置位置值得讨论。原始 Transformer（来自“ [Attention is all you need](https://arxiv.org/abs/1706.03762) ”论文）将两个规范化层分别放置在 Transformer 模块中的注意力模块和前馈模块**之后**。这也称为 Post-LN 或 Post-Norm。

GPT 和之后出现的大多数 LLM 将归一化层置于注意力模块和前馈模块**之前** ，这被称为 Pre-LN 或 Pre-Norm。

![](img/Comparision-20250722162432.png)

[2020 年，熊等人。](https://arxiv.org/abs/2002.04745) 表明 Pre-LN 在初始化时会导致更良好的梯度。此外，研究人员提到，Pre-LN 甚至可以在没有仔细的学习率预热的情况下运行良好，这是 Post-LN 的重要工具。

 OLMo 2 采用了一种 Post-LN 形式（但使用 RMSNorm 而不是 LayerNorm，所以我称之为 _Post-Norm_）。在 OLMo 2 中，它们不是将归一化层放在注意力层和前馈层之前，而是将它们放在后面，如上图所示。但是，请注意，与原始 Transformer 架构相比，归一化层仍在残差层（跳过连接）内。

那么，他们为什么要移动归一化层的位置呢？ 原因是它有助于训练稳定性，如下图所示。

![](img/Comparision-20250722162944.png)

不幸的是，此图显示了与 QK-Norm 一起重新排序的结果，这是一个单独的概念。因此，很难判断归一化层重新排序本身贡献了多少。

## QK-Norm

QK-Norm 本质上是另一个 RMSNorm 层。它放置在多头注意力 （MHA） 模块中，并在应用 RoPE 之前应用于查询 （q） 和键 （k）。为了说明这一点，下面是我为我的 [Qwen3 从头开始实现](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)编写的分组查询注意力 （GQA） 层的摘录 （GQA 中的 QK-norm 应用程序类似于 OLMo 中的 MHA）：

```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups,
        head_dim=None, qk_norm=False, dtype=None
    ):
        # ...

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x) 
        keys = self.W_key(x)
        values = self.W_value(x) 

        # ...

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        # ...
```

如前所述，QK-Norm 与 Post-Norm 一起稳定了训练。请注意，QK-Norm 不是由 OLMo 2 发明的，而是可以追溯到 [2023 年 Scaling Vision Transformers 论文](https://arxiv.org/abs/2302.05442) 。

下图进一步并排比较了 OLMo 2 和 Llama 3;可以看出，除了 OLMo 2 仍然使用传统的 MHA 而不是 GQA 之外，这些架构在其他方面相对相似。（但是， OLMo 2 团队在 3 个月后发布了使用 GQA 的 [32B 变体](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) 。

![](img/Comparision-20250722163515.png)

# Gemma 3

Gemma 的显着特点之一是相当大的词汇量（以更好地支持多种语言），并且更注重 27B 大小（相对于 8B 或 70B）。但请注意，Gemma 2 也有更小的尺寸：1B、4B 和 12B。27B 尺寸达到了一个非常好的最佳位置：它比 8B 型号功能强大得多，但不像 70B 型号那样占用资源.

如前所述，其他模型（如 Deepseek-V3/R1）使用混合专家 （MoE） 架构来减少推理时的内存需求，前提是模型大小固定。（我们稍后将讨论的其他几种模型也使用了 MoE 方法。

Gemma 3 使用了不同的“技巧”来降低计算成本，即滑动窗口注意力。

## **Sliding Window Attention**

通过滑动窗口注意力（最初在 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150)中引入，[Gemma 2](http://arxiv.org/abs/2408.00118) 也已经使用 ），Gemma 3 团队能够大幅减少 KV 缓存中的内存需求，如下图所示。

![](img/Comparision-20250722163710.png)

![](img/Comparision-20250722163749.png)

请注意，滑动窗口注意力可以与 Multi-Head Attention 和 Grouped-Query Attention 一起使用;Gemma 3 使用分组查询注意力。

Gemma 2 的前身架构之前也使用了滑动窗口注意力。Gemma 3 的不同之处在于它们调整了全局（常规）和局部（滑动）注意力之间的比率。

例如，Gemma 2 使用混合注意力机制，将滑动窗口（局部）和全局注意力以 1：1 的比例组合在一起。每个 Token 都可以关注附近上下文的 4k Token 窗口。

Gemma 2 在每隔一层中使用滑动窗口注意力，而 Gemma 3 现在的比率为 5：1，这意味着每 5 个滑动窗口（局部）注意力层只有 1 个完整的注意力层;此外，滑动窗口尺寸从 4096 （Gemma 2） 减少到仅 1024 （Gemma 3）。这将模型的关注点转移到更高效的本地化计算上。

根据他们的消融研究，使用滑动窗口注意力对建模性能的影响最小，如下图所示

![](img/Comparision-20250722163933.png)

## **Normalization Layer Placement in Gemma 3**

需要强调的一个小而有趣的花絮是，Gemma 3 在其分组查询注意力模块的 Pre-Norm 和 Post-Norm 设置中使用了 RMSNorm。

这与 Gemma 2 类似，但仍然值得强调，因为它不同于 （1） 原始transformer中使用的 Post-Norm（2） Pre-Norm，它由 GPT-2 推广，并在之后的许多其他架构中使用，以及 （3） 我们之前看到的 OLMo 2 中的 Post-Norm 风格。

![](img/Comparision-20250722164035.png)

我认为这种归一化层放置是一种相对直观的方法，因为它可以两全其美：Pre-Norm 和 Post-Norm。在我看来，一点额外的规范化不会有什么坏处。在最坏的情况下，如果额外的规范化是多余的，这会通过冗余增加一些效率低下。实际上，由于 RMSNorm 在宏伟的计划中相对便宜，因此这应该不会产生任何明显的影响。

## **Gemma 3n**

在 Gemma 3 发布几个月后，Google 分享了 [Gemma 3n](https://developers.googleblog.com/en/introducing-gemma-3n/)，这是一款 Gemma 3n 模型，针对小型设备效率进行了优化，目标是在手机上运行。

Gemma 3n 中为提高效率而进行的其中一项更改是所谓的每层嵌入 （PLE） 参数层。这里的关键思想是仅将模型参数的子集保留在 GPU 内存中。然后，令牌层特定的嵌入（例如用于文本、音频和视觉模态的嵌入）会按需从 CPU 或 SSD 流式传输。

下图说明了 PLE 内存节省，列出了标准 Gemma 3 模型的 54.4 亿个参数。这可能指的是 Gemma 3 40 亿的变体。

![](img/Comparision-20250722165900.png)

5.44 与 40 亿参数的差异是因为 Google 有一种有趣的方法来报告 LLM 中的参数计数。它们通常排除嵌入参数以使模型看起来更小，但在这种情况下除外，在这种情况下，可以方便地包含它们以使模型看起来更大。这并不是谷歌独有的，因为这种方法已成为整个领域的普遍做法。

另一个有趣的技巧是 [MatFormer](https://arxiv.org/abs/2310.07707) 概念（Matryoshka Transformer的缩写）。例如，Gemma 3n 使用单个共享 LLM（转换器）架构，该架构可以切片为更小的、独立可用的模型。每个切片都经过训练以独立运行，因此在推理时，我们可以只运行您需要的部分（而不是大型模型）。


