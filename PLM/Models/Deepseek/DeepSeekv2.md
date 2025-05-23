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

# 论文基本信息

标题：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

作者：

链接：

代码：

框架图：


# 背景
我们提出了 DeepSeek-V2，一种强大的专家混合 (MoE) 语言模型，其特点是经济的训练和高效的推理。它总共包括236B个参数，其中每个令牌激活21B个参数，并支持128K令牌的上下文长度。 DeepSeek-V2采用多头潜在注意力（MLA）和DeepSeekMoE等创新架构。 MLA 通过将键值 (KV) 缓存显着压缩为潜在向量来保证高效推理，而 DeepSeekMoE 则可以通过稀疏计算以经济的成本训练强大的模型。与 DeepSeek 67B 相比，DeepSeek-V2 性能显着增强，同时节省了 42.5% 的训练成本，减少了 93.3% 的 KV 缓存，最大生成吞吐量提升至 5.76 倍。我们在由 8.1T 代币组成的高质量多源语料库上对 DeepSeek-V2 进行预训练，并进一步进行监督微调（SFT）和强化学习（RL）以充分释放其潜力。评估结果显示，即使只有21B个激活参数，DeepSeek-V2及其聊天版本仍然达到了开源模型中顶级的性能

![](img/Pasted%20image%2020250122112029.png)

我们构建了由 8.1T 令牌组成的高质量、多源预训练语料库。与DeepSeek 67B（我们之前版本）（DeepSeek-AI，2024）使用的语料库相比，该语料库的数据量特别是中文数据量更大，数据质量更高。我们首先在完整的预训练语料库上预训练 DeepSeek-V2。然后，我们收集 150 万个对话会话，其中涵盖数学、代码、写作、推理、安全等各个领域，为 DeepSeek-V2 聊天 (SFT) 执行监督微调 (SFT)。最后，我们遵循 DeepSeekMath (Shao et al., 2024) 采用组相对策略优化 (GRPO) 进一步使模型与人类偏好保持一致，并生成 DeepSeek-V2 Chat (RL)。


# 架构

![](img/Pasted%20image%2020250122113329.png)

总的来说，DeepSeek-V2 仍然采用 Transformer 架构（Vaswani 等人，2017），其中每个 Transformer 块由一个注意力模块和一个前馈网络（FFN）组成。然而，对于注意力模块和 FFN，我们设计并采用了创新的架构。为了引起注意，我们设计了MLA，它利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效的推理。对于 FFN，我们采用 DeepSeekMoE 架构（Dai et al., 2024），这是一种高性能 MoE 架构，能够以经济的成本训练强大的模型。 DeepSeek-V2 的架构如图 2 所示。

## Multi-Head Latent Attention

![](img/Pasted%20image%2020250122114057.png)

设 d 为embedding维度，$n_h$ 为注意力头的数量，$d_h$ 为每个头的维度，$h_t$ ∈ Rd 为注意力层第 t 个标记的注意力输入。$q_t, k_t, v_t ∈ R^{d_hn_h}$

MHA 需要为每个 token 缓存 $2n_hd_hl$ 元素。（l是层数）

![](img/Pasted%20image%2020250122114929.png)

![](img/Pasted%20image%2020250122114552.png)

MLA的核心是对key和value进行低秩联合压缩，以减少KV缓存

![](img/Pasted%20image%2020250122115535.png)

解释：
- **每个k_head附带有不同的信息，它将用这份独有的信息和对应的q_head进行attn的计算**。
- 当前我要存的K cache是4个k_head（图中深绿色框），**但如果我能从这4个k_head中抽取出1份共有的信息**，然后在做attn计算时，**每个head都用这1份共有的信息做计算**，那么我也只需存这1份共有信息作为K cache了。这样我就**把K cache从原来[num_heads](https://zhida.zhihu.com/search?content_id=252939704&content_type=Article&match_order=2&q=num_heads&zhida_source=entity) = 4变成num_heads = 1**，这不就能节省K cache了吗？
- 但是等等，**现在共有的k_head信息是抽取出来了，那么相异的k_head信息呢？**（**简单来说，就是由**WK**不同head部分学习到的相异信息**）。我们当然是希望k_head间相异的信息也能保留下来，那么该把它们保留至哪里呢？当你回顾attn_weights的计算公式时，一个想法在你脑中闪现：**q部分不是也有heads吗！我可以把每个k_head独有的信息转移到对应的q_head上吗！写成公式解释就是：**
![](img/Pasted%20image%2020250124160346.png)


为了减少训练期间的激活记忆，也对查询进行低秩压缩，即使它不能减少 KV 缓存

![](img/Pasted%20image%2020250122115743.png)

ROPE

![](img/Pasted%20image%2020250122120013.png)

我们在表 1 中展示了不同注意力机制中每个 token 的 KV 缓存的比较。MLA 仅需要少量的 KV 缓存，相当于只有 2.25 个组的 GQA，但可以实现比 MHA 更强的性能。

![](img/Pasted%20image%2020250122120132.png)

## 手撕MLA

![](img/Pasted%20image%2020250309130156.png)

![](img/Pasted%20image%2020250309130210.png)
![](img/Pasted%20image%2020250309130223.png)

```json
// config_671B
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
```

MLA的输入及配置参数参考如下：

```python
def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    """
    Forward pass for the Multi-Headed Attention Layer (MLA).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size=2, seq_len=1, dim=7168).
        start_pos (int): Starting position in the sequence for caching. (seq_len, d//2)
                         KV cache 起始填充位置以通过：推理过程中记录上一个预测的token生成KV填充至
                         KV cache lists的结束的位置end_pos计算得到。
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
        mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input.

    """
    pass

if __name__ == "__main__":
    args = Model671BArgs
    bs = 2
    seq_len = 1
    d = args.dim  # 7168
    x = torch.randn(bs, seq_len, d)

    attn_norm = RMSNorm(args.dim)
    x = attn_norm(x)

    start_pos = 0
    mask = None
    attn = MLA(args)

    freqs_cis = precompute_freqs_cis(args)
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    # x = x.to(torch.bfloat16)
    x = attn(x, start_pos, freqs_cis, mask=mask)
```

### Step1: 计算Query

对输入的第t个token，先做低秩压缩， 然后右乘升维映射矩阵恢复原始特征维度。Query分两部分：不带位置编码的 qtC 和带RoPE旋转位置编码的 qtR。参考公式 (6)～(8)，由于源代码中输入token的_shape=(batch_size, seq_len, dim)_为行向量。 因此，矩阵右乘更加符合习惯思维。下面我会改写此公式，并更新公式的计算顺序以及各个数学量描述，便于大家结合代码理解。

![](img/Pasted%20image%2020250309134138.png)

![](img/Pasted%20image%2020250309134219.png)

完整的代码实现如下所示：

```python
"""
x (torch.Tensor): Input tensor of shape (batch_size=1, seq_len=1, dim=7168).
self.wq_a.weight^T : 对应公式中的W^{DQ}
self.wq_b.weight^T : 对应公式中的W^{UQR}
ColumnParallelLinear 对Linear做了一层封装，适配分布式训练，切割输出特征，沿着列方向实现并行。
"""
self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # 1536 -> 128*192=24576
# b,s,d=7168 => b,s,d_c=1536 => bs,s,nh*(dh+dq')=128*192=24576
q = self.wq_b(self.q_norm(self.wq_a(x)))
q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)  # b,s,128,192
# (b, s, 128, 192) => (b, s, 128, 128)|(b, s, 128, 64)
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  

# 加上RoPE旋转位置编码得到带相对位置信息的Query(q_pe):
q_pe = apply_rotary_emb(
    q_pe, freqs_cis
)  #  (2, 1, 128, 64) | (1,32) => (2, 1, 128, 64)
```

MLA在给定attention层下，先降维再升维到给定维度的Query参数量计算如下：

_7168 * 1536 + 1536 * 16384 = 36175872_

MHA计算量则为： _7168 * 16384 = 117440512_

MHA参数量是MLA的3.2x，单层attention layer的参数量下降就挺明显的。参数量下降，训练推理过程的计算量和显存都会下降，训练过程其梯度也会下降，有效减小了推理成本并提升了模型的吞吐量。论文中其实给出了另一种加速视角：**减少训练过程中激活占用的显存。**

> For the attention queries, we also perform a low-rank compression, which can reduce the activation memory during training

### Step2：计算KV，更新KV Cache：

对输入的第t个token特征向量 ht ，先做低秩压缩，然后右乘升维映射矩阵恢复原始特征维度。

![](img/Pasted%20image%2020250309132320.png)
![](img/Pasted%20image%2020250309132408.png)

```python
kv = self.wkv_a(x)  # (2,1,7168) => (2,1,512+64=576) 
kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # (2,100,576) => (2,100,512) 和 (2,100,64)
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # (2,1,1,64)
# self.kv_cache预设一个大一点的全零张量(8,16384,512)缓存kv
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)  # (2,1,512) palce into (:2,0:1,512)
# self.pe_cache预设一个大一点的全零张量(8,16384,64) 缓存key的位置编码, 这里命名有点迷惑性, 不如叫:self.k_pe_cache?
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)  # (2,1,64) palce into (:2,0:1,64)
```

_**P.S.**_ MLA KVCache缓存了低秩压缩的 ctKV,ktR ，而不是升维后的Key和Value，前者在单层attention layer下大小为 dc （在DeepSeekV3-671B大模型中=512），后者缓存大小为 2nhdh （在DeepSeekV3-671B大模型中=128x(128+128)=32768）, 前者缓存成本减小了32768/512=64x。额外增加的缓存 ktR 为 dhR（在DeepSeekV3-671B大模型中=64 << 512，增加少量的缓存，但是，却带来相对位置编码信息） 。
### Step3：计算单层Attention

![](img/Pasted%20image%2020250309132615.png)

```python
# 先reshape，再split效果和上述公式一致
q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)  # b, s, 128, 192
q_nope, q_pe = torch.split(
    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
)  # (2,1,128,192) => (2,1,128,128),(2,1,128,64)

# 加上RoPE旋转位置编码得到带相对位置信息的Query(q_pe):
q_pe = apply_rotary_emb(
    q_pe, freqs_cis
)  #  (2, 1, 128, 64) | (1,32) => (2, 1, 128, 64)
```


$∑_{j=1}^tc_j^{KV}$ ：对应源代码中 self.kv_cache[:bsz,:end_pos] ;

```python
# 这里注意下，当前token计算的kv会经过RMSNorm后缓存到KVCache中
self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )  # 512 => 128*(128+128)=32768
kv = self.wkv_a(x)  # (2,1,7168) => (2,1,512+64=576)
kv, k_pe = torch.split(
    kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
)  # (2,1,576) => (2,1,512),(2,1,64)
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)  # (2,1,512) palce into (:2,0:1,512)
```

完整的代码如实现如下:

```python
self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )  # 512 => 128*(128+128)=32768
wkv_b = (
    self.wkv_b.weight
    if self.wkv_b.scale is None
    else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
)
wkv_b = wkv_b.view(
    self.n_local_heads, -1, self.kv_lora_rank
)  # (128*(128+128)=32768, 512) => (128, 256, 512)
q_nope = torch.einsum(
    "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
)  # (2,1,128,128) einsum (128,128,512) => (2,1,128,512)
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(
    kv
)  # (2,1,512) palce into (:2,0:1,512)
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(
    2
)  # (2,1,64)  palce into (:2,0:1,64)
# (2,1,128,512) einsum (2,1,512) => (2,1,128,1)
# (2,1,128,64) einsum (2,1,64) => (2,1,128,1)
scores = (
    torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
    + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
)
```

基于公式（2-6）（3-2），构建完整的attention公式如下：

![](img/Pasted%20image%2020250309132817.png)

大功告成，至此，更新后的公式和源代码可以一一对应了。

```python
# ot attention计算结果如下：
self.softmax_scale = self.qk_head_dim**-0.5
# 需要注意的是，如果推理的序列大于训练的序列长度，需要动态调整softmax_scale
if args.max_seq_len > args.original_seq_len:
    mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale

scores = (
    torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
    + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
) * self.softmax_scale

scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
x = torch.einsum(
    "bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos]
)  # (2,1,128,512)
x = torch.einsum(
    "bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :]
)  #  (2,1,128,512) (128,-128:,512) -> (2,1,128,128)
```

_**P.S.**_源代码中使用了大量的[PyTorch爱因斯坦求和](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.einsum.html%23torch-einsum)方法，给人直观感受是非常简洁。但如果没有熟练掌握它，可能未必能灵活使用。这部分也可以借用torch.bmm转换为传统的矩阵乘法，变成我们熟悉的味道。举个例子：_torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])_的等效计算方法见如下单元测试：

```python
import torch

b, s, h, c = 2, 1, 128, 512  # 示例维度
t = 1  # 目标KVCache序列长度

# 方法一：使用传统的矩阵乘法方法
q_nope = torch.randn(b, s, h, c)
kv_cache = torch.randn(b, t, c)
q_reshaped = q_nope.view(b, s * h, c)  # shape: (b, s*h, c)
kv_transposed = kv_cache.transpose(1, 2)  # shape: (b, c, t)
scores = torch.bmm(q_reshaped, kv_transposed)  # shape: (b, s*h, t)
scores = scores.view(b, s, h, t)  # shape: (b, s, h, t)

# 方法二：使用爱因斯坦求和方法
scores_einsum = torch.einsum("bshc,btc->bsht", q_nope, kv_cache)

# 检查两种方法的结果是否一致
print(torch.allclose(scores, scores_einsum, atol=1e-6))  # 应输出 True
```

### **备注：解耦Query和Key，单独做旋转位置编码：**

为什么需要接耦Query和Key？这个问题酱紫思考，如果我们不做解耦，直接加上旋转位置编码RoPE，会是什么结果呢？参考论文公式(10)，我们需要更新下等号右侧softmax内分子表达式和各个数学量描述，将投影矩阵左乘转换为矩阵右乘：

![](img/Pasted%20image%2020250309132954.png)
![](img/Pasted%20image%2020250309133236.png)

**_P.S._** 虽然，MLA的低秩压缩的KVcache有效的减少了缓存大小。但是，为了引入相对位置编码，又增加了部分的计算量。但是，Q/K/V的低秩压缩带来参数量的下降又弥补了上述缺失，从而有效的减小了训练（减小了梯度和激活大小）/推理过程的显存以及计算量。总体来说，MLA称的上是推理加速非常有效的一种注意力机制变种。
## DeepSeekMoE

对于 FFN，我们采用 DeepSeekMoE 架构（Dai 等人，2024）。 DeepSeekMoE有两个关键思想：将专家细分为更细的粒度，以实现更高的专家专业化和更准确的知识获取；以及隔离一些共享专家，以减轻路由专家之间的知识冗余。在激活和总专家参数数量相同的情况下，DeepSeekMoE 可以大幅优于 GShard（Lepikhin 等人，2021）等传统 MoE 架构。

![](img/Pasted%20image%2020250122135338.png)

我们在自动学习的路由策略中考虑了负载平衡。首先，负载不平衡会增加路由崩溃的风险（Shazeer et al., 2017），导致部分专家无法得到充分的培训和利用。其次，当采用专家并行时，不平衡的负载会降低计算效率。在DeepSeek-V2的训练过程中，我们设计了三种辅助损失，分别用于控制专家级负载平衡（LExpBal）、设备级负载平衡（LDevBal）和通信平衡（LCommBal）。

![](img/Pasted%20image%2020250122135753.png)

![](img/Pasted%20image%2020250122135806.png)

![](img/Pasted%20image%2020250122135817.png)

虽然平衡损失旨在鼓励平衡负载，但重要的是要承认它们不能保证严格的负载平衡。为了进一步减轻负载不平衡造成的计算浪费，我们在训练过程中引入了设备级令牌丢弃策略。该方法首先计算每个设备的平均计算预算，这意味着每个设备的容量因子相当于 1.0。然后，受到里克尔梅等人的启发。 （2021），我们在每台设备上丢弃亲和力分数最低的令牌，直到达到计算预算。此外，我们确保属于大约 10% 训练序列的 token 永远不会被丢弃。这样，我们就可以根据效率要求灵活决定是否在推理过程中丢弃token，始终保证训练和推理的一致性。

# 预训练

在保持与 DeepSeek 67B（DeepSeek-AI，2024）相同的数据处理阶段的同时，我们扩展了数据量并提高了数据质量。为了扩大我们的预训练语料库，我们探索互联网数据的潜力并优化我们的清理流程，从而恢复大量误删除的数据。此外，我们纳入了更多的中文数据，旨在更好地利用中文互联网上的语料库。除了数据量，我们还关注数据质量。我们利用各种来源的高质量数据丰富了我们的预训练语料库，同时改进了基于质量的过滤算法。改进后的算法保证了大量无益数据被剔除，而有价值的数据大部分被保留。此外，我们从预训练语料库中过滤掉有争议的内容，以减轻特定区域文化引入的数据偏差。附录 E 详细讨论了这种过滤策略的影响。

我们采用与 DeepSeek 67B 中使用的相同的分词器，它是基于字节级字节对编码（BBPE）算法构建的，词汇量为 100K。我们的标记化预训练语料库包含 8.1T 个标记，其中中文标记比英文标记多约 12%。

我们将 Transformer 层数设置为 60，隐藏维度设置为 5120。所有可学习参数均随机初始化，标准差为 0.006。在MLA中，我们将注意力头的数量nh设置为128，每个头的维度dh设置为128。KV压缩维度dc设置为512，查询压缩维度d'c设置为1536。对于解耦查询和密钥，我们将每个头的尺寸 dR h 设置为 64。 (2024)，我们用 MoE 层替换除第一层之外的所有 FFN。每个MoE层由2个共享专家和160个路由专家组成，其中每个专家的中间隐藏维度为1536。在路由专家中，每个代币将激活6个专家。此外，低秩压缩和细粒度专家分割也会影响层的输出规模。因此，在实践中，我们在压缩潜在向量之后采用额外的 RMS Norm 层，并在宽度瓶颈处乘以额外的缩放因子（即压缩的潜在向量和路由专家的中间隐藏状态）以确保稳定的训练。在此配置下，DeepSeek-V2总共包含236B个参数，其中每个令牌激活21B个参数。

我们采用 AdamW 优化器（Loshchilov 和 Hutter，2017），超参数设置为 β1 = 0.9，β2 = 0.95，weight_decay = 0.1。使用预热和逐步衰减策略来安排学习速率（DeepSeek-AI，2024）。最初，学习率在前 2K 步中从 0 线性增加到最大值。随后，在训练大约 6​​0% 的标记后，将学习率乘以 0.316，在训练大约 90% 的标记后，再次将学习率乘以 0.316。最大学习率设置为2.4×10−4，梯度裁剪范数设置为1.0。我们还使用批量大小调度策略，在前 225B 个令牌的训练中，批量大小逐渐从 2304 增加到 9216，然后在剩余的训练中保持 9216。我们将最大序列长度设置为 4K，并在​​ 8.1T 令牌上训练 DeepSeek-V2。我们利用管道并行性将模型的不同层部署在不同的设备上，对于每一层，路由专家将统一部署在 8 个设备上（D = 8）。对于设备限制路由，每个令牌最多将发送到 3 个设备（M = 3）。对于余额损失，我们将α1设置为0.003，α2设置为0.05，α3设置为0.02。我们在训练期间采用令牌丢弃策略来加速，但不丢弃任何令牌进行评估。

在 DeepSeek-V2 的初始预训练之后，我们使用 YaRN (Peng et al., 2023) 将默认上下文窗口长度从 4K 扩展到 128K。 YaRN 专门应用于解耦共享密钥 kR t，因为它负责承载 RoPE (Su et al., 2024)。对于 YaRN，我们将尺度 s 设置为 40，α 设置为 1，β 设置为 32，目标最大上下文长度设置为 160K。在这些设置下，我们可以预期模型对于 128K 的上下文长度有良好的响应。与原始 YaRN 略有不同，由于我们独特的注意力机制，我们调整长度缩放因子来调节注意力熵。因子 √ t 的计算公式为 √ t = 0.0707 ln s + 1，旨在最小化困惑度。

我们还对模型进行了 1000 个步骤的训练，序列长度为 32K，批量大小为 576 个序列。尽管训练仅在 32K 的序列长度下进行，但该模型在以 128K 的上下文长度进行评估时仍然表现出稳健的性能。如图 4 所示，“大海捞针”(NIAH) 测试的结果表明 DeepSeek-V2 在高达 128K 的所有上下文窗口长度上都表现良好。

# Alignment

基于我们之前的研究（DeepSeek-AI，2024），我们整理了指令调整数据集以包含 150 万个实例，其中 120 万个用于帮助的实例和 30 万个用于安全的实例。与初始版本相比，我们提高了数据质量，以减轻幻觉反应并提高写作水平。我们对 DeepSeek-V2 进行了 2 个 epoch 的微调，学习率设置为 5 × 10−6。

为了节省强化学习的训练成本，我们采用组相对策略优化（GRPO）（Shao et al., 2024），它放弃了通常与策略模型大小相同的批评家模型，并以小组成绩估计基线。具体来说，对于每个问题q，GRPO从旧策略πθold中采样一组输出{o1，o2，···，oG}，然后通过最大化以下目标来优化策略模型πθ

![](img/Pasted%20image%2020250122141056.png)

在我们的初步实验中，我们发现对推理数据（例如代码和数学提示）的强化学习训练表现出与一般数据训练不同的独特特征。例如，我们模型的数学和编码能力可以在较长时期的训练步骤中不断提高。因此，我们采用两阶段强化学习训练策略，首先进行推理对齐，然后进行人类偏好对齐。在第一个推理对齐阶段，我们为代码和数学推理任务训练奖励模型 RMreasoning，并根据 RMreasoning 的反馈优化策略模型

在第二个人类偏好调整阶段，我们采用多重奖励框架，该框架从有用奖励模型 RMhelp ful、安全奖励模型 RMsa f ety 和基于规则的奖励模型 RMrule 获得奖励。响应oi的最终奖励是

![](img/Pasted%20image%2020250122141318.png)

为了获得在强化学习训练中发挥关键作用的可靠奖励模型，我们仔细收集偏好数据，并精心进行质量过滤和比例调整。我们根据编译器反馈获得代码偏好数据，并根据真实标签获得数学偏好数据。对于奖励模型训练，我们使用 DeepSeek-V2 Chat (SFT) 初始化奖励模型，并使用逐点或成对损失对其进行训练。在我们的实验中，我们观察到强化学习训练可以充分挖掘和激活我们模型的潜力，使其能够从可能的响应中选择正确且令人满意的答案。



# 讨论

围绕大型 SFT 语料库必要性的讨论一直是激烈争论的话题。之前的工作（Young et al., 2024；Zhou et al., 2024）认为少于 10K 个 SFT 数据实例就足以产生令人满意的结果。然而，在我们的实验中，我们观察到如果我们使用的实例少于 10K，IFEval 基准的性能会显着下降。一种可能的解释是，语言模型需要一定量的数据来开发特定技能。尽管所需的数据量可能会随着模型大小的增加而减少，但不能完全消除。我们的观察强调，迫切需要足够的数据来为法学硕士配备所需的能力。此外，SFT 数据的质量也至关重要，特别是对于涉及写作或开放式问题的任务。

在人类偏好调整过程中，我们观察到开放式生成基准的性能显着提高，无论是人工智能还是人类评估者的评分。然而，我们也注意到一种“对齐税”现象（Ouyang et al., 2022），即对齐过程会对某些标准基准（例如 BBH）的性能产生负面影响。为了减轻对齐税，在强化学习阶段，我们在数据处理和改进训练策略方面做出了巨大努力，最终在标准基准和开放基准的性能之间实现了可容忍的权衡。探索如何使模型与人类偏好保持一致，而无需损害其总体性能为未来的研究提供了一个有价值的方向。

在我们的偏好对齐实验中，我们发现在线方法明显优于离线方法。因此，我们投入了巨大的努力来实现在线 RL 框架来对齐 DeepSeek-V2。关于线上或线下偏好调整的结论在不同的背景下可能会有所不同，我们为未来的工作保留对它们之间更彻底的比较和分析。


# 参考资料

[再读MLA，还有多少细节是你不知道的](https://zhuanlan.zhihu.com/p/19585986234)

[deepseek技术解读(1)-彻底理解MLA（Multi-Head Latent Attention）](https://zhuanlan.zhihu.com/p/16730036197)

[手撕DeepSeek-MLA-多头潜在注意力机制](https://zhuanlan.zhihu.com/p/23062701108)