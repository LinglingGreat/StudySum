---
title: MQA&GQA
created: 2024-07-18
tags:
  - attention
---
## MHA&MQA&GQA

MQA 提出时间挺早的，是 **Noam Shazeer** 这位谷歌老炮 19 年提出的。而 Noam 也是 Transformer 结构提出者之一，现在也就理所当然地早就不在 Google，是 Character.ai 的合伙人。

![](img/Pasted%20image%2020240718163122.png)

**MHA(Multi-Head Attention)**，QKV 三部分有相同数量的头，且一一对应。每次做 Attention，head1 的 QKV 就做好自己运算就可以，输出时各个头加起来就行。

而 MQA 则是，让 **Q 仍然保持原来的头数**，但 **K 和 V 只有一个头**，相当于所有的 Q 头共享一组 K 和 V 头，所以叫做 Multi-Query 了。实现改变了会不会影响效果呢？确实会影响但相对它能带来的收益，性能的些微降低是可以接受的。

能带来多大的收益呢，实验发现一般能提高 30%-40% 的吞吐。

收益主要就是由降低了 KV cache 带来的。实际上 MQA 运算量和 MHA 是差不多的，可理解为**读取一组 KV 头**之后，**给所有 Q 头用**，但因为之前提到的内存和计算的不对称，所以是有利的。

而 GQA 呢，是 MHA 和 MQA 的折衷方案，既不想损失性能太多，又想获得 MQA 带来的推理加速好处。具体思想是，不是所有 Q 头共享一组 KV，而是**分组一定头数 Q 共享一组 KV**，比如上面图片就是两组 Q 共享一组 KV。

LLAMA2 中给出了效果对比，可以看到相比起 MQA，GQA的指标看起来还是要好些的。

![](https://pic3.zhimg.com/v2-69c2cc88b213a65d61cee7a9f31d844e_b.jpg)

同时在推理上的加速还和 MQA 类似：

![](https://pic4.zhimg.com/v2-f12de54f6293fb71af044fcb4ec1809f_b.jpg)

MQA 和 GQA 形式在推理加速方面，主要是通过两方面来完成：

1. **降低了从内存中读取的数据量**，所以也就减少了计算单元等待时间，提高了计算利用率；
2. KV cache 变小了 head_num 倍，也就是显存中需要保存的 tensor 变小了，**空出来空间就可以加大 batch size**，从而又能提高利用率。

如果要用 MQA 和 GQA，可以是从头训练的时候就加上，也可以像 GQA 论文里面一样，用已有的开源模型，挑一些头取个 mean 用来初始化 MQA 或 GQA 继续训练一段时间。


# GQA

Paper：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints  

Abs：https://arxiv.org/abs/2305.13245

grouped-query attention 指出，Multi-Query Attention提高了推理速度的同时，却可能极大地降低回复质量。因此根据上图，GQA 在推理速度和质量之间作了权衡。

Mistral，Llama2 的部分模型使用 GQA 时，采用的 kv head 数量似乎都是 8。

> [为什么现在大家都在用 MQA 和 GQA？](https://zhuanlan.zhihu.com/p/647130255)文中提到 MQA 和 GQA 能获得巨大加速的一个点在于：GPU 内存强的限制。由于 MQA 和 GQA 都降低了内存中数据的读取量，减少了计算单元的等待时间，因此推理速度的提高比想象中的要快更多。

## 实验

以下为 GQA 文中的实验结果，值得注意的是论文中使用原 MHA checkpoint 转换为 GQA 权重后，还进行了额外的预训练：

![](img/Pasted%20image%2020240801114005.png)


## 代码实现

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math
import torch.nn.functional as F
```

定义参数. n_kv_heads如果等于1，那么就是MQA，如果大于1小于n_heads，那么就是GQA。

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
args = ModelArgs()
```

Attention实现，输入维度（2，32，4096），输出的q是（2，32，4096），k是（2，32，1024），v是（2，32，1024）

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False,)            
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False,)

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Thereby, we are doing a “grouped” attention, because 4 queries get grouped to work a single key & value pair.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

X = torch.randn(2, 32, 4096)
attn = Attention(args)
attn(X).shape
# torch.Size([2, 32, 4096])
```


# 参考资料


[In this post, we take a deep dive into the architectural components of Gemma 2 such as Grouped Query Attention, Sliding Window Attention, RoPE Embeddings, Logit soft-capping & Model-merging!](https://amaarora.github.io/posts/2024-07-07%20Gemma.html)

[为什么现在大家都在用 MQA 和 GQA？](https://mp.weixin.qq.com/s/_4OxoRLxhOcjGf0Q4Tvp2Q)



