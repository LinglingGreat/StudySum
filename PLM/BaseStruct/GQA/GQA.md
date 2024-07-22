---
title: GQA
created: 2024-07-18
tags:
  - attention
---
![](img/Pasted%20image%2020240718163122.png)

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


## 参考资料


[In this post, we take a deep dive into the architectural components of Gemma 2 such as Grouped Query Attention, Sliding Window Attention, RoPE Embeddings, Logit soft-capping & Model-merging!](https://amaarora.github.io/posts/2024-07-07%20Gemma.html)

