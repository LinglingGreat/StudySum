---
title: SWA
created: 2024-07-18
tags:
  - attention
---
sliding window attention

![](img/Pasted%20image%2020240718165301.png)

In the traditional sense, $Attention(𝑄,𝐾,𝑉)=softmax(𝑄𝐾^𝑇/sqrt(𝑑_𝑘))𝑉$

Each token in the Query vector 𝑄 can attend to all tokens in the Key vector 𝐾.

But, this leads to a computational complexity of 𝑂(𝑛^2). As a result, memory requirements grow by a factor of 𝑛^2 for a sequence of length 𝑛.

This limits the traditional Transformer architecture from having long context length. The solution is to use Sliding window attention where each token in the Query vector 𝑄 only attends to it’s neighbouring tokens with an overlap of window length 𝑤.

So, a token at position 𝑖 in 𝑄, can attend to tokens in range (𝑖−𝑤,𝑖+𝑤) in 𝐾.


𝑄.𝐾^𝑇 using Einstein summation is as easy as doing:

```python
import torch 
q = torch.arange(1, 9).reshape(4,2)
k = torch.arange(1, 9).reshape(4,2)
out = torch.einsum('xd,yd->xy', q, k)
out.shape.  # torch.Size([4, 4])
```


sliding window attention

```python
q = torch.randn(1, 8, 768)
k = torch.randn(1, 8, 768)

def _chunk(hidden_states, window_overlap):
    """convert into overlapping chunks. Chunk size = 2w, overlap = w"""
    chunk_size = [
        hidden_states.size(0), #bs
        torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1, #n_chunks
        window_overlap * 2,
        hidden_states.size(2),
    ]

    overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
    for chunk in range(chunk_size[1]):
        overlapping_chunks[:, chunk, :, :] = hidden_states[
            :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
        ]
    return overlapping_chunks

query = _chunk(q, window_overlap=2)
key   = _chunk(k, window_overlap=2)
query.shape, key.shape
# (torch.Size([1, 3, 4, 768]), torch.Size([1, 3, 4, 768]))

diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key)) 
diagonal_chunked_attention_scores.shape
# torch.Size([1, 3, 4, 4])
```

![](img/Pasted%20image%2020240718170427.png)


## 参考资料

[Sliding Window Attention](https://amaarora.github.io/posts/2024-07-04%20SWA.html)

