---
title: FlashMLA
created: 2025-03-03
tags:
  - deepseek
  - 推理加速
---
[GitHub - deepseek-ai/FlashMLA: FlashMLA: Efficient MLA decoding kernels](https://github.com/deepseek-ai/FlashMLA/tree/main)

FlashMLA 是一个针对 **Hopper GPU 优化的高效MLA 解码内核**，专门用于处理**可变长度序列**的服务。实现了**3000GB/s 的显存带宽利用率。** 适用于推理，能够减少延迟和提高吞吐量。

与标准的注意力机制相比，MLA 的 KV 缓存大小**减少了接近九成**。

这样就大大减少了需要缓存的键值对数量，**让 kv 缓存的存储成本被压到了极低，从而大大降低内存占用。**

在推理时，通过缓存之前计算的键值对，避免重复计算，从而节省计算资源和时间。比如 DeepSeek-V3 在长文本生成任务中推理速度提高了 1.5 倍。

也是凭借这个技术，**DeepSeek-V2 大名鼎鼎的磁盘缓存技术**，kv 向量从存放在显存/内存变成可以存放在磁盘。当缓存命中的时候，DeepSeek 只收取十分之一的价格。

因为 MLA 和传统的注意力机制不同，该项目在 MLA 结构的基础上参考了 **FlashAttention** 的处理逻辑进行优化。

它让 MLA 结构的模型也能同样通过 tilling 分片在显卡的 SRAM 上进行快速计算，从而达到推理加速的效果。

传统注意力面对可变序列的时候，往往会因为输入文本的长度不同，面临**显存碎片和访存延迟**的问题。

**FLashMLA**，另一个重要优化的点就是针对不同的文本输入，即可变长序列，做了优化。

可变序列（**variable-length**）主要是指在同一个批次（batch）中，不同输入样本可以拥有**不同的序列长度**，而不需要将所有序列统一填充到相同的长度。

FlashMLA 做了和 PagedAttention 思想类似的工作，进行了分页 KV 缓存管理，它实现了基于 64-block 粒度的分页 KV 缓存，极大地缓解了内存访问瓶颈。

同时，它设计了**双缓冲预加载机制**，在计算当前块的时候，会异步加载下一个块到共享内存，让显存访问和计算过程同步进行，减小延迟开销。

H800 上可以达到 3000 GB/s 的内存带宽和 580 TFLOPS 的计算性能，是逼近什么程度呢？

H800 内存带宽：最高约 3.35 TB/s (使用 HBM3 内存)，**确切说 3000+ GB/s 已接近现有商业化产品的极限。**