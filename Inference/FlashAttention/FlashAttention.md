FlashAttention的核心原理是通过将输入**分块**并在每个块上执行注意力操作，从而减少对高带宽内存（HBM）的读写操作。具体而言，FlashAttention使用平铺和重计算等经典技术，将输入块从HBM加载到SRAM（快速缓存），在SRAM上执行注意力操作，并将结果更新回HBM。FlashAttention减少了内存读写量，从而实现了**2-4倍**的时钟时间加速。

![](img/Pasted%20image%2020240424194314.png)

传统Attention，每次完整的矩阵运算的复杂度为O(n^2)

![](img/Pasted%20image%2020240424194349.png)


## 参考资料

[Flash Attention原理详解(含代码讲解) - 知乎](https://zhuanlan.zhihu.com/p/676655352)

[flash attention V1 V2 V3 V4 如何加速 attention - 知乎](https://zhuanlan.zhihu.com/p/685020608)

[[FlashAttention系列][2w字]🔥原理详解: 从Online-Softmax到FlashAttention-1/2/FlashDecoding，再到FlashDecoding++ - 知乎](https://zhuanlan.zhihu.com/p/668888063)

