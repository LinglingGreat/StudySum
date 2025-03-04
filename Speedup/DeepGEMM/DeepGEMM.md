---
title: DeepGEMM
created: 2025-03-03
tags:
  - deepseek
---
[GitHub - deepseek-ai/DeepGEMM: DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling](https://github.com/deepseek-ai/DeepGEMM)

DeepGEMM，是一个矩阵计算库，核心代码只有 300 多行，专注提升大模型**矩阵乘法的**计算效率。仅 300 行代码的 FP8 矩阵乘法库，在 Hopper GPU 上实现了 **1358 TFLOPS 峰值计算性能**，相比 CUTLASS 等主流库提速最高达 **2.7 倍**。适用于训练和推理阶段。

大模型的每一层神经网络本质都是**矩阵变换**，不管是输入的是文本还是图像，都是通过权重矩阵变换来实现特征提取的，比如大家熟知的 Transformer，自注意力机制就是通过 Query、Key、Value 三个矩阵的乘法完成语义关联计算。矩阵乘法占了运行时长的 45%-60%。

直白点说，千亿参数规模的模型，90% 以上的计算量来自矩阵乘法。所以，矩阵乘法的效率直接决定了模型的推理和训练速度。

而且实际过程中，还会因为硬件、结构，面对很多问题。

比如，模型结构各异，矩阵大小差异性就会非常大，不同的长度序列具有不同的 shape、不同的 batchsize，实际矩阵乘法有数百个形状。矩阵太大，没办法一次性放入到寄存器或内存缓存里，还要把矩阵分解成大小合适的 block 或者 tiles，就是为了最大限度地利用内存显存。

所以，GPU/TPU 的架构设计，都是高度针对矩阵乘法去优化的，以 NVIDIA GPU 为例，其张量核心（Tensor Core）就是专门加速矩阵乘累加运算的。还有专门的 GPU 加速库 cuBLAS。

核心代码只有 300 行左右，最高可达 1358 TFLOPS，达到了超越大多数人工专家调优。

先说一下缺点，这次更新依旧是依赖硬件的，仅支持英伟达最新 Hopper 架构显卡 H800。通俗讲，就是高端显卡才能用，不兼容老型号。

它特意说明了一个地方，就是 **JIT，**  Just-In-Time 也就是**即时编译，**在运行时会自动生成需要的代码，不需要配置环境，能快速部署。所以相当友好。

我们都知道大模型一般用 FP8 量化瘦身，目的是提速，虽然读取速度快、存储空间也小，但是低精度带来的问题就是有误差累积，大模型中的矩阵乘法可能涉及数百万次乘加，误差会被指数级放大。

DeepGEMM 巧妙地解决了这个问题，通过一种叫做“**两级累加**”的技术，保证了计算结果的准确性。

先做高精度做乘法和累加，当高精度累加结果超过 FP8 范围时，再转回 FP8 存储。实现在速度和精度之间的平衡。

DeepGEMM 不仅支持普通稠密矩阵乘法，还支持混合专家模型 (MoE) 。

为什么这么说呢？

简单地理解，普通的 AI 通常是稠密矩阵，MOE 因为不是全部激活，所以还会有稀疏矩阵，一般会分组计算，因此 MoE 中的矩阵乘法更加复杂。

官方测试的结果，在 H800 上使用 NVCC 12.8 测试了 DeepSeek-V3/R1 推理中可能使用的所有形状（包括预填充和解码），能获得 1.1 倍~2.7 倍左右的速度提升（与官方 CUTLASS 实现相比）

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qHaz4BTMszXGSnibOSEaAdoshN2W8icSMGiaUzyNOYUeXhLmd2PPZlMrWVVvBZ6czvIM9mckVLomVKBg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

在MoE的两种GEMM场景中：

**a. Contiguous layout：**适合训练或推理时把不同Expert的Token数据按行连续拼接，这样可以利用DeepGEMM的分组功能一次性进行多组运算。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qHaz4BTMszXGSnibOSEaAdosBHDkcWz9NnyFryJs2nmUFhfZrXz6xXgK0Hia1rc4ib87ic0OiagsW0S3zw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

**b. Masked layout：适合推理阶段**，当Expert间的Token数大多并不均匀或实时变化时，可通过一个mask来只计算有效行，从而减少无用算力浪费。DeepGEMM针对这种分组也提供了高效实现。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qHaz4BTMszXGSnibOSEaAdoseTQ0n80Q5009OGocV1TiaZT5Y7Kia9Yexg6ZcFKstAZ1MJWZ0qVicR1Dw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

