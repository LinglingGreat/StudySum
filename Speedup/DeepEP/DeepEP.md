---
title: DeepEP
created: 2025-03-03
tags:
  - deepseek
  - 专家模型
  - 推理加速
  - 训练加速
---
[GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library](https://github.com/deepseek-ai/DeepEP)

DeepEP，是首个用于 MoE 模型训练和推理的开源 EP 通信库，用于训练和推理的高吞吐量和低延迟

- 实现高效的 all-to-all 通信
    
- 提供**高吞吐**（NVLink + RDMA）与**低延迟**（纯 RDMA）两套通信内核，兼顾大批量训练与实时推理场景。
    
- 支持 NVLink 和 RDMA 的**节点内** / **跨节点**通信。
    
- 提供 SM 数量控制接口，可在计算与通信之间灵活分配 GPU 资源。
    
- 集成可以重叠通信和计算的 hook 机制，允许在解码时后台并行接收数据，不占用任何 SM 资源。
    
- 原生 FP8 调度支持

**专家并行**(Expert Parallelism，**EP**)是一种并行计算的方式，主要用在混合专家模型(Mixture of Experts, MoE)中。

- 将不同的"专家"(模型的子网络)分配到不同的计算设备(如 GPU)上
    
- 根据输入数据的特点，选择合适的专家来处理
    
- 多个专家可以同时并行工作，提高整体计算效率

MoE 模型在每次推理时，并不是所有专家都参与计算，而是根据输入数据的需求，只激活一部分。

比如 DeepSeek-R1，它的实际参数大小为 671B（实际还有 14B 的 MTP 投机解码模块），在每次推理的时候，只激活 37B 的参数量，256 个专家里激活 8 个。

在这个过程中，每个设备会根据路由规则，将自身的数据发送到相应专家所在的设备上，然后等待专家完成计算后再将结果返回到原设备。

这个过程中，每两个专家之间需要进行通信来同步各自的计算结果，这个通信过程被称为**All-****to****-All 通信**。

打个比方——

想象一个横跨多个城市的巨型物流中心，在每一个城市就是一个节点 Node，每一个城市里有多个仓库（GPU），而每个仓库里都有物流专家负责接收包裹。

物流中心每天要处理巨量的包裹（token），每个包裹需要同时派发给指定的 8 个物流专家（top-8 experts）。

专家通常分布在**不同的仓库（ GPU）** 上，我们需要依赖高速通信来完成包裹在“物流中心”以及“专家”之间的流转。

此时，

- **NVLink 相当于“城内高速路”**  
    专门负责同一个城市内不同仓库的联系，也就是**同一节点内 GPU-GPU 之间**的高速通信，带宽可达百 GB/s 级，极大提升**单机多卡**之间的数据交换效率。
    
- **RDMA（Remote Direct Memory Access）相当于“跨城高速公路”**  
    用于跨城市通信，也就是跨节点通信，直接让一台服务器的 GPU 与另一台服务器的 GPU 之间进行远程读写，跳过 CPU，有效避免多层网络的额外延迟，带宽通常在**几十 GB/s** 级。
    

在 MoE（混合专家）模型中，专家并行（Expert Parallelism）需要不停地对 token 进行分发（dispatch）和聚合（combine）。

即每一次输入的 token 都需要发送给不同的专家，专家处理结束后再将各自的结果组合在一起。

这就像大型物流中心“分拣包裹”，同时要在站内（NVLink）和跨节点（RDMA）两大“运输通道”上来回调配。

当 Token 数量暴增时，如果通信方案不够高效，就会像人工分拣一样堵成“长龙”，让价值百万的 H800 集群陷入“等传输”的尴尬局面。

所以，DeepEP 就是让 H800 高效通信的解决方案。

那么 DeepEP 具体做了哪些改动呢？

它设计了两种内核，分别应对高吞吐和低延迟两种场景，前者用于训练和快速处理用户输入的文本，后者用于加速大模型一个字一个字生成的速度。

### **高吞吐内核优化**

针对高吞吐场景，DeepEP 提供了一组同时适用于训练和推理预填充任务的通用内核。

它可以直接将数据从 NVLink 域转发到 RDMA 域，也就是可以直接“跨城高速公路”转“城内高速路”，提供非常高的吞吐量。

已知 NVLink 的最高带宽是 160GB/s，DeepEP 在实现专家并行时，实测带宽最高可以达到 153GB/s！无限接近极限值。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qFRxZHFYpwgoUp41PBG5479qALxNa8UvknDXkXcK9meu0ia9TxRYmibcZjRiaumBMbsZaWheAjFMOnvA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

### **低延迟内核**

**而在对延迟敏感的大模型解码环节，DeepEP 提供了另外一组低延迟内核，它只使用 RDMA（远程直接显存访问）以最小化延迟。**

可以看到，完全依照 RDMA 后的延迟在各个专家并行的程度下均达到了微秒级，最高达到了 46GB/s 的带宽（理论极限 50GB/s）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qFRxZHFYpwgoUp41PBG5479EzYwbS37S8JR1ElaC8HrYYYJznhCk5y6GicUoTo29wMylz1XvPbScFw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

这俩恐怖的数字，估计连英伟达都没想到自己的卡还能这么用。

由于在实际分发阶段，往往无法提前知道当前这个阶段究竟会接收多少 tokens。

因此，这里会涉及一个隐式的 CPU 等待 GPU 接收计数信号的过程。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qFRxZHFYpwgoUp41PBG5479ibDsiaw0UKzNz613hPMoc4iaicIOP3ryTZrpFlALedPU4nAGSR01DjQYxQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

图中可以看到 CPU 端先发起相应的操作，然后在等待状态下监听 GPU 的进度。

当 GPU 收到分发或合并指令并完成相应的工作后，会将结果或状态告知 CPU。CPU 收到信号后，再启动下一阶段的调度和计算。

通过这种显式和隐式的等待-通知机制，能够保证更高效的并行与资源利用。

针对该内核还专门引入了一种**基于hook的通信-计算重叠方法**，不会占用任何流式多处理器（SM）资源。

在 GPU 内部，还有一个重要概念：流式多处理器（Streaming Multiprocessor，SM）。

它是 GPU 的基本计算单元，每个 SM 都能并行运行成百上千个线程，为模型运算提供惊人的吞吐力。然而，如果通信速度无法跟上 Token 和专家并行度的需求，SM 等不到数据就会“闲置”，从而导致 GPU 资源浪费。

正因如此，如何在保证大规模分发 / 聚合的同时，让 SM 最大化利用，成为 MoE 并行实践中的一大挑战。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qFRxZHFYpwgoUp41PBG5479rfkZPRTu4uJKkRJmmEibrORVXtWicBP2oDBfjNaAJZqiaJs6wVssnYGNQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

DeepSeek 在用 NVIDIA 的 H800 GPU 训练 V3 时，对 GPU 的核心计算单元（SM，流多处理器）进行了定制化调整,他们把 132 个 SM 中的 20 个专门用来处理服务器之间的**通信任务**，而不是计算任务。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qFRxZHFYpwgoUp41PBG5479iaYrmHJCUoqFDxd2707o3Qh3yJxNQzNj8iabNnNWJrQV9kFkMttHzCCA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

它做到了以下几点：

1. **降低通信开销**：RDMA 在后台传输数据，减少了因数据分发和收集带来的阻塞或等待。
    
2. **释放计算资源**：通信不占用 GPU SM，可让更多计算内核专注于模型的前向或后向计算过程。
    
3. **双批次（或多批次）重叠**：利用接收钩子进一步将不同批次的计算与通信交错执行，在不干扰计算的情况下完成数据传输，从而提升整体吞吐量。
    
4. **灵活适配**：可根据实际负载大小或场景需求，调整通信与计算的重叠比例，从而获得更好的性能表现。
    

# PTX 指令挖掘

最后，DeepEp 发现并使用了一个未记录在英伟达文档之中的**PTX 指令**：

> _ld.global.nc.L1::no_allocate.L2::256B_

虽然此指令会导致未定义行为：使用非一致性只读 PTX 修饰符.nc 访问不稳定 GPU 内存。

但是在 Hopper 架构上，DeepEp 使用_._

> _L1::no_allocate_

DeepEP 强调这条指令在 Hopper 架构（NVIDIA 最新的 GPU 架构之一）上经过了测试，保证了正确性，并且性能更好。

顺便科普一下——

> - **PTX (Parallel Thread Execution)**：是 NVIDIA 为其 GPU 设计的一种低级虚拟指令集架构（ISA）。它类似于 CPU 的汇编语言，但更接近硬件。开发者可以直接编写 PTX 代码，或者通过 CUDA 编译器（如 nvcc）将 CUDA C/C++ 代码编译成 PTX 代码。
>     

这意味着什么，给 NVIDIA 的护城河开了个口子，并非完全“黑盒”，有可能通过逆向工程或其他手段搞点事情出来。

