
在大模型训练的时代，我们经常听到“DeepSpeed”“ZeRO”“Stage 3”“模型并行”“数据并行”等概念。它们看似相近，却代表着不同层次的分布式训练思想。很多人第一次听到 ZeRO Stage 3 时，会产生一个疑惑：

> 它把参数都分片到不同 GPU 上，那这不就是模型并行吗？

本文将系统解释：

1. 为什么我们需要 DeepSpeed；
    
2. DeepSpeed 的核心思想与 ZeRO 的分级机制；
    
3. 模型并行与数据并行的区别；
    
4. 最重要的——**为什么 ZeRO Stage 3 虽然分片了模型参数，但本质上仍然是数据并行。**
    

---

## 一、为什么我们需要 DeepSpeed

在训练 GPT、BERT 这类大规模 Transformer 模型时，我们面临三大瓶颈：

1. **显存限制（Memory Bottleneck）**：模型参数、优化器状态、激活值、梯度都会占用大量显存。单卡 24GB/48GB 显存远远不够。
    
2. **计算效率（Compute Efficiency）**：即便能放下模型，也要高效利用 GPU 算力，避免通信和等待浪费。
    
3. **通信开销（Communication Overhead）**：多 GPU 训练需要不断同步参数与梯度，通信代价常成为性能瓶颈。
    

传统的 **Data Parallel（数据并行）** 方案，如 PyTorch 的 `DistributedDataParallel (DDP)` 或 Horovod，只解决了**多数据样本的分布式训练**问题，却仍然复制了完整的模型副本，导致显存浪费严重。

于是，微软推出了 **DeepSpeed** —— 一个面向大规模模型训练的系统级优化框架，目标是：

> 让百亿、千亿级参数模型在有限 GPU 上也能高效训练。

---

## 二、DeepSpeed 是什么

DeepSpeed 是微软推出的一整套大规模模型训练系统，它并不仅仅是一个“并行库”，而是一套完整的 **训练引擎（DeepSpeed Engine）**，集成了多个组件：

- **ZeRO（Zero Redundancy Optimizer）**：核心分布式优化机制；
    
- **DeepSpeed Inference**：推理加速；
    
- **DeepSpeed MoE**：Mixture of Experts 模型支持；
    
- **DeepSpeed Compression**：量化与剪枝；
    
- **DeepSpeed Checkpointing / Offload**：显存与存储优化。
    

其中最核心、最具影响力的部分就是 ZeRO，它几乎成为所有超大模型训练的基础设施。

---

## 三、并行策略全景图：模型并行 vs 数据并行

要理解 ZeRO，首先要清楚分布式训练的两大基本思路。

### 1. 数据并行（Data Parallelism）

- 每个 GPU 拥有一份完整模型副本；
    
- 不同 GPU 处理不同的数据样本；
    
- 每个 GPU 独立计算梯度，最后进行梯度平均（all-reduce），每个GPU再独立进行参数更新。
    

优点：实现简单，扩展性强。  
缺点：每个 GPU 都存完整的模型和优化器状态，显存浪费大。

### 2. 模型并行（Model Parallelism）

- 把模型切开，分布在多个 GPU 上；
    
- 同一个样本的前向/反向传播需要跨 GPU 执行。
    

常见的模型并行有：

- **Tensor Parallelism（张量并行）**：同一层的矩阵运算被多个 GPU 分担（如 Megatron-LM）。
    
- **Pipeline Parallelism（流水线并行）**：不同 GPU 存不同层，样本沿层传播（如 GPipe、PipeDream）。
    

优点：突破单卡显存限制。  
缺点：通信复杂、依赖层结构、调度困难。

---

## 四、ZeRO 的核心思想：减少冗余，精细分片

在传统数据并行中，每个 GPU 都会复制模型参数、梯度、优化器状态三份信息，也就是说每个 GPU 存了三份几乎一样的东西！

ZeRO（Zero Redundancy Optimizer）的核心思想是：

> 把这三部分状态分片（shard）到不同 GPU 上，让每个 GPU 只存一部分，而不是完整副本。

它分为三个阶段：

| 阶段          | 分片内容    | 显存节省      | 特点              |
| ----------- | ------- | --------- | --------------- |
| **Stage 1** | 分片优化器状态 | 减少约 4× 显存 | 通信量和传统数据并行相同    |
| **Stage 2** | 额外分片梯度  | 减少约 8× 显存 | 通信量和传统数据并行相同    |
| **Stage 3** | 再分片参数本身 | 可训练超大模型   | 需要动态参数交换，通信量有增加 |

在 Stage 3 中，每个 GPU 仅存一部分参数分片。  
当执行前向传播时，DeepSpeed 会**动态地广播需要的参数分片**到计算 GPU 上；  
计算结束后，参数再被释放或重新分配。

---

## 五、为什么 ZeRO Stage 3 仍然是数据并行

这是本文的核心部分。

### 1. 数据并行的定义核心：

> 每个 GPU 处理不同的数据样本，模型逻辑上完整，只是状态分布在不同设备上。

### 2. 模型并行的定义核心：

> 多个 GPU 共同处理同一个样本的前向或反向传播（例如一层被拆到多卡上）。

### 3. ZeRO Stage 3 的实际情况：

Zero Stage 3 加载时将模型参数进行切片存储到不同的GPU上，每个GPU只保留参数的1/N。计算时，每个GPU跑不同的数据，然后GPU之间进行参数通信，保证每个GPU下的batch都能通过模型全部参数，而不是局部参数。（主要利用all-gather收集参数，reduce-scatter规约计算）。因此

- 每个 GPU 依然在处理不同的数据 mini-batch；
    
- 每个 GPU **逻辑上拥有完整模型**，只是参数暂时存储在不同 GPU 上；
    
- 计算时，通过参数分片广播机制，当前 GPU 拿到自己需要的参数块进行 forward/backward；
    
- 所以它依然属于 **数据并行范畴**，只是通过“分布式状态管理”节省显存。
    

换句话说：

> ZeRO Stage 3 是一种“参数分片版的数据并行”（Fully Sharded Data Parallel，FSDP）。

它并不会让不同 GPU 共同计算同一个样本的某一层，这一点和模型并行本质不同。

---

## 六、ZeRO 与其他并行技术的关系

DeepSpeed 的 ZeRO 设计非常灵活，可以与其他并行方式组合：

|并行类型|案例|可否与 ZeRO 结合|
|---|---|---|
|**Tensor 并行**|Megatron-LM|✅ 常见组合（ZeRO + TP）|
|**Pipeline 并行**|GPipe / PipeDream|✅ 可结合|
|**MoE 并行**|DeepSpeed-MoE|✅ 专门支持|
|**FSDP**|PyTorch 官方|🔁 本质上等价于 ZeRO Stage 3|

现代大模型（如 GPT-4、Mixtral、DeepSeek-V2）几乎都采用 **Hybrid Parallelism**：

> Data + Tensor + Pipeline + MoE + ZeRO 混合架构。

---

## 七、总结与启发

- DeepSpeed 的 ZeRO 是一种革命性的分布式训练思想，通过显存分片，使百亿级模型能在有限 GPU 上训练。
    
- ZeRO Stage 3 并没有改变数据并行的本质，而是让“每个 GPU 拥有完整模型”这件事在显存层面更高效地实现。
    
- 模型并行强调计算切分，ZeRO 强调状态切分。
    
- PyTorch 的 FSDP 本质上就是 ZeRO Stage 3 的思想落地。
    

> **结论：**
> 
> ZeRO Stage 3 是一种“无冗余的数据并行”，而不是模型并行。  
> 它使数据并行能够扩展到原本只有模型并行才能训练的规模。

---

## 参考资料

[DeepSpeed 官方文档](https://www.deepspeed.ai/)

[Zero Redundancy Optimizer: Memory Efficiency in Data-Parallel Training](https://arxiv.org/abs/1910.02054)

[PyTorch Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html)

[Megatron-LM: Model Parallelism at Scale](https://arxiv.org/abs/1909.08053)
    
[ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters - Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

