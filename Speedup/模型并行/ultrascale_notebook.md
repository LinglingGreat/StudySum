---
title: ultrascale_notebook
created: 2025-03-09
tags:
  - 并行训练
---


从基础开始，我们将带您了解将大型语言模型的训练从一个 GPU 扩展到数十个、数百个甚至数千个 GPU 所需的知识，并通过实际的代码示例和可重现的基准来说明理论。

## 概述

本书介绍的所有技术都将解决以下三个关键挑战中的一个或多个，我们将在整本书中不断遇到这些挑战：

1. **内存使用**：这是一个硬性限制 - 如果训练步骤无法容纳在内存中，则训练无法进行

2. **计算效率**：我们希望我们的硬件花费大部分时间进行计算，因此我们需要减少花在数据传输或等待其他 GPU 执行工作上的时间。

3. **通信开销**：我们希望尽量减少通信开销，因为这会使 GPU 保持空闲状态。为了实现这一点，我们将尝试充分利用节点内（快速）和节点间（较慢）带宽，并尽可能将通信与计算重叠。

在很多地方，我们会看到我们可以用其中之一（计算、通信、内存）换取另一个（例如重新计算或张量并行）。找到正确的平衡是扩展训练的关键。

## 第一步：在一个 GPU 上进行训练

在开始扩展到多个 GPU 之前，我们先快速回顾一下模型训练的基础知识。在单个 GPU 上训练模型时，训练通常包括三个步骤：

1. 前向传递，将输入传递至模型以产生输出，
2.   反向传递以计算梯度，以及
3. 使用梯度更新参数的优化步骤

**批量大小（bs )** 是模型训练的重要超参数之一，影响模型收敛和吞吐量。

在训练初期，小批量大小可能很有用，可以快速沿着训练场景移动，达到最佳学习点。然而，在模型训练的后期，小批量大小将使梯度保持嘈杂，模型可能无法收敛到最佳的最终性能。在另一个极端，大批量虽然能提供非常准确的梯度估计，但往往会减少对每个训练令牌的充分使用，导致收敛速度变慢，并可能浪费计算。你可以在 OpenAI 关于[大批量训练的论文](**An Empirical Model of Large-Batch Training**  [[PDF]](http://arxiv.org/pdf/1812.06162.pdf))或 MiniMax-01[技术报告](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)第 4.2 节中找到关于这个主题的讨论。

批次大小还会影响在给定文本数据集上进行训练所需的时间：较小的批次大小将需要更多的优化器步骤来对相同数量的样本进行训练。优化器步骤成本高昂（在计算时间方面），因此与使用较大的批次大小相比，总训练时间将会增加。话虽如此，请注意，批次大小通常可以在最佳批次大小附近进行相当大的调整，而不会对模型的性能产生重大影响，即最终模型性能对确切批次大小值的敏感度通常在最佳批次大小附近相当低。

在预训练社区中，批次大小通常以token而非样本数量 ( bst = 批次大小token) 来报告，这使得训练数量通常与训练期间使用的确切输入序列长度无关。

bst=bs∗seq

近期训练的最佳规模通常为每批次 400 万到 6000 万个标记。多年来，批次大小以及训练语料库一直在稳步增长：Llama 1 的训练批次大小约为 400 万个标记，共 1.4 万亿个标记，而 DeepSeek 的训练批次大小约为 6000 万个标记，共 14 万亿个标记。

**当我们将模型训练扩展到如此大的批次大小时，我们的第一个挑战已经出现：内存不足问题。当我们的 GPU 没有足够的内存来容纳目标批次大小的完整批次时，我们该怎么办？**

首先，让我们快速了解一下导致内存不足问题的原因。这将有助于我们获得一些有关训练模型的内存需求的有用直觉。

### Transformers 中的内存使用情况

训练神经网络模型时，内存中需要存储以下内容：

- Model weights  模型权重
- Model gradients  模型梯度
- Optimizer states  优化器状态
- Activations needed to compute the gradients  计算梯度所需的激活

	您可能会认为，对于一个模型，您可以精确地计算内存需求，但是还有一些额外的内存占用因素使得计算起来很难精确：
	
	CUDA 内核通常需要 1-2 GB 的 GPU 内存，您可以通过运行 `import torch; torch.ones((1, 1)).to("cuda")` 然后使用`nvidia-smi`检查 GPU 内存来快速验证。
	
	缓冲区、中间结果和一些由于碎片而无法使用的内存使用情况
	
	我们将忽略最后两个因素，因为它们通常是较小且恒定的因素。


这些项目存储为具有不同_形状_和_精度的_张量。_形状_由超参数决定，例如批量大小、序列长度、模型隐藏维度、注意力头、词汇量和潜在的模型分片，我们稍后会看到。_精度_是指 FP32、BF16 或 FP8 等格式，它们分别需要 4、2 或 1 个字节来存储张量中的每个单个值。我们将在[混合精度训练](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#mixed_precision_training)部分全面讨论不同的精度及其权衡，现在我们只需记住，这些不同格式的内存要求会有所不同，这将影响我们需要存储的项目的内存使用情况。

那么如何才能根据这些变量快速确定内存使用情况呢？一种简单的方法是根据经验进行测量。

#### 分析内存使用情况

使用 Pytorch 分析器，我们可以了解整个训练过程中内存的分配情况。我们可以看到，内存利用率并不是静态的，而是在训练期间和训练步骤中变化很大：

![](img1/Pasted%20image%2020250309154359.png)

显然，第一步与后续步骤非常不同，但让我们先看看步骤的一般结构：首先，当我们进行前向传递时，激活值会快速增加，然后在后向传递期间，梯度会累积，随着后向传递的传播，用于计算梯度的存储激活值会逐渐清除。最后，我们执行优化步骤，在此过程中我们需要所有梯度，然后在开始下一个前向传递之前更新优化器状态。

为什么第一步看起来不同：激活迅速增加，然后稳定一段时间。在第一步中，torch 缓存分配器进行了大量准备，准备内存分配以加快后续步骤，这样它们就不需要在之后搜索空闲内存块（参见[Zach 的博客](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)）。在第一步之后，我们还会看到优化器状态出现，这通常会抵消进一步训练步骤的内存使用量。

现在我们对内存有了初步的了解，让我们看看如何扩大训练规模通常是一个最大化计算效率的问题，同时将这些不同项目（激活、参数、梯度、优化器状态）的内存需求保持在 GPU 的内存限制范围内。

#### 权重/梯度/优化器状态记忆

让我们从前 3 项开始：模型的权重、梯度和优化器状态。我们实际上可以很容易地估算它们所需的内存。

对于简单的变压器LLM，参数数量由[以下公式](https://michaelwornow.net/2024/01/18/counting-params-in-transformer)给出：

N=h∗v+L∗(12∗h^2+13∗h)+2∗h

h 是隐藏维度， vv 是词汇量， LL 是模型中的层数。请注意，查看该等式我们可以看到，在较大的隐藏维度中占主导地位的项是 h^2项，因为它是唯一一个随着参数缩放而呈二次增长的项。

参数和梯度的内存需求只是参数数量乘以每个参数的字节数。在传统的全精度 (FP32) 训练中，参数和梯度都需要 4 个字节，而优化器（如果我们使用 Adam）需要存储动量和方差，这会为每个参数增加另外两个 4 个字节。总结：

![](img1/Pasted%20image%2020250309154758.png)

现在，让我们看看如果我们使用较低的精度，情况会如何变化。出于稳定性原因（请参阅[下面的混合精度训练部分](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#mixed_precision_training)），我们通常不使用完全低精度训练，而是使用高精度和低精度的混合，称为“混合精度”。目前混合精度训练的默认设置是通常使用 BF16 进行大多数计算 - 每个参数和梯度需要 2 个字节 - 以及在 FP32 中额外复制模型权重和梯度，因此每个参数总共需要 12 个字节。除了参数和梯度之外，我们还需要存储优化器状态：对于 Adam 优化器，这需要动量和方差通常存储在 FP32 中以实现数值稳定性，每个使用 4 个字节。

![](img1/Pasted%20image%2020250309154856.png)

	一些库将梯度存储在 fp32 中，这需要额外的 mparams_fp32=4∗N内存。例如，在 nanotron 中就是这样做的，因为`bf16`对于较小的值是有损的，而我们始终优先考虑稳定性。有关更多信息，请参阅[此 DeepSpeed 问题](https://github.com/microsoft/DeepSpeed/issues/1773)。
	
	在文献和代码库中，参数的 FP32 副本 ( mparams_fp32) 有时被称为“主权重”。


有趣的是，混合精度本身并不能节省整体内存，因为它只是在三个组件之间以不同的方式分配内存，事实上，如果我们在 FP32 中积累梯度，它会比全精度训练增加另外 4 个字节。它仍然具有优势，因为以半精度计算前向/后向传递允许我们 (1) 在 GPU 上使用更快的优化低精度操作，以及 (2) 减少前向传递期间的激活内存要求，这是内存使用量的很大一部分，正如我们在上图和下图中看到的那样。

让我们了解一下模型需要多少通用内存（全精度和混合精度给出相同的总体值）：

![](img1/Pasted%20image%2020250309155148.png)

我们可以看到，一旦达到**7B** （！），权重和优化器要求就会开始显著增加，并超过典型 GPU 内存的大小，例如 H100 GPU 是 80GB。

但是现在，让我们从仍然适合单个 GPU 的模型开始，看看我们内存预算的最后一个重要贡献者：激活内存。

#### 激活内存

激活内存的计算比权重、梯度和优化器状态稍微复杂一些，部分原因是它取决于模型的输入。如果您不确定我们为什么需要存储反向传递的激活，[此参考资料](https://www.determined.ai/blog/act-mem-2)是一个很好的快速复习。在仔细检查反向传递的计算方式后，我们可以估算混合精度激活所需的总内存，并得出以下公式：

![](img1/Pasted%20image%2020250309155406.png)

这里 L 是层数， seq 是序列长度， bs 是样本的批量大小，h 是模型的隐藏维度， nheads​ 是头的数量。

有关数字的精确推导，你可以关注 NVIDIA 关于重新计算的原始论文（**Reducing Activation Recomputation in Large Transformer Models**  [[PDF]](http://arxiv.org/pdf/2205.05198.pdf)），它本质上要求你对转换器层中每个操作之间的所有中间激活的大小进行一些计算。

这里有一个有趣的观察，即对于给定的模型，内存使用量并不是静态的；相反，它与批处理大小成线性关系，与序列长度成二次关系。这意味着当我们增加批处理大小或使用更长的序列进行训练时，激活内存将会爆炸。我们可以使用此方程来查看内存使用量如何随不同序列长度而变化，例如对于 Llama 模型（ `bs=1` ）：

![](img1/Pasted%20image%2020250309155555.png)

该图表讲述了一个引人注目的故事：对于短序列（或类似的小批量），激活几乎可以忽略不计，但从大约 2-4k 个标记开始，它们会占用大量内存，而参数、梯度和优化器状态的使用（我们将稍后讨论）大致与序列长度和批量大小无关。

**对于较大的输入标记（又名较大的批量大小/序列），激活成为迄今为止最大的内存负担。**

有没有办法控制这种“激活爆发”？

现在是时候解释我们的第一种技术了——**_激活重新计算__——_** 它将帮助我们限制激活内存占用。这是当今大型模型训练工具箱中必不可少的工具。

### 激活重新计算

**_激活重新计算_**（也称为_梯度检查点_或_重新实现_，_gradient checkpointing_ or _rematerialization_）背后的一般思想是在正向传递期间丢弃一些激活以节省内存，并在反向传递期间花费一些额外的计算来动态重新计算这些激活。如果不进行重新计算，我们会存储两个可学习操作（例如feed-forward, layernorm等）之间的每个隐藏状态，以便我们可以在反向传递期间使用它们来计算梯度。当我们使用重新计算时，我们通常只会在模型架构的几个关键点存储激活，丢弃其余激活并在反向传递期间从最近保存的激活中动态重新计算它们，基本上再次执行正向传递的子部分以权衡内存和计算。它通常看起来像这样：

![](img1/Pasted%20image%2020250309155841.png)

有几种策略可以选择要存储的关键激活：

**完整**：我们在 Transformer 模型的每一层之间的转换点处检查激活。这通常称为`full`策略，因为它需要通过每一层进行前向传递，本质上是在后向传递期间添加完整的前向传递。此策略节省了最多的内存，但在计算方面是最昂贵的。它通常会将计算成本和时间增加高达 30-40%，这是非常明显的。

**选择性**：一般来说，我们可以做得比完全性更好。重新计算论文的作者（**Reducing Activation Recomputation in Large Transformer Models**  [[PDF]](http://arxiv.org/pdf/2205.05198.pdf)）进行了详细的分析，研究了哪些激活增长最大，并且 FLOP 方面的重新计算成本最低。结果表明，注意力计算属于这一类，因此我们通常可以丢弃它们，并专注于检查昂贵的前馈计算。对于 GPT-3 (175B) 模型，这意味着**在 2.7% 的计算成本下，激活内存减少 70%** 。

让我们看看重新计算策略在实践中如何大幅度减少内存占用，以及选择性重新计算如何在节省内存和重新计算成本之间取得良好的平衡：

![](img1/Pasted%20image%2020250309160138.png)

![](img1/Pasted%20image%2020250309160147.png)

![](img1/Pasted%20image%2020250309160208.png)

注意：更多模型的效果可以查看原文

这里显而易见的另一个趋势是，长序列的激活对于较小的模型起着更大的作用，因此重新计算的效果变得更加明显。

	当您测量训练设置使用 GPU/TPU/加速器的效率时，通常需要将重新计算考虑在内，以计算总 FLOPS（每秒浮点运算次数），并将其与 GPU/TPU/加速器的理论最大 FLOPS 进行比较。在计算训练步骤的 FLOPS 时考虑重新计算会给出一个称为“硬件 FLOPS”的值，即在加速器上执行的实际操作次数。将此数字除以训练步骤的持续时间和最大加速器 FLOPS 可得出**_硬件 FLOPS 利用率 (HFU)。_**
	
	然而，归根结底，真正重要的是在给定数据集上训练模型所需的从开始到结束的时间。因此，当比较各种 GPU/TPU/加速器时，如果其中一个加速器提供了足够的内存来跳过重新计算，从而每秒执行更少的操作（较低的 HFU），但是为了更快的训练，它应该得到奖励而不是惩罚。因此，另一种方法是计算所谓的**_模型 FLOPS 利用率 (MFU)_** ，与 HFU 相比，它仅考虑模型的前向 + 后向传递所需的操作，并且不包括测量的 FLOP 中的重新计算。因此，该值比训练实现更具体到模型。

如今，大多数训练框架都使用 FlashAttention（我们将[在下文中进一步](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#flash_attention_1-3)介绍），该框架通过重新计算后向传递中的注意力分数和矩阵（而不是存储它们），将激活重新计算原生地集成到其优化策略中。因此，大多数使用 Flash Attention 的人都已经利用了选择性重新计算。

**正如您现在已经了解的，激活重新计算由于重新计算而略微增加了 FLOP 的数量，同时显著减少了内存访问开销。**

现在我们已经了解了重新计算，我们可以控制激活内存的使用，如上图所示！

然而，激活仍然与批次大小呈线性相关，并且上面的条形图中的所有配置文件都使用`bs=1`因此当我们转向更大的批次大小时，这可能会再次成为一个问题。不要绝望，因为我们的盒子里还有第二个工具——**_梯度累积_**来拯救你！

### 梯度累积

梯度累积是一种非常简单的避免内存爆炸的方法，其方法是将批次分成微批次。我们将在每个微批次上依次执行前向和后向传递，计算梯度，并且顾名思义，在执行优化器步骤之前将所有微批次的梯度相加。实际上，优化步骤不是基于梯度之和，而是基于梯度的平均值，因此结果与梯度累积步骤的数量无关。

我们将每次前向传递的批次大小称为`micro batch size` (mbs)。我们将每次优化步骤之间的总体批次大小称为`global batch size` (gbs)。如果我们每 8 次前向/后向传递执行一次优化步骤，则`global batch size`将是`micro batch size` 8 倍。

因此，我们现在所说的`global batch size`与我们迄今为止为了简单起见所称的`batch size`相对应（我们现在使我们的术语更加精确以避免歧义）。

通过梯度积累，全局批量大小可以简单地计算如下：

bs=gbs=mbs×grad_acc

梯度累积使我们能够有效地将批处理大小增加到无穷大（甚至更大！），同时内存占用保持不变。梯度累积还与激活重新计算兼容，以进一步减少内存。

![](img1/Pasted%20image%2020250309160740.png)

使用梯度累积意味着我们需要保留缓冲区，在其中累积梯度，这些梯度会在整个训练步骤中持续存在。而如果没有梯度累积，则会在释放激活内存的同时计算后向梯度，这意味着峰值内存较低。

通过梯度积累，我们可以仅计算部分微批次，从而减少随批次大小线性增长的激活内存。

**然而，梯度累积的一个缺点是，每个优化步骤需要多次连续的前向/后向传递，从而增加了计算开销并减慢了训练速度。没有免费的午餐！**

但如果你仔细观察，你可能会注意到每个微批次的前向/后向传递实际上可以并行运行。前向/后向传递彼此独立，唯一的区别是独立的输入样本。看来是时候开始将我们的训练扩展到多个 GPU 了！

在此之前，让我们快速了解一下如何通过分布式训练工具箱中最有用的工具之一：**分析器**来可视化计算和通信。此工具对于理解和验证 GPU 和计算之间的通信方式以及瓶颈所在非常有用。

#### 分析 GPU 计算和通信

PyTorch 的[分析器](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)允许我们精确跟踪和可视化训练期间 CPU 和 GPU 上发生的情况。它原生集成在 PyTorch 中。让我们看看如何使用它：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    with_stack=True
) as prof:
    for step in range(steps):
        train_step() 
        prof.step()
```

这会生成一条跟踪记录，我们可以在 TensorBoard 或 Chrome 的跟踪查看器中对其进行可视化。跟踪记录显示：

- CPU 线程以异步方式向 GPU 启动内核
- 多个 CUDA 流并行处理计算和通信
- 内核执行时间和内存分配

![](img1/Pasted%20image%2020250309160956.png)

示例跟踪显示 CPU 线程以异步方式向 GPU 启动内核，计算内核和通信在不同的 CUDA 流之间并行进行

跟踪有助于识别如下瓶颈：

- 可重叠的顺序计算和通信
- 等待数据传输的空闲 GPU 时间
- CPU 和 GPU 之间的内存移动
- CPU 的内核启动开销

了解这些模式对于优化分布式训练性能至关重要。例如，轨迹将清楚地显示梯度同步是否与后向计算正确重叠，我们将在后面讨论。

现在让我们获得一个配备几个 GPU 的更大的工作站🖥️，并开始研究我们的第一个称为_**数据并行的**扩展技术，正如我们将看到的，它只是梯度积累的并行版本_。

## 数据并行

数据并行 (DP) 背后的理念是在多个 GPU 上复制模型（我们称副本为“模型实例”），并在每个 GPU 上并行对不同的微批次数据进行前向和后向传递，因此得名数据并行。您可能已经在简单的训练示例中看到了数据并行，但您很快就会看到，我们将在本节中更深入地探讨，因此即使您知道一般方法，也请继续关注。

如果您不熟悉广播、收集或全归约等分布式通信模式，我们在[A0：并行编程速成课程](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#a0%3A_parallel_programming_crash_course)中整理了一个小型速成课程。

![](img1/Pasted%20image%2020250309161330.png)

对每个 GPU 使用不同的微批次意味着我们在每个 GPU 上都会有不同的梯度，因此为了使模型实例在不同的 GPU 上保持同步，来自模型实例的梯度将使用称为“all-reduce”的操作进行平均，该操作发生在优化器步骤之前的后向传递期间。

这涉及到我们的第一个“分布式通信”原语： _**all-reduce**_ ，它处理 GPU 实例和节点之间的同步和通信。

![](img1/Pasted%20image%2020250309161555.png)

一个简单的 DP 实现会等待反向传递完成，这样我们就有了所有的梯度，然后它会触发所有 DP 等级的全归约，以同步这些梯度。但是，这种先计算后通信的连续步骤是**绝对不行的！** 因为我们不希望我们的 GPU 在通信时保持空闲状态，就像上图所示。

相反，我们应该尽可能地让通信和计算重叠，以便它们尽可能同时发生。

让我们看看三种优化，它们可以让我们做得比我们最初的实现更好！

#### 第一次优化：重叠梯度同步与后向传递

我们刚刚描述的简单 DDP 方法的主要缺点是，在反向传递（_计算_）之后，我们必须等待梯度同步（_通信_）才能更新参数。我们可以将这种通信与计算重叠吗？答案是肯定的！

如上图所示，甚至可以在计算出较早层（左侧的红色框）的梯度之前，就收集并求和某一层的梯度（红色框）。例如，最后一层的反向传递一完成（右侧的最后一个框），这些梯度就可以被收集并求和，同时继续向左移动较早层的反向计算。

![](img1/Pasted%20image%2020250309161840.png)

在 pytorch 中，可以通过将_all-reduce 钩子函数_附加到每个参数来实现。一旦该参数的梯度准备好，就会触发 all-reduce 操作，而其他参数的梯度仍在计算中。这种方法将大多数 all-reduce 操作与梯度计算重叠，从而提高效率。这是一个附加钩子的简单函数：

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that 
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

重叠计算和通信减少了等待整个模型的梯度同步所花费的时间。梯度同步可以（至少部分地）与反向传递并行发生，从而显著加快数据并行性。以下是具有同步重叠的简单 DP 的完整实现：

在 Picotron 中具有重叠的简单 DP 实现

```python
class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting point to understand how data parallelism works.
    It implements a simple all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """
    def __init__(self, module):
        """
        Initializes the DataParallel wrapper for a given module.

        Args:
            module (nn.Module): The model to be wrapped for data parallelism.
            process_group (torch.distributed.ProcessGroup): The process group used for gradient synchronization. 
                                                            It could be a data parallel or context parallel group.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require gradients.    
        """
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """
        Performs an all-reduce operation to synchronize gradients across multiple processes.    
        """
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization. 
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing 
        gradients in between.
        """
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

```

这是我们的第一个“_重叠计算和通信_”示例，我们将在本博文中多次讨论它，这是实现最大扩展效率的一项必不可少的技术。但我们可以进一步提高效率！

#### 第二次优化：梯度下降

GPU 操作在大型张量上执行时通常比在小型张量上运行许多操作更高效。通信操作也是如此。因此，我们可以有利地将梯度分组到存储桶中，并为同一存储桶内的所有梯度启动单个 all-reduce，而不是为每个梯度执行独立的 all-reduce。它通常如下所示：

![](img1/Pasted%20image%2020250309162108.png)

想象一下在运输前将物品打包到箱子里。发送几个大箱子比发送很多小箱子更有效率。通过对每个 bucket 执行单个 all-reduce 操作，我们可以显著减少通信开销并加快通信操作。

以下是使用 bucketing 的代码实现：

Picotron 中的 Bucket DP 实现

```python
class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        """
        Initialize the DataParallelBucket module.
        
        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either 
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes. 
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 

```

#### 第三次优化：与梯度积累的相互作用

最后，正如我们之前所见，梯度积累的工作原理是在使用`optimizer.step()`更新参数之前执行多次前向和后向传递。当将梯度积累与数据并行相结合时，我们在想要同步梯度时应该小心。

在简单版本中，在累积过程中的每次向后传递之后都会自动触发全归约操作，这不是最优的，因为最后一步之后的单次归约会具有相同的效果，同时减少开销。

在 PyTorch 中，通常通过在不需要减少的后向传递中添加[`model.no_sync()`](https://github.com/pytorch/pytorch/blob/5ea67778619c31b13644914deef709199052ee55/torch/nn/parallel/distributed.py#L1408-L1435)装饰器（禁用梯度同步）来解决此问题。

执行通信操作时，张量必须在内存中连续，以避免冗余内存复制。为了最佳地执行此操作，我们通常会预先分配与激活或模型参数大小相同的连续缓冲区，专门用于通信。虽然这可以加快通信速度，但也在一定程度上导致了训练期间的峰值内存使用量。

现在让我们看看这对于全局批量大小意味着什么。

### 重新审视全局批次大小

我们可以使用新添加的数据并行和梯度累积参数来更新批量大小方程：

bs=gbs=mbs×grad_acc×dp

这里 grad_acc 是梯度累积步骤的数量， dp 是用于数据并行的并行实例的数量。

给定一个目标全局批量大小，我们就可以用梯度积累步骤来交换数据并行过程，以加快训练速度。

在实践中，人们倾向于尽可能地增加梯度累积的数据并行节点 (DP) 数量，因为它本质上是并行的，而不像梯度累积的顺序性质。当在 GPU 耗尽之前仅扩展数据并行性是不够的时，梯度累积就会添加到数据并行性之上，以实现目标全局批处理大小。

能够将训练分布到不同的样本上，为我们提供了并行化的第一个维度，从而实现了一维并行性（我们将逐步覆盖另外 4 个维度）。

### 我们迄今为止的旅程

让我们快速总结一下如何使用最佳数据并行设置的草稿方案来设置我们的第一个 1D 并行训练：

1. 我们应该首先通过查阅文献或运行测量模型收敛性的实验来确定最佳（全局）标记批量大小（ `GBST` ）。
2. 然后，我们再次通过查阅文献或进行实验来选择训练的序列长度。通常，2-8k 个标记对于我们今天的评估来说效果很好（我们不会在这里深入讨论训练方法，但团队通常会在训练结束时增加序列，在组合中添加一些较长的上下文数据样本以达到今天的较长上下文大小）。
3. 我们现在知道了批处理大小 (gbs)。我们可以通过增加本地批处理大小直到内存耗尽来找到单个 GPU 上的最大本地批处理大小 (mbs)。
4. 最后，我们确定目标 DP 可用的 GPU 数量。GBS 与 DP 的比率为我们提供了所需 GBS 所需的剩余梯度累积步骤数。

如果梯度累积率低于一，即我们的 GPU 太多，又称 GPU 丰富 🤑 (!)，我们可以选择不使用所有 GPU，探索更大的全局批处理大小，或者测试较低的 MBS 是否会加快训练速度。在后一种情况下，我们最终会优先考虑吞吐量而不是单个 GPU 计算效率，使用尽可能小的 MBS 来加快训练速度。

现在来举一个具体的例子：假设我们想要训练一个 GBS 为 4M 个 token 且序列长度为 4k 的最新模型。因此，我们的批处理大小将为 1024 个样本（我们选择最接近的 2 的幂）。假设我们观察到单个 GPU 只能在内存中容纳 MBS=2，并且我们有 128 个 GPU 可用于训练。这意味着通过 4 个梯度累积步骤，我们将实现每个训练步骤 1024 个样本或 4M 个 token 的目标。现在，如果我们突然有 512 个 GPU 可用，该怎么办？我们可以通过保持 MBS=2 并将梯度累积步骤设置为 1 来实现相同的 GBS，从而实现相同的训练，并实现更快的训练！

请记住，在 512+ GPU 规模下，根据所使用的网络，通信操作将开始受到_环延迟_（信号在环上传播一次所需的时间）的限制，这意味着我们无法再完全重叠 DP 通信。这会降低我们的计算效率并影响我们的吞吐量。在这种情况下，我们应该开始探索其他维度来进行并行化。

虽然数据并行性可以很好地将全归约梯度同步与后向计算相结合以节省时间，但这种优势在大规模上开始消失。为什么？因为随着我们添加越来越多的 GPU（数百或数千个），它们之间的协调开销会显著增加，并且网络要求变得太大而无法带来好处。因此，随着我们向系统添加每个额外的 GPU，我们的设置将变得越来越低效。

让我们通过一些基准来观察实践中这种情况的发生：

![](img1/Pasted%20image%2020250309162836.png)

我们发现，超过某个限制后，吞吐量开始大幅下降，而每个 GPU 的内存使用量保持不变，并且不受添加更多 DP 等级的影响。

**数据并行是我们扩展更多 GPU 训练的第一个（简单）策略。该技术的工作原理类似于梯度累积，但并行化了微批次的前向和后向传递，从而提高了吞吐量！**

然而，敏锐的读者可能已经注意到，这假设我们可以将至少一个输入样本前向传递（mbs _=1）_放入我们的 GPU 内存中。情况并非总是如此！正如我们所见，即使激活了激活重新计算，较大的模型也无法放入单个 GPU 中：

提示：您可以通过乘以 2 来快速估算模型参数所需的最小内存，例如 70B → 140GB (=133GiB)

![](img1/Pasted%20image%2020250309163013.png)

我们还发现，在一定扩展水平以上，数据并行开始产生一些限制性通信开销。对于这些较大的模型或较大的批量大小，我们还有其他选择吗？幸运的是，我们确实有一些解决方案。它们将涉及将一些张量移至 CPU 或将权重/梯度/优化器状态张量拆分到 GPU 设备之间！让我们开始深入研究它们。

拆分主要有两种方法：并行（张量、上下文或管道并行）和共享（DeepSpeed Zero 或 PyTorch FSDP）。这两种方法在某种程度上是正交的，实际上可以结合起来！

共享范式与 DP 密切相关，因此我们将首先通过研究 ZeRO 方法来了解它！

### ZeRO（**零****冗余****优化**器）

在本节中，我们将介绍 DeepSpeed ZeRO（ **Ze** ro **R** edundancy **O** ptimizer），这是一项旨在减少训练中的内存冗余的内存优化技术。

虽然数据并行是一种有效的训练扩展方法，但在每个 DP 等级上对优化器状态、梯度和参数的简单复制会引入显著的内存冗余。ZeRO 通过在数据并行维度上对优化器状态、梯度和参数进行分区来消除内存冗余，同时仍然允许使用全套参数进行计算。这有时需要 DP 等级(rank)之间进行更多通信，这些等级可能完全重叠，也可能不完全重叠，我们接下来会看到！

该方法分为 ZeRO 的三个可能的优化阶段：

- ZeRO-1: optimizer state partitioning  
    ZeRO-1：优化器状态分区
- ZeRO-2: optimizer state + gradient partitioning  
    ZeRO-2：优化器状态 + 梯度分区
- ZeRO-3 (also called FSDP for “Fully-Sharded Data Parallelism”): optimizer state + gradient + parameter partitioning  
    ZeRO-3（也称为 FSDP，即“全分片数据并行”）：优化器状态 + 梯度 + 参数分区

当我们说分区时，它意味着沿着 DP 轴，因为 ZeRO 是数据并行的一部分。我们稍后会看到我们可以沿着其他轴进行分区。

您可能遗漏了我们可以分片的内容中的激活。由于模型的每个 DP 副本都会收到不同的微批次，因此每个 DP 等级上的激活也不同，因此它们不会重复，因此无法分片！

让我们仔细看看通过对每个 ZeRO 阶段进行分区我们可以节省多少！

#### 重新审视内存使用情况

您可能还记得[我们上一节](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#memory_usage_in_transformers)中介绍的标准训练期间优化器状态、梯度和参数的内存使用情况。我们将模型的参数数量称为 Ψ （以前为 N，但这里我们使用原始 ZeRO 论文符号）。在使用 Adam 优化器的[混合精度训练](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#mixed_precision_training)（后面部分将详细介绍）中，我们需要存储的每个项目的内存使用量为：


- Model’s parameters (half precision i.e. bf16/fp16): 2Ψ  
    模型参数（半精度即 bf16/fp16）： 2Ψ
- Model’s gradients (half precision i.e. bf16/fp16): 2Ψ 
    模型的梯度（半精度，即 bf16/fp16）： 2Ψ
- Model’s parameters in fp32 and optimizer states: 4Ψ+(4Ψ+4Ψ) 
    fp32 中的模型参数和优化器状态： 4Ψ+(4Ψ+4Ψ)
- Model’s gradients in fp32: 4Ψ (optional, only accounted if we want to accumulate grads in fp32)  
    fp32 中的模型梯度： 4Ψ （可选，仅当我们想在 fp32 中累积梯度时才考虑）

如果我们不在 fp32 中累积梯度，则总内存消耗为 2Ψ+2Ψ+12Ψ，如果我们累积，则为 2Ψ+6Ψ+12Ψ 。为了简单起见，我们现在重点讨论没有 fp32 梯度累积的情况，但您只需将受 ZeRO-2 和 3 影响的额外字节添加到梯度项中即可。

ZeRO 的想法是将这些对象在 DP 等级之间进行分片，每个节点仅存储一部分项目，这些项目在需要时进行重建，从而将内存使用量除以数据并行度 N_d ：

![](img1/Pasted%20image%2020250309163558.png)

这里 Ψ 表示参数的数量， k 表示优化器状态的内存乘数（对于 Adam 来说为 k=12 ，正如我们刚刚看到的）， Nd​ 表示 DP 度。

#### ZeRO-1：分区优化器状态

在 vanilla DP 中，所有等级在反向传播后收集相同的梯度并同时执行相同的优化器步骤。这似乎是很多重复的工作。我们能否避免这种情况并同时减少内存使用量？

在 ZeRO-1 中，优化器状态被划分为 Nd​ 个相等的部分，其中 Nd 是 DP 度。这意味着分布在每个 DP 等级上的每个模型副本仅跟踪优化器状态的 1/Nd​ 。在优化步骤中，仅更新 float32 权重的 1/Nd​ 。

但是在前向传递过程中，每个副本都需要所有参数，因此我们需要在优化器步骤之后添加一个额外的**_全收集_** （我们遇到的第二种集体通信原语！），以便每个模型副本都有完整的更新权重集。

这解释了我们在上图中看到的 2Ψ+2Ψ+kΨ/Nd​ 的记忆公式！以下是单个训练步骤的操作序列摘要

- 每个副本上使用相同的全套 bf16 参数进行前向传递，但各个副本之间的微批次不同
- 每个副本上使用相同的全套梯度进行反向传递，但各个副本上的微批次不同
- 对梯度执行减少散射reduce-scatter（我们将在下图中解释减少散射原语）
- 每个副本在其本地优化器步骤（仅 1/Nd 个优化器状态）上执行优化器步骤以获取更新的 1/Nd​ fp32 参数，然后可以将其转换为 1/Nd 组完整的 bf16 参数。
- 在 bf16 参数之间执行全收集，以将缺失的切片发送回每个副本。这是 ZeRO 中的新操作，在 vanilla DP 中未使用。

您可能想知道这个“减少散射”操作是什么，以及这一切看起来如何，所以让我们尝试使用下图使其更加图形化。我们将介绍前向/后向传递循环的所有步骤：

  
![dp_zero1.gif](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero1.gif)

在实际通信方面，与 vanilla DP 相比，Zero-1 将我们的“全归约”梯度通信更改为“归约-散射”操作，并在优化器步骤之后对所有参数添加全收集操作。它看起来如下：

![](img1/Pasted%20image%2020250309164220.png)

如果您一直在关注，您会记得，从 vanilla DP 中，我们可以将全归约梯度通信与后向传递计算重叠。在 ZeRO-1 中，我们还可以研究如何有效地重叠新添加的 bf16 参数全集合。为此，有两种主要策略：

- 在优化器步骤期间：我们可以在优化器更新部分参数后立即启动全部收集。这允许通信潜在地与其他参数更新重叠。
- 在前向传播过程中：我们可以将每层参数的全部集合与前向传播重叠。

不幸的是，这些技术并不容易实现，需要复杂的 hooks/bucketing 使用。实际上，我们可以使用 PyTorch 原生的 ZeRO-3/FSDP 实现，并将 FSDPUnit 设置为整个模型，稍后将详细介绍这一点。

在 ZeRO-1 中，优化器状态已被分区，这意味着每个副本仅更新优化器状态的 1/Nd​ 。敏锐的读者一定已经注意到，一开始并不需要在所有 DP 等级上拥有所有梯度，因为优化步骤只需要一个子集。来见识一下 ZeRO-2！

#### ZeRO-2：添加梯度分区

由于我们只需要在每个副本上将梯度分片与优化器状态分片相对应，因此将梯度分片为优化器状态也是有意义的。在反向传递过程中，我们只执行**_减少散射操作，而不是对梯度执行全部减少_**！我们只在内存中分散所需的 1/Nd​ 梯度，因此与 ZeRO-1 相比节省了更多内存。


对于 FP32 梯度累积，我们只需要保留 1/Nd​ fp32_grads，其中我们累积了来自 Reduce-Scatter 的 bf16 梯度。在优化器步骤中，我们使用 1/Nd fp32_grads。

![dp_zero2.gif](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_zero2.gif)

现在很容易看出，对梯度进行分片会导致 2Ψ+(2Ψ+kΨ)/Nd​ ，而随着 Nd​ 的增加，我们可以比基线节省高达 8 倍的内存。在通信方面，与 ZeRO-1 相同的过程适用，唯一的区别是我们即时通信和释放。总的来说，ZeRO-2 也相当于 vanilla DP 训练的通信。

在通信方面，ZeRO-2 与 ZeRO-1 类似，它们都需要对梯度进行减少散射，并对所有参数进行全聚集。

![](img1/Pasted%20image%2020250309164451.png)

注意：您可能会注意到，使用 ZeRO-2 而不是 ZeRO-1 并没有实际开销，事实上 ZeRO-2 通常是最佳选择。

现在我们已经对梯度进行了分片，那么我们完成了吗？或者我们可以继续这样做吗？好吧，差不多。ZeRO-3 来了！

#### ZeRO-3：添加参数分区

对于第 3 阶段，我们扩展了上述对 DP 副本进行分片优化器状态和梯度的方法，直至对模型的参数进行分片。

在 PyTorch 原生实现中，此阶段也称为 FSDP（完全共享数据并行）。我们在本博文中仅提及 ZeRO-3，但无论你在哪里看到它，你都可以想到 FSDP。

那么，如果模型的所有部分都是分布式的，我们在实践中如何进行前向或后向传递呢？很简单，我们在需要时按需收集它们。在前向传递中，这看起来如下：

![](img1/Pasted%20image%2020250309164552.png)

因此，当我们执行前向传递并按顺序遍历各层时，我们会根据需要检索必要的参数，并在不再需要它们时立即将它们从内存中清除。后向传递的工作方式相同，只是流程反转，然后我们生成梯度碎片：

![](img1/Pasted%20image%2020250309164613.png)

另一个问题是，我们需要在整个前向和后向步骤中连续进行这些全收集，与 Zero-2 相比，这相当于在**训练步骤**中进行 2⋅num_layers−1 次额外的全收集，每次收集都会带来较小的**基本延迟**开销，如下图所示：

![](img1/Pasted%20image%2020250309164644.png)

在前向传递过程中，我们会在需要参数时执行全收集操作，因此会产生 Ψ 通信成本。由于我们在前向传递过程中需要参数后立即丢弃它们，因此我们需要在后向传递过程中再进行一次全收集，这又会产生 Ψ 通信成本。最后，我们需要与 ZeRO-2 中相同的梯度**_减少散射，_** 这也会产生 Ψ 通信成本，因此我们得出的总通信成本为 3Ψ ，而 Zero-2 的成本为 2Ψ 。

这听起来可能有很多通信开销，但实际上这很好，因为我们可以将下一层的参数通信与当前层的前向传递重叠，这称为**预取**。通过预取，我们将在前向执行第_n 层_的当前前向时“收集”第 n+1 层的权重，同样，我们将在对_第 n 层_进行后向执行时“收集”_第 n-1 层_的权重。当然，这种重叠只有在我们不过度扩展 DP 的情况下才成立。（根据经验，DP 不应超过 512）

在内存方面，我们可以看到我们的方程现在已达到其最终形式 (2Ψ+2Ψ+kΨ)/Nd​ ，这意味着如果我们可以增加 DP 等级，至少对于模型相关参数而言，我们可以无限期地降低内存使用量。请注意，它对中间激活没有帮助，为此我们可以使用激活检查点和梯度累积，正如我们在前几章中看到的那样。

**让我们总结一下我们迄今为止对 DP 和 ZeRO 的探索历程：我们已经看到，我们可以通过 DP 显著提高训练吞吐量，只需通过添加更多模型副本来扩展训练即可。借助 ZeRO，我们可以通过将参数、梯度和优化器状态分片到 DP 中来训练通常无法放入单个 GPU 的模型，同时产生少量通信成本。**

但是，这里有一个限制，DP 仅在模型的某一层适合单个 GPU 时才有效，并且 ZeRO 只能划分参数、梯度和优化器状态，​​而不能划分激活内存！我们从[激活内存讨论](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#memory_usage_in_transformers)中回想一下，这部分内存随序列长度和批处理大小而变化。当然，我们可以限制这些，但实际上，我们不想受到硬件的限制，只能使用较短的序列长度进行训练。

![](img1/Pasted%20image%2020250309165151.png)

为了克服这个问题，是时候探索一种新的正交并行轴 - 张量并行 (TP)。与依赖大量参数通信的 ZeRO3 不同，TP 建议跨设备分片参数、梯度、优化器状态和激活，而无需 GPU 之间进行任何模型参数通信。

## 张量并行

因此，我们已使用 ZeRO 对模型的参数、梯度和优化器状态进行了分片，但一旦激活内存超出内存预算，我们就会达到极限。欢迎使用张量并行 (TP)，这是一种对权重、梯度和优化器状态以及激活进行分片的方法，无需在计算之前将它们全部收集起来。这似乎是一场梦！让我们首先看看张量并行如何与简单的矩阵乘法一起工作。

张量并行利用了矩阵乘法的数学特性 A×B 。为了了解其工作原理，让我们来看一下实现这种并行化的两个基本方程：

![](img1/Pasted%20image%2020250309165510.png)

这意味着我们可以通过 1) 分别乘以 B 的每一列或 2) 分别乘以每一行并合并结果来计算矩阵乘积。在神经网络中，矩阵乘法通常以以下格式表示： X×W .

让我们看看如何并行化此操作！在张量并行中，张量将沿特定维度拆分为 N 个分片，并分布在 N 个 GPU 上。矩阵可以拆分为列部分或行部分，从而实现行和列并行。我们将在下文中看到，选择行或列分片将需要不同的通信原语。

我们的第一个选择是使用列式分片（也称为**_列线性分片_**）：我们将完整的输入矩阵复制到每个工作器，这需要一项称为**_广播_**的操作，并将权重矩阵拆分为列。然后将输入与部分权重矩阵相乘，最后使用**_全聚集_**操作将结果合并。

![](img1/Pasted%20image%2020250309165639.png)

以下是列式张量并行的代码实现：

```python
class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.output_size_per_partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        if self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias) 
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias) 
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)
        return output
```

第二种选择称为按行分片（也称为**_行线性分片_**）：细心的读者可能会猜到，行线性分片意味着我们将权重矩阵拆分成行块。但是，这也要求我们拆分输入，这需要**_分散_**操作，而不是像列线性分片中使用的广播。每个工作器上的结果已经处于正确的形状，但需要对最终结果进行求和，因此在这种情况下需要全归约操作。

我们在这里看到第四个分布式原语：**_散射_** scatter！

![](img1/Pasted%20image%2020250309165748.png)

以下是按行张量并行的实现：

```python
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # X_i * W_i^T + b
        output_parallel = F.linear(x, self.weight)
        # All-reduce across all the partitions.
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output if self.bias is None else output + self.bias
```

现在我们有了 TP 的基本构建块，让我们看看如何在 Transformer 层内有效地组合它们！

### Transformer 块中的张量并行

为了制定一个可遵循的策略，让我们从一个玩具示例转向一个真正的模型构建块。Transformer 模型由两个主要构建块组成：前馈层 (MLP) 和多头注意力 (MHA)。我们可以将张量并行应用于两者。

前馈部分可以通过“列线性”后接“行线性”来实现并行化，这相当于广播复制输入，并在前向执行全归约。请注意，在实际训练中不需要广播，因为我们可以确保输入已在 TP 等级之间同步。此设置比从“行线性”开始，然后是“列线性”更有效，因为我们可以跳过两个拆分操作之间的中间全归约。

![](img1/Pasted%20image%2020250309165903.png)

现在我们已经找到了 Transformer 前馈部分的有效模式，让我们看一下多头注意力模块（MHA）。

我们通常可以采用类似的方法，其中 Q、K 和 V 矩阵以列并行方式拆分，输出投影沿行维度拆分。对于多头注意力，列并行方法具有非常自然的解释：每个工作者计算单个或部分头的注意力。同样的方法也适用于[**_多查询_**(MQA)](https://arxiv.org/abs/1911.02150)或[**_分组查询注意力_**(GQA)，](https://arxiv.org/abs/2305.13245)其中键和值在查询之间共享。

但值得注意的是，张量并行度不应超过 Q/K/V 头的数量，因为我们需要每个 TP 等级的完整头（否则我们无法在每个 GPU 上独立计算注意力，并且需要额外的通信操作）。如果我们使用 GQA，TP 度实际上应该小于 K/V 头的数量。例如，LLaMA-3 8B 有 8 个 Key/Value 头，因此张量并行度最好不超过 8。如果我们对此模型使用 TP=16，我们将需要在每个 GPU 上复制 K/V 头并确保它们保持同步。

![](img1/Pasted%20image%2020250309165937.png)

最后请注意，Tensor Parallelsim 仍然不是训练的灵丹妙药。我们在模型的计算路径中直接添加了几个分布式通信原语，因此很难完全隐藏/与计算重叠（就像我们在 ZeRO 中所做的那样），我们的最终性能将是计算和内存增益与增加的通信开销之间权衡的结果。让我们来说明一下：

![](img1/Pasted%20image%2020250309170034.png)

通过查看张量并行 MLP 中的操作时间线（同样适用于 Attention），我们可以更好地理解其中涉及的权衡。在每个解码器层的前向中，我们与 AllReduce 操作达到同步点，该同步点不能与计算重叠。在应用最终的 LayerNorm 之前，这种_暴露的通信_开销对于跨张量并行等级合并部分结果是必不可少的。

张量并行确实有助于减少矩阵乘法的激活内存，因为中间激活在 GPU 之间分片。但是，我们仍然需要收集 LayerNorm 等操作的完整激活，这意味着我们无法获得应有的全部内存优势。此外，TP 引入了重要的通信要求，这些要求严重依赖于网络基础设施。无法完全隐藏这个特定的 AllReduce 背后的计算意味着它直接增加了前向传播的关键路径。

让我们在扩展 TP 度时更好地看看权衡：

![](img1/Pasted%20image%2020250309170105.png)

虽然增加 TP 会导致每个 GPU 的吞吐量降低（左），但它可以处理更大的批量大小（右），说明了分布式训练中计算效率和内存可用性之间的权衡。

在实践中，如上图左图所示，当我们扩展到 8 个 GPU 以上时，张量并行的通信开销变得尤为明显。虽然单个节点内的张量并行可以利用快速的 NVLink 互连，但跨节点需要较慢的网络连接。我们观察到从 TP=8 移动到 TP=16 时出现显著下降，从 TP=16 移动到 TP=32 时下降幅度更大。在更高的并行度下，通信开销变得如此之高，以至于它很快就会占据计算时间的主导地位。

话虽如此，张量并行通过在 GPU 上分配模型参数、梯度、优化器状态和激活（在一定程度上）为内存使用提供了重要的好处。让我们在 70B 参数模型上检查一下这种影响：

![](img1/Pasted%20image%2020250309170136.png)

增加张量并行性可减少每个 GPU 上模型参数、梯度和优化器状态所需的内存，从而使我们可以开始在 8 个 GPU 的单个节点上拟合大型模型。

有没有办法从这项技术中获得更多好处？我们已经看到，层规范化和 dropout 仍然需要在每个 GPU 上收集完整激活，这在一定程度上抵消了内存节省。我们可以通过找到并行化这些剩余操作的方法做得更好。

关于张量并行训练中的层规范化，有一点值得注意：由于每个 TP 等级在全聚集之后都会看到相同的激活，因此层规范权重实际上不需要全归约来在后向传递之后同步其梯度。它们自然会在等级之间保持同步。但是，对于 dropout 操作，我们必须确保在 TP 等级之间同步随机种子以保持确定性行为。

接下来让我们探索张量并行的一个小而自然的扩展，称为**序列并行，**它就是这样做的。

### 序列并行

**序列并行 (SP)**涉及拆分模型中未由张量并行 (TP) 处理的部分（例如 Dropout 和 LayerNorm）的激活和计算，而是沿着输入序列维度而不是跨越隐藏维度。

术语“序列并行”有点过分：本节中的序列并行与张量并行紧密相关，适用于 dropout 和层规范操作。但是，当我们转向更长的序列时，注意力计算将成为瓶颈，这就需要 Ring-Attention 等技术，这些技术有时也称为_序列并行，_但我们将其称为_上下文并行_以区分这两种方法。因此，每次看到序列并行时，请记住它与张量并行一起使用（与可以独立使用的上下文并行相反）。

这是必要的，因为这些操作需要访问完整的隐藏维度才能正确计算。例如，LayerNorm 需要完整的隐藏维度来计算均值和方差：

![](img1/Pasted%20image%2020250309170249.png)

因此，尽管这些操作在计算上很便宜，但它们仍然需要大量的激活内存，因为它们需要完整的隐藏维度。 SP 允许我们通过沿序列维度进行拆分，在 GPU 之间分担此**内存**负担。

实际上，我们将从左图到右图：

![](img1/Pasted%20image%2020250309170311.png)

该图显示了我们如何使用不同的集体操作（标记为“f”和“g”）在张量并行和序列并行区域之间转换。关键挑战是高效管理这些转换，同时保持较低的内存使用率并保持正确性。

在前向传递中：

- “f” 是无操作 (no operation)，因为激活已在各个等级之间重复
- “f*” 是一个 all-reduce，用于同步激活并确保正确性

在反向传播中：

- “f*” 是无操作，因为梯度已经在各个等级之间重复
- “f” 是一个 all-reduce，用于同步梯度

这些操作“f”和“f*”被称为**共**轭对，因为它们相互补充——当一个在前向中为无操作时，另一个在后向中为全归约，反之亦然。

对于序列并行 (SP)，我们使用标记为“g”和“g*”的不同操作。具体来说，我们避免在 SP 区域使用 all-reduce，因为这需要收集完整激活并增加我们的峰值内存使用量，从而违背 SP 的目的。

那么这里到底发生了什么？正如一位名人所说，让我们一步一步来：

**初始 LayerNorm（SP 区域）**

- 输入张量 X1_和 X2_ (b,s/2,h) 进入 LayerNorm，已跨序列维度分割
- 每个 GPU 在其序列块上独立计算 LayerNorm，并给出 Y1_和 Y2_

**第一次转换 (SP → TP)**

- “g”操作（全聚集）将 Y1_和 Y2_合并回完整序列长度
- 由于列线性需要全隐藏维度 h，因此恢复 Y (b,s,h)

**第一条线状图（TP 区域）**

- A1 是列线性的，因此它沿隐藏维度分割 Y
- GeLU 在每个 GPU 上独立应用
- Z1* 是 (b,s,h/2)

**第二线性（TP 区域）**

- B1 是行线性的，因此它恢复了隐藏维度
- W1 是 (b,s,h)

**最终过渡（TP → SP）**

- “g*”操作（减少散射）减少了先前行的线性正确性，同时沿序列维度散射
- W1* 是 (b,s/2,h)

序列并行的一个关键优势是它减少了我们需要存储的最大激活大小。仅在张量并行中，我们就必须在各个点存储形状为 (b,s,h) 的激活。但是，使用序列并行时，最大激活大小会减少到 （b⋅s⋅h）/tp​ ，因为我们总是沿着序列或隐藏维度进行拆分。

跟踪 TP 和 TP/SP 中不同分片的所有部分有点困难 —— 相信我们，我们也发现很难映射，因此我们制作了这个小表来总结激活（又名`hidden_states` ）形状在前向传递过程中如何在隐藏维度 h 和序列维度 s 上变化：

![](img1/Pasted%20image%2020250309170731.png)

对于嵌入层：

![](img1/Pasted%20image%2020250309170757.png)

通过使用序列并行，我们可以实现更大的激活内存节省，使我们能够将批处理大小和序列长度推得比仅使用张量并行所能达到的程度更高。让我们看看这对我们之前的 70B 模型示例意味着什么：

![](img1/Pasted%20image%2020250309170813.png)

我们可以看到，我们再次大幅降低了每个 GPU 的最大内存使用量，使我们能够适应 TP/SP=16 的 16k 个 token 的序列长度，这比普通的 TP 情况有所改进！（正如我们在上一节中看到的那样，TP=16 仍然有点大，但我们将在下一节中看到如何改进这一点）。

您可能会问自己一个问题，即使用 TP+SP 是否比 vanilla TP 需要更多通信？答案是既是也不是。在 vanilla TP 的前向传递中，每个 transformer 块有两个 all-reduce，而在 SP 中，每个 transformer 块有两个 all-gather 和两个 Reduce-scatter。因此，SP 的通信操作数量是 TP 的两倍。但由于 all-reduce 操作可以分解为 all-gather + Reduce-scatter（请参阅附录中的[“快速关注 Ring AllReduce”](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#a_quick_focus_on_ring_allreduce)部分），因此它们在通信方面实际上是等效的。反向推理也是一样，因为我们只使用每个操作的共轭（no-op ↔ allreduce 和 allgather ↔ Reducescatter）。

如果你一直密切关注，你会注意到我们谈论的是每层中的 4 个通信操作（2 个用于 Attention，2 个用于 MLP）。使用张量 + 序列并行时，MLP 分析如下所示：

![](img1/Pasted%20image%2020250309170843.png)

与 vanilla TP 一样，TP+SP 不能轻易与计算重叠，这使得吞吐量严重依赖于通信带宽。同样，与 vanilla TO 一样，TP+SP 通常仅在节点内完成（将 TP 度保持在每节点 GPU 数量以下，例如 TP≤8）。

我们可以对这种通信开销在扩大张量并行性时如何变得越来越成问题进行基准测试。让我们测量一下，在为具有 4096 个序列长度的 3B 模型扩展 TP 和 SP 时，吞吐量和内存利用率：

![](img1/Pasted%20image%2020250309170906.png)

同样，计算效率（左）和内存容量（右）之间存在权衡。虽然更高的并行度可以通过减少激活内存来处理更大的批量大小，但它们也会降低每个 GPU 的吞吐量，尤其是在超过与每个节点的 GPU 数量相对应的阈值时。

让我们总结一下我们的观察：

- 对于这两种方法，我们注意到从 TP=8 移动到 TP=16 时性能下降幅度最大，因为那时我们从仅在单个节点内通信（NVLink）转变为节点间通信（EFA）
- 使用 TP 和 SP 时，激活中的内存节省有助于我们适应比单独使用 TP 更大的批次

**我们已经看到 TP 如何通过沿隐藏维度分割注意力和前馈操作来帮助我们在多个 GPU 上分片激活，以及 SP 如何通过沿序列维度分割来自然补充剩余操作。**

由于 SP 区域中的 LayerNorms 作用于序列的不同部分，因此它们的梯度在 TP 等级之间会有所不同。为了确保权重保持同步，我们需要在反向传播过程中降低它们的梯度，类似于 DP 确保权重保持同步的方式。然而，由于 LayerNorm 的参数相对较少，因此这只是一个很小的通信开销。

然而，TP 和 SP 有两个限制：1）如果我们缩放序列长度，激活记忆仍会在 TP 区域爆炸式增长；2）如果模型太大而无法适应 TP=8，那么我们会看到由于节点间连接而导致的速度大幅下降。

我们可以用上下文并行来解决问题 1)，用管道并行来解决问题 2)。我们先来看看上下文并行！

## 上下文并行

借助张量并行和序列并行，我们可以显著降低每个 GPU 的内存需求，因为模型权重和激活都分布在 GPU 上。但是，当在越来越长的序列上训练模型时（例如，当扩展到每个序列 128k 或更多标记时），我们仍然可能超出单个节点上的可用内存，因为当我们在 TP 区域内时，我们仍然必须处理完整的序列长度。

此外，即使我们对激活进行完全重新计算（这会带来约 30% 的大量计算开销），我们仍然需要在内存中保存层边界处的一些激活，这些激活会随序列长度线性扩展。让我们来看看上下文并行如何帮助我们：

![](img1/Pasted%20image%2020250309171104.png)

上下文并行的核心思想是将与序列并行方法类似的思想（即沿序列长度分割）应用于我们已经应用张量并行的模块。因此，我们将沿两个维度分割这些模块，从而也减少序列长度的影响。毕竟我们已经讨论过了，你会发现这种方法非常直观，但是……这里有一个技巧，所以要保持清醒！

对于上下文并行；就像序列并行一样，我们将沿着序列维度分割输入，但现在我们将这种分割应用到整个模型，而不是像我们之前对张量+序列并行所做的那样，只应用模型的序列并行区域。

拆分序列不会影响大多数模块，例如 MLP 和 LayerNorm，其中每个 token 都是独立处理的。它也不需要像 TP 那样昂贵的通信，因为只拆分输入而不是权重矩阵。就像数据并行一样，在计算梯度后，会启动全归约操作以同步上下文并行组中的梯度。

不过有一个重要的例外，因为我们需要特别注意**注意力模块**（哈哈……双关语：D）。在注意力模块中，每个标记都需要访问**所有**其他序列标记中的键/值对，或者在因果注意力的情况下，至少关注每个先前的标记。

由于上下文并行沿序列维度在 GPU 之间分割输入，因此注意力模块将需要 GPU 之间进行充分通信以交换必要的键/值数据。

如果我们天真地这样做，这听起来会非常昂贵。有没有办法可以更高效、更快速地做到这一点！值得庆幸的是，有一种方法可以有效地处理这种键/值对通信，这种核心技术称为_Ring Attention_ 。

	上下文并行与 Flash Attention 具有一些概念上的相似性（有关更多详细信息，请参阅下文） - 这两种技术都依赖于在线 softmax 计算来减少内存使用量。Flash Attention 专注于在单个 GPU 上优化注意力计算本身，而上下文并行则通过将序列分布在多个 GPU 上来实现内存减少。

### 发现 Ring Attention

在这个注意力机制的实现中，每个 GPU 首先启动一个异步通信操作，将其键/值对发送给其他 GPU。在等待其他 GPU 数据的同时，它会计算内存中已有数据部分的注意力分数。理想情况下，在此计算完成之前，会从另一个 GPU 收到下一个键/值对，这样 GPU 就可以在完成第一次计算后立即开始下一轮计算。

让我们来说明一下。假设我们有 4 个 GPU 和 4 个 token 的输入。最初，输入序列沿序列维度均匀分割，因此每个 GPU 将只有一个 token 及其对应的 Q/K/V 值。Leyt 表示 Q1、K1 和 V1 代表第一个 token 的查询、键和值，它们位于第一个 GPU 上。注意力计算将需要 4 个时间步骤才能完成。在每个时间步骤中，每个 GPU 执行以下三个连续操作：

1. 除了最后一个时间步骤以外，以非阻塞方式将“当前键和值”发送到下一台机器，以便我们可以在此步骤完成之前开始下一步
2. 在本地计算已有的“当前键和值”的注意力分数，这通常涉及执行 Softmax(QK^T/sqrt(d))∗V 。
3. 等待从前一个 GPU 接收键和值，然后回到步骤 1。其中“当前键和值”现在是刚从前一个 GPU 接收的键/值。

我们执行这3个步骤四次，以完成注意力计算。

以下动画显示了使用 4 个 GPU 的整个过程：

![ring-attention.gif](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ring-attention.gif)

从这个动画中你可能已经明白为什么作者选择将这种方法称为“Ring Attention”。

然而，有一个大问题，那就是 Ring Attention 的简单实现会导致 GPU 之间因果注意力矩阵的形状而产生一些严重的不平衡。让我们通过考虑带有因果注意力掩码的注意力得分矩阵来看一下 SoftMax 计算：

![](img1/Pasted%20image%2020250309171546.png)

SoftMax 是按行计算的，这意味着只要 GPU 收到一行的所有标记，就可以计算它。我们看到 GPU1 可以立即计算它，因为它从标记 1-4 开始，并且 GPU1 实际上不需要从任何其他 GPU 接收任何信息。但是，GPU2 需要等待第二轮才能收到 1-4，从而获得标记 1-8 的所有值。此外，GPU1 似乎比所有其他 GPU 执行的工作要少得多。

让我们看看是否可以更好地平衡我们的计算：

### Zig-Zag 环注意力机制——一种平衡计算的实现

我们需要一种更好的方法来分配输入序列。这可以通过将 token 分配给 GPU 来实现，而不是完全按顺序分配，而是通过稍微混合顺序，这样我们就可以在每个 GPU 上很好地混合早期和晚期 token。这种方法称为 Zig-Zag 注意力.在这种新的安排下，注意力掩码将显示均匀的计算分布，但如果您计算彩色方块的数量，您会发现计算现在在所有 GPU 上是平衡的。

![](img1/Pasted%20image%2020250309172127.png)

同时我们还会看到，为了完成所有行，每个 GPU 都需要来自所有其他 GPU 的信息。

我们有两种常规方式来重叠计算和通信，要么通过执行常规全收集，同时重新分组每个 GPU 上的所有 KV（以 Zero-3 类型的方式），要么我们根据需要将它们从每个 GPU 逐一收集到每个 GPU：

![](img1/Pasted%20image%2020250309172157.png)

这两种实现之间的关键区别在于它们的通信模式和内存使用情况：

**1.AllGather 实现：**

- 所有 GPU 同时从所有其他 GPU 收集完整的键/值对
- 需要更多临时内存，因为每个 GPU 需要同时存储完整的键值对
- 通信只需一步即可完成，但内存开销较大

**2. All-to-All (Ring) 实现:  

- GPU 以环形模式交换 KV 对，每次一个块
- 内存效率更高，因为每个 GPU 只需要临时存储一个额外的块
- 通信是分散的，并且与计算重叠，尽管多个通信步骤会带来一些额外的基本延迟开销

All-to-All 方法通常以稍微复杂的通信模式为代价提供更好的内存效率，而 AllGather 方法更简单，但在注意力计算期间需要更多的临时内存。

现在我们已经了解了如何使用 TP 将模型拆分到一个节点来驯服大型模型，以及如何使用 CP 来驯服长序列的激活爆炸。

然而，我们仍然知道 TP 无法很好地跨节点扩展，那么如果模型权重不能轻易地适应 1 个节点，我们该怎么办？另一种并行度，即我们的第四种并行度，称为**管道并行度**，可以解决这个问题！

## 流水线并行

[模型并行训练](模型并行训练.md)

## 专家并行性

混合专家模型基本思想是，我们可以有多个并行模块，并通过其中一个或另一个路由标记以进行不同的处理，而不是每层只有一个前馈模块。

![](img/Pasted%20image%2020250309175727.png)

MoE 层的设计使得在专家维度上实现并行性变得非常容易，我们称之为**专家并行性**(EP)。由于前馈层完全独立，我们可以简单地将每个专家的前馈层放在不同的工作器上。与 TP 相比，它更轻量，因为我们不需要拆分矩阵乘法，我们只需要将 token 的隐藏状态路由到正确的专家。

在实践中，EP 通常与其他形式的并行性结合使用 - 例如数据并行性。这是因为 EP 仅影响 MoE 层，而不会对输入标记进行分片（与沿序列长度维度分片标记的上下文并行性不同）。这意味着如果我们仅使用 EP，我们的 GPU 将对所有非 MoE 块进行冗余计算。通过将 EP 与 DP 相结合，我们可以有效地在 GPU 上对专家和输入批次进行分片，如下方简化图所示：

![](img/Pasted%20image%2020250309175813.png)

在实践中，有一些技巧可以让 EP 高效工作，它们与模型设计密切相关。例如，DeepSeek-V3 在路由器中强制执行约束，确保每个令牌最多发送到 M 个节点（在他们的例子中是 4 个），以将令牌保持在单个节点上并减少通信开销。

## 5D 并行性概述

恭喜读者，您现在已经了解了可用于扩展模型训练的所有 5 种并行策略：

1. Data Parallelism (DP) – along the batch dimension  
    数据并行（DP）——沿着批次维度
2. Tensor Parallelism (TP) - along the hidden dimension  
    张量并行（TP）——沿着隐藏维度
3. Sequence and Context Parallelism (SP/CP) - along the sequence dimension  
    序列和上下文并行（SP/CP）——沿着序列维度
4. Pipeline Parallelism (PP) - along the model layers  
    管道并行性（PP）- 沿着模型层
5. Expert Parallelism (EP) - along the model experts  
    专家并行性（EP）- 沿着模型专家

以及可以与数据并行结合以减少内存的 3 种 ZeRO 策略：

1. ZeRO-1 – sharding optimizer states among the DP replicas  
    ZeRO-1 – DP 副本之间的分片优化器状态
2. ZeRO-2 – sharding optimizer states and gradients among the DP replicas  
    ZeRO-2 – DP 副本之间的分片优化器状态和梯度
3. ZeRO-3 – sharding optimizer states, gradients and parameters among the DP replicas  
    ZeRO-3 – DP 副本之间的分片优化器状态、梯度和参数

在这个阶段，您可能想知道的是所有这些并行性和 ZeRO 策略如何相互比较和交互。换句话说，我们应该使用哪些并有效地将它们结合起来，而我们应该将哪些策略分开？

让我们来看看相似之处和相互作用。我们首先将流水线并行性和 ZeRO-3 进行比较，因为它们有一些非常相似的相似之处，但也有一些重要的区别。

**流水线并行与 ZeRO-3 -** PP 和 ZeRO-3 都是将模型权重划分到多个 GPU 上并沿模型深度轴执行通信/计算的方法（例如，在 ZeRO-3 中，我们在计算时预取下一层）。这意味着在这两种情况下，全层操作都在每个设备上计算，而不是像 TP 或 EP 那样在子层单元上执行计算。

![](img/Pasted%20image%2020250309180025.png)

如您所见，ZeRO-3 和 PP 解决了相同的挑战，但涉及不同的方法，两者之间的选择取决于您决定将通信重点放在权重还是激活上。虽然它们可以组合使用，但在实践中通常不会这样做，因为这样做需要显着增加全局批处理大小以摊销通信成本，从而在全局批处理大小、模型大小、网络带宽和训练效率之间进行权衡。如果您决定将它们组合使用，则应将 ZeRO-3 配置为在一系列 PP 微批处理期间将权重保留在内存中，以尽可能减少不必要的通信开销。

另一方面，ZeRO-1 和 ZeRO-2 专注于优化器状态和梯度，可以轻松与管道并行性相结合，并与之互补。结合它们不会带来任何特别的新挑战。例如，DeepSeek-v3 的训练使用了 PP 与 ZeRO-1（原文如此）的结合。

**张量并行**（与序列并行）具有天然的互补性，可以与流水线并行和 ZeRO-3 相结合，因为它依赖于矩阵乘法的分布特性，允许在组合之前对权重和激活进行分片和独立计算。

![](img/Pasted%20image%2020250309180109.png)

我们不想仅将 TP 用于并行性的主要原因是，在实践中，TP 具有我们在前面几节中讨论过的两个限制：首先，由于其通信操作是计算关键路径的一部分，因此很难扩展到通信开销开始占主导地位的某个点之外。其次，与与模型无关的 ZeRO 和 PP 不同，TP 需要仔细处理激活分片 - 有时沿着隐藏维度（在 TP 区域），有时沿着序列维度（在 SP 区域） - 这使得正确实施更加麻烦，并且需要特定于模型的知识来确保始终采用正确的分片模式。

因此，在组合并行策略时，TP 通常用于高速节点内通信，而 ZeRO-3 或 PP 可用于跨越低速节点间通信的并行组，因为它们的通信模式需要的带宽较少（对于 PP）或更容易与计算重叠（对于 ZeRO-3）。组合这些技术时的主要考虑是针对每个并行维度将 GPU 有效地分组，以最大限度地提高吞吐量并最大限度地降低通信开销，同时注意 TP 的扩展限制。例如，为 TP 通信的 GPU 组应保留在节点内。

**上下文并行**和**专家并行**也有助于我们分片激活，可以看作是 TP 的补充。前者处理长序列，而后者支持分布式混合专家训练，两者可以结合在一起而不会出现任何特殊问题。

**上下文并行 (CP)** 通过在 GPU 上沿序列维度对激活进行分片，专门针对非常长序列的训练挑战。虽然大多数操作（如 MLP 和 LayerNorm）可以独立处理这些分片序列，但注意层需要通信，因为每个标记都需要访问完整序列中的键/值。正如我们在[CP 部分](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html?section=high_level_overview#context_parallelism)中看到的那样，这可以通过重叠计算和通信的环形注意模式有效地处理。在扩展到极端序列长度（128k+ 标记）时，CP 特别有价值，在这种情况下，即使使用完整的激活重新计算，单个 GPU 上的注意内存要求也会令人望而却步。

![](img/Pasted%20image%2020250309180213.png)

**专家并行 (EP)** 专门针对混合专家 (MoE) 模型的训练挑战，方法是将专业“专家”分片到 GPU 上，并在计算过程中动态地将令牌路由到相关专家。EP 中的关键通信操作是“全对全”操作，将令牌路由到其指定的专家并收集结果。虽然此操作会带来一些通信开销，但它可以显著扩展模型容量，因为每个令牌在推理（和训练）期间仅由总参数的一小部分处理。就分布式训练/推理而言，当模型扩展到大量专家时，跨 GPU 划分专家变得非常重要。

![](img/Pasted%20image%2020250309180237.png)

EP 和 DP 在输入处理方面的相似性就是为什么一些实现将专家并行视为数据并行的一个子组的原因，其关键区别在于 EP 使用专门的专家路由，而不是让所有 GPU 通过相同的模型副本处理输入。

让我们快速总结一下模型的子部分，其中一些不同的并行策略影响最大：

- 张量并行（和序列并行）通过分片权重和激活来影响整个模型的计算。
- 上下文并行主要影响注意层，因为注意层需要跨序列通信，而其他层在分片序列上独立运行。
- 专家并行性主要影响 MoE 层（取代标准 MLP 块），而注意力和其他组件则保持不变
- 流水线并行和 ZeRO 并不特别针对任何子模块或组件，但流水线并行中需要平衡模块和层，因此由于额外的嵌入层，第一层和最后一层通常会被区别对待。

![](img/Pasted%20image%2020250309180329.png)

在此摘要图中，您将找到单个 Transformer 层的激活和模块图示（在其 MoE 变体中）。我们还说明了并行性的各个方向以及我们在前面所有章节中讨论过的通信操作。

![](img/Pasted%20image%2020250309180337.png)

我们还可以并排展示每种策略的内存**节省情况**。我们将绘制不同序列长度以及选择性（顶部）和完全（底部）重新计算的结果，以便您了解它们如何与激活一起发挥作用：

![](img/Pasted%20image%2020250309180404.png)

让我们以高层次的视角来结束本节，了解所有这些技术、它们的主要基本思想和主要瓶颈：

| **Method  方法** | **Memory savings applies specifically on  <br>内存节省特别适用于**            | **Parallel/sharding dimension  <br>并行/分片维度**      | **Disadvantage  缺点**                                                           |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------ |
| DP             | Activations (reduce local batch size)  <br>激活（减少本地批次大小）              | Batch  批                                          | Limited by max batch size  <br>受最大批次大小限制                                       |
| PP             | Model parameters  模型参数                                               | Model layers  模型层                                 | Idle bubble and complex schedules  <br>空闲时间与复杂日程安排                             |
| TP/SP          | Model parameters and activations  <br>模型参数和激活                        | Hidden dimension / Sequence length  <br>隐藏维度/序列长度 | Requires high bandwidth communication  <br>需要高带宽通信                             |
| CP             | Activations  激活                                                      | Sequence length  序列长度                             | Add communication overhead in attention modules  <br>在注意力模块中添加通信开销             |
| EP             | Experts parameters  专家参数                                             | Expert dimension  专家维度                            | Requires MoE layers, add routing communication overhead  <br>需要 MoE 层，增加路由通信开销 |
| ZeRO-1         | Optimizer states  优化器状态                                              | Sharded among DP replicas  <br>在 DP 副本之间进行分片      | Params communication overhead  <br>参数通信开销                                      |
| ZeRO-2         | Optimizer states and gradients  <br>优化器状态和梯度                         | Sharded among DP replicas  <br>在 DP 副本之间进行分片      | Params communication overhead  <br>参数通信开销                                      |
| ZeRO-3         | Optimizer states, gradients, and model parameters  <br>优化器状态、梯度和模型参数 | Sharded among DP replicas  <br>在 DP 副本之间进行分片      | Params communication overhead  <br>参数通信开销                                      |
显然，这些技巧都不是魔法扩展的灵丹妙药，我们经常需要以某种方式将它们结合起来。我们能否真正想出一些规则来帮助我们找到一个好的起点来选择并结合它们？这将是我们下一节的主题。

## 寻找最佳训练配置

我们在上一节中稍微提到了这一点，但现在让我们逐步详细了解一个可能的决策过程，同时记住，您总是必须运行一些实验才能根据计算集群的各种物理属性、网络带宽、每个节点的 GPU、每个 GPU 的内存等找到最终的最佳设置。

### 步骤 1：在内存中拟合训练步骤

首先，我们需要弄清楚如何在 GPU 上安装完整的模型实例。一般有两种情况。

**GPU 丰富的情况🤑** - 当你有大量可用的 GPU 时：

- 对于 10B 参数以下的模型，您可以使用单一并行技术，例如张量并行或跨 8 个 GPU 进行完全重新计算的 ZeRO-3/DP
- 对于需要 8 个以上 GPU 的 10B-100B 参数之间的模型，您有以下几种选择：

- 将张量并行 (TP=8) 与流水线并行相结合
- 将张量并行 (TP=8) 与数据并行 (ZeRO-3) 相结合
- 仅使用 ZeRO-3（即仅纯数据并行）

- 在 512+ GPU 规模下，纯数据并行/ZeRO-3 将因通信成本而开始变得效率低下 - 更好的方法是将 DP 与张量或管道并行相结合
- 在 1024+ GPU 规模下，建议的设置是张量并行 TP=8、数据并行（ZeRO-2）和流水线并行


我们目前专注于拟合单个实例 - 即使我们可以使用 DP for ZeRO 来实现这一目标 - 但我们只对它与 ZeRO-3 一起使用时提供的模型参数内存节省感兴趣。

特别注意事项：

- 对于非常长的序列，您可能需要跨节点添加上下文并行 (CP)。
- 对于混合专家架构，您将有利地使用跨节点的专家并行性 (EP)。


**GPU 匮乏的情况**- GPU 资源可能不足的情况：

- 您可以启用完全激活重新计算，以用一些计算来换取内存（并且训练速度稍慢）。
- 您可以增加梯度积累，以使用有限的内存来处理更大的批次。


现在我们有了第一个模型实例训练，我们需要确保我们有正确的批量大小。

### 第 2 步：实现目标全局批次大小  

根据步骤 1 中微批次大小和 DP 的设定，我们当前的批次大小可能太小或太大。现在是时候达到我们的目标批次大小了。

为了增加我们当前的全局批量大小：

- 我们可以扩大数据并行或梯度积累步骤
- 对于长序列，我们可以利用上下文并行

为了减少当前的全局批次大小：

- 我们可以减少数据并行性，转而采用其他并行策略
- 对于长序列，我们可以减少上下文并行

好的，现在我们已经在模型大小和批量大小方面按照我们想要的一般配置运行模型，但是我们是否以最快的方式训练它？现在让我们开始尽可能优化吞吐量。

### 步骤 3：优化训练吞吐量  

因此，我们希望确保训练尽可能快地运行，以便我们所有宝贵的 GPU 始终得到充分利用。只要内存和通信不是瓶颈，我们就可以尝试以下方法：

- 扩大张量并行度（使用快速节点内带宽），直到达到接近节点大小的程度，这样我们就可以减少其他并行度
- 使用 ZeRO-3 增加数据并行性，同时保持目标批次大小
- 当数据并行通信开始成为瓶颈时，转换到使用管道并行
- 尝试逐个扩展不同的并行性
- 尝试几种微批量大小（mbs）以达到最大 GBS、模型大小、计算和通信之间的最佳平衡。

### 对数千种配置进行基准测试

现在我们已经逐步介绍了步骤，让我们在现实生活中实现这个搜索过程。

您将在[nanotron](https://github.com/huggingface/nanotron)存储库中找到几个脚本，您可以使用它们来运行我们上面讨论的所有实验，并能够在现实生活中对您自己的模型和集群进行基准测试。

实际上，我们对**数千个分布式配置**运行了基准测试，涵盖了我们上面讨论过的每个模型大小，以及大量的集群配置（即 1-64 个 8xH100 节点），我们可以尝试这些配置来产生本书迄今为止涵盖的结果。

现在让我们退一步来收集和分析所有基准测试的结果，看看除了理论之外，我们是否真的可以在真实数据上发现各种配置之间的相互影响。

以下所有基准测试均以 4096 的序列长度和 1M 个 token 的全局批处理大小进行。我们收集了每个模型和集群大小的所有顶级配置，并将它们绘制在以下热图中：

![](img/Pasted%20image%2020250309180855.png)

热图可视化显示了不同模型大小和计算节点数（每个节点有 8 个 GPU）的最佳训练配置。对于每个组合，配置详细信息包括数据并行 (DP)、张量并行 (TP)、管道并行 (PP)、梯度累积步骤 (GAS)、微批次大小 (MBS) 和 ZeRO 优化阶段。颜色强度表示模型 FLOP 利用率 (MFU)，颜色越亮表示效率越高。

从这个高级可视化中，我们可以得出几个重要的见解：

首先，随着节点数量的增加（并行度更高），我们观察到效率下降。这种影响对于较小的模型尤其明显，因为较小的模型具有较低的计算与模型大小比率。虽然我们通常可以通过增加批处理大小来弥补模型大小较小的问题，但我们受到 1M 的全局批处理大小限制的限制。

其次，更大的模型带来了不同的挑战。随着模型大小的增加，内存需求大幅增加。这会导致节点数减少的两种情况：要么模型根本无法拟合，要么勉强拟合但由于接近 GPU 内存限制而运行效率低下（例如，参见 4 个节点上的 80B 参数模型训练）。

最后，我们的基准测试表明，性能在很大程度上取决于实现质量。当我们首次实现这两种并行策略时，张量并行 (TP) 的表现优于流水线并行 (PP)。在优化我们的 PP 代码后，它成为了更快的选择。现在我们正在改进 TP 实现中的通信重叠，我们预计它将重新获得性能领先地位。

### 基准测试的经验教训

我们写这本书的目的不仅是讨论理论和实现，还要提供实际的数据点。因此，计划很简单：让我们为每个模型和多个集群大小（即 1-64 个 8xH100 节点）运行所有可能的分布式配置。即使排除了不可能的配置，我们仍然需要运行数千次实验。

从理论上看，这听起来很容易：我们可以轻松地在集群上启动大量作业。然而，当我们启动第一批实验时，麻烦就开始了：

- PyTorch 进程有时无法正确清理
- Slurm 作业管理器会强制终止作业，导致节点故障
- 简单的基准测试本应只需几分钟，但却要花上几个小时
- 有些工作会无限期地搁置

在有限的时间内运行所有实验需要额外的工程设计，我们最终在以下方面花费了大量的时间：

- 最小化集群重启时间并优化空闲时间
- 分析详细的 NCCL 调试日志
- 了解内存使用模式和 CUDA 内存分配器行为
- 提高多节点的流水线并行性能

这些挑战值得我们用自己的故事来讲述，但它们也教会了我们关于分布式训练基础设施复杂性的宝贵经验。理论上看似简单的事情在实践中往往需要仔细关注许多变化的部分。

在实践中重现理论结果具有挑战性，尤其是在生产训练代码有限的情况下。通过[nanotron](https://github.com/huggingface/nanotron)和[picotron](https://github.com/huggingface/picotron)等开源项目，我们希望能够帮助使分布式训练技术更易于访问，并在简单高效的代码库上进行协作，帮助研究人员和从业者充分利用他们的硬件资源。

至此我们对 5D 并行分布方法的深入研究就结束了。

退一步来说，我们迄今为止的讨论往往依赖于一个关键假设——计算和通信可以在 GPU 上有效重叠，而不会对计算吞吐量产生任何影响。现实情况则更加微妙。当使用 NCCL send/recv 等常见通信原语时，我们面临着计算和通信资源之间的隐藏争用，因为通信内核通常会使用用于计算的相同 GPU 流式多处理器 (SM)，当通信与计算重叠时会导致吞吐量下降。为了真正优化我们的分布式训练，我们需要更深入地研究 GPU 架构本身。

是时候关灯并激活 CUDA 模式了！

## 深入 GPU – 融合、线程、混合

到目前为止，我们的讨论主要集中在模型操作的高级组织上。我们在各种加速器上移动计算，同时考虑到一般的内存限制和计算单元的高级调度。

但这忽略了我们可以在较低级别进行的所有优化，即通过仔细了解我们的模型操作在每个 GPU 上是如何调度和执行的。

本节将深入探讨 GPU 架构的更多细节，特别是 NVIDIA 的 GPU 架构，但一般思想通常可以在类似的加速器单元上重复使用。

在介绍 Flash-Attention 革命之前，我们将简要解释 GPU 的组织方式、如何有效地在 GPU 上调度工作负载，最后解释如何在 GPU 上有效地使用各种精度。

### GPU 入门

一般来说，GPU 的组织结构非常分层。在本入门指南中，我们将讨论概念层面的内容，这对于我们接下来的演示是必要的。

在计算方面，GPU 由一组称为**流式多处理器**(SM) 的计算单元组成。每个 SM 包含并控制一组流式处理器，也称为核心。例如，Nvidia H100 GPU 有 132 个 SM，每个 SM 有 128 个核心，总​​共有 16,896 个核心（有关详细信息，请参阅[张量核心的文档](https://resources.nvidia.com/en-us-tensor-core)），每个核心都可以同时处理多个线程。

![](img/Pasted%20image%2020250309181225.png)

内存方面也是高度分层的，具有多层缓存和内存：**寄存器**是最小的单位，在执行期间对线程是私有的；**共享内存**和**L1 缓存**在单个 SM 上运行的线程之间共享；更上一层楼的是所有 SM 共享的**L2 缓存**；最后是**全局内存**，它是 GPU 上最大的内存（例如，宣传的 H100 为 80 GB），但访问和查询速度也是最慢的。

![](img/Pasted%20image%2020250309181251.png)

GPU 的目标是利用这种计算/内存的分层组织，在 GPU 核心上并行运行尽可能多的工作负载。

要运行内核，您还需要一个特定的代码部分，称为**主机代码**，它在**CPU/主机**上执行，负责准备数据分配以及加载数据和代码。

```python
// Host code                
void vecAdd(float* h_A, float *h_B, float *h_c, int n) {
    // Allocate vectors in device memory
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}// 主机代码
void vecAdd(float* h_A, float*h_B, float*h_c, int n) {
// 在设备内存中分配向量
int 大小 = n * sizeof(float);
浮动*d_A，*d_B，*d_C；
cudaMalloc（＆d_A，大小）；
cudaMalloc（＆d_B，大小）；
cudaMalloc（＆d_C，大小）；

// 将向量从主机内存复制到设备内存
cudaMemcpy（d_A， h_A，大小，cudaMemcpyHostToDevice）；
cudaMemcpy（d_B， h_B，大小，cudaMemcpyHostToDevice）；

// 调用内核
int每个块的线程数 = 256;
int 每网格块数 =
（N + 每块线程数 - 1）/每块线程数；
向量添加<< >>(d_A，d_B，d_C，N)；

// 将结果从设备内存复制到主机内存
// h_C 包含主机内存中的结果
cudaMemcpy（h_C， d_C， 大小， cudaMemcpyDeviceToHost）；

// 释放设备内存
释放（d_A）；
CUDA释放（d_B）；
CUDA释放（d_C）；
}
```

内核一般按如下方式安排：

- 线程被分组为大小为 32 的**warp。warp**中的所有线程都经过同步，以同时执行指令，但针对数据的不同部分。
- **线程束**被分组为更大、大小更灵活的**块**（例如大小为 256），每个块仍分配给单个 SM。SM 可以并行运行多个块，但是，根据资源情况，并非所有块都可以立即分配执行，有些块可能会被列入等待资源的候补名单。

从这些细节中要记住的主要一点是，存在各种大小和分配限制（各种内存的大小、包装中的并发块和线程的数量），需要将这些限制考虑在内才能以最有效的方式使用 GPU 架构。

大多数情况下，您不需要降低到这种精度水平，幸运的是，您可以重用社区其他成员准备的内核和代码。但无论如何，我们都想为您提供有关如何开始使用内核的入门知识！

### 如何利用内核来提高性能？

如果您希望添加缺少优化内核的新操作或加速现有的 PyTorch 函数，从头开始编写内核似乎是最直接的途径。但是，从头开始创建高性能 CUDA 内核需要丰富的经验和陡峭的学习曲线。通常，更好的入门方法是利用`torch.compile` ，它通过捕获您的操作并在 triton 中生成较低级别的高性能内核来动态优化 PyTorch 代码。

假设你想为名为“指数线性单元”的激活函数编写一个内核：

![](img/Pasted%20image%2020250309181418.png)

您可以从简单的 pytorch 实现开始，然后在顶部添加`@torch.compile`装饰器：

```python
@torch.compile
def elu(x, alpha=1.0):
    return torch.where(x < 0, alpha * (torch.exp(x) - 1), x)
```

编译版本和非编译版本之间的区别非常明显，尤其是考虑到我们只添加了一个装饰器。下图说明了这一显著差异（N 是列数）：

![](img/Pasted%20image%2020250309181457.png)

但是，如果这种性能提升还不够，您可以考虑实现 Triton 内核。作为起点，您可以查看由 @torch.compile 生成的 triton 内核。为此，您只需将环境变量`TORCH_LOGS`设置为`"output_code"` ：

```bash
export TORCH_LOGS="output_code"
```

一旦使用`@torch.compile`装饰器运行 Python 脚本，它将生成并输出相应的 Triton 内核，在本例中为：

```python
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 < tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = 1.0
    tmp5 = tmp3 - tmp4
    tmp6 = tl.where(tmp2, tmp5, tmp0)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
```

为了增强可读性，我们可以修改变量名称、添加注释并进行细微调整（或者请别人帮我们做），如下所示：

```python
@triton.jit
def elu_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create an array of indices for this block
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)[:]
    # Create a mask to ensure only valid indices are processed
    valid_mask = block_indices < num_elements
    # Load input values from the input pointer based on valid indices
    input_values = tl.load(input_ptr + block_indices, valid_mask)
    # Define the ELU parameters
    zero_value = 0.0  # Threshold for ELU activation
    negative_mask = input_values < zero_value
    exp_values = tl.math.exp(input_values)
    # Define the ELU output shift
    one_value = 1.0
    shifted_exp_values = exp_values - one_value

    output_values = tl.where(negative_mask, shifted_exp_values, input_values)

    # Store the computed output values back to the output pointer
    tl.store(output_ptr + block_indices, output_values, valid_mask)
```

这里， `tl.program_id(0)`提供了一个唯一的块 ID，我们用它来确定该块将处理哪个数据部分。使用此块 ID， `block_start`计算每个块部分的起始索引，而`block_indices`指定该部分`valid_mask`的索引范围。valid_mask 确保只处理`num_elements`内的索引，使用`tl.load`安全地加载数据。然后应用 ELU 函数，根据值是否为负数来修改值，并使用`tl.store`将结果写回内存。

当我们使用`triton.testing.Benchmark`对生成的内核进行基准测试时，我们得到以下性能：

![](img/Pasted%20image%2020250309181603.png)

与`@torch.compile`相比，这个独立内核甚至在更小的尺寸下表现出卓越的性能，但这可能只是`torch.compile`编译时间的产物。无论如何，请记住，您可以从这样生成的内核开始，并将注意力集中在优化其性能上，而不是从头开始，从而节省大量时间。

即使在 Triton 中，有时我们也无法完全实现设备的峰值性能，因为语言限制无法处理共享内存和流式多处理器 (SM) 内的调度等低级细节。Triton 功能仅限于块和跨 SM 的块调度。为了获得更深层次的控制，您需要直接在 CUDA 中实现内核，这样您就可以访问所有底层的低级细节。

谈到 CUDA，可以采用各种技术来提高内核的效率。我们在这里只介绍其中几种：优化内存访问模式以减少延迟、使用共享内存存储经常访问的数据以及管理线程工作负载以最大限度地减少空闲时间。

在深入研究 CUDA 示例之前，让我们总结一下我们所见过的可以让我们编写内核代码来在 GPU 上执行指令的工具：

1. Pytorch：简单但缓慢
2. torch.compile：简单，快速，但不灵活
3. triton：更坚固、更快、更灵活
4. CUDA：最难、最快、最灵活（如果你做对了）

让我们来谈谈我们在 CUDA 中可以使用的最常用技术之一：优化内存访问。与缓存相比，GPU 中的全局内存（上图中最大的内存）具有较长的延迟和较低的带宽，这通常会给大多数应用程序造成主要瓶颈。高效地从全局内存访问数据可以大大提高性能。

#### 内存合并

为了有效利用全局内存的带宽，了解其架构至关重要。在 CUDA 设备中，全局内存是使用 DRAM 实现的。

内存合并利用了 DRAM 在访问内存地址时以突发或连续内存位置范围的形式提供数据的方式。每次访问 DRAM 位置时，DRAM 芯片中的多个传感器都会并行读取一系列连续位置（包括请求的位置）。读取后，这些数据可以以突发的形式快速传输到处理器。在 CUDA 中，合并使用这种突发行为来最大限度地提高内存访问效率，方法是确保 warp 中的线程（32 个以锁步 (SIMD) 执行相同指令的线程）访问连续的内存位置。例如，如果线程 0 访问位置 M，线程 1 访问 M + 1，线程 2 访问 M + 2，依此类推，则 GPU 硬件会将这些请求合并或组合成一个大型、高效的 DRAM 突发访问请求，而不是单独处理每个访问。

让我们以矩阵乘法为例。一个简单直接的实现是让每个线程计算输出矩阵的单个元素，如下所示：

```clike
__global__ void matmul_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}
```

这是来自这篇[精彩的博客文章](https://siboehm.com/articles/22/CUDA-MMM)的内核的出色可视化效果：

![](img/Pasted%20image%2020250309181739.png)

然而，当使用`ncu`之类的工具对该内核进行分析时，我们可以看到一些问题，包括低内存吞吐量和未合并的内存访问。

![](img/Pasted%20image%2020250309181755.png)

原因是，在这个内核中，同一个块中的两个线程，线程 ID 分别为`(0, 0)`和`(1, 0)` （最终会进入同一个 warp），都将从矩阵`B`的同一列但矩阵`A`的不同行加载。由于矩阵元素按行主序存储（即行元素位于连续的内存地址中，如下图所示），在第一次迭代`i = 0`中，线程`(0, 0)`将加载 A0,0A0,0​ ，线程`(1, 0)`将加载 A1,0A1,0​ 。这些元素在内存中存储的位置并不靠近，并且这种错位会在每次迭代中出现，从而阻止内存访问合并。

![](img/Pasted%20image%2020250309181809.png)

为了提高内核的性能，我们可以改变坐标方式x 和`y`计算如​​下：

```clike
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
float tmp = 0.0;
for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
}
C[x * N + y] = tmp;
}
```

我们不再使用 2D 块，而是改用 1D 块，并重新定义如何确定`x`和`y`的值。在这种新方法中，同一 warp 中的线程（具有接近的`threadIdx.x`值）将共享相同的`x`值，但具有不同的`y`值。这意味着它们将加载矩阵`A`的同一行，但加载矩阵`B`的不同列。因此，可以为行主矩阵合并内存访问。

当我们分析新内核时，我们注意到有关未合并内存访问的警告已经消失，并且**GPU 的内存吞吐量增加了大约 10 倍**。

![](img/Pasted%20image%2020250309181839.png)

我们还注意到内核的执行时间**减少了 10 倍**！太惊人了。

现在让我们介绍一下文献中经常提到的另一种技术：**平铺**。

#### 平铺


####  螺纹粗化


#### 最小化控制分歧


### 融合内核

我们在很多地方都提到了 GPU 和 CPU 操作如何异步。具体来说，CPU 上的主机代码可以以非阻塞方式调度 GPU 上的工作负载。

非阻塞对于重叠通信和计算很有用 - 正如我们在旅途中多次看到的那样 - 但可以扩展到更一般的想法，即尽量避免不惜一切代价在主机和 GPU 内核命令之间来回切换。

[Horace He](https://horace.io/brrr_intro.html)在这些图表中完美地说明了这个想法：

![](img/Pasted%20image%2020250309182034.png)

我们如何避免这种来回反复？最好的办法是让我们的 GPU 尽可能地自主。这是通过将尽可能多的连续计算操作打包到单个内核中供 GPU 运行来实现的，该内核称为“融合内核”。

对于连续执行的点状操作，融合内核尤其高效且易于编写，这些操作在每个输入标记上彼此独立地执行。在这种情况下，没有必要在将计算值移至 SM 内存并启动新内核之前将其带回全局内存中。将所有值保留在本地直到执行连续计算会更有效。

Transformer 模型中有很多地方可以应用这种“融合”方法：每次我们都有一系列逐点操作，例如在涉及层范数的计算中。

现在，我们已经了解了所有必要的知识，可以惊叹于内核工程的真正杰作： **_Flash Attention_**

### Flash Attention 1-3

[FlashAttention](../FlashAttention/FlashAttention.md)

### 混合精度训练

[混合精度训练](../量化/混合精度训练.md)

