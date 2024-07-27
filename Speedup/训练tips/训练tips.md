---
title: 训练tips
created: 2024-07-27
tags:
  - 训练tips
---
## BF16 优化器

用 FP16 训练巨型 LLM 模型是一个禁忌。

我们已经通过花费几个月的时间 [训练 104B 模型 1](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide) 自证了这一点，你可以从 [Tensorboard 1](https://huggingface.co/bigscience/tr8-104B-logs/tensorboard) 发现，彻头彻尾地失败了。在与不断发散的 lm-loss 作斗争的过程中，我们学到了很多:

![104B - 失败](https://devrel.andfun.cn/devrel/posts/2023/03/7b2vQR.jpg)

我们也从 Megatron-LM 和 DeepSpeed 团队那里得到了相同的建议，在他们训得 [530B 模型 5](https://arxiv.org/abs/2201.11990) 后。最近发布的 [OPT-175B 1](https://arxiv.org/abs/2205.01068) 也报告说他们在 FP16 上训练得非常艰难。

所以早在一月份，我们就知道我们要在支持 BF16 格式的 A100 上进行训练。 Olatunji Ruwase 开发了一个用来训练 BLOOM 的 “BF16Optimizer”。

如果您不熟悉这种数据格式，请查看 [它的位布局 3](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#bfloat16_floating-point_format)。 BF16 格式的关键是它的指数位数与 FP32 相同，因此不会溢出，但 FP16 经常溢出！FP16 的最大数值范围为 64k，您只能进行较小数的乘法。例如你可以做 `250*250=62500`，但如果你尝试 `255*255=65025`，你就会溢出，这是导致训练出现问题的主要原因。这意味着你的权重必须保持很小。一种称为损失缩放 (loss scaling) 的技术有助于缓解这个问题，但是当模型变得非常大时，FP16 较小的数值范围仍然是一个问题。

BF16 没有这个问题，你可以很容易地做 `10_000*10_000=100_000_000`, 完全没问题。


当然，由于 BF16 和 FP16 的大小相同，均为 2 个字节，因此，没有免费的午餐，当使用 BF16 时，代价就是它的精度非常差。然而，你应该还记得我们在训练时采用的随机梯度下降法及其变体，该方法有点像蹒跚而行，如果你这步没有找到完美的方向其实没关系，你会在接下来的步骤中纠正自己。

无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中 —— 这是由优化器更新的内容。因此 16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。

所有 PyTorch 组件都已更新，以确保它们在 FP32 中执行任何累加，因此不会发生精度损失。

一个关键问题是梯度累积，它是流水线并行的主要特征之一，因为每个 micro batch 处理的梯度都会累积。在 FP32 中实现梯度累积以保证训练的精确性至关重要，这正是 `BF16Optimizer` 所做的。

除了其他改进之外，我们认为使用 BF16 混合精度训练将潜在的噩梦变成了一个相对平稳的过程，这可以从以下 lm 损失图中看出:

![176B - 损失](https://devrel.andfun.cn/devrel/posts/2023/03/8YjcJH.jpg)

## CUDA 融合核函数


GPU 主要做两件事。它可以将数据写到显存或从显存读数据，并对这些数据执行计算。当 GPU 忙于读写数据时， GPU 的计算单元就会空闲。如果我们想有效地利用 GPU，我们希望将空闲时间降至最低。

核函数是一组实现特定 PyTorch 操作的指令。例如，当你调用 `torch.add` 时，它会通过一个 [PyTorch 调度器 1](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)，它会根据输入张量及其他变量的取值来决定它应该运行哪些代码，最后运行它。 CUDA 核函数使用 CUDA 来实现这些代码，因此只能在 NVIDIA GPU 上运行。

现在，当使用 GPU 计算 `c = torch.add (a, b); e = torch.max ([c,d])` 时，一般情况下，PyTorch 将执行的操作是启动两个单独的核函数，一个执行 `a` 和 `b` 的加法，另一个执行取 `c` 和 `d` 两者的最大值。在这种情况下，GPU 从其显存中获取 `a` 和 `b`，执行加法运算，然后将结果写回显存。然后它获取 `c` 和 `d` 并执行 `max` 操作，然后再次将结果写回显存。

如果我们要融合这两个操作，即将它们放入一个 “融合核函数” 中，然后启动那个内核，我们不会将中间结果 `c` 写到显存中，而是将其保留在 GPU 寄存器中，并且仅需要获取 `d` 来完成最后的计算。这节省了大量开销并防止 GPU 空闲，因此整个操作会更加高效。

融合核函数就是这样。它们主要将多个离散的计算和进出显存的数据移动替换为有很少数据移动的融合计算。此外，一些融合核函数会对操作进行数学变换，以便可以更快地执行某些计算组合。

为了快速高效地训练 BLOOM，有必要使用 Megatron-LM 提供的几个自定义 CUDA 融合核函数。特别地，有一个 LayerNorm 的融合核函数以及用于融合缩放、掩码和 softmax 这些操作的各种组合的核函数。Bias Add 也通过 PyTorch 的 JIT 功能与 GeLU 融合。这些操作都是瓶颈在内存的，因此将它们融合在一起以达到最大化每次显存读取后的计算量非常重要。因此，例如，在执行瓶颈在内存的 GeLU 操作时同时执行 Bias Add，运行时间并不会增加。这些核函数都可以在 [Megatron-LM 代码库 4](https://github.com/NVIDIA/Megatron-LM) 中找到。




## 数据集

Megatron-LM 的另一个重要特性是高效的数据加载器。在首次训练启动前，每个数据集中的每个样本都被分成固定序列长度 (BLOOM 为 2048) 的样本，并创建索引以对每个样本进行编号。基于训练超参，我们会确定每个数据集所需要参与的 epoch 数，并基于此创建一个有序的样本索引列表，然后打乱它。举个例子，如果一个数据集中有 10 个样本并应参与 2 个 epoch 的训练，则系统首先按 `[0, ..., 9, 0, ..., 9]` 顺序排好样本索引，然后打乱该顺序为数据集创建最终的全局顺序。请注意，这意味着训练不会简单地遍历整个数据集然后重复，你有可能在看到另一个样本之前看到同一个样本两次，但在训练结束时模型将只看到每个样本两次。这有助于确保整个训练过程中的训练曲线平滑。这些索引，包括每个样本在原始数据集中的偏移量，被保存到一个文件中，以避免每次开始训练时都重新计算它们。最后，可以将其中几个数据集以不同的权重混合到训练最终使用的数据中。

## 嵌入 LayerNorm

在我们努力阻止 104B 模型发散的过程中，我们发现在第一个层词嵌入层之后添加一个额外的 LayerNorm 可以使训练更加稳定。

该洞察来自对 [bitsandbytes](https://github.com/facebookresearch/bitsandbytes) 的实验，bitsandbytes 有一个 `StableEmbedding` 操作，它是一个带有 LayerNorm 的普通嵌入，其使用均匀 xavier 函数来初始化。

## 位置编码

基于论文 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation 4](https://arxiv.org/abs/2108.12409)，我们还用 AliBi 替换了普通的位置嵌入，它允许外推比训练模型的输入序列更长的输入序列。因此，即使我们训练时使用长度为 2048 的序列，模型也可以在推理过程中处理更长的序列。


## 训练中的困难

随着架构、硬件和软件的就位，我们得以在 2022 年 3 月上旬开始训练。然而，从那时起，事情其实并非一帆风顺。在本节中，我们将讨论我们遇到的一些主要障碍。

在训练开始之前，有很多问题需要弄清楚。特别是，我们发现了几个问题，这些问题只有在我们开始在 48 个节点上进行训练后才会出现，而不会在小规模时出现。例如，需要设 `CUDA_LAUNCH_BLOCKING=1` 来防止框架挂起，我们需要将优化器组分成更小的组，否则框架会再次挂起。你可以在 [训前编年史 9](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) 中详细了解这些内容。

训练期间遇到的主要问题类型是硬件故障。由于这是一个拥有大约 400 个 GPU 的新集群，平均每周我们会遇到 1-2 个 GPU 故障。我们每 3 小时 (100 次迭代) 保存一个检查点。因此，我们每周因硬件崩溃平均损失 1.5 小时的训练成果。 Jean Zay 系统管理员随后将更换有故障的 GPU 并恢复节点。与此同时，我们有备用节点可供使用。

我们还遇到过多次导致 5-10 小时停机的各种其他问题，其中一些与 PyTorch 中的死锁错误有关，另一些则是由于磁盘空间不足。如果您对具体细节有兴趣，请参阅 [训练编年史 17](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)。

在对训练这个模型进行可行性分析时，所有这些停机时间都被计划在内了，我们也据此选择了合适的模型大小和我们希望模型消耗的数据量。因此，即使存在这些停机问题，我们还是成功地在预计时间内完成了训练。如前所述，它需要大约 100 万个计算时才能完成。

另一个问题是 SLURM 并非设计为供一组人使用。 SLURM 作业由单个用户拥有，如果他们不在身边，则该组的其他成员无法对正在运行的作业执行任何操作。我们制定了一个终止方案，允许组中的其他用户终止当前进程，而不需要启动该进程的用户在场。这在 90% 的问题上都很有效。如果 SLURM 设计者读到这篇文章，请添加一个 Unix 组的概念，这样一个 SLURM 作业就可以由一个组拥有。

由于训练是全天候 24/7 进行的，我们需要有人随叫随到 - 但由于我们在欧洲和加拿大西海岸都有人，因此不需要有人携带传呼机，我们能很好地互相备份。当然，周末的训练也得有人看着。我们自动化了大部分事情，包括自动从硬件崩溃中恢复，但有时仍需要人工干预。

## 参考资料
[千亿参数开源大模型 BLOOM 背后的技术 - Hugging Face - 101.dev 社区](https://101.dev/t/bloom/921)
- [主训练文档 16](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md)
- [Tensorboard 4](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard)
- [训练用的 slurm 脚本 4](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm)
- [训练编年史 17](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)


Megatron-LM:

- [Efficient Large-Scale Language Model Training on GPU Clusters 3](https://arxiv.org/abs/2104.04473).
- [Reducing Activation Recomputation in Large Transformer Models 2](https://arxiv.org/abs/2205.05198)

DeepSpeed:

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models 1](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training 1](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning 1](https://arxiv.org/abs/2104.07857)
- [DeepSpeed: Extreme-scale model training for everyone 7](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

Megatron-LM 和 Deepspeeed 联合:

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model 5](https://arxiv.org/abs/2201.11990).

ALiBi:

- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation 4](https://arxiv.org/abs/2108.12409)
- [What Language Model to Train if You Have One Million GPU Hours? 1](https://openreview.net/forum?id=rI7BL3fHIZq) - 你会在那里找到最终使得我们选择 ALiBi 的实验。

BitsNBytes:

- [8-bit Optimizers via Block-wise Quantization 1](https://arxiv.org/abs/2110.02861) (我们使用了该论文中的嵌入 LaynerNorm，但是论文的其他部分及其技术也很妙，我们没用 8 位优化器的唯一原因是我们已经使用 DeepSpeed-ZeRO 节省了优化器内存)。

