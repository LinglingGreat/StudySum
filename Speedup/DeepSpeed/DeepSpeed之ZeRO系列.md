虽然数据并行是一种有效的训练扩展方法，但在每个 DP 等级上对优化器状态、梯度和参数的简单复制会引入显著的内存冗余。ZeRO 通过在数据并行维度上对优化器状态、梯度和参数进行分区来消除内存冗余，同时仍然允许使用全套参数进行计算。这有时需要 DP 等级(rank)之间进行更多通信，这些等级可能完全重叠，也可能不完全重叠。

## DeepSpeed集成

DeepSpeed支持

1. ZeRO-1：优化器状态分区
2. ZeRO-2：优化器状态 + 梯度分区
3. ZeRO-3（也称为 FSDP，即“全分片数据并行”）：优化器状态 + 梯度 + 参数分区
4. 自定义混合精度训练处理
5. 一系列基于CUDA扩展的快速优化器
6. ZeRO-Offload 到 CPU 和 NVMe

ZeRO-Offload有其自己的专门论文：[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)。而NVMe支持在论文[ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)中进行了描述。

当我们说分区时，它意味着沿着 DP 轴，因为 ZeRO 是数据并行的一部分。我们稍后会看到我们可以沿着其他轴进行分区。

您可能遗漏了我们可以分片的内容中的激活。由于模型的每个 DP 副本都会收到不同的微批次，因此每个 DP 等级上的激活也不同，因此它们不会重复，因此无法分片！

DeepSpeed ZeRO-2主要用于训练，因为它的特性对推理没有用处。

DeepSpeed ZeRO-3也可以用于推理，因为它允许将单个GPU无法加载的大模型加载到多个GPU上。

🤗 Transformers通过以下两种方式集成了[DeepSpeed](https://github.com/microsoft/DeepSpeed)：

1. 通过`Trainer`集成核心的DeepSpeed功能。这是一种“为您完成一切”式的集成 - 您只需提供自定义配置文件或使用我们的模板配置文件。本文档的大部分内容都集中在这个功能上。
2. 如果您不使用`Trainer`并希望在自己的Trainer中集成DeepSpeed，那么像`from_pretrained`和`from_config`这样的核心功能函数将包括ZeRO stage 3及以上的DeepSpeed的基础部分，如`zero.Init`。要利用此功能，请阅读有关[非Trainer DeepSpeed集成](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#nontrainer-deepspeed-integration)的文档。

集成的内容：

训练：

1. DeepSpeed ZeRO训练支持完整的ZeRO stages 1、2和3，以及ZeRO-Infinity（CPU和NVMe offload）。

推理：

2. DeepSpeed ZeRO推理支持ZeRO stage 3和ZeRO-Infinity。它使用与训练相同的ZeRO协议，但不使用优化器和学习率调度器，只有stage 3与推理相关。更多详细信息请参阅：[zero-inference](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#zero-inference)。

此外还有DeepSpeed推理 - 这是一种完全不同的技术，它使用张量并行而不是ZeRO（即将推出）。

## 为什么需要单GPU启动

```python
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

为什么要在仅使用一张 GPU 的情况下使用 DeepSpeed 呢？

1. 它具有 ZeRO-offload 功能，可以将一些计算和内存委托给主机的 CPU 和 内存，从而为模型的需求保留更多 GPU 资源 - 例如更大的批处理大小，或启用正常情况下无法容纳的非常大模型。
2. 它提供了智能的 GPU 内存管理系统，最小化内存碎片，这再次允许您容纳更大的模型和数据批次。

虽然接下来我们将详细讨论配置，但在单个 GPU 上通过 DeepSpeed 实现巨大性能提升的关键是在配置文件中至少有以下配置：

{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}

这会启用`optimizer offload` 和一些其他重要功能。您可以尝试不同的buffer大小

注意：

- 如果您需要在特定的 GPU 上运行，而不是 GPU 0，则无法使用 `CUDA_VISIBLE_DEVICES` 来限制可用 GPU 的可见范围。相反，您必须使用以下语法：
    
    Copied
    
    deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
    
    在这个例子中，我们告诉 DeepSpeed 使用 GPU 1（第二个 GPU）。
    

## ZeRO

[Zero Redundancy Optimizer (ZeRO)](https://www.deepspeed.ai/tutorials/zero/) 是 DeepSpeed 的工作核心。它支持3个不同级别（stages）的优化。Stage 1 对于扩展性来说不是很有趣，因此本文档重点关注Stage 2和Stage 3。Stage 3通过最新的 ZeRO-Infinity 进一步改进。你可以在 DeepSpeed 文档中找到更详细的信息。

配置文件的 `zero_optimization` 部分是最重要的部分（[文档](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)），因为在这里您定义了要启用哪些 ZeRO stages 以及如何配置它们。您可以在 DeepSpeed 文档中找到每个参数的解释。

### offload_optimizer 和 offload_param 

1. offload_optimizer:
   - 这个选项主要指的是将优化器状态卸载到CPU。
   - 优化器状态包括诸如动量、方差等，这取决于所使用的优化器（如Adam、AdamW等）。
   - 也包括梯度，因为优化器需要使用梯度来更新参数。

2. offload_param:
   - 这个选项指的是将模型参数卸载到CPU。
   - 模型参数是指神经网络的权重和偏置。

更详细的解释：

- offload_optimizer:
  - 当启用时，优化器的状态（如Adam优化器中的一阶矩和二阶矩）会存储在CPU内存中。
  - 在反向传播过程中，梯度会先在GPU上计算，然后传输到CPU。
  - 参数更新在CPU上进行，然后更新后的参数再传回GPU。

- offload_param:
  - 当启用时，模型的参数在不需要时会存储在CPU内存中。
  - 在前向传播时，需要用到的参数会从CPU传输到GPU。
  - 在反向传播完成后，参数又会被移回CPU。

这两个选项的主要区别：
- offload_optimizer 主要影响训练过程中优化器相关的内存使用。
- offload_param 直接影响模型参数的存储位置，可能对推理和训练都有影响。

使用这些选项可以显著减少GPU内存的使用，但可能会增加CPU-GPU之间的数据传输，从而影响训练速度。因此，在使用时需要权衡内存使用和训练速度。

在大多数情况下，首先尝试 offload_optimizer 可能就足够了。如果GPU内存仍然不足，再考虑使用 offload_param。

通过将 `pin_memory` 设置为 `true` 启用固定内存。此功能会以减少可用于其他进程的内存为代价来提高吞吐量。固定内存被分配给特定请求它的进程，通常比普通 CPU 内存访问速度更快。

### ZeRO-2 配置

```
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    }
}
```

**性能调优：**

- 启用 `offload_optimizer` 应该减少 GPU 内存使用（需要 `"stage": 2`）。
- `"overlap_comm": true` 通过增加 GPU 内存使用来降低all-reduce 的延迟。 `overlap_comm` 使用了 `allgather_bucket_size` 和 `reduce_bucket_size` 值的4.5倍。因此，如果它们设置为 `5e8`，这将需要一个9GB的内存占用（`5e8 x 2Bytes x 2 x 4.5`）。因此，如果您的 GPU 内存为8GB或更小，为了避免出现OOM错误，您需要将这些参数减小到约 `2e8`，这将需要3.6GB。如果您的 GPU 容量更大，当您开始遇到OOM时，你可能也需要这样做。
- 当减小这些buffers时，您以更慢的通信速度来换取更多的 GPU 内存。buffers大小越小，通信速度越慢，GPU 可用于其他任务的内存就越多。因此，如果更大的批处理大小很重要，那么稍微减慢训练时间可能是一个很好的权衡。

此外，`deepspeed==0.4.4` 添加了一个新选项 `round_robin_gradients`，您可以通过以下方式启用：

{
    "zero_optimization": {
        "round_robin_gradients": true
    }
}

这是一个用于 CPU offloading 的stage 2优化，通过细粒度梯度分区在 ranks 之间并行复制到 CPU 内存，从而实现了性能的提升。性能优势随着梯度累积步骤（在优化器步骤之间进行更多复制）或 GPU 数量（增加并行性）增加而增加。

### ZeRO-3 配置

```
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

如果您因为你的模型或激活值超过 GPU 内存而遇到OOM问题，并且您有未使用的 CPU 内存，可以通过使用 `"device": "cpu"` 将优化器状态和参数卸载到 CPU 内存中，来解决这个限制。如果您不想卸载到 CPU 内存，可以在 `device` 条目中使用 `none` 代替 `cpu`。将优化器状态卸载到 NVMe 上会在后面进一步讨论。

通过将 `pin_memory` 设置为 `true` 启用固定内存。此功能会以减少可用于其他进程的内存为代价来提高吞吐量。固定内存被分配给特定请求它的进程，通常比普通 CPU 内存访问速度更快。

**性能调优：**

- `stage3_max_live_parameters`: `1e9`
- `stage3_max_reuse_distance`: `1e9`

如果遇到OOM问题，请减小 `stage3_max_live_parameters` 和 `stage3_max_reuse_distance`。它们对性能的影响应该很小，除非您正在进行激活值checkpointing。`1e9` 大约会消耗 ~2GB。内存由 `stage3_max_live_parameters` 和 `stage3_max_reuse_distance` 共享，所以它不是叠加的，而是总共2GB。

`stage3_max_live_parameters` 是在任何给定时间要在 GPU 上保留多少个完整参数的上限。“reuse distance” 是我们用来确定参数在将来何时会再次使用的度量标准，我们使用 `stage3_max_reuse_distance` 来决定是丢弃参数还是保留参数。如果一个参数在不久的将来（小于 `stage3_max_reuse_distance`）将被再次使用，那么我们将其保留以减少通信开销。这在启用激活值checkpoing时非常有用，其中我们以单层粒度进行前向重计算和反向传播，并希望在反向传播期间保留前向重计算中的参数。

以下配置值取决于模型的隐藏大小：

- `reduce_bucket_size`: `hidden_size*hidden_size`
- `stage3_prefetch_bucket_size`: `0.9 * hidden_size * hidden_size`
- `stage3_param_persistence_threshold`: `10 * hidden_size`

因此，将这些值设置为 `auto`，`Trainer` 将自动分配推荐的参数值。当然，如果您愿意，也可以显式设置这些值。

`stage3_gather_16bit_weights_on_model_save` 在模型保存时启用模型的 fp16 权重整合。对于大模型和多个 GPU，无论是在内存还是速度方面，这都是一项昂贵的操作。目前如果计划恢复训练，这是必需的。请注意未来的更新可能会删除此限制并让使用更加灵活。

如果您从 ZeRO-2 配置迁移，请注意 `allgather_partitions`、`allgather_bucket_size` 和 `reduce_scatter` 配置参数在 ZeRO-3 中不被使用。如果保留这些配置文件，它们将被忽略。

- `sub_group_size`: `1e9`

`sub_group_size` 控制在优化器步骤期间更新参数的粒度。参数被分组到大小为 `sub_group_size` 的桶中，每个桶逐个更新。在 ZeRO-Infinity 中与 NVMe offload一起使用时，`sub_group_size` 控制了在优化器步骤期间在 NVMe 和 CPU 内存之间移动模型状态的粒度。这可以防止非常大的模型耗尽 CPU 内存。

当不使用 NVMe offload时，可以将 `sub_group_size` 保留为其默认值 _1e9_。在以下情况下，您可能需要更改其默认值：

1. 在优化器步骤中遇到OOM：减小 `sub_group_size` 以减少临时buffers的内存利用
2. 优化器步骤花费很长时间：增加 `sub_group_size` 以提高由于增加的数据buffers而导致的带宽利用率。

### ZeRO-0 配置

请注意，我们将 Stage 0 和 1 放在最后，因为它们很少使用。

Stage 0 禁用了所有类型的分片，只是将 DeepSpeed 作为 DDP 使用。您可以通过以下方式启用：

{
    "zero_optimization": {
        "stage": 0
    }
}

这将实质上禁用 ZeRO，而无需更改其他任何内容。

### ZeRO-1 配置

Stage 1 等同于 Stage 2 减去梯度分片。您可以尝试使用以下配置，仅对优化器状态进行分片，以稍微加速：

Copied

{
    "zero_optimization": {
        "stage": 1
    }
}

### ZeRO++

[ZeRO++ - DeepSpeed](https://www.deepspeed.ai/tutorials/zeropp/)

![](img/Pasted%20image%2020250310203520.png)
图 1：ZeRO++ 项目亮点图片。左上角子图显示，与 ZeRO 第 3 阶段相比，ZeRO++ 将通信量减少了 4 倍。右上角子图显示了 ZeRO++ 在 RLHF 模型训练中的表现，其中 ZeRO++ 实现了 RLHF 训练速度提高 1.3 倍，令牌生成速度提高 2.x 倍。



## NVMe 支持

ZeRO-Infinity 通过使用 NVMe 内存扩展 GPU 和 CPU 内存，从而允许训练非常大的模型。由于智能分区和平铺算法，在offload期间每个 GPU 需要发送和接收非常小量的数据，因此 NVMe 被证明适用于训练过程中提供更大的总内存池。ZeRO-Infinity 需要启用 ZeRO-3。

以下配置示例启用 NVMe 来offload优化器状态和参数：

{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}

您可以选择将优化器状态和参数都卸载到 NVMe，也可以只选择其中一个，或者都不选择。例如，如果您有大量的 CPU 内存可用，只卸载到 CPU 内存训练速度会更快（提示：“device”: “cpu”）。

这是有关卸载 [优化器状态](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) 和 [参数](https://www.deepspeed.ai/docs/config-json/#parameter-offloading) 的完整文档。

确保您的 `nvme_path` 实际上是一个 NVMe，因为它与普通硬盘或 SSD 一起工作，但速度会慢得多。快速可扩展的训练是根据现代 NVMe 传输速度设计的（截至本文撰写时，可以达到 ~3.5GB/s 读取，~3GB/s 写入的峰值速度）。

为了找出最佳的 `aio` 配置块，您必须在目标设置上运行一个基准测试，具体操作请参见[说明](https://github.com/microsoft/DeepSpeed/issues/998)。

### ZeRO-2 和 ZeRO-3 性能对比

如果其他一切都配置相同，ZeRO-3 可能比 ZeRO-2 慢，因为前者除了 ZeRO-2 的操作外，还必须收集模型权重。如果 ZeRO-2 满足您的需求，而且您不需要扩展到几个 GPU 以上，那么您可以选择继续使用它。重要的是要理解，ZeRO-3 以速度为代价实现了更高的可扩展性。

可以调整 ZeRO-3 配置使其性能接近 ZeRO-2：

- 将 `stage3_param_persistence_threshold` 设置为一个非常大的数字 - 大于最大的参数，例如 `6 * hidden_size * hidden_size`。这将保留参数在 GPU 上。
- 关闭 `offload_params`，因为 ZeRO-2 没有这个选项。

即使不更改 `stage3_param_persistence_threshold`，仅将 `offload_params` 关闭，性能可能会显著提高。当然，这些更改将影响您可以训练的模型的大小。因此，这些更改可根据需求帮助您在可扩展性和速度之间进行权衡。

### 如何选择最佳性能的ZeRO Stage和 offloads

通常，以下规则适用：

- 速度方面（左边比右边快）
    
    stage 0（DDP） > stage 1 > stage 2 > stage 2 + offload > stage 3 > stage3 + offload
    
- GPU内存使用方面（右边比左边更节省GPU内存）
    
    stage 0（DDP） < stage 1 < stage 2 < stage 2 + offload < stage 3 < stage 3 + offload
    

所以，当您希望在尽量使用较少数量的GPU的同时获得最快的执行速度时，可以按照以下步骤进行。我们从最快的方法开始，如果遇到GPU内存溢出，然后切换到下一个速度较慢但使用的GPU内存更少的方法。以此类推。

首先，将批量大小设置为1（您始终可以使用梯度累积来获得任何所需的有效批量大小）。

1. 启用 `--gradient_checkpointing 1`（HF Trainer）或直接 `model.gradient_checkpointing_enable()` - 如果发生OOM（Out of Memory），则执行以下步骤。
2. 首先尝试 ZeRO stage 2。如果发生OOM，则执行以下步骤。
3. 尝试 ZeRO stage 2 + `offload_optimizer` - 如果发生OOM，则执行以下步骤。
4. 切换到 ZeRO stage 3 - 如果发生OOM，则执行以下步骤。
5. 启用 `offload_param` 到 `cpu` - 如果发生OOM，则执行以下步骤。
6. 启用 `offload_optimizer` 到 `cpu` - 如果发生OOM，则执行以下步骤。
7. 如果仍然无法适应批量大小为1，请首先检查各种默认值并尽可能降低它们。例如，如果使用 `generate` 并且不使用宽搜索束，将其缩小，因为它会占用大量内存。
8. 绝对要使用混合半精度而非fp32 - 在Ampere及更高的GPU上使用bf16，在旧的GPU体系结构上使用fp16。
9. 如果仍然发生OOM，可以添加更多硬件或启用ZeRO-Infinity - 即切换 `offload_param` 和 `offload_optimizer` 到 `nvme`。您需要确保它是非常快的NVMe。作为趣闻，我曾经能够在一个小型GPU上使用BLOOM-176B进行推理，使用了ZeRO-Infinity，尽管速度非常慢。但它奏效了！

当然，您也可以按相反的顺序进行这些步骤，从最节省GPU内存的配置开始，然后逐步反向进行，或者尝试进行二分法。

一旦您的批量大小为1不会导致OOM，就测量您的有效吞吐量。

接下来尝试将批量大小增加到尽可能大，因为批量大小越大，GPU的效率越高，特别是在它们乘法运算的矩阵很大时。

现在性能优化游戏开始了。您可以关闭一些offload特性，或者降低ZeRO stage，并增加/减少批量大小，再次测量有效吞吐量。反复尝试，直到满意为止。

不要花费太多时间，但如果您即将开始一个为期3个月的训练 - 请花几天时间找到吞吐量方面最有效的设置。这样您的训练成本将最低，而且您会更快地完成训练。在当前快节奏的机器学习世界中，如果您花费一个额外的月份来训练某样东西，你很可能会错过一个黄金机会。当然，这只是我分享的一种观察，我并不是在催促你。在开始训练BLOOM-176B之前，我花了2天时间进行这个过程，成功将吞吐量从90 TFLOPs提高到150 TFLOPs！这一努力为我们节省了一个多月的训练时间。

这些注释主要是为训练模式编写的，但它们在推理中也应该大部分适用。例如，在推理中，Gradient Checkpointing 是无用的，因为它只在训练过程中有用。此外，我们发现，如果你正在进行多GPU推理并且不使用 [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/)，[Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts) 应该提供更优越的性能。

其他与性能相关的快速注释：

- 如果您从头开始训练某个模型，请尽量确保张量的形状可以被16整除（例如隐藏层大小）。对于批量大小，至少尝试可被2整除。如果您想从GPU中挤取更高性能，还有一些硬件特定的[wave和tile量化](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)的可整除性。

## Activation Checkpointing 或 Gradient Checkpointing

Activation Checkpointing和Gradient Checkpointing是指相同方法的两个不同术语。这确实让人感到困惑，但事实就是这样。

- 在标准的反向传播中，需要保存所有中间激活值以计算梯度。在训练神经网络时，反向传播需要访问前向传播时的中间激活值（即神经网络每一层的输出）。通常，所有的激活值都会在内存中保存，以便在计算梯度时使用。然而，当模型较大时，这会消耗大量内存，尤其是在处理长序列或大型批次时。
- Gradient Checkpointing通过只保存部分中间结果（检查点），并在需要时重新计算其他值来减少内存使用。

Gradient Checkpointing允许通过牺牲速度来换取GPU内存，这要么使您能够克服GPU内存溢出，要么增加批量大小来获得更好的性能。

HF Transformers 模型对DeepSpeed的Activation Checkpointing一无所知，因此如果尝试在DeepSpeed配置文件中启用该功能，什么都不会发生。

因此，您有两种方法可以利用这个非常有益的功能：

1. 如果您想使用 HF Transformers 模型，你可以使用 `model.gradient_checkpointing_enable()` 或在 HF Trainer 中使用 `--gradient_checkpointing`，它会自动为您启用这个功能。在这里使用了 `torch.utils.checkpoint`。
2. 如果您编写自己的模型并希望使用DeepSpeed的Activation Checkpointing，可以使用[规定的API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)。您还可以使用 HF Transformers 的模型代码，将 `torch.utils.checkpoint` 替换为 DeepSpeed 的API。后者更灵活，因为它允许您将前向激活值卸载到CPU内存，而不是重新计算它们。

## Optimizer 和 Scheduler

只要你不启用 `offload_optimizer`，您可以混合使用DeepSpeed和HuggingFace的调度器和优化器，但有一个例外，即不要使用HuggingFace调度器和DeepSpeed优化器的组合：

|Combos|HF Scheduler|DS Scheduler|
|:--|:--|:--|
|HF Optimizer|Yes|Yes|
|DS Optimizer|No|Yes|

在启用 `offload_optimizer` 的情况下，可以使用非DeepSpeed优化器，只要该优化器具有CPU和GPU的实现（除了LAMB）。

### Optimizer

DeepSpeed的主要优化器包括Adam、AdamW、OneBitAdam和Lamb。这些优化器已经与ZeRO进行了彻底的测试，因此建议使用它们。然而，也可以导入`torch`中的其他优化器。完整的文档在[这里](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)。

当与DeepSpeed的CPU Adam优化器一起使用时，offload的效果最好。如果您想在offload时使用不同的优化器，自 `deepspeed==0.8.3` 起，您还需要添加：

{
   "zero_force_ds_cpu_optimizer": false
}

到顶层配置中。

### Scheduler

DeepSpeed支持`LRRangeTest`、`OneCycle`、`WarmupLR`和`WarmupDecayLR`学习率调度器。完整文档在[这里](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)。

以下是🤗 Transformers 和 DeepSpeed 之间的调度器重叠部分：

- 通过 `--lr_scheduler_type constant_with_warmup` 实现 `WarmupLR`
- 通过 `--lr_scheduler_type linear` 实现 `WarmupDecayLR`。这也是 `--lr_scheduler_type` 的默认值，因此，如果不配置调度器，这将是默认配置的调度器。

如果在配置文件中不配置 `scheduler` 条目，`Trainer` 将使用 `--lr_scheduler_type`、`--learning_rate` 和 `--warmup_steps` 或 `--warmup_ratio` 的值来配置其🤗 Transformers 版本。

## fp32精度

DeepSpeed支持完整的fp32和fp16混合精度。

由于fp16混合精度具有更小的内存需求和更快的速度，唯一不使用它的时候是当您使用的模型在这种训练模式下表现不佳时。通常，当模型没有在fp16混合精度下进行预训练时（例如，bf16预训练模型经常出现这种情况），会出现这种情况。这样的模型可能会发生溢出或下溢，导致 `NaN` 损失。如果是这种情况，那么您将希望使用完整的fp32模式，通过显式禁用默认启用的fp16混合精度模式：

Copied

{
    "fp16": {
        "enabled": false,
    }
}

如果您使用基于Ampere架构的GPU，PyTorch版本1.7及更高版本将自动切换到使用更高效的tf32格式进行一些操作，但结果仍将以fp32格式呈现。有关详细信息和基准测试，请参见[TensorFloat-32(TF32) on Ampere devices](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)。如果出于某种原因您不希望使用它，该文档包括有关如何禁用此自动转换的说明。

在🤗 Trainer中，你可以使用 `--tf32` 来启用它，或使用 `--tf32 0` 或 `--no_tf32` 来禁用它。默认情况下，使用PyTorch的默认设置。

## NCCL集合

在训练过程中，有两种数据类型：`dtype`和用于通信收集操作的`dtype`，如各种归约和收集/分散操作。

所有的gather/scatter操作都是在数据相同的`dtype`中执行的，所以如果您正在使用bf16的训练模式，那么它将在bf16中进行gather操作 - gather操作是非损失性的。

各种reduce操作可能会是非常损失性的，例如当梯度在多个gpu上平均时，如果通信是在fp16或bf16中进行的，那么结果可能是有损失性的 - 因为当在一个低精度中添加多个数字时，结果可能不是精确的。更糟糕的是，bf16比fp16具有更低的精度。通常，当平均梯度时，损失最小，这些梯度通常非常小。因此，对于半精度训练，默认情况下，fp16被用作reduction操作的默认值。但是，您可以完全控制这个功能，如果你选择的话，您可以添加一个小的开销，并确保reductions将使用fp32作为累积数据类型，只有当结果准备好时，它才会降级到您在训练中使用的半精度`dtype`。

要覆盖默认设置，您只需添加一个新的配置条目：

{
    "communication_data_type": "fp32"
}

根据这个信息，有效的值包括”fp16”、“bfp16”和”fp32”。

注意：在stage zero 3中，bf16通信数据类型存在一个bug，该问题已在`deepspeed==0.8.1`版本中得到修复。

## apex

配置apex AMP-like模式：

Copied

"amp": {
    "enabled": "auto",
    "opt_level": "auto"
}

并且，`Trainer`将根据`args.fp16_backend`和`args.fp16_opt_level`的值自动配置它。

当传递`--fp16 --fp16_backend apex --fp16_opt_level 01`命令行参数时，此模式将被启用。

## ZeRO-3 和 Infinity Nuances

ZeRO-3与ZeRO-2有很大的不同，主要是因为它的参数分片功能。

ZeRO-Infinity进一步扩展了ZeRO-3，以支持NVMe内存和其他速度和可扩展性改进。

尽管所有努力都是为了在不需要对模型进行任何特殊更改的情况下就能正常运行，但在某些情况下，您可能需要以下信息。

### 构建大模型

DeepSpeed/ZeRO-3可以处理参数量达到数万亿的模型，这些模型可能无法适应现有的内存。在这种情况下，如果您还是希望初始化更快地发生，可以使用_deepspeed.zero.Init()_上下文管理器（也是一个函数装饰器）来初始化模型，如下所示：

```python
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

如您所见，这会为您随机初始化一个模型。

如果您想使用预训练模型，`model_class.from_pretrained`将在`is_deepspeed_zero3_enabled()`返回`True`的情况下激活此功能，目前这是通过传递的DeepSpeed配置文件中的ZeRO-3配置部分设置的。因此，在调用`from_pretrained`之前，您必须创建**TrainingArguments**对象。以下是可能的顺序示例：

```python
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

如果您使用的是官方示例脚本，并且命令行参数中包含`--deepspeed ds_config.json`且启用了ZeRO-3配置，那么一切都已经为您准备好了，因为这是示例脚本的编写方式。

注意：如果模型的fp16权重无法适应单个GPU的内存，则必须使用此功能。

有关此方法和其他相关功能的完整详细信息，请参阅[构建大模型](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models)。

此外，在加载fp16预训练模型时，您希望`from_pretrained`使用`torch_dtype=torch.float16`。详情请参见[from_pretrained-torch-dtype](https://huggingface.co/docs/transformers/zh/main_classes/deepspeed#from_pretrained-torch-dtype)。

### 参数收集

在多个GPU上使用ZeRO-3时，没有一个GPU拥有所有参数，除非它是当前执行层的参数。因此，如果您需要一次访问所有层的所有参数，有一个特定的方法可以实现。 您可能不需要它，但如果您需要，请参考[参数收集](https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination)。

然而，我们在多个地方确实使用了它，其中一个例子是在`from_pretrained`中加载预训练模型权重。我们一次加载一层，然后立即将其分区到所有参与的GPU上，因为对于非常大的模型，无法在一个GPU上一次性加载并将其分布到多个GPU上，因为内存限制。

此外，在ZeRO-3下，如果您编写自己的代码并遇到看起来像这样的模型参数权重：

tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)

强调`tensor([1.])`，或者如果您遇到一个错误，它说参数的大小是`1`，而不是某个更大的多维形状，这意味着参数被划分了，你看到的是一个ZeRO-3占位符。

## ZeRO 推理

“ZeRO 推断” 使用与 “ZeRO-3 训练” 相同的配置。您只需要去掉优化器和调度器部分。实际上，如果您希望与训练共享相同的配置文件，您可以将它们保留在配置文件中，它们只会被忽略。

您只需要传递通常的`TrainingArguments`参数。例如：

```python
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

唯一的重要事情是您需要使用ZeRO-3配置，因为ZeRO-2对于推理没有任何优势，因为只有ZeRO-3才对参数进行分片，而ZeRO-2则对梯度和优化器状态进行分片。

以下是在DeepSpeed下运行`run_translation.py`启用所有可用GPU的示例：

```python
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --output_dir output_dir \
--do_eval --max_eval_samples 50 --warmup_steps 50  \
--max_source_length 128 --val_max_target_length 128 \
--overwrite_output_dir --per_device_eval_batch_size 4 \
--predict_with_generate --dataset_config "ro-en" --fp16 \
--source_lang en --target_lang ro --dataset_name wmt16 \
--source_prefix "translate English to Romanian: "
```

由于在推理阶段，优化器状态和梯度不需要额外的大量内存，您应该能够将更大的批次和/或序列长度放到相同的硬件上。

此外，DeepSpeed目前正在开发一个名为Deepspeed-Inference的相关产品，它与ZeRO技术无关，而是使用张量并行来扩展无法适应单个GPU的模型。

## 内存要求

由于 DeepSpeed ZeRO 可以将内存卸载到 CPU（和 NVMe），该框架提供了一些工具，允许根据使用的 GPU 数量告知将需要多少 CPU 和 GPU 内存。

让我们估计在单个GPU上微调”bigscience/T0_3B”所需的内存：

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

因此，您可以将模型拟合在单个80GB的GPU上，不进行CPU offload，或者使用微小的8GB GPU，但需要约60GB的CPU内存。（请注意，这仅是参数、优化器状态和梯度所需的内存 - 您还需要为CUDA内核、激活值和临时变量分配更多的内存。）

然后，这是成本与速度的权衡。购买/租用较小的 GPU（或较少的 GPU，因为您可以使用多个 GPU 进行 Deepspeed ZeRO）。但这样会更慢，因此即使您不关心完成某项任务的速度，减速也直接影响 GPU 使用的持续时间，从而导致更大的成本。因此，请进行实验并比较哪种方法效果最好。

如果您有足够的GPU内存，请确保禁用CPU/NVMe卸载，因为这会使所有操作更快。

## Troubleshooting

### 启动时 deepspeed 进程被终止，没有回溯

如果启动时`deepspeed`进程被终止，没有回溯，这通常意味着程序尝试分配的CPU内存超过了系统的限制或进程被允许分配的内存，操作系统内核杀死了该进程。这是因为您的配置文件很可能将`offload_optimizer`或`offload_param`或两者都配置为卸载到`cpu`。如果您有NVMe，可以尝试在ZeRO-3下卸载到NVMe。这里是如何[估计特定模型所需的内存](https://deepspeed.readthedocs.io/en/latest/memory.html)。

### 训练和/或评估/预测loss为 NaN

这种情况通常发生在使用bf16混合精度模式预训练的模型试图在fp16（带或不带混合精度）下使用时。大多数在TPU上训练的模型以及由谷歌发布的模型都属于这个类别（例如，几乎所有基于t5的模型）。在这种情况下，解决方案是要么使用fp32，要么在支持的情况下使用bf16（如TPU、Ampere GPU或更新的版本）。

另一个问题可能与使用fp16有关。当您配置此部分时：

{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}

并且您在日志中看到Deepspeed报告`OVERFLOW`如下

```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

这意味着Deepspeed损失缩放器无法找到一个克服损失溢出的缩放系数。

在这种情况下，通常需要提高`initial_scale_power`的值。将其设置为`"initial_scale_power": 32`通常会解决问题。

## 注意事项

- 尽管 DeepSpeed 有一个可安装的 PyPI 包，但强烈建议从源代码安装它，以最好地匹配您的硬件，如果您需要启用某些功能，如 1-bit Adam，这些功能在 pypi 发行版中不可用。
- 您不必使用🤗 Transformers的 `Trainer` 来使用 DeepSpeed - 您可以使用任何模型与自己的训练器，您还需要根据 [DeepSpeed 集成说明](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models) 调整后者。

## DeepSpeed Zero Stage 3 到底是什么并行？数据并行还是模型并行？

> 大模型训练通常会用到：  
> 1、[数据并行](https://zhida.zhihu.com/search?content_id=241897162&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C&zhida_source=entity)（Data Parallelism）  
> 2、[模型并行](https://zhida.zhihu.com/search?content_id=241897162&content_type=Article&match_order=1&q=%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C&zhida_source=entity)：包括[张量并行](https://zhida.zhihu.com/search?content_id=241897162&content_type=Article&match_order=1&q=%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C&zhida_source=entity)（Tensor Parallelism）和流水线并行（Pipeline Parallelism）

DeepSpeed Zero Stage 本质上是一种“节省显存”的数据并行，是一种 Fully Sharded Data Parallelism。

**例如，Zero Stage 3 加载时将模型参数进行切片存储到不同的GPU上，每个GPU只保留参数的1/N。计算时，每个GPU跑不同的数据，然后GPU之间进行参数通信，保证每个GPU下的batch都能通过模型全部参数，而不是局部参数。（主要利用all-gather收集参数，reduce-scatter规约计算）**

[ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters - Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

## zero3和megatron对比
Zero3（DeepSpeed ZeRO Stage 3）和Megatron的模型并行是两种不同的大规模模型训练优化技术，主要区别体现在并行策略、通信机制和应用目标上：

---

### **1. 并行维度的区别**

- **Zero3（数据并行扩展）**  
    Zero3属于**数据并行**的增强技术，核心目标是通过**参数分片**减少内存冗余。它将模型的参数（Parameters）、梯度（Gradients）和优化器状态（Optimizer States）分片到所有GPU上，每个GPU仅维护一部分数据。在正向/反向传播时，动态收集所需的参数分片，从而显著降低单卡内存占用。
    
- **Megatron（模型并行）**  
    Megatron属于**模型并行**技术，核心是将模型结构（如Transformer层）**拆分到不同GPU上**。例如，将矩阵乘法运算按行或列切分到多个GPU，每个GPU负责部分计算，通过通信聚合结果。这种方式直接拆分计算图，属于张量并行（Tensor Parallelism）。
    

---

### **2. 通信模式的区别**

- **Zero3的通信特点**
    
    - **按需收集参数**：在正向和反向传播时，各GPU需通过All-Gather操作收集当前计算所需的参数分片，结束后释放冗余副本。
    - **通信频率高**：每个训练步骤均需多次通信，但单次通信量较小（仅涉及当前层的参数）。
- **Megatron的通信特点**
    
    - **层内通信**：在单个Transformer层的计算中（如Self-Attention或FFN），拆分后的GPU需频繁交换中间结果（如通过All-Reduce或All-Gather）。
    - **通信密集**：每层计算均伴随通信，通信频率与模型深度成正比，但单次通信量较大（涉及激活值或梯度）。

---

### **3. 内存优化的区别**

- **Zero3的内存优化**
    
    - **参数分片**：参数、梯度、优化器状态均被分片存储，内存占用与GPU数量成反比。
    - **临时副本**：仅在计算时保留当前层的完整参数副本，计算后释放，适合超大模型训练。
- **Megatron的内存优化**
    
    - **计算图拆分**：每个GPU仅存储模型的一部分结构和对应参数，但需保存完整的优化器状态和梯度（若未结合ZeRO）。
    - **内存节省有限**：主要减少单卡的计算负载，内存优化不如ZeRO显著。

---

### **4. 适用场景**

- **Zero3**  
    适合**参数规模极大**的场景（如万亿参数模型），尤其是当单卡无法容纳模型参数时。通常与数据并行结合使用，扩展性强。
    
- **Megatron**  
    适合**计算密集型**场景（如超大Transformer），通过拆分计算图缓解单卡算力瓶颈。常用于模型本身结构可拆分的情况（如矩阵乘法的天然并行性）。
    

---

### **5. 协同使用**

两者可结合使用（如**DeepSpeed + Megatron**），形成混合并行策略：

- **Megatron处理模型并行**：拆分Transformer层的计算。
- **Zero3处理数据并行**：进一步分片参数、梯度和优化器状态。  
    这种组合能同时优化计算负载和内存占用，支持训练超大规模模型（如GPT-3、Megatron-Turing NLG）。

### **总结对比表**

|**特性**|**Zero3**|**Megatron**|
|---|---|---|
|**并行类型**|数据并行（参数分片）|模型并行（计算图拆分）|
|**内存优化重点**|参数、梯度、优化器状态分片|拆分计算负载，内存优化有限|
|**通信频率**|每层计算时动态收集参数|每层计算时同步中间结果|
|**适用场景**|参数规模极大，内存受限|计算密集，模型结构可拆分|
|**典型应用**|结合数据并行训练超大模型|加速Transformer类模型训练|

通过理解两者的差异，可以根据具体任务需求（模型规模、硬件配置）选择合适的并行策略，或结合两者实现更高效的训练。




Zero3（DeepSpeed ZeRO Stage 3）和Megatron的模型并行虽然都涉及参数分片，但两者的设计目标、实现方式和底层逻辑有本质区别。**参数分片并不等同于模型并行**，关键在于分片的目的是什么，以及分片后如何影响计算流程和通信模式。以下是核心区别：

---

### **1. 分片的目标不同**
- **Zero3：分片是为了优化内存，而非拆分计算**  
  - **目标**：通过参数分片（参数、梯度、优化器状态）**减少单卡内存占用**，让超大模型能够被训练。  
  - **本质**：Zero3是**数据并行的扩展**，每个GPU仍然处理完整的输入数据（数据并行），但只保存部分模型参数。  
  - **关键逻辑**：分片是为了消除数据并行中的冗余存储（参数副本），而不是拆分计算任务。

- **Megatron：分片是为了拆分计算任务**  
  - **目标**：将模型的计算图拆分到不同GPU上，**分摊计算负载**，解决单卡算力不足的问题。  
  - **本质**：属于**模型并行（张量并行）**，每个GPU负责模型的一部分计算，处理同一批输入数据的不同部分。  
  - **关键逻辑**：分片是为了让多个GPU协作完成单次前向/反向传播的计算任务。

---

### **2. 分片对计算流程的影响**
- **Zero3的计算流程**  
  - **动态收集参数**：在计算某一层时，GPU通过All-Gather操作从其他GPU收集该层的完整参数，计算完成后释放分片外的参数副本。  
  - **计算逻辑**：每个GPU独立完成该层的完整计算（输入数据是完整的，参数是临时完整的）。  
  - **示例**：GPU0负责参数分片A，GPU1负责分片B。计算某一层时，GPU0和GPU1均需收集A+B形成完整参数，再各自用完整参数处理自己的数据批次。

- **Megatron的计算流程**  
  - **拆分计算任务**：每个GPU只计算模型某一部分的输出，通过通信聚合结果。  
  - **计算逻辑**：输入数据被拆分到不同GPU，每个GPU处理输入的一部分，计算结果需通过通信（如All-Reduce）合并。  
  - **示例**：将矩阵乘法 \( Y = XW \) 按列拆分为 \( W = [W_1, W_2] \)，GPU0计算 \( Y_1 = XW_1 \)，GPU1计算 \( Y_2 = XW_2 \)，最终通过All-Gather合并得到 \( Y = [Y_1, Y_2] \)。

---

### **3. 通信模式的区别**
| **特性**         | **Zero3**                              | **Megatron**                        |
|------------------|----------------------------------------|-------------------------------------|
| **通信触发条件** | 每层计算前收集参数，计算后释放         | 每层计算中同步中间结果（如激活值）  |
| **通信内容**     | 参数分片（All-Gather）                 | 中间计算结果（All-Reduce/All-Gather）|
| **通信频率**     | 每层两次通信（收集参数 + 释放分片）    | 每层多次通信（依赖计算拆分方式）    |
| **通信量**       | 与参数大小相关                         | 与激活值/梯度大小相关               |

---

### **4. 为什么Zero3的参数分片不是模型并行？**
- **模型并行的定义**：将模型的计算图拆分到不同设备，每个设备负责一部分计算任务，输入数据是同一批次的子集。  
- **Zero3的定位**：  
  - Zero3的参数分片是为了**消除数据并行中的内存冗余**，而非拆分计算任务。  
  - 每个GPU仍然处理完整的输入数据，且计算逻辑与单卡训练一致（只是参数是临时收集的）。  
  - 因此，Zero3属于**数据并行的内存优化技术**，而非模型并行。

---

### **5. 类比解释**
- **Megatron模型并行**：  
  类似于工厂的流水线生产，每个工人（GPU）负责产品（模型计算）的一个环节，所有工人协作完成同一件产品。  

- **Zero3数据并行**：  
  类似于多个独立工厂（GPU）同时生产完整的产品（处理完整数据），但每个工厂只存储部分原材料（参数分片），需要时从其他工厂临时借用材料。

---

### **6. 实际应用中的结合**
两者可以协同使用，形成**混合并行策略**（如DeepSpeed + Megatron）：  
1. **Megatron拆分计算图**：将Transformer层的矩阵运算按行/列拆分到多个GPU（模型并行）。  
2. **Zero3分片参数**：在每个模型并行组内，进一步分片参数、梯度和优化器状态（数据并行）。  
这种组合同时优化了计算负载和内存占用，支持训练超大规模模型（如GPT-3、Turing-NLG）。

---

### **总结：关键区别**
| **维度**       | **Zero3**                              | **Megatron模型并行**                |
|----------------|----------------------------------------|-------------------------------------|
| **核心目标**   | 减少数据并行的内存冗余                 | 拆分计算任务，缓解单卡算力瓶颈      |
| **并行类型**   | 数据并行的内存优化技术                 | 模型并行（张量并行）                |
| **计算逻辑**   | 每个GPU处理完整数据，参数临时完整      | 每个GPU处理部分数据，协作完成计算   |
| **通信内容**   | 参数分片                               | 中间计算结果（激活值、梯度）        |
| **适用场景**   | 参数极大，内存受限                     | 计算密集，模型结构可拆分            |

---

### **最终回答**
Zero3的参数分片是**数据并行的内存优化手段**，而Megatron的模型并行是**计算任务的拆分方式**。两者的本质区别在于：  
- Zero3通过分片参数减少冗余存储，但每个GPU仍独立完成完整计算；  
- Megatron通过拆分计算图，让多个GPU协作完成单次计算。  
参数分片本身不改变计算逻辑，因此不属于模型并行。


## 参考资料

DeepSpeed之ZeRO系列：将显存优化进行到底 - basicv8vc的文章 - 知乎
https://zhuanlan.zhihu.com/p/513571706

huggingface的DeepSpeed集成文档： [DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)
