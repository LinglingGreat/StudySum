
DeepSpeed之ZeRO系列：将显存优化进行到底 - basicv8vc的文章 - 知乎
https://zhuanlan.zhihu.com/p/513571706

huggingface的DeepSpeed集成文档： [DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed)


## offload_optimizer 和 offload_param 

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


