
训练指南：[Megatron-SWIFT训练 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html)

常见问题：[常见问题整理 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%95%B4%E7%90%86.html)


## swift 3.6.4版本

一开始以为最新的megatron-LM有问题，用这个就会保存不成功。不用这个就可以。

- 但是我用的是分支啊，这个分支又没变应该。
    

用swift3.6.4镜像，训练脚本examples/train/megatron/sft.sh可以正常训练和保存checkpoint。

但是在脚本里加上export MEGATRON_LM_PATH='/app/Megatron-LM'，其中Megatron-LM是克隆的Megatron-LM.git@core_r0.13.0，理论上和前一种训练状况是一样的，但是保存的时候就会报错，且报错信息不完善，原因未知
    
- swift 3.6.4版本训练Qwen-A3B DPO，特别慢，不支持expert_tensor_parallel_size，老是OOM


## swift 3.7.1

用swift 3.7.1呢，总是到保存模型哪一步就报错，太奇怪了！

更新了swift 3.7.1，训练qwen2.5-7b模型是可以的。也可以正常保存。Qwen-A3B DPO也可以正常保存了！是镜像的问题！

- pip install git+[https://github.com/modelscope/ms-swift.git@release/3.7](https://github.com/modelscope/ms-swift.git@release/3.7)
    
- https://github.com/modelscope/ms-swift/issues/5435

但是Qwen-A3B DPO训练保存的时候会报错cannot allocate memory，看起来是内存不足。单节点训练，内存一共是1T。

不用Megatron训练，用swift训练，