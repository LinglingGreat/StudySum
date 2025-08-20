
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

不用Megatron训练，用swift训练，会报错。

## 多节点训练

在每个节点上运行命令，其中在主节点上设置NODE_RANK=0，工作节点上设置NODE_RANK=1

```bash
docker run --rm -it --gpus all -p 29500:29500 \
  --shm-size=50G \
  -v xxx/ms-swift:/app \
  -v xxx/models:/models \
  -v xxx/datasets:/datasets \
  -e NNODES=2 \
  -e NODE_RANK=0 \
  -e MASTER_ADDR=主节点IP \
  -e MASTER_PORT=29500 \
  -e NPROC_PER_NODE=8 \
  modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1 \
  bash -c "cd /app && /bin/bash examples/train/megatron/multi-node/node1_moe.sh"
```

node1_moe.sh

```bash
# For more information on multi-node training launch methods, refer to:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node

export MEGATRON_LM_PATH='/app/Megatron-LM'
export MODELSCOPE_CACHE='/app/shared'
timestamp=$(date +%Y%m%d-%H%M%S)

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type dpo \
    --load /models/Qwen3-30B-A3B-Instruct-2507-mcore \
    --dataset '/datasets/02.parquet' \
            '/datasets/01.parquet' \
    --custom_register_path /app/examples/custom/mydataset.py \
    --split_dataset_ratio 0.01 \
    --pipeline_model_parallel_size 2 \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --rpo_alpha 1 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 10 \
    --save_interval 10 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --use_precision_aware_optimizer true \
    --beta 0.1 \
    --loss_type sigmoid #> megatron_output/Qwen3-30B-A3B-Instruct-2507/moe_qwen_${timestamp}.log &
```

主节点上启动后，去工作节点的宿主机上看看能不能连通：

```bash
nc -vz <MASTER_ADDR> <MASTER_PORT>
```

之前试过docker run的时候用`--network=host`，却发现无法连通，即使单节点也无法启动训练脚本。


