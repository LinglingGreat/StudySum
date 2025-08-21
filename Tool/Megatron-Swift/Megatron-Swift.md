
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

## 多节点训练-docker

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

## 多节点训练-k8s-gpt版本-不好用

脚本

```
# Kubernetes YAML for Megatron multi-node training (StatefulSet + Headless Service)
# - Replace <YOUR_IMAGE> with your container image (must contain megatron, torch, CUDA, etc.).
# - Adjust replicas and NNODES together (default below: 2 nodes).
# - This file combines Service + StatefulSet so you can `kubectl apply -f megatron_pytorch_statefulset.yaml`.

---
apiVersion: v1
kind: Service
metadata:
  name: megatron
  labels:
    app: megatron
spec:
  clusterIP: None  # headless service -> stable DNS for each pod: megatron-0.megatron
  selector:
    app: megatron
  ports:
    - port: 29500
      name: rdzv

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: megatron
spec:
  serviceName: "megatron"      # must match the headless service above
  replicas: 1                    # <== change this to the number of nodes (NNODES)
  selector:
    matchLabels:
      app: megatron
  template:
    metadata:
      labels:
        app: megatron
    spec:
      # restartPolicy: Always
      # 本地硬盘挂载路径
      volumes: 
      - name: workdir
        hostPath:
          path: /xxx/ms-swift
          type: Directory
      - name: models
        hostPath:
          path: /xxx/models
          type: Directory
      - name: datasets
        hostPath:
          path: /xxx/datasets
          type: Directory
      containers:
      - name: trainer
        # REQUIRED: put your image here (with CUDA + PyTorch + Megatron installed)
        image: modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1
        imagePullPolicy: IfNotPresent
        # Request/limit 4 GPUs per Pod to match your NPROC_PER_NODE=4
        resources:
          limits:
            nvidia.com/gpu: 4
        env:
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True"
        - name: MEGATRON_LM_PATH
          value: "/app/Megatron-LM"   # you mount your repo here in the container
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        - name: MASTER_PORT
          value: "29500"
        - name: NPROC_PER_NODE
          value: "4"                 # matches CUDA_VISIBLE_DEVICES count
        - name: NNODES
          value: "1"                 # must match .spec.replicas
        # Command: compute NODE_RANK from hostname, set MASTER_ADDR to pod-0 headless DNS,
        # launch torchrun which starts NPROC_PER_NODE processes and runs your `megatron sft` command.
        args:
        - /bin/bash
        - -c
        - |
          # compute ordinal (pod index) from HOSTNAME like 'megatron-0'
          ordinal=$(echo $HOSTNAME | awk -F'-' '{print $NF}')
          export NODE_RANK=${ordinal}

          # Use the StatefulSet pod 0 as rendezvous/master address
          export MASTER_ADDR="megatron-0.megatron"
          export MASTER_PORT=${MASTER_PORT}

          echo "NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"

          # Optional: change to working dir where your code and checkpoints live
          mkdir -p /app
          cd /app || exit 1

          # Note: modify the megatron CLI args below (--load ... and other args) as needed.
          # We're using torchrun to spawn processes per GPU. Replace `--your-other-args`.

          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
          NPROC_PER_NODE=2 \
          CUDA_VISIBLE_DEVICES=0,1 \
          megatron sft \
              --load /models/Qwen2.5-7B-Instruct-mcore \
              --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
                        'AI-ModelScope/alpaca-gpt4-data-en#500' \
                        'swift/self-cognition#500' \
              --tensor_model_parallel_size 2 \
              --sequence_parallel true \
              --micro_batch_size 16 \
              --global_batch_size 16 \
              --recompute_granularity full \
              --recompute_method uniform \
              --recompute_num_layers 1 \
              --finetune true \
              --cross_entropy_loss_fusion true \
              --lr 1e-5 \
              --lr_warmup_fraction 0.05 \
              --min_lr 1e-6 \
              --max_epochs 1 \
              --save megatron_output/Qwen2.5-7B-Instruct \
              --save_interval 10 \
              --max_length 2048 \
              --system 'You are a helpful assistant.' \
              --num_workers 4 \
              --no_save_optim true \
              --no_save_rng true \
              --dataset_num_proc 4 \
              --model_author swift \
              --model_name swift-robot

        volumeMounts:
        - name: models
          mountPath: /models   # mount your code here inside the image or via PVC
        - name: datasets
          mountPath: /datasets # mount your datasets here inside the image or via PVC
        - name: workdir
          mountPath: /app
      # volumes:
      # - name: workdir
      #   emptyDir: {}
  volumeClaimTemplates:
  # Uncomment and modify the section below if you want per-pod persistent storage (PVC).
  # - metadata:
  #     name: megatron-pvc
  #   spec:
  #     accessModes: [ "ReadWriteOnce" ]
  #     resources:
  #       requests:
  #         storage: 200Gi

# Notes / Next steps:
# 1) Build an image that contains megatron and all dependencies and push to your registry.
# 2) If your image doesn't already include the Megatron-LM code at /app/Megatron-LM,
#    you can mount a ConfigMap, an initContainer to git-clone, or use a PVC (see comments).
# 3) Adjust replicas and NNODES together (replicas=NNODES). If you change replicas, update NNODES env.
# 4) Apply: kubectl apply -f megatron_pytorch_statefulset.yaml
# 5) Check pods: kubectl get pods -l app=megatron -o wide
# 6) Check logs: kubectl logs megatron-0 -c trainer
# 7) If you use NetworkPolicy, allow traffic between pods on port 29500.

```


```bash
# 查看状态
kubectl describe pod megatron-0

# 查看日志
kubectl logs megatron-0 -c trainer

# 查看pod 状态
kubectl get pods -l app=megatron -o wide

# 删除 StatefulSet 对象
kubectl delete statefulset megatron

```

但是这个会一直重启运行，即使成功了也会再次运行。

## 多节点训练-k8s-deepseek版本

