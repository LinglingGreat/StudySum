
训练指南：[Megatron-SWIFT训练 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html)

常见问题：[常见问题整理 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%95%B4%E7%90%86.html)


## swift 3.6.4版本

一开始以为最新的megatron-LM有问题，用这个就会保存不成功。不用这个就可以。

- 但是我用的是分支啊，这个分支又没变应该。
    

用swift3.6.4镜像，训练脚本examples/train/megatron/sft.sh可以正常训练和保存checkpoint。

但是在脚本里加上export MEGATRON_LM_PATH='/app/Megatron-LM'，其中Megatron-LM是克隆的Megatron-LM.git@core_r0.13.0，理论上和前一种训练状况是一样的，但是保存的时候就会报错，且报错信息不完善，原因未知
    
- swift 3.6.4版本训练Qwen-A3B DPO，特别慢，不支持expert_tensor_parallel_size，老是OOM

## swift 3.7.1安装

官方指南：

```bash
# Recommended PyTorch version: 2.5 / 2.6
pip install pybind11

# transformer_engine
# If an installation error occurs, you can refer to this issue for resolution: https://github.com/modelscope/ms-swift/issues/3793
pip install --no-build-isolation transformer_engine[pytorch]
# Or install using the following command
# pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]

# apex
git clone https://github.com/NVIDIA/apex
cd apex
# https://github.com/modelscope/ms-swift/issues/4176
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# megatron-core
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0

# If you are using multi-node training, please additionally set the `MODELSCOPE_CACHE` environment variable to a shared storage path.
# This will ensure that the dataset cache is shared, thereby speeding up preprocessing.
export MODELSCOPE_CACHE='/xxx/shared'

# Megatron-LM
# The training module in the dependent library Megatron-LM will be cloned and installed by swift via `git clone`. Alternatively, you can use the environment variable `MEGATRON_LM_PATH` to point to the path of an already downloaded repository (in offline environments, use the [core_r0.13.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.13.0)).
export MEGATRON_LM_PATH='/xxx/Megatron-LM'
```

安装transformer-engine

- 用的是`conda install -c conda-forge transformer-engine-torch`，pip安装总是不成功。
- 但是这样安装之后似乎torch又有问题。。。会报错ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?


## swift 3.7.1-docker

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

但是如果不加的话虽然可以启动，但是无法通信。

网上有一个建立docker之间通信的方案：[docker容器中deepspeed多机多卡集群分布式训练大模型\_deepspeed多机多卡训练-CSDN博客](https://blog.csdn.net/Q2024107/article/details/146428595)

但是运行`docker swarm join`的时候就报错无法连接了，可能是防火墙的问题。没有root权限比较难搞。

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

```yaml
# direct-gpu-training.yaml
apiVersion: v1
kind: Service
metadata:
  name: training-master
spec:
  selector:
    role: master
  ports:
    - protocol: TCP
      port: 29500
      name: training
  clusterIP: None
---
apiVersion: v1
kind: Service
metadata:
  name: training-worker
spec:
  selector:
    role: worker
  ports:
    - protocol: TCP
      port: 29500
      name: training
  clusterIP: None
---
# Master Pod
apiVersion: v1
kind: Pod
metadata:
  name: training-master
  labels:
    role: master
spec:
  restartPolicy: OnFailure
  containers:
  - name: training-container
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    command: ["/bin/bash", "/data/train.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "0"
    - name: NPROC_PER_NODE
      value: "4"
    - name: NNODES
      value: "3"
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1,2,3"
    resources:
      requests:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # 请求 GPU
      limits:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # GPU 上限
    volumeMounts:
    - name: local-data
      mountPath: /data
  volumes:
  - name: local-data
    hostPath:
      path: /path/to/your/local/data
      type: Directory
---
# Worker 1 Pod
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-1
  labels:
    role: worker
spec:
  restartPolicy: OnFailure
  containers:
  - name: training-container
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    command: ["/bin/bash", "/data/train.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "1"
    - name: NPROC_PER_NODE
      value: "4"
    - name: NNODES
      value: "3"
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1,2,3"
    resources:
      requests:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # 请求 GPU
      limits:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # GPU 上限
    volumeMounts:
    - name: local-data
      mountPath: /data
  volumes:
  - name: local-data
    hostPath:
      path: /path/to/your/local/data
      type: Directory
---
# Worker 2 Pod
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-2
  labels:
    role: worker
spec:
  restartPolicy: OnFailure
  containers:
  - name: training-container
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    command: ["/bin/bash", "/data/train.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "2"
    - name: NPROC_PER_NODE
      value: "4"
    - name: NNODES
      value: "3"
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1,2,3"
    resources:
      requests:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # 请求 GPU
      limits:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4  # GPU 上限
    volumeMounts:
    - name: local-data
      mountPath: /data
  volumes:
  - name: local-data
    hostPath:
      path: /path/to/your/local/data
      type: Directory

```


```bash
# 启动任务
kubectl apply -f xxx.yaml

# 删除所有训练相关的资源
kubectl delete -f direct-gpu-training.yaml

# 或者逐个删除
kubectl delete pod training-master training-worker-1 training-worker-2
kubectl delete service training-master training-worker
```

能跑通，但是训练特别特别慢，等了1、2个小时才出第一个step的日志。。。单节点就很快。可能是没用上宿主机的IB卡。

# 使用 py-spy 诊断训练卡住问题

`py-spy dump --pid xxx` 是一个非常强大的 Python 诊断工具命令，**确实可以帮助您排查训练卡在哪个位置**。

## py-spy 是什么

`py-spy` 是一个 Python 程序的性能分析工具，它可以在不修改代码、不重启进程的情况下：
1. 查看 Python 程序的当前调用栈（堆栈跟踪）
2. 分析性能瓶颈
3. 诊断死锁或卡住的问题

## 如何使用 py-spy 诊断训练卡住问题

### 1. 首先进入容器并安装 py-spy

```bash
# 进入训练容器
kubectl exec -it training-master -- /bin/bash

# 安装 py-spy (如果容器内没有)
pip install py-spy

# 或者直接下载预编译版本 (如果没有pip)
curl -L https://github.com/benfred/py-spy/releases/download/v0.3.14/py-spy-v0.3.14-x86_64-unknown-linux-gnu.tar.gz | tar -xz
cp py-spy /usr/local/bin/
```

### 2. 找到 Python 进程的 PID

```bash
# 在容器内查看进程
ps aux | grep python

# 或者直接查找训练进程
pgrep -f "python.*train"

# 输出示例:
# root        123   0.0  0.0   1234  1234 pts/0    S+   00:00   0:00 python train.py
# 这里的 123 就是 PID
```

### 3. 使用 py-spy 查看调用栈

```bash
# 查看当前所有线程的堆栈
py-spy dump --pid 123

# 或者生成火焰图 (更直观)
py-spy record -o profile.svg --pid 123 --duration 10
```

### 4. 分析 py-spy 输出

`py-spy dump` 的输出会显示每个线程当前执行到的代码位置，例如：

```
Thread 0x7F8A1B7FE700 (idle): "MainThread"
    train (train.py:123)
    main (main.py:45)
    <module> (__main__.py:1)

Thread 0x7F8A1AFFD700 (running): "DataLoader"
    __getitem__ (dataset.py:67)
    _get_batch (dataloader.py:234)
    run (threading.py:870)
```

从输出中您可以看到：
- 主线程卡在 `train.py` 的第 123 行
- 数据加载线程正在运行

## 常见卡住的原因及解决方案

根据您的描述卡在 `[INFO:swift] The training of Epoch 0 starts...`，可能的原因包括：

### 1. 数据加载问题
```bash
# 如果py-spy显示卡在数据加载相关代码
py-spy dump --pid xxx | grep -i data
```

**解决方案**：
- 检查数据路径是否正确
- 减少数据加载 workers 数量
- 使用更小的测试数据集验证

### 2. 分布式通信问题（NCCL）
```bash
# 检查是否有NCCL相关线程卡住
py-spy dump --pid xxx | grep -i nccl
```

**解决方案**：
- 检查网络连通性
- 调整 NCCL 环境变量
- 增加 NCCL 超时时间

### 3. 资源等待
```bash
# 检查是否在等待某些资源
py-spy dump --pid xxx | grep -i wait
```

**解决方案**：
- 检查 GPU 内存是否足够
- 检查共享内存设置

### 4. 死锁或条件竞争

**解决方案**：
- 检查多线程/多进程同步代码
- 使用更简单的配置测试

## 完整诊断流程

```bash
# 1. 进入容器
kubectl exec -it training-master -- /bin/bash

# 2. 安装py-spy (如果需要)
pip install py-spy || curl -L https://github.com/benfred/py-spy/releases/download/v0.3.14/py-spy-v0.3.14-x86_64-unknown-linux-gnu.tar.gz | tar -xz && cp py-spy /usr/local/bin/

# 3. 找到PID
PID=$(pgrep -f "python.*train")
echo "Python PID: $PID"

# 4. 获取堆栈信息
py-spy dump --pid $PID

# 5. 如果还需要更多信息，生成火焰图
py-spy record -o /data/profile.svg --pid $PID --duration 30

# 6. 退出容器并将火焰图拷贝出来
exit
kubectl cp training-master:/data/profile.svg ./profile.svg
```

## 其他有用的诊断命令

```bash
# 查看进程状态
cat /proc/$PID/status

# 查看进程打开的文件
ls -la /proc/$PID/fd

# 查看进程内存映射
cat /proc/$PID/maps

# 查看进程限制
cat /proc/$PID/limits

# 使用strace跟踪系统调用 (高级)
strace -p $PID -f -o strace.out
```

## 预防措施

1. **添加超时机制**：在训练代码中添加超时检查
2. **增加日志**：在关键步骤添加更详细的日志
3. **使用检查点**：定期保存训练状态，便于恢复
4. **监控资源**：实时监控 GPU、内存、网络使用情况

## 示例：分析后的行动

假设 `py-spy dump` 显示卡在数据加载器，您可以：

```bash
# 临时减少数据加载workers
export NUM_WORKERS=1

# 或者跳过数据预处理
export SKIP_PREPROCESS=true

# 重新启动训练
```

通过使用 `py-spy`，您应该能够精确地找到代码卡住的位置，从而有针对性地解决问题。这个工具特别适合诊断这种"看似正常但实际已卡住"的情况。