
## 总结

目标是DPO训练Qwen3-A3B，transformers训练速度太慢了。

先后尝试过

1. 安装配置conda环境：已跑通。一开始安装transformer-engine遇到问题没有细究，想着不如用docker方便，就放弃了此方案。后来其他方案问题也不少，因此也同步在继续安装环境。其实当时如果细究下环境安装的问题原因，其实都能解决，也不必尝试其他方案了。。

2. Docker镜像训练：单机已跑通，多机训练速度慢无法接受。单机很快就跑通了，但是多机器训练很慢，找了很久的原因，始终无法加速。因为没有root权限，一些可能行得通的方案也没有再试。

3. k8s集群提交任务训练：已跑通。也是用Docker镜像，但是需要额外安装IB配置以支持多机多卡训练。

最后采取的是安装配置的conda环境去训练的，对于我这种没有root权限的条件下比较适用。而且神奇的是，单节点就能跑了（之前试过docker单节点训练，会在保存模型的时候报错）。

个人用transformers框架的训练速度是13.6w的数据，2个节点，训练了152小时。

用Megatron-Swift的训练速度是7.6w的数据，1个节点，训了了16个小时。

都是采用A100*8的机器，使用Megatron-Swift是transformers速度的10倍！


官方文档也披露了SFT的速度对比，也是10倍的提升：

使用 `megatron sft` 和 `swift sft` 在对 Qwen3-30B-A3B 模型进行全参数微调的训练速度和 GPU 显存使用对比情况如下：

|Megatron-LM|DeepSpeed-ZeRO2|DeepSpeed-ZeRO3|
|---|---|---|---|
|训练速度|9.6s/it|-|91.2s/it|
|显存使用|16 * 60GiB|OOM|16 * 80GiB|

## 官方文档

训练指南：[Megatron-SWIFT训练 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html)

Qwen3最佳实践：[Qwen3最佳实践 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html)

支持的模型：[支持的模型和数据集 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html)

常见问题：[常见问题整理 — swift 3.8.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%95%B4%E7%90%86.html)



训练脚本是官方提供的[ms-swift/examples/train/megatron/rlhf/dpo/moe.sh at main · modelscope/ms-swift · GitHub](https://github.com/modelscope/ms-swift/blob/main/examples/train/megatron/rlhf/dpo/moe.sh)：

```bash
# 8 * 65GiB; 13s/it
# Note: "ms-swift<3.8" does not support DPO packing; please remove --packing true.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron rlhf \
    --rlhf_type dpo \
    --load Qwen3-30B-A3B-Instruct-2507-mcore \
    --dataset AI-ModelScope/orpo-dpo-mix-40k \
    --split_dataset_ratio 0.01 \
    --packing true \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 1 \
    --beta 0.1 \
    --loss_type sigmoid
```

注意这里的模型Qwen3-30B-A3B-Instruct-2507-mcore是用Qwen3-30B-A3B-Instruct-2507转换得到的，转换的命令如下(替换成你自己的模型路径)

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model /models/Qwen3-30B-A3B-Instruct-2507 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /models/Qwen3-30B-A3B-Instruct-2507-mcore \
    --test_convert_precision true
```


## swift 3.6.4版本

	这部分主要是记录踩的坑，可以不看

先尝试了swift 3.6.4，本地安装总是有各种各样的问题（参考下方swift 3.7.1本地安装），所以用了docker镜像去跑。

用swift3.6.4镜像，先尝试了训练脚本examples/train/megatron/sft.sh（因为速度比较快）可以正常训练和保存checkpoint。

但是在脚本里加上export MEGATRON_LM_PATH='/app/Megatron-LM'，其中Megatron-LM是克隆的Megatron-LM.git@core_r0.13.0，理论上和前一种训练状况是一样的，但是！一到保存的时候就会报错！而且报错信息非常地无效，也就是说看不出来是为什么报的错！

一开始以为最新的megatron-LM有问题，用这个就会保存不成功。不用这个就可以。

- 但是我用的是分支啊，这个分支又没变。到现在也不知道原因是什么，也不想去追究了，于我无益。
    
后来又尝试训练Qwen-A3B DPO，特别慢，这个版本不支持expert_tensor_parallel_size，所以老是OOM。这个时候看到出来了3.7.1的镜像，于是放弃3.6.4了。

## swift 3.7.1本地安装

	这部分主要是记录踩的坑，可以不看

参考官方指南安装：

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

安装环境之前可以进行`module load cuda/12.6` 以及高版本gcc的load，防止安装过程中的版本不匹配、gcc版本过低等报错。

先安装pytorch，因为2.7版本下安装的flash-attention用不了会报错，所以安装的是2.6版本

`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126`

然后可以安装swift `pip install ms-swift -U`，也可以源码安装

接着安装transformer-engine

- 看了看官方README，用`conda install -c conda-forge transformer-engine-torch`就成功了，pip安装总是不成功。
- 但是后来发现！这样安装之后torch又有问题。。。会报错ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?，以及前面还有个torch的报错
- 网上说需要重新安装torch和transformers，但是重新安装torch的时候又报了别的错，人都麻了。
- 还是用朴素的pip吧，`pip install --no-build-isolation transformer_engine[pytorch]`，仔细看看报错torch/include/ATen/cudnn/cudnn-wrapper.h:3:10: fatal error: cudnn.h: No such file or directory。确实我们的机器上没有安装cudnn
- 那就手动装一个，装到虚拟环境里`conda install -c nvidia cudnn`
- 然后再装transformer_engine，居然成功了？就这么简单？还是得仔细看看报错信息，不懂就问DeepSeek。

如果还是不行的话，设置一下环境变量：

```bash
# 查找 cudnn.h 文件位置
find $CONDA_PREFIX -name "cudnn.h" 2>/dev/null

# 查找 libcudnn 库文件位置
find $CONDA_PREFIX -name "libcudnn*" 2>/dev/null | head -1

# 设置头文件路径（通常为 $CONDA_PREFIX/include）
export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/include

# 设置库文件路径（通常为 $CONDA_PREFIX/lib）
export CUDNN_LIBRARY_DIR=$CONDA_PREFIX/lib

# 验证设置
echo "CUDNN_INCLUDE_DIR: $CUDNN_INCLUDE_DIR"
echo "CUDNN_LIBRARY_DIR: $CUDNN_LIBRARY_DIR"
```

在verl环境安装的时候也遇到问题，verl的安装命令是`NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1`
总是失败，看原因也看不出来什么。
后来就换成这样安装：`pip install --no-build-isolation transformer_engine[pytorch]`（可以指定版本）
前提是要module load cuda/12.4, gcc 13.3.0, 对应的nccl版本.

安装flash_attention：`pip install flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`

测试flash_attention正常：`import flash_attn_2_cuda as flash_attn_gpu`（因为每次都是这里报错，不行的话就下载源码安装）

安装apex：

```bash
# apex
git clone https://github.com/NVIDIA/apex
cd apex
# https://github.com/modelscope/ms-swift/issues/4176
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

安装Megatron

`pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0`

然后就可以训练啦～

## swift 3.7.1-docker

更新了swift 3.7.1，训练qwen2.5-7b模型是可以的。也可以正常保存。Qwen-A3B DPO也可以正常保存了！是镜像的问题！相关issue：

- https://github.com/modelscope/ms-swift/issues/5435

- pip install git+[https://github.com/modelscope/ms-swift.git@release/3.7](https://github.com/modelscope/ms-swift.git@release/3.7)

另外还需要手动下载Megatron-LM的core_r0.13.0分支（参考官方指南），然后export MEGATRON_LM_PATH='/xxx/Megatron-LM'，否则会报错No module named 'megatron.training'

但是新的问题又来了：Qwen-A3B DPO训练保存的时候会报错cannot allocate memory，看起来是内存不足。单节点训练，内存一共是1T。

不用Megatron训练，用swift训练，也不行，要么报错，要么特别慢。自从看过Megatron的训练速度，真是忍受不了别的速度了。也有其他人尝试用swift训练，真是特别慢：
- [使用deepspeed zer3训练Qwen3-30B-A3B-Instruct-2507时 加载完模型和数据 训练进度条会卡住不动 · Issue #5400 · modelscope/ms-swift · GitHub](https://github.com/modelscope/ms-swift/issues/5400)

所以接下来要尝试用多节点训练，宿主机上多节点训练倒是会，docker的多节点训练怎么弄呢？

	多节点训练这块，紧急求助了DeepSeek和ChatGPT（都是官网上用，GPT也用思考模式，非会员），用下来总体感受是，DeepSeek还是更好用，它给我的方案更好用，正确率更高，出错的概率比GPT要低。

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
  -e MASTER_ADDR=主节点宿主机IP \
  -e MASTER_PORT=29500 \
  -e NPROC_PER_NODE=8 \
  modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1 \
  bash -c "cd /app && /bin/bash examples/train/megatron/multi-node/node1_moe.sh"
```

此处的node1_moe.sh和前面的训练脚本其实是一样的

主节点上启动后，去工作节点的宿主机上看看能不能连通：

```bash
nc -vz <MASTER_ADDR> <MASTER_PORT>
```

发现无法连通，求助AI助手，都说加上`--network=host`，但是我试过也不能连通。

网上有一个建立docker之间通信的方案：[docker容器中deepspeed多机多卡集群分布式训练大模型\_deepspeed多机多卡训练-CSDN博客](https://blog.csdn.net/Q2024107/article/details/146428595)

但是运行`docker swarm join`的时候就报错无法连接了，可能是防火墙的问题。没有root权限比较难搞。感觉这条路又死了，嗯还有一个方案就是在k8s上弄。

## 多节点训练-k8s-gpt版本-不好用

总结来说，gpt先后给了我2个方案，第一个是使用StatefulSet，第二个是使用Job。

前者不好用的原因是restartPolicy只支持Always，也就是说，**StatefulSet PodTemplate 中的容器一旦退出，无论成功还是失败，都会被 Kubernetes 自动重启**，即使训练完成了也还是会启动。显然这不是我想要的。

后者是一次性任务，训练结束 Pod 停掉，不重启。方案其实是对的，但是gpt给我的yaml配置文件不对，导致我在测试的时候报错，解析不了MASTER_ADDR，这显然不可行，因为我是要多节点训练的。

**StatefulSet vs Job**
- StatefulSet 会保证 Pod 名称固定（`megatron-0.megatron`），DNS 解析可靠。
- Job 创建 Pod 名称是随机的（如 `megatron-job-xxxxx`），没有固定的 `megatron-0`，所以用 `megatron-0.megatron` 解析失败。

StatefulSet 主要用于：

- **有状态服务**（比如数据库、分布式训练节点）
- **保证 Pod 姓名固定** (`megatron-0`、`megatron-1`)
- **保证卷和 DNS 稳定**

相关脚本

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

### 基础脚本

DeepSeek给的版本，很不错，能够成功运行了。

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

能跑通，但是训练特别特别慢，等了1、2个小时才出第一个step的日志。。。单节点就很快。

后来想了想，可能是没用上宿主机的IB卡，所以训练很慢。

那么就研究如何在k8s里用上宿主机的IB卡，提升多节点训练的效率。

### IB配置

还是问DeepSeek，它告诉我需要安装 NVIDIA 的 Kubernetes Device Plugin 来暴露 IB 设备，虽然它给的安装方法我跑不起来，但是根据它给的链接，我找到了相关的Github仓库，就是这个：[GitHub - Mellanox/k8s-rdma-shared-dev-plugin](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin)

按照指南操作：
> 
> **1.** Use CNI plugin such as Contiv, Calico, Cluster（这一步啥也没做）
> 
> Make sure to configure ib0 or appropriate IPoIB netdevice as the parent netdevice for creating overlay/virtual netdevices.
> 
> **2.** Create ConfigMap and deploy Device Plugin
> 
> Deploy device plugin and create config map to describe mode as "hca" mode. This is per node configuration.
> 
> ```
> cd deployment/k8s/base
> kubectl apply -k .
> ```
> 
> **3.** Create Test pod
> 
> Create test pod which requests 1 vhca resource.
> 
> ```
> kubectl create -f example/test-hca-pod.yaml
> ```
> 
> Deploy the device plugin with CDI support
> 
> To use the device plugin with [CDI](https://github.com/cncf-tags/container-device-interface) support, do the following:
> 
> ```
> cd deployment/k8s/base/overlay
> kubectl apply -k .
> ```

让DeepSeek解释每一步在做什么：

#### 1. **使用 CNI 插件（如 Contiv, Calico, Cluster）**

**含义**：

- **CNI (Container Network Interface)**：是 Kubernetes 的网络插件标准，Calico、Cilium、Flannel 等都是具体的 CNI 插件，负责为 Pod 分配 IP 地址并管理网络连通性。
    
- **`ib0`**：这是宿主机上 IP over InfiniBand (IPoIB) 网络接口的名称。它是在物理 IB 网卡之上创建的一个 IP 协议接口。
    
- **`parent netdevice`**：在很多 CNI 配置中，你需要指定一个物理网络接口作为“父接口”，CNI 插件会基于它为 Pod 创建虚拟网络接口（veth pair 等）。
    

**操作目的**：  
这一步是**配置 Kubernetes 集群的底层网络**，确保 Pod 的网络流量（尤其是 RDMA 流量）能够正确地通过高速的 IB 网络 (`ib0`) 进行传输，而不是通过普通的以太网（如 `eth0`）。你需要查阅你所使用的 CNI 插件（如 Calico）的文档，来配置如何将 `ib0` 作为其底层父接口。
#### 2. **创建 ConfigMap 并部署设备插件**

**含义**：

- **Device Plugin**：这是一个运行在每个节点上的 DaemonSet Pod。它的作用是向 Kubernetes 汇报节点上的特殊硬件资源（这里指 RDMA 设备），并负责在 Pod 启动时将设备挂载到容器中。
    
- **ConfigMap**：一种 Kubernetes 资源，用于保存配置文件。这里用它来告诉设备插件应以哪种模式工作。
    
- **`"hca" mode`**：`HCA` (Host Channel Adapter) 是 InfiniBand 网卡的术语。这种模式意味着插件将**直接暴露物理 RDMA 设备**给 Pod。
    

**操作目的**：  
这一步是**安装并配置 RDMA 设备插件本身**。通过运行 `kubectl apply -k .`（`kustomize` 命令），你会部署一个包含以下内容的 DaemonSet：

- 设备插件的 Pod
    
- 所需的 RBAC 权限（ServiceAccount, Role, RoleBinding）
    
- 一个 ConfigMap，其中设置了 `config.json` 指定模式为 `"hca"`
    

部署后，每个节点上都会运行一个设备插件 Pod，它会让 Kubernetes 调度器知道这个节点有 RDMA 设备资源（例如 `rdma/hca: 1`）。

#### 3. **创建测试 Pod**

**含义**：

- **`requests 1 vhca resource`**：在 Pod 的资源配置中，你需要像请求 CPU、内存一样，请求 RDMA 资源。`vhca` 可能指一个虚拟化的 HCA 资源。Pod 会向 Kubernetes 申请一个 `rdma/hca` 资源。
    

**操作目的**：  
这一步是**验证设备插件是否工作正常**。`example/test-hca-pod.yaml` 是一个示例 Pod 的配置文件。

创建这个 Pod 后，如果它能成功启动（`Running` 状态），并且你能 `exec` 进入容器执行 `ibv_devices` 等命令看到 RDMA 设备，就证明整个流程从调度到设备挂载都是成功的。

#### 4. **部署支持 CDI 的设备插件**

**含义**：

- **CDI (Container Device Interface)**：这是一个更新的、更通用的标准，用于将设备注入到容器中。它旨在取代和扩展最初的 `device plugin` 框架的功能。
    
- **Overlay**：在 Kustomize 中，`overlay` 通常用于基于一个基础配置（`base`）生成一个具有差异化配置的新版本。
    

**操作目的**：  
这一步提供了**另一种部署设备插件的方式**。它部署的是同一个设备插件，但可能是配置为使用更新的 CDI 接口来向容器提供设备，而不是传统的设备插件方式。这通常能提供更好的灵活性和功能。命令 `kubectl apply -k .` 会应用 `overlay` 目录下的配置，部署支持 CDI 的设备插件版本。

### 更新训练脚本

配置完成后，更新训练文件中的参数，新版配置：

```
# direct-gpu-training.yaml
# 为了让节点能够相互发现,需要创建服务
apiVersion: v1
kind: Service
metadata:
  name: training-master
  namespace: my-training
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
  namespace: my-training
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
  namespace: my-training
  labels:
    role: master
spec:
  restartPolicy: OnFailure
  # 这是节点调度的策略，可以忽略
  tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "my-training"
    effect: "NoSchedule"
  containers:
  - name: training-container
    image: modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1
    # 添加安全上下文和设备请求
    securityContext:
      capabilities:
        add: ["IPC_LOCK", "SYS_RESOURCE", "NET_RAW", "NET_ADMIN"]
    command: ["/bin/bash", "/app/examples/train/megatron/rlhf/dpo/moe_qwen.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master.my-training.svc.cluster.local"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "0"
    - name: NPROC_PER_NODE
      value: "8"
    - name: NNODES
      value: "2"
    # - name: CUDA_VISIBLE_DEVICES
    #   value: "0,1"
    - name: MEGATRON_LM_PATH
      value: "/app/Megatron-LM"
    - name: MODELSCOPE_CACHE
      value: "/app/shared"
    - name: NCCL_DEBUG
      value: "INFO"
    - name: NCCL_IB_DISABLE
      value: "0"  # 确保启用IB
    # - name: NCCL_IB_HCA
    #   value: "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4"  # 根据实际情况调整
    # - name: NCCL_IB_GID_INDEX
    #   value: "3"
    # - name: NCCL_IB_TC
    #   value: "136"
    # - name: NCCL_IB_QPS_PER_CONNECTION
    #   value: "4"
    # - name: NCCL_IB_RETRY_CNT
    #   value: "7"
    # - name: NCCL_IB_TIMEOUT
    #   value: "22"
    # - name: NCCL_IB_SL
    #   value: "0"
    # - name: NCCL_SOCKET_IFNAME
    #   value: "ib0"  # 使用IB接口而不是以太网
    resources:
      requests:
        cpu: "32"
        memory: 512G
        nvidia.com/gpu: 8  # 请求 GPU
        rdma/hca_shared_devices_a: 1  # 请求IB设备
      limits:
        cpu: "64"
        memory: 512G
        nvidia.com/gpu: 8  # GPU 上限
        rdma/hca_shared_devices_a: 1  # 请求IB设备
    volumeMounts:
      - name: dshm
        mountPath: /dev/shm  # 挂载更大的共享内存
      - name: workdir
        mountPath: /app
      # - name: ib-devices
      #   mountPath: /dev/infiniband
      # - name: ib-config
      #   mountPath: /etc/libibverbs.d
      # - name: ib-libraries
      #   mountPath: /usr/lib64/libibverbs
  volumes:
    # - name: ib-devices
    #   hostPath:
    #     path: /dev/infiniband
    # - name: ib-config
    #   hostPath:
    #     path: /etc/libibverbs.d
    # - name: ib-libraries
    #   hostPath:
    #     path: /usr/lib64/libibverbs
    - name: dshm
      # emptyDir:
      #   medium: Memory
      #   sizeLimit: 50Gi  # 根据需要调整
      hostPath:
        path: /dev/shm
        type: Directory
    - name: workdir
      hostPath:
        path: /xxx/ms-swift
        type: Directory
---
# Worker 1 Pod
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-1
  namespace: my-training
  labels:
    role: worker
spec:
  restartPolicy: OnFailure
  tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "my-training"
    effect: "NoSchedule"
  containers:
  - name: training-container
    image: modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1
    securityContext:
      capabilities:
        add: ["IPC_LOCK", "SYS_RESOURCE", "NET_RAW", "NET_ADMIN"]
    command: ["/bin/bash", "/app/examples/train/megatron/rlhf/dpo/moe_qwen.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master.my-training.svc.cluster.local"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "1"
    - name: NPROC_PER_NODE
      value: "8"
    - name: NNODES
      value: "2"
    # - name: CUDA_VISIBLE_DEVICES
    #   value: "0,1"
    - name: MEGATRON_LM_PATH
      value: "/app/Megatron-LM"
    - name: MODELSCOPE_CACHE
      value: "/app/shared"
    - name: NCCL_DEBUG
      value: "INFO"
    - name: NCCL_IB_DISABLE
      value: "0"  # 确保启用IB
    resources:
      requests:
        cpu: "32"
        memory: 512G
        nvidia.com/gpu: 8  # 请求 GPU
        rdma/hca_shared_devices_a: 1  # 请求IB设备
      limits:
        cpu: "64"
        memory: 512G
        nvidia.com/gpu: 8  # GPU 上限
        rdma/hca_shared_devices_a: 1  # 请求IB设备
    volumeMounts:
      - name: dshm
        mountPath: /dev/shm  # 挂载更大的共享内存
      - name: workdir
        mountPath: /app
  volumes:
    - name: dshm
      # emptyDir:
      #   medium: Memory
      #   sizeLimit: 50Gi  # 根据需要调整
      hostPath:
        path: /dev/shm
        type: Directory
    - name: workdir
      hostPath:
        path: /xxx/ms-swift
        type: Directory

```


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