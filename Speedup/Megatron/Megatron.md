# 源码解读

CodeGeeX

![](img/Pasted%20image%2020240727172338.png)

![](img/Pasted%20image%2020240727172348.png)

**codegeex采用的是8头TP，192头DP，共1536块GPU进行训练，采用的训练框架为Megatron + DeepSpeed ZeRO2**

## 预训练代码整体架构

`pretrain`函数主要包含以下4个内容：

- **初始化Megatron**：设置分布式训练环境。主要目的是设置DP/TP/PP进程组，并为每一个进程分配GPU。
- **设置model，optimizer和lr schedule**：在CPU上定义好模型，再将其按照第1步中定义好的分布式框架，把模型切割并搬运到GPU上。
- **处理train/val/test数据集**：按第1步定义好的分布式框架，对数据集进行切分。
- **训练模型**：在分布式环境中定义每个step的训练方式。

## 初始化Megatron

假设我们有2台机器（node0和node1），每台机器上有8块GPU，GPU的编号为0~15。  
我们使用这16块GPU，做**DP/TP/PP混合并行**，如下图：

![](img/Pasted%20image%2020240727172721.png)

- **MP：模型并行组（Model Parallism）**。假设一个完整的模型需要布在8块GPU上，则如图所示，我们共布了2个model replica（2个MP）。MP组为：`[[g0, g1, g4, g5, g8, g9, g12, g13], [g2, g3, g6, g7, g10, g11, g14, g15]]`
- **TP：张量并行组（Tensor Parallism）**。对于一个模型的每一层，我们将其参数纵向切开，分别置于不同的GPU上，则图中一共有8个TP组。TP组为：`[[g0, g1], [g4, g5],[g8, g9], [g12, g13], [g2, g3], [g6, g7], [g10, g11], [g14, g15]]`
- **PP：流水线并行组（Pipeline Parallism）**。对于一个模型，我们将其每一层都放置于不同的GPU上，则图中一共有4个PP组。PP组为：`[[g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]]`
- **DP：数据并行组（Data Parallism）**。经过上述切割，对维护有相同模型部分的GPU，我们就可以做数据并行，则图中共有8个DP组。DP组为`[[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]]`

**（1）分组的原则是什么？**

- **MP设定原则：**MP其实由TP+PP共同决定。在开始训练前，需要我们根据实际模型，预估训练时显存消耗（特别注意峰值显存），来为模型安排GPU资源。
- **TP、DP和PP设定原则**：在这三种并行模式的原理篇中，我们分析过三者的通讯量。一般而言，TP>DP>PP。通讯量大的尽量放入一台机器内，因为机器内带宽高。所以在图例中，TP和DP不跨机，PP跨机。再提一点，在使用Megatron时，很多项目是不用PP，仅用TP+DP的，此时一般将TP放入一台机器内，令DP跨机（比如codegeex）

**（2）分组的目的是什么？**

- **分配进程**：

- 确认分组方案后，在每块GPU上启动一个进程（process），每个进程**独立执行**自己所维护的那部分模型的计算，实现并行训练。
- 进程0~15，为一个**进程大组（group**），其下的每一个DP/MP/PP组，为一个**进程子组（subgroup）**

- **组间通讯：**确认好DP/TP/PP组，并分配好进程后，我们就能进一步设置不同进程间的通讯方案。例如属于一个DP组的g0和g2需要进行梯度通讯，属于一个PP组的g4和g8需要进行层间输出结果的通讯。

```python
def _initialize_distributed():
    """Initialize torch.distributed and mpu.
                |    Node1  |   Node2    |
    ____________| p1 |  p2  |  p3  |  p4 |
    local_rank  | 0  |   1  |  0   |   1 |
    rank        | 0  |   1  |  2   |   3 |  

    node: 物理结点，1台机器或者1个容器。图中2个物理结点
    rank：进程在全局上的序号。图中4个进程
    local_rank：进程在node上的序号。
    torch.cuda.device_count()：当前进程所在的node上可使用的GPU的数量
    device：GPU在某个node上的编号

    该函数作用：
    1、设置分布式环境：初始化进程，分配GPU，并设置进程大组（group）
    2、制定DP/TP/PP分组策略，设置进程子组（subgroup）
    3、设置DeepSpeed ZeRO-R，对activation进行优化
    """
    args = get_args()

    device_count = torch.cuda.device_count() # 当前进程所在的node上可使用的GPU的数量
    if torch.distributed.is_initialized(): # 如果已创建好分布式环境
        if args.rank == 0: # 在0号进程上打印出“创建完毕”的日志
            print( 
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank() # 取得当前进程的全局序号
        args.world_size = torch.distributed.get_world_size() # 取得全局进程的个数

    else: # 如果未创建好分布式环境
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        
        # 1. 初始化进程，分配GPU，并设置进程大组（group）
        if device_count > 0: 
            device = args.rank % device_count # 1块进程1个GPU。device为GPU编号。例如图例中的进程9，其所在机器上有8块卡。因此进程9使用的gpu编号为8%9=1
            if args.local_rank is not None: 
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device 
           
            if args.force_device is not None: 
                print(
                    f"  > forcefully set the device to {args.force_device}, originally {device}"
                )
                device = args.force_device
            torch.cuda.set_device(device) # 为当前进程分配GPU
        
        # 设置进程大组
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost") # 获取rank=0进程的ip
        master_port = os.getenv("MASTER_PORT", "6000") # 获取rank=0进程的端口
        init_method += master_ip + ":" + master_port 
        print( 
            f"  > (rank={args.rank}) initializing process group: "
            f"world_size={args.world_size} "
            f"backend={args.distributed_backend} " 
            f"init_method={init_method}",
            flush=True,
        )
        timeout = datetime.timedelta(minutes=args.dist_timeout)
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            init_method=init_method,
            timeout=timeout
        )
        print(f"  > (rank={args.rank}) process group initialized")

    # 2、制定DP/TP/PP分组策略，设置进程子组（subgroup）
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel( # megatron/mpu/initialize.py
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
            )
    
    # 设置DeepSpeed ZeRO-R，对activation进行优化
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)
```

总体来说，这个代码实现了3个目的：

- 设置分布式环境：初始化进程，分配GPU，并设置**进程大组（group）**。也即例子中的0~15号进程同属一个分布式进程大组
- 制定DP/TP/PP分组策略，设置**进程子组（subgroup）**
- 设置DeepSpeed ZeRO-R，对activation进行优化



# 实践记录

代码框架：[GitHub - epfLLM/Megatron-LLM: distributed trainer for LLMs](https://github.com/epfLLM/Megatron-LLM)

使用指南：[Getting started — Megatron-LLM 0.1.0 documentation](https://epfllm.github.io/Megatron-LLM/guide/getting_started.html)

按照官方指南安装环境，安装apex


# 参考资料

[[源码解析] 模型并行分布式训练Megatron (1) --- 论文 & 基础-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1997455)

[以 GPT-175B 为例，聊聊大语言模型分布式训练的最佳实践](https://mp.weixin.qq.com/s/SN6_uOsj18_5JZIKRYFkzA)

[图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇）](https://mp.weixin.qq.com/s/ak1wg7jerJrD51xzVmmUiw)

**[猛猿：图解大模型系列之：Megatron源码解读1，分布式环境初始化](https://zhuanlan.zhihu.com/p/629121480)**

**[猛猿：图解大模型训练之：Megatron源码解读2，模型并行](https://zhuanlan.zhihu.com/p/634377071)**

**[猛猿：图解大模型训练系列之：Megatron源码解读3，分布式混合精度训练](https://zhuanlan.zhihu.com/p/662700424)**
