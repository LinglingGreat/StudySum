---
title: PPO
created: 2024-06-17
tags:
  - rlhf
  - alignment
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## 背景



## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

PPO的训练流程

![](img/Pasted%20image%2020240617194446.png)

需要4个模型
- Actor/policy：学习者
- Critic/value model：教练
- Reward model：评判员，打分
- Reference model/ref_policy：参考模型，不要走的太远

模型初始化
- Reward model和Critic通常用同一个基础模型。Critic在训练过程中会更新。
- Actor和Reference model也是同一个，只不过Actor会一直学习进步，Reference是保持初心的。
数据
- 训练时，数据只需要有prompt字段，也就是输入即可。回复会用Actor实时生成。

学习率参考
- actor模型的学习率一般是SFT模型最后的学习率的1/10。
- critic模型的学习率是SFT模型最后的学习率的将近2倍。


## 实验



## 实现

需要的核心参数（以OpenRLHF为例）

```python
    # 训练数据，有prompt字段即可
    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument("--pretrain_data", type=str, default=None)
    # actor和reference模型，也就是你要训练的模型
    parser.add_argument("--pretrain", type=str, default=None)
    # reward模型
    parser.add_argument("--reward_pretrain", type=str, default=None)
    # 更新迭代数，相当于过了多少遍数据
    parser.add_argument("--num_episodes", type=int, default=1)
    # 生成的总batch_size，生成总量达到这个batch_size会进行PPO更新
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    # 每次batch生成时候的size，也就是单张卡generate时候的输入batch大小
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    # 进行PPO更新的时候的epochs
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.02)
    ## Make EMA as an optional feature, moving average
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
```


```python
# len(prompts_dataloader) = len(prompts_dataset) // (args.micro_rollout_batch_size * num_process)
num_update_steps_per_episodes = (
        int(len(prompts_dataloader) * (args.micro_rollout_batch_size / args.micro_train_batch_size))
        * args.max_epochs
        // strategy.accumulated_gradient
    )

max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
```


Loss
- PolicyLoss
- ValueLoss
- GPTLMLoss：交叉熵. pretrain_data会用到
- aux_loss： Mixtral 8x7b
- kl_ctl

## 未来方向



## 主要收获


## 参考资料

[ppo_trainer](https://huggingface.co/docs/trl/ppo_trainer)

[The 37 Implementation Details of Proximal Policy Optimization · The ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
[Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f)


