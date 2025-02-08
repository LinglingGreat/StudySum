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

### 策略（policy）

策略分成两种：确定性策略和随机性策略。我们用 θ 表示策略的参数。

![](img/Pasted%20image%2020250206161929.png)

### 奖励（Reward）

奖励由当前状态、已经执行的行动和下一步的状态共同决定。

![](img/Pasted%20image%2020250206162058.png)

### 运动轨迹（trajectory）和状态转移

![](img/Pasted%20image%2020250206162150.png)

### Policy-based强化学习优化目标

![](img/Pasted%20image%2020241018211631.png)

**抽象来说，强化学习的优化过程可以总结为：**

- **价值评估**：给定一个策略 π ，如何准确评估当前策略的价值 Vπ ？
- **策略迭代**：给定一个当前策略的价值评估 Vπ ，如何据此优化策略 π ？

整个优化过程由以上两点交替进行，最终收敛，得到我们想要的最优策略 π∗ 和能准确评估它的价值函数 Vπ∗ 。

**这是否意味着强化学习过程中一定存在** π **和** Vπ **两个实体呢？** 例如，这是否意味我们一定要训练两个神经网络，分别表示策略和价值评估？**答案是否定的：**

- 你可以只有一个价值实体 Vπ ，因为它的输入和状态与动作相关（这里我们不区分V和Q，留到后文细说）。这意味着只要我们知道状态空间S 和动作空间A ， Vπ 就可以作用到这两个空间上帮助我们衡量哪个状态/动作的价值最大，进而隐式地承担起制定策略的角色，**我们也管这种方法叫value-based。
- 你可以只有一个策略实体 π ，在对策略的价值评估中，我们可以让策略和环境交互多次，采样足够多的轨迹数据，用这些数据去对策略的价值做评估，然后再据此决定策略的迭代方向，我们也管这种方法叫**policy-based**。
- 你可以同时有价值实体 Vπ 和策略实体 π ，然后按照上面说的过程进行迭代，**我们也管这种方法叫actor-critic，其中actor表示策略，critic表示价值。**

![](img/Pasted%20image%2020250206162916.png)


### 策略的梯度上升

![](img/Pasted%20image%2020250206164046.png)

![](img/Pasted%20image%2020250206164433.png)

![](img/Pasted%20image%2020250206164721.png)

![](img/Pasted%20image%2020250206164731.png)

这里 R(τ) 表示一整条轨迹的累积奖励或者累积折扣奖励。

### 价值函数（Value Function）

当你端详策略的梯度公式时，你可能会有这样的疑问： R(τ) 是整条轨迹的奖励，但是 πθ(at|st) 却是针对单步的。我用整条轨迹的回报去评估单步的价值，然后决定要提升/降低对应 at 的概率，是不是不太合理呢？例如：

- 一条轨迹最终的回报很高，并不能代表这条轨迹中的每一个动作都是好的。
- 但我们又不能完全忽视轨迹的最终回报，因为我们的最终目标是让这个回合的结果是最优的。
- **综上，在衡量单步价值时，我们最好能在【单步回报】和【轨迹整体回报】间找到一种平衡方式**。

![](img/Pasted%20image%2020250206165124.png)

#### 衡量价值的不同方式

![](img/Pasted%20image%2020250206165305.png)

![](img/Pasted%20image%2020250206165431.png)

![](img/Pasted%20image%2020250206170425.png)

![](img/Pasted%20image%2020250206170539.png)

通过上面的例子，我们已经引出一些关于价值函数的基本概念了：

- $V_π(s_t)$ ：状态价值函数

- $Q_π(s_t,a_t)$ ：动作价值函数

- $A_π(s_t,a_t)=Q_π(s_t,a_t)−V_π(s_t)$ ：优势

#### 回报

在前面的例子中，我们说过，当我们从 某一帧，顶金币(st=某一帧，at=顶金币) 出发后，我们玩游戏一直到回合结束，然后我们执行 $rt+r_{t+1}+...r_{T−1}$ ，作为这个回合的累积奖励。

但其实，我们计算这个累积奖励的目的是衡量从 某一帧，顶金币(st=某一帧，at=顶金币) 这一【单步】出发后带来的未来收益。而对于这一个【单步】来说，一般离它越近的timestep受到它的影响越大，离它越远的timestep受到它的影响越小。在这个直觉的启发下，**我们采用【累积折扣奖励】来定义单步（也就是某个t时刻）的回报：**

$G_t=r_t+γr_{t+1}+...+γ^{T−t−1}r_{T−1}$

在接下来的讲解中，提到某一个回合中【单步】的奖励，我们说的都是【累积折扣奖励】

#### 状态价值函数（State-Value Function）

**状态价值函数的原子定义如下**：

$V_π(s_t)=E_π(G_t|s_t)$

我们先来解释相关的符号：

- 首先，状态价值函数一定是和策略相关的。相同的状态 st 下（例如“同一帧游戏画面”），不同的策略 π 产生的结果也不一样（例如不同的人玩这个游戏）。所以我们带上了下标 π 。
- 其次， st 不是随机变量，而是一个确定值。这是因为此时我们衡量的就是从某个确定的状态 st 出发带来的累积奖励期望。
- 但是， Gt 却是一个随机变量，这是因为因为我们的策略 π(.|st) 和环境转移 P(.|st,at) 都是随机的。所以尽管每次智能体都从 st 出发，但采样到的轨迹却不一样。所以这里我们谈的是 Gt 的期望。

![](img/Pasted%20image%2020250206171145.png)

上面这个展开细节帮助我们从理论上理解上面举的例子：从马里奥游戏的某一帧 st 出发，如何求这一帧的累积回报期望，也就是求这一帧下所有动作的累积回报期望。我们从第4行推导开始讲起：

![](img/Pasted%20image%2020250206171240.png)

#### 动作价值函数（Action-Value Function）

![](img/Pasted%20image%2020250206171520.png)

#### 动作价值函数和状态价值函数的互相转换

![](img/Pasted%20image%2020250206171617.png)

#### 优势函数和TD error

![](img/Pasted%20image%2020250206171944.png)

- **假设这里的** Vπ **可以准确衡量策略** π **的价值，那么TD_error就是优势函数的无偏估计**。这意味着在期望的意义下，使用TD_error近似优势函数不会引起系统性的偏差。
- **假设这里的** Vπ **不能准确衡量策略** π **的价值**，**那么TD_error对于优势函数则是有偏的**。这意味着由于 Vπ 的不准确，我们无法用 $r_t+γV_π(s_{t+1})$ 去近似那个真实的优势函数，因为我们将引入系统性偏差。（读到这里，你可能已经开始联想到在actor-critic算法下，用于估计 Vπ 的critic网络在没有收敛之前都是偏离真值的，这就意味着此时我们用TD-error去近似优势是有偏的，所以这时我们就要请GAE出场了，这是后话，我们放在后文细说）


## Actor-Critic

![](img/Pasted%20image%2020250206172646.png)

接下来我们来看actor loss和critic loss的具体表达式。

### Actor优化目标

![](img/Pasted%20image%2020250206172830.png)

![](img/Pasted%20image%2020250206172926.png)

### Critic优化目标

![](img/Pasted%20image%2020250206173014.png)

### Actor和Critic之间的关系

actor这个优化目标的改写我们已经很熟悉了，**但对于critic loss你可能还是满腹疑惑，例如**：看样子，critic loss是在让优势趋于0，但是如此一来，每个动作的好坏不就都差不多了？那你怎么能选出那个最好的动作呢？

为了解答这个问题，我们先回想前面提到的“价值评估->策略迭代”这样一个循环的过程：

- **价值评估**：给定一个策略 π ，如何准确评估当前策略的价值 Vπ ？
- **策略迭代**：给定一个当前策略的价值评估 Vπ ，如何据此优化策略 π

![](img/Pasted%20image%2020250206173328.png)

## PPO

### 朴素Actor-Critic的问题

![](img/Pasted%20image%2020250206174639.png)

**观察这个梯度表达式，我们会发现如下问题：**

**_问题1_**：每次执行这个梯度更新时，我们都需要对 πθ 进行若干次回合采样。我们知道智能体和环境交互的时间成本（fwd）比较高，**也就是整个训练过程会比较慢。同时由于采样过程具有随机性，我们可能偶发采样到了一些方差特别大的样本，如果我们直接信任这些样本去做更新，就可能使得更新方向发生错误。**

**_问题2_**：**我们在前面说过，实际训练的过程中，用critic网络拟合出来** Vπ **并不一定是能准确衡量** π **的那个价值函数，所以这里我们用TD error去估计优势其实是有偏的**。为了降低这种偏差，我们需要对 Aϕ(st,at) 进行改造，改造的方法之一就是GAE。

接下来我们就详细来看如何解决这两个问题。

### 重要性采样

在朴素的方法中，我们使用 πθ 和环境交互若干次，得到一批回合数据，然后我们用这个回合数据计算出来的奖励值去更新 πθ 。**我们管这个过程叫on-policy（产出数据的策略和用这批数据做更新的策略是同一个）**

**而现在，为了降低采样成本，提升训练效率，同时更加“谨慎”地更新模型，我们想做下面这件事**：

- 假设某次更新完毕后，我们得到策略 πold
- 我们用 πold 和环境交互，得到一批回合数据。
- **我们将把这一批回合数据重复使用k次**：即我们先把这批数据喂给 πold ，更新得到 πθ0 ；我们再把这批数据喂给 πθ0 ，更新得到 πθ1 ；以此类推，做k次更新后，我们得到 πθ 。**我们管这个过程叫off-policy（产出数据的策略和用这批数据做更新的策略不是同一个）**。
- 在这k次更新后，我们令 πold=πθ 。重复上面的过程，直到达到设定的停止条件为止。

我们从更理论的角度来看待这个off-policy的过程：

![](img/Pasted%20image%2020250206174952.png)

**虽然数学上是有办法改写了，但是实际操作中，我们可能遇到p(x)和q(x)分布差异较大的问题。**这里我直接引用李宏毅老师的课堂ppt来说明这一点：

![](https://pica.zhimg.com/v2-bcac958cfda20f0f1784313166fa7c38_1440w.jpg)

- 我们假设 Ex∼p(x)[f(x)] 的真值是负数。
- 由于p(x)和q(x)差异较大。在某次采样中，我们从q(x)里进行采样，大概率会采集到图中绿色曲线的高处，此时f(x)是正的。也就是说，在单次采样中，我们大概率会得到一个正的f(x)。
- 所以，只有经过尽可能多次的采样，让某次能命中q(x)这个绿色曲线的低处。这时p(x)/q(x)较大，也就赋予这个好不容易采样到的负的f(x)非常大的权重，这才足以抵消之前正f(x)的影响。
- **综上所述，当p(x)和q(x)差异较大时，我们需要通过足够多的采样来抵消这种差异对期望的最终影响**。我们先记住这一点，在后面我们再来说这一点对我们策略的影响。

知道了重要性采样的过程，现在我们又可以根据它重写我们的优化目标了。

![](img/Pasted%20image%2020250206175212.png)

### GAE：平衡优势函数的方差和偏差

![](img/Pasted%20image%2020250206175610.png)

### （1）方差与偏差

![](https://pic3.zhimg.com/v2-36e602603c154fad31b81ad6091cd820_1440w.jpg)

  

- **低方差，低偏差**：`E(射击点) = 靶心`，且射击点密集分布在靶心周围。此时我们随机选一个射击点就能很好代表靶心
- **高方差，低偏差**：`E(射击点) = 靶心`，但射击点们离靶心的平均距离较远。此时随机一个射击点不能很好代表靶心，我们必须使用足够多的射击点才能估计靶心的坐标
- **高/低方差，高偏差**：`E(射击点)!=靶心`，无论你做多少次射击，你都估计不准靶心的位置。

![](img/Pasted%20image%2020250206175916.png)

![](img/Pasted%20image%2020250206180002.png)

### （2）GAE

![](img/Pasted%20image%2020250206180037.png)

### PPO前身：TRPO

![](img/Pasted%20image%2020250206183515.png)

### PPO做法1：PPO-Clip

![](img/Pasted%20image%2020250206183531.png)

![](img/Pasted%20image%2020250206183540.png)

### PPO做法2：PPO-Penalty

![](img/Pasted%20image%2020250206183555.png)

### PPO中的critic loss

在PPO的原始论文中，并没有对critic和actor拆分成两个网络以后的critic loss形式做详细介绍，所以这部分的解读我直接以deepspeed-chat的rlhf实现为例，讲一下critic loss的实现。  
  
  
我们知道，PPO的更新步骤是：

```python
# 对于每一个batch的数据
for i in steps: 
    # 先收集经验值
    exps = generate_experience(prompts, actor, critic, reward, ref)
    # 一个batch的经验值将被用于计算ppo_epochs次loss，更新ppo_epochs次模型
    # 这也意味着，当你计算一次新loss时，你用的是更新后的模型
    for j in ppo_epochs:
        actor_loss = cal_actor_loss(exps, actor)
        critic_loss = cal_critic_loss(exps, critic)
        
        actor.backward(actor_loss)
        actor.step()
        
        critc.backward(critic_loss)
        critic.step()
```

在以上的讲解中，我们已经知道actor是在PPO epoch中使用同一批数据做迭代更新的。但是同时，critic也是在迭代更新的。为什么critic要做迭代更新？**因为** Vπ **是和** π **挂钩的，如果你的策略参数已经更新了，你必须有一个新的critic来衡量新策略的价值。**

![](img/Pasted%20image%2020250206183651.png)

![](img/Pasted%20image%2020250206183704.png)



### PPO的训练流程

![](img/Pasted%20image%2020240617194446.png)

需要4个模型
- Actor/policy：学习者
- Critic/value model：教练，预估总收益Vt
- Reward model：评判员，打分，预估即时收益Rt
- Reference model/ref_policy：参考模型，给语言模型增加一些约束，防止训练歪

模型初始化
- Critic模型通常用Reward model初始化。Critic在训练过程中会更新，Reward model不更新。
- Actor和Reference model也是同一个，只不过Actor会一直学习进步，Reference是保持初心的。
数据
- 训练时，数据只需要有prompt字段，也就是输入即可。回复会用Actor实时生成。

学习率参考
- actor模型的学习率一般是SFT模型最后的学习率的1/10。
- critic模型的学习率是SFT模型最后的学习率的将近2倍。

![](img/Pasted%20image%2020240621141211.png)

总收益包含即时收益和未来收益。
- 即时收益，指语言模型当下产生token At 带来的收益
- 实际期望总收益（即时+未来），指对语言模型“当下产生token At ，一直到整个response生产结束”后的期望收益预估。因为当下语言模型还没产出 At 后的token，所以我们只是对它之后一系列动作的收益做了估计，因而称为“期望总收益”。

### actor模型
目的是学习人类偏好。

策略是，先喂给Actor一条prompt （这里假设batch_size = 1，所以是1条prompt），让它生成对应的response。然后，我们再将“prompt + response"送入我们的“奖励-loss”计算体系中去算得最后的loss，用于更新actor。

### Reference model

主要作用是防止Actor”训歪”。**我们希望训练出来的Actor模型既能达到符合人类喜好的目的，又尽量让它和SFT模型不要差异太大**。简言之，**我们希望两个模型的输出分布尽量相似**。

![](img/Pasted%20image%2020241018211116.png)

### Critic model
**Critic Model用于预测期望总收益** Vt **，和Actor模型一样，它需要做参数更新**。实践中，Critic Model的设计和初始化方式也有很多种，例如和Actor共享部分参数、从RW阶段的Reward Model初始化而来等等。

![](img/Pasted%20image%2020241018211253.png)

### reward模型

Reward Model用于计算生成token At 的即时收益，它就是RW阶段所训练的奖励模型，在RLHF过程中，它的参数是冻结的。

![](img/Pasted%20image%2020241018211423.png)

**Reward Normalization and Clipping**: The Bradley-Terry model tends to overfit easy samples while neglecting hard ones, resulting in a larger reward gap between easy response pairs compared to hard pairs. This variation causes gradient imbalance. Reward normalization and clipping mitigate this effect by equalizing the reward distribution across all samples. In practice, we employ Z-score normalization method, $r = (r - \mu) / \delta$, where $\mu$ is the mean and $\delta$ is the standard deviation of the reward training dataset.

```python
def _get_reward_model(base_pretrained_model, base_llm_model, head_prefix="value_head"):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.head_prefix = head_prefix
            setattr(self, head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                try:
                    self.mean[0] = config.mean
                    self.std[0] = config.std
                except:
                    self.mean[0] = config.mean[0]
                    self.std[0] = math.sqrt(config.var[0])

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.head_prefix)(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward

    return RewardModel
```


## 实验



## 实现

### 参数

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
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )

max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
```


### 训练代码

- 第一步，我们准备一个batch的prompts
- 第二步，我们将这个batch的prompts喂给Actor模型，让它生成对应的responses
- 第三步，我们把prompt+responses喂给我们的Critic/Reward/Reference模型，让它生成用于计算actor/critic loss的数据，按照强化学习的术语，我们称这些数据为经验（experiences）。
- 第四步，我们根据这些经验，实际计算出actor/critic loss，然后更新Actor和Critic模型。为了不浪费，**1个batch的经验，会用来计算ppo-epochs次loss，更新ppo-epochs次Actor和Critic模型**（这个会影响到actor模型的loss设计，那部分会说）。


```python
def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        args,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        global_step = 1

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                experience = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)
                # print prompt/answer in each update step
                if global_step % update_timesteps == 0:
                    output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                    self.strategy.print(repr(output[0]))
                self.replay_buffer.append(experience)

                if global_step % update_timesteps == 0:
                    torch.cuda.empty_cache()
                    self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.ppo_train()
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    # logs/checkpoints
                    self.save_logs_and_checkpoints(args, global_step // update_timesteps, pbar, status)

                pbar.update()
                global_step = global_step + 1
```

- actor优化时候的loss：PolicyLoss。
- ptx loss（也是优化actor，传入pretrain_data的时候会用到）：GPTLMLoss，也就是交叉熵。
- critic优化的loss：ValueLoss。
- 如果是Mixtral 8x7b会加一个aux_loss到上述每个loss中。

```python
def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean
```


### 计算每个token的奖励和KL惩罚-token粒度的即时奖励


![](img/Pasted%20image%2020240622154108.png)


**Token Level KL-Penalty: RL模型和SFT模型的response的KL散度，作为reward函数中的一个惩罚项。**

$$r(s_t, a_t) = \textbf{I}(s_t =[\text{EOS}])r(x,y)-\beta \text{KL}(t) \ \ \ (1)$$

$$\text{KL}(t) = \log({\pi_{\theta_{\text{old}}}(a_t|s_t)^{\text{RL}}}/{\pi^{\text{SFT}}(a_t|s_t)}）\ \ \ (2)$$

where x is the prompt, y is the response, and $\textbf{I}(s_t = [\text{EOS}])$ is the identity function that represents whether t is the last token.

注意这里是KL散度的蒙特卡洛近似，**采样动作的权重已经隐含在数据分布中**。无需显式乘以 πold(a∣s)。

汇总的$r(s_t, a_t)$是 PPO 想要优化的奖励。

给定一个 transformer 和任何一个 string，我都可以将整个 string 输入给 reward model 做一次 forward pass，得到每个位置的 token 的 logit。我们取出最后一个 token 的 logit，经过 logit processor 处理，再过一次 softmax 并取 log，得到此处的 log prob。此外，我们也可以对最后一个 token 的 logit 进行其他操作，譬如 pooling 和 projection 等等，拿到 embedding、reward 或者 value。由此可见，对于 string 里的每个 token，我们都可以得到前述所有计算值，但是在 RLHF 中，我们会用到 response 中每个 token 的 log prob 和 value，但是 reward 只会用最后一个 token 的 reward。

reward的计算公式：

![](img/Pasted%20image%2020241018211907.png)

为什么只有最后一个时刻的 Rt 被纳入了考量呢？这是因为在Reward模型训练阶段，就是用这个位置的 Rt 来表示对完整的prompt + response的奖励预测（但不妨碍你理解成是执行完 AT 的即时奖励），然后用这个指标来做模型eval的（但是Reward训练阶段算loss时，还是考虑了response部分所有token输出的reward值）。所以到了RLHF的场景下，其余时刻的即时奖励，我们就用“Actor是否遵循了Ref的约束”来进行评价。  
  
需要注意的是， Rt 的设计并不只有这一种。deepspeed在自己的代码注释中也有提过，可以尝试把最后一个时刻的 RT 替换成所有token的即时奖励的平均值。如果站在这个角度理解的话，我们同样也可以尝试在每一个位置的奖励衡量上引入 Rt 。

伪代码解析：

```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        """
        reward_function：计算最终的reward分数
        复习一下几个相关参数的默认值：
        self.kl_ctl = 0.1
        self.clip_reward_value = 5

        对于batch中的某个prompt来说，它最终的reward分数为：
        (1) 先计算actor和ref_model的logit相似度： -self.kl_ctl * (log_probs - ref_log_probs)
            其实写成self.kl_ctl * (ref_log_probs - log_probs)更好理解些
            这个值越大，说明ref_model对actor生成的结果的认可度越高（即表明rlhf没有训歪），
            没有训歪的情况下我们也应该给模型一些奖励，这个奖励就是self.kl_ctl * (ref_log_probs - log_probs)

        （2）由于我们只取最后一个token对应位置的分数作为reward_score，因此我们只需要：
            self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score

         (3) 同时我们对reward_score也做了大小限制，最大不超过self.clip_reward_value（超过统一给成self.clip_reward_value），
             最小不低于-self.clip_reward_value（低于统一给成-self.clip_reward_value）

         (4) 最后返回的rewards大小为：（batch_size, 各条数据的长度），对batch中的每条数据来说：
             - response的最后一位：self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
             - response的其余位置：self.kl_ctl * (ref_log_probs - log_probs)

        """

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # ---------------------------------------------------------------------------------------------------
        # response开始的位置
        # （因为我们对prompt做过padding处理，因此batch中每个prompt长度一致，也就意味着每个response开始的位置一致）
        # （所以这里start是不加s的，只是一个int）
        # ---------------------------------------------------------------------------------------------------
        start = prompts.shape[1] - 1
        # ---------------------------------------------------------------------------------------------------
        # response结束的位置
        # （因为一个batch中，每个response的长度不一样，所以response的结束位置也不一样）
        # （所以这里end是加s的，ends的尺寸是(batch_size,)
        # ---------------------------------------------------------------------------------------------------
        ends = start + action_mask[:, start:].sum(1) + 1
        # ---------------------------------------------------------------------------------------------------
        # 对rewards_score做限制
        # ---------------------------------------------------------------------------------------------------
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j] # 

        return rewards
```

注意输入输出的维度，`prompts` 是一个 `[batch size, padded prompt length]` 的 matrix，`ref_log_probs` 和 `log_probs` 是 `[batch size, padded prompt with response length]` 大小的矩阵，然后只有从 `prompt` 结束到 `response` 结束这一块儿的 `reward` 才会实际有作用，`prompt` 的 `reward` 是不计算的。

`prompt` 有统一的 `padding`，所以 `response` 的 `start` 位置是唯一的，而 `ends` 则通过 `action_mask` 中的 1 元素的截止为止计算得到。最后，在这个 `batch` 中，每个 `prompt` 的 `reward` 的结尾那个 `token` 加上 `reward_score` 经过 clip 得到的 `reward`。


代码实现：

```python
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -kl_coef * kl

    r = r.clamp(min=-10, max=10)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
    return reward, kl
```


### 优势计算代码

输入
- `values`：形状为 `(batch_size, response_size)` 的张量，表示每个时间步的价值估计。
- `rewards`：形状为 `(batch_size, response_size)` 的张量，表示每个时间步的奖励。
- `action_mask`：形状为 `(batch_size, response_size)` 的张量，用于掩盖无效的动作或时间步。
- `gamma`：折扣因子，通常用于折扣未来的奖励。
- `lambd`：GAE 参数，控制优势函数的平滑度。
输出
- `advantages`：形状为 `(batch_size, response_size)` 的张量，表示每个时间步的优势函数。
- `returns`：形状为 `(batch_size, response_size)` 的张量，表示每个时间步的回报。


```python
def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...


 .      没有引入GAE前的t时刻的优势值：
        detal_t = r_t + gamma * V_t+1 - V_t
        其中：
            - r_t表示t时刻的即时收益
            - V_t+1表示未来时刻的预期收益
            - r_t + gamma * V_t+1可理解成t时刻的实际预期收益
            - V_t可理解成t时刻的预估预期收益（是模型，例如critic model自己估算出来的）
        
        引入GAE后的t时刻的优势值：
        A_t = delta_t + gamma * lambda * A_t+1
        粗暴理解为在t时刻时，不仅考虑当下优势，还考虑了未来的优势
        为了知道A_t, 我们得知道A_t+1，所以在本算法中采取了从后往前做动态规划求解的方法，也即：
        假设T是最后一个时刻，则有A_T+1 = 0, 所以有: A_T = delta_T
        知道了A_T, 就可以依次往前倒推，把A_t-1, A_t-2之类都算出来了
        
        引入GAE后t时刻的实际预期收益
        returns_t = A_t + V_t
                  = delta_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * V_t+1 - V_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * (V_t+1 + lambda * A_t+1)
        
        注意，这里不管是advantages还是returns，都只算response的部分
        
        Input:
        - values: Tensor of shape (batch_size, response_size)，表示每个时间步的价值估计。
        - rewards: Tensor of shape (batch_size, response_size)，表示每个时间步的奖励。

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
```


**Advantage Normalization**: During the training of the value network using mean square loss, the algorithm is sensitive to some large values. Normalizing the advantage can mitigate the impact of these large values. In practice, we also employ Z-score normalization method, $r = (r - \mu) / \delta$, where $\mu$ is the mean and $\delta$ is the standard deviation of the samples within a batch.

```python
self.replay_buffer.normalize("advantages", self.strategy)

def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()
        action_masks_vector = torch.cat(action_masks).flatten()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), action_masks_vector.sum()], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
```


### PolicyLoss-actor模型

![](img/Pasted%20image%2020250207115148.png)

```python
class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        # `surr2` 通过裁剪比率来限制策略更新的幅度。
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        # 通过取 `surr1` 和 `surr2` 的最小值来确保策略更新不会偏离太远。
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss
```


### GPTLMLoss-actor模型

基于交叉熵损失的，用于训练语言模型，使其能够更好地预测下一个词。

Incorporating an additional supervised next-token prediction loss, alongside the KL divergence, into PPO can preserve the pre-existing abilities of the SFT model

```python
class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

### ValueLoss-critic模型

![](img/Pasted%20image%2020241018214906.png)

![](img/Pasted%20image%2020241018215026.png)

通过裁剪机制来控制价值函数更新的幅度。通过计算两个损失项并取其最大值来实现价值函数更新的裁剪，防止价值函数更新幅度过大。

**ValueLoss**：PPO clips the value function like the PPO’s clipped surrogate objective. Given $V_{targ} = returns = advantages + values$, PPO fits the value network by minimizing the following loss:

$$ Loss_v = \max[(V_{\theta_t} - V_{targ})^2, (\text{clip}(V_{\theta_t}, V_{\theta_{t-1}} - \epsilon, V_{\theta_{t-1}} + \epsilon) - V_{targ})^2] \ \ \ (3) $$
![](img/Pasted%20image%2020250207115320.png)


代码实现

```python
class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss # 这是为了与标准的均方误差（MSE）损失函数保持一致，通常 MSE 损失函数在优化过程中会乘以 0.5。
```

最终我们就取实际收益和预估收益的MSE做为loss就好，这里注意，计算实际收益时 Advt,Vt 都是老Critic（真正吃了batch的那个）产出的结果，而预估收益是随着ppo_epochs而变动的。
## 中间结果计算汇总

```python

这个函数的主要作用是：
1. 生成并更新注意力掩码，使模型只关注有效的 token。
2. 确保每个序列的结束位置标记为 `eos_token_id`。尽管 `model.generate` 通常会在生成序列的末尾自动添加 `eos_token_id`，但在某些情况下（如生成的序列过长或模型行为异常），可能不会正确添加。因此，需要手动确保每个序列都以 `eos_token_id` 结尾。
3. 生成动作掩码，用于标记状态序列中的有效 token。

def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
		# 生成一个初始的注意力掩码，其中每个位置上的值为 `1` 表示该位置的 token 既不是 `eos_token_id` 也不是 `pad_token_id`，值为 `0` 表示该位置的 token 是 `eos_token_id` 或 `pad_token_id`。
		# attention_mask的维度是(batch_size, seq_length)
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        
        # `eos_indices`：每个序列中最后一个非 `pad` 和非 `eos` token 的索引加一的位置。
        # `first_token_indices`：每个序列中第一个非 `pad` 和非 `eos` token 的索引。
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        # `mask`：一个掩码矩阵，用于标记每个序列中从第一个有效 token 到 `eos_indices` 位置的范围。
        # 使用 `mask` 更新 `attention_mask`，确保这些范围内的值都为 `1`。
        # 使用 `scatter_` 方法将 `eos_token_id` 写入 `eos_indices` 位置。
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1)
        mask = mask.to(sequences.device)
        mask = (mask <= eos_indices) & (mask >= first_token_indices)

        attention_mask.masked_fill_(mask, 1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # `state_seq`：从输入序列中截取的状态序列，范围是从 `input_len - 1` 到倒数第二个位置。
        # `action_mask`：一个掩码，用于标记 `state_seq` 中的有效 token（既不是 `eos_token_id` 也不是 `pad_token_id`）。
        state_seq = sequences[:, input_len - 1 : -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask


# generate seq
inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
# sequences是actor_model.generate()得到的output_ids，再经过上面的process_sequences函数得到sequences, attention_mask, action_mask
sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
num_actions = action_mask.size(1)

# log probs
# 计算输出 logits 对应目标序列的对数概率 `log_probs`，只取response部分的
action_log_probs = self.actor(sequences, num_actions, attention_mask, use_cache=False)

# init log probs，同上
base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

# values，只取response部分的
value = self.critic(sequences, action_mask, attention_mask)

# rewards，只提取每个序列中最后一个有效位置（即 `attention_mask` 中最后一个 1 的位置）的reward
r = self.reward_model(sequences, attention_mask)

# 计算汇总每个token的奖励和KL惩罚
reward, kl = compute_reward(
	r,
	self.kl_ctl.value,
	action_log_probs,
	base_action_log_probs,
	action_mask=action_mask,
)
# 计算advantage和returns
advantage, returns = self.get_advantages_and_returns(
	value,
	reward,
	action_mask,
	generate_kwargs["gamma"],
	generate_kwargs["lambd"],
)

info = {
	"kl": masked_mean(kl, action_mask, dim=-1),
	"reward": r,
	"return": reward.sum(dim=-1),
	"response_length": action_mask.float().sum(dim=-1),
	"total_length": attention_mask.float().sum(dim=-1),
}

return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )
```


### 采用不同的tokenizer

reward模型和其他模型可以用不同的tokenizer，但是critic模型和actor模型必须用同一个tokenizer，因为必须保证他们的token ID是一一对应的，critic模型需要计算token级别的奖励（优势函数那块）。



## old

### PolicyLoss-actor模型

主要用到了优势advantages，新旧策略的对数概率。

![](img/Pasted%20image%2020241018211716.png)

![](img/Pasted%20image%2020241018205717.png)

![](img/Pasted%20image%2020241018210332.png)



### 重新设计优势(可以不看)
![](img/Pasted%20image%2020241018212148.png)

![](img/Pasted%20image%2020241018212240.png)

一些额外的信息：

优势函数可以用以下方式定义：`Advantage(s, a) = Q(s, a) - V(s)`

其中，`Advantage(s, a)`表示在状态 `s` 下采取动作 `a` 的优势函数值，`Q(s, a)` 表示状态动作对 `(s, a)` 的动作值函数（也称为动作优势函数），`V(s)` 表示状态值函数。

优势函数的作用在于帮助评估当前动作的相对价值，以便在策略更新过程中确定应采取的动作。通过比较不同动作的优势函数值，可以决定哪些动作是更好的选择。正的优势函数值表示执行的动作比平均水平更好，而负的优势函数值表示执行的动作比平均水平更差。

状态动作对的长期价值涉及更长时间尺度上的评估，它考虑了智能体在当前状态下选择不同动作所导致的未来回报的累积。长期价值可以表示为状态值函数（State Value Function）或动作值函数（Action Value Function）。

状态值函数（V-function）表示在给定状态下，智能体从该状态开始执行一系列动作，然后按照某个策略进行决策，从而获得的预期累积回报。**状态值函数估计了智能体处于某个状态时所能获得的长期价值，反映了状态的优劣程度。**

动作值函数（Q-function）则表示在给定状态下，智能体选择某个动作后，按照某个策略进行决策，从该状态转移到下一个状态并获得预期累积回报的价值。**动作值函数估计了在给定状态下采取不同动作的长期价值，可以帮助智能体选择在每个状态下最优的动作。**

长期价值考虑了智能体在未来的决策过程中所能获得的累积回报，相比之下，即时奖励只提供了当前动作的即时反馈。长期价值对智能体的决策具有更全面的影响，可以帮助智能体更好地评估当前状态和动作的长期效果，并指导智能体在长期时间尺度上作出更优的决策。

**Generalized Advantage Estimation (GAE)**: GAE [5], a $\text{TD}(\lambda)$ return estimation method, is used to estimate token-wise rewards in PPO. In practice, we typically set $\lambda = 1$, transforming the GAE method into a Monte Carlo estimation method.

- **优势（Advantages）**：表示在特定状态下采取某个动作相对于平均水平的好坏程度。通过计算优势函数，可以更好地指导策略的更新，使得策略更倾向于选择优势较大的动作。
- **回报（Returns）**：表示从某个时间步开始到未来所有时间步的折扣累积奖励。回报用于衡量整个序列的表现，通过回报可以评估策略的整体效果。

![](img/Pasted%20image%2020240621175633.png)

![](img/Pasted%20image%2020241018202821.png)


### 引入新约束(可以不看)

前面提到过，如果我们想让一个batch的经验值被重复使用ppo_epochs次，等价于我们想要Actor在这个过程中，模拟和环境交互ppo_epochs次。举个例子：

- 如果1个batch的经验值只使用1次，那么在本次更新完后，Actor就吃新的batch，正常和环境交互，产出新的经验值
- 但如果1个batch的经验值被使用ppo_epochs次，在这ppo_epochs中，Actor是不吃任何新数据，不做任何交互的，所以我们只能让Actor“模拟”一下和环境交互的过程，吐出一些新数据出来。

![](img/Pasted%20image%2020241018214652.png)

![](img/Pasted%20image%2020241018214707.png)


### 总结

![](img/Pasted%20image%2020241018214823.png)


## 主要收获


## 参考资料

[ppo_trainer](https://huggingface.co/docs/trl/ppo_trainer)

[The 37 Implementation Details of Proximal Policy Optimization · The ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
[Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f)

[Reinforcement Learning From Human Feedback — My sample book](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)

[reward model learning papers - Shiyu\_Huang - 博客园](https://www.cnblogs.com/huangshiyu13/p/17203355.html)

[为什么RLHF中，PPO需要Critic模型而不是直接使用RewardModel - 风生水起 - 博客园](https://www.cnblogs.com/end/p/17481052.html)

[解析 RLHF 微调三阶段](https://mp.weixin.qq.com/s/t41vKSLJ0p-oKMosFEqagQ)

[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)

[如何让 RLHF 训练更稳定？](https://zhuanlan.zhihu.com/p/16734946629)

[授人以渔：巧用 o1 学 ppo](https://zhuanlan.zhihu.com/p/12621744754)

[图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)

[人人都能看懂的RL-PPO理论知识](https://zhuanlan.zhihu.com/p/7461863937)

[浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](https://zhuanlan.zhihu.com/p/16370000391)

[图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)

[The 37 Implementation Details of Proximal Policy Optimization · The ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (PPO训练的trick)

