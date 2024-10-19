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

### 强化学习

![](img/Pasted%20image%2020241018211631.png)

### 价值函数

![](img/Pasted%20image%2020241018211716.png)



## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

PPO的训练流程

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
- 实际期望总收益（即时+未来），指对语言模型“当下产生token At ，一直到整个response生产结束”后的期收益预估。因为当下语言模型还没产出 At 后的token，所以我们只是对它之后一系列动作的收益做了估计，因而称为“期望总收益”。

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
        int(len(prompts_dataloader) * (args.micro_rollout_batch_size / args.micro_train_batch_size))
        * args.max_epochs
        // strategy.accumulated_gradient
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



### PolicyLoss

主要用到了advantages，新旧策略的对数概率。

![](img/Pasted%20image%2020241018205717.png)

![](img/Pasted%20image%2020241018210332.png)



那么怎么得到这个优势advantages呢？

先来看看如何计算公式中的即时奖励Rt
#### 计算每个token的奖励和KL惩罚-token粒度的即时奖励


![](img/Pasted%20image%2020240622154108.png)


**Token Level KL-Penalty: RL模型和SFT模型的response的KL散度，作为reward函数中的一个惩罚项。**

$$r(s_t, a_t) = \textbf{I}(s_t =[\text{EOS}])r(x,y)-\beta \text{KL}(t) \ \ \ (1)$$

$$\text{KL}(t) = \log({\pi_{\theta_{\text{old}}}(a_t|s_t)^{\text{RL}}}/{\pi^{\text{SFT}}(a_t|s_t)}）\ \ \ (2)$$

where x is the prompt, y is the response, and $\textbf{I}(s_t = [\text{EOS}])$ is the identity function that represents whether t is the last token.

汇总的$r(s_t, a_t)$是 PPO 想要优化的奖励。

![](img/Pasted%20image%2020241018211907.png)

为什么只有最后一个时刻的 Rt 被纳入了考量呢？这是因为在Reward模型训练阶段，就是用这个位置的 Rt 来表示对完整的prompt + response的奖励预测（但不妨碍你理解成是执行完 AT 的即时奖励），然后用这个指标来做模型eval的（但是Reward训练阶段算loss时，还是考虑了response部分所有token输出的reward值）。所以到了RLHF的场景下，其余时刻的即时奖励，我们就用“Actor是否遵循了Ref的约束”来进行评价。  
  
需要注意的是， Rt 的设计并不只有这一种。deepspeed在自己的代码注释中也有提过，可以尝试把最后一个时刻的 RT 替换成所有token的即时奖励的平均值。如果站在这个角度理解的话，我们同样也可以尝试在每一个位置的奖励衡量上引入 Rt 。

代码实现

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

### 重新设计优势
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

### 引入新约束

前面提到过，如果我们想让一个batch的经验值被重复使用ppo_epochs次，等价于我们想要Actor在这个过程中，模拟和环境交互ppo_epochs次。举个例子：

- 如果1个batch的经验值只使用1次，那么在本次更新完后，Actor就吃新的batch，正常和环境交互，产出新的经验值
- 但如果1个batch的经验值被使用ppo_epochs次，在这ppo_epochs中，Actor是不吃任何新数据，不做任何交互的，所以我们只能让Actor“模拟”一下和环境交互的过程，吐出一些新数据出来。

![](img/Pasted%20image%2020241018214652.png)

![](img/Pasted%20image%2020241018214707.png)

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

### 总结

![](img/Pasted%20image%2020241018214823.png)

#### GPTLMLoss

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

#### ValueLoss

![](img/Pasted%20image%2020241018214906.png)

![](img/Pasted%20image%2020241018215026.png)

通过裁剪机制来控制价值函数更新的幅度。通过计算两个损失项并取其最大值来实现价值函数更新的裁剪，防止价值函数更新幅度过大。

**ValueLoss**：PPO clips the value function like the PPO’s clipped surrogate objective. Given $V_{targ} = returns = advantages + values$, PPO fits the value network by minimizing the following loss:

$$ Loss_v = \max[(V_{\theta_t} - V_{targ})^2, (\text{clip}(V_{\theta_t}, V_{\theta_{t-1}} - \epsilon, V_{\theta_{t-1}} + \epsilon) - V_{targ})^2] \ \ \ (3) $$
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



## 未来方向



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


