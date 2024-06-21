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
- Critic/value model：教练，预估总收益
- Reward model：评判员，打分，预估即时收益
- Reference model/ref_policy：参考模型，给语言模型增加一些约束，防止训练歪

模型初始化
- Reward model和Critic通常用同一个基础模型。Critic在训练过程中会更新。
- Actor和Reference model也是同一个，只不过Actor会一直学习进步，Reference是保持初心的。
数据
- 训练时，数据只需要有prompt字段，也就是输入即可。回复会用Actor实时生成。

学习率参考
- actor模型的学习率一般是SFT模型最后的学习率的1/10。
- critic模型的学习率是SFT模型最后的学习率的将近2倍。

![](img/Pasted%20image%2020240621141211.png)


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


## 更新前的准备工作

```python
# generate seq
inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
num_actions = action_mask.size(1)

# log probs
action_log_probs = self.actor(sequences, num_actions, attention_mask, use_cache=False)

# init log probs
base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

# values
value = self.critic(sequences, action_mask, attention_mask)

# rewards
r = self.reward_model(sequences, attention_mask)

reward, kl = compute_reward(
	r,
	self.kl_ctl.value,
	action_log_probs,
	base_action_log_probs,
	action_mask=action_mask,
)
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

#### reward模型

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

#### 计算每个token的奖励和KL惩罚

**Token Level KL-Penalty: RL模型和SFT模型的response的KL散度，作为reward函数中的一个惩罚项。**

$$r(s_t, a_t) = \textbf{I}(s_t =[\text{EOS}])r(x,y)-\beta \text{KL}(t) \ \ \ (1)$$

$$\text{KL}(t) = \log({\pi_{\theta_{\text{old}}}(a_t|s_t)^{\text{RL}}}/{\pi^{\text{SFT}}(a_t|s_t)}）\ \ \ (2)$$

where x is the prompt, y is the response, and $\textbf{I}(s_t = [\text{EOS}])$ is the identity function that represents whether t is the last token.

汇总的$r(s_t, a_t)$是 PPO 想要优化的奖励。

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

#### advantages

**Generalized Advantage Estimation (GAE)**: GAE [5], a $\text{TD}(\lambda)$ return estimation method, is used to estimate token-wise rewards in PPO. In practice, we typically set $\lambda = 1$, transforming the GAE method into a Monte Carlo estimation method.

![](img/Pasted%20image%2020240621175633.png)

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

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

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

### 开始更新，其中的Loss


- actor优化时候的loss：PolicyLoss。
- ptx loss（也是优化actor，传入pretrain_data的时候会用到）：GPTLMLoss，也就是交叉熵。Incorporating an additional supervised next-token prediction loss, alongside the KL divergence, into PPO can preserve the pre-existing abilities of the SFT model
- critic优化的loss：ValueLoss。
- 如果是Mixtral 8x7b会加一个aux_loss到上述每个loss中。
#### PolicyLoss

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
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss
```

#### **GPTLMLoss**

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
        return 0.5 * loss
```


## 未来方向



## 主要收获


## 参考资料

[ppo_trainer](https://huggingface.co/docs/trl/ppo_trainer)

[The 37 Implementation Details of Proximal Policy Optimization · The ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
[Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f)

[Reinforcement Learning From Human Feedback — My sample book](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)

[reward model learning papers - Shiyu\_Huang - 博客园](https://www.cnblogs.com/huangshiyu13/p/17203355.html)


