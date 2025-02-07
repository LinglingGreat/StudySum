---
title: Loss函数总结
created: 2025-02-07
tags:
  - loss
---
## 基础

在研究 loss 函数前，建议把下面几个公式和图先焊死在脑子中。

![](img/Pasted%20image%2020250207115807.png)

![](img/Pasted%20image%2020250207115837.png)

![](img/Pasted%20image%2020250207115849.png)

## SFT 家族

### GPTLMLoss

```python3
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

没啥多说的，最常见的 gpt loss 函数，也就是 pretrain / sft 的 loss 函数，通过 self.IGNORE_INDEX 来实现 prompt 的 loss_mask 。

### KDLoss

```python3
# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss
```

第二种 sft 的 loss 函数：[知识蒸馏](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F&zhida_source=entity)的 loss 函数。需要在同源 tokenizer 的情况下，利用一个大模型的 logits 分布结果，来让小模型学习软标签。当然，[embedding](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=embedding&zhida_source=entity) 毕竟只是一个线性层，可以考虑再给模型外挂一个线性层，把 model_A 的 [tokenizer](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=2&q=tokenizer&zhida_source=entity) 映射到 model_B 的 tokenizer，进而实现利用 qwen 蒸馏 llama 的美好愿景，不知道有没有大佬做过类似的尝试。

言归正传，我们都知道知识蒸馏是用 KL 散度作为 loss 函数的，但代码里也没看见 KL [散度公式](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E6%95%A3%E5%BA%A6%E5%85%AC%E5%BC%8F&zhida_source=entity)啊，不妨一起简单推导下。

KL(P||Q)=∑iP(i)(log⁡P(i)−log⁡Q(i))

其中，P(i) 是教师模型的概率分布， Q(i) 是学生模型的概率分布。在实际的优化过程中，KL 散度中的第一项 ∑iP(i)log⁡P(i) 是关于教师模型的熵，对于学生模型的[参数优化](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E5%8F%82%E6%95%B0%E4%BC%98%E5%8C%96&zhida_source=entity)是一个常数，可丢弃。因此，我们通常只需要最小化第二项，即[交叉熵损失](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1&zhida_source=entity)：

CrossEntropy=−∑iP(i)log⁡Q(i)

知识蒸馏本身没啥痛点，只要能解决 seq_len * vocab_size 大小的 logits 通讯问题，这就是个简单纯粹有效的优化小模型的极佳方案。不过传统的 KL 往往是 [soft_label](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=soft_label&zhida_source=entity) 和 hard_label 的加权组合，这在 OpenRLHF 的代码中没有体现出来，大家有需要的话可以自行实践：

```python3
lm_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    label.view(-1),
    ignore_index=self.IGNORE_INDEX
)
total_loss = alpha * lm_loss + beta * distil_loss
```

## DPO 家族

### DPOLoss

```python3
class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
```

我们熟悉的 dpo 的 loss 函数，看上去没提供任何 trick，实践中如果 chosen_rewards 和 rejected_rewards 都下降，可以考虑给正例 / 负例再加一个系数。

除了原始的 loss 函数，OpenRLHF 为我们提供了额外两个选项：

**IPO**：论文中的 loss 表达式 E(yw,yl)∼D[(hπ(yw,yl)−τ−12)2]

我没有实践过这个算法就不多评价了，似乎重点是加了一个正则项。

**CDPO**：大概就是给 DPO 加了个 label_smoothing。

标签平滑是一种[正则化方法](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95&zhida_source=entity)，它通过将硬标签转换为软标签来防止模型过度自信。具体来说，对于二分类问题：原本的样本是正例就是正例，是负例就是负例，平滑后变成了： (1 - self.label_smoothing) 的概率是正例，self.label_smoothing 的概率是负例。

具体在 DPO 算法中的含义，一个 pair 对，以 (1 - self.label_smoothing) 的概率认为 good_sentence 比 bad_sentence 质量高，以 self.label_smoothing 的概率认为 bad_sentence 比 good_sentence 质量高。从而避免了模型对训练数据的过度拟合和过度自信。

理解这个平滑代码实现的关键点在于下面这两个公式，相信负例的 loss 可以动手笔划一下：

- 相信正例的 loss： −log⁡(σ(z))=−log⁡sigmoid(z)
- 相信负例的 loss： −log⁡(1−σ(z))=−log⁡sigmoid(−z)

### KTOLoss

```python3
# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL
```

kto 的 loss 函数，对标 dpo 的一个工作。这个算法我之前也没有实践过，但是“kto 不需要偏好 pair 对”这一优点吸引了我，所以也简单研究了一下它的原理和实现代码。

kto 的算法思想说是借鉴了“[前景理论](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E5%89%8D%E6%99%AF%E7%90%86%E8%AE%BA&zhida_source=entity)”，这个概念对我一个程序员来说太高深了，还是直接去讨论一下算法怎么实现的吧。

kto 的训练数据是 prompt + response + label，这个 label 就是 1 或者 -1，代表着 response 的质量是否被认可。label 是 1 的被称为正例，label 是 -1 的被称为负例。我们看到 loss 函数中要做一个判断 if policy_chosen_logps.shape[0] != 0 的操作，这是因为如果该条训练数据为负例，那么 policy_chosen_logps 这个变量就是一个空 [tensor](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=tensor&zhida_source=entity)，反之亦然。和 dpo 相比最大的区别是：dpo 的每一条 prompt 需要同时具有正例和负例，kto 的每一条 prompt 则只需要有正例或负例中的一个即可。

kto 正例和负例的 loss 函数分别如下所示：

- Lchosen=1−σ(β(log⁡πθ(a|s)πref(a|s)−KL))
- Lrejected=1−σ(β(KL−log⁡πθ(a|s)πref(a|s)))

1 - sigmoid 是一个单调递减函数，这说明：kto 的 loss 函数在正例中鼓励策略模型尽量大于参考点 KL，在负例中则鼓励模型尽量小于参考点 KL，也是一个比较明显的学习正例打压负例的损失函数。self.desirable_weight 和 self.undesirable_weight 则是正向和负向样本各自的权重损失，调参用的。

kto 代码的理解难点是，这个 KL 并不是一条训练样本的 KL，而是一批样本的平均 KL（代码中的 dist.all_reduce），并且为了训练稳定这个 KL 也是不进行反向传播的（代码中的 detach），只是拿来控制损失的饱和度，并且做了 clamp(min=0) 处理。至于这么设计的原因，反正原论文就这么写的，我没具体看公式是怎么推的，不敢瞎分析，感兴趣的可以自己推推公式。

### VanillaKTOLoss

```python3
# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards
```

kto 的变种，看代码实现的话，主要 diff 应该是去掉了参考点 KL ，并且正负样本要 1:1 均衡（OpenRLHF 代码库没写，但是注释里的 [https://github.com/ContextualAI/HALOs](https://link.zhihu.com/?target=https%3A//github.com/ContextualAI/HALOs) 这个代码库写了）

乍一看，这个均匀采样 kto 的 loss 函数和 dpo 已经很相似了，但其实还是有本质区别的。dpo 的重点是 margin，也就是正例和负例的 loss 是要做减法的，均匀采样 kto 用的是 torch.cat()，也就是说正例和负例的 loss 相互之间毫无影响，各自朝着各自的 label 去优化。

需要留意的细节是，chosen_KL 和 rejected_KL 也做了 clamp(min=0) 的操作。这里给出我对 RLHF 代码的一条学习心得：**不要放过任何一个 clamp / clip 操作背后的原因**。

## RLHF 家族

### PolicyLoss

```python3
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

rlhf 中，actor_model （也就是被优化的模型）的 loss 函数，大概是三个步骤：

- 计算新旧策略的概率比例 ratio=exp⁡(log⁡πθ(a|s)−log⁡πθold(a|s))=πθ(a|s)πθold(a|s)
- 利用[优势函数](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E4%BC%98%E5%8A%BF%E5%87%BD%E6%95%B0&zhida_source=entity)指导更新方向；
- 限制策略更新幅度 LCLIP(θ)=Et[min(ratiot⋅At,clip(ratiot,1−ϵ,1+ϵ)⋅At)]
- -torch.min(surr1, surr2)，选择未剪辑和剪辑后损失项的最小值。取负号是因为在最小化损失函数，但 PPO 的目标是最大化期望收益。

代码写的很清晰简洁，和 ppo 论文完全吻合，上面的两个公式也都是 ppo 论文的原始公式。对这里的代码实现有疑惑的，可以结合 ppo 论文一起读。

### ValueLoss

```python3
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

rlhf 中，critic_model 的 loss 函数。

- 如果不剪辑损失函数： LVF(θ)=12Et[(Vθ(st)−Rt)2]
- 如果要剪辑损失函数，便需要对价值函数的更新进行剪辑，防止价值估计发生过大的变化，和上面的 policy model 的剪辑是一个道理： Vclipped=Vθold(s)+clip(Vθ(s)−Vθold(s),−ϵ,ϵ)
- 剪辑的损失函数： LVF(θ)=12Et[max((Vθ(st)−Rt)2,(Vclipped−Rt)2)] 。也就是说，如果有剪辑，通过 torch.max(surr1, surr2) 选择让 loss 最大化的更新策略；
- loss 最终乘以 0.5，也算是平方误差损失函数中常见的缩放因子了。

这里我之前被一个地方绊住过，可以分享一下我曾经的疑惑点：clamp 的意义既然是防止模型进行较大的参数更新，那为什么 value function 的 loss 还要选 torch.max() 呢，不应该是 torch.min() 更合理吗？

我目前的观点：在策略函数中，模型更新幅度的大与小，和 loss 的大小并无直接关系。新的 values 距离 old_values 越近，代表着价值估计的目标更新幅度越小。显然，old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps) 对应的 values_clipped ，是一定比原始的 values 更接近 old_values 的。

surr1，surr2 分别代表使用 values_clipped 和 values 进行模型更新的 loss：

- surr1 < surr2：说明 clip 的过分了，导致 loss 变小可能会更新不动，那就放弃 clip，选择 values 来更新；
- surr1 > surr2：说明用更保守的更新策略 values_clipped，得到了更大的 loss。模型期望的更新幅度小，训练动力还大，没有比这更好的事情了。

### PairWiseLoss

```python3
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
```

主角登场，reward_model 的 loss 函数，以最简单的形式干最多的活！

这里有意思的点是 OpenRLHF 提供了一个 margin 的选项。还记得文章开头给大家画出来的 −log(sigmoid(x)) 求导后的曲线吗？x 越小，梯度的绝对值越大，就越能避免梯度消失。原本 positive_reward - negative_reward = 2 的时候，已经没梯度训不动了，现在 positive_reward - negative_reward = 2 + margin 的时候才会训不动。

这个 margin 和 dpo 的 reference_model 非常类似，都是常量。我曾经疑惑过这种常量是不是没啥大用，后来动手求了求导就明白了：这些被 logsigmoid() 包裹起来的常量，会影响梯度的大小，决定梯度在什么情况下趋近于零，进而也会影响模型训练的动力。

### LogExpLoss

```python3
class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
```

看上去是用 log(1+e−x) 取代了 −log(sigmoid(x)) ，这妥妥的就是一个等价变化啊。

### PRMLoss

```python3
class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
```

[当红炸子鸡](https://zhida.zhihu.com/search?content_id=250280707&content_type=Article&match_order=1&q=%E5%BD%93%E7%BA%A2%E7%82%B8%E5%AD%90%E9%B8%A1&zhida_source=entity)，学界认为的通往 o1 的钥匙，process_reward_model 的 loss 函数。出于对 o1 的尊重，我逐行解读一下这个 loss 。

首先看下 PRM 训练集合的样子：

```json
{
    "inputs": "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. ки Step 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. ки Step 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. ки Step 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. ки Step 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 ки",
    "labels": "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. + Step 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. + Step 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. + Step 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. + Step 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 -",
    "values": [ "+", "+", "+", "+", "-" ]
}
```

- 在 inputs 中，每个 step 后面会有一个 special_token：ки
- 在 labels 中，每个 step 后面会有一个 label_token：+ / - （代表着当前 step 的推理是否正确）

```python3
placeholder_mask = inputs == self.placeholder_token_id
logits = logits[placeholder_mask]
labels = labels[placeholder_mask]
```

logits 就是整个 inputs 过了一遍 llm 后得到的输出，形状为 seq_len * vocab_size （不考虑 batch_size），self.placeholder_token_id 就是 “ки” 对应的 id。使用这几行代码，上面的 case 中，logits 会变成 5 * vocab_size， label 会变成 5 * 1

```python3
logits = logits[..., self.reward_token_ids]
for i, token in enumerate(self.reward_token_ids):
    labels = torch.where(labels == token, i, labels)
```

紧接着，先理解常规的 hard label，self.reward_token_ids 就是["+"对应的 id, "-"对应的 id]，labels 就是["+"对应的 id, "+"对应的 id, "+"对应的 id, "+"对应的 id , "-"对应的 id]。这几行代码成功提取出了每个 step 下，两个 label 各自对应的 logits，以及每个 step 的 label 是什么。

```python3
if labels.dtype == torch.float:
    logits = logits[..., self.reward_token_ids]
    assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
    positive_labels = labels.to(logits.dtype)
    negative_labels = 1 - positive_labels
    negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
    labels = torch.stack([positive_labels, negative_labels], dim=-1)
```

再理解非常规的 soft_label，此时 labels 不再是 id，而是 float 类型，比如 labels = [0.8, 0.85, 0.9, 0.78, 0.1]，代表着每个 step 正确的概率（ assert len(self.reward_token_ids) == 2 是为了确保可以通过减法算出 step 错误的概率）。

理解了 soft_label 和 hard_label 分别是如何获得的之后，后面的 loss 计算和 acc 计算就没什么好多说的了。哦对，如果用 qwen 去跑这份代码，会遇到一个 tokenizer 的 bug，请教了下代码作者朱小霖大佬，大概是说不想为了 math-shepherd 加太多冗余逻辑，后续会给出一个优化版的代码（期待ing）。

## 参考资料

[OpenRLHF学习笔记-loss篇](https://zhuanlan.zhihu.com/p/6290579087)

