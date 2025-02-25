---
title: RewardModel
created: 2024-07-26
tags:
  - reward模型
---
Reward模型一般也是使用的LLM模型，由于最终需要生成scores，因此通常在模型最后会有一个线性层Linear(dims, 1)，将output_hidden转成为一个score。Reward模型可以是任意SFT模型+Linear层，也可以是其他预训练好的Reward模型（也可以在基础上继续微调）。

Loss函数有2种：

```python
# PairWiseLoss
loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
# LogExpLoss
loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
```

为什么用logsigmoid？由于在数据非常大或者非常小时，sigmod函数会存在数值溢出问题，且在结果0时容易存在梯度消失问题，可以将sigmod换成logsigmod函数。

将chosen和reject_ids拼接起来做forward，会快一些

```python
def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
```


输入最后一个token得到的reward_value结果，作为Reward模型的最终打分。

```python
def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

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
```

## reward hacking

[Amodei et al. (2016)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1606.06565) 指出了在 RL 训练中缓解reward hacking的一些方向：

1. _对抗性奖励函数。_ 我们将奖励函数本身视为一个自适应 AI 智能体，它可以适应模型发现的新技巧，即在奖励很高但人类评分很低的情况下。
2. _模型前瞻。_ 可以根据未来预期状态给予奖励；例如，如果 AI 智能体要替换奖励函数，它将获得负奖励。
3. _对抗性盲化。_ 我们可以用某些变量来蒙蔽模型，这样 AI 智能体就无法学习使其能够破解奖励函数的信息。
4. _精心设计。_ 通过精心设计可以避免某些针对系统设计的reward hacking；例如，对 AI 智能体进行沙箱化，以将其行为与奖励信号隔离开来。
5. _奖励上限 (Reward capping)._ 这种策略旨在简单地限制最大可能的奖励，从而有效防止 AI 智能体通过破解来获得超高回报策略的罕见情况。  
    
6. _反例抵抗 (Counterexample resistance)._ 提高对抗鲁棒性应有助于增强奖励函数的鲁棒性。  
    
7. _多种奖励的组合 (Combination of multiple rewards)._ 结合不同类型的奖励可以增加破解的难度。  
    
8. _奖励预训练 (Reward pretraining)._ 我们可以从 (状态, 奖励) 样本的集合中学习奖励函数。然而，这种监督训练设置的质量会影响结果，可能带来其他问题。[RLHF](https://link.zhihu.com/?target=https%3A//lilianweng.github.io/posts/2021-01-02-controllable-text-generation/%23rl-fine-tuning-with-human-preferences) 依赖于此，但学习到的标量奖励模型很容易学习到不期望的特性。  
    
9. _变量漠视 (Variable indifference)._ 目标是要求 AI 智能体优化环境中的某些变量，而忽略其他变量。  
    
10. _绊线 (Trip wires)._ 我们可以故意引入一些漏洞，并设置监控和警报，以便在任何漏洞被利用进行reward hacking时发出警报。 在强化学习（RL）设置中，如果人类反馈以对AI 智能体行为的_批准_形式出现，[Uesato et al. (2020)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2011.08827) 提出了使用**解耦批准**来防止奖励篡改。如果反馈以 $(s, a)$ (状态，行为) 为条件，一旦这个状态-行为对发生了奖励篡改，我们就永远无法获得状态 $s$ 下行为 $a$ 的未被破坏的反馈。解耦意味着用于收集反馈的查询行为是从在世界中采取的行为独立采样的。甚至在行为在世界中执行之前就收到了反馈，从而防止了该行为破坏其自身的反馈。
## 参考资料

[让 LLM 来评判 | 奖励模型相关内容](https://mp.weixin.qq.com/s/xiQblwvDY8cGz1A42Yb82g)

