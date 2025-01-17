---
title: DPO
created: 2024-06-15
tags:
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


## 损失函数
对于同一个 propmt，给定一个好的回答 𝑦𝑤 和一个不好的回答 𝑦𝑙，**通过降低不好回答被采样的概率，提升好回答的概率**，从而进行模型训练。这个数据和训练 Reward Model 的 pair 数据格式完全一致，都是同一个 prompt 对应两个不同质量的 responses。

![](img/Pasted%20image%2020240615170451.png)

[[源码](https://link.zhihu.com/?target=https%3A//github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)] 中计算 loss 的部分（最简单的sigmoid损失函数）：

```python
def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        return losses
```

rewards的计算方法。所以DPO的loss也可以理解为：

`L_DPO​=−log(sigmoid(chosen_rewards−rejected_rewards))`

```python
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

计算logps的函数

```python
def get_batch_logps(
        self, 
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """

        if logits.shape[:-1] != labels.shape:
            logger.info(f"logits shape: {logits.shape}; label shape: {labels.shape}")
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
```

### chosen_logits和chosen_logps的关系

1. **`chosen_logits`**

- **定义**:
    - `chosen_logits` 是模型对 **chosen responses** 的原始输出（未归一化的 logits）。
    - 形状为 `(batch_size, sequence_length, vocab_size)`，其中 `vocab_size` 是词汇表的大小。
- **含义**:
    - 表示模型对每个 token 的预测分数（logits），即未经过 softmax 归一化的原始分数。
    - 这些 logits 可以用于进一步计算概率分布或损失函数。
- **用途**:
    - 通常用于计算模型的预测分布或与其他模型的输出进行比较。
        

---

2. **`chosen_logps`**

- **定义**:
    - `chosen_logps` 是模型对 **chosen responses** 的对数概率（log probabilities）。
    - 形状为 `(batch_size, sequence_length)`，表示每个 token 的对数概率。
- **含义**:
    - 表示模型对 **chosen responses** 中每个 token 的预测概率的对数值。
    - 这些值是通过对 `chosen_logits` 计算 log-softmax 得到的。
- **用途**:
    - 通常用于计算损失函数（如负对数似然损失）或评估模型的预测质量。


## 核心亮点


## 主要收获


## 参考资料
