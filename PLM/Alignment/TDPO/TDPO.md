---
title: TDPO
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

标题：Token-level Direct Preference Optimization

作者：

链接：

代码：

框架图：


## 背景

在 PPO 训练的时候，我们通常会加上 KL 惩罚来约束模型不要偏离 reference model 过远，

但在 DPO 的实现中却没有并没有添加这一项。

[[TDPO](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2404.11999)] 提出了这一改进，在原来的 DPO loss 上新增了 kl 惩罚项：

![](img/Pasted%20image%2020240615170821.png)

不过，不同于 PPO 中使用 reverse KL，**TDPO 则是使用 forward KL 来计算 KL 惩罚**，

因为 KL 是一个非对称的距离函数，所谓 forward 和 reverse 其意思就是「以 SFT 计算采样概率」还是「以 Policy Model 计算采样概率」。

在 [[源码](https://link.zhihu.com/?target=https%3A//github.com/Vance0124/Token-level-Direct-Preference-Optimization/blob/4f533a7bf8944d287c89a451c2006027e3353f56/trainers.py%23L145)] 中我们能更直观的看到 forward KL 的计算方式：

```python
vocab_logps = logits.log_softmax(-1)

reference_vocab_ps = reference_logits.softmax(-1)
reference_vocab_logps = reference_vocab_ps.log()

# forward kl 计算
# backward kl (PPO) 应为: vocab_logps - reference_vocab_logps
per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
```

由于 reverse KL 的目标是拟合整个分布中的「一部分」，而 forward KL 的目标是尽可能 cover 整个分布中的大部分。因此，**TDPO 训练后的模型会比 PPO 训练后的模型，在输出多样性上更加自由**。

> **PS：**经过 PPO 后的模型基本一眼就能看出来，输出风格都非常一致，因为此时输出分布已经「聚集」到一个局部分布上了，reward 方差会比 SFT 小很多。

完成 loss 函数如下：

```python
def tdpo_loss(
        chosen_logps_margin,
        rejected_logps_margin,
        chosen_position_kl,
        rejected_position_kl,
        beta: float, 
        alpha: float = 0.5, 
        if_tdpo2: bool = True
    ):
    """Compute the TDPO loss for a batch of policy and reference model log probabilities.

    Args:
        chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the TDPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        alpha: Temperature parameter for the TDPO loss, used to adjust the impact of sequential kl divergence.
        if_tdpo2: Determine whether to use method TDPO2, default is True; if False, then use method TDPO1.
    """
    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl
    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)    # tdpo1
    else:
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards
```

## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
