---
title: ORPO
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


## 背景

不管是哪种 DPO，除了 policy model 外，都还有一个 reference model，我们能不能把 ref_model 也干掉。

回想一下，在 DPOP 中，我们使用 ref_model 来保证模型在 chosen 上的概率不要过低，

如果只是为了保证模型能够拟合 chosen 答案，那我们是不是直接把 chosen 答案拿出来做 SFT 就好，

这不就不需要 ref_model 来吗？

[[ORPO](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2403.07691)] 的目标函数一共由两部分组成（SFT Loss + Odds Ratio Loss）：

![](img/Pasted%20image%2020240615171020.png)

其中 SFT Loss 就是拿 chosen 答案算 CrossEntropy Loss，这很好理解，剩下的就是这个 Odds Ratio 是什么。

在统计学和概率论中，odds 指的是「某事件发生与不发生的比例」，

比如，如果一件事情发生的概率是 𝑝，那么它不发生的概率就是 1−𝑝，其 odds 计算公式就为：

![](img/Pasted%20image%2020240615171039.png)

当一件事情的发生概率越大，其对应的 odds 值就越大。

知道 odds 的概念后，我们再一起上述 loss function 的后半部分 𝐿𝑂𝑅 的定义：

![](img/Pasted%20image%2020240615171055.png)

通过 minimize 这个 loss 值，我们就需要 maximize 括号内的值，**也就是尽可能的让「好句子」发生的概率增大，「坏句子」发生的概率减小**。

由此可见，**ORPO 通过定义了一个神奇的 odds 值来提升好样本的概率，降低坏样本的概率，并通过一个 SFT loss 来保证模型对 chosen response 的基本拟合**。

[[源码](https://link.zhihu.com/?target=https%3A//github.com/huggingface/trl/blob/main/trl/trainer/orpo_trainer.py%23L667)] 中对 odds_ratio 的计算如下：

```python
def odds_ratio_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
    ):
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """
        # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using 
        # log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (
            policy_chosen_logps - policy_rejected_logps
            ) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - 
            torch.log1p(-torch.exp(policy_rejected_logps))
        )
        sig_ratio = F.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = self.beta * ratio
        return losses
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
