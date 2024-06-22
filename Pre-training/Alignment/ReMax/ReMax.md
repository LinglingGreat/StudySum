---
title: ReMax
created: 2024-06-15
tags:
  - rlhf
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

ReMax认为，我们可以丢掉 Critic（教练），Actor 不再需要受到 Critic 的指导，而是直接去对齐 RM（裁判），这样一来，我们就只用载入 3 个模型.

我们只让 actor 去生成行为，然后利用所有行为共同获得分数来训练模型，但是，因为每一个行为（对应生成句子中的每一个 token）都是一个随机变量，N 个随机变量加在一起，**方差就会非常巨大，这通常会导致整个 RL 训练崩掉**。

为了解决这个问题，**我们可以让每一个随机变量都减掉一个 baseline，这样就可以降低方差，稳定训练**。

那么这个 baseline 如何得到呢？

一种很直觉的想法是：我们随机采样 N 次，将这 N 次采样结果的得分「求均值」并作为 baseline，

但这个方法的缺陷也很明显，只有当 N 足够大时，方差才能足够小。

对此，PPO 的处理方式是：使用一个神经网络 Critic 去拟合这个均值（而不是直接叠加），从而减小方差。

而 ReMax 的思路就比较有趣：**使用「当前策略」认为最好的行为来当作 baseline 值**。

![](img/Pasted%20image%2020240615165301.png)

可以看到，在 PPO 中我们计算 actor 分数时是：r - V(s)，而在 ReMax 中变成了：r - r(greedy).

**其中，r(greedy) 是指对于一个 prompt，LLM 在 greedy sample 的情况下生成一个句子，该句子的得分**。

> **PS：**通常情况下我们在 On Policy 训练过程中，LLM 在做 generate 的时会采用 top_p = 1.0, top_k = -1 的采样方式，以增强模型的探索。

使用 greedy 策略生成句子的得分做为 basline，这之所以能够降低方差，是默认认为通常 SFT 模型已经经过一部分对齐，对于同一个 prompt 模型不太会输出差异性过大的答案。

这样看来，ReMax 优化思路也很直觉：模型每次只需要和当前 greedy 策略下进行比较，当这次「探索」的句子的得分大于 greedy 策略生成的句子，那么就鼓励模型朝着这次探索的句子分布进化。于是，很有可能在下一次 greedy 采样时，当前被探索出来的优秀答案就能被采出。

除此之外，ReMax 最大的优势是在于：它丢掉了一个巨大的 Critic 网络。

因此，**在只有 4 张 A800-80G 的情况下，ReMax 也能在不使用 offload 的情况下训练 Llama-7B

训练一步的时间对比如下：

![](img/Pasted%20image%2020240615165549.png)

PPO 只用做一次 generation，需要更新 2 次参数（actor + critic）；

ReMax 需要做两次 generation（训练 sample 1 次 + greedy sample 1 次），需要更新 1 次参数（actor）。

> **PS：**论文中讨论的 PPO 是 actor 和 critic 串行 backward 的情况，事实上由于 actor 和 critic 的 loss 是没有相互依赖的，通常我们可以做成异步更新，其实也就只有 1 个 t_back。




## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

![](img/Pasted%20image%2020240618173735.png)


[源码](https://github.com/liziniu/ReMax/blob/master/step3_rlhf_finetuning/remax_trainer.py) 中计算 loss 的部分如下：

```python
def compute_loss(self, inputs):
    prompts = inputs["prompts"]
    log_probs = inputs["logprobs"]
    ref_log_probs = inputs["ref_logprobs"]
    reward_score = inputs["rewards"]
    baseline_reward_score = inputs["baseline_rewards"]
    attention_mask = inputs["attention_mask"]
    seq = inputs["input_ids"]

    start = prompts.size()[-1] - 1
    action_mask = attention_mask[:, 1:]

    with torch.no_grad():
        kl_divergence = -(log_probs - ref_log_probs)
        kl_divergence = self.kl_ctl * kl_divergence

        reward_score = reward_score - baseline_reward_score         # 真实 reward
        returns, kl_ratio = self.compute_returns(
            prompts, kl_divergence, reward_score, action_mask
        )

    # process the new outputs
    batch = {"input_ids": seq, "attention_mask": attention_mask}
    logits = self.actor_model(**batch, use_cache=False).logits
    log_probs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])

    actor_loss = self.actor_loss_fn(
        log_probs[:, start:], returns[:, start:], action_mask[:, start:]
    )
    return actor_loss, returns[:, start:], kl_ratio


# reward & basline_reward_score 计算如下:
seq = self._generate_sequence(
    self.actor_model,
    prompts,
    ...
)
baseline_seq = self._generate_sequence(
    self.actor_model,
    prompts,
    ...
    do_sample=False,
)
reward_score = self.reward_model.forward_value(
    seq, action_mask, prompt_length=self.prompt_length
)
baseline_reward_score = self.reward_model.forward_value(
    baseline_seq, baseline_action_mask, prompt_length=self.prompt_length
)
```



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
