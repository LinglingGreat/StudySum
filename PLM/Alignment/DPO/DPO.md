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

[源码](https://link.zhihu.com/?target=https%3A//github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) 中计算 loss 的部分（最简单的sigmoid损失函数）：

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


## DPO 是如何简化 RLHF 的

![](img/Pasted%20image%2020250126174234.png)

![](img/Pasted%20image%2020250126174250.png)

![](img/Pasted%20image%2020250126174303.png)

**DPO算法的目的是最大化奖励模型(此处的奖励模型即为训练的策略)，使得奖励模型对chosen和rejected数据的差值最大，进而学到人类偏好。**

dpo 从头到尾都在以 reward_model 的方式让模型学习 evaluate 能力，但是却并没有证明一个重要假设：“**模型的 evaluate 能力和 generate 能力到底是不是相互促进的？**” dpo 后的模型具有了更强的 evaluate 能力，但我们的目标是提升模型的 generate 能力啊。如果这个基本假设不成立，那 dpo 的学习过程就没有什么价值。

也正是因为 dpo 是在让模型具有 reward_model 的能力，所以它并不在乎模型能不能说出一个好的句子，只在乎 loss margin 是否在变大。大家训练 dpo 的时候，基本都遇到过 good_sentence 和 bad_sentence 的 loss 都上升的尴尬现象，往往需要我们加系数和调参数才能解决。

reward_model 的训练方式根本不在乎模型的 generate 能力，因此稳定训练的 dpo 需要魔改 loss 函数。

## reference模型的作用

从DPO最初始的RL优化目标来看，Reference model起到的第一个作用就是**在KL散度中限制Policy Model，让它不要偏离Reference Model太远**。事实上DPO的灵魂也在于此，没有了Reference Model的偏好优化损失，其实就是一个普通的Ranking Loss罢了。

[From r to Q∗: Your Language Model is Secretly a Q-Function](https://link.zhihu.com/?target=https%3A//cn.bing.com/fd/ls/GLinkPing.aspx%3FIG%3D5498F8C437B342958A4C2424A72A6AE5%26%26ID%3DSERP%2C5175.2%26SUIH%3DKWsjordAO3H-wMQHtnpZpw%26redir%3DaHR0cHM6Ly96aHVhbmxhbi56aGlodS5jb20vcC82OTM2NjUxNjg)来进一步理解Reference Model的重要性。

**DPO中的细粒度对齐**

我们首先来看看作者的第一个重要结论:

![图片](https://mmbiz.qpic.cn/mmbiz_png/wAPfqDgY33pwmtfcFiajggichSFEpkmiapJSC3UIlJf1dc2JRRXctjgbDnAS0fibacvrVmz4lr4woaJrUnYOURxxBA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

公式左边是DPO对于第t个token的损失函数，右边就是一个非常标准的Advantage函数定义（详见Actor Critic)。这个结论就非常地强，意思就是说，我虽然没有Value Model，并且训练还是样本级别的pair-wise数据，但是我证明了加上Reference Model之后，相当于引入了一个**细粒度的Value Model**和一个**细粒度的Reward Model**，等价于PPO中Advantage的计算。PPO的优点，其中很重要的一个点就是Value Model可以带来细粒度的监督，而似乎DPO看来也具备同样的性质？ 我们进一步来分析看一下这个结论:

  
![](https://pic3.zhimg.com/v2-9ac8eecdce77adc18b192e2c04ed66be_1440w.jpg)

![](https://pic2.zhimg.com/v2-e6a9223522c376ff8b9b74c600916273_1440w.jpg)

观察上面的公式5和7，将公式7带入公式5，给公式5两边取log，再移项，便可以得到第一个重要结论。所以我们进一步分析一下这两个式子的含义。

先看公式5，原文说公式5是公式2的固定点解(the fixed point solution)，所以我们贴出公式2：

![](https://pic2.zhimg.com/v2-d37779fdd928b95d027c37282db52993_1440w.jpg)

公式2是PPO+KL约束的优化目标，注意H是熵的意思，结合一下熵的定义-plog(p)，可以发现，这里是故意将KL Penalty改写成了 log⁡πref+entropy 的形式，所以这两项之前都有一个 β 。注意这里是非常重要的一个技巧，从而成功把优化目标转化成了 r+log⁡πref ，并且套进了entropy bonus形式的PPO公式。下图是带entropy bonus的PPO，S就是熵。

![](https://pica.zhimg.com/v2-1e43e53742f4cfd1759e32b7eb812ade_1440w.jpg)

优化目标转变了之后，公式7也就顺利成章可以得出了，5和7一结合，就自然完成了公式1的推导。完成了推导之后，再来看一个实际的例子:

![](https://picx.zhimg.com/v2-4bd54655dfe6c56323c473b494ddd645_1440w.jpg)

左边是一个DPO的测试正样本，右边是一个DPO的测试负样本，颜色越深代表token对应的奖励越高，可以看到负样本中有问题的token "250", "great management" 都得到了相对更低的奖励值(Q值)，从而细粒度级别的优化。

**与搜索的联系**

现有的很多工作都在考虑利用value model来指导搜索，由于DPO训练后的对数概率直接就能代表Q函数，

![](https://pic3.zhimg.com/v2-9a77b89cfe0ebff69a9388a5b5eaaa6e_1440w.jpg)

通过公式13的推导，可以很容易发现只需要基于DPO优化后的policy模型来进行搜索，就可以达到利用value model进行搜索的效果。这一点其实也很容易通过一个事实来验证: DPO训练后的模型的pass@k总是明显高于SFT模型。作者同样也观察到用beam size为5进行beam search，就可以有10~15%的胜率提升，和value model guided beam search接近。

本文是DPO原作者为DPO书写的正名之作，深刻理解本文胜过阅读十篇所谓DPO的改进，非常有益于进一步做出好的Offline RL工作。并且我觉得本文可能更加能够印证**[GRPO](https://zhida.zhihu.com/search?content_id=249317057&content_type=Article&match_order=1&q=GRPO&zhida_source=entity)等no-critic方法的意义**(可能REINFORCE+KL真的就够了)，也印证了为啥目前多个大厂最后还是选择了GRPO而不是PPO。最后，在我看来DPO的细粒度优化还是有很大的改善空间的，尽管理论很美好，但现实总是残酷，混沌，并且不可测。未来DPO的改进，相信还是要落在细粒度建模上的。

## DPO训练时，为什么chosen和rejected的reward一起下降

[百面LLM-7](https://zhuanlan.zhihu.com/p/686122806)

[【不靠谱】有关DPO训练时，为什么chosen和rejected的reward一起下降的猜想](https://zhuanlan.zhihu.com/p/694381064)

[DPO正例概率不应该下降？DPO在实践中的反思与改进](https://zhuanlan.zhihu.com/p/698852522)

在以下情况中正例的概率就可能下降：

1. 如果正例并不是一个绝对意义上好的回复而仅仅是相对于负例而言更好，正例的概率降低才是正常的，因为**当前样本的正例可能也是其他样本中的负例（如果正例的某个模式出现在其他样本的负例中也会导致该正例的概率下降）**。
2. 即使数据中的正例可以看作是绝对意义上的好的回复，**但如果query存在多个绝对意义上好的回复，该正例的概率也可能因为其他好回复概率的上升而下降**（参考章节三思考2中提到的场景）。

此外，文无第一，**对于很多任务而言不存在绝对的正确性，不同模型的偏好可能不同，即使某个正例在某个评估标准下没有正确性问题，逻辑也很好，它的概率在训练过程中仍然可能会被降低，因为模型受到其他数据的激发可能认为其他形式的输出更好（比如把解释放在后面而不是放在前面），提升了其他形式输出的概率，进而导致该正例概率的下降**。我们在实验中观察到，正例概率的下降很多时候不是核心答案概率下降导致的，而是模型倾向的回复话术改变了，标注正例的回复话术和模型倾向不一致导致的概率下降。如标注正例是以『这句话。。。』开头，而采用greedy search策略，模型倾向于以『根据文本内容。。。』开头，且该话术在DPO中相较于[SFT](https://zhida.zhihu.com/search?content_id=243430776&content_type=Article&match_order=1&q=SFT&zhida_source=entity)的概率是提升的，此时标注正例开头tokens的概率很多时候就下降了，而这些tokens并不会影响核心答案的正确性。


## 主要收获


## 参考资料

[DPO 是如何简化 RLHF 的](https://zhuanlan.zhihu.com/p/671780768)

[dpo 的局限性](https://zhuanlan.zhihu.com/p/1082394115)

[理解DPO的Reference Model](https://mp.weixin.qq.com/s/60jnAfy6AXA-mjwbB92JtQ)

