---
title: GRPO
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

在 ReMax 中我们提到：使用一种好的方法来计算 baseline 是丢掉 Critic 网络的关键。

在 [[DeepSpeek-v2](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2405.04434)] 的 RLHF 过程中，这个思路也有被使用，

不过计算 baseline 的方式稍有不同，文章中将其称为 [[GRPO](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.03300)]。

GRPO 认为，直接退化为 Policy Gradient 是不是有点过于原始，

虽然天下苦 Critic 久矣，PPO 中其他先进 features 咱们还是可以保留的：比如 **importance sampling** 和 **clip**。

于是，整个优化目标就变成这样：

![](img/Pasted%20image%2020240615165906.png)

上图中绿色部分是不是非常眼熟，这不就是 PPO 的优化目标嘛。

但现在的问题是：公式中的Ai 在 PPO 中是需要通过 Critic 去参与计算的$(r+V_s{next}-V_s)$, 
可是GRPO 里没有 Critic 啊，这咋计算！

我们回想一下：Critic 的目标是去估计一个状态的期望值（从而降低方差），而期望的近义词是均值，

**那我们直接暴力的去采样 N 次求均值来代替这个期望不就好了！**

没错，这就是 GRPO 暴力且有效的方法：

![](img/Pasted%20image%2020240615170232.png)

这里有几个值得注意的细节：

1. GRPO 中也加入了 KL Penalty，只不过不像 PPO 的实现是每个 token 位置上加一个惩罚，而是直接一并计算完后加到最后的 loss 中去。
2. KL Penalty 使用 [Schulman 近似值](https://link.zhihu.com/?target=https%3A//github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/trainer/accelerate_ppo_trainer.py%23L458) 用以保证 KL 始终为正数，即： 𝑟𝑎𝑡𝑖𝑜−1−𝑙𝑜𝑔𝑟𝑎𝑡𝑖𝑜 。
3. 句子的最终得分为： 𝐴𝑖=(𝑟𝑖−𝑚𝑒𝑎𝑛(𝑟))/𝑠𝑡𝑑(𝑟) ，由于在 LLM 里我们通常将 GAE 中的 𝛾 设置为 1.0，因此在这里 GRPO 也直接将这个最终得分复制到句子中的每一个 token 上进行训练。

尽管这种方法确实可以省掉一个 Critic，但成功需要具备 2 个关键：

1. SFT 对给定的 prompt 不能有着太 diverse 的输出，否则方差会比较大。
2. 对同一个 prmopt 采样的数量要可能大，这样才能降低方差。

这可能是论文选择在「数学任务」上使用这种方式进行训练的原因。


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

[Site Unreachable](https://zhuanlan.zhihu.com/p/20021693569?utm_psn=1871540714904616960)

