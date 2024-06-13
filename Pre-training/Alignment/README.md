---
title: README
created: 2024-06-12
tags:
  - alignment
---

[Reinforcement Learning from Human Feedback 全家桶（RL 侧）](https://zhuanlan.zhihu.com/p/700149886) 介绍了PPO，ReMAX, GRPO, DPO, DPOP, TDPO, ORPO等方法。

#alignment [Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts | RLHFlow](https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/)

[Token-level Direct Preference Optimization](https://papers.cool/arxiv/2404.11999) TDPO试图通过在token级别上直接优化策略，同时控制KL散度，来提高语言模型与人类偏好的对齐度，并保持生成响应的多样性。论文通过在多种文本任务上的实验结果表明，TDPO在平衡对齐度和生成多样性方面优于DPO和基于PPO的RLHF方法。 #alignment 

#rlhf #alignment Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint https://arxiv.org/pdf/2312.11456v4  这篇一个是第一个做了RLHF 的理论, 处理了 KL-regularized contextual bandit (不同于之前的dueling bandit) 的数学原理; 第二个是从理论insight 出发说明online iterative RLHF 的好处; 第三个就是自然导出了 online iterative DPO 这样一个算法, 用我们最近开源的reward model (reward bench 上现在sota的开源 rm), 可以很轻松把 Zephyr-7B-SFT 在 Alpaca-eval 4.63% -> 35.95%, mt bench 5.3 -> 7.5。

