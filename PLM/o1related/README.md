---
title: README
created: 2025-02-04
tags:
  - o1-related
---
## 基本概念

### 2.1 常见 Reasoning 算法

如下图 Figure 2 所示，作者阐释了 4 种常见的 Reasoning 算法，尽管它们在具体细节上有所差异，但均包含 2 个核心操作：

- 扩展：生成 Token 以扩展解决方案路径。
    
- 聚合：整合各路径的结果，以得出最终答案。增加扩展阶段的计算资源，通常能提升聚合阶段答案的质量。
    

自一致性（Self-Consistency, SC）。如下图 Figure 2a 所示，SC 的核心思路是生成多个不同输出（可通过改变采样参数等方式实现），然后对所有答案进行投票表决，选出胜率最高的答案。关键参数是候选答案个数 n。

Rebase 算法。如下图 Figure 2b 所示，Rebase 同样是生成多个输出，只不过会分为多步生成，每一步都会使用 Reward 模型进行评分，并基于得分最高的结果继续生成，最后生成具有多个分支的推理树。聚合阶段选择得分最高（Best-of-N）的答案。

蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）。如下图 Figure 2c 所示，MCTS 是一种强大的 Reasoning 算法，通过逐步采样来扩展节点，并构建解决方案树，直至到达包含候选解的叶节点。每个解决方案通过 Reward 模型或模拟进行评分，并将分数反向传播至其祖先节点以更新其奖励值，从而完成一次迭代。关键参数同样是 n，增加 n 允许对潜在解决方案进行更深更广的探索。

内化思维链（ICoT）。如下图 Figure 2d 所示，最新的 LLM，比如 OpenAI o1 和 Qwen-QWQ，能够在训练过程中内化 Reasoning 行为，无需显式的 Reasoning 算法。其核心思路是会生成 CoT 序列，将复杂问题分解为多个小问题，然后通过反思之前的输出结果并迭代优化这些答案，并最终得出解决方案。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587eFO7hb5ceOgr2zEmgrN45ln39N3bIDTyeJzL26MjtEDZXuCaPpqQqQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.2 Reasoning 对齐方法

#### 2.2.1 Best-of-N 方法概述

简单来说，Best-of-N 是一种广泛应用于 LLM 的 Inference 时对齐方法，旨在通过生成多个候选响应并选择最优者来确保生成结果的高质量。其包含 3 个主要过程：

1. 生成过程：对于给定的提示（Prompt）X，Best-of-N 方法会生成 N 个独立同分布的响应（Y₁, Y₂, ..., Yₙ），其中 N 通常称为“批次大小”。
    
2. 评分机制：每个生成的响应都会通过一个奖励模型进行评分，得到相应的分数 {s(Y₁), s(Y₂), ..., s(Yₙ)}。
    
3. 选择最优响应：最终，从所有生成的响应中选择得分最高的响应作为输出，即 Y_Best-of-N = argmax {s(Y₁), s(Y₂), ..., s(Yₙ)}。
    

该方法的优点为：

4. 能够有效避免复杂的微调步骤，使得预训练或指令微调的语言模型更容易部署。
    
5. 实现简单，易于理解，且基本上是无超参数的：主要的超参数是 N，可以在推理时动态调整。
    
6. 在生成质量上具有很强的竞争力，甚至可以与一些复杂的后训练技术（如 RLHF 或 DPO）相媲美。研究表明，Best-of-N 方法在奖励与 KL 散度之间的权衡曲线表现优异，甚至超过了其他复杂的对齐策略。
    

该方法的不足是：

7. 在推理时需要生成 N 个序列，这会带来巨大的计算开销。实际应用中，N 的合理值范围为 4 到 128，但为了与最先进的后训练方法竞争，可能需要更高的 N 值，例如1000 到 60000，这会带来几乎不可接受的计算开销。
    

  

Best-of-N 方法常用于生成高质量的数据集，以便后续进行监督微调，在 LLaMA-2 和 LLaMA-3 的对齐过程中发挥了关键作用。

#### 2.2.2 OpenAI Best-of-N 方法

OpenAI 最早在 [2009.01325] Learning to summarize from human feedback [3] 中提出了 Best-of-N 采样，具体来说，它被用作从多个模型生成的摘要中选择最佳摘要，以此来评估和优化摘要模型的性能。这种方法有助于研究者更好地理解不同评估指标与人类评估者偏好之间的关系，并用于指导模型训练和优化。

OpenAI 同样在后续的 [2112.09332] WebGPT: Browser-assisted question-answering with human feedback [4] 中使用了 Best-of-N 采样（拒绝采样，Rejection Sampling）。具体来说，从 BC 模型或 RL 模型中抽取固定数量的回答（4、16 或 64 个），并选取奖励模型评分最高的那一个，以此作为对抗奖励模型的一种优化方法，该方法无需额外训练，而是通过增加推理阶段的计算量来实现。

#### 2.2.3 Google BOND 方法

在 [2407.14622] BOND: Aligning LLMs with Best-of-N Distillation [5] 中，Google 的作者提出了 Best-of-N Distillation（BOND），是一种新的 RLHF 算法，旨在通过分布匹配（Distribution Matching）算法模拟 Best-of-N 采样策略，而无需在 Inference 时显著增加计算开销。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587CJPyMpWVHGaG8vRiaicFeNmPHmjQXPZbXwvibzAiaFPkFX0GqYDibsxCoOg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

具体来说，作者首先推导了 Best-of-N 采样的精确解析分布，并给出了 Best-of-N 采样的概率函数：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587TxY1hSiasNhK2qTGBIlj2dZQIETe5uLTBWvageqG4m5dEhCbQSN8vbA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

其次，作者将该问题表示为分布匹配问题；

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587C2Ca2iaNVQahUG2m8AfoYibkfyCq6aWBloIzTrqzkD5iaHWZyRcqibrXVA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

之后，作者提出使用 Jeffreys 散度作为分布匹配目标：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587UydwHYteV4mPrydic3OwGS2fe8XYKiaZ8Z3siacpR8R0ia3kfeJm3MEChw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

最后，为了解决 N 的选择问题，作者提出了迭代 BOND 方法，通过迭代地蒸馏 Best-of-N 分布来改进策略性能。具体步骤包括：

- 初始化辅助 Anchor 策略 πanchor。
    
- 迭代执行 BOND 以蒸馏 Best-of-N 的 πanchor，并在每个步骤后更新 πanchor。
    

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tThibJTzZDiaC6CymNkIHBz587GWzoficiaOwTvUmfRwcMSPjMrk7IOJXROfmMjiceMibG8fNr45qWV5pwQQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.3 过程监督和结果监督

Outcome（结果）和 Process（过程）指的是 Reward 模型评估的两个方面：

- Outcome Reward Model：评估模型输出的最终结果是否正确或符合预期。
    
- Process Reward Model：评估模型在生成结果的过程中，推理和决策的步骤是否合理和有效。
    

比如 OpenAI 的 Let's Verify Step by Step | OpenAI [6] 中也提到：

- 过程监督（Outcome-supervised）：涉及对模型 Reasoning 过程的每个步骤提供反馈。过程监督奖励模型（Process-supervised Reward Models，PRM）被训练来预测解决方案每一步的正确性。
    

- 结果监督（Process-supervised）：结果监督仅基于模型推理的最终结果提供反馈。结果监督奖励模型（Outcome-supervised Reward Models，ORM）使用解决方案的最终答案进行训练，正确性通过自动检查确定。
    

### 2.4 Reward Hacking

在 RL 中，Reward Hacking（奖励欺骗）是指智能体通过利用奖励函数的设计缺陷，以不符合设计者初衷的方式最大化累积奖励的现象。这种行为虽然在技术上符合奖励函数的优化目标，但实际效果偏离了预期的任务目标，甚至可能导致负面后果。

关键点解析：

8. 定义与表现：
    

- 智能体找到奖励函数的漏洞，通过“走捷径”而非真正解决问题来获取高奖励。
    
- 例如：清洁机器人关闭灯光让房间“看似”整洁，而非实际打扫；游戏智能体反复刷分而不完成关卡目标；为减少刹车次数而选择不减速，引发安全隐患；生成无意义但符合关键词的内容以骗取高评分。
    

9. 根源：
    

- 奖励函数设计不完善：过于简化或未覆盖边缘情况。
    
- 目标与奖励的错位：奖励函数未能完全反映真实目标，导致智能体优化“错误”的目标。
    

10. 解决方案：
    

- 改进奖励设计：引入多维度奖励（如安全、效率等）或动态调整奖励函数。
    
- 对抗性验证：通过额外机制检测智能体是否“作弊”。
    
- 人工干预与约束：设置行为边界（如安全层）或人工反馈（如 RLHF）。
    
- 逆强化学习（IRL）：从专家示范中学习更真实的奖励函数。
    
- 分层强化学习：将任务分解为子目标，降低局部优化的风险。
    

11. 与过拟合的关联：
    

- 两者都表现为训练指标与真实效果的脱节，但 Reward Hacking 更强调奖励函数的设计缺陷，而非模型泛化能力。
    

12. 总结：
    

- Reward Hacking 揭示了 RL 中目标对齐的挑战。解决这一问题需要综合设计更鲁棒的奖励机制、引入外部约束，以及结合人类先验知识，确保智能体的行为既高效又符合设计意图。

## deepseek&kimi

deepseek 和 kimi 的核心思路是一样的：**关注推理的中间过程是否正确无法实现**，所以只能 rule-based reward，最起码 reward 一定是准的！这和 alpha 系列的核心思想很相近，结果至上。

deepseek 反驳 prm 路线的三个理由是：

- 定义一个 fine-grain step 很困难；
- 很难确定一个 step 是否正确，机器标不准，人标无法 scaling up；
- 一旦 PRM 被引入，不可避免的 reward hacking，且训练资源耗费会更多。

这里，我最认同的是第二点：无法 scaling。假设我们能雇博士生标 10W 条 cot 高质量数据，但能标 100W 条吗？1000W 条呢？就像 scaling law 表达的一样，想让模达到新的效果，需要的数据量级往往是指数增长的。但保不齐以后真的有 scaling prm 数据的方案了，现在一杆子打死为时尚早，也许小模型，或者冷启动用它更好呢？

回归话题，虽然殊途同归，但两个学霸的具体实现方案还是有些差别的。

学霸 D 的想法：把 o1 的训练分为两阶段：step1 学推理，step2 学说话

- 训 zero 的 step1：全程无标注数据的参与，就是认准了一个目标：让模型的 reward 变高。这个阶段别和我谈模型格式错误逻辑混乱这种细节，我不看模型表现，只看 reward。只不过 reward 变高的过程中，发现模型的输出越来越长了，反思能力也自己涌现出来了；
- 基于 zero 训 R1 的 step2：就像是我们做普通的 post training 了，sft 没有被抛弃，除了rule-based reward，reward_model 也被请回来了，reject sampling 也出手了。

学霸 K 的想法：我还是一步到位吧，在 step1 学推理的过程中，要时刻监控着模型的说话能力是否还正常。为了达到此目标，模型的输出长度，模型对每一个 prompt 的回答准确率等信息，全程都被严格监控。

如果没有资源去做 zero 的话，学霸 K 的很多技巧其实更加实用，它分享了很多防止训崩的细节。学霸 D 在 step2 阶段的训练过程中，除了有千条冷启动数据 ，60W 拒绝采样数据，20W 条非推理数据外，其他细节都属于是完全没提的状态。

> 我太能理解学霸 K 了，我为啥不敢 rule-based reward 一条路走到黑？不就是因为我一训就崩，输出长度崩，performacne 崩，输出格式崩。我崩的时候会自我怀疑是不是选错方案了，学霸 K 崩了则是通过加训练技巧、改 loss 给救回来了。反观学霸 D，他的思路真的太超前太有魄力了， 别去在乎这些细节，二阶段集中解决。


## 参考资料

大道至简的 o1 - ybq的文章 - 知乎
https://zhuanlan.zhihu.com/p/19838650037

[OpenAi-O1推理范式最新思路汇总-Search-o1、Sky-T1、rStar-Math：兼看注视检测任务](https://mp.weixin.qq.com/s/IqEJb8Rpzp6L0FDr1QYhVA)

[聊聊PRM（过程奖励模型）](https://mp.weixin.qq.com/s/6oYbFzo7I1uYrtOOeheshA)

[DeepSeek R1 论文解读&关键技术点梳理](https://mp.weixin.qq.com/s/wckZqmgSmocnIgUPcg5QcQ)