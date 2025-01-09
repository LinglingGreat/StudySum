---
title: README
created: 2024-06-12
tags:
  - alignment
  - 总结
---
## RL: Policy-Based & Value Based

强化学习（Reinforcement Learning, RL）的核心概念可简单概括为：一个机器人（Agent）在看到了一些信息（Observation）后，自己做出一个决策（Action），随即根据采取决策后得到的反馈（Reward）来进行自我学习（Learning）的过程。

RL 的最终目标其实就是要让机器人（Agent）学会：**在一个给定「状态」下，选择哪一个「行为」是最优的**。

一种很直觉的思路就是：我们让机器人不断的去玩游戏，当它每次选择一个行为后，如果这个行为得到了「正奖励」，那么下次就多选择这个行为；如果选择行为得到了「负惩罚」，那么下次就少选择这个行为。

为了实现「多选择得分高的行为，少选择得分低的行为」，早期存在 2 种不同的流派：Policy Based 和 Value Based。

其实简单来说，这 2 种流派的最大区别就是在于将行为量化为「概率」还是「值」，具体来讲：

1. **Policy Based：**将每一个行为量化为**「概率分布」**，在训练的时候，好行为的概率值将被不断提高（向右走，0.9），差行为的概率将被不断降低（向上走，0.1）。当机器人在进行行为选择的时候，就会按照当前的概率分布进行采样，这样就实现了「多选择得分高的行为，少选择得分低的行为」。
2. **Value Based：**将每一个行为量化为**「值」**，在训练的时候，好行为的行为值将被不断提高（向右走，1分），差行为的行为值将被不断降低（向上走，-1）。当机器人在进行行为选择的时候会选择「行为值最大的动作」，这样也实现了「多选择得分高的行为，少选择得分低的行为」。

	比较出名的代表算法：[Policy Gradient](https://www.researchgate.net/publication/2503757_Policy_Gradient_Methods_for_Reinforcement_Learning_with_Function_Approximation)（Policy Based）和 [Q-Learning](https://paperswithcode.com/task/q-learning)（Value Based）。

## 序列决策（Sequence Decision）以及单步奖励（Step Reward）的计算

「单步决策」：机器人只做一次决策，GPT 也只生成一个字。

但事实上，机器人想要拿到钻石，通常需要做出 N 次行为选择。

在这种情况下我们最终只有 1 个得分和 N 个行为，但是最终 RL 更新需要每个行为都要有对应的分数，

我们该如何把这 1 个总得分对应的分配给所有的行为呢？

答案是计算「折扣奖励（discount reward）」。

我们认为，越靠近最末端的行为对得分的影响越大，于是从后往前，每往前行为就乘以 1 次折扣因子 γ.

同样，GPT 在生成一个完整句子的过程中，也会做出 N 个行为（续写 N 个字），而我们在评分的时候，只会针对**最后生成的完整句子进行一个打分**（而不是生成一个字打一个分），最后，利用上述方法通过完整句子的得分倒推出每个字的对应得分

值得注意的是：通常在对 GPT 生成句子进行得分拆解的时候，折扣因子（γ）会取 1.0，这意味着，在句子生成任务中，**每一个字的生成都会同等重要地影响着最后生成句子的好坏**。

> 我们可以这么理解：在找钻石的游戏中，机器人采取了一些「不当」的行为后是可以通过后续行为来做修正，比如机器人一开始向右走（正确行为），再向左走（不当行为），再向右走（修正行为），再向上走（正确行为），这个序列中通过「修正行为」能够修正「不当行为」带来的影响；但在句子生成任务中，一旦前面生成了一个「错别字」，后面无论怎么生成什么样的字都很难「修正」这个错别字带来的影响，因此在文本生成的任务中，每一个行为都会「同等重要」地影响最后句子质量的好坏。


## 加入概率差异（KL Penalty）以稳定 RL 训练

除了折扣奖励，在 OpenAI 的 [Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325.pdf) 这篇工作中指出，

在最终生成句子的得分基础上，我们还可以在每生成一个字时候，计算 RL 模型和 SFT 模型在生成当前字的「概率差异」，并以此当作生成当前字的一个 step reward

通常在进行 RL 训练时，初始都会使用 SFT 模型做初始化，随即开始探索并学习。

由于 RL 的训练本质就是：探索 + 试错，加上「概率差异」这一限制条件，就相当于限制了 RL 仅在初始模型（SFT）的附近进行探索，这就大大缩小了 RL 的探索空间：既避免了探索到那些非常差的空间，又缓解了 Reward Model 可能很快被 Hacking 的问题。

## On Policy + Off Policy

可以简单理解为：**凡是需要 LLM 在训练过程中做 generation 的方法就是 On Policy，反之为 Off Policy**。

On Policy 的核心思路就是：**让模型自己做生成，我们根据模型生成结果的好坏来打分，用于指导模型进行更新**。

一个标准 PPO 所需要的 4 个模型：

- Actor：用于生成句子的模型，也就是正在被训练玩游戏的你。
- Critic：指导你进步的教练模型，注意，这个教练模型也会随着你的进步来调整自己的指导策略。比如，当你很菜的时候，突然打出了一个很强的操作时，会给你一个较高的分数（Vs 较低，因此 r - Vs 就较大，看不懂这句话没关系，我只是尝试证明这个例子的存在一定合理性），当你本身比较强了，再打出同样操作的时候给的奖励就没有之前那么高。因此，训练过程中 Critic 是和 Actor 一起训练的。
- Reward Model：用于给出最终分数的模型。虽然教练能够给你一定的指导，但最终游戏获胜与否还是要靠裁判说了算，可以说教练在教你的同时也在尝试学习裁判的偏好。裁判一般是固定的，因此 Reward Model 在整个训练过程中参数是被冻结的。
- Reference Model：这是 PPO 在 LLM 中独有的概念，目的是为了让 actor 不要训练偏离太远，主要是缓解 reward hacking + 稳定训练使用的。

通常来讲，这 4 个模型都是同样规模参数的模型.

Online:

- [ReMax](ReMax/ReMax.md)

- [GRPO](GRPO/GRPO.md)

Offline:

- [DPO](DPO/DPO.md)
- [DPOP](DPOP/DPOP.md)
- [TDPO](TDPO/TDPO.md)
- [ORPO](ORPO/ORPO.md)


## 提高性能

[FollowComplexInstruction](FollowComplexInstruction/FollowComplexInstruction.md) 
- 具有复杂约束的数据相比单一约束具有更好的效果。
- 先用弱LLM输出再由强LLM对错误的输出进行纠正细化，要优于直接用强LLM的输出（前者数据的微调效果更佳）。前者的数据还可以用来做偏好训练。

[LearnFromFeedback](LearnFromFeedback/LearnFromFeedback.md)
- 利用大型语言模型（LLM）(Mixtral-8x7B) 来识别和分类对话中包含反馈的文本片段。
- 使用正面反馈样本进行微调，使用包括正面和负面样本的KTO偏好训练。

[ArenaLearning](ArenaLearning/ArenaLearning.md)
将数据分成多个批次。
- 给定数据1，让模型v0和几个优秀的LLM PK，选出弱模型失败的那些数据，将强模型的输出作为groud truth去微调得到SFTv1。
- 给定数据2，让模型SFTv1和几个优秀的LLM PK，选出弱模型失败的那些数据pair对，DPO训练得到DPOv1
- 给定数据3，让模型DPOv1和几个优秀的LLM PK，选出弱模型失败的那些数据pair对，训练reward模型和PPOv1
- 再用PPOv1重复迭代上述过程

## 经验分享

来自character.ai
- 分析数据；收集用户反馈；更好地评估；更好地AB测试
- 高质量数据非常重要，真的高质量吗？能不能更高质量？
- 用户反馈数据作为分类，一直在变化，高质量数据也在随着变化，SFT模型也随着变化
- 最终能够在我看来最有效帮助到这样一个过程，就是怎么样建立起一个尽可能高效的迭代过程。这个迭代过程可以说是管线非常的robust，所以我有大量的用户，或者我的用户量并不是很大。但是我用的AB测试的工具，能够快速的让我高效的收集到各种模型的小的变化对用户测的的影响。然后我能做一定的分析，积累性能好，或者有人说我在评估上特别努力，对吧？
- 我内部做的这个评估机，它非常像真实的用户。我可能说内部圈了一个特殊的模型才能模拟。现在这个用户在说的话，他能用这个模型去跟这个新的模型去对话，然后来告诉你说这个模型是不是会被用户更加喜欢。也有人说我在数据的利用上，我做的特别的高效。只要你今天给我点个赞，可能明天这个模型在跟这个用户在聊的时候，它的效果就会更好。
- 就是我工作的主要目要方式，其实就是看大量的数据，另外一方面是研究数据，那边就是分析最近的几次迭代的效果。怎么样去理解里面可能说模型是应该调数据还是调算法。然后在实际上工程那边就会思考我们宪。这个管线里面是不是有些用户数据的使用方式还是不够优秀。或者比如说我们要做偏好对齐的话，DPU这个算法最近有没有什么业界的新的研究，发现它有一些缺陷可以去改善。当然少不了就大量的跟研究员去讨论，看看研究员那边对于最新的业绩的方法有没有什么新的见解。一般可能一天8个小时里面，我觉得真正的在写代码程序实现的里面，大概不会超过2个小时。6个小时基本都是在各种交流，还有分析各种数据。
- 我觉得最后能分享的东西就是说你怎么能最快的把用户反馈带着飞起来，对吧？就是上一代的AI模型，大家都会说有一个数据飞轮，我觉得这一代同样也有个数据飞轮，而且这一代的数据飞轮效应更加强烈。因为大模型本身就是个数据黑洞，就你喂他一堆数据，然后他吐出来一堆数据给你。而且这里面有很强的一个随机性，很大的不可控性。所以你可能在快速迭代的时候，你得不停的去改变这个数据配比。

## 参考资料

[【RLHF】RL 究竟是如何与 LLM 做结合的？](https://zhuanlan.zhihu.com/p/675329917) (已整理)

[Reinforcement Learning from Human Feedback 全家桶（RL 侧）](https://zhuanlan.zhihu.com/p/700149886) 介绍了PPO，ReMAX, GRPO, DPO, DPOP, TDPO, ORPO等方法。

#alignment [Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts | RLHFlow](https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/)

[Token-level Direct Preference Optimization](https://papers.cool/arxiv/2404.11999) TDPO试图通过在token级别上直接优化策略，同时控制KL散度，来提高语言模型与人类偏好的对齐度，并保持生成响应的多样性。论文通过在多种文本任务上的实验结果表明，TDPO在平衡对齐度和生成多样性方面优于DPO和基于PPO的RLHF方法。 #alignment 

#rlhf #alignment Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint https://arxiv.org/pdf/2312.11456v4  这篇一个是第一个做了RLHF 的理论, 处理了 KL-regularized contextual bandit (不同于之前的dueling bandit) 的数学原理; 第二个是从理论insight 出发说明online iterative RLHF 的好处; 第三个就是自然导出了 online iterative DPO 这样一个算法, 用我们最近开源的reward model (reward bench 上现在sota的开源 rm), 可以很轻松把 Zephyr-7B-SFT 在 Alpaca-eval 4.63% -> 35.95%, mt bench 5.3 -> 7.5。

[人类偏好对齐训练技术解析](https://mp.weixin.qq.com/s/Zo274CCITKGRn0dKD8WNJA)

[Alignment Guidebook](https://efficient-unicorn-451.notion.site/Alignment-Guidebook-e5c64df77c0a4b528b7951e87337fa78)

[A recipe for frontier model post-training](https://www.interconnects.ai/p/frontier-model-post-training)：Apple、Meta 和 Nvidia 都同意——合成数据、迭代训练、人类偏好标签和大量过滤。

[Llama3.1，DeepSeek-V3，TÜLU 3，Qwen2.5后训练合集](https://zhuanlan.zhihu.com/p/12862210431)

[拒绝采样](https://zhuanlan.zhihu.com/p/3907736367)

