# 强老师救不了OPD

卷友们好，我是rumor。

现在做大模型后训练，应该没人能绕开On-Policy Distillation(OPD)了。一众主流模型都把它放进了核心训练管线，OPD已经成了继SFT、结果奖励式RL之后，大模型的第三大标配技术。

但很多用过OPD的人，都踩过同一个让人崩溃的坑：**换了个参数量更大、benchmark分数更高的强老师模型，结果蒸馏效果不仅没变好，反而直接训崩了；甚至有时候，一个分数更低的小老师，蒸馏效果反而吊打强老师**。

为什么会这样？是学习率没调好、batch size没配对，还是OPD本身就有一套我们从未搞懂的底层规则？

清华最新的Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe[1]，把OPD这个工业界天天用、却没人彻底搞懂的黑箱，完完整整拆开了。

**从OPD成败的核心条件，到token级的底层机制，再到训崩了怎么救，最后到OPD的天生天花板，给了一套逻辑闭环、可直接落地的完整答案**。

## OPD是什么

我们都熟悉的**传统off-policy蒸馏，本质是「老师提前写好标准答案，学生死记硬背」**：老师先提前生成一批固定的回答序列，学生照着这些序列拟合学习。但这里有个解不开的死结——训练时学生学的是老师的分布，推理时却要从自己的分布里生成内容，学生自己写出来的步骤，和老师提前写的标准答案完全不一样，之前背的东西不一定能用得上。这就是传统蒸馏越训越偏、长序列效果拉胯的核心原因。

而OPD的核心革新，就是彻底解决了这个分布不匹配的问题，它的逻辑是「学生先自己写解题步骤，老师全程跟着做一对一指导」：

先让学生模型自己生成完整的回答轨迹(rollout)，走到它自己会访问的状态里；

在学生生成的每一步前缀上，都用老师模型的token级对数概率，作为密集奖励信号；

优化目标是最小化学生自己生成的轨迹上，师生分布的反向KL散度，只教学生真正走到的地方。

简单说，传统蒸馏是老师让学生硬背自己的解题思路，OPD是老师顺着学生的解题思路，一步步纠正优化。这也是为什么OPD能快速成为工业界标配——它从根源上缓解了分布不匹配的问题，训练稳定性和效果上限都远超传统蒸馏。

## OPD的两个核心条件

论文最核心的贡献，就是用严谨的对照实验证明：**OPD的成败，和老师的绝对分数几乎无关，只看两个核心条件有没有被满足**。用强老师训崩了，大概率是没满足这两个条件。

### 1. Thinking-pattern consistency

论文的第一组对照实验，直接锁定了这个核心变量。

实验设置非常干净：固定学生为Qwen3-1.7B-Base，对比两个参数量完全相同、benchmark分数几乎持平的老师：

老师A：官方发布的Qwen3-4B(Non-thinking)版本

老师B：基于Qwen3-4B-Base，用GRPO做了一轮RL训练的版本

两个老师在数学推理benchmark上的准确率相差不到3%，但用它们分别做OPD蒸馏，效果天差地别——RL训练后的老师B，给学生带来的准确率提升，显著超过老师A（见下图）。

![Thinking-pattern consistency](https://mmbiz.qpic.cn/mmbiz_png/2oicuz5vRaSsgcS52WliaHtUv8cic6ICv74uP6JMUmIKCdxY1yh1D9ibibx1lOc1Q67j0FXvM06ED8X0qD2CCBTLxyfGOz1TzGQQ9XIrvzda0Llk/640?wx_fmt=png&from=appmsg)

分数差不多，为什么效果差这么多？

论文追踪了一个全新的核心指标：**初始重叠率(overlap ratio)**，也就是在学生生成的前缀上，师生模型预测的top-k高概率token集合的平均重叠比例。如上图右所示，RL训练后的老师B，和学生的初始重叠率远高于老师A，这意味着，它和学生的「思维模式」是高度兼容的。

这就是OPD成功的第一个核心条件：**老师和学生的思维模式必须足够兼容，初始重叠率必须足够高**。

### 2. Higher scores ≠ New knowledge

论文的第二组实验，给出了第二个核心条件。

实验同样做了严格的变量控制：固定学生为DeepSeek家族的R1-Distill-1.5B，对比两个同家族的老师：

老师A：R1-Distill-7B，和学生来自完全相同的训练数据、完全相同的训练管线，只是参数量更大，benchmark分数更高

老师B：Skywork-OR1-Math-7B，在R1-Distill-7B的基础上，做了额外的RL数学后训练，获得了新的推理能力

![Higher scores ≠ New knowledge](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSuLMnKkdXATwcaFNzDHVNGKhXcw3SqWbDHgwRl6LpY499QWc2Wsziber46ERh48rduxeyXVY4nJ6mTrYEFcmQhSibzrbdc9Lvtyo/640?wx_fmt=png&from=appmsg)

结果见上图，论文用**差距回收率(gap recovery rate)**来量化蒸馏的真实效果，也就是OPD后学生的性能提升量，占师生初始性能差距的比例，结果非常极端：

同管线的7B强老师A，差距回收率仅5.3%，几乎没给学生带来任何有效提升

做了额外RL训练的老师B，差距回收率达到16.9%，是老师A的3倍多

在Qwen家族的重复实验里，这个差距更夸张：同管线的Qwen3-4B老师，差距回收率只有15.6%；而做了额外RL数学训练的Qwen3-4B老师，差距回收率直接达到58.6%，提升超过3倍。

核心逻辑非常简单：同数据、同训练管线训出来的大模型，哪怕参数量翻了几倍，本质上只是对同一套数据的拟合度更高、熟练度更强，没有任何学生没见过的、真正的新能力。而只有当老师通过额外的RL后训练，获得了学生在之前的训练中从未接触过的新能力、新的推理范式时，它才能给学生提供真正有效的驱动信号，让学生获得质的提升。

这就是OPD成功的第二个核心条件：**老师必须具备学生从未接触过的、真正的新能力**。

## 反向蒸馏实验

为了同时验证这两个条件，论文设计了一个绝妙的反向蒸馏实验，推翻了「老师benchmark分数越高，蒸馏效果一定越好」的直觉。

实验步骤是这样的：

先把R1-Distill-1.5B用RL做训练，得到能力更强的JustRL-1.5B，它的benchmark分数远超原始的1.5B模型；

反转蒸馏方向：用能力更强的JustRL-1.5B当学生，分别用两个老师做蒸馏；

老师A：R1-Distill-1.5B，也就是JustRL-1.5B做RL之前的checkpoint，能力更弱；

老师B：R1-Distill-7B，同管线的大模型，benchmark分数比JustRL-1.5B还要高。

![Reverse distillation](https://mmbiz.qpic.cn/mmbiz_png/2oicuz5vRaSte6icvkF8ib4nNAyZtf5BOZlm0jibRnAAYO5wGwKTyQz9zGcUqny8ic2Ylv6Yl8cy75a2cJrI4g3wQRgGJUrG2j9icb5OjEfXGcpnY/640?wx_fmt=png&from=appmsg)

结果如上图，**两个蒸馏实验的训练轨迹几乎完全一致，学生都退化回了RL之前的性能水平**。

分数更高的7B强老师，和更弱的1.5B老师，蒸馏效果没有任何区别。核心原因就是：

两个老师和学生的思维模式，都来自同一条训练管线，高度兼容，满足第一个条件；

两个老师都没有给学生提供任何新能力——1.5B老师是学生的过去版本，7B老师和学生是同数据同管线训练的，能力分布高度重叠，都没有新东西可教，满足不了第二个条件。

## OPD的底层机制

搞懂了OPD成败的两个条件，接下来的问题是：为什么这两个条件能决定成败？在token层面，OPD到底是怎么生效的？

论文用大量的消融实验，揭开了OPD的底层机制，核心结论一句话就能说清：**成功的OPD，本质上是师生高概率重叠token的渐进式对齐；OPD 97%-99%的有效梯度，全来自这些重叠token**。

### 成功vs失败的训练动态，天差地别

![训练动态对比](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSuZAZib69My9yfbeQCCuLnDzvInqEC4QVwuR3zgDSw1Dh8K1NLtiapiaJa4Fd8HibsFGrSNExapkAoHKiaZQMbv5j4wsja3OsWpLJibk/640?wx_fmt=png&from=appmsg)

论文定义了三个核心动态指标，全程监控OPD的训练过程，成功和失败的训练，呈现出完全相反的特征：

**Overlap Ratio**：即每一步里学生和老师top-k token的重叠率。成功的OPD，重叠率会从初始的72%稳步上升到91%；失败的OPD，重叠率从头到尾停滞不动，甚至会下降。

**Overlap-Token Advantage**：衡量重叠token集合内，师生分布的一致性，成功的OPD里，这个指标会稳步向0收敛，说明学生在重叠token上的置信度，和老师越来越匹配；失败的OPD里，这个指标全程保持较大的负值，没有任何收敛趋势。

**Entropy and Entropy Gap**：师生在同一生成状态上的熵的差值，成功的OPD里，熵差会持续缩小，说明学生逐步匹配了老师在对应状态上的不确定性分布；失败的OPD里，熵差全程保持高位，没有任何缩小的迹象。

更关键的是，论文通过统计发现：**在训练全程，师生模型给整个词表分配的总概率里，有97%-99%的概率，都集中在了两者共同认可的重叠token上。** 也就是说，这些重叠token，就是师生分布里绝对的核心区域，OPD的优化，本质上就是在这个核心区域里做对齐。

### 重叠token，才是OPD的有效梯度核心

为了验证重叠token是不是真的决定了OPD的效果，论文做了一个决定性的消融实验：把top-k token集合，拆成「师生重叠部分」和「非重叠部分」，分别单独计算损失、做优化。

![重叠token消融实验](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSsVUPeuTibLovmDBErQgRBDsQ6mHvaGqdhFRmbHBG2cMMU7IhmaXN230xf4Gt5PszHK4WuoAjYsKHicVd0u05YxektOLIUgZiaR0M/640?wx_fmt=png&from=appmsg)

实验结果一目了然：

只在重叠token上做优化，效果和完整的top-k OPD几乎完全一致，没有任何性能损失；

只在非重叠token上做优化，效果大幅跳水，几乎没有给学生带来任何有效提升。

这个实验直接证明：**OPD几乎所有的有效梯度信号，都来自师生的重叠token，非重叠token对优化的贡献几乎可以忽略不计**。

同时，论文还发现了重叠token的自我强化效应：随着训练推进，学生把更多的概率质量集中到重叠token上，师生的重叠区域会进一步扩大，老师的引导信号也会越来越精准，形成正向循环，训练会越来越稳、效果越来越好。

到这里，我们就能彻底解释「强老师为什么训崩了」：**如果师生的思维模式不匹配，初始重叠率极低，没有足够的有效梯度信号，训练全程停滞，自然不会有任何效果；哪怕老师的benchmark分数再高，它的有效信号也落不到学生能接住的区域里，全是无用功**。

## 落地指南

搞懂了OPD成败的核心，解决训练失败的方案就呼之欲出了——核心就是拉高师生的初始重叠率，补齐思维模式的差距。

论文给出了两个零算法改造、可直接落地的修复策略，能把原本失败的OPD，拉回正轨，甚至突破效果上限。

### 1. Off-policy cold start

**在正式启动OPD训练之前，先用老师生成的轨迹，给学生做一轮有监督微调(SFT)，先把学生的思维模式拉到和老师同频，大幅拉高初始重叠率，再开OPD训练。**

![Off-policy cold start](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSstZa4qbRGrQBic9jqkeXpVl8RZe5UnoDMnj1MkeNDXgA4gWCDlia9k5Ozttcmbhjg36muZFDBcSK7z1CjPbEqPGF9ZMkicMwTgLU/640?wx_fmt=png&from=appmsg)

论文实验证明，这个「SFT冷启动+OPD」的两阶段策略，不仅能把原本失败的OPD救回来，还能稳定超越纯OPD的效果上限，同时大幅降低训练的不稳定性。

P.S. 冷启动策略实验中，SFT阶段用了额外的200K条老师数据，论文没有隔离「数据量增加」和「思维模式对齐」的影响，没法确定效果各有多少提升。

### 2. Teacher-aligned prompt selection

这个方案是从数据侧入手，核心逻辑是：**老师的思维模式，是被它后训练阶段用的Prompt塑造的。如果用老师后训练时用过的Prompt来做OPD，能让学生生成的状态，更贴合老师熟悉的场景，在高概率token上形成更精准的对齐信号，大幅提升训练效果。**

![Teacher-aligned prompt selection](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSuY8njs7dBIxeRbwb1ws7E0qGfwdvEgxDRxiaicyaEJibvK7g5YmMYKxKsq3nKv9otZ7niaX157FPBTkzicBTorXUj2Tr2FvrM4J8WE/640?wx_fmt=png&from=appmsg)

但这里有一个必须注意的坑：**不能全用老师对齐的Prompt，否则学生的熵会严重坍缩**，生成多样性直接崩掉。实际使用时，必须混合一部分分布外(OOD)Prompt，平衡蒸馏效果和生成多样性。

P.S. 教师对齐Prompt策略，一定要控制好OOD Prompt的混合比例，否则很容易把学生的生成多样性训崩，论文里也没有给出最佳的混合比例参考，需要大家根据自己的场景做消融。

## OPD的局限

作者发现，哪怕你把两个条件都满足、两个方案都用对，OPD也有它解不开的固有局限：**OPD引以为傲的token级密集奖励，是有固有代价的——奖励质量会随着轨迹深度的增加，系统性下降；训练的不稳定性，会从序列的尾部开始产生，逐步向前传播，最终带崩整个训练**。

![OPD局限](https://mmbiz.qpic.cn/mmbiz_png/2oicuz5vRaSuMD4EaEqGd4pPkEXicjljdWRjGnSGntXticjEhwZh7zZPBYJuOXsbKwt3oJP29JQZMZoPuibtEibWicMdvO3PbAVSMucG7qVOLJB4g/640?wx_fmt=png&from=appmsg)

如图所示，OPD的效果，在3K-7K的序列长度里达到峰值；超过10K之后，效果就会停滞甚至下降；到15K长度时，训练后期会直接出现重叠率暴跌、熵值飙升的崩溃现象。大白话解释就是：学生写的解题步骤越长，后面的步骤离老师熟悉的路径就越远，老师给的指导就越不准，甚至会给错信号。轨迹越长，老师的奖励信号可靠性越差，最终整个序列的训练都会被带崩。

不过也要客观说明，论文的实验均基于1.5B-7B的中小模型完成，这类模型本身的长文本能力有天然上限；工业级大模型场景下，OPD的长序列天花板到底在哪里、是否有可行的缓解方案，还有待进一步探索。

## 总结

这篇论文给出了OPD一套可预判、可解释、可落地的完整框架。论文的两个条件，拆到最底层，其实就是一句话：**你得能接住，对方得有新东西可给**。

这非常符合直觉和常识。但常识成立不代表它容易满足——随着学生越来越强，同时满足这两个条件的老师越来越难找。真正有新能力的老师，大概率和你的思维模式差太远，overlap ratio从头就上不去；能和你无缝对齐的老师，又往往只是同一条训练管线的放大版，没有任何新东西可教。甜蜜区间，在模型变强的过程中是系统性收窄的。

论文给的冷启动SFT补丁，是在用工程手段强行撑大这个甜蜜区间，但治标不治本。真正能持续产生「学生没见过的新能力」的，只有RL。

所以现在所有走在前面的后训练管线，本质上都是同一个循环：RL生产新知识，OPD把新知识廉价复制给更小的模型，等学生追上老师，再跑一轮RL。

OPD的上限，从来不是OPD本身，是RL能跑多远。

---

参考资料

[1] Rethinking On-Policy Distillation: https://arxiv.org/abs/2604.13016

---

作者：rumor（北航本硕，大模型算法工程师，谷歌开发者专家）

原文链接：https://mp.weixin.qq.com/s/dRGhsyts9GxMJexTnH1JMA
