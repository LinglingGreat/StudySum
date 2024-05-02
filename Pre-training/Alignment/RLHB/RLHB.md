---
title: RLHB
created: 2024-05-02
tags:
  - alignment
  - 在线对齐
type: 论文
papername: The Real, the Better-Aligning Large Language Models with Online Human Behaviors
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 百度
---

## 论文基本信息

标题：[The Real, the Better: Aligning Large Language Models with Online Human Behaviors](https://papers.cool/arxiv/2405.00578)

作者：[Guanying Jiang](https://arxiv.org/search/?searchtype=author&query=Guanying%20Jiang) ; [Lingyong Yan](https://arxiv.org/search/?searchtype=author&query=Lingyong%20Yan) ; [Haibo Shi](https://arxiv.org/search/?searchtype=author&query=Haibo%20Shi) ; [Dawei Yin](https://arxiv.org/search/?searchtype=author&query=Dawei%20Yin)

链接：https://papers.cool/arxiv/2405.00578

代码：

框架图：

![](img/Pasted%20image%2020240502155233.png)

## 背景
这篇论文试图解决的问题是如何将大型语言模型（LLMs）与在线人类行为更好地对齐，以避免生成无益甚至有害的回应。具体来说，论文提出了以下几个关键点：

1. **在线行为对齐**：论文提出了一种新的对齐框架，名为“Reinforcement Learning with Human Behaviors”（RLHB），该框架直接利用在线人类行为来对齐LLMs，而不是依赖预先定义的偏好偏差。
    
2. **训练过程的挑战**：传统的LLMs对齐方法通常需要长时间的训练过程，并且受限于预先设定的偏好信号，这限制了模型适应在线用户多样化偏好的能力。
    
3. **实时在线适应性**：RLHB框架通过生成对抗网络（GAN）的方式，训练生成器（即LLM）以遵循预期的人类行为，同时训练鉴别器来验证查询、回应和人类行为三元组是否来自真实的在线环境。
    
4. **自然语言行为建模**：论文提出了将人类行为以自然语言形式建模，并将其作为条件信息输入到生成指令中，这样做可以充分利用LLM理解和遵循自然语言的能力。
    
5. **持续学习机制**：与RLHF（Reinforcement Learning from Human Feedback）相比，RLHB消除了注释需求，可以推广到各种场景和应用，并且由于其多模型同时训练机制和自然语言形式的行为建模，能够随着人类行为的更新而持续学习。
    
6. **实验验证**：通过一系列实验，论文验证了RLHB框架的有效性，包括人类评估和自动评估（使用GPT4）。
    

总结来说，这篇论文旨在通过直接利用在线人类行为来改进LLMs的对齐过程，使其能够更好地适应和响应用户的多样化需求。


## 相关研究
论文中提到了与大型语言模型（LLMs）对齐相关的多项研究，这些研究主要集中在以下几个方面：

1. **强化学习方法**：使用强化学习方法来对齐LLMs，如Stiennon等人（2020）和Ouyang等人（2022）的工作。
    
2. **排名学习方法**：如RRHF（Yuan等人，2023b）、DPO（Rafailov等人，2023）、PRO（Song等人，2023）和ΨPO（Azar等人，2023）等。
    
3. **人类注释和原则**：研究如何使用人类注释和原则来对齐LLMs，包括不同粒度的注释（Wu等人，2023）、不同来源的融合（Zeng等人，2023；Rame等人，2023）以及多步骤过程（Lightman等人，2023；Uesato等人，2022；Yuan等人，2023a；Luo等人，2023）。
    
4. **AI辅助**：利用其他强大的LLMs（例如GPT-4）的反馈作为对齐指导（Chang等人，2023），或者通过蒸馏这些强大LLMs的反馈到较小的奖励模型中（Bai等人，2023；Lee等人，2023；Tunstall等人，2023；Yang等人，2023）。
    
5. **在线演示**：使用在线演示来对齐语言模型，如通过逆强化学习（IRL）直接模拟人类偏好（Casper等人，2023；Ji等人，2023）。
    
6. **信息检索和推荐系统**：研究如何利用在线匿名人类行为来提高内容质量和用户体验（Joachims等人，2017；Mitra等人，2018；Huang等人，2020；Hu等人，2008；Liu等人，2010；Zhao等人，2018；Xie等人，2021；Wu等人，2022）。
    
7. **其他对齐方法**：包括使用规则基础（OpenAI，2023）和原则遵循（Sun等人，2023）的奖励模型，以及使用自我批评作为条件信息（Dong等人，2023b；Hu等人，2023）。
    

这些研究为LLMs的对齐提供了不同的方法和视角，包括人工注释、AI辅助、在线行为数据等，旨在提高LLMs的生成质量和与人类偏好的一致性。论文提出的RLHB框架是在这些现有研究的基础上，尝试直接利用在线人类行为来对齐LLMs，以适应在线用户多样化的偏好。



## 核心亮点
论文提出了一个名为“强化学习与人类行为”（Reinforcement Learning with Human Behaviors，简称RLHB）的框架来解决大型语言模型（LLMs）与在线人类行为对齐的问题。具体解决方案包括以下几个关键步骤：

1. **在线人类行为收集**：首先，论文展示了如何在搜索引擎中收集用户对LLM生成的回答的在线行为，包括页面浏览量、点击、点赞或不点赞等。
    
2. **行为信号处理**：将收集到的在线人类行为匿名化，并处理成数值或自然语言形式，以便更好地与LLMs的对齐。我们通过 log(1 + x) 平滑指标并将每个指标离散成 N 个相等的部分。在强化学习训练期间，我们执行奖励塑造（Likes - Dislikes）/（PV + Clicks），将多头奖励组合成整体标量。

3. **RLHBC**：一个Naive的方法是直接训练一个预测用户行为bij的反馈模拟器，基于query和LLM的response。但是由于现实中很难收集到基于同一个问题、不同的答案下的人类反馈行为，因此，可以构建一个具有cross entropy损失的multi-head pointwise classifier。这个模拟器作为RLHF中的reward model，这种方法称为RLHBC(Reinforcement Learning with Human Behavior through Classifier)。

![](img/Pasted%20image%2020240502164112.png)
    
3. **RLHB框架**：提出了RLHB框架，其中包括一个目标LLM作为生成器，另一个辅助LLM作为鉴别器。鉴别器的任务是判断查询、回答和人类行为三元组是否来自真实的在线环境。生成器则需要根据指定的人类行为生成回答，使得三元组尽可能真实，以混淆鉴别器。这里面包括actor, critic, discriminator的训练，为了防止判别器过度拟合，使其更加鲁棒，在判别器更新中，我们bootstrap一定比例的highly-rewarded demonstrations作为'fake'的online demonstrations。

![](img/Pasted%20image%2020240502170702.png)

![](img/Pasted%20image%2020240502170714.png)


    
4. **自然语言行为建模**：利用LLM理解自然语言的能力，将人类行为以自然语言形式表达，并将其作为条件信息输入到生成指令中。
    
5. **多模型联合训练**：通过对抗性训练，同时训练生成器和鉴别器。生成器在给定查询和最偏好的行为信号输入下，生成与人类行为相匹配的回答。
    
8. **在线部署**：在推理阶段，经过良好对齐的生成器可以直接在线部署，接受用户的查询和最偏好的行为信号作为输入。
    

通过这些步骤，论文提出的RLHB框架能够实现LLMs与在线人类行为的有效对齐，生成更符合用户偏好的回答，并且能够适应用户偏好的动态变化。

![](img/Pasted%20image%2020240502163457.png)


## 实验
论文中进行了多项实验来评估提出的RLHB框架的有效性，具体实验包括：

1. **数据收集**：
- 用户行为数据：从百度搜索引擎中收集了约100k个<查询，回答，反馈>三元组作为在线人类行为数据。
- 用户偏好数据：参考InstructGPT收集了用于reward model的用户偏好数据，随机采样query，然后用LLM生成多个回答，经验丰富的标注员会标注出每2个回答的偏好标签。
    
2. **模型设置**：
- 使用了一个内部的13B参数规模的LLM，用了数百万的平台数据微调，作为基线模型
- 奖励模型（RM）：用户偏好数据训练得到
- 分类器模型（CM）：RLHBC中的分类器，用数字形式拟合人类偏好
- 鉴别器模型（DM）：warmup一个判别器，用线上用户行为数据作为real，随机替换其中的反馈作为fake，训练得到。
    
3. **评估指标**：采用了三种类型的评估指标，包括GPT4的评估、人类评估系统（质量得分和满意度得分）。我们利用RL指标来评估模型训练的稳定性，并利用LLM指标来评估LLM能力。
- **GPT4评估**：使用GPT4对不同模型生成的相同查询的回答进行排名，得到Win-Tie-Loss（WTL）结果。
- **人类评估**：通过质量得分和满意度得分对模型生成的回答进行评估，关注优秀和良好回答的提升。
    
6. **模型训练指标**：研究了超参数设置对模型性能的影响，包括RLHF、RLHBC和RLHB模型的训练指标。
    
7. **消融研究**：进行了多个消融实验，包括研究伪造数据比例对鉴别器的影响、不同的rollout数量和批量大小对演员（即LLM策略代理）的影响。
    
8. **在线行为对齐**：评估了基于在线人类行为的LLM对齐是否能够接近基于离线注释偏好的对齐。
    
9. **预定义偏好对齐的改进**：评估了在线行为对齐是否能够进一步提升已经通过离线偏好对齐的LLM。
    
10. **直接对齐与基于分类器的对齐**：比较了直接使用在线人类行为对齐（RLHB）与通过训练一个分类器来预测人类行为再用于对齐（RLHBC）的方法。
    

这些实验全面地评估了RLHB框架在不同方面的表现，包括其在模仿在线人类行为、提升回答质量、用户满意度以及模型训练稳定性等方面的表现。通过这些实验，论文证明了RLHB框架能够有效地对齐LLMs以匹配在线人类行为，并且可以作为一个持续学习和适应用户偏好的在线对齐方法。

Q1 Whether the generative model can be directly aligned on real online human behaviors? 

Q2 Whether the predefined preferences alignment can be further improved by online alignment?

![](img/Pasted%20image%2020240502165833.png)

![](img/Pasted%20image%2020240502170143.png)

![](img/Pasted%20image%2020240502170152.png)

![](img/Pasted%20image%2020240502170224.png)

boostrap比例达到50%时候，判别器急剧崩溃，奖励接近1.0

rollouts从1增加到4到6，稳定性显著提升，尤其是mean returns。

batch size从16到8，除了波动性较大之外，训练效果的差异并不明显。

## 未来方向


## 主要收获


## 参考资料
