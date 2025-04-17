---
title: Tulu
created: 2025-01-06
tags:
  - 论文
  - alignment
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - AllenAI
---

## 论文基本信息

标题：TÜLU 3: Pushing Frontiers in Open Language Model Post-Training

作者：allenai团队

链接：https://arxiv.org/abs/2411.15124

亮点：报告详细，并且开源了用于训练的所有数据集、用于数据管理和评估的强大工具包、训练代码、模型。
- Tulu 3 405B: https://hf.co/allenai/Llama-3.1-Tulu-3-405B
- TÜLU 3 70B https://hf.co/allenai/Llama-3.1-Tulu-3-70B  
- TÜLU 3 8B https://hf.co/allenai/Llama-3.1-Tulu-3-8B  
- TÜLU 3 DATA https://hf.co/collections/allenai/tulu-3-datasets673b8df14442393f7213f372  
- TÜLU 3 Code https://github.com/allenai/open-instruct  
- TÜLU 3 EVAL https://github.com/allenai/olmes  
- Demo https://playground.allenai.org/

模型和代码，模型包括SFT后的、DPO后的以及最终版本的，还有RM模型。

![](img/Pasted%20image%2020250416173522.png)

数据集，不管是模型用了的还是没用的，都在这里了。

![](img/Pasted%20image%2020250416173538.png)

这是把能开源的全都开源了！

核心框架：

![](img/Pasted%20image%2020250416173744.png)

包括仔细的数据管理、严格的实验和评估、创新的训练方法，改进的训练infra。这些也是Tulu3成功的关键。

团队先是确定了一组训练后需要改进的核心技能（例如，推理、数学、编码、安全、精确指令跟随、知识回忆等），并建立了一个评估框架，以建立明确的绩效目标，并指导模型改进。

![](img/Pasted%20image%2020250416180051.png)

模型的训练包括SFT、DPO、RLVR。通过识别技能缺陷并完善数据组合、方法和参数，确保核心技能在整个训练过程中的平衡表现。

模型评测结果概览：

![](img/Pasted%20image%2020250416175913.png)

![](img/Pasted%20image%2020250417113830.png)

![](img/Pasted%20image%2020250417115352.png)

![](img/Pasted%20image%2020250417115403.png)
## 核心亮点

上述简单过了一遍Tulu3的主要贡献点、核心框架、评测结果等，接下来就是详细看看核心框架部分做了哪些工作。

主要有四个步骤:

1. 数据管理：构建多样化、高质量的Prompt，主要来自开源数据集和自己合成。
2. 有监督微调：通过评估框架的指导和仔细的实验，确定最终的 SFT 数据和训练超参数。
3. 偏好优化：和SFT类似，通过评估框架和仔细的实验确定数据配比。
4. 强化学习：选择具有可验证结果的任务，例如数学问题，使用可验证的奖励而不是传统的奖励模型来强化训练。

### Prompt构建

![](img/Pasted%20image%2020250119145934.png)

选取了几个公开的数据集，包括:WildChat, OpenAssistant, NoRobots, FLAN v2等，用以确保多样性。除此之外，为了确保一些特定能力，还用了OpenMathInstrct, Evol-CodeAlpaca等数学代码相关的数据集。 基本上这一步就是一个大大的数据收集工。质量好点的开源数据都给用上了。

除此之外还做了一些数据合成，借助的是所谓的persona-driven的方法，通过让LLM来扮演不同角色从而生成不同的指令。

最后还做了Prompt的去污染，防止污染测试集。

### 有监督微调

主要也是对已有公开数据的过滤清洗，以及对于没有response的数据，利用GPT4o来进行合成。

![](https://picx.zhimg.com/v2-bb23336bd2d80587b167ad78901e3d29_1440w.jpg)

对SFT数据的消融，比较有趣的是Persona数据确实改善了对应技能如Math/Coding上的能力。(不过其实也没有很显著)

- 多样化对话数据(WildChat)，对大多数技能有积极影响，特别提升Alpaca Eval性能。
- 安全性 SFT 数据通常与其他数据集正交。

![](https://pic1.zhimg.com/v2-5b069139e246486b2eb4d1f2f51a9bea_1440w.jpg)

一个比较抽象的发现是: 不同的prompt template看起来对结果的影响还不小....

此阶段学习率5e-6，只需要训两轮，更多轮数不能带来提升。

![](img/Pasted%20image%2020250119150102.png)

**Trick：Batch Aggregation**

TÜLU 3注意到Open-Instruct框架训练的SFT模型与在其他环境(如TPU)上训练的模型之间存在性能差距。这个问题主要是由于Transformers中loss aggregation的一个问题：在不考虑梯度累积或分布式训练设置的情况下对padding tokens的损失进行平均。

报告用一个例子来说明这个问题。假设批次中有两个样本，分别有 n1 、 n2 个non-padding tokens和 m1 、 m2 个填充标记。如果同时将两个样本输入默认的Transformers forward pass:

$L=(l_{n1}+l_{n2})/(n1+n2)$

然而，如果应用gradient accumulation分别输入两个样本，计算损失，然后除以2:

$L=(l_{n1}/n1+l_{n2}/n2)/2$

第二种情况下平等地对待每个样本,，而在第一种情况下平等地对待每个token。因此改变梯度累积可能会由于有效地改变样本权重而对性能产生重大影响。由于跨设备平均,分布式训练中也会出现类似的问题。

所以TÜLU 3在训练时普遍选择使用求和损失（sum loss）而不是平均损失（mean loss）。即通过简单地移除上述方程中的分母，同时调整学习率。这使得所有token被赋予相同的权重。TÜLU 3通过使用各种学习率、训练轮数和损失类型在TÜLU 2 SFT混合数据集上微调Llama 3.0来验证各种设置的性能。最终发现使用lr = 5e-6的sum loss效果最好。TÜLU 3还发现更长时间的训练并没有带来进一步的改进，因此确定使用2个训练epoch。
### 偏好优化

这块在技术上无甚可说，主要也是讲讲数据: 通过从一个model pool中选择模型来生成多样的回答(模仿Ultrafeedback)，并且也做了on-policy的数据采样，大致保持了on-policy和off-policy的数据量1:1。最后使用GPT4o来进行标注。(本GPT4o企业级用户表示: 这看起来感觉并不是一个很强的做法。。)

![](img/Pasted%20image%2020250119150825.png)

最后得出了几个结论:

1. Prompts的多样性大大影响了DPO的效果。(SFT，DPO的Scaling Law)
2. 只增加数量，但是Prompt的多样性不增加，其实模型效果是会退化的。
3. DPO阶段，复用SFT阶段的Prompt，会带来一定收益，但还是采用新的Prompt效果更佳。
4. On-policy Data（模型采样出来的pair）相比off-policy data效果更好。
5. GPT-4o-2024-08-06是标注能力最强的模型 (用它标注的结果做DPO效果最佳，和Llama405b打平)。这里还把Llama 3.1 405b拿出来试了下，看来4o的参数效率还是很领先的。

![](img/Pasted%20image%2020250119151148.png)



本阶段最后选择的学习率是5e-7，并且只需要训练一轮。另外作者用的是Length-normalized DPO，顾名思义，给对数概率和除了个权重。TÜLU 3场景中，不同RL算法的实验结论是，length-normalized DPO效果最好，SimPO甚至性能不如SFT-base。

![](img/Pasted%20image%2020250119150632.png)

![](img/Pasted%20image%2020250119151214.png)

![](img/Pasted%20image%2020250119150651.png)



### 强化学习

RLVR（RL with verifiable rewards）

![](https://pic4.zhimg.com/v2-420ceb691fc2ee3e98c252d8469b3bdd_1440w.jpg)

![](img/Pasted%20image%2020250119151321.png)

直接基于GroundTruth来判断答案是否正确，然后应用PPO来进行训练。其实就是基于Rule-Based RM做RL的另一种说法。不同于DeepSeek-V3和Qwen2.5采取的GRPO，RLVR的算法采取了PPO。PPO需要value model，但reward model目前是一个verifier，所以TÜLU 3使用General RM来初始化value model。

发现:

1. 这样可以直接在目标领域(比如数学)改善效果，其实这也是一个趋势了，代码生成，数学推理这些可自动验证的领域后面应该都会跟上。
2. Value Model最好从一个通用的RM上去初始化。
3. 用RM产生的分数反而会产生噪音。(所以在能用规则验证的地方，还是不要用模型了吧)

### 其它发现

1. 在线的DPO没生效。
2. 拒绝采样也没怎么生效。

### 尾声:

本文看起来朴实无华，但对于整个后训练链路的探索，还是很有价值的。不管是RLVR，还是在线DPO的失败，其实应该都体现的是在本文中RM的失败，RM的过拟合，噪音，Hacking等问题，依然是值得大家去警惕的。

## 实验



## 未来方向



## 主要收获


## 参考资料

[TÜLU 3: 拒绝RM的后训练技术总结](https://zhuanlan.zhihu.com/p/8589852586)

