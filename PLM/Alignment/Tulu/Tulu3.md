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

Tulu3先是确定了一组训练后需要改进的核心技能（例如，推理、数学、编码、安全、精确指令跟随、知识回忆等），并建立了一个评估框架，以建立明确的绩效目标，并指导模型改进。

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

训练数据的**多样性**对于模型的泛化、避免模型遗忘以及使模型对不常见的输入具有鲁棒性至关重要，据此选取了几个公开的数据集，包括:
- WildChat（真实的用户和模型交互的数据）
- OpenAssistant（由志愿者创建，用于一般聊天）
- NoRobots（由专家为广泛的开放式类别进行标注）
- FLAN v2（经典 NLP 任务的大型汇总）
-  UltraFeedback 的去污染子集（这里面包括FalseQA、UltraChat、Evol-Instruct、FLAN v2，在早期研究中显示出对通用偏好调整有很强的性能表现）

 根据早期研究，一些功能，**如复杂推理、编码和精确的指令跟随，受益于混合额外的数据**。因此为了确保一些特定能力，还包括了以下数据集：用于数学推理的 OpenMathInstruct和 NuminaMath，用于编码的 Evol-CodeAlpaca，用于精确教学跟踪的 Daring-Anteater 的子集，用于多语言的 Aya，用于科学文献理解的 SciRIFF和 TableGPT处理与表相关的任务。

此外还做了一些数据合成，采取论文Scaling synthetic data creation with 1,000,000,000 personas.中的**persona-driven的方法来合成**。关键思想是使用不同的角色（例如，“专注于神经网络的机器学习研究人员”）和数据合成提示（例如，“创建编码问题”）来引导 LLM 以相应的视角合成数据。具体来说，以来自 Persona Hub 的 ∼250K 角色为条件，以生成针对特定技能的提示，例如**精确指令遵循、数学和编码**。
- 精确指令遵循：涵盖 IFEval 基准测试中定义的 25 种不同的约束类型。具体地说，首先为每个约束（例如，字数）手动编写 1-2 条示例指令，总共生成33个可验证指令作为种子提示。然后，使用GPT-4o在给定数据合成提示、人设和单个可验证指令示例的情况下生成新指令。总共收集了29,980个可验证指令-响应对，称之为If-Persona-Sft。还生成另一种针对受约束指令遵循的提示，通过从Tülu 2 SFT混合中随机抽样指令，并将其与Zhou等人（2023年，Tablegpt）的分类法中的约束相结合。这个集合称为IF-augmented。这些提示仅用于DPO和RLVR阶段。
- 数学和代码：数学问题包括需要高级数学技能的问题以及小学问题。编码包括初级到中级程序员可以解决的 Python 编程问题。zero-shot提示 GPT-4o 生成独特且特定于给定角色输入的问题。然后使用 GPT-4o 生成多步数学解决方案，并使用 claude-3-5-sonnet 生成 Python 程序。总共收集了 ∼220K 和 35K 实例用于数学推理和编码。
- 不合规和安全：策划了一组模型不应遵守的不合规提示，以及与安全相关的直接和对抗性提示，覆盖良性和有害情况。不合规提示是根据 Brahman 等人（2024 年）的上下文不合规分类法获得的，涵盖多个类别，包括不完整、不支持、不确定和人性化请求（以及不安全请求）。安全相关提示是在合成对抗性提示、合成普通（直接）请求、真实世界的用户-LLM 交互 （In-The-Wild） 和精选的注释者编写的示例中精心挑选的，以最大限度地提高覆盖范围、多样性和平衡性。

最后还做了Prompt的去污染，防止污染测试集，主要采取文本匹配（ngram）的方法。

![](img/Pasted%20image%2020250417193826.png)



### 有监督微调

首先是对数据集的过滤清洗、重新生成
- 对于有response的提示（比如公开数据集），如果response是人类编写的或者是前沿模型（比如GPT-4o）生成的，那么就保留
- 过滤调空回复，或者包含模型信息、开发者信息的回复
- 对于弱模型生成的回复，或者没有回复的提示（比如persona prompts），使用GPT-4o生成回复。

为了设计最终的 SFT 数据组合，首先构建了特定于技能的数据混合和模型，保留导致单个技能最佳表现的混合，而忽略其他评估。这样做是为了在给定设置下近似每个评估的上限。

然后，将这些混合物混合在一起，创建了最初的 Tülu 3 预览混合物。然后继续迭代混合物，添加或删除数据集以提高落后技能的性能，对评估进行净化，并对特别大的数据集进行下采样。

![](img/Pasted%20image%2020250418150727.png)

最终的SFT数据混合结果如表9所示，超越了Tulu 2以及其他开源的SFT模型。

![](img/Pasted%20image%2020250418150901.png)

表10是一些SFT数据的关键实验。
- 多样化的聊天数据（WildChat），对大多数技能都有正向影响，特别是Alpaca Eval
- 安全性数据通常与其他数据集正交，去掉它基本上只会影响安全得分。另外添加对比提示（例如 CoCoNot 中的提示）有助于防止模型过度拒绝安全提示。
- 删除 Persona 数据集后，HumanEval（+）、GSM8K、MATH 和 IFEval 的性能下降
- 特定的数学SFT 数据（前面提到的针对特定技能会混合一些额外的数据）大大提高了 GSM8K 和 MATH的性能。
- （图4）SFT 数据越多，模型平均性能会不断改进。有趣的是，TruthfulQA 性能实际上会随着混合数据量的增加而下降。

![](img/Pasted%20image%2020250418151852.png)

（表14）还尝试了SFT阶段采用不同的随机种子（真·玄学调参），最终选择效果最好的SFT模型。这里的model soup指的是不同模型权重合并后得到的模型，Tulu3采取的是线性合并，mergekit工具。

SFT阶段的超参数如下，最大长度居然只有4096。

![](img/Pasted%20image%2020250418151939.png)


不同的prompt template对结果的影响：

![](img/Pasted%20image%2020250418152533.png)


**Batch Aggregation**

TÜLU 3注意到Open-Instruct框架训练的SFT模型与在其他环境(如TPU)上训练的模型之间存在性能差距。这个问题主要是由于Transformers中loss aggregation的一个问题：在不考虑梯度累积或分布式训练设置的情况下对padding tokens的损失进行平均。

用一个例子来说明这个问题。假设批次中有两个样本，分别有 n1 、 n2 个non-padding tokens和 m1 、 m2 个padding tokens。如果同时将两个样本输入默认的Transformers forward pass，会得到:

$L=(l_{n1}+l_{n2})/(n1+n2)$

然而，如果应用gradient accumulation分别输入两个样本，计算损失，然后除以2，得到:

$L=(l_{n1}/n1+l_{n2}/n2)/2$

第二种情况下平等地对待每个样本，而在第一种情况下平等地对待每个token。因此改变梯度累积可能会由于有效地改变样本权重而对性能产生重大影响。由于跨设备平均,分布式训练中也会出现类似的问题。

所以TÜLU 3在训练时普遍选择使用求和损失（sum loss）而不是平均损失（mean loss）。即通过简单地移除上述方程中的分母，同时调整学习率。这使得所有token被赋予相同的权重。TÜLU 3通过使用各种学习率、训练轮数和损失类型在TÜLU 2 SFT混合数据集上微调Llama 3.0来验证各种设置的性能。最终发现使用lr = 5e-6的sum loss效果最好。TÜLU 3还发现更长时间的训练并没有带来进一步的改进，因此确定使用2个训练epoch。

![](img/Pasted%20image%2020250418160352.png)


### 偏好优化

首先是方法上，标准的DPO长这样：

![](img/Pasted%20image%2020250418162058.png)

而Tulu3使用了length-normalized DPO，在标准DPO基础上，对数概率对长度进行归一化，这有助于减轻人类和模型偏好中常见的长度偏差。

![](img/Pasted%20image%2020250418162129.png)


然后是数据的构造，通过调整和改进 UltraFeedback pipeline来产生偏好数据。包括3个步骤：prompt选择，模型池中的模型生成回答，用LLM-as-a-judge的方法来做偏好标注，形成偏好数据集。
- prompt选择：给定表 7 （prompt数据合集）中的一组提示，精选了包括 SFT 期间使用的提示，以及从相同来源进行二次采样但未用于 SFT 的提示。还包括来自其他来源的提示，例如没有 TruthfulQA 实例的 Ultrafeedback 版本，或者通过向提示添加新的 IF 约束。
- 回复生成：对于给定的提示，从模型池（包括开源模型、闭源模型）中随机抽样四个模型以生成响应。也生成了一批on-policy数据，其中一个回复是on-policy model生成的，另一个回复是off-policy models生成的。
- 偏好标注：用GPT-4o-2024-0806打分，从帮助性、遵循指示、诚实和真实几个维度，1-5分。将评分最高的响应作为chosen，并从具有较低平均值的响应中随机抽样作为reject。

![](img/Pasted%20image%2020250418162904.png)



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

