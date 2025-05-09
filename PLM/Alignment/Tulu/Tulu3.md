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

#### 数据和实验

首先是对数据集的过滤清洗、重新生成
- 对于有response的提示（比如公开数据集），如果response是人类编写的或者是前沿模型（比如GPT-4o）生成的，那么就保留
- 过滤空回复，或者包含模型信息、开发者信息的回复
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

#### 训练参数

![](img/Pasted%20image%2020250418151852.png)

（表14）还尝试了SFT阶段采用不同的随机种子（真·玄学调参），最终选择效果最好的SFT模型。这里的model soup指的是不同模型权重合并后得到的模型，Tulu3采取的是线性合并，mergekit工具。

SFT阶段的超参数如下，最大长度居然只有4096。

![](img/Pasted%20image%2020250418151939.png)


不同的prompt template对结果的影响：

![](img/Pasted%20image%2020250418152533.png)


#### Batch Aggregation

TÜLU 3注意到Open-Instruct框架训练的SFT模型与在其他环境(如TPU)上训练的模型之间存在性能差距。这个问题主要是由于Transformers中loss aggregation的一个问题：在不考虑梯度累积或分布式训练设置的情况下对padding tokens的损失进行平均。

用一个例子来说明这个问题。假设批次中有两个样本，分别有 n1 、 n2 个non-padding tokens和 m1 、 m2 个padding tokens。如果同时将两个样本输入默认的Transformers forward pass，会得到:

$L=(l_{n1}+l_{n2})/(n1+n2)$

然而，如果应用gradient accumulation分别输入两个样本，计算损失，然后除以2，得到:

$L=(l_{n1}/n1+l_{n2}/n2)/2$

第二种情况下平等地对待每个样本，而在第一种情况下平等地对待每个token。因此改变梯度累积可能会由于有效地改变样本权重而对性能产生重大影响。由于跨设备平均,分布式训练中也会出现类似的问题。

所以TÜLU 3在训练时普遍选择使用求和损失（sum loss）而不是平均损失（mean loss）。即通过简单地移除上述方程中的分母，同时调整学习率。这使得所有token被赋予相同的权重。TÜLU 3通过使用各种学习率、训练轮数和损失类型在TÜLU 2 SFT混合数据集上微调Llama 3.0来验证各种设置的性能。最终发现使用lr = 5e-6的sum loss效果最好。TÜLU 3还发现更长时间的训练并没有带来进一步的改进，因此确定使用2个训练epoch。

![](img/Pasted%20image%2020250418160352.png)


### 偏好优化

#### 数据构造和消融实验

首先是数据的构造，通过调整和改进 UltraFeedback pipeline来产生偏好数据。包括3个步骤：prompt选择，模型池中的模型生成回答，用LLM-as-a-judge的方法来做偏好标注，形成偏好数据集。
- prompt选择：给定表 7 （prompt数据合集）中的一组提示，精选了包括 SFT 期间使用的提示，以及从相同来源进行二次采样但未用于 SFT 的提示。还包括来自其他来源的提示，例如没有 TruthfulQA 实例的 Ultrafeedback 版本，或者通过向提示添加新的 IF 约束。
- 回复生成：对于给定的提示，从模型池（包括开源模型、闭源模型）中随机抽样四个模型以生成响应。也生成了一批on-policy数据，其中一个回复是on-policy model生成的，另一个回复是off-policy models生成的。
- 偏好标注：用GPT-4o-2024-0806打分，从帮助性、遵循指示、诚实和真实几个维度，1-5分。将评分最高的响应作为chosen，并从具有较低平均值的响应中随机抽样作为reject。

![](img/Pasted%20image%2020250418162904.png)

然后做数据消融实验，表15是最终的数据混合情况，表16是包含/排除某些数据集对性能的影响情况。8B模型最终用了271k的偏好数据，70B模型用了334k的偏好数据。

![](img/Pasted%20image%2020250418180730.png)

![](img/Pasted%20image%2020250418180757.png)

最后得出了几个结论:
1. Prompts的多样性很重要。增加唯一提示(Unique Prompts)的数量可以提高下游 DPO 性能，但是增加重复提示的数量不一定会提高下游DPO性能，建议花更多精力在收集唯一的提示数据和合适的数据混合上。
2. 重复使用 SFT阶段中的提示会带来收益，但使用SFT阶段未使用的提示效果更好。在Tulu3中，把两者结合效果是最好的。
3. On-policy数据（偏好对中肯定有一个是从当前模型采样生成的回复）相比off-policy 数据效果更好，两者结合效果最好（把on-policy回复和off-policy回复放一起，平等对待，通过打分选取pair对）。
4. GPT-4o-2024-08-06做裁判的效果最好。Tulu3测试了GPT-4（GPT-4-turbo-2024-04-09、GPT-4o-2024-08-06、gpt-4o-mini-2024-07-18）和 Llama 3.1（70B 和 405B）。GPT-4o、Llama 3.1 405B 和 GPT-4 Turbo的表现类似，GPT-4o略微领先。
5. Tulu3的数据混合性能超过了UltraFeedback和Helpsteer2，论文中将其归因于UltraFeedback回复中使用的模型性能普遍低于Tulu3中普遍使用的70B模型。
6. 在针对指令遵循、编码和数学技能的三个角色偏好数据集中，只有 Tülu 3 Persona IF 提高了平均评估分数和目标 IFEval 分数，其他两个数据没有对评估产生影响。
7. 添加由 WildChat 提示和使用Tulu3的合成偏好数据pipeline获得的pair对组成的偏好数据通常会提高 DPO 性能。将 SFT 训练期间看到的 WildChat 提示添加到 DPO 组合中，比将未使用的提示与重复使用的 WildChat 提示相结合，平均性能更好。
8. 比较了使用几个数据集的原始偏好对，和Tulu3的合成偏好数据pipeline相比，哪个效果更好。实验了Helpsteer2、Ultrafeedback 和 MultiPref这几个数据集，Tulu3的合成偏好数据pipeline生成的数据DPO性能更好。
9. 指令遵循数据集。IF-persona 偏好数据显著提高了 IFEval 分数，同时对平均性能的损害最小。IF-augmented数据集仅将 IFEval 性能提高了 1 分，同时也略微损害了平均性能。将 IF-persona 与 IF-augmented-verified 相结合可获得最佳的 IFEval 性能，但平均值略低。
	1. Persona IF：将If-Persona-Sft变成偏好数据，使用GPT-4o改写prompt中的约束，并生成新回复，新回复不符合原始prompt中的全部约束，将新回复作为rejected，和原始回复组成pair对。
	2. IF-augmented：前面prompt构建部分已经说过构造方法了，(chosen, rejected)来自合成偏好数据pipeline。通过约束验证函数只保留chosen符合约束的数据，得到IF-augmented-verified.
	3. WildChat IF：用GPT-4从WildChat筛选出那些prompt中含约束的数据。

![](img/Pasted%20image%2020250419100851.png)

![](img/Pasted%20image%2020250419102355.png)

![](img/Pasted%20image%2020250419103305.png)

![](img/Pasted%20image%2020250419105321.png)

#### 偏好优化算法和超参数

Tulu3使用早期的 SFT 检查点和 UltraFeedback 数据集做了算法和超参数的消融选择。尝试了DPO 、 SimPO 和length-normalized DPO，根据实验效果最终选择了length-normalized DPO。

标准的DPO长这样：

![](img/Pasted%20image%2020250418162058.png)

length-normalized DPO在标准DPO基础上，对数概率对长度进行归一化，这有助于减轻人类和模型偏好中常见的长度偏差。

![](img/Pasted%20image%2020250418162129.png)

顺便看看SimPO的公式，虽然Tulu3的实验中看SimPO的效果很差，还不如SFT Base.

![](img/Pasted%20image%2020250419110110.png)

在此基础了做了超参数的选择，降低了 70B 训练的学习率并增加了批量大小，因为在对较大的模型进行 SFT 时，降低学习率并增加批量大小是很常见的。

不同的数据混合下，最佳的学习率是有差异的，一个是2.0 × 10-7，一个是5.0 × 10-7。

![](img/Pasted%20image%2020250419105928.png)

Tulu3还尝试了PPO，用DPO的早期偏好数据混合训练RM，保持DPO和PPO使用相同的prompt数据混合，没怎么调参数，也只训练了一次RM。训练结果是PPO和DPO性能相似，但是PPO需要更多的计算资源。鉴于资源有限且RM不好评估，因此Tulu3还是使用了DPO训练，在RLVR中会使用PPO。

![](img/Pasted%20image%2020250419110939.png)

![](img/Pasted%20image%2020250419111210.png)

另外，在DPO训练过程中，缓存DPO Log Probs，以及对Chosen，Rejected序列进行Separate Forward，会降低GPU显存的占用。

### 强化学习

RLVR（Reinforcement Learning with Verifiable Rewards） 利用了现有的 RLHF 目标，但用验证函数取代了奖励模型，如图 18 上所示。当应用于具有可验证答案的领域时，例如数学和可验证的指令遵循，RLVR 展示了对 GSM8K 等基准测试的针对性改进，同时保持其他任务的性能。

![](img/Pasted%20image%2020250419113249.png)

![](img/Pasted%20image%2020250119151321.png)

直接基于Rule-Based RM，答案正确才有奖励得分，然后应用PPO来进行训练。

训练数据集采用GSM8K, MATH, IFEval（表22），将数据集合并得到30,000个prompts及其ground truth。训练细节：
- 使用General RM来初始化value model。
- 禁用Dropout
- 使用 SFT 数据集进行训练并在 epoch 之间shuffle，Tulu3训练了100, 000/7, 473 ≈ 13 epochs，并在最后一个epoch中，每40-100步跑一次评估，选择在development evaluation set上评估结果最好的checkpoint。
- 对于没有EOS token结束的response给予惩罚
- 对Advantage做标准化

做了超参数选择、RM初始化模型对比、RM模型+可验证奖励、从更弱的SFT模型初始化训练等实验。

![](img/Pasted%20image%2020250419114825.png)

![](img/Pasted%20image%2020250419115026.png)

![](img/Pasted%20image%2020250419115043.png)


发现：
1. RLVR 可以提高目标领域中的性能。
2. 从通用的RM上去初始化Value Model，比从SFT模型初始化好。
3. 仅使用可验证的奖励优于使用奖励模型中的分数。使用 RM 的分数进行可验证奖励的训练似乎会引入更多噪音，尤其是在平均分数中。
4. 从 SFT 和 DPO 开始可以带来相同水平的可验证奖励，但与从 DPO 模型开始相比，从 SFT 模型开始会产生更大的 KL。因为 SFT 模型在 GSM8K 上的表现远不及 DPO 模型。然而，从更强的模型开始通常会带来更好的测试集性能。
5. 更大的KL偏离通常会导致更低的平均分数。

此外，采取了异步训练方式，使得模型推理的时候同时进行模型训练，减少GPU空闲时间。

最终采用DPO模型作为初始化模型训练，训练后提升了MATH, GSM8k, and IFEval的性能。

![](img/Pasted%20image%2020250419121229.png)



### 讨论
将上述流程应用到405B模型上，训练需要32个节点，因此通信、速度等也是一个挑战。

由于405B模型经过 SFT 和 DPO 训练就达到了 GSM8K性能的饱和，以及 IFEval 数据在初始 RLVR 运行中没有太大帮助。因此，对于 Tülu 3 405B RLVR，只使用了 MATH训练集。只经过 25 个 RLVR 训练步骤，MATH 性能就提高了 5 分以上，并且随着训练的增加而继续提高。

其他发现
1. 在线的DPO没生效。估计是RM没训练好。
2. 拒绝采样（生成n个回复，用LLM排名，不断迭代）也没怎么生效。比较依赖于强大的judges。


## 未来方向

1. 长上下文和多轮次数据。Tulu3的训练数据长度都很短，基本上低于2048 tokens，在长上下文成为趋势的现在显得太短了。
2. 多语言训练。Tulu3基本上只考虑了英文，多语言的训练涉及到跨语言对齐、数据平衡等。
3. 工具使用和Agent。训练模型去使用工具是一个很自然的事情，你不能期待依赖模型权重就能够做到所有事情。

## 主要收获

论文中还花了比较多的篇幅介绍Tulu3的评估框架，包括使用了哪些任务/数据集做评估，评估的方式（zero-shot/few-shot/cot，prompt等），这部分内容因为我不太感兴趣就没仔细看了。有感兴趣的可以自行去看论文。

我觉得这篇论文值得我们学习的就是把所有能开源的东西都开源了，数据不一定能直接拿来用去训练，但作为prompt来源是没问题的，不需要再去大力搜集各种prompt了。论文也写的很详细，实验做的很仔细。包括PPO、在线DPO的失败，也毫不避讳地说明了RM训练的失败，虽然是2024年的论文（提交了很多个arxiv版本，最早是2024年11月，最新一版是2025年4月份的），但目前也没有一个公认的训练出好的RM以及如何评估RM的方法。仍旧是后续需要探索的方向。

## 参考资料

[TÜLU 3: 拒绝RM的后训练技术总结](https://zhuanlan.zhihu.com/p/8589852586)

