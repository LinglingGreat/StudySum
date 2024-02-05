---
title: CUT
created: 2024-02-05
tags:
  - alignment
type: 论文
papername: Reasons to Reject? Aligning Language Models with Judgments
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 腾讯
  - 香港中文大学
---

## 论文基本信息

标题：Reasons to Reject? Aligning Language Models with Judgments

作者：

链接：https://arxiv.org/abs/2312.14591

代码：https://github.com/wwxu21/CUT

框架图：


## 背景

香港中文大学和腾讯 AI Lab 的研究者们提出了一项名为对比式非似然训练（Contrastive Unlikelihood Learning，CUT）的创新研究，利用语言反馈来对齐语言模型，让模型像人类一样从不同的批评意见中学习成长。

CUT 简单有效。仅凭 1317 条语言反馈数据，CUT 就能使 LLaMA2-13b 在 AlpacaEval 上的 win rate 从 1.87% 飙升至 62.56%，击败 175B 的 DaVinci003。更令人兴奋的是，CUT 能像其他 RLHF 框架一样形成探索 -> 批评 -> 改进的反复迭代，其中批评可由自动的评语模型来完成，实现整个系统“自产自评自提升”。

作者对 LLaMA2-chat-13b 进行了四轮迭代，将模型在 AlpacaEval 上的性能从 81.09% 逐步提升至 91.36%。相较于基于分数反馈的对齐技术（DPO），CUT 在同等数据规模下表现更佳。


## 相关研究

![](img/Pasted%20image%2020240205153932.png)

两种常见的大模型对齐方式：
1. 从**示例**中学习 (Learning from Demonstration)：基于现成的指令 - 回复对，利用监督式训练的方法来对齐大模型。
- 优点：训练稳定；实现简单。
    
- 缺点：收集高质量、多样化的示例数据成本高；无法从错误回复中学习；示例数据往往和模型无关。

2. 从**分数反馈**中学习 (Learning from Rewards)：给指令 - 回复对打分，利用强化学习训练模型最大化其回复的得分。
- 优点：能同时利用正确回复和错误回复；反馈信号与模型相关。
    
- 缺点：反馈信号稀疏；训练过程往往比较复杂。
    
此研究关注的则是从**语言反馈**中学习 (Learning from Judgments)：给指令 - 回复对写评语，基于该语言反馈改进模型存在的瑕疵，保持模型的优点，从而提升模型性能。

可以看出，语言反馈继承了分数反馈的优点。与分数反馈相比，语言反馈的信息量更大：与其让模型去猜哪里做对了和哪里做错了，语言反馈可以直接指出详细的不足之处和改进方向。然而，令人遗憾的是，研究者们发现目前尚无有效方法能充分利用语言反馈。为此，研究者们提出了一种创新性的框架 CUT，旨在充分发挥语言反馈的优势。

## 核心亮点

CUT 的核心思想是从对比中学习。研究者们通过对比大模型在不同条件下的回复去启发哪些部分是令人满意的，应该保持，哪些部分是有瑕疵，需要修改。基于此，研究者们利用最大似然估计（MLE）来训练令人满意的部分，利用非似然训练（UT）来修改回复中的瑕疵。

![](img/Pasted%20image%2020240205154050.png)

1. **对齐场景**：如上图所示，研究者们考虑了两种对齐场景：

a)x->y：这是通常理解的对齐场景，在该场景下，回复需要忠实地遵循指示并符合人类的期望和价值观。

b)[x, j] -> y: 该场景引入了语言反馈作为额外的条件。在该场景下，回复要同时满足指令和语言反馈。例如，当收到一个消极反馈，大模型需要根据对应的反馈中提到的问题去犯错。

2. **对齐数据**：如上图所示，基于上述两者对齐场景，研究者们构造了三类对齐数据：
a) Align-P：大模型生成了**令人满意的**回复，因此获得了积极的反馈。显然，Align-P 在x->y和[x, j] -> y场景下都是满足对齐的

b) Align-N：大模型生成了有瑕疵（蓝色加粗）的回复，因此获得了消极的反馈。对于 Align-N，x->y中是不满足对齐。但考虑该消极反馈后，Align-N 在[x, j] -> y场景下仍是对齐的。

c) Misalign：Align-N 中真实的消极反馈被替换为一条伪造的积极反馈。显然，Misalign 在x->y和[x, j] -> y场景下都不满足对齐。

3. **从对比中学习**：

![](img/Pasted%20image%2020240205154525.png)

a) **Align-N v.s. Misalign**：两者的区别主要在于[x, j] -> y下的对齐程度。鉴于大模型强大的上下文内学习能力（in-context learning），从 Align-N 到 Misalign 的对齐极性翻转通常伴随着特定词的生成概率的显著变化，尤其是那些与真实消极反馈密切相关的词。如上图所示，在 Align-N（左通路）的条件下，大模型生成 “a” 的概率明显高于 Misalign（右通路）。而这概率显著变化的地方刚好是大模型犯错的地方。

为了从该对比中学习，研究者们将 Align-N 和 Misalign 数据同时输入给大模型，以获取输出词分别在两种条件下的生成概率![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8icNsT3zQwjo5rbM8JFF4QaI8ZW0DVK54Kia1o6dxRQkBwCoYKfK27dFA/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)和![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8QbHto71mllFxuMdiaK2U64F1dTdJO8b5IS5zOzx8878LQwPHuSm500g/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)。那些在$j^-$条件下有着明显高于$j^+$条件下的生成概率的词被标记为不合适的词。具体而言，研究者们采用如下标准来量化不合适词的界定：
  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8AMyexhw3vHql3qgkBTnMgUgGZiaK2eAficv4dOCQpyScuMrn1FPzwR0g/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)

  

其中$\lambda$是权衡不合适词识别过程中精度和召回的超参数。

研究者们对这些识别出来的不合适词采用非似然训练（UT），从而迫使大模型去探索更加令人满意的回复。对于其他回复词，研究者们仍采用最大似然估计（MLE）来优化：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8PRibsKwVPqajfahB30QoPh82EwC5oibPEeTGb5rD6z3ibWf8egUU5gN5A/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)

  
其中$\alpha$是控制非似然训练的比重的超参数，N是回复词数。

b) **Align-P v.s. Align-N**：两者的区别主要在于x->y下的对齐程度。本质上，大模型通过引入不同极性的语言反馈来控制输出回复的质量。因此该二者的对比能启发大模型去区分令人满意的回复和有瑕疵的回复。具体而言，研究者们通过以下最大似然估计（MLE）损失来从该组对比中学习：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8QnZrobaCZQCIyibHuqBSmUBRFEQU1GzeJV4MYibfyPMFFOfmmZMibfm3g/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)

  

其中![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gWicvBJX4ZN9UPXZwuDJ5tKG8icLv0bKliaGMGfyYKdeHnd5W4QTt4MPicibvQccDln095pELSlhsG0Qxxg/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&wx_co=1)是指示函数，如果数据满足x->y对齐返回 1，否则返回 0。

CUT 最终的训练目标结合了上述两组对比：$L_{CUT}=L_1+L_2$。
## 实验
**1. 离线对齐**

为了省钱，研究者们首先尝试了利用现成的语言反馈数据来对齐大模型。该实验用以证明 CUT 在利用语言反馈的能力。

a) 通用模型

![](img/Pasted%20image%2020240205155131.png)

如上表所示，在通用模型对齐上，研究者们使用 Shepherd 提供的 1317 条对齐数据，分别在冷启动（LLaMA2）和热启动（LLaMA2-chat）的条件下比较了 CUT 与现有从语言反馈学习的方法。

在基于 LLaMA2 的冷启动实验下，CUT 在 AlpacaEval 测试平台上大幅超越现有对齐方法，充分证明了其在利用语言反馈方面的优势。并且 CUT 在 TruthfulQA 上相比于基座模型也取得了大幅提升，这揭示了 CUT 在缓解大模型幻觉（hallucination）问题上有巨大潜力。

在基于 LLaMA2-chat 的热启动场景中，现有方法在提升 LLaMA2-chat 方面表现不佳，甚至产生了负面影响。然而，CUT 却能在此基础上进一步提升基座模型的性能，再次验证了 CUT 在利用语言反馈方面的巨大潜力。

b) 专家模型

![](img/Pasted%20image%2020240205155209.png)

研究者们同时测试了在特定专家任务（文本摘要）上 CUT 的对齐效果。如上表所示，CUT 在专家任务上相比现有对齐方法也取得了明显的提升。

**2. 在线对齐**

离线对齐的研究已经成功证明了 CUT 的强大对齐性能。现在，研究者们进一步地探索了更贴近实际应用的在线对齐场景。在这个场景中，研究者们迭代地对目标大模型的回复进行语言反馈标注，使该目标模型能够根据与其相关的语言反馈进行更精确的对齐。具体流程如下：

- 步骤 1：收集指令x，并获得目标大模型的回复y。
    
- 步骤 2：针对上述指令 - 回复对，标注语言反馈j。
    
- 步骤 3：采用 CUT，基于收集到的三元组数据{x, y, j}微调目标大模型。

![](img/Pasted%20image%2020240205155319.png)

如上图所示，经过四轮在线对齐迭代后，CUT 在仅有 4000 条训练数据和较小的 13B 模型规模的条件下，仍然能够取得令人瞩目的 91.36 分数。这一成绩进一步展示了 CUT 卓越的性能和巨大的潜力。

**3. AI 评语模型**

![](img/Pasted%20image%2020240205155342.png)

考虑到语言反馈的标注成本，研究者尝试训练评语模型（Judgement Model）来自动地为目标大模型标注语言反馈。如上图所示，研究者们分别使用 5000 条（AI Judge-5000）和 3000 条（AI Judge-3000）语言反馈数据来训练了两个评语模型。这两个评语模型在优化目标大型模型方面都取得了显著成果，尤其是 AI Judge-5000 的效果更为突出。

这证明了利用 AI 评语模型对齐目标大模型的可行性，同时也突显了评语模型质量在整个对齐过程中的重要性。这组实验还为未来降低标注成本提供了有力支持。

**4. 语言反馈 vs. 分数反馈**

为了深入挖掘语言反馈在大型模型对齐中的巨大潜力，研究者们将基于语言反馈的 CUT 与基于分数反馈的方法（DPO）进行了对比。为了确保比较的公平性，研究者们选取了 4000 组相同的指令 - 回复对作为实验样本，让 CUT 和 DPO 分别从这些数据所对应的分数反馈和语言反馈中进行学习。

![](img/Pasted%20image%2020240205155430.png)

如上表所示，在冷启动（LLaMA2）实验中，CUT 的表现明显优于 DPO。而在热启动（LLaMA2-chat）实验中，CUT 在 ARC、HellaSwag、MMLU 和 TruthfulQA 等任务上能取得与 DPO 相媲美的成绩，并在 AlpacaEval 任务上大幅度领先 DPO。这一实验证实了在大型模型对齐过程中，相较于分数反馈，语言反馈具有更大的潜力和优势。

## 未来方向
该工作中，研究者们系统地探讨了语言反馈在大模型对齐中的现状并创新性地提出了一种基于语言反馈的对齐框架 CUT，揭示了语言反馈在大型模型对齐领域所具有的巨大潜力和优势。此外，语言反馈的研究还有着一些新的方向和挑战，例如：

**1. 评语模型的质量**：尽管研究人员已成功地证实了训练评语模型的可行性，但在观察模型输出时，他们仍然发现评语模型经常给出不够准确的评价。因此，提升评语模型的质量对于未来大规模利用语言反馈进行对齐具有举足轻重的意义。 

**2. 新知识的引入**：当语言反馈涉及到大模型所缺乏的知识时，大模型即使能准确地识别出错误的地方，但也没有明确的修改方向。因此在对齐的同时补足大模型缺乏的知识非常重要。

**3. 多模态对齐**：语言模型的成功促进了多模态大模型的研究，如语言、语音、图像和视频的结合。在这些多模态场景下，研究语言反馈以及对应模态的反馈迎来了新的定义和挑战。

## 主要收获


## 参考资料

[像人类一样在批评中学习成长，1317条评语让LLaMA2胜率飙升30倍](https://mp.weixin.qq.com/s/nyXJanCZYpZhRBNtTT1DJg)

