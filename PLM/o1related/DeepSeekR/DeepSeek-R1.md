---
title: DeepSeek-R1
created: 2025-01-21
tags:
  - cot
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - DeepSeek
---

## 论文基本信息

标题：

作者：

链接：https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

代码：https://huggingface.co/deepseek-ai

框架图：


## 背景

我们介绍我们的第一代推理模型 DeepSeek-R1-Zero 和 DeepSeek-R1。 

我们迈出了**使用纯强化学习（RL）（无需监督学习）提高语言模型推理能力**的第一步。我们的目标是探索法学硕士在没有任何监督数据的情况下发展推理能力的潜力，重点关注它们通过纯强化学习过程的自我进化。

我们使用 DeepSeek-V3-Base 作为基础模型，并采用 GRPO (Shao et al., 2024) 作为 RL 框架来提高模型的推理性能。在训练过程中，DeepSeek-R1-Zero自然而然地表现出了许多强大而有趣的推理行为。经过数千个 RL 步骤后，DeepSeek-R1-Zero 在推理基准测试中展现出超强的性能。例如，AIME 2024 上的 pass@1 分数从 15.6% 提高到 71.0%，通过多数投票，分数进一步提高到 86.7%，与 OpenAI-o1-0912 的性能相当。

然而，DeepSeek-R1-Zero 遇到了可读性差、语言混合等挑战。为了解决这些问题并进一步提高推理性能，我们推出了 DeepSeek-R1，**它结合了少量的冷启动数据和多阶段训练管道**。具体来说，我们首先收集数千个冷启动数据来微调 DeepSeek-V3-Base 模型。接下来，我们像 DeepSeek-R1Zero 一样执行面向推理的强化学习。当 RL 过程接近收敛时，我们通过 RL 检查点上的拒绝采样，结合来自 DeepSeek-V3 在写作、事实 QA 和自我认知等领域的监督数据来创建新的 SFT 数据，然后重新训练 DeepSeek-V3 -基础模型。在使用新数据进行微调后，检查点会经历额外的强化学习过程，同时考虑所有场景的提示。经过这些步骤，我们获得了一个名为 DeepSeek-R1 的检查点，其性能与 OpenAI-o1-1217 相当。

我们进一步探索从 DeepSeek-R1 到更小的密集模型的蒸馏。使用 Qwen-2.5-32B（Qwen，2024b）作为基础模型，**从 DeepSeek-R1 直接蒸馏的性能优于对其应用 RL**。这表明较大的基础模型发现的推理模式对于提高推理能力至关重要。我们的蒸馏 14B 模型大幅优于最先进的开源 QwQ-32B-Preview（Qwen，2024a），并且蒸馏 32B 和 70B 模型在密集模型的推理基准上创下了新记录。

为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1 以及基于 Qwen 2.5和 Llama3 从 DeepSeek-R1 中蒸馏出来的六个密集模型（1.5B、7B、8B、14B、32B、70B）。

![](img/Pasted%20image%2020250121103056.png)


## 相关研究

process-based reward models (Lightman et al., 2023; Uesato et al., 2022; Wang et al., 2023)
reinforcement learning (Kumar et al., 2024)
search algorithms such as Monte Carlo Tree Search and Beam Search (Feng et al., 2024; Trinh et al., 2024; Xin et al., 2024).


## DeepSeek-R1-Zero: Reinforcement Learning on the Base Model

### **GRPO**

![](img/Pasted%20image%2020250121112104.png)

### **Reward模型**
- Accuracy rewards：评估响应是否正确。
- Format rewards：强制模型将其思维过程放在“<think>”和“</think>”标签之间。

我们在开发 DeepSeek-R1-Zero 时没有应用结果或过程神经奖励模型，因为我们发现神经奖励模型在大规模强化学习过程中可能会遭受奖励黑客攻击，并且重新训练奖励模型需要额外的训练资源它使整个训练流程变得复杂。

### **训练模板**

为了训练 DeepSeek-R1-Zero，我们首先设计一个简单的模板，指导基本模型遵守我们指定的指令。如表 1 所示，该模板需要 DeepSeek-R1-Zero 首先产生推理过程，然后得出最终答案。我们有意将限制限制在这种结构格式上，避免任何特定于内容的偏差——例如强制反思推理或促进特定的问题解决策略——以确保我们能够在强化学习（RL）过程中准确观察模型的自然进展。

![](img/Pasted%20image%2020250121112634.png)

### **Performance of DeepSeek-R1-Zero**

图 2 描绘了 DeepSeekR1-Zero 在整个强化学习 (RL) 训练过程中在 AIME 2024 基准上的性能轨迹。如图所示，随着 RL 训练的进展，DeepSeek-R1-Zero 表现出稳定且一致的性能增强。值得注意的是，AIME 2024 上的平均 pass@1 分数显示出显着的增长，从最初的 15.6% 跃升至令人印象深刻的 71.0%，达到了与 OpenAI-o1-0912 相当的性能水平。这一重大改进凸显了我们的 RL 算法随着时间的推移优化模型性能的有效性。

表 2 提供了 DeepSeek-R1-Zero 和 OpenAI 的 o1-0912 模型在各种推理相关基准上的比较分析。RL赋能的DeepSeek-R1-Zero 无需任何监督微调数据即可获得强大的推理能力。它强调了模型仅通过 RL 有效学习和泛化的能力。此外，DeepSeekR1-Zero 的性能可以通过多数投票的应用进一步增强。例如，当在 AIME 基准上采用多数投票时，DeepSeek-R1-Zero 的性能从 71.0% 提升到 86.7%，从而超过了 OpenAI-o1-0912 的性能。 DeepSeek-R1-Zero 能够在有或没有多数投票的情况下实现如此有竞争力的性能，凸显了其强大的基础能力及其在推理任务中进一步进步的潜力。

![](img/Pasted%20image%2020250121113457.png)

### **Self-evolution Process of DeepSeek-R1-Zero**

DeepSeek-R1-Zero 的自我进化过程精彩地展示了 RL 如何驱动模型自主提高其推理能力。通过直接从基础模型启动强化学习，我们可以密切监控模型的进展，而不受监督微调阶段的影响。这种方法提供了模型如何随时间演变的清晰视图，特别是在其处理复杂推理任务的能力方面。

DeepSeek-R1-Zero 自然地获得了通过利用延长的测试时间计算来解决日益复杂的推理任务的能力。

![](img/Pasted%20image%2020250121113742.png)

这种自我进化最显着的方面之一是随着测试时间计算的增加而出现复杂的行为。诸如反思（模型重新审视并重新评估其先前步骤）以及探索解决问题的替代方法等行为会自发出现。这些行为不是明确编程的，而是模型与强化学习环境交互的结果。这种自发的发展显着增强了 DeepSeek-R1-Zero 的推理能力，使其能够以更高的效率和准确性处理更具挑战性的任务。

### **Aha Moment of DeepSeek-R1-Zero**

在 DeepSeek-R1-Zero 训练过程中观察到的一个特别有趣的现象是“顿悟时刻”的发生。如表 3 所示，这一时刻发生在模型的中间版本中。在此阶段，DeepSeek-R1-Zero 学会通过重新评估其初始方法来为问题分配更多思考时间。这种行为不仅证明了模型推理能力不断增强，而且也是强化学习如何带来意想不到的复杂结果的一个迷人例子。

它强调了强化学习的力量和美妙之处：我们不是明确地教导模型如何解决问题，而是简单地为其提供正确的激励，它就会自主开发先进的问题解决策略。 “顿悟时刻”有力地提醒人们，强化学习有潜力解锁人工智能系统的新水平，为未来更加自主和自适应的模型铺平道路。

![](img/Pasted%20image%2020250121114033.png)

### **Drawback of DeepSeek-R1-Zero**

DeepSeek-R1-Zero 面临着可读性差和语言混合等挑战。

## DeepSeek-R1: Reinforcement Learning with Cold Start

受到 DeepSeek-R1-Zero 令人鼓舞的结果的启发，自然而然地出现了两个问题：1）通过纳入少量高质量数据作为冷启动，能否进一步提高推理性能或加速收敛？ 2）我们如何训练一个用户友好的模型，不仅能产生清晰、连贯的思想链（CoT），而且能表现出强大的通用能力？为了解决这些问题，我们设计了一个训练 DeepSeek-R1 的管道。该管道由四个阶段组成，概述如下。

### Cold Start

与 DeepSeek-R1-Zero 不同，为了防止从基础模型开始的 RL 训练的早期不稳定冷启动阶段，对于 DeepSeek-R1，我们构建并收集少量长 CoT 数据，以来微调模型作为初始 RL actor。为了收集此类数据，我们探索了几种方法：以长 CoT 为例，使用少样本提示，通过反馈和验证直接提示模型生成详细答案，以可读格式收集 DeepSeek-R1Zero 输出，并通过人类注释者的后处理细化结果。

收集了数千个冷启动数据，该数据的优势是
- 可读性：DeepSeek-R1-Zero 的一个关键限制是其内容通常不适合阅读。响应可能混合多种语言或缺乏markdown格式来突出显示用户的答案。相比之下，在为 DeepSeek-R1 创建冷启动数据时，我们设计了一种可读模式，其中在每个响应末尾包含摘要，并过滤掉对读者不友好的响应。这里，我们将输出格式定义为`|special_token|<reasoning_process>|special_token|<summary>`，其中推理过程是问题的CoT，summary用于总结推理结果。
- 潜力：通过根据人类先验仔细设计冷启动数据的模式，我们观察到比 DeepSeek-R1-Zero 更好的性能。我们相信迭代训练是推理模型的更好方法。

### Reasoning-oriented Reinforcement Learning

在对冷启动数据进行微调 DeepSeek-V3-Base 后，我们应用了与 DeepSeek-R1-Zero 相同的大规模强化学习训练过程。这一阶段的重点是增强模型的推理能力，特别是在编码、数学、科学和逻辑推理等推理密集型任务中，这些任务涉及明确的问题和明确的解决方案。在训练过程中，我们观察到 CoT 经常表现出语言混合，特别是当 RL 提示涉及多种语言时。为了缓解语言混合问题，我们在 RL 训练期间引入了语言一致性奖励，其计算方式为目标语言单词在 CoT 中的比例。尽管消融实验表明，这种对齐会导致模型性能略有下降，但这种奖励符合人类偏好，使其更具可读性。最后，我们将推理任务的准确性和语言一致性的奖励结合起来，直接求和形成最终的奖励。然后，我们对微调模型应用强化学习（RL）训练，直到其在推理任务上实现收敛。

### Rejection Sampling and Supervised Fine-Tuning

当面向推理的 RL 收敛时，我们利用生成的检查点来收集下一轮的 SFT（监督微调）数据。与最初的冷启动数据主要侧重于推理不同，此阶段融合了其他领域的数据，以增强模型在写作、角色扮演和其他通用任务方面的能力。具体来说，我们生成数据并微调模型，如下所述。

Reasoning data: 我们通过从上述 RL 训练的检查点执行拒绝采样来策划推理提示并生成推理轨迹。在前一阶段，我们仅包含可以使用基于规则的奖励进行评估的数据。然而，在这个阶段，我们通过合并额外的数据来扩展数据集，其中一些数据使用生成奖励模型，将ground-truth和模型预测输入 DeepSeek-V3 进行判断。此外，由于模型输出有时比较混乱且难以阅读，因此我们过滤掉了混合语言、长段落和代码块的思维链。对于每个提示，我们都会对多个响应进行采样，并仅保留正确的响应。我们总共收集了大约 60 万个推理相关的训练样本。

Non-Reasoning data: 对于非推理数据，例如写作、事实问答、自我认知和翻译，我们采用 DeepSeek-V3 管道并重用 DeepSeek-V3 的 SFT 数据集的部分内容。对于某些非推理任务，我们在通过提示回答问题之前调用 DeepSeek-V3 生成潜在的思维链。然而，对于更简单的查询，例如“hello”，我们不提供 CoT 响应。最终，我们一共收集了大约20万个与推理无关的训练样本。

我们使用上述约 80 万个样本的精选数据集对 DeepSeek-V3-Base 进行了两个epoch的微调。

### Reinforcement Learning for all Scenarios

为了进一步使模型符合人类偏好，我们实施了二次强化学习阶段，旨在提高模型的有用性和无害性，同时完善其推理能力。具体来说，我们使用奖励信号和不同提示分布的组合来训练模型。
- 对于推理数据，我们遵循 DeepSeek-R1-Zero 中概述的方法，该方法利用基于规则的奖励来指导数学、代码和逻辑推理领域的学习过程。
- 对于一般数据，我们采用奖励模型来捕捉复杂而细致的场景中的人类偏好。我们以 DeepSeek-V3 管道为基础，并采用类似的偏好对和训练提示分布。
- 对于有用性，我们只关注最终摘要，确保评估强调响应对用户的实用性和相关性，同时最大限度地减少对潜在推理过程的干扰。
- 对于无害性，我们评估模型的整个响应，包括推理过程和摘要，以识别和减轻生成过程中可能出现的任何潜在风险、偏见或有害内容。
最终，奖励信号和不同数据分布的整合使我们能够训练一个擅长推理的模型，同时优先考虑有用性和无害性。

## Distillation: Empower Small Models with Reasoning Capability

为了为更高效的小型模型配备 DeekSeek-R1 等推理功能，我们使用 DeepSeek-R1训练中 的 80 万个样本直接对 Qwen (Qwen, 2024b) 和 Llama (AI@Meta, 2024) 等开源模型进行微调。我们的研究结果表明，这种简单的蒸馏方法显着增强了较小模型的推理能力。我们在这里使用的基本模型是 Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.514B、Qwen2.5-32B、Llama-3.1-8B 和 Llama-3.3-70B-Instruct。我们选择Llama-3.3，因为它的推理能力比Llama-3.1略好。  对于蒸馏模型，我们仅应用 SFT，不包括 RL 阶段，尽管合并 RL 可以显着提高模型性能。我们的主要目标是证明蒸馏技术的有效性，将 RL 阶段的探索留给更广泛的研究社区。



## 实验

DeepSeek R1的指标大部分比R1-Zero好

![](img/Pasted%20image%2020250121125013.png)

![](img/Pasted%20image%2020250121125111.png)




## 讨论

### Distillation v.s. Reinforcement Learning

![](img/Pasted%20image%2020250121125623.png)

我们使用数学、代码和 STEM 数据在 Qwen-32B-Base 上进行大规模 RL 训练，训练超过 10K 步骤，最终得到 DeepSeek-R1-Zero-Qwen-32B。实验结果如图 6 所示，表明 32B 基本模型经过大规模的RL 训练，达到与 QwQ-32B-Preview 相当的性能。

然而，从 DeepSeek-R1 中提炼出来的 DeepSeek-R1Distill-Qwen-32B 在所有基准测试中的表现都明显优于 DeepSeek-R1-Zero-Qwen-32B。因此，我们可以得出两个结论：首先，将更强大的模型蒸馏成更小的模型会产生优异的结果，而本文提到的依赖于大规模强化学习的较小模型需要巨大的计算能力，甚至可能无法达到蒸馏的性能。其次，虽然蒸馏策略既经济又有效，但超越智能的边界可能仍然需要更强大的基础模型和更大规模的强化学习。

### Unsuccessful Attempts

Process Reward Model (PRM)：PRM 是一种合理的方法，可以引导模型找到更好的方法来解决推理任务（Lightman et al., 2023；Uesato et al., 2022；Wang et al., 2023）。然而，在实践中，PRM 存在三个主要局限性，可能会阻碍其最终成功。首先，在一般推理中明确定义细粒度步骤具有挑战性。其次，确定当前中间步骤是否正确是一项具有挑战性的任务。使用模型进行自动标注可能无法得到满意的结果，而手动标注则不利于规模化。第三，一旦引入基于模型的PRM，就不可避免地会导致奖励黑客攻击（Gao et al., 2022），并且重新训练奖励模型需要额外的训练资源，这使得整个训练流程变得复杂。总之，虽然 PRM 展示了对模型生成的前 N ​​个响应进行重新排序或协助引导搜索的良好能力（Snell 等人，2024），但与大规模过程中引入的额外计算开销相比，其优势有限。

Monte Carlo Tree Search (MCTS)：受 AlphaGo（Silver 等人，2017b）和 AlphaZero（Silver 等人，2017a）的启发，我们探索使用蒙特卡罗树搜索（MCTS）来增强测试时计算可扩展性。这种方法涉及将答案分解为更小的部分，以使模型能够系统地探索解决方案空间。为了促进这一点，我们提示模型生成与搜索所需的特定推理步骤相对应的多个标签。对于训练，我们首先使用收集的提示通过预训练价值模型引导的 MCTS 寻找答案。随后，我们使用生成的问答对来训练actor模型和价值模型，迭代地完善该过程。

然而，这种方法在扩大培训规模时遇到了一些挑战。首先，与搜索空间相对明确定义的国际象棋不同，令牌生成提供了指数级更大的搜索空间。为了解决这个问题，我们为每个节点设置了最大扩展限制，但这可能导致模型陷入局部最优。其次，价值模型直接影响生成的质量，因为它指导搜索过程的每一步。训练细粒度的价值模型本质上是困难的，这使得模型迭代改进具有挑战性。虽然 AlphaGo 的核心成功依赖于训练价值模型来逐步提高其性能，但由于代币生成的复杂性，这一原则在我们的设置中很难复制。

总之，虽然 MCTS 与预先训练的价值模型配合使用时可以提高推理过程中的性能，但通过自搜索迭代地提高模型性能仍然是一个挑战。

## 未来计划

- 通用能力：DeepSeek-R1的通用能力仍然不及DeepSeekV3。接下来，DeepSeek团队计划探索如何利用长CoT来提升这些领域的任务表现。
- 语言混合：DeepSeek-R1目前针对中文和英文进行了优化，但是在处理其他语言以及语言遵循方面还是会有问题。
- PE：DeepSeek-R1对Prompt非常敏感。few-shot提示会持续降低其性能。这里建议用户直接描述问题并指定输出格式（采用zero-shot，不要加示例），以获得最佳结果。
- 软件工程任务：由于长时间的评估会影响RL过程的效率，大规模RL尚未在软件工程任务中广泛应用。因此，DeepSeek-R1在软件工程基准测试上未显示出比DeepSeek-V3更大的改进。未来版本将通过在软件工程数据上实施拒绝采样或在RL过程中引入异步评估来提高效率。


## 参考资料

[There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study](https://oatllm.notion.site/oat-zero)

1. **There may NOT be Aha moment in R1-Zero-like training.** Instead, we found Aha moment (such as self-reflection patterns) appears at epoch 0, namely base models. We observe that all models (except Llama-3.x series) already exhibit self-reflection patterns without any post-training.
2. We found **Superficial Self-Reflection (SSR)** from base models’ responses, in which case self-reflections do not necessarily lead to correct final answers.
3. We took **a closer look at R1-Zero-like training via RL**, and found that the increasing response length phenomenon is not due to the emergence of self-reflection, but a consequence of RL optimizing well-designed rule-based reward functions.


[DeepSeek R1 论文解读&关键技术点梳理](https://mp.weixin.qq.com/s/wckZqmgSmocnIgUPcg5QcQ)

DeepSeek-R1的100问： https://blog.sciencenet.cn/blog-439941-1469698.html

[Deepseek R1可能找到了超越人类的办法](https://mazzzystar.com/2025/01/30/chatgpt-to-deepseek-r1-zh)

