---
title: demystify-long-cot
created: 2025-03-03
tags:
  - longcot
type: 论文
papername: Demystifying Long Chain-of-Thought Reasoning in LLMs
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - 清华
  - CMU
---

## 论文基本信息

标题：Demystifying Long Chain-of-Thought Reasoning in LLMs

作者：Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig,  Xiang Yue

链接：

代码： https://github.com/eddycmu/demystify-long-cot

框架图：


## 背景

long CoT的定义：

![](img/Pasted%20image%2020250303194927.png)





## 相关研究




## 核心亮点



## 实验

基座模型：Llama-3.1-8B和Qwen2.5 -7B-Math

训练数据：SFT和RL都用的是MATH的7,500条训练样本的prompt

评估benchmark：in-domain (MATH-500 test set)和out-of-domain (AIME 2024, TheoremQA, MMLU-Pro-1k).

### Impact of SFT on Long CoT

SFT数据来自拒绝采样：
- long CoT数据蒸馏自QwQ-32B-Preview
- short CoT数据蒸馏自Qwen2.5-Math-72B-Instruct

PPO训练：
- PPO算法，规则验证结果
- 采取余弦长度缩放奖励和重复惩罚

基座模型是Llama-3.1-8B

可以看到用long CoT数据SFT后模型能取得更高的性能上限，并且更容易通过RL进一步提升性能。

![](img/Pasted%20image%2020250303195412.png)

两种获取long CoT数据的方法：（1）Construct：通过提示短COT模型生成原始动作并顺序组合来构建长的COT轨迹； （2）Emergent：从现有的长COT模型中提取long CoT轨迹，这些模型已经有long CoT模式。

实验证明，高质量的涌现的的long CoT模式会有更好的泛化性和RL收益。

![](img/Pasted%20image%2020250303200428.png)

### Impact of Reward Design on Long CoT

探讨reward的设计对CoT长度和性能的影响。

#### CoT Length Stability

Classic Reward：基于规则的reward，正确答案给1分。

微调时设置上下文长度为16K，使用Classic Reward。

我们观察到，这两种模型在训练过程中都增加了CoT的长度，最终达到了上下文窗口限制。由于COTS超过允许的窗户尺寸，这导致训练准确性下降。此外，不同的基本模型表现出不同的缩放行为。与QWEN-2.5-MATH-7B相比,Llama-3.1-8B长度的波动更大。

![](img/Pasted%20image%2020250303201210.png)

我们还发现，思维链（CoTs）超过上下文窗口大小的比率在低于1的某个阈值处趋于平稳。这表明超过限制开始对COT长度分布施加显着的向下压力，并突出了上下文窗口大小在隐式长度惩罚中的作用。值得注意的是，即使没有奖励或优势归一化导致的明显超出长度惩罚，轨迹也可能受到惩罚。

#### Active Scaling of CoT Length

我们发现奖励塑形可以用来稳定新兴的长度缩放。我们设计了一个奖励函数，将思维链（CoT）长度作为额外输入，并观察了一些排序约束。

1. 首先，正确的思维链（CoTs）获得比错误的思维链更高的奖励。这意味着模型会被鼓励生成正确的推理过程。

2. 其次，较短的正确思维链获得比较长的正确思维链更高的奖励，这激励模型高效地使用推理计算。这一原则鼓励模型在保证正确性的同时尽可能简洁，避免不必要的冗长。

3. 第三，较短的错误思维链应该受到比较长的错误思维链更高的惩罚。这鼓励模型在不太可能得到正确答案时延长其思考时间。这个原则鼓励模型在面对困难问题时投入更多的思考，而不是草率地给出错误答案。

这些原则共同构成了一个精心设计的奖励机制，旨在优化模型的推理过程。它不仅鼓励正确性，还促进了效率和深度思考之间的平衡。这种方法可以帮助模型在各种复杂度的任务中都能表现出色，既能快速解决简单问题，又能在复杂问题上投入足够的思考。

我们发现使用分段余弦函数很方便，它易于调整且平滑。我们将这个奖励函数称为余弦奖励，如图3所示。这是一个稀疏奖励，仅在思维链（CoT）结束时根据答案的正确性给出一次奖励。

![](img/Pasted%20image%2020250304094914.png)

![](img/Pasted%20image%2020250304095005.png)

![](img/Pasted%20image%2020250304095026.png)

实验设置：
- Classic Reward和Cosine Reward
- 使用long CoT数据微调过的Llama3.1-8B作为起始点

Cosine Reward显著稳定化了RL阶段的长度scaling行为，也稳定了训练准确率，提高了RL的效率。模型在下游任务也有提升。

![](img/Pasted%20image%2020250304095443.png)

![](img/Pasted%20image%2020250304095613.png)
#### Cosine Reward Hyperparameters

调整Cosine Reward的超参数。

如果正确答案的奖励随COT长度（$r_0^c <r_L^c$）增加，则COT长度会爆炸。

正确奖励相对于错误奖励越低，CoT长度就越长。我们将此解释为一种训练出来的风险规避行为，在这种风险规避中，其中正确和错误奖励的比率决定了模型在给出答案终止其CoT时需要达到的自信度，以获得正面的期望值。

![](img/Pasted%20image%2020250304100048.png)

#### Context Window Size

我们知道，较长的上下文为模型提供了更多的探索空间，并且随着培训样本的更多，该模型最终学会了利用更多的上下文窗口。这提出了一个有趣的问题 - 是否需要更多的培训样本来学习使用更大的上下文窗口？

我们发现，具有上下文窗口大小为8K的模型的性能要比预期的4K模型更好。但是，我们观察到的16K性能低于8K。请注意，这三个实验均使用相同数量的训练样本（图6）。我们认为这表明模型需要更多的培训计算以学习完全利用更长的上下文窗口大小。这与Advancing language model reasoning through reinforcement learning and inference scaling中的发现一致。

![](img/Pasted%20image%2020250304100726.png)

#### Length Reward Hacking

通过足够的培训计算，模型开始显示奖励黑客的迹象，它通过重复而不是学习解决问题来增加困难问题的CoT长度。

我们还注意到模型的分支频率下降，我们通过计算CoT中关键词'alternatively'出现的次数来估计这一点。

![](img/Pasted%20image%2020250304101157.png)

我们通过实现一个简单的N -gram重复惩罚来缓和这一点。

![](img/Pasted%20image%2020250304101219.png)

我们观察到，对重复的标记（tokens）直接施加惩罚比对整个轨迹施加稀疏奖励更有效。同样，我们发现在计算回报时折扣重复惩罚也是有效的。具体指出重复发生的位置，可能使模型更容易学会避免重复。

从图5看到，重复惩罚导致了更好的下游任务性能和较短的CoT，这意味着可以更好地利用推理计算.

我们的实验揭示了重复惩罚，训练准确性和余弦奖励之间的关系。当训练准确性较低时，余弦奖励对COT的长度施加了更大的向上压力，从而通过重复增加了奖励黑客攻击。反过来，这需要更强大的重复惩罚。未来的工作可以进一步研究这些相互作用，并探索动态调整方法以更好地优化。

#### Optimal Discount Factors

我们假设，在时间局部性的基础上应用重复惩罚（即使用低折扣因子）将最为有效，因为它为特定的违规标记提供了更强的学习信号。然而，我们也观察到，当正确性（余弦）奖励的折扣因子过低时，性能会下降。

为了最佳调整这两种奖励类型，我们修改了PPO中的GAE公式以适应多种奖励类型

![](img/Pasted%20image%2020250304102108.png)

较低的折现因素有效地强制执行重复惩罚，而较高的折现因子则提高了正确的奖励和超出长度的惩罚。

从图5中看到，较高的因子使该模型可以得到足够的奖励，以便在COT中选择正确的答案。

我们观察到了一个相当有趣的现象，降低了正确性（余弦）奖励的折现因子γ增加了模型CoT中的分支频率，从而使模型迅速放弃了似乎不会立即导致正确答案的方法.

![](img/Pasted%20image%2020250304102446.png)

我们推测，这种短期思维是由于在正确答案之前只有相对少量的标记（tokens）接收到奖励，这意味着通向正确答案的中间步骤被低估了。这种行为导致了性能下降。（图5）



### Scaling up Verifiable Reward

我们使用WebInstruct dataset数据探索（从Web Corpora中提取的与推理有关的QA对），通过MinHash去重得到WebInstruct-462k。

#### SFT with Noisy Verifiable Data

加入SFT阶段。由于数据没有显著的监督信号，但是数据量大，我们在不过滤的情况下从教师模型中对每个提示生成了一个响应。RL使用MATH数据集。

表2显示，包含silver-supervised数据可改善平均性能。将Webintruct数据添加到长COT SFT中，相比仅使用数学，MMLU-PRO-1K的绝对准确度获得了510％的增益。此外，混合数学和Webintruct数据可实现跨基准的最佳平均精度。

![](img/Pasted%20image%2020250304103330.png)

#### Scaling up RL with Noisy Verifiable Data

我们比较了两种主要方法，以从嘈杂的可验证数据中获得奖励：1）提取短形式答案并使用基于规则的验证者； 2）使用基于模型的验证符能够处理自由形式响应。

这里的关键因素是QA对是否有short-form答案。因此，我们还比较了是否仅保留具有短形式答案的数据集。

我们通过使用原始参考解答来提示Qwen2.5-Math-7B-Instruct，从而实现基于模型的验证器。为了提取简短形式的答案，我们首先使用Llama-3.1-8B-Instruct从原始响应中提取，然后使用QwQ-32B-Preview进行拒绝采样。具体来说，我们从WebInstruct-462k中为每个提示生成两个响应，并丢弃两个响应都不符合提取的参考答案的情况。这个过程在115k个独特提示中产生了大约189k个响应。我们的案例研究表明，拒绝采样丢弃了许多提示，原因是：1）许多WebInstruct提示缺乏我们的基于规则的验证器可以有效处理的简短形式答案，2）一些提示即使对于QwQ-32B-Preview来说也太困难。对于监督微调（SFT），我们在过滤后的数据集上训练Llama-3.1-8B，作为强化学习（RL）的初始化。在RL阶段，我们在未过滤的设置中使用完整的462k提示集，在过滤后的设置中使用115k子集，使用30k个提示进行训练，每个提示4个响应。

![](img/Pasted%20image%2020250304104359.png)

在包含简短答案的筛选提示集上，带有基于规则的验证器的 RL 在相同数量的 RL 样本下的大多数基准测试中实现了最佳性能。这可能表明，经过适当过滤后，基于规则的验证程序可以从嘈杂的可验证数据中生成最高质量的奖励信号。

与在人工注释的验证数据 （MATH） 上训练的模型相比，利用嘈杂但多样化的可验证数据仍然可以显著提高 O.O.D. 基准测试的性能，TheoremQA 的绝对增益高达 2.9%，MMLU-Pro-1k 的绝对增益高达 6.8%。相比之下，将基于规则的验证程序应用于未筛选的数据会导致性能最差。这可能是由于它在自由格式答案上的训练准确率较低造成的，基于模型的验证器可实现更好的性能。

### Exploration on RL from the Base Model

自我验证行为有时会被模型的探索标记为涌现行为或“顿悟时刻”，因为这种模式在简短的 CoT 数据中很少见。但是，我们注意到，有时 Base Model 中已经存在自我验证行为，通过 RL 强化这些行为需要严格的条件，例如强大的 Base Model。

我们遵循 Zeng 等人（2025 年）的设置，在大约 8k MATH 3-5 级问题上使用 PPO 和基于规则的验证器来训练 Qwen2.5-Math-7B，但我们使用自己的基于规则的验证器实现。对于推理，我们采用温度 t = 0（贪婪解码），因为我们的初步实验表明，对于通过 Qwen2.5-Math-7B 的直接 RL 获得的模型，t = 0 通常明显优于 t > 0。考虑到 4096 个令牌的训练上下文长度，我们使用 4096 个令牌的最大输出长度。请注意，我们对基本模型使用 zero-shot prompting，以避免在输出模式中引入偏差。我们从以往作品中的长 CoT 案例中选择了五个代表性关键词，“wait”, “recheck”, “alternatively”, “retry” and “however”（OpenAI，2024 年;DeepSeek-AI，2025 年;Pan et al.， 2025;Zeng et al.， 2025），并计算它们的频率以量化模型进行自我验证的程度。

图 7 显示，Qwen2.5Math-7B 的 RL 有效地提高了准确性，但没有增加基础模型输出中存在的 “recheck” 模式的频率，也没有有效地激励其他反射模式，如 “retry” 和 “alternatively”。这表明来自基本模型的 RL 不一定会激励反射模式，尽管显著提升了性能。有时，基本模型的输出中存在此类行为，而 RL 并未显著增强它们。因此，我们可能需要更加小心地识别涌现行为。

![](img/Pasted%20image%2020250304104856.png)


长度放大被认为是模型有效探索的另一个重要特征。然而，我们注意到，有时长度放大可能伴随着 KL 散度的减小，这增加了长度受 KL 惩罚影响的可能性，并且只是恢复到基本模型的较长输出，而不是反映获得长 CoT 能力。

设置与上述的相同。除了输出 token 长度外，我们还计算 “coding rate”。如果模型的输出包含 “'''python”，我们会将其归类为 “coding”，因为 Qwen2.5-Math-7B 同时使用自然语言和编码来解决数学问题。请注意，这里的 “coding” 输出实际上是一种特殊形式的自然语言输出，其中的代码不被执行，代码的输出由模型生成。

图 8 （1） 显示，输出标记的长度在初始下降后增加，但从未超过基本模型的初始长度。

Zeng et al. （2025） 认为，最初的下降可能是由于模型从生成长编码输出过渡到较短的自然语言输出。然而，图 8 （2） 表明自然语言输出实际上比编码输出更长，并且长度的初始下降发生在两种类型的输出中。此外，图 8 （3） 显示编码率随后再次增加，这表明编码和自然语言之间的区别可能不会显着影响优化过程。

此外，我们怀疑随后的长度放大不是来自模型的探索，因为当长度放大时，策略与基本模型的 KL 散度会下降，如图 8 （4） 所示。这可能表明影响长度的是 KL 罚。如果是这种情况，则策略输出长度超过基本模型长度的可能性很小，因为探索受到 KL 约束的限制。

![](img/Pasted%20image%2020250304105135.png)


我们对 Qwen2.5-Math-7B 的 RL 的详细分析表明，它无法完全复制 DeepSeek-R1 的训练行为。我们确定了以下潜在原因：1） 基础模型相对较小（7B 参数），在受到激励时可能缺乏快速发展如此复杂能力的能力。2） 模型可能在（持续）预训练和退火期间过度暴露于类似 MATH 的短指令数据，导致过度拟合并阻碍长 CoT 行为的发展。

我们比较了基本模型的 RL 和long CoT SFT 后的 RL 的性能，发现long CoT SFT 的 RL 通常表现更好。

![](img/Pasted%20image%2020250304110924.png)

我们使用生成式搜索引擎 Perplexity.ai 来识别明确包含问题解决步骤的网页，这些步骤从多个角度处理问题或在提供答案后执行验证。使用 GPT-4o 生成了“顿悟时刻”特征的短语列表，然后采取MinHash算法去通过 OpenWebMath（从CommonCrawl清洗得到，经常用于预训练） 进行搜索。我们发现，在论坛帖子中有大量的匹配，其中多个用户之间的对话显示出与长 CoT 的相似性，讨论了多种方法以及回溯和纠错。这提出了一个有趣的可能性，即长期 CoT 起源于人类对话，尽管我们还应该注意，论坛是 OpenWebMath 中的常见数据源。




## 未来方向

1. 使用更大的模型
2. 如何有效地扩展验证信号？在设计 RL 环境的上下文中，是否有等效的预训练？
3. 还有哪些其他能力等待从预训练数据中嵌入的大量人类知识和经验中得到启发？

## 主要收获


## 参考资料
