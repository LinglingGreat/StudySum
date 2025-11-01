---
title: Smol训练手册-构建世界级LLMs的秘诀
created: 2025-11-01
tags:
  - 预训练
---

### 第一阶段：训练之前先决策

训练一个高性能的 LLM 是一个高风险项目，所以在训练之前一定要从战略层面回答三个核心问题：**为什么训练（Why）**、**训练什么（What）** 和**如何训练（How）**。

在投入大量计算资源之前，必须进行严格的自我审视。很多时候，失败不是参数错，而是**不该训练却训练了**。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/vI9nYe94fsFfuCK1S6y4aIUdbcA5OwO2YE5Dp5HHgfCia3LEtyG6DGjkhDFCS5WrcUM0GYfXhkBQrSbiaAAibqqgQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=1)

#### 1. 明确目标与定制化需求（Why → What）

训练 LLM 的理由必须具体且不可替代。面对日益强大的开源生态，团队必须首先确认：**现有的模型是否能通过提示词工程（Prompting）或微调（Fine-tuning）来解决问题？** 只有当目标集中在**研究**（探索新架构）、**生产**（处理领域特有词汇或满足**边缘设备部署**等约束）或**战略开源**（填补生态空白，如强大的端侧模型）时，定制预训练才具有价值。

Hugging Face团队在训练 SmolLM3 时的战略目标便是填补市场对**强大、高效端侧小模型**的空白，因此模型类型被确定为 **3B 密集的 Llama 风格架构**。

#### 2. “去风险”的消融实验

大模型的行为往往并**不直观**，仅凭理论推断是无法做出正确决策的。所以，消融实验（Ablations）是确定最终训练方案的唯一途径。

例如，使用看似“最高质量的数据”并不总是产生更强大的模型。以 arXiv 为例，它是人类科学知识的大量集合。直观地说，对如此丰富的 STEM 数据进行训练应该会产生更好的模型，对吧？在实践中，它不会，尤其是对于较小的型号，它甚至会损害性能（Shao 等人，2024 年）。为什么？原因是，虽然 arXiv 论文充满了知识，但它们高度专业化，并且以狭隘的学术风格编写，与模型从中学习最多的多样化、通用文本有很大不同。

要让消融真正有用，实验必须快速迭代且**有**强大的**区分力**。为此，**必须严格遵循“一次只修改一个变量”的核心规则。在 LLM 训练的复杂环境中，不同的组件往往以非线性方式相互作用。如果一次改动多个配置，即便结果变好，也无法判断究竟是哪一项起作用，后续决策就失去依据。只有当**单独改动**被验证为有效，才能把它并入**基线 **，再在新的基线上测试下一个改动，形成可解释、可回滚的累积式进步。**

在采纳标准上，遵循“去风险（Derisking）”原则：而任何架构或超参数的修改，只有在通过实验证明其有助于目标性能或训练效率时，才会被采纳。

这里的“有帮助”，不仅指**目标能力提升**，也可以是**可测量、有意义**的工程收益，比如更快的推理、更低的内存、更高的稳定性，并且**不得以牺牲其他关键性能为代价**。

之所以要这么严格，是因为训练 LLM 的成本极高。以 SmolLM3 为例，**预训练阶段的消融与调试就消耗了超过一半的总成本（共 161,280 GPU·h）**。在这样的压力下，团队必须实行**战略性实验（strategic experimentation）**：只针对那些能对模型性能产生实质性影响的修改进行测试。

在 SmolLM3 的整个开发过程中，我们总共运行了 100 多次消融：我们花了 20 天进行训练前消融，10 天用于训练中期消融，7 天从意外的训练问题中恢复.

#### 设置我们的消融框架

消融的目标是小规模进行实验，并获得我们可以自信地推断到最终生产运行的结果。

主要有两种方法。首先，我们可以采用目标模型大小并在更少的代币上训练它。对于 SmolLM3 消融，我们在 100B 标记上训练了完整的 3B 模型，而不是最终的 11T。其次，如果我们的目标模型太大，我们可以训练一个更小的代理模型进行消融。例如，当 Kimi 开发具有 32B 主动参数的 1T 参数 Kimi K2 模型时，对所有消融使用全尺寸将非常昂贵，因此他们在具有 0.5B 主动参数的 3B MoE 上运行了一些消融（Team et al.， 2025）。

一个关键问题是这些小规模的发现是否真的转移了。根据我们的经验，如果某些东西在小规模上损害了性能，您可以自信地将其排除在大规模之外。但是，如果某些东西在小规模上有效，您仍然应该确保您已经训练了合理数量的标记，以便得出这些发现很有可能推断到更大规模的结论。训练时间越长，消融模型越接近最终模型越好。

#### [**了解什么是有效的：评估**](https://huggingfacetb-smol-training-playbook.hf.space/#understanding-what-works-evaluation)

任何训练模型的人的第一反应可能是查看损失，是的，这确实很重要。您希望看到它平稳下降，而不会出现剧烈的峰值或不稳定。对于许多架构选择，损耗与下游性能密切相关，并且可能足够（Y. Chen et al.， 2025）。然而，只看损失并不总是可靠的。以数据消融为例，你会发现在维基百科上训练比在网页上训练损失更低（下一个标记更容易预测），但这并不意味着你会得到一个更强大的模型。同样，如果我们在运行之间更改分词器，则损失无法直接比较，因为文本的拆分方式不同。一些变化还可能特别影响推理和数学等某些功能，并在平均损失中被冲走。最后但并非最不重要的一点是，即使在预训练损失收敛之后，模型也可以继续改进下游任务（Liu et al.， 2022）。

我们需要更细粒度的评估来了解全貌并理解这些细微差别的影响，一种自然的方法是使用下游评估来测试知识、理解、推理以及对我们重要的任何其他领域。

对于这些消融，最好专注于提供良好早期信号并避免嘈杂基准的任务。在 FineTasks 和 FineWeb2 中，可靠的评估任务由四个关键原则定义：

单调性：随着模型训练时间的延长，基准分数应该会持续提高。

低噪声：当我们使用相同的设置但不同的随机种子训练模型时，基准分数应该不会有很大差异。

高于随机性能：许多功能仅在训练后期出现，因此长时间显示随机级别性能的任务对消融没有用处。例如，对于多项选择格式的 MMLU，我们稍后将解释。

排名一致性：如果一种方法在早期阶段优于另一种方法，则随着训练的继续，这种排序应该保持稳定。

任务的质量还取决于任务的表述（我们如何向模型提问）和指标选择（我们如何计算答案分数）。

三种常见的任务公式是多项选择格式 （MCF）、完形填空公式 （CF） 和自由格式生成 （FG）。多项选择格式要求模型从提示中显式显示并以 A/B/C/D 为前缀的多个选项中选择一个选项（例如，在 MMLU 中所做的那样）。在完形填空公式中，我们比较不同选择的可能性，看看哪一个更有可能，而无需在提示中提供它们。在 FG 中，我们查看给定提示的贪婪生成的准确性。FG 需要模型中的大量潜在知识，并且对于模型来说通常太困难了，在完全训练之前的短训练前消融中，模型无法真正有用。因此，在运行小尺寸消融（MCF 或 CF）时，我们专注于多项选择配方。

研究还表明，模型在训练早期与 MCF 作斗争，只有在广泛训练后才学会这项技能，这使得 CF 更适合早期信号（Du 等人，2025 年;Gu 等人，2025 年;J. Li 等人，2025 年）。因此，我们将 CF 用于小消融，并将 MCF 集成到主运行中，因为一旦模型通过阈值以获得足够高的 MCF 信噪比，它就会提供更好的训练中期信号。还需要快速说明的是，为了在 CF 等序列似然评估中对模型的答案进行评分，我们将准确性计算为正确答案具有最高对数概率的问题百分比，按字符数归一化。这种标准化可以防止偏向于较短的答案。

#### 训练框架

| Framework       | Features                                | Battle-tested                       | Optimised                                               | Lines of Code (core / total) | Extensibility & Debugging                 |
| --------------- | --------------------------------------- | ----------------------------------- | ------------------------------------------------------- | ---------------------------- | ----------------------------------------- |
| **Megatron-LM** | ✅ Extensive                             | ✅ Kimi-K2, Nemotron                 | ✅ Pioneers of 3D parallelism                            | 93k / 269k                   | ⚠️ Hard for beginners                     |
| **DeepSpeed**   | ✅ Extensive                             | ✅ BLOOM, GLM                        | ✅ Pioneers of ZeRO & 3D parallelism                     | 94k / 194k                   | ⚠️ Hard for beginners                     |
| **TorchTitan**  | ⚡ Growing feature set                   | ⚠️ Newer but tested by PyTorch team | ⚡Optimised for dense models, MoE improvements underway. | 7k / 9k                      | ⚡ Moderate: requires parallelism know-how |
| **Nanotron**    | 🎯 Minimal, tailored for HF pretraining | ✅ Yes (StarCoder, SmolLM)           | ✅ Optimised (UltraScale Playbook)                       | 15k / 66k                    | ⚡ Moderate: requires parallelism know-how |


#### 模型架构

|Architecture Type|Model Family|Sizes|
|---|---|---|
|**Dense**|[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)|8B, 70B|
|**Dense**|[Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)|1B, 3B|
|**Dense**|[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)|0.6B, 1.7B, 4B, 14B, 32B|
|**Dense**|[Gemma3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)|12B, 27B|
|**Dense**|[SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9), [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)|135M, 360M, 1.7B, 3B|
|**MoE**|[Qwen3 MoE](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)|30B-A3B, 235B-A122B|
|**MoE**|[GPT-OSS](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)|21B-A3B, 117B-A5B|
|**MoE**|[Kimi Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)|16B-A3B|
|**MoE**|[Kimi-k2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d)|1T-A32B|
|**MoE**|[DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)|671B-A37B|
|**Hybrid**|[Zamba2](https://huggingface.co/Zyphra/models?search=zamba2)|1.2B, 2.7B, 7B|
|**Hybrid**|[Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df)|0.5B, 1.5B, 3B, 7B, 34B|
|**MoE + Hybrid**|[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)|80B-A3B|
|**MoE + Hybrid**|[MiniMax-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)|456B-A46B|


|Model|Architecture|Parameters|Training Tokens|Attention|Context Length (final)|Position Encoding|Precision|Init (std)|Optimizer|Max LR|LR Schedule|Warmup Steps|Batch Size|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|DeepSeek LLM 7B|Dense|7B|2T|GQA|4K|RoPE|BF16|0.006|AdamW|4.2×10⁻⁴|Multi-Step|2K|9.4M|
|DeepSeek LLM 67B|Dense|67B|2T|GQA|4K|RoPE|BF16|0.006|AdamW|3.2×10⁻⁴|Multi-Step|2K|18.9M|
|DeepSeek V2|MoE|236B (21B active)|8.1T|MLA|128K|Partial RoPE|-|0.006|AdamW|2.4×10⁻⁴|Multi-Step|2K|9.4M→37.7M (warmup 225B)|
|DeepSeek V3|MoE|671B (37B active)|14.8T|MLA|129K|Partial RoPE|FP8|0.006|AdamW|2.2×10⁻⁴|Multi-Step + Cosine|2K|12.6M→62.9M (warmup 469B)|
|MiniMax-01|MoE + Hybrid|456B (45.9 active)|11.4T|Linear attention + GQA|4M|Partial RoPE|-|Xavier init with deepnorm scaling|AdamW|2×10⁻⁴|Multi-Step|500|16M→32M→64M→128M|
|Kimi K2|MoE|1T (32B active)|15.5T|MLA|128K|Partial RoPE|BF16|likely 0.006|MuonClip|2×10⁻⁴|WSD|500|67M|
|OLMo 2 7B|Dense|7B|5T|MHA|4K|RoPE|BF16|0.02|AdamW|3×10⁻⁴|Cosine|2K|4.2M|
|SmolLM3|Dense|3B|11T|GQA|128K|NoPE|BF16|0.02|AdamW|2×10⁻⁴|WSD|2K|2.3M|
NoPE（No Position Embedding）（Kazemnejad 等人，2023）在没有任何显式位置编码的情况下训练 transformer，允许模型通过因果掩蔽和注意力模式隐式学习位置信息。作者表明，与 ALiBi 和 RoPE 相比，这种方法表现出更好的长度泛化。如果没有显式位置编码来外推超出训练长度，NoPE 自然会处理更长的上下文。但在实践中，与 RoPE 相比，NoPE 模型在短上下文推理和知识任务上往往表现出较弱的性能（Yang 等人）。这表明，虽然显式位置编码可能会限制外推，但它们为训练上下文长度内的任务提供了有用的归纳偏差。



**关键架构选择（以 SmolLM3 为例）：**

- ****注意力机制：采用 **GQA（Grouped Query Attention）**，在不显著损失效果的前提下压缩 KV 缓存、提升推理效率；同时整体性能与 MHA 基本匹配。****
    
- ******嵌入层：**共享输入/输出嵌入可减少参数量。对于小型模型而言，把有限的参数预算优先投入到**增加网络深度（层数）**，通常比不共享嵌入带来的收益更高。********
    
- ******位置编码：采用 **RNoPE**（交替使用 RoPE 和 NoPE）。该方案在保持短上下文表现的同时，为更高效的**长上下文外推**打好基础。******RoPE 层提供显式位置信息，并处理具有新近偏差的本地上下文，而 NoPE 层则改善了长距离信息检索。该技术最近用于 Llama4、Command A 和 SmolLM3 。
    
- ******数据处理：**启用**文档内掩码**——阻断训练序列中跨文档 token 的相互关注，有助于训练稳定性，并利于长上下文能力的扩展。********
    
- ******超参数：优化器选 **AdamW**，稳定可靠；学习率采用 **WSD（Warmup–Stable–Decay）** 调度，比 Cosine Decay 更灵活，便于在运行中调整总训练时长。******
    

### 第二阶段：预训练先打底

**数据是决定模型能力的首要因素。因此，专业的训练流程应该围绕一套有规划的多阶段训练课程（Curriculum）展开：**

前期用****覆盖更广、体量更大****的通用数据建立基础分布；****等到进入**学习率衰减的后期，**再注入**少量但高质量**的数据集（如 _Stack-Edu_、_FineMath4+_）。原因很简单：在低学习率阶段，模型对新信息的吸收更稳、更持久，高质量样本能在不破坏早期已学能力的前提下，最大程度地塑形模型的最终行为。

**多语言能力则从**分词器**开始就要算清账。我们不凭感觉选词表，而是用两个直接的指标来衡量：生育率（fertility）看一个词平均被切成多少个 token（越低说明越紧凑、高效），连续词比例（proportion of continued words）查看常见词是否被过度切碎（比例越低越好）。**

**在这些指标的约束下，SmolLM3 选择了 **Llama-3.2** 的分词器：它在多语言覆盖与模型体积之间给出了合理平衡，让训练出来的模型既不会被冗余 token 拖慢，也能在多语场景下保持有效表达。**

### 第三阶段：大规模跑起来

大规模训练的现实是：系统故障和性能瓶颈是必然发生的。SmolLM3 的 11T-token 训练也不例外：

#### 1. 吞吐量和数据瓶颈

一开始吞吐量莫名下滑，排查后发现根因在共享 **FSx** 存储。随着负载上升，FSx 会驱逐数据集碎片，训练端频繁读到“缺页”，IO 抖动把吞吐量直接拉崩。团队没有在远端存储上继续死磕，而是立刻把**完整的 24TB 语料**整体下放到各节点的本地 **NVMe RAID（/scratch）**，让训练直接从高速本地盘顺序读取，吞吐瞬间回到稳定高位。

问题并未就此结束。随后大家注意到：即便 IO 正常，吞吐量仍会随着 **step count** 增长而缓慢走低。把监控细节和火焰图对齐后，矛头指向了内置 **nanosets** 数据加载器——步数越大，它维护索引的开销越高，正把计算时间吞掉。与其继续给加载器打补丁，团队直接换用线上验证过的 **TokenizedBytes** 加载器，绕开索引热区，数据就地按字节切分与映射，训练立刻恢复到目标吞吐。

这两次处置背后的共同要点是：**先把瓶颈“物理化”**（IO/存储拓扑、加载器复杂度），再做最短路径的结构性改动；不要把算力浪费在与基础设施“拔河”上。

2. 最微妙的错误：张量并行性 Bug

训练推进到**1 万亿个 token**时，评测曲线开始掉队，与预期不符。团队没有立刻调参，而是按模块做系统性的排查：数据、优化器、学习率日程、评测管线都被逐一排除后，问题最终指向了最不显眼的一层——**Tensor Parallel（TP）** 的播种逻辑。

我们发现各个 TP 秩竟然**复用了同一个随机种子**，导致权重初始化在并行分片间**高度相关**；表面上一切正常，实际却让有效表示空间被“挤”在一起，训练效率被悄悄拖慢。确认根因后，团队果断在 **1T token** 处**重启训练**，并确保**每个 TP 秩使用独立种子**。

这次事故的教训很直白：哪怕所有大部件和小规模消融（包括 TP=2 的实验）看起来都正常，底层并行化设置中的微小错误也会在规模效应下被放大。

### 第四阶段：后训练最后打磨

预训练赋予了 SmolLM3 原始能力，但要把它变成真正可用的助手，还得靠**后训练**把这些能力收束成稳定、可控的行为。

SmolLM3 的目标是做一款混合推理（Hybrid Reasoning）的模型——用户可以通过系统消息切换思考模式（`/think` 打开链式思考，`/no_think` 直接给结论），而不是把所有请求都强行走一条推理路径。

整条后训练链路从 **SFT（监督微调）** 起步。它便宜、稳定，是所有后训练流程的起点。关键的做法是：**只在助手 token 上计算损失**，把用户输入完全掩码掉。这样训练信号就集中在“如何产出一段高质量回答”，而不会把“自动续写用户问题”的坏习惯学进去。

为了让模型在一开始就具备更强的推理骨架，团队会在 SFT 之前加一段 **mid-training（继续预训练）**：用大规模的“蒸馏推理数据”喂一遍，让模型先把推理范式学扎实，再进入指令对齐。实操上，这一步可以使推理基准的性能提高近三倍。

在具备基本行为后，接下来是**偏好优化（Preference Optimization）**，用成对或对比反馈把模型朝人类偏好“拧紧”。这一步对超参数非常敏感，**学习率通常要比 SFT 再低一个量级**，否则很容易把前面累出来的知识“洗掉”，出现灾难性遗忘。

最后，当任务允许自动验证时，可以引入 **RLVR** 一类的强化学习，让模型在闭环里自己找更优策略。不过这里存在奖励欺骗（Reward Hacking）的典型风险：模型可能通过没**被提示也狂写冗长 CoT** 来赚奖励，而不是更好地解决问题。对应的缓解方法是引入**超长完成惩罚（overlong completion penalty）的机制**，把输出长度分布拉回正常区间，逼着模型把思考用在刀刃上，而不是用字数刷分。

把以上环节连起来，就形成了清晰的一条线：**mid-training** 夯实推理底座，**SFT** 定义基本助手行为，**偏好优化**把风格与取舍对齐到人类偏好，必要时再用 **RLVR** 在自动可验场景里精修。通过系统消息控制的混合推理机制贯穿其上，让模型在需要时展开思考、在简单请求时干净利落地给出答案。

整个过程的要点只有一个：每一步都围绕“可控与可验证”来设计信号，让能力不只是更强，而且更可用。



# 参考资料

[The Smol Training Playbook: The Secrets to Building World-Class LLMs - a Hugging Face Space by HuggingFaceTB](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)

[214页内部秘籍《Smol训练手册：构建世界级LLMs的秘诀》](https://mp.weixin.qq.com/s/l1bb4FrWm4NlHOS4bgNCXA)

