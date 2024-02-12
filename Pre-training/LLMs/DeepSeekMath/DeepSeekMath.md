---
title: DeepSeekMath
created: 2024-02-12
tags:
  - 数学大模型
type: 论文
papername: DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - DeepSeek
  - 北京大学
  - 清华
---

## 论文基本信息

标题：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

作者：

链接：

代码： https://github.com/deepseek-ai/DeepSeek-Math

框架图：


## 背景
![](img/Pasted%20image%2020240212114659.png)


## 构建数据集

创建了 DeepSeekMath 语料库，这是一个包含 120B 个数学标记的大规模高质量预训练语料库。该数据集是使用基于 fastText 的分类器从 Common Crawl (CC) 中提取的。

使用OpenWebMath作为种子语料。在初始迭代中，采样50万的 OpenWebMath中的实例作为正例，采样50万的CC中的网页数据作为负例，来训练分类器。词向量维度256，学习率0.1，单词n-gram的最大长度为3，word occurrences最大为3，epoch为3。

为了减少原始 Common Crawl 的大小，我们采用了基于 URL 的重复数据删除和近重复数据删除技术，生成了 40B 的 HTML 网页。然后，我们使用 fastText 模型从重复数据删除的 Common Crawl 中召回数学网页。为了过滤掉低质量的数学内容，我们根据 fastText 模型预测的分数对收集的页面进行排名，并且只保留排名靠前的页面。保留的数据量是通过对前 40B、80B、120B 和 160B 代币进行预训练实验来评估的。在第一次迭代中，我们选择保留前 40B 代币。

在数据收集的第一次迭代之后，许多数学网页仍未收集，主要是因为 fastText 模型是在一组缺乏足够多样性的正例上进行训练的。因此，我们确定了额外的数学网络资源来丰富种子语料库，以便我们可以优化 fastText 模型。具体来说，我们首先将整个 Common Crawl 组织成不相交的域；域被定义为共享相同基本 URL 的网页。对于每个域，我们计算在第一次迭代中收集的网页的百分比。收集超过 10% 网页的域被归类为与数学相关的域（例如 mathoverflow.net）。随后，我们手动标注这些已识别域内的数学内容相关的 URL（例如 mathoverflow.net/questions ）。链接到这些 URL 的网页（尚未收集）将被添加到种子语料库中。这种方法使我们能够收集更多正面示例，从而训练改进的 fastText 模型，使其能够在后续迭代中调用更多数学数据。经过四次数据收集迭代后，我们最终得到了 3550 万个数学网页，总计 120B 个代币。在第四次迭代中，我们注意到第三次迭代中已经收集了近 98% 的数据，因此我们决定停止数据收集。

为了避免基准污染，我们遵循Guo等人的做法过滤掉包含 GSM8K 和 MATH 等英语数学基准以及 中文基准CMATH和 AGIEval等的问题或答案的网页。过滤标准如下：任何包含与评估基准中的任何子字符串完全匹配的 10 gram字符串的文本段都会从我们的数学训练语料库中删除。对于短于 10 gram但至少有 3 gram的基准文本，我们采用精确匹配来过滤掉受污染的网页。

![](img/Pasted%20image%2020240212120811.png)

和其他数学数据集相比，DeepSeekMath数据集如何呢？通过预训练DeepSeekLLM 1.3B实验对比。
MathPile (Wang et al., 2023c): a multi-source corpus (8.9B tokens) aggregated from textbooks, Wikipedia, ProofWiki, CommonCrawl, StackExchange, and arXiv, with the majority (over 85%) sourced from arXiv; 

OpenWebMath (Paster et al., 2023): CommonCrawl data filtered for mathematical content, totaling 13.6B tokens; 

Proof-Pile-2 (Azerbayev et al., 2023): a mathematical corpus consisting of OpenWebMath, AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers (28.0B tokens). When experimenting on Proof-Pile-2, we follow Azerbayev et al. (2023) to use an arXiv:Web:Code ratio of 2:4:1.

我们在每个数学语料库上分别训练 150B 个 token 的模型。所有实验均使用高效、轻量级的 HAI-LLM (High-flyer, 2023) 训练框架进行。𝛽1 = 0.9, 𝛽2 = 0.95, and weight_decay = 0.1, along with a multi-step learning rate schedule where the learning rate reaches the peak after 2,000 warmup steps, decreases to its 31.6% after 80% of the training process, and further decreases to 10.0% of the peak after 90% of the training process. We set the maximum value of learning rate to 5.3e-4, and use a batch size of 4M tokens with a 4K context length.

![](img/Pasted%20image%2020240212122436.png)

![](img/Pasted%20image%2020240212122548.png)

DeepSeekMath语料库质量高，涵盖多语言数学内容（中文数学基准的改进），规模最大。

## 预训练

DeepSeekMath-Base 使用 DeepSeek-Coder-Base-v1.5 7B进行初始化，训练了500B token。56% is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github code, and the remaining 10% is natural language data from Common Crawl in both English and Chinese. 训练参数设置和1.3B的模型一样，只有最大学习率设置成了4.2e-4，batch size是10M tokens.

我们对DeepSeekMathBase 7B的数学能力进行全面评估，重点考察其不依赖外部工具产生自成一体的数学解、利用工具解决数学问题以及进行形式化定理证明的能力。除了数学之外，我们还提供了基础模型的更一般的概况，包括其自然语言理解、推理和编程技能的表现。

![](img/Pasted%20image%2020240212123200.png)

我们的基础模型 DeepSeekMath-Base 7B 在 GSM8K 上达到了 64.2%，在竞赛级 MATH 数据集上达到了 36.2%，优于 Minerva 540B。

![](img/Pasted%20image%2020240212123236.png)

![](img/Pasted%20image%2020240212123311.png)

如表 4 所示，相比其前身 DeepSeek-Coder-Base-v1.5，DeepSeekMath-Base 7B 在 MMLU 和 BBH 上的性能显着增强，说明了数学训练对语言的积极影响理解和推理。此外，通过包含用于持续训练的代码令牌，DeepSeekMath-Base 7B 有效地保持了 DeepSeek-Coder-Base-v1.5 在两个编码基准上的性能。总体而言，DeepSeekMath-Base 7B 在三个推理和编码基准上显着优于通用模型 Mistral 7B。

## SFT

预训练后，我们将数学指令调整应用于 DeepSeekMath-Base，其中包括chain-of-thought、program-of-thought、和tool-integrated reasoning数据，不同数学领域的英文和中文问题，一共776K。

我们使用工具集成解决方案标注 GSM8K 和 MATH 问题，并采用 MathInstruct 的子集以及 Lila-OOD 训练集，其中问题通过 CoT 或 PoT 解决。我们的英文数据涵盖数学的各个领域，例如代数、概率、数论、微积分和几何。

我们收集了涵盖线性方程等 76 个子主题的中国 K-12 数学问题，并以 CoT 和工具集成推理格式标注了解决方案。

 DeepSeekMath-Instruct 7B是在 DeepSeekMath-Base 的基础上进行数学指令调优的。训练示例随机连接，直到达到 4K 令牌的最大上下文长度。我们对模型进行 500 个步骤的训练，批量大小为 256，恒定学习率为 5e-5。

如表5所示，在不允许使用工具的评估设置下，DeepSeekMathInstruct 7B表现出强大的分步推理性能。值得注意的是，在竞赛级别的 MATH 数据集上，我们的模型绝对超过所有开源模型和大多数专有模型（例如 Inflection-2 和 Gemini Pro）至少 9%。即使对于更大的模型（例如 Qwen 72B）或通过以数学为重点的强化学习（例如 WizardMath-v1.1 7B）专门增强的模型也是如此。虽然 DeepSeekMath-Instruct 在 MATH 方面可与中国专有模型 GLM-4 和 Baichuan-3 相媲美，但它的性能仍然低于 GPT-4 和 Gemini Ultra。在允许模型集成自然语言推理和基于程序的工具使用来解决问题的评估设置下，DeepSeekMath-Instruct 7B 在 MATH 上的准确率达到 60%，超越了所有现有的开源模型。在其他基准测试中，我们的模型与 DeepSeek-LLM-Chat 67B 具有竞争力，后者是之前最先进的模型的 10 倍。

![](img/Pasted%20image%2020240212124140.png)
![](img/Pasted%20image%2020240212124153.png)

## 强化学习

我们引入了Group Relative Policy Optimization（GRPO），这是Proximal Policy Optimization（PPO）的一种变体强化学习（RL）算法。 GRPO 放弃了批评家模型，而是根据小组分数(针对同一问题产生的多个采样输出的平均奖励)估计基线，从而显着减少了训练资源。

![](img/Pasted%20image%2020240212130521.png)

![](img/Pasted%20image%2020240212130827.png)

RL 的训练数据是来自 SFT 数据的与 GSM8K 和 MATH 相关的思维链格式问题，由大约 144K 个问题组成。我们排除了其他 SFT 问题，以调查 RL 对整个 RL 阶段缺乏数据的基准的影响。我们构建了以下奖励模型的训练集（Wang et al., 2023b）。我们基于 DeepSeekMath-Base 7B 训练初始奖励模型，学习率为 2e-5。对于GRPO，我们将策略模型的学习率设置为1e-6。 KL系数为0.04。对于每个问题，我们采样 64 个输出。最大长度设置为 1024，训练批量大小为 1024。策略模型在每个探索阶段后仅进行一次更新。我们根据 DeepSeekMath-Instruct 7B 的基准评估 DeepSeekMath-RL 7B。对于 DeepSeekMath-RL 7B、GSM8K 和具有思想链推理的 MATH 可以视为域内任务，所有其他基准测试可以视为域外任务。

通过仅使用英语指令调优数据的子集，GRPO 比强大的 DeepSeekMath-Instruct 获得了实质性改进，包括域内（GSM8K：82.9％→88.2％，MATH：46.8％→51.7％）和域外强化学习阶段的领域数学任务（例如 CMATH：84.6% → 88.8%）。

## 讨论

值得注意的是，在本节中引用 DeepSeekMath 语料库时，我们使用数据收集过程第二次迭代中的 89B-token 数据集。

**1.1 代码训练有利于数学推理**

我们注意到，与一般的 LLM 相比，从代码训练模型开始是更好的选择（数学训练之前的代码训练可以提高模型在使用或不使用工具的情况下解决数学问题的能力）。

![](img/Pasted%20image%2020240212131519.png)

![](img/Pasted%20image%2020240212131642.png)

general tokens (sampled from a large-scale general corpus created by DeepSeek-AI)

代码训练之后的数学训练会降低编码性能。

无论是在两阶段训练还是一阶段训练设置下，代码训练都有利于程序辅助数学推理。如表6所示，在两阶段训练设置下，仅代码训练就已经显着增强了使用Python解决GSM8K和MATH问题的能力。第二阶段的数学训练有进一步的提高。有趣的是，在一阶段训练设置下，混合代码标记和数学标记有效缓解了两阶段训练带来的灾难性遗忘问题，并且还协同编码（表7）和程序辅助数学推理（表6）。

代码训练还可以在不使用工具的情况下提高数学推理能力。在两阶段训练设置下，代码训练的初始阶段已经产生了适度的增强。它还可以提高后续数学训练的效率，最终达到最佳表现。然而，将代码标记和数学标记结合起来进行一阶段训练会在不使用工具的情况下损害数学推理。一种猜测是，DeepSeek-LLM 1.3B 由于其规模有限，缺乏同时完全同化代码和数学数据的能力。

**1.2 ArXiv 论文在提高数学推理方面似乎无效**

尽管对 arXiv 论文进行训练很常见，尤其是在许多数学相关论文中，但它对本文采用的所有数学基准并没有带来显着的改进。

![](img/Pasted%20image%2020240212131825.png)

MathPile超过85%是arxiv，8.9B tokens。ArXiv-RedPajama是28B tokens

当在仅限 arXiv 的语料库上进行训练时，这两个模型在本研究中使用的不同复杂性的各种数学基准上都没有表现出显着的改进甚至恶化。这些基准包括 GSM8K 和 MATH（表 8）等定量推理数据集、MMLU-STEM（表 8）等多项选择题以及 miniF2F 等形式数学。

待研究：arXiv 代币对本研究中未包含的特定数学相关任务的影响，例如定理的非正式化，即将正式陈述或证明转换为其非正式版本； arXiv 代币与其他类型数据结合时的效果； arXiv 论文的好处是否会在更大的模型规模上体现出来。

**Insights of Reinforcement Learning**

我们提供了一个统一的范式来理解不同的方法，例如拒绝采样微调（RFT）（Yuan et al., 2023a）、直接偏好优化（DPO）（Rafailov et al., 2023）、PPO 和 GRPO。基于这样一个统一的范式，我们发现所有这些方法都被概念化为直接或简化的强化学习技术。我们还进行了广泛的实验，例如在线与离线训练，结果与过程监督，单轮与迭代强化学习等，深入研究这一范式的基本要素。

![](img/Pasted%20image%2020240212134040.png)

在本文中，我们基于指令调优数据的子集进行强化学习，并且在指令调优模型上实现了显着的性能增强。进一步解释为什么强化学习有效。我们在两个基准上评估了 Instruct 和 RL 模型的 Pass@K 和 Maj@K 准确性。如图 7 所示，RL 增强了 Maj@K 的性能，但没有增强 Pass@K。这些发现表明，强化学习通过使输出分布更加鲁棒来增强模型的整体性能，换句话说，这种改进似乎归因于增强 TopK 的正确响应，而不是基本能力的增强。同样，（Wang et al., 2023a）发现了 SFT 模型中推理任务的错位问题，表明SFT模型的推理性能可以通过一系列偏好调整策略来提高

![](img/Pasted%20image%2020240212135146.png)



## 未来方向



## 主要收获


## 参考资料
