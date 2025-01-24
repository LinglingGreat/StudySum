---
title: aya-expanse
created: 2025-01-24
tags:
  - 多语言
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## 背景
随着 Aya Expanse 系列的发布，该系列具有[8B](https://huggingface.co/CohereForAI/aya-expanse-8b)和[32B](https://huggingface.co/CohereForAI/aya-expanse-32b)参数模型，我们正在解决人工智能领域最紧迫的挑战之一：缺乏能够与单语模型相媲美的高性能多语言模型。虽然人工智能取得了巨大的进步，但跨多种语言的模型性能仍然存在巨大差距。Aya Expanse 是 C4AI 多年专注研究的成果[——](https://cohere.com/research)数据[套利](https://arxiv.org/abs/2408.14960)、[多语言偏好训练](https://arxiv.org/abs/2407.02552)、[安全调整](https://arxiv.org/abs/2406.18682)和[模型合并](https://arxiv.org/abs/2410.10801)。

这些综合突破在多语言性能方面取得了新的领先成绩。我们根据一系列评估来评估我们的模型，包括[Arena-Hard-Auto](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto-v0.1)数据集（[论文](https://arxiv.org/abs/2406.11939)），该数据集已翻译成 23 种语言，我们[已发布这些语言供其他人使用](https://huggingface.co/datasets/CohereForAI/m-ArenaHard)。在成对比较中，[Aya Expanse 32B 的](https://huggingface.co/CohereForAI/aya-expanse-32b)表现优于 Gemma 2 27B、Mistral 8x22B 和 Llama 3.1 70B（一个比其大小高出 2 倍多的模型），为多语言性能树立了新的领先水平。我们还发布了[Aya Expanse 8B](https://huggingface.co/CohereForAI/aya-expanse-8b)，其表现优于其参数类别中领先的开放权重模型，例如 Gemma 2 9B、Llama 3.1 8B 和最近发布的 Ministral 8B，胜率从 60.4% 到 70.6% 不等。我们观察到在难度较低的评估中收益甚至更大。

[  
![Aya Expanse 8B 胜率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-8b-win-rates.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-8b-win-rates.png)[![Aya Expanse 8B 语言特定胜率 vs Gemma 2 9B](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-8b-language-specific-win-rates-vs-gemma-2-9b.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-8b-language-specific-win-rates-vs-gemma-2-9b.png)

我们将这两种模型作为开放权重发布给研究社区，希望这能进一步加速多语言的进展。

[![Aya Expanse 32B 胜率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-32b-win-rates.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/aya-expanse-32b-win-rates.png)


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 避免合成数据中的模型崩溃

从教师模型池中进行策略性抽样。这种方法具有重要的意义，因为它挑战了传统上依赖单一教师模型来生成合成数据的做法。相反，_数据套利_利用了模型池之间的性能差异。虽然这种技术适用于任何领域，但它特别适合多语言环境，因为在多语言环境中，缺乏一个在所有语言方面都表现优异的普遍有效的教师会带来重大挑战。在创建高质量的合成多语言数据集时，_多语言套利_通过利用多样化的模型池对数据分布的不同部分进行策略性抽样以改进多语言生成，从而证明其价值。

我们首先为语言组训练一个模型池，并使用_**仲裁器**_来评估和选择最佳生成。这里的仲裁器是一个内部奖励模型 (RM)，用于对模型生成进行评分。在基于奖励的路由中，对于给定语言的每个提示，我们从池中的所有模型生成完成，并使用奖励模型对其进行评分。得分最高的完成被选为该提示的最终完成。我们的 8B 模型，即使在使用多语言套利训练的 SFT 阶段，与[之前的 Aya 23 模型](https://arxiv.org/abs/2405.15032)相比，与 Gemma 2 9B 相比，胜率提高了 9.1% 以上，证明了这种方法在利用跨语言的多样化模型优势方面的有效性。

[![逐步提高 Gemma 2 9B 的胜率](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/step-by-step-improvements-in-win-rates-against-gemma-2-9b.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/step-by-step-improvements-in-win-rates-against-gemma-2-9b.png)





## 利用全局偏好不断改进

在监督微调之后，与人类偏好保持一致是训练当今最先进的 LLM 的关键步骤。尽管被广泛采用，但众所周知，[在单语环境中进行偏好训练已经具有挑战性](https://arxiv.org/abs/2307.15217)。在多语环境中最大化偏好训练的收益带来了更多的挑战。现有的绝大多数偏好数据集都是英语，而现有的少数多语言偏好数据集通常质量较低。此外，众所周知，同时对多种不同语言进行建模是一个困难的优化问题，因为单纯地优化某些语言的性能往往会导致其他语言性能的下降。

在[_LHF 能说多种语言：为 LLM 解锁多语言偏好优化_](https://arxiv.org/abs/2407.02552)中，我们利用一种新颖的合成数据生成技术来构建高质量的多语言偏好数据对，方法是对比来自高性能多语言 LLM 的语言完成与由较弱模型生成的英语翻译成的较低质量完成。这使我们的模型避免生成低质量的多语言完成，这些完成通常包含不良伪像，例如由糟糕的翻译引入的伪像。我们表明，这种方法可以显著提高所有语言的性能，并且通常还可以提高偏好训练数据中未包含的语言的性能。

虽然这项[研究](https://arxiv.org/abs/2407.02552)还表明，使用在线数据进行偏好训练的效果优于离线训练，但在 Aya Expanse 的训练过程中，我们发现，先使用离线数据进行偏好训练，然后再使用在线数据进行偏好训练，这种组合的效果优于单独使用在线或离线训练。在第一个偏好训练阶段，我们通过从套利阶段选取最高和最低奖励响应作为选定和拒绝的完成，对精选数据进行训练，这使得 DPO 训练的第一阶段_处于离线状态_。

离线偏好训练之后，我们运行_在线_迭代 DPO，其中我们从上次迭代期间训练的模型中为每个提示抽取多个在线代，使用奖励模型对这些代进行排序，然后进一步训练这些偏好对。对于这两个模型，我们将此过程重复 3 次迭代，因为我们发现超过 3 次迭代会导致最小的收益，但需要重新调整正则化系数 (beta) 等参数，有时还会引入奖励黑客行为。总体而言，对于 Aya Expanse 8B，在使用套利训练的模型之上结合离线和在线偏好训练，导致 Gemma 2 9B 的胜率额外提高 7.1%。


## 通过模型合并最大化性能

任何训练后（和训练前）流程中都会反复出现一个问题，无论它是由 SFT 这样的单个阶段组成，还是由更复杂的多阶段优化流程（例如我们上面的流程）组成，都是选择正确的数据混合进行训练。这个过程非常复杂，需要在微调超参数和数据组合方面付出相当大的努力。合并多个模型是一种以较低的总计算成本实现复杂多任务处理的替代方法。在 Aya Expanse 中，我们直接以我们最近的研究论文《[_混合数据还是合并模型？优化多样化多任务学习》的成果为基础_](https://arxiv.org/abs/2410.10801)，并在套利阶段和偏好训练的每次迭代中应用合并。

当训练多个独立模型并希望合并时，最大化检查点之间的多样性非常重要。但是，这应该与确保池中的每个模型都达到高性能相平衡。为了平衡这些目标，我们通过为不同的语系训练模型来最大化检查点之间的多样性。这利用了[跨语言迁移](https://aclanthology.org/2024.acl-long.845.pdf)，它通常可以提供显著的性能优势，同时确保语言差异在检查点之间提供足够的区分。

简单来说，我们可以对每种语言的模型进行拆分训练，然后进行合并，但这无法实现跨语言迁移所带来的好处。为了提高合并的稳健性，我们在每个集群中都包含了一些共同的语言（这里是英语、西班牙语和法语）。在最终方案中，我们使用了多个阶段来合并在不同数据集群上训练的运行，并在同一运行中使用检查点。

除了加权线性平均之外，我们还尝试了多种合并技术，即[SLERP](https://dl.acm.org/doi/10.1145/325165.325242)、[TIES-merging](https://arxiv.org/pdf/2306.01708)和[DARE-TIES](https://arxiv.org/abs/2311.03099)。但是，我们发现加权平均是最一致的方法。因此，我们在整个流程中使用加权平均。有趣的是，我们观察到 35B 规模的合并收益明显高于 8B 规模的合并收益，高达 3 倍。这与[最近的研究](https://arxiv.org/pdf/2410.03617)表明合并在规模上更有效的研究一致。

## 整合所有

[![成分](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/components.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/components.png)

这些图表展示了我们端到端的训练后流程，从而实现了前面讨论的逐步收益。回顾 Aya 模型系列的发展历程，真的很特别，从[Aya 101](https://huggingface.co/CohereForAI/aya-101)开始，伴随着[Aya Collection](https://huggingface.co/datasets/CohereForAI/aya_collection)，它拓展了开源协作的极限，到现在结合了关键开放基础研究问题的稳步进展，为多语言性能树立了新标准。

[![综合](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/combined.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-expanse/combined.png)


## 主要收获


## 参考资料

[A Deepdive into Aya Expanse: Advancing the Frontier of Multilinguality](https://huggingface.co/blog/aya-expanse)

