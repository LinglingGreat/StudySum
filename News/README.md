## 2024.4

## 模型

以中文为核心而不是英文为核心依旧可以训一个"现在是不错，以后可能是很好"的LLM！
	我们开源了主要通过清洗CC和OCR的中文预训练语料800B MAP-CC，这是当前中文NLP开源社区最大的经过深度清洗的中文开源预训练数据集之一！
	我们开源了数据清理的全流程pipeline！
	我们开源了经过SFT与DPO后的CT-LLM！
	我们开源了一个较小的类MT-Bench中文效果衡量的Benchmark，CHC-Bench！
	我们开源了训练全流程的intermediate ckpts供大家分析！
	最重要的是，我们证明了中文为核心也可以训一个很不错的LLM，也可以在英文(其他语言上)涌现能力！
	CT-LLM is now available:
	paper: https://arxiv.org/pdf/2404.04167.pdf
	twitter: https://twitter.com/GeZhang86038849/status/1777163413183193296
	huggingface collection: https://huggingface.co/collections/m-a-p/chinese-tiny-llm-660d0133dff6856f94ce0fc6


### MOE相关的模型

[A21 Labs宣布开源520亿参数的全新混合专家大模型（Mixture of Experts，MoE）Jamba：单个GPU的上下文长度是Mixtral 8x7B的三倍](https://www.datalearner.com/blog/1051711641710005)

[开源大模型再上台阶：Databricks开源1320亿参数的混合专家大模型DBRX-16*12B，评测超Mixtral-MoE！](https://mp.weixin.qq.com/s/dkx0UU2PgR_CpaVa88KcZQ)

[重磅！阿里开源自家首个MoE技术大模型：Qwen1.5-MoE-A2.7B，性能约等于70亿参数规模的大模型Mistral-7B](https://mp.weixin.qq.com/s/XHFjybR3GIg4RIpBlndVGg)

[马斯克旗下xAI发布Grok-1.5，相比较开源的Grok-1，各项性能大幅提升，接近GPT-4！](https://www.datalearner.com/blog/1051711675314896#google_vignette)

[MoE架构模型大爆发！元象科技XVERSE开源256亿参数模型XVERSE-MoE-A4.2B，评测结果接近Llama1-65B](https://mp.weixin.qq.com/s/g1le9yGBSGwe6WqeeaVSEw)

### 长文本

[超长文本无损能力压测！中文大模型“大海捞针”首批结果公布](https://mp.weixin.qq.com/s/QgoRf2LB-7vc3vTFOHJkpw)

## 2024.4.22-26

开源了15T的高质量网络数据FineWeb，对2013-2014期间的cc进行过滤和去重：https://twitter.com/gui_penedo/status/1781953413938557276  #数据 

 [好样本，事半功倍：使用样本设计工程 (SDE) 来构造更好的大模型下游微调样本](https://mp.weixin.qq.com/s/QbiTwDvXLJ_Bbsi3xFOgkQ).          [Sample Design Engineering: An Empirical Study of What Makes Good Downstream Fine-Tuning Samples for LLMs](https://papers.cool/arxiv/2404.13033)   #微调 

多头专家混合模型：[[2404.15045v1] Multi-He. d Mixture-of-Experts](https://arxiv.org/abs/2404.15045v1)   #专家模型 

[**Continual Learning of Large Language Models: A Comprehensive Survey**](https://papers.cool/arxiv/2404.16789) 关于大型语言模型（Large Language Models，简称LLMs）在持续学习（Continual Learning，简称CL）领域的研究综述。它试图解决的问题是如何将预训练的LLMs有效地融入到动态变化的数据分布、任务结构和用户偏好中。主要挑战在于平衡模型的适应性和知识保留。当预训练的LLMs被定制以满足特定需求时，它们在以前知识领域的性能往往会显著下降，这种现象被称为“灾难性遗忘”（catastrophic forgetting）。 #ContinualLearning 

[Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding](https://papers.cool/arxiv/2404.16710) 训练使用层dropout和早期退出损失（Layer Dropout & Early Exit Loss）、使用早期退出进行推理、使用自我推测解码进行验证和纠正等方法来提升推理速度 #推理加速

[XC-Cache: Cross-Attending to Cached Context for Efficient LLM Inference](https://papers.cool/arxiv/2404.15420)  XC-CACHE模型通过使用跨注意力机制和紧凑的缓存策略，旨在提高大型语言模型在条件生成任务中的效率和性能，同时减少所需的存储空间。 #推理加速 

[Make Your LLM Fully Utilize the Context](https://papers.cool/arxiv/2404.16811) IN2训练利用了一个合成的长文本问答数据集，其中的答案需要（1）对合成长文本中的一个短片段（约128个令牌）进行细粒度的信息感知；（2）整合和推理来自两个或更多短片段的信息。通过在Mistral-7B模型上应用这种信息密集型训练，作者们提出了FILM-7B（FILl-in-the-Middle）模型。通过合成长文本数据的训练可以泛化到真实世界的场景中，并且FILM-7B在训练过程中没有损害短文本能力。 #长文本 

[TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](https://papers.cool/arxiv/2404.11912) TriForce通过利用原始模型权重和动态稀疏KV缓存，以及一个轻量级模型来进一步推测，从而实现了对长序列生成的高效支持。论文还展示了TriForce在不同设置下的性能，包括在单个A100 GPU上达到2.31倍的加速，以及在两个RTX 4090 GPU上实现7.78倍的加速。 #推理加速 

[Fewer Truncations Improve Language Modeling](https://papers.cool/arxiv/2404.10830) 这篇论文试图解决大型语言模型（LLM）训练中的数据截断问题。在传统的训练方法中，输入文档被连接在一起，然后分割成等长的序列，以适应模型的上下文长度。这种方法虽然高效，但会导致许多文档被不完整地截断，从而损害了数据的完整性。截断会影响模型学习逻辑连贯、事实一致的内容的能力，因为模型无法获取完整的上下文信息。
为了解决这个问题，论文提出了一种名为“Best-fit Packing”的方法，这是一种可扩展且高效的方法，通过长度感知的组合优化将文档打包进训练序列。这种方法完全消除了不必要的截断，同时保持了与连接方法相同的训练效率。实证结果表明，使用Best-fit Packing训练的模型在多种任务上表现出更好的性能，并且在减少封闭域幻觉方面效果显著。 #预训练 

[Token-level Direct Preference Optimization](https://papers.cool/arxiv/2404.11999) TDPO试图通过在token级别上直接优化策略，同时控制KL散度，来提高语言模型与人类偏好的对齐度，并保持生成响应的多样性。论文通过在多种文本任务上的实验结果表明，TDPO在平衡对齐度和生成多样性方面优于DPO和基于PPO的RLHF方法。 #alignment 

[SnapKV: LLM Knows What You are Looking for Before Generation](https://papers.cool/arxiv/2404.14469) SnapKV提供了一种有效的方法来压缩KV缓存，同时保持了LLMs在处理长文本输入时的性能。 #长文本 #推理加速 

[Sequence can Secretly Tell You What to Discard](https://papers.cool/arxiv/2404.15949)  这篇论文试图解决大型语言模型（LLMs）在部署时面临的一个关键内存瓶颈问题，即关键值（KV）缓存的内存占用问题。尽管LLMs在多种自然语言处理任务上表现出色，但它们需要大量的GPU内存，并消耗大量的计算资源。除了模型权重外，KV缓存占用的内存随着序列长度的增加而线性增长，成为推理过程中的主要瓶颈。
论文中指出，即使是一个参数量为7亿的模型，批量大小为128，序列长度为4096，其KV缓存的内存占用也高达256GB，远远超过了模型本身14GB的内存消耗。因此，研究者们提出了一种优化KV缓存的方法，以减少其内存占用，同时尽量保持模型性能。
为了解决这个问题，论文提出了一种名为CORM（Cache Optimization with Recent Message）的方法，这是一种KV缓存逐出策略，它基于最近的查询注意力信息动态保留对推理重要的键值对，而无需对模型进行微调。通过广泛的实验评估，研究表明CORM能够在不显著降低性能的情况下，将KV缓存的推理内存使用量减少高达70%。 #推理加速 

[From Complex to Simple: Enhancing Multi-Constraint Complex Instruction Following Ability of Large Language Models](https://papers.cool/arxiv/2404.15846)  使用包含多个约束的指令来训练LLMs可以增强它们对复杂指令的理解，尤其是对低复杂度级别的指令。论文提出了一种基于歧视的方法（Discrimination method）来生成高质量的组合数据（compositional data），并通过对比方法结合强化学习微调（RLFT）来利用这些数据。这种方法比直接使用高级模型生成输出更有效。论文介绍了如何从三个广泛使用的指令调整数据集中收集种子指令，并将多个约束整合到这些种子指令中，以生成复杂的指令。 #指令遵循

[Let's Think Dot by Dot: Hidden Computation in Transformer Language Models](https://papers.cool/arxiv/2404.15758) 这篇论文试图深入理解Transformer语言模型中链式思考和填充标记的作用，以及它们如何影响模型的计算能力和性能。同时，它也提出了关于模型透明度和未来发展方向的思考。 #cot 

[Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs](https://papers.cool/arxiv/2404.15676) 这篇论文提供了一个全面的调查，探讨了大型语言模型（LLMs）中的一种称为“Chain-of-X（CoX）”的方法。CoX方法受到了“Chain-of-Thought（CoT）”提示方法的启发，CoT方法通过将复杂问题分解为一系列中间子任务来显著提高LLMs的推理能力。CoX方法将CoT的概念扩展到了更广泛的任务和领域，通过构建不同组件的序列来解决各种挑战。 #cot 

[Retrieval Head Mechanistically Explains Long-Context Factuality](https://papers.cool/arxiv/2404.15574) 在长文本上下文中，大型语言模型（LLMs）是如何获取并检索相关信息的。论文提出了“检索头”这一概念，这是模型中负责从长文本上下文中检索相关信息的特殊类型的注意力头（attention heads）。这些检索头对于模型能否成功找到并使用输入文本中的信息至关重要。 #长文本 

[FireAttention — Serving Open Source Models 4x faster than vLLM by quantizing with \~no tradeoffs](https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs) 没有开源，需要购买。 #推理加速 

[LLM能否依据角色的过去预测未来？一篇有趣的研究](https://mp.weixin.qq.com/s/wwuayFtqWuy0ByEe4HVf8w)  #roleplay 









