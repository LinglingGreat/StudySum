## 好文


[垂直领域大模型的思考 - 知乎](https://zhuanlan.zhihu.com/p/652645925)   #垂域模型

[MiniCPM：揭示端侧大语言模型的无限潜力](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)   #端侧模型

[符尧：别卷大模型训练了，来卷数据吧！](https://mp.weixin.qq.com/s/jnQjMDbSV2L9OzA7rFWAUA)  #数据 #fuyao

[XG纪要 | Compression theory for Ms](https://mp.weixin.qq.com/s/DSfZzOFexAfSSrAj6d3q5w) #fuyao

[如何从零开始训练大模型（minicpm分享&讨论） - 知乎](https://zhuanlan.zhihu.com/p/686664720) #预训练 

[ChatGPT 负责人：GPT-4 越来越聪明是因为 post-training，大模型短期没有数据瓶颈](https://mp.weixin.qq.com/s/I_-RXtMAy5YXPRQ7XsBNRQ) 

[大模型训练十戒](https://mp.weixin.qq.com/s/kSSXKPxTyBj9QFUnnMcEDA)

[ [2024智源大会速览] 大语言模型和AGI篇](https://zhuanlan.zhihu.com/p/706173121)

[LLM Continue Pretrain（2024版）](https://mp.weixin.qq.com/s/Uq8EPuh9AgOb-d3ZthoK9A)   #ContinualLearning 

[预训练LLM副本攻略：结构优化与超参数调整](https://zhuanlan.zhihu.com/p/707993464)   #预训练 

[Extrinsic Hallucinations in LLMs | Lil'Log](https://lilianweng.github.io/posts/2024-07-07-hallucination/) #幻觉

[梳理一下MiniCPM](https://mp.weixin.qq.com/s/Tm06k77DNW0KI2-eqy6niA)

[从零训练的 1B 以下小模型汇总](https://mp.weixin.qq.com/s/d1ypjLwaJKEV8Edfz83tVw)

[Exploring the Potential of In-Context Learning: New Pathways for Enhancing Chat-Based Large Language Model Performance (In Refinement)](https://www.notion.so/c31d141411be4d0eb50473fe6abae1db?v=50264a9824494b6c836ba0c6f3bebd2f)

[Thinking about High-Quality Human Data | Lil'Log](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/)

[GitHub - PeterH0323/Streamer-Sales: Streamer-Sales 销冠 —— 卖货主播 LLM 大模型🛒🎁，一个能够根据给定的商品特点从激发用户购买意愿角度出发进行商品解说的卖货主播大模型。🚀⭐内含详细的数据生成流程❗ 📦另外还集成了 LMDeploy 加速推理🚀、RAG检索增强生成 📚、TTS文字转语音🔊、数字人生成 🦸、 Agent 使用网络查询实时信息🌐、ASR 语音转文字🎙️](https://github.com/PeterH0323/Streamer-Sales)




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

## 2024.5.10-5.31

#预训练 我们宣布了第一个工业级的透明中英文双语大模型-Neo的开源，我们提供了全部的4.7T 预训练数据，训练pipeline，基于spark的预训练数据pipeline，OCR pipeline，以及复现的deepseek-math提出的迭代地从预训练数据中召回高质量数据的直接可用的pipeline。我们的模型在7B大小，MMLU达到约58，CMMLU和C-Eval约为55，GSM-8k达到50，ARC-C约为68，HumanEval约为25，与OLMo和Amber相比，Neo作为基座基本达到了工业级comparable的水准
twitter：https://twitter.com/GeZhang86038849/status/1788874345927889203
hf model collection: https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04
dataset: https://huggingface.co/datasets/m-a-p/Matrix
github: https://github.com/multimodal-art-projection/MAP-NEO


 #模型融合 我们用ExPO增强的模型上传两周已经8k+ downloads了
https://huggingface.co/collections/chujiezheng/weak-to-strong-extrapolation-expedites-alignment-662b69fbe7850e722e10ff70.  

#rlhf #alignment 我们最近做的开源的 online iterative RLHF recipe, https://x.com/CaimingXiong/status/1790379121719361776 . TL;DR: 我们从pretrained 出发，纯用开源数据集做了 (1) SFT, (2) reward modeling, preference modeling, (3) online iterative DPO (实际code 实现也可以换成slic ipo). 最终得到的reward model 和 preference model 现在是 reward benchmark 上开源模型的 SOTA，并且最终得到的模型和meta自己做的RLHF模型在测试的academic benchmark上 comparable 并且在指令跟随的benchmark上还要更好一些。我们开源了代码，模型，数据，和具体用的超参数，很轻松就能复现，technical report 在RLHF Workflow: From Reward Modeling to Online RLHF https://arxiv.org/pdf/2405.07863.  [仅靠开源数据复刻出LLaMA3指令学习效果，在线迭代RLHF全流程解决方案来了](https://mp.weixin.qq.com/s/bRxdSCCPIrgNBgtDfyzhAA)

#长文本 #推理加速   [Full Stack Transformer Inference Optimization Season 2: Deploying Long-Context Models](https://yaofu.notion.site/Full-Stack-Transformer-Inference-Optimization-Season-2-Deploying-Long-Context-Models-ee25d3a77ba14f73b8ae19147f77d5e2)


#sft 我们今年的NAACL做过一个让大模型意识到自己的知识边界的工作，通过sft 使得模型 尽量只回答对于自己知识边界以内的问题，对于知识边界以外的拒绝回答，teach LLMs to say i dont know.  以此来减少幻觉对真实世界造成的影响。 算是抛砖引玉，欢迎关注~  https://arxiv.org/abs/2311.09677

[Apr 2024 | Llama 3 Opens the Second Chapter of the Game of Scale.](https://yaofu.notion.site/Apr-2024-Llama-3-Opens-the-Second-Chapter-of-the-Game-of-Scale-efff1c0c185f4008af673b78faf83b61)

#数据集 #资源汇总[Artifacts Log 1: Announcement, Llama 3 fine-tunes, SOTA reward model, human prompt datasets](https://www.interconnects.ai/p/f1b83a34-18cd-4507-b4b0-560902eb3275) 包括模型、数据集介绍等

#训练参数 [大 Batch 训练 LLM 探索 - 知乎](https://zhuanlan.zhihu.com/p/666997679) batch

#推理加速  [图解Mixtral 8 \* 7b推理优化原理与源码实现 - 知乎](https://zhuanlan.zhihu.com/p/691066049)

#推理加速  [腾讯PCG自研高性能大语言模型推理引擎「一念LLM」正式开源](https://mp.weixin.qq.com/s/rlyJwaOfDfNYMZEH7kfKGA)

Introducing OpenChat 3.6.   Surpassed official Llama3-Instruct—with 1-2M synthetic data compared to ~10M human labels. GPTs are close to limits—excel at generation but fall short at complex tasks !We are training next gen—capable of deterministic reasoning and planning !   Explore OpenChat-3.6 (20240522 Llama 3 Version): HuggingFace:https://huggingface.co/openchat/openchat-3.6-8b-20240522  Live Demo: https://openchat.team GitHub: https://github.com/imoneoi/openchat
我们为 LLM 开发了一种新的连续预训练方法 Meta-Alignment，它实现了与 Meta 使用 Llama3 Instruct 进行的广泛 RLHF 训练类似的结果。此过程在数据和计算方面都非常高效，主要使用合成数据，占用数据集的不到 10%


## 2024.6.1-6.7

#预训练 [高能力全透明双语大语言模型MAP-Neo完全开源，开放所有细节！](https://mp.weixin.qq.com/s/hKdufVyzAhxFKFIScT9YQA) 

#专家模型 [MoE门控网络最新创新！性能对标Llama 3，源2.0-M32大幅提升模型算力效率](https://mp.weixin.qq.com/s/Z1hK9Xds9XUnmPHqvKrsRw)

#专家模型 [面壁新模型：早于Llama3、比肩 Llama3、推理超越 Llama3！](https://mp.weixin.qq.com/s/BAeFq-jXuyXiGMF7MMy5qw)

#推荐系统 [当推荐系统遇见大语言模型：通往未来的三条路径](https://mp.weixin.qq.com/s/H2Relpo8FW6q8vqmNtn5Rg)

#专家模型 [单个4090可推理，2000亿稀疏大模型「天工MoE」开源](https://mp.weixin.qq.com/s/h5bxuWca65t3LsQwqGq-Og)


#tts [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models](https://bytedancespeech.github.io/seedtts_tech_report/#applications-samples)

#资源汇总 [魔搭社区每周速递（5.26-5.31）](https://mp.weixin.qq.com/s/csPVepZKWFLDWsbr4N00CQ)
- **89**个模型：****MAP-Neo、YOLOV10、Yuan2-M32-hf、M2_Encoder_Huge、ChatTTS等；
- **47****个数据集：**Matrix、中国民族五声调式数据集、orpo-dpo-mix-40k等；
- **49****个创新应用****：**FaceChain-FACT人物写真生成、ChatTTS-demo、Grounded-SAM 自动分割生成蒙版遮罩等；
- **7篇****文章：**
- 高能力全透明双语大语言模型MAP-Neo完全开源，开放所有细节！
- ChatTTS：专为对话场景设计的文本转语音模型，底模开源！
- 【报名】GLM 法律行业大模型挑战赛 | 冠军最高 52 万现金奖励  
- YOLOv10发布，性能效率双提升，魔搭社区最佳实践来啦！
- 社区供稿 | YuanChat全面升级：知识库、网络检索、适配CPU，手把手个人主机部署使用教程！
- Data Is All You Need! 生成式图文对数据最佳实践，数据集魔搭开源！
- PAI x ModelScope：在PAI使用ModelScope模型

#roleplay 李沐：这一年多攒了不少技术，想陆续给大家分享心得。今天放一个70B模型展示post-training。1）是为复杂场景的roleplay设计，但通用能力同样很重要，2）从Llama3-70B base开始训练，做了完整了SFT和RLHF。尤其是RLHF可以提升巨大。3）很多LLM都多少overfit了测试集，所以尽量是post-training数据不包含benchmark的数据。4）在两个新的还没被刷爆评测上（MMLU-pro, Arena-hard）分数挺好。更流行的评测上面也比llama3-70B-instruct的要好。5）blog [https://boson.ai/higgs-opensource/](https://boson.ai/higgs-opensource/) 模型：[https://huggingface.co/bosonai/Higgs-Llama-3-70B](https://huggingface.co/bosonai/Higgs-Llama-3-70B)

#训练参数 [腾讯混元、北大发现Scaling law「浪涌现象」，解决学习率调参难题](https://mp.weixin.qq.com/s/ff5_O0H5VQNkArKroJkEZQ)

#roleplay [当GPT-4o遇上情感陪伴：多巴胺的胜利，催产素的挑战](https://mp.weixin.qq.com/s/JC3Mr6uFo_TXE3P92Va00g)

#训练参数 [用最酷的LR，训最猛的模型](https://mp.weixin.qq.com/s/2bNYBaJOLxuBaomv0Iu3gQ)

#大模型 [开源模型进展盘点：最新Mixtral、Llama 3、Phi-3、OpenELM到底有多好？](https://mp.weixin.qq.com/s/bgdDYkGHbPZMMSJPIutFSQ)

## 2024.6

[here is my meticulously curated (and highly biased) summer paper reading list](https://x.com/jxmnop/status/1800292343343693934)

## 2025.9

#多模态 #多模态训练[信息量很大：谷歌核心团队最新分享实录，揭秘Nano-Banana如何训练](https://mp.weixin.qq.com/s/LuDIEz3E-nbOUOWJmq8gAg)

- 当模型学会如何正确生成文本的结构时，它其实也学会了如何生成图像中的其他结构。在一张图像中，存在着不同频率的信息，你可以将其看作结构，但同时也有纹理等其他元素。所以，文本渲染能力为了解模型生成场景结构的优劣提供了非常重要的信号。

- 在缺乏其他有效的图像质量评估指标的情况下，因为很多指标很快就会饱和，文本渲染是一个衡量整体图像质量的绝佳方式。我曾经对使用人类评估员来进行图像生成评测的方法持怀疑态度。但随着时间的推移，我至少认识到，当你让足够多的人类，针对各种类别的足够多的提示词进行评估时，你确实能得到相当不错的信号。但显然，这种方法成本高昂。你不想总是让一大群人来给图片打分。因此，在模型训练过程中关注文本渲染这样的指标，能为你提供很好的信号，判断其表现是否符合预期。
- 交错生成的奇妙之处在于，它为图像生成提供了一种全新的范式。假设你有一个非常复杂的提示，比如包含六项不同的编辑要求，甚至可以设想包含五十项。现在，模型拥有一个非常出色的机制，能够以像素级精度从上下文中抓取信息，并应用在下一轮生成中。因此，你可以要求模型将一个复杂的提示——无论是用于图像编辑还是图像生成——分解为多个步骤，然后在不同步骤中逐一完成这些编辑。例如，第一步完成五项编辑，下一步再完成另外五项，以此类推。这与我们在语言模型领域采用的逐步推理和计算模式非常相似。你投入更多算力，让模型在像素空间中进行一种“思考”，并把复杂任务分解成更小的部分，从而能够精准地完成每个特定阶段的任务。当这些步骤累积起来，你就能完成任何复杂的任务。我认为，这再次体现了交错生成的魅力所在：你可以通过增量的方式生成极其复杂的图像，这与传统方式截然不同。传统方式力求“一步到位”地生成最佳图像。但模型的单次处理能力终究有上限，当指令细节多达上百个时，模型可能就无法一次性完成了。然而，当你拥有了交错生成这种分步处理的机制，你便可以应对几乎任何量级、任何复杂度的生成需求。
- 我们确实花了大量时间在 X (前 Twitter) 平台上，逐一查看用户的反馈。我记得很清楚，我和 Kaushik 还有团队里的其他人，一起收集了所有用户报告的失败案例，并基于这些案例构建了我们的评测集。所以，我们有一个专门的评测基准，完全来自于真实用户的反馈。











