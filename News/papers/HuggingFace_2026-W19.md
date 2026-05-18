# HuggingFace Papers — 2026-W19（5/4 – 5/10）

> 来源：https://huggingface.co/papers/week/2026-W19
> 统计日期：2026-05-12
> 筛选条件：upvotes ≥ 30
> 论文数：25

## 目录

1. [MolmoAct2: Action Reasoning Models for Real-world Deployment](#1-molmoact2)（👍 273）
2. [From Context to Skills: Can Language Models Learn from Context Skillfully?](#2-ctx2skill)（👍 152）
3. [Stream-R1: Reliability-Perplexity Aware Reward Distillation for Streaming Video Generation](#3-stream-r1)（👍 122）
4. [RLDX-1 Technical Report](#4-rldx-1)（👍 115）
5. [ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration](#5-aris)（👍 107）
6. [Stream-T1: Test-Time Scaling for Streaming Video Generation](#6-stream-t1)（👍 102）
7. [OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents](#7-opensearch-vl)（👍 95）
8. [Beyond Semantic Similarity: Direct Corpus Interaction for Agentic Search](#8-dci)（👍 90）
9. [UniVidX: Unified Multimodal Framework for Versatile Video Generation](#9-unividx)（👍 81）
10. [HERMES++: Unified Driving World Model for 3D Scene Understanding and Generation](#10-hermes)（👍 71）
11. [Continuous Latent Diffusion Language Model (Cola DLM)](#11-cola-dlm)（👍 69）
12. [Skill1: Unified Evolution of Skill-Augmented Agents via RL](#12-skill1)（👍 68）
13. [OpenSeeker-v2: Pushing the Limits of Search Agents](#13-openseeker-v2)（👍 64）
14. [MiniCPM-o 4.5: Real-Time Full-Duplex Omni-Modal Interaction](#14-minicpm-o-45)（👍 64）
15. [MiA-Signature: Approximating Global Activation for Long-Context Understanding](#15-mia-signature)（👍 53）
16. [PRISM: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL](#16-prism)（👍 46）
17. [RaguTeam at SemEval-2026 Task 8: Judge-Orchestrated LLM Ensemble](#17-raguteam)（👍 40）
18. [When to Trust Imagination: Adaptive Action Execution for World Action Models (FFDC)](#18-ffdc)（👍 39）
19. [Web2BigTable: Bi-Level Multi-Agent LLM for Internet-Scale Search & Extraction](#19-web2bigtable)（👍 38）
20. [SkillOS: Learning Skill Curation for Self-Evolving Agents](#20-skillos)（👍 37）
21. [MARBLE: Multi-Aspect Reward Balance for Diffusion RL](#21-marble)（👍 36）
22. [Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration (LoPE)](#22-lope)（👍 35）
23. [PhysForge: Physics-Grounded 3D Assets for Interactive Virtual World](#23-physforge)（👍 35）
24. [Rethinking Reasoning-Intensive Retrieval (BRIGHT-Pro / RTriever-4B)](#24-bright-pro)（👍 35）
25. [Let ViT Speak: Generative Language-Image Pre-training (GenLIP)](#25-genlip)（👍 32）

---

## <a id="1-molmoact2"></a>1. MolmoAct2: Action Reasoning Models for Real-world Deployment
**👍 273** · [arXiv:2605.02881](https://arxiv.org/abs/2605.02881) · [GitHub](https://github.com/allenai/molmoact2) · [Project](https://allenai.org/blog/molmoact2)

### 问题与动机
VLA（Vision-Language-Action）模型希望成为通用机器人控制器，但当前格局令人难受：前沿模型闭源，开源权重的方案绑死昂贵硬件，带推理的策略 grounding 延迟高得离谱，微调成功率离"能放心部署"还差一截。AI2 希望从架构、数据、tokenizer 三个层面把这条线彻底打开。

### 方法与核心创新
MolmoAct2 在前作基础上推进五个轴：(1) MolmoER 视觉语言主干，3.3M 样本、specialize-then-rehearse 课程专门强化空间和具身推理；(2) 三套新数据集，含 MolmoAct2-BimanualYAM——720 小时双臂遥操作，是目前最大开源双臂语料；(3) OpenFAST：跨 5 种本体百万级轨迹训练的开权 action tokenizer；(4) 把 flow-matching 连续动作专家"嫁接"到离散 token VLM 上，通过 per-layer KV-cache 条件注入；(5) MolmoThink 自适应深度推理——只对帧间变化的场景区域重算深度 token，保留几何 grounding 同时大幅降延迟。

### 关键实验结果
在 7 套仿真和真实基准的史上最大规模开源 VLA 实证里：MolmoAct2 击败 Pi-0.5；MolmoER 在 13 个具身推理基准上超过 GPT-5 和 Gemini Robotics ER-1.5。权重、代码、训练数据全开。

### 局限性与开放问题
论文未直接给出失败模式分析，但 720 小时双臂数据虽是最大开源集，仍远小于工业闭源规模；flow-matching 嫁接 KV-cache 是否对 long-horizon 任务保持稳定需要验证；MolmoThink 的"变化区域"判断在快速运动场景下可能误判。

### 启发与应用前景
对 Builder 来说，这是当前最值得 fork 的开源具身基座——数据 + tokenizer + 主干全开，相当于把"自己训 VLA"的门槛砍掉一半。对工程派最大的启发是 KV-cache 条件注入这个技巧，可迁移到任意"离散主干 + 连续头"的混合架构。

---

## <a id="2-ctx2skill"></a>2. From Context to Skills: Can Language Models Learn from Context Skillfully?
**👍 152** · [arXiv:2604.27660](https://arxiv.org/abs/2604.27660) · [GitHub](https://github.com/S1s-Z/Ctx2Skill)

### 问题与动机
很多真实任务的上下文超出 LM 参数化知识范围，需要"上下文学习"——从给定 context 直接抽规则与流程变成可复用 skill。但人工标注长技术文档代价高，自动构造又缺反馈。

### 方法与核心创新
Ctx2Skill 提出无人工监督、无外部反馈的自演化框架。多智能体自博弈循环：Challenger 出题与评分细则、Reasoner 在演化的 skill 集合下解题、中立 Judge 给二值反馈。专门的 Proposer 和 Generator agent 分析失败 case 合成两边都用的 skill 更新。为防止对抗坍缩（出题越来越极端、skill 过专化），引入 Cross-time Replay：在代表性样本上选出对 Reasoner 最平衡的 skill 集合。

### 关键实验结果
CL-bench 上四个上下文学习任务，Ctx2Skill 持续提升各 backbone 解决率。具体数字论文未在摘要给出。

### 局限性与开放问题
对抗自博弈的稳定性高度依赖 Judge 质量；二值反馈信号稀疏；摘要未披露绝对解决率，难判断"提升幅度"含金量；skill 是否在 OOD 任务上仍有效是个开放问题。

### 启发与应用前景
Skill1、SkillOS、本篇构成本周三篇"skill 库自演化"系列——把"经验沉淀成可复用 skill 文件"这件事真正变成 RL 问题。对个人 AI 工具开发的启发：把 Claude/Cursor 调用经验沉淀成 skill 库是有方法论的，不止是 prompt engineering。

---

## <a id="3-stream-r1"></a>3. Stream-R1: Reliability-Perplexity Aware Reward Distillation for Streaming Video Generation
**👍 122** · [arXiv:2605.03849](https://arxiv.org/abs/2605.03849) · [GitHub](https://github.com/FrameX-AI/Stream-R1) · [Project](https://stream-r1.github.io/)

### 问题与动机
流式视频扩散模型加速的事实标准是 DMD（distribution matching distillation），但它把每个 rollout、每帧、每像素当作同样可靠的监督——这压制了蒸馏上限。事实上两个维度差异巨大：rollout 间可靠性不同（Inter-Reliability），rollout 内空间帧间贡献也不同（Intra-Perplexity）。

### 方法与核心创新
Stream-R1 用一套共享 reward 机制在两个尺度上自适应重加权：rollout 级，按预训练 video reward 分数指数加权，让靠谱 rollout 主导优化；时空像素级，把同一 reward 模型反向传播得到 per-pixel 梯度显著性，分解成空间和时间权重，把优化压力集中在"还有提升空间"的区域和帧。自适应平衡机制避免单一质量维度（视觉/动作/对齐）独霸。

### 关键实验结果
在标准流式视频生成基准上，视觉质量、动作质量、文本对齐三维度一致提升，且不需要架构改动或额外推理代价。摘要未列具体数字。

### 局限性与开放问题
依赖预训练 video reward 模型质量；reward 反向传播的梯度显著性在长视频上是否稳定未知；可能继承 reward model 的偏见。

### 启发与应用前景
"reward-guided 加权蒸馏"这个思路可迁移到任何蒸馏场景——把统一权重换成两层 reward 信号几乎是免费午餐。Stream-R1 / Stream-T1 同周双发，FrameX-AI 这组人在流式视频生成上走得很激进，值得 follow。

---

## <a id="4-rldx-1"></a>4. RLDX-1 Technical Report
**👍 115** · [arXiv:2605.03269](https://arxiv.org/abs/2605.03269) · [GitHub](https://github.com/RLWRLD/RLDX-1) · [Project](http://rlwrld.ai/rldx-1)

### 问题与动机
VLA 借由预训练 VLM 拿到了场景理解和语言泛化，但在复杂真实任务上仍卡在运动感知、记忆决策、物理感知这类"超出通用智能"的功能能力上。

### 方法与核心创新
RLDX-1 是通用灵巧操作策略，基于 Multi-Stream Action Transformer (MSAT)——通过 modality-specific stream + 跨模态联合自注意力统一异构模态。系统级设计包括：罕见操作场景的合成数据生成、面向人形操作的专用学习流程、实时部署的推理优化。

### 关键实验结果
在仿真和真实任务上一致超越 π_{0.5}、GR00T N1.6 等前沿 VLA。ALLEX 人形任务 86.8% 成功率，而 π_{0.5} 和 GR00T N1.6 都只在 40% 左右——46 个百分点的差距。

### 局限性与开放问题
86.8% vs 40% 的差距太大，需要警惕是否任务设置偏向 RLDX-1 的训练分布；技术报告对模型规模、训练算力披露不足；MSAT 的可复现性取决于 stream 配置细节。

### 启发与应用前景
和 MolmoAct2 形成本周双开源 VLA 对照组——MolmoAct2 走"开权重 + flow-matching 嫁接"，RLDX-1 走"多流 Transformer + 数据合成"。两条路线都值得对比试。对工程派启发：高 DoF 人形控制现在已不是"只能闭源做"的领域了。

---

## <a id="5-aris"></a>5. ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration
**👍 107** · [arXiv:2605.03042](https://arxiv.org/abs/2605.03042) · [GitHub](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)

### 问题与动机
长周期自主科研工作流最危险的不是显式崩溃，而是"看似可信的无支撑成功"——长跑 agent 输出的 claim 证据链不完整、被错误转述、或默默继承了 executor 的 framing。需要一个 harness 把这种"silent failure"暴露出来。

### 方法与核心创新
ARIS 跨模型对抗协作：executor model 推进进度，**不同模型家族**的 reviewer 默认要批评中间产物并要求修订。三层架构：执行层提供 65+ Markdown skill、MCP 模型集成、持久研究 wiki、确定性绘图；编排层协调 5 个端到端工作流，可调 effort 和 reviewer 路由；assurance 层有三阶段实验声明验证（完整性核验 → 结果到声明映射 → 声明审计交叉对照证据账本），加上五遍科学编辑流水线、数学证明检查、PDF 视觉检视。原型 self-improvement 循环记录研究轨迹、提议 harness 改进，只有过 reviewer 批准才合并。

### 关键实验结果
报告以系统描述为主，未在摘要给出基准数字。

### 局限性与开放问题
"跨模型 reviewer 默认配置"成本高，对算力受限的个人开发者不友好；assurance 层规则化检查难覆盖所有 silent failure 模式；self-improvement 提议需 reviewer 批准——批准本身依赖 reviewer 质量。

### 启发与应用前景
对个人 Builder 极有参考价值——把"对自己的 AI 工具做 claim 审计"这件事系统化。GitHub 仓库名"Auto-claude-code-research-in-sleep"暗示这是 Claude Code 上跑的；可以借鉴它的三阶段 claim 审计流程，套到自己写的 agent 上。

---

## <a id="6-stream-t1"></a>6. Stream-T1: Test-Time Scaling for Streaming Video Generation
**👍 102** · [arXiv:2605.04461](https://arxiv.org/abs/2605.04461) · [GitHub](https://github.com/FrameX-AI/Stream-T1) · [Project](https://stream-t1.github.io/)

### 问题与动机
扩散视频 TTS 候选探索成本高、缺时间引导。流式视频生成的 chunk 级合成 + 少步去噪天然适合 TTS——能在显著降低算力的同时提供细粒度时间控制。

### 方法与核心创新
三模块统一框架：(1) Stream-Scaled Noise Propagation，用历史已验证高质量 chunk 的噪声去主动 refine 当前 chunk 的初始 latent，建立时间依赖、用历史 Gaussian 先验引导当前生成；(2) Stream-Scaled Reward Pruning，结合即时短期评估和滑窗长期评估，对候选做局部美学 + 全局时序的双重权衡；(3) Stream-Scaled Memory Sinking，根据 reward 反馈把从 KV-cache 驱逐的 context 动态路由到不同更新通路，确保历史视觉信息持续锚定后续生成。

### 关键实验结果
在 5s 和 30s 综合视频基准上时间一致性、运动流畅性、帧级视觉质量全面提升。摘要未给具体数字。

### 局限性与开放问题
依赖高质量 reward 模型；滑窗长期评估的窗口大小对效果影响未知；30s 已经是上限，更长视频的稳定性需要进一步验证。

### 启发与应用前景
和 Stream-R1 是同组（FrameX-AI）配套——蒸馏 + TTS 双管齐下做流式视频。对要做 long-form 视频生成或实时视频应用的人，这两篇是必读。

---

## <a id="7-opensearch-vl"></a>7. OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents
**👍 95** · [arXiv:2605.05185](https://arxiv.org/abs/2605.05185) · [GitHub](https://github.com/shawn0728/OpenSearch-VL) · [Project](https://huggingface.co/OpenSearch-VL)

### 问题与动机
多模态 deep search agent 难以复现——缺开源高质量训练数据、透明的轨迹合成 pipeline、详细训练 recipe。本文给一个完整开源 recipe。

### 方法与核心创新
专用数据管线：Wikipedia 路径采样 + 模糊实体重写 + source-anchor 视觉 grounding，减少捷径和一步检索坍缩。两份数据集：SearchVL-SFT-36k（SFT）和 SearchVL-RL-8k（RL）。工具环境融合文本搜索、图搜索、OCR、裁剪、锐化、超分、透视矫正，让 agent 主动感知 + 外部知识获取。算法上提出"多轮致命错误感知 GRPO"——掩盖工具失败后的 token、但用 one-sided advantage clamping 保留失败前有用推理。

### 关键实验结果
七个基准平均超 10 个点；多个任务追平闭源商业模型。数据、代码、模型全开。

### 局限性与开放问题
"超 10 点"是平均，单基准差异未知；Wikipedia 路径采样的语料偏置可能让 agent 在非百科域表现下滑；致命错误感知 GRPO 对短轨迹任务是否同样有效需验证。

### 启发与应用前景
本周 search-agent 主题三连发之一（另两篇是 #8 DCI 和 #13 OpenSeeker-v2）。对要做"研究助手 / 投资研究 agent"的人，这套数据合成方法可以直接借鉴——尤其是"source-anchor 视觉 grounding"思路对要处理含图表的金融研报场景很有用。

---

## <a id="8-dci"></a>8. Beyond Semantic Similarity: Direct Corpus Interaction for Agentic Search
**👍 90** · [arXiv:2605.05242](https://arxiv.org/abs/2605.05242) · [GitHub](https://github.com/DCI-Agent/DCI-Agent-Lite)

### 问题与动机
现代检索系统（无论 lexical 还是 semantic）都把语料压成 top-k 单步检索接口。这种抽象对 agentic search 是瓶颈：精确词法约束、稀疏线索合取、局部上下文核验、多步假设修正都难以通过现成 retriever 实现；早期被过滤掉的证据后续强推理也救不回。

### 方法与核心创新
**Direct Corpus Interaction (DCI)**——agent 直接用通用终端工具（grep、文件读、shell、轻量脚本）搜原始语料，**完全不用** embedding、向量索引、检索 API。无需离线索引，天然适配本地语料演化。

### 关键实验结果
在 IR 基准和端到端 agentic 任务上显著超过强 sparse、dense、reranking baseline；BRIGHT 和 BEIR 多个数据集上领先；BrowseComp-Plus 和 multi-hop QA 上不靠任何传统语义检索器也拿到强精度。

### 局限性与开放问题
极度依赖语料规模与 agent 的 tool-use 成本——大规模语料下 grep 成本能否扛住存疑；论文展示主要在结构化或半结构化文本，纯非结构 web 场景的表现未充分评估；接口设计自由度高也意味着 agent 行为更难预测。

### 启发与应用前景
**这篇是本周观点最炸的一篇**：直接挑战"必须先建索引再检索"的常识。对 Claude Code 用户特别有共鸣——我们用 Read/Grep 工具就是 DCI。对要做"本地知识库 + AI"的人是颠覆性提示：可能根本不用 RAG，给个 grep 加几个 read 工具就够。

---

## <a id="9-unividx"></a>9. UniVidX: Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors
**👍 81** · [arXiv:2605.00658](https://arxiv.org/abs/2605.00658) · [GitHub](https://github.com/houyuanchen111/UniVidX) · [Project](https://houyuanchen111.github.io/UniVidX.github.io/)

### 方法与核心创新
统一框架，把像素对齐任务统一为多模态共享空间的条件生成，保留 VDM 原生先验、促进跨模态一致性。三大设计：(1) Stochastic Condition Masking 训练时随机把模态分成"干净条件"和"噪声目标"，实现全方向条件生成；(2) Decoupled Gated LoRA，每个模态一个 LoRA，仅在该模态为目标时激活，保护 VDM 主干强先验；(3) Cross-Modal Self-Attention 共享 K/V 但保留模态特定 Query，促进跨模态对齐。

### 关键实验结果
两个实例化：UniVid-Intrinsic（RGB + albedo + irradiance + normal）和 UniVid-Alpha（混合 RGB + RGBA 分层）。即使只用 < 1000 个视频训练，仍达到 SOTA 水平并在 wild 数据上稳健泛化。

### 问题与动机
现有 VDM 重用方法各任务训各的，固定 I/O 映射，损失跨模态关联建模。

### 局限性与开放问题
< 1000 视频的"数据高效"很惊艳但需独立复现；模态扩展性（加更多模态时 LoRA 数量线性增长）有上限。

### 启发与应用前景
"用 LoRA 做模态保护 + 注意力共享做模态对齐"这个范式可直接拿到多模态生成的任何场景。

---

## <a id="10-hermes"></a>10. HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation
**👍 71** · [arXiv:2604.28196](https://arxiv.org/abs/2604.28196) · [GitHub](https://github.com/H-EmbodVis/HERMESV2) · [Project](https://h-embodvis.github.io/HERMESV2/)

### 问题与动机
自动驾驶 world model 主要做未来场景生成，忽视 3D 场景理解；LLM 推理强但缺几何演化预测——语义解读和物理模拟之间脱节。

### 方法与核心创新
单一框架整合两端：(1) BEV 表征把多视图空间信息整合成 LLM 兼容结构；(2) LLM-enhanced world queries 让理解分支的知识转移到生成分支；(3) Current-to-Future Link 用语义上下文为几何演化建条件桥；(4) Joint Geometric Optimization 显式几何约束 + 隐式 latent 正则化，对齐内部表征到几何先验。

### 关键实验结果
多基准上，未来点云预测和 3D 场景理解都超过专家模型。摘要未给具体数字。

### 局限性与开放问题
驾驶域以外的迁移性未验证；BEV 对垂直方向信息有损（楼层、桥洞场景）；联合优化的训练稳定性细节未披露。

### 启发与应用前景
"理解 + 生成"统一这个思路在驾驶之外（家用机器人、无人机）也适用。BEV + LLM query 这套接口设计可借鉴。

---

## <a id="11-cola-dlm"></a>11. Continuous Latent Diffusion Language Model (Cola DLM)
**👍 69** · [arXiv:2605.06548](https://arxiv.org/abs/2605.06548) · [Project](https://hongcanguo.github.io/Cola-DLM/)

### 问题与动机
高质量文本生成不一定非要绑死自回归从左到右。现有非自回归方案在效率、可扩展表示学习、全局语义建模三者之间难以兼得。

### 方法与核心创新
分层连续 latent 扩散语言模型：Text VAE 先学稳定的"文本-latent"映射；block-causal DiT 在连续 latent 空间建模全局语义先验；最后条件解码生成文本。统一 Markov-path 视角下，扩散过程做的是 latent 先验传输而非 token 级观测恢复——把全局语义组织和局部文本实现分离。

### 关键实验结果
4 个研究问题、8 个基准、严格匹配 ~2B 参数的自回归和 LLaDA baseline，scaling 曲线推到约 2000 EFLOPs，验证了 scaling 行为。摘要未给绝对分数。

### 局限性与开放问题
2B 规模够不够说明 scaling 趋势在 70B+ 仍成立，存疑；和强 AR baseline 的真实差距摘要未明示；Text VAE 的稳定性是整条链的瓶颈。

### 启发与应用前景
分层 + 连续 + 扩散的语言建模可能是"统一连续文本和多模态"的桥——尤其在 dLLM 这条路线开始升温的当下，值得跟。

---

## <a id="12-skill1"></a>12. Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning
**👍 68** · [arXiv:2605.06130](https://arxiv.org/abs/2605.06130)

### 问题与动机
持久 skill 库让 LLM agent 跨任务复用策略，但维护要 select / utilize / distill 三能力耦合。现有方法孤立优化或用分离 reward，导致演化部分且冲突。

### 方法与核心创新
Skill1 用**单一策略**对三能力共同演化、共享同一任务结果目标。流程：生成 query 检索 skill 库 → 重排候选并选一个 → 在该 skill 条件下解题 → 从轨迹蒸馏新 skill。所有学习来自单一任务结果信号——低频趋势归因选择能力，高频变化归因蒸馏能力。

### 关键实验结果
ALFWorld 和 WebShop 上超过 skill-based 和 RL baseline；训练动态确认三能力共演化；消融移除任一 credit 信号都退化。

### 局限性与开放问题
"低频/高频信号归因"在更复杂任务（如代码、长推理）是否成立没验证；单一任务结果信号可能让稀疏奖励任务训不动。

### 启发与应用前景
和 Ctx2Skill、SkillOS 构成本周"skill 自演化"三件套，但本篇是唯一"共享 reward 单策略"做法。要做"AI 工具自我升级"的人值得对比这三种范式。

---

## <a id="13-openseeker-v2"></a>13. OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories
**👍 64** · [arXiv:2605.04036](https://arxiv.org/abs/2605.04036) · [GitHub](https://github.com/PolarSeeker/OpenSeeker)

### 问题与动机
工业前沿 search agent 是 pre-train + CPT + SFT + RL 的重型流水线，纯学术团队几乎玩不动。本文证明：只要数据够"信息量大 + 难度高"，**纯 SFT** 也能上 SOTA。

### 方法与核心创新
三个简单数据合成改进：(1) 放大知识图谱规模做更丰富探索；(2) 扩大工具集；(3) 严格 low-step 过滤（剔除"太容易解"的轨迹）。

### 关键实验结果
**仅 10.6k 数据点**，30B 规模 ReAct 范式下：BrowseComp 46.0%、BrowseComp-ZH 58.1%、Humanity's Last Exam 34.6%、xbench 78.0%——全面超越走重型 CPT+SFT+RL 的 Tongyi DeepResearch（43.4 / 46.7 / 32.9 / 75.0）。

### 局限性与开放问题
"low-step 过滤"可能让 agent 偏向短轨迹解，长 horizon 推理是否被牺牲？10.6k 数据的多样性瓶颈在哪？工业 pipeline 真正的护城河可能不在数据量而在 RL 阶段。

### 启发与应用前景
**对个人/小团队是巨大利好**：search agent 不一定非要 RL。建议关注 OpenSeeker 这个组——本周 Stream-T1 / OpenSeeker-v2 都是用"更聪明的数据"打"更暴力的 pipeline"。

---

## <a id="14-minicpm-o-45"></a>14. MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction
**👍 64** · [arXiv:2604.27393](https://arxiv.org/abs/2604.27393) · [GitHub](https://github.com/OpenBMB/MiniCPM-o)

### 问题与动机
MLLM 的瓶颈不再是模态覆盖或延迟，而是交互范式本身：感知/响应仍是交替阶段、模型只在显式请求时响应。

### 方法与核心创新
**Omni-Flow** 统一流式框架把全模态 I/O 对齐到共享时间轴，把传统轮换交互变成全双工时间对齐过程——可以同时感知和响应，主动行为自然涌现。

### 关键实验结果
9B 总参数，视觉-语言能力逼近 Gemini 2.5 Flash；全模态理解超过 Qwen3-Omni-30B-A3B 并语音生成更好、算力远低；高效架构 + 推理优化让模型能在 **<12GB RAM 的边缘设备**上做实时全双工。

### 局限性与开放问题
"接近 Gemini 2.5 Flash"是按基准均值还是子集？长对话下"主动行为"的合适度（不打扰用户）需要长跑评估；中文场景表现摘要未提。

### 启发与应用前景
**做端侧 AI 助手的人必看**——12GB 全双工意味着 Mac 16GB 内存可跑。这是"AI 陪伴 / 角色扮演"产品最值得关注的开源底座，端侧低延迟交互是体验质变点。

---

## <a id="15-mia-signature"></a>15. MiA-Signature: Approximating Global Activation for Long-Context Understanding
**👍 53** · [arXiv:2605.06416](https://arxiv.org/abs/2605.06416)

### 问题与动机
认知科学提示：reportable conscious access 关联到分布式记忆系统的"全局点火"，但个体无法直接枚举所有激活内容——存在压缩表征近似全局影响。

### 方法与核心创新
Mindscape Activation Signature (MiA-Signature)：query 诱导的全局激活模式的压缩表征。LLM 实现上用 submodular 选择覆盖激活上下文空间的高层概念，可选 working memory 轻量迭代精化。Signature 作为条件信号近似全状态影响但保持算力可控。

### 关键实验结果
集成到 RAG 和 agentic 系统，多个 long-context 理解任务一致提升。摘要无绝对数字。

### 局限性与开放问题
"submodular 选择"具体实现摘要未明示；和现有 long-context 方法（YaRN、CEPE、RAG）的对比基准未充分披露；认知科学类比是否真有效果增益还是只是叙事框架，需独立验证。

### 启发与应用前景
"压缩激活成 signature 作为条件"这个抽象对要做 long-context agent 的人有借鉴；和 DCI、SkillOS 这类"agent 接口设计"是同一脉络。

---

## <a id="16-prism"></a>16. PRISM: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL
**👍 46** · [arXiv:2604.28123](https://arxiv.org/abs/2604.28123) · [GitHub](https://github.com/XIAO4579/PRISM) · [Project](https://xiao4579.github.io/PRISM/)

### 问题与动机
SFT → RLVR 的标准 post-training 配方有问题：SFT 引入的分布漂移既不保留原始能力也没忠实匹配监督分布，多模态场景下感知错误和推理失败漂移模式不同还会在 RL 阶段相互放大。

### 方法与核心创新
**在 SFT 和 RLVR 之间插入显式分布对齐阶段**。基于 on-policy distillation，PRISM 把对齐定义为黑盒响应级对抗博弈：policy vs MoE 判别器（专门设感知专家和推理专家），给出解耦的纠错信号、不需要 teacher logits 访问权。1.26M 公开示范用于 SFT 初始化；额外 113K Gemini 3 Flash 示范（含密集视觉 grounding 和分步推理）用于对齐阶段。

### 关键实验结果
Qwen3-VL 上，GRPO / DAPO / GSPO 多种 RL 算法和多模态基准均一致提升；4B 模型平均 +4.4 点、8B +6.0 点（相对 SFT-to-RLVR baseline）。

### 局限性与开放问题
"+4.4 / +6.0"是均值，子任务方差未知；113K Gemini 数据依赖商业模型 distill，自主性受限；MoE 判别器训练成本未披露。

### 启发与应用前景
对做后训练的人是直接武器：把 SFT-to-RL 的 pipeline 改成 SFT → distill 对齐 → RL 几乎是 free improvement。MoE 判别器解耦感知/推理这个 idea 也可迁移到纯文本场景。

---

## <a id="17-raguteam"></a>17. RaguTeam at SemEval-2026 Task 8: Judge-Orchestrated LLM Ensemble
**👍 40** · [arXiv:2605.04523](https://arxiv.org/abs/2605.04523) · [GitHub](https://github.com/RaguTeam/ragu_mtrag_semeval)

### 问题与动机
SemEval-2026 Task 8（MTRAGEval）的多轮 RAG 生成任务：要在有参考段落条件下做忠实的多轮回复。

### 方法与核心创新
异构集成：7 个 LLM × 2 种 prompting 变体，GPT-4o-mini 当 Judge 逐 instance 选最佳。同时引入 Meno-Lite-0.1，一个 7B 的域适应小模型，做成本-性能权衡。

### 关键实验结果
26 队第 1 名；conditioned harmonic mean 0.7827，远超最强单模型 baseline gpt-oss-120b 的 0.6390。消融证明模型家族、规模、prompt 策略多样性都不可或缺。

### 局限性与开放问题
比赛系统的工程复杂度高（7 LLM + Judge），实际部署成本高；Meno-Lite-0.1 的训练细节摘要未详；"Judge 选最佳"在 OOD 场景下的稳定性需验证。

### 启发与应用前景
"Judge 调度的异构集成"是当下 LLM 系统设计的实用范式——和 ARIS 的"跨模型 reviewer"同根同源。要做生产级 RAG 的可直接借鉴 7+2+1 这套配置思路。

---

## <a id="18-ffdc"></a>18. When to Trust Imagination: Adaptive Action Execution for World Action Models (FFDC)
**👍 39** · [arXiv:2605.06222](https://arxiv.org/abs/2605.06222)

### 问题与动机
World Action Models 同时预测未来视觉观察和动作，但执行固定数量动作会让机器人对"想象的未来是否还和真实一致"完全失明。

### 方法与核心创新
把自适应 WAM 执行定义为**未来-现实校验问题**：可信就多执行、偏离就早重规划。提出 **Future Forward Dynamics Causal Attention (FFDC)**——轻量校验器联合预测未来动作、预测视觉动态、真实观察、语言指令，估计剩余 rollout 是否还可信。chunk size 自适应是预测-观察一致性的涌现结果。再加 Mixture-of-Horizon Training 提升长 horizon 覆盖。

### 关键实验结果
RoboTwin 上 WAM 前向传播 -69.10%、执行时间 -34.02%、成功率 +2.54%（相对短 chunk baseline）；真实世界成功率 +35%。

### 局限性与开放问题
35% 真实世界提升的任务集大小未明示；FFDC 本身是个轻量但额外的网络，对端侧部署的额外开销需评估；语言指令依赖度可能让纯视觉任务的迁移受限。

### 启发与应用前景
"自适应 chunk size"思路可迁移到任意 model-based control 场景——LLM 工具调用里"chunk 长度由置信度决定"也是同一范式。

---

## <a id="19-web2bigtable"></a>19. Web2BigTable: Bi-Level Multi-Agent LLM for Internet-Scale Search & Extraction
**👍 38** · [arXiv:2604.27221](https://arxiv.org/abs/2604.27221) · [GitHub](https://github.com/web2bigtable/web2bigtable)

### 问题与动机
Agentic web search 同时面临两类需求：单目标深推理 + 跨实体异源结构化聚合。深度任务要长轨迹连贯推理；宽度任务要 schema 对齐、宽覆盖、跨实体一致。现有系统两头都吃力。

### 方法与核心创新
**Bi-Level Multi-Agent**：上层 orchestrator 把任务分解成子问题，下层 worker agent 并行解。Run-verify-reflect 闭环联合改进分解和执行，借助持久人类可读外部 memory 做自演化 single-agent 更新。Worker 通过共享 workspace 公开局部发现——减少重复探索、调和矛盾证据、补覆盖空缺。

### 关键实验结果
WideSearch 上 SOTA：Avg@4 Success Rate **38.50（次优 5.10 的 7.5×）**、Row F1 63.53（+25.03）、Item F1 80.12（+14.42）。XBench-DeepSearch 上深推理任务 73.0 准确率。

### 局限性与开放问题
"7.5×"的对照组水平本身偏低（5.10），数字看着惊艳但要看 WideSearch 任务本身的难度分布；多 agent 并行成本未明示；外部 memory 增长后的读取效率是潜在瓶颈。

### 启发与应用前景
做"信息收集到表格"的 agent（招聘、竞品研究、投资标的清单）可直接用这套 bi-level 框架。

---

## <a id="20-skillos"></a>20. SkillOS: Learning Skill Curation for Self-Evolving Agents
**👍 37** · [arXiv:2605.06614](https://arxiv.org/abs/2605.06614)

### 问题与动机
LLM agent 处理流式任务时常常是一次性解题器，不学经验。Skill 库是自演化的天然载体，但 curation 是瓶颈——现有方法靠人工、靠启发式、或只训短 horizon skill 操作。

### 方法与核心创新
SkillOS：experience-driven RL 训练 skill curation。冻结 agent executor 检索 + 应用 skill；可训练 skill curator 根据累计经验更新外部 SkillRepo。复合奖励 + 基于 skill 相关任务依赖的分组任务流——早期轨迹更新 SkillRepo，后续相关任务评估更新效果。

### 关键实验结果
多轮 agentic 任务和单轮推理任务上都超过 memory-free 和强 memory baseline，效果和效率双优。Curator 跨 executor 主干和任务域泛化。Skill 在 repo 里演化成结构更丰富的 Markdown 文件、编码高层 meta-skill。

### 局限性与开放问题
"复合奖励"具体设计未在摘要披露；冻结 executor 限制 skill 利用上限；分组任务流构造对真实流式场景的代表性需验证。

### 启发与应用前景
和 Skill1、Ctx2Skill 互为对照——本篇用**冻结 executor + 可训 curator**，Skill1 用**单一策略全演化**，Ctx2Skill 用**对抗自博弈无外部反馈**。哪条路对个人 AI 工具最实用？冻结 + curator 这套门槛最低、最容易复刻。

---

## <a id="21-marble"></a>21. MARBLE: Multi-Aspect Reward Balance for Diffusion RL
**👍 36** · [arXiv:2605.06507](https://arxiv.org/abs/2605.06507) · [GitHub](https://github.com/aim-uofa/MARBLE) · [Project](https://aim-uofa.github.io/MARBLE/)

### 问题与动机
扩散模型 RL 微调对齐多评估准则时，主流"加权求和 reward"会失败——大多数 rollout 是某些 reward 维度的 specialist 样本，对其他维度无关，加权求和稀释监督。

### 方法与核心创新
MARBLE：梯度空间优化框架。为每个 reward 维护独立 advantage 估计器，计算 per-reward 策略梯度，通过解二次规划问题协调成单一更新方向——**完全无需手调 reward 权重**。摊销公式利用 DiffusionNFT loss 的仿射结构，把单步代价从 K+1 次反传降到接近单 reward baseline；EMA 平滑平衡系数避免单 batch 波动。

### 关键实验结果
SD3.5 Medium + 5 个 reward 上：5 维度同时提升；加权求和下 80% mini-batch 里"最差对齐 reward"的梯度余弦是负的，MARBLE 把它扭成持续正；训练速度是 baseline 的 **0.97×**——几乎没额外代价。

### 局限性与开放问题
QP 求解的开销随 reward 数量扩展性如何？5 个 reward 还能扩到 10、20 吗？

### 启发与应用前景
**多目标 RL 的通用解药**——LLM 多目标对齐（helpfulness + harmlessness + honesty）的人立刻该试。梯度空间优化 + QP 这个思路可迁移到任何多 reward 微调。

---

## <a id="22-lope"></a>22. Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration (LoPE)
**👍 35** · [arXiv:2605.05566](https://arxiv.org/abs/2605.05566)

### 问题与动机
GRPO 在复杂任务上常遇"零优势问题"——所有 sample 都失败时相对优势坍缩到 0，模型失去训练信号。简单加大采样预算治标不治本，静态采样策略约束了探索。

### 方法与核心创新
**LoPE (Lorem Perturbation for Exploration)**：把 Lorem Ipsum 伪拉丁词汇随机拼接成序列，**前置到 prompt 前面**再重采样。任务无关的 prompt 空间扰动能让输出分布偏移、打开正交推理路径。

### 关键实验结果
1.7B / 4B / 7B 模型上一致显著超过原 prompt 重采样。进一步分析显示其他低 perplexity 的拉丁随机序列也有效。

### 局限性与开放问题
"低 perplexity 拉丁序列"为啥有效，机制解释还不够深；中文模型上是否一样有效？拉丁扰动对指令遵循能力的副作用未评估。

### 启发与应用前景
**最让人意外的一篇**——胡话居然能稳定 broaden exploration。对 RL 微调实践的启发：探索性问题不一定要改算法，prompt 空间的随机扰动可能是更便宜的 escape hatch。

---

## <a id="23-physforge"></a>23. PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World
**👍 35** · [arXiv:2605.05163](https://arxiv.org/abs/2605.05163) · [GitHub](https://github.com/HKU-MMLab/PhysForge) · [Project](https://hku-mmlab.github.io/PhysForge/)

### 问题与动机
交互式虚拟世界和具身 AI 的瓶颈是物理 grounded 的 3D 资产。现有方法只关心静态几何，忽略交互必需的功能属性。

### 方法与核心创新
解耦两阶段框架 + PhysDB（15 万资产、四层物理标注）。第一阶段 VLM 当"物理建筑师"规划"层级物理蓝图"，定义材料、功能、运动学约束；第二阶段物理 grounded 扩散模型通过 KineVoxel Injection (KVI) 机制实现蓝图，合成高保真几何 + 精确运动学参数。

### 关键实验结果
实验证明 PhysForge 生成功能合理、可仿真就绪的资产，作为交互式 3D 内容和具身 agent 的数据引擎。

### 局限性与开放问题
摘要未给基准对比数字；15 万资产规模仍远小于 Objaverse；四层标注的人工成本未披露。

### 启发与应用前景
"VLM 当 architect + 扩散当 builder"这个两阶段范式可推广到任何需要"先规划后生成"的场景——比如 PPT、代码、3D 设计。

---

## <a id="24-bright-pro"></a>24. Rethinking Reasoning-Intensive Retrieval (BRIGHT-Pro / RTriever-4B)
**👍 35** · [arXiv:2605.04018](https://arxiv.org/abs/2605.04018) · [GitHub](https://github.com/yale-nlp/Bright-Pro)

### 问题与动机
推理密集检索要找的是支撑下游推理的证据而非主题相似。但 BRIGHT 等基准给的 gold 集太窄、孤立评估 retriever；合成训练语料只优化单段相关性而非证据组合。

### 方法与核心创新
(1) **BRIGHT-Pro**：专家标注基准，每个 query 扩展多 aspect gold 证据，在静态和 agentic 协议下都评估。(2) **RTriever-Synth**：aspect 解耦合成语料，生成互补正例 + positive-conditioned 难负例。(3) 用它 LoRA 微调 Qwen3-Embedding-4B 得 **RTriever-4B**。

### 关键实验结果
跨 lexical、通用、推理密集 retriever 的实验显示 aspect-aware 和 agentic 评估暴露标准指标隐藏的行为差异；RTriever-4B 大幅超过 base。

### 局限性与开放问题
专家标注扩展成本高、跨域可扩展性差；"大幅超过"具体数字摘要未明示。

### 启发与应用前景
推理密集 RAG 实践的人立刻能用 RTriever-4B 当 embedding。aspect 分解 + positive-conditioned 难负例的数据合成方法也可迁移到任何场景做 embedding 微调。

---

## <a id="25-genlip"></a>25. Let ViT Speak: Generative Language-Image Pre-training (GenLIP)
**👍 32** · [arXiv:2605.00809](https://arxiv.org/abs/2605.00809) · [GitHub](https://github.com/YanFangCS/GenLIP) · [Project](https://yanfangcs.github.io/vitspeak/)

### 问题与动机
为了让 ViT 视觉编码器更好对接 LLM 的自回归本质，需要重新设计预训练目标——传统 contrastive 对齐不够"原生"。

### 方法与核心创新
**GenLIP**：训 ViT 直接从视觉 token 用标准 LM 目标预测语言 token，无需对比 batch 构造、无需额外文本 decoder。三优势：简单（单 transformer 联合建模视觉文本 token）、可扩展（数据和模型规模都 scale 良好）、性能（多模态基准上有竞争力或更强）。在 Recap-DataComp-1B 的 8B 样本训练；继续在多分辨率原始宽高比图像上预训练后，OCR、图表理解等细节敏感任务进一步提升。

### 关键实验结果
匹配或超过强 baseline，且用了显著更少的预训练数据。

### 局限性与开放问题
"显著更少"和"匹配"的具体对比数据未列；多分辨率继续预训练成本和收益的权衡未明示。

### 启发与应用前景
"训 ViT 让它'说话'"是非常优雅的极简思路——单目标、单架构、自然支持 MLLM 集成。对要从头训自己的 vision encoder 的人极有参考价值。

---

## 🗺️ 趋势洞察

### 1. Agentic Search 全面升级：数据合成、接口革命、多 agent 协作三线并进
**涉及论文**：#7 OpenSearch-VL、#8 DCI、#13 OpenSeeker-v2、#19 Web2BigTable、#24 BRIGHT-Pro
**核心观点**：本周一口气 5 篇 search agent，但路线分化明显：OpenSeeker-v2 证明**纯 SFT + 高质量数据**能打过工业 RL pipeline；DCI 直接挑战"必须先建向量索引"的成见，让 agent 用 grep/shell 直接和原始语料对话；Web2BigTable 用 bi-level 多 agent 解决宽度+深度双需求；OpenSearch-VL 给多模态 search agent 完整开源 recipe；BRIGHT-Pro 提供新评估协议。**信号**：search agent 已从"做 SOTA"进入"暴露隐藏问题 + 重新设计接口"阶段。

### 2. Skill 库自演化：三种路线同周对照
**涉及论文**：#2 Ctx2Skill、#12 Skill1、#20 SkillOS
**核心观点**：让 agent 把经验沉淀成可复用 skill 这件事在本周成为显学。三种范式：(a) Ctx2Skill 用多 agent 对抗自博弈，无人工无外部反馈；(b) Skill1 用单一 RL 策略同时演化 selection / utilization / distillation；(c) SkillOS 冻结 executor、单独训 curator。对个人 Builder：SkillOS 路线门槛最低、最容易复刻；Skill1 的"低频/高频信号归因"思路最优雅。

### 3. 流式视频生成进入 "TTS + 蒸馏" 双轴优化阶段
**涉及论文**：#3 Stream-R1、#6 Stream-T1、#9 UniVidX
**核心观点**：FrameX-AI 同周双发 Stream-R1（蒸馏端 reward-aware 加权）和 Stream-T1（推理端 test-time scaling），把流式视频的训练和推理两端都优化了。UniVidX 在多模态视频生成统一框架上做出 < 1000 视频高效训练。**信号**：流式范式已成共识，竞争点转向"如何榨干每一帧的监督信号"。

### 4. 具身智能：开源 VLA 走向真正可部署
**涉及论文**：#1 MolmoAct2、#4 RLDX-1、#18 FFDC、#23 PhysForge
**核心观点**：MolmoAct2（AI2）和 RLDX-1 双开源 VLA 同周发布，前者走"开权重 + flow-matching 嫁接 + 双臂数据"，后者走"多流 Transformer + 数据合成"。FFDC 解决"动作执行什么时候该信任想象"的根本问题，PhysForge 把"VLM 当 architect"用到 3D 资产生成。**信号**：具身 VLA 不再是闭源专利，且开源方案开始在硬指标（86.8% vs 40%）上拉开差距。

### 对比与张力
- **DCI（#8 grep 直查）vs 传统 RAG（#15 #24 retriever 优化）**：一边是"丢掉所有 retriever 用通用工具直接交互"，一边是"在 retriever 框架内做更精细评估和数据合成"。两边都在涨点，**接口设计自由度**和**针对性优化**的张力会持续。
- **OpenSeeker-v2（#13 纯 SFT）vs PRISM（#16 SFT → 对齐 → RL）**：一边是"数据够好 SFT 就能 SOTA"，一边是"SFT-to-RL 的漂移必须显式对齐"。问题是：OpenSeeker-v2 的 SOTA 是否只是因为基准本身没榨干 SFT 能力上限？
- **三种 skill 自演化（#2 #12 #20）**：对抗自博弈 vs 单策略共演化 vs 冻结-curator 解耦。
- **MARBLE（#21 多 reward 梯度空间）vs Stream-R1（#3 reward 加权蒸馏）**：都是"reward 不再统一权重"，但一个是用 QP 解多目标，一个是用 reward 自身做权重指导器。同一周不同子领域同时摸到了"reward 加权机械化失败"这个问题。

### 值得关注的研究方向
1. **Search agent 的接口革命**：DCI（#8）这条线如果成立，意味着 RAG 这套技术栈的相当一部分可以被绕开。值得花周末复现一下小规模实验。
2. **Skill 库自演化**：三种范式同周对照，对要做"会自我进化的个人 AI 工具"的人是绝佳学习窗口。优先看 SkillOS（#20）的 curator 实现。
3. **端侧全双工多模态**：MiniCPM-o 4.5（#14）的 <12GB 全双工把"端侧实时陪伴 AI"的可行性大幅推进，做角色扮演 / 心理陪伴产品的应该立刻评估。
4. **多 reward 梯度协调**：MARBLE（#21）的 QP 框架是"对齐多目标"的通解，LLM 后训练做多目标对齐的可立刻借鉴。
5. **Prompt 扰动作为探索 boost**：LoPE（#22）这个简单到不可思议的 trick，几乎所有 GRPO 训练都该试一下。
