# HuggingFace 周榜论文深度总结 — 2026 第 22 周

> 来源：https://huggingface.co/papers/week/2026-W22
> 统计日期：2026-06-18
> 筛选条件：upvotes ≥ 50
> 论文数：26

## 目录

1. [Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players](#1-gamma-world-generative-multi-agent-world-modeling-beyond-two-players) 👍423
2. [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](#2-skillopt-executive-strategy-for-self-evolving-agent-skills) 👍234
3. [AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security](#3-agentdog-15-a-lightweight-and-scalable-alignment-framework-for-ai-agent-safety-and-security) 👍142
4. [Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments](#4-qwen-vla-unifying-vision-language-action-modeling-across-tasks-environments-and-robot-embodiments) 👍142
5. [LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding](#5-locateanything-fast-and-high-quality-vision-language-grounding-with-parallel-box-decoding) 👍141
6. [DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning](#6-dvao-dynamic-variance-adaptive-advantage-optimization-for-multi-reward-reinforcement-learning) 👍136
7. [Rethinking Cross-Layer Information Routing in Diffusion Transformers](#7-rethinking-cross-layer-information-routing-in-diffusion-transformers) 👍110
8. [Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models](#8-lens-rethinking-training-efficiency-for-foundational-text-to-image-models) 👍110
9. [WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation](#9-wbench-a-comprehensive-multi-turn-benchmark-for-interactive-video-world-model-evaluation) 👍102
10. [Agent Explorative Policy Optimization for Multimodal Agentic Reasoning](#10-agent-explorative-policy-optimization-for-multimodal-agentic-reasoning) 👍91
11. [ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation](#11-prorl-effective-reinforcement-learning-for-proactive-recommendation-via-rectified-policy-gradient-estimation) 👍87
12. [Macaron-A2UI: A Model for Generative UI in Personal Agents](#12-macaron-a2ui-a-model-for-generative-ui-in-personal-agents) 👍82
13. [EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation](#13-evalverse-pipeline-aware-and-expert-calibrated-benchmarking-for-professional-cinematic-video-generation) 👍80
14. [Foundation Protocol: A Coordination Layer for Agentic Society](#14-foundation-protocol-a-coordination-layer-for-agentic-society) 👍80
15. [OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources](#15-omniretrieval-unified-retrieval-across-heterogeneous-knowledge-sources) 👍76
16. [From Pixels to Words -- Towards Native One-Vision Models at Scale](#16-from-pixels-to-words----towards-native-one-vision-models-at-scale) 👍73
17. [SpatialBench: Is Your Spatial Foundation Model an All-Round Player?](#17-spatialbench-is-your-spatial-foundation-model-an-all-round-player) 👍72
18. [MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research](#18-mobilegym-a-verifiable-and-highly-parallel-simulation-platform-for-mobile-gui-agent-research) 👍64
19. [CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation](#19-collectionlora-collecting-50-effects-in-1-lora-via-multi-teacher-on-policy-distillation) 👍61
20. [Why Far Looks Up: Probing Spatial Representation in Vision-Language Models](#20-why-far-looks-up-probing-spatial-representation-in-vision-language-models) 👍60
21. [Self-Improving Language Models with Bidirectional Evolutionary Search](#21-self-improving-language-models-with-bidirectional-evolutionary-search) 👍59
22. [minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models](#22-minwm-a-full-stack-open-source-framework-for-real-time-interactive-video-world-models) 👍58
23. [SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research](#23-sciatlas-a-large-scale-knowledge-graph-for-automated-scientific-research) 👍58
24. [YoCausal: How Far is Video Generation from World Model? A Causality Perspective](#24-yocausal-how-far-is-video-generation-from-world-model-a-causality-perspective) 👍54
25. [TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction](#25-trisplat-simulation-ready-feed-forward-3d-scene-reconstruction) 👍52
26. [ResearchMath-14K: Scaling Research-Level Mathematics via Agents](#26-researchmath-14k-scaling-research-level-mathematics-via-agents) 👍50

---

## 1. Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players
**👍 423** · https://huggingface.co/papers/2605.28816

### 问题与动机
交互式视频世界模型几乎都停在「单智能体」设定：未来画面由单一控制信号生成。但很多真实环境是多人/多机器人同时在共享空间里行动——单控制信号的范式根本表达不了。把世界模型扩展到多智能体，难点在于：每个智能体要能独立控制、彼此地位对称（置换不变）、推理还要高效，同时跨时间和视角保持一致。这是当前世界模型最缺的一块。

### 方法与核心创新
两个关键设计。其一 **Simplex Rotary Agent Encoding**：把 3D RoPE 做了一个无参数扩展，把每个智能体放在「旋转角空间里正单纯形的一个顶点」上。这样每个智能体有独立相位、却又彼此置换等价——不需要学习「第几号槽位」的身份，也不依赖固定排序，从而可扩展。其二 **Sparse Hub Attention**：用可学习的 hub token 做中介，避免智能体之间稠密的 all-to-all 注意力，把跨智能体注意力成本从平方降到线性。最后用扩散教师蒸馏出因果学生，配合 KV 缓存做时间块的顺序生成，实现 24 FPS 的实时动作响应生成。

### 关键实验结果
在多人虚拟环境里，相比 slot-based 和稠密注意力基线，模型在视频保真度、动作可控性、智能体间一致性上都更好；最有说服力的是**从两人泛化到四人无需额外训练**。生成速率 24 FPS，达到实时交互可用。（具体的 FID/可控性数值摘要未给出，但定性结论明确。）

### 局限性与开放问题
摘要未给出与具体强基线的量化差距，难判断领先幅度。线性注意力靠 hub token 中介，hub 容量是否会成为多智能体信息瓶颈、上限到几个智能体仍未知（摘要未明确，推断）。蒸馏成因果学生通常会损失部分长程一致性，24 FPS 下的累积漂移情况也未交代。

### 启发与应用前景
最大价值是给「多智能体世界模型」提供了一套可扩展的身份编码范式——单纯形 RoPE 这个无参数 trick 简洁且可迁移到任何需要置换对称实体编码的场景（多目标追踪、群体仿真）。直接应用是多人游戏/具身仿真训练环境，NVIDIA 出品也暗示其面向机器人仿真管线。

---

## 2. SkillOpt: Executive Strategy for Self-Evolving Agent Skills
**👍 234** · https://huggingface.co/papers/2605.23904 · GitHub: https://github.com/microsoft/SkillOpt

### 问题与动机
今天的 agent skill（技能文档）要么手写、要么一次性生成、要么靠松散的自我修订进化——没有一种像「深度学习优化器」那样，能在反馈下可靠地比起点更好。作者的洞察很犀利：应该把 skill 当作冻结 agent 的「外部状态」来训练，用让权重优化可复现的那套纪律来训练它。

### 方法与核心创新
SkillOpt 是首个系统化、可控的「文本空间技能优化器」。一个独立的优化器模型把打分后的 rollout 转成有界的 add/delete/replace 编辑，作用在单一技能文档上；**只有当编辑严格提升留出验证分时才接受**——这正是把 SGD 的「单调改进」纪律搬到文本上。配套有：文本学习率预算、被拒编辑缓冲、epoch 级的慢/元更新，让技能训练稳定，且部署时**零额外推理调用**（技能就是一份文档）。

### 关键实验结果
跨 6 个 benchmark、7 个目标模型、3 种执行环境（直接对话、Codex、Claude Code），在全部 52 个 (模型,benchmark,环境) 格子上都是最好或并列最好，击败 human / one-shot LLM / Trace2Skill / TextGrad / GEPA / EvoSkill 所有逐格竞品。在 GPT-5.5 上：直接对话提升 **+23.5 分**、Codex 循环内 **+24.8 分**、Claude Code 内 **+19.1 分**（相对无技能基线）。迁移实验显示优化后的技能跨模型规模、跨 Codex/Claude Code 环境、迁到邻近数学 benchmark 仍保留价值。

### 局限性与开放问题
依赖「可打分的留出验证集」，对难以自动评分的开放任务可能失效（摘要未明确，推断）。优化阶段需大量 rollout 打分，训练成本被转移到了 offline 而非消失。技能文档单文件的容量上限、技能间冲突如何组合，摘要未涉及。

### 启发与应用前景
对所有在用 Claude Code/Codex 做 agent 的人极有现实意义：技能可以像模型一样被「训练」而非靠手写。零推理开销 + 跨环境迁移意味着可以一次优化、到处部署。微软出品 + 8000+ star，工程成熟度高，是 agent skill 工程化的标杆。

---

## 3. AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security
**👍 142** · https://huggingface.co/papers/2605.29801

### 问题与动机
现代开放世界 agent（如 OpenClaw）有强大的跨环境执行能力，却带来大量新的安全风险来源；同时前沿模型大幅降低了攻击门槛，现有对齐框架已不够用于真实部署。需要一个既轻量又可扩展的 agent 安全对齐框架。

### 方法与核心创新
三步。其一，更新 agent 安全分类法，纳入 Codex、OpenClaw 这类执行场景下的新兴风险。其二，构建「分类法引导的数据引擎」+ 影响函数净化，仅用约 1k 样本就训练出多个轻量版本（0.8B/2B/4B/8B），达到与领先闭源模型（如 GPT-5.4）相当的性能——**1k 样本达成闭源级表现**是最反直觉的点。其三，搭建高效的 agentic 安全 SFT+RL 训练环境，把 Docker 级环境的部署开销**降低两个数量级**。最终作为免训练在线护栏做实时安全审查。

### 关键实验结果
在多样且复杂的交互式 agent 场景中达到 SOTA。核心数字：仅约 1k 样本、最小 0.8B 参数即可比肩 GPT-5.4 级闭源模型；部署开销下降两个数量级。模型与数据集全部开源。（与各基线的具体分差摘要未给出。）

### 局限性与开放问题
1k 样本训练 + 影响函数净化的可复现性高度依赖数据引擎质量，泛化到全新攻击面是否成立存疑（摘要未明确，推断）。作为「免训练护栏」的延迟开销、误杀正常操作的假阳性率未交代。安全分类法本身需随攻击演进持续更新。

### 启发与应用前景
对正在部署 agent 的团队是即插即用的安全层。「小模型 + 少样本 + 影响函数净化」做护栏的路线，证明安全审查不必靠大模型，可低成本嵌入推理链路。开源 0.8B 版本尤其适合边缘/本地部署。

---

## 4. Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
**👍 142** · https://huggingface.co/papers/2605.30280

### 问题与动机
具身智能长期被「一个任务一个专用模型」割裂：操作、导航各自为政，跨任务/环境/机器人形态的泛化差。本文问：异构的具身决策问题能否统一进单一 VLA 模型？

### 方法与核心创新
Qwen-VLA 把 Qwen 的视觉-语言栈从「感知-理解-推理」延伸到「连续动作与轨迹生成」，靠一个 **DiT 结构的动作解码器**。大规模联合预训练融合了机器人操作轨迹、人类第一视角演示、合成仿真、视觉语言导航、轨迹监督、辅助 VL 数据。为支持多机器人平台，引入**具身感知的 prompt 条件**：用机器人专属文字描述指定当前形态和控制约定。再把操作、导航、轨迹预测都铸成统一的「动作-轨迹预测」框架，实现跨形态的视觉锚定、空间推理与连续动作迁移。

### 关键实验结果
数据扎实：LIBERO **97.9%**、Simpler-WidowX **73.7%**、RoboTwin-Easy/Hard **86.1%/87.2%**、R2R 上 OSR **69.0%**、RxR 上 SR **59.6%**、真实 ALOHA 平均 OOD 成功率 **76.9%**、DOMINO 动态操作零样本 **26.6%**。在场景布局、背景、光照、物体配置、机器人形态变化下保持 OOD 泛化。

### 局限性与开放问题
DOMINO 动态操作零样本仅 26.6%，说明真正高动态/接触丰富任务仍是短板。摘要未给出与其他 VLA（如 π0、OpenVLA）的逐项对比，领先幅度难判。靠文字描述区分形态，新形态的零样本迁移效果未充分验证（摘要未明确，推断）。

### 启发与应用前景
统一 VLA 基座的代表作，背靠 Qwen 生态，工程落地潜力大。「embodiment-aware prompt」是低成本支持多机器人的实用 trick，对做异构机器人车队的团队直接可借鉴。

---

## 5. LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
**👍 141** · https://huggingface.co/papers/2605.27365

### 问题与动机
VLM 通常把视觉定位/检测当作「坐标 token 生成」问题，把一个 2D 框拆成多个 1D token 逐个独立学习和解码。这与框几何的耦合结构不匹配，且严格顺序生成造成推理瓶颈——又慢又损精度。

### 方法与核心创新
LocateAnything 基于 **Parallel Box Decoding (PBD)**：把边界框、点等几何元素作为「原子单位」一步解码，而非 token-by-token。这保留了框内几何一致性，同时解锁大量并行——既提吞吐又提定位精度。配套一个可扩展数据引擎，构建 **LocateAnything-Data，超 1.38 亿训练样本**，大幅增加高精度定位的数据多样性。

### 关键实验结果
显著推进了速度-精度前沿：解码吞吐明显更高，同时在多 benchmark 上**高 IoU 定位质量**也提升。论文强调 PBD 与大规模训练数据的互补效应。（具体加速倍数与各 benchmark 绝对数值摘要未给出。）

### 局限性与开放问题
摘要缺乏具体加速比和精度数字，难以量化「显著」到什么程度。一步并行解码框对「框数目可变」「框间重叠/遮挡」场景如何保证不冲突，未说明（摘要未明确，推断）。1.38 亿样本的数据引擎本身成本高。

### 启发与应用前景
「把结构化输出当原子单位并行解码」这个思路可推广到任何几何/结构化预测（关键点、多边形、布局）。对需要实时检测的 agent 视觉、GUI 定位、机器人感知有直接价值。NVIDIA 出品，定位偏底层基础能力。

---

## 6. DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning
**👍 136** · https://huggingface.co/papers/2605.25604

### 问题与动机
RL 已成 LLM 对齐标准范式，GRPO 作为免价值模型的高效替代很受欢迎，但适配到真实的「多奖励」设定很难。标准标量化做法各有硬伤：Reward Combination 常产生平方幅度过大的 advantage 导致训练不稳；Advantage Combination 靠静态超参、忽略跨目标相关性。

### 方法与核心创新
DVAO 根据每个目标在一个 rollout 组内的**经验奖励方差**动态调整组合权重——上调学习信号强的目标、压制噪声大的目标。这是关键直觉：方差大=信号弱，就该少信它。作者从数学上证明 DVAO **保持 advantage 幅度有界**（稳定训练），并引入自适应的跨目标正则机制处理目标相关性。

### 关键实验结果
在数学推理和工具使用 benchmark、Qwen3 与 Qwen2.5 模型上，DVAO 显著优于基线，取得更优的多目标 Pareto 前沿和更稳的训练。（具体提升点数摘要未给出，但有理论的有界性保证 + Pareto 前沿改进双重证据。）

### 局限性与开放问题
摘要无具体数值，领先幅度未知。基于「方差→权重」的自适应假设了「方差大即信号弱」，但高方差也可能来自任务本身难度而非噪声，这种混淆未讨论（摘要未明确，推断）。目标数量很多时的可扩展性也未交代。

### 启发与应用前景
多奖励 RLHF 的实用改进，对同时优化「正确性+格式+安全+简洁」等多目标的训练直接可用。「用组内方差自适应加权」思路简单且有理论支撑，易集成进现有 GRPO 管线。

---

## 7. Rethinking Cross-Layer Information Routing in Diffusion Transformers
**👍 110** · https://huggingface.co/papers/2605.20708

### 问题与动机
DiT 已是视觉生成主流骨干，分词、注意力、条件、目标、latent VAE 各轴都被反复重构过，唯独「跨层信息如何累积」的残差流，直接照搬了原始 Transformer 没人动。作者认为这是被忽视的设计轴。

### 方法与核心创新
先做系统实证：沿深度和去噪时间步分析跨层信息流，诊断出传统残差加法的三个症状——**前向幅度单调膨胀、反向梯度急剧衰减、块间显著冗余**。据此提出 **Diffusion-Adaptive Routing (DAR)**：一个即插即用的残差替换，对子层输出历史做「可学习、时间步自适应、非增量」的聚合（而非简单相加）。DAR 还与 REPA 等现代增强方法兼容。

### 关键实验结果
ImageNet 256×256 上，DAR 把 SiT-XL/2 的 FID **改善 2.11（7.56 vs 9.67）**，并用 **少 8.75 倍训练迭代**达到基线收敛质量。叠加 REPA 后早期阶段 **2 倍训练加速**。还能用于大规模 T2I 微调，在 DMD 蒸馏中保留高频细节。

### 局限性与开放问题
DAR 聚合历史子层输出会增加显存/计算（非增量聚合需保留历史），摘要未交代推理开销。在更大分辨率/视频 DiT 上的收益是否保持未验证（摘要未明确，推断）。

### 启发与应用前景
指出「残差流」是 DiT 被长期忽视的设计轴，是有启发性的研究方向。8.75 倍训练效率提升对算力受限的训练极有吸引力，且与 REPA 正交可叠加。即插即用属性意味着易于被现有 DiT 训练采纳。

---

## 8. Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models
**👍 110** · https://huggingface.co/papers/2605.21573 · GitHub: https://github.com/microsoft/Lens

### 问题与动机
T2I 模型越做越大、越训越贵。Lens 反其道：用 **3.8B 参数**达到甚至超过 6B+ SOTA 模型，且训练算力大幅更少——只用了 Z-Image 约 **19.3%** 的训练算力。核心问题是：训练效率能否系统性提升而非靠堆参数。

### 方法与核心创新
两条主线。其一最大化每个 batch 的数据信息密度：(i) 在 **Lens-800M**（8 亿密集描述图文对，caption 由 GPT-4.1 生成、平均约 109 词）上训练，远比短 caption 语义监督更丰富；(ii) 每个 batch 混合多分辨率、多长宽比图像，扩大每步优化的视觉覆盖。其二靠架构选择提收敛速度：语义 VAE 提供更好的 latent、强语言编码器加速优化并实现「仅英文训练→多语言泛化」。预训练后用带分类法 prompt 的 RL（Lens-RL-8K）+ 结构化奖励抑制瑕疵，reasoner 模块做免训练 system prompt 搜索，蒸馏加速到 4 步推理。

### 关键实验结果
3.8B 比肩/超越 6B+ 模型，训练算力仅 Z-Image 的 19.3%。泛化到 1:2~2:1 任意长宽比、最高 1440² 分辨率、多语言。速度：单张 H100 上生成 1024² 图 **3.15 秒**，蒸馏 turbo 版 4 步生成 **0.84 秒**。

### 局限性与开放问题
高度依赖 GPT-4.1 生成的 109 词密集 caption，数据构建成本与对教师模型的依赖被转移而非消除。与 6B+ 模型对比的具体 benchmark 分差摘要未细列（仅称 competitive/surpassing）。多语言泛化的非英语质量未量化（摘要未明确，推断）。

### 启发与应用前景
对算力有限的团队是范本：「数据信息密度 + 架构选择」可换取数倍算力。密集 caption + 多分辨率 batch 是可直接复用的训练技巧。微软开源 + HF 模型可用，落地门槛低。

---

## 9. WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation
**👍 102** · https://huggingface.co/papers/2605.25874 · GitHub: https://github.com/meituan-longcat/WBench

### 问题与动机
交互式世界模型进展飞快，但现有 benchmark 只覆盖部分能力，没有统一的系统评测标准。需要一个多维度、多轮交互的基准。

### 方法与核心创新
WBench 沿**五个维度**评测：视频质量、设定遵循、交互遵循、一致性、物理合规。含 **289 个测试用例、1058 个交互轮次**，每个用例给定世界设定+多轮交互序列，覆盖多样场景/风格/主体、第一与第三人称视角，四种交互类型（导航、主体动作、事件编辑、视角切换）。导航上统一了文本、6-DoF 位姿、离散动作三种控制，使不同原生输入接口的模型都能评。评测用 **22 个自动子指标**（专家视觉模型 + 大型多模态模型组合），全部对齐人类判断。

### 关键实验结果
评测 **20 个 SOTA 模型**，核心发现：**没有任何单一模型在所有维度都强**。论文给出每个模型的特征性优势、弱点、开放挑战的诊断洞察。（各模型具体分数摘要未列。）

### 局限性与开放问题
22 个自动指标虽称对齐人类，但多模态 LLM 做评判本身有偏差/不稳定风险。289 用例对「世界模型」这种高维空间覆盖是否充分存疑（摘要未明确，推断）。物理合规维度的自动评判最难，可靠性未细说。

### 启发与应用前景
本周多篇世界模型论文（#1 #22 #24）都缺统一评测，WBench 正好补位，是该方向的基础设施。五维度 + 多轮 + 多控制接口统一的设计，可作为后续世界模型论文的标准评测。美团出品，已开源代码数据。

---

## 10. Agent Explorative Policy Optimization for Multimodal Agentic Reasoning
**👍 91** · https://huggingface.co/papers/2605.28774

### 问题与动机
带扩展推理的 VLM 在复杂问题上表现好，但很多真实问题需要外部工具，光靠内部推理解决不了。Agentic 推理交织两种行为且存在结构性不对称：thinking（自包含的默认）和 tool use（高方差的辅助行动）。作者称之为 **Thinking-Acting Gap**。在 GRPO 下表现为两个症状：仅约 **30%** 的 rollout 会尝试工具；尝试时，约 **40%** 的问题上组内工具 rollout 全错——恰好在最需要工具的地方学习信号被压没了。

### 方法与核心创新
**AXPO**：对每个「全错的工具使用子组」，固定 thinking 前缀、只重采样工具调用及其后续，配合基于不确定性的前缀选择。直觉很准：问题不在思考，而在工具调用环节，那就专门给工具调用补足探索信号。

### 关键实验结果
跨 9 个多模态 benchmark、3 种规模的 Qwen3-VL-Thinking，**SFT+AXPO** 平均优于 **SFT+GRPO**（8B 上平均 Pass@1 **+1.8pp**、Pass@4 **+1.8pp**）。更亮眼的是 **8B 的 SFT+AXPO 在 Pass@4 上超过 32B Base，参数少 4 倍**。

### 局限性与开放问题
平均 +1.8pp 提升偏温和，是否在所有任务上稳定不退化未细说。方法专门针对「全错工具子组」，对部分对/部分错的中间情况是否最优未讨论（摘要未明确，推断）。重采样增加训练成本。

### 启发与应用前景
精准诊断 + 对症下药的范例：识别出 GRPO 在工具学习上的信号稀疏问题，用前缀固定+局部重采样解决。对所有训练「会用工具的推理 agent」的团队直接有用，尤其是工具调用稀疏/失败率高的场景。

---

## 11. ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation
**👍 87** · https://huggingface.co/papers/2605.28293 · GitHub: https://github.com/hongruhou89/ProRL

### 问题与动机
主动推荐系统（PRS）目标是通过生成一连串中间推荐，引导用户偏好向目标物品迁移。RL 天然适合这种序列决策（路径奖励能同时刻画短期接受度和长期引导效果），但朴素套用策略梯度会产生有缺陷的梯度估计。

### 方法与核心创新
作者诊断两个缺陷：(1) 路径级奖励分解成步级奖励后均值为正，造成**长度依赖偏置**——梯度偏好「延长路径」而非有意义探索；(2) 用整条路径奖励给每步加权，忽略分解结构，导致**高梯度方差**。ProRL 两个机制对症：**Stepwise Reward Centering** 减去期望奖励中和长度偏置，保证路径延长的期望梯度为零；**Position-Specific Advantage Estimation** 利用奖励分解结构算步依赖基线，降方差。

### 关键实验结果
三个真实世界数据集上，ProRL 显著优于 SOTA 的 PRS 方法。（具体提升幅度摘要未给出。）代码已开源。

### 局限性与开放问题
摘要无具体数值，难判断「显著」程度。方法针对 PRS 的特定奖励分解结构设计，迁移到其他序列推荐范式的通用性未知（摘要未明确，推断）。「引导用户偏好迁移」本身存在操纵伦理争议，论文未涉及。

### 启发与应用前景
对推荐系统工程团队有直接价值，把「长度偏置」和「方差」这两个策略梯度老问题在推荐场景下精确化处理。两个机制（reward centering、position-specific baseline）思路通用，可借鉴到其他路径式 RL 任务。

---

## 12. Macaron-A2UI: A Model for Generative UI in Personal Agents
**👍 82** · https://huggingface.co/papers/2605.24830

### 问题与动机
个人 agent 处理越来越复杂的用户任务，静态纯文本聊天正成为瓶颈。生成式 UI（Generative UI）作为新接口层，能从交互上下文实时合成正确的控件、选项和状态。

### 方法与核心创新
Macaron-A2UI 让 agent 在生成自然语言的同时，生成轻量、可执行的 UI 动作，用于信息收集、偏好细化、确认、多目标组织。从异构对话源构建大规模 Generative UI 语料，提出 **A2UI-Bench** 做受控评测，训练 **30B/235B/754B** 三档模型，用 LoRA 参数高效 SFT + 奖励驱动 RL。

### 关键实验结果
最佳 Macaron-A2UI 在 A2UI-Bench 上达 **75.6 总分**，且是在**无显式 schema 提示**下取得，**超过最强的「全 schema」前沿基线**——即不给它 UI 结构模板它也能赢给了模板的对手，含金量更高。模型、benchmark、评测协议全部开源。

### 局限性与开放问题
75.6 分的绝对值意味着仍有约四分之一差距，生成 UI 的可用性/正确性尚未到生产级。从对话源构建语料，覆盖的 UI 交互类型是否全面未知（摘要未明确，推断）。754B 模型的推理成本对「个人 agent」场景偏重。

### 启发与应用前景
指向 agent 交互形态的下一步：从纯文本到「文本+生成式 UI」。对做个人助理/对话 agent 产品的团队有前瞻价值。A2UI-Bench 提供了该方向的早期评测标准。

---

## 13. EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation
**👍 80** · https://huggingface.co/papers/2605.23271

### 问题与动机
生成式视频模型迈向专业级电影合成，但可靠评测成了关键瓶颈。现有 benchmark 多评「对不对」（基本 prompt 遵循），却忽略「好不好」（电影质感、表演、美学）；自动指标缺乏领域严谨性，人类美学感知与机器打分之间存在严重可信度鸿沟。

### 方法与核心创新
EvalVerse 把视频生成评测当核心科学问题——**主观电影专业知识的系统数字化**。三步：(1) 把领域知识组织成与专业电影工作流（前期、制作、后期）对齐的评测分类法；(2) 把人类专家判断蒸馏成大规模人工标注的数据集；(3) 用「专家校准微调」把知识注入 VLM，使其能做显式 CoT 推理。既兼容「对不对」的基础指标，又扩展到「好不好」，并覆盖复杂多镜头排序和视听整合。

### 关键实验结果
EvalVerse 超越静态排行榜，提供细粒度诊断信号，可作为奖励模型和评估器 agent 的基础设施。（评测的模型数量与具体指标数值摘要未给出。）

### 局限性与开放问题
摘要偏理念阐述，缺具体数据（覆盖多少模型、与人类一致性多少）。「电影美学」高度主观，专家标注的代表性和文化偏差难以消除（摘要未明确，推断）。VLM 做 CoT 美学评判的稳定性未验证。

### 启发与应用前景
随着视频生成转向 RL 和 agentic 工作流，「评什么是好」直接决定奖励信号质量，EvalVerse 切中要害。对训练视频生成奖励模型的团队是关键基础设施。「把专业主观知识数字化」的方法论可推广到其他需要专家审美的生成任务。

---

## 14. Foundation Protocol: A Coordination Layer for Agentic Society
**👍 80** · https://huggingface.co/papers/2605.23218 · GitHub: https://github.com/FoundationAgents/foundation-protocol

### 问题与动机
自治 agent 正从工具变成社会基础设施层：浏览、购买、部署软件、管理系统、彼此交互。系统规模化后，瓶颈从「模型原始能力」转向「协调」——agent 需要形成可靠关系、组织多智能体协作、交换价值、支撑 AI 经济，并在真实监管下安全可问责。

### 方法与核心创新
Foundation Protocol (FP) 是一个**图优先的协调层**。统一异构实体（agent、工具、资源、人、机构、组织），支持原生的多方组织和事件驱动协作。提供经济原语：计量、收据、结算；把策略、溯源、审计作为一等公民。设计上是**包装并桥接现有协议而非取代**，支持增量采用、降低集成与治理开销。核心理念：让自治 agency 保持可组合，同时让问责不可妥协。

### 关键实验结果
本文是协议/愿景型论文，无传统实验指标（摘要未给出基准评测）。贡献在于协调层的概念架构与经济/治理原语设计。

### 局限性与开放问题
作为协议提案，缺乏实证验证——真实多 agent 系统采用后的可扩展性、性能开销、安全性都未验证。「图优先」协调在 agent 数量极大时的扩展性、经济结算的防作弊机制等核心问题未细说（摘要未明确，推断）。协议成功高度依赖生态采纳。

### 启发与应用前景
站在「agent 社会」的高度思考基础设施，前瞻性强。把「问责、溯源、审计」作为一等公民，对监管和企业落地有现实意义。「包装而非取代」的增量策略务实。对构建多 agent 平台/AI 经济的团队是值得跟踪的标准候选。

---

## 15. OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources
**👍 76** · https://huggingface.co/papers/2605.29250 · GitHub: https://github.com/JinheonBaek/OmniRetrieval

### 问题与动机
真实信息需求要访问结构各异的知识源：非结构化文本、关系表、知识图谱、属性图。现有检索器一次只处理一种源、固定查询语言，把知识割裂在不兼容的接口后面。直接「塞进共享空间」做统一，又会抹掉 schema、本体、组合算子这些让每种源有表达力的结构特性。

### 方法与核心创新
OmniRetrieval 的关键洞察：有效检索需要的不是同质化，而是一个**在各源自身条件上与之对接的总览层**。框架接受任意自然语言查询，识别合适的知识源，把**源原生的查询**分派给各自的原生执行引擎——而非把所有东西压成向量。这样既统一了接口，又保留了各源的结构区分。

### 关键实验结果
跨 **13 个数据集、309 个不同知识库**（文本、关系、图结构源）的大规模 benchmark 上，OmniRetrieval 超越单源基线，证明可作为异构源的通用接口同时保留结构价值。（与具体基线的分差摘要未给出。）

### 局限性与开放问题
摘要无具体提升数值。「识别合适源 + 生成源原生查询」依赖 LLM 的路由与查询生成准确性，路由错误的影响未量化（摘要未明确，推断）。支持新源需为其编写原生查询适配，扩展成本存在。

### 启发与应用前景
对 RAG/agent 检索是重要方向：现实知识本就异构，「保留结构而非同质化」的理念纠正了「全塞向量库」的惯性。对构建企业知识库 agent（同时有数据库、文档、知识图谱）极有实用价值。已开源。

---

## 16. From Pixels to Words -- Towards Native One-Vision Models at Scale
**👍 73** · https://huggingface.co/papers/2605.28820

### 问题与动机
当前 VLM 通常把独立的图像编码器和语言解码器靠多阶段对齐「缝合」起来，这种模块化框架不可避免地把像素级信号在帧间割裂、把早期像素-词交互打散。同时原生 VLM 虽在单图上表现亮眼，但在多图、视频理解、空间智能上几乎没被探索。

### 方法与核心创新
NEO-ov 是一个**原生基础模型**，端到端学习跨帧和像素-词对应，**不用任何外部编码器、辅助适配器或事后融合**。通过彻底消除模块边界，让细粒度、统一的时空建模在模型内部原生涌现。论文同时给出系统的架构分析和详细训练配方，以推动后续原生多模态建模。

### 关键实验结果
NEO-ov **大幅缩小了与模块化对手的差距**，同时在细粒度视觉感知上表现出色，验证了原生「one-vision」架构在规模上不仅可行而且有竞争力。（与模块化 VLM 的具体 benchmark 分差摘要未给出。）

### 局限性与开放问题
摘要坦承是「缩小差距」而非全面超越模块化方案——原生架构仍未在所有任务上领先。具体 benchmark 数值、模型规模、训练数据量摘要未给出，难判断代价。原生端到端训练通常数据/算力需求更高（摘要未明确，推断）。

### 启发与应用前景
回答了一个根本问题：VLM 是否必须「缝合」？NEO-ov 证明原生统一架构可扩展，是多模态架构路线之争的重要一票。公开的架构分析和训练配方对研究者价值高。指向「消除模块边界」的长期方向。

---

## 17. SpatialBench: Is Your Spatial Foundation Model an All-Round Player?
**👍 72** · https://huggingface.co/papers/2605.27367 · GitHub: https://github.com/Ropedia/SpatialBench

### 问题与动机
空间基础模型在标准数据集上表现亮眼，但一个关键问题悬而未决：它们真能跨多样下游任务、任意视角、变化场景域、不同输入密度、特定硬件约束稳健泛化吗？现有模型多在为其专门设计/训练的特定域上评测，受限于窄范式覆盖、有限场景域、随意帧采样，难以评估真实泛化。

### 方法与核心创新
SpatialBench 是跨范式、域多样、**确定性采样**的基准。规模空前且设计严谨：含 **19 个数据集、546 个场景、5 个空间域**，评测 **41 个模型、6 种范式、5 个任务套件、4 种输入密度设置**。确定性采样消除了随意帧采样的不公平性。此外还引入大规模数据集 **DA-Next-5M** 和强基线模型 **DA-Next**，超越纯评测。

### 关键实验结果
广泛评测揭示：当前模型**还不是全能选手**。关键洞察：**全上下文注意力最大化精度，而有界内存策略解锁长序列可扩展性**；在具身/第一视角任务上，**严格域对齐和高数据质量远比单纯数据集扩展更关键**——这是对「scaling 万能论」的有力反驳。

### 局限性与开放问题
评测了 41 个模型但摘要未列具体排名/分数。「确定性采样」虽更公平，但是否充分覆盖空间任务的高维变化空间存疑（摘要未明确，推断）。DA-Next 作为论文自家基线，与第三方 SOTA 的对比公正性需关注。

### 启发与应用前景
对空间基础模型领域是重要基础设施，且给出反直觉洞察（域对齐 > 数据扩展），直接影响后续训练策略。「全上下文 vs 有界内存」的权衡发现对部署有实用指导。开源 + 自带强基线，可用性高。

---

## 18. MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research
**👍 64** · https://huggingface.co/papers/2605.26114 · GitHub: https://github.com/Purewhiter/mobilegym

### 问题与动机
移动 GUI agent 研究缺乏既可验证又可大规模并行的训练环境。真机执行慢、不可复现、难提供密集 RL 奖励；复刻专有后端又不现实。需要一个轻量、完全可控、能给确定性奖励信号的环境。

### 方法与核心创新
MobileGym 是**浏览器托管**的轻量环境，追求交互保真而非复刻专有后端。两大能力：(1) **可验证的结果信号**——对结构化 JSON 状态做确定性、基于状态的判定；(2) **可扩展的在线 RL**——低成本并行 rollout。全部环境状态以 JSON 捕获/配置/fork/比较，单台服务器可托管**数百个并行实例**，每实例约 **400 MB 内存、约 3 秒冷启动**。配套 **MobileGym-Bench**：**416 个参数化任务模板**（256 测试 + 160 训练）、覆盖 **28 个 app**，确定性判定 + 结构化 AnswerSheet 协议避免自由文本匹配失败。

### 关键实验结果
Sim-to-Real 案例：GRPO 训 Qwen3-VL-4B-Instruct 在 256 任务测试集上 **+12.8 个百分点**；在 59 任务真机信号子集上，**真机执行保留了 95.1% 的仿真侧训练增益**——这个 sim-to-real 保留率是最关键的可信度证据。

### 局限性与开放问题
浏览器模拟 app 与真实 app 行为必有差距，覆盖 28 个 app、状态用 JSON 抽象，对复杂动态界面的保真度有上限（摘要未明确，推断）。59 任务的真机验证子集偏小。

### 启发与应用前景
对 GUI agent / 移动自动化研究是关键基础设施——解决了「可验证 + 可并行 RL」的核心痛点，95.1% sim-to-real 保留率说明仿真训练真能迁移。627 star 说明社区认可度高，可直接用于训练移动 agent。

---

## 19. CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation
**👍 61** · https://huggingface.co/papers/2605.25378 · GitHub: https://github.com/Qwen-Applications/CollectionLoRA

### 问题与动机
定制化图像编辑常用 LoRA 给扩散模型加特定视觉效果。但效果数量增多时，存储和动态加载大量 LoRA 显著增加部署开销；且当前流程把效果 LoRA 与加速模块级联，触发严重参数干扰，导致概念串扰和风格退化。

### 方法与核心创新
CollectionLoRA 是多教师 on-policy 蒸馏框架，能把**多达 50 个不同效果 LoRA** 连同少步生成能力**蒸馏进单个 LoRA**，从根本解决特征干扰、大幅降部署成本。三个机制：(i) **概率双流路由**，训练时随机切换数据源，增强对未见场景的泛化；(ii) **非对称正交 Prompting**，在 prompt 空间实现概念隔离；(iii) **由粗到细蒸馏目标**，缓解师生分布差。

### 关键实验结果
单个 LoRA 蒸馏了全部定制效果 + 少步生成，降部署开销，同时概念保真度**比肩或优于独立训练的教师模型**。（具体 FID/用户偏好数值摘要未给出。）

### 局限性与开放问题
摘要无具体量化指标，「比肩或优于」程度难判。50 个效果是上限还是可继续扩展未说明，效果数继续增多时单 LoRA 容量是否饱和存疑（摘要未明确，推断）。概念隔离对高度相似效果的有效性未验证。

### 启发与应用前景
对图像编辑产品的部署是实用工程方案：50 合 1 + 少步生成直接降存储和延迟成本。「概念隔离 + 多教师蒸馏」组合可推广到其他需要合并多个专用适配器的场景。Qwen-Applications 出品，工程导向明确。

---

## 20. Why Far Looks Up: Probing Spatial Representation in Vision-Language Models
**👍 60** · https://huggingface.co/papers/2605.30161 · GitHub: https://github.com/cheolhong0916/contrastive-probing

### 问题与动机
VLM 在空间推理 benchmark 上表现强，但不清楚这反映的是结构化 3D 理解，还是依赖自然图像里的统计捷径。这是「benchmark 高分 ≠ 真理解」的典型可解释性问题。

### 方法与核心创新
作者提出表征层分析框架：构造**最小对比对**来度量空间轴在 VLM embedding 内如何组织和解耦。跨多个模型家族的分析揭示一致的**垂直-距离纠缠**——模型把垂直图像位置与距离混淆，正是自然照片的透视偏置。为把这种偏置与评测集偏斜分离，引入合成基准 **SpatialTunnel**，移除自然图像里常见的相关性，专门暴露空间捷径偏置。

### 关键实验结果
该偏置在「透视一致」与「反启发式」样本间产生**显著精度差**，且**随数据扩展而加剧**，即便整体 benchmark 精度在提升——这点最警醒：scaling 不仅没修复反而放大了捷径。还证明 benchmark 分数相近的模型内部表征可不同，且这些差异能预测跨基准的精度和稳健性；空间轴分离良好的模型更稳健。

### 局限性与开放问题
分析基于「最小对比对」和合成 SpatialTunnel，合成基准与真实任务的差距可能引入新偏置（摘要未明确，推断）。给出了诊断却未提出修复纠缠的训练方法，是开放问题。具体精度差数值摘要未列。

### 启发与应用前景
重要的可解释性工作，戳破「空间 benchmark 高分=真 3D 理解」的假象，且发现 scaling 放大捷径偏置，对空间推理模型的训练和评测都有警示意义。「表征解耦程度预测稳健性」可作为模型选择/诊断的新指标。

---

## 21. Self-Improving Language Models with Bidirectional Evolutionary Search
**👍 59** · https://huggingface.co/papers/2605.28814 · GitHub: https://github.com/Embodied-Minds-Lab/BES

### 问题与动机
搜索被认为是自改进 LM 和 agent 系统的有效方法（用于后训练样本生成和推理）。但 best-of-N、树搜索有两个根本局限：靠稀疏的验证信号引导；且主要靠自回归扩展构造候选，把探索限制在模型概率质量大的区域。

### 方法与核心创新
**Bidirectional Evolutionary Search (BES)** 耦合「前向候选进化」和「后向目标分解」。前向搜索在标准扩展外加入**进化算子**，重组部分轨迹生成单次 rollout 难以得到的候选；后向搜索把原任务**递归分解成可检查的子目标**，产生密集中间反馈引导前向。理论上证明：纯扩展搜索的候选被困在**狭窄熵壳**内，进化算子能逃出；后向搜索能**指数级减少**找到正确答案所需的样本数。

### 关键实验结果
在主流后训练算法都无法提升的困难后训练任务上，BES 带来一致增益；在三个开放问题求解 benchmark 的推理时，BES 在平均和最佳表现上都**超越现有开源框架**。（具体提升数值摘要未给出，但有理论的「指数级减样本」保证。）

### 局限性与开放问题
摘要无具体数值。后向目标分解依赖任务能被分解成「可检查子目标」，对难以分解的开放问题适用性受限（摘要未明确，推断）。进化算子重组轨迹 + 双向搜索的计算开销可能较高。

### 启发与应用前景
「双向搜索 + 进化重组」突破了自回归扩展的探索局限，有扎实理论支撑（熵壳、指数减样本），对 test-time scaling 和自改进方向有启发。对做推理时搜索、后训练样本生成的团队直接可用，已开源。

---

## 22. minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models
**👍 58** · https://huggingface.co/papers/2605.30263 · GitHub: https://github.com/shengshu-ai/minWM

### 问题与动机
视频扩散基础模型在高质量生成上进展显著，但把它们变成实时交互式视频世界模型仍难。交互式世界模型需要可控、因果、低延迟的 rollout，实践中需要贯穿数据构建、可控微调、自回归训练、少步蒸馏、流式推理的完整管线——门槛高且零散。

### 方法与核心创新
minWM 是**全栈开源框架**，提供端到端管线，把现有双向 T2V/TI2V 视频基础模型转成相机可控的少步自回归世界模型。先用相机控制微调双向视频扩散模型，再应用 **Causal Forcing / Causal Forcing++** 管线（AR 扩散训练、因果 ODE 或因果一致性蒸馏、非对称 DMD），蒸馏成少步自回归生成器实现低延迟。框架模块化、架构可扩展，已在 **Wan2.1-T2V-1.3B、HY1.5-TI2V-8B** 等开放骨干上实例化，覆盖交叉注意力条件注入和 MMDiT 两类架构，还支持把 HY-WorldPlay 适配到新数据分布/训练配方/延迟目标。

### 关键实验结果
本文是框架/工具型贡献，发布可运行脚本、checkpoint、文档、推理代码，并提供相机轨迹质量、可控性训练步数、最小 batch 需求的实用消融。（无传统 benchmark 横评数值。）

### 局限性与开放问题
作为框架论文，缺乏与其他世界模型在统一指标上的对比（可用 #9 WBench 评测）。少步蒸馏 + 自回归通常牺牲长程一致性，累积漂移情况摘要未量化（摘要未明确，推断）。

### 启发与应用前景
对世界模型研究是降门槛的基础设施：把分散的全栈管线整合成可复现配方，且支持把现成视频扩散模型「升级」为交互式世界模型。608 star + 多骨干支持说明实用性强，与 #1 #24 #9 共同构成本周世界模型生态。

---

## 23. SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research
**👍 58** · https://huggingface.co/papers/2605.22878 · GitHub: https://github.com/zjunlp/SciAtlas

### 问题与动机
全球学术产出指数增长造成「信息爆炸」，碎片化、非结构化的知识组织阻碍深度跨学科整合。现有学术检索工具靠浅层关键词匹配或向量语义检索，缺乏导航复杂逻辑关系的拓扑推理能力；基于 agentic 深度研究的框架又易出逻辑幻觉、推理成本高。

### 方法与核心创新
SciAtlas 是大规模、多学科、异构的学术知识图谱，作为「全景科学演化网络」。整合 **26 个学科超 4300 万篇论文**，共 **1.57 亿实体、30 亿三元组**，提供结构化拓扑认知基底，打破学科壁垒。配套**神经-符号检索算法**（三路协同召回 + 图重排），实现从语义匹配到确定性关联发现的过渡。应用方向含文献综述、自动研究趋势综合、idea 定位、学术轨迹探索。

### 关键实验结果
本文是知识图谱/系统型贡献，核心数字在规模：43M 论文、157M 实体、3B 三元组、26 学科。论文称能赋能自动科研全流程并显著降低推理成本，已在 GitHub 发布 KG 检索和下游任务接口。（与基线检索器的具体精度对比摘要未给出。）

### 局限性与开放问题
摘要重规模轻评测，缺乏检索质量的量化对比。30 亿三元组的构建质量、实体消歧准确率、知识时效性更新机制摘要未交代（摘要未明确，推断）。神经-符号检索在超大图上的延迟也未说明。

### 启发与应用前景
对「自动科研」「AI4Science」方向是重量级基础设施，给 AI agent 提供全局科学认知地图，对治深度研究 agent 的逻辑幻觉和高成本问题。三路协同召回 + 图重排的神经-符号思路对知识密集检索有借鉴价值。浙大 NLP 出品，已开源接口。

---

## 24. YoCausal: How Far is Video Generation from World Model? A Causality Perspective
**👍 54** · https://huggingface.co/papers/2605.30346 · GitHub: https://github.com/youzhe0305/YoCausal

### 问题与动机
视频扩散模型（VDM）迈向世界模型时，一个关键问题：它们真懂因果，还是只是过拟合了统计的时间模式？现有 benchmark 多用合成数据，受 sim-to-real gap 限制泛化差。

### 方法与核心创新
YoCausal 是受认知科学「违反预期（VoE）」范式启发的两级 benchmark。巧思在于**把真实视频时间反转，零成本得到自然反事实样本**，建立可任意扩展的评测协议。**Level 1** 引入 Reverse Surprise Index (RSI)，用去噪损失量化对「时间之箭」的感知；**Level 2** 引入 Causality Cognition Index (CCI)，用 VLM 把数据集分层成因果/非因果子集，把真正的因果推理与单纯的时间偏置解耦。

### 关键实验结果
评测 **13 个 SOTA VDM**，核心发现：**能感知时间之箭并不意味着理解因果**，且相对人类水平的因果认知仍存在显著差距。这个「感知≠理解」的结论与 #20 的精神一致——揭穿表面能力。

### 局限性与开放问题
「时间反转作为反事实」是聪明的零成本设计，但反转视频有时本身物理合理（如水流双向），这类样本会污染信号（摘要未明确，推断）。CCI 依赖 VLM 做因果分层，VLM 自身因果能力有限会引入偏差。各模型具体 RSI/CCI 分数摘要未列。

### 启发与应用前景
切中「视频生成 ≠ 世界模型」的核心争论，用因果视角提供了可证伪的评测。「时间反转造反事实」零成本且可扩展，方法论很优雅，可作为世界模型因果性评测标准。与 #1 #22 #9 共同构成本周世界模型的「能力-评测」生态。

---

## 25. TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction
**👍 52** · https://huggingface.co/papers/2605.26115 · GitHub: https://github.com/ziplab/TriSplat

### 问题与动机
稀疏视图 3D 重建越来越多用前馈 splatting 网络直接从图像预测显式基元。但多数方法以高斯基元为中心、只间接暴露表面——要拿到可用于下游仿真/物理推理/具身交互的 mesh，仍需昂贵的事后步骤，破坏了「前馈」的承诺。无位姿设定下更难，结构和相机参数都要从稀疏观测联合估计。

### 方法与核心创新
TriSplat 用**有向三角形基元**表示场景，从单次前向直接导出仿真就绪的 mesh 场景。给定输入图像，网络预测局部 3D 点图、三角形属性、相机位姿和可选内参。关键 trick：不把三角形朝向当无约束的隐变量回归，而是**从预测点图构造几何法线**，用图像条件法线头精修，再转成稳定局部坐标系做三角形参数化。配 mono-normal bootstrap 调度稳定早期训练，opacity/blur 调度渐进锐化表面以便直接提取 mesh。

### 关键实验结果
在 **RealEstate10K 和 DL3DV** 上，该表示比高斯前馈基线产生**更几何忠实**的重建，同时保持有竞争力的新视图渲染质量。因为渲染基元本身就是表面三角形，输出可被物理引擎、碰撞检测、标准渲染管线**直接摄取无需转换**。（具体 PSNR/Chamfer 数值摘要未给出。）

### 局限性与开放问题
三角形基元在渲染质量上可能略逊纯高斯（摘要称「有竞争力」而非超越）。摘要无具体量化指标，几何忠实度的提升幅度难判。复杂细薄结构、透明/反射表面的三角形重建效果未验证（摘要未明确，推断）。

### 启发与应用前景
直击具身 AI / 机器人仿真的痛点：从图像到仿真就绪 mesh 一步到位，省去高斯转 mesh 的昂贵后处理。「用点图法线约束三角形朝向」是稳定可借鉴的几何先验。对需要从稀疏图像快速构建可交互 3D 场景的机器人/仿真团队价值高，已开源。

---

## 26. ResearchMath-14K: Scaling Research-Level Mathematics via Agents
**👍 50** · https://huggingface.co/papers/2605.28003

### 问题与动机
数学前沿由「解尚未知」的问题定义，但 LM 能否无需人类干预地有意义地处理这类问题仍不清楚。主要障碍是缺乏大规模的研究级数学数据集。

### 方法与核心创新
作者用**多智能体 pipeline** 从学术源精选出 **14,056 个问题**，成为迄今最大的研究级数学问题集。进一步从两个开放模型生成 **ResearchMath-Reasoning（22 万条教师轨迹）**，并观察到反复出现的回避行为——不作答和**捏造引用**。有意思的发现：跨 8 个开源权重模型，**越新的世代每条轨迹产生多 5.6 倍的引用、多 5.0 倍的假引用**——能力越强幻觉越多，反直觉且警醒。经 agentic 过滤后，微调 4B~30B 的 Qwen3 模型比基础模型**平均提升 9.2 分**。

### 关键实验结果
14,056 个研究级问题（最大规模）；220K 教师轨迹；新世代模型假引用增长 5.0 倍；agentic 过滤后微调 Qwen3 (4B-30B) 平均 **+9.2 分**。核心结论：**即便推理轨迹不完全正确，过滤后的开放问题尝试仍能提供有用监督**。ResearchMath-14k 已公开。

### 局限性与开放问题
「捏造引用随能力增强而增多」虽是亮点发现，但论文未给出根治方法，是开放问题。研究级问题本身「无已知答案」，监督信号质量难保证、过滤标准的可靠性存疑（摘要未明确，推断）。+9.2 分提升在「研究级」难度下绝对水平可能仍低。

### 启发与应用前景
对 AI4Math、前沿数学推理是稀缺的大规模数据贡献。「假引用随能力增强而增多」的发现对所有用 LLM 做科研辅助的人都是重要警示——越强的模型越会一本正经编造引用。「不完全正确的轨迹也有监督价值」对数据稀缺领域的训练策略有启发。

---

## 🗺️ 趋势洞察

### 1. 视频世界模型从「能生成」转向「能交互、懂因果、可评测」
**涉及论文**：#1 Gamma-World, #9 WBench, #22 minWM, #24 YoCausal（外加 #7 #8 #13 的视频生成基础）
**核心观点**：本周最热的方向是交互式视频世界模型，且已形成完整生态闭环。#1 解决「多智能体如何共存」（单纯形 RoPE + 稀疏 hub 注意力，24 FPS 实时、两人泛化到四人）；#22 解决「如何把现成视频扩散模型升级成交互式世界模型」（minWM 全栈开源管线）；#9 解决「如何统一评测」（五维度 289 用例，发现无单一模型全维度强）；#24 从因果视角追问「视频生成离真正的世界模型还有多远」（发现感知时间之箭 ≠ 理解因果）。能力建设、工具链、评测、批判性反思四件事同周出现，说明该方向正快速成熟。值得注意的是 #9 和 #24 都泼了冷水——能力提升的同时，评测和因果理解的短板被同步暴露。

### 2. Agent 工程化：技能、安全、协调、训练环境全面基础设施化
**涉及论文**：#2 SkillOpt, #3 AgentDoG 1.5, #14 Foundation Protocol, #18 MobileGym, #10 AXPO, #15 OmniRetrieval, #23 SciAtlas
**核心观点**：Agent 研究的重心从「单个 agent 更强」转向「让 agent 可训练、可治理、可协调、可被供养」。#2 把技能当外部状态用优化器纪律训练（GPT-5.5 上 +19~25 分）；#3 用 1k 样本训出比肩 GPT-5.4 的轻量安全护栏；#14 提出 agent 社会的图优先协调层把问责作为一等公民；#18 提供可验证+高并行的 GUI agent 训练环境（95.1% sim-to-real 保留）；#10 精准修复工具学习的信号稀疏；#15/#23 给 agent 喂异构检索和科学知识图谱。一个清晰信号：底层模型能力不再是唯一瓶颈，技能优化、安全护栏、协调协议、训练环境、知识供给这些「agent 基础设施」成了新战场。多篇出自微软、浙大等机构且高 star，工程成熟度明显上升。

### 3. 效率优先：用更聪明的设计而非更大的规模换性能
**涉及论文**：#7 DAR, #8 Lens, #5 LocateAnything, #19 CollectionLoRA, #6 DVAO
**核心观点**：在算力受限的现实下，本周多篇论文系统性地用「设计巧思」替代「规模堆叠」。#8 Lens 用 3.8B 参数、仅 19.3% 算力比肩 6B+ 模型（靠密集 caption + 多分辨率 batch）；#7 DAR 重新设计 DiT 被忽视的残差流，8.75 倍训练效率提升；#5 用并行框解码同时提速提精度；#19 把 50 个 LoRA 蒸进 1 个降部署成本；#6 用方差自适应稳定多奖励 RL。共同主题是「找被忽视的设计轴」——残差流、解码并行、数据信息密度、advantage 组合，都是惯例照搬却未优化的环节。

### 4. 可解释性与诚实性：揭穿「高分≠真能力」
**涉及论文**：#20 Why Far Looks Up, #24 YoCausal, #26 ResearchMath-14K, #17 SpatialBench
**核心观点**：一股「祛魅」思潮——benchmark 高分不等于真理解，而且 scaling 可能放大问题。#20 发现 VLM 把垂直位置混淆为距离的捷径，且随数据扩展而加剧；#24 发现视频模型感知时间之箭却不懂因果；#26 发现越强的模型越会捏造引用（新世代假引用多 5 倍）；#17 发现空间模型还不是全能选手，且域对齐比数据扩展更关键。四篇都在用更精巧的诊断工具（对比探针、反事实、过滤轨迹、确定性采样）戳破表面能力的假象。

### 对比与张力
- **能力 vs 评测的张力**：世界模型（#1 #22 ）和 agent（#2 #3 ）在拼命提升能力，而 #9 #24 #20 #26 #17 在揭示这些能力的虚高和盲区。两股力量同周出现，反映领域进入「狂奔后回头审视」的阶段。
- **规模 vs 设计的张力**：#17 明确说「域对齐和数据质量远比数据集扩展关键」，#26 发现「越大越爱编造引用」，#8 #7 用小模型/巧设计赢大模型——多篇论文集体质疑「scaling 万能」，与过去几年的扩展叙事形成张力。
- **统一 vs 保留结构的张力**：#4 Qwen-VLA、#16 NEO-ov 主张「统一/消除模块边界」，而 #15 OmniRetrieval 明确反对「同质化」、主张「在各源自身条件上对接」。统一到什么程度、何时该保留异构性，是悬而未决的设计哲学之争。

### 值得关注的研究方向
1. **世界模型的因果性与长程一致性**：#24 指出感知≠因果，#1 #22 的实时少步生成都面临累积漂移，「如何让世界模型真正理解因果并保持长时一致」是下一个硬骨头。
2. **Agent 技能/安全的可训练化与可迁移性**：#2 证明技能能像权重一样被优化且跨环境迁移，#3 证明安全护栏能低成本训练，这条「agent 外部状态工程化」路线潜力巨大。
3. **被忽视的架构设计轴**：#7 对 DiT 残差流的重审是范例，提示还有多少「照搬 Transformer 惯例」的环节值得系统重构。
4. **可解释性驱动的诊断指标**：#20 提出「表征解耦程度可预测稳健性」，把可解释性从事后分析变成模型选择/训练的前置指标，是有前景的方向。
5. **仿真就绪与 sim-to-real**：#25 TriSplat、#18 MobileGym 都指向「让仿真产物直接可用于真实/下游」，95.1% sim-to-real 保留率和一步出 mesh 说明这条路在加速成熟。
