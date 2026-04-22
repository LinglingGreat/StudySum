# HuggingFace 2026-03 热门论文深度总结

> 来源：`https://huggingface.co/api/daily_papers?month=2026-03&limit=100`
> 统计日期：2026-04-22
> 筛选条件：点赞 ≥ 100
> 论文数：37 篇（覆盖 2026-02-24 至 2026-03-30 发布）

## 目录

1. [AI Can Learn Scientific Taste](#1-ai-can-learn-scientific-taste) (👍424)
2. [Demystifying Video Reasoning](#2-demystifying-video-reasoning) (👍369)
3. [CARLA-Air: Fly Drones Inside a CARLA World](#3-carla-air) (👍340)
4. [InCoder-32B: Code Foundation Model for Industrial Scenarios](#4-incoder-32b) (👍308)
5. [SocialOmni: Audio-Visual Social Interactivity Benchmark](#5-socialomni) (👍248)
6. [GOLF: Group-Level NL Feedback in RL](#6-golf) (👍210)
7. [HACRL: Heterogeneous Agent Collaborative RL](#7-hacrl) (👍194)
8. [Helios: Real Real-Time Long Video Generation](#8-helios) (👍186)
9. [MiroThinker-1.7 & H1: Heavy-Duty Research Agents](#9-mirothinker) (👍185)
10. [Utonia: One Encoder for All Point Clouds](#10-utonia) (👍185)
11. [Attention Residuals (AttnRes)](#11-attention-residuals) (👍182)
12. [HyDRA: Hybrid Memory for Dynamic Video World Models](#12-hydra) (👍156)
13. [ShotStream: Streaming Multi-Shot Video Generation](#13-shotstream) (👍155)
14. [Seoul World Model](#14-seoul-world-model) (👍153)
15. [Qianfan-OCR: Unified Document Intelligence](#15-qianfan-ocr) (👍153)
16. [dLLM: Simple Diffusion Language Modeling](#16-dllm) (👍153)
17. [HSImul3R: Physics-in-the-Loop HSI Reconstruction](#17-hsimul3r) (👍152)
18. [OpenClaw-RL: Train Any Agent by Talking](#18-openclaw-rl) (👍152)
19. [OmniLottie: Vector Animation Generation](#19-omnilottie) (👍151)
20. [OpenSeeker: Open-Source Frontier Search Agents](#20-openseeker) (👍149)
21. [EnterpriseOps-Gym: Enterprise Agent Benchmark](#21-enterpriseops-gym) (👍148)
22. [ReBalance: Efficient Reasoning with Balanced Thinking](#22-rebalance) (👍148)
23. [RL3DEdit: Geometry-Guided RL for 3D Scene Editing](#23-rl3dedit) (👍145)
24. [LongCat-Next: Discrete Native Multimodal](#24-longcat-next) (👍144)
25. [TAPS: Task-Aware Speculative Sampling](#25-taps) (👍142)
26. [MetaClaw: Continual Meta-Learning Agent](#26-metaclaw) (👍139)
27. [ADE-CoT: Adaptive Test-Time Scaling for Image Editing](#27-ade-cot) (👍138)
28. [MinerU-Diffusion: Document OCR as Inverse Rendering](#28-mineru-diffusion) (👍135)
29. [Intern-S1-Pro: 1T Scientific Multimodal Model](#29-intern-s1-pro) (👍131)
30. [Omni-WorldBench: Interaction-Centric 4D Benchmark](#30-omni-worldbench) (👍126)
31. [daVinci-MagiHuman: Single-Stream Audio-Video](#31-davinci-magihuman) (👍123)
32. [T2S-Bench & Structure-of-Thought](#32-t2s-bench) (👍121)
33. [Penguin-VL: LLM-based VLM Encoder](#33-penguin-vl) (👍119)
34. [PixelSmile: Fine-Grained Facial Expression Editing](#34-pixelsmile) (👍117)
35. [Astrolabe: Forward-Process RL for AR Video](#35-astrolabe) (👍109)
36. [HopChain: Multi-Hop VL Reasoning Data](#36-hopchain) (👍109)
37. [Beyond Language Modeling: Multimodal Pretraining](#37-beyond-language-modeling) (👍103)

[🗺️ 趋势洞察](#-趋势洞察)

---

## 1. AI Can Learn Scientific Taste
**👍 424** · [arXiv:2603.14473](https://huggingface.co/papers/2603.14473) · [GitHub (391★)](https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste)

### 问题与动机
现有 AI-for-Science 研究几乎全押在「执行能力」上——让模型跑实验、写代码、查文献。但真正区分大科学家的是**科学品味**：在海量可能性里挑出高影响力方向的判断力。这种 taste 无法通过指令微调灌进去，因为它本质上是对"哪些想法值得做"的隐式偏好。过去几乎没人认真建模这件事。

### 方法与核心创新
作者提出 **RLCF (Reinforcement Learning from Community Feedback)**——把科学社区的引用信号当作群体偏好信号。关键拆成两步：
1. **Scientific Judge**：在 70 万对"同领域、同时间段、高引 vs 低引"论文对上训练偏好模型，学会判断想法潜力
2. **Scientific Thinker**：以 Scientific Judge 作为 reward model，用 RL 训练策略模型产出高影响力 idea

与直接用引用数作 label 的粗糙做法相比，**时间和领域配对**避免了马太效应污染（老论文自然引用多），让信号真正反映 idea 质量。

### 关键实验结果
- Scientific Judge 在偏好预测上**超过 GPT-5.2 和 Gemini 3 Pro**
- 泛化性强：在未来年份的测试集、未见领域、同行评审偏好上都成立
- Scientific Thinker 生成的 research idea 被 Judge 评分显著高于所有 baseline

### 局限性与开放问题
- 引用数只是影响力的**代理指标**，短期内被高引的未必真有品味（热点效应）
- 依赖"已发表论文对"训练，对完全开辟新领域的 taste 可能判断失灵
- 没公开报告 reward hacking 的防御——LLM 很容易学会写"看起来高引"的套话

### 启发与应用前景
- **RL from Community Feedback** 是继 RLHF / RLAIF 之后的新范式，任何有群体偏好信号的场景（代码仓库 star 数、产品评论、论文引用）都能套
- 对个人研究者：可以用 Scientific Judge 给自己的 idea 打分作为早期筛选
- 开源权重 + 391 star GitHub 说明社区反响好，follow-up 机会大

---

## 2. Demystifying Video Reasoning
**👍 369** · [arXiv:2603.16870](https://huggingface.co/papers/2603.16870) · [GitHub (22★)](https://github.com/OpenSenseNova/Demystifying_Video_Reasoning)

### 问题与动机
视频扩散模型最近被观察到有 "reasoning 能力"——能解迷宫、做推理任务。主流解释是 **Chain-of-Frames (CoF)**：推理沿着时间帧一帧帧展开。但这个解释有个尴尬：如果推理靠帧间传递，为什么单帧模型也能做推理？论文质疑了这个定论。

### 方法与核心创新
作者提出**全新机制 Chain-of-Steps (CoS)**：推理发生在**扩散去噪步骤轴**上，而非时间帧轴上。具体观察：
1. **早期去噪步骤**探索多个候选解
2. **逐步收敛**到最终答案——类似思考过程
3. 在 DiT 内部还发现**功能分化**：前层做感知、中层做推理、后层做表征整合

同时识别了三个涌现行为：**工作记忆**（持久引用）、**自我修正**、**感知先于行动**。基于这些洞察，他们还给出一个 training-free 方案——用不同随机种子的 latent 轨迹做 ensemble，直接提升推理质量。

### 关键实验结果
- 通过定性分析 + 目标探测实验（targeted probing）验证 CoS 存在
- Training-free ensemble 方案在多个推理任务上直接涨点，无需重训
- 清晰区分了早/中/后层的功能分化（有 probing 数据支撑）

### 局限性与开放问题
- 没给出 CoF 完全失效的严格证明，只是给出 CoS 作为**更主要**的解释——两种机制可能共存
- Ensemble 需要多次前向，成本线性增长
- DiT 功能分化是否对所有视频扩散架构通用？未验证 UNet 等

### 启发与应用前景
- 重新定义了"视频模型推理"的研究路径——以后做视频推理不用再死磕 CoF
- **Ensemble latent trajectories** 这个 trick 极简但有效，几乎所有扩散任务都可以试
- DiT 层级功能分化的发现，对**模型压缩**（不同层用不同精度）、**层级并行训练**有直接价值

---

## 3. CARLA-Air
**👍 340** · [arXiv:2603.28032](https://huggingface.co/papers/2603.28032) · [GitHub (603★)](https://github.com/louiszengCN/CarlaAir)

### 问题与动机
低空经济和空地协同是热门方向，但仿真器生态**严重割裂**：CARLA 等驾驶仿真器缺空中动力学，AirSim 等多旋翼仿真器缺真实地面场景。桥接方案（co-simulation）有同步开销，时空一致性无法保证。对要训空地协同 policy 的研究者来说，这是卡脖子问题。

### 方法与核心创新
作者把 CARLA 和 AirSim 的能力**融合到同一个 Unreal Engine 进程里**，不是桥接而是原生整合：
- 共享物理 tick 和渲染管线，保证严格时空一致
- 同时保留 **CARLA 和 AirSim 原生 Python API + ROS 2 接口**，已有代码零修改可用
- 单 tick 同步捕获 **18 种传感器模态**（多平台统一输出）
- 可插拔 asset pipeline 支持自定义机器人

值得一提的是：AirSim 官方上游已归档（停更），CARLA-Air 相当于接管了 AirSim 的空中能力持续演进。

### 关键实验结果
- 支持四大 workload：空地协同、具身导航 + VLA、多模态感知/数据集构建、RL 策略训练
- 18 传感器模态同步采集（论文没给具体 FPS 数字，但强调保证一致性）
- **GitHub 603 star** 印证社区刚需

### 局限性与开放问题
- 基于 Unreal Engine，单机性能要求高；大规模多 agent 场景的 scalability 未公开测试
- 接管 AirSim 意味着后续维护负担巨大（嵌入式/机载侧代码积累多年）
- 物理真实度没有与专业工具（如 Gazebo + PX4 SITL）做严谨对比

### 启发与应用前景
- **空地协同数据集**长期稀缺，这个平台直接降低数据生成门槛
- 对做具身智能 / VLA 的团队：CARLA + AirSim 原生 API 意味着现有代码迁移几乎零成本
- 预构建二进制 + 全开源，可以立刻用于项目搭建

---

## 4. InCoder-32B
**👍 308** · [arXiv:2603.16790](https://huggingface.co/papers/2603.16790) · [GitHub (100★)](https://github.com/CSJianYang/Industrial-Coder)

### 问题与动机
通用 code LLM 在 LeetCode 和一般工程任务上已经很强了，但遇到**工业场景**性能断崖下跌：芯片设计（Verilog/SystemVerilog）、GPU kernel 优化（CUDA/Triton）、嵌入式（紧内存约束）、编译器优化、3D 建模——这些场景需要理解**硬件语义 + 专用语言构造 + 严格资源约束**，通用代码预训练语料几乎不覆盖。

### 方法与核心创新
从头训 **32B 参数模型**，四阶段训练 pipeline：
1. **通用代码预训练** — 打基础
2. **工业代码退火 (annealing)** — 精选工业语料微调
3. **中训练阶段** — 从 8K 上下文渐进扩展到 128K，同时喂合成的工业推理数据
4. **后训练** — execution-grounded verification（代码实际跑起来验证正确性，而非仅仅语法对）

关键决策：不是通用模型继续训，而是**专门 from scratch** 训工业向——在语料配比阶段就侧重工业场景。

### 关键实验结果
- 在 **14 个通用代码 benchmark + 9 个工业 benchmark（覆盖 4 个专业领域）** 上评估
- 通用任务上「highly competitive」（没给具体数字 — 常见的隐藏话术，实际可能略逊于专用通用模型）
- 工业领域建立了**开源 baseline**——这是关键，之前工业代码开源模型几乎空白

### 局限性与开放问题
- 没公布与 GPT-5 / Claude 等闭源模型的直接对比
- 论文摘要里"highly competitive"这种措辞通常意味着**没能全面超过** SOTA 通用模型
- 128K 上下文对工业代码够不够？芯片设计项目动辄数百万行

### 启发与应用前景
- 对芯片 / GPU / 嵌入式公司：立刻可用的开源底座，省掉 from scratch 成本
- **execution-grounded verification 作为 RLHF 替代** 是值得关注的思路——有明确 ground truth 的领域都能套
- 工业 LLM 的 "from scratch 比 continue pretrain 好"——挑战了"通用底座 + 领域微调"的主流路线

---

## 5. SocialOmni
**👍 248** · [arXiv:2603.16859](https://huggingface.co/papers/2603.16859) · [GitHub (47★)](https://github.com/MAC-AutoML/SocialOmni)

### 问题与动机
全模态 LLM (OLM) 现在能同时处理语音 / 视觉 / 文本，但评测基准还停留在**静态问答**的老思路上——测"能不能看懂这张图"、"能不能听懂这句话"。现实人机交互的核心能力——**什么时候插话、怎么插话、能不能分辨谁在说话**——完全没有 benchmark。这个 gap 导致 OLM 研发方向被带偏。

### 方法与核心创新
作者定义 **社交交互性**三维度：
1. **说话人分离与识别**（who is speaking）
2. **打断时机控制**（when to interject）
3. **打断话术生成**（how to phrase）

构建 SocialOmni benchmark：**2000 个感知样本** + **209 个严格时空约束的交互生成样本**，还设计了**音视频不一致场景**测鲁棒性（如嘴型和声音对不上）。这是首个把"**感知能力** vs **交互生成能力**"分开打分的 benchmark。

### 关键实验结果
- 评测了 **12 个 leading OLM**
- 关键发现：**感知准确率 ≠ 交互能力**——模型能听懂「谁在说什么」，但不知道什么时候该插嘴、怎么插
- 揭示"感知 - 交互"脱节，说明当前 OLM 在对话能力上的本质缺陷

### 局限性与开放问题
- 209 个生成样本规模偏小，统计显著性有限
- "合适的打断"本身主观，评估依赖人工 + LLM judge，引入评估噪声
- 中英文混合 / 多人场景覆盖度未知

### 启发与应用前景
- 对做**语音助手、客服机器人、AI 主播**的公司：直接可用的诊断工具
- "感知-行动脱节"是个**通用信号**——VLA、具身智能、游戏 agent 都可能有类似病灶
- 数据集 + 代码全开源（HuggingFace 上），门槛极低

---

## 6. GOLF
**👍 210** · [arXiv:2603.04597](https://huggingface.co/papers/2603.04597) · [GitHub (16★)](https://github.com/LuckyyySTA/GOLF)

### 问题与动机
现行 RL 把环境反馈压缩成**一个标量 reward**，但 LLM 与环境交互时实际收到的是**丰富的自然语言反馈**——"这里逻辑错了"、"换个思路"、"你漏掉了某个 case"。这些信息被暴力压扁为 scalar 后，探索效率低下，稀疏奖励区域尤其吃亏。

### 方法与核心创新
提出 **GOLF**：显式利用**组级自然语言反馈**引导定向探索。核心机制：
1. **外部批评**：指出具体错误或修正建议
2. **组内尝试**：从同组其他候选的尝试中抽取另一种思路、不同失败模式

这两类反馈被聚合成**高质量修正 (refinements)**，作为 **off-policy scaffold** 注入训练——等于在稀疏奖励区手动给模型"指路"。同时在统一 RL 环内共同优化生成能力和修正能力，形成正反馈循环。

### 关键实验结果
- 在可验证和不可验证任务上都超基线
- **样本效率提升 2.2×**（对比仅用 scalar reward 的 RL）
- 在 sparse-reward benchmark 上优势更明显

### 局限性与开放问题
- "组级"要求同时跑多个候选 — 训练成本显著上升
- NL feedback 质量依赖外部 critic（大概率也是 LLM），存在二阶误差累积
- 可验证任务上效果明显，不可验证任务上的提升幅度未详述

### 启发与应用前景
- NL feedback 作为 off-policy scaffold 是个**通用技巧**——RLHF / RLAIF 流水线都能套
- 对做 **agent 训练** 的人直接可用：环境很自然就会吐 NL 信号（报错、工具返回等）
- 和 RLCF (#1)、OpenClaw-RL (#18) 互相呼应——"scalar reward 太穷"是这波论文的共识

---

## 7. HACRL
**👍 194** · [arXiv:2603.02604](https://huggingface.co/papers/2603.02604) · [Project](https://zzx-peter.github.io/hacrl/)

### 问题与动机
on-policy RL 每个 agent 各自采样各自训，完全浪费了**异构 agent 之间的多样性**。LLM-based MARL 要求协调部署，on/off-policy 蒸馏又只能单向（teacher→student）。能不能让异构 agent **训练时互相学、推理时独立跑**？

### 方法与核心创新
提出 **HACRL** 范式：异构 agent 共享"已验证 rollout"互相提升，推理时独立执行。关键算法 **HACPO**：
- **Verified rollout 共享**：只共享通过验证的优质轨迹
- **四个针对性机制**解决能力差异和策略分布漂移，**理论上保证 unbiased advantage estimation**
- 双向互学（区别于单向蒸馏）——弱 agent 学强的，强 agent 也能从弱的异构行为中获益

本质是把"异构性"从负担转为**训练信号来源**。

### 关键实验结果
- 多种异构模型组合 × 多个推理 benchmark 评估
- 比 **GSPO 平均高 3.3%**，**rollout 成本只有一半**
- 所有参与 agent 都有提升（不是零和博弈）

### 局限性与开放问题
- 需要"已验证"机制——可验证域（数学、编程）好做，开放域验证成本高
- 异构组合策略如何选？理论保证不等于工程调参好做
- 没和更多基线（如 mixing、合成 KD）做对比

### 启发与应用前景
- 大模型训练范式级创新：以后训 agent 不必"一个模型一套数据"
- 对开源社区：可以把多个不同开源模型组队训，互补短板
- **rollout 成本减半** 是硬收益 — 大模型训练中这个省钱效应非常可观

---

## 8. Helios
**👍 186** · [arXiv:2603.04379](https://huggingface.co/papers/2603.04379) · [GitHub (1716★)](https://github.com/PKU-YuanGroup/Helios)

### 问题与动机
视频生成模型长期被三个魔咒困住：
1. 长视频容易 **drifting**（漂移、退化）
2. 实时生成需要 KV-cache / 稀疏注意力 / 量化等花招
3. 训练要各种 sharding 框架，复杂度高

而且这三个问题通常互相制约，改了一个伤另一个。

### 方法与核心创新
**Helios：14B 视频生成模型**，单张 H100 **19.5 FPS**、分钟级生成、质量媲美强基线。三个维度全部突破：
1. **抗漂移**：不用 self-forcing / error-banks / keyframe sampling 等启发式，而是训练时**主动模拟漂移**，并消除重复运动的根源
2. **实时**：不用 KV-cache / 稀疏 attention / 量化，通过**重度压缩历史与噪声上下文 + 减少采样步数**，计算成本**接近甚至低于 1.3B 模型**
3. **训练**：不用并行/sharding 框架，支持图像扩散级别的 batch size，**单机 80GB GPU 能装下 4 个 14B 模型**

统一表示天然支持 T2V / I2V / V2V。

### 关键实验结果
- 14B 参数 + **19.5 FPS @ H100**（实时阈值 24FPS 差一点但非常接近）
- 分钟级生成不漂移
- 短视频和长视频生成都超越主流方法
- GitHub **1716 star** — 社区刚需验证

### 局限性与开放问题
- 19.5 FPS 没达到完全实时（≥24 FPS）
- "分钟级"未说明具体多长，与 Sora、Veo 等闭源模型缺直接对比
- "图像扩散级 batch size" 具体多大没披露

### 启发与应用前景
- 开源 14B 实时长视频模型 — **立刻改变视频生成模型的 compute 预算曲线**
- 对做视频 AIGC 产品的公司：单机部署成本骤降
- "主动训练漂移" 替代启发式 trick 的思路可以泛化到其他 AR 生成任务

---

## 9. MiroThinker-1.7 & H1
**👍 185** · [arXiv:2603.15726](https://huggingface.co/papers/2603.15726) · [Blog](https://www.miromind.ai/blog/mirothinker-1.7-h1-towards-heavy-duty-research-agents-via-verification)

### 问题与动机
当前研究型 agent（Deep Research 一类）问题是**长链条可靠性差**：单步有错、全局答案就崩；中间步骤没核验，最终答案缺乏证据链。要做「重型研究 agent」，必须引入显式的验证机制。

### 方法与核心创新
两代模型递进：
- **MiroThinker-1.7**：通过 **agentic mid-training** 阶段强化结构化规划、上下文推理、工具交互，**每步**更可靠
- **MiroThinker-H1**：在推理过程中直接嵌入**验证**——
  - **局部验证**：中间决策可以被评估并 refine
  - **全局验证**：整条推理轨迹被审计，确保最终答案有证据链支持

与纯 CoT 相比，**验证是一等公民**而非后处理。

### 关键实验结果
- 在 open-web research、科学推理、金融分析 benchmark 上 **SOTA**
- 专业域任务保持强表现
- 同时开源 MiroThinker-1.7 和 1.7-mini，提供高效率版本

### 局限性与开放问题
- "验证" 本身由谁做没交代清楚 — 如果是 LLM 自评，存在 echo chamber 风险
- 全局审计成本高，推理 latency 可能激增
- 没给效率-精度 tradeoff 曲线

### 启发与应用前景
- "local + global verification" 是**深度研究 agent 标配**的方向
- 对搭 research assistant 的团队：直接可抄的架构蓝本
- 与 OpenSeeker (#20)、MetaClaw (#26) 一起，构成 2026-Q1 search agent 开源新一波

---

## 10. Utonia
**👍 185** · [arXiv:2603.03283](https://huggingface.co/papers/2603.03283) · [GitHub (616★)](https://github.com/Pointcept/Utonia)

### 问题与动机
点云的"基础模型"迟迟没到位。根本原因：各域（遥感、室外 LiDAR、室内 RGB-D、CAD、RGB 视频升维）的**传感几何、密度、先验完全不同**，难以共享一个编码器。做 AR/VR、机器人、自动驾驶都在各训各的。

### 方法与核心创新
**Utonia**：首个跨域**自监督点云 Transformer encoder**。关键不是新架构，而是**联合训练所有域**——尽管域差异巨大，模型能学到**一致的表征空间**并跨域迁移。

更惊喜的是**涌现行为**：只有联合训练才会出现的能力（论文强调"intriguing emergent behaviors"），说明不同域之间有互补先验。

### 关键实验结果
- 跨域感知能力全面提升
- **赋能具身任务**：把 Utonia 特征喂给 VLA policy → 机器人操作提升
- **赋能多模态推理**：集成到 VLM → 空间推理涨点
- GitHub **616 star**

### 局限性与开放问题
- 具体涌现行为是什么没详说（论文藏了）
- 不同域在 joint training 里的"权重配方"可能敏感
- 不是 scale law（没给数据量 vs 性能曲线）

### 启发与应用前景
- 点云终于有了自己的 "CLIP 时刻"——后续 3D 任务都可以 encode 一下试试
- 对 AR/VR、自动驾驶公司：**现成通用 3D backbone**
- 呼应 LongCat-Next (#24)、Beyond LM (#37)：**"all-in-one modality encoder" 是 2026 的大趋势**

---

## 11. Attention Residuals (AttnRes)
**👍 182** · [arXiv:2603.15031](https://huggingface.co/papers/2603.15031) · [GitHub (3176★)](https://github.com/MoonshotAI/Attention-Residuals)

### 问题与动机
现代 LLM 清一色 **PreNorm + residual** — 每层输出等权累加到残差流。这个"固定权重 1"看似简单，实则问题大：
- **hidden state 随深度无控制增长**，每层贡献被稀释
- 深层网络中，前层信号几乎淹没在累加里

这是个被大家忽略的基础缺陷。

### 方法与核心创新
**Attention Residuals (AttnRes)**：把"固定累加"换成 **softmax attention over previous layer outputs**——每层可以**有条件、输入相关地**选择性聚合前层表征。

工程优化：
- 全量 AttnRes 内存和通信开销大
- **Block AttnRes**：把层分块，块级注意
- **缓存式 pipeline 通信 + 两阶段计算策略**
- 成为"drop-in replacement"，几乎零额外成本

与 Kimi Linear 架构结合（**48B 总参 / 3B 激活**），**1.4T token 预训练**。

### 关键实验结果
- **Scaling law 一致**：不同模型尺寸都获益（重要 — 证明不是小模型偏方）
- PreNorm 稀释缓解：各层输出幅度、梯度分布更均匀
- 所有下游任务都提升
- GitHub **3176 star** — 受关注度极高

### 局限性与开放问题
- Block AttnRes 的"块大小"是新超参，调优复杂
- 只在 Kimi Linear 验证，Dense 架构 + 标准 Transformer 上的效果？
- 训练加速比未披露

### 启发与应用前景
- 残差连接 15 年没大改动，**AttnRes 是可能的新范式**
- 对所有在训 LLM 的团队：潜在性价比极高的底层升级点
- Moonshot 开源 + 3K star — 表明头部实验室押注这条路

---

## 12. HyDRA (Hybrid Memory)
**👍 156** · [arXiv:2603.25716](https://huggingface.co/papers/2603.25716) · [GitHub (234★)](https://github.com/H-EmbodVis/HyDRA)

### 问题与动机
视频世界模型把环境当作**静态画布**——物体一旦移出视野，再出现时就崩（冻结、扭曲、消失）。现实世界里遮挡-再出现是常态（人走进房间、物体滑出桌面又回来），静态记忆机制根本不够用。

### 方法与核心创新
提出 **Hybrid Memory** 范式：模型同时扮演两个角色——
- **静态背景**的精确档案管理员
- **动态主体**的机警追踪者

即使主体不在视野内，也要保持其运动的连续性。

**HyDRA 架构**：
- 把记忆压缩成 token
- **时空相关性驱动的检索机制**：选择性 attend 相关运动线索
- 保持隐藏主体的身份和运动

**数据集 HM-World**：**59K 高保真 clip**，**17 场景 × 49 主体**，解耦相机和主体轨迹，专门设计进/出场事件来测 hybrid 一致性。

### 关键实验结果
- 在 HM-World 上显著超越 SOTA
- 动态主体一致性 + 整体生成质量双赢
- GitHub 234 star，数据集和代码开源

### 局限性与开放问题
- 59K clip 只有 17 场景 — 多样性不够
- "相关性驱动检索" 的计算复杂度未详述
- 极长时间（分钟级）遮挡的保持度未测

### 启发与应用前景
- **视频世界模型从静态→动态**的标志性工作
- 对 AR/VR、游戏引擎、机器人仿真：直接可用于提升遮挡鲁棒性
- 和 Helios (#8)、ShotStream (#13)、Seoul World Model (#14) 一起构成 2026-Q1 视频世界模型的全面升级

---

## 13. ShotStream
**👍 155** · [arXiv:2603.25746](https://huggingface.co/papers/2603.25746) · [GitHub (126★)](https://github.com/KlingAIResearch/ShotStream)

### 问题与动机
多镜头长叙事视频生成对**交互性**要求高（用户随时改剧情），但**双向架构**注定高延迟、低交互。换成**因果 (causal) 架构**又面临两个老问题：镜头间一致性差、**误差累积**。

### 方法与核心创新
**ShotStream**：因果多镜头架构 + 流式提示。把任务**重构为"next-shot generation"**，以历史为条件，用户可动态插入 streaming prompt 改剧情。

关键两步：
1. 先把 T2V 模型微调为**双向 next-shot generator**
2. 用 **Distribution Matching Distillation (DMD)** 蒸馏成因果学生

应对两个老问题：
- **双缓存机制**：全局 context cache（镜头间一致性） + 局部 cache（镜头内一致性）；**RoPE 不连续指示符**显式区分两个缓存
- **两阶段蒸馏**：先 intra-shot self-forcing（用真实历史），再 inter-shot self-forcing（用自生成历史），弥合 train-test gap

### 关键实验结果
- **16 FPS @ 单 GPU，亚秒级延迟**
- 质量**媲美或超越**更慢的双向模型
- 支持实时交互式叙事

### 局限性与开放问题
- 16 FPS 在视频生成里不错，但离"真正实时"（≥24）仍差
- DMD + 双缓存 + 两阶段蒸馏 — 训练 pipeline 很重
- 流式交互性的实际用户体验（延迟感、修改响应）未做用户研究

### 启发与应用前景
- **短视频 / 剧本互动**平台：ShotStream 是可产品化模板
- Kling AI 出品（快手阵营），开源 + 126 star，商业背景强
- 因果 + distill + 双缓存这套 trick 可以拆解用到别的 AR 视频任务

---

## 14. Seoul World Model (SWM)
**👍 153** · [arXiv:2603.15583](https://huggingface.co/papers/2603.15583) · [GitHub (533★)](https://github.com/naver-ai/seoul-world-model)

### 问题与动机
现有 world model 生成的是**虚构环境**——视觉合理但不真实。如果能让 world model 渲染**真实存在的城市**，应用场景（自动驾驶、城市规划、AR）立刻展开。但这面临几个坎：时间错位（参考图 vs 动态目标）、轨迹多样性差、车载采集数据稀疏。

### 方法与核心创新
**Seoul World Model (SWM)**：锚定真实首尔的世界模型。
- **自回归视频生成 + retrieval-augmented 条件**：用附近街景图作为条件
- **Cross-temporal pairing**：解决参考图和动态场景的时间错位
- **大规模合成数据集**：让相机轨迹多样化
- **View interpolation pipeline**：从稀疏街景合成连贯训练视频
- **Virtual Lookahead Sink**：持续用未来位置的检索图重新锚定，稳定长视距生成

### 关键实验结果
- 在**首尔 / 釜山 / Ann Arbor** 三城市上评估
- 超越现有 world model 在**空间保真、时间一致、长视距**三维度
- 支持**上百米的轨迹**，相机运动多样，文本 prompt 可调场景

### 局限性与开放问题
- 仅三个城市，泛化到任意城市需要更多数据
- retrieval-augmented 依赖街景数据库 — 没有 Google Street View 级别数据的地区难复制
- 计算开销（retrieval + generation）未披露

### 启发与应用前景
- 自动驾驶仿真：可以**在真实城市数据上训练**而非虚构地图
- AR/VR 导航：真实城市的生成式渲染
- Naver AI 出品，533 star — 工业级背书
- "RAG for world models" 是个全新角度，扩散 + 检索的结合可能成为范式

---

## 15. Qianfan-OCR
**👍 153** · [arXiv:2603.13398](https://huggingface.co/papers/2603.13398) · [GitHub (383★)](https://github.com/baidubce/Qianfan-VL)

### 问题与动机
传统 OCR 流水线（版面分析 → OCR → 理解）管道长、误差累积。端到端 VLM 模型虽然简洁，但**丢失了显式的版面结构**——复杂排版（表格、多栏、公式）容易崩。百度想要在**端到端 + 复杂版面**上都做好。

### 方法与核心创新
**Qianfan-OCR：4B 参数 E2E VLM**，统一文档解析、版面分析、文档理解。关键创新 **Layout-as-Thought (LaT)**：
- 由 `<think>` token 触发的**可选思考阶段**
- 先生成结构化版面表征（bbox、元素类型、阅读顺序）
- 再产出最终输出
- 相当于把 CoT 搬到版面理解上，**把丢失的版面 grounding 救回来**

支持多种 prompt 驱动任务：表格抽取、图表理解、DocQA、关键信息抽取。

### 关键实验结果
- **OmniDocBench v1.5：93.12，端到端第一**
- **OlmOCR Bench：79.8**
- 关键信息抽取多个公开 benchmark 上**平均分第一**，**超过 Gemini-3.1-Pro、Seed-2.0、Qwen3-VL-235B**
- 通用能力（OCRBench、CCOCR、DocVQA、ChartQA）与同规模 VLM 竞争

### 局限性与开放问题
- 4B 参数领跑 235B — 可能意味着在**更大模型上的差距会缩小**
- `<think>` 是可选的 — 触发条件的一致性如何保证？
- 部署在百度千帆平台（闭源云服务），开源程度未说

### 启发与应用前景
- "可选 thinking phase" 是个非常**精打细算**的设计：需要时才多花 token
- 对做 OCR / 文档理解的公司：端到端 + layout-grounded 是可抄的架构
- 小模型在垂直任务上**超越大模型** — 这是 2026 的一个持续信号

---

## 16. dLLM
**👍 153** · [arXiv:2602.22661](https://huggingface.co/papers/2602.22661) · [GitHub (2404★)](https://github.com/ZHZisZZ/dllm)

### 问题与动机
扩散语言模型 (DLM) 最近发展很快，LLaDA、Dream 都已开源。问题是核心组件**散落在各 ad-hoc 代码库**里、没统一实现——想复现、扩展、魔改都很难。没有类似 HuggingFace Transformers 对 AR LLM 那种统一框架。

### 方法与核心创新
**dLLM**：DLM 的**统一开源框架**，涵盖训练、推理、评估全链路。特点：
- 标准化 pipeline，复现/微调/部署/评估 LLaDA、Dream 等模型
- 提供**可复现 recipe**：从 BERT-style encoder 或 AR LM 转换成 DLM
- 低算力门槛**从头构建小 DLM**
- 开源小型 DLM 权重

本质是做了 DLM 界的"标准仓库"。

### 关键实验结果
- 这是工程类论文，不追 SOTA
- 社区反响极强：**GitHub 2404 star**（发表仅数周）
- 多个小型 DLM checkpoint 可直接下载

### 局限性与开放问题
- 不涉及新架构或新训练法
- 工具好不好用要看社区长期反馈
- "从 BERT/AR LM 转 DLM" 的性能损耗未详细量化

### 启发与应用前景
- 扩散 LLM 研究的**加速器**——以后这个方向的论文都会依赖它
- 对想入场 DLM 的研究者：**从零训练门槛骤降**
- 呼应 DLM 2025-2026 的崛起，是不可忽视的基础设施

---

## 17. HSImul3R
**👍 152** · [arXiv:2603.15612](https://huggingface.co/papers/2603.15612) · [GitHub (41★)](https://github.com/yukangcao/HSImul3R)

### 问题与动机
从普通拍摄（稀疏视角 / 单目视频）重建 **人物-场景交互 (HSI) 3D**，现有方法有个致命 gap：**感知-仿真鸿沟**——视觉上看起来对，但一扔进物理引擎就崩（穿模、脱离地面、接触不稳）。具身 AI 根本没法直接用这些重建。

### 方法与核心创新
**HSImul3R：物理在环的双向优化**。把**物理仿真器当作主动监督者**，前向后向联合优化：
- **前向（人 → 场景）**：**Scene-targeted RL** 优化人体运动，双重监督（运动保真 + 接触稳定性）
- **反向（场景 → 人）**：**Direct Simulation Reward Optimization** 用仿真反馈（重力稳定性、交互成功率）refine 场景几何

同步发布新 benchmark **HSIBench**，多样物体和交互场景。

### 关键实验结果
- **首批可直接部署到真实人形机器人的 HSI 重建**
- 在 HSIBench 上显著超越感知 only 方法
- 真实人形机器人部署演示（关键 — 证明不是纸面指标）

### 局限性与开放问题
- 依赖物理仿真器 — 仿真器本身的 fidelity 决定上限
- 双向优化训练慢
- 对刚体效果好，变形物体（布、液体）未处理

### 启发与应用前景
- **具身 AI 数据管线**的关键补全：拍摄 → 可用仿真数据
- 对机器人 / 游戏公司：用户拍视频 → 直接生成 training asset
- "物理仿真器作为 RL 监督者" 的思路可泛化到 HSI 之外

---

## 18. OpenClaw-RL
**👍 152** · [arXiv:2603.10165](https://huggingface.co/papers/2603.10165) · [GitHub (5075★)](https://github.com/Gen-Verse/OpenClaw-RL)

### 问题与动机
每个 agent 动作后都有一个 **next-state signal**（用户回复、工具输出、GUI 状态变化），这些信号**极其丰富**——既能评估动作好坏，又能指导"应该怎么改"。但现有 agentic RL 系统没一个把它作为在线学习源来用。浪费。

### 方法与核心创新
**OpenClaw-RL**：把 next-state signal 当作**统一的训练源**。个人对话、终端执行、GUI 交互、SWE 任务、工具调用——**都不是独立问题，都是同一个 policy 的训练数据**。

两类信号：
- **评估型 (evaluative)**：由 PRM judge 转化为标量 reward
- **指导型 (directive)**：通过 **Hindsight-Guided On-Policy Distillation (OPD)** 回收——从 next-state 抽文本提示，构建强化版 teacher context，做 token 级方向性优势监督

**异步架构**：模型服务 / PRM 评审 / trainer 更新**同时进行**，零协调开销。

### 关键实验结果
- 个人 agent：**被用就能自动改进**（从用户 re-query、修正、显式反馈中学）
- 通用 agent：终端、GUI、SWE、工具调用全域支持
- GitHub **5075 star** — 2026 年最热门 agent 开源项目之一

### 局限性与开放问题
- PRM judge 质量决定一切 — 自评估的可靠性仍是黑洞
- token 级 directional advantage 比 scalar reward 复杂，调优难
- 离线到在线的切换稳定性未详述

### 启发与应用前景
- "agent 在生产环境边服务边学" 终于落地
- 对做产品级 agent 的公司：**零下行时间优化**是刚需
- PRM + on-policy distillation 的组合可能成为 agent RL 标配

---

## 19. OmniLottie
**👍 151** · [arXiv:2603.02138](https://huggingface.co/papers/2603.02138) · [GitHub (646★)](https://github.com/OpenVGLab/OmniLottie)

### 问题与动机
矢量动画在 web、广告、UI 里无处不在（Lottie 格式是事实标准），但生成式 AI 集中在像素视频。直接让 LLM 生成 Lottie JSON？JSON 包含大量**不变的结构元数据和格式 token**，模型学不到真正的动画表达。

### 方法与核心创新
**OmniLottie**：多模态指令生成高质量矢量动画。核心 trick：**Lottie tokenizer**——把 JSON 转成结构化的 (命令, 参数) 序列，剥离无关格式，只保留 shape / animation function / control parameter。

在 tokenizer 基础上建在**预训练 VLM** 上，支持**多模态交错指令**（文字 + 参考图）。

配套发布 **MMLottie-2M**：2M 专业设计的矢量动画 + 文本 + 视觉标注。

### 关键实验结果
- 生成**生动且语义对齐**的矢量动画
- 紧贴多模态指令
- GitHub **646 star**

### 局限性与开放问题
- 矢量动画评估本质主观，缺客观指标
- MMLottie-2M 质量和多样性未深度分析
- 与传统 T2V 像素视频的应用边界模糊

### 启发与应用前景
- UI / web 设计生产力工具的**直接杀手级应用**
- 矢量 tokenizer 的思路可延伸到 SVG、CAD 等结构化表征
- 小领域 × 大数据 × 预训练 VLM = 仍能出新工作

---

## 20. OpenSeeker
**👍 149** · [arXiv:2603.15594](https://huggingface.co/papers/2603.15594) · [GitHub (615★)](https://github.com/rui-ye/OpenSeeker)

### 问题与动机
Deep Search（深度搜索 agent）是 frontier 能力，但**高质量训练数据被工业巨头垄断**，开源社区一直追不上。想破局必须**把训练数据也完全开源**。

### 方法与核心创新
**OpenSeeker：首个模型 + 数据完全开源的深度搜索 agent**。两大技术：
1. **Fact-grounded 可控 QA 合成**：通过**拓扑扩展 + 实体混淆**反向工程 web graph，生成复杂多跳推理任务，覆盖度和复杂度可控
2. **去噪轨迹合成**：**回溯式摘要机制**给 teacher LLM 吐高质量动作

关键设计：**只用 SFT**（不做 RL），证明数据质量本身就能打。

### 关键实验结果
- **只用 11.7K 合成样本**（一次训练）
- BrowseComp: **29.5% vs DeepDive 15.3%**（近乎翻倍）
- BrowseComp-ZH: **48.4% vs Tongyi DeepResearch 46.7%**（后者用了大量 continual pretraining + SFT + RL）
- 多 benchmark SOTA

### 局限性与开放问题
- 11.7K 样本数量很小，是否真覆盖搜索全景？
- "反向工程 web graph" 对小众长尾主题效果？
- SFT-only 的天花板未测（配 RL 可能更高）

### 启发与应用前景
- "数据合成 > 算力堆料" 的又一次佐证
- 对所有做 agent 训练的开源团队：**可直接拿 11.7K 数据起飞**
- 数据集 + 权重全开源 — 降低进入门槛极其明显

---

## 21. EnterpriseOps-Gym
**👍 148** · [arXiv:2603.13594](https://huggingface.co/papers/2603.13594) · [Project](https://enterpriseops-gym.github.io/)

### 问题与动机
LLM agent 要在**企业环境**部署，但现有 benchmark 远不够真实——企业场景需要**长时程规划 + 持久状态变化 + 严格访问协议**。没这个 benchmark，无法评估 agent 真实能力。

### 方法与核心创新
**EnterpriseOps-Gym**：ServiceNow 出品的企业级 agent 评测。
- **容器化沙箱**：**164 个数据库表 + 512 个功能工具**，模拟真实的搜索摩擦
- **1150 个专家策划任务**
- **8 个关键垂直**：客服、HR、IT 等
- 可扩展

### 关键实验结果
评测 14 个 frontier 模型：
- **Claude Opus 4.5 最高：37.4%**（还是非常低！）
- 给 oracle 人类 plan 后提升 **14-35 个百分点** — 证明**策略推理是主要瓶颈**
- 拒绝不可行任务能力：**最好的模型 53.9%** — 很多任务被硬上，产生副作用

### 局限性与开放问题
- 沙箱再真实也不是真企业
- 1150 任务由专家标注 — 主观性和 coverage bias
- 没覆盖协作、审批流等长流程

### 启发与应用前景
- **企业 agent 产品化**的诊断工具，ServiceNow 背书 — 工业权威
- Top 模型 37.4% — 说明**企业级 agent 至少还要 1-2 年**才能大规模可用
- 策略推理是瓶颈 → 下一波 agent 研究方向应聚焦规划而非工具使用

---

## 22. ReBalance
**👍 148** · [arXiv:2603.12372](https://huggingface.co/papers/2603.12372) · [GitHub (119★)](https://github.com/yu-lin-li/ReBalance)

### 问题与动机
大推理模型 (LRM) 两病并发：
- **过度思考 (overthinking)**：简单题上浪费 token
- **思考不足 (underthinking)**：能力足够却不做充分探索

现有方法压制反思关键词 / 限制推理长度，反而引发 underthinking — 此消彼长。

### 方法与核心创新
**ReBalance**：training-free，通过**置信度**这个**连续信号**实现平衡思考。
- **高置信方差 → 过度思考**（模型在反复纠结）
- **持续过高置信 → 思考不足**（模型没认真考虑）

在小规模数据集上**聚合 hidden state 成"推理模式原型"**，计算 **steering vector** 引导推理轨迹。动态控制函数根据实时置信度调节强度方向 — 过度思考时剪冗余、思考不足时促进探索。

### 关键实验结果
- **4 个模型（0.5B ~ 32B）× 9 个 benchmark**（数学推理、通用 QA、代码）
- 同时降低冗余、提升准确率
- Training-free + plug-and-play

### 局限性与开放问题
- "推理模式原型"的构建方式对 dataset 敏感
- 0.5B 到 32B 的涵盖有限，更大模型效果未知
- "置信度方差" 的阈值需调

### 启发与应用前景
- **LRM 部署时的必备优化**：立即加在推理时省 token
- "置信度作为连续信号" 可以泛化到 agent 动作选择、VLA 规划
- Training-free — 零成本部署，99% 的团队都能直接用

---

## 23. RL3DEdit
**👍 145** · [arXiv:2603.03143](https://huggingface.co/papers/2603.03143) · [GitHub (195★)](https://github.com/AMAP-ML/RL3DEdit)

### 问题与动机
用 2D 扩散先验做 3D 编辑很有前景，但**多视图一致性**是老大难。监督微调最有效，但**3D 一致编辑数据极度稀缺** — 训不了。

### 方法与核心创新
**关键观察**：**生成 3D 一致内容很难，但验证 3D 一致很容易**。这天然适合 RL。

**RL3DEdit**：用 VGGT（3D foundation model）作为 reward source：
- 编辑图像喂给 VGGT
- **置信度图 + 姿态估计误差**作为 reward
- 把 2D 编辑先验**锚定到 3D 一致流形**

单 pass 框架，不需要配对编辑数据。

### 关键实验结果
- **多视图一致性稳定**
- 编辑质量超越 SOTA
- 效率高

### 局限性与开放问题
- VGGT 的先验上限就是 RL3DEdit 的上限
- 单 pass 对复杂编辑够用吗？
- 没给效率的具体数字

### 启发与应用前景
- "生成难、验证易 → RL 合适" 是个**通用启发** — 3D、蛋白质、代码都能套
- 对 3D 内容创作：视频编辑 / VR 场景编辑直接受益
- VGGT 作为 reward source 是对 foundation model 新用法

---

## 24. LongCat-Next
**👍 144** · [arXiv:2603.27538](https://huggingface.co/papers/2603.27538) · [GitHub (403★)](https://github.com/meituan-longcat/LongCat-Next)

### 问题与动机
多模态模型普遍**语言中心**：非语言模态（视觉、音频）当外挂处理，架构碎片化、整合次优。要不要走"**所有模态都 tokenize 成离散 token，共享自回归目标**"这条路？

### 方法与核心创新
**DiNA (Discrete Native Autoregressive)** 框架 + **LongCat-Next** 模型。关键：
- **dNaViT (Discrete Native any-resolution ViT)**：任意分辨率 tokenize/de-tokenize，连续视觉 → 层次离散 token
- 文本、视觉、音频统一在**单一自回归目标**下，极少模态特定设计
- 工业级基础模型

### 关键实验结果
- "看、画、说"一个框架搞定
- **突破了离散视觉建模在理解任务上的长期瓶颈**
- 统一了**理解-生成**的冲突（通常只能顾一头）
- GitHub 403 star，美团出品

### 局限性与开放问题
- 离散 tokenize 的信息损失？
- 多分辨率 token 的效率曲线未披露
- 对视频、3D 扩展？

### 启发与应用前景
- "All modalities as discrete tokens" 是近年重要范式之争 — LongCat-Next 是新一轮投票
- 对做多模态的公司：可参考的开源架构
- 和 Utonia (#10)、Beyond LM (#37) 一起构成 2026-Q1 的"native multimodal"浪潮

---

## 25. TAPS
**👍 142** · [arXiv:2603.27027](https://huggingface.co/papers/2603.27027) · [GitHub (5★)](https://github.com/Moe-Zbeeb/TAPS)

### 问题与动机
speculative decoding 加速自回归生成 — draft 模型提议 token，大模型并行验证。但大家训 draft 都用**通用语料** (ShareGPT 等)，到底 draft **训练分布**对效果影响多大？没人认真研究。

### 方法与核心创新
**系统化 study + TAPS 方法**：
- 用 HASS 和 EAGLE-2 两类 drafter，在 MathInstruct、ShareGPT、混合数据上训
- 在 MT-Bench、GSM8K、MATH-500、SVAMP 上评估 acceptance length
- 发现 draft 能力**强烈依赖分布匹配**

推理时组合专用 drafter：
- **checkpoint averaging 差**
- **confidence-based routing 好**
- **merged-tree verification 最强**（最高 acceptance length）
- **confidence 是更好的路由信号**（比 entropy）

### 关键实验结果
- **task-specific draft 显著特化**：MathInstruct 训的数学 benchmark 最强，ShareGPT 训的 MT-Bench 最强
- 混合数据增强鲁棒性，但堆规模不通用
- confidence-routed merged-tree verification 总体最优

### 局限性与开放问题
- GitHub 只 5 star — 评估代码/可复现性疑虑
- 局限在两个 drafter 架构
- 路由的计算开销未详细分析

### 启发与应用前景
- **speculative decoding** 的从业者必读
- "推理时组合专用 drafter" 的思路可以做进 serving framework
- Confidence > entropy 作为路由信号的发现是细节宝藏

---

## 26. MetaClaw
**👍 139** · [arXiv:2603.17187](https://huggingface.co/papers/2603.17187) · [GitHub (3452★)](https://github.com/aiming-lab/MetaClaw)

### 问题与动机
部署的 agent **静态**，不能跟着用户需求演进。现有方案：
- 存原始轨迹不蒸馏知识
- 静态技能库
- 需要停机重训

都不理想。

### 方法与核心创新
**MetaClaw**：持续元学习框架，同时进化 LLM policy 和**可复用行为技能库**。两个互补机制：
1. **技能驱动快速适应**：LLM evolver 分析失败轨迹 → 合成新技能 → 零停机即时提升
2. **机会主义策略优化**：用户非活跃窗口（**OMLS 调度器**监控系统空闲 + 日历）触发云 LoRA 微调 + RL-PRM

两者**相互强化**：refined policy 生成更好轨迹给技能合成；richer skills 提供更高质量数据给 policy 优化。防污染：版本化 support/query 分离。proxy 架构扩展到生产规模 LLM，无本地 GPU 需求。

### 关键实验结果
- 技能适应提升**最高 32% 相对精度**
- 全 pipeline 把 Kimi-K2.5 从 **21.4% → 40.6%**
- 复合鲁棒性提升 **18.3%**
- GitHub **3452 star**

### 局限性与开放问题
- 技能库膨胀后的 retrieval 效率？
- LoRA 更新频率和一致性
- OMLS 对 24/7 高负载场景的适用性

### 启发与应用前景
- 生产级 agent **运行时持续进化**的参考模板
- "非活跃时段训练" 是非常 pragmatic 的工程思路
- 和 OpenClaw-RL (#18)、MiroThinker (#9) 呼应：agent 进化是 2026 的大主题

---

## 27. ADE-CoT
**👍 138** · [arXiv:2603.00141](https://huggingface.co/papers/2603.00141)

### 问题与动机
Image-CoT (test-time scaling for image gen) 主要针对 T2I，但图像**编辑**是目标导向的——解空间被源图和指令约束。直接套 Image-CoT 有三个问题：固定采样预算低效、早期验证不可靠（依赖通用 MLLM 分数）、大规模采样出冗余结果。

### 方法与核心创新
**ADE-CoT (Adaptive Edit-CoT)**：按需 test-time scaling。三大策略：
1. **难度感知资源分配**：基于估计的编辑难度动态分配预算
2. **编辑特定的早期剪枝验证**：用**区域定位 + 字幕一致性**选候选
3. **深度优先机会停止**：指令一致的结果出现就立刻停

针对的是编辑而非生成的特点。

### 关键实验结果
- 在 **Step1X-Edit、BAGEL、FLUX.1 Kontext** 三个 SOTA 上都涨
- 相同采样预算下 **> 2× 加速 over Best-of-N**
- 同时性能更好（不是只求快）

### 局限性与开放问题
- "编辑难度估计"的准确性是关键 — 论文细节未展开
- 仅在三个模型上验证
- 没 GitHub 链接，复现性存疑

### 启发与应用前景
- 图像编辑类产品的**推理加速法宝**
- "Best-of-N 的智能版"可以泛化到其他条件生成任务
- 区域定位 + 字幕一致性作为早期 verifier 的设计可以借鉴

---

## 28. MinerU-Diffusion
**👍 135** · [arXiv:2603.22458](https://huggingface.co/papers/2603.22458) · [GitHub (566★)](https://github.com/opendatalab/MinerU-Diffusion)

### 问题与动机
文档 OCR 已经从"逐行转写"进化到"结构化解析"——版面 + 表格 + 公式。但主流 VLM 都用**自回归解码**，存在：
- 顺序延迟（一 token 一 token 出）
- 长文档上误差累积严重

作者质疑：**左到右因果生成是序列化遗留**，不是 OCR 本质。

### 方法与核心创新
**MinerU-Diffusion**：把文档 OCR 重新定义为 **inverse rendering 问题**，用**扩散去噪并行解码**代替自回归。
- **块级扩散 decoder**
- **不确定性驱动的 curriculum learning**
- 视觉条件下的并行去噪

配套 benchmark **Semantic Shuffle** — 打乱语义顺序的图像，验证是否依赖语言先验。

### 关键实验结果
- **> 3.2× 解码加速** vs 自回归基线
- 鲁棒性全面提升
- Semantic Shuffle 上**降低语言先验依赖**、视觉 OCR 能力更强

### 局限性与开放问题
- 扩散解码的训练成本通常高于 AR
- 块大小、迭代步数等超参需调
- 与 Qianfan-OCR (#15) 等 SOTA 未直接 head-to-head

### 启发与应用前景
- "AR → 扩散" 在 OCR 跑通，可能推广到其他结构化输出任务（如代码、结构化抽取）
- OpenDataLab (上海 AI Lab) 出品，开源强背书
- 对批量 OCR 场景（文档数字化、科研文献）：3.2× 加速是真金白银

---

## 29. Intern-S1-Pro
**👍 131** · [arXiv:2603.25040](https://huggingface.co/papers/2603.25040)

### 问题与动机
通用 LLM 越来越强，但**科学专业深度**仍是短板——化学、材料、生命、地球科学都有专门知识。能不能在**一万亿参数规模**打造**科学专用多模态基础模型**？

### 方法与核心创新
**Intern-S1-Pro：首个 1T 参数科学多模态基础模型**。通用 + 科学双强：
- 通用推理 + 图文理解增强
- **100+ 专业科学任务**（化学、材料、生命、地球）
- Agent 能力加持
- **XTuner + LMDeploy** 基础设施：1T 规模 RL 训练，严格训练/推理精度一致

定位为**"可专业化通才" (Specializable Generalist)**。

### 关键实验结果
- 通用能力：开源模型顶级梯队
- 专业科学任务：**超越闭源模型**
- 没给非常具体的 benchmark 数字（论文摘要级描述偏虚）

### 局限性与开放问题
- 1T 参数的部署成本极高，谁能用？
- "超越闭源" 需要更细粒度对比
- 没说开不开源权重

### 启发与应用前景
- 与 OpenAI / Anthropic 的旗舰模型规模相当的**开源科学专用**选项
- 上海 AI Lab 出品 — 国家队背书
- 跟 AI-for-Science 的 #1 呼应：**2026 AI-for-Science 是大主题**

---

## 30. Omni-WorldBench
**👍 126** · [arXiv:2603.22212](https://huggingface.co/papers/2603.22212) · [GitHub (105★)](https://github.com/AMAP-ML/Omni-WorldBench)

### 问题与动机
视频世界模型两大流派：视频生成（只看视觉保真/文本对齐）、3D 重建（用静态 3D 指标，完全忽略时间动态）。**两派 benchmark 都不测"交互响应"**——即"动作如何驱动跨时空的状态转换"，而这才是 4D 世界模型的核心。

### 方法与核心创新
**Omni-WorldBench**：交互中心的 4D 评测。两个组件：
1. **Omni-WorldSuite**：系统化 prompt 集，跨交互等级、场景类型
2. **Omni-Metrics**：**基于 agent 的评估框架** — 量化交互动作对最终结果和中间状态演化轨迹的**因果影响**

### 关键实验结果
- 评测 **18 个代表性世界模型**（跨多种范式）
- 揭示现有模型在**交互响应**上的关键缺陷
- 为未来研究提供可操作洞察

### 局限性与开放问题
- "因果影响"怎么精确测量？agent-based eval 有主观性
- 18 个模型覆盖哪些具体架构未详列
- 交互场景复杂度分层不够清晰

### 启发与应用前景
- **4D 世界模型**的第一个权威评测 — 行业标准潜在候选
- 对研究者：明确未来改进方向（交互响应）
- 和 HyDRA (#12)、Seoul World Model (#14) 呼应：**动态/交互是世界模型下一战场**

---

## 31. daVinci-MagiHuman
**👍 123** · [arXiv:2603.21986](https://huggingface.co/papers/2603.21986) · [GitHub (1918★)](https://github.com/GAIR-NLP/daVinci-MagiHuman)

### 问题与动机
音视频联合生成架构复杂：多流、交叉注意力、模态桥接 … 优化困难。**能不能用最简单的单流 Transformer 解决？**

### 方法与核心创新
**daVinci-MagiHuman**：**单流 Transformer** 统一 token 序列处理文本、视频、音频，**仅用 self-attention**。优势：
- 避免多流/交叉注意力的复杂性
- 标准训练推理基础设施就行
- 人物中心场景特别强：面部表演、语音-表情协调、真实身体运动、精确音视频同步
- **多语种语音**：普通话、粤语、英、日、韩、德、法

高效推理：单流 backbone + 蒸馏 + 潜空间超分 + Turbo VAE — **单 H100 GPU 2 秒生成 5 秒 256p 视频**。

### 关键实验结果
- 自动评估：开源最强视觉质量 + 文本对齐；**最低 WER 14.60%**（语音清晰度）
- 人类评估：**vs Ovi 1.1 胜率 80.0%**，**vs LTX 2.3 胜率 60.9%**（2000 次比对）
- 全栈开源：基础模型、蒸馏模型、超分模型、推理代码

### 局限性与开放问题
- 256p 分辨率较低
- 5 秒时长受限（不是长视频）
- WER 14.60% 仍有提升空间（商用 TTS 通常 < 5%）

### 启发与应用前景
- AI 数字人 / AI 主播的**开源新基线**
- 单流架构：简洁胜过复杂的又一个案例
- 直接可产品化（2 秒生成 5 秒视频是极具吸引力的速度）

---

## 32. T2S-Bench & Structure-of-Thought
**👍 121** · [arXiv:2603.03790](https://huggingface.co/papers/2603.03790) · [Project](https://t2s-bench.github.io/T2S-Bench-Page/)

### 问题与动机
人类处理复杂文本时会**圈重点、推关系、构建结构**。但 LLM 只在纯文本里硬拼，没显式结构化过程。能不能让模型显式"文本→结构"？

### 方法与核心创新
两件事：
1. **Structure of Thought (SoT)**：prompting 技巧 — 显式引导模型构建中间文本结构
2. **T2S-Bench**：首个评估文本到结构能力的 benchmark — **1.8K 样本**，**6 个科学领域 × 32 种结构类型**

### 关键实验结果
- **45 个主流模型** 评估，提升空间大：
  - 多跳推理任务平均准确率 **仅 52.1%**
  - 最强模型端到端抽取 **节点精度 58.1%**
- 在 Qwen2.5-7B 上：
  - **SoT 单独 +5.7%**（跨 8 任务）
  - **T2S-Bench 微调后 +8.6%**

### 局限性与开放问题
- 32 种结构是否覆盖所有下游结构？
- 科学领域偏 STEM — 人文、法律、金融未测
- SoT prompting 和 CoT 的详细对比未展开

### 启发与应用前景
- **结构化推理**的补全 — CoT → SoT
- 对做**信息抽取 / 知识图谱 / RAG** 的产品：即时可用
- 提示了 LLM 的一个系统性短板：**隐式处理结构的能力不足**

---

## 33. Penguin-VL
**👍 119** · [arXiv:2603.06569](https://huggingface.co/papers/2603.06569) · [GitHub (177★)](https://github.com/tencent-ailab/Penguin-VL)

### 问题与动机
紧凑 VLM（2B、8B 规模）的发展被 vision encoder 的**对比预训练** (CLIP / SigLIP) 锁死。但对比学习是为**判别**优化的 — 类别级不变性**抑制细粒度视觉线索**，这恰恰是 dense captioning 和复杂 VLM 推理需要的。目标不匹配。

### 方法与核心创新
**关键挑战**：VLM 的 vision encoder **不必**源自对比预训练。**Penguin-VL**：**vision encoder 从纯文本 LLM 初始化**！

这是个反常识选择：文本 LLM 从没见过图像，为什么它能初始化 vision encoder？答：LLM 学到的结构化推理能力比 CLIP 的"类别级不变性"更适合细粒度 VLM 任务。

### 关键实验结果
- 小模型（2B、8B）
- 和 Qwen3-VL 等 leading VLM 在数学推理上**相当**
- 在**文档理解 / 视觉知识 / 多视角视频理解**上**超越**
- 消融：Penguin-Encoder 在所有任务上稳定打败对比编码器

### 局限性与开放问题
- 为什么 LLM 初始化的视觉编码器更好？机理解释不够
- 大模型（30B+）上效果未测
- 视频任务仅限短片段？

### 启发与应用前景
- 挑战 CLIP 作为**默认视觉 backbone** 的主流地位
- 为资源受限端侧（手机、机器人）提供强 VLM 基础
- 如果这套路线成立，整个 VLM 预训练流水线可能被重构

---

## 34. PixelSmile
**👍 117** · [arXiv:2603.25728](https://huggingface.co/papers/2603.25728) · [GitHub (358★)](https://github.com/Ammmob/PixelSmile)

### 问题与动机
细粒度面部表情编辑困难 — 表情之间**语义重叠**（笑和惊讶都可能张嘴），直接编辑容易混淆、身份漂移。现有方法难做到**精准 + 保 ID + 可控强度**。

### 方法与核心创新
1. **FFE 数据集**：连续情感标注（不是离散分类）
2. **FFE-Bench**：评估结构混淆、编辑准确、线性可控性、表情-身份 tradeoff
3. **PixelSmile** 框架：扩散模型 + **完全对称联合训练** disentangle 表情语义
4. **强度监督 + 对比学习**：表情更强、更易区分
5. 通过**文本潜空间插值**实现**精确稳定的线性表情控制**

### 关键实验结果
- Disentanglement 出色，身份保持鲁棒
- 支持连续、可控、细粒度表情编辑
- 自然支持表情平滑混合

### 局限性与开放问题
- 数据规模未披露
- 和 head pose、光照的解耦度未测
- 对遮挡面部 / 极端角度未报告

### 启发与应用前景
- AI 换脸、数字人、动画 character 驱动的直接升级
- "连续情感标注"替代离散分类，思路可延伸到其他情感建模
- 358 star — 社区已验证好用

---

## 35. Astrolabe
**👍 109** · [arXiv:2603.17051](https://huggingface.co/papers/2603.17051) · [GitHub (127★)](https://github.com/franklinz233/Astrolabe)

### 问题与动机
蒸馏后的自回归 (AR) 视频模型支持流式生成，但**与人类偏好对齐差**。RL 对齐要么需要**重新蒸馏**（贵）要么**求解器耦合的反向过程优化**（内存和计算开销大）。

### 方法与核心创新
**Astrolabe**：针对蒸馏 AR 模型的**高效在线 RL**。关键突破 **Forward-process RL 公式** — 基于 negative-aware fine-tuning：
- 直接在**推理端点**对比正负样本
- 建立隐式 policy improvement 方向
- **无需反向过程展开**

对齐到长视频：
- **流式训练**：rolling KV-cache 渐进生成序列
- **RL 更新仅限局部 clip 窗口**，条件化在历史上下文以保长程一致性

抗 reward hacking：多 reward 目标 + 不确定性感知选择性正则 + 动态参考更新。

### 关键实验结果
- 多个蒸馏 AR 视频模型上一致提升生成质量
- 鲁棒可扩展的对齐方案

### 局限性与开放问题
- 没给加速比、内存节省的具体数字
- 与直接 re-distillation 的质量天花板对比
- 多 reward 权重调优复杂

### 启发与应用前景
- 视频对齐技术的**工程化**关键一步
- "forward-process RL" 的思路可泛化到其他蒸馏模型对齐问题
- 和 Helios (#8)、ShotStream (#13)、daVinci (#31) 一起：**视频 AR 流生成是 2026 的热战区**

---

## 36. HopChain
**👍 109** · [arXiv:2603.17024](https://huggingface.co/papers/2603.17024)

### 问题与动机
VLM 在**细粒度视觉-语言推理**上仍弱。长 CoT 推理暴露多种失败模式（感知、推理、知识、幻觉错误）层层累积。问题：现有 RLVR 训练数据**不涉及复杂推理链**，弱点未被暴露，更别说修复。

### 方法与核心创新
**HopChain**：可扩展框架合成**多跳视觉-语言推理数据**（专为 RLVR 训练）。每个合成查询：
- **逻辑依赖的实例锚定 hop 链**
- 早期 hop 建立后期 hop 所需实例/集合/条件
- 最终答案是**可验证的确切数字**

把合成数据加入 **Qwen3.5-35B-A3B、Qwen3.5-397B-A17B** 的 RLVR 原始数据中。

### 关键实验结果
- **24 benchmark**（STEM、puzzle、VQA、OCR、视频）
- **20/24 benchmark 提升**（两个模型都如此）— 广泛泛化
- 半多跳 / 单跳变体分别降低 **5.3 / 7.0 点** — 证明完整链条的价值
- 超长 CoT 场景**增益峰值 > 50 个准确率点**

### 局限性与开放问题
- 多跳合成的**难度分布**需要仔细控制
- 不同 backbone 的普适性未充分测
- 没给合成数据规模

### 启发与应用前景
- RLVR 数据合成的**新 pipeline** — 可直接抄进多模态训练
- 对做 VLM 推理的团队：50 点增益是大招
- 呼应 SoT (#32)：**显式推理链 / 结构化推理**是 2026 的共性

---

## 37. Beyond Language Modeling
**👍 103** · [arXiv:2603.03276](https://huggingface.co/papers/2603.03276) · [Project](https://beyond-llms.github.io/)

### 问题与动机
**原生多模态模型**的设计空间仍不清晰——视觉如何和语言协同预训练？用什么视觉表征？模态间规模律怎么走？没系统化研究。

### 方法与核心创新
受控 from-scratch 预训练实验——隔离多模态预训练的**独立因素**（不被语言预训练干扰）。**Transfusion 框架**：语言用 next-token prediction，视觉用扩散。数据：text、video、image-text、action-conditioned video。

四个核心洞察：
1. **Representation Autoencoder (RAE)** 是最优统一视觉表征（理解+生成双强）
2. **视觉和语言互补**，下游任务有协同
3. 统一多模态预训练**自然通向世界建模**
4. **MoE 实现高效多模态 scaling**，天然诱导模态特化

**IsoFLOP 分析** + 规模律：**视觉比语言更数据饥渴**（scaling asymmetry）。MoE 恰好能同时提供**高容量**（语言需要）和**数据容纳**（视觉需要）。

### 关键实验结果
- 严谨受控的预训练规模律分析
- **发现模态间规模不对称** — 视觉数据需求远高
- MoE 协调模态规模差异 — 关键架构建议

### 局限性与开放问题
- "from-scratch" 实验规模相对学术界可行，但与工业级预训练差几个数量级
- RAE 具体形式未充分公开
- Action-conditioned video 部分权重较低

### 启发与应用前景
- "MoE for multimodal" 有了**理论基础** — 不再是工程直觉
- 对所有做 native multimodal 的团队：**视觉更饥渴**的发现改变数据配比
- 和 Utonia (#10)、LongCat-Next (#24) 呼应：**2026 多模态预训练科学化**

---

# 🗺️ 趋势洞察

## 1. 视频世界模型：从静态渲染迈向动态交互 & 实时流式
涉及论文：**#2, #8, #12, #13, #14, #30, #31, #35**

2026-Q1 视频模型的主线：**实时 + 交互 + 动态一致**。
- 推理机制被重解：Chain-of-Steps (#2) 颠覆 CoF，指出推理沿**去噪步**展开
- 实时：Helios (#8) 14B 模型单 H100 19.5 FPS；ShotStream (#13) 亚秒级延迟 16 FPS；daVinci (#31) 2 秒生成 5 秒视频
- 动态一致：HyDRA (#12) 处理遮挡 re-emerge；Astrolabe (#35) 为流式 AR 模型做 RL 对齐
- 物理/真实世界 grounding：Seoul World Model (#14) 锚定真实城市；Omni-WorldBench (#30) 首个 4D 交互评测

**核心观点**：视频生成的评价维度从"单帧美观"扩展到"交互响应 / 长程一致 / 流式延迟"——**4D 视频世界模型**成为新赛道。

## 2. Agent 范式从"执行工具"转向"学习、适应、生成品味"
涉及论文：**#1, #6, #7, #9, #18, #20, #21, #26**

过去 agent 研究集中于"更强工具、更长链条"。2026-Q1 转向**训练与适应机制**：
- 新型 reward 源：RLCF 用社区引用 (#1)、GOLF 用 NL feedback (#6)、OpenClaw-RL 用 next-state signal (#18)
- 异构协同：HACRL (#7) 让异构 agent 互相学 rollout
- 验证驱动：MiroThinker (#9) 局部 + 全局验证
- 持续进化：MetaClaw (#26) 在非活跃时间自动蒸馏新技能
- 开源追赶：OpenSeeker (#20) 证明 11.7K 精合成数据能打败巨头大 SFT+RL
- 现实压力测试：EnterpriseOps-Gym (#21) 揭示最强 Claude Opus 4.5 也只有 37.4%

**核心观点**：Agent 的"天花板"不在工具更多，而在**能否从环境信号连续学习**。"scalar reward 太穷" 已成共识。

## 3. 多模态预训练走向"原生 & 统一 & 科学化"
涉及论文：**#10, #15, #16, #24, #28, #33, #37**

- 原生离散化：LongCat-Next (#24) 所有模态 tokenize 到共享离散空间
- 通用 encoder：Utonia (#10) 一个编码器跨所有点云域
- 规模律科学化：Beyond LM (#37) 揭示视觉比语言更数据饥渴、MoE 天然解耦
- 反直觉：Penguin-VL (#33) vision encoder 从纯文本 LLM 初始化**超过** CLIP
- 扩散取代 AR：MinerU-Diffusion (#28) 把 OCR 重新定义为 inverse rendering；dLLM (#16) 统一扩散语言模型基础设施
- 端到端 + 版面思考：Qianfan-OCR (#15) Layout-as-Thought 小模型打大模型

**核心观点**：多模态底座的设计思路从"加模态"升级为"重架构"——**MoE + 离散统一 + 扩散解码** 是 2026 的主流范式。

## 4. 推理 & 训练效率的精细化控制
涉及论文：**#11, #22, #25, #27, #36**

- 残差机制革新：AttnRes (#11) 用 soft attention 替代固定累加残差
- 推理平衡：ReBalance (#22) 用置信度做连续信号控制过度/不足思考
- 推测解码特化：TAPS (#25) draft 模型的训练分布与下游工作负载匹配是关键
- Test-time scaling 自适应：ADE-CoT (#27) 按编辑难度动态分配预算
- 数据合成驱动：HopChain (#36) 多跳合成数据 RLVR 训练，超长 CoT 增益 +50 点

**核心观点**：效率优化不再是"减参数 / 加速卡"，而是**在推理流中做条件/难度/分布感知的细粒度调度**。

## 对比与张力

- **视频生成：因果 (交互) vs 双向 (质量)** — ShotStream (#13) 通过 DMD 蒸馏尝试兼得；Helios (#8) 则在双向基础上做重压缩
- **agent reward：scalar vs 语义** — GOLF (#6)、RLCF (#1)、OpenClaw-RL (#18) 都在摆脱 scalar 的束缚，但评估的客观性难保证
- **VLM encoder：对比预训练 vs LLM 初始化** — Penguin-VL (#33) 挑战 CLIP，但大模型效果未验证
- **多模态：添加模态 vs 原生离散统一** — LongCat-Next (#24)、Beyond LM (#37) 推 native path；Utonia (#10) 单模态走通用 encoder 路线
- **推理：冗长推理 vs 精简推理** — ReBalance (#22) 用置信度动态平衡，而不是二选一

## 值得关注的研究方向

1. **交互式 4D 世界模型** — Omni-WorldBench (#30) 揭示了短板，Seoul World Model (#14)、HyDRA (#12) 提供起点，是未来 1-2 年的主战场
2. **从自然语言/社区信号学习 reward** — RLCF (#1)、GOLF (#6)、OpenClaw-RL (#18) 代表一个新范式：**社区偏好、next-state signal 是 scalar reward 的"超集"**，适用范围极广
3. **原生多模态的规模律 & MoE 架构** — Beyond LM (#37) 开了理论头；LongCat-Next (#24) 给出工业实例；"视觉数据比语言饥渴"是个可泛化的设计原则
4. **端到端 + 结构化思考 (Layout-as-Thought, Structure-of-Thought)** — Qianfan-OCR (#15) 和 T2S-Bench (#32) 示范如何把结构化推理做进任务中，**CoT → SoT** 是一个清晰的路径
5. **小模型垂直超越大模型** — Qianfan-OCR (#15) 4B 打 235B、Penguin-VL (#33) 小打大、OpenSeeker (#20) 11.7K 数据超越巨头，**精调 × 结构化数据 > 算力堆料**这个信号越来越强

---

*本报告由 [paper-digest skill](~/.claude/skills/paper-digest/) 自动生成 · API 方案（非 CDP）· 生成耗时 < 1 分钟（数据抓取）*
