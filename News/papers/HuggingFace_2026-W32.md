# HuggingFace 周榜论文深度总结 — 2026-W32

> 来源：https://huggingface.co/papers/week/2026-W32
> 统计日期：2026-08-11
> 筛选条件：upvotes ≥ 30（周榜共 105 篇，入选 53 篇）
> 论文数：53

## 目录

1. [Recursive Synthesis for Long-Horizon Terminal Tasks](#1-recursive-synthesis-for-long-horizon-terminal-tasks) 👍229
2. [LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks](#2-longhorizon-harness-advancing-long-horizon-agents-for-real-world-tasks) 👍165
3. [SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks](#3-swantale-unified-multi-speaker-speech-and-audio-generation-for-instruct-and-zero-shot-tasks) 👍155
4. [Deferred Exposure of Future Trajectories for Verifiable Reasoning in Autonomous Driving VLMs](#4-deferred-exposure-of-future-trajectories-for-verifiable-reasoning-in-autonomous-driving-vlms) 👍141
5. [DAPD: Dual-Anchored Policy Distillation](#5-dapd-dual-anchored-policy-distillation) 👍108
6. [From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement](#6-from-rlvr-to-rlsvr-task-transformation-induces-self-verifiable-rewards-for-open-ended-llm-self-improvement) 👍105
7. [Mental World Modeling](#7-mental-world-modeling) 👍103
8. [MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations](#8-merchantbench-benchmarking-llm-agents-for-long-term-coherence-in-e-commerce-operations) 👍96
9. [JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion](#9-joyai-video-edit-real-time-open-ended-video-editing-with-autoregressive-diffusion) 👍90
10. [AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning](#10-agentopsd-recursive-self-distillation-for-agentic-reinforcement-learning) 👍89
11. [Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing](#11-hunyuan3d-buffalo-10-a-unified-multimodal-model-for-scalable-3d-generation-understanding-and-editing) 👍88
12. [AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling](#12-aurora-lm-autoencoding-unified-representation-for-continuous-latent-diffusion-language-modeling) 👍79
13. [N_0-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens](#13-n-0-vtla-scaling-vision-tactile-language-action-model-with-latent-tactile-tokens) 👍76
14. [Interpretable MEG Decoding of Perceived Speech: Cortical Sources and the Stimulus Features That Drive Retrieval](#14-interpretable-meg-decoding-of-perceived-speech-cortical-sources-and-the-stimulus-features-that-drive-retrieval) 👍68
15. [OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models](#15-osreward-instituting-standardized-evaluation-for-cross-platform-computer-use-reward-models) 👍68
16. [InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis](#16-infinisplat-implicit-gaussian-decoding-for-large-baseline-monocular-view-synthesis) 👍65
17. [ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment](#17-abseeker-training-long-horizon-search-agents-via-answer-backtracked-credit-assignment) 👍64
18. [WorldClaw: Agentic 3D Open-World Generation at Scale](#18-worldclaw-agentic-3d-open-world-generation-at-scale) 👍61
19. [Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes](#19-towards-physics-of-multimodal-pretraining-knowledge-flow-modality-synergy-early-unification-and-recipes) 👍58
20. [Progressive Agent Skill Generation via Reinforcement Learning](#20-progressive-agent-skill-generation-via-reinforcement-learning) 👍58
21. [ToolArtist: Tool-Using Unified Multimodal Models for Agentic Image Generation](#21-toolartist-tool-using-unified-multimodal-models-for-agentic-image-generation) 👍56
22. [Meshy T2: Fast Native Mesh Generation with Flow Matching](#22-meshy-t2-fast-native-mesh-generation-with-flow-matching) 👍56
23. [Weak-to-Strong On-Policy Distillation](#23-weak-to-strong-on-policy-distillation) 👍56
24. [Video-DeepResearch: Towards the Next-Generation Multimodal Deepresearch Agent](#24-video-deepresearch-towards-the-next-generation-multimodal-deepresearch-agent) 👍50
25. [UEmbed: Unified Sparse and Dense Multimodal Embeddings](#25-uembed-unified-sparse-and-dense-multimodal-embeddings) 👍50
26. [Knowledge-Geometry Decoupling: Refreshable Pretrained Transfer for Streaming Recommendation](#26-knowledge-geometry-decoupling-refreshable-pretrained-transfer-for-streaming-recommendation) 👍47
27. [N_0-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation](#27-n-0-twam-scaling-tactile-native-world-action-model-for-contact-rich-manipulation) 👍47
28. [VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation](#28-vad-attributing-visual-evidence-for-target-reconstruction-in-multimodal-on-policy-distillation) 👍46
29. [GST-Bench: Can VLMs Develop Global Spatial Awareness from Video?](#29-gst-bench-can-vlms-develop-global-spatial-awareness-from-video) 👍43
30. [EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning](#30-envace-internalizing-environment-dynamics-via-world-rehearsal-for-agentic-reinforcement-learning) 👍39
31. [The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads](#31-the-personalization-mirage-how-llms-fabricate-user-profiles-and-why-self-monitoring-misleads) 👍39
32. [Learning from Failures: Retrieval-Centric CoT via Hard Negatives for Unified Multimodal Retrieval](#32-learning-from-failures-retrieval-centric-cot-via-hard-negatives-for-unified-multimodal-retrieval) 👍38
33. [ChronoVision: Temporal Reasoning via Latent State Reconstruction](#33-chronovision-temporal-reasoning-via-latent-state-reconstruction) 👍38
34. [PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning](#34-pcsd-persistent-consistency-for-self-distillation-in-agentic-reinforcement-learning) 👍38
35. [CADENA: Stepwise CAD Reverse Engineering](#35-cadena-stepwise-cad-reverse-engineering) 👍38
36. [Scaling Properties of Text Conditioning in Visual Generation](#36-scaling-properties-of-text-conditioning-in-visual-generation) 👍38
37. [AISPA: User-Centric System Prompt Auditing for Large Language Model Applications](#37-aispa-user-centric-system-prompt-auditing-for-large-language-model-applications) 👍37
38. [Quo Vadis, World Modeling?](#38-quo-vadis-world-modeling) 👍36
39. [DiffusionGemma Technical Report](#39-diffusiongemma-technical-report) 👍36
40. [HelloWorld: Enabling Socially Interactive Characters in Video World Models](#40-helloworld-enabling-socially-interactive-characters-in-video-world-models) 👍35
41. [OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents](#41-onedayagent-towards-a-long-horizon-harness-for-autonomous-agents) 👍34
42. [WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity](#42-worldexam-benchmarking-world-models-from-apparent-appearance-to-inherent-reactivity) 👍34
43. [SAF-OPD: Stable Advantage Fusion for On-Policy Distillation](#43-saf-opd-stable-advantage-fusion-for-on-policy-distillation) 👍34
44. [HarnessOpt-Bench: Evaluating LLMs at Harness Optimization](#44-harnessopt-bench-evaluating-llms-at-harness-optimization) 👍33
45. [From Economic Agents to Agentic Economies: A Systems Blueprint for Economic World Models](#45-from-economic-agents-to-agentic-economies-a-systems-blueprint-for-economic-world-models) 👍33
46. [PAST-Bench: Benchmarking the Foundations of Recursive Self-Improvement in Personal Agents](#46-past-bench-benchmarking-the-foundations-of-recursive-self-improvement-in-personal-agents) 👍33
47. [Fewer Clarifications, Better Code: Benchmarking Cross-Session Personalized Ambiguity Adaptation in Coding Assistants](#47-fewer-clarifications-better-code-benchmarking-cross-session-personalized-ambiguity-adaptation-in-coding-assistants) 👍32
48. [LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models](#48-llada-moe-v2-scaling-mixture-of-experts-diffusion-language-models) 👍31
49. [SKT: Skill-Use Training at Scale via Verified Synthetic Data Generation](#49-skt-skill-use-training-at-scale-via-verified-synthetic-data-generation) 👍31
50. [On-Policy Delta Distillation for Multilingual Math Reasoning](#50-on-policy-delta-distillation-for-multilingual-math-reasoning) 👍30
51. [Teaching Nemotron Greek: Mining a Corpus, Adapting Retrieval, and Grounding Generation for Modern Greek across Specialist Domains](#51-teaching-nemotron-greek-mining-a-corpus-adapting-retrieval-and-grounding-generation-for-modern-greek-across-specialist-domains) 👍30
52. [DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces](#52-dataspace-benchmarking-data-agents-for-verifiable-analytics-over-heterogeneous-workspaces) 👍30
53. [QQWorld: Quantile-Quantile Matching for World Model Regularization](#53-qqworld-quantile-quantile-matching-for-world-model-regularization) 👍30

---
## 1. Recursive Synthesis for Long-Horizon Terminal Tasks
**👍 229** · 🏛 腾讯 / 佐治亚大学 / 马里兰大学 / 宾夕法尼亚大学 · [arXiv:2608.05466](https://arxiv.org/abs/2608.05466) · [项目页](https://zhongzhi660.github.io/recursive-verified-synthesis-site/)

### 问题与动机
终端 Agent（在真实 shell 里装依赖、跑测试、修配置的智能体）的训练数据贵得离谱——一条高质量长程任务人工造价数百到数千美元。原因不是写指令难，而是一条任务里 **instruction / Docker 环境 / 参考解 / 私有 verifier 四者必须互相一致**：verifier 检查的每一项都得能从公开指令推出来，参考解在干净沙箱里必须真能跑通。人写不动，直接让 LLM 生成又会把这四者的依赖关系写崩（典型症状是 verifier 检查了指令里根本没提的要求，Agent 无论如何都做不对）。现有终端数据集（TermiGen 3.5K、Endless Terminals 3.3K、TMax-15K）要么规模上不去，要么难度是一次性设定的，无法随模型变强而继续加码。

### 方法与核心创新
RST（Recursive Synthetic Terminal Tasks）的核心机制是**递归改写 + 沙箱验证 + 接受即复用为下一轮种子**：从 639 条已验证的引导任务出发，每轮对种子任务做四件事——(1) 选一个可行的改写算子（论文定义了 5 大族 40 个算子，如 `unit_test_failure_repair`、`container_build_alignment`、`environment_variable_resolution`）并定义预期结果；(2) 延长参考解，把新增工作量写进 `solve.sh`；(3) **同步重写 verifier 和公开指令**使三者重新对齐；(4) 在全新沙箱里验证。

接受标准是两条硬约束：**oracle validity**（参考解必须在干净沙箱里通过私有 verifier）和 **contract validity**（verifier 检查的每一项都必须在公开指令里写明或能从 workspace 推断）。第二条是防止「隐藏要求」的关键——它把「任务是否公平」变成了可自动检查的属性。通过的任务同时进入下一轮种子池和 RL 任务池，成功 rollout 则作为 SFT 轨迹。

与 SWE-Gym / TMax 这类一次性构造的数据集相比，关键区别在于 **reseeding**：难度不是设计出来的，是递归长出来的。

### 关键实验结果
15 轮递归产出 **37,484 条可执行任务，单条约 $0.05**（每千条约 $50）。难度增长在结构和 Agent 两端都被验证：中位参考解从 67 行涨到 374 行（5.6×），执行命令数 40 → 244（6.1×），用到的 CLI 工具 17 → 71，verifier 断言 17 → 57，而**指令长度只从 85 词涨到 122 词（1.4×）**——说明增长的是执行工作量而非文字啰嗦。

固定 solver 下 DeepSeek-V4-Pro 的 pass@4 从 R1 的 90% 单调掉到 R15 的 **2.5%（36 倍衰减）**，平均 partial credit 从 0.970 掉到 0.170。更说明问题的是「差一点做完」的比例：failed attempt 中满足 ≥75% 检查项的从 86.4% 掉到 1.2%，低于 50% 检查项的任务从 0% 涨到 97.5%——后期任务不是「稍微难一点」，是 Agent 根本推进不动。

训练侧：SFT 三轮后 Qwen3.5-27B 在 Terminal-Bench 2 从 41.20% → 47.94%，Terminal-Bench Hard 22.67% → 28.33%；122B-A10B 从 43.82% → 49.44%。agentic PPO 把 27B 拉到 TB2 49.44 / TB-Hard 32.00 / LHTB 22.07，相对基座提升 20.0% / 41.2% / 21.9%——**TB-Hard 上的 32.00 已接近 DeepSeek-V4-Pro 的 36.00**。污染审计做得干净：与三个 benchmark 的 13-token 精确重叠为 0/89、0/46、0/100，且 unigram JSD 随轮次上升（越练越不像 benchmark）。

### 局限性与开放问题
论文自承的主要问题是**相似度长尾**：轮内最近邻相似度中位数从 R1 的 0.223 涨到 R15 的 0.464，p95 已到 0.703，作者承认这是「未来需要去重的可管理尾部」——换句话说 R15 已经开始出现近重复簇，再往下递归大概率要先做去重才能继续。

我观察到的三个软肋：其一，**RL 只做到 27B 一个规模、一个 setting**，PPO 的 value head 还是从既有 critic checkpoint 热启的，这条曲线的可复现性存疑；其二，SFT 表格只到「Round 3」，而合成做了 15 轮——**最难的 R7–R15 数据实际没有用于训练**（pass@4 只有 2.5%，rejection sampling 根本采不到成功轨迹），所以「没有天花板」这个结论只对**合成端**成立，对**训练端**并未验证；其三，接受率稳定在 74.5–81.5% 看着漂亮，但它衡量的是「任务能不能自洽」，不是「任务有没有意义」——第 15 轮的任务本质上是同一条 JSON-diff 流水线套了五层配置修复（见论文 Table 5 的 lineage 案例），是否真的对应真实工程分布是个开放问题。

### 启发与应用前景
最值得抄的是 **contract validity 这个检查**——把「verifier 是否公平」变成可自动判定的条件，这套思路能直接搬到 SWE 任务合成、工具调用合成、GUI 任务合成上，比单纯的「LLM 生成 + LLM 判分」硬得多。第二个可迁移点是**用固定 solver 的 pass@4 曲线来标定合成难度**，比人工分级或 token 长度靠谱。

follow-up 的切入角度：(1) 把 R7+ 的高难任务用「过程奖励 / 部分学分」而非 0-1 成功率引入训练，绕开 rejection sampling 采不到样本的瓶颈；(2) 在递归里加入去重压力（把新颖度做进接受判据），看能否把 15 轮推到 30 轮；(3) 复用这套 recursive verified synthesis 到浏览器/GUI 环境，那里 verifier 更难写，contract validity 的价值也更大。数据集已在 HuggingFace 开放。

---

## 2. LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks
**👍 165** · 🏛 阿里巴巴 · [arXiv:2608.01964](https://arxiv.org/abs/2608.01964) · [GitHub](https://github.com/AMAP-ML/LongHorizon-Harness) · [项目页](https://lh-harness.pages.dev)

### 问题与动机
长程 Agent 的失败模式常被误诊为「模型能力不够」。这篇论文的诊断是**harness 的架构问题**：现有 harness（Claude Code、OpenClaw、Codex CLI）把任务执行、任务状态、完成度判断全塞在同一个不断膨胀的上下文里。后果有两个——状态被埋在几十万 token 的交互记录里越来越难追踪；更致命的是**模型对自己的错误自评会直接沉淀进上下文，成为后续决策的「事实」**。Agent 说「我已经装好了」，这句话就成了它后面所有推理的前提，没有任何机制去环境里核对一遍。

### 方法与核心创新
把长程执行重构成**任务状态管理问题**，用 Manage-Execute-Audit（MEA）三角色循环：
- **Manager** 维护显式的、独立于执行上下文的任务状态，并据此签发一个有界的子任务 contract；
- **Executor** 用**全新上下文**执行该 contract（每轮执行完，原始交互轨迹直接丢弃）；
- **Auditor** 用**只读工具**去环境里独立核实执行结果，产出 audit report。

关键设计是「**任务状态只用环境独立验证过的事实更新**」——Executor 的自我报告不能直接进状态，必须过 Auditor 这一关。跨轮持久化的只有任务状态 + audit reports，执行轨迹全扔。这直接切断了错误自评的传播链。另外配了个轻量 AgentAdapter，可以把 Claude Code 等现成 harness 当作 Executor 后端插进来，不改它们的原生 agent loop。

### 关键实验结果
同模型同 Executor 后端的对照（这点很重要，隔离了模型能力变量）：Qwen 3.7-Plus + Claude Code 在 WeaveBench 上 PassRate 51.8%、均分 0.702，换成 LongHorizon-Harness（内部仍用 Claude Code 执行）后升到 **80.7% / 0.835**。Terminal-Bench 2.1 从 69.7% → 77.2%。OSWorld 2.0 二值完成率 2.8% → 8.3%（近 3 倍），partial 21.5% → 35.2%。

换强模型也成立：Claude Opus 4.7 在 OSWorld 2.0 的 34 题子集上从 20.6% → **35.3%**，partial 55.8% → 66.9%。

token 成本的分解很有说服力：Manager 只占总 token 的 2.0–8.1%（显式状态维护几乎不要钱），**Auditor 占 19.4–38.1%（独立验证才是主要开销）**。而总成本并非固定倍数——WeaveBench 上是基线的 2.3×，OSWorld 2.0 是 3.6×，但 **Terminal-Bench 2.1 上反而少花 24% token 且成功率更高**。原因是执行者越强，需要的返工轮次越少。

### 局限性与开放问题
论文没有独立的 Limitations 章节，但细粒度表格里的负向结果很诚实，值得单独拎出来：Terminal-Bench 2.1 上 data-science（-0.125）、mathematics（-0.167）、video-processing（-0.333）、mteb（-0.333）全是**退步**；WeaveBench 的 Desktop 域均分 -0.0205。作者自己的解释是「harness 主要抬高长程轨迹的下尾——把很多濒临失败转成部分/完全成功，但少数本来就强的基线轨迹会被额外的验证和修复步骤搞坏」。这个 trade-off 是真实的：**对短任务和一次成型的任务，MEA 循环是净损耗**。

我观察到的问题：其一，**Auditor 用同一个 backbone 模型**，「独立验证」只是上下文独立而非能力独立——模型看不出来的错误，换个上下文照样看不出来，剩余失败集中在「隐藏性能阈值、具身视觉精度、时序视频证据」这类审计器闭不了环的条件上，正是这个局限的体现；其二，OSWorld 2.0 上 8.3% 的绝对值仍然极低，2.8→8.3 的「三倍」放在 8% 的基数上参考价值有限；其三，25 轮 MEA 上限 + 每轮 Executor 1800 秒预算，**单任务墙钟时间可能是基线的数倍**，论文只报了 token 没报 wall-clock。

### 启发与应用前景
最实用的结论是「**长程能力 = 模型 × harness**，而 harness 这一侧还有很大空间没挖」——用 Qwen 3.7-Plus + 好 harness 打过 Claude Opus 4.7 + Claude Code 的裸配置，这个结果对做产品的团队很有价值：不换模型也能拿到大幅提升。

工程上直接可抄的三件事：(1) **状态外置**，把「任务进展」从对话历史里拎出来变成结构化对象；(2) **执行上下文一次性**，每个子任务开新上下文，避免污染累积；(3) **只读审计器**，用独立角色去环境里核实，而不是信执行者的自述。第三点是 ReAct/Reflexion 系框架普遍缺的一环。

follow-up 角度：用**更小更便宜的模型做 Auditor**（既然它只需要只读核对，不需要规划能力），能把 20–38% 的审计开销压下来；或者研究 Auditor 该在什么条件下触发（自适应审计而非每轮必审），针对上面 data-science / math 类任务的退步做门控。代码已开源。

---

## 3. SwanTale: Unified Multi-Speaker Speech and Audio Generation for Instruct and Zero-Shot Tasks
**👍 155** · 🏛 字节跳动 / 浙江大学 · [arXiv:2608.02023](https://arxiv.org/abs/2608.02023) · [项目页](https://swanaigc.github.io/#swantale)

### 问题与动机
动画配音、有声剧、短剧、广告这类场景对语音生成的要求，和学术 TTS benchmark 的要求是两回事：创作者需要**无参考音频、纯靠自然语言描述就能设计一个音色**（instruct 任务），需要**同一段波形里同时生成多说话人对白 + 环境声场 + 局部音效**，事后还要能通过参考音频复用之前设计好的声音（zero-shot 任务）。现有系统要么只做 zero-shot 克隆，要么只做单说话人 instruct 控制，二者不统一；更根本的问题在数据侧——训练数据只有转录文本和说话人切分，没有描述环境、音色、局部情绪变化的**多层级 caption**，模型压根没见过这种监督信号。

### 方法与核心创新
数据和模型两侧同时下手。

**SwanData-Caption（约 7000 万条 caption 记录）**是四段式流水线：覆盖设计 → 语音预处理 → caption 标注 → 数据精炼。其中三个设计有借鉴价值：
- **定向合成补盲区**：老年语音、中英文超短语句（均长 1.5 秒）、难发音目标（多音字/品牌名/中英混排）各补 10 万条，用带发音提示的 TTS teacher 生成。这三类都是人耳极敏感但真实语料极稀缺的。
- **三字段 caption schema**：Environment（场景声床、混响、背景噪）/ Speakers（每个说话人的稳定属性：性别、年龄段、音色、语速、口音）/ Content（按时间顺序的内容，用 `<S1>` 标说话人、`<Audio>` 标音效，并在标签周围描述瞬时的情绪、音量、停顿、强调变化）。**稳定属性和瞬时属性被明确分开**，这是控制粒度的关键。
- **风格-人设库作为软先验**：给动画、短剧/影视、广告/数字人三类媒体各建一套风格矩阵（触发条件、描述符优先顺序、稳定 vs 瞬时的边界）。作者发现不给这个先验，标注模型产出的说话人风格极度贫乏。

模型侧：SwanVAE（96 维连续 latent、25 Hz、每步 40ms，标称 38.40 kbps）作为统一声学表征 + flow-based Transformer + reward-conditioned 质量控制 + Engram conditioning + Unified MoE（统一处理多任务多模态）+ 课程学习 + GRPO 后训练。

### 关键实验结果
**SwanVAE 重建**：语音上 PESQ 4.1683 / MCD 0.9638 双第一（对比 DAC 的 4.1178 / 1.1963），歌声上 PESQ、STOI、MCD 全部第一（3.9821 / 0.9001 / 1.5661）；通用音频 ViSQOL 第一（4.1269），音乐 ViSQOL 第二（4.2623，输给 EnCodec 的 4.2976）。**同一个 checkpoint 跑四个域，没有做域特化选型**——25 Hz 这么低的帧率还能守住 MCD，是这篇里最扎实的数字。

**zero-shot（SwanBench-Speech）**：独白和双人对话两个设定下，音色一致性、表现力丰富度、表现力层次感均第一。相对自家前代 SwanVoice：独白音色一致性 0.95、内容错误率降到 0.086、SpeechJudge 3.75；对话设定 0.94 / 0.120 / 3.92。

**instruct（InstructTTSEval）**：中文 APS 86.1 第一，英文 APS 84.2 并列第一，中文 DSD 80.1 第二。

### 局限性与开放问题
作者在结论里直接列了三条：(1) **复杂背景音乐生成仍然困难**，尤其是音乐类型需要随情绪切换的场景；(2) **超过两分钟的多说话人 + 音效长程 instruct 生成还做不好**；(3) **精细局部控制不到位**——指定说话人的连续情绪变化、重音和节奏的精确控制、停顿与音效的精确对时，数据标注和模型两侧都是难题。

实验表格里还暴露了两个作者说得比较轻的问题：**内容准确率和音质并非最优**——独白设定下 FishSpeech 的 Content Error 更低、Sound Fidelity 更高，对话设定下 SoulX-Podcast 两项都更好。也就是说 SwanTale 赢在表现力，输在保真度和吐字准确度，这在配音这类容错率低的商业场景是硬伤。另外 **RP（角色扮演）指标在中英文都是弱项**（中文 64.1、英文 63.6，均低于 Qwen3-TTS 和 MOSS），作者归因于风格矩阵对长尾职业/角色原型覆盖不足——这说明「风格-人设库」这个软先验既是优势也是天花板，泛化到库外角色就掉。

消融里另有一个值得注意的点：**换更大的 caption encoder（8B → 32B）带来的提升（3.82 → 3.98 总体表现力）比 MoE 本身（3.56 → 3.82）小不了多少**，说明相当一部分收益其实来自文本侧理解能力，不全是音频模型的功劳。

### 启发与应用前景
对做多模态生成数据的团队，**三字段 caption schema 和「稳定/瞬时属性分离」是可以直接复用的设计**——同样的思路可以搬到视频生成（场景 / 角色外观 / 逐帧动作描述）和音乐生成上。**风格矩阵作为标注软先验**也是个通用技巧：与其指望标注模型自己想出丰富描述，不如给它一张按媒体类型组织的描述符清单。

工程上值得注意的是 SwanVAE 的 25 Hz / 96 维配置——每秒只有 25 个 latent step，对下游 LM 的序列长度友好得多，这是它能在一个模型里同时塞下语音+音效+音乐的前提。

follow-up：论文自己指出的方向是**统一生成与编辑**（改已有音频的内容、说话人、情绪）；我更看好的切入点是补上保真度短板——用 FishSpeech 那类系统的内容准确率约束做 reward，在 GRPO 阶段把 Content Error 一起优化下去。目前未见权重开源，只有 demo 页。

---

## 4. Deferred Exposure of Future Trajectories for Verifiable Reasoning in Autonomous Driving VLMs
**👍 141** · 🏛 北京航空航天大学 / 卓驭科技 / 浙江大学 / 上海交通大学 · [arXiv:2608.01755](https://arxiv.org/abs/2608.01755) · [GitHub](https://github.com/hzx122/DEFT-RLVR)

### 问题与动机
自动驾驶 VLA 模型普遍用 CoT 监督来提升 VLM 部分的推理能力，而 CoT 的标注流水线几乎都会把**记录下来的真实未来轨迹（GT）喂给 teacher 模型**。这篇论文指出这会造成 **trajectory anchoring bias（轨迹锚定偏差）**：teacher 不是从场景证据推出决策，而是**为已知结果编理由**。

作者做了一个设计得相当干净的对照实验来证明这点：100 个强因果场景（急刹、由行至停、急转），同一个 teacher（Qwen3.5-397B-A17B）、同样 12 帧视觉输入、字节级完全相同的 system prompt、temperature 0、关闭 thinking，**唯一变量是一个包含 10 个 GT waypoint 的文本块插不插入**。两名标注员独立按四个维度（grounding / 无幻觉 / 具体性 / 因果连贯）打分，共 1600 个维度级评分。

### 方法与核心创新
去掉 GT 能消除捷径，但开放式轨迹生成又会把「高层决策」和「精确几何合成 + 底层动力学」纠缠在一起——论文的 Figure 7 显示直接做轨迹 token 生成，SFT 拟合度上升但 ADE 始终远高于码本重建下限 0.279 m，且**通用视觉能力显著退化，加 CoT 也救不回来**。

于是提出 **AD-MCQ**：把规划变成从少量显式候选轨迹中选一条。用 K-means 在 489,042 条真实未来轨迹上建 K=8192 的原型码本，把 GT 量化到最近原型作为唯一正确选项，再按规则构造干扰项（Dev/Test 用结构化干扰：2 个尺度匹配 + 1 个匀速外推 + 2 个 hard negative）。这样规划变成**精确可验证**的选择题，同时保留轨迹级粒度，且难度可通过候选数 M 和相似度上限 ρ 精细调节。

**DEFT-RLVR** 则把未来轨迹从「决策前的锚」变成「决策后的验证目标」：第一轮模型在**看不到候选**的条件下从场景推出高层决策，第二轮才暴露候选并要求做细粒度匹配。RLVR 奖励 = 精确的候选正确性 + 实例级 rubric 奖励，且 **rubric 离线预生成、线上只做文本打分**，process reward 以 outcome 正确为门控。

### 关键实验结果
**锚定偏差的证据**：暴露 GT 后严重幻觉率从 29.0% 飙到 **50.0%**，成对偏好胜率从 60.5% 掉到 24.0%。这是全文最有说服力的数字——同一个模型、同样输入，只多给一段未来轨迹，幻觉率翻倍。

**主结果（Qwen3-VL-8B）**：训练自由的 DEFT 把 ACC 从 28.1% 提到 56.6%；同样只用正确性奖励，DEFT+RLVR 比 JEFT+RLVR 高 **15.3 个点**（76.4% vs 61.1%）；完整 DEFT-RLVR 达 77.9%，CFS 0.658、HLD 0.501。蒸馏路线更高：DEFT Distillation (Mixed Targets) 到 **84.1%**，CFS 0.934、HLD 0.627，而等数据量的 JEFT Distillation 只有 64.0%。Qwen3.5-4B 上趋势一致（34.0% → 65.6% → 79.0%）。

**通用能力不退反升**：DEFT+RLVR 让 12 个视觉基准的均分从 54.81 升到 56.39，RefSpatial 从 38.52 升到 44.26——这在领域微调里相当少见。

**跨域泛化**：500 个 nuScenes 场景上，训练自由 DEFT 39.6%，Mixed Targets 蒸馏 55.8%，DEFT-RLVR 49.5%。

**成本**：DEFT-RLVR 相比纯正确性奖励只增加 0.5% 的 step 时间（424.5 → 426.5 秒），比在线 rubric 变体快 41.1%（724.4 秒）。离线 rubric 这一招性价比极高。

### 局限性与开放问题
论文没有独立 Limitations 章节，这是它最明显的短板——一篇提出新 benchmark 的工作不讨论 benchmark 自身的局限，说不过去。我从数据里读出三点：

其一，**MCQ 形式本身是双刃剑**。Table 11 的候选集消融显示，当 ρ_max 提到 0.95（候选彼此高度相似）时，DEFT-RLVR 只有 55.6%，训练自由基线 45.8%，**增益从 M=6/ρ=0.50 时的 +20.0 掉到 +9.8**。也就是说方法在候选易区分时收益大，候选难区分时收益腰斩——而真实驾驶里的难例恰恰是后者。

其二，**蒸馏路线普遍损伤通用能力**（JEFT Distillation 均分从 54.81 掉到 49.80，Mixed Targets 51.80），而 RLVR 路线不损伤。论文把 84.1% 这个最高分给了蒸馏，但那条路线的通用视觉能力掉了 3 个点，两个指标不能同时拿——这个 trade-off 被表格藏起来了。

其三，**「选对轨迹」不等于「能开车」**。AD-MCQ 只验证 VLM 的决策，完整 VLA 的轨迹生成、控制、闭环执行全不在评测范围内。oracle 本身还有量化误差（平均 ADE 0.45 m，p95 达 1.63 m）。另外码本 K=16384 时样本外原型利用率掉到 73.7%，说明码本继续放大会失效。

其四，训练集只有 5000 个场景，Test 500 个，**数据规模在自动驾驶里算小的**，跨域实验只做了 nuScenes 一个。

### 启发与应用前景
最通用的启发是那个对照实验揭示的原理：**给 teacher 看答案会系统性败坏 CoT 质量**——这不是自动驾驶独有的，任何用「GT + LLM 生成解释」构造推理数据的流水线都中招（数学题给答案让模型编步骤、代码给正解让模型编推理，都是同一个坑）。「延迟暴露」这个矫正手段可以直接迁移：**先让模型在无答案条件下承诺一个决策，再暴露答案做验证**。

第二个可迁移点是**把开放式生成降维成可验证选择**——当目标空间可以用码本/原型离散化时（轨迹、动作、布局、配置），这招能把不可验证的任务变成 RLVR 可用的任务，同时避开生成任务对通用能力的侵蚀。

第三个是**离线 rubric**：实例级评分标准提前生成好、线上只做文本判分，41% 的加速几乎白捡。

follow-up 切入点：把 ρ_max=0.95 那档的困难候选做成课程学习的后期阶段；或者验证「DEFT 后的 VLM 接上真实轨迹生成头」端到端是否还成立。代码已开源。

---

## 5. DAPD: Dual-Anchored Policy Distillation
**👍 108** · 🏛 上海交通大学 / 上海人工智能实验室 / 中国科学技术大学 · [arXiv:2608.01735](https://arxiv.org/abs/2608.01735) · [GitHub](https://github.com/uanu2002/DAPD)

### 问题与动机
on-policy 自蒸馏（OPSD）是当下后训练的热门配方：让 teacher 拿到「特权信息」（比如参考解），再把它的分布蒸馏给学生。问题在于**特权幻觉（privilege illusion）**——学生学到的是**依赖特权信息才成立的行为**，但推理时根本没有那些信息，它却表现得好像还有。

论文把病因精确定位为**信息不对称**：teacher 在「见过参考解」的条件下产生分布，学生在「没见过」的条件下被要求匹配它。这个诊断比之前的方案（Purified OPSD 用互信息修正、DOPD 做动态路由）更根本——那些方法是在**过滤或路由一个不对称的信号**，而不是消除不对称本身。

失败的具体形态论文给了很生动的例子：OPSD 训出来的 Qwen3-4B 在 AIME25 第 20 题上写「既然是竞赛题，答案大概是个好看的数，比如 360°」然后交了 360；在 HMMT25 上写「我记得答案可能是 1/7 或 2/7」。**它学会了「跳到一个看起来对的答案」这个动作，因为 teacher 有参考解时确实可以直接跳。**

### 方法与核心创新
DAPD 用两层锚定消除不对称：

**Dual-Path Anchoring（DPA）**——引入一个 self-conditioned 桥接分布，在**信息条件匹配**的两条路径上对齐参考行为和 rollout 行为。具体是在每个 completion 的 token 上构造三种视图：`None`（无特权，就是推理时的条件）、`Cross`（给参考解）、`Self`（自条件桥）。Entangled Distillation、Inference Anchor、Privileged Anchor 三个目标在两个方向上同时评估，teacher 分布全部 detach，梯度只走学生侧。

**Dual-Source Anchoring（DSA）**——把这两条路径同时用在 reference→rollout 和 rollout→reference 两个方向，既保留正确性监督，又降低对特权参考指导的依赖。

实现细节：LoRA rank 64 / scale 128，divergence 用**分量裁剪的全词表前向 KL**（每个词表分量的贡献上限 c=0.05），学生 rollout 温度 1.1。

### 关键实验结果
Qwen3-4B 六任务均分 **57.34**，比 OPSD 高 **+2.00**，比 Purified OPSD 高 +1.09，比 DOPD 高 +3.85。分项看增益不是集中在一处：AIME25 +4.44、HMMT25 +3.06、IFBench +2.33、LCB v5 +1.05。

**规模趋势是这篇最关键的结果**：OPSD 相对基座的增益随规模迅速消失——1.7B 时 +5.19，4B 时 +1.39，**8B 到 32B 只剩最多 +0.28**（等于完全失效）。而 DAPD 在 8B/14B/32B 仍保持 +2.41 / +2.13 / +3.06。相对 OPSD 的领先在五个规模上分别是 +1.94 / +2.69 / +2.41 / +2.04 / +2.78，**没有随规模衰减**。这个对比说明白了一件事：模型越大，rollout 本身越有价值，硬灌特权分布的边际收益就越小，而匹配信息条件的做法不受这个影响。

**训练动态**直接证实了「特权幻觉」的存在：每万 token 的错误断言数，OPSD 从第 100 步的 14.81 一路涨到第 300 步的 **37.04**，同期 Reasoning Avg@12 从 61.50 掉到 53.24（**训得越久越差**）；DAPD 同期错误断言只到 11.11，均分维持在 60.90。

**OOD 泛化**：只在数学数据上训，评测代码和指令跟随——DAPD 均分 49.64，OPSD 48.27，Base 47.72。

消融：两条路径都必要（Entangled 单独 < 加 Inference Anchor < 再加 Privileged Anchor，达到 63.89/65.09）；两个来源结合（65.28）优于任一单独来源；前向 KL + 分量裁剪 65.28 vs 不裁剪 62.50 vs 反向 KL 62.50——**裁剪贡献了 2.78 点，几乎和整个方法相对 OPSD 的增益一个量级**。

### 局限性与开放问题
作者自承三条：(1) DAPD 训练时要构造多个锚定分布，**训练计算量增加**（推理时无额外开销）；(2) **权重可能需要跨规模/架构重新校准**——从 Table 5 看这不是虚言，五个规模的 λ、β_infer、β_priv 组合各不相同，4B 用 (.2, 1, 1, 1, 2)、8B 用 (.2, 1, 2, 1, .5)，**每换一个规模就得重新调 5 个超参**，实用性打折；(3) verified-rollout 扩展需要自动正确性信号，开放式任务用不了。

我另外观察到：其一，**全部实验都用 LoRA**（rank 64），全参微调下这套 anchoring 是否还成立没有验证，而 OPSD 类方法在全参下的表现可能完全不同；其二，**+2.00 的均分提升里有相当部分来自 AIME25 单项的 +4.44**，AIME 只有 30 题、Avg@12 采样，这个量级的波动需要多 seed 才能确认——论文用的是固定 seed 42 的匹配协议，没报 seed 间方差；其三，分量裁剪贡献 2.78 点这件事有点尴尬——它是个纯粹的数值稳定性技巧，与「信息不对称」这个理论故事无关，说明**方法收益里有不小比例来自实现层面的 trick 而非理论洞见**。

### 启发与应用前景
理论上最值得记住的一句话：**蒸馏时让 teacher 和 student 处在同样的信息条件下，比事后过滤或路由不对称信号更有效**。这个原则适用范围远超 OPSD——RLHF 里 reward model 见过参考答案、多模态蒸馏里 teacher 见过额外模态、agent 蒸馏里 teacher 见过环境反馈，都是同一类信息不对称。

工程上直接可用的是**「训得越久越差 + 错误断言数上升」这个诊断指标**：如果你的自蒸馏跑到后期指标下滑，先去数一下模型有没有开始写「答案大概是」这类话术，这是特权幻觉的典型信号，比看 loss 直观得多。

follow-up 角度：(1) 把权重校准自动化（用验证集在线搜 β），解决跨规模重调的实用性问题；(2) 全参微调下复现，验证不是 LoRA 特有的现象；(3) 把 matched-information 原则推到 agent 场景——teacher 见过环境完整状态、学生只有观测，这个不对称比参考解更严重。代码已开源。

---

## 6. From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement
**👍 105** · 🏛 杜克大学 / Adobe / 俄勒冈州立大学 / 新加坡国立大学 · [arXiv:2607.23802](https://arxiv.org/abs/2607.23802) · [GitHub](https://github.com/wangqinsi1/RLSVR)

### 问题与动机
RLVR 把推理模型推上了新台阶，但它的适用范围被死死卡在数学和代码这类**正确性可确定性判定**的领域。开放式任务（摘要、创意写作）只能退回人类偏好、reward model 或 LLM judge，这三条路各有硬伤：评估偏差、judge 能力天花板（judge 比 policy 弱就没法继续提升）、以及额外推理成本。

作者的切入角度很聪明——借自监督学习的思路：**自监督不是找到了标签，而是构造了一个 pretext task 让标签从数据本身长出来**。那 RL 能不能也做「任务变换」，把不可验证的目标转化成一个规则内生、结果可自动判定的代理环境？

### 方法与核心创新
提出 **RLSVR**（Reinforcement Learning with Self-Verifiable Rewards）范式，并用 **SpyRL** 实例化——原型是《谁是卧底》：

5 个玩家中 4 个「平民」拿到完整输入，1 个「卧底」拿到被遮蔽的输入（摘要/写作遮 20% 的连续片段，数学遮 40%）。所有人完成**同一个目标任务**（写摘要 / 写故事 / 出题并解题），然后集体投票指认谁是卧底。

关键在于**卧底身份是环境预先指定的**，所以投票结果**天然可验证**，同时「谁被投票」又与输出质量高度相关——信息缺失会在输出里露馅。于是「输出质量」这个不可验证的目标被换算成了「输出是否暴露信息缺陷」这个可计算的代理量。两个阶段的奖励是耦合的：预设身份监督 detection 阶段，detection 的结果又决定 performing 阶段的奖励。

还有一个必要组件 **Role-Advantage Estimation（RAE）**：卧底和平民面对的任务难度结构性不同，原始奖励不可比，必须减去角色特定的 baseline。

### 关键实验结果
**代理奖励与真实质量的对齐性验证**（这是全文最关键的证据）：跑 100 局，记录每个玩家的得票数，同时用 GPT-4o 对 5 份输出按质量排序。结果得票数与质量排名正相关——**质量差的更容易被怀疑、拿到更小奖励**。这一步做实了「投票奖励 ≈ 质量奖励」。

**摘要**：Qwen3-4B 在 GovReport 的 ROUGE-L 从 30.2 提到 **36.7**（R-Zero 32.1、Absolute Zero 33.2），五个数据集全面领先，GPT-4o A/B 对基座胜率 74.6%。跨域复现：在 PubMed 上训练，arXiv/PubMed/BillSum 三个集平均 ROUGE-L 提升 **4.9 点**（33.2 → 38.1）。

**创意写作**：对基座胜率 81.3%（WritingPrompt）/ 75.1%（WritingBench），对 R-Zero 78.9%、对 Absolute Zero 75.6%。**人工评测独立复现**了这个结论（对基座 80.0%），不是只靠 GPT-4o 自评。

**数学（无外部 verifier 却在可验证任务上也赢）**：GSM8K 84.5 → 93.4，Math500 68.2 → 79.5，**AIME25 6.7 → 20.0（三倍）**，GPQA-D 26.3 → 41.3。这个结果比开放式任务的更让人意外——它说明信息不对称自博弈提供的信号，比 Absolute Zero 那类自出题范式更有效。

**成本对比**很有说服力：对 Qwen3.5-27B-RaR（rubric-as-reward）胜率 59.3%，对 GPT-4o-RaR 胜率 48.9%——**基本打平 GPT-4o 做裁判的方案，而后者在他们的实验里额外花了约 900 美元的 verifier 成本，SpyRL 花 0**。

**消融**：只训 performing 阶段（冻结检测器）会在 72 分左右迅速饱和振荡——因为静态检测器跟不上越来越精细的输出；只训 detection 几乎无提升；**去掉卧底机制同样卡在 71.6**。完整 SpyRL 到 79.5。RAE 的消融最惊人：**去掉后七基准均分从 50.4 崩到 37.5，低于基座的 41.4**——即梯度是有害的，而不只是变慢。

### 局限性与开放问题
论文没有独立 Limitations 章节。我读出的几个真问题：

其一，**遮蔽算子 g(·) 是唯一需要按任务人工指定的部件**，虽然作者做了 20% vs 40% 的敏感性实验（差异很小），但「哪种降级方式适合哪类任务」仍是人工设计，范式的通用性没有被真正验证——摘要和写作都用「遮连续片段」，换成对话、多轮 agent、代码这类任务该怎么降级并不显然。

其二，**跨任务迁移是单向的**：写作 ↔ 摘要互相有正迁移（55%–64%），但**数学训出来的模型在两个写作任务上都跌破 50%**（38.5%–45.6%），说明这套自博弈学到的东西有明显的能力分区，不是通用的「输出质量」提升。

其三，**规模只做到 4B/8B**，而自博弈类方法的一个经典问题正是：模型越强，卧底越会伪装、检测越难，信号会不会退化？论文的玩家数消融显示 3→5 人边际收益最大、6→8 人已明显递减，这暗示环境复杂度的收益是有上限的。

其四，**AIME25 从 6.7 到 20.0** 这类数字在 30 题的基准上等于多做对 4 题，波动区间很大，论文没报多 seed 方差。

### 启发与应用前景
这篇最有价值的是那句总结：**可验证性不必是任务的内在属性，而可以通过任务变换工程化地造出来**。这个思路的适用面远超论文本身——任何「质量不可判定但缺陷可暴露」的任务都可以套：多轮对话（谁的回复缺少上下文信息）、代码审查（谁的 review 漏看了关键 diff）、检索增强生成（谁的答案没读到关键文档）。核心配方是**制造信息不对称 + 让群体投票指认**。

第二个可迁移点是 **RAE 的教训**：多角色自博弈里，不同角色的任务难度不同，直接用原始奖励做优化会把「信息劣势」误判为「策略差」，梯度直接反向。任何非对称多智能体 RL 都该检查这一点。

follow-up 切入角度：(1) 把 g(·) 自动化——用 RL 学一个「降级策略」，让卧底的难度自适应；(2) 验证规模上限，看 30B+ 上信号是否退化；(3) 把这套机制用到 agent 轨迹质量上（卧底 agent 少看一个工具返回值，看能否被识别）。模型和代码已开源。

---

## 7. Mental World Modeling
**👍 103** · 🏛 牛津大学 / 新加坡国立大学 · [arXiv:2607.27201](https://arxiv.org/abs/2607.27201) · [GitHub](https://github.com/mental-world/Mentis) · [项目页](https://mental-world.github.io/)

### 问题与动机
现有世界模型——不管是 Dreamer 系的表征世界模型、Sora/Genie 系的视频生成世界模型，还是 POMDP 系的隐状态模型——回答的都是**物理问题**：有什么、在哪、会怎么演化。但人的行为由**隐藏心理状态**驱动：他相信什么、想要什么、打算做什么、感受如何、认为什么是社会可接受的。

作者的核心论断很锋利：**一个能追踪物理场景但不追踪「每个人对场景知道什么、相信什么」的模型，会在一个看起来完全正确的场景上预测出错误的动作。** 场景看对了，人算错了。

### 方法与核心创新
**MWM（Mental World Modeling）**把 POMDP 做了一个关键的形式化扩展：标准 POMDP 隐藏的是**物理状态的一部分**，MWM 把**心理和社会变量本身放进隐世界状态**。

联合状态 s_t = (s_t^phy, s_t^ment)：物理侧是物体、角色、物理关系、环境；心理侧是**每个个体的心理状态**（身份、信念、注意、目标、意图、情绪、性情、规范、行为约束九个字段）+ 群体心理状态 + 心理关系 + 场景氛围。

两个计算角色明确分离：**TargetAgent** 是第一人称的，只能基于渲染给它的**局部观测** o_t^ε 行动；**WorldModel** 是第三人称的，维护联合状态、接收目标动作、模拟后继状态和下一个观测。动作本身也是耦合的：a = (物理载体, 心理/语义内容)。

**Mentis** 是训练无关、完全可检视的基线实现，把过程拆成五步：状态解析 → 目标观测生成 → 动作分解 → 物理心理耦合转移 → 分支级价值评估。价值评估器给每个分支打三个归一化分数——**心理一致性**（这个人以他的信念和目标会不会选这个动作）、**物理可行性**、**社会得体性**，外加一个二值的安全/合法否决票。决策模块是确定性的、放在语言模型之外，保证可复现可审计。

### 关键实验结果
Menti-Bench 448 条人工构造的情境决策记录（320 文本 / 100 图像 / 28 有声视频），六选一，**人类参考线 98.5**。

**必要性阶梯**（8 个 LLM 的平均 F1）：选项-only 地板 31.3 → 直接作答 63.3 → +CoT 74.6 → +自洽性 SC@6 77.9 → +自由文本状态 80.3 → +结构化状态 82.6 → **完整 MWM 87.9**。最强配置 gpt-5.6-sol 达 90.7。关键结论：**结构化建模比直接作答高 21–28 点，而且加多少测试时采样都补不上这个差距**（SC@6 只到 77.9）。

**通道消融**证明两个通道都必需：去掉心理侧掉到 75.8，去掉物理侧掉到 71.4，**把物理和心理转移解耦预测掉到 81.5**——耦合本身值 6.4 点。

**Oracle 瓶颈定位**（最有方法论价值的部分）：给 gpt-5.6-sol 逐个换上金标准中间产物，S6 的 90.7 分别提升到——gold state +2.8、gold observation +1.7、gold action +0.7、**gold transition +3.5（最大）**；四个全给到 97.0，离人类只差 1.5。这直接给出了改进优先级：**瓶颈在后继状态模拟，不在状态解析**。

**模态泛化**：直接作答在文本 70.8 / 图像 67.0 / 视频 64.8，有明显模态惩罚；到 S6 时三者变成 90.5 / 91.2 / 90.9，**模态差距完全消失**，而且这个收敛在 S5（结构化状态、还没做模拟）就已完成。通道干预验证了证据确实被用到：图像换成中性 caption 让 S6 掉 6.4 分（S1 只掉 2.8），只留音频掉 18.8 分——**结构化系统对证据的依赖比直接作答更强，说明它不是靠文本先验蒙的**。

### 局限性与开放问题
论文有 B.3「Limitations of the Baseline」明确说明 Mentis 只是基线实现而非 MWM 的最终架构，继承了 prompted pipeline 的固有弱点。作者在 M.1 里还有一段很清醒的自我批评：**当分数逼近天花板（97.0 vs 人类 98.5），outcome 指标本身就变成了弱工具**——一个「因错误理由而答对」的系统（正确选择建立在不忠实的状态之上）恰恰是输入漂移时最不能信的，所以 outcome-process 背离必须单独报告而不能被平均掉。这个反思水平在 benchmark 论文里少见。

我观察到的问题：其一，**数据规模太小**——448 条，其中视频只有 28 条。表 8 里视频通道干预的 delta 都建立在 n=28 上，作者自己也标注「视为方向性结论」。整个 modality 结论的统计功效不足。

其二，**六选一的形式限制了结论的外推**。选项-only 地板已经 31.3（远高于随机的 16.7），说明选项集本身泄漏了不少信息；而真实的行为预测是开放的。

其三，**Mentis 是 training-free 的 prompted pipeline**，五阶段每步都要调一次 LLM，成本相当高（还要 batched 比较打分），论文没报 token 开销或延迟——一个要跑五步 LLM 才能预测一个动作的「世界模型」，在需要快速 rollout 的规划场景里基本不可用。

其四，**「金标准心理状态」这个标注对象本身有效度问题**：人的信念和意图是不可观测的，人工标注的 gold mental state 到底是真实心理还是标注者的合理化，论文没有做标注者间一致性分析。

### 启发与应用前景
方法论上最值得学的是 **oracle 干预做瓶颈定位**这套实验设计——逐阶段替换金标准，量化每一步贡献多少误差。任何多阶段 pipeline 都该这么做，比笼统的消融信息量大得多。

概念上，「心理状态作为世界状态的一等公民而非事后 rationale」这个立场，对做 role-play、社交 agent、用户建模的团队有直接价值。特别是**目标观测渲染**这个抽象——从第三人称联合状态渲染出第一人称局部观测，天然处理了信息不对称，这比让 LLM「扮演角色」要严谨得多（后者极易泄漏角色不该知道的信息）。

follow-up 角度：(1) 瓶颈已经指明在转移模拟，可以专门针对「心理状态如何随动作演化」做训练而非 prompting；(2) 把 MWM 接到真实交互环境（SOTOPIA 类）做闭环而非静态选择；(3) 用更小的模型跑 Mentis 的各个阶段，做成本-效果曲线。代码已开源。

---

## 8. MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations
**👍 96** · 🏛 阿里巴巴 / 浙江大学 · [arXiv:2607.28956](https://arxiv.org/abs/2607.28956) · [GitHub](https://github.com/KhanCold/merchantbench)

### 问题与动机
Agent benchmark 绝大多数是**有界任务 + 即时成功判定**：做完一件事，判对错，结束。但真实部署要的是**长期一致性（Long-Term Coherence）**——在很长的时间跨度里保持目标导向的行为，同时根据不断累积的证据调整决策。

评这个能力需要一个特定形态的环境：动作要**约束未来选择**（今天进的货明天还占着资金和货位）、反馈要**以不同延迟到达**（供应商涨价立刻可见，退货和差评几周后才发生）、不一致的行为要产生**可测量的累积后果**。卖家侧电商恰好全占：选品、上架定价、现金流管理、混合延迟反馈适应四类决策反复交织。

### 方法与核心创新
MerchantBench 是一个**365 天订单级模拟**，基于 1688 平台的 **98,843 个真实商品记录（36,576 个供应商）**，覆盖 2025-06-01 到 2026-05-31 的真实日级需求历史，包含 618、双十一双峰、春节低谷等真实季节结构，另附 365 份日度市场报告作为选品信号。给 agent 26 个工具，分四类（选品、上架定价、现金流、店铺运营）。

环境设计的关键是**双时间尺度反馈耦合**：上游供应商事件（涨价、下架、发货延迟）即时可见，下游订单结果（取消、退款、仅退款、差评）延迟发生。agent 必须跟踪单个订单的完整生命周期，并回头修正早先的决策。

初始条件统一：现金 2000 元、保证金 1000 元、最多 50 个在架商品。罚款规则照搬平台真实规则（退款退货 8 元、差评/缺货/余额不足 5 元、迟发 3 元）。

指标分三组：业务表现（终期净资产、GMV、净利率、订单数）、店铺可靠性（罚款、评分、异常率）、**长程活跃度**——其中 **SWR（Sustained Window Rate）** 是最有意思的：所有滚动 30 天窗口中，「包含至少一次环境工具调用的决策窗口占比」的最小值，直接测量 agent 有没有半途摆烂。

### 关键实验结果
8 个模型 × 2 个框架（ReAct / Hermes）× 3 次重复 = 48 次 365 天完整运行，另有规则基线和 3 名无电商经验的人类参与者。

**核心结论**：**最好的 LLM 配置只拿到人类平均终期净资产的 27.3%**。人类没有电商经验，只在 5 个日历日内操作完 365 个模拟日。

单项：ReAct 下 GPT-5.6 Sol 最高（净资产 40.89K）；Hermes 下 Qwen3.7-Max 最高（59.46K），也是 16 个配置的总冠军。但**Qwen3.7-Max 的变异系数高达 55.1%**，而 GPT-5.6 Sol under ReAct 只有 3.3%——最高分和最稳分不是同一个模型，这个对比很值得注意。

**框架效应显著**：Hermes 相比 ReAct 平均高 53.3% 净资产、71.5% GMV、71.2% 订单，8 个模型里 7 个受益（Qwen3.7-Max 提升 187.8%，Claude Opus 4.8 只有 11.5%），Kimi K2.6 反而低 4.1%——**框架收益强依赖底层模型**。

**失效模式的刻画是这篇最有价值的部分**：
- **Operational Coherence 崩溃**：人类 SWR 保持 100%，LLM 在 ReAct 下 10.6%–99.4%、Hermes 下 17.8%–66.1%。Qwen3.7-Max 的季度有效窗口率在 ReAct 下从 68% 掉到 23%。
- **Control-Loop Narrowing**：运营循环退化成被动响应供应商事件，几乎不再主动选品、调价、诊断。ReAct 下 Qwen3.7-Max 的 SWR 11.1% 伴随供应链检查从 14% 涨到 34% 的剩余工具调用占比——**它没有停下来，它只是变成了一个只会看警报的机器**。
- **Premature Abandonment**：一次 Hermes Kimi K2.6 的运行里，agent 在**第 104 天就判定店铺无法挽救**，此后 523 个决策窗口中的 355 个没有采取任何环境动作——尽管可行动作依然存在。

### 局限性与开放问题
论文没有独立 Limitations 章节。我看到的问题：

其一，**人类基线只有 3 个人、每人 1 次**，而 LLM 是 8 模型 × 2 框架 × 3 次。「27.3%」这个标题数字的分母是 3 个样本的均值，方差完全没报。这是全文最需要补强的地方——用 3 个人的平均值去锚定一个 benchmark 的人类天花板，说服力不足。

其二，**人类和 agent 的操作条件不可比**：人类通过 dashboard 在 5 天内跑完 365 天，可以随时回看全局；agent 受上下文压缩限制（ReAct 到 16 万 token 就压到 3 万）。这个差距有多少来自「能力」、多少来自「界面和记忆」是分不开的。

其三，**模拟器的隐藏参数（cancel_rate、refund_rate、ref_price 等）是从真实数据标定的，但需求模型本身是构造的**，agent 的最优策略在多大程度上是「摸清模拟器」而非「学会做生意」，论文没有讨论。

其四，**没有做失效模式的定量归因**——Operational vs Strategic Coherence 的划分是基于 trace 的定性观察，论文自己也用了「suggests」「may contribute」这类措辞。

### 启发与应用前景
**SWR 这个指标值得单独抄走**：用「所有滚动窗口中活跃度的最小值」而非平均值来测长程一致性，能抓住平均值掩盖的中途摆烂。任何长程 agent 评测都可以加这一项。

「**Premature Abandonment**」和「**Control-Loop Narrowing**」这两个失效模式的命名和刻画，对做生产环境 agent 的团队有直接诊断价值——比「效果不好」精确得多。前者可以用「禁止自我终止 + 强制最小动作频率」缓解，后者需要在 harness 层面区分「响应式工具」和「主动式工具」并对后者做配额。

框架 × 模型的交互效应（同一个 Hermes 让 Qwen 涨 187.8%、让 Kimi 跌 4.1%）也提醒一件事：**agent 框架的评测必须跨模型做，单模型结论不可外推**。

follow-up：把人类基线扩到 20+ 人并报方差；或者用这个环境研究「记忆架构」——上下文压缩策略与 SWR 衰减的关系，是最直接可做的一个消融。代码已开源。

---

## 9. JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion
**👍 90** · 🏛 京东 · [arXiv:2608.03974](https://arxiv.org/abs/2608.03974) · [GitHub](https://github.com/jd-opensource/JoyAI-Video-Edit)

### 问题与动机
实时视频编辑要同时满足几件互相打架的事：**低延迟因果生成**（不能看未来帧，也不知道视频总长）、**有界算力**、**保持源视频保真度**、**长时序一致性**。现有方案两极分化——离线编辑器（VACE、Kiwi-Edit、Bernini-R）质量好但必须拿到完整视频；流式编辑器（StreamDiffusionV2、SANA-Streaming、LiveEdit）能实时但质量差得离谱（OpenVE-Bench 上 1.23–2.62 分，满分 5）。

技术上的三个具体障碍：chunk-wise 自回归带来的**训练-推理不匹配**、两步生成下的**源保真度流失**、长 rollout 的**误差累积漂移**。

### 方法与核心创新
16B 参数的自回归扩散框架，三个组件：MLLM（从首帧 + 编辑指令抽条件 token）、因果视频 VAE（时空压缩 8×24×24，即每个 latent 帧代表 8 个视频帧）、MM-DiT 扩散主干。

三个针对性设计：
1. **Chunk-wise 自回归适配**——降低训练与推理的分布不匹配。
2. **SA-DMD（Source-Anchored Distribution Matching Distillation）**——把 teacher 锚定到**对齐的源 chunk** 上，在两步生成的极限压缩下保住源保真度。这是解决「蒸馏后画面越编越偏离原视频」的关键。
3. **LHAD（Long-Horizon Autoregressive Distillation）**——针对长 rollout 后期误差累积的状态做优化，稳定漂移。

训练走四阶段课程：T2I 预训练（256²→512²，4.3B+793M 样本）→ T2V（256p 12/24fps → 360p → 480p SFT/CT）→ 图像编辑 SFT（720p，3.2M）→ 双向视频编辑（V2V/IV2V 联合，720p，5.3M + 1.1M）。

数据侧有个务实的解法：高质量成对视频编辑数据极难规模化收集，于是**把成熟的图像编辑监督迁移过来**——选代表性关键帧编辑后，用 IV2V 模型把编辑传播到全视频；或者用 latent-shared I2V 生成，两个分支共享早期去噪 latent（保运动和构图一致）、后期条件于不同图像（引入编辑）。

### 关键实验结果
**短视频（OpenVE-Bench，Gemini 判分 1–5）**：总分 **3.60**，比 SANA-Streaming 高 0.98、比 LiveEdit 高 1.60、比 XMax-X2.0 高 1.73、比 StreamDiffusionV2 高 **2.37**。更值得注意的是它已经追平离线系统：Kling-3.0 Omni 3.64、Kling-O1 3.62、Bernini-R 3.72（27B），而 JoyAI 是**因果推理**的。局部移除项拿到全场最高分。

**长视频（自建 LongV2VBench，229 个一分钟任务）**：总分 **3.30，吞吐 30.19 FPS**——两项都是流式方法第一，而且是压倒性的：第二名 XMax-X2.0 是 1.71 分 / 20.90 FPS，SANA-Streaming 1.64 分 / 14.51 FPS。**质量翻倍的同时还更快**。

**延迟分解**很有意思：81 帧全流水线 2.68 秒（30.19 FPS），其中 DiT 2.18 秒（37.21 FPS）、**VAE 只要 0.405 秒（200 FPS）**——对比 StreamDiffusionV2 的 VAE 要 2.19 秒（37 FPS）。也就是说 JoyAI 的加速有相当大一块来自 VAE 侧（8×24×24 的激进压缩），不全是 DiT 的功劳。

**人工评测**：对 LiveEdit 90%、SANA-Streaming 87%、StreamDiffusionV2 87%、XMax-X2.0 81% 的偏好率；对离线的 Bernini-R 是 48% vs 44%（基本打平），对 Kling-3.0 Omni 和 Seedance 2.0 各拿 56%。

**消融**：SA-DMD 单独贡献最大（总分 2.81 → 3.23，global style 3.61 → 4.24），LHAD 单独 2.81 → 3.06（主要改善背景替换和局部移除），两者合起来 3.30。

### 局限性与开放问题
论文**没有任何 Limitations 章节**——对一个宣称 30 FPS 实时的系统论文，这是明显缺失。我从数据里读出几点：

其一，**硬件门槛被轻描淡写了**。30 FPS 是在**单张 Nvidia B200** 上测的，这是当前最贵的数据中心卡之一。「实时」这个卖点在消费级硬件上完全没有验证，对「直播、实时数字人、游戏内容」这些论文自己列举的应用场景来说，B200 的成本结构基本排除了 C 端可能性。

其二，**消融数字有点微妙**。看 Table 5：单独 SA-DMD 的 global style 是 4.24，**加上 LHAD 后反而降到 3.85**；单独 SA-DMD 的 background change 2.49，完整版还是 2.49。也就是说 LHAD 在部分维度是负贡献，完整配置赢在 local add（2.74 → 3.10）和 local remove（2.67 → 2.99）。作者说两者「高度互补」，实际上更像是**在不同维度上此消彼长，总分略胜**。

其三，**LongV2VBench 是自建的**，判分也是 Gemini。自建长视频基准 + 自家模型第一，缺少第三方交叉验证。而 OpenVE-Bench 这个公开基准上，它其实**没有超过 Bernini-R（3.72 vs 3.60）**。

其四，人工评测「全部是 10 秒以内的单镜头视频」——**长视频的质量优势恰恰没有经过人工验证**，而长视频正是它的主打场景。

### 启发与应用前景
最有工程价值的洞见是 **VAE 侧的时空压缩是流式生成的隐藏瓶颈**。多数工作盯着 DiT 做加速，而 JoyAI 的数据显示对手的 VAE 解码耗时和 DiT 相当甚至更多。8×24×24 这个激进压缩率（每 latent 帧 = 8 视频帧、空间 24 倍）是 200 FPS VAE 的来源，值得任何做实时视频生成的团队重新审视自己的 VAE 配置。

**SA-DMD 的思路可迁移**：蒸馏到极少步数时，teacher 应该锚定在**与当前 chunk 对齐的源内容**上，而不是自由分布——这个原则对图像编辑、音频编辑的少步蒸馏同样适用。

**用图像编辑监督合成视频编辑数据**（关键帧编辑 + IV2V 传播 / latent-shared 双分支 I2V）也是务实的可复制方案，成对视频编辑数据稀缺是全行业的问题。

follow-up：把模型压到消费级显存（16B + B200 显然不是终点），或者验证 LHAD 在什么条件下真正必要。代码已开源。

---

## 10. AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning
**👍 89** · 🏛 清华大学 / 浙江大学 / 美团 · [arXiv:2608.05987](https://arxiv.org/abs/2608.05987) · [GitHub](https://github.com/ZethWang/AgentOPSD)

### 问题与动机
长程多轮 agentic RL 的核心痛点是**信用分配**：GRPO 把轨迹级 advantage 均匀广播到所有 token，但一条 50 轮的轨迹里，真正决定成败的可能只有两三个关键决策。把同样的信用发给「打开冰箱」和「误判了目标物体位置」，学习效率必然低下。

近期有工作把**特权自蒸馏**引入信用分配（teacher 拿到额外信息，用 teacher-student log-prob gap 作为稠密监督），但论文指出一个未被回答的问题：**这种局部信号该如何表达「序列性」的信用？** 一个 turn 的 teacher-student gap 大，不一定意味着这个 turn 关键——可能只是这个 turn 恰好难。

### 方法与核心创新
**AgentOPSD** 是 critic-free 的递归式 turn 级信用分配，核心机制分三步：

1. **turn 级聚合**：把 token 级的 teacher-student log 概率差在**环境对齐的 turn 边界**上聚合成 turn 级证据 e_k。理由很直接——环境反馈对应的是一个完整动作，不是单个 token。

2. **log-odds 空间的递归贝叶斯信念更新**：维护一个「轨迹最终成功」的信念状态 B_k，用 turn 级证据递归更新。**先验 B_0 锚定在该 GRPO group 的成功率均值 R̄** 上——这是个 verifier-grounded 的任务难度估计。

3. **用相邻状态间的边际修正 ΔB_k 识别关键 turn**：一个 turn 的信用不取决于它的局部 gap 多大，而取决于**这个 gap 让「最终会成功」这个信念改变了多少**。同样大小的局部 gap，在结果未定时是决定性的，在信念已经明确指向某个结果后就是冗余的。再乘上与最终 outcome 对齐的**符号方向**——成功轨迹里向上的信念修正该加分，失败轨迹里同样的修正该扣分。

整套机制不需要额外 rollout，也不需要学习 critic，与标准 policy optimization 完全兼容。

### 关键实验结果
三个环境（ALFWorld / WebShop / Search-QA）× 两个规模（Qwen2.5-3B/7B）。特权 skill 从 SkillRL 的 SkillBank 按关键词检索，**只在训练时用，推理时不用任何外部 skill**。

**主结果**：ALFWorld 上 Qwen2.5-7B 达 **89.1%** 成功率。在 8 组聚合对比中全面超过 GRPO+OPSD、Skill-SD、RLSD，8 组中 6 组超过 SDAR。3B 上 ALFWorld 从 GRPO 的 75.0 提升（Skill-GRPO* 80.5、GRPO+OPSD 81.2、SDAR 84.4）。

**最有说服力的证据是长度依赖分析**：在 ALFWorld 上把子任务成功率对「成功 episode 的平均轮数」做回归，报告每多一轮损失多少成功率点——RLSD **-3.59**、GRPO **-2.91**，而 **AgentOPSD 只有 -0.54**。这条曲线直接验证了方法的动机：轨迹越长，单一广播 advantage 要覆盖的决策越多，历史依赖的信念修正就越有价值。

**消融把每个设计的贡献切得很干净**（ALFWorld / Qwen2.5-7B，完整版 89.1）：
- 换成 per-token 累积 → **85.9**（-3.2）：环境反馈对应完整动作，token 级累积把一个决策拆碎了；
- 用原始局部 gap e_k 代替递归修正 ΔB_k → **82.8**（-6.3）：这是**验证核心论点的关键消融**——局部 gap 本身不是序列信用；
- 只用幅度 |ΔB_k| 丢掉符号 → **80.5**（-8.6）：幅度知道信念在哪变了，但不知道变的方向和最终结果一不一致；
- 去掉 B_0 经验先验锚定 → **78.9**（-10.2，影响最大）：没有这个锚，轨迹从任意不确定度起步，早期信念修正会被错误缩放。

### 局限性与开放问题
论文**没有 Limitations 章节**。我看到的问题：

其一，**规模只到 7B，环境只有三个经典 benchmark**。ALFWorld 和 WebShop 都是文本模拟环境，Search-QA 的最大交互轮数只设了 4 轮——**真正长程的只有 ALFWorld（50 轮）**，而长度依赖那条最有力的曲线也只在 ALFWorld 上做。方法的卖点是长程，验证却主要靠一个环境。

其二，**依赖特权 skill 检索**。teacher 的优势来自从 SkillBank 按关键词匹配到的 skill，这意味着**方法的适用前提是存在一个覆盖该任务域的技能库**。论文强调「增益来自信用构造而非特权访问」（所有特权基线用同样的 skill），这个对照是成立的，但不改变一个事实：**没有 SkillBank 就跑不起来**，这在新领域是硬约束。

其三，消融里 **B_0 锚定的影响（-10.2）比递归机制本身（-6.3）还大**，而 B_0 = group 成功率均值本质上就是 GRPO 已有的量。这让人怀疑相当一部分收益来自「用 group 难度校准了 advantage 的缩放」这个相对朴素的效果，而不是贝叶斯信念递归这个精巧的故事。

其四，**超参 λ=0.5、b=0.2、γ=0.95 三个新超参**，论文只给了一组配置，没有敏感性分析。

### 启发与应用前景
核心洞见很值得记住：**局部信号 ≠ 序列信用**。判断一步是否关键，要看它**相对于此前累积的历史**改变了多少对最终结果的预期，而不是它自身的绝对幅度。这个原则可以脱离自蒸馏这个具体载体——任何有「逐步局部信号 + 稀疏最终结果」结构的问题（多轮对话质量、代码生成中的中间编辑、长文档推理链）都能套上「信念修正即信用」这个框架。

「**符号方向必须与 outcome 对齐**」这个细节（贡献 8.6 点）也值得单独记：如果你在做基于置信度/不确定度变化的信用分配，只用幅度会把「成功时的正确转折」和「失败时的错误转折」混为一谈。

follow-up：(1) 去掉对 SkillBank 的依赖，用轨迹内 hindsight 信息构造 teacher；(2) 把 B_0 锚定和递归修正做因子化对比，弄清收益来源；(3) 推到真实工具环境（浏览器、终端）验证长程优势。代码已开源。

---

## 11. Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing
**👍 88** · 🏛 腾讯混元 · [arXiv:2608.02711](https://arxiv.org/abs/2608.02711) · [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0) · [项目页](https://tencent-hunyuan.github.io/Hunyuan3D-Buffalo1.0/)

### 问题与动机
图像领域已经证明了统一多模态模型（理解 + 生成 + 编辑一体）的价值，3D 领域却卡在数据上——特别是**大规模、几何一致的编辑数据几乎不存在**。现有 3D 编辑数据的两条构造路线各有硬伤：2D-to-3D lifting（先在图像域编辑再抬回 3D，如 ShapeLLM-Omni）缺少源资产与目标资产之间的显式 3D 对应，结果是身份漂移、几何幻觉、多视图不一致、以及不该改的区域被改了；显式 3D 空间约束路线（Nano3D、VoxHammer）虽然把修改限制在局部区域，但编辑结果的几何质量被底层模块封顶。

### 方法与核心创新
框架把 **Hunyuan3D-VLM**（负责语义、结构、空间理解）和 **Hunyuan3D DiT**（负责高保真 3D 合成）组合起来：VLM 为生成提供多模态语义条件；编辑和部件生成任务额外把扩散过程**条件在源物体表示上**，以保住整体结构和未编辑区域。

真正的重头是 **8700 万规模的 3D 多模态语料**，三块：
- **2500 万理解样本**：700 万纯文本 + 300 万图文（防止 3D 适配过程中语言和 2D 能力退化）+ **1500 万点云-文本**。3D 任务覆盖 captioning、QA、**grounding（用 `<boxs>/<boxe>` token 序列输出量化轴对齐包围盒）**、编辑指令合成（把粗糙的编辑请求转成基于几何的可执行指令）、编辑结果 captioning。
- **5000 万 text-to-3D 对**：全自动五阶段流水线，其中**四层提示词分类法**把「生成什么」和「怎么组合」解耦（L0 生成模式 / L1 二十个类别 / L2-L3 子类型与细粒度实例），再叠加风格、颜色、材质、状态、姿态五类受控词表；每个资产配 **6 档 caption**（4–6 句详述 / 24–30 token 主描述 / 简化 / 改写 / 6–10 token 短描述 / ≤8 关键词标签）。
- **1200 万编辑对**：由 **Nano3D-v2** 生成——锚点视角选择 + 学习式 3D 编辑区域定位 + 体素级局部修改 + 细粒度几何精修 + 多模态质量过滤。

### 关键实验结果
**3D 理解（UniPart-Bench）**：部件级 QA 的 SBERT 85.47 / SimCSE 89.06 / BLEU-1 49.95，对比最强基线 UniVerse3D 的 83.11 / 87.16 / 46.79；物体级 captioning 差距更大——SBERT **72.94 vs 65.18**、BLEU-1 **50.93 vs 42.75**、METEOR **50.47 vs 41.11**。

**text-to-3D 用户研究**（四选一并列展示、随机顺序）：**56.6% 总体偏好率**，第二名 Omni123 只有 18.4%，TRELLIS 14.4%，Universe3D 8.3%。

**3D 编辑（Edit3D-Bench）**是差距最悬殊的一项：平均 Chamfer Distance **0.0091**、F1 **0.6515**，而此前最好的 Omni123 是 CD 0.0684 / F1 0.2001，Steer3D 是 0.1190 / 0.2729。**CD 降低了 7.5 倍，F1 提高了 3.3 倍**。这个量级的领先在成熟 benchmark 上很少见，说明编辑数据确实是此前的瓶颈而非模型能力。

**数据规模消融**（用户偏好率）：300 万样本 8.4% → 1500 万 28.6% → **5000 万 57.5%**，还在陡峭上升段，没有饱和迹象。

另有一个值得注意的对照：编辑任务用 **3D-VLM 条件比用 CLIP 条件**更好（CD 0.0091 vs 0.0158，F1 0.6515 vs 0.6336），验证了「更强的 3D 指令理解能提升编辑」这个统一建模的核心假设。

### 局限性与开放问题
作者在结论里列了 **6 条**，是本周里最诚实的 limitations 之一：
1. **单阶段高质量几何表示尚未收敛**——当前高质量几何生成普遍依赖多阶段 DiT（如 TRELLIS），而多阶段对统一模型是灾难：要编辑就得做多阶段编辑，难以规模化。3D 能否像图像那样做到单阶段高质量，是开放的根本问题。
2. **caption 质量受限于 MLLM**——目前用 Gemini 做 3D 资产标注，描述仍有大量歧义，text-to-3D 数据对是有噪声的。
3. **纹理编辑基本没做**——当前只聚焦几何编辑，缺训练数据。几何与纹理能否联合建模是更大的问题。
4. **编辑数据构造流水线不够鲁棒**——Nano3D-v2 靠替换 box mask 外的 token 来保一致性，**mask 内部的非编辑区域一致性无法保证**，这种不一致会传播进端到端模型、拉低编辑质量。
5. 当前是级联 AR + DiT 架构，Transfusion 式深度融合值得探索。
6. 数据的量和质都还没到位。

我要补充的是：**编辑指标的 7.5 倍领先很可能部分来自「训练数据和评测的构造方式同源」**。Edit3D-Bench 的编辑对与 Nano3D-v2 的构造范式（局部区域替换、保留外部）高度契合，而 2D-lifting 类方法本来就不保证这种局部性。这个领先有多少是真正的能力、有多少是范式匹配，论文没有拆开。另外**没有报任何推理成本或模型规模**，级联 VLM + DiT 的实际部署代价不明。

### 启发与应用前景
最有借鉴价值的是**六档 caption 分级**和**四层提示词分类法**——前者让同一资产同时支持长描述精确控制和短标签快速检索，后者把「生成什么」与「怎么组合」正交化，避免了随机组合出不合理提示。这套数据设计方法可以直接搬到视频、音频等其他生成模态的合成数据构造上。

**「生成和理解都能提升编辑」**这个统一训练的实证（3D-VLM 条件优于 CLIP 条件）是这篇的核心论点，对判断「要不要做统一多模态」有参考价值：至少在 3D 编辑这个任务上，更强的指令理解能力确实转化成了更好的编辑几何。

数据规模消融显示 5000 万还在陡峭上升段，这对做 3D 生成的团队是个明确信号：**当前瓶颈仍在数据量而非架构**。

follow-up：作者列的 6 条本身就是路线图，其中「单阶段几何表示」和「纹理-几何联合建模」是最有价值的两条。代码已开源。

---

## 12. AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling
**👍 79** · 🏛 南京大学 / 南洋理工大学 / 帝国理工学院 · [arXiv:2608.02602](https://arxiv.org/abs/2608.02602) · [GitHub](https://github.com/fyv587/AURORA-LM) · [项目页](https://aurora-lm-project.github.io/)

### 问题与动机
图像、视频、音频都在连续隐空间里建模，**只有语言仍然是离散 token 的孤岛**。这个不对称还塑造了多模态系统的架构：连续观测被硬转成离散 token 序列去喂 Transformer，使离散性成了通用接口，可能压制了本该连续表达的结构。

现有连续语言模型走了两条路，各有一个共同的妥协：embedding-space 扩散（Diffusion-LM、PLAID、TEncDM）**继承了不是为「生成+解码」联合设计的表征**；latent-space 扩散（LD4LG、COSMOS、Cola-DLM）为了让扩散好学而**压缩隐空间，代价是 token 级保真度**。核心矛盾在于：紧凑平滑的 latent 好生成，但丢掉了恢复确切词、语法、局部顺序所需的区分度；保留更多信息则让 latent 先验面对更宽更复杂的分布。**生成难度和重建保真度被同一个表征瓶颈耦合在一起。**

### 方法与核心创新
AURORA-LM 的核心立场是**拒绝这个妥协**：不是简化表征去迁就生成模型，而是**保住高容量可解码的文本 latent，让扩散模型去学它的分布**。两阶段解耦：

**第一阶段——Query-based Encoder-Decoder** 构造高容量、前缀对齐的 latent 序列。因果编码器让后续 latent 位置读取逐渐变长的 token 前缀，天然诱导出适合分块从左到右生成的前缀序。解码器把 latent 前缀映射到 token logits，只用文本重建训练。训完**冻结**。

**第二阶段——Block-causal Diffusion Transformer** 用 flow matching 学 full-width latent 的分布，块间从左到右生成、块内并行去噪。为了在不牺牲解码容量的前提下学好这个更难的分布，用了三个设计：
1. **只对含噪输入通路做低秩投影（D_b=128），而预测目标仍是 full-width（D=1024）的干净 latent**——这是全文最巧的一招：模型不需要以全宽读取被污染的输入，但保留了干净预测所需的全部容量。
2. **把噪声水平分布校准到 latent 宽度**——越宽的 latent 越需要把训练质量分配到高噪声状态。
3. **self-trajectory consistency**——对齐去噪轨迹上相邻状态的干净预测，弥合「训练时独立采样噪声状态」与「推理时迭代去噪」的差距。

### 关键实验结果
**受控消融把每个设计的理由说得很清楚**：
- *latent 宽度*：把编码器输出加噪后直接解码（不去噪），窄 latent 的 token 恢复准确率很快崩溃，宽 latent 在更大的污染范围内保持精确解码；生成侧的 MAUVE 也随宽度单调上升，**D=1024 最优**。序列压缩上保留 90% 位置 MAUVE 只从 0.82 掉到 0.81，压到 70% 掉到 0.67。
- *含噪输入瓶颈*：D_b=128 的平均 MAUVE **0.808** 最高，32 太窄（0.786）、超过 128 无进一步收益（256/512/1024 分别 0.760/0.748/0.790）。
- *预测目标与损失空间*：x₀ 预测 + x₀ 空间损失 **0.815**，而 x₀ 预测 + v 空间损失崩到 **0.059**（坐标转换给干净 latent 误差加了 1/σ² 权重，把监督集中到低噪声态，与高噪声分配直接冲突）；v 预测 + x₀ 损失 0.729。
- *块大小*：Q=4 时 MAUVE 0.909，Q=64 掉到 0.631；选 Q=16（0.816）作为平衡点，把顺序生成阶段从 32 降到 8。

**系统对比（130M 的 AURORA-LM-S）**：OpenWebText 1024 token 无条件生成，Gen-PPL **23.56**（AR 85M 是 39.40，MDLM 121.36，ELF-B 24.11），MAUVE **0.890**——注意 ELF-B 的 Gen-PPL 24.11 已经很接近，但 **MAUVE 只有 0.229**，说明它生成的文本流畅却分布严重偏离人类文本。XSum 条件摘要 ROUGE-1/2/L **36.6 / 13.4 / 28.9**，全面超过 ELF-B 的 36.0 / 12.2 / 27.8。

**规模化**：1B 参数、约 1500 EFLOPs，九个语言基准宏平均 **32.6**，对比 1.8B 参数、约 2000 EFLOPs 的 Cola-DLM 的 25.1，**九项全胜**。差距最大的是 HellaSwag（18.4 vs 5.7）和 StoryCloze（54.8 vs 33.8）。全部实验在**昇腾 NPU** 上完成。

### 局限性与开放问题
论文**没有 Limitations 章节**——一篇提出新范式的工作不讨论边界，是明显缺失。我的观察：

其一，**绝对能力仍然很弱**。1B 模型九基准宏平均 32.6，其中 MMLU 22.2（四选一随机是 25，**低于随机**）、ARC-C 21.2、HellaSwag 18.4。它赢的只是同类连续扩散语言模型，而**同规模的自回归模型在这些基准上是碾压级别的**。论文只对比了 Cola-DLM 一个 baseline，回避了与 AR 的规模对等比较——这是最关键的缺失对照。

其二，**131M 那组的 AR baseline 只有 85M**，参数量对不齐，而 AURORA-LM-S 是 130M（还不算 autoencoder）。「backbone 参数量」这个口径把冻结的 encoder-decoder 排除在外，实际总参数量并不透明。

其三，**Gen-PPL 用 GPT-2 Large 当评分器**，这个指标对「生成文本是否符合 GPT-2 的偏好」比对「是否好」更敏感；MAUVE 也是分布相似度而非质量。整套评测缺少人工评估或强 LLM 判分。

其四，**块大小的质量-并行度取舍很不利**：Q=4 时 MAUVE 0.909 但需要 32 个顺序阶段，Q=16 掉到 0.816——**扩散语言模型的核心卖点是并行解码，而这里并行度每提高一档，质量就掉一截**。选择 Q=16 意味着仍有 8 个顺序阶段，相对 AR 的加速优势论文完全没有量化（没报任何吞吐或延迟数字）。

### 启发与应用前景
最值得单独记住的技术点是**「只压缩噪声输入通路、不压缩干净预测目标」**——这个非对称设计把「让分布好学」和「让解码保真」这两个此前耦合的需求解开了。同样的思路可以迁移到图像/视频 latent 扩散：VAE latent 的宽度不必因为扩散难学而被压窄，可以只在 denoiser 的输入侧做瓶颈。

第二个可迁移的是**噪声分配要随 latent 宽度校准**——宽表征需要把更多监督放在高噪声态。这在图像扩散领域已有零星证据（分辨率越高越需要 shift），这篇给出了在语言 latent 上的系统验证。

第三个是那张 x₀/v 预测 × x₀/v 损失的 2×2 表（0.815 / 0.059 / 0.729 / 0.653）——**在高维表征空间里直接回归干净数据显著优于速度预测**，而且损失空间的选择比预测目标本身影响更大。

follow-up：最该做的是**与同规模 AR 的对等对比 + 吞吐量测量**——如果连续扩散语言模型不能在「同参数、同算力、更快」上占住至少一条，这条路线的价值就只是概念上的。代码已开源。

---

## 13. N₀-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens
**👍 76** · 🏛 NeoteAI / 复旦大学 · [arXiv:2607.23782](https://arxiv.org/abs/2607.23782) · [GitHub](https://github.com/neoteai/N0-VTLA) · [项目页](https://research.neoteai.com/n0-vtla/)

### 问题与动机
VLA 模型在抓取、搬运这类任务上已经不错，但**接触密集（contact-rich）的精细操作**——插插头、拔钥匙、插 HDMI、叠碗——依然很差，因为纯视觉看不见接触力。加触觉的直觉做法是把触觉当成「又一路观测」塞进视觉-语言前缀，但这有两个问题：触觉信号高频且局部，混进前缀会稀释语义；而且**反应式的触觉只能在接触发生后纠正，来不及**。

### 方法与核心创新
最核心的设计判断是：**把触觉当作预测目标而非观测输入**。

架构上，在 PaliGemma + flow-matching action expert 的 π₀.₅ 骨干之外，只加一条**latent tactile pathway**：冻结的触觉编码器（DINOv2）把每根手指的**接触差分图像**转成 token，一个小预测器结合场景和指令，把这些 token 蒸馏成 latent tactile token **z**——z 估计的是**未来 action chunk 内预期的净接触变化**。action expert 直接条件在 z 上，而**触觉永远不进入视觉-语言前缀**。

训练是三阶段递进（每阶段从上阶段 checkpoint 出发，可训练面只在新接口被 grounding 之后才扩大）：
- **Stage 1 grounding**：冻结整个基础策略，只训预测器、触觉投影和一个轻量重建头。用对称 InfoNCE 把 z 拉向未来触觉目标 z*，同时用 ℓ1 重建头解码出粗粒度的未来接触变化场。
- **Stage 2**：冻结触觉感知栈、屏蔽视觉-语言通路，把 latent 与 action expert 对齐。
- **Stage 3**：全策略联合训练。

另有 **ALTER**（Advantage Labeling from Trajectory Events and Relative Progress）做部署数据上的离线策略改进：干净演示提供密集进度监督（阶段边界用触觉接触变化、末端运动学、夹爪状态、视觉事件线索定位），不完美的部署轨迹提供稀疏的前后偏好（来自**触觉检测到的掉落事件**和记录下来的人工干预）。二者联合训练一个成对进度模型，转成二值 advantage 标签做 advantage-conditioned 离线 RL。

跨本体统一在一个 32 维状态/动作容器里（前 20 维每臂 10 维：3 维位置 + 6 维 rot6d 旋转 + 1 维夹爪），单臂数据补零，单双臂共存于一个固定宽度目标下。

### 关键实验结果
**真机 NeoReal 九任务**：**九项全胜**，成功率均值 47.2% vs π₀.₅ 的 29.4%；100 分进度分 56.8 vs 42.3。ACT 一项都没完成，平均只有 10.2 进度分。差距最明显的是 Socket Plugging（精密插座插入）：**85% vs 60%**，且成功 rollout 在日光变化下仍稳定，还能从失败插入中恢复。

**仿真 20 任务**：总均值 **63.8% vs π₀.₅ 的 44.0%**（+19.8 点）。拆开看更有信息量——UniVTAC 八任务 83.1%（最强外部基线 InternVLA-A1 是 67.1%），NeoSim 十二任务 50.8%（π₀.₅ 45.8%）。**单双臂差距极大**：四个单臂任务 73.8%，八个双臂任务只有 39.4%，而专用基线在双臂上直接崩到个位数（InternVLA-A1 仅 1.0%）。

**表征分析是这篇最扎实的部分**，它排除了两个平凡解释：
- 378 个留出查询中，预测的 z 在约 32 个候选池里 top-1 检索到匹配的未来触觉目标 **92.3%**（随机 3.2%）；
- **反自相关对照**：只用当前触觉编码 g（预测器自己的输入）排序只有 57% top-1，128 候选池里 40%，而预测器有 81%——**说明 z 携带了超出输入的未来接触信息，且池子越难差距越大**；
- **反视觉捷径对照**：交换触觉输入让 z 移动约 0.9（中心化余弦距离），交换 RGB 视图和指令只移动约 0.2，触觉/视觉-语言敏感度比在 Stage 1 末为 4.3，联合训练后仍有 1.4。

**ALTER**：三个长程真机任务达到 75%–95% 成功率。

### 局限性与开放问题
论文的 Future Work 只提了两条（预测式 latent 是更广设计空间的一个点；ALTER 可扩展到更多任务），**没有独立 Limitations 章节**。我的观察：

其一，**双臂是明显短板**。39.4% 的双臂均值里，Insert Screw 24%、Unstack Bowl 17%、Unstack Cup 10%——这些正是接触最密集的任务。更尴尬的是，在 Stack Plates（94 vs 96）和 Stack Bowls（96 vs 93）这两项上 N₀-VTLA 与 π₀.₅ 基本打平甚至略输，**触觉通路在双臂协调任务上没有体现价值**。

其二，**UniVTAC 上的 Insert HDMI 只有 25%**，而 Xiaomi-Robotics-0 是 69%——在一个明确的接触密集插入任务上被专用基线打了近 3 倍。均值 83.1% 掩盖了这个单点失败。

其三，**硬件依赖是根本约束**：整套方法建立在自研视觉-触觉传感器上，每根参与手指都要装。NeoData 的规模、组成、传感器规格全部推给了「companion data report」，**这篇论文本身无法独立评估数据规模**。复现门槛极高。

其四，**ALTER 的 75%–95% 只在三个任务上报了区间，没给基线对比**——不知道这个数字相对不做 ALTER 提升了多少。

### 启发与应用前景
最有价值的设计原则是**「把新模态当预测目标而非观测输入」**。这个思路可以推广到任何「高频、局部、且未来值比当前值更有用」的模态——力矩、音频、热成像、甚至网络延迟。相比把新模态 token 拼进前缀，预测式 latent 既不稀释语义，又天然带前瞻性。

**三阶段渐进解冻**（先 grounding 新接口、再对齐、最后联合）也是可复用的配方，专门针对「给预训练大模型加新模态而不破坏它」这个普遍问题。

**表征分析的两个对照实验设计值得抄**：反自相关（只用输入本身能达到多少）和反捷径（扰动各模态看表征移动多少）——任何声称「学到了跨模态预测能力」的工作都该做这两个对照，否则无法排除模型只是在复读输入或走视觉捷径。

follow-up：双臂协调是明确的空白；把 ALTER 的「触觉检测掉落事件」这个自动失败标注思路推广到其他可自动检测的失败模式，是低成本高回报的方向。代码已开源，配套的 N₀-TWAM 见 [27]。

---

## 14. Interpretable MEG Decoding of Perceived Speech: Cortical Sources and the Stimulus Features That Drive Retrieval
**👍 68** · 🏛 HSE 大学 / ITMO 大学 · [arXiv:2608.01481](https://arxiv.org/abs/2608.01481) · [GitHub](https://github.com/ivsemenkov/LISA) · [项目页](https://ivsemenkov.github.io/LISA/)

### 问题与动机
用 CLIP 式目标训练深网络、把 MEG 信号对齐到 wav2vec 2.0 音频嵌入，已经能从非侵入脑磁图中检索出感知到的语音片段。但这类工作有个根本缺陷：**权重无法转译成经典电生理学的概念**。Défossez 等人的空间滤波器原则上能映射到源拓扑图，但**这些源的动态特性（时间频谱）拿不到**；语音流中究竟是哪些属性驱动了解码，同样不清楚。

换句话说，现有神经解码工作证明了「能解出来」，但没回答「解的是脑子的哪里、用的是语音的什么」——而这恰恰是神经科学关心的。

### 方法与核心创新
构造一个**前端受测量物理和源生理双重约束**的解码器（命名 LISA）：
1. 把 Défossez 的 2D 傅里叶空间注意力换成**球谐函数参数化**的层（L=24，每个虚拟通道 576 个实基函数）——这是关键，因为它让空间投影受 MEG 传感器阵列的**实际三维几何**约束，而不是拍平成 2D 网格；
2. 把被试特定表征从 270 个分支**降到 K=25**；
3. **加一层可训练时域滤波器**（150 ms 核），使每个分支在时间上也匹配一个神经源。

理论基础来自 Petrosyan 等人的框架：在因子化的空间-时间架构里，空间和时间滤波器是**同时自适应**的，所以要正确解释权重，必须考虑二者的相互依赖——此前的解释尝试都忽略了这一点。每个分支被视为一个匹配到特定神经源（有其空间位置和二阶动态特性）的空间-时间匹配滤波器。

还有一个方法论上的重要处理：**训练前先用 ICA 去掉眼动和心电成分**。作者说得很直白——这不是常规卫生操作，而是因为「以检索损失为目标的网络是机会主义的：它没有理由偏好皮层信号而非同样能预测目标的外周信号，会挑最容易提取的那个，这就是最朴素的捷径学习」。眼动已知会跟踪语言结构和注意语音，心律会随叙事强度变化——不去掉，就分不清解出来的是大脑还是眼球。

最后是 **paired MEG occlusion**：把带特征标记的语音片段替换成匹配的对照片段，测 19 个声学/音位/语言学特征（静音、七类音素、词起始、伪词、随机词表、surprisal、预测熵、词频、响度、声学起始强度等）各自对检索的贡献。

### 关键实验结果
清理后的 MEG-MASC 数据集（27 名被试、49 个约一小时的记录）上，1005 候选中 **Top-1 准确率 39.75 ± 0.34%**（六个训练解的均值），**解码器可训练参数少约 20 倍**。

参数量对比很说明问题：LISA 主模型 486,619 参数、Top-1 40.01% / Top-10 70.60%；Défossez 等人 9,565,054 参数、41.30% / 70.70%（但候选池是 1363 而非 1005，不完全可比）。更关键的是那个**同容量对照**——把 LISA 设成 K=270、五个卷积块（7,210,224 参数，最接近 Défossez 的配置），Top-1 反而掉到 **36.41%**。也就是说**参数少不是妥协，是增益**：K=25 的紧凑分支空间比 K=270 好 3.6 个点。

架构消融的排序也清楚：**去掉被试条件化空间映射的损失最大**（多被试 MEG 里同一皮层活动在不同人身上的传感器拓扑不同，纯共享投影要同时解决对齐和提特征两件事，太受限）；去掉空间注意力次之；**3D 球谐注意力优于 2D 注意力**（证明约束于传感器物理布局有实质价值，而不只是「有个注意力层」）；把 15 采样点时域滤波换成可学习的单采样点滤波会降性能，说明分支确实受益于局部时域变换。

可解释性结果：权重能映射到源空间，**恢复出与经典语音感知网络一致的发生源**，且**定位到左侧的分支携带更高频的节律成分，右侧没有**——这是一个符合语言偏侧化先验的独立验证。

### 局限性与开放问题
论文没有集中的 Limitations 段落，但在 Discussion 里对预处理选择做了充分论证。我的观察：

其一，**数据集只有一个（MEG-MASC，27 人）**，且是英语被试听虚构故事。结论能否推广到其他语言、其他任务（产出语音而非感知语音）完全未知。

其二，**39.75% 的绝对值放在 1005 候选里看起来很高，但这是检索而非重建**——它证明的是「MEG 中含有能区分 1005 个 3 秒片段的信息」，离「读出说了什么」还很远。而且论文自己注明**测试片段没有对齐到词起始**（与 Défossez 不同），所以候选集比对方更无结构，两个 Top-1 数字不能直接比。

其三，**occlusion 分析的因果解释有边界**：替换 MEG 片段测的是「解码器是否用了该特征关联的神经信息」，不等于「大脑用该特征编码语音」。这两件事在论文里的措辞是克制的，但容易被读者过度解读。

其四，**主结果基于 seed 42**，虽然附录 B 用 seed 43–47 重复了 occlusion 分析验证稳健性，但架构消融和解释性分析仍是单 seed。

### 启发与应用前景
这篇最大的方法论价值不在神经科学，而在**「可解释性应该来自架构约束，而非事后归因」**这个立场。把物理（传感器几何 → 球谐参数化）和生理（源的时空匹配滤波）编码进网络结构，使得训练完的权重**天然可以翻译成领域概念**（源拓扑图、激活时间序列、功率谱）。这比训一个黑箱再上 SHAP/attention 可视化靠谱得多，而且**代价是负的**——参数少 20 倍且性能不降。

**「去掉眼动和心电是防捷径学习而非数据清洗」**这个论述值得任何做生理信号解码的人记住：只要有一条外周通路能同样预测目标，优化器就会走它。同类问题在医疗影像（学到扫描仪型号）、语音（学到录音环境）里都存在。

**paired occlusion 的对照设计**（用匹配片段替换而非置零/加噪）也比常见的 ablation 干净——置零会引入分布外输入，替换保持了信号统计特性。

follow-up：把这套架构约束思路用到 EEG（便宜得多、几何更规则）；或者验证跨语言、跨任务的迁移。代码和清理用的 ICA 成分都已开放（OSF）。

---

## 15. OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models
**👍 68** · 🏛 香港大学 / 复旦大学 / 南京大学 / 西安交通大学 · [arXiv:2607.28609](https://arxiv.org/abs/2607.28609) · [GitHub](https://github.com/OS-Copilot/OSReward) · [项目页](https://os-copilot.github.io/OSReward-Home/)

### 问题与动机
计算机使用 agent（CUA）的评测、数据筛选、RL 训练，每一环都要判断「这条轨迹到底完成任务了没有」。人写 verifier 一个任务一个、真实场景里很多任务根本没有可检查的结果；人工标注更不可能规模化。于是整个领域默认转向 **VLM 当裁判**。

但一个根本问题一直没人系统检验：**这些 VLM 裁判到底可不可靠？**

已有的零星研究停留在单一平台、复用现成 benchmark 的指令和轨迹，问题不小：复用轨迹继承了产出它们的 rollout 设置（agent 骨干、预算），金标签可能来自不完美的 verifier，有些指令模糊到根本不存在确定判决。而平台差异极大——动作空间不同、应用不同、运行长度从移动端十来步到桌面端上百步。

### 方法与核心创新
OSReward 从零构建：
- **环境准备**：桌面和移动端远超基准镜像的标配——每台机器装上真实用户会有的日常应用（Ubuntu 覆盖 VS Code、Blender、GIMP、LibreOffice、Wireshark 等 40+ 应用），并**丰富初始化每个任务**：用户配置、待编辑的真实文件、预置的应用数据库、干扰内容。目的很明确：让轨迹「有值得判断的内容」——有真实状态可失败、成功必须改变环境而非口头宣称完成、有具体状态供机器 verifier 检查。Web 任务跑在**实时网站**上。
- **指令编写**：标注员探索环境后现场编写，故意包含超出规则 verifier 能力的开放式任务；每条候选由**非作者的标注员交叉筛查**，剔除模糊、无依据、无法回答的。约 1500 条候选，约 800 条进入下一阶段。
- **轨迹收集**：每条指令由 1–3 个 agent 骨干执行，横跨 Claude、Gemini、Kimi、Qwen 四个家族——**不同骨干的动作习惯、思考冗余度、失败模式各异**，避免 benchmark 退化成单一 agent 的接口风格。自动预筛掉严重采集问题（反爬拦截、网络故障、执行冻结），确保 benchmark 里的 fail 反映的是 agent 真的失败了。
- 再派生 **OSReward-Hard**（浓缩真正困难的样例）和 **OSReward-Multi**（细粒度的效率与意图对齐评分，各用三级 rubric）。

在此之上构建 **OS-Shepherd-100K** 开放语料（32 万判断实例，8 个来源），训出 **OS-Shepherd 9B / 35B**：SFT 用 9.66 万经一致性过滤、平台和标签均衡的样本（截图设置故意混合：last-5 帧 45.1%、first-1+last-2 26.0%、last-3 18.9% 等，让裁判耐受输入形式变化）；RL 阶段专门针对残余的**假成功**错误——挖约 3.1K 贪心解码失败但采样能对的样本做 GRPO。

### 关键实验结果
**27 个裁判的评测是全文最有价值的产出**，暴露了一个系统性问题：**所有模型都有宽容偏差（leniency bias），把失败当成功**。

看 OSReward 主集：Claude-Opus-4-8 准确率 89.7%（sRec 91.1 / fRec 88.9，比较均衡），GPT-5.5 89.5%。但便宜模型的失衡触目惊心——**Doubao-2.0-Lite 成功召回 98.5% 而失败召回只有 72.1%**，GPT-5-nano 97.0% vs 71.1%，Gemini-2.5-Flash 95.5% vs 74.0%。**它们几乎把所有轨迹都判成成功**，准确率看着还行只是因为成功样本多。

**OSReward-Hard 上全线崩溃**：最好的 Claude-Opus-4-8 只有 **69.7%**，GPT-5.5 67.3%，Gemini-3.1-Pro 61.6%。而失衡更极端：**Claude-Sonnet-4-6 成功召回 90.7% 但失败召回仅 45.5%**，Qwen3.5-397B 是 91.9% vs 43.9%，Doubao-2.0-Lite 是 96.1% vs **24.3%**——在困难集上，四分之三的失败轨迹被判成了成功。用这样的裁判做 RL 奖励，等于在系统性奖励失败。

**OS-Shepherd 的价值在于修正偏差而非单纯提分**：Qwen3.5-9B 基座在主集上 76.7%（sRec 98.9 / fRec 59.9，典型的极端宽容），训练后 OS-Shepherd-9B 到 **86.1%（86.6 / 86.0，几乎完全平衡）**。Hard 集上从 39.4%（97.7 / **14.1**）到 60.2%（66.3 / 57.6）。35B 版本主集 85.6%、Hard 集 62.7%——**已超过 Gemini-3.1-Pro 的 61.6%**，而成本据称比前沿模型低 **30–60 倍**。

### 局限性与开放问题
论文没有集中的 Limitations 章节。我看到的问题：

其一，**OS-Shepherd-9B 在主集上 86.1% 其实不如 35B 的 85.6% 差多少，但 Hard 集差 2.5 点**——规模收益很小，说明瓶颈不在模型容量而在数据或任务本身的困难度。作者自己也说 SFT 后「假成功样本明显比普通样本更难拟合，案例检查归因于真实的判断困难而非标签噪声」。

其二，**训练数据的金标签来自「ensemble 一致性过滤」而非人工**——训练语料 32 万实例不可能全人工，用 Gemini-3.1-Pro 等裁判的一致响应作为监督。这意味着 **OS-Shepherd 的能力上界被这些教师裁判封住了**，它学到的是「多数裁判会怎么判」，而多数裁判恰恰有宽容偏差。RL 阶段针对假成功做了矫正，但训练信号本身仍有这个天花板。

其三，**alignment 指标实际只有两级**——440 条轨迹里只有 1 条拿了 0 分（还被当作安全问题移除了），59 条 0.5 分、380 条 1.0 分。这个分布下 alignment 这一维的区分度非常有限。

其四，**Hard 集只有几百条**（从 sRec/fRec 的粒度推断），最好的模型也只有 69.7%，**接近但仍高于随机的 50%**——这个集合的难度设定使得所有模型都聚在 45–70% 区间，区分度也有限。

### 启发与应用前景
最重要的实践结论：**如果你在用 VLM 判分做 agent 的 RL 或数据筛选，请立刻测一下你的裁判的失败召回率**。准确率是个骗人的指标——在成功样本占多数的数据上，一个「全判成功」的裁判也能有 70%+ 准确率，但它提供的奖励信号是零信息甚至负信息。这篇给出的 sRec/fRec/BalAcc 三件套应该成为裁判评测的标配。

第二个实践结论：**便宜模型不能当裁判**。Doubao-2.0-Lite、GPT-5-nano 这类在 Hard 集上失败召回只有 24–46%，而它们恰恰是大规模数据筛选时最常被选中的（因为便宜）。这可能是当前很多 CUA 训练数据质量问题的根源。

方法上，**「用多个不同骨干 agent 产出轨迹」**这个 benchmark 设计原则值得推广——单一 agent 产出的轨迹会让裁判过拟合到那个 agent 的思考风格和失败模式。

**RL 专攻假成功**这个做法（挖贪心失败但采样能对的样本）是针对特定错误模式的定向修正，比笼统的 RLHF 精准，可以推广到任何有已知系统性偏差的判别任务。

follow-up：最该做的是**摆脱教师裁判的天花板**——用少量人工金标签做 RL 而非用 ensemble 一致性做 SFT，看能否突破 70% 这道坎。语料和模型都已开源。

---

## 16. InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis
**👍 65** · 🏛 浙江大学 / Udeer.ai / 深圳大学 · [arXiv:2608.02437](https://arxiv.org/abs/2608.02437) · [GitHub](https://github.com/zju3dv/InfiniSplat) · [项目页](https://zju3dv.github.io/InfiniSplat/)

### 问题与动机
单图前馈 3D Gaussian Splatting 想从一张图直接得到可渲染的 3D 场景，绕开多视图采集和逐场景优化。但现有方法几乎都被**像素对齐表征**卡住：高斯基元从固定的图像网格位置预测出来。这样做在邻近视角渲染尚可，但基元与真实场景表面**只是弱耦合**——高斯的分布由图像网格的离散化决定，而不是由场景表面的几何决定。视角一大，网格离散化造成的散乱基元就暴露成裂缝和空洞。

### 方法与核心创新
把表征从**像素对齐**推向**表面对齐**，拆成两个子问题分别解决：

**(1) 几何引导采样**决定「高斯该放在哪」——用冻结的单目深度模型（DepthPro）预测稠密深度和相机内参，反投影得到每个 patch 的 3D 表面面积，**按面积比例分配支撑点**（surface area 大的地方多放），每个支撑点初始化为一个 base Gaussian。这一步把高斯布局锚定在几何上，而不是图像网格上。

**(2) 隐式高斯解码**决定「在不规则位置上怎么预测参数」——一个共享 MLP 在支撑点坐标处**双线性查询** DINO 语义特征图和 CNN 纹理特征图，经门控机制融合后，预测位置、尺度、旋转、颜色、不透明度的**有界偏移量**。

关键在于「查询式」这个设计：因为支撑点不再落在像素中心，传统的 DPT 式稠密预测解码器用不了，必须用可以在任意连续坐标处求值的隐式解码器。

另有 LiDAR 条件变体：额外接收 1500 个稀疏深度采样作为 prompt，用 InfiniDepth-Metric 得到更强的几何条件。

### 关键实验结果
**全部在 Hypersim 室内合成数据上训练，四个真实数据集零样本评测**（ETH3D、ScanNet++、Tanks-and-Temples、DL3DV，无一用于训练）。

RGB-only：平均 PSNR **20.394** / SSIM **0.806** / LPIPS **0.277**，对比最强基线 SHARP 的 18.475 / 0.758 / 0.299。**DL3DV 上差距最大**——21.685 vs 18.305（+3.38 dB），SSIM 0.772 vs 0.652。

LiDAR 条件下更夸张：平均 PSNR **22.548**，而 ADGaussian 只有 12.249——**差 10 dB 以上**，几乎是「能用 vs 不能用」的区别。ETH3D 上 25.880 / 0.946 / 0.151。稀疏深度 prompt 的鲁棒性也测了：加 5% 乘性噪声，PSNR 从 25.880 只掉到 23.595。

**消融把每个组件的作用切得很清楚**（ETH3D / ScanNet++ PSNR）：
- **去掉 DINO 分支最致命**：11.465 / 12.501（掉 9 dB 以上），模型失去主要语义骨干、无法泛化，产生大洞和结构缺失；
- 去掉隐式解码器换成像素对齐 DPT：18.678 / 20.811；
- 去掉学习式更新（只渲染几何初始化的 base 高斯）：18.819 / 19.861——说明**模型不是简单渲染一张抬升的深度图**；
- 去掉 CNN 分支：18.646 / 20.368（低层外观和边界细节变糊）；
- 去掉几何引导采样（换成像素对齐支撑点）：19.934 / 21.576——这是**最小的一项消融**（只掉 0.6 / 0.66 dB）。

支撑点预算：0.5M → 18.779、1.0M → 21.180、1.5M → 22.240、2.0M → 22.237（**已饱和**）。1.5M 时推理 0.598 秒、渲染 0.009 秒。

### 局限性与开放问题
论文有专门的 Limitations 章节且配了失败案例图，写得诚实：(1) **单视图歧义**——看不到遮挡区域，目标视角暴露大片未见区域时会产生不完整几何、拉伸结构或幻觉外观；(2) **继承深度先验的错误**——几何引导采样依赖预测深度构造支撑分布，深度先验在反射面、透明物体、细结构、无纹理区域失效时，支撑点就被放在错误的几何脚手架上；(3) 极端视角外推、非朗伯表面、重复细结构、强深度不连续仍然困难。

我要补充一个论文没点破的问题：**消融数据其实削弱了论文的核心叙事**。标题和方法都在强调「从像素对齐走向表面对齐」，但几何引导采样这个「表面对齐」的核心组件，消融掉只损失 0.6–0.66 dB，是所有组件里最小的；而 DINO 特征掉了 9 dB。换句话说，**这篇的性能主要来自「用了强语义特征 + 隐式解码器」，而不是来自「表面对齐」这个概念本身**。表面对齐带来的更多是定性改善（裂缝少、法线干净），这在图上看得见，但定量贡献远小于标题的分量。

另外，**训练数据只有 Hypersim（室内合成）**，虽然零样本泛化到真实场景是亮点，但也意味着结果的上限被这一个数据源的分布决定。而 Tanks-and-Temples 上所有方法的 PSNR 都只有 15–17 dB，说明大基线室外场景仍然远未解决。

### 启发与应用前景
最实用的洞见是那个**支撑点预算饱和曲线**：1.5M 到 2.0M 完全没有增益。做前馈 3DGS 的团队可以直接用这个数字定预算，不必盲目堆基元。

**「查询式隐式解码器解耦了预测位置和像素网格」**这个模式值得推广——任何「输出要放在不规则位置」的稠密预测任务（点云补全、稀疏体素预测、非均匀采样的神经场）都可以用「特征图 + 连续坐标双线性查询 + 有界偏移预测」这套组合，而不是被卷积的规则网格绑死。

**LiDAR 变体的 10 dB 领先**是个很强的工程信号：在自动驾驶、机器人这些本来就有稀疏深度的场景，单图 3DGS 的实用性比纯 RGB 高一个量级。而 5% 噪声只掉 2.3 dB 说明它对真实 LiDAR 的噪声容忍度够用。

follow-up：论文自己指出的方向是结合生成先验补遮挡、不确定性感知的几何估计，或引入稀疏多视图。我更看好的是**把「表面对齐」这个概念真正做实**——当前的几何引导采样只是按面积配额，如果能引入曲率、边界、语义显著性等更强的表面结构信号，那 0.6 dB 的增益可能还有很大空间。代码已开源。

---

## 17. ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment
**👍 64** · 🏛 上海交通大学 · [arXiv:2608.05102](https://arxiv.org/abs/2608.05102)

### 问题与动机
长程搜索 agent 要走几十上百步去搜索、检索、验证、整合证据。但现有训练方法在 SFT 和 RL 阶段**都把轨迹内所有步骤同等对待**——成功轨迹的每一步都被强化，失败轨迹的每一步都被惩罚。

问题在于这个假设是错的。论文对 8.5K 条轨迹做了奖励分布分析，结果很直接：**成功轨迹里约 4% 的步骤是低质量的**（奖励低于 1.0），而**失败轨迹里近 10% 的步骤奖励高于 1.0**——它们确实发现或验证了有用线索，只是最终答案错了。轨迹级监督会**在成功轨迹里强化错误动作，在失败轨迹里惩罚有用动作**，等于往两个方向同时污染梯度。

### 方法与核心创新
**ABC（Answer-Backtracked Credit Assignment）**的核心思路是：既然有验证过的标准答案，就**从答案倒推回去**，恢复出解题必需的中间线索，再拿这些线索去量每一步。

两阶段：
1. **Answer-Backtracked Clue Recovery**——给定一个可能很晦涩的查询和它的标准答案，从答案回溯，恢复出到达答案所需的中间证据线索集合。这一步用 DeepSeek-V4-Flash 完成。
2. **Clue-Anchored Step Scoring**——拿每一步的搜索动作去对照这个线索集合打分，把稀疏的二值结果监督转成**稠密的步级奖励**。

然后两条训练路径：
- **ABC-SFT**：用 sigmoid 把步级奖励映射成 loss 权重 w(r_t) = σ(α(r_t − β))，实现上是 w(r) = 2σ(2(r−1))，让中性奖励 1.0 映射到权重 1.0。高分步贡献更强梯度，低分步几乎不贡献。
- **ABC-GRPO**：步级分数直接当奖励，组内归一化后计算**折扣步级 advantage** A_{i,t} = Σ_k γ^{k−t} R̂_{i,k}（γ=0.25），该 advantage 赋给该步的所有策略生成 token，工具返回被屏蔽。

**成功和失败轨迹全部保留**——这是整个设计的关键，失败轨迹里的好步骤才有机会被强化。

### 关键实验结果
基座 Qwen3.5-4B，**只用 8.5K 条轨迹**（5.5K 正确 + 3.0K 错误）。

**主结果**：BrowseComp **37.3%**（无上下文管理）/ **55.3%**（有上下文管理），BrowseComp-ZH 39.1% / 52.9%，xbench-2505 **77.0%**，xbench-2510 46.0%，GAIA-text **81.6%**。

对比很有说服力：同为 4B 的 DR-Venus 是 29.1%/37.7%，AgentCPM-Explore 24.1%/29.1%，QUEST-4B 40.0%。而**跨规模看**——ABSeeker 的 xbench-2505 77.0% 和 GAIA-text 81.6% **超过了所有报告的 30B agent**（MiroThinker-1.7-mini 80.3、Tongyi-DeepResearch 75.0/70.9、OpenSeeker 74.0）；BrowseComp 的 55.3% 也超过 RedSearcher（57.4 略高）之外的多个 30B 系统，比 DeepMiner-32B 的 33.5% 高出一大截。

**消融把两个阶段的贡献分开了**（均无上下文管理）：
- 标准 SFT → ABC-SFT：BrowseComp 28.5 → 30.8，xbench-2510 **27.0 → 35.0**，GAIA-text 66.0 → 72.8；
- 标准 GRPO → ABC-GRPO：33.5 → 37.3、41.0 → **46.0**、77.7 → 81.6。

两个阶段各自都有增益，且 RL 阶段的增益更大。**只在 BrowseComp 风格问题上训练却能泛化到 xbench 和 GAIA**，是跨基准泛化的证据。

### 局限性与开放问题
作者的 Future Work 承认**受算力限制只做了 4B**，规模化未验证。我另外看到几个问题：

其一，**「有/无上下文管理」这个变量把主表搞得很不干净**。ABSeeker 报的是 37.3\*/55.3——18 个点的差距全部来自上下文管理这个与 ABC 方法无关的组件。而对比的 4B 基线只报了带 \* 的数字（无上下文管理）。**用 55.3 去和 30B agent 比、用 37.3 去和 4B 基线比**，这个选择性对齐让「4B 打平 30B」的结论打了折扣。更公平的读法是：ABC 方法本身带来 33.5 → 37.3（消融表里的 +3.8 点），上下文管理带来 +18 点。

其二，**方法依赖一个强 LLM 做线索恢复和步骤打分**（DeepSeek-V4-Flash）。这既是成本，也是能力上界——ABC 的奖励质量被这个打分器封顶。论文没有做「换一个更弱/更强的打分器」的敏感性分析。

其三，**只适用于「答案可回溯出中间线索」的任务**。作者自己也把这一点写进了 future work。对于开放式写作、多目标优化这类没有唯一答案链的任务，整套方法不成立。

其四，**RL 训练只用了 1000 个问题**（200 个 <100 轮、800 个 ≥100 轮），每题 8 rollout。这个数据量很小，加上评测虽然跑了三次取平均但没报方差，37.3 vs 33.5 这个 3.8 点的差距需要看波动区间才好判断。

### 启发与应用前景
最核心的可迁移洞见是那组奖励分布数字：**失败轨迹里有 10% 的好步骤，成功轨迹里有 4% 的坏步骤**。任何做长程 agent RL 的人都该先测一下自己数据里的这个比例——如果类似，那纯轨迹级 advantage 就是在系统性地污染梯度。

**「从已验证的答案倒推中间线索，再拿线索当过程奖励的锚」**这个配方，本质上是一种**免训练的过程奖励模型**：不需要训 PRM，也不需要 MCTS rollout，只需要一个答案和一个能倒推的 LLM。这比 Math-Shepherd 那类靠 rollout 估计步骤价值的方法便宜得多，适用范围是「有 ground truth 且解题路径可分解」的所有任务——多跳 QA、代码调试、数据分析都能套。

工程上，**同时保留成功和失败轨迹**这个做法几乎零成本却常被忽略。多数 SFT 流水线直接丢掉失败 rollout，等于扔掉了 10% 的有效监督。

follow-up：把 ABC 推到更大规模验证；或者做打分器的能力消融，看这套方法对 judge 质量有多敏感。

---

## 18. WorldClaw: Agentic 3D Open-World Generation at Scale
**👍 61** · 🏛 腾讯混元 · [arXiv:2608.05248](https://arxiv.org/abs/2608.05248) · [项目页](https://tencent-hunyuan.github.io/Hunyuan3D-WorldClaw/)

### 问题与动机
从开放式文本生成大规模、可自由探索的 3D 世界，难点在于要**同时**满足三件常常互相冲突的事：全局空间连贯性（地形要连成一体、区域组织要合理）、丰富的局部内容、以及**显式可编辑可复用的资产**（不是一整块烘焙好的网格或高斯点云，而是能拿进 Blender/游戏引擎单独改的实例）。

一次性整体生成（holistic generation）能保连贯但产出不可编辑；逐个物体生成能保可编辑但难保全局一致。

### 方法与核心创新
WorldClaw 是**全 agentic 的粗到细框架**，三阶段串行，各由专门 agent 执行、通过共享的结构化中间表示通信：

**(1) 意图分析与规划**——刻意拆成两个 agent：**意图分析 agent 只提取和规范化用户明确表达的约束**（场景类型、主题风格、关键区域和物体、空间关系、偏好），**不引入新内容、不补全未指定属性**；**场景规划 agent** 才负责消歧和按预定义 schema 补全下游所需的信息。这个「显式约束提取」与「缺失信息补全」的分离是为了保住原始用户意图不被规划阶段的想象覆盖。输出结构化场景规格 P = (区域, 地形规格, 物体规格)。

**(2) 全局地形生成**——从语义布局构建全局连贯的地形基座：可复用资产 + 生成式或程序化材质 + **区域感知的高度场**。

**(3) 区域物体生成与放置**——对需要细节的区域生成地形条件下的构图，重建**可编辑的带纹理网格**，并恢复它们在地形上的摆放；再由**基于渲染的 agent** 迭代精修地形、物体、外观和接触关系。

最终产物是显式地形 + 独立可管理的纹理网格，支持自由视点探索、实例级编辑复用、以及接入常规渲染/动画/游戏引擎流程。

### 关键实验结果
**这是本周论文里实验最薄弱的一篇——正文没有任何定量表格。**

论文展示了 11 个以上的定性场景：热带海盗据点岛屿、有部落聚落的河谷峡谷、沙漠战场、《红警》风格的雪山谷地、中世纪村庄、雪地河畔村庄、有龙环绕的沙漠营地、日式小镇岛屿、火山魔窟、宝石矿场、霍比特村谷地。每个场景配全局环绕视图、区域特写、局部漫游视图，以及对应的**实例分割图、深度图、法线图渲染**——后三者是用来证明底层确实是显式可编辑表示，而不是一整块烘焙几何。

论文提到「与代表性文本驱动 3D 场景生成方法做了对比」，但**正文抓取到的内容里没有任何量化对比数据、用户研究或指标**。

### 局限性与开放问题
论文的 Limitations 章节写得相当坦率，三条都是真问题：

1. **强依赖底层模型**——流水线由异构模型协作（LLM 做规划和程序化设计、图像生成模型做语义布局和区域构图、3D 生成模型重建资产）。作者直言：**当前开源语言模型经常无法生成既可执行又符合用户要求的程序化地形和材质；开源图像生成模型经常产不出可用的语义布局图，或在物体图像生成和提取时保不住外观和姿态**。最终视觉质量还直接受 3D 生成骨干封顶。因此**「在现阶段完整验证这套解耦流水线仍然需要 Claude Opus 4.8、GPT-Image-2、Hunyuan3D 这个级别的模型」**——这句话等于承认整套方法目前是闭源模型专属。

2. **代码生成的稳定性风险**——地形构建、程序化材质、资产放置、局部精修多个阶段依赖 LLM 生成程序。尺度估计、数值参数、节点连接上的错误会直接在 3D 场景里表现为地貌不一致、材质效果失真、布局偏离意图，往往需要多轮「渲染-检查-精修」迭代。对 Blender 这类专业软件尤其明显——**艺术家用复杂节点图能构建的材质效果，在生成的程序里往往退化成相当简单的近似**。

3. **效率开销**——为了实例级可编辑性，每个物体单独生成重建，再加多轮 agentic 精修，长程流水线带来可观的推理延迟和算力成本，且随物体数量和迭代轮数增长。**对于整体生成方法几步就能合成的简单场景，这套流水线不必要地冗长低效**。

我要补的最大问题就是**定量评测的缺失**。一篇声称「at Scale」的系统论文，既没有报生成一个世界的耗时、成本、资产数量，也没有和基线的量化对比或用户研究。「diverse open-world prompts 上表现好」全靠图片支撑，读者无法判断失败率、一致性、可编辑性到底达到什么水平。

### 启发与应用前景
最值得借鉴的是**「意图分析」与「场景规划」的职责分离**这个 agent 设计模式——前者只做无损的约束提取，后者才做有创造性的补全。这个分离能防止一个常见问题：规划 agent 一边补全一边把用户的明确要求也改掉了。任何「短提示 → 复杂结构化规格」的场景（PPT 生成、网站生成、游戏关卡生成）都适用。

**用实例/深度/法线渲染作为「显式表示」的证据**也是个好做法——它把「我的输出是可编辑的」从口头声明变成了可视化的证明。

对做 3D 内容工具的团队，这篇的真正价值在于它的 Limitations：**当前 agentic 3D 生成的瓶颈不在架构，而在「LLM 写不好 Blender 程序」和「开源图像模型产不出可用语义布局」**。这两个具体卡点比任何架构讨论都更有指导意义。

follow-up：把最脆弱的一环（LLM 生成 Blender 节点图）替换成受限的 DSL 或模板库，很可能大幅提升稳定性；或者做混合策略——简单场景走整体生成，复杂场景才走 agentic 流水线，解决效率问题。目前未见代码开源。

---

## 19. Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes
**👍 58** · 🏛 Meta / 牛津大学 · [arXiv:2608.05000](https://arxiv.org/abs/2608.05000) · [项目页](https://junlinhan.github.io/projects/physics_of_mm_pretrain/)

### 问题与动机
原生统一多模态预训练（不是先训语言再对齐视觉，而是从头一起训）正在成为主流，Gemini、Llama 4、Kimi 都在往这个方向走。但**设计空间和模态在联合训练中如何相互作用的基本机制，几乎没有被系统研究过**——数据怎么配比、哪些参数该共享、什么时候引入视觉，业界靠的是启发式。

这篇的定位是做「统一多模态预训练的物理学」：用受控实验（合成 + 真实大规模数据）把机制拆开。

### 方法与核心创新
默认骨干是 1.5B 的 Llama-3 式解码器（文本和图像用**分离 FFN**，总计 2.3B 参数，16 层、hidden 2048、GQA 32 查询头 8 KV 头），采用 **Transfusion** 框架——文本走离散 next-token 预测、视觉生成走连续 flow matching，统一在一个模型里。视觉 tokenization 支持四种配置（RAE 默认 / 原始像素 / CLIP+VAE / AR UniTok），用来验证结论不依赖特定表征。

四个核心发现：

**(i) 知识流是不对称的。** 语言是**所有视觉任务的通用增益器**；视觉理解是**生成的强先验**；而**视觉生成对其他能力是中性的**——既不帮忙也不损害。

**(ii) 协同 vs 竞争由数据复杂度决定。** 简单任务充当跨模态增益器，复杂任务引发容量竞争、盖过协同。架构上，**共享注意力和归一化层促进协同，解耦 FFN 缓解容量竞争**。

**(iii) 早期统一优于后期对齐或顺序训练。** 延迟整合会导致**「视觉懒惰（vision laziness）」**——模型转而依赖语言先验，而不去真正用视觉证据。

**(iv) 高效配方**：用**仅 5% 的算力预算**就能达到强生成性能。

### 关键实验结果
**语言比例扫描**（视觉固定 50B token，语言从 0% 加到 80%，即额外 0/12.5/33/75/200B）：所有视觉理解评测轴**单调提升**，Knowledge 和 OCR&Chart 提升最剧烈；视觉生成的文图对齐和组合生成质量也一致提升，**条件和无条件的 diffusion loss 都下降**——注意无条件生成也受益，说明语言不只是提供了更好的文本条件，而是提升了模型的原生视觉建模能力本身。这个结果在四种 tokenizer 设计下**全部复现**。

**数据配比网格搜索**（表 1，L/U/G 三轴）：语言需要占主导份额——PPL 从 10% 语言时的 19.27 单调降到 80% 时的 15.57；但**视觉能力在高度不对称的特定配比处达峰**，70/15/15 时视觉理解均分 38.1、DPG 0.399、GenEval 0.219 都接近最优，而生成的 DiffLoss 反而在 50/25/25 时最低（0.2804）。三个目标的最优配比不重合，这是配方设计的核心张力。

**规模化验证与受控基线对比**（表 2，13.5B MoE、2T token）：Full 配置在**所有指标上全胜**——PPL 11.67（Balanced Recipe 11.97、Dense 12.14、Late-Fusion 12.25），视觉理解均分 **43.08**（41.42 / 40.49 / 40.66），GenEval **0.482**（0.467 / 0.459 / 0.471）。三个消融基线分别验证了三个设计选择：数据配方（Balanced Recipe）、架构风格（Dense Model）、视觉对齐策略（Late-Fusion）。

**概念迁移的留一实验**（CLEVR 合成数据）设计得很干净：把含某个概念（如黄色、球体、front-left、数量 4+）的所有场景从**生成训练数据**中外科手术式移除，但**理解数据完全不动**，然后用明确要求该概念的提示词测模型能否生成它——直接测量「理解 → 生成」的概念级迁移。

### 局限性与开放问题
论文有专门的 Limitations 附录，两条：(1) **实证局限在文本和静态图像**，视频和音频这类动态连续模态未探索。视频引入时间维度，会改变数据复杂度和算力需求，知识流和模态协同的原理是否适用、时间推理是否需要不同的架构解耦或数据课程，是关键的下一步。(2) 配方虽在 13.5B / 2T 上验证，但**前沿模型的极端规模（>1T 参数）下机制可能演化**——模态从协同转向竞争的阈值可能移动；更有趣的是，作者推测**超大规模下可能出现新的双向知识流**：目前观察到「视觉生成对理解几乎无反向迁移」的严格不对称，但足够大的模型或许能把生成建模当成内部世界模拟器来用。

我补充两点：其一，**「视觉生成是中性的」这个发现如果成立，其含义比论文说的更尖锐**——它意味着当前把大量算力投给视觉生成的统一模型，在语言和理解能力上得不到任何回报，生成能力是「白加的」而非「协同的」。论文把这解读为「生成可以低成本集成」，但反过来读也可以是「生成没有为统一性做出贡献」。

其二，**受控实验大多在 1.5B/100B token 规模**，而规模化验证只有一组 13.5B/2T 的配置对比。四个发现里有多少能在 1.5B 到 13.5B 的跨度上保持，论文只对最终配方做了验证，没有对每个 finding 单独做规模检查。

### 启发与应用前景
这是本周对**训练决策**最有直接价值的一篇。三条可以立刻用上的结论：

1. **语言数据是最便宜的视觉能力提升手段**——加语言 token 能单调提升所有视觉指标，包括无条件生成。做多模态模型时不要为了「多模态」而削减语言数据。
2. **架构上：共享注意力和归一化、解耦 FFN**。这个组合在四种视觉 tokenizer 下都成立，是相当稳的经验结论。
3. **早期统一，不要延迟引入视觉**——「视觉懒惰」这个现象命名得好：延迟整合的模型学会了用语言先验蒙答案，而不是真看图。这解释了很多 late-fusion VLM 在需要细粒度视觉证据的任务上表现差的原因。

方法论上，**「留一概念 + 单侧移除」的迁移测量设计**值得抄——它把「模态间是否真的迁移了知识」从相关性证据变成了因果证据。

follow-up：作者指出的视频/音频扩展是显然的方向；我更感兴趣的是**验证「视觉生成中性」这个结论在更强推理任务上是否成立**——如果生成真的能当内部世界模拟器，那应该在空间推理、物理推理这类任务上先显现，而不是在 MMLU 上。

---

## 20. Progressive Agent Skill Generation via Reinforcement Learning
**👍 58** · 🏛 香港中文大学 / LIGHTSPEED · [arXiv:2608.01678](https://arxiv.org/abs/2608.01678) · [GitHub](https://github.com/ejhshen/skill-alpha)

### 问题与动机
Agent skill（模块化的过程性知识单元，如 Claude Skills 那种 SKILL.md）已经成为提升复杂任务表现的重要机制，于是「从文档或经验里自动生成高质量 skill」变成了实际问题。

现有方法要么是启发式、要么是流水线式的整合，**每种证据来源都要单独设计**。学习式方法本来更统一，但卡在一个根本困难上：**skill 没有天然的监督信号**——它没有「相关性」或「正确性」可言，它的价值**只能由它是否改善了 agent 在下游任务上的行为来决定**。

### 方法与核心创新
**Skill-α** 的关键设计是把 skill 生成**形式化为顺序编辑过程**，从而把「整个 skill 好不好」这个不可分解的问题，拆成一串「这一次编辑好不好」的可单独评估的问题。

编辑动作空间有五个（结构化 JSON）：**Create**（新增可复用章节）、**Update**（整节替换）、**Merge**（合并重叠章节）、**Prune**（删除有害或冗余内容）、**Noop**（保持不变）。

核心创新是 **rollback reward（回滚奖励）**：对每次编辑，在一个**锚定查询**上比较「用原 skill 执行」和「用编辑后 skill 执行」的下游结果。具体流程是——先用控制 skill z_{t−1} 在锚定查询上跑一遍拿到 r_ctrl，再采样 G 个候选编辑动作，每个应用后在**同一个锚定查询**上评估拿到 r_edit，两者之差就是这次编辑的信用。这样每个局部编辑都直接被「它对下游行为的改变」定价，而不是被文本合理性定价。

策略从 Qwen3-8B 初始化，先用监督编辑数据热启，再用 GRPO 优化。被编辑的 skill 是一份不断演化的 SKILL.md；worker agent（GPT-4o / Claude-Sonnet-4.5）是**固定的、不参与训练**。

### 关键实验结果
两个设置：document-to-skill（CL-Bench）和 experience-to-skill（SpreadsheetBench、tau2-bench）。

**document-to-skill（GPT-4o worker）**：最大增益在 Procedural Task Execution——无 skill 时 4.30，Skill-α 到 **9.68**，而 Anthropic Skill-Creator 只有 5.38、Ctx2Skill 4.30、AutoSkill **3.23（比不用 skill 还差）**。作者的解读是 Skill-α 不只是在压缩长上下文，而是**把过程性约束转成了可执行的 skill**。换 Claude-Sonnet-4.5 做 worker，四个类别全部最优或并列最优——说明生成的 skill 不只适配用来构造它的骨干，而是**可迁移的结构化知识**。

**experience-to-skill（GPT-4o）**优势更明显：SpreadsheetBench 18.00 → **27.50**，tau2 Airline 40.00 → **65.00**（+25 点），Telecom 12.50 → 22.50，Retail 打平最好的 80.00，tau2 均分 **55.83**。

**消融是这篇最有说服力的部分**（CL-Bench 均分 / SpreadsheetBench / tau2 均分，完整版 10.38 / 27.50 / 55.83）：
- **SFT only**（去掉 RL）→ 3.46 / 15.50 / 44.17，说明收益不是来自学会输出格式；
- **去掉 rollback reward**（换成直接 verifier 奖励）→ 3.68 / 17.00 / 46.67——**几乎回到 SFT only 的水平**，这是全文最关键的消融：rollback reward 是把局部编辑和下游改善绑定起来的唯一信号，去掉它 RL 基本失效；
- **去掉 Merge/Prune** → 4.74 / 20.00 / **39.17**（tau2 上比 SFT only 还差），说明 skill 构建必须有显式的合并和删除，光靠不断累加不行；
- **去掉 Noop** → 9.55 / 22.00 / 53.33，仍然不错（排除了「模型只是学会不编辑」的平凡解释），但完整版更好。

**训练动态**的观察很有意思：完整模型**初始奖励不是最高的，但随训练逐步提升**，说明它在学习更好的编辑策略而非利用某个固定捷径；其动作分布是均衡的（Create 主导但其余四个都活跃）。而**去掉 rollback reward 的变体，动作分布被 Noop 主导**——弱信用分配下策略退化成保守的「什么都不改」。

**证据粒度分析**也值得注意：证据顺序影响不大（source 27.50/55.83、shuffled 28.00/51.67、reverse 26.00/57.50），但**每步证据批量大小影响很强**——1 个 22.00/47.50、2 个 23.50/53.33、**4 个 27.50/55.83**、8 个 20.00/48.33。太小则编辑器短视、对局部失败过度反应、skill 碎片化；太大则多个模式和候选编辑混在一起，编辑焦点被削弱、信用分配变模糊。

### 局限性与开放问题
作者自承两条：**奖励和 verifier 接口仍然依赖具体 benchmark**；**skill 表示是纯文本的**。

我补充几点：其一，**CL-Bench 上的绝对分数低得离谱**——Procedural Task Execution 最好也只有 9.68（满分应该是 100 制的 LLM-as-judge 分数），CL-Bench 均分 10.38。在这个量级上，「+3.3 点」的领先意味着什么很难直观判断，而且多个基线（AutoSkill 3.23）比不用 skill（4.30）还差，说明**这个 benchmark 上 skill 生成整体还在很低的水平**，排名的稳定性存疑。

其二，**rollback reward 的成本很高**：每个候选编辑都要在锚定查询上完整跑一遍 worker agent，G 个候选就是 G 次完整 agent 执行。论文没报训练总成本或 GPU 时，但这个开销随 group size 线性增长，规模化会很贵。

其三，**锚定查询的选取有偏**：experience 设置下，锚点池从「不成功或部分成功的轨迹」构造。这让 rollback reward 偏向「修复失败」，可能低估那些让成功案例更稳健的编辑。

其四，**worker 固定为 GPT-4o**（评测时换过 Claude 验证迁移），但**skill 生成器只有 Qwen3-8B 一个规模**，没有规模消融。

### 启发与应用前景
最有价值的设计原则是**「把不可评估的产物拆成可评估的增量」**——skill 整体好坏没法打分，但「这一次编辑是否改善了下游」可以直接测。这个思路能推广到任何「产出一个 artifact，其价值只能通过使用来体现」的场景：prompt 优化、文档编写、配置调优、知识库构建。

**rollback reward 本身是一个可复用的机制**：A/B 对比「改动前 vs 改动后」在同一个下游任务上的表现。它比让 LLM 判断「这次编辑好不好」可靠得多，因为它测的是实际效果而非表面合理性。

**Merge/Prune 必须存在**这个消融结论对做记忆系统/知识库的人特别重要——**只增不删的知识累积会主动变坏**（去掉 Merge/Prune 后 tau2 分数比不训还低）。这与 agent memory 领域普遍只关注「怎么写入」的现状形成对照。

**证据批量大小的 U 型曲线**（4 个最优）也是个实用参数：做增量式知识提炼时，一次喂太少会碎片化，喂太多会失焦。

follow-up：作者指出的方向是更通用的 verifier 和多模态 skill 格式。我认为更紧迫的是**降低 rollback 的评估成本**——用一个学到的代理模型预测「这次编辑会带来多少下游改善」，只对高不确定性的编辑做真实 rollback。代码已开源。

---

## 21. ToolArtist: Tool-Using Unified Multimodal Models for Agentic Image Generation
**👍 56** · 🏛 中国人民大学 / 香港科技大学（广州）/ 新加坡国立大学 · [arXiv:2608.04436](https://arxiv.org/abs/2608.04436)

### 问题与动机
文生图模型画得漂亮，但在需要**复杂语义理解、多步推理、外部世界知识**的开放世界任务上仍然不行——比如「画一张对比 WGS 84 和 GRS 80 参考椭球的技术图，铭牌上要列出 WGS 84 半短轴的精确米数和维护该标准的国际组织名称，两者都必须正确」。模型不知道那个数字，也不知道那个组织。

已有的 agentic 图像生成工作有个共同缺陷：**要么规定固定工作流（先搜后画），要么只把开放世界生成过程的一部分交给 agent 控制**。结果是推理、工具调用、图像生成**不由单一策略协调**——搜索完了就交给一个独立的画图模型，中间没有反馈回路。

### 方法与核心创新
把整个开放世界图像生成过程放进**一个统一策略**里。基座是 Emu3.5（原生自回归统一多模态模型，文本和图像都是 token，在同一个自回归上下文里）。

**SFT 阶段的数据构造有个巧妙的转换**：先让 teacher agent 带着 TextSearch、ImageSearch 和一个**外部图像生成工具**（gemini-3-pro-image-preview）做多轮 rollout，得到含推理、工具调用、工具返回、最终图像的完整轨迹；然后在 convert 阶段，**把外部图像生成这一步「隐藏」掉**——不再把它当作独立的工具输出，而是重写成原生多模态生成格式：调用图像生成工具用的 prompt 写成 visual-caption span，后面跟对应的图像 token。也就是说，**teacher 轨迹里每一次外部图像生成，都被转换成「模型自己在同一条自回归轨迹里完成生成」的形式**。这样学生模型学到的是端到端的原生生成能力，而不是「学会调一个画图 API」。

搜索工具也做了针对性增强：TextSearch 用 Google Search + LLM Reader 检索、摘要、在候选页面间回退；ImageSearch 先过滤掉下不下来的页面，用 LLM Reader 判断相关性，最后返回**可直接使用的参考图 + 结构化摘要**。

**RL 阶段**是 **RAD-GRPO（Reason-Act-Draw GRPO）**，用互补的 intent reward 和 quality reward 联合优化完整策略。

### 关键实验结果
**SFT 语料 7,132 条轨迹**，主题分布多样（地理/建筑 18.9%、科学/工程 14.8%、历史 11.6%、音乐/影视 10.7%、IP/游戏 10.4% 等），**平均输入 20.5k token、中位 20.2k**，工具调用均值和中位都是 4.0——说明这不是单步图像提示的集合，而是带大量检索证据和视觉上下文的多轮轨迹。

**WISE**（1000 条提示、25 个子域，测世界知识整合）：总分 **0.79**，超过此前的 agentic 图像生成模型。知识密集的自然科学类目表现好——物理 0.81、化学 0.79。但**闭源前沿模型仍然领先**：Nano Banana 0.89、Nano Banana-Pro 0.87，尤其在 Time（0.87/0.80 vs 0.62）和 Space（0.95/0.89 vs 0.75）上差距明显。

**WorldGenBench-Humanities**（244 个国家和地区、732 条提示，用知识清单打分）：平均 KCS **22.10**，是非闭源最好，对比 Qwen-Image 21.76、Unify-Agent 15.58、GenSearcher-Qwen-Image 13.66。但闭源的 Nano Banana 是 29.67、Nano Banana-Pro 28.62，**差距约 7 个点**。且大洲级别的最好成绩是分散的——本方法领先非洲、南极、亚洲，Qwen-Image 领先欧洲、北美、大洋洲，Unify-Agent 领先南美。

**最有说服力的消融是 image_search 的 source-aware 摘要**：去掉后总分从 0.79 掉到 **0.61（-0.18）**，其中**生物类目从 0.75 崩到 0.25（-0.50）**、化学 -0.29、空间 -0.21。这说明模型的知识增益**高度依赖检索结果被结构化摘要的质量**，而不只是「拿到了图片」。

**RL 训练动态**：整体奖励从初期短暂下降后，Step 10–20 快速上升，中期在 0.37–0.40 波动，末期升到约 0.41；策略熵从 12.03 缓降到 11.87–11.91 且**没有熵坍缩**。

### 局限性与开放问题
论文**没有 Limitations 章节**。我看到几个问题：

其一，**与闭源模型的差距是结构性的，不是边际的**。WISE 上 0.79 vs 0.89、WorldGenBench 上 22.10 vs 29.67。论文的叙述重心放在「超过其他 agentic 方法」，但从案例表看，即使是本方法表现最好的 Socotra 案例，8 个清单项也只满足了 5 项（KCS 0.625）；Moorea 和 Santo Domingo 都只有 0.400（10 项中满足 4 项）。**绝对水平离「可用」还很远**。

其二，**训练数据规模极小**——7,132 条 SFT 轨迹，RL 阶段的规模论文没报。而每条轨迹平均 20k token，总量约 1.5 亿 token。这个量级下，模型学到的可能更多是「调用工具的格式」而非真正的开放世界知识整合能力。

其三，**SFT 的 teacher 是 gemini-3-pro-image-preview**，转换后的图像 token 来自 Gemini 生成的图。这意味着 **ToolArtist 的图像质量上界由 Gemini 封顶**，而它对比的闭源基线正是同类模型。这是个循环依赖：用闭源模型的输出训开源模型，再和闭源模型比。

其四，**评测的判分器是 LLM-as-judge**（WISE 的 WiScore 用三个准则加权 0.4/0.3/0.3），WorldGenBench 的清单也由模型判定，缺人工验证。

### 启发与应用前景
最值得抄的是那个**「工具隐藏 + 轨迹重写」的数据转换**：用一个强外部工具产生高质量结果，然后**在训练数据里抹掉工具调用、把结果重写成模型自己产出的形式**。这个技巧可以把任何「外部工具能做但自己做不好」的能力蒸馏成原生能力——语音合成、代码执行、结构化输出都适用，而且比直接学「调 API」更能内化能力。

**image_search 的 source-aware 摘要贡献 0.18 分**这个消融给出了一个很实用的信号：**agentic 生成的瓶颈往往在检索结果的结构化质量，而非模型本身**。做 RAG 类系统时，把「返回原始网页」换成「返回结构化摘要 + 可用素材」的投入产出比可能远高于换模型。

follow-up：把 SFT 数据规模从 7K 扩到十万级；或者验证 RAD-GRPO 在不依赖闭源 teacher 的冷启动下是否还成立。论文声明会发布训练数据和完整后训练基础设施。

---

## 22. Meshy T2: Fast Native Mesh Generation with Flow Matching
**👍 56** · 🏛 Meshy AI · [arXiv:2607.28675](https://arxiv.org/abs/2607.28675) · [GitHub](https://github.com/meshy-dev/meshy-t2)

### 问题与动机
游戏引擎、AR/VR、机器人仿真都跑在多边形网格上，而**生产可用的网格**要求用尽可能少的图元捕捉形状，同时保住锐边、薄结构和有语义的部件边界——冗余面会推高带宽、显存、光栅化成本和编辑工作量。

问题在于**主流 3D 生成系统不直接生成这样的网格**。TRELLIS、3DShape2VecSet、LATTICE 这类方法先学隐式或体素几何表示，再用 Marching Cubes 抽表面。抽取是按**网格分辨率**而非几何结构来做三角剖分的，产出通常是几十万个近乎均匀的三角形——对渲染、编辑、动画流程来说太密。后处理简化算法能降面数，但作为纯几何后处理，它产生的不规则三角剖分**不尊重锐特征和部件边界**。

另一条路是自回归地把网格序列化成 token 逐个解码（MeshAnything、BPT、DeepMesh），但推理慢且对误差累积敏感。

### 方法与核心创新
Meshy T2 在一个**每个 token 恰好对应一个顶点**的隐空间里生成网格。

这个表示由 **vertex-set mesh VAE** 建立：编码器把显式网格映射成逐顶点的 latent 集合，解码器**一次性**从这个集合恢复出顶点、边和带朝向的面。

生成本身是**两级 flow matching 的粗到细级联**（用 Rectified Flow 的线性插值调度）：给定参考图像，**voxel flow** 先在 64³ 的占据栅格上勾出整体形状；**latent flow** 再往这个脚手架里填充逐顶点 latent token，同时受图像、体素脚手架和**请求的顶点预算**三重引导。最后用 VAE 解码器解出网格。

技术上最有意思的细节是**位置编码的最优传输匹配**：vertex-set VAE 处理的是无序 latent token 集合，但 Transformer 骨干依赖 3D RoPE 坐标。作者不是简单按 token 索引绑定 RoPE，而是**在顶点坐标和 Sobol 候选点之间求解最小代价指派（OT）**。

### 关键实验结果
**位置编码消融**验证了 OT 的价值：Chamfer 距离从 index PE 的 0.0127 降到 **0.0070（-45%）**，Hausdorff 从 0.0783 降到 **0.0225（3 倍以上）**，非流形边比例从 0.0102 降到 0.0043。而且**Sobol + Morton 排序配对只到 0.0091/0.0534**——说明增益主要来自完整的传输指派，而非 Sobol 采样本身。

**高模重拓扑**（把 10 万面的密集网格重拓扑到约 4000 面）是这篇最亮眼的结果：Meshy T2 的 CD **0.020** / HD **0.044** / NC **0.860** / **非流形边 0.14** / 三转四率 70.1% / **3 秒** / 成功率 100%。对比——MeshFlow 是 0.319/0.534/304.5 非流形边/94 秒；MeshAnything V2 是 0.037/0.151/69.6/49 秒；BPT 0.061/0.136/39.0/**210 秒**/成功率 95.7%；DeepMesh 0.453/0.719/233.0/**636 秒**/成功率仅 **28.7%**；MeshSilksong 更是 1210 秒、成功率 49.6%、非流形边 **19823**。

也就是说 Meshy T2 在**几何精度、拓扑合法性、速度、成功率四个维度同时领先**，而且速度是最快自回归基线的 16 倍、最慢的 400 倍。非流形边 0.14 vs 对手的几十到上万，这个差距是「能不能进生产流程」的分界。

**图像到网格生成**：FD(Inception) 255.77 / FD(DINOv2) **2312.01** / **6 秒** / 100% 成功率。FD(DINOv2) 是全场最好（Tripo P1 2442.27、MeshFlow 2577.00、MeshAnything V2 2499.05）；FD(Inception) 上 MeshFlow 的 254.06 略好，但它要 94 秒。参考基线 Meshy 6 高分辨率网格是 240.65/2021.68。

### 局限性与开放问题
论文**完全没有 Limitations 章节**，抓到的正文也异常简短（HTML 版只有 7.4KB，是本周 53 篇里最短的），实验部分只有位置编码消融被完整展开。这本身就是个信号——**这更像一份产品技术报告而非研究论文**。

我看到的具体问题：

其一，**FD(Inception) 上并非最优**（255.77 vs MeshFlow 254.06、Tripo P1 255.76），论文对此没有讨论。两个 FD 指标给出不同排名，说明生成质量的判断并不稳健。

其二，**参考基线 Meshy 6 的地位很微妙**——高模重拓扑实验的 ground truth 就是「Meshy 6 高分辨率网格降采样到 10 万面」，也就是说**评测的金标准来自自家的上一代产品**。这让 CD 0.020 这个数字的含义变成「和自家高模有多接近」，而不是「和真实艺术家网格有多接近」。

其三，**没有报模型规模、训练数据来源和规模、训练成本**。对一篇声称建立新范式的工作，这些信息的缺失让复现和评估都无从谈起。

其四，**顶点预算是显式输入**，这在实用上是优点（可控），但也意味着评测里的「约 4000 面」是人为设定的——不同预算下的质量曲线没有报告。

### 启发与应用前景
**「每个 token 一个顶点」这个表示选择**是最值得记住的设计。它把网格生成从「序列化-解码」（自回归，慢且累积误差）和「隐式场-抽表面」（后处理丢失结构）两条路里解放出来，直接在与目标结构同构的空间里生成。同样的思路可以推广到任何「输出是有结构的集合」的生成任务——场景图、分子、电路网表。

**用最优传输给无序集合分配位置编码**是个通用的小技巧，值得单独记：当你要给一个无序 latent 集合上 RoPE 或其他位置编码时，随便按索引绑会显著损失几何精度（这里是 45% 的 Chamfer 差距）。用 OT 把 latent 匹配到低差异序列（Sobol）的候选坐标上，代价不高但收益明显。

工程价值上，**3 秒重拓扑 + 100% 成功率 + 0.14 非流形边**这组数字对 3D 内容生产是实质性的——它意味着自动重拓扑第一次可以放进无人值守的流水线。DeepMesh 28.7% 的成功率显然不行。

follow-up：最该补的是**与真实艺术家网格（而非自家高模）的对比**，以及顶点预算-质量的完整曲线。代码已开源。

---

## 23. Weak-to-Strong On-Policy Distillation
**👍 56** · 🏛 马里兰大学 / 微软研究院 / MBZUAI · [arXiv:2607.26246](https://arxiv.org/abs/2607.26246) · [GitHub](https://github.com/Yu-Fangxu/W2S-OPD) · [项目页](https://w2s-opd.github.io)

### 问题与动机
on-policy 蒸馏（OPD）现在是能力迁移的主流范式，但它有个隐含前提：**teacher 至少要和 student 一样强**。这个前提在两个场景下崩塌——训前沿模型时**根本不存在更大的 teacher**；而「训多个领域专家再合并进一个 student」的做法**需要在 student 的规模上训练专家，成本极高**。

这篇要回答的问题是：**能不能只用比 student 更小更弱的模型来提升 student？**

### 方法与核心创新
**W2S-OPD** 的机制非常简洁，可以一句话说清：**在 logit 空间里，用一对「正模型减负模型」的差值分离出「能力方向」，再把这个方向加到 student 自己的 base 模型上，构造出一个代理 teacher。**

这个代理 teacher 有两个关键性质：它**携带了被分离出来的能力方向**，同时因为锚定在 student 自己的 base 上，**分布上与 student 相邻**——不会因为 teacher 分布太远而让 student 学不动。然后 student 在自己的 rollout 上最小化逐 token 的反向 KL。

论文给了三种对比对的实例化：
1. **Pre-RL / Post-RL**：正模型是小 base 模型经 RL 得到的领域专家，负模型是它的 RL 前初始化，差值分离出**RL 所注入的技能**。这是在 student 规模上训专家的廉价替代。
2. **Smaller / Larger**：正负模型是两个不同规模的现成 base 模型（Qwen3-4B 和 Qwen3-0.6B），差值分离出**纯粹来自规模的能力**。**不需要任何额外训练，已发布的模型里就免费带着这个信号。**
3. **Correct / Wrong Hints**：正负模型是同一个 base 模型分别条件在正确和错误的解题提示上。因为提示条件是共享的，**差值抵消了提示带来的风格偏移**，分离出实例级的「指向正确解」的方向。只需要一个小模型和一份参考解。

### 关键实验结果
Student 统一是 Qwen3-8B（non-thinking），训 100 步。

**Pre-RL/Post-RL 设定**（正模型是 4B-RL 专家）：数学四基准均分 **51.8**，OPD 46.5，SFT 45.7——**+5.3 点**。最关键的是**W2S-OPD 超过了领域 teacher 本身（4B-RL 的 48.8），而 OPD 仍在其之下**。分项：AIME24 68.9（OPD 62.1）、AIME25 60.1（54.8）、HMMT25 Nov. 43.1（38.3）。代码上 60.9 vs 58.7。多 teacher 设定（数学 + 代码专家按域路由）同样领先：数学 52.1 vs 46.5（+5.6）。

**Smaller/Larger 设定是最反直觉的结果**：正模型 4B（数学均分 14.9）、负模型 0.6B（1.7），**两个都远弱于 8B student 的 17.0**，但 W2S-OPD 把 student 提到 **23.0（+6.0）**。HMMT25 Nov. 从 9.0 提到 **20.1（翻倍多）**。代码从 57.6 到 58.8。**用两个更弱的模型把更强的模型显著提升了**，这是全文最强的论点。

**Correct/Wrong Hints**：只用一个 4B 模型 + 参考解，数学 17.0 → 18.4（+1.4），代码 57.6 → 58.7（+1.1）。增益最小但成本也最低。

**能力差距越大信号越强**：固定 4B 正模型，负模型换成 1.7B 时数学增益 +4.7，换成 0.6B 时 **+6.0**。作者的解释是差距越大，减法跨越的「能力随规模增长的方向」越长，提取的信号越强。**实践建议：用手头差距最大的一对**。

**OOD 泛化**（只在数学上训，测科学推理和指令跟随）：GPQA-Diamond 从 38.9 提到 **56.5**（OPD 54.4）；IFBench 上**OPD 把 student 降到了 25.9（低于基座 26.3）**——这是 student 吸收了小专家的缺陷，而 **W2S-OPD 反而提升到 27.0**。这个对比很关键：直接蒸馏小专家会传染它的局限，而只蒸馏「能力方向」不会。

**放大系数 α 的 U 型曲线**：α 太小则能力方向注入不足、信号弱于 OPD；α 太大则代理 teacher 偏离 student 分布太远、监督难以学习。中等 α 最优（实测数学 1.0 / 代码 0.75）。

**成本**：每步 1043 秒 vs OPD 的 868 秒，**只多 20%**——尽管要前向 3 个冻结模型（8B 锚 + 4B 对比对）而非 1 个 4B teacher。原因是单步成本由 on-policy rollout 生成主导，而额外的 teacher 打分是纯前向且可并行。

### 局限性与开放问题
论文的 Discussion 提了开放问题（弱监督能把强 student 推多远才饱和、如何从弱源提取更多信息），但**没有独立 Limitations 章节**。我的观察：

其一，**只做了一个 student 规模（8B）和一个模型家族（Qwen3）**。「logit 差值分离能力方向」这个操作依赖正负模型和 student base **共享词表和 logit 空间**——跨家族、跨词表怎么办，论文没有讨论。这是方法适用性的硬边界。

其二，**Correct/Wrong Hints 那一档的增益（+1.4/+1.1）小到接近噪声**。数学基准是 AIME/HMMT，每个只有 30 题左右，虽然采样了 32 个解，但 +1.4 的均分提升需要看方差才能判断。论文全篇没报置信区间。

其三，**α 需要按域调**（数学 1.0、代码 0.75），加上正负模型对的选择，实际有 3 个自由度需要调，这削弱了「简单有效」的宣称。

其四，**理论解释是启发式的**。「logit 差分离能力方向」建立在一个未经证明的线性假设上（能力在 logit 空间里是可加的方向）。论文用 Schoenfeld 阶段分布分析（表 6）展示了高 Δ token 集中在 Analyze 和 Plan 阶段，这是好的经验证据，但不构成机制解释。

### 启发与应用前景
这是本周对**前沿模型后训练**最有战略价值的一篇。核心命题：**当没有更强的 teacher 时，监督信号可以从「更弱模型之间的差」里造出来。**

三种实例化里，**Smaller/Larger 那一档几乎是免费的**——你手上已有的 Qwen3-0.6B 和 4B 权重，什么都不用训，就能给 8B 提供 +6.0 的数学增益。任何有多个规模开源权重的模型家族都能立刻试。

**「差值锚定在 student base 上」这个设计**是关键，值得单独理解：直接蒸馏小 teacher 会把它的局限一起传过来（IFBench 上 OPD 降到基座以下就是证据），而只取差值再锚回 student，等于**只搬能力增量、不搬绝对水平**。这个思路可以推广到模型合并、LoRA 组合、任务向量等一系列「在参数/logit 空间做算术」的技术。

follow-up：跨家族适配（词表不同怎么办）；以及作者自己提的关键问题——**弱监督的饱和点在哪**。如果能证明弱到强监督可以持续迭代（用提升后的 student 再造新的对比对），那这就是一条真正的自举路径。代码已开源。

---

## 24. Video-DeepResearch: Towards the Next-Generation Multimodal Deepresearch Agent
**👍 50** · 🏛 中国科学技术大学 · [arXiv:2608.03979](https://arxiv.org/abs/2608.03979) · [项目页](https://costaliya.github.io/Video-DeepResearch/)

### 问题与动机
把多模态 agent 从静态图像扩展到连续视频流，要求**稠密的时空 grounding + 开放网络探索**同时进行。作者的前期诊断发现了两个具体的失效模式，都很有价值：

1. **模态偏见（modality bias）**——agent 会绕开视觉工具，转而依赖文本搜索。数据很刺眼：Qwen3.5-35B-A3B 在 VideoDR 上平均只调用 **0.04 次视觉工具**、0.58 次文本工具；Qwen3.5-397B-A17B 是 0.10 / 1.27；**GPT-5 是 0.00 / 0.12——完全不用工具**。
2. **参数化知识泄漏（parametric knowledge leakage）**——模型靠内部记忆而非真正的工具增强执行来答题。GPT-5 在零工具调用的情况下拿到 57% 的竞争力准确率，这意味着**benchmark 分数与真实视频理解已经脱钩**。

### 方法与核心创新
**解耦的感知-探索流水线 + 阶段式工具解锁**：强制 agent 在做网络检索之前，**先完成穷尽的跨帧视觉 grounding**。具体在工具设计上体现为——探索阶段**只暴露 `select_crop_search` 一个工具**（挑帧、从每帧裁一个 bbox、对每个裁片跑反向图搜/视觉网络搜索，一次调用可批量 1–8 个选择），`search` 和 `visit` 到回答阶段才加进来。这个「先看后搜」的硬约束是纠正模态偏见的核心机制。

训练是两阶段：**SFT**（7K 合成轨迹建立解耦范式的冷启动，**另加 7K 纯文本 QA** 来强化基础深度研究能力）+ **GRPO**（2K 中等难度数据，让 agent 自主探索、突破模仿学习天花板）。

另外构建了 **VideoDR-Bench**：200 条人机协作的复杂多跳 VQA，视频长度分布为短（≤2 分钟）46%、中（2–10 分钟）34%、长（≥10 分钟）20%。设计原则是**每道题都可证明地同时需要视觉搜索和外部知识推理**。

### 关键实验结果
**主结果**：Video-DeepResearch-35B-A3B 平均 **64.0%**，超过 Claude-4.5-Sonnet（59.0%）**5.0 点**，显著优于 Gemini 2.5 Pro（57.5%）和 GPT-5（52.5%）。30B-A3B 变体 59.3%，与 Claude-4.5-Sonnet 持平。

相对各自基座的提升极大：Qwen3-VL-30B-A3B 从 40.5 → **59.3（+18.8）**，VideoDR 单项从 38.0 → 62.0（+24.0）；Qwen3.5-35B-A3B 从 42.8 → **64.0（+21.2）**，VideoDR 从 42.0 → 68.0（+26.0）。

**工具使用行为的转变是最有说服力的证据**：基线 Qwen3.5-397B 每任务只有 0.10 次视觉操作、1.27 次文本；Video-DeepResearch-30B 在 VideoDR 上是 **2.33 次视觉 + 4.24 次文本**——**一个 30B 模型的工具使用多样性超过了 397B 基线**。同时 VideoDR-Bench 也确实逼出了更多操作：GPT-5 的视觉调用从 0.00 涨到 0.31、文本从 0.12 涨到 1.43，证明新 benchmark 确实绕开了参数化知识泄漏。

**消融把每一步的贡献切得很清楚**（30B，平均分）：基座 40.5 → 4K 轨迹 SFT **46.0（+5.5）** → 7K 轨迹 SFT **53.0（+7.0）** → 加 7K 纯文本 SFT **56.8（+3.8）** → 加 2K RL **59.3（+2.5）**。视觉 grounding 数据贡献最大（+12.5 累计），文本增强修正工具调用偏见，RL 突破模仿天花板。

### 局限性与开放问题
论文有 Limitation 段落，承认**为了保证高质量数据合成和稳健训练，当前性能水平带来了可观的算力开销**（4 节点 32×80GB GPU、80k token 上下文、TP4/CP2/EP8 混合并行、3 epoch）。

我另外看到几个更实质的问题：

其一，**VideoDR-Bench 的规模和口径有矛盾**。摘要说 200 条实例，但 4.1 节写「VideoDR-Bench 是包含 **100 条**人工标注 VQA 对的评测基准」，而表 2 的视频长度分布加起来是 92+68+40 = **200**。同一篇论文里三个数字对不上。而且六个细分类目里 OTH（other）在多个模型上都是 42.9%——这个反复出现的数字暗示该类目样本极少（可能只有 7 条，3/7 = 42.9%）。**在这样的规模上报六个类目的细分分数，统计意义很弱**。

其二，**「超过 Claude-4.5-Sonnet 5 个点」的比较不对等**。本方法是在 VideoDR 和 VideoDR-Bench 的同分布数据上做了 SFT + RL 的专用模型，而 Claude 是零样本。这个对比只能说明「专门训练有效」，不能说明模型能力更强。表格里 30B 在 DLY（daily）类目上只有 40.5%，**低于 Claude 的 54.1% 和 Gemini 的 51.4%**，NWS 和 OTH 类目也不占优，说明优势集中在训练分布覆盖的类目。

其三，**Discussion 部分的措辞过于夸张**。「emergence over scaling」「30B 打败 397B 说明训练方法比规模重要一个数量级」这类论断，建立在一个 200 条（或 100 条）的自建 benchmark 上，且比较的是「专门训练 vs 零样本」。这是本周论文里 claim 与证据落差最大的一篇。

其四，**关键帧提取上限硬编码为 20 帧**（CLIP 相似度 >0.8 的连续帧丢弃），对 ≥10 分钟的长视频（占 20%）来说，20 个关键帧的覆盖度存疑。

### 启发与应用前景
抛开叙述上的夸张，**两个诊断本身是这篇最有价值的产出**：

**模态偏见**——多模态 agent 会系统性地绕开视觉工具走文本捷径。0.00–0.10 次视觉调用这个数字应该让所有做多模态 agent 的人去测一下自己的系统。修正手段（阶段式工具解锁，探索期只给视觉工具）简单且立刻可用。

**参数化知识泄漏**——GPT-5 零工具调用拿到竞争力分数，说明现有视频 benchmark 大量题目可以靠背知识答对。构建 benchmark 时应该像本文一样，**验证每道题确实需要工具**（可以用「零工具基线的准确率」作为 benchmark 质量的检验指标）。

工程上，`select_crop_search`（挑帧 + 裁 bbox + 反向图搜）这个工具设计对做视频 agent 的团队直接可用——它把「看视频」变成了一个有明确参数的可组合动作，比「给模型 20 帧让它自己看」结构化得多。

follow-up：把 benchmark 扩到千条量级并公开统计口径；以及验证解耦范式在不做同分布 SFT 时的零样本迁移能力。

---

## 25. UEmbed: Unified Sparse and Dense Multimodal Embeddings
**👍 50** · 🏛 阿里巴巴 / 中国科学院 / 耶鲁大学 · [arXiv:2608.02583](https://arxiv.org/abs/2608.02583) · [GitHub](https://github.com/Alibaba-NLP/UEmbed) · [项目页](https://alibaba-nlp.github.io/UEmbed/)

### 问题与动机
稀疏检索是现代搜索系统的地基（倒排索引、BM25 的继承者），Learned Sparse Retrieval（LSR，如 SPLADE）把它从精确词面匹配推向了更丰富的语义。但 LSR 至今**绑死在 encoder 式双向架构上**，扩展到多模态还严重依赖辅助的跨模态模块。

技术障碍很具体：SPLADE 的稀疏权重靠**对所有位置的隐状态做 max-pooling**得到，而 decoder-only 模型的**单向注意力使每个位置看不到后面的 token**，max-pooling 直接失效。所以在 LLM 时代，稀疏检索被留在了上一代架构里。

### 方法与核心创新
**UEmbed 的核心机制一句话讲清**：在输入末尾追加 **N=16 个可学习的特殊 token**，同时把词表**划分成 16 个不相交子集**；每个特殊 token 的因果隐状态负责预测它所分配子集上的稀疏权重，16 个子集拼接起来就是完整的稀疏向量。

因为特殊 token 在序列末尾，它们能 attend 到全部前文，等于各自对完整输入做了一次摘要——**用「多个末位摘要 token」替代了「所有位置 max-pooling」**，绕过了因果注意力的限制。

两个配套设计：
- **词表压缩**：LLM 分词器有大量冗余 token（空白、重音变体）。用 NLTK 做去重音、小写化、空白折叠，同形归并（Hello / HELLO / héllò → hello），打分时取最大权重。**词表从 248,320 压到 184,016**。
- **语义聚类划分**：用 k-means 把词表划成 16 个大小相近的子集，让每个特殊 token 成为一个「软主题专家」。

训练数据来自三个公开集共 394 万样本（Echo-embedding、MLDR、MMEB 训练集），多模态部分用 Qwen3-VL-Embedding-8B 挖硬负例。发布 2B/4B/9B 三个规模，**一次因果前向同时产出稠密和稀疏两种表示**。

### 关键实验结果
**MMEB-v2**：UEmbed-9B 稠密 **71.8** / 稀疏 **71.0**。稠密上领先所有公开数据训练的模型（RzenEmbed-V2-7B 71.1、Ops-MM-Embed-7B 67.1），但**被 Qwen3-VL-Embedding-8B 的 77.8 超过**（后者用大规模私有数据多阶段训练）。4B 稠密 70.4 超过同级所有模型（Embed-RL-4B 68.1），2B 的 66.5 甚至超过一些 7B 开源基线（UniME-7B 64.1）。

**最关键的结果是稀疏和稠密的差距**：所有规模上**最大只差 1.0 点**（9B 是 71.8 vs 71.0）。作者称这是多模态稀疏嵌入在该基准上的首次报告结果，且 **UEmbed-4B 稀疏的 69.7 超过了多个已确立的稠密模型**。

**BEIR**（9 个数据集 nDCG@10）：稠密 9B 平均 **56.3**（最高），4B 56.0，均领先 Qwen3-VL-Embedding-8B 和 GME-7B；稀疏 9B 55.2，与 Echo-Mistral-SPLADE 的 55.2 持平，显著超过 SPLADE-v3 的 50.5。

**对照双向 SPLADE 基线**（同 Qwen3.5 骨干、同数据、同 FLOPS 正则，只是换成双向注意力 + 标准 SPLADE max-pooling 头）：UEmbed-2B 稠密 **+3.2**（61.3 → 64.5）、稀疏 **+2.1**（61.3 → 63.4），**IMG-QA 上差距最大（稠密 +8.2、稀疏 +6.0）**。作者的解读是标准 SPLADE 配方没有充分利用自回归骨干固有的 QA 能力——这说明**统一因果形式化本身就有益于嵌入质量，而不只是部署方便**。

**Agentic 搜索（BrowseComp-Plus）**给出了稀疏检索的实际优势：UEmbed-9B 稀疏准确率 49.76%、召回 64.33%，**平均搜索轮数 31.05**，而稠密版是 33.68 轮；**校准误差稀疏 8.16% vs 稠密 7.06%**（4B 上更明显：稀疏 8.79% vs 稠密 14.34%）。对比 BM25 的 36.87% 准确率 / 17.91 轮和 Qwen3-VL-Embedding-8B 的 31.45% / 34.19 轮。

**消融**：联合训练几乎不损失单模式专家的性能（负迁移可忽略）；语义聚类划分 63.4 > 最大距离 63.2 > 随机 63.0；特殊 token 太多会伤稀疏检索；稀疏训练温度高一些更好。另有一个训练技巧值得记：**混入文本数据能显著稳定和加速多模态稀疏训练**——有文本数据时 <100 步就学出有效稀疏表示，纯多模态数据要近 500 步才能可靠区分正负例。

### 局限性与开放问题
论文有明确的 Limitations，三条都是真问题：(1) **语言和文化偏见**——训练语料偏向中英文，稀疏头激活的跨语言泛化有限；(2) **词表稳定性和伪影**——在现代 LLM 的巨大词表上操作偶尔会产生异常 token 激活（如 `_alt` 这类非标准子词），生产环境需要词表剪枝或后处理过滤；(3) **模态特定的性能差距**——稀疏和稠密在文本和静态视觉文档上接近，但**视频域差距更明显**，作者归因于视频帧的高信息密度和时序动态，怀疑把时空数据平铺进一个稀疏向量会遇到容量瓶颈。

我补充两点：其一，**与 Qwen3-VL-Embedding-8B 的 6 点差距（71.8 vs 77.8）被「私有数据」这个理由带过去了**，但这个差距不小，读者无法判断有多少来自数据、多少来自方法。其二，**BEIR 上稀疏只是持平 Echo-Mistral-SPLADE（55.2 vs 55.2）**，在纯文本这个 LSR 的主场并没有优势，卖点主要在多模态和统一性。

### 启发与应用前景
这篇的工程价值很实在：**一个 checkpoint、一次前向，同时产出可进倒排索引的稀疏向量和可进向量库的稠密向量**。对已有搜索基建的团队，这意味着可以在不推翻倒排索引的前提下升级到多模态语义检索——这比「全面迁移到向量数据库」的迁移成本低得多。

「**用末位多 token + 词表分区替代 max-pooling**」这个技巧是解决「因果注意力下如何做全局池化」的通用解法，可以迁移到任何需要在 decoder-only 模型上做全序列聚合的任务（分类、多标签预测、结构化输出）。

**agentic 搜索里稀疏检索校准误差更低、搜索轮数更少**这个发现有点意外，也很实用——稀疏表示的可解释性（每一维对应一个词）可能让 agent 更容易判断「这次检索是否找到了」，从而减少无效轮次。

**混文本数据稳定多模态稀疏训练**（500 步 → 100 步）是个零成本的训练技巧，做多模态对比学习的人可以直接用。

follow-up：视频域的稀疏容量瓶颈是论文自己点出的最有价值的开放问题——把「一个视频压成一个稀疏向量」换成「多向量稀疏表示」可能是解法。模型和代码已开源。

---

## 26. Knowledge-Geometry Decoupling: Refreshable Pretrained Transfer for Streaming Recommendation
**👍 47** · 🏛 厦门大学 / Shopee · [arXiv:2608.02738](https://arxiv.org/abs/2608.02738) · [GitHub](https://github.com/FuCongResearchSquad/KGD4REC)

### 问题与动机
工业推荐系统正在采用「预训练-再迁移」范式（GPSD 首先形式化），但行为分布持续漂移带来两个此前没被回答的问题：**从行为序列里学什么**，以及**在预训练模型被持续刷新的情况下如何迁移已学的知识**。

具体到机制层面，两个病灶都很具体：
1. **邻接不等于依赖**。常规的 next-token 预测把序列相邻当成依赖关系，会把跨不相关会话的**伪转移**也编码进去——用户看完手机去看袜子，NTP 会认为这是一个有意义的转移。
2. **预训练目标和任务目标对共享表征提出冲突的几何要求**。论文给了直接证据：联合优化时测两个 loss 的梯度余弦相似度，**在 embedding 和 Transformer 参数上都很弱、有时为负**；而且是双向干扰——只回传预训练 loss 时任务 loss 在多个数据集上不降反升，开启任务梯度后任务 loss 降了但预训练 loss 又回升。

### 方法与核心创新
**KGD（Knowledge–Geometry Decoupling）**的前提是：预训练知识和任务几何**不必竞争同一套表征**，可以作为分属不同所有权的两层共存。

**「学什么」→ BMTP（Behavioral Multi-Token Prediction）**：只保留**协同相关或语义相关**的未来 item 作为监督，把基于邻接的监督沿协同和语义两个轴过滤掉，给编码器一个更干净的基础几何。

**「怎么迁移」→ 解耦读写所有权**：
- **可刷新的编码器**拥有行为知识；
- **任务学习器**通过**只读的 cross-attention** 读取编码器的上下文化状态，通过 **ACR（Anchored Calibration Residual）**——一个**正交于预训练 embedding** 的残差——写入任务特定几何。

这个「读写分离」的关键效果是：**编码器可以持续刷新而不受任务梯度干扰，也不会让下游适配失效**。日常流水线是——每天先用新数据刷新编码器一遍，然后冻结编码器参数、只更新学习器（编码器只做推理）。训练成本约为纯编码器一遍的两倍，A100 上约两小时，与增量训练成本相当。

### 关键实验结果
**八个公开基准（Amazon 5-core）**：KGD + BMTP 在**每个数据集上都最优**，比最强已发表基线高 **4–12%**。

论文把增益拆得很清楚：
- **BMTP 的贡献依赖于迁移方式**。固定迁移策略、把 NTP 换成 BMTP 在不同架构和数据集上都有帮助；但**冻结程度越高，BMTP 相对 NTP 的增益越大**（Arts 上 TE&FE/TA&FE 是 +6.5%/+7.1%，TA&FD 和 TA&FA 下达到两位数），而**全量微调下同样的优势会缩小甚至反转**（Phones 上 TE&FT 是 **-8.2%**）。作者的解读很到位：任务梯度被允许重写预训练参数时，**它恰好抹掉了 BMTP 装进去的结构**。这既证明 BMTP 编码了真实可迁移的知识，也证明保住它需要一种不覆写的迁移方式。
- **控制预训练目标后，架构本身的贡献是 0.4–7.0%**（同样用 BMTP 预训练的编码器，KGD 仍超过最好的共享参数基线）。

**工业流数据（28 天 + 90 天）**是这篇最有价值的部分：
- **决定性因素是所有权而非刷新时间表**。对比 S2（预训练一次后冻结）和 S3（每日刷新）：**在共享参数上，刷新根本没用**——TA&FT 从 0.7852 反而掉到 0.7837，TA&FE 几乎不动（0.7841）。同样的 S3 时间表下，KGD 靠解耦达到 **0.7867**；而**去掉 KGD 的接口（无 ACR、无只读编码器）会让 S3 崩到 0.7785，低于从零训练的 0.7806**。刷新只有在解耦所有权下才有回报——参数纠缠时交替更新会反复覆写几何，使其无法稳定。
- **各种替代方案各有各的失败方式**：IncCTR（0.7810）蒸馏前一天的 checkpoint，几乎不加结构信号；**buffer replay 更差（0.7732，低于从零训练）**——重放历史等于多轮训练，触发稀疏 embedding 的「超过一轮就过拟合」问题；LoRA（0.7818）低秩适配器留给几何重塑的自由度太少。
- **90 天轨迹揭示了快照掩盖的东西**：冻结迁移在短窗口内看着有竞争力，所以刷新性和稳定性无法从静态表格判断。

**线上 A/B（Shopee 首页搜索，已全量部署）**：**人均 GMV +1.75%，广告收入 +1.53%**。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我看到的问题：

其一，**AUC 提升的绝对值很小**（0.7867 vs 0.7841/0.7852，即千分之二左右）。工业推荐场景下千分位 AUC 提升确实能转化成可观收入（线上 +1.75% GMV 也印证了），但这也意味着**结论对实验噪声很敏感**，而论文没有报多次运行的方差。

其二，**公开数据集的选择被作者自己承认是妥协**。附录 C.1 明确说更接近工业场景的公开数据集都不适用——有的只跟踪高活跃用户（无法反映真实流量的用户 ID 分布漂移）、有的特征加密（无法做语义 BMTP）、有的交互序列不足或时间戳不可靠。所以八个 Amazon 基准**主要用来验证「学什么」，而「怎么迁移」的核心主张只在自家工业流上验证**——不可复现。

其三，**BMTP 的「协同或语义相关」筛选标准**是这套方法的关键，但正文抓取的部分没有给出具体的筛选阈值和敏感性分析。

其四，**训练成本翻倍**（编码器刷新 + 学习器训练两遍）。虽然作者说与增量训练成本相当，但这是相对基线而言，绝对成本仍然是每天两小时 A100。

### 启发与应用前景
最有普适价值的是那个**梯度冲突诊断**：测预训练 loss 和任务 loss 在共享参数上的梯度余弦相似度。**如果接近零或为负，就说明两个目标对表征提出了冲突的几何要求，参数共享会互相污染**。这个诊断可以直接用在任何「预训练 + 下游微调」的场景，包括 LLM 的持续预训练与 SFT 之间的冲突。

**「读写分离的所有权设计」**是个通用架构模式：一方拥有知识（只读暴露），另一方通过正交残差写入任务几何。相比 LoRA（在同一空间里低秩微调）和冻结迁移（只重投影、不塑造自己的几何），这个设计的独特之处在于**允许知识侧持续更新而不使任务侧失效**——这对任何需要「基座模型持续迭代 + 下游应用不能每次重训」的场景都有价值，包括 LLM 的基座升级问题。

**「刷新只在解耦下有回报」**这个结论对做持续学习的团队是个警告：单纯加上刷新时间表可能让效果变差（论文里从 0.7852 掉到 0.7837）。

follow-up：把这套解耦思路搬到 LLM 的持续预训练 + 指令微调；或者研究 ACR 的正交约束能否放宽成更一般的低干扰约束。核心实现已开源。

---

## 27. N₀-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation
**👍 47** · 🏛 NeoteAI / 复旦大学 · [arXiv:2607.23783](https://arxiv.org/abs/2607.23783) · [GitHub](https://github.com/neoteai/N0-TWAM) · [项目页](https://research.neoteai.com/n0-twam)

### 问题与动机
这是 [13] N₀-VTLA 的姊妹工作，但走的是**世界模型**而非策略模型的路线。视频世界-动作模型（预测未来场景再从中读出动作）在接触密集任务上表现很差——论文的数据很直接：**纯视觉世界-动作基线在 UniVTAC 上甚至落后于普通 VLA 策略**（LingBot-VA 31.4、FastWAM 48.0、GigaWorld-Policy 16.5，而 π₀.₅ 是 41.4、InternVLA-A1 是 67.1）。

原因很清楚：**预测未来场景不足以完成由接触定义的任务**。插 HDMI、拔钥匙、叠碗这些任务的成败发生在像素看不见的地方。

### 方法与核心创新
N₀-TWAM 的三个设计选择与「视频世界模型 + 加触觉」的常规做法明确区分：

**(i) 触觉与视觉联合建模**——未来触觉信号在**同一个 flow-matching 目标、同一个因果步**下与未来视频一起生成，模型**直接预测触觉**而非从像素推导。

**(ii) 容量隔离在权重而非注意力**——每个模态有自己的 expert，但**共享一个 self-attention**。这样视觉和触觉保有私有权重，同时保持完全互相注意。论文明确对比：同期的触觉感知模型是靠**门控或屏蔽触觉 token 来保护视觉流**，那等于承认两个模态会互相干扰。

**(iii) 触觉扮演双重角色**——既是**提前预测的前瞻目标**，也是**当下观测**（一条轻量的 observed pathway 编码当前触觉，在动作头之前 cross-attend 进动作 token）。

架构是**非对称 Mixture-of-Transformers**：30 层，视频 expert 是全宽的（hidden 3072，5.00B，从预训练视频模型热启），动作和触觉 expert 是瘦的（hidden 1024，各 1.13B / 1.03B，从零训练），总计 **7.16B**——对比全宽变体的约 15B，**参数量减半**。流式推理便宜的关键在于每个动作去噪步只重跑轻量的 action expert。

另有 **NeoForce**（统一的力空间触觉表示）作为物理接地的接触信号，以及**触觉接触事件用于任务分段**，支持长程多阶段操作。预训练覆盖**六种本体、450 个任务**。

### 关键实验结果
**UniVTAC（公开触觉操作基准，8 任务）**：平均成功率 **84.5%**，比任何类型的最强基线（InternVLA-A1 67.1%）高约 **17 点**。

**NeoSim（自建，12 任务）**：**49.4%**，超过最强基线 π₀.₅ 的 45.8%，在接触密集的堆叠和插入任务上领先明显（Insert USB 100、Plate Stack 98、Bowl Stack 97）。但作者诚实指出**触觉优势在仿真里比真机小**——仿真触觉不是 N₀-TWAM 预训练用的传感器，observed pathway 只能退回轻量的仿真编码器而非 NeoForce 力空间表示。

**真机（8 任务）**：平均 **46.3%**，领先所有 VLA 和纯视觉世界-动作基线。

**消融（UniVTAC/NeoSim 平均）**：
- **20% 预训练数据** → 65.4（-19.1）：**数据规模是最大的单一因素**，Pull-out Key 从 79 崩到 25、Put Bottle in Shelf 从 87 崩到 46；
- **去掉 predicted 触觉** → 71.8 / 41.1；
- **去掉 observed 触觉** → 70.5 / 29.6。

两条通路都必要，且**在 NeoSim 上 observed 通路更关键**（-19.8 vs -8.3）。

**触觉真实感实验设计得很讲究**：UniVTAC 默认用仿真器的 clean 输出（基于标定的 GelSight 渲染，接触表现为干净的色块），但真实视觉触觉传感器返回的是「银灰、有斑点、柔和阴影 + 传感器噪声」的凝胶图像。作者在同一信号上额外渲染了更真实的 gel 图像（重建银色凝胶底、叠加密集无标记彩色斑点、按接触做切向和法向平流、加深度阴影和弹性体颗粒）。结果：**训练评测都在这个真实渲染上，成绩只从 84.5% 掉到 82.4%**（NeoSim 单臂 63.8→58.5、双臂 42.3→39.3）。更关键的是，因为它与真实传感器共享视觉域，**可以被预训练的 NeoForce 力编码器读取——读了之后反而涨到 88.1%**（Put Bottle in Shelf 82→96、Insert HDMI 63→75、Pull-out Key 74→83）。

另一个实用发现：**delta vs 绝对末端位姿参数化的权衡**。四个对齐敏感任务上，delta 平均 50.0%、绝对位姿 **82.5%**（Lift Can 24→93、Grasp Chip 38→92）——delta 从移动的 chunk 起点锚测量，每 chunk 的小误差会累积且目标从不绑定固定位置；绝对位姿把每个目标直接接地在世界系。**统一多本体策略用 delta（平移不变、跨本体迁移好），亚厘米级放置主导成败的任务特化设定用绝对位姿。**

### 局限性与开放问题
论文的 Future Work 提了三个方向（更快推理、等），但**没有独立 Limitations 章节**。我的观察：

其一，**泛化能力其实是弱项**。Table 4 显示在未见物体上 N₀-TWAM 只有 **65%，明显低于 π₀.₅ 的 80%**；只在视觉扰动上领先（45% vs 25%），综合 51.7% vs 50.0% 基本打平。**触觉带来的是精度而非泛化**，这个 trade-off 论文没有正面讨论。

其二，**NeoSim 上的绝对水平仍然很低**：Unplug & Plug Charger **0%**、Place Gears 0%、Cup Stack 12%、Cup Unhandover 14%、Cup Unstack 16%。12 个任务里有 5 个在 20% 以下。49.4% 的均值主要被 Plate Stack（98）和 Bowl Stack（97）两个任务撑着。

其三，**与姊妹论文 [13] 的关系没有说清**。N₀-VTLA 在 UniVTAC 上是 83.1%、NeoSim 50.8%，N₀-TWAM 是 84.5% / 49.4%——**两个方法在同一批基准上几乎打平**，一个是策略模型加预测式 latent，一个是世界模型加原生触觉。同一团队同期发两篇拿到相同水平，说明**收益可能主要来自共享的 NeoData 预训练和 NeoForce 表示，而非各自的架构创新**。两篇论文都没有对方的对照实验。

其四，**7.16B 模型 + 自研传感器 + 六本体 450 任务的自采数据**，复现门槛极高。

### 启发与应用前景
**「预测未来接触」这个目标本身**是最有价值的抽象。它把触觉从「反应式的当前读数」变成「前瞻式的预测目标」，让策略能在接触发生前就调整——这与 [13] 的 latent tactile token 是同一个洞见的两种实现。

**非对称 MoT（全宽视觉 expert + 瘦动作/触觉 expert）**是个很实用的工程设计：多模态世界模型里各模态的建模难度差别巨大，给所有模态同样宽度是浪费。参数减半而性能不降，这个配比值得任何做多模态 expert 架构的团队参考。

**gel-rendered 触觉图像的做法**解决了一个真实的 sim2real 问题：仿真触觉太干净，训出来的模型读不了真传感器。而这里的做法是**在不重新仿真的前提下，把 clean 场重渲染成有噪声有纹理的真实外观**，代价只有 2 个点，却换来了「能被真实传感器的预训练编码器读取」这个大收益。这个「渲染层做域适配、不动物理仿真」的思路适用于所有 sim2real 场景。

**delta vs 绝对位姿的 32.5 点差距**是个应该被广泛知道的工程细节——做机器人策略时，泛化和精度的参数化选择不是无关紧要的实现细节。

follow-up：最该做的是 N₀-VTLA 和 N₀-TWAM 的直接对照，弄清架构创新和数据/表示各贡献多少。代码和权重承诺开源。

---

## 28. VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation
**👍 46** · 🏛 上海交通大学 / 小红书 / 香港中文大学 / 浙江大学 · [arXiv:2607.28590](https://arxiv.org/abs/2607.28590) · [GitHub](https://github.com/DeepExperience/VAD_Multimodal_OPD)

### 问题与动机
多模态 on-policy 蒸馏靠一个**特权视角 teacher**（能看到更清晰/更完整视觉证据的教师）监督学生自己生成的轨迹，来迁移细粒度视觉知识。问题在于 teacher 的 next-token 修正是**源混杂的（source-mixed）**：它把视觉信号、语言先验、以及 teacher 特有的效应全混在一起。

论文把关键挑战说得很精确：**难点不是判断「在哪里蒸馏」或「蒸多强」，而是估计「哪些修正真的由视觉证据支撑」**。此前的方法（visual-advantage weighting）只是给 token 加权重，仍然在蒸馏一个混杂的信号。

### 方法与核心创新
**VAD（Visual Attribution Distillation）**用**反事实目标重构**来分离视觉可归因的部分：

在学生生成的每个前缀上，**用同一个固定的 teacher 评估两次——相关证据在场（x⁺）和证据移除（x⁻）**。两次的中心化对数概率之差定义 **u_t**，这是一个**有符号的视觉证据方向代理**：它估计「暴露证据」对每个候选 token 是支持还是反驳。

然后把原始修正 r_t **投影到这个代理方向上**，分解成**与干预对齐的分量**和**代理无法解释的残差**，再从前者重构一个**锚定在学生上的目标** q^VAD。训练时这个重构目标提供主要监督（标准的 token 级 JS 散度，散度形式不变、只换目标），特权 teacher 只贡献一个**弱正则**——正则的 token 权重在「完整修正中被视觉归因的比例越小」时越大。

弱正则是必需的：论文的 visual-only 消融显示，只用 L_vis 会导致明显的**语言和输出漂移**——回复变长变重复、答案承诺被延迟、格式和停止变得不稳定。原因很清楚：视觉归因保住了对证据敏感的修正，但**不直接恢复语言实现、答案格式和 EOS 行为**。

推理时不用 teacher 也不用辅助视角，就是普通的全图学生。

### 关键实验结果
六个细粒度视觉基准（V*、Zoom、HR-Bench 4K/8K、MME-RW EN/CN），4B 和 9B 两个规模。

**目标构造消融是这篇最核心的证据**（Qwen3.5-4B，Avg₆）：
- 直接用特权 teacher p_T⁺：**75.92**
- 标量收缩：76.19
- 单边（只保留支持方向）无正则：77.06
- 单边 + 正则：77.52
- **VAD 无正则：78.06**
- **完整 VAD：78.32**

从 75.92 到 78.32 是 **+2.40**，而且是逐步递进的——每一层设计都有贡献。V* 上从 89.53 提到 92.15，HR-8K 从 80.12 提到 83.38。

**留出泛化是这篇最有价值的对照**（MMVP/CV/MMStar/POPE 四个未训练基准，相对基座的变化）：
- 4B：GRPO **-4.20**、VA-OPD **-2.72**、V-Zero -0.66、Vision-OPD -0.89、Decomposed OPD +0.05、**VAD +0.24**
- 9B：GRPO -0.93、VA-OPD **-3.18**、V-Zero -1.86、Vision-OPD -1.95、Decomposed OPD -1.36、**VAD +0.23**

**除了 VAD，所有方法在两个规模上都让留出性能低于基座**。也就是说细粒度视觉能力的提升是拿通用能力换来的，而 VAD 是唯一不用换的。这个对比比主表的 +2.4 更有说服力。

**训练效率**：4B 上 VAD 每步 7.84 分钟、67.9 GPU 小时，对比 GRPO 的 9.34 分钟/81.0 小时、**V-Zero 的 13.10 分钟/113.5 小时**，与最快的 Vision-OPD（7.55/65.5）基本持平。反事实的双次 teacher 前向没有带来明显开销。

**散度选择**：JSD 在 4B 上 78.32 略优于前向 KL 77.62 和反向 KL 77.85；9B 上 79.93 vs 79.21/79.56。差距不大但一致。

### 局限性与开放问题
论文有明确的 Limitations，**两条都很到位、且是方法论层面的诚实**：

1. **每次干预只用单一视角对产生的一个对比向量表示**，这可能对**组合式证据**有偏——如果答案依赖多处视觉线索的组合，单个对比方向捕捉不全。多视角或学习式的方向基可能给出更丰富的估计。
2. **当前的投影只带来语义富集，而非可辨识的分离**——归因出来的分量**仍可能保留非视觉的 teacher 效应，残差也仍然是源混杂的**。带 grounding 约束的学习式分解可能给出更干净的归因。

第二条尤其值得肯定：作者明确承认「视觉归因」这个名字承诺的东西（干净分离）并没有真正做到，实际做到的是「富集」。这在方法论论文里是罕见的克制。

我补充两点：其一，**证据移除（x⁻）的构造方式没有在抓取的正文里展开**，而这是整个方法的支点——如何「移除相关证据」（遮挡？模糊？替换？）直接决定了 u_t 的质量。不同的移除方式可能给出完全不同的归因方向。

其二，**绝对水平上，4B 的 78.32 和 9B 的 79.93 仍低于 Gemini 3.1 Pro 的 78.04……实际上是持平甚至略超**（Gemini 3 Flash 77.32、GPT-5.4 75.45），这点值得肯定；但对比 235B 的 Qwen3-VL-Instruct（74.80）时要注意后者未做同类后训练，不是对等比较。

### 启发与应用前景
最有价值的方法论洞见：**当 teacher 的监督信号是多来源混合的时候，加权（决定蒸多强）不如重构（决定蒸什么）**。VAD 的做法是用反事实干预造出一个「因果方向」，再把原信号投影上去。这个模式可以推广到任何「teacher 有特权信息」的蒸馏场景——teacher 见过标准答案、见过环境状态、见过额外模态，都可以用「有/无该信息的两次前向之差」构造干预方向。这与 [5] DAPD 处理特权幻觉、[23] W2S-OPD 用 logit 差分离能力方向，构成了本周关于「在 logit 空间做因果分解」的一组呼应工作。

**「留出集相对基座的变化」应该成为蒸馏工作的标配指标**。这篇的表 3 显示，几乎所有细粒度视觉蒸馏方法都在悄悄损害通用能力（-0.9 到 -4.2），而只报主基准的提升会完全掩盖这一点。

工程上，**反事实双前向几乎不增加成本**（7.84 vs 7.55 分钟）这个数字很实用——teacher 是冻结的，两次前向可以 batch 在一起，比想象中便宜得多。

follow-up：作者指出的两条限制都是好方向，其中「学习式分解 + grounding 约束」如果做成，就能把「富集」变成真正的「分离」。代码已开源。

---

## 29. GST-Bench: Can VLMs Develop Global Spatial Awareness from Video?
**👍 43** · 🏛 字节跳动 Seed / 浙江大学 / 新加坡国立大学 · [arXiv:2608.05747](https://arxiv.org/abs/2608.05747) · [项目页](https://qwerirwq.github.io/GST-Bench/)

### 问题与动机
空间智能是具身 agent 的基础，但现有基准聚焦于**单视角或少数视角下的局部空间感知**，忽略了**在连续长程视觉流上的全局空间意识**——也就是「走过一圈房子之后，能不能在脑子里建立起这个空间的整体地图」。

这个区分很重要：能判断「杯子在桌子左边」（局部）和能判断「我现在站在客厅，刚才在卧室看到的那个杯子在我的右后方」（全局）是两种不同的能力。

### 方法与核心创新
**GST-Bench** 从 **6,790 分钟合成视频**中导出人工验证的 VQA 问题，要求模型：(1) **从输入视频中未出现的新视角做准确空间推断**；(2) **把第一人称观测映射到全局俯视图上**。

数据生成流水线的设计有三个明确特征：
- **按构造强制全局推理**：物体定位题要求目标物体**在探索视频里可见、但从当前视角不可见**，且**当前视角采样自视频轨迹之外**。这样模型既不能靠在单张图里识别目标来解题，也不能靠把查询图直接匹配到某个视频帧。
- **利用仿真的可扩展性和可控性**：场景来自 BEHAVIOR-1K、HyperSim、ArtVIP 等室内仿真资产；相机位姿、物体位姿、可见性、距离、角度、俯视投影全部可从场景几何精确获得。
- **自动生成 + 人工验证**：标注员从两方面验证每个样本的可答性——目标物体在视频里是否可辨识、轨迹外的当前视角是否仍能从探索视频中定位。

配套还有 **GST-Bench-Local**（同样任务形式的局部变体，用于定位差距成因）和 **GST-Train**（训练集，场景与评测集不相交）。

### 关键实验结果
**22 个 SOTA VLM（2B 到 38B，含闭源、开源、具身理解模型）**，人类基线 **79.08**。

**核心发现是差距极大**：最强的 Gemini-3-Pro 只有 **42.68**，落后人类 **36.4 点**。GPT-5 40.85、Gemini-2.5-Pro 40.95、GPT-4o 30.33、Seed1.8 34.04。

差距在各能力上普遍存在，**朝向估计（Ori）最悬殊：21.52 vs 人类 85.00**；全局位置估计 GP_v 是 42.23 vs 93.00。一个值得注意的例外是**自我中心距离估计**——人类基线本身就低（约 41 MRA），最强模型几乎追平（EDist_v 42.00 vs 41.50）。作者的解读很克制：这不表示模型强，而是**恢复绝对度量距离对人和模型都本质困难**。

**开源模型基本在随机线附近挣扎**：Qwen3-VL-32B（30.43）和 InternVL3.5-38B（30.71）看起来领先，但这个优势**主要来自 Top-Down Selection 的 easy/medium 子任务**（TDS_E 上多个模型有 88–99 分），在剩下十个任务上只比随机基线高几个点。**四个模型甚至低于随机**（LLaVA-OV-1.5-8B 19.58、InternVL3.5-2B 19.72、InternVL3.5-4B 19.83、Cosmos-Reason2-2B 19.99），17 个开源和具身模型里有 9 个在随机线三点以内。**具身微调模型相对通用模型没有任何优势**——这个结论对具身领域是个不小的打击。

**GST-Bench-Local 的对照定位了成因**（自我中心方向任务）：Gemini-2.5-Pro 从全局的 23.08 提到 Local-Video 的 41.62（+18.54）、Local-Image 的 66.49（**+43.41**）；GPT-5 从 31.09 提到 60.32（+29.23）/ 65.61（+34.52）。**同样的任务形式下，只要不要求跨帧整合，模型就能做对**——所以失败**不在空间推理本身，而在把长程观测整合成全局一致的场景表示**。

**针对性训练**：Qwen3-VL-8B 在 GST-Train 上微调（混通用多模态指令数据以保住指令跟随能力），平均分从 **25.89 提到 53.52**，**超过所有零样本模型包括闭源系统**。但仍然离人类的 79.08 有 25.6 点差距。

### 局限性与开放问题
论文**没有独立 Limitations 章节**（只有 Conclusion）。这是明显的缺失，因为有几个问题需要正面讨论：

其一，**全部数据来自仿真**。6,790 分钟合成视频、仿真室内场景。虽然仿真给了精确的几何真值（这正是能自动生成答案的前提），但**在仿真视频上测出的空间能力差距，有多少会迁移到真实视频**是完全未知的。真实视频有运动模糊、光照变化、纹理复杂度，也可能让模型表现更差或更好。

其二，**人类基线的采集协议没有说明**。79.08 这个数字是几个人、什么背景、看多长时间视频、能不能回看，全部没交代。而全文最重要的论断（36.4 点差距）完全建立在这个数字上。考虑到 Ori 上人类 85.00 而模型 21.52，这个基线的可靠性至关重要。

其三，**微调后的 53.52 超过所有零样本模型**这个比较不对等——GST-Train 与 GST-Bench 由**同一条流水线生成**，只是场景不相交。同流水线的模板化 QA 意味着微调模型学到的可能包含题型模式而非纯粹的空间能力。论文没有报微调模型在其他空间基准（如 VSI-Bench）上的表现来验证泛化。

其四，**TDS_E 子任务上多数模型能拿 88–99 分**，这个子任务显然过于简单，把它算进 12 项均值会系统性抬高所有模型的分数，掩盖真实差距。

### 启发与应用前景
最有价值的是那个**局部-全局对照实验**——它把「模型空间能力差」这个笼统结论，精确定位到「**跨帧空间整合**」这一个环节。同样的任务形式，给单图能做到 66.49，给视频要求全局整合就掉到 23.08。这个诊断方式（构造同任务形式的局部变体）值得任何做能力评测的人借鉴：**先证明模型有原子能力，再证明它缺的是组合/整合能力**，比笼统地报一个低分有用得多。

**「具身微调模型没有优势」**这个发现对具身 AI 领域有实际意义：Cosmos-Reason2、RoboBrain2.5、Robix 这些专门为物理环境空间推理调过的模型，在全局空间任务上并不比通用 VLM 强。这说明当前的具身微调数据可能都集中在局部感知上。

**「按构造强制全局推理」的数据设计**（目标物体必须视频可见但当前视角不可见，且当前视角在轨迹之外）是个可复用的模板——它用几何约束而非人工判断来保证任务不可被捷径解决。

follow-up：最该补的是**真实视频版本**和**微调模型的跨基准泛化验证**。GST-Train 已作为资源发布。

---

## 30. EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning
**👍 39** · 🏛 上海交通大学 / 腾讯 / 浙江大学 / 新加坡国立大学 · [arXiv:2608.06197](https://arxiv.org/abs/2608.06197) · [GitHub](https://github.com/Within-yao/EnvACE) · [项目页](https://within-yao.github.io/EnvACE/)

### 问题与动机
训练长程工具使用 agent，要么依赖真实或合成的可执行环境（构造和验证成本极高），要么依赖外部模拟器（难以接地）。两条路都把**环境动力学留在策略之外**。

这篇提出一个更激进的问题：**能不能让策略自己把环境动力学装进参数里？**

### 方法与核心创新
**世界排练（world rehearsal）**：训练时策略在**行动**和**排练**两个角色间交替——先生成一个工具调用，然后**自己扮演环境**，产出该动作诱导的响应，再把这个排练出来的响应追加进交互历史，条件在它上面做后续决策。**整条轨迹在没有外部环境的情况下展开。**

两个角色由**同一个共享策略**承担，用 **role-wise GRPO** 联合优化（两个角色用各自的 baseline，但联合更新策略），奖励是任务成功。

通过反复排练，「动作如何塑造环境响应」这个知识被吸收进策略参数，形成一个**直接支持决策的 agent 世界模型**。

**测试时**这个内化的世界模型还能再用一次：策略先做**私有排练**（并行或串行），把排练结果总结成 rehearsal memory，再用这份记忆引导一次在真实外部环境里的**承诺执行**——**不需要额外的外部交互就能获得增益**。

### 关键实验结果
四个基准（BFCL V4、τ²-Bench、VitaBench、FinMCP-Bench），主力是 Qwen3-8B。

**主结果**：Overall **32.91%**，超过所有在三个基准上有完整结果的环境扩展基线——比 EnvScaler-8B 高 0.99%、比 AWM-14B 高 0.37%。分项：BFCL V4 **46.04%**（超 Qwen3-8B 基座 2.00%、超 AWM-8B 1.75%，但**比 EnvScaler-8B 低 1.03%**）；τ²-Bench **36.7%**（第二高，超 EnvScaler-8B 3.8%、AWM-8B 5.5%、AWM-14B 6.0%）；VitaBench **16.0%**（7B–8B 方法里最好，超 EnvScaler-8B 0.2%）。

值得一提的是基线里的 **Simulator-8B 的失败模式很有教育意义**：它在 τ²-Bench 上拿到最高的 38.5% 和 BFCL 的 Irrelevance 86.54%，但 **BFCL V4 整体只有 19.78%**（Multi 项 1.47%、NoLive 32.46%）——一个专门学环境模拟的模型，把行动能力学没了。这正好反证了 EnvACE「两个角色共享参数」设计的必要性。

**规模效应**：从 1.7B 到 8B，BFCL V4 均值从 31.81% 提到 46.04%（**+14.23**），τ²-Bench 从 15.3% 提到 36.7%（**+21.4**）；且**EnvACE 在两个规模上都超过标准 GRPO，8B 上优势更明显**——说明世界排练随模型容量增强而更有效。

**测试时缩放（N=2）**是这篇最有意思的结果：
- 不做 TTS：τ² 均值 31.4、BFCL Multi-Turn 均值 41.9、Overall 36.7
- TTS + 并行 + **基座排练**：32.2 / 41.4 / 36.8（**几乎没有增益**）
- TTS + 并行 + **EnvACE 排练**：**38.0 / 43.9 / 40.9（+4.2）**
- TTS + 串行 + 基座排练：31.0 / 38.8 / 34.9（**低于不做 TTS**）
- TTS + 串行 + EnvACE 排练：34.8 / 42.3 / 38.5

关键对比是「基座排练 vs EnvACE 排练」：**同样的测试时排练流程，用没经过世界排练训练的基座模型来排练，几乎没有收益甚至变差；用 EnvACE 训过的模型排练，才有 +4.2 的提升**。这直接证明了增益来自内化的世界模型，而非「多想一会」这个流程本身。

### 局限性与开放问题
论文有 Limitation 章节，两条：**受算力限制只评到 8B 规模**；**评测主要集中在工具交互任务**，把世界排练扩展到更广的 agentic 设定是未来方向。

我补充几点：

其一，**Overall 的领先幅度极小**——32.91% vs EnvScaler-8B 的 31.92%（+0.99）、AWM-14B 的 32.54%（+0.37）。而且**在 BFCL V4 上实际输给 EnvScaler-8B（46.04 vs 47.07）**。「outperforming environment-scaling baselines in the overall evaluation」这个说法成立，但靠的是在 τ²-Bench 上的优势拉平了 BFCL 上的劣势。0.37% 的领先幅度需要多 seed 方差才能确认，论文没有报。

其二，**「排练出来的环境响应可能是错的」这个根本风险没有被量化**。策略自己扮演环境，如果它对环境动力学的理解有偏差，那它就是在一个**自己想象的、可能不真实的世界里**做 RL。论文没有报排练响应与真实环境响应的一致性指标（比如在有真实环境的任务上对比排练响应和真实响应的匹配度）。这是整个方法最需要验证的地方，却是缺失的。

其三，**训练完全不接触真实环境**是卖点也是隐患：模型只能学到「它已经知道的环境动力学」的强化版本，**无法学到它不知道的**。这从原理上限制了它对新工具、新 API、新领域的适应能力。论文报了 FinMCP-Bench 上的迁移性，但抓取的正文里没有具体数字。

其四，**测试时排练的成本**：N=2 意味着每个任务要多跑两遍完整的想象轨迹，论文没报 token 开销或延迟。

### 启发与应用前景
最有价值的概念是**「让策略同时是环境」**。这跟 [6] SpyRL 的「把不可验证任务变成可验证代理游戏」、[38] 的世界模型综述路线呼应，但走得更远——**它不是造一个外部模拟器，而是把模拟器折叠进策略本身**。好处是环境和策略天然对齐（不存在模拟器与策略能力不匹配的问题），坏处是失去了外部真值的锚。

**「基座排练无效、训练过的排练有效」这个对照**是全文最漂亮的实验设计，值得任何做测试时缩放的人学：要证明增益来自你的方法而非「多算一会」，就该做这个对照。很多 TTS 工作缺的正是这一步。

工程上，**测试时私有排练 + rehearsal memory + 承诺执行**这个三段式，对**外部交互昂贵或有副作用**的场景（真实 API 调用、金融交易、生产环境操作）特别有价值——先在脑子里试几遍，再动手。

follow-up：最紧要的是**验证排练响应的保真度**，以及**混合训练**（大部分排练 + 少量真实环境校准）能否兼得两者优势。代码已开源。

---

## 31. The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads
**👍 39** · 🏛 腾讯 LIGHTSPEED / 香港科技大学 · [arXiv:2608.04570](https://arxiv.org/abs/2608.04570)

### 问题与动机
带持久记忆的个性化 LLM 正在大规模部署，但**它们构建的用户模型是否忠实于证据，从来没被系统检验过**。这篇研究的现象叫 **over-inference（OI，过度推断）**：**LLM 编造超出证据支持范围的用户属性**。

举个具体的：用户只透露了三个事实，模型却在写「周末行程建议」时给你安排了一整套基于刻板印象推断出的兴趣爱好、消费水平和社交偏好。这些属性会**写进记忆**，然后成为后续所有交互的「事实」。

### 方法与核心创新
**MirageBench** 的设计有几个值得注意的地方：

- **150 个 persona，在刻板印象、反刻板印象、中性三类上平衡**——这个平衡很关键，它让「模型是不是在套刻板印象」变成可测量的。
- **6 个个性化任务构成一条「想象力梯度」**——从只需引用已陈述偏好就能完成的（生日礼物），到必须超出已透露的 3 个事实才能完成的（公寓/住房推荐）。
- **四类忠实度分类法**（Grounded 有据 / Reasonable 合理推理 / Stereotype 刻板印象 / Fabrication 编造），由独立 judge（Claude-Opus-4-7）执行。**信度验证做得扎实**：与盲标注员在 400 条 claim 上比对，四分类 Cohen's κ = **0.863**，二分类 κ = **0.900**。
- 排行榜覆盖 **7 个家族的 12 个模型、143,616 条被判定的 claim**。

### 关键实验结果
**OI 是普遍的，没有例外**：12 个模型的 OI 率在 **35%–49%** 之间，跨模型均值 41.6%、claim 加权 41.8%。最好的 Gemini-3.1-pro 也有 35.1%，Claude-Opus-4-6 是 35.4%；最差的 Qwen3-8B 是 48.7%。**Fabrication（纯编造）单项就占了均值的 31.1%**——注意这不是「合理推理」，是判定为编造的。

**最震撼的发现是 Self-Monitoring Inversion（自我监控反转）**：在**模型选择层面**，模型自评的 OI 与 judge 测出的 OI **负相关**（ρ = **-0.60**，p = 0.044；探索性结论，bootstrap 置信区间较宽 [-0.90, +0.06]，n=12）。

具体看 Table 2 的模式：**Qwen3-8B 自评 13.0% 而实测 48.7%（Δ = +35.7，严重低估）**，GPT-4o-mini 自评 20.1% 实测 45.1%（+25.0）；而 Kimi-K2.5 自评 58.2% 实测 43.1%（**-15.1，高估自己的问题**），GPT-5.4-nano -11.4、GLM-5.1 -9.3、Gemini-3.1-pro -8.1。

**也就是说：报告自己过度推断最少的模型，恰恰最容易被判定为编造最多。** 用自我报告的置信度来比较模型是**误导性的**。

但论文的结论很有分寸：**在单个模型内部，自审仍然能不错地给自己的 claim 排序**（AUROC 0.58–0.83，Qwen3.6-plus 是 ρ=0.64/AUROC 0.83，GPT-5.5 是 0.63/0.80）。所以「自审无用」是错的，正确的说法是**「自审的分数不可跨模型比较，但可以在模型内部做相对排序」**。

**任务依赖性符合「可接地梯度」的预测**：公寓/住房 OI 57.8%（有据只占 15.0%、编造 38.0%）、推荐信 48.2%、压力来源 39.8%、周末行程 38.7%、约会资料 28.5%、**生日礼物 27.0%（有据占 40.0%）**。必须超出三个已知事实的任务被刻板印象和编造主导，能靠引用已陈述偏好回答的任务则主要是有据的。

**多轮累积试点**（8 轮对话，2 个 persona 平均）显示推断属性**近似线性增长且很少被修正**：GPT-5.5 从 18.5 涨到 **125.0**（每轮 +15.2），GLM-5.1 从 16.5 到 121.5，Claude-Opus-4-6 从 15.5 到 104.5。**能力越强的模型累积越快**——GPT-5.4-nano 只从 14.5 涨到 15.0（每轮 +0.1），Qwen3-8B 每轮 +1.4。这个反向关系值得警惕：**越好用的模型，越会往你的档案里堆没根据的东西**。

### 局限性与开放问题
论文没有独立 Limitations 章节（只有 Conclusion），但在正文里对统计强度做了诚实标注（把 ρ=-0.60 明确标为「探索性」并给出宽置信区间）。我的观察：

其一，**n=12 的秩相关，p=0.044，置信区间跨越 0**。作者自己标注了这一点，但「Self-Monitoring Inversion」被写进标题和摘要，作为核心发现推出。**这个结论的统计支撑是薄弱的**——12 个点的 Spearman 相关，置信区间 [-0.90, +0.06] 包含正值，意味着不能排除正相关。读者应该把它当作一个值得追查的假设，而非已确立的结论。

其二，**judge 是 Claude-Opus-4-7，而被评的模型里有 Claude-Opus-4-6**。同家族评判存在潜在偏袒，虽然 Claude-Opus-4-6 的 OI 排名第二好（35.4%）不算异常，但这个混淆没有被讨论或用第二个 judge 交叉验证。κ=0.863 只验证了 judge 与人类的一致性，没有验证跨 judge 的一致性。

其三，**多轮累积试点只有 2 个 persona**。论文自己用了「pilot」这个词，但表 4 的数字（+106.5 vs +0.5，差 200 倍）在正文和摘要里被当作实质发现引用。2 个样本的均值不足以支撑「能力越强累积越快」这个跨模型规律。

其四，**「编造」的判定标准依赖 judge 对「证据支持范围」的理解**。而个性化本身就是要做超出字面证据的推断——「合理推理」和「编造」的边界是模糊的，且随场景变化。四分类 κ=0.863 说明标注者之间一致，但不说明这个边界划得对。

### 启发与应用前景
这篇最有实践价值的结论是**「用外部验证而非模型自报作为可信个性化的基础」**。具体到产品设计：

1. **不要用模型的自评置信度做记忆写入的门控**。这是当前很多记忆系统的默认做法（模型说「我有信心用户喜欢 X」就写进去），而这篇的数据说明自评在跨模型层面是反向信号。
2. **做 provenance tracking（来源追踪）**——每条写入记忆的用户属性都标注它来自「用户明确陈述」「合理推理」还是「模型推断」，让下游可以按可信度过滤。
3. **对高想象力梯度的任务加约束**。公寓推荐这类任务 57.8% 的 claim 是过度推断，而生日礼物只有 27.0%——**任务类型是 OI 风险的强预测因子**，可以据此分级管控。

**「可接地梯度」这个任务设计思路**也值得抄：把任务按「能否仅凭已知事实完成」排序，就能得到一条清晰的风险曲线，比笼统地测「幻觉率」精确得多。

**近似线性累积、很少修正**这个观察对做长期记忆的系统是个警报：如果没有主动的修正和过期机制，8 轮对话后记忆里就有上百条无据属性，而它们会持续影响所有后续交互。这与本周 [20] Skill-α 的「只增不删的知识累积会主动变坏」形成呼应。

follow-up：最该做的是**把 n 从 12 扩到更多模型**来确认 Self-Monitoring Inversion，以及**多轮累积扩到几十个 persona**。MirageBench 承诺全量发布。

---

## 32. Learning from Failures: Retrieval-Centric CoT via Hard Negatives for Unified Multimodal Retrieval
**👍 38** · 🏛 格灵深瞳 / 中国人民大学 · [arXiv:2608.06060](https://arxiv.org/abs/2608.06060) · [GitHub](https://github.com/deepglint/UniME-R1)

### 问题与动机
统一多模态检索里，直接把原始多模态输入编码成一个向量，常常丢失细粒度的判别线索，导致语义相似的候选混淆。近期的 **Reasoner–Embedder** 方法用 LVLM 先生成 caption、查询扩展或 CoT 再做嵌入来缓解这个问题。

但论文指出一个精准的缺陷：**这些推理通常只从查询本身导出**——它解释的是「查询描述了什么」，而不是「检索器误解了什么」。产出的 CoT 可能只是复述显著内容，却漏掉了把目标与混淆候选区分开的那个微妙差别。

### 方法与核心创新
**UniME-R1** 是一个 **embedder–adviser 框架**，核心主张是：**有效的检索推理应该条件在检索反馈上**。

流程是：先做一次初始检索，adviser **逐个分析检索回来的候选**，识别出 embedder 混淆的判别线索，然后二选一：
- 如果目标已在初始 top-k 里 → **直接重排**；
- 否则 → 生成 **RC-CoT（Retrieval-Centric Chain-of-Thought）**修正检索方向，用双模式 embedder 做**全库重检索**。

输出格式上，RC-CoT 明确分成 `<cot_focus>`（总结候选暴露出的混淆点）和 `<cot_answer>`（修正后的描述）两部分。

训练侧三件事：挖硬负例来**模拟真实的检索失败**、联合优化直接检索和 RC-CoT 增强检索、用监督学习 + **面向检索的强化学习**（judge / rerank / RC-CoT 三个奖励）把 adviser 与检索结果对齐。

### 关键实验结果
**RC-CoT 的核心主张被消融直接验证**（固定 adviser 为 Qwen3-VL-235B，MMEB-V2 Overall）：
- 只用查询的 CoT：**65.6**
- 条件在**随机候选**上：66.5（只有小幅提升）
- **RC-CoT（条件在真实 top-k 上）：68.5**

比只用查询高 **2.9 点**，比随机候选高 2.0 点。**「真实检索反馈是必要的」这个论断被随机候选这个对照坐实了**——不是「多给点上下文」有用，而是「给真实的失败证据」有用。去掉 `<cot_focus>`（显式的失败诊断）会从 68.5 掉到 67.9，说明**总结混淆点的价值超出了最终的修正描述本身**。

**rerank-or-retrieve 策略消融**（MMEB-V2 Overall）：
- 初始检索：63.5
- 总是重排：69.1；总是重检索：69.0（两条路各自都有效且**互补**）
- **UniME-R1 自适应路由：69.9**（比两个固定策略高 0.8/0.9，比初始检索高 **6.4 点**，视频任务高 **13.0 点**）
- **Oracle 路由：72.2**——比学到的路由还高 **2.3 点**，视频上差 3.2 点。这个上界很诚实地说明**路由决策还有可观空间**。

**双模式 embedder 消融**：判别损失 62.3 → 加生成损失 62.6（图像几乎不变，视频 +0.7、视觉文档 +0.9）→ **加挖掘的硬负例 63.5**。两个组件角色不同：生成损失教 embedder 把 RC-CoT 转成检索有效的表示，硬负例锐化两种模式的决策边界。

**面向检索的 RL 消融**：SFT adviser 68.8 → 完整 GRPO **69.9**（+1.1）。去掉任一奖励都会退化——去掉 judge 或 rerank 奖励各降 0.4，去掉 RC-CoT 奖励降 0.3。

**GRPO 对路由的校准效果**很直观：把「路由到重排的比例」与「GT@Top-5（目标确实在初始候选里的比例）」的差距，图像上从 17.3 点压到 **5.3 点**、视频上从 18.7 压到 **4.7 点**。但 VisDoc 上差距反而从 2.4 涨到 5.7（轻微过度路由到重排）。

**推理效率是很实在的卖点**：MMEB-V1 上 3,600 查询 × 111,384 候选，Embed-RL 给查询和候选都生成 CoT，**每候选 0.27 秒**；UniME-R1 只给查询生成 RC-CoT，每个候选用 `<dis_emb>` 编码一次，**候选延迟降到 0.01 秒（快 27 倍）**。查询侧从 0.28 秒略增到 0.34 秒。对大规模或频繁更新的候选池，候选侧成本主导索引构建和维护，这个差距是决定性的。

**多轮推理**：从 1 轮到 2 轮，Overall 从 69.9 到 70.5（视频 +1.3 最大）。但作者判断增益相对额外推理成本不划算，**默认仍用单轮**——这个克制值得肯定。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我的观察：

其一，**Oracle 路由的 72.2 说明学到的路由离最优还差 2.3 点**，作者诚实报了这个上界，但没有分析路由失败的模式。而路由是整个框架的枢纽——判错了就走了错误的分支。

其二，**adviser 用 Qwen3-VL-235B 做消融**（虽然最终系统的 adviser 规模没在抓取部分明确），这个规模的 adviser 对每个查询做候选分析，实际部署成本不低。查询侧 0.34 秒是在什么硬件上、用多大的 adviser 测的，没有交代。

其三，**RL 的总增益只有 1.1 点**（68.8 → 69.9），而三个奖励各自的贡献是 0.3–0.4 点，**接近实验噪声量级**。论文没有报多 seed 方差。

其四，**VisDoc 上 GRPO 让路由校准变差**（2.4 → 5.7），作者提了一句「轻微过度路由」就带过了，没有分析为什么这个模态方向相反。

### 启发与应用前景
最核心的洞见值得单独记：**推理应该条件在系统的实际失败上，而非条件在输入上**。「随机候选只提升 0.9 点、真实候选提升 2.9 点」这个对照把这件事说透了——**同样是给上下文，给真实的错误证据才有信息量**。

这个原则的适用面远超检索：RAG 系统可以让模型看到「检索回来的错误文档」再改写查询；代码 agent 可以让模型看到「失败的测试输出」再改代码（这已是常规做法）；推荐系统可以让模型看到「曝光未点击的候选」再修正用户意图。共同点是**用系统的失败反馈闭环，而不是让模型对着输入空想**。

**「只给查询侧做 CoT、候选侧一次编码」这个非对称设计**是纯粹的工程智慧：候选池大且需要预建索引，查询是实时的且数量少。27 倍的候选侧加速几乎是免费的架构选择，任何做检索的团队都该检查自己有没有在候选侧做昂贵计算。

**自适应路由 + oracle 上界**这个报告方式也值得学——同时给出「学到的策略」和「完美策略」，让读者知道还剩多少空间。

follow-up：路由失败模式的分析是最直接的改进入口（2.3 点的空间摆在那）；另外可以试着把 RC-CoT 的失败诊断做成显式的结构化输出（而非自由文本），可能更容易被 embedder 利用。模型和代码已开源。

---

## 33. ChronoVision: Temporal Reasoning via Latent State Reconstruction
**👍 38** · 🏛 伊利诺伊大学厄巴纳-香槟分校 / 宾夕法尼亚大学 / PediaMed AI · [arXiv:2608.05631](https://arxiv.org/abs/2608.05631)

### 问题与动机
多模态大模型擅长**被动感知**，但在需要多步时序推理的复杂视觉认知任务上很差。论文的诊断是：**这种退化很大程度源于基于语言的推理本身的模糊性**——语言常常无法准确表达连续的视觉变换。

任务形态是 **Vbvr-VQA**：给定视频的第一帧和 6 个打乱的候选帧，重建视频的时序顺序。评价是**精确匹配**——只有整个 6 帧序列完全正确才算对。典型例子是汉诺塔式的积木重排：需要理解「每次只能移动最上面的块」这个规则，再推断出唯一合法的操作序列。

### 方法与核心创新
两阶段训练，两个机制都是把推理**锚定到视觉表示**上而非任其在语言空间漂移：

**(1) SFT + ROI Attention Locating**：为防止注意力分散和语义噪声，在输出序列之前先让模型生成一个被特定结构 token 包裹的**简洁语义定位线索**。然后在指定的中间层 ℓ 上，用生成的 locate token 作 query、初始查询图像 token 作 key，提取 text-to-image self-attention 权重，用真值包围框 R_tar 做**注意力凝聚损失**：L_AC = −log(s(R_tar))，其中 s 是框内归一化平均注意力占比。总目标是 L_SFT + α·L_AC。

论文明确在附录讨论了**为什么选注意力对齐而非显式回归包围框坐标**——这是个有意识的设计选择而非偷懒。

**(2) RL + 隐式过程接地**：用 GRPO，奖励解耦成三个各自有界于 [0,1] 的分量，**全部无需人工标注**：
- **最终结果奖励 R_out**：整个序列精确匹配才给 1；
- **隐空间接地过程奖励 R_latent**：在句子级提供稠密监督。第 k 个推理步时，把视觉 token 的上下文化隐状态喂进 RVH 生成隐空间特征 Z_k，再与 6 个打乱候选的视觉特征 V_i 算**最大余弦相似度**（归一化到 [0,1]）。**这把文本推理锚定到底层图像特征上**——模型说的每一步都必须对应某个真实候选帧。
- **无监督视觉聚焦奖励 R_focus**：用视觉 token 上自注意力分布的**香农熵**构造归一化负熵奖励，防止注意力坍缩。

### 关键实验结果
**Vbvr-VQA 主结果**：ChronoVision **总体 73.2%**（ID 74.8%、OOD 71.6%），而最强闭源基线 Claude Opus 4.6 是 55.8%（ID 50.8、OOD 60.8）。其余：Qwen 3.5 397B 49.4%、Gemini 3.0 Flash 46.6%、GPT o3 46.0%、Gemini 3.1 Pro 41.8%、GPT-5.4 33.8%、GLM-4.6V 21.4%、**基座 Qwen 3.5 9B 只有 14.2%**。

也就是说：**一个 9B 模型从 14.2% 提到 73.2%，超过了 397B 的开源模型和所有闭源前沿模型**。分类目看，最难的 Flu.（流体？）类目上基座是 10.8%，ChronoVision 达到 72.3%，而 Claude Opus 4.6 只有 30.8%。

**通用能力不退化**：7 个标准多模态基准上，ChronoVision 与原版 Qwen 3.5 9B **严格可比**（MMMU 78.8 vs 78.4，MathVista 85.9 vs 85.7）——注意力定位和重构目标提升时空推理**没有损害通用视觉-语言理解**。

论文还做了相当广泛的附加消融（附录 C）：部分匹配评估、隐表示演化、CoT 影响、前缀敏感性、中间状态扰动、隐序列干预、线性探测、与相关时序推理方法的对比。

### 局限性与开放问题
论文有 Limitation 章节（正文抓取里位置标注了但内容未完整展开）。我从数据和设计里看到的问题：

其一，**73.2% vs 14.2% 这个 5 倍提升的对比性质要看清**。基座 Qwen 3.5 9B 是零样本，ChronoVision 是在**同一个任务上做了 SFT + RL** 的专用模型。和闭源模型的比较同样是「专门训练 vs 零样本」。这个数字证明的是「针对性训练极其有效」，不是「模型能力更强」。**OOD 分割也来自同一个 Vbvr-VQA 基准**（只是类目不同），不是真正的跨任务泛化。

其二，**Vbvr-VQA 是什么、多大规模、怎么构造的，正文抓取部分完全没有交代**。从类目缩写（Flu./Cry./Vis./Men./Trans.）和汉诺塔的例子看，任务偏合成和规则化。评测是**整序列精确匹配**，6 帧全排列有 720 种，随机基线约 0.14%——所以 14.2% 的基座已经远好于随机，但 73.2% 在这样一个高度结构化的任务上是否代表通用时序推理能力，存疑。

其三，**R_latent 这个过程奖励有循环风险**：它奖励「推理步的隐表示与某个候选帧的视觉特征相似」，但**这不保证相似的是正确的那一帧**（取的是 max over V_i）。模型完全可以学会让每一步都强烈对应某个（可能错误的）候选来刷奖励。论文没有报这个奖励与最终正确性的相关性。

其四，**三个奖励的权重 ω₁/ω₂/ω₃ 没给具体值**，也没有敏感性分析。

其五，**通用能力「严格可比」的说法**（78.8 vs 78.4、85.9 vs 85.7）实际是略微提升，但差距在 0.2–0.4 点，属于噪声范围，说「不退化」是准确的，说「增强」则言过。

### 启发与应用前景
最有价值的机制是 **R_latent 这个「把文本推理锚定到视觉特征」的过程奖励**。它解决的是一个真问题：**思维链在多模态任务里会脱离视觉证据自由漂移**。用「推理步的隐表示必须能对应某个真实视觉输入」作为约束，是个比「让模型输出包围框」更轻量、更不依赖标注的做法。这个思路可以推广到任何「推理必须持续接地在感知输入上」的场景——视频问答、具身规划、图表推理。

**注意力凝聚损失（L_AC）**也是个可复用的组件：与其让模型显式输出坐标（需要额外的检测头和标注格式），不如**直接约束某个中间层的注意力落在目标区域内**。代价只是一个 log 项，且不改变输出格式。

**无监督视觉聚焦奖励（负熵）**防注意力坍缩，是个零标注成本的正则，适用于任何长程多模态推理的 RL。

follow-up：最该补的是**跨基准泛化验证**——在 Vbvr-VQA 之外的时序推理基准（如 TempCompass、VSI-Bench 的时序子集）上测，才能判断学到的是通用时序推理还是任务特定模式。另外 R_latent 的 max 操作应该换成与正确候选对齐的监督形式，或至少报告它的对齐率。

---

## 34. PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning
**👍 38** · 🏛 北京理工大学 / 美团 / 中国科学院自动化研究所 / 清华大学 · [arXiv:2608.01837](https://arxiv.org/abs/2608.01837) · [项目页](https://pcsd.vercel.app/)

### 问题与动机
Agent RL 的稀疏奖励问题很直接：一条几十轮的轨迹只拿到一个结果级信号。on-policy 自蒸馏（OPSD）用特权 teacher 提供稠密的 token 级监督来补这个缺口，但**teacher 并非在每个位置都可靠**。

已有方法的处理方式各有缺陷：**依赖孤立的 token 级差异**（对噪声敏感——某个位置 teacher 恰好更自信，不代表这里真的有信息）；或者**给整个 step 分配一个共享权重**（忽略了位置差异——同一个动作里不同 token 的重要性并不相同）。

### 方法与核心创新
**PCSD** 的核心思路一句话：**token 级蒸馏权重应该来自「teacher 支持信号的局部持续性」，而非单点强度。**

具体机制四层：
1. **自适应窗口 + 指数衰减聚合**——捕捉「持续的相对 teacher 支持」。窗口在 N_min=1 和 N_max=8 之间自适应（由局部方差决定，阈值 τ_low=0.05 / τ_high=0.5），衰减因子 α=0.8 使邻近 token 权重更高（proximity-aware）。
2. **趋势感知调制**——单边地衰减「局部正在下降的支持」（负趋势调制强度 γ=0.3）。teacher 的支持如果正在减弱，说明这段信号不可信。
3. **sigmoid 门控**产生连续权重（锐度 β_gate=5.0）。
4. 该目标与 **GRPO 联合优化**，把稠密的 teacher 指导和稀疏的环境反馈结合。

关键是**推理时不用任何 skill**——teacher 的特权信息只在训练时用。

### 关键实验结果
**ALFWorld Overall（Qwen2.5-3B-Instruct）**：PCSD **90.6%**，超过 GRPO 的 75.0%（**+15.6**）、SDAR 的 84.4%（+6.2）、GRPO+OPSD 的 81.2%、RLSD 的 79.7%。Qwen3-1.7B 上相对 GRPO **+13.3**、相对 SDAR +5.5。

分类目看差距集中在**最难的几类**：Cool 从 SDAR 的 75.0 提到 **94.4**，**Pick2 从 84.2 提到 100.0**，Heat 从 61.9 提到 83.3。而 Look 类目上 PCSD 只有 63.6，**低于 GRPO+OPSD 的 82.4 和 RLSD 的 75.0**——这个反向结果论文没有讨论。

**WebShop** 上 Score 85.0（最好）、Acc 67.2（与 SDAR 并列最好），但**领先幅度很小**（SDAR 是 83.4/67.2）——作者的措辞「remaining competitive」是准确的。

**未见 ALFWorld 分割上比 GRPO 高 15.8 点**，泛化性得到验证。

**组件消融**（ALFWorld Overall，完整版 90.6）：
- 固定 N=1（逐点加权）→ **82.8**（-7.8）：这正是「孤立 token 差异」的做法，掉得最多；
- 固定 N=4（固定局部窗口）→ 88.3（-2.3）：窗口有用，但自适应更好；
- 去掉趋势调制 → 83.6（-7.0）；
- 去掉指数衰减（改用均匀加权）→ 85.1（-5.5）。

**蒸馏系数敏感性**是个 U 型：λ=0 时 75.0（就是纯 GRPO），λ=0.005 时 87.5，**λ=0.01 时 90.6**，λ=0.05 时掉回 83.6。**过强的蒸馏会破坏 teacher 监督和奖励优化之间的平衡。**

### 局限性与开放问题
论文有 Discussion 章节明确讨论了设计取舍：**PCSD 对局部聚合和门控使用固定超参，并保持特权 teacher 冻结**。这是为了隔离「持续一致性加权」这一个变量的效果，避免把策略优化与变化的 teacher 或加权机制耦合在一起。**代价是对演化中的轨迹统计和 teacher 可靠性变化的适应能力降低。**这个自我评估是准确且克制的。

我补充几点：

其一，**超参数量多且全部固定**：N_min、N_max、α、τ_low、τ_high、γ、β_gate、λ 共 **8 个**，只有 λ 做了敏感性分析。而 λ 的曲线本身就很陡（0.005/0.01/0.05 对应 87.5/90.6/83.6），说明这套机制对超参敏感。剩下 7 个超参的鲁棒性完全未知，跨环境迁移大概率需要重调。

其二，**评测环境只有 ALFWorld 和 WebShop 两个**，而**优势几乎全部集中在 ALFWorld**（WebShop 上只是持平）。ALFWorld 是文本模拟的具身环境，WebShop 是购物环境，两者都是相对成熟的小规模基准。方法在真实工具环境、代码 agent 上是否成立，完全没有验证。

其三，**规模只到 3B 和 1.7B**。这在本周的 agent RL 论文里算是最小的。

其四，**与同期工作的关系**：本周有 [10] AgentOPSD（用递归贝叶斯信念修正做 turn 级信用分配）、[5] DAPD（用信息条件匹配消除特权幻觉）、[28] VAD（用反事实归因重构目标）、[43] SAF-OPD——**至少五篇都在处理「特权 teacher 的信号该怎么用」这个问题**。PCSD 与 AgentOPSD 的环境（ALFWorld/WebShop）和基线（SDAR、RLSD）高度重合，但**两篇没有互相引用或对比**（AgentOPSD 的 ALFWorld 7B 成绩是 89.1%，PCSD 的 3B 成绩是 90.6%，规模不同无法直接比）。这个领域正在快速拥挤，缺少统一的对照。

### 启发与应用前景
最有价值的洞见是**「持续性比强度更可信」**：判断一个 token 位置上 teacher 的指导是否可信，看的不是该位置的单点差异有多大，而是**这个支持信号在局部邻域内是否持续存在**。这是个很朴素但有力的去噪原则，适用于任何「逐点信号噪声大、但真信号有局部连续性」的场景——过程奖励模型、注意力显著性、异常检测。

**趋势调制（单边衰减下降中的支持）**这个细节贡献了 7.0 点，值得单独记：不只看当前值，还看它的变化方向。正在减弱的支持应该被打折。

**λ 的 U 型曲线**（0 → 0.005 → 0.01 → 0.05 对应 75.0 → 87.5 → 90.6 → 83.6）给出了一个实用的调参起点，也提醒：**蒸馏和 RL 的配比有明确的最优点，过强会互相拖累**。

follow-up：作者自己提的方向很对——**从轨迹统计、teacher 不确定性和环境反馈中学习上下文相关的聚合和门控参数**，把 7 个固定超参变成自适应的。这既解决超参问题，也解决论文承认的适应性缺陷。

---

## 35. CADENA: Stepwise CAD Reverse Engineering
**👍 38** · 🏛 莫斯科国立大学 / Innopolis 大学 / FusionBrain Lab · [arXiv:2608.00799](https://arxiv.org/abs/2608.00799) · [GitHub](https://github.com/zhemdi/cadena)

### 问题与动机
CAD 是现代工程的基础，但把已有形状转成可编辑模型仍然需要大量专家工作。**多数 AI 系统一次性吐出整个 CAD 程序，从不检查中间几何**。而人类工程师是一个特征一个特征地搭，**每做完一个操作就看一眼还剩什么没建模**。

这个差别不是风格问题，而是能力问题：单遍生成无法根据「当前建到哪一步了、还差什么」来决定下一步。

### 方法与核心创新
**CADENA**（西班牙语「链」）把 3D 网格重建为参数化 CAD 程序，**一次生长一个操作，每一步都把目标与当前预测几何做对比**。

监督来自一个**基于规则的程序化生成器**，它采样有效的 CAD 程序连同它们产生的网格，覆盖 extrude、revolve、loft、sweep、shell、hole、gear、spring、edge 等操作。因为生成器**拥有完整的构造历史**，它可以直接产出逐步监督：**把程序在某个检查点切开，冻结前缀并执行得到当前状态，后续部分提供目标操作**。每个训练样本因此是一个单步——目标形状 + 前缀产生的部分构造 + 下一个操作的代码，**与推理时策略消费的形式完全一致**。

训练两阶段：**热身阶段 186 万样本**（最多两个操作的短程序，教 DSL 和「残差几何 ↔ 操作参数」的对应关系）；**主阶段 1800 万样本**（最多十二个操作，池化自长度递增的生成器运行：2/4/6/8/10/12 操作），让模型见到**各种深度的部分构造**而非只是程序开头。策略是 Qwen2-VL，teacher-forced 下一操作预测，每阶段 2 epoch。

之后用 **RL 针对程序环境微调**：奖励是**已建实体与目标网格之间的体积 IoU**——直接对着执行出来的几何优化。

另建 **CADENA-Bench**：3396 个机械零件，六个零件族。

### 关键实验结果
**外部基准（DeepCAD / Fusion360 / MCB）**：CADENA-RL 全面领先。DeepCAD 上 IoU **96.1%**（次优 CADEvolve 92.4）、GMS 97.0、**无效率 0.3%**；Fusion360 IoU **94.1%**（次优 CADFit 88.2）、无效率 1.2%；**MCB 上差距最大**——IoU **88.2%** vs 次优 CADFit 的 75.8，CD₃₀ₖ **0.093** vs 0.365（**快 4 倍的精度**），无效率 **0.7% vs CADFit 的 29.7%**。

**CADENA-Bench 揭示了公开基准的严重低估**（这是全文最有价值的发现）：从 DeepCAD 转到 CADENA-Bench，**每个学习式方法都损失约一半的 GMS**——cadrille 94.8 → 49.8、CAD-Recode 92.9 → 48.9、CADEvolve 95.3 → 52.9、CADReasoner 94.9 → 52.9。CADENA-RL 从 97.0 掉到 67.0，**但它的领先幅度从 DeepCAD 上的 1.7 点扩大到 CADENA-Bench 上的 12.2 点**。

作者的结论说得很到位：**「从 sketch–extrude 语料构建的测试集，低估了它们被认为在衡量的那个任务的难度。」**

**RL 的贡献**（贪心解码，DeepCAD/Fusion360/MCB）：IoU 从 SFT 的 91.7/88.8/75.2 提到 **96.1/94.1/88.3**；**无效率从 2.63/3.77/12.04 降到 0.35/1.22/0.74**。对着执行几何做 RL，**同时提升了精度和合法率**——这是很干净的双赢。

**Vision2Code（BenchCAD）**：CADENA-RL voxel IoU **0.910**、无效率 0.9%，而前沿视觉模型 GPT-5.6 Sol 是 0.706、Gemini 3.1 Pro 0.355、Claude Opus 4.7 0.279。不过论文明确标注**「这个对比不是同类比较，不应被读作排名」**——CADENA 输入的是网格，其他模型输入的是图像。

### 局限性与开放问题
论文的 Limitations 是本周写得最好的之一，**主动列出对自己不利的证据**：

1. **不是每个零件族都领先**——齿轮和轴承上 CADFit 的直接曲面拟合达到 61.3 GMS，超过 CADENA 的 58.1。作者还解释了原因：**对着目标优化几何的方法不依赖训练分布覆盖旋转阵列零件**。
2. **还没有报告在自己语料上训练的单遍模型**，这本来能分离「逐步推理」和「训练数据」各自的贡献。作者承诺后续版本补上——**这是最关键的缺失对照，作者自己点破了**。
3. **产出 CadQuery，继承其局限**：角落情况下 CadQuery 能执行的构造树在工业 CAD 软件里没有忠实对应，导出的程序不保证能迁移。

另外还有一段**「回路之外的失败模式」**：策略只观察固定八视角协议，**从所有标准视点都看不见的几何对它是不可见的**；DSL 无法表达的特征被用拉伸堆叠近似；**齿轮齿数这类可数特征是被近似而非被计数的**——少一个齿留下的残差几乎和正确零件一样；**早期操作若定错了平面，后续步骤是绕过它而非撤销它**。逐步推理还要每个操作后执行和渲染，**每个零件的成本高于单遍前向**。

甚至连**「无法对比的方法」都单列了一节**（A.6），逐个说明为什么：SOV-CAD 只放了评测代码没有权重；CADFS 用 Onshape 专有语言需要每个零件走一次托管内核；Zero-to-CAD 没放渲染器，作者自己复现的渲染只达到作者原图 82% 的效果（55.6 vs 68.0 IoU），**「82% 的天花板下列出的数字会把他们的方法和我们对他们渲染器的复现混淆，所以我们省略它」**。作者写道：「**沉默的省略与不利的结果无法区分**。」

这个诚实度在本周 53 篇里是最高的。

我唯一想补充的是：**训练数据完全来自程序化生成器**，1986 万样本全是合成的。这既是优势（有完整构造历史，能产出逐步监督）也是风险——**真实机械零件的设计意图分布与规则生成器的采样分布可能有系统差异**，而 CADENA-Bench 也是作者自建的。零件族上的表现差异（springs & fasteners 63.4 强、gears & bearings 58.1 弱）可能正是这种分布偏差的体现。

### 启发与应用前景
**「用拥有构造历史的程序化生成器产出逐步监督」**是这篇最可复制的方法论。它绕开了逐步监督最难的部分——**你通常拿不到中间状态的标注**。但如果数据是你自己生成的，中间状态是免费的：把程序在任意点切开，前缀执行得到状态，后缀就是目标。这个思路适用于任何「产出是程序/序列且可执行」的任务：代码生成、SQL 构造、化学合成路径、机器人技能序列。

**对着执行结果做 RL 同时提升精度和合法率**（IoU +12 点、无效率从 12% 降到 0.7%）是个强证据：**当环境能真正执行你的输出时，用执行结果做奖励比任何代理指标都好**。

**CADENA-Bench 的教训值得整个领域记住**：所有方法在 sketch–extrude 语料上都是 90+ GMS，换到真实机械零件全部腰斩到 50 左右。**基准饱和不代表任务解决**，只代表基准太简单。任何在公开基准上刷到 95% 的领域都该问一句：我的测试集是不是从一个过于狭窄的语料构造的？

follow-up：作者自己指出的最重要缺失就是**单遍 vs 逐步的对照**（同样训练数据）；另外「CADENA 与 CADFit 的失败模式互补，两者结合优于择一」这个观察是个明确的工程方向。代码、权重、benchmark 全部开源。

---

## 36. Scaling Properties of Text Conditioning in Visual Generation
**👍 38** · 🏛 字节跳动 Seed · [arXiv:2607.29679](https://arxiv.org/abs/2607.29679) · [GitHub](https://github.com/heheyas/context-scaling) · [项目页](https://heheyas.github.io/context-scaling)

### 问题与动机
LLM 靠模型规模、数据、算力三轴 scaling 前进，文生图也照搬了这套配方（更大的扩散主干、更重的训练、更大的图文语料）。但这个类比掩盖了一个基本的不对称：**扩散损失并不随自然语言提示的 token 数增长**。也就是说，文本条件这一轴从来没有被真正测量过 scaling 性质。

论文的实证起点很尖锐：**把提示词单纯拉长，会让所有被评测的开源模型性能下降**——Qwen-Image、HunyuanImage 3.0、BAGEL、FLUX.1 Dev、Emu3 **无一例外，最终都低于它们自己最短 caption 的结果**。甚至在同一条散文阶梯上训练的对照模型，也只是略有提升就饱和了。

### 方法与核心创新
核心发现是：**收敛后的扩散损失随提示中「结构化语言的量」而 scale，而非随 token 数。**

为了量化「结构化语言」，论文引入两个互补度量：**GPG（白盒似然度量）**和 **ED（黑盒属性度量）**。控制训练实验显示：**收敛扩散损失随 GPG 近似线性下降（r = 0.984），随 ED 遵循幂律（∝ ED^0.207）**。

Figure 1 右侧的两组图把机制说得很清楚：**散文在信息量上饱和（约 90 GPG 后每加字段才有新事实，散文则停滞），而结构化提示持续增益**；但在**信息量匹配时，两者落在同一条拟合线上**——说明起作用的确实是信息，不是形式本身。

基于这个 scaling 性质，论文分两头改进：
- **可扩散性（diffusability）**：用**从图像导出的语义和几何标注**构造结构化提示（scene + style + N 个元素，每个元素带 bbox、material 等字段）；
- **可提示性（promptability）**：训练一个 prompter，走**监督微调 → 冷启动 → verifier-gated on-policy 蒸馏**三步，把用户的简短请求扩展成信息密集的结构化提示。

一个副产品是**零样本编辑能力**：因为结构化提示把图像因子暴露成可编辑字段，改一个字段再重新生成，就能只改变物体位置、材质、场景或全局风格，而保留其余构图。

### 关键实验结果
**固定主干探针实验设计得很干净**：用同一个 Qwen-Image 主干和同一个种子，从自然语言（NL）和结构化提示（SP）两种 caption 重建一张留出的参考图，各取四个长度档。**NL caption 保持相同的实体和关系但长度递增（470 → 695 → 1322 → 2130 token），重建质量却是平的**（DINOv3 0.51/0.48/0.46/0.51，SigLIP2 0.36/0.36/0.33/0.36，LPIPS 0.68/0.69/0.69/0.68）；而**逐步恢复 SP 字段，三个指标全部改善**（DINOv3 0.41 → 0.60 → 0.61 → 0.65，LPIPS 0.72 → 0.62 → 0.60 → 0.59）。

**这个对照是全文的核心证据**：同样的实体、同样的关系、同样的主干、同样的种子，**只有信息组织方式不同，重建质量差 0.24 个 DINOv3 余弦**。

**GSB 净偏好曲线**（VLM 好/同/差净偏好，150 条提示，每个系统与**自己最短 caption 的输出**比较，所有提示增强器关闭）：五个开源模型的曲线**全部下行到负值**；在结构化提示 schema 下，**净偏好随 caption 长度单调上升**；再把写提示的 prompter 微调（finetuned structured），相对零样本结构化 prompter 又有一大截领先。

**最终系统**在几乎所有组合性、推理和世界知识基准上**超过所有被评测的开源模型**，在多数评测上**匹配或超过最强的闭源模型**。

### 局限性与开放问题
论文是技术报告，抓取到的内容里**没有独立 Limitations 章节**。我的观察：

其一，**结构化提示的构造依赖从图像导出的标注**（bbox、material 等）。训练时这没问题（有图），但**推理时用户只给一句话**——所以整个系统的能力最终压在那个训练出来的 prompter 上：它要凭空「想象」出合理的 bbox 和材质字段。**prompter 编错了字段，扩散器就会忠实地渲染错误的构图**。论文报了 finetuned prompter 的增益，但没有报 prompter 的字段准确率或错误模式。

其二，**GSB 是「与自己最短 caption 比」**，这个设计的好处是隔离了「拉长带来的增益」，但**它不衡量绝对质量**。所以「所有开源模型拉长都变差」这个结论是关于**相对变化**的，不能直接推出「短提示更好」。

其三，**FLUX.1 Dev 的文本编码器在 512 token 后截断**，论文用虚线标注了这一点，处理得诚实——但这也意味着它的下行曲线部分是架构限制而非信息饱和。

其四，**r = 0.984 这个相关系数**是在「控制训练运行」上测的，训练运行的数量没有在抓取部分给出。单一超参配置下的高相关，未必意味着跨模型跨规模的普适定律。

### 启发与应用前景
这是本周对**提示工程**最有理论价值的一篇。核心结论应该被广泛知道：**「把提示词写长」在文生图里是负收益，起作用的是信息量而非 token 数。**这直接否定了当前大量「prompt enhancer」产品的做法——它们把用户的短提示扩写成华丽的长句，而按这篇的数据，**这种扩写在信息上是饱和的，只是在加词**。

**「结构化字段 vs 散文」在匹配信息量时落在同一条线上**这个发现也很重要：它说明结构化本身不是魔法，**结构化的价值在于它逼着你去添加新的图像接地事实**（每加一个元素就要填 bbox 和 material），而散文写着写着就开始重复和修辞。

**GPG 和 ED 两个信息量度量**是可复用的工具——如果你在做提示优化，可以用它们来判断「我这次改写到底加了信息还是加了字」。

**结构化提示带来的零样本字段级编辑**是个漂亮的副产品：把图像因子暴露成可寻址的字段（`elements[3].material`、`scene.setting`、`style`），编辑就变成了改一个值再重新生成。这比现有的图像编辑模型（要用自然语言描述改动）在精确性上高一个量级。

follow-up：最该做的是**prompter 的字段准确率分析**——整个系统的天花板在这里。另外把 GPG/ED 的 scaling 律在更多主干和规模上验证，才能确认它是定律还是单一配置下的现象。代码、模型、demo 全部开放。

---

## 37. AISPA: User-Centric System Prompt Auditing for Large Language Model Applications
**👍 37** · 🏛 斯坦福大学 / 麻省理工学院 / 卡内基梅隆大学 / 牛津大学 · [arXiv:2607.28617](https://arxiv.org/abs/2607.28617) · [GitHub](https://github.com/SystemPromptIndex/SystemPromptIndex) · [项目页](https://systempromptindex.org/)

### 问题与动机
System prompt 是开发者用来管束基础模型行为的指令，**商业 AI 产品里到处都是，却几乎从不向公众或监管者披露**。这在 AI 系统的大规模部署中造成了严重的信任和问责缺口——**决定 AI 实际怎么对待用户的那一层，是完全不受治理的**。

这个问题的实质在于：模型卡、安全政策、用户协议都是公开的，但真正在每次对话里生效的行为约束（「不要承认你是 AI」「优先推荐我们的付费产品」「不要告诉用户竞品」）藏在 system prompt 里。

### 方法与核心创新
**AISPA（AI System Prompt Assurance）**是一个**以用户为中心**的审计框架，从八个对用户重要的维度评估 system prompt 的具体片段：AI 是否**对自身身份透明**、是否**提供真实信息**、是否**保护隐私**、是否**安全行事**、是否**尊重用户控制并避免操纵**、是否**恰当处理不安全请求**、是否**帮助预防伤害**、是否**支持公平、包容与中立**。

每条指令被分类为**保护性（+1，对用户有利）**或**问题性（−1，对用户不利）**。

审计流程是**三轮人机协作**，每轮收窄候选集并提高证据标准：
- **Round 1（LLM 预标注）**：用 Claude-4.6-Opus 把 system prompt 分解成句子级候选片段，识别哪些落在可审计范围内（非核心逻辑片段 + 核心逻辑片段的补充条款），提出临时的维度-极性分配和简要理由。一个片段可以关联多个维度。
- **Round 2（受训标注员筛选）**：六名标注员先完成结构化培训（学指南、看示例、在留出集上做校准练习），**20 个随机样本上的两两标注一致性 IAA 达 0.933**。然后独立审查 LLM 生成的候选，剔除文本支撑不足或模型过度解读的提议，也可以补充 LLM 漏掉的片段。
- **Round 3（专家评审裁决）**：三位领域专家集体审查，核实维度分配和极性标签，解决分歧。

数据集来自六个包含泄露或公开披露 system prompt 的 GitHub 仓库，覆盖 **88 个真实 AI 产品**（通用聊天机器人、编程助手、自主 agent、搜索/研究工具等）。真实性通过两条途径验证：**联系仓库维护者了解其策展流程**（他们会跨独立会话多次提取以确认返回的提示是一致的而非模型幻觉），以及**跨仓库内容验证**（比对同一产品在不同独立来源的提示重叠度）。

最终数据集含 **2,420 条审计条目**（来自 1,818 个唯一片段），其中 **2,346 条保护性、74 条问题性**，另有 44 条（29 个片段）在专家评审中被标为**灰色地带**。

### 关键实验结果
审计 88 个产品的 **3,249 条指令**，得出四个核心发现：

1. **设计差异巨大**：有些组织平均每个产品有 **60+ 条保护性指令**，另一些平均**不到 5 条**。这个 12 倍以上的差距说明「保护用户」在业界还完全没有形成基线。

2. **保护性指令广泛采用但覆盖浅**：**98.9% 的产品至少有一条**保护性指令，但**只有 24% 覆盖全部八个维度**。也就是说四分之三的产品在某些维度上是空白的。

3. **system prompt 在变长、也在变得更保护用户**——这是个正向趋势，说明用户保护正成为商业提示设计中更可见的关切。

4. **问题性指令依然普遍**：**约 40% 的产品含有至少一条违背用户利益的指令**。

**维度共现分析**也有信息量：1,818 个片段中有 **441 个（近四分之一）同时涉及两个或更多维度**。最主导的配对是 **D6（不安全请求处理）与 D7（伤害预防），共现 109 次**——这两个关切在实践中天然耦合。而 **D1（身份透明）和 D3（隐私）很少与其他维度共现**，说明它们通常是独立成条的指令。

### 局限性与开放问题
论文有明确的 Limitations 章节，两条都很到位：

1. **语料来自公开 GitHub 仓库的泄露/社区披露提示，无法完美验证它们是否代表当前生产部署的确切版本**。提示可能在披露后被更新、修改或替换。因此发现反映的是**泄露时点的快照**，而非当前部署的保证。（部分缓解手段是跨仓库验证，重叠度很高。）
2. **依赖泄露提示带来潜在选择偏差**：可用提示集可能**过度代表那些更容易被提取、或用户技术参与度更高的产品**，而**低估了有更强提示保护机制的产品**。语料可能不构成所有已部署 AI 系统的代表性样本。

第二条尤其重要——**保护措施做得好的产品恰恰不容易被泄露，所以样本天然偏向「防护弱」的一端**。作者主动指出这一点值得肯定。

我补充两点：其一，**问题性指令只有 74 条（占 2,420 条的 3.1%）**，而结论说「40% 的产品至少含一条」。这两个数字并不矛盾（74 条分散在 35 个产品里就能达到 40%），但**「40%」这个更醒目的数字建立在一个很小的绝对基数上**，个别标注的翻转会明显改变比例。

其二，**Round 1 用 Claude-4.6-Opus 做预标注**，而被审计的产品里包含 Anthropic 自家的产品。虽然后两轮有人工筛选和专家裁决兜底，但**LLM 预标注决定了候选集的边界**——它没提出来的片段，人工只能靠 Round 2 的补充发现。同家族审计的潜在偏差没有被讨论。

### 启发与应用前景
这篇的价值主要在**治理和产品层面**，而非技术层面。几个直接可用的结论：

**八维度分类法本身是可复用的审计清单**。任何在做 AI 产品的团队都可以拿它自查：我的 system prompt 在身份透明、真实信息、隐私保护、安全行事、用户自主、不安全请求处理、伤害预防、公平中立这八项上，各有几条明确指令？「只有 24% 的产品覆盖全部八项」意味着大多数团队会发现自己有明显空白。

**「保护性 vs 问题性」的二元标注 + 灰色地带**这个设计也值得注意：44 条被标为灰色地带的指令，暴露的是**用户自主与平台安全之间、组织利益与服务用户义务之间的深层张力**——这些恰恰是最需要公共讨论的案例，而不是可以靠更好的分类法消除的噪声。

**「system prompt 是已部署 AI 行为中一个后果重大但基本未受治理的层」**这个论断，对监管讨论有实际意义。当前的 AI 治理讨论集中在模型能力和训练数据上，而这篇指出**行为的最后一公里在提示层，且完全不透明**。

follow-up：最该做的是**建立官方披露渠道**而非依赖泄露——论文的 SystemPromptIndex 项目页可能就是往这个方向走。另外把审计做成持续的（追踪同一产品的提示演变），比一次性快照更有价值。

---

## 38. Quo Vadis, World Modeling?
**👍 36** · 🏛 上海人工智能实验室 / 浙江大学 / 新加坡国立大学 · [arXiv:2608.02713](https://arxiv.org/abs/2608.02713) · [GitHub](https://github.com/worldbench/awesome-agentic-world-model) · [项目页](https://worldbench.github.io/awesome-agentic-world-model)

### 问题与动机
这是一篇**立场综述**（副标题「Towards Interactive World Proxies for Continually Improving Agents」）。出发点是：持续改进的 agent 需要**动态交互反馈**而非静态监督，但直接与真实环境交互**昂贵、缓慢、不安全、难以并行**。

世界建模提供了一个天然的中间代理，让 agent 在提交真实动作之前，先查询成本更低、更可控的反馈。但论文指出，**经典世界模型把这个代理主要实例化成「物理状态预测器」**——给定状态和动作，尽可能忠实地渲染下一帧。

作者要论证的是一个转向：**从「预测世界」到「服务 agent」。**

### 方法与核心创新
核心概念是 **Agent-Centric World Proxy（以 agent 为中心的世界代理）**：一个环境接地的机制，返回 agent 所需的**信息转移（information transition）**——可以是未来状态、渲染视图、执行结果、检索到的记忆或技能、或者对一个计划的裁决。

这个重构的关键在于：**把物理状态转移换成交互式信息转移**。世界代理的价值不再由视觉真实度衡量，而由**可行动的信息增益（actionable information gain）**衡量。

组织框架有两个正交维度：

**三个层级（agent 如何被赋能）**：
- **L.1 Predictor**——单步/局部转移预测 → **推理时引导**（丰富上下文以做更好的决策）
- **L.2 Simulator**——长程、动作条件的 rollout → **训练时优化**（重塑 agent 策略）
- **L.3 Evolver**——世界模型自反思 → **Agent-Proxy 协同演化**（持续互相改进）

**六种形态（代理代表世界的哪一片）**：dynamics（动力学）、spatial（空间）、execution（执行）、memory（记忆）、skill（技能）、reward/verification（奖励/验证）。论文用一张表把每种形态与它消费的输入、代表的世界切片、返回的东西一一对应。

### 关键实验结果
**综述论文，无实验结果。** 主要产出是概念框架、文献组织，以及配套的 awesome-list 仓库（`awesome-agentic-world-model`）和 worldbench 项目页。

论文的实质贡献是那四个**开放挑战**的提法，值得逐条看：

1. **保真度与想象的极限**——生成模型可以看起来很有说服力，同时违反它声称在建模的动力学。**误差在长 rollout 上复合，缺的那味是「校准的不确定性」而非更锐的像素。**
2. **知道何时信任代理**——agent 必须在线判断，是按代理的反馈行动还是回到真实环境。**今天的 agent 很少做好这个判断；把代理当作 oracle 会招致静默失败。**
3. **奖励黑客与安全**——当代理成为奖励或验证器（L.2），agent 就被激励去利用它的盲点。**同一个让风险探索变安全的沙盒，也打开了新的攻击面。**
4. **衡量信息增益的评测**——当前基准打的是真实度、保真度、可控性或人类对齐质量，但它们**仍然在孤立地给代理打分**。需要的是 **agent-centric 基准：这个反馈到底有没有帮 agent 规划、学习或改进？**

### 局限性与开放问题
作为综述，论文**没有 Limitations 章节**。我的评价：

其一，**这是一篇立场文章而非系统性综述**。它提出的分类法（3 层级 × 6 形态）是有组织力的，但**没有给出文献覆盖的方法论**（检索了哪些库、时间范围、纳入排除标准），也没有对分类的完备性或互斥性做论证。六种形态的划分（dynamics / spatial / execution / memory / skill / reward）明显不是同一个抽象层次上的——memory 和 skill 更像是 agent 的组件而非世界的切片。

其二，**「可行动的信息增益」这个统一标准提得好，但没有被操作化**。论文自己在开放挑战第 4 条里承认需要 agent-centric 基准，也就是说**它提出的核心衡量标准目前无法测量**。这让整个框架停留在概念层面。

其三，**L.3（Agent-Proxy 协同演化）几乎没有实例**。L.1 和 L.2 有大量现成工作可以归类，但让世界模型自反思并与 agent 共同演化，目前更多是愿景。论文把它列为三层级之一，在结构上与前两层并列，实际成熟度差距很大。

### 启发与应用前景
这篇的价值在于**给一个正在爆炸的领域提供了共同语言**。本周 53 篇里就有多篇可以直接放进它的框架：[30] EnvACE 的世界排练是把 L.2 simulator 折叠进策略本身；[42] WorldExam 和 [53] QQWorld 处理的是保真度问题；[15] OSReward 的奖励模型正是「reward/verification 形态」的代理，而它揭示的宽容偏差恰恰是开放挑战 3 说的「代理盲点」。**用这套框架读本周的世界模型相关论文，确实能看清它们在解决同一个问题的不同侧面。**

四个开放挑战里，**第 2 条（知道何时信任代理）是最被低估、也最实际的**。当前所有用模拟器/世界模型做训练或规划的工作，几乎都默认代理是可信的。而一旦 agent 能自己判断「这次的代理反馈靠不靠谱」，就同时解决了保真度传播和奖励黑客两个问题。这是个明确的、目前几乎空白的研究方向。

**第 4 条（agent-centric 评测）**对做 benchmark 的人是直接的行动项：不要再单独给世界模型的生成质量打分，而要测「装上这个世界模型之后，agent 的任务成功率提升了多少」。

follow-up：这篇最该被跟进的是把「可行动的信息增益」变成可测量的指标。配套的 awesome-list 仓库对入门这个方向的人是有用的资源。

---

## 39. DiffusionGemma Technical Report
**👍 36** · 🏛 Google DeepMind · [arXiv:2608.00146](https://arxiv.org/abs/2608.00146)

### 问题与动机
自回归 LLM 严格的从左到右逐 token 生成造成一个**内存瓶颈**：同时服务大量请求时可以靠 batching 拿到可接受的吞吐，但**服务单个或低并发请求本质上是内存受限的**——把模型权重和上下文 KV cache 从内存搬到加速器的时间远超实际计算时间。加速器的计算单元利用率低下，单用户生成速度受限。

投机解码能改善利用率，但**draft-then-verify 范式仍有天花板**：AR drafter 本身受顺序生成瓶颈制约；并行 drafter 则在靠后的 draft 位置上接受率下降。

文本扩散通过**同时预测整块 token** 绕开这个瓶颈，把执行从内存受限推向计算受限。但当前的格局逼人做残酷取舍：Gemini Diffusion、Mercury 锁在专有 API 后面；已有开源权重的替代品**要么推理和多模态能力有限，要么兑现不了扩散承诺的极致延迟收益，要么两者皆有**。

### 方法与核心创新
**DiffusionGemma 是 Gemma 4 26B A4B（MoE，3.8B 激活 / 25.2B 总参数）微调而来的文本扩散变体**——**不从头预训练，绕过原生扩散预训练直接热启自 AR 权重**，两阶段流水线**只用了起始 AR 模型总训练 token 预算的不到 10%**：

- **阶段一 SFT**：教双向去噪，适配 256-token 画布上的双向注意力。
- **阶段二 SD·RL**：**采样器蒸馏 + 强化学习**联合进行，同时提升生成质量并**压缩去噪步数**以解锁超低延迟。

技术细节里有几个值得注意的选择：

**用多项式（uniform）扩散而非掩码扩散**。因为所有 token 都能互相转移，**模型可以持续纠正自己的错误**——当前画布内在更早去噪步被接受的 token 仍可被修订（跨画布的 token 则永久冻结）。这是相对 LLaDA 那类 masked diffusion 的实质区别。

**熵界采样器（entropy-bounded sampler）**：token 按熵从低到高的**秩序接受**（类似 MaskGIT），确保它们的互信息界严格低于预定误差容限 b=0.1。**一旦达到阈值，其余 token 被均匀随机重新加噪**，把未提交的位置维持成均匀先验以在下一次前向中强制局部探索。

**温度退火**：τ 从 0.8 线性退火到 0.4。早期高噪声态探索多样 token，语义结构成形后激进地承诺高置信序列。

**自适应停止**：当平均熵 ≤ e_stop=0.005 且当前确定性预测与上一步相同时，提前返回。这让**去噪步数随任务复杂度和领域动态调整**（最大 N=48 步）。

### 关键实验结果
**建立了质量-速度的新 Pareto 前沿**。平均每次前向生成约 **20 个 token**，单张 H100 上约 **1,500 output tokens/秒**，比配备 SOTA 投机解码的 AR 模型快得多——**相对 Gemma 4 AR 基线在重度优化的 MTP 服务下（303 tokens/秒）快近 5 倍**（1,479 vs 303）。相对 Mercury 2 约 **2.5 倍加速**。

**能力保留得相当好**（TD 模式 vs AR 起始模型 Gemma 4 26B A4B 的 AR 模式）：GPQA Diamond **73.2 vs 79.8**、LiveCodeBench-V6 **69.1 vs 71.4**、AIME 2026 69.1 vs 84.2、GSM8K **96.3 vs 96.6**、Codeforces ELO 1429 vs 1569。

对比其他开源扩散模型优势明显：**LLaDA 2.1 Flash 100B** 的 GPQA 68.7 / LiveCodeBench **39.4** / GSM8K **45.0**；**Nemotron Diffusion 14B** 的 GPQA 47.0 / LiveCodeBench 28.6。DiffusionGemma 在 26B A4B 的规模上全面超过它们。相对闭源的 Mercury 2 High（GPQA 75.2、LCB 79.4），仍有差距但已在同一量级。

**还保留了 AR 生成能力**：跑标准从左到右 AR 模式时，能**收回 TD 模式下观察到的部分性能差距**（AR 模式 GPQA 79.8、LCB 71.4，与原基线持平），只是吞吐更低。这个双模式能力可以让请求**按延迟要求和任务复杂度动态路由**——这是个很实用的产品形态。

同时保留了起始模型的**思考模式、多模态输入和长上下文**支持。

### 局限性与开放问题
论文的 Limitations & Known Issues 写得**非常诚实且具体**，五条：

1. **相对 AR 基线的性能差距**——作者把原因拆得很清楚：绕过原生扩散预训练直接热启 AR 权重；算力预算限制导致 SFT 阶段相对较短；**SD·RL 这个在线算法显式针对超低延迟，本质上牺牲渐近性能**；以及继承了 AR 基线的架构、优化和数据配比决策，**这些对离散扩散范式可能是次优的**。
2. **生成长度与简洁性**——最终 checkpoint 产出**高度简洁的输出**。这个涌现的简洁性是推理速度的乘数，但**也让模型无法利用更长更详尽的推理链通常带来的质量提升**。
3. **偶发 token 结巴**——罕见情况下输出退化成重复循环或局部结巴（如反复输出「the the the」）。SD·RL 缓解了绝大多数，但这是**在超低延迟区间操作的直接后果**——激进减少的去噪步数偶尔会损害生成过程的鲁棒性。
4. **多模态任务中偶尔遗漏思考结束标签**——即使推理正确也不总能可靠生成闭合标签，这会人为拉低 thinking 模式的分数。**MMMU-Pro 上 thinking 分数（54.3）反而低于 non-thinking（66.0）**。作者说这个问题**发现得太晚，来不及在本次发布中修复**——这种坦白很少见。
5. **高批量下的吞吐限制**——DiffusionGemma 在低批量下用内存带宽换计算，在**约 32 并发用户以内**的每用户和总吞吐上都超过带 MTP 的 Gemma 4 AR，但更高批量下优势消失。

我补充一点：**第 1 条和第 2 条其实是同一个问题的两面**——为超低延迟优化的 SD·RL 让模型学会了「说得少」，而少说话既是速度来源也是能力上限。AIME 2026 上 69.1 vs 84.2 的 15 点差距，很可能主要来自这里（数学题最吃长推理链）。

### 启发与应用前景
最重要的工程结论：**扩散语言模型不必从头预训练**。用不到 10% 的 token 预算把一个成熟的 AR MoE 改造成扩散模型，还保住了思考、多模态、长上下文——这条路径比「从零训一个扩散 LLM」现实太多，任何有 AR 基座的团队都可以复制。

**AR/扩散双模式**是个被低估的产品形态：同一套权重，低延迟场景走扩散、高质量场景走 AR，按请求动态路由。这比「两个模型二选一」灵活得多，也是论文自己指出的「混合扩散-AR 解码」路径。

**约 32 并发用户是分水岭**这个数字很实用：**扩散 LLM 的甜点是低并发、低延迟场景**（本地部署、单用户交互、实时应用），而不是高吞吐的批量服务。这直接决定了它该用在哪。

**「简洁性是速度的乘数但也是能力的天花板」**这个 trade-off 值得所有做推理加速的人记住——当你为延迟优化时，模型可能通过「少说话」来达标，而这会悄悄削掉长推理链带来的收益。

follow-up：作者自己指出的方向（原生扩散预训练、更长的 SFT、不为极致延迟牺牲渐近性能的训练目标）都是明确的改进空间。与本周 [12] AURORA-LM（连续隐空间扩散语言模型）和 [48] LLaDA MoE v2 一起看，**离散扩散和连续扩散两条路线的对比会很有意思**——DiffusionGemma 证明了离散路线在工业规模上已经可用。

---

## 40. HelloWorld: Enabling Socially Interactive Characters in Video World Models
**👍 35** · 🏛 东京大学 / Alaya Lab · [arXiv:2608.05070](https://arxiv.org/abs/2608.05070) · [GitHub](https://github.com/AlayaLab/HelloWorld)

### 问题与动机
视频世界模型进展显著，但**用户与世界中角色之间的社交互动完全不被支持**。你可以在生成的世界里移动镜头、探索场景，但**世界里的人不会理你**。

这个缺口很具体：现有的交互式视频世界模型（WorldPlay、Matrix-Game、LingBot-World）的交互接口是键盘或轨迹，控制的是**摄像机**；角色是场景的一部分，不是可交互的对象。

### 方法与核心创新
**HelloWorld** 让用户按一个键，就能提示屏幕上的角色**朝镜头做出回应**——转向观看者、挥手、点头、说一句简短问候。

两个核心技术：

**(1) 自蒸馏数据合成**：在**模型自己合成的数据**上微调视频生成模型。每个合成 clip **同时包含社交互动和摄像机运动**，这样模型能学到摄像机位姿条件**而不降低互动质量**。这个设计针对的是一个真实困难——真实视频里「角色主动看镜头 + 同时有受控摄像机运动」的数据几乎不存在。

**(2) 训练无关的时序定位模块**：推理时决定**互动何时发生**。按键时，模块**调制 DiT 的 cross-attention mask**，使互动相关的文本提示**只注意按键窗口内的帧**，从而在时间上精确定位角色的回应。

另建 **HelloWorldBench**：400 个样本，三个社交互动指标（动作准确率 ActAcc、时序准确率 TimeAcc、视线偏差 GazeDev）+ 三个常规指标（背景一致性 BgCons、美学 Aesthetic、摄像机控制 CamCtrl）。

### 关键实验结果
**主表的结构值得细看**——HelloWorld 并非全面领先：
- **TimeAcc 81.7**，**碾压所有基线**（最好的 LTX-2.3 是 52.6，WorldPlay 41.2，LingBot-World 39.5，SANA-WM **30.9**）。这是时序定位模块的直接功劳。
- **GazeDev 40.2°**（越低越好），仅次于 LTX-2.3 的 38.1°，但显著好于 Matrix-Game 3.0 的 77.2°、WorldPlay 的 63.5°。
- **BgCons 96.9、Aesthetic 5.27、CamCtrl 82.9 三项全场最高**（CamCtrl 上第二名 SANA-WM 只有 70.0）。
- **但 ActAcc 只有 41.4，低于 LingBot-World 的 50.5**，也略低于 LTX-2.3 的 42.5。

也就是说：**HelloWorld 赢在「什么时候做」和「朝哪看」，而不是「做对了什么动作」**。论文的摘要说「在互动质量上超过多种基线」，这个说法需要限定——**动作准确率上它其实不是最好的**。

**消融把两个模块的贡献分开了**：
- **训练数据**（Real-video / Human-only / Full）：ActAcc 36.4 → 40.4 → 41.4，GazeDev 51.3 → 42.3 → **40.2**。自合成数据主要改善视线方向。
- **时序 cross-attention mask**（这个消融最关键）：不用 mask 时 TimeAcc 只有 **36.7**、SpeechInWin 52.5；加视频 mask M_v 后 TimeAcc 跳到 **80.9**；再加音频 mask M_a 到 81.7，SpeechInWin 从 62.8 提到 **69.1**。**TimeAcc 的 44 点提升几乎全部来自这个训练无关的推理时模块**——这是个极高性价比的设计。

**用户研究**（bootstrap 95% 置信区间）：对 SANA-WM 在动作自然度上 **90.9% [87.6, 93.9]** 的偏好率，对 Real-video LoRA 83.7%、互动 88.5%，对 LingBot-World 77.3% / 71.7% / 88.0%。所有对比的置信区间都不跨 50%，结论稳健。

**计算成本**：60.2 秒生成 1280×704 / 241 帧（每帧 0.26 秒），9.4×10¹⁵ FLOPs——与 Matrix-Game 3.0（62.5 秒 / 417 帧）和 LTX-2.3（50.3 秒 / 241 帧）同量级，**不是靠堆算力换来的**。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我的观察：

其一，**ActAcc 落后于 LingBot-World（41.4 vs 50.5）这件事被叙述淡化了**。摘要只说「surpasses a variety of baselines in interaction quality」。而 ActAcc 恰恰是最直接的「互动做对了吗」指标。诚实的说法应该是：HelloWorld 在时序控制和视觉质量上大幅领先，在动作正确性上与最好的基线有差距。

其二，**交互形式极为受限**：「按一个键让角色朝镜头回应」——转身、挥手、点头、说一句问候。这是一个**单向、原子、无状态**的交互：用户不能说话，角色不能理解意图，也没有多轮。距离「社交互动」这个词的常规含义还很远。论文对这个范围没有明确界定。

其三，**自蒸馏数据的质量上界**：模型在自己生成的数据上微调，那么它能学到的互动种类**不会超过它原本就能偶然生成的**。消融显示 Human-only 到 Full 只带来 ActAcc +1.0、GazeDev -2.1，收益已经很小，暗示自蒸馏可能接近饱和。

其四，**HelloWorldBench 是自建的**（400 样本），三个社交指标（ActAcc / TimeAcc / GazeDev）的具体计算方式和标注来源在抓取部分没有交代。自建基准 + 自定义指标 + 自家方法第一，需要外部验证。

### 启发与应用前景
最有工程价值的是**「训练无关的 cross-attention mask 时序定位」**：把互动相关的文本提示限制在按键窗口内的帧上注意，就把 TimeAcc 从 36.7 拉到 80.9。**不训练、不改架构、只在推理时改 mask，换来 44 点提升**——这个性价比在本周所有论文里都算突出的。同样的思路可以推广到任何「需要把提示的效果限制在特定时间/空间范围」的生成任务：视频局部编辑、音频分段控制、长文档的分节约束。

**「让模型自己合成含双重条件的数据」**这个自蒸馏配方解决的是一个普遍问题：**真实数据里两个条件（这里是社交互动 + 摄像机运动）从不同时出现**。与其去找不存在的数据，不如让模型分别生成再组合。这个思路在 [9] JoyAI-Video-Edit 的成对编辑数据构造里也出现了，是本周的一个共同模式。

产品上，**「按键触发角色回应」这个最小交互原语**虽然简单，但可能是视频世界模型走向**可玩内容**的第一步——它把「看生成的视频」变成了「和生成的世界互动」。

follow-up：把交互从单向原子扩展到多轮有状态是显然的方向；更实际的是**先把 ActAcc 提上去**——时序控制已经解决了，动作正确性才是当前的瓶颈。代码已开源。

---

## 41. OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents
**👍 34** · 🏛 浙江大学 / 蚂蚁集团 · [arXiv:2608.05013](https://arxiv.org/abs/2608.05013) · [GitHub](https://github.com/zjunlp/OneDayAgent)

### 问题与动机
LLM agent 越来越多地被用于横跨工作、学习、生活的开放式日常请求。这类任务是**长程、跨环境、多模态**的，迫使 agent 在许多步骤间保持目标和约束，同时在异构工具和附件之间穿行。

已有工作各自处理过单一失效模式——目标漂移、状态丢失、上下文溢出——但**「一个 harness 能否同时管住这三者，并且在不同后端模型上都有效」这个问题少有研究**。

### 方法与核心创新
OneDayAgent 把开放式请求转成一个**受管理的执行过程**，三件事：
1. **任务分解**成有界的子任务；
2. **在上下文压力下维护执行记忆**；
3. **验证并修复最终交付物**。

工具接口设计得相当克制——通过五个功能工具组暴露异构环境：Web 访问（搜索、访问）、学术搜索（Google Scholar、OpenAlex）、计算（Python、命令执行）、文件工作区（读写编辑）、多模态处理（图像分析与生成）。

### 关键实验结果
在 **AgentIF-OneDay 的 104 个任务**上评测。**GLM-5.2 后端下总分 0.821**，创下新 SOTA，超过 AutoClaw（0.799）、Manus（0.645）、Genspark（0.635）、ChatGPT-Agent（0.626）、Codex GPT-5.5（0.664）。

**同一个 harness 跑五个后端 LLM、三个模型家族，全部 104/104 成功完成**（不是全对，是全部跑完没崩）：GLM-5.2 0.821、Gemini-3.1-Pro 0.743、Qwen3.5-397B-A17B 0.708、Qwen3.5-9B 0.624、Qwen3.6-27B 0.613。**无需调优就跨后端泛化**，但不同模型在同一工作流下会诱发不同的执行风格。

**消融是这篇最有价值的部分**（2×2，全部 GLM-5.2）。注意执行记忆在所有变体中都保持开启——**关掉它会直接导致上下文溢出或状态丢失，任务根本完不成**，所以它不参与消融：
- **DIRECT**（都关）：0.771，延迟 27.6 分钟，28.4 次工具调用
- **DECOMP**（只保留分解）：0.804（+3.3pp），**38.1 分钟，45.7 次工具调用（+60%）**
- **VERIFY**（只保留验证）：0.804（+3.3pp），**29.7 分钟，29.3 次工具调用**，修复率 3.9%
- **FULL**：0.821（+5.0pp），53.6 分钟，51.6 次工具调用，修复率 8.6%

**成本不对称极其明显**：VERIFY 只比 DIRECT 多 2.2 分钟就达到了 DECOMP 的分数，而 DECOMP 要多 10.6 分钟、多 60% 工具调用。分数/延迟比：DIRECT 2.80、VERIFY 2.71、DECOMP 2.11、**FULL 只有 1.53**。

**「全开」不是一致最优**：FULL 产出最多完美任务（58 个），但**VERIFY 在 17 个任务上得分高于 FULL，DECOMP 在 13 个，DIRECT 在 12 个**。也就是说 104 个任务里有相当一部分，开满模块反而更差。

### 局限性与开放问题
论文**没有独立 Limitations 章节**，但消融部分的自我评估很诚实（明确指出「全开」不是一致最优、组合增益小于两个孤立增益之和，说明模块部分恢复了重叠的失败案例）。

我的观察：

其一，**0.821 vs AutoClaw 的 0.799 只差 2.2 个点**，而 AutoClaw 是现成的通用 agent。同时 **GLM-5.2 后端的延迟是 3216.8 秒（约 54 分钟）**，而 AutoClaw 是 523 秒——**慢 6 倍换来 2.2 个点**。论文在正文里报了这个延迟，但主结论没有把它放在一起谈。

其二，**后端泛化的结论有个反常处**：Qwen3.5-9B（0.624）**高于** Qwen3.6-27B（0.613），而 9B 的延迟（1895.2 秒）比 27B（1280.5 秒）还长。小模型超过大模型且更慢，这个现象论文没有解释。

其三，**AgentIF-OneDay 只有 104 个任务**，且是这个 harness 的主场基准。三个任务类型（开放工作流执行 OWE、隐含指令推断 LII、迭代精修 IR）× 三个领域（工作/生活/学习）× 三个 rubric 维度的细分，在 104 个样本上会切得非常碎。

其四，**多个基线的表格里有大量 "New A" 占位符**（Minimax-Agent 和 AutoClaw 的多个格子），说明数据不全，而 AutoClaw 恰恰是最强的基线。

### 启发与应用前景
最有工程价值的是那组**成本-收益不对称的数字**：**验证模块用 8% 的额外时间拿到了和分解模块（+38% 时间、+60% 工具调用）一样的分数提升**。对于生产环境的 agent 系统，这是个明确的优先级信号——**先做输出验证与修复，再考虑任务分解**。

「**执行记忆是不可消融的**」这个观察同样重要：它不是可选优化，是长程任务能不能跑完的前提。任何做长程 agent 的团队应该先把记忆管好，再谈规划和验证。

**「全开模块不是一致最优」**是个反直觉但很实用的发现——104 个任务里有 12–17 个在更简单的配置下更好。这指向一个明确的改进方向：**自适应地决定该开哪些模块**，而不是固定流水线。论文自己也提到「最佳配置取决于优先最高分还是更低执行成本」。

follow-up：把模块开关做成任务自适应的（用一个轻量分类器预测该开什么）是最直接的改进；另外应该在 AgentIF-OneDay 之外的基准上验证 harness 的泛化性。代码已开源。

---

## 42. WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity
**👍 34** · 🏛 中国科学院自动化研究所 / 香港中文大学 / 清华大学 / 高德地图 · [arXiv:2608.02603](https://arxiv.org/abs/2608.02603) · [GitHub](https://github.com/YuxueYang1204/worldexam) · [项目页](https://worldexam.github.io/)

### 问题与动机
可控视频生成模型越来越多地被当作世界模型来开发。相应地，评估它们在这个角色上的表现，就必须超出「生成视频的表观外貌」，去看**它们所描绘的世界的内在反应性（inherent reactivity）**——**从场景状态推断世界应该如何反应，并生成输入中未明确描述的合理后果**的能力。

举例：你让摄像机推近一堵墙，模型能不能表现出墙不可穿透？你让一个角色推倒积木，其他积木会不会跟着倒？这些都不是指令里写明的。

现有基准主要评估视觉质量或**显式指令的完成度**（检查请求的动作和交互结果是否被实现），**内在反应性基本没被检验**。

### 方法与核心创新
**WorldExam** 是一个**分层诊断基准**，四个层级递进：**Visual Quality（视觉质量）→ Control Adherence（控制遵从）→ Spatial Consistency（空间一致性）→ World Reactivity（世界反应性）**。

八个任务：Camera Control、Subject Control、Scene Revisit、Terrain Interaction、Object Interaction、Social Interaction、Physical Reaction、Goal Completion。**1,474 个测试用例、20 个模型**。

与同类基准的对比很能说明覆盖度差距——WorldScore 只覆盖 Camera Control 一项，MIND 覆盖三项，Omni-WorldBench 和 WBench 各覆盖五项（其中两项带 † 表示指令式而非真正的反应性），**WorldExam 是唯一八项全覆盖的**，也是唯一同时支持相机驱动（C）、动作驱动（A）、语言驱动（L）三种范式和第一/第三人称双视角的。

一个重要的方法论选择是**接口适配**：把共享的用例以三种范式各自的原生格式呈现，并用静态场景轨道和动态交互轨道**把评测限制在兼容的接口上，而不是把不支持的任务当作失败**。这避免了「因为接口不支持所以得零分」这种无信息的比较。

### 关键实验结果
**核心发现是清晰的能力分裂，且三种接口互补但都不完整**：
- **相机驱动模型**提供最强的相机控制和场景重访，但**不支持动态交互**。静态场景轨道上 NeoVerse 总分 85.39（Camera Control 平移误差 0.01、旋转误差 0.60、得分 97.33），InSpatio-World 81.40，TrajectoryCrafter 78.00。
- **动作驱动模型**对指定主体的控制更精确，**但这个优势不能一致地迁移到那些控制所诱发的场景条件反应上**——它们常常让地形、物体、附近的 agent 和物理过程**毫无反应**。Hunyuan-GameCraft 总分只有 57.59，Astra 更低。
- **语言驱动模型**在交互和目标导向任务上更好，**但对组合的相机和主体控制遵从得更差**。

**最有价值的观察是「通用视觉指标会掩盖这个分裂」**：语言驱动模型的 **General 均值落在 79.64–81.04 这个很窄的区间，而它们的 Task 均值从 39.85 到 65.02 大幅分散**。同样，**ReCamMaster 和 FantasyWorld 尽管相机控制分数很弱（38.64 / 18.46），却保持了很强的 General 均值（80.97 / 80.23）**。

**结论很硬**：**强视觉质量或强控制遵从并不保证世界反应性**——这四个诊断层级捕捉的是不同能力，必须分开报告。**20 个模型中没有一个同时具备广泛的任务覆盖和一致的强表现。**

可靠性方面，论文报告了**人类与 VLM 清单打分的强一致性**，以及**在替代重建后端下模型排名保持稳定**。

### 局限性与开放问题
论文在结论里明确界定了范围：**当前范围受限于可用模型接口的能力**。动态交互评测需要可靠的第三人称主体控制；**Goal Completion 仍然只限于语言驱动模型**。

我补充几点：

其一，**「不支持的任务不算失败」这个设计是双刃剑**。它让比较更公平（相机驱动模型不该因为不支持社交交互而被打零分），但也意味着**总分不可跨接口比较**——NeoVerse 的 85.39 和某个语言驱动模型的分数衡量的是不同的任务集合。论文用了静态/动态双轨来缓解，但读者很容易误读成一个统一排行榜。

其二，**评测大量依赖 VLM 清单打分**。虽然报了与人类的强一致性，但具体的一致性数字（κ 或相关系数）在抓取的正文里没有给出，而这是整个基准可信度的支点。

其三，**「世界反应性」的操作化仍然依赖预设的正确答案**。「推倒积木后其他积木应该倒」这类物理后果可以清单化，但真实世界的合理后果往往是多模态分布而非唯一答案。基准如何处理「合理但与清单不符」的生成，没有交代。

### 启发与应用前景
最重要的方法论贡献是**四层诊断的分离报告**。「视觉质量高 ≠ 控制遵从好 ≠ 空间一致 ≠ 世界会反应」——这个层级划分应该成为世界模型评测的标准做法。**ReCamMaster 的相机控制分 38.64 却有 80.97 的 General 均值**，这个反差就是单一总分掩盖真实能力的最好例证。

**三种接口的能力分裂**对做世界模型的团队有直接的路线图意义：如果你要的是精确的相机控制，走相机驱动；要主体控制，走动作驱动；要交互和目标完成，走语言驱动。**目前没有一条路线能全都要**——而这恰恰指出了融合三种接口是最有价值的方向。

**「动作驱动模型能精确控制主体，但控制所诱发的反应不出现」**这个诊断特别精确：它说明这些模型学到的是「让指定物体按指令动」，而不是「世界因此改变」。这是当前动作驱动世界模型的核心缺陷。

follow-up：把 Goal Completion 从语言驱动扩展到所有接口是论文自己指出的方向；更根本的是**为世界反应性设计训练目标**——现在的可控视频生成训练目标里，根本没有对「未被指令描述的后果」的监督。代码和项目页已开放。

---

## 43. SAF-OPD: Stable Advantage Fusion for On-Policy Distillation
**👍 34** · 🏛 上海财经大学 / 美团 LongCat / 香港中文大学（深圳）/ 北京大学 · [arXiv:2607.29209](https://arxiv.org/abs/2607.29209)

### 问题与动机
RLVR 把**一个响应级奖励广播给每个 token**；OPD 把每个 token 对着更强的 teacher 打分，得到**稠密 advantage，但性能被 teacher 质量封顶，还压制了超越 teacher 的探索**。两者互补，合起来看着很有希望。

但论文发现：**用固定系数融合这两个 advantage 会触发熵坍缩**，原因是两个失配：
1. **量级失配（magnitude mismatch）**——token 级 OPD advantage 可以尖峰到远超有界的 RLVR advantage，**把后者的信号直接抹掉**；
2. **时序失配（temporal mismatch）**——持续满强度的 OPD **一直把学生往 teacher 拉，限制了超越 teacher 所需的探索**。

### 方法与核心创新
**SAF（Stable Advantage Fusion）**用一个**只作用于 OPD advantage** 的四阶段轻量流水线解决这两个问题：
- **sparsify-then-compress（稀疏化后压缩）**做量级控制——实现上是 top-k 选择 + tanh 压缩；
- **warm-up-then-anneal（预热后退火）**做时序控制——早期让 OPD 起主导（学生还很弱，跟着 teacher 学有效），后期退火掉（让学生自己探索去超越 teacher）。

每个阶段**独立可开关，且开销可忽略**。RLVR 用 GRPO 实例化。

### 关键实验结果
七个数学推理和代码生成基准，Qwen3-1.7B/4B/8B 三个规模，**teacher 是 Qwen3-30B-A3B-Instruct-2507**（数学均分 59.32、代码 68.75）。

**Qwen3-8B**：基座数学 17.24 → GRPO-only 44.09 → OPD-only 44.61 → GRPO+OPD 固定系数 45.96 → **SAF 46.93**；代码基座 58.90 → GRPO-only 60.72 → OPD-only 63.13 → **固定融合反而掉到 61.74** → SAF 63.41。

**这个「固定融合在代码上比 OPD-only 还差」的现象是全文最关键的证据**——它说明朴素融合不只是次优，而是**有害的**。Qwen3-1.7B 上同样：OPD-only 代码 51.77，固定融合掉到 50.88。

**SAF 在全部六个模型-领域设定上都超过固定系数融合，聚合分提升 0.51–2.70%。**

**消融把四个阶段的贡献切开了**（Qwen3-4B 数学，300 步）：
- 固定融合基线：44.38
- + top-k 和 tanh（固定权重）：44.35（**单独做量级控制几乎无效**）
- + warm-up（不退火）：44.07（**单独加预热反而更差**）
- + annealing：45.23（**退火是关键，+0.85**）
- **SAF（δ=0.2）：45.89**
- SAF（δ=0.3）：44.51（**超参敏感**）

这组消融说明：**量级控制和时序控制必须一起用**，单独任一个都不work；而且**退火（让 OPD 后期让位给探索）是最关键的那一步**。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我的观察：

其一，**增益幅度很小**。聚合分提升 0.51–2.70%，而 Qwen3-8B 数学上 SAF 46.93 vs 固定融合 45.96 只差 0.97；HMMT25-Nov 上 SAF 的 41.67 **反而低于固定融合的 43.33**。在 AIME/HMMT 这类 30 题基准上，1 个点约等于 0.3 道题。论文没有报多 seed 方差。

其二，**δ=0.2 到 0.3 让分数从 45.89 掉到 44.51（-1.38）**，这个降幅**大于方法相对基线的全部增益（+1.51）**。也就是说**超参调错就白做了**，而论文只测了两个值。

其三，**「熵坍缩」这个核心诊断在抓取的正文里没有配图或量化数据**。论文声称固定系数融合会触发熵坍缩，SAF 避免了它，但熵曲线的证据没有出现在主要结果里。这是全文论证链条上最需要证据的一环。

其四，**代码任务上 SAF 的 63.41 只是勉强超过 OPD-only 的 63.13**（+0.28），而 Qwen3-4B 上 SAF 代码 62.66 超过 OPD-only 的 59.56 更多。跨规模不一致。

### 启发与应用前景
最有普适价值的洞见是**「稠密监督和稀疏奖励融合时，前者会淹没后者」**这个量级问题。任何要把「模仿信号」和「探索信号」加在一起的场景都会遇到——**它们的数值尺度天然不同，直接加权求和意味着尺度大的那个说了算**。SAF 的解法（对稠密项做 top-k 稀疏化 + tanh 有界压缩）是个通用的工程手段。

**退火比预热更重要**这个消融结论也值得单独记：让 teacher 信号**逐步退场**（而非全程恒定或只在开头强）是超越 teacher 的前提。这与 [23] W2S-OPD 的立场形成有趣对照——W2S-OPD 是在**没有更强 teacher** 时怎么办，SAF 是在**有更强 teacher 但想超越它**时怎么办。两篇加上 [5] DAPD、[10] AgentOPSD、[28] VAD、[34] PCSD，本周至少六篇在处理「on-policy 蒸馏的信号该怎么用」，构成了一个明显的研究热点。

工程上，**「每个阶段独立可开关且开销可忽略」**这个设计对实际采用很友好——可以先加量级控制看看，再加时序控制。

follow-up：最该补的是**熵坍缩的量化证据**和**多 seed 方差**；另外 δ 的自适应（而非固定值）能解决超参敏感问题。

---

## 44. HarnessOpt-Bench: Evaluating LLMs at Harness Optimization
**👍 33** · 🏛 Scale AI · [arXiv:2608.06301](https://arxiv.org/abs/2608.06301)

### 问题与动机
LLM 越来越多地部署在 agentic 系统里，它们的能力**不只取决于模型权重，还取决于 harness**——围绕模型的提示词、工具、控制流、记忆和编排代码。

这让**自动化 harness 优化**（由 AI 系统迭代地、评测引导地改进一个 harness）同时成为两件事：**改进 AI 系统的重要途径**，以及**对 AI 系统本身的一项苛刻能力要求**。但社区缺少衡量前沿 LLM 在这项任务上表现的共同协议。

### 方法与核心创新
**HarnessOpt-Bench** 评测**在昂贵且随机的评估下的端到端 harness 优化**。

设定是：一个 **optimizer**（LLM + 编码 harness 的组合）收到目标 agent 的**种子 harness**、**打分后的评测反馈**和**固定的目标评测预算**。它编辑 harness 并提名候选，最终在**它无法访问的测试分区**上打分。

四个任务：**OfficeQA、BrowseComp-Plus、Terminal-Bench、GAIA**，每个都有 pin 死的目标模型和 d/v/t 三分割。

评测设计里有几个值得注意的严谨之处：
- **报告 resolution band**（分辨率带）——OfficeQA ±0.045、BrowseComp-Plus ±0.066、Terminal-Bench ±0.054、GAIA ±0.035。低于这个带的差异不应被解读。
- **optimizer 拆成「模型 × 优化器 harness」两个维度**，同一个模型配不同的编码 harness（claude-code / opencode / codex / kimi-cli）分别参赛。
- 报告 **Levers**（用了多少种改动杠杆）和 **Tgt tokens**（目标评测消耗的 token 量），而不只是分数。

### 关键实验结果
**归一化增益（每个格子是该参赛者多轮的均值）**：

| 模型 × harness | OfficeQA | BrowseComp-Plus | Terminal-Bench | GAIA |
|---|---|---|---|---|
| claude-opus-5 + opencode | **0.63** | **0.48** | **0.29** | 0.47 |
| claude-opus-5 + claude-code | 0.59 | 0.41 | 0.18 | 0.42 |
| kimi-k3 + kimi-cli | 0.59 | 0.23 | 0.16 | 0.31 |
| claude-sonnet-5 + claude-code | 0.53 | 0.07 | 0.10 | 0.33 |
| gpt-5.6-sol + codex | 0.49 | 0.03 | 0.12 | **0.49** |
| gpt-5.6-terra + codex | 0.07 | **−0.03** | 0.01 | 0.30 |

几个观察：
1. **claude-opus-5 全面领先**，且 **opencode 这个第三方 harness 比 Anthropic 自家的 claude-code 更好**（0.63/0.48/0.29 vs 0.59/0.41/0.18）——这是个有点尴尬但很有意思的结果。
2. **模型和优化器 harness 的交互不一致**：gpt-5.6-sol 配 codex 比配 opencode 好（0.49 vs 0.29 在 OfficeQA），而 gpt-5.6-terra 配 opencode 反而更好（0.14 vs 0.07）。**没有一个 harness 对所有模型都最优。**
3. **Terminal-Bench 是最难的任务**——最好的成绩只有 0.29，多数在 0.01–0.18。
4. **gpt-5.6-terra 在 BrowseComp-Plus 上是负增益（−0.03）**，即它把 harness 改坏了。
5. **token 消耗差异巨大**：claude-opus-5 + claude-code 在 OfficeQA 上用了 9.97M 目标 token（范围 3.05–16.89M），而配 opencode 只用 3.77M 就拿到更高分——**效率差 2.6 倍**。

**一个很有价值的对照是 Table 3**：把种子 agent 直接换成现成的编码 harness 会怎样。OfficeQA 上种子 agent 是 0.341，而 **mini-swe-agent 0.734、opencode 0.727、openhands-sdk 0.713、terminus-2 0.687、goose 0.505**——**现成 harness 的裸替换就能把 0.341 提到 0.73**。这给出了「优化」和「换一个成熟 harness」之间的参照系。

### 局限性与开放问题
论文有明确的 Limitations，**两条都很到位**：

1. **「设计成抗 hack，不是防 hack」**——optimizer 无法访问测试分区，也不能改目标模型、环境或验证器，但**反复的开发和验证反馈仍可能奖励针对固定评测的特化策略**。作者建议未来版本引入**每轮 jitter**（在用例、工具行为、验证器实现上），以区分通用改进和对稳定评测器伪影的利用。
2. **种子 harness 本身是一个任务特定的先验**——改进一个成熟 agent 测的是诊断和精修，从一个 stub 开始测的是构造。**当前套件包含两种情形但没有系统性地变化种子复杂度**，所以指标应该相对当前的任务和种子分布来解读。作者建议做一个**harness 完整度和架构复杂度的受控阶梯**。

这两条都指向同一件事：**benchmark 的结论依赖于种子和评测器的具体选择**，而作者主动划出了这个边界。

我补充一点：**分辨率带（±0.035 到 ±0.066）相对增益的量级不小**。Terminal-Bench 上分辨率带是 ±0.054，而多数参赛者的增益在 0.01–0.18 之间——**接近一半的格子落在分辨率带的两三倍以内**。论文报了这个带是负责任的做法，但也意味着 Terminal-Bench 上除了 claude-opus-5 的 0.29 之外，其他排名的可靠性有限。

### 启发与应用前景
这篇最重要的意义是**把「harness 优化」确立为一项可测量的能力**。当前 AI 系统的能力提升有两条路——改模型和改 harness——而后者一直没有评测标准。本周 [2] LongHorizon-Harness 和 [41] OneDayAgent 都在证明 harness 的价值，这篇则给出了「AI 自己改 harness」的度量。

**「现成 harness 裸替换就能把 0.341 提到 0.73」**这个对照数字对所有做 agent 的团队都有直接价值：**在花力气自研 harness 之前，先试试把 opencode / mini-swe-agent / openhands-sdk 这些成熟框架接上**。0.341 → 0.734 这个跨度比绝大多数算法改进都大。

**模型 × harness 的交互不一致**（同一个 harness 对不同模型效果相反）再次印证了 [8] MerchantBench 的结论：**agent 框架的评测必须跨模型做**。

**报告 resolution band** 这个做法值得所有 benchmark 学——它把「多大的差异才值得解读」明确写出来，避免读者过度解读小数点后的排名。

**每轮 jitter 防 benchmark 特化**这个建议也很实用：在用例、工具行为、验证器实现上加扰动，是区分「真改进」和「拟合评测器」的直接手段。

follow-up：作者自己提的两条（jitter 和种子复杂度阶梯）都是明确的改进方向。另外把 optimizer 的改动**按杠杆类型分类**（改提示 vs 改工具 vs 改控制流），能看出模型擅长改什么、不擅长改什么。

---

## 45. From Economic Agents to Agentic Economies: A Systems Blueprint for Economic World Models
**👍 33** · 🏛 香港中文大学（深圳）/ 香港大学 / 南洋理工大学 · [arXiv:2608.06020](https://arxiv.org/abs/2608.06020) · [GitHub](https://github.com/FreedomIntelligence/Awesome-Economic-World-Models) · [项目页](https://economic-world-model.github.io/)

### 问题与动机
**经济世界模型（Economic World Models, EWM）**是生成式经济模型，通过建模**异质 agent、他们的信念与行动、以及他们互动所经由的市场和制度机制**，来模拟经济如何从内部演化。

这篇是**实现路线图**而非新方法：把 EWM 议程翻译成一份工程蓝图，目标是加速下一代经济模拟环境的开发——既作为人类决策者的高保真沙盒，也作为 AI agent 的训练、规划、评测和安全基底。

### 方法与核心创新
核心产出是**六级能力阶梯**，每一级是对工程要求的更强实现：
- **L1 固定规则 agent 世界**
- **L2 自适应 agent 世界**
- **L3 基于 LLM 的 agent 世界**
- **L4 自演化 agent 世界**
- **L5 演化的经济世界**（制度内生化）
- **L6 Sim-to-real 经济孪生**（与真实观测对齐）

四个工程要求（desiderata）在各级的满足情况被明确列表：**内生闭合（endogenous closure）L1–L6 全部满足；行为保真度（behavioral fidelity）从 L3 起满足；演化动力学（evolving dynamics）从 L4 起满足；现实对齐（reality alignment）只有 L6 满足**。

配套还有**分层评测目标**（这部分对实操最有用）：
- **Agent 层**：agent 是否按其角色、目标、信念和约束行事 → 动作有效性、角色一致性、信念校准、行为多样性
- **环境层**：可执行世界是否正确执行约束、机制和结算 → 约束违反率、市场出清误差、会计一致性、结算正确性
- **协同演化层**：agent 和环境在反复互动中是否有效适应 → 适应增益、稳定性、漂移、政策变化、机制重校准质量
- **现实对齐层**：模拟状态是否贴近观测到的经济证据 → 状态误差、趋势匹配、价格误差、成交量误差、波动率误差、修正幅度
- **效率层**：系统在 agent、市场、rollout 长度增长时是否仍可扩展 → 运行成本、内存、并行效率、随 agent 数量的可扩展性

### 关键实验结果
**综述/立场论文，无实验。**

系统性文献调研的核心结论是：**现有工作仍然集中在较低层级的 agent 和模拟环境上**，而**具备自演化 agent、内生制度、持续经验对齐、以及经过验证的经济机制的系统依然罕见**。

论文还有一张表把「工程浪潮」与能力层级对应起来——早期的浪潮主要改进 EWM 的 agent 侧组件，而**要实现交互式和更高层级的经济世界，需要的是环境工程**。

### 局限性与开放问题
论文**没有 Limitations 章节**。作为一篇路线图性质的工作，我的评价：

其一，**六级阶梯的划分标准不完全正交**。L1–L4 是按 agent 能力递进（固定规则 → 自适应 → LLM → 自演化），L5 换成了按「规则是否演化」，L6 又换成了「是否与现实对齐」。**三个不同的划分维度被压进一条线性阶梯**，这会造成分类困难——一个「LLM agent + 内生制度但不与现实对齐」的系统该算 L3 还是 L5？

其二，**「现实对齐」只在 L6 出现，而这恰恰是最难也最重要的一环**。整个阶梯把最关键的验证问题推到了最后一级，而前五级都可以在完全不接触真实经济数据的情况下宣称进展。这个结构可能会鼓励「在模拟里越做越复杂但从不验证」的研究路径。

其三，**没有给出文献调研的方法论**（检索范围、时间窗口、纳入标准），所以「较高层级的系统罕见」这个结论无法被独立核验。

其四，**经济学的验证标准与 AI 的评测标准之间的张力没有被讨论**。经济模拟的经典难题是「你的模型能复现历史数据不代表它捕捉了因果机制」（卢卡斯批判的现代版本），而论文提出的「现实对齐」指标（状态误差、趋势匹配、价格误差）全是拟合度指标，正好落在这个陷阱里。

### 启发与应用前景
这篇对**做经济/社会模拟**的人是有价值的组织框架，对更广的 AI 社区，价值在两个地方：

**分层评测目标是可以直接借用的**。特别是「环境层」的四个指标（约束违反率、市场出清误差、会计一致性、结算正确性）——**它们衡量的是「模拟器本身有没有 bug」，而这是所有基于模拟器的 agent 训练工作都该先测但很少测的**。本周 [30] EnvACE 让策略自己扮演环境，正好缺少这类环境保真度指标。

**「环境工程是更高层级的瓶颈」**这个判断与本周多篇论文呼应：[1] RST 的核心贡献是可执行任务环境的合成，[15] OSReward 的核心投入是环境准备（给每台机器装满真实应用、丰富初始化）。**当前 agent 领域的进展越来越受制于环境而非模型**，这篇给了一个明确的表述。

**经济作为 agent 训练基底**这个定位值得注意：与 [8] MerchantBench（365 天电商经营）对照看，后者正是这篇框架里 L3 级别的一个具体实例——LLM agent + 固定规则环境 + 真实数据标定但无内生制度演化。用这个阶梯给现有 agent 环境定级，能看清各自的边界。

follow-up：把六级阶梯拆成正交维度（agent 能力 / 制度内生性 / 现实对齐）会比线性阶梯更有用；另外「现实对齐」需要引入经济学的机制验证标准而非只看拟合误差。策展的论文列表已开放。

---

## 46. PAST-Bench: Benchmarking the Foundations of Recursive Self-Improvement in Personal Agents
**👍 33** · 🏛 普林斯顿大学 / 芝加哥大学 · [arXiv:2608.04003](https://arxiv.org/abs/2608.04003) · [GitHub](https://github.com/Gen-Verse/PAST-Bench)

### 问题与动机
递归自我改进要求 agent **把积累的经验转化为更好的未来行为**。个人 AI agent 提供了研究这个能力的具体场景，因为它们跨会话保留偏好、任务历史、工具惯例和习得技能。

但一个基本问题没被系统检验过：**保留下来的经验，真的让它们随时间变好了吗？**

### 方法与核心创新
**PAST-Bench** 的核心设计是**性能归因**：每个 agent 跑过有序的全新会话任务序列，在**把「保留经验」打开和关闭的匹配条件下**对比。覆盖 **26 个场景、204 个 episode**，四个能力维度：**memory（记忆）、procedural reuse（流程复用）、information gathering（信息收集）、update（更新）**。

关键创新在于**同时报告两件事**：
1. **后续任务的增益**（persistence-on 减 persistence-off 的差 Δ）；
2. **这些增益是否遵循了预期的「保存 → 检索 → 更新」路径**（Mech 分数）。

这个双报告是为了区分「分数涨了」和「因为正确的机制涨了」。论文明确指出：**headline 增益相同的 agent，在「这个增益是否有预期路径的证据支撑」上可能差别很大**。

基准还支持**模型侧和框架侧的双重隔离**——固定框架换模型、固定模型换框架，这在同类基准里是独有的（对比表里 GAIA、AgentBench、OSWorld 都不支持保留经验维度，LongMemEval 和 LoCoMo 支持保留经验但不支持框架比较）。

基于诊断发现，论文还开发了 **Hermes+**，在 agent loop 的各阶段加了五个针对性干预。

### 关键实验结果
**七个基座模型 × 四个 agent 框架。改进是真实的，但在各能力上很不均匀。**

**模型侧**（Hermes 框架，persistence off → on）：
- GLM-5.1：总体 0.52 → **0.71（Δ +0.20）**，其中信息收集 **+0.36（占 46%）**、状态 +0.23、流程 +0.11、记忆 +0.09，Mech 0.70
- Kimi K2.6：0.57 → 0.73（Δ +0.17），**状态 +0.33（49%）、信息 +0.27（40%）**，但**记忆只有 +0.03（4%）、流程 +0.05（7%）**，Mech 0.72
- DeepSeek-V4-Pro：0.50 → 0.67（Δ +0.17），状态 +0.33、信息 +0.18、记忆 +0.12、流程 +0.06，Mech 0.71

**注意 Kimi K2.6 和 DeepSeek-V4-Pro 的总体 Δ 都是 +0.17，但构成完全不同**——这正是论文强调的「相同 headline 增益可能来自不同路径」。

**框架侧**（固定 MiniMax-M2.7）：
- nanobot：总体 Δ **+0.13**，但**流程 −0.06**，Mech 0.57
- ZeroClaw：+0.12，记忆 **+0.29** 最强，流程 −0.04，Mech 0.55
- **Agent-Zero：总体 Δ −0.08**——**开启持久化反而让它变差**，记忆 **−0.27**、更新 −0.13，Mech 0.39
- Hermes：+0.13，Mech **0.64**
- **Hermes+：记忆 +0.27、信息 +0.12、更新 +0.24，但流程 −0.02**

**Agent-Zero 的负增益是全文最值得注意的单点结果**：一个 agent 框架在打开经验保留后总体变差，记忆能力掉了 0.27。这说明**持久化机制设计不当会主动伤害性能**。

**四个框架里有三个在「流程复用」上是负的或接近零**（−0.06 / −0.04 / −0.02），只有 Agent-Zero 是 +0.11——**流程复用是当前持久化 agent 最普遍的短板**。

### 局限性与开放问题
论文的**自我评估极其克制，值得单独表扬**。结论里明确写道：Hermes+ 把总体 Δ 从 +0.13 提到 +0.15、Mech 从 0.64 提到 0.73，**「这个 +0.02 的总体差异小于运行间波动」**，效果在各能力和基座模型上不一致，**所以把 Hermes+ 当作诊断脚手架而非普适改进**。

一篇提出自家框架的论文主动说「我的主要指标提升小于噪声」，这在本周 53 篇里是唯一的。

Future Work 也列了四条实质性的：
1. **生态效度和时间跨度**——当前任务族是**合成构造且孤立评测**的，需要人类撰写和交互衍生的场景、更长的任务序列、以及**一个任务族积累的经验影响另一个任务族**的设定，才能测试长程状态维持、跨域迁移、以及独立习得的记忆/流程/修正之间的**互相干扰**。
2. **能力空间需要扩展**——当前四个能力是在线自演化的必要基础，但**不覆盖更强形式的递归改进**（获取此前不可用的工具策略、构建修订长程计划、跨多 agent 协调经验、改进「决定存什么/取什么/验什么/更新什么」的机制本身）。
3. **机制归因需要加强**——当前的机制证据分衡量的是**与预期持久化路径的一致性，而非因果必然性**。更强的评测应结合轨迹证据与**反事实干预**（删除、替换或破坏候选产物，测量行为变化）。还应支持**多条语义有效的持久化路径**，因为不同 agent 可能把同一经验编码成记忆、技能、结构化产物或修订后的策略。
4. **持久化机制不应被当作可统一组合的**——未来 agent 可以学会**动态路由经验**到记忆、技能、会话历史之间，同时检测这些基底之间的冲突、冗余和陈旧状态。

我唯一要补充的是：**204 个 episode、26 个场景、3 次运行**的规模，配合「+0.02 小于运行间波动」这个自陈，说明**大部分框架间的排名差异（0.12 vs 0.13）都不可靠**。真正稳健的结论只有两个：**Agent-Zero 的 −0.08 是真的负**，以及**流程复用普遍不 work**。

### 启发与应用前景
最重要的方法论贡献是**「结果增益 + 机制证据」的双报告**。只看分数涨了多少，无法区分「agent 真的用上了保存的经验」和「它碰巧在后面的任务上表现更好」。**Mech 分数（0.39 到 0.73 的跨度）暴露了这个差异**——Agent-Zero 的 Mech 只有 0.39，与它的负增益一致。任何做记忆/持久化系统的团队都该加这一项。

**「持久化会让 agent 变差」这个反例（Agent-Zero −0.08、记忆 −0.27）**是本周对 agent memory 领域最有价值的警告。与 [20] Skill-α 的「去掉 Merge/Prune 后表现低于不训练」、[31] 的「推断属性线性累积很少修正」放在一起看，**三篇独立工作指向同一个结论：只增不删、不加验证的经验累积是主动有害的**。

**「流程复用普遍是负的」**这个发现指出了一个具体的技术空白：agent 能记住事实（memory Δ 高达 +0.29），但**记不住怎么做事**。这与 [20] 用 RL 学习 skill 编辑的工作正好互补。

follow-up：作者自己提的**反事实干预做机制归因**（删掉那条记忆看行为变不变）是最直接的改进，也是把「相关证据」升级成「因果证据」的必经之路。代码已开源。

---

## 47. Fewer Clarifications, Better Code: Benchmarking Cross-Session Personalized Ambiguity Adaptation in Coding Assistants
**👍 32** · 🏛 东南大学 / 香港科技大学 · [arXiv:2607.26611](https://arxiv.org/abs/2607.26611)

### 问题与动机
AI 辅助编程把非正式的用户意图翻译成可执行软件，但**编码请求常常含有歧义，而这些歧义以用户特定的方式跨任务跨会话反复出现**——同一个用户总是省略同一类信息（比如从不说要不要错误处理、总是默认用某个库、习惯性地不指定返回格式）。

现有的消歧方法**在当前会话内孤立地处理每个歧义请求**，通常靠追问澄清。但**同一用户已解决的历史会话能否作为记忆，用来解决新会话中反复出现的个性化歧义**，这个问题少有研究。

### 方法与核心创新
论文把**个性化歧义适应**形式化为一个新任务：给定用户此前已解决的编码会话和一个新的歧义请求，助手应该**识别出反复出现的歧义模式、产出符合意图的可执行方案、并最小化澄清次数**。

**CAPA 基准**通过六种机制刻画个性化编码歧义，并用**受控的三阶段生成流水线**把这些机制注入到无歧义的可执行任务里。包含 **600 个编码会话、60 个平衡的「用户×歧义」单元**，其中 300 个是留出评测会话。

三个指标：**ES（可执行成功率）、FT-ES（首轮成功率）、TTC（完成所需轮数）**。

### 关键实验结果
12 个 LLM 在无历史和同用户历史两个条件下评测。

**最关键的发现是「历史主要提升的是首轮成功率，而非最终成功率」**：
- **Claude Opus 4.8**：ES 88.0% → 90.0%（**只 +2.0pp**），但 **FT-ES 24.3% → 60.3%（+36.0pp）**，TTC 2.800 → 2.113
- **GPT-5.5**：ES 74.3% → 84.3%（+10.0pp），**FT-ES 2.3% → 31.0%（+28.7pp）**，TTC 4.417 → 2.970
- **GLM-5.2**：ES 85.3% → 89.7%（+4.4pp），**FT-ES 18.7% → 46.7%（+28.0pp）**
- **ChatGPT-5.6-Sol**：**ES 79.0% → 78.7%（−0.3pp，唯一负值）**，但 FT-ES 2.7% → 18.3%（+15.6pp）

也就是说：**有没有历史，最终都能做对；有历史则少问几轮就能做对。**这个区分很重要——它说明**历史的价值是效率而非能力**。

**GPT-5.5 无历史时 FT-ES 只有 2.3%** 这个数字特别刺眼：100 次里只有 2 次能首轮就猜对用户的隐含偏好。而 Claude Opus 4.8 无历史就有 24.3%，说明**不同模型对「不问就猜」的倾向差别巨大**。

**小模型受益最大**：Llama-3.3-70B ES 从 10.0% 涨到 30.0%（**+20.0pp**），Qwen3.5-27B 从 56.0% 到 74.3%（+18.3pp）。这些模型无历史时基本做不了这个任务。

**难度分层**（按无历史时的会话长度定义）显示复杂任务上历史的帮助明显减弱：GPT-5.5 简单 91.75%、中等 93.26%、**复杂只有 71.05%**；GLM-5.2 是 97.94% / 96.63% / **75.44%**。

**打乱历史的对照实验**是最有说服力的：GPT-5.5 用**打乱的（错误用户的）历史** ES 是 85.00%，而无历史是 74.33%——**即使历史配错了人，也比没有历史好**。这说明**相当一部分增益来自「看到了同类任务的解法范例」，而非「学到了这个用户的偏好」**。论文提出的轻量用户历史门控方法正是针对这一点。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我的观察：

其一，**打乱历史的对照实际上削弱了核心论断**。GPT-5.5 在打乱历史下拿到 85.00 ES，而正确匹配历史（表 1）是 84.3——**打乱的比正确的还略高**。GLM-5.2 打乱是 89.00 vs 正确 89.7，DeepSeek 打乱 77.67 vs 正确 79.0。三个模型上，**「个性化」相对「随便给点历史」的增量在 0.7 个点以内**。论文把这个对照放在表 3 而非主结论，但它其实是最重要的发现：**当前所谓的个性化适应，大部分是通用的 few-shot 效应**。

其二，**歧义是「注入」的**——用受控流水线把六种机制注入到原本无歧义的任务里。这保证了可控性和可验证性，但**注入的歧义模式是否反映真实用户的歧义习惯**没有验证。60 个「用户×歧义」单元是合成的用户画像。

其三，**FT-ES 的绝对值普遍很低**。即使有历史，最好的 Claude Opus 4.8 也只有 60.3%，多数模型在 20–47%。这意味着**大多数情况下助手仍然需要多轮**，「fewer clarifications」的标题成立，但离「no clarifications」很远。

### 启发与应用前景
最实用的结论是**「历史提升的是首轮命中率而非最终正确率」**。对做编程助手的团队，这直接指明了历史/记忆功能的价值主张：**不是让 AI 写得更对，而是让它少问几句**。TTC 从 4.4 降到 3.0 意味着用户少打两轮字——这在体验上是实质改善，但不应被宣传成「更准确」。

**打乱历史对照的启示更重要**：**在你宣称做了「个性化」之前，先测一下随便给点别人的历史效果如何**。如果差不多，那你做的是 few-shot 而非个性化。这个对照应该成为所有个性化/记忆系统的标配消融。

**复杂任务上历史帮助减弱**（GLM-5.2 从 97.94% 掉到 75.44%）也是个实际约束：记忆机制在简单重复性任务上最有效，而复杂任务的歧义可能是**任务特有的而非用户特有的**。

follow-up：最该做的是**用真实用户的历史会话**验证，而非合成用户画像；以及把「用户特有的歧义」和「任务通用的歧义」在数据层面分开，才能真正量化个性化的贡献。

---

## 48. LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models
**👍 31** · 🏛 中国人民大学 / 蚂蚁集团 · [arXiv:2608.03457](https://arxiv.org/abs/2608.03457)

### 问题与动机
扩散语言模型（dLLM）是自回归建模的替代路线，但 **MoE dLLM 的 scaling 行为几乎无人理解**。当前的 dLLM 工作要么是 dense 架构，要么直接照搬 AR 的超参和配比经验——而这些经验是否适用于扩散目标，没人验证过。

### 方法与核心创新
论文做了**系统的受控扫描**，覆盖优化超参、算力分配、MoE 架构三块，并把发现与 AR 模型已报告的 scaling 趋势做定量对比。核心发现有六条：

**优化侧**：
- **最优 nominal batch size 随算力增长得比 AR 更陡**；
- **最优学习率随算力衰减得比 AR 更快**。

**算力分配**：
- IsoFLOP 分析显示**接近平衡但略偏数据侧**：**M\* ∝ C^0.475、D\* ∝ C^0.525**。对比其他定律——Kaplan 是 0.73/0.27（极度偏模型）、Chinchilla 0.49/0.51、Ling MoE(AR) 0.5095/0.4905、SMDM Diffusion(Dense) 0.634/0.366、Quokka 0.514/0.486、DLMs 0.566/0.434。**MoE dLLM 的前沿比 dense dLLM 的前沿更偏数据**。

**MoE 架构**：
- **规模越大越倾向更低的激活比**（固定激活容量下用更大的专家池）；
- **中等专家粒度 G=8–16 跨规模都稳健**；
- **共享专家占激活容量的最优比例 S=33.3% 跨规模保持稳定**。

据此训练了 **LLaDA MoE v2**，一个 **30B-A3B 的 dLLM，从零在 23.5T token 上训练**。

### 关键实验结果
**预训练**：用**约 Qwen3 的 65% 的预训练 token**（即少 35%），在多个知识、推理和编码基准上**接近 Qwen3**。

**SFT 后**（仅监督微调，无 RL）：在八个推理和编码基准中的**七个上超过 SDAR Chat**，多个任务上仍接近 Qwen3。

**架构效率**：匹配或超过 7B-A1B 配置，但算力显著更低。

**这两个数字合起来的意义**：一个扩散语言模型用更少的 token 预算逼近同代 AR 模型，说明扩散路线在 scaling 上并没有天然劣势——至少在这个规模上。

### 局限性与开放问题
论文**没有独立 Limitations 章节**（抓取的正文里未见）。我的观察：

其一，**「接近 Qwen3」这个措辞需要具体数字才能判断**。抓取的表 3 被截断，只看到列头（LLaDA MoE v2 / SDAR Sci / LLaDA MoE / Dream 7B / LLaDA 8B / Qwen3）。「approaches」可以是差 1 分也可以是差 10 分，而这决定了整个工作的说服力。**用 65% 的 token 达到 90% 的能力和达到 99% 的能力，是完全不同的结论。**

其二，**没有报推理速度**。扩散语言模型的核心卖点是并行解码带来的速度，而这篇通篇在讲 scaling 和能力，**没有任何吞吐或延迟数字**。对比本周 [39] DiffusionGemma 明确报了 1,479 tokens/秒和 5 倍加速，这篇的定位就变成了「扩散模型也能训得不错」，而没有回答「那为什么要用扩散」。

其三，**scaling 律的拟合细节缺失**。M\* ∝ C^0.475 这个指数是在什么算力范围内拟合的、用了多少个点、拟合的置信区间是多少，抓取的正文没有给出。scaling 律最容易在外推时失效，而 30B-A3B 的验证点相对拟合范围可能是外推的。

其四，**「只做 SFT 就超过 SDAR Chat 七项」**——SDAR Chat 是否做了 RL 没有说明。如果对方做了 RL 而这边只做 SFT，那是有利于本文的比较；如果对方也只有 SFT，则是对等的。

### 启发与应用前景
最有实用价值的是那三条 **MoE 架构的经验法则**，它们和 AR MoE 的常识有明确差异，值得任何要训 MoE dLLM 的团队直接采用：
- **规模越大越该降激活比**（大专家池 + 少激活）；
- **专家粒度 G=8–16 是稳健区间**，不用为不同规模重调；
- **共享专家占激活容量 33.3%** 这个比例跨规模稳定——这是个可以直接抄的数字。

**「最优 batch size 增长更陡、最优学习率衰减更快」**这两条对训练配置有直接影响：**照搬 AR 的超参 scaling 经验会在大算力下同时把 batch 开小、把学习率设高**，两个都错。

**数据侧倾斜（D\* ∝ C^0.525）**这个结论意味着：**训 MoE dLLM 时，同样的算力应该比 AR 更多地花在数据上而非模型上**。而且 MoE dLLM 比 dense dLLM 更偏数据（0.525 vs 0.434），说明 MoE 架构在扩散目标下对数据更饥渴。

follow-up：最该补的是**吞吐/延迟数据**和**与同规模 AR 的完整对照表**。与 [39] DiffusionGemma（AR 微调成扩散）和 [12] AURORA-LM（连续隐空间扩散）合看，本周三篇扩散语言模型分别代表了从零训练、AR 转换、连续隐空间三条路线——**这个方向正在快速分化**。

---

## 49. SKT: Skill-Use Training at Scale via Verified Synthetic Data Generation
**👍 31** · 🏛 上海人工智能实验室 · [arXiv:2608.02287](https://arxiv.org/abs/2608.02287)

### 问题与动机
Agent Skill 已经成为给语言模型 agent 装配可复用过程知识的重要机制（Claude Skills 那种 SKILL.md）。但论文指出一个被忽视的前提问题：**光给 skill 并不保证当前模型能有效地识别、应用和协调它们**。

换句话说，社区在造 skill，但**没人训练模型「怎么用 skill」**。给一个不会用工具箱的人一箱工具，工具再好也没用。

### 方法与核心创新
**SKT** 是一个**经验证的数据合成流水线**，构造 skill 使用的训练数据。配置里几个设计值得注意：

- **skill 基数 K = {1, 2, 3}**——任务被设计成需要 1 到 3 个 skill 协同，直接训练「协调多个 skill」这个能力，而不只是「用一个 skill」。
- **任务侧模型组件**：DeepSeek V4 Pro + Claude Agent harness 生成任务；
- **难度控制器**：Qwen3.5-35B-A3B + OpenCode，参数 N=5、θ_easy=0.6——即用一个中等模型采样 5 次，成功率高于 0.6 的任务被判为太简单而过滤掉。**这是一个自动的难度校准机制。**
- **教师模型**是四个的集合：MiniMax-M2.5、GLM-5、Qwen3.5-397B-A17B、DeepSeek V4 Pro。

### 关键实验结果
在**匹配 harness、外部提供 skill** 的条件下评测（四次完整运行的均值 ± 样本标准差）：

**Qwen3.5-9B + OpenCode**：
- SkillsBench **5.80 → 15.79**（近 3 倍）
- SkillEval **55.24 → 72.48**（+17.2）
- MolBench-Bind（分子科学）**33.11 → 44.59**（+11.5）
- AgentSkillOS（产物生成）79.61 → 84.70（+5.1）

**Qwen3.5-9B + DeepAgents**：4.94 → 13.62、51.62 → 70.53、31.76 → 47.30、74.03 → 79.84

**Gemma 4 E4B-IT + OpenCode**：7.08 → 10.28、61.01 → 72.85、34.46 → 45.27、49.41 → 56.74

**跨两个 harness、两个模型家族，四个基准全部提升**，而且提升幅度都远超标准差（多数标准差在 1–2.6 之间）。

**SkillsBench 上的绝对值极低（原始 5.80、训练后 15.79）**很值得注意——这说明当前模型在通用 skill 使用上几乎是不及格的，即使训练后也只有 15.79 分。

**混合 harness 训练的消融**（Qwen3.5-9B）：用评测 harness 自己的轨迹训练（Specialist）vs 用两个 harness 的轨迹混合训练（Mixed）：
- OpenCode 上：SkillsBench Specialist 15.79 > Mixed 14.24；SkillEval **Mixed 74.05 > Specialist 72.48**；MolBench **Mixed 45.27 > Specialist 44.59**；AgentSkillOS Specialist 84.70 > Mixed 82.25
- DeepAgents 上：SkillsBench **Mixed 13.93 > Specialist 13.62**；MolBench Specialist 47.30 > Mixed 44.59

**混合训练和专用训练互有胜负，没有一致优劣**——这说明 skill 使用能力有相当一部分是 harness 特定的，跨 harness 迁移不完全。

### 局限性与开放问题
论文**没有独立 Limitations 章节**（抓取范围内）。我的观察：

其一，**教师模型是四个前沿模型的集合**（MiniMax-M2.5、GLM-5、Qwen3.5-397B-A17B、DeepSeek V4 Pro），学生是 9B 和 E4B。这本质上是**多教师蒸馏**，「合成数据」的能力上界由这四个教师封顶。论文标题强调「verified synthetic data generation」，但验证机制（难度控制器 θ_easy=0.6）过滤的是**太简单**的任务，而不是验证**教师轨迹的正确性**。轨迹质量如何保证，抓取部分没有交代。

其二，**只训到 9B**，且两个模型都是相对小的（Qwen3.5-9B、Gemma 4 E4B）。**大模型是否还需要专门的 skill 使用训练**是开放的——有可能强模型本来就会用。论文没有在更大规模上验证。

其三，**SkillsBench 上 15.79 的绝对值**说明问题远未解决。三倍提升听起来很多，但从 5.80 到 15.79 意味着**依然有 84% 的任务做不对**。

其四，**混合 vs 专用的不一致结果**没有被深入分析。这其实是个有价值的现象：**skill 使用能力有多少是通用的、多少是 harness 特定的**，值得单独研究，而论文只是报了表格。

### 启发与应用前景
最重要的问题意识值得单独记：**「有 skill」和「会用 skill」是两回事**。当前 agent skill 生态（Claude Skills、各种 skill 市场）的注意力全在「造更多更好的 skill」上，而这篇指出**模型侧的 skill 使用能力才是瓶颈**——SkillsBench 上 5.80 分说明模型基本不会主动识别和协调 skill。

**难度控制器（用中等模型采样 N 次，成功率超阈值就丢弃）**是个便宜好用的合成数据过滤器。它把「这个任务是否有训练价值」变成可自动判定的，比人工分级或靠 LLM 打分靠谱。这与 [1] RST 用固定 solver 的 pass@4 标定难度是同一个思路。

**K = {1, 2, 3} 的 skill 基数设计**也值得抄：如果你要训练「协调多个组件」的能力，就必须在数据里显式构造需要多个组件协同的任务，否则模型只学会单点调用。

**跨 harness 的迁移不完全**这个发现对实践有直接影响：如果你的 agent 用 OpenCode，就该用 OpenCode 的轨迹训练；混合训练不是免费午餐。

follow-up：最紧要的是**验证轨迹质量的机制**（当前只过滤难度不验证正确性），以及**在更大模型上确认 skill 使用训练是否仍然必要**。

---

## 50. On-Policy Delta Distillation for Multilingual Math Reasoning
**👍 30** · 🏛 NAVER AI Lab · [arXiv:2608.05802](https://arxiv.org/abs/2608.05802)

### 问题与动机
on-policy 蒸馏（OPD）正在成为 RL 之外有前景的后训练替代方案，但**它在多语言场景下的有效性少有研究**。

这个问题有实际意义：多语言推理有个特有的失败模式——**模型可能推理对了，但用错误的语言回答**。而 OPD 是否会加剧这个问题，没人测过。

### 方法与核心创新
研究 OPD 及其进阶变体 **OPD²（On-Policy Delta Distillation）**在英语、韩语、日语数学推理上的表现。

**OPD² 的机制**：用**后训练 teacher 与其 base 模型之间的概率差**作为学习信号，而非直接用 teacher 的分布。

这个思路与本周 [23] W2S-OPD 高度同构——都是**用「差」而非「绝对值」作为监督信号**，只不过 W2S-OPD 是用两个弱模型的差去提升强学生，OPD² 是用同一个模型后训练前后的差。共同的直觉是：**「差」隔离出了「后训练/规模带来的能力增量」，而不携带 teacher 的绝对水平和它的缺陷。**

### 关键实验结果
**Qwen3-1.7B 多语言 OPD 训练**：

**英语基准**（非思考模式）：基座均分 47.6 → OPD 61.7 → **OPD² 65.7**。分项 M-MMLU 从 17.9 → 47.0 → **55.3**，KSM 21.6 → 42.7 → **48.8**，MATH 74.4 → 88.0 → **91.1**。思考模式下：70.4 → 71.4 → **73.4**（OPD 只提升 1.0，OPD² 提升 3.0）。

**目标语言回复率是这篇最有价值的发现**（PolyMath 上用问题语言回答的比例）：

| 训练数据 | OPD-KO | OPD-JA | OPD²-KO | OPD²-JA |
|---|---|---|---|---|
| 仅英语 | 83.1% | **36.9%** | **36.1%** | **30.8%** |
| 多语言 | 72.9% | 70.1% | **97.6%** | **95.2%** |

两个关键观察：
1. **只用英语数据做 OPD，也能提升韩语和日语的性能，但会把回复推向英语**——日语回复率掉到 36.9%（OPD）和 30.8%（OPD²），即**三分之二以上的日语问题被用英语回答了**。
2. **用多语言数据时，OPD² 的语言保持能力远超 OPD**——韩语 97.6% vs 72.9%，日语 95.2% vs 70.1%。

第二张表（另一组配置）里 OPD 多语言的韩日回复率是 99.7%/99.8%，OPD² 是 90.5%/90.9%——**方向反了**。论文抓取部分对这个不一致没有解释，这是需要注意的地方。

总体结论：**OPD² 一致优于原始 OPD，在韩语和日语上提升尤其明显，并且普遍缩小了英韩性能差距**。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。这是一篇较短的工作（arXiv 上体量不大），我的观察：

其一，**规模只有 Qwen3-1.7B**。1.7B 是相当小的模型，多语言能力本来就弱（基座 M-MMLU 只有 17.9），提升空间大。**在更大模型上 OPD² 相对 OPD 的优势是否保持**完全未知。

其二，**两张回复率表的方向矛盾**。同一篇论文里，一处显示多语言 OPD² 的语言保持（97.6%/95.2%）远好于 OPD（72.9%/70.1%），另一处显示 OPD（99.7%/99.8%）好于 OPD²（90.5%/90.9%）。抓取的表格标注不完整（可能是思考/非思考模式的区别），但**这个不一致削弱了「OPD² 更好地保持目标语言」这个结论**。

其三，**只测了三种语言**，且英语、韩语、日语在资源丰富度上都属于中高资源。真正的低资源语言（本周 [51] 处理的现代希腊语就是一例）上是否成立，没有验证。

其四，**「英语训练也能提升韩日性能」这个发现虽然有用，但代价（语言漂移）被量化得很清楚——日语回复率 30.8%**。这意味着**跨语言迁移在这个设定下是「能力迁移了但输出语言没跟上」**，实用价值有限。

### 启发与应用前景
最有价值的实践结论：**做多语言后训练时，必须单独测「目标语言回复率」，而不能只看准确率**。一个在日语数学题上准确率很高但 69% 的时候用英语回答的模型，在产品里是不可用的。这个指标应该成为多语言评测的标配。

**「用后训练前后的概率差作为蒸馏信号」**这个思路与 [23] W2S-OPD 独立收敛到同一个原理，这本身是个信号：**在 logit/概率空间做减法来隔离「能力增量」，可能是一个比直接蒸馏绝对分布更普适的范式**。本周还有 [28] VAD 用反事实前向差分离视觉归因、[5] DAPD 用信息条件匹配——**「做差」正在成为蒸馏领域的共同工具**。

**「英语数据能提升多语言能力但会造成语言漂移」**这个 trade-off 对资源受限的团队很实际：如果你只有英语高质量数据，可以用它提升多语言推理，**但必须配合少量目标语言数据来锚住输出语言**。

follow-up：最该做的是**在更大规模上复现**，以及**澄清两张回复率表的矛盾**。另外把 OPD² 的「delta 信号」思路推到低资源语言，是更有价值的方向。

---

## 51. Teaching Nemotron Greek: Mining a Corpus, Adapting Retrieval, and Grounding Generation for Modern Greek across Specialist Domains
**👍 30** · 🏛 Sophea AI / KIEFER SA（雅典）· [arXiv:2608.05138](https://arxiv.org/abs/2608.05138)

### 问题与动机
**现代希腊语不在 NVIDIA Nemotron 检索模型的支持语言列表里，也不在任何主流多语言检索基准（BEIR、MIRACL）里**——而希腊的法律、能源、金融和临床文档恰恰是「又长又术语密集」的那种文本，正是 RAG 该服务的对象。

论文的立场很尖锐：「稠密检索取代了词法检索」这个信念是**承重的**——生产 RAG 栈正是因为相信它，才装一个多语言 embedder 然后**把 BM25 完全丢掉**。而这个信念**碰到希腊语就不成立**。

### 方法与核心创新
端到端适配 Nemotron 家族到希腊语：**挖语料、训检索栈、合成 reader 监督、建评测基准**，并**测量每个阶段实际买到了什么**。

数据阶段就有两个独立发现：
- **基本没有原生希腊语指令数据可训**，任何有用规模的池子都**必须靠翻译**；
- **希腊文档 chunk 足够长，以至于 max_len=512 会静默截断 87% 的训练对**——这是个纯工程细节，但会毁掉整个训练。

### 关键实验结果
**最不舒服的建模结论是**：**在全部五个领域上，一个没有任何学习参数的 BM25 词法基线，跑赢了所有测试过的现成多语言 embedder，包括一个 8B 的**。域外验证同样成立——BM25 的 0.6602 击败了 8B 的 0.5936 和 4B 的 0.6119。

**微调有效且幅度巨大**：在 65,773 个希腊检索对上微调一个 1B embedder，**nDCG@10 从 0.362 提到 0.835**。

**适配迁移为「语言能力」**：在一个两个模型都没训练过的通用希腊语料上（HERA 检索轨道，4,946 查询 vs 30 万希腊维基段落），适配后的 1B 比它自己的未适配基座**高 +0.399**（0.1651 → 0.5637，95% CI [+0.387, +0.410]），0.6B 高 +0.049。作者的评论很精准：**「1B 基座停在 0.165，勉强能用，适配把它推到了距离一个骨干本来就会希腊语的 0.6B 模型只差 0.007 的位置。这是本文中关于『暴露量而非容量』最干净的证据。」**

**但相对 BM25 的优势不迁移**：域外 BM25 是 0.6442，**击败两个适配模型**（比 1B 高 +0.081、比 0.6B 高 +0.069），现成的 Qwen3-Embedding-4B 也比适配 1B 高 +0.055。**「我们在适配过的领域领先，在通用希腊语上落后，两者都报。」**

**正确的做法是加上 BM25 而非替换它**：用 RRF（k=60，各系统 top 100，等权重不调参）融合，域外得 **0.6715 vs BM25 的 0.6442（+0.027，CI [+0.019, +0.035]）**，Recall@10 从 0.733 提到 0.782。域内融合到 0.6798（比稠密模型单独高 +0.013）。**权重在留出折上选、在另一折上打分，且调参版（0.6749）与不调参版不可区分**——增益不依赖调参。另一个 embedder（微调 Qwen3-0.6B）行为一致，说明**这是「稠密+词法融合在希腊语上」的属性，而非某个 checkpoint 的偶然**。

**reader 侧**：LoRA 微调一个 30B-A3B MoE 作为接地 reader，**判定的答案正确率从 29.4% 提到 66.9%**。

### 局限性与开放问题
**这篇的自我批判是本周 53 篇里最彻底的**。论文明确**「报告了我们自己的仪器在这一路上骗我们的四种方式，包括我们自己的两个论断在第二次更大规模的评测中没有复现」**。

具体的自陈局限：
- **域内相对 BM25 的优势在更大评测中站不住**：+0.027，95% CI **[−0.0005, +0.054]，区间触及零**——「在这个集合上，以这个样本量，我们无法证明相对词法检索的优势」。
- **两个评测的一个共同性质限制了结论**：**HERA 的查询和自建集的查询都是 LLM 从金标段落生成的**，这让查询和金标共享词汇，**在两边都偏袒词法匹配**。作者指出这个共同偏差不能解释两者之间的反转，但确实意味着 **BM25 的绝对位置在两个评测里都可能被抬高了**。
- 论文标题里「across specialist domains」这个限定**在做实际工作**——「一个在通用希腊语上部署这个 embedder 的读者，应该预期输给词法基线」。

我几乎没有可补充的批评。唯一想指出的是：**LLM 生成查询这个偏差如果被消除，BM25 的强势可能会大幅减弱**——那样整篇论文最反直觉的结论（BM25 打败 8B embedder）就需要重新评估。作者已经点出了这一点，但没有做无偏查询的验证。

### 启发与应用前景
这是本周**方法论诚实度最高**的一篇，值得所有做工程报告的人当范本。三个具体做法值得学：
1. **报告置信区间并接受「区间触及零 = 无法证明」**；
2. **主动列出「我们的仪器骗了我们的四种方式」**，包括自己没复现的论断；
3. **把打败自己的基线当成组件而非对手**——「A baseline that beats you is a component, not only a rival」。

**技术上最有价值的结论**：**在分布外语言上，不要丢掉 BM25**。生产 RAG 栈默认「装个多语言 embedder，删掉词法检索」的做法，在支持语言列表之外是危险的。而**RRF 融合等权重不调参就有 +0.027 增益**，成本几乎为零。

**「max_len=512 静默截断 87% 训练对」**这个工程细节值得单拎出来——它不会报错，只会让训练效果莫名其妙地差。任何在长文档语言（希腊语、德语、以及大量非英语语言）上训检索模型的人都该检查这一点。

**「暴露量而非容量」**这个论断（1B 基座 0.165 → 适配后 0.5637，逼近本来就会希腊语的 0.6B）对低资源语言工作是个鼓舞：**问题往往不是模型太小，而是它没见过这门语言**。

follow-up：用非 LLM 生成的真实查询重做评测，是验证 BM25 强势是否成立的关键。模型和 HERA 基准已开放。

---

## 52. DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces
**👍 30** · 🏛 香港科技大学（广州）/ 清华大学 · [arXiv:2608.03451](https://arxiv.org/abs/2608.03451) · [GitHub](https://github.com/HKUSTDial/DataSpace) · [项目页](https://dataspace-bench.github.io)

### 问题与动机
数据 agent 让人可以用自然语言在组织的工作空间上做分析，而**相关证据可能散落在数据库、结构化文件、长文档和多媒体里**。

现有基准**大多把结构化查询、检索、开放式分析各自孤立起来**：Spider/BIRD 只有数据库，HotpotQA/CRAG 只有文档，MMLongBench-Doc 只有长文档，DABStep/KramaBench 覆盖文件+文档但没有媒体和跨制品发现。**异构证据发现、完整表格输出、确定性评测三者的统一，一直不充分。**

### 方法与核心创新
**DataSpace** 要求数据 agent 从**任务局部的异构工作空间**产出**可验证的表格结果**。410 个任务。

对比表显示它是唯一同时覆盖以下全部维度的：**输入制品**（DB / 文件 / 文档 / 媒体）、**工作空间要求**（跨制品、发现、长文档、文档→记录、跨语言）、**输出与评测语义**（完整表格、model-free、schema 不变）。

其中几个设计特别值得注意：
- **完整表格输出**——不是回答一个数字或一段话，而是产出完整的结果表。这让评测可以是确定性的。
- **model-free 评测**——不需要 LLM 判分。
- **schema 不变性**——答案的正确性不依赖列名和列序的具体形式。
- **文档→记录**——要求从非结构化文档里抽出结构化记录再参与计算，这是真实数据分析的常见环节。

### 关键实验结果
六个多模态模型 × 五个常用 agent harness 的评测中，**最好的准确率只有 66.34%**（另一处提到受控对比中 GPT-5.6 Sol 达 64.63%、Claude Code harness 44.63%、Claude Sonnet 5 只有 32.93%）。

**多模态证据整合和 join 一致地降低准确率**——这两个是最难的环节。

**失败分析是这篇最有价值的部分**（对最强骨干 Grok 4.5 的 136 次失败做 trace 级分析）：
- **答案物化（answer materialization）是最大类，占 71/136（52.2%）**。其中 **60 次是「所需的内部结果已经拿到了，但输出的表格多了或少了列」**。
- **任务规格与意图占 31/136（22.8%）**，其中 17 次是**错误地形成了请求的输出或行粒度**。
- 这两条通向错误答案 schema 的不同路径合计 **77/136（56.6%）**。
- **只有 3 次失败源于选错了证据源**；抽取和语义接地合计 21 次。

**这个分布颠覆了直觉**：大家以为异构工作空间上的数据 agent 难在「找不到数据」，而实际上**找到制品几乎不是问题（3/136），问题在于「拿到了正确的中间结果却输出了错误的表格结构」**。

作者还强调**「症状不是诊断」**——同一个评测器输出可以由不同原因造成。

### 局限性与开放问题
论文**没有独立 Limitations 章节**。我的观察：

其一，**「完整表格 + 确定性评测」这个设计既是优点也是约束**。它让评测无需 LLM 判分（避免了 [15] OSReward 揭示的裁判宽容偏差问题），但也意味着**评测对输出格式极其敏感**——而失败分析恰恰显示 52.2% 的失败是格式问题。这里有个循环：**基准的严格性本身制造了它测出的主要失败模式**。「多了一列」到底是 agent 的能力缺陷还是任务规格表述不够明确，论文用 schema 不变性做了部分缓解，但 60 次「加列或漏列」的失败仍然可能部分归因于任务表述。

其二，**410 个任务**要覆盖 DB/文件/文档/媒体四类制品 × 多种工作空间要求，每个组合的样本量会很小。跨类别的细分结论需要谨慎。

其三，**成本没有充分报告**。表格里提到「视频渲染 Claude Sonnet 4.6 约 $0.58」，说明构建成本被跟踪了，但**运行一次完整评测的成本**（六个模型 × 五个 harness × 410 任务）没有汇总。

其四，**Claude Sonnet 5 的 32.93% 远低于 GPT-5.6 Sol 的 64.63%**，这个近乎两倍的差距很反常，论文抓取部分没有分析原因（是能力问题还是 harness 适配问题）。

### 启发与应用前景
最有实用价值的是那个**失败分布**：**52.2% 的失败是「算对了但输出格式错了」，而只有 2.2% 是「找错了数据源」**。对做数据 agent 产品的团队，这直接指明优先级——**不要再花力气优化检索和发现，去优化输出 schema 的对齐**。具体手段可以是：让 agent 在产出前显式声明输出 schema 并与任务对照、或者提供 schema 模板。

**「完整表格 + model-free 确定性评测」**这个基准设计范式值得推广。当前 agent 评测大量依赖 LLM 判分，而 [15] OSReward 已经证明 LLM 裁判有系统性的宽容偏差。**把任务设计成输出可确定性验证的结构化结果**，是绕开这个问题的根本办法——虽然会限制任务类型。

**「症状不是诊断」**这个方法论提醒对所有做失败分析的人有用：同一个错误表现可以由完全不同的环节造成，只统计评测器输出的错误类型会误导。

follow-up：把「输出 schema 对齐」单独做成一个可干预的模块，很可能带来立竿见影的提升（52.2% 的失败里有一大半可能是可修复的）。代码和项目页已开放。

---

## 53. QQWorld: Quantile-Quantile Matching for World Model Regularization
**👍 30** · 🏛 西安交通大学 · [arXiv:2607.28415](https://arxiv.org/abs/2607.28415)

### 问题与动机
隐空间世界模型通过在紧凑表示空间里预测未来状态来实现高效规划，但**它们的表现关键取决于学到的隐分布的质量**。近期一条工作线用**完整的正态性检验作为可微惩罚**来把隐变量的边缘分布正则化到各向同性高斯——LeWorldModel（LeWM）用的是 **Epps–Pulley（EP）检验**，一个基于特征函数的经典正态性检验。

但论文观察到：**即便有这个正则，LeWM 学到的隐变量仍有明显的重尾。**这在世界模型里是有害的——**极端隐变量值会把学到的动力学推进表示很差的区域，在多步 rollout 中放大误差**，同时加大学到的隐分布与目标高斯先验的失配。

**关键在于：这发生在模型被显式惩罚非正态性的情况下**，说明问题出在所用检验的优化几何上。

### 方法与核心创新
论文的分析分两步，做得很扎实：
1. 利用 **EP 统计量与平方 MMD 的等价性**，把 EP 惩罚重新解读为**相对 N(0,1) 的单位带宽核差异**；
2. 分析这个目标诱导的梯度，**证明它的修正力对远离主体的隐变量值迅速衰减**。一旦某个隐坐标移出核的交互尺度，EP 正则**就几乎不提供回复力**，因此无法有效抑制重尾偏差。

**解法是把 EP 惩罚换成 QQ（Quantile–Quantile）匹配目标**：把排序后的隐值与对应的高斯分位数对齐。

L_QQ = Σ (x_n − Φ⁻¹((ρ(n) − 0.5)/N))²，其中 ρ(n) 是 x_n 在批内的秩。

**Proposition 2（非消失回复力）**：梯度是 ∂L_QQ/∂x_n = 2(x_n − q_ρ(n))——**负梯度更新直接指向秩匹配的高斯分位数，幅度是 2|x_n − q_ρ(n)|**。与 EP 梯度对极值消失相反，**QQ 梯度随分位数差异增大而增强**。

论文还处理了一个技术疑虑：**秩交换时梯度会不连续**。作者证明**平局构型在 QQ 目标下是局部排斥的**——两个相邻秩样本在平局处的单侧方向导数是 −2(q_{k+1} − q_k) < 0，所以梯度下降会**扩大它们的间隔**。这个局部反坍缩行为解释了为什么排序带来的梯度不连续在实践中不构成困难。

**Proposition 3（QQ 对 EP 的单向控制）**：L_EP ≤ C(L_QQ + log N / N)，因此 **L_QQ → 0 蕴含 L_EP → 0，但反之不成立**——EP 损失可以很小而尾部偏差任意大，因为它们的高斯核贡献随距离饱和。

还有 **Cross-Batch QQ**：用一个 FIFO 队列保留前 K 次迭代的**detach 后的**投影特征来扩大排序池（从 N 扩到 M = (K+1)N），**只用于排秩，梯度仍只通过当前 N 个特征传播**。这样在不增加反向传播 batch 的前提下提高分位数目标的精度，论文还刻画了它的偏差-方差权衡。

### 关键实验结果
**四个控制环境**上，QQWorld **有效提升了 LeWM 的平均规划成功率**，并**一致地产生更好的高斯对齐和更薄的隐尾**。

需要指出的是：**抓取的正文里没有给出具体的成功率数字**（PDF 提取只到方法部分，实验表格未在抓取范围内）。摘要用的措辞是「effectively improves」和「consistently yielding」，没有量化。

### 局限性与开放问题
论文**没有独立 Limitations 章节**（抓取范围内）。我的观察：

其一，**这是本周唯一一篇没能抓到量化实验结果的论文**。摘要只说「有效提升平均规划成功率」，没有给出数字。**理论分析（三个命题）扎实且优雅，但实验证据的强度无法判断**——「四个控制环境」是标准的 DMC 类基准还是自建的，提升是 2 个点还是 20 个点，都不清楚。对一篇以「换一个正则项」为核心贡献的论文，实验幅度是决定性的。

其二，**排序引入的计算开销**没有讨论。每个 batch 都要对投影特征排序，Cross-Batch QQ 还要维护队列并在 M = (K+1)N 个样本上排秩。虽然排序是 O(M log M) 不算贵，但在大 batch 下不是零成本。

其三，**Cross-Batch QQ 的偏差-方差权衡被「刻画」了，但 K 该取多少没有实践指导**。历史特征是 detach 的，意味着它们代表的是旧模型的分布——K 太大会引入分布漂移的偏差。

其四，**方法只解决了「边缘分布的正态性」**。而世界模型隐空间的质量还涉及时序一致性、动作可控性、信息保留等维度，正态性只是其中一个代理指标。**「更薄的尾 → 更好的规划」这条因果链，论文用理论论证了前半段（EP 梯度消失导致重尾）和实验验证了后半段（QQ 提升成功率），但中间「重尾确实是规划失败的原因」这一环是假设而非证明。**

### 启发与应用前景
这篇的**理论分析质量在本周是突出的**——从 EP↔MMD 等价性出发，识别出「核带宽外梯度消失」这个具体的优化几何缺陷，再给出一个梯度随偏差线性增长的替代目标，并证明单向控制关系。这个分析范式（**把正则项翻译成它诱导的梯度场，然后看梯度在哪里失效**）适用于任何用「统计检验作可微惩罚」的场景。

**「基于核的分布匹配对尾部无能为力」**这个洞见有普适性。MMD、核两样本检验、以及各种基于 RBF 核的正则项，都有同样的问题：**核在远处饱和，所以离群点不受约束**。而 QQ 匹配（本质上是 2-Wasserstein 距离的求积近似）没有这个问题。任何在用 MMD 类正则的工作（域适配、表示学习、生成模型）都值得检查一下自己的尾部行为。

**「平局构型是局部排斥的」**这个证明很漂亮，也很实用——它打消了「排序不可微所以不能用」这个常见顾虑。同类的基于排序的损失（排序学习、分位数回归）都可以用这个论证。

**Cross-Batch QQ 用 detach 的历史样本扩大排序池**是个便宜的技巧：**排秩不需要梯度，所以可以用比反向传播 batch 大得多的样本集**。这个「只用于统计估计的部分可以 detach 并跨批累积」的思路，适用于任何需要批内统计量的损失（BatchNorm 的统计、对比学习的负样本池、分布匹配）。

follow-up：最紧要的是**补上量化实验结果**并验证「重尾 → 规划失败」这条因果链（比如人为注入重尾看规划成功率如何变化）。

---

## 🗺️ 趋势洞察

### 1. On-policy 蒸馏的「信号提纯」成为独立研究方向

本周最拥挤的赛道。**七篇论文**在处理同一个问题：**teacher 给出的稠密监督是源混杂的（source-mixed），该怎么用？**

它们的答案惊人地一致——**做差，而非直接用**：
- [5] DAPD 发现病因是**信息不对称**（teacher 见过参考解、学生没有），解法是让两者处在匹配的信息条件下；
- [23] W2S-OPD 用**两个弱模型的 logit 差**分离出「能力方向」，再锚回学生 base——**两个都比学生弱的模型能把学生提升 6 点**；
- [28] VAD 用**证据在场/移除两次前向的差**分离视觉可归因的修正；
- [50] OPD² 用**后训练 teacher 与其 base 的概率差**做多语言蒸馏信号；
- [10] AgentOPSD 用**贝叶斯信念的边际修正**而非局部 gap 定义 turn 级信用；
- [34] PCSD 用**支持信号的局部持续性**而非单点强度定权重；
- [43] SAF-OPD 处理**稠密 OPD advantage 会淹没稀疏 RLVR advantage** 的量级失配。

**涉及论文**：[5], [10], [23], [28], [34], [43], [50]

**核心观点**：绝对分布携带 teacher 的全部特性（包括它的缺陷和它的特权），而**差值只携带增量**。[23] 的 OOD 实验给了最干净的证据——直接蒸馏小专家会把它的缺陷一起传染（IFBench 掉到基座以下），而只蒸馏「差」则不会。这个原理已经跨越了具体载体（自蒸馏、多模态、多语言、agent RL），有望成为后训练的通用工具。

同时也要看到风险：**这七篇的增益普遍在 1–6 点之间，多数没报多 seed 方差**，且环境和基线高度重合（[10] 和 [34] 都用 ALFWorld/WebShop 和 SDAR/RLSD 基线却互不引用）。这个方向正在快速拥挤而缺少统一对照。

---

### 2. 长程 Agent 的瓶颈从「模型」转移到「harness 与环境」

本周有 **11 篇**在做长程 agent，而它们指向一个共同结论：**当前的能力天花板不在模型权重，而在包裹模型的那层工程**。

最直接的证据来自 [2] LongHorizon-Harness——**同模型同执行后端**，只是把任务状态外置、执行上下文一次性、加一个只读审计器，WeaveBench 就从 51.8% 涨到 80.7%，Qwen 3.7-Plus 配好 harness 打过了 Claude Opus 4.7 配裸 Claude Code。[44] HarnessOpt-Bench 提供了另一个参照系：**把种子 agent 换成现成的成熟 harness（opencode / mini-swe-agent），OfficeQA 从 0.341 直接到 0.73**——这个跨度大于绝大多数算法改进。

环境侧同样：[1] RST 的核心贡献是**可执行任务的递归合成**（15 轮 37,484 条，单条 $0.05），[15] OSReward 的核心投入是**环境准备**（给每台机器装满真实应用、丰富初始化），[45] 明确提出「**要实现更高层级的世界，需要的是环境工程**」。

**涉及论文**：[1], [2], [8], [15], [20], [30], [41], [44], [45], [46], [49]

**核心观点**：模型能力被 harness 组织和转化的效率所限制。这一转移带来两个后果——其一，**agent 框架的评测必须跨模型做**（[8] 里同一个 Hermes 让 Qwen 涨 187.8%、让 Kimi 跌 4.1%；[44] 里同一个 harness 对不同模型效果相反）；其二，**「AI 自己改 harness」成为一项需要被度量的能力**，这正是 [44] 的定位。

---

### 3. 记忆与经验累积的「负收益」被反复独立证实

本周至少**四篇互不相关的工作**得出同一个反直觉结论：**不加验证、只增不删的经验累积会主动损害性能**。

- [46] PAST-Bench：**Agent-Zero 框架打开经验保留后总体 Δ 是 −0.08，记忆能力掉 0.27**；四个框架里三个在「流程复用」上是负的；
- [20] Skill-α：**去掉 Merge/Prune（只保留 Create/Update）后，tau2 分数 39.17，低于完全不训练的 44.17**；
- [31] Personalization Mirage：8 轮对话后 GPT-5.5 往记忆里堆了 **125 条推断属性（每轮 +15.2）且几乎不修正**，而 12 个模型中 35–49% 的 claim 是过度推断；
- [26] KGD：**在共享参数上做每日刷新反而让 AUC 从 0.7852 掉到 0.7837**，去掉解耦接口后更是低于从零训练。

**涉及论文**：[20], [26], [31], [46]

**核心观点**：**「持久化」不是免费的加法**。写入机制远比读取机制受关注，而删除、合并、修正、以及「哪块经验该归哪个存储基底」几乎无人处理。[46] 提出的「结果增益 + 机制证据」双报告，和 [31] 提出的 provenance tracking，是这个方向目前最具体的两个抓手。

---

### 4. 世界模型从「预测像素」转向「服务 agent」，但保真度问题浮出水面

本周有 **8 篇**世界模型相关工作，[38] Quo Vadis 给出了统一的框架表述：**从预测世界到服务 agent**，衡量标准从视觉真实度换成「可行动的信息增益」。

这个转向在具体工作里已经发生：[30] EnvACE 把模拟器**折叠进策略本身**（世界排练），[13][27] 把触觉从「观测」变成「预测目标」，[7] Mental World Modeling 把心理状态提升为世界状态的一等公民。

但 [42] WorldExam 的评测把代价摆了出来：**20 个模型里没有一个同时具备广泛覆盖和一致强表现**；更关键的是「**强视觉质量或强控制遵从并不保证世界反应性**」——语言驱动模型的 General 均值落在 79.64–81.04 的窄区间，而 Task 均值从 39.85 到 65.02 大幅分散。[53] QQWorld 从优化几何层面找到一个具体缺陷：基于核的分布正则（EP/MMD）**对尾部梯度消失**，导致隐空间重尾、多步 rollout 误差放大。

**涉及论文**：[7], [13], [27], [30], [38], [40], [42], [53]

**核心观点**：世界模型作为 agent 的代理，其价值取决于**可信度**而非逼真度。[38] 列出的四个开放挑战里，「**agent 如何在线判断该不该信任代理**」是最被低估也最实际的——当前所有用模拟器/世界模型做训练的工作几乎都默认代理可信，而 [30] 的策略自扮演环境正是把这个假设推到了极致（论文也没有量化排练响应的保真度）。

---

### 对比与张力

- **「让评测更难」vs「让评测更可信」**：本周基准工作分两派。一派在提高难度——[35] CADENA-Bench 显示**所有方法从 sketch-extrude 语料换到真实机械零件都损失一半 GMS**（cadrille 94.8→49.8），[29] GST-Bench 上最强模型 42.68 vs 人类 79.08，[24] 指出 GPT-5 零工具调用就能拿竞争力分数说明现有视频基准大量泄漏。另一派在提高可信度——[52] DataSpace 用 model-free 确定性评测绕开 LLM 判分，[15] OSReward 直接量化了 LLM 裁判的宽容偏差（**困难集上 Doubao-2.0-Lite 的失败召回只有 24.3%**）。**两派都对，但方向相反**：更难的任务往往更难确定性验证，而可确定性验证的任务往往形式受限（[52] 有 52.2% 的失败是输出格式问题，部分正源于它的严格性）。

- **扩散语言模型的三条路线**：[39] DiffusionGemma 走 **AR 微调转换**（<10% token 预算，1,479 tokens/s，但 AIME 掉 15 点）；[48] LLaDA MoE v2 走**从零训练 MoE**（23.5T token，65% 于 Qwen3 的预算逼近其能力，但**没报任何速度数字**）；[12] AURORA-LM 走**连续隐空间**（1B 规模九基准均分 32.6，但 MMLU 22.2 低于随机）。三条路线的取舍完全不同，而**只有 DiffusionGemma 同时报了能力和速度**——这恰恰是判断这条路线价值的唯一方式。

- **「信息量」vs「token 数」**：[36] 的发现值得单列——**把提示词拉长会让所有开源文生图模型变差**，起作用的是结构化信息量（GPG/ED）而非 token 数，且在信息量匹配时结构化和散文落在同一条拟合线上。这与 LLM 领域「长上下文更好」的直觉正好相反，也直接否定了当前大量 prompt enhancer 的做法。

---

### 值得关注的研究方向

1. **在 logit/概率空间做因果分解**：本周七篇独立收敛到「做差」这个操作（[5][10][23][28][34][43][50]），但都停留在启发式层面。[28] VAD 自己承认「当前的投影只带来语义富集，而非可辨识的分离」——**带 grounding 约束的学习式分解**如果做成，就能把这一系列工作从经验技巧升级为方法论。

2. **经验持久化的「删除与仲裁」机制**：写入已经有很多方案，而 [46] 的负增益案例、[20] 的 Merge/Prune 消融、[31] 的线性累积观察共同指向同一个空白。[46] 提出的**反事实干预做机制归因**（删掉那条记忆看行为变不变）是把相关证据升级为因果证据的必经之路。

3. **代理可信度的在线判断**：[38] 明确把它列为开放挑战，而 [30] EnvACE 的世界排练、[15] OSReward 揭示的裁判宽容偏差、[53] QQWorld 的重尾误差放大，都是这个问题的不同侧面。**让 agent 学会「这次代理反馈靠不靠谱」**，能同时解决保真度传播和奖励黑客两个问题。

4. **合成数据的难度自校准**：[1] RST 用固定 solver 的 pass@4 曲线标定难度（90% → 2.5%），[49] SKT 用中等模型采样 N 次的成功率做阈值过滤，[35] CADENA 用程序化生成器的构造历史直接产出逐步监督。**三种独立发明的机制指向同一个配方：用一个可控的求解器把「任务难度」变成可测量、可调节的量**。这比人工分级或 LLM 打分都可靠，值得系统化。

5. **分布外语言/领域上的「不要丢掉旧基线」**：[51] 的希腊语工作证明**BM25 在支持语言列表之外打败所有现成多语言 embedder（包括 8B）**，而 RRF 等权重不调参融合就有稳定增益。这个教训大概率适用于所有「主流评测覆盖不到」的场景——**当分布外时，把打败你的基线当组件而非对手**。
