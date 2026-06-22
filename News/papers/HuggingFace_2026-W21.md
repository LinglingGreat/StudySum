# HuggingFace 周榜论文深度总结 — 2026 第 21 周

> 来源：https://huggingface.co/papers/week/2026-W21
> 统计日期：2026-06-18
> 筛选条件：upvotes ≥ 50
> 论文数：36

## 目录

1. [HRM-Text: Efficient Pretraining Beyond Scaling](#1-hrm-text-efficient-pretraining-beyond-scaling) 👍315
2. [CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence](#2-citevqa-benchmarking-evidence-attribution-for-trustworthy-document-intelligence) 👍271
3. [Code as Agent Harness](#3-code-as-agent-harness) 👍219
4. [DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards](#4-delta-discriminative-token-credit-assignment-for-reinforcement-learning-from-verifiable-rewards) 👍204
5. [Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information](#5-anti-self-distillation-for-reasoning-rl-via-pointwise-mutual-information) 👍195
6. [AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration](#6-autoresearchclaw-self-reinforcing-autonomous-research-with-human-ai-collaboration) 👍189
7. [TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation](#7-transitlm-a-large-scale-dataset-and-benchmark-for-map-free-transit-route-generation) 👍177
8. [Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?](#8-perception-or-prejudice-can-mllms-go-beyond-first-impressions-of-personality) 👍169
9. [When Vision Speaks for Sound](#9-when-vision-speaks-for-sound) 👍159
10. [Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](#10-video2gui-synthesizing-large-scale-interaction-trajectories-for-generalized-gui-agent-pretraining) 👍145
11. [PhysBrain 1.0 Technical Report](#11-physbrain-10-technical-report) 👍143
12. [Mega-ASR: Towards In-the-wild² Speech Recognition via Scaling up Real-world Acoustic Simulation](#12-mega-asr-towards-in-the-wild-speech-recognition-via-scaling-up-real-world-acoustic-simulation) 👍134
13. [SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution](#13-skillsvote-lifecycle-governance-of-agent-skills-from-collection-recommendation-to-evolution) 👍127
14. [MMSkills: Towards Multimodal Skills for General Visual Agents](#14-mmskills-towards-multimodal-skills-for-general-visual-agents) 👍118
15. [LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation](#15-longlive-20-an-nvfp4-parallel-infrastructure-for-long-video-generation) 👍113
16. [π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows](#16-π-bench-evaluating-proactive-personal-assistant-agents-in-long-horizon-workflows) 👍105
17. [Active Learners as Efficient PRP Rerankers](#17-active-learners-as-efficient-prp-rerankers) 👍98
18. [Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps](#18-full-attention-strikes-back-transferring-full-attention-into-sparse-within-hundred-training-steps) 👍95
19. [Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](#19-enhancing-train-free-infinite-frame-generation-for-consistent-long-videos) 👍92
20. [IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools](#20-indusagent-reinforcing-open-vocabulary-industrial-anomaly-detection-with-agentic-tools) 👍83
21. [OpenComputer: Verifiable Software Worlds for Computer-Use Agents](#21-opencomputer-verifiable-software-worlds-for-computer-use-agents) 👍82
22. [Lance: Unified Multimodal Modeling by Multi-Task Synergy](#22-lance-unified-multimodal-modeling-by-multi-task-synergy) 👍78
23. [AI for Auto-Research: Roadmap & User Guide](#23-ai-for-auto-research-roadmap--user-guide) 👍67
24. [FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization](#24-fashionchameleon-towards-real-time-and-interactive-human-garment-video-customization) 👍65
25. [OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization](#25-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-quantization) 👍64
26. [ACC: Compiling Agent Trajectories for Long-Context Training](#26-acc-compiling-agent-trajectories-for-long-context-training) 👍60
27. [GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment](#27-golongrl-capability-oriented-long-context-reinforcement-learning-with-multitask-alignment) 👍59
28. [Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation](#28-learning-to-foresee-unveiling-the-unlocking-efficiency-of-on-policy-distillation) 👍59
29. [A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook](#29-a-survey-of-large-audio-language-models-generalization-trustworthiness-and-outlook) 👍56
30. [DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo](#30-dexjoco-a-benchmark-and-toolkit-for-task-oriented-dexterous-manipulation-on-mujoco) 👍54
31. [Auditing Agent Harness Safety](#31-auditing-agent-harness-safety) 👍54
32. [PhysX-Omni: Unified Simulation-Ready Physical 3D Generation](#32-physx-omni-unified-simulation-ready-physical-3d-generation-for-rigid-deformable-and-articulated-objects) 👍53
33. [Process Rewards with Learned Reliability](#33-process-rewards-with-learned-reliability) 👍53
34. [CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?](#34-chi-bench-can-ai-agents-automate-end-to-end-long-horizon-policy-rich-healthcare-workflows) 👍53
35. [You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories](#35-you-only-need-minimal-rlvr-training-extrapolating-llms-via-rank-1-trajectories) 👍50
36. [EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL](#36-envfactory-scaling-tool-use-agents-via-executable-environments-synthesis-and-robust-rl) 👍50

---

## 1. HRM-Text: Efficient Pretraining Beyond Scaling
**👍 315** · https://huggingface.co/papers/2605.20613 · GitHub: https://github.com/sapientinc/HRM-Text

### 问题与动机
当前 LLM 预训练范式严重依赖海量算力和互联网级原始文本，把基础研究的门槛抬到普通团队望尘莫及的高度。作者反其道而行，从生物学习的高样本效率（如额顶环路的多时间尺度处理）找灵感：能不能用极小的算力，从零训出一个有竞争力的模型？这是对"scaling 就是一切"主流叙事的直接挑战。

### 方法与核心创新
核心是用**分层循环模型 HRM** 替代标准 Transformer，把计算解耦为"慢演化的战略层"和"快演化的执行层"，模拟大脑多时间尺度。为稳定这种深度循环，引入 MagicNorm 和 warmup 式深度信用分配。更关键的是抛弃原始文本预训练，**只用 instruction-response 配对数据**配合 task-completion 目标和 PrefixLM 掩码训练——本质上把"预训练"和"对齐"合并了。

### 关键实验结果
一个 1B 参数模型，从零训练，只用 **400 亿** unique token、**1500 美元**预算，达到 MMLU 60.7%、ARC-C 81.9%、DROP 82.2%、GSM8K 84.5%、MATH 56.2%。相比标准 baseline 少用 **100–900 倍** token、**96–432 倍**算力，性能却与 2–7B 开源模型相当。这是惊人的算力-性能比压缩。

### 局限性与开放问题
1B 规模下成立，能否 scale 到 10B/100B 仍是开放问题（摘要未明确，推断）。只用指令数据训练可能牺牲了原始文本带来的世界知识广度，在长尾知识上的表现存疑。HRM 的循环结构对推理延迟的影响也未充分讨论。

### 启发与应用前景
若结论稳健，将极大降低基础模型研究门槛，让学术界和小团队重新进入预训练赛道。"架构-目标协同设计"的思路可能比单纯堆数据更有性价比，对算力受限场景（边缘、垂直行业）意义重大。

---

## 2. CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence
**👍 271** · https://huggingface.co/papers/2605.12882 · GitHub: https://github.com/opendatalab/CiteVQA

### 问题与动机
多模态大模型在文档理解上进步很快，但现有 Doc-VQA 评测**只看最终答案对不对，不查证据**。这掩盖了一个致命失败模式：模型答对了，但依据的是错误段落——在法律、金融、医疗这类高风险领域，每个结论都必须可追溯到具体来源，这种"蒙对"是不可接受的。

### 方法与核心创新
提出 CiteVQA，要求模型在给出答案的同时返回**元素级 bounding-box 引用**，二者联合评估。数据集含 **1,897 个问题、711 份 PDF**，跨 7 个领域、2 种语言，文档平均 **40.6 页**（长文档）。真值引用由"掩码消融自动管线"生成（遮住关键证据看答案是否崩）再经专家复核。核心指标 **Strict Attributed Accuracy (SAA)** 只在答案和引用都对时才给分。

### 关键实验结果
审计 20 个 MLLM 揭示普遍的"归因幻觉"：模型经常答对却引错区域。最强系统 Gemini-3.1-Pro-Preview 的 SAA **仅 76.0**，最强开源 MLLM 更是只有 **22.5**——开源与闭源差距悬殊，且整体远未及格。

### 局限性与开放问题
SAA 对引用边界的严格度可能略偏苛刻（部分正确的引用得 0 分）。自动管线生成的真值虽经复核，但"掩码消融=关键证据"的假设在多证据题上可能不完备（摘要未明确，推断）。

### 启发与应用前景
为"可信文档智能"提供了缺失的测量工具。直接推动 RAG、合规审查、智能投研等场景的可追溯性需求，也提示厂商：答案准确率不等于可信，归因能力是下一个竞争维度。

---

## 3. Code as Agent Harness
**👍 219** · https://huggingface.co/papers/2605.18747 · GitHub: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers

### 问题与动机
LLM 写代码的能力很强，但作者观察到一个范式转变：在 agentic 系统里，**代码不再只是输出目标，而是 agent 推理、行动、环境建模和执行验证的"操作底座"**。这篇综述要回答：如何系统地把"代码作为 agent harness（执行框架）"这个视角组织起来？

### 方法与核心创新
提出统一视角并分三层组织综述：(1) **harness 接口层**——代码如何连接 agent 与推理、动作、环境建模；(2) **harness 机制层**——长程执行的规划、记忆、工具使用，加上反馈驱动的控制与优化；(3) **harness 扩展层**——从单 agent 到多 agent，共享代码 artifact 支撑协调、审查与验证。这是一个把零散实践框架化的"地图"贡献。

### 关键实验结果
综述类论文，无定量实验。覆盖编程助手、GUI/OS 自动化、具身 agent、科学发现、个性化推荐、DevOps、企业工作流等大量应用场景（摘要未给出具体数值）。

### 局限性与开放问题
作者自己列出了开放挑战：超越"最终任务成功"的评估、不完整反馈下的验证、无回归的 harness 改进、多 agent 间一致的共享状态、安全关键动作的人类监督、向多模态环境的扩展。

### 启发与应用前景
为正在做 agent 基础设施的团队提供了清晰的概念框架和文献地图。"代码即 harness"的提法有助于统一当前碎片化的 agent 工程实践，指向"可执行、可验证、有状态"的 agent 系统路线图。

---

## 4. DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards
**👍 204** · https://huggingface.co/papers/2605.21467 · GitHub: https://github.com/RUCBM/DelTA

### 问题与动机
RLVR（可验证奖励强化学习）是提升 LLM 推理的核心技术，但**响应级奖励如何转化为 token 级概率变化，机理一直不清楚**。这导致优化信号被"格式 token"等高频共享模式稀释，真正区分好坏答案的稀疏方向被淹没。

### 方法与核心创新
作者提出 RLVR 更新的"判别器视角"：策略梯度更新方向本质上是 token 梯度向量上的**线性判别器**，由正负侧 centroid（优势加权平均）构成。问题在于 centroid 被高频格式 token 主导。DelTA 估计 token 系数，**放大侧特异性的梯度方向、降权共享或弱判别方向**，重塑 RLVR 更新方向，使有效 centroid 更具对比性。

### 关键实验结果
在 7 个数学基准上，DelTA 相比同规模最强 baseline 在 Qwen3-8B-Base 上平均高 **3.26 分**、Qwen3-14B-Base 上高 **2.62 分**。在代码生成、不同 backbone 和域外评测上同样有效，验证了泛化性。

### 局限性与开放问题
提升幅度（2–3 分）属中等，对超大规模模型是否仍有效未充分验证（摘要未明确，推断）。token 系数估计本身引入的额外计算开销未量化。

### 启发与应用前景
为 RLVR 提供了可解释的 token 级信用分配机制，"判别器视角"是个有洞察力的分析工具。可直接用于改进数学/代码推理的后训练，对所有依赖 GRPO 类算法的团队都有借鉴价值。

---

## 5. Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information
**👍 195** · https://huggingface.co/papers/2605.11609 · GitHub: https://github.com/FloyedShen/AntiSD

### 问题与动机
On-policy 自蒸馏（学生被拉向"看过答案的自己"）本是无需更强外部老师的自我提升路径，但在数学推理上**收益时灵时不灵**。作者用点互信息分析定位了原因。

### 方法与核心创新
PMI 分析发现：特权上下文（已验证的解）会**抬高老师在"已被答案蕴含的 token"（结构连接词、可验证断言）上的置信度，却压低在"思考 token"（"Wait""Let""Maybe"）上的置信度**——而后者正是驱动多步搜索的关键。于是提出 **AntiSD**：不是最小化学生-老师的散度，而是**上升（反向）这个散度**，反转每个 token 的符号，得到天然有界的优势。再加一个熵触发门控，老师熵崩塌时关闭该项，成为默认自蒸馏的即插即用替代。

### 关键实验结果
在 4B 到 30B 的 5 个模型上，AntiSD 用 **2–10 倍更少的训练步数**就达到 GRPO baseline 的准确率，最终准确率最多提升 **11.5 个百分点**。

### 局限性与开放问题
方法专门针对数学推理的"思考 token"现象，在非推理任务上是否同样有效未知（摘要未明确，推断）。熵门控阈值的鲁棒性依赖调参。

### 启发与应用前景
"反向蒸馏"是个反直觉但深刻的发现——有时该远离而非靠近教师信号。为可扩展的模型自举提供了新路径，启发我们重新审视蒸馏中"哪些 token 该被强化"。

---

## 6. AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration
**👍 189** · https://huggingface.co/papers/2605.20025 · GitHub: https://github.com/aiming-lab/AutoResearchClaw

### 问题与动机
自动化科研不只是"从想法生成论文"。真实研究是迭代的：假设被多角度挑战、实验失败后指导下一次尝试、教训跨周期累积。现有自主科研系统把这个过程建模成**线性管线**——单 agent 推理、执行失败就停、不跨 run 携带经验。

### 方法与核心创新
AutoResearchClaw 是多 agent 自主科研管线，建立在五个机制上：(1) **结构化多 agent 辩论**做假设生成与结果分析；(2) **自愈执行器**，带 Pivot/Refine 决策环，把失败转化为信息；(3) **可验证结果报告**，防止编造数字和幻觉引用；(4) **人在环协作**，含从全自主到逐步监督的 7 种干预模式；(5) **跨 run 进化**，把过去错误转为未来防护。

### 关键实验结果
在 ARC-Bench（25 主题实验阶段基准）上，AutoResearchClaw 比 AI Scientist v2 **高出 54.7%**。人在环消融揭示一个重要发现：**在高杠杆决策点做精准定向协作，持续优于全自主和穷尽式逐步监督**——即"该出手时才出手"最优。

### 局限性与开放问题
54.7% 的提升基于特定基准，能否泛化到真实顶会标准仍存疑（参见 #23 的悲观结论）。多 agent 辩论的算力成本未量化。

### 启发与应用前景
"研究放大器而非替代者"的定位务实。7 种干预模式的设计和"高杠杆点协作最优"的结论，对设计任何人机协作系统都有普适价值。

---

## 7. TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation
**👍 177** · https://huggingface.co/papers/2605.22355 · GitHub: https://github.com/HotTricker/TransitLM

### 问题与动机
公共交通路线规划传统上依赖结构化地图基础设施和复杂路由引擎。作者问：**能否让模型完全绕过地图依赖，直接从数据学会路线规划？** 此前没有任何数据集支持训练这种模型。

### 方法与核心创新
发布 TransitLM——超过 **1300 万**条交通路线规划记录，覆盖 4 个中国城市、**120,845 个站点、13,666 条线路**，作为持续预训练语料和 3 个评测任务的基准。核心思路是把路线规划当成纯序列建模问题，让 LLM 从 OD（起点-终点）信息端到端生成路线。

### 关键实验结果
在 TransitLM 上训练的 LLM 能高准确率产出**结构有效的路线**，且能在无任何显式映射的情况下**隐式地把任意 GPS 坐标 grounding 到合适的站点**（摘要未给出具体准确率数值）。证明了交通路线规划可完全从数据学习。

### 局限性与开放问题
仅覆盖 4 个中国城市，跨城市/跨国泛化能力未知。"隐式 grounding GPS"在边缘坐标（新建站点、坐标噪声）下的鲁棒性未给数字（摘要未明确，推断）。与传统路由引擎在实时性、最优性上的硬碰硬比较缺失。

### 启发与应用前景
为"map-free"导航打开思路，尤其在地图数据缺失或快速变化的区域有价值。也是 LLM 处理结构化时空数据能力的有趣探针。

---

## 8. Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?
**👍 169** · https://huggingface.co/papers/2605.22109 · GitHub: https://github.com/kkkcx/MM-OCEAN

### 问题与动机
MLLM 越来越多用于人际场景，性格感知很关键。但现有基准**只评 Big Five 数值预测**，无法区分模型是真的通过行为理解感知性格，还是仅凭表面模式匹配"以貌取人"。

### 方法与核心创新
三大贡献：(1) 新任务 **Grounded Personality Reasoning (GPR)**，要求模型把每个 Big Five 评分锚定到可观察证据，形成"评分-推理-grounding"链；(2) 新数据集 **MM-OCEAN**（1,104 视频、5,320 道选择题），含时间戳行为观察、证据 grounded 的特质分析、7 类线索-grounding 题；(3) 三层评估（评分/推理/grounding）加 4 个失败模式指标：偏见率 PR、虚构率 CR、整合失败率 IR、整体 grounding 率 HR。基准 27 个 MLLM。

### 关键实验结果
揭示惊人的"偏见鸿沟"：全行业 **51% 的正确评分并未 grounded 在检索到的线索上**，整体 grounding 率仅覆盖 **0–33.5%** 区间。即"答对分数"和"为对的理由推理"严重脱节。

### 局限性与开放问题
Big Five 本身作为性格金标准存在争议；视频性格标注的主观性可能影响真值可靠性（摘要未明确，推断）。

### 启发与应用前景
与 #2 CiteVQA 异曲同工——都揭示"答案对≠依据对"。对招聘、心理评估、社交机器人等敏感场景敲响警钟：MLLM 的性格判断可能本质上是偏见，需要 grounding 验证才能负责任地部署。

---

## 9. When Vision Speaks for Sound
**👍 159** · https://huggingface.co/papers/2605.16403 · GitHub: https://github.com/rakanWen/wvs-code

### 问题与动机
视频 MLLM 看似能理解音频，但作者发现它们的"音频理解"往往是**视觉驱动**的：模型靠视觉线索推断甚至幻觉出声学信息，而非真的核验音频流。这个问题在顶级开源 omni 模型和 Google/OpenAI 的闭源模型上都存在。

### 方法与核心创新
把这种失败命名为**视听"聪明汉斯效应"**——模型看似 audio-grounded，实则利用视听相关性而不核验两路是否真对齐。提出 **Thud** 探测框架，用 3 种反事实音频编辑诊断：**Shift**（测时间同步）、**Mute**（测声音存在性）、**Swap**（测视听一致性）。进而给出两阶段对齐方案：干预派生的偏好对教模型核验音频，事件级通用视频偏好做正则防止过拟合。

### 关键实验结果
最佳的 **10K 样本**方案在三个干预维度上平均性能**提升 28 个百分点**，同时在通用视频和视听 QA 基准上略有提升（无副作用）。

### 局限性与开放问题
诊断和修复都基于反事实编辑，真实世界更复杂的视听错位场景覆盖度未知。28 个点的提升后绝对水平如何未明（摘要未明确，推断）。

### 启发与应用前景
"聪明汉斯效应"框架可推广到任何多模态捷径学习诊断。对自动驾驶、安防、内容审核等真正需要音频核验的场景至关重要——不能让模型"看图说声"。

---

## 10. Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining
**👍 145** · https://huggingface.co/papers/2605.14747 · GitHub: https://github.com/WeiminXiong/Video2GUI

### 问题与动机
GUI agent 的泛化受限于**大规模跨应用训练数据稀缺**。现有数据集严重依赖昂贵的人工标注，且局限于狭窄领域。

### 方法与核心创新
Video2GUI 是全自动框架，**直接从无标注互联网视频提取 grounded 的 GUI 交互轨迹**。用 coarse-to-fine 过滤策略识别高质量 GUI 教程视频，转成结构化 agent 轨迹。把管线应用到 **5 亿**视频元数据条目上，构建出 **WildGUI** 数据集——含 **1200 万**条交互轨迹，跨 **1,500+** 应用和网站。

### 关键实验结果
在 WildGUI 上预训练 Qwen2.5-VL 和 Mimo-VL，在多个 GUI grounding 和动作基准上**一致提升 5–20%**，达到或超过 SOTA。

### 局限性与开放问题
教程视频可能偏向"标准操作流程"，对异常/错误恢复场景的覆盖不足（摘要未明确，推断）。从视频提取的动作精度受限于视频质量。

### 启发与应用前景
"用互联网视频自动造 agent 训练数据"是个可规模化的好思路，与 #11 PhysBrain（从人类视频学物理）思路相通。直接降低 GUI/OS 自动化 agent 的数据门槛。

---

## 11. PhysBrain 1.0 Technical Report
**👍 143** · https://huggingface.co/papers/2605.15298 · GitHub: https://github.com/Phys-Brain/PhysBrain-VLA

### 问题与动机
视觉-语言-动作（VLA）模型进步快，但**机器人轨迹数据对学习广泛物理理解覆盖有限**。作者探索一条互补路线：在机器人适配前，先从大规模人类第一人称视频学结构化物理常识。

### 方法与核心创新
数据引擎从人类 egocentric 视频提取场景元素、空间动态、动作执行、深度感知关系，转成 QA 监督来训 PhysBrain VLM。再通过"能力保留+语言敏感"的适配设计，把物理先验迁移到 VLA 策略。本质是用人类视频补足机器人数据的物理常识缺口。

### 关键实验结果
在多模态 QA 基准（ERQA、PhysBench）和具身控制基准（SimplerEnv-WidowX、LIBERO、RoboCasa）上均达 **SOTA**，在 SimplerEnv 上**域外泛化尤其强**（摘要未给出具体分数）。

### 局限性与开放问题
具体提升幅度摘要未明确。从被动观看视频学到的物理常识与主动交互获得的常识之间是否有本质差距，是个开放问题（推断）。

### 启发与应用前景
与 #10 Video2GUI 共享"人类视频→agent 能力"范式。为机器人学习提供了缓解数据稀缺的有效桥梁，指向"先看后做"的具身智能训练路线。

---

## 12. Mega-ASR: Towards In-the-wild² Speech Recognition via Scaling up Real-world Acoustic Simulation
**👍 134** · https://huggingface.co/papers/2605.19833 · GitHub: https://github.com/xzf-thu/Mega-ASR

### 问题与动机
尽管 ASR 和大音频语言模型进步快，真实环境鲁棒识别仍受"声学鲁棒性瓶颈"限制：模型在严重、复合的失真下会失去声学 grounding，产生漏字或幻觉。

### 方法与核心创新
Mega-ASR 统一框架，结合可扩展复合数据构建与渐进式声学-语义优化。推出 **Voices-in-the-Wild-2M** 数据集，覆盖 **7 种经典声学现象**和 **54 种物理合理的复合场景**。训练采用"声学-语义渐进式 SFT"加"双粒度 WER 门控策略优化"。

### 关键实验结果
在恶劣条件 ASR 基准上显著领先：VOiCES R4-B-F 上 **45.69% vs. 54.01%**（WER 越低越好），NOIZEUS Sta-0 上 **21.49% vs. 29.34%**。在复杂复合声学场景上，相比强开源/闭源 baseline 进一步实现 **30%+ 相对 WER 降低**。

### 局限性与开放问题
54 种合成场景能否覆盖真实世界的长尾噪声仍是问题；合成数据与真实失真的分布差异可能在极端场景失效（摘要未明确，推断）。

### 启发与应用前景
"in-the-wild²"（野外的平方）强调复合失真，对车载、会议、户外等真实 ASR 部署直接有用。声学仿真规模化的思路可复用到其他音频任务。

---

## 13. SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution
**👍 127** · https://huggingface.co/papers/2605.18401 · GitHub: https://github.com/MemTensor/skills-vote

### 问题与动机
长程 LLM agent 留下的轨迹本可成为可复用经验，但原始轨迹**嘈杂、难治理**。开放技能生态充斥冗余、参差、环境敏感的 artifact，盲目更新会污染未来上下文。

### 方法与核心创新
把 Agent Skills 当作"经验 schema"——耦合可执行脚本与非可执行的流程指导。SkillsVote 是覆盖"收集-推荐-进化"的全生命周期治理框架：先 profile **百万级**开源语料的环境需求/质量/可验证性，为可验证技能合成任务。执行前做 agentic library search 暴露技能上下文；执行后把轨迹分解为技能关联子任务，把结果归因到技能使用/探索/环境/结果信号，**只接纳成功可复用的发现做"证据门控更新"**。

### 关键实验结果
离线进化在 Terminal-Bench 2.0 上提升 GPT-5.2 最多 **7.9 pp**；在线进化在 SWE-Bench Pro 上提升最多 **2.6 pp**。证明受治理的外部技能库可在不更新模型的情况下改进冻结 agent。

### 局限性与开放问题
提升幅度依赖任务类型；"证据门控"的严格度与覆盖率之间的权衡未充分探讨（推断）。百万级语料的 profiling 成本未量化。

### 启发与应用前景
与 #14 MMSkills 共同推动"技能复用"成为 agent 提升的核心范式。对运营 Claude/agent 技能生态的团队（如本仓库这类 skill 体系）极有参考价值：如何防止技能库污染、如何做证据门控更新。

---

## 14. MMSkills: Towards Multimodal Skills for General Visual Agents
**👍 118** · https://huggingface.co/papers/2605.13527 · GitHub: https://github.com/zkangning/MMSkills_for_Visual_Agents

### 问题与动机
可复用技能已成提升 agent 的核心，但现有技能包主要把行为编码为**文本提示、代码或学到的例程**。对视觉 agent 而言，程序知识本质是**多模态**的：复用不仅取决于"做什么操作"，还取决于识别相关状态、解读进度/失败的视觉证据、决定下一步。

### 方法与核心创新
形式化"多模态程序知识"，解决三个问题：多模态技能包应含什么、从哪派生、如何在推理时用而不爆图像上下文。**MMSkill** 是紧凑的状态条件包，耦合文本流程+运行时状态卡+多视角关键帧。用 agentic "轨迹→技能" Generator（工作流分组、流程归纳、视觉 grounding、元技能审计）从公开非评测轨迹生成技能。用时通过 **branch-loaded** 多模态技能 agent：在临时分支检查状态卡和关键帧，与实时环境对齐后蒸馏成结构化指导给主 agent。

### 关键实验结果
在 GUI 和游戏视觉 agent 基准上，MMSkills **一致提升前沿和较小的多模态 agent**（摘要未给出具体提升数值），表明外部多模态程序知识能补足模型内部先验。

### 局限性与开放问题
具体数值缺失，难判断提升幅度。关键帧选取的代表性和 branch-loading 的延迟开销未量化（推断）。

### 启发与应用前景
把"技能"从纯文本扩展到多模态，是视觉 agent 工程的重要一步。"临时分支检查证据再蒸馏"避免上下文爆炸的设计很实用。

---

## 15. LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
**👍 113** · https://huggingface.co/papers/2605.18739 · GitHub: https://github.com/NVlabs/LongLive

### 问题与动机
长视频生成在训练和推理上都遭遇速度与显存瓶颈。NVIDIA 团队要解决：如何在全流程上用低精度并行基础设施突破这些瓶颈。

### 方法与核心创新
LongLive-2.0 是首个面向长视频生成的 **NVFP4**（4-bit 浮点）训练+推理系统。训练侧引入序列并行自回归训练（Balanced SP），把 clean-history 和 noisy-target 时间块在每个 rank 上配对，配合 SP 感知的分块 VAE 编码。结合 NVFP4 精度降显存、加速 GEMM。与依赖 ODE 初始化+DMD 蒸馏的 Self-Forcing 系列不同，**直接把扩散模型调成长、多镜头、交互式 AR 扩散模型**，还可用独立 LoRA 权重转为实时生成（4→2 去噪步）。推理侧在 Blackwell GPU 上启用 W4A4 NVFP4 推理、KV cache 量化为 NVFP4、异步流式 VAE 解码。

### 关键实验结果
训练**最高 2.15 倍加速**，推理 **1.84 倍**。LongLive-2.0-5B 实现 **45.7 FPS** 推理，同时在基准上保持强性能。

### 局限性与开放问题
NVFP4 强依赖 Blackwell 架构，非 Blackwell GPU 需用 SP 推理才能匹配速度，硬件绑定较强。4-bit 量化在极长视频上的质量退化边界未充分探讨（推断）。

### 启发与应用前景
为长视频生成的工业化部署提供了完整的低精度基础设施模板。NVFP4 训练+推理全流程打通，对追求实时交互视频生成（如直播换装、游戏）意义重大。

---

## 16. π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows
**👍 105** · https://huggingface.co/papers/2605.14678 · GitHub: https://github.com/Simplified-Reasoning/Pi-Bench

### 问题与动机
个人助理 agent（如 OpenClaw）潜力巨大，核心挑战是**主动协助**：用户初始请求常欠规约，重要需求/约束/偏好不说出来。现有基准很少评估 agent 能否在用户明说前识别并行动于这些隐藏意图，尤其在需求逐步浮现的多轮交互中。

### 方法与核心创新
π-Bench 含 **100 个多轮任务**，跨 **5 个领域特定用户画像**。通过纳入隐藏用户意图、任务间依赖、跨会话连续性，**联合衡量主动性和任务完成度**在长程轨迹中的表现——更贴近真实使用。

### 关键实验结果
实验显示：(1) 主动协助仍很有挑战；(2) **任务完成度与主动性之间存在清晰区分**（能完成不等于会主动）；(3) 先前交互对后续任务的主动意图解析有价值（摘要未给出具体分数）。

### 局限性与开放问题
仅 100 个任务、5 个画像，规模偏小；主动性评估的主观性如何控制未明（推断）。"过度主动"（猜错意图）的负面影响未纳入评分。

### 启发与应用前景
与 #34 CHI-Bench、#21 OpenComputer 同属"长程 agent 能力评测"浪潮。对个人助理产品设计直接有用：主动性是独立于任务完成的能力维度，需专门优化。

---

## 17. Active Learners as Efficient PRP Rerankers
**👍 98** · https://huggingface.co/papers/2605.14236 · GitHub: https://github.com/jerecoder/IReranker

### 问题与动机
成对排序提示（PRP）从 LLM 获取成对偏好判断再聚合成排序，通常用经典排序算法。但 LLM 判断**嘈杂、对顺序敏感、有时不传递**，排序算法的假设根本不匹配；而排序追求完整置换，截断它以满足调用预算无法产出可靠 top-K。

### 方法与核心创新
把 PRP 重排**重新框定为"从嘈杂成对比较中主动学习"**，证明 active ranker 是即插即用替代，在调用受限场景下提升每次调用的 NDCG@10。还引入**随机方向 oracle**，每对只用一次 LLM 调用——把系统性位置偏差转成零均值噪声，从而无需双向调用即可实现无偏聚合排序。

### 关键实验结果
在调用受限（call-constrained）场景下，active ranker 提升每次调用的 NDCG@10（摘要未给出具体数值）。随机方向 oracle 把双向调用成本砍半（2 次→1 次）同时去偏。

### 局限性与开放问题
具体 NDCG 提升幅度缺失。随机方向去偏依赖"位置偏差近似对称"的假设，强偏差场景下可能失效（推断）。

### 启发与应用前景
对 RAG、搜索重排这类 LLM-as-reranker 场景直接降本——用更少 LLM 调用拿到更可靠的 top-K。"主动学习视角"重构经典问题的思路有启发性。

---

## 18. Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps
**👍 95** · https://huggingface.co/papers/2605.16928

### 问题与动机
长上下文推理受全注意力二次成本瓶颈。现有高效替代要么依赖原生稀疏训练（贵），要么靠启发式 token 驱逐（损精度），在效率、训练成本、精度间形成糟糕的三角权衡。

### 方法与核心创新
核心洞察：**全注意力 LLM 本身就内在稀疏**，只需极少适配就能转成高度稀疏模型。基于三个观察：(1) 只有一小部分注意力头真需要全长程处理；(2) 长程检索主要由低维子空间主导，用 **16 维 indexer** 即可高效检索相关 token；(3) 有用 token 预算强依赖 query，动态 top-p 比固定 top-k 更合适。据此提出 **RTPurbo**：只为检索头保留全 KV cache，引入轻量 token indexer 做稀疏注意力，**仅几百训练步**即可稀疏化。

### 关键实验结果
在长上下文和推理任务上近乎无损保持精度，效率大增：**1M 上下文下 prefill 最高 9.36 倍加速**，decode **约 2.01 倍加速**。证明无需昂贵原生稀疏预训练即可获得强稀疏推理。

### 局限性与开放问题
"内在稀疏"假设在所有模型/任务上是否普适未知；16 维 indexer 的检索召回在极难长程依赖任务上的边界未明（推断）。

### 启发与应用前景
与 #25 OSCAR（KV 量化）、#26 ACC、#27 GoLongRL 共同攻坚长上下文效率。"模型本就稀疏，只需轻量适配"的洞察极具实用价值——百步即可改造现成模型，部署成本极低。

---

## 19. Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos
**👍 92** · https://huggingface.co/papers/2605.18233 · 项目页：https://xiaokunfeng.github.io/miga_homepage/

### 问题与动机
免训练长视频生成希望让基础视频模型产出更长视频而不增大算力。帧级自回归框架（如 FIFO-diffusion）能以恒定显存生成无限长视频，但**训练-推理不匹配**和**长期一致性难维持**限制了基础模型的有效利用。

### 方法与核心创新
提出 **MIGA**：(1) **两阶段对齐机制**，通过减少喂给模型的过量噪声跨度来缓解训练-推理 gap；(2) **双一致性增强机制**——自反思方法纠正早期高噪声帧，长程帧引导方法利用后期低噪声、广覆盖帧来引导生成，联合改善时间一致性。均为免训练。

### 关键实验结果
在 VBench 和 NarrLV 上达到 **SOTA** 性能（摘要未给出具体分数）。

### 局限性与开放问题
免训练方法的上限通常受基础模型能力约束；具体一致性指标提升幅度缺失，与训练型长视频方法的差距未明（推断）。

### 启发与应用前景
与 #15 LongLive-2.0（训练型）形成对比——一个免训练靠推理技巧，一个重基础设施。MIGA 对算力受限、想直接复用现成视频模型的场景很实用。

---

## 20. IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools
**👍 83** · https://huggingface.co/papers/2605.20682

### 问题与动机
MLLM 能桥接视觉感知与文本推理，但在**开放词汇工业异常检测（IAD）** 上常受限于领域错配推理和幻觉式结构推断——工业质检场景下这些幻觉是致命的。

### 方法与核心创新
提出 **IndusAgent**，工具增强的 agentic 框架。先构建 **Indus-CoT** 数据集，整合全局视觉观察、高分辨率局部 patch、专家正常态先验，监督模型在严谨工业检测轨迹上微调。IndusAgent 动态编排外部工具：动态区域裁剪、高频特征增强、先验检索，让 agent 主动解决视觉歧义、辨别细微异常。引入**门控强化学习目标**，联合优化异常分类、定位、类型推理和高效工具使用，确保工具调用只在有益时发生。

### 关键实验结果
在 **5 个工业异常基准**（MVTec-AD、VisA、MPDD、DTD、SDD）上达到 **SOTA 零样本性能**（摘要未给出具体数值）。

### 局限性与开放问题
具体数值缺失。依赖专家正常态先验，对全新产品线的冷启动能力存疑（推断）。工具编排的延迟对产线实时性的影响未讨论。

### 启发与应用前景
"门控 RL 让工具按需调用"是控制 agent 工具滥用的好设计。对智能制造质检直接落地，把 MLLM 从"会看"推向"会查"。

---

## 21. OpenComputer: Verifiable Software Worlds for Computer-Use Agents
**👍 82** · https://huggingface.co/papers/2605.19769 · GitHub: https://github.com/echo0715/OpenComputer

### 问题与动机
Computer-use agent 的评估难在"验证"：一个 agent 可能给出看似正确的答案，但实际软件状态并不对。如何构建可验证的软件世界来训练和评估这类 agent？

### 方法与核心创新
OpenComputer 是 verifier-grounded 框架，四组件：(1) **应用特定状态验证器**，暴露真实应用的结构化检查端点；(2) **自进化验证层**，用执行 grounded 反馈提升验证器可靠性；(3) **任务生成管线**，合成真实且机器可检的桌面任务；(4) **评估 harness**，记录完整轨迹并算可审计的部分得分。当前覆盖 **33 个桌面应用、1,000 个任务**，跨浏览器、办公、创意软件、开发环境、文件管理、通讯应用。

### 关键实验结果
OpenComputer 的硬编码验证器比 **LLM-as-judge** 更贴合人类裁决，尤其在成功依赖细粒度应用状态时。前沿 agent 在端到端完成上挣扎（尽管有部分进展），开源模型相比其 OSWorld-Verified 分数**大幅下滑**，暴露稳健计算机自动化的持久 gap（摘要未给出具体分数）。

### 局限性与开放问题
硬编码验证器需为每个应用定制，扩展到新应用成本高。具体 agent 完成率数值缺失（推断偏低）。

### 启发与应用前景
"硬编码验证器 > LLM-as-judge"的发现对所有 agent 评测有警示意义——别过度依赖 LLM 当裁判。为 computer-use agent 提供可信训练/评估环境，与 #36 EnvFactory（合成可执行环境）思路互补。

---

## 22. Lance: Unified Multimodal Modeling by Multi-Task Synergy
**👍 78** · https://huggingface.co/papers/2605.18678 · GitHub: https://github.com/bytedance/Lance

### 问题与动机
统一多模态模型（理解+生成+编辑，图像+视频）通常靠模型容量 scaling 或 text-image 主导设计。字节团队探索一条更务实的路径：**靠协同多任务训练实现统一，而非堆容量**。

### 方法与核心创新
Lance 是轻量原生统一模型，基于两原则：**统一上下文建模**和**解耦能力通路**。从零训练，在共享交织多模态序列上用**双流 MoE 架构**——联合学上下文的同时解耦理解与生成的通路。引入**模态感知旋转位置编码**缓解异构视觉 token 间干扰、提升跨任务对齐。训练采用分阶段多任务范式，配能力导向目标和自适应数据调度。

### 关键实验结果
在图像和视频生成上**大幅超越现有开源统一模型**，同时保持强多模态理解能力（摘要未给出具体数值）。

### 局限性与开放问题
具体数值缺失，难与闭源统一模型对比。"双流 MoE+模态 RoPE"的设计复杂度对训练稳定性的影响未充分讨论（推断）。

### 启发与应用前景
"多任务协同 > 容量 scaling"的思路与 #1 HRM-Text 异曲同工，都在挑战 scaling 主义。对想用有限算力做统一多模态的团队有借鉴价值。

---

## 23. AI for Auto-Research: Roadmap & User Guide
**👍 67** · https://huggingface.co/papers/2605.18661 · GitHub: https://github.com/worldbench/awesome-ai-auto-research

### 问题与动机
AI 辅助研究正跨过临界点：全自动系统能以**低至 15 美元**生成一篇研究论文。但这个生产力前沿暴露了更深的诚信问题：在科学压力下，即便前沿 LLM 仍会编造结果、漏掉隐藏错误、无法可靠判断新颖性。

### 方法与核心创新
综述截至 2026 年 4 月的进展，把 AI 在完整研究生命周期的应用组织成四个认识论阶段：**Creation**（想法生成、文献综述、编码实验、表图）、**Writing**（论文写作）、**Validation**（同行评审、rebuttal、修订）、**Dissemination**（海报、幻灯片、视频、社媒、项目页、交互 agent）。识别出**阶段依赖的"可靠辅助 vs 不可靠自主"清晰边界**。

### 关键实验结果
综述类，核心结论：AI 擅长结构化、检索 grounded、工具中介的任务，但在**真正新颖的想法、研究级实验、科学判断**上脆弱。生成的想法实现后常退化，研究代码远落后于模式匹配基准，端到端自主系统尚未稳定达到顶会接收标准。更大自主性反而会掩盖（而非消除）失败模式。

### 局限性与开放问题
综述本身依赖 2026 年 4 月前的快照，进展快可能迅速过时。"15 美元生成论文"的质量边界需结合 #6 AutoResearchClaw 的乐观结论辩证看。

### 启发与应用前景
与 #6 形成有趣张力（一个乐观给出 +54.7%，一个强调自主性的脆弱与诚信风险）。结论务实：**人类治理的协作**是最可信部署范式。对所有用 AI 做科研的人是必读的清醒剂。

---

## 24. FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization
**👍 65** · https://huggingface.co/papers/2605.15824 · GitHub: https://github.com/quanjiansong/FashionChameleon

### 问题与动机
人物服装级视频定制有巨大商业价值（电商、内容创作），但现有方法**不支持低延迟交互式服装控制**——用户没法在生成过程中实时换装。且训练通常需要多服装视频数据。

### 方法与核心创新
FashionChameleon 仅用**单服装视频数据**实现交互式多服装定制，三技术：(1) 用单参考-服装对+In-Context Learning 训 Teacher Model，保留 image-to-video 范式同时强制参考与服装图不匹配，隐式学会单服装切换的连贯性；(2) **Streaming Distillation with ICL**，用 in-context teacher forcing 微调，靠梯度重加权的分布匹配蒸馏提升外推一致性；(3) **免训练 KV Cache 重调度**（服装 KV 刷新、历史 KV 撤回、参考 KV 解耦），实现换装同时保运动连贯。

### 关键实验结果
单 GPU 上实现 **23.8 FPS** 实时生成，比现有 baseline **快 30–180 倍**，且独特支持交互式定制和一致长视频外推。

### 局限性与开放问题
仅用单服装数据训练，复杂服装（多层、透明、复杂纹理）的切换保真度可能受限（推断）。23.8 FPS 下的画质与离线方法的差距未明确量化。

### 启发与应用前景
"免训练 KV 重调度实现交互控制"的设计很巧妙。直接服务电商虚拟试衣、直播带货实时换装，30–180 倍加速是落地的关键。

---

## 25. OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization
**👍 64** · https://huggingface.co/papers/2605.17757 · GitHub: https://github.com/FutureMLS-Lab/OSCAR

### 问题与动机
INT2 KV-cache 量化对长上下文 LLM 服务很有吸引力，但**又准又可部署很难**。简单旋转（如 Hadamard）能减异常值，但在 INT2 下仍退化，因为它们**没和下游注意力对齐**。

### 方法与核心创新
OSCAR 离线估计**注意力感知的协方差结构**，据此导出固定旋转和裁剪阈值，使 KV 量化与注意力实际消费的协方差结构对齐。不只给理论证明，还开发了**完全可部署的系统**：自定义 INT2 注意力核，兼容 paged KV-cache 服务和 fused 核管线，可无缝集成进 SGLang、vLLM。

### 关键实验结果
在含 32k 推理轨迹的 5 个任务上：Qwen3-4B-Thinking-2507 上把 BF16 精度差距缩到 **3.78 分**，Qwen3-8B 上 **1.42 分**，而朴素旋转 INT2 **崩溃到接近零**。scale 到 Qwen3-32B 和 GLM-4.7（358B）仍与 BF16 持平。长上下文 RULER-NIAH 128K 上保持鲁棒。系统侧：**KV cache 显存降约 8 倍**，大批量下吞吐**最高 7 倍**，batch-1 解码**最高 3 倍**加速。

### 局限性与开放问题
依赖离线协方差估计，对分布漂移大的新任务可能需重新校准（推断）。INT2 在创意生成等非推理任务上的表现未覆盖。

### 启发与应用前景
"量化对齐注意力实际消费的结构"是深刻洞察，配可部署系统极具工程价值。与 #18 RTPurbo 共同攻克长上下文服务成本，对 LLM 推理降本直接有用。

---

## 26. ACC: Compiling Agent Trajectories for Long-Context Training
**👍 60** · https://huggingface.co/papers/2605.21850

### 问题与动机
agent 发展重燃对 LLM 长上下文推理的需求，但训练这种能力需要昂贵的长文档策划或启发式上下文合成。作者观察到：agent 解题时产生海量轨迹，证据散落在多轮工具调用和环境观察中——而**标准 agent SFT 屏蔽工具响应、只训轮级工具选择，造成监督盲区**。

### 方法与核心创新
提出 **Agent Context Compilation (ACC)**：把搜索、软件工程、数据库查询 agent 的轨迹转成**长上下文 QA 对**，将原问题与跨多轮收集的工具响应和环境观察组合，训模型**不用工具直接作答**。这让问题与证据的依赖显式化，无需额外标注即可直接监督跨远段的长上下文推理，且可与任意现有长上下文扩展/训练方法组合。

### 关键实验结果
用 ACC 训 Qwen3-30B-A3B：MRCR 达 **68.3（+18.1）**，GraphWalks 达 **77.5（+7.6）**，结果可比 **Qwen3-235B-A22B**（用约 1/8 参数达到近似效果），同时保持 GPQA、MMLU-Pro、AIME、IFEval 的通用能力。机理分析显示模型出现任务自适应的注意力重构和专家专门化。

### 局限性与开放问题
依赖现有 agent 轨迹的质量与多样性；"直接作答不用工具"的训练是否会削弱实际工具调用能力，需 trade-off 验证（推断）。

### 启发与应用前景
"把 agent 轨迹回收成长上下文训练数据"是个优雅的数据飞轮思路——agent 跑任务本身就在产生训练料。与 #13 SkillsVote（回收轨迹成技能）共享"轨迹即资产"理念。

---

## 27. GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment
**👍 59** · https://huggingface.co/papers/2605.19577 · GitHub: https://github.com/xiaoxuanNLP/GoLongRL

### 问题与动机
现有长上下文 RL 方法常把数据构建当成"设计越来越复杂的检索路径"，导致任务覆盖同质化、奖励形式不能充分反映实际长上下文需求。

### 方法与核心创新
GoLongRL 是全开源、能力导向的长上下文 RLVR 后训练配方，两贡献：(1) **能力导向数据构建+全量开源**——开放 **23K** RLVR 样本、完整管线和全部训练代码，按长上下文能力分类法覆盖 **9 种任务类型**，每种配自然评测指标，含策划的开源样本和从真实文档（书、论文、多轮对话）生成 QA 的合成样本；(2) **TMN-Reweight** 处理异构多任务优化——任务级均值归一化对齐跨任务奖励尺度，加难度自适应加权做更可靠的优势估计。

### 关键实验结果
同样 vanilla GRPO 设置下，其数据集**单独就超过闭源 QwenLong-L1.5 数据集**。Qwen3-30B-A3B 训后长上下文性能可比 **DeepSeek-R1-0528 和 Qwen3-235B-A22B-Thinking-2507**。TMN-Reweight 进一步超越 vanilla GRPO 且保持通用能力。

### 局限性与开放问题
23K 样本规模相对有限；9 种任务分类法的完备性可能未覆盖所有真实长上下文需求（推断）。

### 启发与应用前景
全开源（数据+管线+代码）对社区价值大。"覆盖广度+奖励多样性 > 复杂检索路径"的结论，与 #26 ACC 都指向长上下文能力的关键在数据构建而非单一技巧。

---

## 28. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation
**👍 59** · https://huggingface.co/papers/2605.11739 · GitHub: https://github.com/caiyuchen-ustc/EffOPD

### 问题与动机
On-policy 蒸馏（OPD）是高效的后训练范式，但**为何高效，参数级机理一直不清楚**——以往只归因于"更密更稳的监督"。

### 方法与核心创新
作者论证 OPD 的高效源于一种"远见（foresight）"：它在训练早期就建立了朝向最终模型的稳定更新轨迹。表现在两方面：(1) **模块分配层**——OPD 识别低边际效用区域，把更新集中在对推理更关键的模块；(2) **更新方向层**——OPD 表现出更强的低秩集中，其主导子空间早期就紧密对齐最终更新子空间。据此提出 **EffOPD**：即插即用加速法，自适应选外推步长、沿当前更新方向移动，无需额外可训练模块或复杂调参。

### 关键实验结果
EffOPD 实现平均 **3 倍训练加速**，同时保持可比的最终性能。

### 局限性与开放问题
"远见"机理在不同模型规模/任务上的普适性需更多验证；外推步长的自适应在不稳定训练中的鲁棒性未充分探讨（推断）。

### 启发与应用前景
"OPD 早期就锁定最终方向"是有洞察的发现，配 EffOPD 的即插即用加速直接实用。与 #35 RELEX（RLVR 轨迹低秩可外推）共享"训练轨迹低秩可预测"的深层观察。

---

## 29. A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook
**👍 56** · https://huggingface.co/papers/2605.20266 · GitHub: https://github.com/Kwwwww74/Awesome-Trustworthy-AudioLLMs

### 问题与动机
大音频语言模型（LALM）是实现通用听觉智能的关键，但其**能力扩张远超确保可信度的系统框架发展**。向统一端到端框架的转变和连续声学信号的整合，内在地扩大了攻击面。

### 方法与核心创新
综述深入 LALM 的内生机制，详述促成涌现推理的架构创新和对齐算法。建立可信度分类法，归类关键漏洞如**跨模态越狱、潜在声学后门、生物特征隐私泄露**。从六个分析支柱审视 SOTA：幻觉、鲁棒性、安全、隐私、公平、认证。

### 关键实验结果
综述类，无定量实验。核心论断：**成熟的攻击格局与欠发展的防御之间存在深刻失衡**（摘要未给出具体数值），验证了音频中心智能面临的可信度 gap 和多维风险。

### 局限性与开放问题
综述快照性质，攻防进展快易过时。提出的"纵深防御"架构、因果听觉世界建模、内在表征工程仍是路线图而非已验证方案。

### 启发与应用前景
与 #2、#8、#9 共同构成"可信多模态"主题。对部署语音助手、声纹认证的团队是安全清单。声学后门、跨模态越狱等新攻击面值得安全研究者关注。

---

## 30. DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo
**👍 54** · https://huggingface.co/papers/2605.16257 · GitHub: https://github.com/brave-eai/dexjoco

### 问题与动机
人类级操作需要灵巧机械手处理复杂物体交互，但现有灵巧操作基准**缺乏反映灵巧手相对平行夹爪独特能力的任务**，也缺综合评估管线。

### 方法与核心创新
DexJoCo 是面向任务的灵巧操作基准+工具包，含 **11 个功能 grounded 任务**，评估工具使用、双手协调、长程执行和推理。开发低成本数据采集系统，采集 **1.1K 轨迹**，支持域随机化以评估鲁棒性。在视觉/动力学随机化、多任务训练、动作头适配等多样设置下基准现代模型。

### 关键实验结果
通过大量实证分析识别出当前策略在灵巧操作上的若干重要洞察和共性局限（摘要未给出具体数值），凸显灵巧手机器人学习的关键挑战。

### 局限性与开放问题
基于 MuJoCo 仿真，sim-to-real gap 未涉及；1.1K 轨迹规模对复杂长程任务可能偏少（推断）。具体模型成功率数值缺失。

### 启发与应用前景
与 #11 PhysBrain、#32 PhysX-Omni 同属具身智能浪潮。为灵巧操作提供标准化评测，"功能 grounded 任务"（工具使用、双手协调）的设计填补了平行夹爪基准的空白。

---

## 31. Auditing Agent Harness Safety
**👍 54** · https://huggingface.co/papers/2605.14271 · GitHub: https://github.com/eric-ai-lab/HarnessAudit

### 问题与动机
LLM agent 越来越多运行在执行 harness 内（调度工具、分配资源、路由消息）。但 harness 可能在**访问未授权资源或泄露上下文给错误 agent 的轨迹上返回看似正确的良性答案**。输出级评估看不到这些失败，而多数安全基准只评最终输出或终态——可很多违规发生在**轨迹中途**而非终止时。

### 方法与核心创新
提出 **HarnessAudit**，审计完整执行轨迹，跨边界合规、执行保真、系统稳定三维度，聚焦风险最突出的多 agent harness。推出 **HarnessAudit-Bench**：**210 个任务**跨 8 个真实领域，单/多 agent 配置都实例化，嵌入安全约束。评估 10 种 harness 配置（跨前沿模型和 3 个多 agent 框架）。

### 关键实验结果
发现：(1) **任务完成与安全执行错位，违规随轨迹长度累积**；(2) 安全风险随领域/任务类型/agent 角色而变；(3) 多数违规集中在资源访问和 agent 间信息传递；(4) **多 agent 协作扩大安全风险面，而 harness 设计设定了安全部署的上限**。

### 局限性与开放问题
210 任务、8 领域的覆盖能否代表所有企业场景存疑；嵌入式安全约束的真实性如何验证未明（推断）。

### 启发与应用前景
与 #3 Code as Agent Harness 直接呼应——一个建框架，一个审安全。"违规随轨迹长度累积""多 agent 扩大风险面"对生产部署 agent 系统是重要警示，呼吁过程级（而非仅输出级）安全审计。

---

## 32. PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects
**👍 53** · https://huggingface.co/papers/2605.21572 · GitHub: https://github.com/physx-omni/PhysX-Omni

### 问题与动机
仿真就绪的物理 3D 资产应用广泛，但多数 3D 生成方法**要么忽略物理属性，要么局限于单一资产类别**（刚体/可变形/铰接其一）。

### 方法与核心创新
PhysX-Omni 是跨多样资产类型的统一仿真就绪物理 3D 生成框架。开发为 VLM 量身定制的高效几何表示，**直接编码高分辨率 3D 结构而不压缩**，显著提升生成性能。构建首个通用仿真就绪 3D 数据集 **PhysXVerse**（覆盖室内外多类）。提出 **PhysX-Bench**，从六属性评估：几何、绝对尺度、材质、affordance、运动学、功能描述。

### 关键实验结果
在常规指标和 PhysX-Bench 上，PhysX-Omni 在生成和理解上**表现强劲**（摘要未给出具体数值）。额外研究验证了其在仿真就绪场景生成和机器人策略学习的潜力。

### 局限性与开放问题
具体定量结果缺失。"不压缩直接编码高分辨率 3D"对 VLM 上下文长度和算力的压力未讨论（推断）。

### 启发与应用前景
统一三类物体（刚体/可变形/铰接）是重要进展。"仿真就绪"直接服务具身 AI 和物理仿真——与 #30 DexJoCo、#11 PhysBrain 共建具身智能的物理基础设施。

---

## 33. Process Rewards with Learned Reliability
**👍 53** · https://huggingface.co/papers/2605.15529 · GitHub: https://github.com/JinYuanLi0012/Beta-Binomial-PRM

### 问题与动机
过程奖励模型（PRM）提供步级反馈，但通常每步**只输出单一奖励分**。下游方法只能把不完美的步级预测当可靠信号用，**没有"何时该信任"的指示**。

### 方法与核心创新
提出 **BetaPRM**，分布式 PRM，**同时预测步级成功概率和该预测的可靠性**。给定来自蒙特卡洛续推的步成功监督，BetaPRM 学一个 Beta 信念，通过 Beta-Binomial 似然解释观察到的成功续推数，而非回归到有限样本成功比的点目标。这个可靠性信号指示何时该信任某步奖励。作为应用，提出**自适应计算分配 ACA**用于 PRM 引导的 Best-of-N：高奖励解可靠时停止，对不确定候选前缀花更多算力。

### 关键实验结果
在 4 个 backbone、4 个推理基准上，BetaPRM 改善 PRM 引导的 Best-of-N 选择同时保持标准步级错误检测。ACA 改善精度-token 权衡：相比固定预算 Best-of-16，**token 用量减少最多 33.57%**同时提升最终答案准确率。

### 局限性与开放问题
Beta-Binomial 假设可靠性可由续推次数良好刻画，续推预算有限时可靠性估计本身可能不准（推断）。

### 启发与应用前景
"奖励不仅给分还给置信度"是个重要扩展，与 #4 DelTA、#5 AntiSD 共同深化奖励/信用分配研究。ACA 的"可靠就停、不确定才花算力"对推理时算力分配（test-time compute）直接有用。

---

## 34. CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?
**👍 53** · https://huggingface.co/papers/2605.16679 · GitHub: https://github.com/actava-ai/chi-bench

### 问题与动机
真实医疗运营的端到端自动化考验三种现有基准未充分覆盖的能力：**政策密度**（决策须 grounded 在大量医疗/保险/运营规则）、**多角色组合**（单任务需扮多角色并交接）、**多边交互**（中间步骤是多轮对话，如同行评审、患者外联）。

### 方法与核心创新
推出 **χ-Bench**，长程医疗工作流基准，跨三领域：provider 事前授权、payer 使用管理、护理管理。每个任务把临床病例交给 agent，在含 **20 个医疗应用、87 个 MCP 工具**的高保真模拟器中驱动至终态，由 **1,290+ 文档**的管理式护理运营手册 skill 引导。

### 关键实验结果
跨 30 个 agent harness/模型配置，**最佳 agent 仅解决 28.0% 任务**，无 agent 在严格 pass³ 上超 20%，**单会话执行所有任务时性能暴跌至 3.8%**。这暗示类似 gap 可能出现在其他政策密集、角色组合、不可逆的企业领域。

### 局限性与开放问题
基于模拟器，真实医疗系统的复杂度和合规约束可能更高；模拟器保真度本身是结论可信度的前提（推断）。

### 启发与应用前景
与 #16 π-Bench、#21 OpenComputer、#31 HarnessAudit 同属长程 agent 评测浪潮。28%/3.8% 的低分对医疗 AI 落地是清醒剂——政策密集、不可逆领域远未到自主部署。

---

## 35. You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories
**👍 50** · https://huggingface.co/papers/2605.21468 · GitHub: https://github.com/weizhepei/RELEX

### 问题与动机
RLVR 是提升 LLM 推理的主导范式，但**其参数轨迹的几何结构一直未被充分探索**。作者要揭示：RLVR 权重轨迹到底长什么样，能否预测？

### 方法与核心创新
证明 RLVR 权重轨迹**极度低秩且高度可预测**：大部分下游性能增益由参数 delta 的 **rank-1 近似**捕获，且该投影的幅度随训练步数**近线性演化**。据此提出 **RELEX**（RL 外推）：从短观察窗口估计 rank-1 子空间，用线性回归外推未来 checkpoint，**无需任何学习模型**。

### 关键实验结果
在 Qwen2.5-Math-1.5B、Qwen3-4B-Base、Qwen3-8B-Base 三模型上，RELEX 产出的 checkpoint 匹配或超过 RLVR 性能（域内外都是），**仅需 15% 步数**。更惊人的是能远超观察窗口外推——**观察前 50 步即可外推到 1000 步**（10–20 倍），且持续改进。消融证明 rank-1 已充分（增秩或非线性建模无额外收益）。成功源于"去噪"效应：投影到 rank-1 子空间丢弃了会损害外推的随机优化噪声。

### 局限性与开放问题
在三个数学/推理模型上成立，对更复杂任务或更大模型的 rank-1 假设是否仍成立未知（推断）。外推过远是否最终发散需边界探讨。

### 启发与应用前景
"RLVR 轨迹本质 rank-1 可外推"是极漂亮的发现，与 #28 EffOPD（OPD 低秩集中）共同揭示训练轨迹的低秩本质。直接大幅降低 RLVR 训练成本——只跑一小段再外推。

---

## 36. EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL
**👍 50** · https://huggingface.co/papers/2605.18703 · GitHub: https://github.com/LARK-AI-Lab/EnvFactory

### 问题与动机
用 Agentic RL 给 LLM 装工具使用能力，受两瓶颈卡住：**缺可扩展、稳健的执行环境**，以及**缺捕捉隐式人类推理的真实训练数据**。现有方法依赖昂贵真实 API、易幻觉的 LLM 模拟器，或常为单轮/依赖预收集文档的合成环境；且合成轨迹常过度规约，像指令序列而非自然人类意图。

### 方法与核心创新
EnvFactory 全自动框架同解两难题：**自主探索并验证有状态、可执行的工具环境**（从真实资源），通过拓扑感知采样和校准精炼**合成自然多轮轨迹**，产出带隐式意图的 grounded query。仅用 **85 个验证环境**跨 7 领域，生成 **2,575 条 SFT 和 RL 轨迹**。

### 关键实验结果
尽管环境数远少于先前工作（常多 **5 倍**），EnvFactory 训练效率和下游性能更优：Qwen3 系列在 BFCLv3 上 **+15%**、MCP-Atlas 上 **+8.6%**、对话基准（τ²-Bench、VitaBench）上 **+6%**。

### 局限性与开放问题
85 个环境的领域覆盖能否泛化到极多样的真实工具生态存疑；"拓扑感知采样"对环境结构复杂度的依赖未充分讨论（推断）。

### 启发与应用前景
与 #21 OpenComputer 共享"合成可验证环境训 agent"思路。"少而精的环境 > 多而泛"的结论对降低 agentic RL 数据成本很有价值，自动化环境构建是 tool-use agent 规模化的关键基础设施。

---

## 🗺️ 趋势洞察

### 1. Agent 工程从"能跑"转向"可验证、可治理、可审计"
**涉及论文**：#3, #13, #14, #21, #31, #36, #16, #34

本周 agent 类论文密集，但焦点已从"让 agent 完成任务"转向**基础设施层的成熟化**。#3 Code as Agent Harness 把代码确立为 agent 的"操作底座"并系统化；#31 HarnessAudit 揭示违规随轨迹长度累积、多 agent 扩大风险面，呼吁**过程级安全审计**而非只看输出；#21 OpenComputer 和 #36 EnvFactory 都在解决"可验证执行环境"的稀缺；#13 SkillsVote 和 #14 MMSkills 把"技能复用+证据门控更新"做成 agent 自我提升的范式。评测侧 #16 π-Bench（主动性）、#34 CHI-Bench（政策密集医疗，最佳仅 28%）暴露长程、不可逆领域离自主部署还很远。核心信号：**agent 的瓶颈不再是模型本身，而是 harness、环境、验证和治理。**

### 2. "答案对≠依据对"——可信多模态成为独立评测维度
**涉及论文**：#2, #8, #9, #29

多篇高赞论文不约而同攻击同一个失败模式：模型给出正确输出却基于错误依据。#2 CiteVQA 发现 MLLM 普遍"归因幻觉"（最强系统 SAA 仅 76，开源仅 22.5）；#8 揭示 51% 正确性格评分未 grounded 在线索上的"偏见鸿沟"；#9 命名视听"聪明汉斯效应"——模型看图说声而不核验音频；#29 系统梳理 LALM 的可信度 gap。这股潮流意味着**评测正从"结果准确率"分化出"过程可信度/可归因性"这一独立且更难的维度**，对法律、医疗、金融等高风险落地至关重要。

### 3. 高效训练的两条路：架构-目标协同 vs. 训练轨迹的低秩可预测性
**涉及论文**：#1, #22, #5, #28, #35, #18, #25

对抗"scaling 主义"的努力分两支。**第一支重新设计架构与目标**：#1 HRM-Text 用分层循环+纯指令数据，1B 模型仅 1500 美元、40B token 就媲美 2-7B 模型；#22 Lance 用多任务协同而非堆容量做统一多模态；#5 AntiSD 反向蒸馏，2-10 倍更少步数。**第二支揭示训练轨迹的内在低秩结构**：#35 RELEX 证明 RLVR 轨迹 rank-1 可外推（15% 步数、观察 50 步外推 1000 步），#28 EffOPD 发现 OPD 早期就低秩对齐最终方向——二者共同指向"训练动力学可预测，故可大幅压缩"。推理侧 #18、#25 则揭示模型本就稀疏、KV 可激进量化。**共同主题：性能增益的"有效自由度"远小于参数规模，效率红利尚未挖尽。**

### 4. 用人类/互联网数据为具身与 agent 能力补课
**涉及论文**：#10, #11, #30, #32, #7

数据稀缺是具身与垂直 agent 的共性瓶颈，本周出现一致的破解思路：**从已有的非专门数据中提炼能力**。#10 Video2GUI 从 5 亿互联网视频造 1200 万 GUI 轨迹；#11 PhysBrain 从人类第一人称视频学物理常识再迁移到机器人；#7 TransitLM 从 1300 万真实交通记录学 map-free 路线规划。配合 #30 DexJoCo、#32 PhysX-Omni 构建仿真就绪的物理基础设施，具身 AI 正在用"先看后做""从真实痕迹学习"绕开昂贵的专门标注。

### 对比与张力

最尖锐的张力在 **#6 AutoResearchClaw（乐观）vs. #23 AI for Auto-Research（清醒）**：前者声称比 AI Scientist v2 高 54.7%、把自动科研推向可用；后者综述截至 2026 年 4 月的全景，强调即便 15 美元能生成论文，前沿 LLM 仍会编造结果、研究代码远落后基准、端到端系统未达顶会标准，且"更大自主性反而掩盖失败"。两者的共识落点是**"人类治理的协作"而非全自主**——#6 自己的消融也证明"高杠杆点精准协作 > 全自主 or 穷尽监督"。

另一处张力是**长视频生成的两条路线**：#15 LongLive-2.0 重金打造 NVFP4 训练+推理基础设施（绑定 Blackwell 硬件），#19 MIGA 则纯靠免训练推理技巧。前者追求工业级吞吐（45.7 FPS），后者追求零额外算力复用现成模型——反映了"投入基础设施"与"巧用现有能力"的永恒权衡。

### 值得关注的研究方向

1. **过程级评估与归因**：从 #2/#8/#9/#31 看，"输出对"已不够，可归因性、过程合规性将成为下一代基准的标配。做 RAG、合规、医疗的团队应提前布局可追溯性。
2. **训练动力学的低秩/可预测性**：#35/#28 揭示的"轨迹 rank-1 可外推"若在更大模型成立，将颠覆训练成本结构——值得密切跟踪能否 scale。
3. **agent 技能/轨迹的回收与治理**：#13/#14/#26 共享"轨迹即资产"理念（轨迹→技能、轨迹→长上下文数据），结合证据门控防污染，是构建 agent 数据飞轮的关键，对运营 skill 生态的团队尤其相关。
4. **长上下文的效率红利**：#18/#25/#26/#27 多管齐下（内在稀疏、KV 量化、轨迹编译、数据构建），长上下文服务成本仍有大幅下降空间。
