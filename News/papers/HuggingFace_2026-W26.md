# HuggingFace 周榜论文深度总结 — 2026 第 26 周

> 来源：https://huggingface.co/papers/week/2026-W26
> 统计日期：2026-07-09
> 筛选条件：upvotes ≥ 50
> 论文数：18

## 目录
1. [MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision](#1-memslides-a-hierarchical-memory-driven-agent-framework-for-personalized-slide-generation-with-multi-turn-local-revision) 👍176
2. [Qwen-AgentWorld: Language World Models for General Agents](#2-qwen-agentworld-language-world-models-for-general-agents) 👍146
3. [Are We Ready For An Agent-Native Memory System?](#3-are-we-ready-for-an-agent-native-memory-system) 👍124
4. [Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models](#4-wan-streamer-v01-end-to-end-real-time-interactive-foundation-models) 👍115
5. [PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems](#5-planbench-xl-evaluating-long-horizon-planning-of-llm-tool-use-agents-in-large-scale-tool-ecosystems) 👍95
6. [DanceOPD: On-Policy Generative Field Distillation](#6-danceopd-on-policy-generative-field-distillation) 👍81
7. [EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions](#7-enterpriseclawbench-benchmarking-agents-from-real-workplace-sessions) 👍79
8. [Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention](#8-grouped-query-experts-mixture-of-experts-on-gqa-self-attention) 👍79
9. [OpenRath: Session-Centered Runtime State for Agent Systems](#9-openrath-session-centered-runtime-state-for-agent-systems) 👍77
10. [DataClaw0: Agentic Tailoring Multimodal Data from Raw Streams](#10-dataclaw0-agentic-tailoring-multimodal-data-from-raw-streams) 👍74
11. [DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation](#11-domainshuttle-freeform-open-domain-subject-driven-text-to-video-generation) 👍67
12. [PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models](#12-perceptiondlm-parallel-region-perception-with-multimodal-diffusion-language-models) 👍64
13. [In-Context World Modeling for Robotic Control](#13-in-context-world-modeling-for-robotic-control) 👍62
14. [NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?](#14-naturebench-can-coding-agents-match-the-published-sota-of-nature-family-papers) 👍62
15. [World Action Models: A Survey](#15-world-action-models-a-survey) 👍56
16. [OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning](#16-opid-on-policy-skill-distillation-for-agentic-reinforcement-learning) 👍54
17. [Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning](#17-escaping-the-self-confirmation-trap-an-execute-distill-verify-paradigm-for-agentic-experience-learning) 👍52
18. [Unlimited OCR Works](#18-unlimited-ocr-works) 👍51

---

## 1. MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision
**👍 176** · 🏛 北京邮电大学 / 清华大学 / 上海交通大学 · https://huggingface.co/papers/2606.17162 · GitHub: https://github.com/huohua325/Memslides

### 问题与动机
个性化 PPT 生成不能只靠「当前这句 prompt + 模板」：Agent 既要跨任务记住用户的稳定偏好（喜欢什么风格），又要在多轮修改中记住这一轮新提出的约束，还要能可靠地做局部小改而不推翻重来。现有做法要么把偏好塞进单一上下文（多轮后遗忘），要么每次改都整份重生成（浪费且破坏未改部分），三者难以兼顾。

### 方法与核心创新
MemSlides 的核心是把记忆做成分层结构：长期记忆再拆成「用户画像记忆」（intent-conditioned profile，为第 0 轮初次生成提供人设对齐）和「工具记忆」（存可复用的执行经验，用于可靠的局部编辑）；工作记忆则承载当前会话内跨轮次的活跃偏好与约束。配套 scoped slide-local revision（局部化修订）——只对受影响的最小区域动刀，而非整份 deck 重生成。三类记忆分别对应「持久人设 / 会话级状态 / 可复用操作经验」。

### 关键实验结果
在受控实验中：用户画像记忆在多人设、多意图的 profile bank 上改善了人设对齐判断；工具记忆注入在诊断性配对（matched-pair）设置下改善了闭环修改行为；定性案例展示了工作记忆的偏好延续能力。摘要以定性和配对对比为主，未给出准确率等硬指标数字（摘要未明确）。GitHub 已收获 558 stars，社区关注度高。

### 局限性与开放问题
最大短板是缺乏统一量化基准——三种记忆各自用不同诊断设置验证，缺一个端到端的「个性化 PPT 质量」总分，难以与其他方案横向比较。工具记忆随任务增长如何避免陈旧/冲突、局部修订的「最小影响区域」如何自动界定，均未展开。

### 启发与应用前景
把「持久画像 / 会话工作记忆 / 可复用执行经验」三层解耦的思路可迁移到任何长期服务同一用户的创作类 Agent（文档、海报、代码助手）。它与本周 [3]、[9] 共同指向「记忆是 Agent 一等系统对象」这一主线。开源代码可直接用于 PPT 自动化产品原型。

## 2. Qwen-AgentWorld: Language World Models for General Agents
**👍 146** · 🏛 阿里巴巴 · https://huggingface.co/papers/2606.24597 · GitHub: https://github.com/QwenLM/Qwen-AgentWorld

### 问题与动机
世界模型（world model，根据当前观测和动作预测环境将如何变化的模型）是推理与规划的核心认知机制。问题是：能否用语言模型来做世界模型，从而拓展通用 Agent 的能力边界？现有 Agent 大多直接在真实环境里试错，采样昂贵、不可控，缺一个「能被语言模型模拟出来」的环境引擎。

### 方法与核心创新
推出 Qwen-AgentWorld-35B-A3B 与 397B-A17B（MoE 架构，A3B=总参 35B 但每 token 只激活 3B；A17B 同理激活 17B），号称首个能通过长链思维（long CoT）模拟 7 大领域 agentic 环境的语言世界模型。基于 1000 万+ 真实环境交互轨迹，用三阶段训练：CPT（持续预训练，从状态转移动态注入通用世界建模能力）→ SFT（激活「下一状态预测」推理）→ RL（用 rubric 与规则混合奖励打磨模拟保真度）。配套 AgentWorldBench，从 5 个前沿模型在 9 个既有基准上的真实交互构建。

### 关键实验结果
在 AgentWorldBench 上显著超越现有前沿模型（摘要为定性表述，未列具体分差）。两条落地路径均验证有效：作为解耦的环境模拟器，可规模化、可控地模拟上千真实环境供 agentic RL 训练，收益超过纯真实环境训练；作为统一 Agent 底座，世界模型训练充当高效「预热」，改善 7 个 agentic 基准的下游表现。GitHub 801 stars。

### 局限性与开放问题
摘要未给出模拟保真度的量化指标（如与真实环境 rollout 的一致率），「显著超越」缺具体数字（摘要未明确）。语言世界模型对物理连续动力学（精确碰撞、流体）的建模精度、以及 CoT 模拟带来的推理开销与延迟，均未讨论。

### 启发与应用前景
「用语言模型当环境模拟器，再喂给 agentic RL」是绕开真实环境采样瓶颈的关键路线，且实测收益超过真实环境训练，工业价值明确。它与 [13]、[15] 共同构成本周「世界模型」主线。Qwen 开源全套代码，可直接用于 Agent 训练的仿真沙盒。

## 3. Are We Ready For An Agent-Native Memory System?
**👍 124** · 🏛 上海交通大学 / 清华大学 / 记忆张量 · https://huggingface.co/papers/2606.24775 · GitHub: https://github.com/OpenDataBox/MemoryData

### 问题与动机
LLM Agent 的记忆已从简单的检索增强（RAG）演化成一个完整的数据管理系统——支持持久存储、检索、更新、整合、动态生命周期治理。但现有评测仍只用端到端任务成功率（F1、BLEU）来衡量，把底层系统当黑盒。结果是运营成本、各记忆模块间的架构权衡、知识动态更新下的鲁棒性这些系统级问题被严重忽视。

### 方法与核心创新
提出一个「数据管理视角」的分析框架，把 Agent 记忆解耦成 4 个核心模块：记忆表示与存储、抽取、检索与路由、维护。在此框架下系统性评测 12 个代表性记忆系统 + 2 个参考基线，覆盖 5 类基准工作负载、共 11 个数据集。关键不是又造一个系统，而是把「记忆系统」拆开做模块级的细粒度消融，量化各模块对表示保真度、检索精度、更新正确性、长程稳定性的独立贡献。

### 关键实验结果
核心结论是「没有单一架构通吃」——效果高度取决于记忆结构与工作负载瓶颈的匹配程度。成本-性能权衡上，一个反直觉但实用的发现是：局部化维护（localized maintenance）比全局重组（global reorganization）更省成本。这为「记忆系统该怎么选型」给出了以负载为导向的判据，而非盲目追一个通用 SOTA。

### 局限性与开放问题
评测仍基于既有 12 个系统的实现，新型 agent-native 架构（论文自己呼吁的方向）尚未落地验证；「模块匹配负载」的结论需要一套负载画像工具才能落地，论文未提供自动选型方法（摘要未明确）。11 个数据集是否覆盖真实长程 Agent 的记忆压力也待检验。

### 启发与应用前景
这是本周「记忆」主线里最偏「基础设施与方法论」的一篇，4 模块框架 + awesome-agent-memory 清单对做记忆系统选型的工程团队是直接可用的地图。与  [1] （分层记忆）、[9]（session 运行时状态）呼应，共同把「Agent 记忆」从功能点抬升为独立系统学科。

## 4. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
**👍 115** · 🏛 阿里巴巴 · https://huggingface.co/papers/2606.25041

### 问题与动机
实时、低延迟、全双工（full-duplex，双方可像打电话一样同时说话）的音视频交互，传统靠级联管线实现：VAD（语音活动检测）→ ASR（语音识别）→ 语言模型 → TTS（语音合成）→ 音频驱动的形象动画 → 视频生成，一堆独立模块串起来。问题是每级都加延迟、且误差层层累积，天花板难破。

### 方法与核心创新
Wan-Streamer 把语言、音频、视频都当作输入也当作输出，统一进单个 Transformer——序列由交错的视觉/音频/文本输入 token 与视觉/音频/文本输出 token 组成，靠 block-causal attention（块因果注意力，只能看过去的块）实现增量流式。不依赖任何外部语言/语音/形象/视频生成模块：感知、推理、生成、响应时机、轮次管理、跨模态同步全部在一个模型里联合学习。整套技术栈围绕「可流式」重做：因果编码器、因果解码器、低延迟多模态 token 调度，流式单元短至 160ms（25fps）。

### 关键实验结果
模型侧响应延迟约 200ms，叠加 350ms 双向网络延迟后，端到端总交互延迟约 550ms，实现亚秒级的双工音视频通信。相比级联管线动辄秒级以上的累积延迟，这是数量级的改善，且避免了跨模块误差累积。摘要未给出与具体级联基线的对话质量对比数值（摘要未明确）。无 GitHub，仅项目页。

### 局限性与开放问题
v0.1 定位为早期版本，摘要未报告生成质量（画面真实度、口型同步、语义正确率）的量化指标，只强调延迟（摘要未明确）。单模型联合训练的数据与算力成本、长对话下的稳定性、多说话人场景均未涉及。

### 启发与应用前景
「把整条音视频交互管线塌缩进一个流式 Transformer」是数字人、实时语音助手、视频客服的方向性设计，200ms 模型延迟已达可用门槛。对做实时多模态交互的团队，其 block-causal + 160ms 流式单元的调度设计值得借鉴。

## 5. PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems
**👍 95** · 🏛 伊利诺伊大学厄巴纳-香槟分校 · https://huggingface.co/papers/2606.22388 · GitHub: https://github.com/JiayuJeff/PlanBench-XL

### 问题与动机
真实场景里 Agent 面对的是庞大的工具生态——需要先检索出相关工具，再推断隐含子目标，还要在动态环境里长程适应。但现有基准几乎不评测「工具可见性受限（retrieval-limited）」下的规划：它们通常把工具直接摆在 Agent 面前，而现实是工具太多、必须先检索才看得到。

### 方法与核心创新
PlanBench-XL 是一个交互式基准：327 个零售任务、1665 个工具，考验 Agent 能否迭代式地检索出可用工具、调用它们来挖出中间证据、再据此发起后续调用直至最终目标。关键创新是可选的 blocking 机制——模拟真实世界的不可预测性，注入缺失、失败或干扰性的工具函数，逼迫 Agent 检测出被打断的路径并在运行时临场适应。这把「静态工具调用」升级为「检索-发现-在故障中重规划」的闭环考验。

### 关键实验结果
10 个主流 LLM 的结果暴露了严峻脆弱性：GPT-5.4 在无阻塞设置下准确率 51.90%，但在最严苛的阻塞条件下崩塌到 11.36%——落差近 40 个百分点。进一步分析显示，Agent 在「故障没有显式错误信号」或「恢复需要更长的替代工具路径」时尤其脆弱。这量化了「大工具生态 + 不完美环境」下长程规划远未解决。

### 局限性与开放问题
任务限于零售域，跨域泛化性未验证；11.36% 的极低分意味着区分度在低端可能被压缩，难分辨「差一点」和「完全不行」的系统（摘要未明确，推断）。「隐式故障」的构造是否覆盖真实工具失败的分布也存疑。

### 启发与应用前景
这是一把诊断 agentic 规划失败的手术刀，尤其把「无显式错误信号的故障恢复」立为核心难点，对做工具调用 Agent（如 MCP 生态）的鲁棒性设计有直接指导。它与 [7] 、[14] 共同构成本周「用真实困难任务给 Agent 降温」的评测主线。

## 6. DanceOPD: On-Policy Generative Field Distillation
**👍 81** · 🏛 字节跳动 / 新加坡国立大学 / 马里兰大学 / 香港科技大学 · https://huggingface.co/papers/2606.27377 · GitHub: https://github.com/worldbench/DanceOPD

### 问题与动机
现代图像生成想要一个模型统一多种能力：文生图（T2I）、局部编辑、全局编辑。但这些能力天然不对齐、常互相打架——加了编辑能力会拖累 T2I，全局编辑和局部编辑又互相干扰。如何把多种能力「组合」进单一模型而不彼此损耗，是训练中的中心难题。

### 方法与核心创新
DanceOPD 是面向 flow-matching 模型（一类通过学习「速度场」把噪声搬运成数据的生成模型）的 on-policy 生成式场蒸馏框架。核心机制：把每个训练样本路由到一个「能力场」（如 T2I 场、编辑场），在学生自己 rollout 出的低噪声状态上查询该场，用简单的速度 MSE（均方误差）目标训练。因为每种专家能力都被定义为共享 flow 状态空间上的一个速度场，学生就在自己走出的状态上向对应专家场学习，从而组合多种专家能力。这个形式还能顺带吸收算子定义的场，比如 classifier-free guidance（CFG，无分类器引导，一种提升条件生成质量的推理技巧）。

### 关键实验结果
在 T2I、编辑、真实感场吸收、CFG 吸收上的实验表明：DanceOPD 改善了多能力组合，在强化目标能力的同时保住了锚点（anchor）生成质量，缓解了「加编辑就掉 T2I」的顽疾。摘要以定性对比为主，未给出 FID 等具体分数（摘要未明确）。GitHub 120 stars。

### 局限性与开放问题
「on-policy 在学生自身 rollout 状态上蒸馏」意味着效果受学生当前分布约束，冷启动阶段学生状态质量低时可能学偏（摘要未明确，推断）。能力场数量增多时的路由冲突、以及缺乏与其他统一图像模型的量化横评，是明显缺口。

### 启发与应用前景
「把每种能力抽象成速度场、让学生在自身轨迹上组合专家」提供了 flow-matching 模型做多能力融合的一条务实路线。值得注意的是它与 [16] OPID 都以「on-policy 蒸馏」为核心——一个在图像生成、一个在 agentic RL，暗示 on-policy 蒸馏正成为跨模态的通用工具。

## 7. EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions
**👍 79** · 🏛 Frontis.AI · https://huggingface.co/papers/2606.23654 · GitHub: https://github.com/FrontisAI/EnterpriseClawBench

### 问题与动机
企业 Agent 越来越多地在真实工作空间里干活：读各种异构文件、调工具、交付业务产物（报告、表格、方案）。但缺一个源自真实企业会话的基准——学术基准的任务往往和真实办公工作流脱节，无法反映企业落地的真实难度。

### 方法与核心创新
EnterpriseClawBench 从大量真实企业 Agent 会话档案构建出 852 个可复现任务，每个任务都配了恢复出来的 fixtures（运行所需的文件/环境）、重写后的 prompt、角色类别、技能子类、硬性规则和语义评分 rubric。由于会话含企业内部内容，作者不释放数据本身，而把「构建与评测协议」作为可复用贡献。设计哲学是：企业评测必须是多轴的，不能塌缩成一个分数。

### 关键实验结果
最强配置（Codex 搭 GPT-5.5）也只拿到 0.663——离「可靠交付」还有明显距离。作者据此主张：企业 Agent 评测必须报告 harness-模型组合、产物交付、视觉质量、成本、运行时、技能迁移行为等多个维度，而不是压成单一总分。这把「harness 和成本」抬为一等评测轴，呼应了近期评测哲学的转向。

### 局限性与开放问题
不释放数据导致外部无法独立复现或审计任务质量，社区只能信任其协议；0.663 这个总分本身又与「反对单一分数」的主张略有张力（摘要未明确，推断）。企业会话的采样偏差（来自哪些行业/岗位）也影响结论泛化。

### 启发与应用前景
「从真实工作会话逆向构建可复现任务 + 多轴报告」对企业 Agent 产品的验收标准有直接价值。它与 [5]、[14] 共同表明：2026 年的 Agent 评测正从刷榜转向「真实、困难、多维度」。数据不开源但协议开源，团队可用其方法自建内部基准。

## 8. Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention
**👍 79** · 🏛 FrontiersMind · https://huggingface.co/papers/2606.20945

### 问题与动机
自注意力是 Transformer 性能的核心，也是长上下文下最贵的部分——token 两两交互，计算量随序列长度平方增长。更浪费的是：标准稠密注意力对每个 token 都用同一套注意力头，不管这个 token 难不难、信息量大不大，一刀切地激活所有头，序列越长越浪费算力。

### 方法与核心创新
提出 Grouped Query Experts（GQE）——在 grouped-query attention（GQA，多个查询头共享同一组 KV 头以缩小 KV cache 的注意力变体）之上叠一层 mixture-of-experts（MoE，用路由器为每个 token 只挑一部分专家激活）。具体做法：在每个 GQA 组内，路由器为每个 token 选出 k 个「查询头专家」，而所有 KV 头保持稠密不变。这样既保住了 GQA 缩小 KV cache 的好处，又只削减「查询头」这部分计算——把 MoE 的稀疏激活思想第一次用到注意力的查询侧。

### 关键实验结果
在固定 30B token 预算、250M 参数规模下，GQE 在下游准确率上追平「全激活 GQA」基线，但每 token 只激活一半的查询头。也就是说，用一半的查询头计算换来了同等性能——这是注意力侧稀疏化的一个干净验证。摘要未给出具体加速比或墙钟时间（摘要未明确）。无 GitHub。

### 局限性与开放问题
实验规模较小（250M 参数、30B token），能否 scale 到数十亿参数、KV 头保持稠密是否会成为新瓶颈，均未验证（摘要未明确，推断）。路由器本身的开销、以及查询头专家选择在极长序列上的稳定性也待考察。

### 启发与应用前景
「稀疏化注意力的查询侧、保留 KV 稠密」是一条与稀疏注意力（选 KV 块）正交的省算路线，可与 GQA/MoE 生态直接组合。它与 [12] 、[18] 共同构成本周「削减注意力/KV 成本」的效率主线，对长上下文推理的算力预算有直接意义。

## 9. OpenRath: Session-Centered Runtime State for Agent Systems
**👍 77** · 🏛 未标注 · https://huggingface.co/papers/2606.19409 · GitHub: https://github.com/Rath-Team/OpenRath

### 问题与动机
现代 Agent 系统的运行时状态是碎片化的：对话记录、工具副作用、记忆事件、工作空间位置、分支来源、回放证据各自分散记录，事后极难检查或复现。想 debug 一次多智能体运行，往往要从一堆外部日志里拼凑现场，fork/merge/replay 都不是一等操作。

### 方法与核心创新
OpenRath 借鉴 PyTorch 的编程模型思路（类比的是「中心化一等运行时抽象」这个角色，不是张量计算）。核心抽象是 Session——在 Agent 与 workflow 之间传递的运行时值，它可分支、可检查、可回放、感知后端、可组合。Session 记录对话块、沙箱位置、血缘元数据、token 用量、待办工作、工具证据，并定义记忆交互在何处进入运行时记录。因为状态由「程序执行时传递的同一个值」承载，fork/merge/replay 就变成显式的运行时操作，而非从外部 trace 事后重建。此外还定义了 Sandbox、Tool、Agent、Memory、Workflow、Selector，其中 Selector 把控制流变成运行时路由的决策。

### 关键实验结果
这是一份系统/工程报告，呈现的是编程模型、架构、经审计的里程碑和证据协议，claim 明确限定在「受控的运行时属性」，把广泛的量化对比、真实 provider 质量、记忆质量等留给后续评估。因此没有基准分数（摘要未明确）。但社区反响强烈——GitHub 1074 stars，是本周这批论文里工程采纳信号最高的之一。

### 局限性与开放问题
作者自陈没有量化对比，无法判断相对现有 Agent 框架（LangGraph 等）的性能/开销优劣；「Session 承载全部状态」在长程运行下的内存与序列化成本、以及可组合性在复杂多智能体拓扑下的表现，均待评估。

### 启发与应用前景
「把 Session 做成一等运行时值，让 fork/merge/replay 成为原语」是对治 Agent 系统可审计性、可复现性的务实工程范式，对做多智能体框架的团队有直接借鉴价值。它从「运行时状态」角度，与 [1] 、[3] 的「记忆」一起，把 Agent 的持久状态管理推向系统化。

## 10. DataClaw0: Agentic Tailoring Multimodal Data from Raw Streams
**👍 74** · 🏛 西安交通大学 / 中国科学院大学 / 深圳理工大学 / 清华大学 · https://huggingface.co/papers/2606.21337 · GitHub: https://github.com/vancyland/DataClaw0

### 问题与动机
海量非结构化多模态流有很高的「数据熵」（data entropy，信息杂乱无序程度高），既妨碍人类高效获取知识，也拖累 AI 后训练的数据质量。现有的被动标注范式——靠启发式规则或通用 VLM（视觉语言模型）——成本高、单调，且无法解锁原始数据里嵌着的深层流程逻辑。

### 方法与核心创新
提出「Agentic Data Tailoring（数据裁剪）」范式：把数据处理本身升级为一种可学习能力，主动地精炼、结构化数据以对齐多样的用户与下游意图。为解决训练这种高阶能力的数据稀缺，设计两阶段管线——把生成式语义合成锚定在确定性的 Factual Anchors（事实锚点）上，产出跨 5 个核心物理与数字领域的大规模数据集。在此之上，DataClaw_0-9B 模型把 SFT（监督微调）与 GRPO（Group Relative Policy Optimization，一种组内相对比较的强化学习算法）协同，实现对复杂裁剪意图的稳健对齐。还构建了 DataClaw_0-val，首个专测数据精炼能力的基准。

### 关键实验结果
关键方法论是「以下游后训练为终极验证」：在视频生成、真实世界 VQA（视觉问答）、GUI 导航三类下游任务上，用 DataClaw_0 裁剪出的高信息密度数据训练，确认能在有限训练数据下高效适配新任务。摘要以定性/下游验证为主，未给出各任务的具体提升点数（摘要未明确）。GitHub 115 stars。

### 局限性与开放问题
「生成式合成锚定在事实锚点」能压多少幻觉、锚点覆盖不到的领域是否会引入偏差，未量化（摘要未明确，推断）。5 个领域的选择依据、以及裁剪能力对分布外数据的泛化，也待检验。

### 启发与应用前景
「把数据处理当作可 RL 训练的 Agent 能力，并以下游后训练效果为验证」是数据工程的范式升级，对做多模态后训练数据管线的团队有直接价值。它把「数据质量」从人工规则问题转成可学习问题，是数据中心 AI 的一个有意思的方向。

## 11. DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation
**👍 67** · 🏛 香港科技大学 · https://huggingface.co/papers/2606.26058 · GitHub: https://github.com/HKUST-C4G/DomainShuttle

### 问题与动机
开放域「主体驱动的文生视频」（S2V，subject-driven text-to-video，给一张参考主体图 + 文本，生成含该主体的视频）有两种场景：in-domain 要尽量保留参考主体特征；cross-domain 要保住主体的内在特征、但允许与主体无关的属性（风格、场景）随文本灵活变化。现有方法主要在 in-domain 追求主体保真度，导致在 cross-domain（新风格、语义组合、域属性变化）下可编辑性和适应性差。

### 方法与核心创新
DomainShuttle 主张理想 S2V 应能在不同域间自由「穿梭」，两端都强。三个核心组件：Domain-MoT（mixture-of-transformers）把视频与参考特征解耦，用 domain-aware AdaLN（自适应层归一化，按域调制特征）对参考图做域特异建模；Video-Reference DualRoPE 把参考图 token 和视频 token 放进各自独立的 RoPE（旋转位置编码）空间，实现精确的主体级空间建模；Cross-Pair Consistent Loss 则专门抽取不受无关特征污染的主体内在特征。三者合力兼顾「保真」与「灵活」。

### 关键实验结果
大量实验显示 DomainShuttle 相比现有方法有显著提升，在多样的开放域应用场景下同时展现高主体保真度与生成灵活性。摘要为定性表述，未给出主体一致性/CLIP 分数等具体数值（摘要未明确）。GitHub 157 stars。

### 局限性与开放问题
「in-domain 保真」与「cross-domain 灵活」本质是一对权衡，摘要未给出量化的权衡曲线，无法判断两端是否真的都达到 SOTA（摘要未明确，推断）。DualRoPE 分离空间带来的额外计算开销、以及多主体场景的表现也未涉及。

### 启发与应用前景
「让同一主体在保真与风格自由间可控穿梭」对个性化视频生成、广告/电商素材制作有实用价值。DualRoPE 把参考与视频 token 放进独立位置编码空间的做法，对做 subject-driven 生成的研究者是可复用的结构设计。它与 [6] 共同代表本周视觉生成方向。

## 12. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models
**👍 64** · 🏛 北京大学 / 字节跳动 · https://huggingface.co/papers/2606.19534 · GitHub: https://github.com/MSALab-PKU/PerceptionDLM

### 问题与动机
多模态大模型（MLLM）在视觉理解上进步巨大，但绝大多数用自回归生成（autoregressive，一个 token 一个 token 顺序产出）。这对「给一张图里的多个区域分别写描述」这类感知任务效率很低——区域只能一个接一个串行处理，区域越多越慢。

### 方法与核心创新
PerceptionDLM 是面向高效并行区域感知的多模态扩散语言模型（DLM，diffusion language model，通过「去噪」并行生成多个 token，而非顺序生成）。基于 PerceptionDLM-Base（在开源扩散 MLLM 中达到 SOTA 的强基线），充分利用 DLM 的并行解码天性。核心是引入高效 prompting 与结构化注意力掩码，让模型能同时感知多个被 mask 的区域，在序列级和 token 级双重并行地生成区域描述。还构建了 ParaDLC-Bench——把 DLC-Bench 扩展到「每张图多个区域 mask」，联合评测描述质量和推理效率。

### 关键实验结果
实验表明 PerceptionDLM 在区域描述质量上保持竞争力的同时，对多区域感知任务实现了显著的速度提升。作者称是首个利用扩散语言模型实现并行区域描述与感知的工作。摘要强调「substantial speed improvements」但未给出具体加速倍数（摘要未明确）。GitHub 73 stars，代码/模型/数据集均已释放。

### 局限性与开放问题
未给出并行相对串行的具体加速比和精度损失曲线，难以判断「并行」是否以质量为代价（摘要未明确，推断）。扩散语言模型本身在复杂推理任务上通常弱于自回归，区域数量极多时的可扩展性与质量稳定性也待验证。

### 启发与应用前景
「用扩散语言模型的并行解码天性做多区域感知」对密集标注、图像区域理解（如自动打标、医学影像多病灶描述）有效率价值。它把扩散 LM 的并行优势从文本生成迁移到视觉感知，是本周效率主线（[8]、[18]）里独特的「换范式提速」思路。

## 13. In-Context World Modeling for Robotic Control
**👍 62** · 🏛 复旦大学 / 上海创智学院 / 同济大学 · https://huggingface.co/papers/2606.26025

### 问题与动机
现代 VLA（Vision-Language-Action，视觉-语言-动作模型）常无法泛化到新配置——比如换了相机视角或机器人本体（morphology）。原因是它们通常只以当前观测和语言指令为条件，把底层系统配置当成固定不变，等于默认了训练时那套执行环境，一旦变了就得靠数据密集的微调。

### 方法与核心创新
提出 In-Context World Modeling（ICWM），把「系统辨识」（system identification，推断出系统的物理/配置参数）当成一个上下文内适应问题。ICWM 让机器人策略从一小段「自己生成的、与任务无关的交互历史」中自主推断出关键系统变量。与传统 in-context learning（用示范告诉模型「做什么任务」）不同，ICWM 用上下文窗口去理解「这个系统是怎么运作的」。在正式执行任务前先处理这些交互，模型就隐式捕获了当前系统的世界动力学，从而不更新任何参数就适应新配置。

### 关键实验结果
在仿真和真实机器人平台上的大量实验表明，ICWM 在「新相机视角」上显著超越标准 VLA 基线。摘要未给出具体成功率提升点数（摘要未明确），但「无需参数更新即可适应新视角」本身是相对「必须微调」基线的实质进步。无 GitHub。

### 局限性与开放问题
「一小段自生成交互」需要多长、对高维系统变量（复杂本体差异）是否够用，未量化（摘要未明确，推断）。摘要主要强调相机视角，对更剧烈的本体变化（自由度不同的机器人）泛化效果如何、以及自生成交互本身的安全性，都待验证。

### 启发与应用前景
「把系统辨识变成上下文内适应，用交互历史理解系统而非指定任务」是让机器人策略免微调泛化的巧妙思路，对机器人快速部署到新硬件/新场景有实用价值。它与 [2]、[15] 共同构成本周「世界模型」主线，且是其中最贴近真实机器人控制的一篇。

## 14. NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?
**👍 62** · 🏛 Frontis.AI / 清华大学 · https://huggingface.co/papers/2606.24530 · GitHub: https://github.com/FrontisAI/NatureBench

### 问题与动机
要回答一个尖锐问题：AI 编码 Agent 能否超越「复现」、走向真正的「发现」？现有「让 Agent 做科研」的基准长期受困于环境碎片化——每篇论文的运行环境难以标准化复现，导致评测可信度低。

### 方法与核心创新
NatureBench 是跨学科基准，含 90 个从同行评审的 Nature 系期刊论文提炼出的任务，专门评测 Agent 能否在真实科学问题上从复现走向发现。它建在 NatureGym 之上——一个自动化管线，能从源论文构建标准化的、每任务独立容器化的环境，直击环境碎片化难题。评测 10 种前沿 Agent 配置，且用严格的「禁用网络搜索」协议（防止 Agent 直接搜到论文答案）。

### 关键实验结果
最强模型在 g>0.1 判据（g 指 Hedges' g 效应量，即相对已发表 SOTA 的改进幅度需超过 0.1 才算数）下，也只在 17.8% 的任务上超越了 SOTA——离「AI 自主做科研」还很远。更关键的发现是路径分析：Agent 主要靠「方法论翻译」取胜（把科学任务转化成熟悉的监督预测问题），而非真正的科学发明；失败主要源于选错方法和算力预算不足，而非没读懂任务。

### 局限性与开放问题
17.8% 的低分说明区分度集中在少数成功任务上；「禁用网络搜索」虽防作弊，但也偏离了真实科研可查文献的常态（摘要未明确，推断）。90 个任务对「发现」的定义仍受限于「超越已发表 SOTA」，未覆盖开放式的原创假设生成。

### 启发与应用前景
「Agent 靠方法论翻译而非科学发明取胜」是对「AI Scientist」热潮的冷静校准，对评估 AI 科研能力的真实边界很有价值。NatureGym 的「从论文自动容器化」管线是可复用的基础设施。它与 [5]、[7] 共同把本周评测主线钉在「真实困难任务暴露 Agent 天花板」。

## 15. World Action Models: A Survey
**👍 56** · 🏛 新加坡国立大学 · https://huggingface.co/papers/2606.20781 · GitHub: https://github.com/world-action-models/awesome-world-action-models

### 问题与动机
World Action Models（WAMs，世界-动作模型）是一类具身的「预测-动作」模型：它对未来做出预测，并让这个预测服务于动作决策。近期 WAM 一支复用大型视频生成模型，另一支则用语言/视觉语言骨干、不含视频生成核心。这种快速扩张把「广义世界模型、视频生成模型、动作接地的视频世界模型、VLA 策略、WAM」之间的边界搅浑了，急需一次梳理。

### 方法与核心创新
这是一篇综述，先厘清上述概念边界，再用两个互补视角组织现有工作。视角一问「每种方法被要求生成什么」——渲染出的未来、潜在（latent）的未来、还是免视频生成的动作推理。视角二把每种方法按「预测基底、骨干、动作耦合方式、部署形态」四维拆解。这套解剖学支撑了对可交互性、因果性、持久性、物理合理性、泛化性的统一讨论，后接数据、评测与开放挑战。

### 关键实验结果
综述提炼出一个一致的设计规律：WAM 不是「视频生成器 + 动作头」那么简单，而是一类「预测-动作」方法，其设计选择本质是在「表征丰富度」与「算力、内存、延迟、动作标注成本」之间做权衡。领域正朝「少生成未来、但保留控制所需的部分」演进——即不必渲染完整未来画面，够用于决策即可。作为综述无基准分数。GitHub 312 stars（awesome 清单）。

### 局限性与开放问题
综述本身不产出新方法或实验，其分类框架能否稳定容纳后续新工作、四维拆解是否遗漏关键维度，需时间检验。「少生成未来」的最优粒度（到底该保留多少未来信息）仍是开放问题。

### 启发与应用前景
「WAM 是预测-动作方法而非带动作头的视频生成器」这一定性，对做具身智能的研究者厘清了方向；四维解剖框架是入门与选型的好地图。它与 [2]、[13] 共同标记了本周「世界模型」主线，且提供了该主线的理论坐标系。

## 16. OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
**👍 54** · 🏛 清华大学 / 浙江大学 / 香港中文大学 / 南洋理工大学 · https://huggingface.co/papers/2606.26790 · GitHub: https://github.com/jinyangwu/OPID

### 问题与动机
基于结果的强化学习（outcome-based RL，只按最终成败给奖励）为语言 Agent 提供了稳定的优化骨干，但轨迹级的稀疏奖励几乎不告诉模型「中间哪一步该被强化、哪一步该被抑制」。on-policy 自蒸馏能提供稠密的 token 级监督，但现有「技能条件化」变体往往依赖外部技能库或检索来的特权上下文——维护成本高，且常与当前策略在多轮交互中诱导出的状态分布不匹配。

### 方法与核心创新
OPID 直接从「已完成的 on-policy 轨迹」里抽取技能监督，不靠外部库。它把轨迹的「事后总结」（hindsight）表示成分层技能：episode 级技能捕获全局工作流或规避失败的规则，step 级技能捕获关键时间步的局部决策知识。critical-first 路由机制在识别到关键决策时用 step 级技能、否则回退到 episode 级作默认引导。选中的技能被注入交互历史，让旧策略在「原始上下文」和「技能增强上下文」下对同一采样响应重新打分，由此产生的 log 概率偏移就构成 token 级自蒸馏优势，再与结果优势结合做策略优化。这样 RL 仍是主目标，同时引入了稠密、分布匹配的事后监督。

### 关键实验结果
在 ALFWorld、WebShop 和基于搜索的 QA 三个基准上，OPID 相比「纯结果 RL」和现有技能蒸馏基线，普遍改善了 Agent 性能、样本效率和鲁棒性。摘要为定性表述，未给出具体成功率提升点数（摘要未明确）。GitHub 82 stars。

### 局限性与开放问题
「从自身轨迹抽技能」意味着技能质量受当前策略水平上限约束，策略弱时抽出的「经验」可能本身是错的（这正是 [17] EDV 警惕的自确认陷阱）；critical 决策的识别准确性、以及跨任务的技能可迁移性未量化（摘要未明确，推断）。

### 启发与应用前景
「把轨迹事后总结成分层技能、用 log 概率偏移做自蒸馏优势」是对治稀疏奖励的务实密化手段，且不依赖外部技能库，工程上更轻。它与 [6] DanceOPD 同属「on-policy 蒸馏」家族（一个 RL、一个图像生成），共同说明该范式的跨领域普适性。

## 17. Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning
**👍 52** · 🏛 浙江大学 / 清华大学深圳国际研究生院 / 西北工业大学 / 中国科学技术大学 · https://huggingface.co/papers/2606.24428

### 问题与动机
经验驱动的自我进化对 LLM Agent 在开放世界中改进至关重要。但现有经验学习方法大多是单 Agent 闭环——同一个 Agent 既执行任务、又总结结果、又决定写什么进记忆。这让 Agent 陷入「自确认陷阱」（Self-Confirmation Trap）：那些「错误但自洽」的轨迹被误判为成功经验，在后续检索复用中导致误差累积、越错越自信。

### 方法与核心创新
提出 EDV（Execute-Distill-Verify，执行-蒸馏-验证）框架，把经验学习从「孤立的自我反思」变成「协作式构建」。Execute 阶段：多个异构 Agent 并行探索同一任务空间，产出多样化的候选轨迹。Distill 阶段：一个专门的第三方 Agent 对这些轨迹做比较分析，产出候选经验，从而削减「执行者中心」的总结偏差。Verify 阶段：执行组通过共识机制验证候选经验，只有被批准的才写入共享或私有记忆。三阶段解耦，在经验入库前就过滤掉错误和噪声内容。

### 关键实验结果
在三个有挑战的长程基准 tau2-bench、Mind2Web、MMTB 上，EDV 一致超越强基线，验证了「可靠的经验构建是 Agent 稳健自我进化的前提」。摘要为定性表述，未给出各基准的具体提升点数（摘要未明确）。作者称代码已开源（github.com/shidingz/EDV）。

### 局限性与开放问题
需要多个异构 Agent 并行执行 + 第三方蒸馏 + 共识验证，计算成本明显高于单 Agent 自反思，摘要未给出成本-收益权衡（摘要未明确，推断）。共识机制在多数 Agent 都犯同一系统性错误时是否仍会误判、异构性从何而来，都是开放问题。

### 启发与应用前景
「用异构执行 + 第三方蒸馏 + 共识验证对治自确认偏差」直击 Agent 自我进化的可信度根源，是本周记忆/经验主线里最有批判性的一篇。它与 [16] OPID 形成有意思的张力：一个警惕单 Agent 自蒸馏的偏差、一个正是单 Agent 自蒸馏——共同勾勒出「自我改进的自偏差」这一未解难题。

## 18. Unlimited OCR Works
**👍 51** · 🏛 百度 · https://huggingface.co/papers/2606.23050 · GitHub: https://github.com/baidu/Unlimited-OCR

### 问题与动机
近期端到端 OCR 模型（以 DeepSeek OCR 为代表）让 OCR（光学字符识别，把图片里的文字转成文本）重回聚光灯。用 LLM 当解码器能借语言先验提升识别质量，但代价明显：输出序列越长，累积的 KV cache（注意力缓存的键值对）越大，内存飙升、生成越来越慢。这与人类形成鲜明对比——人抄写长文档时效率并不会随长度下降。

### 方法与核心创新
提出 Unlimited OCR，模拟人类的「解析工作记忆」。以 DeepSeek OCR 为基线，把解码器里所有注意力层换成自研的 Reference Sliding Window Attention（R-SWA，参考式滑动窗口注意力，只关注固定窗口内的近期内容加少量参考）。R-SWA 在降低注意力计算成本的同时，让 KV cache 在整个解码过程中保持恒定大小，不再随输出长度膨胀。把 DeepSeek OCR 编码器的高压缩率与这个恒定 KV cache 结合，Unlimited OCR 能在标准 32K 最大长度下、单次前向就转写几十页文档。更重要的是，R-SWA 是通用的解析注意力机制，不止 OCR，也适用于 ASR（语音识别）、翻译等。

### 关键实验结果
核心成果是效率突破：单次前向转写「几十页」文档、KV cache 恒定不膨胀、标准 32K 长度即可。摘要以效率/工程能力为主，未给出与 DeepSeek OCR 基线的字符准确率对比数字（摘要未明确）。但社区反响极其强烈——GitHub 高达 13698 stars（百度出品），是本周这批论文里最高，远超第二名。

### 局限性与开放问题
「恒定 KV cache + 滑动窗口」本质是牺牲全局注意力换效率，超长文档中跨页的长程依赖（如前后文一致的表格、跨页引用）是否受损，摘要未量化（摘要未明确，推断）。R-SWA 在 OCR 之外任务上的实际增益也仅是宣称，未给数据。

### 启发与应用前景
「用恒定 KV cache 的滑动窗口注意力破解长文档 OCR 的内存/速度瓶颈」对文档数字化、批量票据/合同解析有直接工业价值，13698 stars 的采纳信号极强。R-SWA 作为通用解析注意力，与 [8] GQE、[12] PerceptionDLM 共同构成本周「削减注意力/KV 成本」的效率主线。

---

## 🗺️ 趋势洞察

### 1. Agent 的记忆、经验与运行时状态正被抬升为「一等系统对象」
**涉及论文**：[1], [3], [9], [16], [17]
**核心观点**：本周最密集的主题不是模型能力，而是「怎么管好 Agent 的状态」。MemSlides（[1]）把记忆分层为持久画像 / 会话工作记忆 / 可复用工具经验；《Are We Ready For An Agent-Native Memory System?》（[3]）干脆用数据管理视角把记忆拆成表示、抽取、检索、维护 4 模块，评测 12 个系统后得出「没有单一架构通吃、局部维护比全局重组更省成本」；OpenRath（[9]）把运行时状态收敛成可 fork/merge/replay 的一等 Session 值（1074 stars）。经验侧，OPID（[16]）从自身轨迹抽分层技能做自蒸馏，EDV（[17]）则警惕单 Agent 自蒸馏的「自确认陷阱」、用异构执行 + 第三方蒸馏 + 共识验证来对治。共同信号：2026 年做 Agent，记忆/经验/状态管理已从功能点升级为独立的系统学科。

### 2. 世界模型：从「环境模拟」走向「服务于动作的预测」
**涉及论文**：[2], [13], [15]
**核心观点**：世界模型这一周从多个层面被系统化。Qwen-AgentWorld（[2]）证明语言模型可以当环境模拟器，用它模拟出的环境训练 agentic RL，收益甚至超过纯真实环境训练；ICWM（[13]）把「系统辨识」变成上下文内适应，让机器人策略从自生成交互里推断系统变量、免微调适应新相机视角；World Action Models 综述（[15]）则厘清了「WAM 不是带动作头的视频生成器，而是预测-动作方法」，并指出领域正朝「少生成未来、够用于决策即可」演进。三篇共同表明：世界模型的价值不在于渲染逼真未来，而在于为决策和 RL 提供可控、可扩展的预测。

### 3. Agent 评测集体转向「真实、困难、多轴」，齐齐给能力降温
**涉及论文**：[5], [7], [14]
**核心观点**：三个新基准不约而同地暴露 Agent 天花板并反对「单一分数」。PlanBench-XL（[5]）在 1665 个工具的生态里加入故障阻塞机制，GPT-5.4 从无阻塞的 51.90% 崩塌到 11.36%；EnterpriseClawBench（[7]）从真实工作会话构建 852 个任务，最强配置也只有 0.663，并主张必须多轴报告 harness、成本、产物质量；NatureBench（[14]）让最强模型在超越已发表 SOTA 上仅得 17.8%，且揭示 Agent 靠「方法论翻译」而非科学发明取胜。共同信号：刷榜时代结束，评测正变成「衡量 Agent 在真实困难任务上到底能不能干活」的仪器。

### 4. 推理效率的多路线攻坚：稀疏注意力、恒定 KV cache、并行解码
**涉及论文**：[4], [8], [12], [18]
**核心观点**：降低注意力/KV 成本本周出现多条互补路线。查询侧稀疏：GQE（[8]）把 MoE 用到 GQA 的查询头，只激活一半查询头就追平全激活基线。KV 恒定：Unlimited OCR（[18]）用 R-SWA 让 KV cache 在整个解码中不随长度膨胀，单次前向转写几十页文档（13698 stars）。换范式提速：PerceptionDLM（[12]）用扩散语言模型的并行解码天性，把多区域感知从串行变并行。塌缩管线：Wan-Streamer（[4]）把整条音视频交互管线塞进一个流式 Transformer，端到端延迟压到约 550ms。四条路线（削查询、压 KV、换并行、塌管线）说明推理效率已无单一银弹。

### 对比与张力
- **记忆做加法 vs 效率做减法**：[1] / [3] / [9] 拼命给 Agent 增加记忆/状态以支撑长程能力，而 [8] / [18] 则拼命削减 KV/注意力以压部署成本——「能力上限」与「部署成本」始终是拉锯的两端，如何让丰富记忆不反噬推理效率尚无定论。
- **自我改进的自偏差 vs 多智能体验证**：OPID（[16]）正是让单 Agent 从自身轨迹自蒸馏，而 EDV（[17]）明确指出这条路会掉进「自确认陷阱」，需异构 Agent + 第三方蒸馏 + 共识验证来纠偏。同一周两篇针锋相对，凸显「用模型自己的判断改进自己」这一范式的未解风险。
- **on-policy 蒸馏成为跨模态通用工具**：DanceOPD（[6]）在 flow-matching 图像生成里做 on-policy 生成式场蒸馏，OPID（[16]）在 agentic RL 里做 on-policy 技能蒸馏——同一范式跨越了图像与 Agent 两个截然不同的领域，暗示「在学生自身 rollout 状态上蒸馏专家」正在成为通用训练工具。
- **生成式补全信息 vs 保真约束**：DomainShuttle（[11]）要在保主体和随文本变之间穿梭，DataClaw0（[10]）用生成式语义合成锚定事实锚点——都在「生成灵活性」与「保真/防幻觉」间走钢丝，可信度边界普遍未被量化。

### 值得关注的研究方向
1. **Agent-native 记忆系统的标准化抽象**：[3] 给出 4 模块框架、[9] 给出 session 运行时抽象、[1] 给出分层记忆，但三者各说各话。把「表示-抽取-检索-维护」与「可 fork/replay 的运行时值」统一进一个可复用的记忆系统标准，是下一代 Agent 框架的明确空白。
2. **语言世界模型作为 agentic RL 的模拟器**：[2] 已证明模拟环境训练能超过真实环境，[13] 给出免微调的上下文内适应——把「语言世界模型当训练沙盒」与「in-context 系统辨识」组合，可能大幅降低 Agent 训练的真实环境采样成本。
3. **on-policy 蒸馏的跨模态迁移**：[6] 与 [16] 已在图像与 RL 各自验证，值得系统研究其在语音、视频、多模态统一模型上的适用边界与失效模式。
4. **真实任务评测的多轴报告标准**：[5] / [7] / [14] 都主张不要把 Agent 性能塌缩成单一分数，但各自的多轴（成本、harness、产物质量、效应量）尚不统一。建立一套跨基准可比的「多轴 Agent 评测报告规范」，是可信 Agent 落地的前提。
