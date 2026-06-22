# HuggingFace 周榜论文深度总结 — 2026 第 23 周

> 来源：https://huggingface.co/papers/week/2026-W23
> 统计日期：2026-06-18
> 筛选条件：upvotes ≥ 50
> 论文数：20

## 目录

1. [On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters](#1-on-the-scaling-of-peft-towards-million-personal-models-of-trillion-parameters) 👍229
2. [Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs](#2-crafter-a-multi-agent-harness-for-editable-scientific-figure-generation-from-diverse-inputs) 👍193
3. [Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding](#3-domino-decoupling-causal-modeling-from-autoregressive-drafting-in-speculative-decoding) 👍146
4. [Cosmos 3: Omnimodal World Models for Physical AI](#4-cosmos-3-omnimodal-world-models-for-physical-ai) 👍121
5. [Audio Interaction Model](#5-audio-interaction-model) 👍112
6. [COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation](#6-colleagueskill-automated-ai-skill-generation-via-expert-knowledge-distillation) 👍112
7. [GrepSeek: Training Search Agents for Direct Corpus Interaction](#7-grepseek-training-search-agents-for-direct-corpus-interaction) 👍109
8. [OCC-RAG: Optimal Cognitive Core for Faithful Question Answering](#8-occ-rag-optimal-cognitive-core-for-faithful-question-answering) 👍91
9. [Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution](#9-code2lora-hypernetwork-generated-adapters-for-code-language-models-under-software-evolution) 👍86
10. [A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks](#10-a-matter-of-taste-improving-coverage-and-difficulty-of-agent-benchmarks) 👍68
11. [Trust-Region Behavior Blending for On-Policy Distillation](#11-trust-region-behavior-blending-for-on-policy-distillation) 👍66
12. [Masking Stale Observations Helps Search Agents -- Until It Doesn't](#12-masking-stale-observations-helps-search-agents----until-it-doesnt-a-regime-map-and-its-mechanism) 👍63
13. [KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks](#13-kvarn-variance-normalized-kv-cache-quantization-mitigates-error-accumulation-in-reasoning-tasks) 👍62
14. [Representation Forcing for Bottleneck-Free Unified Multimodal Models](#14-representation-forcing-for-bottleneck-free-unified-multimodal-models) 👍60
15. [SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue](#15-swanvoice-expressive-long-form-zero-shot-speech-synthesis-for-both-monologue-and-dialogue) 👍58
16. [K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts](#16-k-browsecomp-a-web-browsing-agent-benchmark-grounded-in-korean-contexts) 👍56
17. [Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories](#17-where-do-deep-research-agents-go-wrong-span-level-error-localization-in-agent-trajectories) 👍54
18. [Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses](#18-harness-1-reinforcement-learning-for-search-agents-with-state-externalizing-harnesses) 👍54
19. [Mellum2 Technical Report](#19-mellum2-technical-report) 👍54
20. [From Activation to Causality: Discovery of Causal Visual Representations in the Human Brain](#20-from-activation-to-causality-discovery-of-causal-visual-representations-in-the-human-brain) 👍52

---

## 1. On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters
**👍 229** · https://huggingface.co/papers/2606.02437

### 问题与动机
PEFT（参数高效微调）一直被当作全量微调的"廉价替代品"。本文要重新定位它：把小型可训练 adapter 当作叠加在强大共享基座之上的"持久化本地状态"，让基座负责通用能力，adapter 承载实例专属行为（偏好、技能、工具习惯、类记忆更新）。核心动机是为"每人一个个性化模型"乃至"百万级个人模型挂万亿参数基座"的形态找到技术底座。

### 方法与核心创新
作者把问题组织成三条 scaling 轴：Scale Up（基座越强，小幅本地更新越有用）、Scale Down（adapter 能压到多小仍可靠）、Scale Out（大量持久化适配实例如何共存）。配套提出 MinT 基础设施，管理 adapter 的身份、版本、来源溯源、评测与服务驻留。这把 PEFT 从"省钱手段"重新框定为"可持久承载个人状态的紧凑基底"。

### 关键实验结果
摘要未给出具体数值（无准确率、压缩比、参数量等量化指标），主要为框架性与基础设施层面的论证。

### 局限性与开放问题
（摘要未明确，推断）三轴 scaling 缺乏量化曲线，Scale Down 的可靠性下界、Scale Out 的服务调度成本均未给出实测边界；MinT 的工程开销与冲突管理也待验证。

### 启发与应用前景
若"基座+海量个人 adapter"成立，将催生类似"模型应用商店"的个性化 AI 形态，把记忆、偏好与隐私本地化到 adapter，避免反复全量微调，对端侧个性化助手有结构性意义。

## 2. Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs
**👍 193** · https://huggingface.co/papers/2605.30611 · GitHub: https://github.com/HaozheZhao/Crafter

### 问题与动机
科研配图是论文最费工的环节之一。现有自动生成系统只能处理单一图类、仅接受纯文本输入，且输出是栅格图无法局部修改。作者洞察：科研图本质是"离散语义组件的结构化组合"，生成器在这类版面上犯的是局部错误，需要的不是更强的骨干模型，而是一个"调度框架（harness）"。

### 方法与核心创新
提出两套互补系统：Crafter 是多 agent harness，无需改架构即可跨图类、跨输入条件生成；CraftEditor 用同一模式把栅格输出转成可局部编辑的 SVG。同时发布 CraftBench 基准，覆盖 3 种图类、4 种输入条件并带人工质量标注。关键思路是把"生成质量"问题转化为"组件级编排与纠错"问题。

### 关键实验结果
Crafter 在 PaperBanana-Bench 和 CraftBench 上显著优于独立生成器与 agent 基线；消融确认各组件独立贡献；CraftEditor 转出的可编辑 SVG 超越所有基线。摘要未给出具体分数差，但定性上为全面领先。

### 局限性与开放问题
（摘要未明确，推断）多 agent 编排带来推理成本与延迟；SVG 转换对复杂含位图/渐变的图可能失真；CraftBench 仅 3 图类 4 条件，覆盖面仍有限。

### 启发与应用前景
"harness 而非更大骨干"的范式对结构化生成任务普遍适用。可编辑 SVG 输出契合科研写作的迭代修改需求，有望嵌入论文写作工具链。

## 3. Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding
**👍 146** · https://huggingface.co/papers/2605.29707 · GitHub: https://github.com/jianuo-huang/Domino

### 问题与动机
投机解码（speculative decoding）靠草稿模型一次性起草多 token、再由目标模型并行验证来加速。其瓶颈是草稿质量与草稿成本的权衡：自回归草稿器建模因果依赖但有串行开销，并行草稿器成本低但块内依赖弱。Domino 要把"因果依赖建模"与"昂贵的自回归执行"解耦。

### 方法与核心创新
先用并行草稿骨干一次性产出整块的初步草稿分布，再用轻量级 Domino head 注入前缀相关的因果信息做精修。为稳定 teacher-forced 因果编码，引入"base-anchored 训练课程"：先强化并行骨干，再逐步把优化重心转向因果修正后的最终分布。本质是"先并行出草稿、再轻量补因果"。

### 关键实验结果
在 Qwen3 模型上，Transformers 后端取得最高 5.49× 端到端加速，SGLang 服务下取得最高 5.8× 吞吐加速——相较常规投机解码方案为显著提升。

### 局限性与开放问题
（摘要未明确，推断）加速比与具体接受率、任务类型相关，长尾任务或低接受率场景增益可能缩水；Domino head 的额外训练成本与跨模型族泛化未充分展开。

### 启发与应用前景
"解耦因果建模与执行成本"思路可推广到其他需要并行起草的序列任务。对 LLM 推理服务降本（高吞吐部署）有直接工程价值。

## 4. Cosmos 3: Omnimodal World Models for Physical AI
**👍 121** · https://huggingface.co/papers/2606.02800 · GitHub: https://github.com/NVIDIA/cosmos

### 问题与动机
具身智能（Physical AI）需要同时理解与生成多模态信号。现有体系把视觉语言模型、视频生成器、世界模拟器、世界-动作模型割裂为独立系统。NVIDIA 想用一个统一框架把它们全部"收编"。

### 方法与核心创新
Cosmos 3 是一族全模态世界模型，用统一的 mixture-of-transformers 架构联合处理与生成语言、图像、视频、音频、动作序列，支持高度灵活的输入输出组合。一个框架即同时充当 VLM、视频生成器、世界模拟器、世界-动作模型，作为具身 agent 的通用骨干。

### 关键实验结果
在多样的理解与生成任务套件上确立新 SOTA。后训练版本被 Artificial Analysis 评为最佳开源文生图与图生视频模型，被 RoboArena 评为撰写报告时最佳策略模型。代码、权重、合成数据集、评测基准在 OpenMDW-1.1 许可下开放。摘要未给出具体分数。

### 局限性与开放问题
（摘要未明确，推断）全模态统一架构训练与推理成本极高；动作模态在真实机器人上的 sim-to-real 差距、长时域世界一致性等典型世界模型难题未在摘要展开。

### 启发与应用前景
全模态世界模型作为"具身智能通用骨干"的路线，配合 10306 stars 的开源生态，可能成为机器人/自动驾驶领域的基础设施级底座。

## 5. Audio Interaction Model
**👍 112** · https://huggingface.co/papers/2606.05121 · GitHub: https://github.com/xzf-thu/Audio-Interaction

### 问题与动机
音频天生是交互模态，但当前大型音频语言模型（LALM）都是离线的，流式音频模型又各只做单一任务（流式 ASR 或语音聊天）。作者主张把它们统一成一个在线 LALM：通过"感知-决策-响应"常开循环，实时听声音、环境与指令并即时反应。

### 方法与核心创新
形式化提出 Audio Interaction Model 范式，并以 Audio-Interaction 统一流式模型实现：保留离线任务能力，同时新增在线通用音频指令跟随，能从流的语义自行决定"何时该响应"。配套 SoundFlow 框架端到端落地该循环（流式原生数据构造、理解感知训练、异步低延迟推理）。

### 关键实验结果
构建 StreamAudio-2M 流式语料（2.6M 条，覆盖 7 项基础能力、28 个子任务）与 Proactive-Sound-Bench。在 8 个基准上，模型在主流音频任务保持有竞争力的性能，同时解锁离线 LALM 做不到的实时 ASR、流式音频指令跟随、主动求助等能力。摘要未给出逐项分数。

### 局限性与开放问题
（摘要未明确，推断）"何时响应"的决策时机准确率、误触发率是主动交互的关键风险；实时低延迟与模型规模/能力的权衡未量化。

### 启发与应用前景
"感知-决策-响应"常开循环是从离线问答迈向真正实时语音 agent 的关键范式，对智能助手、车载与无障碍交互有直接价值。

## 6. COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation
**👍 112** · https://huggingface.co/papers/2605.31264 · GitHub: https://github.com/titanwings/colleague-skill

### 问题与动机
LLM agent 越来越被期待承载某个人或角色的专业判断与交互风格。难点在于：可操作的个人知识通常埋在异构的痕迹（traces）里，而非写成干净指令。现有 memory/persona 系统只抓碎片，skill 框架只提供打包格式，缺乏从痕迹到可用 skill 的端到端流程。

### 方法与核心创新
提出"痕迹到 skill"的自动蒸馏系统：给定某人/角色的材料，COLLEAGUE.SKILL 产出带版本的 skill 包，含两条协调的轨道——能力轨（实践、心智模型、决策启发式）与有界行为轨（沟通风格、交互规则、纠错历史）。skill 包可检视、可纠正、可回滚、可跨 agent 宿主安装，并可选准备受控分发。

### 关键实验结果
撰写时公开仓库约 18.5k GitHub stars；gallery 列出来自 165 位贡献者的 215 个 skill，跨 skill 卡片累计超 10 万 stars。摘要以生态采纳度为主要量化指标，未给出任务准确率类数值。

### 局限性与开放问题
（摘要未明确，推断）蒸馏出的"决策启发式"忠实度与可解释性难评估；个人知识蒸馏涉及隐私与归属授权；"有界行为"边界如何防止越权模仿仍是开放问题。

### 启发与应用前景
把"人的专长"做成可检视、可纠错的便携包，而非黑盒 prompt 或隐藏记忆，对企业知识传承、专家 agent 复制有现实意义，也呼应了当前 skill 生态的快速增长。

## 7. GrepSeek: Training Search Agents for Direct Corpus Interaction
**👍 109** · https://huggingface.co/papers/2605.29307 · GitHub: https://github.com/alirezasalemi7/grepseek

### 问题与动机
主流 LLM 搜索 agent 都靠 retriever（关键词/自然语言查询 → 预计算索引返回排序文档）。本文提出互补视角：让 agent 把语料本身当作搜索环境，通过发可执行 shell 命令（如 grep）直接找证据，即"直接语料交互（DCI）"。

### 方法与核心创新
GrepSeek 训练一个紧凑搜索 agent 去查找、过滤、组合大语料证据。为解决直接在大语料上做 RL 不稳定的问题，用两阶段管线：先用 answer-aware Tutor + answer-blind Planner 造冷启动数据（验证过、因果接地的搜索轨迹），再用 GRPO 精修策略。并用语义保持的分片并行执行引擎加速 shell 检索。

### 关键实验结果
分片并行引擎把 shell 检索加速最高 7.6×，且与串行执行字节级等价。在 7 个开放域问答基准上，GrepSeek 取得最强的 token 级 F1 与 Exact Match。分析指出纯词法交互在表面形式变化大的查询上有局限。

### 局限性与开放问题
作者自承纯词法交互对同义/改写类查询乏力，DCI 更适合作为现有检索范式的补充而非替代；shell 执行的安全沙箱与超大语料下的扩展性也需关注。

### 启发与应用前景
"corpus as environment、用 shell 命令检索"重新定义了 agent 与数据的交互方式，对代码库、日志、私有文档等结构化语料的精确检索尤其契合。

## 8. OCC-RAG: Optimal Cognitive Core for Faithful Question Answering
**👍 91** · https://huggingface.co/papers/2606.00683 · GitHub: https://github.com/optimal-cognitive-core/OCC-RAG

### 问题与动机
近年模型进步由 scale 定义，把世界知识塞进权重。但很多实际应用更需要稳健推理而非海量参数化知识。作者主张任务专用小模型（SLM）是更原则化的设计，提出 OCC（最优认知核）系列，并聚焦 OCC-RAG：基于给定上下文做忠实问答，要求多跳推理且忽略记忆知识。

### 方法与核心创新
实现新颖管线，大规模合成多上下文、多跳 QA 数据，产出超 300 万条样本，针对多跳推理、严格上下文忠实、校准的弃答（abstention）。释出 OCC-RAG-0.6B 与 OCC-RAG-1.7B（在该语料上 mid-train），模型产出带源引用的结构化推理链，引用为上下文中的字面原文。

### 关键实验结果
OCC-RAG 在多跳推理（HotpotQA、MuSiQue、TAT-QA）、忠实度（ConFiQA）与拒答（MuSiQue-Un）基准上，能匹配或超越 2–6× 于自身规模的通用模型——即 0.6B/1.7B 小模型胜过数倍大模型。

### 局限性与开放问题
（摘要未明确，推断）合成数据的覆盖偏差可能限制真实世界泛化；强制字面引用在需要归纳/释义的问答上可能僵化；任务专用 SLM 的通用能力下降未评估。

### 启发与应用前景
"任务专用小模型 + 强上下文忠实"对企业 RAG 落地极具吸引力：低成本、可引用、可校准弃答，是对"无脑堆参数"路线的有力反例。

## 9. Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution
**👍 86** · https://huggingface.co/papers/2606.06492

### 问题与动机
代码模型需要仓库级上下文来解析 import、API 与项目约定。现有做法要么把知识塞进长输入（RAG/依赖分析），要么按仓库微调/LoRA——在仓库规模上昂贵且对演进中的代码库很脆弱。Code2LoRA 要用超网络（hypernetwork）生成仓库专属 LoRA，零推理时 token 开销注入仓库知识。

### 方法与核心创新
Code2LoRA-Static 把单个仓库快照转成 adapter，适合稳定代码库的理解；Code2LoRA-Evo 维护一个由 GRU 隐状态按每次 code diff 更新的 adapter，适合活跃开发中的演进代码库。关键创新是"用超网络直接生成 adapter 权重"，免去逐仓库训练且能随提交增量更新。

### 关键实验结果
构建 RepoPeftBench（604 个 Python 仓库）。静态轨：Code2LoRA-Static 取得 63.8% 跨仓与 66.2% 仓内 exact match，匹配逐仓库 LoRA 上界；演进轨：Code2LoRA-Evo 取得 60.3% 跨仓 exact match，比单一共享 LoRA 高 +5.2 个百分点。

### 局限性与开放问题
（摘要未明确，推断）超网络生成的 adapter 质量受训练仓库分布约束，跨语言（非 Python）泛化未验证；GRU 增量更新对大规模 diff 的累积漂移风险待考。

### 启发与应用前景
"零 token 开销注入仓库知识"且能随代码演进增量更新，对 IDE 代码补全/agentic coding 是工程上很优雅的方案，避免长上下文成本与逐仓库微调负担。

## 10. A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks
**👍 68** · https://huggingface.co/papers/2605.28556 · GitHub: https://github.com/tomerkeren42/TASTE-task-synthesis-from-tool-sequence-evolution

### 问题与动机
随 agent 能力进步，τ²-Bench 等基准日益饱和。新建任务又复杂、昂贵、费人力。且标准做法（先写自然语言场景再映射到工具序列）只覆盖了 agent 实际工具用法的窄子集。作者反转任务构造流程来解决覆盖与难度问题。

### 方法与核心创新
提出 TASTE（Task Synthesis from Tool Sequence Evolution）：用基于 LLM 判定有效性信号训练的 Adaptive Contrastive n-gram 模型，采样覆盖海量工具组合的有效工具序列；再经聚类选代表序列、实例化为完整任务、迭代难度演化精修。即"从工具序列反推任务"。

### 关键实验结果
构建 τc-Bench（τ²-Bench 三域的高难扩展）。评测 11 组 agent/user LLM 配对，发现近乎刷满 τ²-Bench 的模型在新任务上大幅掉分（如 Gemini-3-Flash 从 0.82–0.94 跌到 0.28–0.61）。生成任务令 agent 须执行的唯一工具组合数翻倍以上。

### 局限性与开放问题
（摘要未明确，推断）n-gram 采样的有效性依赖 LLM 判定器质量，可能引入判定偏差；自动生成任务的真实性/可解性需人工抽检；难度演化是否过度针对特定失败模式而非真实能力缺口待考。

### 启发与应用前景
揭示"高分常反映饱和而非稳健解题能力"，对评测体系是重要警示。自动化生成高难高覆盖基准，支撑对未来 agent 的可持续、可扩展评测。

## 11. Trust-Region Behavior Blending for On-Policy Distillation
**👍 66** · https://huggingface.co/papers/2605.31159

### 问题与动机
在线策略蒸馏（OPD）让学生在自身策略采样的前缀上学习并匹配更强教师，解决了离线蒸馏的前缀失配。但早期学生 rollout 质量差，导致教师监督被施加在弱/低质前缀上，浪费监督信号。

### 方法与核心创新
提出 Trust-Region behavior Blending（TRB）热身方法：在以学生为中心的 KL 信任域内，用"最接近教师的行为策略"替换早期 rollout 策略，同时保持每前缀的 reverse-KL OPD 损失不变。KL 预算退火到零，热身后训练回归纯学生 rollout。本质是"早期借教师行为生成更好前缀，再平滑过渡回学生"。

### 关键实验结果
在两个数学推理蒸馏设定上，TRB 取得对比方法中的最强平均表现。摘要未给出具体分数差或基线名称。

### 局限性与开放问题
（摘要未明确，推断）信任域大小与退火调度需调参；仅在数学推理两设定验证，跨任务（代码、对话）泛化未知；"最接近教师的行为策略"的求解成本未量化。

### 启发与应用前景
针对 OPD 早期不稳定这一具体痛点的轻量改进，思路可迁移到任何 on-policy 知识迁移场景，对小模型蒸馏训练稳定性有实用价值。

## 12. Masking Stale Observations Helps Search Agents -- Until It Doesn't: A Regime Map and Its Mechanism
**👍 63** · https://huggingface.co/papers/2606.00408 · GitHub: https://github.com/i-DeepSearch/observation-masking

### 问题与动机
长时域搜索 agent 跨多次工具调用累积大量检索内容，上下文预算效率越来越重要。一个极简干预是随轨迹推进 mask 掉陈旧观察，但何时有用、为何有用并不清楚。本文系统刻画这一上下文管理手段的适用边界与机制。

### 方法与核心创新
在 4B 到 284B 多种 agent 骨干、三种 retriever、离线与实时网络搜索基准上做系统扫描。核心发现：mask 带来的准确率增益相对"无上下文管理时的准确率"呈非对称倒 U 形——弱 retriever 下是平台、强 retriever 配中等能力模型时达峰、模型饱和时急剧崩塌。机制上 mask 实现"token 换轮次"的权衡。

### 关键实验结果
增益曲线随模型/检索能力组合呈倒 U（平台-峰值-崩塌）。该模式反映 retriever 召回与模型隐式过滤能力的交互，而非任一单因素。摘要未给出具体峰值百分点。

### 局限性与开放问题
masking 在模型已能隐式过滤（饱和）时反而删除本可用的证据，导致崩塌；何时触发 mask 缺乏自适应判据。作者将上下文管理重新框定为"依赖 regime 的干预"。

### 启发与应用前景
为"是否该做上下文 masking"提供了清晰的决策地图（看 retriever 强度 × 模型容量），对长时域 agent 的上下文工程是可操作的指导，避免盲目套用 masking。

## 13. KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks
**👍 62** · https://huggingface.co/papers/2606.03458 · GitHub: https://github.com/huawei-csl/KVarN

### 问题与动机
测试时扩展（test-time scaling）能提升推理，但长时域解码时 KV-cache 膨胀造成内存瓶颈。KV-cache 量化可缓解，但现有方法在"类 prefill"设定下评估，而自回归解码下误差行为不同。作者发现该场景下量化误差会跨时间步累积，主因是错误的 token 尺度（token scale）。

### 方法与核心创新
提出 KVarN：免校准（calibration-free）的 KV-cache 量化器，先做 Hadamard 旋转，再对 K、V 矩阵的两个轴做双尺度方差归一化（dual-scaling variance normalization）。该组合修正离群 token 尺度误差，显著降低误差累积。

### 关键实验结果
在生成式基准（MATH500、AIME24、HumanEval）的 2-bit 精度下确立 KV-cache 量化新 SOTA。提供 vLLM 实现。摘要未给出与基线的具体分数差。

### 局限性与开放问题
（摘要未明确，推断）2-bit 极低精度下的极端长序列稳定性、Hadamard 旋转的额外计算开销、对非推理类任务的适用性未充分展开。

### 启发与应用前景
直击"自回归解码下量化误差累积"这一被前人忽视的真问题，免校准设计便于即插即用。对长链推理（reasoning model）的内存降本与端侧部署有直接价值。

## 14. Representation Forcing for Bottleneck-Free Unified Multimodal Models
**👍 60** · https://huggingface.co/papers/2605.31604 · 项目页：https://yuqingwang1029.github.io/RepresentationForcing/

### 问题与动机
统一多模态模型（UMM）想在单模型内处理感知与生成，但现有 UMM 仍依赖冻结的、单独预训练的 VAE 做图像生成，构成结构性瓶颈。直接去掉 VAE 又会有质量缺口，因为模型须同时从原始像素学高层结构与低层细节。

### 方法与核心创新
提出 Representation Forcing（RF）：让模型把"表征预测"变成原生能力——强制 decoder 在生成像素前先自回归预测视觉表征作为中间 token，这些 token 留在上下文中、在同一骨干内引导像素扩散。由此把表征从"感知输出"变成"生成目标"，彻底去掉外部生成式潜空间。

### 关键实验结果
图像生成上，带 RF 的像素空间模型可匹配 SOTA 的基于 VAE 的统一模型；图像理解上，像素空间 RF 普遍优于其 VAE 版本。摘要未给出具体 FID/准确率数值。

### 局限性与开放问题
（摘要未明确，推断）自回归预测中间表征会增加生成步骤与延迟；"匹配"而非超越 VAE 基线说明像素空间生成质量仍是挑战；高分辨率扩展性未展开。

### 启发与应用前景
去 VAE 瓶颈、让理解与生成共享同一表征，是迈向真正端到端统一多模态模型的有意义一步，对简化多模态系统架构有启发。

## 15. SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue
**👍 58** · https://huggingface.co/papers/2605.30993 · 项目页：https://swanaigc.github.io/#/swanvoice

### 问题与动机
零样本 TTS 在单说话人合成上已大幅进步，但富表现力的长篇多说话人对话仍困难。常见做法是逐轮用独白模型合成再拼接，这增加成本且常破坏跨轮的声学一致性、对话连贯性与情感连续性。近期对话 TTS 仍难以同时保持表现力连贯、可控说话人切换与独白质量。

### 方法与核心创新
提出 SwanData-Speech（从野外音频建独白与对话语料，用 Swan Forced Aligner 做停顿感知词级对齐、RobustMegaTTS3 处理难发音）与 SwanVoice。SwanVoice 是 1–4 说话人零样本 TTS：含 25 Hz VAE、带停顿符号与拼音替换的原文条件、带说话人轮次条件的 flow-matching DiT；训练从独白起步，经混合与真实对话数据，再用 DiffusionNFT 后训练（音素级 + 说话人相似度奖励）。

### 关键实验结果
在 SwanBench-Speech 上，独白与对话两种设定下，丰富度与层次性得分均高于所有评测的开源基线；内容准确率仍是主要局限。摘要未给出具体分值。

### 局限性与开放问题
作者自承内容准确率（即文本还原正确性）是主要短板；野外数据构建的标注噪声、4 人以上对话扩展性未涉及。

### 启发与应用前景
原生支持长篇多说话人对话且保持情感连续，对有声书、播客生成、对话式数字人有直接应用价值，省去逐轮拼接的工程负担。

## 16. K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts
**👍 56** · https://huggingface.co/papers/2606.02404 · GitHub: https://github.com/prometheus-eval/K-BrowseComp

### 问题与动机
前沿模型评测正从基础能力（指令跟随、推理）转向组合式、agentic 能力，但韩语 agentic 基准稀缺。作者构建 K-BrowseComp，一个扎根韩国语境的网页浏览 agent 基准，共 400 题。

### 方法与核心创新
含 300 题的 K-BrowseComp-Verified 子集由韩语母语者手工构造并校验；另用"hard few-shot 范例 + 针对失败模式的生成"构造 100 题合成集，利用"解题难、出题易"的不对称性。合成集经对抗过滤，作为单独的针对性压力测试报告。

### 关键实验结果
在 Verified 子集上，前沿 LLM（GPT-5.5、DeepSeek-V4-Pro、GLM-5.1）仅达 30.00–45.67%，相比 BrowseComp 大幅下降；通过韩国自主 AI 基础模型计划发布的韩语 LLM 仅得 0.00–10.33%。对抗过滤的合成诊断集上，最强模型仅达 26.00%。

### 局限性与开放问题
（摘要未明确，推断）合成集依赖失败模式定向生成，可能偏向特定难点而非全面能力；网页内容随时间变化导致基准时效性问题；400 题规模对统计稳健性有限。

### 启发与应用前景
揭示前沿模型在非英语（韩语）语境网页浏览上的显著能力鸿沟，尤其本土 LLM 近乎归零，对多语言 agent 评测与本地化能力建设是重要警示。

## 17. Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories
**👍 54** · https://huggingface.co/papers/2606.02060 · GitHub: https://github.com/NJU-LINK/DRIFT · 项目页：https://nju-link.github.io/DRIFT/

### 问题与动机
深度研究 agent 通过搜索、工具调用、证据检视、答案综合的长轨迹解题。基于最终答案的评估只能判断成败，无法定位轨迹中哪部分让答案不可靠。本文研究 span 级错误定位。

### 方法与核心创新
从两个 agent 框架、三个骨干模型、三个基准收集 2,790 条真实轨迹，把原始日志转成语义 span，经 LLM 辅助专家评审标注有害错误 span，构建 TELBench（1,000 实例基准，区分正常探索/失败搜索/试探性假设/无害噪声）。提出 DRIFT：以"声明（claim）为中心"的审计框架，追踪 agent 声明、核对其在轨迹证据中的支持度、标记影响答案路径的无支持/冲突声明的 span。

### 关键实验结果
跨模型族与审计框架的实验显示，DRIFT 把 span 级错误定位与首错（first-error）准确率提升最高 30 个百分点。

### 局限性与开放问题
（摘要未明确，推断）span 标注依赖 LLM 辅助专家评审，存在标注主观性；claim-centric 审计对隐式推理错误（非显式声明）可能漏检；2,790 条轨迹覆盖框架/模型有限。

### 启发与应用前景
把 agent 可靠性从"结果级"推进到"过程级"，对调试、可解释性与可信 agent 部署意义重大，DRIFT 的 claim 审计思路可成为 agent 监控工具的核心组件。

## 18. Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses
**👍 54** · https://huggingface.co/papers/2606.02373 · GitHub: https://github.com/pat-jj/harness-1

### 问题与动机
搜索 agent 常被训练成"在不断增长的 transcript 上的策略"：模型既要决定怎么搜，又要记住看过什么、哪些证据有用、哪些约束未解、哪些声明已核对。作者认为这把太多例行状态管理塞进了策略，逼 RL 同时优化语义搜索决策与本可由环境可靠维护的"可恢复记账"。

### 方法与核心创新
提出 Harness-1，一个 20B 检索子 agent，在"有状态搜索 harness"内用 RL 训练。harness 维护环境侧工作记忆（候选池、重要性标记的精选集、紧凑证据链接、验证记录、压缩去重的观察、预算感知的上下文渲染）；策略只保留语义决策（搜什么、留弃哪些文档、验证什么、何时停）。即"把记账外置给环境，让 RL 专注语义"。

### 关键实验结果
在覆盖网络、金融、专利、多跳 QA 的 8 个检索基准上，Harness-1 取得 0.730 平均精选召回（curated recall），比次强开源搜索子 agent 高 +11.4 个百分点，并与大得多的前沿模型搜索器有竞争力。在留出迁移基准上增益尤其强。

### 局限性与开放问题
（摘要未明确，推断）harness 的状态结构是人工设计的，对不同任务可能需重设计；环境侧记忆维护的工程复杂度与跨域可移植性未充分量化。

### 启发与应用前景
"状态外置 + RL 专注语义决策"是 agent 设计的重要范式，留出迁移上的强泛化暗示该结构能产出可迁移的检索行为，对构建稳健搜索 agent 有方法论价值。

## 19. Mellum2 Technical Report
**👍 54** · https://huggingface.co/papers/2605.31268 · 项目页：https://www.jetbrains.com/mellum/

### 问题与动机
JetBrains 推出 Mellum 2，一个面向软件工程的开放权重通用语言模型，是此前 4B dense、专注补全的 Mellum 的后继。目标是在商用 GPU 上以低推理成本覆盖代码生成与编辑、调试、多步推理、工具调用、agentic coding 与对话式编程辅助。

### 方法与核心创新
12B 参数 MoE（64 专家、激活 8 个），每 token 仅 2.5B 激活参数。架构结合 Grouped-Query Attention（4 KV 头）、每四层中三层用 Sliding Window Attention，以及单个 Multi-Token Prediction 头（既作预训练辅助目标，又内建为投机解码的草稿模型）。每项设计都以商用 GPU 推理效率为约束做了消融验证。

### 关键实验结果
预训练约 10.6 万亿 token，三阶段课程从多样网络数据渐进偏向代码与数学，用 Muon 优化器、FP8 混合精度、WHD 调度训练；经 layer-selective YaRN 扩到 128K 上下文，再两阶段后训练（SFT + RLVR）产出 Instruct 与 Thinking 两个变体。在代码生成、数学推理、工具调用、知识与安全基准上，与 4B–14B 开放权重基线有竞争力，却只跑 2.5B dense 的每 token 算力。Apache 2.0 释出 base/instruct/thinking。

### 局限性与开放问题
（摘要未明确，推断）"与 4B–14B 竞争"说明并未压倒更大模型，绝对能力上限受激活参数所限；MoE 的部署内存占用（需载入全部专家）对端侧仍是负担。

### 启发与应用前景
"MoE 低激活算力 + 内建投机解码草稿头"对成本敏感的编程助手是务实路线，完整公开训练配方（数据、架构、调度）对开源社区复现极具参考价值。

## 20. From Activation to Causality: Discovery of Causal Visual Representations in the Human Brain
**👍 52** · https://huggingface.co/papers/2605.23895 · 项目页：https://yuvalgol123.github.io/BrainCause/

### 问题与动机
识别大脑中哪些区域表征某视觉概念是神经科学核心难题。现有方法用激活最大化定位粗功能区（如人脸、场所），但"强激活"不等于"表征该概念本身"——反应可能由相关的视觉/语义线索驱动。作者要从"激活"推进到"因果"。

### 方法与核心创新
提出 BrainCause 自动框架，结合生成模型与脑模型，合成受控刺激并通过定向因果测试验证神经表征。给定目标概念的查询，框架构造刺激集：概念图、移除目标概念但保留其余内容的反事实编辑图、含候选相关干扰物的图；再用 image-to-fMRI 编码模型预测脑响应，搜索对目标概念（而非相关替代物）特异响应的表征。

### 关键实验结果
方法成功复现已知功能定位，并在数十个概念上识别新候选表征，在预测与实测 fMRI 数据上均得验证。关键地，作者证明若没有因果验证，相当大比例的定位会是假阳性——确认"仅凭激活不足以证明表征"。

### 局限性与开放问题
（摘要未明确，推断）反事实编辑图的生成质量直接影响因果结论的可信度；image-to-fMRI 编码模型本身的误差会传导；"数十个概念"覆盖仍有限，需更大规模实测 fMRI 复核。

### 启发与应用前景
把因果反事实方法引入认知神经科学，为"区分激活与表征"提供可自动化的工具，对脑机接口、视觉认知图谱构建有方法论意义，也示范了生成模型作为受控实验"刺激合成器"的新用法。

---

## 🗺️ 趋势洞察

### 1. Agent 的"状态外置"与"harness 化"成为主线
**涉及论文**：#2, #7, #12, #17, #18
**核心观点**：本周最显眼的范式转变是——不再把所有能力都压进策略/骨干模型，而是把"状态管理、记账、编排、纠错"外置给一个 harness/环境。Crafter（#2）把科研图生成的局部纠错交给多 agent harness 而非更强骨干；Harness-1（#18）把候选池、验证记录、证据链接等工作记忆外置给环境，让 RL 只优化语义决策，换来 +11.4 点召回与强迁移；GrepSeek（#7）把语料本身当环境、用 shell 命令交互；observation masking（#12）研究上下文这一状态该如何管理；DRIFT（#17）则在 claim 级审计 agent 的状态/声明可靠性。共同信念：routine 状态由环境维护更可靠，模型专注语义判断。

### 2. PEFT/Adapter 从"省钱替代"升格为"持久化个性载体"
**涉及论文**：#1, #6, #9
**核心观点**：PEFT 不再被当作全量微调的廉价版。#1 把 adapter 重定义为叠加在万亿参数基座上的"持久个人状态"，并给出三条 scaling 轴与 MinT 基础设施；#9 Code2LoRA 用超网络按 code diff 增量生成仓库专属 LoRA，零 token 开销且随软件演进更新；#6 COLLEAGUE.SKILL 把人的专长蒸馏成可检视、可纠错、可回滚的便携 skill 包。三者共同指向"百万级个性化/角色化模型"的形态：基座共享、个性本地、可版本化管理。

### 3. 小模型 + 任务专用化对抗"无脑堆参数"
**涉及论文**：#8, #13, #19
**核心观点**：在通用大模型军备竞赛之外，本周有一股"务实小模型"逆流。OCC-RAG（#8）用 0.6B/1.7B 任务专用 SLM 在多跳推理与忠实问答上击败 2–6× 大的通用模型；Mellum 2（#19）用 MoE 把激活算力压到 2.5B 却对标 4B–14B；KVarN（#13）用免校准量化在 2-bit 下保住推理质量、降内存。共同逻辑：很多场景需要的是稳健推理与低成本部署，而非海量参数化知识。

### 4. 多模态走向"统一骨干"与"实时交互"
**涉及论文**：#4, #5, #14, #15
**核心观点**：多模态在两个方向收敛。统一化：Cosmos 3（#4）用 mixture-of-transformers 收编 VLM/视频生成/世界模拟/动作模型；RF（#14）去掉 VAE 瓶颈让理解与生成共享原生表征。实时交互化：Audio-Interaction（#5）用"感知-决策-响应"常开循环把离线音频模型变在线；SwanVoice（#15）攻克长篇多说话人对话合成。方向是从"割裂的单任务模块"走向"统一、实时、可交互"的多模态系统。

### 对比与张力
- **大 vs 小**：#4（NVIDIA 全模态巨型世界模型，10306 stars）与 #8 /#19（务实小模型）形成鲜明对照——前者押注"通用骨干 scaling"，后者押注"任务专用 + 低成本"。两条路线并非互斥：#1 的"强基座 + 海量小 adapter"恰是二者的缝合点。
- **能力 vs 评测**：模型能力狂飙的同时，#10（TASTE）、#16（K-BrowseComp）、#17（DRIFT）集中暴露"现有高分多反映基准饱和而非真实能力"——前沿模型在新难基准上从 0.9 掉到 0.3、在韩语网页浏览上近乎归零。评测正成为与建模同等重要的研究战场。
- **激活 vs 表征/因果**：#20 把"强激活≠真表征"的因果反思带进神经科学，与 AI 侧 #17 的"claim 是否真被证据支持"的审计精神同构——两个领域都在追问"表象之下是否有真实机制"。

### 值得关注的研究方向
1. **环境侧工作记忆的标准化**：harness 化是趋势，但各家 harness 的状态结构仍人工设计（#18），亟需可复用、可迁移的状态管理抽象。
2. **个性化 adapter 的服务与治理**：#1/#6/#9 把个性载体做轻了，但 Scale Out（百万 adapter 共存）的调度、隐私、归属授权、版本冲突尚无成熟方案。
3. **过程级可靠性评测**：#17 的 span/claim 级错误定位、#12 的上下文管理 regime map，预示评测从"结果对错"走向"过程可信"，是 agent 落地的关键缺口。
4. **非英语/本土语境 agentic 能力**：#16 揭示的本土 LLM 网页浏览近乎归零，提示多语言 agent 与本地化数据/工具生态是被低估的方向。
