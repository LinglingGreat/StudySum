# HuggingFace 周榜论文深度总结 — 2026 第 25 周

> 来源：https://huggingface.co/papers/week/2026-W25
> 统计日期：2026-07-09
> 筛选条件：upvotes ≥ 50
> 论文数：24

## 目录
1. [Looped World Models](#1-looped-world-models) 👍477
2. [LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](#2-loopcoder-v2-only-loop-once-for-efficient-test-time-computation-scaling) 👍209
3. [JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence](#3-joyai-vl-interaction-real-time-vision-language-interaction-intelligence) 👍209
4. [Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance](#4-moebius-02b-lightweight-image-inpainting-framework-with-10b-level-performance) 👍139
5. [Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories](#5-data-journalist-agent-transforming-data-into-verifiable-multimodal-stories) 👍130
6. [VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models](#6-vibethinker-3b-exploring-the-frontier-of-verifiable-reasoning-in-small-language-models) 👍122
7. [Geometric Action Model for Robot Policy Learning](#7-geometric-action-model-for-robot-policy-learning) 👍117
8. [From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI](#8-from-chatbot-to-digital-colleague-the-paradigm-shift-toward-persistent-autonomous-ai) 👍116
9. [DreamX-World 1.0: A General-Purpose Interactive World Model](#9-dreamx-world-10-a-general-purpose-interactive-world-model) 👍113
10. [OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data](#10-omnidirector-general-multi-shot-camera-cloning-without-cross-paired-data) 👍113
11. [FastContext: Training Efficient Repository Explorer for Coding Agents](#11-fastcontext-training-efficient-repository-explorer-for-coding-agents) 👍93
12. [Ling and Ring 2.6 Technical Report: Efficient and Instant Agentic Intelligence at Trillion-Parameter Scale](#12-ling-and-ring-26-technical-report-efficient-and-instant-agentic-intelligence-at-trillion-parameter-scale) 👍87
13. [APPO: Agentic Procedural Policy Optimization](#13-appo-agentic-procedural-policy-optimization) 👍79
14. [Learning from the Self-future: On-policy Self-distillation for dLLMs](#14-learning-from-the-self-future-on-policy-self-distillation-for-dllms) 👍76
15. [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](#15-memory-is-reconstructed-not-retrieved-graph-memory-for-llm-agents) 👍75
16. [DragMesh-2: Physically Plausible Dexterous Hand-Object Interaction with Articulated Objects](#16-dragmesh-2-physically-plausible-dexterous-hand-object-interaction-with-articulated-objects) 👍74
17. [Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients](#17-zone-of-proximal-policy-optimization-teacher-in-prompts-not-gradients) 👍63
18. [Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages](#18-multi-lcb-extending-livecodebench-to-multiple-programming-languages) 👍60
19. [Measuring Epistemic Resilience of LLMs Under Misleading Medical Context](#19-measuring-epistemic-resilience-of-llms-under-misleading-medical-context) 👍60
20. [GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?](#20-gamecraft-bench-can-agents-build-playable-games-end-to-end-in-a-real-game-engine) 👍58
21. [ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining](#21-ace-ego-0-unifying-egocentric-human-and-robotic-data-for-vla-pretraining) 👍54
22. [MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction](#22-molmomotion-forecasting-point-trajectories-in-3d-with-language-instruction) 👍53
23. [Playful Agentic Robot Learning](#23-playful-agentic-robot-learning) 👍50
24. [Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games](#24-beyond-the-current-observation-evaluating-multimodal-large-language-models-in-controllable-non-markov-games) 👍50

---

## 1. Looped World Models
**👍 477** · 🏛 FaceMind Research Asia · https://huggingface.co/papers/2606.18208

### 问题与动机
世界模型（world model，学习环境动态、能预测「下一步会发生什么」的模型）有一个根本矛盾：忠实的长时程模拟需要很深的计算，但更深的模型部署昂贵、且会累积误差（compounding errors，每步小错逐步放大）。要解决的正是「既要深、又要省、还要稳」这三者不可兼得的困境，这是世界模型走向长时程可用的核心瓶颈。

### 方法与核心创新
提出 Looped World Models（LoopWM），首个用于世界建模的循环架构（looped architecture，反复调用同一组共享参数的 transformer 块，而非堆叠更多不同层）。它通过参数共享的块迭代地精修潜在环境状态，关键机制是「自适应计算」——按每一步预测的复杂度自动决定循环多少次（简单步少循环、难步多循环），把「迭代潜在深度」确立为一条独立于「模型规模、训练数据」的全新扩展轴。

### 关键实验结果
最亮眼的数字是相比传统方法达到最高 100 倍的参数效率——即用 1% 的参数量逼近同等模拟能力（摘要未给出具体基准与对比模型名称，推断）。作者强调这是与「加大模型、加多数据」正交的第三条路。

### 局限性与开放问题
摘要几乎没有给出量化基准（用哪个环境、误差如何、与哪个 SOTA 比），「100 倍参数效率」缺乏参照系，可信度待代码/权重释出后验证（本文无 GitHub）。自适应循环次数如何稳定收敛、极难步是否会无限循环，也未讨论。

### 启发与应用前景
「迭代深度作为 scaling 轴」若成立，对具身仿真、自动驾驶世界模型是普惠性利好——用小模型换深计算。它与 [2] LoopCoder-v2 构成本周最鲜明的「循环」主题呼应，但两者对「循环几次最优」给出了相反的直觉，值得对照追踪。

## 2. LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling
**👍 209** · 🏛 北京航空航天大学 / IQuest Research / 澜舟科技 / 中国人民大学 · https://huggingface.co/papers/2606.18023

### 问题与动机
循环 Transformer 通过重复应用共享块来扩展潜在计算，但串行循环会随循环次数线性推高延迟与 KV 缓存显存（KV cache，注意力缓存的键值，是长上下文显存大户）。并行循环 Transformer（PLT）用跨循环位置偏移（CLP）和共享 KV 的门控滑窗注意力缓解成本，使「循环几次」成为可调设计。问题是：到底该循环几次？

### 方法与核心创新
用「收益—成本」视角研究 PLT 的循环次数：多循环一次可能精修表示，但 CLP 会在每个循环边界引入位置错配。作者从零训练 LoopCoder-v2——一族 7B 的 PLT 编码模型、不同循环次数、在 18T token 上训练并做同配置指令微调。核心发现是循环收益强烈非单调：主要精修发生在第 2 次循环，之后是递减、震荡的更新。

### 关键实验结果
两循环变体全面超越非循环基线：SWE-bench Verified 从 43.0 升到 64.4（+21.4 点）、Multi-SWE 从 14.0 升到 31.0（翻倍多）。而三循环及以上反而退化——因为 CLP 错配成本大致固定，当精修收益缩小后，偏移成本逐渐主导，解释了 PLT 在两循环处饱和。

### 局限性与开放问题
结论「两次最优」是在 PLT + CLP 这一特定架构下得出的，换一种位置编码方案是否仍饱和于 2 未知（摘要未明确，推断）。18T token 从零训练成本极高，社区难以复现其循环次数消融。

### 启发与应用前景
给「循环深度」提供了可操作的选择诊断，直接对治盲目加循环。它与 [1] LoopWM 形成有趣张力：[1] 主张自适应深度、循环越多越省，[2] 却发现两次即饱和——说明「循环是否值得」高度依赖任务与位置编码设计，是循环架构落地的关键开放问题。

## 3. JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence
**👍 209** · 🏛 京东 · https://huggingface.co/papers/2606.14777 · GitHub: https://github.com/jd-opensource/JoyAI-VL-Interaction

### 问题与动机
真实世界的很多时刻不会等你发问：监控里起火、视频通话里一闪而过的表情、直播里想买的商品。但当今大模型基本是「回合制」——被点名才回答，连看似交互的视频通话 app 也只在被轮询/提示时反应。要解决的是让模型像人一样「在场」：持续观察当下、自己决定说还是不说。

### 方法与核心创新
开源 JoyAI-VL-Interaction，一个 8B、视觉优先的 VL 交互模型。最关键机制是把「回应决策」内化到模型里——它每一秒自主选择「沉默 / 回应 / 委派给后台大模型」（难题甩给背景大脑）。配套一套可迁移训练配方，并涌现出未专门训练过的能力，如引导购物者在切换的 app 界面中操作、看着幻灯片即兴讲课。第二个贡献是完整可部署系统：把任意实时视频流喂进模型，ASR/TTS、记忆、UI、背景大脑均可插拔。

### 关键实验结果
在六个真实场景中，人类评分者对 JoyAI-VL-Interaction 的偏好大幅超过豆包和 Gemini 的应用内视频通话助手（摘要用「by a wide margin」定性，未给具体分差，推断）。作者称这是首个开源的视觉驱动交互模型，且连训练配方、数据、可部署系统一并释出。

### 局限性与开放问题
「大幅领先」缺乏量化胜率与置信区间，六个场景的选择是否有利于自家模型未知（推断）。持续观察 + 每秒决策的算力/延迟成本、以及「何时该沉默」的误报率（该说话时沉默、该沉默时插嘴）都未量化。

### 启发与应用前景
从「回合制问答」到「持续在场」是交互范式的实质跃迁，对安防监控、直播电商、陪伴助手、无障碍辅助价值直接。「主模型秒级决策 + 委派后台大脑」的分工，是把小模型的实时性与大模型的深度结合的务实架构。全套开源（含数据与系统）显著降低了社区做实时交互智能的门槛。

## 4. Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance
**👍 139** · 🏛 华中科技大学 / vivo AI Lab · https://huggingface.co/papers/2606.19195 · GitHub: https://github.com/hustvl/Moebius

### 问题与动机
图像修复（image inpainting，把图像里被抹除或缺失的区域自然填补回来）近年靠 10B 级工业基础模型冲到高质量，但其高昂算力严重阻碍落地。做一个任务专用的小专家是出路，但极端结构压缩必然触发「表示瓶颈」——参数太少装不下复杂的图像先验，质量骤降。

### 方法与核心创新
提出 Moebius，系统性重构扩散骨干，核心是 Local-λ Mix Interaction（LλMI）块，由 Local-λ 与 Interactive-λ 两个模块组成，把空间上下文和全局语义先验概括成固定大小的线性矩阵——在大幅砍参数的同时保留复杂的潜在交互。为释放这个紧凑架构的全部能力，配一套自适应多粒度蒸馏：严格在潜空间操作（避免昂贵的像素级解码），动态平衡多个基于梯度的损失以做高保真对齐。

### 关键实验结果
在自然与人像基准上，Moebius 用不到 2% 的参数（0.22B vs 11.9B）就媲美甚至超越 10B 级工业通才 FLUX.1-Fill-Dev 的生成质量，同时总推理时间加速 >15 倍。这把「小 55 倍、快 15 倍、质量不降」立成了高保真修复的新效率标杆。

### 局限性与开放问题
「媲美或超越」的具体质量指标（FID、用户偏好率）摘要未逐项列出（推断）。线性矩阵概括是一种有损压缩，面对超大缺失区域或极复杂纹理时是否仍稳住，边界未量化。

### 启发与应用前景
「专用小专家 + 潜空间蒸馏」逼近 10B 通才，是本周「效率优先」主题（[1] / [6] / [12]）的图像生成代表作，对移动端/端侧修复、批量电商图处理有直接商业价值。LλMI 的「把上下文压成固定线性矩阵」思路可迁移到其他扩散任务的轻量化。

## 5. Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories
**👍 130** · 🏛 牛津大学 / 斯坦福大学 · https://huggingface.co/papers/2606.11176 · GitHub: https://github.com/QinghongLin/data2story-skill

### 问题与动机
数据新闻的价值在于把原始信息变成非专家可信赖的故事，而一篇高质量特稿需要新闻编辑部团队数周：找背景、跑统计、定角度、做可视化。现有 agent 只擅长其中单步（数据科学 agent 做分析、设计 agent 生成网页）。问题是：一个 agent 能端到端胜任数据记者吗？

### 方法与核心创新
提出 Data2Story，把多个专职角色编排成一个虚拟编辑部的多智能体框架。两个创新：(i) 论断证据可溯——一个 Inspector 把每个数字、角度、素材都回链到数据、代码或外部引用；(ii) 多模态生成——不默认纯文本 + 静态图表，而是推理「读者想看什么」再调用多模态工具，如地理用交互地图、音乐用音频。

### 关键实验结果
在 18 篇文章上评测（每篇对标已发表的专家原稿），四条轴：人-agent 角度覆盖、53 名参与者的五维评分、用「电脑操作 agent」当评委作为读者导航的省钱代理、以及可验证性（编码验证器重跑语句核对数据与引用）。结果是产出有竞争力、证据可追溯的多媒体故事，尤其在透明度与可审计性上突出；人类稿件仍在编辑角度、创意设计、呈现上占优。

### 局限性与开放问题
仅 18 篇、53 人评分，样本偏小，统计显著性未报告（推断）。「用 agent 当读者评委」本身是代理指标，其与真实读者行为的一致性未验证。角度创意上仍输人类，说明「找到好故事」这一新闻内核尚难自动化。

### 启发与应用前景
把「每个论断回链证据」做成一等公民，正面回应了 AI 生成内容的可信危机，定位为「记者的协作者」而非替代者更务实。这套「证据可溯 + 多模态按需生成」范式可迁移到财报解读、科普、政务数据披露等需要可审计叙事的场景。

## 6. VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models
**👍 122** · 🏛 新浪微博 · https://huggingface.co/papers/2606.16140 · GitHub: https://github.com/WeiboAI/VibeThinker

### 问题与动机
探究「可验证推理」（verifiable reasoning，答案对错可自动判定的任务，如数学、代码）在严格的小模型（3B）体制内能被推到多远。行业默认小模型只是「部署高效的替代品」，本文要挑战这一假设。

### 方法与核心创新
基于「Spectrum-to-Signal」后训练范式，用一条优化流水线系统增强 3B 稠密模型：课程式监督微调（由易到难排课）、多领域强化学习、离线自蒸馏（用模型自己的高质量输出反哺自己）。由此提出「参数化压缩-覆盖假说」：可验证推理能被压进紧凑的「推理核」，而开放域知识与通用能力需要在事实、概念、长尾场景上的广覆盖参数。

### 关键实验结果
数字极具冲击力：AIME26 达 94.3（配合论断级测试时扩展升到 97.1）、LiveCodeBench v6 的 Pass@1 达 80.2、在近期未见过的 LeetCode 竞赛上 96.1% 通过率（强分布外泛化）。这把一个 3B 模型放进了一线推理系统的性能带，匹配甚至超越体量大几个数量级的 DeepSeek V3.2、GLM-5、Gemini 3 Pro；IFEval 93.4 说明极致推理增强没有牺牲指令可控性。

### 局限性与开放问题
「压缩-覆盖假说」也自承边界：3B 只在可验证推理上追平巨头，开放域知识/通用能力仍需大参数——即它是「互补路径」而非全面替代。测试时扩展（94.3→97.1）的额外算力成本未计入公平对比（推断）。

### 启发与应用前景
「推理可压缩、知识需覆盖」是对「参数即能力」的重要祛魅，指导用小模型专攻数学/代码等可验证域。全套开源 + 1.5B 前作的延续，对做端侧推理助手、低成本竞赛级解题器价值高，也是本周「小模型逼近巨头」主题（[4] / [1]）的推理代表作。

## 7. Geometric Action Model for Robot Policy Learning
**👍 117** · 🏛 KAIST / 苏黎世联邦理工学院 / ETH AI Center · https://huggingface.co/papers/2606.17046 · GitHub: https://github.com/cvlab-kaist/Geometric-Action-Model

### 问题与动机
通用机器人策略要一边听指令、一边推理物体/相机/动作在 3D 物理世界里如何交互。近期的 VLA 模型（Vision-Language-Action，输入图像和语言、直接输出机器人动作）和视频世界-动作模型继承了强语义/时序先验，但主要在 2D 图像帧或 2D 派生的潜空间上操作，把接触密集操作（contact-rich manipulation，如插拔、拧转）真正需要的 3D 几何留成了隐式。

### 方法与核心创新
提出 Geometric Action Model（GAM），直接把预训练的几何基础模型（GFM，专门理解三维几何结构的大模型）当作感知、时序预测、动作解码共用的底座。机制是把 GFM 在中间层「切开」：浅层当观测编码器；在切口插入一个因果未来预测器，以语言、本体感觉、动作历史为条件预测未来潜 token；预测出的 token 再送回 GFM 剩余块做特征传播与解码——一个骨干同时产出未来几何和动作。改动极小却给 GFM 装上了语言条件的时序世界建模，同时保住其丰富几何先验。

### 关键实验结果
在一大批仿真与真机操作基准上，GAM 比当前基础模型规模的基线更准、更鲁棒、更快、更轻（摘要未给出具体分数与对比模型名，推断——四项定性优势同时成立较少见）。

### 局限性与开放问题
缺乏量化数字（提升几点、快几倍）是明显短板，「更快更轻」相对哪个基线未说明。GFM 的几何先验对透明/反光/无纹理物体是否退化，接触动力学是否真被几何潜空间捕捉，尚待验证。

### 启发与应用前景
「复用几何基础模型当感知-预测-动作统一底座」是 VLA 架构的一个优雅方向，与本周多篇具身工作（[16] / [21] / [22] / [23]）共同指向「让机器人策略真正吃进 3D 几何」。切层插预测器的做法可迁移到其他有强预训练骨干的具身任务。

## 8. From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI
**👍 116** · 🏛 腾讯优图实验室 / 清华大学 / 中山大学 / 中南大学 · https://huggingface.co/papers/2606.14502

### 问题与动机
这是一篇立场/综述论文。它要梳理的是 LLM 正从「对话式生成器」转变为具备推理、行动、记忆、自我改进的集成系统这一根本变迁，并给这场变迁一个统一的概念框架：从 Chatbot（对话式回答）到 Digital Colleague（持续性工作）。痛点是当前讨论碎片化，缺乏组织这场转变的坐标系。

### 方法与核心创新
沿两条紧耦合维度组织转变。其一在「认知内核」层：从 Chatbot 时代由下一 token 预测驱动的「快思考」，走向利用推理时计算、思维链（Chain-of-Thought）、反思、过程监督、强化学习的「Thinking LLM」，实现更审慎可靠的认知。其二在「工具增强的任务执行」层：从临时调用外部资源的 tool-calling Agent，走向 OpenClaw 式工作站系统——配备持久工作区（Workspace）、技能、验证循环、治理。「Workspace + Skill」范式靠状态持久化、可复用流程、任务闭环、经验复用，把零散工具使用变得像同事一样。

### 关键实验结果
作为综述本身无实验，但它指出两个可度量的转变：数据构造从「指令-回答对」转向「状态-动作-观测轨迹」（State-Action-Observation），评测从静态基准转向可沙盒化、可审计、自演化的 AI 生态（无量化数字）。

### 局限性与开放问题
综述性质决定了它给的是地图而非路标——「Digital Colleague」的能力边界、失败模式、成本都未被量化。「持久工作区 + 技能库」在长期运行下的记忆膨胀、技能冲突、治理如何落地，均为开放问题。

### 启发与应用前景
它为本周大量 agent 工作（[3] / [11] / [13] / [15] / [23]）提供了共同的概念坐标：从「回答」到「持续工作」。对正在搭 agent 平台的团队，「Workspace + Skill + 验证循环 + 治理」是一份有用的能力清单，也预示评测将从「答得对」转向「能不能像同事一样把活干完并可审计」。

## 9. DreamX-World 1.0: A General-Purpose Interactive World Model
**👍 113** · 🏛 DreamX Team · https://huggingface.co/papers/2606.16993 · GitHub: https://github.com/AMAP-ML/DreamX-World

### 问题与动机
要一个通用的、可交互的文本/图像到视频世界模型，支持可控的长时程生成：相机导航、回访之前看过的区域、可提示的事件，且要跨写实、游戏风、风格化三类域通用。难点是长时程自回归生成里的风格/色彩漂移与「回到旧场景对不上」的记忆一致性问题。

### 方法与核心创新
数据引擎融合了相机精确的 Unreal Engine 渲染、动作丰富的游戏录像、以及恢复了相机几何的真实视频。相机控制上提出 E-PRoPE（投影位置编码的轻量变体，对空间下采样的 token 做相机感知注意力）。把双向视频生成器改造成少步自回归世界模型，用因果强制、DMD 式蒸馏、长 rollout 训练；在自生成的长时程上下文上训练，让模型见到自己生成的历史，从而抑制跨块累积的漂移。Memory-Conditioned Scene Persistence 用相机几何检索早先视角，Event Instruction Tuning 加可组合事件控制。

### 关键实验结果
配合混精 DiT 执行、残差复用、75% 剪枝的 VAE 解码、异步流水并行，在 8 张 RTX 5090 上达最高 16 FPS。5 秒基础评测里相机控制分 73.75、总分 84.76，总分超过 HY-WorldPlay 1.5（80.79）和 LingBot-World（80.45）。

### 局限性与开放问题
16 FPS 需 8 卡 5090，消费级单卡实时仍远；「5 秒」评测下的一致性到更长时程（分钟级）是否保持未知（推断）。记忆检索依赖相机几何，纯风格化/无几何场景下的回访一致性可能退化。

### 启发与应用前景
「可交互 + 长时程 + 记忆持久」的世界模型对游戏、影视预演、具身仿真沙盒价值直接，与 [1] LoopWM 一同构成本周「世界模型」主线。「在自生成历史上训练以抗漂移」是自回归长视频生成的通用技巧，值得迁移。

## 10. OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data
**👍 113** · 🏛 快手 / 清华大学 / 北京大学 · https://huggingface.co/papers/2606.13432 · GitHub: https://github.com/lisj575/OmniDirector

### 问题与动机
从参考视频克隆相机运动，是视频生成里的重要控制手段（视频比参数更直观精确）。现有方法要么直接用参数化表示、无法处理多镜头（multi-shot，一段视频含多个机位/镜头切换）生成，要么合成跨配对数据、受困于数据稀缺，在复杂相机运动克隆上表现差。

### 方法与核心创新
核心是一种通用相机运动表示：把相机编码成「网格运动视频」（camera grid，用可视化的网格把相机参数表达成视频形式）。这一表示天然支持把多条轨迹整合进多镜头生成。基于此提出 OmniDirector，在百万级「相机网格-视频」配对上训练，为多模态扩散 transformer 协调角色、动作、相机，提供导演级控制。还设计了分层提示扩展 agent，通过理解信号关系、系统描述相机运动与视觉内容，把不同控制信号和谐整合。

### 关键实验结果
大量实验显示其性能与可控性优越（摘要未给出具体量化分数或对比基线名称，推断）。关键卖点是「无需跨配对数据」即实现复杂多镜头相机克隆，绕开了数据稀缺瓶颈。

### 局限性与开放问题
缺乏量化结果是硬伤——「优越」相对谁、领先多少均未说明。「网格运动视频」表示的精度上限（能否精确复现快速甩镜、变焦）、以及百万级配对数据本身的获取成本未讨论。

### 启发与应用前景
「把相机参数可视化成网格视频」是个巧妙的表示统一——让相机控制和视觉内容在同一模态里被扩散模型处理，与 [9] DreamX-World 的相机控制形成同主题对照。对 AI 影视、虚拟制片的分镜/运镜控制有直接价值。

## 11. FastContext: Training Efficient Repository Explorer for Coding Agents
**👍 93** · 🏛 微软 / 上海交通大学 · https://huggingface.co/papers/2606.14066 · GitHub: https://github.com/microsoft/fastcontext

### 问题与动机
LLM 编码 agent 在软件工程任务上已很强，但「仓库探索」仍是主要瓶颈：定位相关代码消耗大量 token 预算，还把无关片段塞进上下文污染 agent。多数 agent 用同一个模型既探索又解题，探索性的读取和搜索全留在解题者的历史里，越滚越脏。

### 方法与核心创新
提出 FastContext，一个专职的探索子 agent，把「仓库探索」从「解题」中剥离。按需调用、发起并行工具调用、只返回精炼的文件路径和行号范围作为聚焦上下文。它由 4B–30B 的专用探索模型驱动：先用强参考模型的轨迹做冷启动，再用任务锚定的奖励精修——奖励覆盖广泛的首轮搜索、多轮取证、精确引用生成三种能力。

### 关键实验结果
在 SWE-bench Multilingual、SWE-bench Pro、SWE-QA 上，把 FastContext 集成进 Mini-SWE-Agent，端到端解决率最高提升 5.5%，同时把编码 agent 的 token 消耗最高降低 60%，额外开销很小。这证明探索可以从解题中分离并由专用小模型高效承担。

### 局限性与开放问题
解决率 +5.5% 与 token -60% 是「最高」值，平均增益未突出（推断）。专用探索模型需单独训练与维护，多一个模型的部署复杂度是隐性成本。子 agent 只返回路径+行号，可能漏掉需要全文才能判断相关性的边界情形。

### 启发与应用前景
「探索/解题分离 + 专用小探索模型」是对 agent 上下文污染的务实解法，微软出品且给出模型权重（HF 上有 FastContext-1.0-4B）。这一「子 agent 只回聚焦上下文」范式可迁移到深度搜索、长文档 QA 等任何受上下文预算约束的 agent 场景。

## 12. Ling and Ring 2.6 Technical Report: Efficient and Instant Agentic Intelligence at Trillion-Parameter Scale
**👍 87** · 🏛 Inclusion AI · https://huggingface.co/papers/2606.15079

### 问题与动机
高效可扩展的 agentic 智能既要低延迟响应、又要强推理，还要在训练、服务、部署上实际可行。要解决的是「快」与「强」与「可落地」三者的统一，尤其在万亿参数规模上。

### 方法与核心创新
提出 Ling-2.6（面向即时响应、单位输出 token 能力高）与 Ring-2.6（面向深度推理与高级 agentic 工作流）。不从零训练，而是通过「架构迁移预训练 + 大规模后训练」升级 Ling-2.0 底座，并用「模型架构、优化目标、服务系统、agent 训练环境」的统一协同设计来指导。架构上引入混合线性注意力，把 Lightning Attention 与 MLA（Multi-head Latent Attention，压缩 KV 缓存的注意力变体）结合，提升长上下文训练/解码效率。为提升单 token 能力，用进化式思维链、语言单元策略优化、双向偏好对齐、最短正确回答蒸馏。agentic 上提出 KPop 强化学习框架，支撑 Ring-2.6-1T（万亿参数）在大规模环境锚定数据上的稳定训练，靠跨编码/搜索/工具/工作流的异步调度提升训练效率。

### 关键实验结果
摘要主打方法与开源，未给出具体基准分数与对比数字（摘要未明确，推断）。核心可度量卖点是「万亿参数规模上同时优化能力与部署效率」，并开源 2.6 全家 checkpoint。

### 局限性与开放问题
缺乏量化基准使「高效」「即时」难以横比。混合线性注意力 + MLA 在超长上下文下的精度损失边界、KPop 在万亿参数上的训练稳定性代价均未量化。工程栈极复杂，社区复现门槛高。

### 启发与应用前景
「快模型 + 慢模型」双档 + 万亿参数全开源，对搭 agentic 服务的团队是重要开放底座，延续本周「效率优先」主线。「架构-优化-服务-环境协同设计」的方法论，比单点 trick 更值得工程团队借鉴。

## 13. APPO: Agentic Procedural Policy Optimization
**👍 79** · 🏛 中国科学技术大学 / 阿里巴巴 / 南方科技大学 · https://huggingface.co/papers/2606.12384 · GitHub: https://github.com/AMAP-ML/APPO

### 问题与动机
agentic 强化学习大幅提升了 LLM agent 的多轮工具使用能力，但多数方法在粗粒度的启发式单元（工具调用边界、固定工作流）上做信用分配（credit assignment，判断最终成败该归功/归咎于哪一步决策），难以识别到底哪个中间决策影响了结果。

### 方法与核心创新
先做试点分析，得两个反直觉观察：有影响力的决策点广泛分布在整段生成里，而非集中在工具调用处；且仅靠 token 熵（不确定性）并不可靠反映其对最终结果的影响。据此提出 APPO，把分支与信用分配从粗粒度交互单元下沉到序列中细粒度的决策点。用「分支分数」结合 token 不确定性与后续续写的策略诱导似然增益来选分支位置，既做更有针对性的探索、又过滤掉虚假的高熵位置；再用「过程级优势缩放」在分支 rollout 间更好地分配信用。

### 关键实验结果
在 13 个基准上，APPO 在强 agentic RL 基线上稳定提升近 4 个点，同时保持高效的工具调用与行为可解释性。「4 点」虽不惊人，但跨 13 个基准的一致性说明是方法性而非调参增益。

### 局限性与开放问题
+4 点相对哪些具体基线均值未逐项拆分（推断）。「分支分数」引入额外计算，其相对 rollout 成本的开销未量化。决策点广泛分布的观察是否泛化到非工具类任务未验证。

### 启发与应用前景
「影响力决策点广泛分布、token 熵不可靠」是对 agentic RL 信用分配的重要经验修正，直接挑战「在工具边界打分」的惯例。细粒度分支 + 过程级优势的思路，可迁移到任何需要在长序列里定位关键决策的 RL 训练。

## 14. Learning from the Self-future: On-policy Self-distillation for dLLMs
**👍 76** · 🏛 清华大学 / 慕尼黑工业大学 / 南洋理工大学 / 英属哥伦比亚大学 · https://huggingface.co/papers/2606.18195 · GitHub: https://github.com/xingzhejun/d-opsd-code

### 问题与动机
在线自蒸馏（OPSD，用模型自己在策略上的输出蒸馏自己）对自回归 LLM 后训练有效，但对扩散 LLM（dLLM，不从左到右逐词生成、而像图像扩散那样并行多步去噪地生成整段文本）尚属空白。现有 OPSD 本质是自回归中心的：靠从左到右的前缀条件注入特权信息、做 token 级散度监督——这与 dLLM 的任意顺序生成根本冲突。

### 方法与核心创新
提出 d-OPSD，首个为 dLLM 定制的 OPSD 框架，两个核心贡献。其一重构「自教师」：用模型自生成的答案作为「后缀条件」，让学生从「自身未来经验」（self future-experience，先看到自己会得出的答案再回头学）而非特权前缀中学习。其二把监督从 token 级移到步级，对齐 dLLM 的迭代去噪过程。

### 关键实验结果
在四个推理基准上，d-OPSD 一致优于 RLVR 和 SFT 基线，且样本效率更高——只需约 RLVR 的 10% 优化步数即达到更好效果。「10% 步数」是最亮眼的效率数字，为 dLLM 后训练开了条低成本路径。

### 局限性与开放问题
「优于 RLVR/SFT」的具体分差摘要未逐项列（推断）。用自生成答案当后缀条件，若模型本身答错，「自未来」会否强化错误（自我偏差）未讨论。仅四个推理基准，泛化到开放生成未知。

### 启发与应用前景
它填补了 dLLM 后训练的方法空白，「后缀条件 + 步级监督」是把自回归时代的自蒸馏正确迁移到扩散范式的关键适配。10% 步数的样本效率对 dLLM 这一新兴方向很有吸引力，代码已开源，是想入局扩散语言模型后训练的良好起点。

## 15. Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents
**👍 75** · 🏛 未标注 · https://huggingface.co/papers/2606.06036 · GitHub: https://github.com/Ji-shuo/MRAgent

### 问题与动机
LLM agent 在超长交互历史上的推理仍吃力。当前记忆增强 agent 依赖静态的「先检索-后推理」范式，这种僵化流水线无法根据推理中途发现的证据动态调整记忆访问——一次性检索错了就没有回旋余地。

### 方法与核心创新
提出 MRAgent，把「联想记忆图」与「主动重构机制」结合。记忆表示为 Cue-Tag-Content 图（线索-标签-内容），其中联想标签充当语义桥梁，把细粒度线索连到记忆内容。在此结构上，主动重构机制把 LLM 推理直接嵌入记忆访问：agent 基于累积证据迭代地探索并剪枝检索路径，既让检索动态适配推理上下文，又用剪枝避免无约束扩展导致的组合爆炸。论文标题点题——记忆是「重构」出来的，不是「检索」出来的。

### 关键实验结果
在 LoCoMo 和 LongMemEval 两个长记忆基准上，相比强基线最高提升 23%，同时大幅降低 token 与运行时成本。「+23% 且更省」的双赢说明主动重构不是靠堆算力换来的。

### 局限性与开放问题
+23% 是「最高」值，平均增益与在哪类查询上收益最大未拆分（推断）。Cue-Tag-Content 图的构建成本、标签质量对性能的敏感度未量化。「主动重构」的迭代探索本身会增加推理轮次，其延迟代价与省下的 token 如何权衡未细究。

### 启发与应用前景
「记忆是重构非检索」是对 RAG 式静态检索的范式挑战，把推理嵌进记忆访问的思路对长程对话助手、个人记忆 agent 价值直接。联想图 + 证据驱动剪枝，可迁移到知识图谱问答、多跳检索等需要动态取证的场景。

## 16. DragMesh-2: Physically Plausible Dexterous Hand-Object Interaction with Articulated Objects
**👍 74** · 🏛 北京大学 · https://huggingface.co/papers/2606.15133 · GitHub: https://github.com/AIGeeksGroup/DragMesh-2

### 问题与动机
用灵巧手（多指机械手）操作铰接物体（articulated object，有可活动关节的物体，如抽屉、剪刀、柜门）对家务、辅助、人形机器人很重要。但它不同于静态物体操作：目标部件不能被直接驱动，其运动必须通过持续的「手-把手」物理接触涌现出来。因此从「以物体为中心的铰接生成」到「手驱动的灵巧交互」并不平凡——几何轨迹回放或开环执行都不建模移动关节所需的接触动力学。而且只为固定动力学下任务完成而训的策略会过拟合名义接触载荷，载荷一变就退化。

### 方法与核心创新
提出 DragMesh-2，一个接触驱动的框架，把铰接交互从「以物体为中心的生成」扩展到「手驱动的灵巧手-物交互」，其中铰接运动必须经由物理接触产生。进一步提出 PICA——一种物理知情、接触感知的训练机制，在没有触觉或力反馈的情况下把物理信号注入策略学习，从而在接触载荷变化时提升鲁棒性与成功率。还提供一份纯几何的灵巧交互资源以支撑后续研究。

### 关键实验结果
在七个 GAPartNet 物体上，DragMesh-2 在接触载荷变化下比对比方法更鲁棒，且在不同阻尼条件下保持高任务成功率（摘要未给出具体成功率数字，推断）。系统性地在多种阻尼与铰接类别上评估了载荷变化下的鲁棒性。

### 局限性与开放问题
缺乏具体成功率数字，「更鲁棒」相对哪些基线、领先多少未量化。仅七个 GAPartNet 物体，类别覆盖有限。「无触觉/力反馈」是卖点也是限制——真正精细的接触控制长远可能仍需力感知。

### 启发与应用前景
「无需触觉/力反馈也能注入物理信号提升接触鲁棒性」对硬件受限的灵巧手很实用，与本周具身工作（[7] / [21] / [23]）共同攻「接触密集操作」。纯几何交互资源对 loco-manipulation、人形手-物交互研究是有价值的公共数据。

## 17. Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients
**👍 63** · 🏛 英伟达 · https://huggingface.co/papers/2606.18216 · GitHub 无

### 问题与动机
知识蒸馏把大教师的能力传给小学生，但在「小学生」体制下很脆：强迫学生模仿远大教师的 logits（logits，输出层未归一化的原始分数），会让它集中到教师最尖锐的模式上，损害在训练语料之外基准族的泛化。强化学习（RL）不做 logit 模仿、在学生自身 rollout 上训练，但在「每个 rollout 都失败」的难题上——优势为零、被静默丢弃——若把更强教师的回答注入策略梯度，就破坏了 on-policy 假设、诱发漂移。

### 方法与核心创新
提出 ZPPO（灵感来自维果茨基的「最近发展区」），核心是把教师留在提示里、而非策略梯度里。对难题构造两种改写提示：BCQ（二元候选题）把一个正确教师回答与一个错误学生回答作为匿名候选，逼学生去分辨；NCQ（负候选题）把学生的错误 rollout 聚合成一个提示，暴露它们共同的失败模式。再用「提示回放缓冲」循环每道难题，直到它「毕业」（学生在该题的平均 rollout 准确率过半）或在有限容量下被 FIFO 淘汰，从而在学生当前的「最近发展区」内放大 BCQ 与 NCQ。

### 关键实验结果
在 Qwen3.5 家族四个学生规模（0.8B–9B）、27B 教师、后训练为视觉语言模型、并在 31 个基准（16 VLM、10 LLM、5 视频）上评估：ZPPO 优于离/在线蒸馏和 GRPO（一种用一组采样相对好坏更新策略的 RL 算法），且在最小规模上增益最大。

### 局限性与开放问题
「增益最大在最小规模」也暗示对较大学生收益递减（推断）。具体领先分数未逐项列。构造 BCQ/NCQ + 提示回放增加了数据构造复杂度，其额外成本未量化。「教师留在提示」是否会让学生学会依赖提示中的候选、而非内化能力，值得追问。

### 启发与应用前景
「把教师从梯度搬进提示」是对小模型蒸馏漂移问题的巧妙规避，与 [14] d-OPSD、[6] VibeThinker 共同构成本周「后训练/蒸馏新范式」主线。BCQ/NCQ 的「让学生分辨对错候选」思路，可迁移到任何 teacher-student 且难题上 RL 失效的场景。

## 18. Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages
**👍 60** · 🏛 GigaCode / Yandex School of Data Analysis / Applied AI Institute · https://huggingface.co/papers/2606.20517 · GitHub: https://github.com/Multi-LCB/Multi-LCB

### 问题与动机
LiveCodeBench（LCB）已成代码生成的主流基准，靠不断加新题、按发布日期过滤实现抗污染（contamination-aware，避免模型训练时已见过测试题），提供全面的编码能力视图。但 LCB 只测 Python，留下一个开放问题：LLM 能否泛化到真实软件工程所需的多样编程语言？

### 方法与核心创新
提出 Multi-LCB，覆盖含 Python 在内的 12 种编程语言。它把 LCB 的 Python 任务转换成其他语言的等价任务，同时保留 LCB 的抗污染控制与评测协议；因完全兼容原 LCB 格式，它会自动跟踪未来的 LCB 更新——即随 LCB 长期演进的「活基准」。

### 关键实验结果
评测了 24 个指令/推理 LLM，揭示三类问题：Python 过拟合（模型在 Python 上明显更强）、语言特定的污染、以及显著的多语言性能差距。这把「LCB 只测 Python」这一主要局限直接补上，暴露了当前 LLM 的关键短板（摘要未给出各语言具体分数，推断）。

### 局限性与开放问题
「把 Python 任务翻译成等价任务」本身可能引入翻译偏差——某些语言的等价实现难度并不真等价（推断）。具体各语言排名、差距幅度摘要未列。「语言特定污染」如何量化与去除仍开放。

### 启发与应用前景
「多语言 + 抗污染 + 自动跟踪」使其成为代码模型的实用新基准，直接校正了「Python 高分 = 会写代码」的错觉。它与 [19] / [20] / [24] 共同构成本周「评测祛魅」主线，对多语言代码模型选型和真实软件工程能力评估有直接价值。

## 19. Measuring Epistemic Resilience of LLMs Under Misleading Medical Context
**👍 60** · 🏛 牛津大学 / 华盛顿大学 / 伦敦大学学院 / 滑铁卢大学 · https://huggingface.co/papers/2606.12291 · GitHub: https://github.com/AI-in-Health/MedMisBench

### 问题与动机
LLM 已在医学执照考试上达专家级分数，这让人误以为「高分 = 安全的医疗判断」，而患者越来越多地用它咨询健康。本文证明这一假设很脆：当把误导性上下文注入到 LLM 原本答对的题目里，它们会放弃正确答案。

### 方法与核心创新
定义「认知韧性」（epistemic resilience，在对抗性上下文下维持正确判断的能力），并推出 MedMisBench 来度量它。基准含 10,932 个医学题项、48,889 个误导性「上下文-选项」对，跨医学推理、agentic 能力、患者旅程三类评估——系统性地测「模型知道什么」之外的「误导下还守不守得住」。

### 关键实验结果
跨 11 种模型配置，平均准确率从原题的 71.1% 暴跌到聚焦误导下的 38.0%（近乎腰斩），攻击成功率 51.5%。最具破坏力的是形式化、规则式的编造：权威框定的谎言攻击成功率达 69.5%，例外中毒（伪造例外规则）达 64.1%。来自 7 国的 14 人临床专家组在 38.2% 的复审案例中识别出严重潜在危害。

### 局限性与开放问题
误导上下文由模板生成，真实临床对话里的误导更微妙，攻击成功率可能高估或低估（推断）。「认知韧性」如何在训练中直接优化、而不牺牲对合理新信息的采纳（过度固执也危险），是未解的张力。

### 启发与应用前景
它精准戳破「医考高分 = 安全」的结构性盲区，对医疗 AI 部署是重要安全警示——现有基准测「知道什么」，却不测「误导下守不守得住」。「认知韧性」作为新评测维度，可迁移到法律、金融等任何高风险、易被对抗性上下文操纵的领域。

## 20. GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?
**👍 58** · 🏛 香港中文大学（深圳） / Shenzhen Loop Area Institute / 腾讯 / 北京科技大学 · https://huggingface.co/papers/2606.17861 · GitHub: https://github.com/tongxuluo/gamecraft-bench

### 问题与动机
游戏生成是编码 agent 的新兴应用，要把自然语言规格变成可玩的交互系统。它不同于传统编码任务：发生在游戏引擎里，脚本、场景、素材、渲染、运行时交互必须共同产出连贯的玩法。现有评测缺乏对「引擎内、可玩、可交互验证」的度量。

### 方法与核心创新
把端到端游戏生成形式化为「产出一个通过可观测玩家-游戏交互实现规格的完整游戏物件」。提出三个必要条件：引擎接地（Engine Grounding）、物件完整性（Artifact Completeness）、交互验证（Interactive Verification）。评测框架用回放的演示 + 量规引导的多模态评判来评估可执行玩法，并实例化为 GameCraft-Bench：140 个 Godot（一个开源游戏引擎）任务、覆盖 15 个游戏族。

### 关键实验结果
评测前沿编码 agent 显示端到端游戏生成仍极具挑战：最强 agent 仅得 41.46%，多数低于 40%。进一步分析发现，agent 常能实现可辨认的玩法机制，却难以交付内容充分、视觉反馈可用、呈现连贯的完整游戏。

### 局限性与开放问题
最强 41.46% 说明区分度尚可，但「量规引导的多模态评判」本身的可靠性、与人类玩家判断的一致性未报告（推断）。仅限 Godot 引擎，迁移到 Unity/Unreal 未知。「可玩性」这一主观维度的评分稳定性是开放问题。

### 启发与应用前景
把「可玩、引擎接地、交互验证」立为评测一等公民，是对「生成代码能跑就行」的重要超越，与本周 [18] / [19] / [24] 共同祛魅 agent 真实能力。41.46% 的天花板给「AI 一键做游戏」降温，对游戏 AI、交互式内容生成设定了务实标尺。

## 21. ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining
**👍 54** · 🏛 ACE Robotics / 香港中文大学 / 香港中文大学（深圳） / 上海交通大学 · https://huggingface.co/papers/2606.17200 · GitHub: https://github.com/ACERobotics-VLA/ACE-Ego-0

### 问题与动机
VLA 模型受益于大规模多样的具身数据，但采集机器人轨迹既贵又费人力。第一人称（egocentric，如头戴相机拍摄）人类视频能提供互补的真实世界监督，但人类与机器人数据联合训练很难：动作空间、本体结构、时序动态、监督质量都不一致。

### 方法与核心创新
提出 ACE-EGO-0，一个统一的 VLA 预训练框架，联合利用异构数据源。为从第一人称人类视频提取大规模预训练监督，构建了可扩展的「第一人称视频到动作」流水线，把原始人类视频转成机器人格式的伪动作轨迹。为让这些标签与机器人演示可比，用统一动作表示：相机空间动作、形态条件化、时间对齐的动作分块。为稳健利用有噪的伪动作监督，设计了可靠性感知的训练目标，配一个人类辅助损失，把监督集中到可靠信号上。

### 关键实验结果
在 4.53K 小时机器人+仿真数据 + 1.48K 小时伪动作标注的第一人称人类数据上实例化。实验显示，在可靠性加权下引入大规模人类监督，一致提升统一联合预训练与监督微调。ACE-EGO-0 在 RoboCasa GR1 TableTop 和 RoboTwin 2.0 上达 SOTA，并强迁移到真实世界双臂操作（摘要未给具体分数，推断）。

### 局限性与开放问题
SOTA 具体领先幅度未列。「伪动作」由视频反推，其标签噪声上限决定了监督质量，可靠性加权能补多少未量化。第一人称人类动作与机器人本体的形态差距，在精细操作上是否仍是瓶颈，未讨论。

### 启发与应用前景
「用海量第一人称人类视频 + 可靠性感知加权」补机器人数据稀缺，是 VLA 扩展的务实路线，与 [7] / [22] 共同指向「用非机器人数据喂具身模型」。可靠性感知训练目标可迁移到任何「大量有噪伪标签 + 少量高质标签」的联合训练场景。

## 22. MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction
**👍 53** · 🏛 Allen Institute for AI / 华盛顿大学 / 北卡罗来纳大学教堂山分校 · https://huggingface.co/papers/2606.18558 · GitHub: https://github.com/allenai/molmo-motion

### 问题与动机
运动预测是视觉智能的核心：agent 必须预判物体如何运动才能规划动作、推理物理交互、合成真实未来。本文主张：世界坐标下的 3D 点是一种通用表示——类别无关、视角稳定、紧凑、且对下游任务直接有用，优于类别特定或 2D 的运动表示。

### 方法与核心创新
形式化「目标条件的 3D 点运动预测」：给定短视觉历史、物体上一组 3D 查询点、以及意图目标的语言描述，预测每个点的未来 3D 轨迹。提供一整套栈：(1) MolmoMotion-1M，从 116 万条无约束视频标注的、动作描述且物体接地的 3D 点轨迹大语料；(2) PointMotionBench，人工核验基准，跨 111 个物体类别、61 种运动类型；(3) MolmoMotion 模型，同时支持自回归坐标预测与基于流匹配（flow-matching）的轨迹生成。

### 关键实验结果
MolmoMotion 能按不同语言指令准确预测多样运动模式，在 PointMotionBench 上显著超越现有运动预测基线（摘要未给具体分差，推断）。更重要的是，学到的 3D 运动先验迁移良好：提升机器人操作的训练效率与泛化，其预测轨迹还能作为运动引导，让生成模型合成物体运动更真实的视频。

### 局限性与开放问题
「显著超越」缺具体分数。3D 查询点依赖准确的 3D 标注/深度，来自「无约束视频」的伪 3D 标签精度未量化（推断）。111 类/61 种运动虽广，长尾罕见运动的覆盖仍有限。

### 启发与应用前景
「3D 点轨迹作为通用运动表示」把感知、机器人、视频生成三端用同一先验打通，是 AllenAI 的一项基础设施型工作。语料 + 基准 + 模型全开源，对具身操作、可控视频生成都提供了可复用的运动先验，与 [7] / [21] 共同强化本周「3D/几何驱动具身」主线。

## 23. Playful Agentic Robot Learning
**👍 50** · 🏛 加州大学伯克利分校 / Impossible Research · https://huggingface.co/papers/2606.19419 · GitHub: https://github.com/Playful-RATs/rats

### 问题与动机
当前 agentic 机器人系统能写可执行的「代码即策略」（Code-as-Policy，让 LLM 直接生成控制机器人的代码）程序、观察反馈、跨多次尝试修正，但它们基本是任务驱动的：只有在明确指令下才习得可复用技能。缺的是「在任务到来前的自主探索学习」。

### 方法与核心创新
研究「玩耍式 agentic 机器人学习」——让具身编码 agent 把自主玩耍当作下游任务前的持续技能学习阶段。提出 RATs（机器人 agent 团队）用于玩耍期技能获取：玩耍中，RATs 提出新颖但可学的探索任务，规划并执行机器人代码策略，验证中间进度，诊断失败，用密集的步级反馈重试，并把成功执行蒸馏进一个持久的代码技能库。测试时，agent 从这个冻结的技能库里复用相关技能来解新任务。

### 关键实验结果
在 LIBERO-PRO 和 MolmoSpaces 上，玩耍学到的技能相比无玩耍/随机玩耍基线提升下游任务，相比 CaP-Agent0 分别 +20.6 和 +17.0 个百分点。而且这些技能可直接插进其他推理时的 CaP agent（只需检索进上下文），把 RoboSuite 与真实世界迁移分别提升 8.9 和 8.8 点，无需微调底座模型。

### 局限性与开放问题
「自主提出可学任务」的质量依赖底座 LLM，玩耍可能在无价值任务上空耗算力（探索效率未量化，推断）。技能库随玩耍增长会否膨胀、冲突、检索变慢，未讨论。玩耍成本 vs 下游收益的性价比未细算。

### 启发与应用前景
「先玩耍、后任务」把开放式探索引入具身 agent，且技能以代码形式持久化、可插拔复用，呼应 [8]「Workspace + Skill」愿景。技能库无需微调即可跨 agent 迁移，对搭建可积累的机器人技能生态很有启发，与 [7] / [16] / [21] 共同丰富本周具身主线。

## 24. Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games
**👍 50** · 🏛 复旦大学 / 上海创智学院 / 上海人工智能实验室 / 浙江大学 · https://huggingface.co/papers/2606.19338 · GitHub: https://github.com/InternLM/RNGBench

### 问题与动机
把多模态基础模型部署为闭环策略，越来越需要「基于已不可见的观测来决定动作」（即非马尔可夫，Non-Markov——当前决策不能只看当前画面、必须依赖历史信息）。但现有基准要么暴露完整状态、要么把「隐藏状态重构」与其他技能混为一谈、要么只在一局结束后测回忆。

### 方法与核心创新
提出 RNG-Bench（Reconstructive Non-Markov Games），专门隔离「在多步交互中重构过去观测并据此行动」的能力。含两个互补游戏：Matching Pairs（卡牌身份在特定位置短暂显示、之后须回忆）和 3D Maze（第一人称视角须整合成空间地图）。统一测评框架下有三条可控难度轴：网格大小、视觉模式、观测模态。还引入「一对一对决协议」控制实例级方差，以及「记忆差距」（Memory Gap）指标，把「遗忘」从「决策差」中拆开。

### 关键实验结果
最难配置每局需约 128K token 上下文和 350 张图像输入，前沿 MLLM 远未饱和。记忆差距分析显示：多数残余错误源于遗忘更早的观测，而非决策次优——这把失败精确归因到「记不住」而非「不会想」。此外，在最优策略 rollout + 过滤后的模型演示上微调 Qwen3.5-9B，可提升 RNG-Bench 表现并迁移到已有基准，且不损害通用多模态能力。

### 局限性与开放问题
两个游戏（配对、迷宫）是否覆盖真实非马尔可夫任务的多样性存疑（推断）。128K token/350 图的极端配置对评测成本要求高。「记忆差距」把失败归因于遗忘，但如何在架构上根治长时记忆仍是开放问题。

### 启发与应用前景
「隔离记忆重构能力 + 用记忆差距归因」是对多模态长时记忆的精细评测工具，与 [15] MRAgent（记忆是重构非检索）从评测与方法两端呼应「记忆」主题。它对把 MLLM 部署为闭环具身策略是重要诊断——先量化「记不住」的短板，再谈决策，与本周 [18] / [19] / [20] 共同祛魅 agent 真实能力。

---

## 🗺️ 趋势洞察

### 1. 「循环 / 迭代潜在深度」成为独立于规模与数据的第三条 scaling 轴
**涉及论文**：[1], [2]
**核心观点**：本周点赞第一（[1], 477）与并列第二（[2], 209）都在做「循环」——反复调用同一组共享参数的块，用迭代深度换算力效率，而非堆更多层或更多数据。LoopWM（[1]）把它用于世界模型，宣称最高 100 倍参数效率、并靠自适应计算按预测难度动态决定循环次数；LoopCoder-v2（[2]）把它用于 7B 编码模型，把 SWE-bench Verified 从 43.0 拉到 64.4。两篇共同把「迭代潜在深度」立为一条正交的新扩展轴。但它们对「循环几次最优」给出相反直觉——[1] 主张自适应、越多越省，[2] 却发现两循环即饱和、之后退化。这条轴是否普适、饱和点由什么决定，是本周最值得追踪的开放问题。

### 2. 「效率优先」：小模型 / 极致压缩逼近甚至超越巨头
**涉及论文**：[1], [4], [6], [11], [12]
**核心观点**：本周一条密集主线是「用远小的模型/远低的成本追平大模型」。Moebius（[4]）用不到 2% 参数（0.22B vs 11.9B）媲美 10B 级 FLUX、推理快 >15 倍；VibeThinker-3B（[6]）以 3B 在 AIME26 达 94.3、匹配甚至超越大几个数量级的 Gemini 3 Pro / DeepSeek V3.2；LoopWM（[1]）宣称 100 倍参数效率；FastContext（[11]）用 4B–30B 专用探索模型把编码 agent 的 token 消耗砍 60%；Ling/Ring 2.6（[12]）在万亿参数上主打「单位输出 token 能力 + 部署效率」。VibeThinker 的「压缩-覆盖假说」给出了理论解释：可验证推理可压进紧凑核，知识才需大参数。共同信号：2026 年「效率」不再只是部署妥协，而是逼近前沿的独立路径。

### 3. 具身智能全栈推进：从 2D 感知走向 3D 几何、接触动力学、人类视频与自主玩耍
**涉及论文**：[7], [16], [21], [22], [23]
**核心观点**：本周有五篇具身工作，共同信号是「让机器人策略真正吃进 3D 与物理」。GAM（[7]）直接复用几何基础模型当感知-预测-动作统一底座，补上 VLA 长期隐式的 3D 几何；DragMesh-2（[16]）攻接触密集的铰接物体操作、无触觉反馈也注入物理信号；ACE-Ego-0（[21]）用 1.48K 小时第一人称人类视频 + 可靠性加权补机器人数据稀缺；MolmoMotion（[22]）把「3D 点轨迹」立为跨感知/操作/视频生成的通用运动表示；RATs（[23]）让 agent「先自主玩耍、后解任务」，把成功蒸馏进可复用代码技能库、+20.6 点。从几何底座、接触动力学、数据来源到技能获取方式，具身智能正在多个层面同时被重写。

### 4. 评测集体给能力「祛魅」：多语言、医疗鲁棒性、游戏生成、长时记忆
**涉及论文**：[5], [18], [19], [20], [24]
**核心观点**：延续上周（W24）的评测反思，本周多篇基准继续戳破「高分 = 能干活」。Multi-LCB（[18]）把 LCB 扩到 12 种语言，揭示「Python 高分」掩盖的过拟合与多语言差距；MedMisBench（[19]）证明医考专家级模型在误导上下文下准确率从 71.1% 暴跌到 38.0%、攻击成功率 51.5%，戳破「医考高分 = 安全」；GameCraft-Bench（[20]）显示最强 agent 端到端做游戏仅 41.46%；RNG-Bench（[24]）用「记忆差距」把 MLLM 的失败精确归因为「记不住」而非「不会想」；Data2Story（[5]）则把「每个论断回链证据」的可验证性做成评测轴。共同信号：评测正从「分数排行」转向「抗误导、抗污染、可归因、可验证」的真实能力诊断。

### 对比与张力
- **循环「越深越省」 vs 「两次即饱和」**：[1] LoopWM 主张自适应循环、深度可换 100 倍效率，[2] LoopCoder-v2 却实证循环收益强烈非单调、三次以上退化。同一「循环」范式在世界模型与编码任务上给出相反的深度-收益曲线，饱和机制尚无统一解释。
- **能力吹捧 vs 评测祛魅**：一边是 [6] VibeThinker-3B 宣称 3B 匹配 Gemini 3 Pro、[1] 宣称 100 倍效率的乐观叙事，另一边是 [19] / [20] / [24] 用暴跌的准确率、41% 的天花板、「记不住」的归因给能力降温。同一周里「小模型追平巨头」与「大模型其实很脆」并存，提醒读者对单一基准的高分保持警惕。
- **记忆：重构 vs 检索**：[15] MRAgent 主张「记忆是重构出来的、不是检索出来的」，把推理嵌进记忆访问；[24] RNG-Bench 则从评测端量化「遗忘」才是 MLLM 失败主因。方法端与评测端共同把「长时记忆」推成 agent 的核心瓶颈。
- **世界模型 / 生成 vs 实时交互**：[1] / [9] 追求可控长时程的世界模拟与生成质量，[3] JoyAI 则追求「每秒自主决定说不说」的实时在场。前者重「模拟得像」，后者重「反应得及时」，是具身/交互智能两种正交的价值取向。

### 值得关注的研究方向
1. **循环深度作为可控 scaling 轴**：[1] / [2] 把「迭代潜在深度」立为新维度，但饱和点、自适应循环的稳定收敛、以及它与位置编码的耦合（[2] 的 CLP 错配）都待厘清。搞清「循环几次最优、由什么决定」可能带来普惠性效率增益。
2. **具身智能的统一表示**：[22] 的「3D 点轨迹」、[7] 的「几何基础模型底座」、[21] 的「相机空间动作」都在找跨感知-预测-动作的统一表示。一个类别无关、视角稳定、能同时喂操作与视频生成的表示，可能成为下一代 VLA 的公共基座。
3. **后训练里的「自监督 / 自未来」**：[14] d-OPSD 用「自身未来经验」、[17] ZPPO 把教师留在提示、[6] VibeThinker 用离线自蒸馏，都在绕开对大教师梯度或稀缺标注的依赖。但它们共享「模型用自己的判断改进自己」的自我偏差风险，何时强化正确、何时放大错误，亟需可验证的边界。
4. **抗误导 / 可归因的评测基础设施**：[19] 的认知韧性、[24] 的记忆差距、[18] 的多语言抗污染，都在把评测从「测知道什么」推向「测守不守得住、记不记得住、跨语言稳不稳」。把这些诊断维度标准化，是可信 agent 落地的前提。
5. **「玩耍 / 探索」驱动的技能积累**：[23] RATs 的「先玩耍后任务 + 持久代码技能库」，与 [8] 的「Workspace + Skill」愿景呼应。一个可积累、可跨 agent 复用、无需微调的技能抽象，可能是让具身 agent 走向「数字同事」的关键组件。
