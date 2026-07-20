# HuggingFace 周榜论文深度总结 — 2026 第 29 周

> 来源：https://huggingface.co/papers/week/2026-W29
> 统计日期：2026-07-20
> 筛选条件：upvotes ≥ 30（本周上榜 101 篇，取 32 篇）
> 论文数：32

## 目录
1. [Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable](#1-harness-handbook-making-evolving-agent-harnesses-readable-navigable-and-editable) 👍198
2. [LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget](#2-longstraw-long-context-rl-beyond-2m-tokens-under-a-fixed-gpu-budget) 👍177
3. [VideoChat3: Fully Open Video MLLM for Efficient and Generalist Video Understanding](#3-videochat3-fully-open-video-mllm-for-efficient-and-generalist-video-understanding) 👍148
4. [Weak-to-Strong Generalization via Direct On-Policy Distillation](#4-weak-to-strong-generalization-via-direct-on-policy-distillation) 👍130
5. [Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation](#5-boogu-image-01-boosting-open-source-unified-multimodal-understanding-and-generation) 👍126
6. [Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models](#6-function-aware-fill-in-the-middle-as-mid-training-for-coding-agent-foundation-models) 👍104
7. [ABot-N1: Toward a General Visual Language Navigation Foundation Model](#7-abot-n1-toward-a-general-visual-language-navigation-foundation-model) 👍100
8. [Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning](#8-ring-zero-scaling-zero-rl-to-a-trillion-parameters-for-emergent-reasoning) 👍91
9. [SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning](#9-seed-self-evolving-on-policy-distillation-for-agentic-reinforcement-learning) 👍88
10. [Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation](#10-search-beyond-what-can-be-taught-evolving-the-knowledge-boundary-in-agentic-visual-generation) 👍85
11. [ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory](#11-abot-agentos-a-general-robotic-agent-os-with-lifelong-multi-modal-memory) 👍83
12. [Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation](#12-read-it-back-pretrained-mllms-are-zero-shot-reward-models-for-text-to-image-generation) 👍80
13. [Video Generation Models are General-Purpose Vision Learners](#13-video-generation-models-are-general-purpose-vision-learners) 👍80
14. [Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading](#14-long-horizon-terminal-bench-testing-the-limits-of-agents-on-long-horizon-terminal-tasks-with-dense-reward-based-grading) 👍74
15. [SynthDocBench: Controlled Benchmark for Long-Context Visual Document Understanding](#15-synthdocbench-controlled-benchmark-for-long-context-visual-document-understanding) 👍69
16. [SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration](#16-searchos-v1-towards-robust-open-domain-information-seeking-agent-collaboration) 👍60
17. [KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill](#17-knowact-guiclaw-know-deeply-act-perfectly-personal-gui-assistant-with-self-evolving-memory-and-skill) 👍56
18. [Scalable Visual Pretraining for Language Intelligence](#18-scalable-visual-pretraining-for-language-intelligence) 👍55
19. [OvisOCR2 Technical Report](#19-ovisocr2-technical-report) 👍52
20. [4D Human-Scene Reconstruction from Low-Overlap Captures](#20-4d-human-scene-reconstruction-from-low-overlap-captures) 👍52
21. [BadWAM: When World-Action Models Dream Right but Act Wrong](#21-badwam-when-world-action-models-dream-right-but-act-wrong) 👍47
22. [LightMem-Ego: Your AI Memory for Everyday Life](#22-lightmem-ego-your-ai-memory-for-everyday-life) 👍46
23. [Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model](#23-xiaomi-robotics-u0-unified-embodied-synthesis-with-world-foundation-model) 👍41
24. [AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities](#24-agentcompass-a-unified-evaluation-infrastructure-for-agent-capabilities) 👍38
25. [KeyFrame-Compass: Towards Comprehensive Evaluation of Keyframe-Conditioned Video Generation](#25-keyframe-compass-towards-comprehensive-evaluation-of-keyframe-conditioned-video-generation) 👍36
26. [PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails](#26-policyshiftguard-benchmarking-and-improving-policy-adaptive-image-guardrails) 👍35
27. [MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors](#27-metaview-monocular-novel-view-synthesis-with-scale-aware-implicit-geometry-priors) 👍34
28. [Trust Region Policy Distillation](#28-trust-region-policy-distillation) 👍33
29. [AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification](#29-advancedmathbench-a-benchmark-suite-for-advanced-mathematical-proof-generation-and-verification) 👍32
30. [KronQ: LLM Quantization via Kronecker-Factored Hessian](#30-kronq-llm-quantization-via-kronecker-factored-hessian) 👍32
31. [MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation](#31-multiref-compass-towards-comprehensive-evaluation-of-multi-reference-to-audio-video-generation) 👍31
32. [Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models](#32-blind-spots-bench-evaluating-blind-spots-in-multimodal-models) 👍30

---

## 1. Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable
**👍 198** · 🏛 印第安纳大学 / 腾讯 · [arXiv](https://arxiv.org/abs/2607.13285)

### 问题与动机
一个 AI agent 的能力，一半在基础模型，另一半在「harness」——负责拼 prompt、管状态、调工具、编排执行流程的那层胶水代码。模型换代、API 改版、需求变更，harness 就得跟着改。但改之前得先找到「实现这个行为的代码在哪」，而这恰恰最难：需求是用行为描述的（「让 agent 超时后别重试那么多次」），代码却是按文件和模块组织的。

举个例子：你要改「agent 超时后重试几次」，逻辑可能散在配置类的默认值、调度循环的 while 条件、异常处理分支这三个地方，grep 「retry」会漏掉那个叫 `max_attempts` 的字段，也会淹没在几十个无关命中里。论文把这个叫「行为到代码的鸿沟」，并认为它才是 harness 演进的真正瓶颈——不是不会写补丁，是不知道往哪打。

### 方法与核心创新
两件东西。一是 Harness Handbook：用静态分析扫出调用关系和数据流，再让 LLM 把它结构化成一份「以行为为索引」的手册，每条行为条目直接挂上对应的源码位置——相当于给代码库配一本按功能查的说明书，而不是按目录查的文件树。二是 BGPD（行为引导的渐进披露）：不把整本手册塞进上下文，而是让 agent 从高层行为逐级下钻到具体实现，且每一步都拿当前源码去校验候选位置是否还成立（防止手册过期误导）。

### 关键实验结果
在两个开源 harness 的多样化修改请求上，Handbook 辅助的规划同时提升了行为定位准确率和编辑方案质量，且**planner 消耗的 token 更少**——通常这两者是此消彼长的，能同时改善说明省下的是无效检索。收益最大的三类场景很有代表性：代码点分散、极少执行的冷路径、跨模块交互。摘要未披露具体百分点数字。

### 局限性与开放问题
论文自己承认评测只覆盖两个开源 harness，泛化性存疑。我的观察：手册是 LLM 合成的，一旦代码演进而手册不同步，就会变成有权威感的错误地图——BGPD 的源码校验是补丁但不是根治；另外静态分析对反射、动态注册、配置驱动的调用天然失效，而这些恰是 harness 代码的重灾区。

### 启发与应用前景
这篇的价值不在具体方法，而在把「找位置」从「生成补丁」里独立出来当成一等公民问题。任何维护大型 agent 系统的团队都可以先做一件低成本的事：给自己的 harness 手写一份行为索引挂到 CLAUDE.md 或 AGENTS.md，效果可能就有一半。项目页：https://ruhan-wang.github.io/Harness-Handbook/

---

## 2. LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget
**👍 177** · 🏛 MindLab / 复旦大学 · [arXiv](https://arxiv.org/abs/2607.14952) · [GitHub](https://github.com/MindLab-Research/longstraw)

### 问题与动机
推理侧已经在冲百万 token 上下文，RL 后训练却还卡在 256K 以下，靠「长度泛化」硬扛部署时的长输入。这个错配对 agent 尤其致命——agent 一条轨迹里堆满了观察、工具输出、文档、历史决策，训练时从没见过这么长的输入，上线就得赌运气。瓶颈是显存：训练要存整张计算图做反向传播，长度一上去直接 OOM。

### 方法与核心创新
LongStraw 是一套「架构感知的执行栈」，在 GRPO（组相对策略优化，一个 prompt 采样一组回答、组内比较好坏来算优势）上落地，核心是三个动作：共享 prompt 的前向**不开 autograd**（不记录梯度图），只保留后续 token 真正需要的模型特定状态（比如 KV、recurrent state），然后把组内的多条短回答**一条一条 replay** 做反向。

举个例子：一个 2M token 的 prompt 配 8 条各 2K 的回答，朴素做法要为 8×(2M+2K) 的完整图留显存；LongStraw 把那 2M 的公共部分算完就「掐断」梯度，只在 2K 的分支上重建图——用重算时间换显存，本质是 gradient checkpointing 思路针对 RL 分组结构的特化。

### 关键实验结果
8 张 H20 上完成了 Qwen3.6-27B 在 **2.1M positions** 上的分组打分与反向；关键数字是**组大小从 2 增到 8，峰值显存只多 0.21 GB**——几乎与组大小解耦，这正是逐条 replay 的直接收益。压力测试触到 4.46M positions。32 卡 H20 上跑通了 GLM-5.2 全部 78 层、2.1M token prompt 的端到端路径。

### 局限性与开放问题
论文相当坦诚：这些实验证明的是**执行容量而非训练正确性**——捕获的 prompt 状态是 detached 的，部分分布式前向和梯度组合路径尚未完成。换句话说这是一份工程可行性报告，不是「我们训出了更好的模型」。我的观察：用时间换显存的代价没有量化披露，replay 一条条串行做，组大小大时吞吐可能塌得很难看。

### 启发与应用前景
对做长上下文 agent RL 的团队，这套思路可以直接借鉴到自己的训练框架里，尤其「共享前缀不建图 + 分支单独 replay」这一条几乎是免费的显存优化。

---

## 3. VideoChat3: Fully Open Video MLLM for Efficient and Generalist Video Understanding
**👍 148** · 🏛 南京大学 / 上海人工智能实验室 / 南洋理工大学 / 北京大学 · [arXiv](https://arxiv.org/abs/2607.14935) · [GitHub](https://github.com/MCG-NJU/VideoChat3)

### 问题与动机
开源视频多模态模型有三个老毛病：只在特定领域好用（会看短动作的不会看两小时长片）、算力开销大到没法规模化、以及「半开源」——放个权重但训练代码、数据配方、训练策略全不给，别人复现不了也改进不了。VideoChat3 要同时解掉这三个。

### 方法与核心创新
效率侧两招。I3D-ViT（Inflated 3D ViT）把预训练好的 2D 图像 ViT 沿时间轴「膨胀」成 3D，直接继承图像模型的视觉先验，省掉从头训视频编码器的代价，同时能建模时空关系。Adaptive Frame Resolution 则按需分配分辨率——举个例子：一段监控视频里 90% 的帧是静止走廊，10% 是有人闯入，前者压到低分辨率糊着看就够，把省下的 token 预算全给关键帧，而不是像固定采样那样每帧一视同仁地烧算力。

效果侧则是一条可规模化的视频数据合成流水线，产出三个数据集分别对应三类场景：VideoChat3-Academic2M（通用，200 万）、LV116K（长视频，11.6 万）、OL617K（流式在线，61.7 万）。三个数据集全部开放，这也是「fully open」的实质内容。

### 关键实验结果
仅 **4B 参数**，在通用、长视频、流式三类 benchmark 上全面超过参数量相当甚至更大的开源模型，且推理效率更高。摘要未披露逐项 benchmark 的具体分数，只给了「同参数量或更大参数量对手全胜」这个定性结论——这是本文报告上的明显短板。

### 局限性与开放问题
论文承认的局限着墨不多。我的观察：训练数据由合成流水线产出，合成数据的分布偏差会不会导致在真实长尾视频上退化，摘要没有交代；另外「流式」场景的延迟指标（首 token 时间、每秒可处理帧数）完全缺席，而这恰恰是流式理解能否落地的决定性指标。

### 启发与应用前景
4B 这个尺寸意味着单卡消费级 GPU 就能跑，对做视频审核、监控分析、剪辑辅助的团队是很现实的起点。全套开放（代码+策略+数据集）的姿态在当前视频模型领域相当罕见，二次开发成本远低于同类。项目页：https://mcg-nju.github.io/VideoChat3

---

## 4. Weak-to-Strong Generalization via Direct On-Policy Distillation
**👍 130** · 🏛 清华大学 / 字节跳动 / 北京大学 · [arXiv](https://arxiv.org/abs/2607.05394)

### 问题与动机
RLVR（用可验证奖励做强化学习，比如数学题答案对了给 1 分错了给 0 分）确实能提升推理能力，但每出一个新的大模型就得重跑一遍，而 RL 训练要求目标模型自己生成海量 rollout——模型越大越烧钱，后训练本身成了瓶颈。能不能在小模型上跑 RL（便宜），然后把学到的东西搬给大模型？

直接蒸馏跑完 RL 的小老师不行，因为老师的最终策略是「RL 带来的增益」和「小模型自身的能力天花板」的混合物，学生照单全收会把老师的短板一起继承。

### 方法与核心创新
Direct-OPD 的关键洞察：不要学老师的策略，要学**老师被 RL 改变的那个「差量」**。具体做法是拿 RL 后的老师和它自己 RL 前的参考模型对比，把两者在同一动作上的对数概率之比当作一个稠密的隐式奖励信号。

举个例子：小模型在做数学题时，RL 训练后「先列方程再代数」这一步的概率从 0.2 涨到 0.6，而「直接猜答案」从 0.3 掉到 0.05——这个升降模式就是 RL 教会它的东西，与小模型算不算得对无关。Direct-OPD 把这个升降信号施加在**学生自己生成的状态上**（on-policy，即学生自己走一遍，老师在学生走过的路上打分纠偏，而不是让学生背老师的标准答案），从而绕开了在大模型上跑稀疏奖励 RL 的高昂代价。

### 关键实验结果
最亮眼的一组：Qwen3-1.7B 在 AIME 2024 上从 **48.3% 提到 58.3%（+10 个百分点）**，代价只有 **8 张 A100 跑 4 小时**——对比直接在目标模型上跑 RL，在相同训练步数下 Direct-OPD 更优。另外多个策略偏移可以**串行叠加**，意味着不同 RL 运行的成果能像补丁一样累积复用。

### 局限性与开放问题
我的观察有两点风险：一是老师和学生必须共享同一套 tokenizer 和大致的动作空间，跨模型族迁移能不能成立摘要没提；二是「RL 增益」和「小模型局限」在对数比里未必真的可分离——如果小模型是靠某种只对小模型有效的捷径拿到奖励的，这个信号传给大模型就是噪声。论文自身承认的局限在摘要中未展开。

### 启发与应用前景
这个思路把 RL 的产出从「一个最终模型」重新定义成「一份可复用的奖励信号」，对算力受限但想跟进推理能力的团队非常实用：在 1.7B 上做实验找配方，把 checkpoint 对保存下来，直接施加到 32B 上。项目页：https://bytedtsinghua-sia.github.io/Direct-OPD/

---

## 5. Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation
**👍 126** · 🏛 未标注 · [arXiv](https://arxiv.org/abs/2607.13125) · [GitHub](https://github.com/boogu-project/Boogu-Image)

### 问题与动机
Nano-Banana-Pro、GPT-Image-2 这类闭源多模态系统效果好，但它们的强并非来自单个模型——而是系统级集成（多阶段流水线、重写 prompt、多轮筛选等），而这套工程实践对外完全不公开。开源社区看到的只是最终效果，学不到怎么做。这篇要回答的是：在算力预算被死死卡住的前提下，靠什么能追上去。

### 方法与核心创新
Boogu-Image-0.1 是一个统一「理解+生成」的模型家族，四个变体分工明确：Base（高质量文生图）、Turbo（快速推理）、Edit（指令式编辑）、Edit-Turbo。论文强调突破点不在某个新架构，而在三处叠加：模型的理解能力、数据质量、训练流水线，再加上**agentic 推理时扩展**——即推理时不是一发定生死，而是让模型像 agent 一样多轮生成、自评、修正。

举个例子：用户输入「一只戴墨镜的柴犬站在霓虹招牌下，招牌写『营业中』」，单次生成常把中文字渲染成鬼画符；agentic 推理会先生成、再让理解侧模型检查招牌文字是否正确、不对就带着「文字错误」的反馈重生成——把生成模型的弱项交给它自己的理解能力去兜底，这是「统一模型」相比纯生成模型的结构性优势。中英双语文字渲染是它主打的能力之一。

### 关键实验结果
两个成本数字是本文的真正卖点：全程只用了 **2.0862 亿张去重图片**，base 模型的**理论训练成本约 40 万美元**——对照动辄十亿级图片、千万美元级投入的闭源系统，这是一到两个数量级的差距。效果上在标准 benchmark 上持平或超过其他开源模型，并逼近领先闭源系统。摘要未披露具体的 benchmark 分数。

### 局限性与开放问题
论文自己承认的是「高度受限的算力预算」这个前提。我的观察：「approaching leading closed-source systems」是全文最模糊的表述，没有任何一个具体指标支撑；另外 40 万美元是「理论训练成本」（大概按 GPU 小时折算的理想值），不含数据清洗、失败实验、调参的实际开销，真实复现成本会显著更高。agentic 推理扩展也意味着推理侧延迟和成本上升，这部分代价未量化。

### 启发与应用前景
Apache 2.0 放出权重、代码和配方，对想自建图像生成能力的团队是可直接商用的基座，尤其中英双语文字渲染在国内电商/海报场景是刚需。「用理解模型给生成模型做闭环反馈」这个模式也值得单独抽出来用在现有 pipeline 上。项目页：https://boogu.org/

---

## 6. Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models
**👍 104** · 🏛 滑铁卢大学 / 不列颠哥伦比亚大学 / 英伟达 / Verdent AI · [arXiv](https://arxiv.org/abs/2607.12463) · [GitHub](https://github.com/TIGER-AI-Lab/FIM-Midtraining)

### 问题与动机
coding agent 干活的循环是「发起动作 → 拿到工具返回 → 把返回值接进后续推理」。但标准的代码预训练是纯左到右的下一 token 预测，模型只学过「往前写」，从没被专门训练过「中间被塞进一段我算不出来的结果，我该怎么接着往下写」。这个能力缺口一直是靠后训练硬补的。

### 方法与核心创新
论文的核心洞察相当漂亮：**agent 的「动作-观察-继续」循环，和代码里的一个函数调用点在结构上是同构的**——调用方绑定参数（发起动作），被调函数在别处算出返回值（工具执行），下游代码消费这个值（接进推理）。而这种结构在互联网规模的普通代码里遍地都是，等于免费的训练信号。

于是做 function-aware FIM（函数感知的填空）中训练：把函数体挖掉让模型补，但不是随便挖——先用程序依赖图分析选出真正参与数据流的函数，再用「复杂度-可推断性」双重标准筛（太简单的挖了没信息量，太难的仅凭上下文根本推不出来，挖了只会教模型瞎猜）。举个例子：一个 `parse_config(path)` 的调用，上下文里下游用了 `cfg.timeout` 和 `cfg.retries`，模型就能从消费方反推返回结构——这正是 agent 看到工具返回 JSON 后要做的事。

### 关键实验结果
在 968 个 GitHub 仓库、**26 亿 token** 的去污染语料上做中训练。SWE-Bench-Verified：7B **+2.8**、14B **+3.0**、Qwen3-8B **+3.2**；SWE-Bench-Lite 更大，分别 **+3.7 / +4.0 / +5.4**。更值得注意的是跨管线成立（R2E-Gym 和 SWE-Smith 两条后训练流水线都涨），且在非 Qwen2.5 基座上也成立。

另一个意外收获：中训练缓解了 agentic 后训练带来的**能力侵蚀**——通常把模型往 agent 方向调，非 agent 编码（LiveCodeBench）和通用工具调用（tau-bench、BFCL）会退化，加了这层中训练反而稳住甚至提升。而且语料只有 Python，这个「函数调用归纳偏置」竟然迁移到了非 Python、非编码任务上。

### 局限性与开放问题
论文承认语料仅限 Python。我的观察：+3 个点在 SWE-Bench 上算扎实但不算颠覆，而 26 亿 token 的中训练不是零成本；另外「可推断性」筛选依赖启发式阈值，这个筛选器本身的敏感性没有消融交代。

### 启发与应用前景
最实用的一点是：这是一个纯自监督目标，不需要任何人工标注或轨迹采集，任何有代码语料的团队都能加进自己的训练流程。对「agent 能力是不是必须靠昂贵的轨迹数据来教」这个默认假设，本文给了一个反例。模型集合已发布在 HuggingFace。

---

## 7. ABot-N1: Toward a General Visual Language Navigation Foundation Model
**👍 100** · 🏛 高德地图 / 阿里巴巴 · [arXiv](https://arxiv.org/abs/2607.10383)

### 问题与动机
视觉语言导航（给机器人一句自然语言指令让它走到目标）现在主流是「单体策略」——观测直接映射到动作的黑箱。问题有三：坐标漂移（模型输出的世界坐标误差会累积）、长尾语义处理差（没见过的地标就懵）、以及完全不可解释（走错了没法 debug）。

### 方法与核心创新
ABot-N1 用「慢-快」双层架构把认知和控制拆开。慢层是个视觉语言推理器，显式做思维链推理，输出不是坐标也不是动作，而是**像素目标**——直接在当前摄像头画面上标出一组锚点。快层是个动作专家，同时吃文本线索和这些像素锚点，以原生控制频率吐出连续路点。

像素目标是全文的题眼。举个例子：指令是「去街对面那家亮着绿灯牌的药店」，传统做法要先估算药店的世界坐标（相机标定一偏、走两步误差就滚起来），ABot-N1 直接在图像上圈出那块绿灯牌所在的像素点——图像空间是天然对齐的，不需要坐标系转换，也不会漂移。而且这套「在图上打点」的接口对五类任务通用：point-goal（走到某点）、object-goal（走到某物体）、poi-goal（走到某商户）、指令跟随、跟人走。顺带把中间推理过程暴露出来，走错了能看到它当时在想什么。

### 关键实验结果
城市尺度导航的提升是断崖式的：**POI 到达率提升 35.0 个百分点，达到 77.3%**——考虑到基线只有 42% 出头，这几乎是把一个不可用的系统做成了可用。复杂室内、室外场景的成功率分别为 **95.4% 和 92.9%**。在物体抵达、跟人、指令跟随上也保持优势。团队同时开源了新的 Point-Goal / POI-Goal benchmark。

### 局限性与开放问题
论文承认这只是「迈向」通用导航基础模型的一步。我的观察：像素目标的软肋是目标必须在当前视野内——绕过拐角、目标被遮挡时锚点无处可打，摘要没交代这种情况的回退策略；另外慢层做思维链推理天然慢，慢快两层的频率如何协调、慢层延迟对实时性的影响，是落地的关键却未量化。

### 启发与应用前景
出自高德，导航和 POI 数据是它的主场，这套东西离配送机器人、园区巡检的实际部署很近。「用图像空间锚点而非世界坐标做高低层接口」这个设计对任何视觉驱动的机器人控制都有借鉴价值。项目页：https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/

---

## 8. Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning
**👍 91** · 🏛 中国人民大学 / 蚂蚁集团 / 清华大学 / 浙江大学 · [arXiv](https://arxiv.org/abs/2607.12395)

### 问题与动机
zero RL——完全不用人工标注数据，只靠可自动验证的奖励（数学题对错）从基座模型直接激发思维链——已经被证明有效，但因为算力限制，几乎所有研究都停在小模型上。1T 参数规模下训练动态是什么样、会不会涌现出小模型没有的能力，是一片空白。而朴素放大规模会撞上三个问题：输出可读性差、token 冗余、推理深度不会自适应（简单题也长篇大论）。

### 方法与核心创新
论文给的是一整套稳定高效的训练流水线，三个关键工程点：clipped importance sampling（裁剪重要性采样，限制训练策略和采样策略偏离过大导致的梯度爆炸）、training-inference ratio correction（修正训练框架和推理框架数值不一致造成的偏差——大模型上这个 gap 会被放大成训练崩溃）、以及混合精度控制。

举个例子：1T 模型的 rollout 通常用 vLLM 之类的推理引擎生成，而梯度更新在训练框架里算，两边的算子实现、精度、kernel 都不同，同一条序列算出的概率会有微小差异。小模型上这点误差可以忽略，1T 规模上它会被重要性采样比值指数放大，直接把训练带偏——这就是第二项修正在解决的事。

### 关键实验结果
三条结论直接支持「苦涩的教训」：[1] 规模到 1T 显著提升样本效率和性能天花板；[2] 训练过程分两个阶段——先是「发现期」（模型在探索各种推理模式），然后是「锐化期」（收敛到有效模式并强化）；[3] 模型自发涌现出高级认知行为，包括拟人化表达、结构化排版、自我验证、并行推理，以及一个很有意思的「上下文焦虑」（context anxiety，模型意识到自己快超长了而调整策略）——这些行为让手工设计的推理启发式规则变得多余。

Ring-2.5-1T-Zero 在七个数学 benchmark 上取得有竞争力的成绩（摘要未披露具体分数）。论文还额外提了一个从可理解性、可复现性、效率三个维度评估思维链质量的框架——因为只看最终答案对错，无法区分「想得清楚」和「蒙对了」。

### 局限性与开放问题
论文承认朴素放大会有可读性和冗余问题，其方案是缓解而非根治。我的观察：全文最大的缺口是缺具体分数——「competitive performance」在 1T 规模上是个相当保守的说法，很可能意味着并未大幅超越更小的对手，那么 1T 的性价比就是个真问题；另外「涌现的认知行为」多为定性观察，没有量化的出现频率或与性能的相关性分析，容易过度解读。

### 启发与应用前景
最值钱的不是模型而是那三条训练动态观察，尤其「发现期→锐化期」的两阶段规律，可以直接指导中小规模 RL 训练的超参调度（发现期给高探索、锐化期收紧）。训练-推理数值一致性这个坑，任何做大规模 RL 的团队都会踩，这篇给了明确的解法方向。

---

## 9. SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning
**👍 88** · 🏛 清华大学 / 浙江大学 / 香港中文大学 / 南洋理工大学 · [arXiv](https://arxiv.org/abs/2607.14777) · [GitHub](https://github.com/jinyangwu/SEED)

### 问题与动机
把大模型训成能跑多轮任务的 agent，主流做法是结果导向的强化学习（RL）：一整局跑完，成功给 1 分、失败给 0 分。问题在于一局可能有二三十步，最后一个 0 分没法告诉模型是第 3 步走错了房间还是第 17 步忘了开抽屉。这就是论文说的「监督鸿沟」——奖励在「整局」这个粒度，而参数更新需要落到「每个 token」这个粒度。

### 方法与核心创新
SEED 的做法是让模型自己给自己写复盘笔记，再把笔记的效果蒸馏回参数里。具体两步：第一，同一个模型既当演员（跑任务）又当分析师（跑完读自己的轨迹，用自然语言总结出可复用的工作流、关键观察或者避坑规则）；第二，把这条笔记的价值转成密集的 token 级信号。

举个例子：模型在 ALFWorld 里跑完「把勺子洗干净放进抽屉」，事后自己写一条笔记「抽屉是关着的，必须先 open 再 put」。接下来对同一批已采样的动作，模型在「带笔记」和「不带笔记」两种上下文里各算一遍概率——如果 `open drawer` 这一步的概率从 0.2 涨到 0.8，这个差值就成了逐 token 的训练目标。这就是 on-policy distillation（学生自己生成答案、老师当场逐词打分纠正）的变体，只不过老师就是加了笔记的自己。因为演员和分析师共享参数，策略变强的同时复盘能力也变强，监督信号跟着策略一起进化，不会像固定教师模型那样越训越过时。

### 关键实验结果
ALFWorld 上比最强的静态基线高 7.4 分（Qwen2.5-3B）、10.2 分（7B），在 Qwen3-1.7B 上高出 38.1 分。在没见过的场景（unseen split）平均成功率从 70.9 提到 86.2，比 GRPO 高 15.3 分。样本效率很扎实：只用 60% 训练数据就达到 80.7，超过 GRPO 用全量数据的 75.0。视觉类任务（Sokoban、EZPoints）平均成功率 77.0%→91.0%。同时平均轨迹长度从 28 轮压到 13 轮，比 GRPO 的 16 轮更短——是走得更准而不是提前放弃。

### 局限性与开放问题
论文自己承认的：演员和分析师是同一个模型，优点互通、盲区也互通——一条错误的复盘可能把偶发失误固化成「可复用规则」，后续更新还会不断强化它；另外训练要多跑一遍分析和成对打分，成本随交互长度增长（部署时无额外开销）。我的观察：这些「技能」是纯自然语言的，没有和环境状态变化做校验，也就是说笔记写得对不对完全由模型自己说了算，缺一个外部锚点。

### 启发与应用前景
这套思路对做工具调用 agent 的团队很实用：不需要更大的教师模型，也不需要人工标中间步骤，只要环境能给终局奖励就能把稀疏信号加密。项目页 https://jinyangwu.github.io/seed/ 提供了完整代码和实验配置。

---

## 10. Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation
**👍 85** · 🏛 香港科技大学 / 滑铁卢大学 / 阿里巴巴 / 帝国理工学院 · [arXiv](https://arxiv.org/abs/2607.05382) · [GitHub](https://github.com/HaozheH3/SearchGen)

### 问题与动机
图像生成模型画得越来越好，但它不知道的东西会自信地编。用户需求是长尾且不断更新的：新角色、热点事件、训练截止之后的一切。这是结构性缺陷——生成器训在固定语料上，世界却在往前跑。论文做了 SearchGen-Bench 来量化：前沿开源生成器只拿到 21–28 分（满分 100），而这个 40 分级别的塌陷在现有 benchmark 上完全看不出来。

### 方法与核心创新
最自然的解法是接搜索工具，但论文发现无脑搜索反而有害：它不加区分地检索，把噪声塞进那些生成器本来就会画的 prompt 里。

举个例子：你让模型画「一只柯基」，它比谁都熟；这时硬塞几张检索来的柯基照片，会把照片里的背景、光线、构图这些你根本没要求的东西一起泄漏进结果。但你让它画一个上个月才火起来的动画角色，不给参考它就只能编。**该搜什么，取决于这个生成器自己会什么**——这就是论文所谓「生成器特定的、会移动的知识边界」。

边界事先没法写死，但可以「教了再搜」地共训出来：先用 DPO 强化生成器（把能内化的知识塞进参数，边界往外推），再用 RFT 重新校准搜索推理器（让它只去搜边界外的东西，搜索范围往内收）。此外检索到的图不直接当条件图，而是先由推理器转成带引用的文字规格，比如「按图 I，把角色渲染成青金配色的长袍」，避免像素级泄漏。

### 关键实验结果
共训三阶段单调提升：Klein-4B 从 26.4 提到 31.8（+5.4 分），Bagel 从 23.4 提到 26.8（+3.4 分），且每个难度档内部都提升。一个有意思的对照：关掉搜索时，Nano Banana Pro 只掉 9.7 分（75.0→65.3），Qwen-Image-1 掉了近 40 分——掉幅大小直接反映这个模型对外部知识的依赖程度。数据规模上，SearchGen-20K 含 20,839 条 prompt、12 类失败模式、22 个领域，外加预执行好的百万级多模态检索语料 SearchGen-Corpus-1M，支持离线可复现实验。

### 局限性与开放问题
论文自己承认这只是最小可用版本：一轮共训、4B 生成器 + 8B 推理器，还没验证多轮迭代会不会持续把边界推远，也不知道大模型内化的知识在小模型上是否根本无法参数化。我的观察：绝对分数仍然只有 31.8/100，离 oracle 检索的天花板还有明显距离，说明瓶颈可能不只在「搜什么」，也在生成器对文字规格的执行力。

### 启发与应用前景
「知识边界」这个框架可以直接迁移到 RAG：判断一个查询该不该走检索，答案取决于这个具体模型会什么，而不是查询本身难不难。项目页 https://haozheh3.github.io/SearchGen/ 放出了完整数据集和可重放的搜索语料。

---

## 11. ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory
**👍 83** · 🏛 高德地图 · [arXiv](https://arxiv.org/abs/2607.10350)

### 问题与动机
VLM（视觉语言模型）和 VLA（视觉语言动作模型）把机器人的感知和单步动作预测提上来了，但长时程任务还缺一层「运行时」：谁来做推理、存记忆、调工具、验证结果、跨不同机体执行。现在的做法是把这些都糊进一个大控制器里，结果是既没法审计也没法复用。

### 方法与核心创新
ABot-AgentOS 把自己定位成坐在底层控制器之上的「机器人操作系统」，提供场景感知的规划、上下文隔离的技能执行、多阶段验证、多模态记忆和端云协同。核心是 Universal Multi-modal Graph Memory——把对话、视觉观察、空间上下文、时间关系、任务轨迹统统转成带类型的节点和边，每条都标着来源。

举个例子：机器人昨天在客厅听你说过「我的药放在电视柜第二格」，今天你说「帮我拿药」。底层 VLA 只看得到当前画面，根本不知道该往哪走；而图记忆里存着一条「药—位于—电视柜第二格」的边，来源标着那次对话的时间戳，规划器直接查图就能生成导航目标，出错了也能顺着来源追溯是哪一步记岔了。

另一个设计是失败驱动的自进化闭环：诊断出的记忆失败会被转成「进化资产」，但这些资产只允许在后续评测切分上生效，不能回流到当前切分——这是为了防止拿测试集答案反哺自己，属于比较克制的工程处理。

### 关键实验结果
配套的 EmbodiedWorldBench 覆盖 16 个室内外及混合场景、4 个难度档、200+ 任务，包含导航、找物、NPC 对话、动态事件，并按轨迹打分。记忆类基准上静态版本拿到 LoCoMo 87.5、OpenEQA EM-EQA 59.9、Mem-Gallery 88.6、NExT-QA 76.5 Acc@All；开自进化后分别提到 88.7 / 60.4 / 89.0。至于在 EmbodiedWorldBench 上比单控制器基线好多少，摘要只说「在任务成功率和目标完成度上都有提升」，未披露具体数字。

### 局限性与开放问题
我的观察是最该打问号的地方：自进化带来的增益只有 1.2 / 0.5 / 0.4 分，跟静态版本相比几乎在噪声范围内，把「持续改进」当卖点还欠说服力；具身基准的关键数字避而不谈，也让人怀疑那部分结果并不好看。论文自己承认 EmbodiedWorldBench 目前只跑了一个初始子集。此外未开源代码和项目页，复现门槛很高。

### 启发与应用前景
「带类型、带来源、可审计的图记忆」这个抽象比向量库检索更适合具身场景——因为空间和时间关系天然是图结构，而不是一堆语义相近的段落。对做长期陪伴型机器人或家庭助手的团队，这层设计值得借鉴。

---

## 12. Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation
**👍 80** · 🏛 香港大学 / 字节跳动 / 北京大学 · [arXiv](https://arxiv.org/abs/2607.11886)

### 问题与动机
用 RL 训图像生成模型，最难的是奖励函数从哪来。现有路线要么让多模态大模型（MLLM）当裁判打分——但打分对提示词和校准极其敏感，噪声大；要么把 prompt 拆成一堆原子问题逐条验证——可靠但工程复杂度爆炸。两条路都要么训奖励模型，要么标偏好数据。

### 方法与核心创新
SpectraReward 的想法非常朴素：不问模型「这张图好不好」，而是问「看着这张图，你还能不能把原始 prompt 一字不差地背出来」。

举个例子：prompt 是「一只戴着红帽子的猫坐在钢琴上」，模型生成一张图。把这张图作为上下文塞给 MLLM，然后用 teacher forcing（把标准答案一个词一个词喂进去，只统计模型对下一个词的把握）强制它复述这句 prompt。猫头上要是空的，「红帽子」这几个 token 的 log 概率就会明显掉下来。把整句的平均 log 概率当奖励——一次前向就出分，不用采样、不用偏好标注、不用微调奖励模型。

更漂亮的是 Self-SpectraReward：对统一多模态模型（同时具备理解和生成两个分支），直接让自己的理解分支给自己的生成分支打分，形成不依赖任何外部模型的闭环自改进。

### 关键实验结果
基座是 BAGEL，默认奖励模型 Qwen3-VL-30B-A3B，用 AWM 算法在 32 张 A100 上训 380 步。512 分辨率下 SpectraReward 把 TIIF-Bench 短/长 prompt 分别拉高 +10.0/+6.2；对比同样用 MLLM 奖励的 AlphaGRPO，短/长各高 6.3/5.3 分、GenEval 高 3.3 分。Self 版本表现相当（85.1/84.3），GenEval 额外涨 5.5 分。虽然只在 512 训，收益能迁移到 1024 推理：Self 版本 GenEval 89.8、GenEval2 34.3、知识型基准 WISE 0.76。覆盖 2 个扩散模型、3 种 RL 算法、4 个家族 9 个 MLLM 骨干（4B 到 235B）、5 个分布外基准。一个反直觉发现：奖励模型不是越大越好，自奖励能追平甚至超过大得多的外部模型——说明奖励与策略的对齐程度比奖励模型的绝对能力更关键。

### 局限性与开放问题
论文自己承认：奖励质量被骨干模型的视觉理解能力封顶；而且似然只算在 prompt 原文上，只能捕捉显式语义，捕捉不到隐含含义——比如「热咖啡」暗示应该有蒸汽，但这个要求不体现在任何一个 prompt token 上。我的观察：似然奖励存在被钻空子的风险，一张构图平庸但语义元素齐全的图可能拿到高分，而「larger is not better」这个现象论文只给了观测没给机制解释。

### 启发与应用前景
「用重建难度当奖励」这个思路可以直接迁到视频生成、3D 生成，甚至代码生成（给定代码能否还原需求描述）。项目页 https://huangrh99.github.io/SpectraReward/。

---

## 13. Video Generation Models are General-Purpose Vision Learners
**👍 80** · 🏛 Google DeepMind / 多伦多大学 / 伦敦大学学院 / 牛津大学 · [arXiv](https://arxiv.org/abs/2607.09024)

### 问题与动机
NLP 靠 next-token prediction 这一个目标，从任务专用模型走到了通才基础模型。计算机视觉的等价物是什么？这篇论文的答案很直白：大规模文生视频。理由是视频生成天然需要三样东西——时空先验（物体怎么动、遮挡怎么恢复）、视觉语言对齐（文字条件）、以及可扩展性（视频数据无限多）。

### 方法与核心创新
GenCeption 把预训练好的视频扩散骨干改造成一个前馈感知模型，靠文字指令切换任务。

举个例子：让视频模型「续拍」一个人转头的镜头，它必须隐含地知道鼻子比耳朵离镜头近、头发遮住的区域转过去会露出什么。这些几何和结构知识已经压在骨干网络的权重里了。GenCeption 做的事是把这个骨干截下来，去掉逐步去噪那个耗时的循环，改成一次前向直接出结果，再用文字指令指定输出类型——「输出深度图」、「分割出那个正在倒水的人」、「给出相机位姿」。同一套权重、同一次前向，换句提示词就换个任务。

### 关键实验结果
在深度估计、表面法线、相机位姿估计、指代表达分割、3D 关键点预测这一堆任务上都做到 SOTA，经常追平或超过各自领域的专用模型（DepthAnything3、SAM3、D4RT、VGGT-Omega、Sapiens、Lotus-2 等）。最能说明问题的是数据效率：追平 D4RT 和 VGGT-Omega 这类领先模型，只用了它们 1/7 到 1/500 的训练数据。同等设置下，视频生成预训练也打赢了 V-JEPA 和 Video MAE 这两条主流自监督路线。还有个有意思的涌现现象：只在合成人体视频上训练的模型，能泛化到真实拍摄画面，甚至泛化到动物和机器人这些完全没见过的类别。摘要未披露各任务的具体指标数值，只给了「达到或超过专用模型」的定性描述和上述数据效率倍数。

### 局限性与开放问题
论文自己说 scaling 性质只是「初步」（preliminary），也就是说数据和模型规模上去之后曲线还没跑满。我的观察：所有任务都是稠密预测或几何类，没测需要高层语义推理的任务（比如视觉问答、复杂场景理解），而这恰恰是「通才视觉智能」最该证明的部分；另外缺了完整的推理成本对比——去掉去噪循环之后是否真的比专用小模型快，摘要没交代。

### 启发与应用前景
如果这条路成立，视觉领域的做法会从「每个任务标一批数据训一个模型」变成「拿视频生成骨干 + 少量数据微调」，专用感知模型的价值会被大幅稀释。对机器人和自动驾驶尤其关键——它们要的正是物理世界的时空先验。项目页 https://genception.github.io/。

---

## 14. Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading
**👍 74** · 🏛 马里兰大学 / 腾讯 · [arXiv](https://arxiv.org/abs/2607.08964) · [GitHub](https://github.com/zli12321/LHTB)

### 问题与动机
现在的终端类 benchmark 有两个毛病：任务几分钟就跑完，而且只看最终结果对不对。这导致奖励信号极度稀疏，也看不出模型到底走到了哪一步。

举个例子：任务是「复现某篇论文的实验」，传统评测只看最后有没有产出一张对得上的结果表。装好了依赖、跑通了数据预处理、训练脚本跑到 90% 崩了——全都算 0 分，跟一行代码都没写的模型分数一模一样。「完全不会」和「差最后一步」被混为一谈。

### 方法与核心创新
LHTB 保持 Terminal-Bench 那套设定（每个任务有参考解或仿真引擎），但把每个任务拆成一串细粒度的、可打分的子任务，做成密集中间奖励和部分得分。任务本身也刻意做长：典型任务需要几百个 episode、几十分钟到几小时的执行时间，逼的是长程规划、长上下文管理和迭代调试，而不是一次性解题。46 个任务覆盖 9 个类别：实验复现、软件工程、多模态分析、交互式游戏、科学计算等。

### 关键实验结果
评了 15 个前沿模型，实测强度非常直观：平均每个任务消耗 990 万 token、约 231 个 episode、85.3 分钟执行时间——这个量级远超此前任何终端 benchmark。成绩相当难看：最强模型在「部分得分 ≥0.95」这个阈值下只有 15.2% 的 pass@1，在「必须满分 1.0」下只有 10.9%；所有模型的平均通过率分别只有 4.3% 和 1.7%。也就是说随便挑一个前沿模型丢进去，五十次里成功不到一次。

### 局限性与开放问题
论文自己承认这个 benchmark 主要用来暴露改进空间，并给了失败模式和错误模式的分析。我的观察有三点：46 个任务对 9 个类别来说样本量偏小，单类只有 5 个左右，类别间比较的统计效力很弱；密集打分依赖人工拆解的子任务，拆解粒度本身就是一种主观建模，不同拆法会显著改变分数；平均 990 万 token 的成本意味着完整跑一遍 15 个模型的开销极高，社区复现和迭代评测的门槛不低。

### 启发与应用前景
对做 coding agent 的团队，这是目前少数能区分「进步了多少」而不只是「过没过」的评测——部分得分曲线可以直接当训练期的进度信号用，而不只是最后验收。项目页 https://zli12321.github.io/LHTB/index.html。

---

## 15. SynthDocBench: Controlled Benchmark for Long-Context Visual Document Understanding
**👍 69** · 🏛 ServiceNow / Mila / 蒙特利尔大学 · [arXiv](https://arxiv.org/abs/2607.10400) · [GitHub](https://github.com/ServiceNow/SynthDocBench)

### 问题与动机
视觉语言模型（VLM）在 DocVQA、ChartQA、MMLongBench-Doc 上分数都不低，但真实文档同时混着长度、版式复杂度、模态构成、问题难度好几个变量，一旦模型答错，你根本不知道该怪谁。

举个例子：某个模型在 MMLongBench-Doc 上掉了 10 分，是因为文档太长撑爆了上下文，还是因为图表太多、还是因为问题需要跨页推理？这三个因素在真实文档里天然纠缠，做不了归因。SynthDocBench 干的事相当于做对照实验：固定长度只变版式，固定版式只变长度，一次只动一个旋钮。

### 方法与核心创新
全合成 benchmark，用组合设计（combinatorial design）让每个因子独立变化——文档长度、版式结构、模态构成、问题类型。文档由 LLM pipeline 端到端生成，覆盖 6 种版式原型，并刻意加入 40% 的随机覆写，防止模型靠「这个 benchmark 的文档长这样」这类伪相关走捷径。文档长度和结构多样性都显著超过现有基准。

### 关键实验结果
评了 7 个前沿 VLM，挖出三个现有 benchmark 完全暴露不出来的失败模式。其一，随文档变长性能陡降。其二，系统性的位置敏感：6 个模型里有 5 个在文档「中间三分之一」表现最差，同样有 5 个呈现从前段到后段的负向趋势，最陡的一条掉了 8.3 个百分点——这跟纯文本长上下文里著名的「lost in the middle」现象严丝合缝对上了，说明它不是文本特有的，视觉文档里同样存在。其三，图表理解在长文档场景下直接崩掉：单独看图表模型答得挺好，一旦这张图埋在长文档中间就不行了。结论是当前模型可能在过拟合 benchmark 的人工痕迹，而不是真的具备稳健的长上下文视觉文档理解能力。

### 局限性与开放问题
我的观察：全合成文档意味着完全不包含扫描噪声、手写、印章、歪斜这些真实文档的脏活，结论向真实场景的迁移性存疑；文档由 LLM 生成，如果被评模型和生成模型同源，可能天然占便宜——那 40% 随机覆写的设计恰恰说明作者也意识到合成 pipeline 自身存在可被利用的规律；7 个模型的样本量做「5 of 6」这类统计陈述，说服力有限。论文自身把这些定位为受控诊断工具而非替代真实评测。

### 启发与应用前景
最直接的用法是当归因工具：产品里的文档理解模块出了问题，先用受控 benchmark 定位是长度、位置还是模态的锅，再决定是切 chunk、重排还是换模型。「中间段最难」这个结论也提示了一个便宜的工程解——把关键信息放在文档头尾，或者对中段做重复注入。

---

## 16. SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration
**👍 60** · 🏛 中国人民大学 / 蚂蚁集团 · [arXiv](https://arxiv.org/abs/2607.15257) · [GitHub](https://github.com/antins-labs/SearchOS)

### 问题与动机
搜索已经是信息检索类 agent 的核心能力，但交互历史一长，agent 就跟丢了自己的进度。一旦搜不到有用证据，单 agent 和多 agent 系统都容易陷进重复循环——同样的关键词换个说法再搜一遍，把预算烧光，最后答案又残缺。多加几个 agent 也解决不了：并行的 worker 会重复劳动，还会对同一个中间状态产生分歧。

### 方法与核心创新
SearchOS 的核心主张是：把隐式、脆弱的搜索进度，变成显式、持久、共享的系统状态。

第一步是把开放域信息检索重新形式化为**关系表填充**。举个例子：「过去五年获得某奖的华人导演、各自的代表作和获奖年份」——这本质就是填一张表。SearchOS 先由探索 agent 决定表结构（一张宽表，还是导演表 + 作品表两张关联表），每个单元格必须挂一条来源引用。

第二步是 SOCM（面向搜索的上下文管理），把进度外化成四个显式对象：Frontier Task（待办）、Evidence Graph（证据图）、Coverage Map（哪些格子还空着）、Failure Memory（哪些查法已经试过且失败）。下一个 sub-agent 直接读 Failure Memory，就不会再撞同一堵墙。

第三步是流水线并行调度：不等一批 agent 全部跑完，谁空出来就立刻塞一个针对未覆盖空缺的新任务。外加一层 Search Tool Middleware Harness 拦截所有模型和工具交互，负责记录证据、检测卡死、执行预算上限。

### 关键实验结果
WideSearch 上 item 级 F1 达 80.3（精确率 83.9 / 召回率 79.7），比最强基线 A-MapReduce 的 76.0 高 4.3 分，增益主要来自召回，跟「补全覆盖空缺」的设计意图一致；row 级 F1 56.5（+2.0）。GISA 的 Set 类问题上 76.5 对基线最好的 63.1，高出 13.4 分——需要完整枚举答案集的场景最吃证据图和覆盖检查。效率上，连续调度把平均端到端时间降了 24.3%，同时 LLM 调用次数更少、F1 反而更高。还有一个消融很说明问题：即便用 oracle 为每道题挑最优的固定表结构，仍比 SearchOS 的自主 schema 规划低 8.2 个 item F1 点。

### 局限性与开放问题
论文自己说下一步要扩展到更多领域和多模态设定，并改进跨 agent、跨数据源的适配。我的观察：「关系表填充」这个形式化对可枚举、结构化的问题特别合适，但对「解释某个现象的成因」这类没有天然表结构的开放问题恐怕水土不服，而后者才是真实检索需求的大头；另外只在 WideSearch 和 GISA 两个基准上评测，覆盖面偏窄。

### 启发与应用前景
Coverage Map + Failure Memory 这对组合是可以直接抄进任何长程 agent 系统的工程模式——只要任务能被拆成「一堆待填的空」，把已试过的失败路径显式落盘，就能省掉大量重复劳动。

---

## 17. KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill
**👍 56** · 🏛 Lychee Team / 哈尔滨工业大学（深圳） / Shenzhen AI Training Platform / Shenzhen Loop Area Institute · [arXiv](https://arxiv.org/abs/2607.12625) · [GitHub](https://github.com/HITsz-TMG/KnowAct)

### 问题与动机
OpenClaw 这类 agent 框架擅长调工具、写代码，但一碰手机屏幕就露怯：跨平台（Android/iOS/鸿蒙/Windows）的 GUI 操作支持薄弱，而且**每次都从零开始**——昨天摸索了 20 步才找到某个 App 的「订单退款」入口，今天同样的任务还是摸索 20 步。这两个缺陷叠加，导致长任务（long-horizon，指需要几十步操作才能完成的任务）成功率上不去。

### 方法与核心创新
论文提出 Know-Route-Act-Reflect 四段式框架。主 agent 先「Know」——调取历史交互经验和任务知识来拆解任务；再把子任务路由给可插拔的 GUI 子 agent。子 agent 有两个核心部件：**经验可归因的记忆系统**（记住哪条经验来自哪次执行，出错时能回溯是哪条记忆坑了自己）和**自演化技能库**（把成功的操作序列固化成「快捷路径」）。

举个例子：第一次帮用户在美团退款，agent 试错找到路径「我的 → 订单 → 展开详情 → 申请退款」；这条路径被写进技能库变成一个原子技能，下次用户说「退掉昨天那单」，agent 直接调技能一步到位，不再逐屏识别。同时用户画像和反馈持续沉淀，让任务拆解和工具调用越用越准。

### 关键实验结果
在长任务基准 MobileWorld 上，搭配开源 Kimi-2.6 的 GUIClaw 拿到 **64.1%**，超过所有对比的 agent 框架，也超过闭源的 Seed-2.0-Pro 和 GPT-5.5——开源模型 + 好框架打赢闭源模型是这篇最硬的卖点。记忆与技能的**可迁移性**也验证了：把积累的经验换到其他底座模型上依然有效，Kimi-2.6 上带来 8.5% 的提升。

### 局限性与开放问题
论文自承跨平台适配依赖子 agent 的可插拔设计，新平台仍需接入成本。我观察到的更大隐患是：技能库会**固化过时路径**——App 一改版，缓存的「快捷路径」就成了错误路径，摘要没说清失效检测机制。另外记忆里存用户画像和反馈，隐私边界如何划定完全没提。

### 启发与应用前景
最实用的启发是「经验即资产」：agent 产品的护城河不在基座模型，而在用户使用过程中沉淀的、可跨模型迁移的技能与记忆。这意味着换底座模型不必推倒重来。项目页 https://shibosusu.github.io/KnowAct-GUIClaw/ 有跨四平台的实机演示。

---

## 18. Scalable Visual Pretraining for Language Intelligence
**👍 55** · 🏛 上海人工智能实验室 / 中国科学技术大学 / 浙江大学 / 上海交通大学 · [arXiv](https://arxiv.org/abs/2607.09657)

### 问题与动机
今天所有大模型的预训练都默认一件事：把网页、PDF、论文先「拍平」成纯文本再喂给模型。这个 OCR/解析步骤是**有损**的。举个例子：一篇论文里的柱状图，转文本后只剩图注「图3：不同方法的准确率对比」——柱子谁高谁低、差多少、趋势是升是降，全丢了；一个排版精美的财报表格，转成文本后行列对应关系经常错位；一个多栏排版的杂志页，纯文本会把两栏内容交叉串成一锅粥。这些信息本来就写在**像素里**，文本管道天生接不住。

### 方法与核心创新
这篇论文的做法很激进：**跳过文本抽取，直接把文档页面当图像做无监督预训练**。核心主张是「视觉预训练本身就是一个可规模化的语言智能学习器」——不是把视觉当作语言能力的补充模态，而是当作语言能力的**替代获取路径**。

关键的实验设计是控制变量：同一份底层语料，一条路线走「转文本 → 文本预训练」，另一条走「保留页面图像 → 视觉预训练」，然后在多个 backbone、多个下游基准上对比。这样排除了「视觉路线只是因为看了更多数据才赢」的干扰，直接检验表征形式本身的差异。

### 关键实验结果
结论是视觉预训练在相同语料上**一致优于**纯文本预训练，且跨多个 backbone 和 benchmark 都成立。**摘要未披露具体数字**——没给出提升几个点、用了多大语料、模型规模多少，这是这篇最遗憾的地方，也让「scalable」的成色难以判断（是在 1B 上验证还是在 70B 上验证，结论的分量完全不同）。

### 局限性与开放问题
论文自承这是一次「系统性研究」而非成品模型，定位偏探索。我观察到的核心疑问是**成本**：图像 token 通常比等价文本贵一个数量级，摘要说「efficient pathway」但没给 FLOPs 或 token 效率的对照，这个账不算清，工业界不会跟。另外纯视觉预训练能否学到长程推理能力，也完全未验证。

### 启发与应用前景
如果结论站得住，影响是釜底抽薪的：整个数据清洗管道里那套 PDF 解析、OCR、表格还原的重工程可能被绕过。对垂直领域（金融研报、医学文献、法律卷宗——这些恰恰是排版信息最密集的地方）尤其有价值。

---

## 19. OvisOCR2 Technical Report
**👍 52** · 🏛 阿里巴巴 · [arXiv](https://arxiv.org/abs/2607.13639)

### 问题与动机
文档解析（把一页 PDF 转成结构化 Markdown）长期由**流水线方法**统治：先做版面分析框出区块，再分别调文字 OCR、公式识别、表格识别模型，最后拼装。流水线的问题是误差逐级累积——版面框错一次，后面全错，而且工程上要维护五六个模型。

举个例子：一页论文里公式紧挨着正文排版，版面分析若把这块判成「文本区」而非「公式区」，后面就会派通用 OCR 去认它，`E=mc²` 输出成 `E=mc2`——上标丢了语义全毁，而真正该干这活的公式识别模型根本没被调用，压根没机会纠正。这就是流水线的死穴：上一环的错误对下一环不可见。端到端模型（一张图进、Markdown 出）本可绕开这点，但一直打不过流水线。OvisOCR2 要证明的就是：端到端也能赢，而且只用 **0.8B** 参数。

### 方法与核心创新
两条主线。**数据引擎**：真实文档标注做过滤，再加合成页面——妙处在于合成数据的图像和 Markdown 标注**来自同一份 HTML 源码**渲染，所以标注天然零误差、零错位，不像人工标注会漏标下标或搞错表格合并单元格。

**训练配方**是四步串联：先监督微调（SFT）；然后在一个更大的 **4B 分支**上做强化学习，用多组件奖励（文本准确率、公式正确性、表格结构等分开打分，而不是笼统一个相似度）；接着做 **on-policy distillation** 蒸馏到 0.8B——注意这不是普通蒸馏，普通蒸馏是让小模型模仿老师的标准答案，on-policy 是**让 0.8B 学生自己先生成一版 Markdown，再由 4B 老师针对学生这版实际输出当场打分纠错**，学生因此学到的是「我常犯的错怎么改」而不是「标准答案长什么样」；最后模型融合。

### 关键实验结果
OmniDocBench v1.6 上总分 **96.58**，SOTA——这是端到端模型第一次登顶这个此前被流水线方法霸占的榜。PureDocBench 上 Avg3 拿到 **75.06**，同样第一。内部长尾难例基准上也是最优。用 0.8B 干掉多模型流水线，相当于把一整套服务压缩进一个能上端侧的小模型。

### 局限性与开放问题
论文没披露推理延迟和显存占用，0.8B 的实际部署优势缺少数据支撑。我观察到的风险是那套四步训练配方（SFT + 4B RL + 蒸馏 + 融合）复现成本极高，几乎只有大厂能跑通；而 96.58 已逼近天花板，剩余误差集中在哪类文档、是否是标注噪声，摘要没有拆解。

### 启发与应用前景
「大模型做 RL、小模型做部署、中间用 on-policy 蒸馏搭桥」是很值得复用的范式——RL 需要容量，服务需要便宜，这套配方把两者解耦了。模型已在 HuggingFace 开放（ATH-MaaS/OvisOCR2），做 RAG 文档入库的团队可以直接替换现有解析管道。

---

## 20. 4D Human-Scene Reconstruction from Low-Overlap Captures
**👍 52** · 🏛 首尔大学 · [arXiv](https://arxiv.org/abs/2607.09125)

### 问题与动机
影视级的动态人体捕捉靠的是几十上百台相机围一圈的摄影棚阵列，效果好但普通人用不起。现实场景里往往只有**三五台稀疏、视野几乎不重叠**的相机。举个例子：一个篮球场四角各装一台监控，球员运动时，他的后背可能在整场比赛中**从来没有任何一台相机拍到过**——这就是「低重叠」的要害，不是分辨率不够，是有些区域**根本没有观测**。现有 4D 重建方法在这些盲区会糊成一团；视频扩散模型能脑补，但生成的人体几何前后不一致（这一帧手臂两条，下一帧长出第三条）。

### 方法与核心创新
StudioRecon 的核心判断是：**人和背景不该用同一套办法补**。背景是静态的、容错高，适合让扩散模型脑补；人是动态的、几何必须自洽，脑补会翻车。于是把两者解耦：

- **背景**：用视频扩散模型合成数百个「相机可控」的新视角，把稀疏的背景监督信号密集化——相当于凭空造出一个虚拟的多机位棚。
- **人体**：用可变形高斯（deformable Gaussians）建模，但初始化做得很扎实——先跨视角做身份关联（多台相机里谁是谁，别把 A 的左手接到 B 的身上），再用多视角关键点三角化拟合出骨架。
- **合成后**：递归增强模块 + 运动自适应一致性注入，把人和背景拼在一起后残留的接缝、闪烁抹平。

### 关键实验结果
在**四个真实世界数据集**上取得新视角合成的 SOTA，并演示了新轨迹渲染（让虚拟相机沿任意路径飞行）和人物替换两个应用。**摘要未披露具体的 PSNR/SSIM/LPIPS 数值**，只给了 SOTA 的定性结论，无法判断领先幅度是压倒性的还是边际的。

### 局限性与开放问题
论文自承在欠观测区域仍需「进一步避免残留伪影」，说明问题是缓解而非根除。我观察到的限制：管线依赖视频扩散模型生成数百视角，推理成本恐怕不低，实时化无望；多视角关键点三角化在人物密集遮挡（如球场混战）时的身份关联稳定性也存疑。

### 启发与应用前景
最直接的落地是把体育场馆、演唱会、安防摄像头这类**既有的稀疏机位**升级成自由视角回放系统，不必新增硬件。「静态部分交给生成、动态部分交给几何」这个分工思路，对所有混合场景重建任务都通用。项目页 https://sisyphm.github.io/studiorecon-page/ 有效果视频。

---

## 21. BadWAM: When World-Action Models Dream Right but Act Wrong
**👍 47** · 🏛 新加坡国立大学 / 香港理工大学 · [arXiv](https://arxiv.org/abs/2607.15207) · [GitHub](https://github.com/LiQiiiii/BadWAM)

### 问题与动机
世界-动作模型（WAM）是具身智能的新宠：机器人不只输出动作，还同时预测「执行完这个动作后世界长什么样」。社区普遍相信这种耦合带来**安全性**——因为可以拿动作去和模型「想象的未来」对账，对不上就拦截。这篇论文专门来砸这块招牌：这个假设很脆弱。

### 方法与核心创新
论文定义了一类新攻击——**世界-动作漂移攻击**（World-Action Drift Attack），用肉眼几乎看不出的图像扰动，去撬开「想象」和「执行」之间的对齐。BadWAM 沿攻击强度和隐蔽性两个维度给出两种实例：

- **纯动作攻击**：不管想象，直接把模型往「任务失败的动作」上推。粗暴有效，但会留下证据——想象的未来会跟着崩坏，安全监控能发现。
- **想象保持攻击**：这个才是真正阴险的。它在优化时**额外加一项约束，强制模型预测的未来画面尽量贴近未被攻击时的样子**，同时让实际输出的动作偏掉。

举个例子：机器人要把杯子放到桌上，它「想象」出的画面是杯子稳稳落在桌面——监控系统看这张想象图，一切正常，放行；但同一时刻它实际输出的关节指令是把杯子松手扔到地上。**监控看到的和机器人做的，被攻击者硬生生撕成了两回事**——「对账式安全」在这里彻底失效。

### 关键实验结果
在多个 WAM 变体上闭环执行测试：纯动作攻击把任务成功率从 **96.5% 打到 43.1%**，掉了 53 个百分点，等于让一个几乎不出错的机器人变成掷硬币。想象保持攻击揭示了更关键的现象：**适度**的未来保持正则化既能维持很强的攻击效果，又能显著压低想象漂移——也就是说攻击者不必在「有效」和「隐蔽」之间做取舍，两者可以兼得。

### 局限性与开放问题
论文自承评估限于若干 WAM 变体。我观察到的关键缺口是**防御方案完全缺席**——只做了攻击面刻画，没给检测或加固手段。另外攻击是白盒还是黑盒、物理世界打印出来的对抗贴纸是否仍然有效（摄像头成像、光照、角度都会削弱扰动），摘要未交代，这直接决定威胁是纸面的还是现实的。

### 启发与应用前景
对做具身安全的团队是当头一棒：**别把模型的自我预测当独立的安全校验层**——它和动作头共享同一个被攻击的输入，不是相互独立的证据源，真正的冗余必须来自模型之外（独立传感器、物理限位、外部监督模型）。项目页 https://liqiiiii.github.io/BadWAM/ 有攻击演示。

---

## 22. LightMem-Ego: Your AI Memory for Everyday Life
**👍 46** · 🏛 浙江大学 / 华南理工大学 / 华中师范大学 / 联想 · [arXiv](https://arxiv.org/abs/2607.11487)

### 问题与动机
AI 眼镜和手机能全天候录下你看到、听到的一切，但**录下来不等于记住**。用户真正会问的是「我钥匙放哪了」「上周老板说的 deadline 是几号」——要答对这类问题，得有一套能持续吞入、自动组织、快速检索的多模态长期记忆。难点在于设备端算力和存储极其有限：一天 16 小时的视频音频流，不可能全量存下来再做全局检索。

### 方法与核心创新
LightMem-Ego 的核心是**分层记忆 + 动态路由**。系统同时接第一人称视角的视频流和音频流，先在**共享时间轴**上对齐（关键的一步：只有对齐了才能回答「他说那句话的时候我正在看什么」），然后组织成三层：当前记忆（正在发生）、短期记忆（近期，保留较多细节）、长期记忆（远期，高度压缩成摘要和结构化事件）。

查询进来时，系统**动态判断该去哪一层捞**。举个例子：问「我刚才把手机放哪了」只需查当前记忆，几秒内命中；问「上周三会议上讨论了什么」会路由到长期记忆，先在压缩摘要里定位到那个时间段，再回原始片段取证据。这个路由机制是省算力的关键——不用每次都扫全量历史。最后答案必须**基于多模态证据生成**，即回答要能指回具体的画面帧和音频段，而不是模型凭印象编。

### 关键实验结果
这是一篇 **demo/系统论文**，重点在可部署性：已在智能手机和 AI 眼镜上跑通，支持找东西、对话回溯、生活总结、作息规律发现、个性化助理五类场景。**摘要未披露任何量化指标**——没有检索准确率、没有端侧延迟、没有内存占用、也没有和其他记忆系统的对比，「lightweight」轻到什么程度完全无从判断。

### 局限性与开放问题
论文本身定位是演示系统，未给出系统性评测。我观察到的两个硬问题：一是**隐私**，全天候录制第一人称视频天然会拍到同事、家人、路人，数据留存和脱敏策略只字未提；二是**遗忘策略**，三层记忆的压缩和淘汰规则是什么？如果长期记忆里丢掉了「钥匙放在鞋柜第二层」这种当时看似无关的细节，整个系统就废了——什么该留什么该扔，恰恰是这类系统最难也最核心的决策。

### 启发与应用前景
分层 + 路由这套结构不止适用于可穿戴，对任何**长上下文 agent** 都适用：把「最近几轮对话」「本次会话摘要」「跨会话的用户画像」分层存放并按需检索，比无脑塞进超长上下文窗口更省更准。代码已开源在 zjunlp/LightMem-Ego。

---

## 23. Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model
**👍 41** · 🏛 小米 · [arXiv](https://arxiv.org/abs/2607.11643)

### 问题与动机
机器人训练最缺的是数据，真机采集又慢又贵。用图像/视频生成模型造数据是自然思路，但直接拿来用不行——具身场景要求多视角一致（同一时刻头部相机和腕部相机看到的必须是同一个物理世界）、几何连贯、还得符合机器人本体约束（机械臂不能拧出人类做不到的角度）。而现有做法是拿少量机器人数据去微调基础模型，结果**把预训练学到的通用视觉知识微调没了**——学会了画机械臂，忘了世界长什么样。

### 方法与核心创新
Xiaomi-Robotics-U0 是一个 **38B 参数的多模态自回归模型**，思路是把具身生成当成基础图像/视频生成的**自然延伸**而非独立任务：五个任务联合优化——文生图、图像编辑、具身场景生成、具身迁移、具身视频生成。前两个是通用能力，后三个是具身能力，一起训所以通用知识不会被冲掉。

最有意思的是**结构化可控的「具身迁移」**。举个例子：你已经采了一段「用 A 型两指夹爪抓取桌上马克杯」的真实数据，现在想要「换成 B 型五指灵巧手、桌面换成金属台、杯子换成玻璃瓶」的数据。U0 可以做这种细粒度编辑，同时保持多视角一致性和交互动力学（手指接触瓶身的形变、抓稳后的运动轨迹都得跟着改对）。一段真机数据因此被放大成几十段不同本体、不同场景的训练数据。

### 关键实验结果
最有说服力的是下游验证：把 U0 生成的数据用于训练，**把 pi_0.5 在真实世界高难度操作任务上的分布外（OOD）成功率从 36.9% 拉到 63.2%**——提升 26.3 个百分点，几乎翻倍，而且是在「没见过的场景」上，说明生成数据真的补上了泛化缺口而非过拟合。此外具身场景生成和迁移的人工评测**优于 GPT-Image-2.0**，具身视频生成在 World Arena 榜单**排名第一**。

### 局限性与开放问题
论文没说清 38B 模型的生成成本——如果造一条数据比真机采一条还贵，性价比就是负的。我观察到的更深层问题是**分布幻觉**：生成数据的物理正确性由模型的世界模型质量兜底，一旦它对某类材质（透明、反光、柔性物体）的接触动力学理解有偏，训出来的策略会在真机上系统性翻车，而且这种错误在仿真评测里看不出来。

### 启发与应用前景
核心结论是「基础世界模型可以同时充当具身世界模型和**可扩展的数据引擎**」——这可能是机器人领域绕开数据瓶颈最现实的路径：少量真机数据 + 大规模结构化迁移生成。代码与 checkpoint 已在 robotics.xiaomi.com/xiaomi-robotics-u0.html 开放。

---

## 24. AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities
**👍 38** · 🏛 上海人工智能实验室 · [arXiv](https://arxiv.org/abs/2607.13705) · [GitHub](https://github.com/open-compass/agentcompass)

### 问题与动机
Agent 评测目前是一地鸡毛：每个 benchmark 自带一整套跑法，评测逻辑、执行框架、运行环境**死死焊在一起**。举个例子：你想在 SWE-bench 上换一个自研的 ReAct 循环试试，会发现它的 harness 直接把 Docker 启动、补丁应用、测试执行、结果判分全写在一个脚本里；换到 WebArena，又是另一套完全不同的耦合结构。想横向对比五个 benchmark，你得把「跑 agent」这件事重复实现五遍——结果就是复现困难、工程量重复浪费。

### 方法与核心创新
AgentCompass 的核心动作是**做正交拆分**，把评测拆成三个互相独立的组件：**Benchmark**（任务和判分标准）、**Harness**（agent 的执行逻辑，比如 ReAct、Plan-and-Execute）、**Environment**（沙箱、浏览器、操作系统等运行时）。三者可以自由组合——想在新 benchmark 上试你的 harness，只写 benchmark 适配，执行逻辑一行不改。

另外两个工程点很戳痛处：**容错的异步运行时**（agent 评测动辄跑几小时，中途某个任务卡死或超时，不能让整批任务陪葬）；**完整的轨迹分析工具**，用来诊断细粒度失败模式，尤其点名了 **reward-hacking**——指 agent 没真正解决问题却骗过了判分器，比如该修 bug 却直接把测试用例删了、或者硬编码一个 if 分支让断言通过。这类失败只看最终分数完全发现不了，必须回看轨迹。

### 关键实验结果
这是基础设施论文，没有模型性能指标。硬数字是覆盖面：原生支持 **20+ 个 benchmark**，横跨**五个能力维度**。**摘要未披露**运行时的吞吐、故障恢复率或与其他评测框架的工程量对比数据。

### 局限性与开放问题
论文自承定位是「轻量、可扩展的基础设施」而非评测方法论创新。我观察到的问题：三组件解耦看着优雅，但现实中 benchmark 和 environment 常有隐式耦合（某些任务的判分依赖特定容器镜像里的工具版本），抽象层能否兜住这些边界情况有待检验；另外 reward-hacking 检测目前靠人工看轨迹还是自动化规则，摘要没说清——如果是人工，那它就不是可规模化的能力。

### 启发与应用前景
对任何自建 agent 评测的团队，这套「Benchmark / Harness / Environment 三层解耦」的抽象值得直接借鉴，能省掉大量重复胶水代码。更值得关注的是轨迹分析这条线：随着 agent 任务越来越长，**「分数对不对」正在让位于「过程对不对」**——只看终局分数会被 reward-hacking 系统性欺骗，轨迹级诊断会成为标配。项目已在 OpenCompass 生态下开源。

---

## 25. KeyFrame-Compass: Towards Comprehensive Evaluation of Keyframe-Conditioned Video Generation
**👍 36** · 🏛 香港科技大学（广州） / 快手 / 北京大学 / 中国人民大学 · [arXiv](https://arxiv.org/abs/2607.14202) · [GitHub](https://github.com/cactusqq/KeyFrame-Compass)

### 问题与动机
现在做 AI 视频的实际工作流很少是「一句话生成一条片子」，而是创作者先画/选好几张关键帧（keyframe，就是分镜里指定「第几秒画面长这样」的参考图），让模型把中间补出来。新一代模型都号称支持多关键帧条件，但没人认真验过：它到底有没有把你给的那几张图**照着做**，还是只是「参考了一下氛围」。这篇论文补的就是这个评测空白。

### 方法与核心创新
KeyFrame-Compass 是第一个专门评关键帧条件视频生成的基准，386 条精挑样本，在五个维度上做交叉：3 个应用领域 × 2 种视频结构 × 2 种提示词粒度 × 2 种条件输入格式 × 4 档关键帧密度——这样才能分离出「是模型不行还是这类输入不行」。核心创新在评测拆解：它把「关键帧执行得好不好」拆成六个互补指标——**是否出现**（presence）、**画面还原度**（fidelity）、**时序顺序对不对**（ordering）、**出现在第几秒**（localization）、**能持续多久**（persistence）、**有没有重复出现**（uniqueness）。整体画质则用多模态大模型当裁判，但要求它给出证据、再配专门的感知模型交叉校验，避免 MLLM 拍脑袋打分。

举个例子：你给模型三张图——男主推门、男主坐下、男主喝水。旧指标算个 CLIP 相似度就说「像」，但实际生成的片子可能是男主先喝水后推门（ordering 错），或者推门那帧只闪了 3 帧（persistence 差），或者坐下的画面出现了两次（uniqueness 差）。这六个指标就是把「像」这个模糊结论拆成能定位问题的六条诊断。

### 关键实验结果
九个代表性视频生成系统上测下来有三个明确结论：一是**忠实执行关键帧和自然流畅的视频合成之间存在明确权衡**——越老实照着关键帧做，运动越僵硬；二是关键帧越密，性能下降越明显；三是大多数开源模型**读不懂 storyboard 网格图**（把多张关键帧拼成一张九宫格输入），它们把网格当成一张普通图片，完全没理解成有先后顺序的帧序列。摘要未披露各模型的具体分数。

### 局限性与开放问题
论文自己指出的是当前模型的能力边界，而非基准的边界。我观察到的问题：386 条样本对五维交叉设计来说偏薄，某些格子可能只有个位数样本，统计结论的稳定性存疑；另外整体画质仍依赖 MLLM 评判，而 MLLM 对「运动自然度」的判断本身就是这批模型的弱项，存在评估者与被评估者同源的风险。

### 启发与应用前景
对做 AI 短视频工具的团队，这套六指标可以直接当线上回归测试用——尤其 ordering 和 persistence 两项，是用户投诉最多但传统指标测不出来的。storyboard 网格读不懂这个发现更实用：如果你的产品让用户传九宫格分镜，现在基本可以确定开源模型会做错，老实拆成独立图片序列传更靠谱。

---

## 26. PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails
**👍 35** · 🏛 复旦大学 / 同济大学 / Virtue AI / 香港中文大学 · [arXiv](https://arxiv.org/abs/2607.05910) · [GitHub](https://github.com/ssmisya/PolicyShiftGuard)

### 问题与动机
现有的图片安全审核模型有个隐含假设：安全性是图片本身的固有属性——这张图要么安全要么不安全。但真实业务完全不是这样。同一张比基尼照片，在电商泳装类目是正常商品图，在儿童教育 App 里就该拦；今天允许的内容，明天政策一改就得下架。论文把这个设定叫 **policy-adaptive guardrailing**：模型收到的不只是图片，还有一份「当前生效的策略文本」，它得按这份策略判，而不是按自己训练时记住的「安全直觉」。

### 方法与核心创新
先建基准 PolicyShiftBench：265 张图，2000 条策略判别样本，**平均每张图配 7.55 条不同的策略提示**——同一张图在不同策略下答案不同，这样才能测出模型是真在读策略还是在背图片。这是整个设计的精髓所在。

方法侧是两阶段训练。第一阶段 **RP-SFT（随机策略微调）**：训练时随机采样各种策略描述配同一张图，逼模型不敢忽略策略文本。第二阶段 **BP-Adapt（边界对适配）**：对同一张图、同一个风险类别，构造一对「该拦的策略」和「该放的策略」，除了常规标签监督，还加一个成对比较损失，直接把这两条推开。

举个例子：一张「有人在喝啤酒」的图，配策略 A「禁止展示酒精饮品」→ 拦，配策略 B「禁止未成年人饮酒画面」→ 放。这两条只差几个字，模型必须真读进去才能分对。消融实验证实，正是这种「配对的通过/拦截边界对」让策略适配变稳定——只给单边样本训不出来。

### 关键实验结果
现成的 VLM 和专用审核模型在策略漂移下普遍脆弱。PolicyShiftGuard 的 **7B 模型在 PolicyShiftBench 上拿到 76.9 平均 F1、72.1 平均 PSS（策略敏感度分数），双双 SOTA**，并且能迁移到 UnSafeBench 和 SafeEditBench 两个外部基准。此外它用了精简输出格式（不写长篇推理），把延迟-效果的权衡也改善了——对每天要过几亿张图的审核链路，这条比精度更值钱。

### 局限性与开放问题
论文承认基准规模有限（265 张图）。我的观察：策略文本都是研究者写的规范表述，真实企业的策略文档往往冗长、自相矛盾、还带大量业务黑话，模型能否在这种噪声下保持策略敏感是个开放问题；另外只做了图片，视频和图文混排还没覆盖。

### 启发与应用前景
这个思路对任何多租户内容平台都直接可用：一套模型服务多个业务方，各家策略不同，不必为每个客户单独训一个审核模型。项目页 https://policyshiftguard.github.io/ 有 demo。

---

## 27. MetaView: Monocular Novel View Synthesis with Scale-Aware Implicit Geometry Priors
**👍 34** · 🏛 南洋理工大学 / 快手 / 香港科技大学（广州） · [arXiv](https://arxiv.org/abs/2607.12000) · [GitHub](https://github.com/KlingAIResearch/MetaView)

### 问题与动机
给一张照片，让 AI 生成「从旁边 90 度看过去是什么样」——这就是单目新视角合成（NVS）。现有做法分两派且各有死穴：**显式几何派**先重建点云再渲染，几何一致但视角一大就露馅（重建不出来的区域直接是洞）；**隐式生成派**让扩散模型自由发挥，画面好看但相机控制不准，你说转 30 度它可能转了 50 度。MetaView 要的是两头都占。

### 方法与核心创新
核心是「隐式几何先验 + 最小必要显式线索」的组合拳。具体做法：用一个前馈几何感知网络（DUSt3R/VGGT 那一类，一次前向就出深度和位姿）抽出**几何 token**，把它作为软约束喂进扩散模型来正则化结构——注意是正则化，不是拿它渲染，所以不受重建失败的拖累；同时引入**度量深度**（metric depth，带真实物理尺度的深度，单位是米不是相对值）把生成锚定在真实尺度上。

举个例子：SLAM 系统里有个经典毛病叫尺度漂移——你估出来的场景是「相对尺度」，可能整体比真实小 3 倍，这时你说「相机前进 1 米」，投影出来的画面其实相当于前进了 3 米。MetaView 用度量深度就是把这个尺子标定死，让「转 30 度」在生成的画面里真的是 30 度。论文还提出新指标 **DMD（Dense Matching Distance，稠密匹配距离）**，因为 PSNR 这类低层指标在大视角变化下基本失效——两张图内容对但位置偏了，PSNR 照样很低，分不出「画错了」和「画对但没对齐」。

### 关键实验结果
在 DL3DV 上按视角重叠率分成 easy（>80% 重叠）/medium（50-80%）/hard（<50%）三档，全面碾压六个基线。最难的 hard 档：**PSNR 12.54 对最佳基线 PE-Field 的 11.97**，提升不算夸张；但 **DMD 从最好的 34.45（HY-World-1.5）降到 20.74，降幅 40%**——这说明它的优势主要在空间对齐而非像素保真。easy 档更明显，DMD **2.56 vs Gen3C 的 6.52，好了 2.5 倍**。论文已被 ECCV 2026 接收。

### 局限性与开放问题
论文附录 C 自己承认两条：受基础模型能力限制，遇到预训练里没见过语义的复杂域外场景会外推失败；在远景开阔场景中尺度估计本身不可靠，锚定也就失效了。我补一条：整套方法依赖前馈几何网络的质量，纯反光/纯白墙这类几何网络本身就崩的场景，先验会变成噪声先验。

### 启发与应用前景
最直接的落地是电商和房产的「一张图转多视角」。代码在快手 KlingAI 名下开源，项目页 https://prototypenx.github.io/MetaView/。

---

## 28. Trust Region Policy Distillation
**👍 33** · 🏛 香港科技大学（广州） / 微软 · [arXiv](https://arxiv.org/abs/2607.04751)

### 问题与动机
On-Policy Distillation（OPD，在线策略蒸馏）是当下小模型后训练的主流范式：**学生自己生成回答，老师模型当场给每个 token 打分，学生照着改**——比传统离线蒸馏（学生背老师的标准答案）更贴合学生的真实分布。但它有个致命毛病：训练极不稳定、方差爆炸。原因在奖励的数学形式：token 级奖励是 log(老师概率 / 学生概率)，当学生给某个 token 的概率趋近 0 时，这个 log 直接冲向负无穷。**一个 token 的离谱奖励就能把整个 batch 的梯度带偏。**

### 方法与核心创新
TOP-D 的解法优雅得近乎取巧：不直接拿老师当监督目标，而是每一步动态构造一个**「近端老师」**——把老师和学生的概率分布做线性插值，π̃ = α·π_teacher + (1−α)·π_student。

这一步改动直接把奖励变成 log(α·ρ + 1−α)，其中 ρ 是老师/学生的概率比。关键在于：**无论 ρ 怎么趋近 0，这个值都不会低于 log(1−α)**，天然有下界，方差爆炸从数学上被堵死了。论文特意说明为什么在概率空间插值而不是对数概率空间——后者算出来还是无界的。

举个例子：这就像教小孩解奥数题。标准 OPD 是直接拿竞赛冠军的解法当标准答案对照，孩子写出个离谱步骤时，「差距」大到没法量化，梯度直接炸。TOP-D 则是每一步都取一个「比孩子当前水平强一点点、但不至于遥不可及」的参照系——摘要开头那句「大目标一次达不成，拆成小步更明智」讲的就是这个。另外还叠加了内部信任域迭代（借鉴 TRPO 的思路，限制每步更新幅度）。论文给出了有界方差、全局收敛、单调改进三个定理，且**零额外计算开销**——只是改了 reward 的计算式，不多跑一次前向。

### 关键实验结果
数字很硬。Qwen3-8B-Base 学生 + Qwen3-30B-A3B-Instruct 老师，**AIME24 上 avg@32 准确率 50.42%，比标准 OPD 的 24.58% 高出 25.84 个百分点**——翻了一倍还多；对比 RLVR 路线的 DAPO（32.92%）也高 17.5 点。AMC23 从 76.88 提到 88.13（+11.25），MATH-500 从 87.98 到 91.23（+3.25）。换成 1.7B 小学生同样成立：AIME24 从 OPD 的 8.96 提到 20.31。注意基座只有 9.38，说明提升确实来自训练而非模型底子。

### 局限性与开放问题
论文没披露插值系数 α 的敏感性分析——α 太小近端老师退化成学生本身，学不到东西；太大又回到原始 OPD 的不稳定，这个甜点区多宽是关键工程问题。另外全部实验只在数学推理上做，代码、Agent 这类奖励更稀疏的任务是否同样有效未验证。

### 启发与应用前景
如果你的团队正在跑 OPD 蒸馏且苦于 loss 曲线抽风，这是零成本的替换项——改 reward 计算那几行即可。暂无开源代码。

---

## 29. AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification
**👍 32** · 🏛 上海人工智能实验室 / 上海交通大学 / 香港中文大学 / 大湾区大学 · [arXiv](https://arxiv.org/abs/2607.11849)

### 问题与动机
大模型刷爆 AIME、IMO 这类竞赛题后，一个尴尬的事实是：这些题都有**唯一的最终答案**，评测只要对一下数字就行。但真正的高等数学不是这样——研究生资格考试和科研里的题目是「证明某某成立」，没有标准答案可对，只有一条推理链的对错。现有基准既覆盖不到本科以上的学科广度，评估又只能靠「最终答案对不对」或粗糙的整体打分，**推理过程本身的有效性完全没被测量**。一个模型可以靠三步错误互相抵消得到正确结论，现有评测给满分。

### 方法与核心创新
这套基准是三件套：**ProverBench** 296 道证明题，覆盖本科（UGD）和博士资格考（QE）两个难度层；**自动验证流水线**，用大规模专家标注训出来的，不只给「对/错」，还能定位具体是哪类证明错误（比如引理误用、边界情况漏讨论），在留出的证明轨迹上与人类专家一致性很高；**VerifierBench** 888 条模型生成的证明轨迹配专家标注的正误，专门测「模型会不会当审稿人」。

举个例子：一道题问「证明某函数在闭区间上一致连续」，模型写了五步。旧评测看到最后写着 QED 就算过；这套流水线会指出「第三步默认了函数可导，但题设只给了连续」——这种错在本科作业里是典型扣分点，模型却经常犯。把「生成证明」和「验证证明」拆成两个独立能力分开测，是这篇最有价值的设计决策。

### 关键实验结果
前沿模型在这上面依然吃力。证明生成上表现最好的 **GPT-5.5-xhigh 在本科题只拿 75.8、博士资格考题掉到 66.1**——对比它在 AIME 上接近饱和的表现，落差非常明显，说明「答对数」和「证得对」是两种能力。证明验证更糟：**最好的模型 Balanced F1 只有 65.1**，且普遍**真阴性率极低**——翻译成人话就是，模型看到一份错误证明时，倾向于说「看起来没问题」。这对当下流行的「用模型当裁判」（LLM-as-judge）和「自我验证」路线是个直接警告：验证者本身是链条上最弱的一环。

### 局限性与开放问题
论文承认 296 题的规模对「高等数学」这个覆盖面来说偏小。我的观察：验证流水线本身也是训出来的模型，用模型评模型存在天花板——它与专家的一致性在留出集上强，但对全新数学分支的错误类型能否泛化，缺乏证据。

### 启发与应用前景
对做数学/科研 Agent 的团队，最实用的结论是别指望模型自查证明——低真阴性率意味着自我验证环节几乎起不到过滤作用，还是得上形式化验证器（Lean 之类）或人工。暂无开源地址。

---

## 30. KronQ: LLM Quantization via Kronecker-Factored Hessian
**👍 32** · 🏛 南加州大学 / 耶鲁大学 · [arXiv](https://arxiv.org/abs/2607.07964) · [GitHub](https://github.com/Intelligent-Computing-Lab-Panda/KronQ)

### 问题与动机
训练后量化（PTQ，不重新训练直接把模型权重压成低比特）现在的主力方法是 GPTQ 一系，它们的量化目标全部只用**输入激活的统计量**来构造。这背后有个从没被质疑的隐含假设：一层里所有输出通道对最终损失的贡献是一样的。KronQ 直接指出这是错的——有些输出通道量化坏了模型无感，有些坏一点就崩，而这个信息藏在**梯度**里，激活统计量根本看不到。

### 方法与核心创新
理论出发点是 Kronecker 因子分解的 Hessian 近似（K-FAC，把巨大的二阶导矩阵拆成「输入协方差 ⊗ 梯度协方差」两个小矩阵的克罗内克积，否则算不动）。这个分解一摊开就说明白了：量化损失同时取决于激活协方差**和**梯度协方差，而现有方法把后一半整个扔了。

KronQ 在两个层面利用这一半信息。其一，**双向非相干处理**：GPTQ 系用随机旋转矩阵把输入维度上的权重「摊平」（减少极端离群值，因为离群值是量化误差的主要来源），KronQ 把这套旋转靠梯度协方差扩展到输出维度，两个方向一起摊。其二，**跨层混合精度分配**：用梯度和激活两个 Hessian 的迹推出一个新的敏感度指标，据此决定哪些层给 3 bit、哪些层能压到 2 bit。

举个例子：把权重矩阵想象成一张地形图，量化就是把海拔离散成有限个高度档。旧方法只沿着东西方向把山削平，南北方向的悬崖还立着——落在悬崖上的权重量化误差极大。KronQ 是东西南北都削一遍。

### 关键实验结果
最能说明问题的是极限场景：**LLaMA-3-70B 做 2-bit 纯权重量化时，GPTQ 和改进版 GPTAQ 直接崩溃——WikiText-2 困惑度超过 2000（等于胡言乱语），而 KronQ 拿到 7.93。** 这不是「高了几个点」的量级差异，是「能用 vs 完全不能用」的分界。2-bit 意味着 70B 模型的权重从 140GB（FP16）压到约 17.5GB，相当于把原本要 2 张 H100 才装得下的模型塞进一张消费级显卡。

### 局限性与开放问题
论文的方法需要梯度协方差，意味着量化流程里得跑反向传播，这比纯前向的 GPTQ 成本高——摘要没披露量化本身的耗时开销，这是实际选型时的关键参数。另外只报了 2-bit 权重量化的极端案例，4-bit 这个工业界主流档位下相对 GPTQ 的增量有多大没在摘要中给出。

### 启发与应用前景
「量化目标里加入梯度信息」这个思路可以直接迁移到 KV cache 量化和激活量化上。对端侧部署团队，2-bit 70B 能跑通意味着单卡本地跑大模型的可行性又前进一步。

---

## 31. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation
**👍 31** · 🏛 南京大学 / 快手 / 新加坡国立大学 / 香港科技大学（广州） · [arXiv](https://arxiv.org/abs/2607.14189) · [GitHub](https://github.com/zxhhh0201/MultiRef-Compass)

### 问题与动机
MR2AV（多参考图转音视频）是个正在冒头的新设定：你给模型好几张参考图——这个人、这只猫、这个房间——再加一句指令「让他在这个房间里逗猫，配上猫叫和笑声」，模型要同步生成画面和声音。现有基准要么只测文生视频，要么只测单参考的主体保持，要么孤立地测音画对齐，**没有一个覆盖「多个参考同时保真 + 正确绑定 + 音画同步」这个组合难题**。

### 方法与核心创新
MultiRef-Compass 用一条可扩展的「素材包组合」流水线构造了 350 条样本——预先收集主体包、物体包、场景包、音频包，再按元数据规则匹配组合，好处是可控且能持续扩产。覆盖三类硬骨头：多视角同一主体的保持、多实体绑定、人-物-场景三者合成。

评估分四个维度共 14 个子指标：基础画质、参考一致性、音画一致性、指令遵循。裁判用的是**带「重判」机制的 MLLM-as-Judge**——大模型第一次打分后，再让它对照证据复核一遍，提升逐条评分的一致性和可审计性。

举个例子：**多实体绑定**这个能力最容易理解——你给了图 A 是穿红衣的女生、图 B 是穿蓝衣的男生，指令说「女生把杯子递给男生」。模型常见的错法是把红蓝衣服穿反了，或者更糟，把两张脸融成一个人。论文里就观察到 Veo 对测试集中若干亚洲人脸参考的身份保持较弱，而后来的 Gemini-Omni 版本在这些样例上有改善——这种细粒度归因正是这类基准的价值。

### 关键实验结果
八个代表性 MR2AV 系统全都有明显短板，没有一家通吃。音画同步维度上 **Gemini-Omni 的 SLS 得分 4.53 最高，明显甩开 Seedance 2.0 的 3.17 和 Kling 3.0 的 2.86**；开源模型差距更大，Wan2.1-VACE 的视觉时序质量 0.1633 是全场最低，参考一致性 0.5204 也垫底，比闭源最好的 Kling 3.0（0.6160）低约 9.5 个百分点。基准与人类偏好的一致性很强：四个维度的皮尔逊相关系数在 **0.898 到 0.964** 之间，指令遵循维度最高。另外 HappyHouse 1.1 的输出「精致但明显合成感重」——这种人眼一看就知道、指标却测不出的问题，正是需要 MLLM 裁判的地方。

### 局限性与开放问题
论文自己承认这是个受控诊断基准，不覆盖所有创作领域、文化风格和参考模态，且为了跨模型可比性把每条样本统一标准化成三个参考。我的观察：Seedance 2.0 和 Gemini-Omni 因内容安全过滤只在 282 和 245 条样本上评测，与其他模型的 350 条不完全可比，这个偏差论文虽已标注但影响排名解读。

### 启发与应用前景
「素材包组合」的数据构造方式值得借鉴——想扩 10 倍样本不用重新收集，加素材包重组即可，这对任何需要持续更新的评测集都适用。

---

## 32. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
**👍 30** · 🏛 洛桑联邦理工学院 · [arXiv](https://arxiv.org/abs/2607.08317) · [GitHub](https://github.com/matteosantelmo/reasoning-blind-spots)

### 问题与动机
一个长期存在但很少被系统测量的怪现象：模型能解博士级数学题，却做不了人类觉得毫无难度的事——比如「把这个字符串倒过来」「画一只有五条腿的狗」。这类失败不是能力不足，是**盲区**：模型的训练分布里这类任务要么太罕见，要么被强先验覆盖（狗就该四条腿，画五条腿会被先验拽回四条）。现有基准清一色测「难任务」，恰恰系统性地漏掉了这些「简单但做不到」的洞。

### 方法与核心创新
数据来源很特别——不是研究者拍脑袋编的，而是**从一门 AI 课程的学生那里征集原始问题**。学生们天天用这些模型，最清楚它们在哪些地方翻车，这种「用户实测吐槽」比实验室设计更容易命中真实盲区。作者把征集来的问题清洗、标注上结构化的参考答案，再根据数据本身归纳出任务分类法（而不是先定分类再填题），最终 235 条样本。配套一条自动打分流水线，覆盖开源和闭源的语言模型、视觉语言模型、图像生成模型三类。

举个例子：「画一只有五条腿的狗」为什么难？因为图像生成模型学到的「狗」这个概念里，四条腿是极强的联合先验，你在提示词里写 five legs，模型的采样过程会不断把腿数拉回训练分布的众数。这跟 LLM 数不清「strawberry 里有几个 r」是同一类问题——**都是强先验压过了显式指令**，只是一个发生在像素空间一个在 token 空间。

### 关键实验结果
最有价值的发现是：**闭源前沿模型比开源模型高出约 10 个百分点，而这两批模型在现有主流基准上分数是接近的。** 换句话说，现有基准已经饱和到分辨不出真实差距，盲区测试反而成了更灵敏的区分器——这对模型选型有直接指导意义。更细的分析显示，**没有任何一个模型在所有任务类型上占优**（各家盲区位置不同），而且**有些任务所有被测模型都做不出来**，说明这不是某家工程没做好，是当前范式的共性缺陷。

### 局限性与开放问题
235 条样本规模确实小，论文把它定位成诊断性压力测试而非综合排行榜。我的观察：来自单一课程的学生征集会带来样本偏差——这批题目大概率集中在字符串操作、计数、反常识生成这几个已经广为人知的失败模式上，真正「无人知晓的盲区」未必能被这种方式挖出来；另外样本量太小时，10% 的开闭源差距落到绝对值只有约 23 题，置信区间不窄。

### 启发与应用前景
最实用的用法是当上线前的 sanity check：你的产品如果涉及精确计数、字符级操作或反常识生成，先跑一遍这类题，别等用户发现。「向重度用户征集失败案例」这个数据收集方法本身也值得任何模型团队直接照搬成内部流程。

---

## 🗺️ 趋势洞察

### 1. On-policy 蒸馏集体爆发：绕开「RL 太贵」的三条路

同一周内出现三篇专攻 on-policy distillation（学生自己生成答案、老师逐 token 当场打分纠正）的工作，且分别从不同角度切入同一个痛点：强化学习（RL）后训练本身已经变成瓶颈——目标模型每训一步都要自己生成大量 rollout，模型越大越烧钱，而最终只拿到一个「对/错」的稀疏信号。[4] 走「弱到强」：在便宜的小模型上跑完 RL，再把成果直接迁移给大模型，省掉大模型的 rollout 开销；[9] 走「自演化」：让 agent 在多轮工具调用中自己产出中间监督，补上「episode 级奖励 vs token 级更新」之间的空档；[28] 走「稳定性理论」：用动态构造的「近端老师」把 OPD 这个出了名高方差的过程约束进信赖域。

**涉及论文**：[4], [9], [28]
**核心观点**：2026 年后训练的主线正在从「怎么把 RL 跑得更好」转向「怎么用更密的监督信号替代 RL 的稀疏奖励」。蒸馏不再是压缩模型的手段，而是 RL 的性价比替代品。

### 2. RL 的两个规模边界同时被推：上下文长度和参数量

[2] LongStraw 把 RL 后训练的上下文从业界常见的 256K 推到 200 万 token 以上，而且是在固定 GPU 预算下做到的——它点破了一个尴尬现实：推理侧已经在跑百万级上下文，训练侧却还停在 256K，中间靠「长度泛化」硬撑。这对 agent 影响最大，因为 agent 的观察、工具返回、历史决策是一路累加的。[8] Ring-Zero 则往参数量方向推，把 zero RL（不用人工标注、纯靠可验证奖励激发思维链）做到万亿参数规模，去看大规模下的训练动态和涌现能力究竟长什么样。

**涉及论文**：[2], [8]
**核心观点**：RL 长期被算力约束在「小模型 + 短上下文」的角落里，这周两篇同时告诉你这两堵墙都能推。推完之后，第一条趋势里「RL 太贵」的前提是否还成立，会是接下来最值得看的问题。

### 3. Agent 的竞争重心从模型移到「模型外面那一层」

[1] 是本周最高赞，论点很直接：agent 的能力不只取决于基础模型，还取决于 harness——那套负责拼 prompt、管状态、调工具、协调执行的代码。而 harness 会随模型、API、环境不断变，改之前你得先找全「实现这个行为的所有代码位置」，这件事在大型耦合代码库里极难。同一主题下，[11] 给机器人做了一个 agent 操作系统层（规划、记忆、工具、验证、跨本体执行），[16] 解决搜索 agent 陷入重复循环、浪费预算的问题，[17] 做带自演化记忆和技能的跨平台 GUI 助手，[22] 做端侧可持续累积的多模态生活记忆，[24] 则把碎片化的 agent 评测管线统一成一套基建。

**涉及论文**：[1], [11], [16], [17], [22], [24]
**核心观点**：六篇论文各做各的，但指向同一件事——agent 的瓶颈已经不在「模型够不够聪明」，而在它周围的运行时：记忆怎么存、状态怎么追、失败怎么退出、代码怎么改、效果怎么测。这一层正在被系统化，甚至开始出现「OS」这种自我定位。

### 4. 视频生成被当成视觉的「预训练范式」，同时挨了一记警告

[13] 直接类比 NLP：next-token prediction 把 NLP 从任务专用模型推成了通用基础模型，那计算机视觉的等价催化剂是什么？它的答案是大规模文生视频——视频生成天然提供时空先验和视觉-语言对齐。[23] 小米把这套思路落到具身：用世界基础模型合成满足多视角一致性和机器人本体约束的数据；[20] 用视频扩散补 4D 人体-场景重建里没被相机看到的区域；[27] 用隐式几何先验做单目新视角合成。但 [21] 在同一周唱了反调：世界-动作模型「把动作和未来预测耦合起来」常被视为鲁棒性和安全性的来源（因为动作可以拿想象中的未来去核对），这篇论文证明这个假设站不住——模型可以「梦对了但做错了」。

**涉及论文**：[13], [20], [21], [23], [27]
**核心观点**：视频生成作为视觉通用先验的路线正在被大量押注，但 [21] 提醒：用「模型能想象出正确未来」来推断「模型会做出正确动作」，是一个没有被验证的跳跃。

### 5. 评测在通胀，但方向变了：从「答案对不对」转向「错在哪、盲在哪」

32 篇里有 8 篇是 benchmark 或评测基建，占比四分之一。更值得注意的是它们的共同转向：[14] 不再只看终端任务的最终结果，而是用密集奖励给中间进度打分；[15] 把文档长度、版面复杂度、模态、问题难度拆成可控变量，好把失败归因到具体原因；[29] 不满足于最终答案正确，要验证数学证明的推理过程是否成立；[32] 专门去测那些人类觉得毫无难度、模型却做不了的任务（比如操作字符串、画一只五条腿的狗）；[26] 则指出安全护栏一直被当成「图片本身是否有害」来训练，而真实部署里同一张图在不同产品、不同政策版本下结论完全不同。[24], [25], [31] 补齐 agent、关键帧视频、多参考音视频三个方向的评测空缺。

**涉及论文**：[14], [15], [24], [25], [26], [29], [31], [32]
**核心观点**：最终答案正确率这个指标正在集体失效——不是因为它不准，而是因为它不再有区分度，也说不清模型为什么错。细粒度归因和盲区挖掘正在成为新的评测共识。

### 对比与张力

- **绕开 RL vs 硬推 RL**：[4] / [9] / [28] 的共同前提是「RL rollout 太贵，得用蒸馏替代」，而 [2] / [8] 的做法恰恰是正面把 RL 的算力墙推开。这两条路线在成本假设上是直接冲突的：如果 LongStraw 式的工程优化能让长上下文 RL 变成常规操作，蒸馏派的性价比论证就会被削弱一大截；反过来，如果万亿参数 zero RL 的收益边际递减，蒸馏路线会成为主流。
- **世界模型的乐观派 vs 怀疑派**：[13] / [23] 押注「生成式世界先验能迁移成通用视觉/具身能力」，[21] 用实证指出耦合世界预测并不等于动作安全。这不是路线之争，而是一个尚未解决的验证缺口——目前没人能说清「想象得对」和「做得对」之间的因果链条。
- **通用大模型 vs 小而专**：[19] 用 0.8B 做端到端文档解析、[22] 做端侧轻量记忆，跟 [8] 的万亿参数形成鲜明反差。前者赌的是特定任务上「小模型 + 好数据引擎」足够，后者赌的是规模本身会带来涌现。
- **视觉信息该进哪一端**：[18] 主张把图表、排版公式这些视觉信号直接喂进**语言模型的预训练**（因为转成纯文本会丢信息），而 [15] / [19] 的路线是训练更强的**视觉文档模型**去解析。同一个问题，一个从预训练侧解，一个从下游模型侧解。

### 值得关注的研究方向

1. **On-policy 蒸馏的理论与工程收敛**：[28] 给了方差控制的理论框架，[4] 给了跨规模迁移的实证，[9] 给了 agentic 场景的适配。三者还没有被统一起来——谁能把「什么时候该用蒸馏、什么时候该用 RL」讲成一个可判定的准则，会是很有价值的工作。
2. **Agent 运行时的标准化**：[1] 的 harness 可维护性、[24] 的评测基建、[11] 的机器人 agent OS，都在往「基础设施层」收敛，但彼此没有互操作标准。这个位置目前是空的。
3. **长上下文 RL 的下游效应**：[2] 打开了 2M token RL 的口子之后，agent 的训练方式会怎么变？现在大量 agent 训练技巧（上下文压缩、记忆外置、分段 rollout）都是在「训练上下文不够长」的约束下发明的，约束一旦松动，这些技巧里哪些会被淘汰值得追踪。
4. **过程级评测与过程级奖励的合流**：[14] 的密集奖励打分、[29] 的证明过程验证，本质上都在造「能评价中间步骤的裁判」。而这正是 [9] 那类工作需要的监督信号来源。评测和训练在这里有很自然的接口，但目前是两拨人在做。
5. **免训练奖励模型**：[12] 用「能否从生成图反推出原始 prompt」当奖励信号，完全不训练；[10] 用检索去补生成模型的世界知识边界。两者都在绕开「训一个奖励模型」这个昂贵环节，思路可以迁移到视频、音频、代码等其他生成任务。
