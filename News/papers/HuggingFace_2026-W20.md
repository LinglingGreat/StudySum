# HuggingFace Papers — 2026-W20（5/11 – 5/17）

> 来源：https://huggingface.co/papers/week/2026-W20
> 统计日期：2026-05-19
> 筛选条件：upvotes ≥ 30
> 论文数：49

## 目录

1. [MinT：百万 LoRA 策略的训管服一体化基础设施](#1-mint)（👍 213）
2. [MV-Split：千层 DiT 的均值塌缩诊断与残差分裂修复](#2-mv-split)（👍 184）
3. [SenseNova-U1：NEO-unify 原生统一多模态理解与生成](#3-sensenova-u1)（👍 172）
4. [SU-01：简化统一配方达到奥赛金牌级推理](#4-su-01)（👍 149）
5. [MemPrivacy：边-云 Agent 的隐私保护型个性化记忆](#5-memprivacy)（👍 143）
6. [MulTaBench：多模态表格学习基准（文本+图像）](#6-multabench)（👍 137）
7. [δ-mem：基于 delta-rule 的轻量在线记忆](#7-delta-mem)（👍 117）
8. [Qwen-Image-2.0 技术报告](#8-qwen-image-20)（👍 105）
9. [SDAR：自蒸馏的 Agentic 强化学习](#9-sdar)（👍 97）
10. [Flow-OPD：Flow Matching 的在线策略蒸馏](#10-flow-opd)（👍 96）
11. [AnyFlow：基于 Flow Map 的任意步视频扩散蒸馏](#11-anyflow)（👍 91）
12. [Causal Forcing++：实时交互视频的 1-2 步自回归扩散蒸馏](#12-causal-forcing-pp)（👍 88）
13. [MMProLong：长上下文视觉语言模型训练配方](#13-mmprolong)（👍 85）
14. [MACE-Dance：音乐驱动舞蹈视频的运动-外观级联专家](#14-mace-dance)（👍 85）
15. [Soohak：研究级数学的数学家策划评测集](#15-soohak)（👍 77）
16. [SANA-WM：分钟级世界模型的混合线性扩散 Transformer](#16-sana-wm)（👍 74）
17. [RubricEM：超越可验证奖励的 Rubric 引导元 RL](#17-rubricem)（👍 74）
18. [MemLens：多模态长期记忆评测基准](#18-memlens)（👍 71）
19. [CollabVR：VLM 与 VGM 协同的视频推理](#19-collabvr)（👍 68）
20. [AutoTTS：测试时扩展策略的自动发现](#20-autotts)（👍 66）
21. [LPO：Listwise 策略优化的几何统一视角](#21-lpo)（👍 65）
22. [World Action Models 综述：具身 AI 的下一个边界](#22-wam-survey)（👍 62）
23. [HyperEyes：并行多模态搜索 Agent 的双粒度效率 RL](#23-hypereyes)（👍 62）
24. [EVA-Bench：端到端语音 Agent 评测框架](#24-eva-bench)（👍 61）
25. [企业级 World Model：运行时发现 vs 离线训练](#25-cascadebench)（👍 60）
26. [MemEye：视觉中心的多模态 Agent 记忆评测](#26-memeye)（👍 59）
27. [Qwen-Image-VAE-2.0 技术报告](#27-qwen-image-vae)（👍 55）
28. [Darwin Family：MRI-Trust 进化合并的训练-free 推理扩展](#28-darwin)（👍 54）
29. [HumanNet：百万小时人本视频学习](#29-humannet)（👍 51）
30. [TMAS：多智能体协同的测试时计算扩展](#30-tmas)（👍 49）
31. [GCWM：几何冲突解释 LLM 持续后训练遗忘](#31-gcwm)（👍 49）
32. [文本-表格建模预测 Agent 决策](#32-agent-decision-pred)（👍 48）
33. [LIFE 综述：多智能体协作、归因与自演化](#33-life-survey)（👍 46）
34. [WildClawBench：真实长程 CLI Agent 评测](#34-wildclaw)（👍 45）
35. [模型合并的扩展律](#35-merging-scaling-law)（👍 43）
36. [MCP-Cosmos：MCP 环境的世界模型增强 Agent](#36-mcp-cosmos)（👍 43）
37. [STALE：Agent 能否识别失效的记忆？](#37-stale)（👍 42）
38. [Token-Superposition Training：高效预训练](#38-tst)（👍 41）
39. [ROPD：Rubric-based 黑盒在线策略蒸馏](#39-ropd)（👍 40）
40. [Warp-as-History：单视频训练的相机可控视频生成](#40-warp-as-history)（👍 38）
41. [TrackCraft3R：复用视频扩散 Transformer 做密集 3D 追踪](#41-trackcraft3r)（👍 36）
42. [AlphaGRPO：UMM 的自反思多模态生成](#42-alphagrpo)（👍 35）
43. [DRoRAE：多层视觉特征融合的 Tokenizer](#43-drorae)（👍 33）
44. [PaperFit：视觉闭环的论文排版优化](#44-paperfit)（👍 32）
45. [Edit-Compass：图像编辑与奖励建模统一基准](#45-edit-compass)（👍 31）
46. [Many-Shot CoT-ICL：上下文测试时学习的扩展规律](#46-many-shot-cot-icl)（👍 30）
47. [WorldReasonBench：视频生成的世界推理压力测试](#47-worldreasonbench)（👍 30）
48. [Pixal3D：图像到 3D 的像素对齐生成](#48-pixal3d)（👍 30）
49. [RouteProfile：LLM 路由的画像设计空间](#49-routeprofile)（👍 30）

[🗺️ 趋势洞察](#-趋势洞察)

---

## 1. MinT：百万 LoRA 策略的训管服一体化基础设施
**👍 213** · [arXiv:2605.13779](https://arxiv.org/abs/2605.13779) · [GitHub](https://github.com/MindLab-Research/mindlab-toolkit) · [Project](https://macaron.im/mindlab/mint)

### 问题与动机
当一个基座模型上要训练并在线服务数百万条 LoRA 策略时，传统做法把每条策略物化成"合并后完整 checkpoint"会带来巨大的存储与切换成本。基座模型加载昂贵、训练/服务/调度/数据流动各自割裂，使得"万策略量级"的实际工程根本跑不通——这是 RLHF/Agent 训练时代正在浮现的核心瓶颈。

### 方法与核心创新
MindLab Toolkit (MinT) 把 base model 常驻、只在 rollout/update/export/eval/serve/rollback 链路中流转 LoRA adapter，三维度扩展：
1. **Scale Up**：把 LoRA RL 扩到 1T 参数级的 dense 与 MoE（含 MLA、DSA 注意力路径）；
2. **Scale Down**：rank-1 时 adapter 不到 base 1%，4B dense 单步缩短 18.3×，30B MoE 缩短 2.85×；并发多策略 GRPO wall-time 1.77× / 1.45×，不增加峰值显存；
3. **Scale Out**：策略可寻址性与 CPU/GPU 工作集解耦，张量并行支持 10^6 量级目录、千 adapter 活跃波，packed MoE LoRA 让 live engine 加载提速 8.5-8.7×。

### 关键实验结果
- 单引擎扫过 100K adapter 实测无瓶颈，集群级千 adapter 活跃；
- 多策略 GRPO 同时跑训练与服务，墙钟降幅 1.4-1.8×；
- 4B 与 30B 两种 backbone 上对单步训练时延实测加速分别为 18.3× 与 2.85×。

### 局限性与开放问题
论文聚焦 LoRA 适配器，对非 LoRA 全参后训练的服务化没回答；million-scale catalog 的冷热分层与跨数据中心同步细节也较少；当 adapter 数继续指数膨胀时，调度公平性与隔离的成本曲线缺位。

### 启发与应用前景
MinT 为"个性化模型即服务"——每个用户/租户/任务一条 LoRA——提供了真正可落地的工程底盘。值得 follow 的方向：(1) 把 MinT 思路扩到 ControlNet/IP-Adapter 这类视觉适配器；(2) adapter 级 KV-cache 共享与 prefix-sharing；(3) 把训练拓扑与服务拓扑统一调度的更激进设计。开源 mindlab-toolkit 是企业 RLHF 平台的现成参考。

---

## 2. MV-Split：千层 DiT 的均值塌缩诊断与残差分裂修复
**👍 184** · [arXiv:2605.06169](https://arxiv.org/abs/2605.06169) · [GitHub](https://github.com/erwold/mv-split) · [Project](https://erwold.github.io/mv-split/)

### 问题与动机
把 Diffusion Transformer 扩到几百层后出现"沉默的塌缩"：训练 loss 看起来正常，但 token 表征被一个共享均值主导、中心化方差被压扁，模型其实已经废了。这种"训练稳定的死亡"在常规监控指标下完全无法察觉，是阻挡 DiT 走向 1000 层的核心障碍。

### 方法与核心创新
作者把这一现象命名为 **Mean Mode Screaming (MMS)**：通过机制审计精确分解残差写入端的梯度，发现 mean-coherent 与 centered 两个分量耦合崩坏；当 token 值同质化后，Softmax Jacobian 零空间又压制了 attention-logit 的梯度，进一步把网络锁死在均值主导态。
提出 **MV-Split Residuals**：把残差写入拆成"独立增益的 centered 残差"和"带 leaky 替换的 trunk-mean"，相当于一种对均值/方差通道分别施加可控反馈的新型 LayerNorm 替代物。

### 关键实验结果
- 400 层单流 DiT 上，未稳定 baseline 训练崩溃，MV-Split 紧贴 baseline 崩溃前轨迹且持续显著优于 token-isotropic 类（如 LayerScale）；
- 推到 **1000 层 DiT** 仍稳定可训，验证了极端深度的可行性。

### 局限性与开放问题
论文以单流 DiT 为主，未广泛验证多流 DiT、文-图条件下的实际生成质量曲线；MV-Split 引入的额外可学参数与显存代价讨论较少；缺乏与 MoE-DiT、Mamba-DiT 等替代深化路径的横向对比。

### 启发与应用前景
"沉默塌缩"在 1000-层 LLM 或大规模视频扩散中很可能更普遍——MV-Split 给出了诊断 + 修复的完整范式。值得 follow：把同样的"均值/方差梯度分解"工具用于 ViT、视频 DiT、3D-DiT；构建"塌缩预警指标"集成到训练监控；探索 mean-coherent 通道是否可被剪掉得到更高效深网络。开源代码与项目页可复用其诊断工具链。

---

## 3. SenseNova-U1：NEO-unify 原生统一多模态理解与生成
**👍 172** · [arXiv:2605.12500](https://arxiv.org/abs/2605.12500) · [GitHub](https://github.com/OpenSenseNova/SenseNova-U1)

### 问题与动机
当前 VLM 把"理解"和"生成"切成两套子系统——独立架构、级联管线、错配表示空间。论文认为这种二分不是工程权宜而是结构性缺陷，阻碍了原生多模态能力涌现：模型只是在"翻译模态"而不是"原生跨模态思考"。

### 方法与核心创新
提出 **SenseNova-U1**，基于 **NEO-unify** 范式：理解与生成被视为同一底层过程的两种协同视角，共享 backbone（不再有独立的生成头）。发布两版：
- **SenseNova-U1-8B-MoT**（dense 8B）
- **SenseNova-U1-A3B-MoT**（MoE 30B-A3B）
论文同时披露完整 data pipeline、pre-/post-train、推理策略，强调"think pattern"是否启用都能切换工作模式。

### 关键实验结果
- 在文本理解、视觉感知、知识推理、agentic decision、空间智能五项上与顶级 understanding-only VLM 持平；
- 在 X2I（任意到图像）合成、知识密集 infographic、交错视觉-语言生成上展示出强语义一致性与视觉保真；
- 早期证据显示其在 VLA（视觉-语言-动作）与世界模型场景同样强势，意味着 NEO-unify 的范式具备外溢能力。

### 局限性与开放问题
论文偏定性论述与多任务展示，缺乏与同期 unified 模型（如 Janus、Show-o、Emu3、Liquid）的逐 benchmark 同台对比；MoT（Mixture of Transformers）切换开销与训练数据 mix 的具体配比披露有限；统一范式下"安全/对齐"如何分摊也未涉及。

### 启发与应用前景
"VLA + WM 直接从 unified 模型涌现"是这篇论文最重要的暗示：未来具身 / 世界模型不必额外搭桥，原生统一 backbone 就足够。值得 follow：用 NEO-unify 思路重做 multilingual VLM、医学图文模型；以及把生成头反向用于"想象式推理"做长程任务。开源 8B 版可作研究 baseline。

---

## 4. SU-01：简化统一配方达到奥赛金牌级推理
**👍 149** · [arXiv:2605.13301](https://arxiv.org/abs/2605.13301) · [GitHub](https://github.com/Simplified-Reasoning/SU-01) · [Project](https://simplified-reasoning.github.io/SU-01)

### 问题与动机
近年频频出现"前沿模型 IMO/IPhO 金牌"的报道，但配方复杂且封闭。社区缺少一个能从 post-trained backbone 出发、可复制达到金牌的统一流程，导致研究无法迭代、教学无法借鉴。

### 方法与核心创新
作者给出一条简单但完整的"成型菜谱"：
1. **Reverse-perplexity curriculum SFT**：用"逆困惑度"难度排序教模型严格证明搜索 + 自我核验；
2. **两阶段 RL**：先 RL with verifiable rewards，再升级到 proof-level RL；
3. **Test-time scaling** 收尾。
全过程基于 30B-A3B backbone，SFT 数据 ~340K 条且每条 <8K token，RL 仅 200 步——比常见百万级 RL 步骤简洁得多。

### 关键实验结果
- 训出的 **SU-01** 在 IMO 2025 / USAMO 2026 / IPhO 2024 / IPhO 2025 上达到金牌级；
- 训练轨迹支持 >100K token 的稳定推理；
- 在数学、物理之外的科学推理任务上展现强泛化能力。

### 局限性与开放问题
论文没披露 RL 的具体 reward 设计、温度与采样细节；30B-A3B 的算力门槛对学术界仍偏高；"金牌级"主要靠 grader 与人工证明审阅，缺乏公开可复跑的全自动评测；340K SFT 数据的来源与版权信息也较模糊。

### 启发与应用前景
"逆困惑度课程"是个被低估的工具，几乎可以无缝迁移到代码、定理证明（Lean/Coq）、医学诊断的 SFT 阶段。短 SFT + 短 RL 的"轻配方"也对 7B/13B 模型友好，值得复现验证。开源仓库提供训练脚本与少量数据，是当前奥赛级开源最接近的入口。

---

## 5. MemPrivacy：边-云 Agent 的隐私保护型个性化记忆
**👍 143** · [arXiv:2605.09530](https://arxiv.org/abs/2605.09530) · [GitHub](https://github.com/MemTensor/MemPrivacy)

### 问题与动机
LLM Agent 在边-云协同场景越来越普遍，"个性化记忆"是长期适配关键。但把记忆推到云端会暴露隐私；用传统激进 masking 又会把"任务相关语义"也抹掉，导致召回与个性化质量崩塌。隐私与效用之间长期被迫二选一。

### 方法与核心创新
**MemPrivacy** 在边端识别隐私敏感片段，用"语义化结构 + 类型感知占位符"替换原文，云端只在占位符上做记忆处理，最终在本地把真实值还原。其核心是"解耦隐私保护与语义破坏"——占位符保留了任务所需信息形态。配套：
- **MemPrivacy-Bench**：200 用户 + 52K+ 隐私实例的系统评测集；
- **四级隐私分类**用于策略可配置。

### 关键实验结果
- 隐私信息抽取上"显著超越 GPT-5.2 与 Gemini-3.1-Pro"等通用强模型；
- 在多种主流记忆系统上，效用损失被限制在 **1.6% 以内**，明显优于各类 baseline masking；
- 同时降低推理时延，证明结构化占位符不是性能负担。

### 局限性与开放问题
四级分类是否覆盖未来更细粒度场景（如医疗 HIPAA、未成年人）尚待验证；端侧识别 model 本身的鲁棒性与对抗样本风险论文涉及不深；占位符的 schema 跨系统迁移性、隐私审计可视化也是后续问题。

### 启发与应用前景
对"端云协同 Agent"的产品来说，MemPrivacy 是一份可直接套用的工程模式，比"全本地推理"或"全云端 GDPR 合规"都更可行。值得 follow：(1) 把占位符思路用于 RAG 知识库的"隐私感知向量"；(2) 与差分隐私 / 同态加密混合；(3) 跨用户场景的"零知识共享"。开源代码与 Bench 是隐私研究的稀缺基座。

---

## 6. MulTaBench：多模态表格学习基准（文本+图像）
**👍 137** · [arXiv:2605.10616](https://arxiv.org/abs/2605.10616)

### 问题与动机
Tabular Foundation Model 已在结构化数据上达到 SOTA，但天然不支持文本和图像非结构化模态——通常只是冻结预训练 embedding 灌入。现有"多模态表格"benchmark 关注模态共现而非任务互补信号，使得"task-specific tuning 是否真的有用"被掩盖。

### 方法与核心创新
作者构建 **MulTaBench**：40 个数据集均分图-表与文-表两类，特意挑选"模态提供互补预测信号、通用 embedding 会丢失关键信息"的场景。配套验证"Target-Aware Representation"在不同 tabular learner、encoder 规模、嵌入维度下的稳定增益。

### 关键实验结果
- 在 40 个数据集上系统证明：对预训练 embedding 做 task-specific tuning 比冻结 + 串接稳定优胜；
- 覆盖医疗、电商等高影响领域，是目前规模最大的图-表 benchmark；
- 文本与图像两侧的增益均能泛化到多个 tabular learner 与 embedding 维度。

### 局限性与开放问题
论文聚焦于评测，没有提出新架构；模态间互补信号的强度本身是数据驱动的，未来要小心 selection bias；"Target-Aware Representation"具体实现细节较省略；中文/多语言表格也未涵盖。

### 启发与应用前景
MulTaBench 让"多模态表格基础模型"研究有了可比较的统一土壤——这是该方向走出实验室的前提。值得 follow：(1) 构建原生"多模态 tabular foundation model"（如统一一个 backbone 处理 num/cat/text/img）；(2) 把 LLM 作为 tabular 端 encoder；(3) 在医疗影像 + EHR 这类组合上做 transfer。

---

## 7. δ-mem：基于 delta-rule 的轻量在线记忆
**👍 117** · [arXiv:2605.12357](https://arxiv.org/abs/2605.12357) · [GitHub](https://github.com/declare-lab/delta-Mem)

### 问题与动机
让 LLM 在长期助手或 Agent 中累积复用历史信息，单靠扩窗口既贵又利用率差，靠 RAG 又被检索质量卡脖子。需要一种"轻量、可在线更新、与注意力直接耦合"的记忆机制。

### 方法与核心创新
**δ-mem** 给冻结 backbone 旁挂一个紧凑的"在线关联记忆状态"：用 **delta-rule** 把过去信息压缩到固定大小的状态矩阵中更新；推理时它的读出通过 **低秩修正**注入 backbone 的 attention 计算。模型无需全参微调、无需替换 backbone，也不要求显式扩窗口。

### 关键实验结果
- 仅 **8×8 在线状态**就把平均分提到 1.10× 冻结 backbone、1.15× 最强非-δ-mem 记忆 baseline；
- 在内存重度任务上提升更大：MemoryAgentBench **1.31×**、LoCoMo **1.20×**；
- 通用能力基本不退化，证明 δ-mem 不是"以广义能力换记忆"。

### 局限性与开放问题
8×8 状态对极端长程多主题对话是否仍足够、是否会冲洗旧记忆缺乏长尾压力测试；delta-rule 与传统 RNN 状态的灾难性遗忘机理是否相同也值得研究；对生成式而非分类式任务的提升曲线偏少。

### 启发与应用前景
"低秩修正 + 极小状态"是非常有吸引力的部署形态：可以为每个用户/会话独立维护一个 8×8 矩阵而几乎无成本，与 LoRA 形成对应的"运行时侧记忆"。值得 follow：把 δ-mem 用到 Agent 工具调用历史压缩、视频长片段缓存；与 Mamba/Linear Attention 的状态合并。

---

## 8. Qwen-Image-2.0 技术报告
**👍 105** · [arXiv:2605.10730](https://arxiv.org/abs/2605.10730)

### 问题与动机
现有图像生成模型在超长文本渲染、多语种排版、高清写实、复杂指令跟随、轻量部署上仍力有未逮，尤其在"文图密集+组合复杂"场景（海报、信息图、漫画）频繁失败。

### 方法与核心创新
**Qwen-Image-2.0** 把 **Qwen3-VL** 当作 condition encoder，配 **Multimodal Diffusion Transformer** 做"条件-目标联合建模"。配以大规模数据策划、多阶段定制训练管线，使其既具备多模态理解，又保留生成与编辑灵活性。最长支持 **1K token** 指令，专门强化幻灯片、海报、信息图、漫画类文本密集生成与多语种字体。

### 关键实验结果
- 多语种文本保真与排版相对前代有显著跃升（人评）；
- 写实细节、纹理、光照一致性大幅改进；
- 跨多样风格的复杂 prompt 跟随更稳；
- 大规模人评显示 Qwen-Image-2.0 在生成与编辑双向都明显超越上代。

### 局限性与开放问题
报告偏工程披露，缺乏与 SD3 / FLUX / Imagen3 / Midjourney 的同 prompt 对照；多语种"字符级"准确率到底相对 SOTA 高多少没有数字；对 NSFW / 著作权风险的防护策略也未涉及。

### 启发与应用前景
"用 VLM 直接做 condition encoder"已经成为 2026 年图像生成的主流范式（参见同周 SenseNova-U1 与 AlphaGRPO），Qwen-Image-2.0 是又一确认。值得 follow：把 1K token 指令能力推向多页 PDF/网页全屏图生成；与视频生成结合做"动态海报"；以及把同一 backbone 推到编辑/inpainting/控制网。

---

## 9. SDAR：自蒸馏的 Agentic 强化学习
**👍 97** · [arXiv:2605.15155](https://arxiv.org/abs/2605.15155) · [GitHub](https://github.com/ZJU-REAL/SDAR)

### 问题与动机
RL 在 LLM Agent post-training 已成主流，但轨迹级 reward 只是粗粒度信号；OPSD（On-Policy Self-Distillation）通过"特权 teacher 分支"提供 token 级密度监督。然而把 OPSD 套到多轮 Agent 上会失败：多轮累积不稳定 + 技能条件下"教师拒绝"未必正确（可能是技能召回/利用不当）。

### 方法与核心创新
**SDAR** 把 OPSD 视作"门控辅助目标"挂在 RL 主干上：把脱离梯度的 token 级信号映射到 sigmoid 门，**对教师认同的正间隙 token 增强蒸馏，对教师拒绝信号则软衰减**。这一非对称处理避免了"naive GRPO + OPSD"的不稳定。

### 关键实验结果
跨 Qwen2.5 / Qwen3 模型族在三类任务上：
- ALFWorld **+9.4%**，Search-QA **+7.0%**，WebShop-Acc **+10.2%** 相对 GRPO；
- 一致优于多种混合 RL-OPSD baseline，跨规模稳定。

### 局限性与开放问题
SDAR 的 sigmoid 门权重如何随训练动态调整未充分披露；teacher 失效率高时门控容易退化为纯 RL；任务集中在 ALFWorld/WebShop 这类相对清晰奖励的场景，对真实复杂工作流的泛化还需验证。

### 启发与应用前景
"非对称蒸馏 + RL"是当前 Agent RL 的一个清晰技术分水岭：单纯 token 级监督正在被"门控的、有条件的、自己评估"的方案取代。值得 follow：把 SDAR 思路用于 tool-use 失败归因、长对话 multi-turn 训练、Coding Agent；与 process reward model 整合做更精细信用分配。开源 SDAR 代码可作 RL Agent 训练新 baseline。

---

## 10. Flow-OPD：Flow Matching 的在线策略蒸馏
**👍 96** · [arXiv:2605.08063](https://arxiv.org/abs/2605.08063) · [GitHub](https://github.com/CostaliyA/Flow-OPD) · [Project](https://costaliya.github.io/Flow-OPD/)

### 问题与动机
Flow Matching 文本-图像模型在多任务对齐时被 reward sparsity（标量 reward 信号太稀）+ gradient interference（异质目标互相打架）夹击，呈现典型的"seesaw effect"：一头提升一头下跌，并伴随 reward hacking。

### 方法与核心创新
**Flow-OPD**：首个把 LLM 社区的 On-Policy Distillation 迁移到 Flow Matching 的统一对齐框架。两阶段：
1. **单 reward GRPO** 培养多个"领域专家"（教师），让每条 reward 单独把模型压榨到天花板；
2. **Flow-based Cold-Start** + **on-policy sampling + task-routing labeling + dense trajectory supervision** 三步把异质专长整合到一个学生。
此外引入 **Manifold Anchor Regularization (MAR)**：用一个"任务无关"教师对全数据监督，把生成锚定到高质量流形上，缓解纯 RL 驱动的审美塌缩。

### 关键实验结果
基于 Stable Diffusion 3.5 Medium：
- **GenEval 63 → 92**；
- **OCR 准确率 59 → 94**；
- 综合相对 vanilla GRPO 约 **+10 点**；
- 还出现"学生超越教师"的涌现现象，保持图像保真与人类偏好对齐。

### 局限性与开放问题
专家模型数量增多时的协调成本未充分讨论；MAR 锚定教师的选择带主观性；OCR 指标这种容易"hack 文字"的项目需小心 reward 设计；对视频或 3D 生成的延展还需验证。

### 启发与应用前景
Flow-OPD 把 LLM 蒸馏哲学完整带入 Flow Matching，对图像/视频 RL 是一个范式升级。值得 follow：把 MAR 应用到 SDXL/Stable Video Diffusion；探究 task-routing label 自动获得而非人工标注；尝试与 [[10-flow-opd]] 一致的多专家融合用于视频生成（参考 [[11-anyflow]]）。

---

## 11. AnyFlow：基于 Flow Map 的任意步视频扩散蒸馏
**👍 91** · [arXiv:2605.13724](https://arxiv.org/abs/2605.13724) · [GitHub](https://github.com/NVlabs/AnyFlow) · [Project](https://nvlabs.github.io/AnyFlow/)

### 问题与动机
Consistency Distillation 让少步视频生成质量大幅提升，但有个怪现象：步数从 4 增加到 8、16 时质量反而下降——consistency 训练把原 ODE 轨迹替换成了 consistency-sampling 轨迹，丢失了"步数越多越好"的天然 scaling。

### 方法与核心创新
**AnyFlow**：首个基于 **flow map** 的任意步视频蒸馏。把蒸馏目标从端点 consistency mapping `z_t → z_0` 改为 **flow-map transition** `z_t → z_r`，覆盖整条 ODE 轨迹任意时段。配套 **Flow Map Backward Simulation**：把完整 Euler rollout 分解为 shortcut flow-map transitions，实现 on-policy 蒸馏并同时压低 (i) 离散化误差和 (ii) causal 生成中的 exposure bias。

### 关键实验结果
跨双向与因果架构、1.3B 到 14B 参数：
- 少步制式上匹配甚至超过 consistency-based 对手；
- **步数继续增加时质量持续上升**——恢复了 ODE 采样的 scaling 属性；
- 大模型上效果尤为明显。

### 局限性与开放问题
flow-map 的训练目标更复杂，训练开销与稳定性比 consistency 更难；在更高分辨率视频上的 GPU 显存与时延曲线披露不足；蒸馏过的 flow map 是否会破坏可控性（如相机控制）需进一步验证。

### 启发与应用前景
AnyFlow 把视频扩散蒸馏从"固定 4 步"解放到"任意步数 + 步数越多越好"，对实时 / 高质量两端都有意义。值得 follow：与 [[12-causal-forcing-pp]] 的 1-2 步 AR 蒸馏合并研究；扩展到 3D / 世界模型（[[16-sana-wm]]）；与 ControlNet 类条件注入耦合验证。

---

## 12. Causal Forcing++：实时交互视频的 1-2 步自回归扩散蒸馏
**👍 88** · [arXiv:2605.15141](https://arxiv.org/abs/2605.15141) · [Project](https://github.com/thu-ml/Causal-Forcing)

### 问题与动机
"实时交互式视频生成"要求低时延、流式、可控 rollout。已有 AR 扩散蒸馏在"chunk-wise 4 步"已取得不错效果，但响应粒度仍粗、采样时延仍可观；要进一步推进必须挑战 **frame-wise + 1-2 步**这一极致设置——而该制式下，少步 AR 学生的初始化是核心瓶颈，已有策略要么目标错位、要么不能少步生成、要么扩展成本爆炸。

### 方法与核心创新
**Causal Forcing++**：用 **causal consistency distillation (causal CD)** 做少步 AR 初始化。核心洞察是 causal CD 学到的 AR-conditional flow map 与 causal ODE distillation 一致，但只需要单步在线教师 ODE 监督，避免预计算并存储完整 PF-ODE 轨迹——初始化既高效又易优化。整条 pipeline 进一步扩展到 Genie3 风格的"动作条件世界模型"。

### 关键实验结果
**frame-wise 2-step 设置**下，相对 SOTA 4-step chunk-wise Causal Forcing：
- **VBench Total +0.1、VBench Quality +0.3、VisionReward +0.335**；
- **首帧时延减半**；
- **Stage 2 训练成本约 4× 降低**。

### 局限性与开放问题
"frame-wise + 1-2 步"对极长视频的累计漂移评估较短；与 [[11-anyflow]] 一类 any-step 蒸馏的优劣关系没有同台对照；动作条件世界模型部分的具体动作粒度与控制接口偏概念化。

### 启发与应用前景
真正的"游戏级实时视频生成"需要的就是 frame-wise 极少步 + 动作条件，Causal Forcing++ 离这个目标比之前任一方案都近。值得 follow：与 [[16-sana-wm]] 类世界模型结合做真正可玩 demo；探索 1-step AR 极限蒸馏；以及与 ControlNet/Video LoRA 链路的兼容。

---

## 13. MMProLong：长上下文视觉语言模型训练配方
**👍 85** · [arXiv:2605.13831](https://arxiv.org/abs/2605.13831)

### 问题与动机
长上下文已成 LVLM 核心能力，用于长文档理解、视频分析、Agent 工具串联。但社区缺一个"data mix 应该长什么样、平衡比例怎么定"的系统配方，导致每家厂商都在重复造轮子。

### 方法与核心创新
对 7B 模型把上下文从 32K 扩到 128K 做系统消融，三项关键发现：
1. **长文档 VQA 远优于 OCR 转录**作为长上下文训练数据；
2. **长度分布要平衡**：刻意瞄准 128K 不如混入各种长度；
3. **检索是主要瓶颈**——检索重头数据更有用，少量推理数据保多样性即可；
4. 纯长 VQA **几乎不损害短上下文能力**。
基于这些发现产出 **MMProLong**：从 Qwen2.5-VL-7B 继续训练，仅用 **5B token** 预算。

### 关键实验结果
- 长文档 VQA **+7.1%**；
- 在 **256K / 512K 上下文**（远超 128K 训练窗口）仍保持强势性能，证明 generalization 良好；
- 进一步泛化到网页多模态 needle retrieval、长上下文视觉-文本压缩、长视频理解，**无需任务专属监督**。

### 局限性与开放问题
研究以 7B 为主，更大模型上的结论不确定；中文等非英文长文档评测覆盖少；视频长度推到何处仍可工作没有明确边界；OCR 失败案例的归因略浅。

### 启发与应用前景
"长 VQA + 长度平衡"是当前社区最稀缺的实证结论，可直接套用到任何继续训练 LVLM 的工程上。值得 follow：把 MMProLong 推到 30B/72B；与 [[18-memlens]] / [[26-memeye]] 的多模态记忆评测交叉；探究"长检索 + 短推理"的最优配比。

---

## 14. MACE-Dance：音乐驱动舞蹈视频的运动-外观级联专家
**👍 85** · [arXiv:2512.18181](https://arxiv.org/abs/2512.18181) · [GitHub](https://github.com/AMAP-ML/MACE-Dance)

### 问题与动机
音乐驱动舞蹈视频生成是当下 AIGC 一个 niche 但商业潜力高的方向。已有方法要么做"音乐到 3D 动作"，要么做"姿态驱动图像动画"，要么做"音频驱动 talking head"，但无法**同时**给出高画质 + 真实人体运动。

### 方法与核心创新
**MACE-Dance**：级联 Mixture-of-Experts 架构。
- **Motion Expert**：音乐到 3D 运动，BiMamba-Transformer 混合 + **Guidance-Free Training (GFT)**，保证 kinematic 合理与艺术性；
- **Appearance Expert**：运动/参考条件下的视频合成，采用"kinematic-aesthetic 解耦微调"，保身份与时空一致。
论文同时贡献了一份大规模、多样的数据集与"运动-外观双轴"评测协议。

### 关键实验结果
- **Motion Expert** 在 3D 舞蹈生成上 SOTA；
- **Appearance Expert** 在 pose-driven image animation 上 SOTA；
- 联合 pipeline 在新提出的协议下也是 SOTA。

### 局限性与开放问题
"Mixture-of-Experts 级联"的实际 GPU 部署成本（两套大型扩散模型）较高；新协议是作者自定义评估，外部第三方验证有限；舞蹈风格覆盖与文化多样性披露不足；GFT 的稳定性表现需更多数据点。

### 启发与应用前景
"音乐 → 运动 → 外观"的明确分解为类似强结构任务（动作教学、康复评估）提供了模板。值得 follow：把 GFT 推广到其他 conditional diffusion；级联专家用于"音频 → 嘴型 → 全身"；与 [[16-sana-wm]] 一类世界模型整合做交互式舞蹈生成。

---

## 15. Soohak：研究级数学的数学家策划评测集
**👍 77** · [arXiv:2605.09063](https://arxiv.org/abs/2605.09063)

### 问题与动机
IMO 已被前沿模型攻克，社区急需"研究级数学"作为下一个北极星目标。但研究级问题"取材难"——Riemann Bench 仅 25 题、FrontierMath-Tier 4 也只有 50 题，统计噪声大。

### 方法与核心创新
**Soohak**：由 **64 位数学家**从零撰写的 **439 题** benchmark，分两个子集：
- **Challenge 子集**：研究级数学；
- **Refusal 子集**（关键创新）：检测模型能否识别"病态/不可解"问题、选择停下来而不是给出"自信但无根据"的答案。
为防污染，数据将延迟至 2026 年底公开发布。

### 关键实验结果
**Challenge** 上前沿模型表现：
- Gemini-3-Pro **30.4%**、GPT-5 **26.4%**、Claude-Opus-4.5 **10.4%**；
- 顶级开源（Qwen3-235B、GPT-OSS-120B、Kimi-2.5）**均 <15%**；
**Refusal** 上：
- 没有任何模型超过 **50%**，说明"知道自己不知道"是新的优化方向。

### 局限性与开放问题
研究级数学定义本身较主观、且部分子领域比另一些更易出 hard problems；评分依赖数学家审稿；439 题相对其他数学 benchmark 算大，但和真实数学研究广度比仍狭窄；模型评测目前仅 on-request。

### 启发与应用前景
**"Refusal subset"** 是这篇论文最重要的方法论贡献——它把"诚实"作为可量化能力提到台前，这对 AI for Science / 法律 / 医学评测都有外溢价值。值得 follow：把 Refusal 思路移植到代码、化学、定理证明；构建跨学科 hard-question pool；研究 Refusal 与 Test-Time Reasoning 的关系（参 [[4-su-01]]）。

---

## 16. SANA-WM：分钟级世界模型的混合线性扩散 Transformer
**👍 74** · [arXiv:2605.15178](https://arxiv.org/abs/2605.15178) · [Project](https://nvlabs.github.io/Sana/WM/)

### 问题与动机
"世界模型"是具身/游戏的下一站，但分钟级、720p、精确相机控制、可在单卡部署同时实现的开源方案缺位；现有大型工业基线（LingBot-World、HY-WorldPlay）质量好但消耗夸张。

### 方法与核心创新
**SANA-WM**：**2.6B** 参数的开源世界模型，原生训练 1 分钟生成、720p。四个关键设计：
1. **Hybrid Linear Attention**：帧级 Gated DeltaNet (GDN) + softmax attention，内存高效；
2. **Dual-Branch Camera Control**：精确 6-DoF 相机轨迹；
3. **Two-Stage Generation Pipeline**：长视频 refiner 修补 stage-1；
4. **Robust Annotation Pipeline**：从公共视频提取 metric-scale 6-DoF 相机姿态做高质量动作标注。

### 关键实验结果
- 仅 **~213K 公共视频片段**带 metric pose 监督；
- **64×H100 训 15 天**完成全部训练；
- 单 GPU 生成 60 秒视频；
- 蒸馏版可在 **单张 RTX 5090** + NVFP4 量化下，**34 秒**完成 60 秒 720p 去噪；
- 相比开源 baseline 动作跟随更准，在视觉质量持平的同时 **吞吐 36×**。

### 局限性与开放问题
"分钟级"在游戏世界模型可能仍不够长；硬件友好评测主要在 H100/5090，对消费级 GPU 仍偏紧；动作语义粒度（如复杂物理交互）仍以相机为主，关节级/工具级动作支持待补；视觉一致性长程评测样本量有限。

### 启发与应用前景
SANA-WM 把"分钟级世界模型"的门槛拉到了真正可复现的水位，是开源具身研究的重要节点。值得 follow：与 [[22-wam-survey]] 配套读；将 hybrid linear attention 思路移到 Vid-DiT；与 [[40-warp-as-history]] 的相机控制交叉对照。

---

## 17. RubricEM：超越可验证奖励的 Rubric 引导元 RL
**👍 74** · [arXiv:2605.10899](https://arxiv.org/abs/2605.10899)

### 问题与动机
训练"深度研究 Agent"——会规划、检索、评证据、写长报告——已超越 verifiable reward 的能力范围：没有标准答案、多工具决策轨迹长、过往尝试无机制转化为可复用经验。

### 方法与核心创新
**RubricEM** 把 rubric 提升为"贯穿策略执行 / judge 反馈 / Agent 记忆的共享接口"：
1. **分阶段策略分解**：planning / evidence gathering / review / synthesis 各自条件于自生成 rubric；
2. **Stage-Structured GRPO**：用 stage-wise rubric 判断分发更密的语义信用；
3. **Reflection meta-policy**：共享 backbone 的元策略把判定过的轨迹蒸馏成"rubric-grounded guidance"反哺下一次尝试。

### 关键实验结果
- **RubricEM-8B** 在 **四个长形式研究 benchmark** 上稳定领先同规模开源模型；
- 性能逼近闭源专属 deep-research 系统；
- 论文还给出关键成分的 ablation。

### 局限性与开放问题
rubric 自生成对模型能力本身敏感，弱模型上可能失败；元策略的 generalization 范围（跨主题、跨语言）评估有限；"逼近闭源"是定性描述，缺少 head-to-head 严格对照；推理时调用成本未细化披露。

### 启发与应用前景
Rubric 既当"评估接口"又当"记忆接口"是该论文最具传播价值的思路，可外溢到 Coding Agent、Math Tutor、医学诊断等"非 verifiable"领域。值得 follow：把 RubricEM 与 [[20-autotts]] 类自动 TTS 发现结合，构建"自发现 rubric + 自发现策略"的双层 Agent；与 [[34-wildclaw]] 长程 Agent 评测交叉验证。

---

## 18. MemLens：多模态长期记忆评测基准
**👍 71** · [arXiv:2605.14906](https://arxiv.org/abs/2605.14906) · [GitHub](https://github.com/xrenaf/MEMLENS)

### 问题与动机
多模态长程对话场景下，长上下文 LVLM 与"记忆增强 Agent"是两种主流路线，但没有同台 benchmark 真正考"必须用视觉证据"的题，导致两派优劣不可见。

### 方法与核心创新
**MEMLENS** 共 **789 题**覆盖五种记忆能力（信息抽取、跨会话推理、时序推理、知识更新、Refusal），四档上下文长度（32K-256K，统一跨模态 token 计数）。最关键的是 **image-ablation 设计**：去掉证据图，两 frontier LVLM 在含图的 80.4% 题上准确率 **掉到 2% 以下**——证明任务确实"视觉必要"。

### 关键实验结果
评测 **27 个 LVLM + 7 个 memory Agent**：
- 长上下文 LVLM 在短上下文内通过直接视觉 grounding 取得高分，但对话变长后退化；
- 记忆 Agent 长度更稳定但"存储期压缩"折损视觉细节；
- 跨会话推理大多卡在 **<30%**；
- 没有任一路线单独解决问题，论文呼吁混合架构。

### 局限性与开放问题
789 题虽合规但任务多样性还能更广；长上下文 vs Agent 的 hybrid 还停留在结论层面，没有给出具体新架构；评测主要英语；视觉证据的"细粒度"分级也偏粗。

### 启发与应用前景
MEMLENS 是当前最干净的"多模态长记忆"评测，几乎所有 Agent / RAG / VLM 厂商都该跑一遍。值得 follow：与 [[26-memeye]] 联读看视觉中心记忆评测；与 [[37-stale]]（记忆是否过期）配套；以及构建"长上下文 + 记忆 retrieval 混合"的实际系统。

---

## 19. CollabVR：VLM 与 VGM 协同的视频推理
**👍 68** · [arXiv:2605.08735](https://arxiv.org/abs/2605.08735) · [GitHub](https://github.com/Joow0n-Kim/CollabVR) · [Project](https://joow0n-kim.github.io/collabvr-project-page/)

### 问题与动机
"用视频生成模型当推理工具"近期热度上升（Chain-of-Frames），但 VGM 在目标导向任务上有两个反复出现的失败：长程漂移与片中模拟错误。原因是 VGM 缺乏"显式推理"，而 VLM 显然合适——但 VLM 该放在哪？放在前面"预先规划"会过早承诺、放在后面"事后批评"又太迟。

### 方法与核心创新
**CollabVR**：闭环、step-level VLM-VGM 协同框架。VLM 在每一步只规划"下一动作 + 检查 VGM 刚生成的一帧"，把诊断直接折叠进下一步 prompt 修复检测到的失败——一种"边走边查、边查边修"的协作范式。

### 关键实验结果
在 **Gen-ViRe / VBVR-Bench** 上：
- 开源和闭源 VGM 均稳定提升，且 hardest task 增益最大；
- 在相同计算预算下超过 single-inference、Pass@k 和已有 test-time scaling baseline；
- **叠加在 reasoning-finetuned VGM 之上仍有进一步增益**——证明 step-level VLM 监督与 fine-tuning 是正交叠加而非互相替代。

### 局限性与开放问题
每步调用 VLM 增加显著推理时延；step-level prompt 设计本身需要工程经验；目前实验主要是 short-horizon 视频任务，长程一致性提升曲线还需要更多样本。

### 启发与应用前景
"VLM 当 verifier + VGM 当 generator"的闭环组合极可能成为未来视频推理标配。值得 follow：CollabVR 思路移植到 3D / 世界模型（[[16-sana-wm]] / [[47-worldreasonbench]]）；与 [[20-autotts]] 协同寻找最佳 step 频率；自动学 step prompting policy。

---

## 20. AutoTTS：测试时扩展策略的自动发现
**👍 66** · [arXiv:2605.08083](https://arxiv.org/abs/2605.08083) · [GitHub](https://github.com/zhengkid/AutoTTS) · [Project](https://zhengkid.github.io/AutoTTS-web/)

### 问题与动机
Test-Time Scaling（TTS）已被广泛证明能拉升 LLM 性能，但目前主流方案靠人工设计推理 pattern、人工调启发式，导致 compute-allocation 空间大部分未被探索。

### 方法与核心创新
**AutoTTS** 换一个研究对象：不再设计 TTS 启发式本身，而是设计"让 TTS 策略可被自动发现的环境"。具体把 **width-depth TTS** 形式化为"控制器合成"问题：在预收集的推理轨迹 + 探针信号上，控制器决定何时分支/继续/探测/剪枝/停止——评估完全在缓存上跑，**不重复调 LLM**。还引入 **beta 参数化**让搜索空间可处理，以及**细粒度执行轨迹反馈**帮 Agent 诊断 TTS 程序为何失败。

### 关键实验结果
- 在数学推理 benchmark 上，自动发现的策略相对强人工 baseline **同时提升 accuracy 与降低 cost**；
- 策略可泛化到 hold-out benchmark 与不同模型规模；
- **整套发现成本仅 $39.9 + 160 分钟**——这种工程经济性在 TTS 论文里非常罕见。

### 局限性与开放问题
预收集轨迹的代表性决定上限；环境抽象到数学推理之外（如多工具 Agent）是否依然奏效未充分验证；自动发现策略的可解释性不强。

### 启发与应用前景
"先造一个便宜环境再让 Agent 自动发现 TTS"是范式级转变，会改变社区的研究方式。值得 follow：把 AutoTTS 用到 Coding Agent / Tool-use Agent；与 [[30-tmas]] 多智能体 TTS 整合；和 [[17-rubricem]] 的 rubric 系统 + 自动发现策略叠加。

---

## 21. LPO：Listwise 策略优化的几何统一视角
**👍 65** · [arXiv:2605.06139](https://arxiv.org/abs/2605.06139)

### 问题与动机
RLVR（reinforcement learning with verifiable rewards）已成 LLM post-training 标配，其中 group-based policy gradient（如 GRPO）最常用。但各种变体的内部关系混乱：到底它们在共同优化什么？

### 方法与核心创新
作者揭示一个**统一几何结构**：所有这些算法都在 response simplex 上**隐式定义一个目标分布**，并用一阶近似把当前策略往那里推。基于这个认识提出 **Listwise Policy Optimization (LPO)**：
- 把 proximal RL 目标**显式投影**到 response simplex 上；
- 通过精确的 divergence minimization 做策略投影；
- 提供 (i) 单调改进、bounded、zero-sum、self-correcting 的 listwise 目标；(ii) 离散步可灵活选 divergence。

### 关键实验结果
在多种推理任务和 LLM backbone 上：
- 在 matched target 下一致优于经典 policy gradient baseline；
- 同时保持 **训练稳定性** 与 **响应多样性**——这正是 group-based 方法常见的痛点。

### 局限性与开放问题
"几何统一"理论很美但与实际超参选择的桥梁还偏抽象；不同 divergence 的实证表现差异还需更多数据；与 [[9-sdar]] 类辅助蒸馏目标如何叠加未涉及。

### 启发与应用前景
LPO 给"为什么 GRPO 经常崩、什么时候改用别的目标"提供了第一性原理的指南。值得 follow：把 LPO 用于 multi-modal RL（[[10-flow-opd]] / [[42-alphagrpo]]）；与 [[17-rubricem]] 的 rubric-judge 联合验证 simplex 投影；探索更高阶 divergence（Wasserstein、Sinkhorn）。

---

## 22. World Action Models 综述：具身 AI 的下一个边界
**👍 62** · [arXiv:2605.12090](https://arxiv.org/abs/2605.12090) · [GitHub](https://github.com/OpenMOSS/Awesome-WAM) · [Project](https://openmoss.github.io/Awesome-WAM/)

### 问题与动机
VLA 模型语义泛化强，但本质上仍是"观测→动作"反应式映射，并不显式建模物理世界在 intervention 下如何演化。近期一批工作把世界模型嵌进 action 生成管线，但这一方向"概念碎片化、术语混乱、taxonomy 缺失"，难以系统比较。

### 方法与核心创新
作者正式定义 **World Action Models (WAMs)**：将预测式状态建模与动作生成统一为对 (future state, action) 联合分布的建模。
- 明确 WAM 与 VLA、world model、policy 的区别；
- 把已有方法分类为 **Cascaded WAM** 与 **Joint WAM**，再按生成模态/条件机制/动作解码策略细分；
- 系统梳理数据生态（远程操控、可穿戴人类示教、仿真、互联网级 egocentric video）；
- 综合 visual fidelity、physical commonsense、action plausibility 三轴的评测协议。

### 关键实验结果
作为综述论文不含主实验，但贡献了：
- 当前最完整的 WAM taxonomy；
- 横跨数据、训练、评测的统一视图；
- 维护中的 Awesome-WAM 资源页。

### 局限性与开放问题
综述本身没回答"哪种 WAM 范式最终会赢"；定量横向比较仍依赖现有工作的不一致评测；对长程任务的"自动评估指标可信度"也无独立验证。

### 启发与应用前景
WAM 综述是 [[16-sana-wm]] / [[25-cascadebench]] / [[29-humannet]] / [[36-mcp-cosmos]] / [[47-worldreasonbench]] 这一波同期工作的最佳"地图"，对从事具身 / 世界模型研究的人是必读路标。值得 follow：把 WAM taxonomy 与 [[3-sensenova-u1]] 的 unified 多模态思路整合；建立"统一 WAM 评测套件"。

---

## 23. HyperEyes：并行多模态搜索 Agent 的双粒度效率 RL
**👍 62** · [arXiv:2605.07177](https://arxiv.org/abs/2605.07177) · [GitHub](https://github.com/DeepExperience/HyperEyes)

### 问题与动机
多模态搜索 Agent 把多个待检索实体逐个串行处理，每实体一次 tool call，遇到可分解 query 时累积大量冗余轮次——"搜得长"而不是"搜得宽"。论文主张该并行的就同时并发。

### 方法与核心创新
**HyperEyes** 把 visual grounding 与 retrieval 合成**单个原子动作**，支持同一轮并发多实体查询，并把推理效率提升为"一等训练目标"。两阶段：
1. **Cold-start**：构建支持并行的合成数据 pipeline，覆盖视觉多实体 + 文本多约束 query，用 Progressive Rejection Sampling 筛 efficiency 友好轨迹；
2. **Dual-Grained Efficiency-Aware RL**：
   - **宏观（TRACE）**：轨迹级奖励，引用基线随训练单调收紧，压低多余 tool call，但不限制真实多跳；
   - **微观**：On-Policy Distillation 在失败 rollout 上注入 token 级 corrective 信号，弥补稀疏 outcome reward 的信用分配缺陷。
另引入 **IMEB**：300 题、首次"准确率 + 推理成本"联评。

### 关键实验结果
- **HyperEyes-30B** 跨 6 个 benchmark 相对最强可比开源 Agent **+9.9% accuracy**；
- **平均 5.3× 更少 tool-call 轮次**——这是同类工作里最显著的成本下降。

### 局限性与开放问题
"原子化并行搜索"对工具协议要求高（要支持 batched call）；TRACE 参考基线的设计依赖任务特性，泛化到新任务需重新校准；IMEB 仅 300 题。

### 启发与应用前景
"准确率 + 成本"联评是 Agent 评测的下一步，HyperEyes 和 IMEB 提供了同时改进两端的范例。值得 follow：把 dual-grained RL 用于 Coding Agent；与 [[34-wildclaw]] 真实 CLI 任务交叉；和 [[20-autotts]] 探索"自动发现并行调度"。

---

## 24. EVA-Bench：端到端语音 Agent 评测框架
**👍 61** · [arXiv:2605.13841](https://arxiv.org/abs/2605.13841) · [GitHub](https://github.com/ServiceNow/eva) · [Project](https://servicenow.github.io/eva/)

### 问题与动机
语音 Agent 在企业部署增长迅速，但没有 benchmark 同时解决"生成真实对话 + 覆盖语音特有失败模式"两大问题。多数评测要么靠人工录制要么用文本翻译，缺乏可比较的客观度量。

### 方法与核心创新
**EVA-Bench** 是端到端框架：
- **仿真层**：协调 bot-to-bot 多轮音频对话，自动检测用户模拟器出错并在打分前重新生成；
- **度量层**：
  - **EVA-A (Accuracy)**：任务完成、忠实度、音频级语音保真；
  - **EVA-X (Experience)**：对话推进、口语简洁度、turn-taking 时机；
- **213 个场景** × 三类企业领域 + 口音/噪声 perturbation 套件；
- 同时给出 pass@1 / pass@k / pass^k，区分"峰值能力"与"可靠能力"。

### 关键实验结果
跨 **三种架构 12 个系统**：
1. 没有任何系统同时在 EVA-A pass@1 与 EVA-X pass@1 都超 0.5；
2. **pass@k 与 pass^k 差距中位数 0.44**——峰值与可靠表现严重背离；
3. 口音和噪声 perturbation 暴露巨大鲁棒性鸿沟（最高 0.314 偏差）。

### 局限性与开放问题
EVA-Bench 没强制公开音频用例（涉及隐私与合规），独立复现依赖框架代码；"企业场景"主要英语，多语种 / 方言扩展待补；perturbation 与真实噪声的覆盖映射不完整。

### 启发与应用前景
EVA-Bench 把语音 Agent 评测从"WER + 客户满意度"升级到"端到端任务 + 可靠性 + 鲁棒性"。值得 follow：把 EVA-X 风格的"体验度量"迁移到 Coding Agent、客服 Agent；与 [[34-wildclaw]] 联立做"GUI/CLI/Voice 三栈" Agent 评测。

---

## 25. 企业级 World Model：运行时发现 vs 离线训练
**👍 60** · [arXiv:2605.12178](https://arxiv.org/abs/2605.12178)

### 问题与动机
企业系统的"环境动力学"由租户级业务逻辑决定，跨部署差异大、随时间演化，导致基于历史 transition 训练的 world model 一上线就 brittle。世界模型文献从未问过：当规则在推理时可读，Agent 还需要把它"学进来"吗？

### 方法与核心创新
作者主张**运行时发现**（runtime discovery）与离线训练应互补：在动力学**可配置且可读**的设置下，Agent 应主动读取系统配置而非完全依赖内化表征。提出 **enterprise discovery agents** 与 **CascadeBench**——按 World of Workflows 评测方法构建的多样合成环境，加上 deployment-shift 评测。

### 关键实验结果
- 离线训练的 world model 在 in-distribution 表现良好，但**动力学变化时显著退化**；
- **discovery-based agents** 在 shift 下更稳健，因为预测被 grounded 到当前实例；
- 论文未给单一聚合数字，但在多组 shift 设置上趋势一致。

### 局限性与开放问题
"可读配置"假设并非所有企业系统都成立；CascadeBench 仍是合成；配置文档与代码混杂场景的 discovery 复杂度未量化；与 RAG 的边界也不够清晰（参 [[37-stale]] 记忆失效问题）。

### 启发与应用前景
对 B2B AI Agent 设计有直接指导：**当业务规则可读，就别去训它**。值得 follow：把 discovery 思路接入 MCP（[[36-mcp-cosmos]]）；与 [[31-gcwm]] 持续学习的几何冲突一道，建立"何时训练 / 何时读配置"决策框架。

---

## 26. MemEye：视觉中心的多模态 Agent 记忆评测
**👍 59** · [arXiv:2605.15128](https://arxiv.org/abs/2605.15128) · [GitHub](https://github.com/MinghoKwok/MemEye) · [Project](https://minghokwok.github.io/MemEye/)

### 问题与动机
长程 Agent 记忆越来越多模态，但既有评测里很多"看似视觉"的题靠 caption 或文本痕迹就能答——视觉证据可被绕过。同时缺少"对随时间变化的视觉状态做推理"的高难题。

### 方法与核心创新
**MemEye** 从两个维度评估记忆：
1. **决定性视觉证据的粒度**：从场景级到像素级；
2. **证据如何被使用**：从单证据到进化合成。
基于此构建 8 类生活场景 benchmark，配置 ablation 驱动的验证 gate（answerability、shortcut resistance、visual necessity、reasoning structure）。

### 关键实验结果
- 评测 **13 种记忆方法 × 4 个 VLM backbone**；
- 当前架构仍难以**保留细粒度视觉细节**与**跨时间状态推理**；
- 论文给出结论："长期多模态记忆依赖 evidence routing、temporal tracking 与 detail extraction"。

### 局限性与开放问题
8 类生活场景有领域偏向；evidence granularity 的离散分级仍偏粗；缺少与 [[18-memlens]] 的同题直接对比；"进化合成"的题目数量较少。

### 启发与应用前景
MemEye + MemLens 是 W20 关于多模态记忆评测的两块拼图——前者强调视觉粒度，后者强调跨会话推理。值得 follow：把 MemEye 与 [[5-memprivacy]] 隐私保护结合，构造"既保护又可视觉记忆"的端云 Agent。

---

## 27. Qwen-Image-VAE-2.0 技术报告
**👍 55** · [arXiv:2605.13565](https://arxiv.org/abs/2605.13565) · [GitHub](https://github.com/alibaba/OmniDoc-TokenBench)

### 问题与动机
高压缩 VAE 同时要"重建保真"和"利于扩散建模"两个目标常常冲突。文本密集场景（文档、海报）下高压缩尤其难，且 latent 维度上升后扩散收敛也会变慢。

### 方法与核心创新
**Qwen-Image-VAE-2.0**：
- **Global Skip Connections (GSC)** + 拓展 latent 通道，缓解高压缩下的重建瓶颈；
- 数据规模扩到 **十亿级图像** + 合成渲染引擎，增强文字场景；
- **语义对齐**增强让高维 latent 更适合 diffusion；
- **非对称、attention-free encoder-decoder** 降编码开销；
- 新提出 **OmniDoc-TokenBench**：真实文档 + OCR 指标。

### 关键实验结果
- 公共重建 benchmark 上 SOTA，尤其在 text-rich 高压缩比下；
- 下游 DiT 实验显示其 **diffusability 显著优于现有高压缩 baseline**，收敛明显加速；
- 提出的 OmniDoc-TokenBench 为社区补上了文档重建评测空缺。

### 局限性与开放问题
具体压缩比与重建 PSNR 数字未在摘要披露，难以横向比较；非对称设计在视频 VAE 的可迁移性待考；与 [[43-drorae]] 的多层融合思路是否互补也是好问题。

### 启发与应用前景
"高压缩 VAE + 文档专评测"对所有做 text-to-X 生成的厂商有用。值得 follow：把 GSC 思路扩展到视频 VAE；OmniDoc-TokenBench 应纳入图像生成模型 release 时的标准评测；与 [[8-qwen-image-20]] 联合理解 Qwen 的影像栈整体战略。

---

## 28. Darwin Family：MRI-Trust 进化合并的训练-free 推理扩展
**👍 54** · [arXiv:2605.14386](https://arxiv.org/abs/2605.14386) · [Project](https://vidraft.net)

### 问题与动机
要在不付出额外训练成本的情况下提升前沿 LLM 推理能力，**模型合并**是最具诱惑力的路径。但传统合并要么过于粗暴（按层平均），要么需要 gradient 监督。能否做到 gradient-free 且跨架构？

### 方法与核心创新
**Darwin Family** 三个关键点：
1. **14 维自适应合并基因组**：细粒度到 component / block 级重组；
2. **MRI-Trust Fusion**：把"层重要性诊断信号"与进化搜索通过一个可学的 trust 参数自适应平衡；
3. **Architecture Mapper**：实现跨架构繁殖——比如 Transformer 与 Mamba 组件可以混在一个孩子里。

### 关键实验结果
- 旗舰 **Darwin-27B-Opus** 在 **GPQA Diamond 86.9%**；
- 在 **1252 个评测模型中排名第 6**，并 **超过自身的完整训练 foundation model**——不用任何梯度训练；
- 4B-35B 各规模一致改进，支持递归多代进化；
- Transformer + Mamba 混合体在 training-free 下可工作。

### 局限性与开放问题
"训练-free 超过 foundation model"的稳健性需要更多次随机种子复现；进化搜索的算力成本本身可能不低；评测主要 GPQA Diamond，多任务一致性如何待补；MRI 度量对新架构的可迁移性需验证。

### 启发与应用前景
和 [[35-merging-scaling-law]] 一起读：前者给出"合并到底能涨多少"的法则，后者给出具体合并算法。值得 follow：把 Darwin 用到代码 / 数学 / 法律领域专家融合；探索 Transformer/Mamba/Mixer 混合 backbone 的自动设计。

---

## 29. HumanNet：百万小时人本视频学习
**👍 51** · [arXiv:2605.06747](https://arxiv.org/abs/2605.06747) · [GitHub](https://github.com/DAGroup-PKU/HumanNet) · [Project](https://dagroup-pku.github.io/HumanNet/)

### 问题与动机
具身智能进展受限于"物理交互"数据稀缺：视觉/语言可以靠互联网语料 scale，但人怎么和物理世界打交道的数据没有同等规模、同等丰富标注。

### 方法与核心创新
**HumanNet**：**一百万小时**的人本视频语料——第一/第三人称都有，覆盖细粒度活动、人-物交互、工具使用、长程行为，跨多样真实环境。除原始视频还有交互中心标注（caption、运动描述、手/身体信号）；论文同时把"数据策划"提升为一等公民——人本过滤、时序结构化、视角多样、标注富化。

### 关键实验结果
首步验证："Qwen VLM 继续训练 **1000 小时 HumanNet egocentric 视频** 已**超过用 100 小时 Magic Cobot 真实机器人数据**继续训练的效果"——意味着 egocentric 人类视频是真实机器人数据的廉价替代。

### 局限性与开放问题
"百万小时"数据是否能完整开放、版权与隐私如何处理仍是关键问题；"1000h vs 100h"的对比并不完全等价，需要更对照实验；评测指标依赖具体 VLA 任务定义。

### 启发与应用前景
HumanNet 直接为 [[22-wam-survey]] 一类世界动作模型提供燃料。值得 follow：把 HumanNet 与 [[16-sana-wm]] 类世界模型联训；探索 egocentric → robot 的可控迁移；与 [[14-mace-dance]] 的人体动作生成结合做"通用人形 backbone"。

---

## 30. TMAS：多智能体协同的测试时计算扩展
**👍 49** · [arXiv:2605.10344](https://arxiv.org/abs/2605.10344) · [GitHub](https://github.com/george-QF/TMAS-code)

### 问题与动机
结构化 TTS 已通过多轨迹/refine/verify 取得进展，但要么并行轨迹间协调弱，要么依赖嘈杂历史而不显式决定什么保留复用——探索 / 利用难以平衡。

### 方法与核心创新
**TMAS** 把推理组织为多智能体协同：
- **Hierarchical memories**：experience bank 复用低层可靠中间结论与局部反馈，guideline bank 记录探索过的高层策略，引导后续 rollout 避开冗余 reasoning 模式；
- **Hybrid reward RL**：兼顾"基础推理能力保留 + 经验利用增强 + 鼓励超越已尝试策略"。

### 关键实验结果
- 在多个挑战性推理 benchmark 上，TMAS 比已有 TTS baseline **迭代扩展效率更高**；
- Hybrid reward 训练进一步提升 scaling 有效性与跨迭代稳定性。

### 局限性与开放问题
hierarchical memory 的具体存储 schema 与召回粒度披露不深；多智能体角色数量到何时收益开始递减没有 ablation；与 [[33-life-survey]] 的多智能体失效归因思路如何整合是个研究问题。

### 启发与应用前景
TMAS 把"TTS"和"多智能体"两条线合并：未来的 reasoning 系统很可能就是"一群专家 + 共享记忆"。值得 follow：与 [[17-rubricem]] 的 rubric-as-memory 共建经验 bank；探索企业场景的轻量多 Agent；以及作为 [[20-autotts]] 自动发现策略空间的一部分。

---

## 31. GCWM：几何冲突解释 LLM 持续后训练遗忘
**👍 49** · [arXiv:2605.09608](https://arxiv.org/abs/2605.09608) · [GitHub](https://github.com/wyy-code/GCWM)

### 问题与动机
持续 post-training（连续接入新任务、新数据）希望让 LLM 增长能力，却频繁出现灾难性遗忘。现有方法（顺序微调、回放、正则、合并）治标，无法回答"何时融合有益、何时有害"。

### 方法与核心创新
作者引入**任务几何**视角：把每个 post-training 任务表示为"参数更新"，研究该更新诱导的**协方差几何**。核心发现：**遗忘 = state-relative update-integration failure**——当任务诱导的协方差几何与不断演化的模型 state 不对齐，整合就失败。基于此提出 **Geometry-Conflict Wasserstein Merging (GCWM)**：通过 Gaussian Wasserstein barycenter 构建共享 Wasserstein 度量，用 geometry conflict 触发 geometry-aware correction。

### 关键实验结果
- 在 **Qwen3 0.6B-14B** 上跨 domain-continual 与 capability-continual 设置；
- **不需回放数据**就稳定优于同类 data-free baseline；
- 同时改进知识保留与最终性能。

### 局限性与开放问题
Gaussian Wasserstein 假设对极端非高斯参数分布的鲁棒性需验证；computational cost 在大模型上的拓展曲线没充分披露；与回放 + 正则的混合方案没充分对比。

### 启发与应用前景
"几何冲突"既是解释信号也是控制信号——这种"双重身份"的研究范式可能也适合解释 RLHF 漂移、视频生成 LoRA 干扰。值得 follow：把 GCWM 整合到 [[35-merging-scaling-law]] 的合并扩展律；与 [[28-darwin]] 进化合并互补使用做"几何感知进化"。

---

## 32. 文本-表格建模预测 Agent 决策
**👍 48** · [arXiv:2605.12411](https://arxiv.org/abs/2605.12411)

### 问题与动机
当一个 AI Agent 与未知对方 Agent 谈判时，对方的 LLM / prompt / 规则不可见，但每个决策点都有金钱后果。能否仅凭"几次互动"就预测对方下一步？

### 方法与核心创新
作者把任务形式化为 **target-adaptive 文本-表格预测**：每个决策点是一行，结合结构化博弈 state、报价历史与对话；K 次以前的同一对手对局作为 prompt 内的"标注适配示例"。模型核心是 tabular foundation model + game-state 特征 + 基于 LLM 的文本表征，并新增 **LLM-as-Observer**：一个小冻结 LLM 读决策时状态与对话，**只取其 hidden state 作 decision-oriented 特征**（不取其文本答案）。

### 关键实验结果
- 训练用 **13 种前沿 LLM Agent**，测试 **91 个 hold-out scaffolded Agent**；
- 在 K=16 时 Observer 特征 **AUC +4 点（response 预测）**、**bargaining offer 预测误差 -14%**；
- 全模型显著优于直接 LLM-as-Predictor prompting 与 game+text 特征 baseline。

### 局限性与开放问题
"bargaining game"仍是相对结构化任务，开放对话决策预测的可迁移性需验证；LLM-as-Observer 选用何种 backbone 影响也未充分披露；可对抗性（对方主动伪装）未涵盖。

### 启发与应用前景
"用 hidden state 当特征"反向利用 LLM 的潜在表征，是个被低估的工程技巧。值得 follow：把 Observer 用于多 Agent 协作中的"对方意图建模"（参 [[33-life-survey]]）；把 prediction 反馈回 [[36-mcp-cosmos]] 的世界模型；探索高风险场景（金融、安全）下的实战部署。

---

## 33. LIFE 综述：多智能体协作、归因与自演化
**👍 46** · [arXiv:2605.14892](https://arxiv.org/abs/2605.14892) · [GitHub](https://github.com/mira-ai-lab/awesome-mas-life)

### 问题与动机
LLM 多智能体系统通过结构化协作放大单 Agent 能力，但更紧的协同也放大风险：**错误会在 Agent 间传播**，且失败诊断与自我改进机制薄弱。已有综述要么只聊单 Agent 能力、要么只聊协作、要么只聊自演化——三条线的**因果依赖**没有人系统性梳理。

### 方法与核心创新
本综述提出 **LIFE 四阶段**因果框架：
1. **Lay** 基础能力；
2. **Integrate** 通过协作整合；
3. **Find** 失效归因；
4. **Evolve** 通过自主自改进演化。
对每阶段给出 taxonomy 并形式化前后阶段的依赖关系，揭示每阶段如何**依赖也约束下一阶段**；并指出阶段交界处的开放问题。

### 关键实验结果
作为综述，主贡献是：跨四个阶段的统一术语 / taxonomy / 依赖图，以及 awesome-mas-life 资源页。

### 局限性与开放问题
"自演化"在实践中仍处于早期阶段，综述无法给出现成胜出方案；归因部分缺乏可量化的失败 attribution benchmark；自我组织形式的失控风险（emergence 一致性）讨论较保守。

### 启发与应用前景
LIFE 框架补齐了 [[30-tmas]] / [[34-wildclaw]] / [[20-autotts]] 一波同期工作的"理论地图"。值得 follow：把"失败归因"作为 [[24-eva-bench]] 类企业评测的一等指标；与 [[31-gcwm]] 几何冲突结合做"协作时的能力遗忘"研究。

---

## 34. WildClawBench：真实长程 CLI Agent 评测
**👍 45** · [arXiv:2605.10912](https://arxiv.org/abs/2605.10912) · [GitHub](https://github.com/internlm/WildClawBench) · [Project](https://internlm.github.io/WildClawBench/)

### 问题与动机
LLM/VLM Agent 越来越多通过 CLI harness 替用户做事，但绝大多数 Agent benchmark 仍在合成沙箱、短任务、mock API、终答检查上转——和真实部署的"原生 runtime"严重脱节。

### 方法与核心创新
**WildClawBench**：**60 个人工撰写、双语、多模态任务**跨 6 个主题；每任务**平均 8 分钟 wall-clock + 20+ tool calls**，运行在可复现 Docker 容器内（内含真实 CLI Agent harness：OpenClaw / Claude Code / Codex / Hermes Agent）。评测混合：deterministic rule-based 检查 + 环境 state 审计 + LLM/VLM judge 语义验证。

### 关键实验结果
评测 **19 个前沿模型**：
- **最强 Claude Opus 4.7 在 OpenClaw 上仅 62.2%**；
- 其他模型全部 **<60%**；
- **同一模型仅更换 harness 评分波动可达 18 个百分点**——harness 不是中立。

### 局限性与开放问题
60 个任务统计噪声仍高；6 主题外的覆盖（如运维、数据科学）有限；某些任务依赖外部 API 真实可用性，结果可重现性可能受时间影响。

### 启发与应用前景
"harness 不是中立"是这篇论文最值得记住的洞察——任何 Agent 模型对比必须报告 harness 版本。值得 follow：把 WildClawBench 与 [[24-eva-bench]] 联合做"GUI+CLI+Voice"全栈评测；用 [[20-autotts]] / [[17-rubricem]] 在 WildClawBench 上跑出新 SOTA。

---

## 35. 模型合并的扩展律
**👍 43** · [arXiv:2509.24244](https://arxiv.org/abs/2509.24244) · [GitHub](https://github.com/InfiXAI/Merging-Scaling-Law) · [Project](https://infix.io/research/MergingScalingLaw)

### 问题与动机
模型合并被广泛使用，但社区缺一条"加专家数 / 升基座规模时回报如何"的定量规则。没有 scaling law，合并就只能停留在 heuristic。

### 方法与核心创新
作者用 cross-entropy 度量合并性能，发现一条**紧凑的幂律**：
- **size-dependent floor** 随基座容量下降；
- **merging tail** 在专家数增加时呈**明显递减回报**；
- 跨架构、方法（Average / TA / TIES / DARE）一致，**in-domain 与 cross-domain 都成立**；
- 解释两个稳健规律：增益主要来自前几个专家；专家越多方差越小。
配套提出简洁理论解释为何回报近似 **1/k**，并把 floor 与 tail 与基座属性、领域多样性挂钩。

### 关键实验结果
- 在多种架构与四种合并方法上紧拟合；
- 提供"达到目标 loss 需要多少专家"、"何时停止加专家"、"在固定预算下 base size vs expert count 如何 trade-off"等可计划公式。

### 局限性与开放问题
所有 scaling law 都受拟合区间限制；对极端规模与极端专家数的外推需谨慎；论文未与 RL/SFT scaling 做联合建模。

### 启发与应用前景
合并从"启发实践"升级为"可规划替代多任务训练"——和 [[28-darwin]] 进化合并合起来构成"合并研究的工程化双件"。值得 follow：把扩展律植入开源工具链做自动停止；与 [[31-gcwm]] 的几何冲突做"何时该停"的二维准则。

---

## 36. MCP-Cosmos：MCP 环境的世界模型增强 Agent
**👍 43** · [arXiv:2605.09131](https://arxiv.org/abs/2605.09131)

### 问题与动机
MCP 已经统一了 LLM 与工具的接口，但 Agent **如何在 MCP 环境中"理解"环境**仍有断层：任务级 planning 不顾执行时动力学，反应式执行又缺长程预见。

### 方法与核心创新
**MCP-Cosmos**：把生成式 **World Model** 注入 MCP 生态。三项统一：
- **MCP**（接口） + **World Model**（动力学先验） + **Agent**（执行）；
- **Bring Your Own World Model (BYOWM)**：Agent 可任意外挂 WM，在 latent 空间模拟 state transition 并 refine 计划再执行；
- 实验跨 ReAct / SPIRAL × 2 个 planning model × 3 个代表性 WM，超过 20 个 MCP-Bench 任务。

### 关键实验结果
- 工具调用成功率、参数准确率等环境交互 KPI 改进；
- 新指标 **Execution Quality** 揭示 WM 相对 baseline 的实际价值差异。

### 局限性与开放问题
"BYOWM"虽灵活但 WM 与策略不一致时表现差距大；20+ MCP 任务规模不大；推理开销（每步都做 latent 模拟）的工程账没细讲。

### 启发与应用前景
MCP-Cosmos 把"世界模型 + MCP"具体化，可能成为下一代生产级 Agent 的标准结构。值得 follow：与 [[25-cascadebench]] 的"runtime discovery"统一研究"何时模拟、何时读规则"；和 [[22-wam-survey]] 的 WAM taxonomy 整合做"接口 + 世界模型 + 动作"统一框架。

---

## 37. STALE：Agent 能否识别失效的记忆？
**👍 42** · [arXiv:2605.06527](https://arxiv.org/abs/2605.06527)

### 问题与动机
长程个性化记忆研究主要测"静态事实检索"，但**新的观测可能让旧记忆失效**——这一"隐式冲突"问题被严重忽视。Agent 若识别不出 stale 记忆，就会把"过期 belief"反复施加到下游。

### 方法与核心创新
作者提出 **Implicit Conflict** 失败模式：后到观察未显式否定，但通过常识推理可推断它使先前记忆失效。**STALE** 是 400 个专家验证冲突场景 + 1200 个评估 query，跨 100+ 日常主题，context 长达 150K token。三维 probing：
- **State Resolution**：识别"belief 已过时"；
- **Premise Resistance**：拒绝带错误前提的 query；
- **Implicit Policy Adaptation**：主动在下游行为应用新状态。

### 关键实验结果
- **最强模型整体准确率仅 55.2%**——前沿模型也存在系统性盲点；
- 模型常常接受用户 query 里嵌入的过期假设；
- 论文还给出 **CUPMem** 原型，通过结构化 state consolidation + propagation-aware search 强化 write-time 修订。

### 局限性与开放问题
400 个场景是中等规模、专家标注成本高；中文 / 跨文化"日常常识"覆盖有限；CUPMem 与商用记忆框架的兼容性需更多 case study。

### 启发与应用前景
STALE 把"记忆是否失效"作为新评测维度抬到台前——这对所有 Agent / 数字员工产品都非常关键。值得 follow：和 [[18-memlens]] / [[26-memeye]] 的多模态记忆评测整合做"过期 + 视觉"双维基准；与 [[25-cascadebench]] 的 runtime discovery 共建"何时丢弃记忆 / 何时重读配置"框架。

---

## 38. Token-Superposition Training：高效预训练
**👍 41** · [arXiv:2605.06546](https://arxiv.org/abs/2605.06546) · [Project](https://nousresearch.com/token-superposition)

### 问题与动机
LLM 预训练越来越贵，已有提速方法多数侵入式：要么改并行、要么改 optimizer / tokenizer / 数据 / 架构，导致工程门槛飙升。能否在**不动这些**的前提下提升 throughput？

### 方法与核心创新
**Token-Superposition Training (TST)**：drop-in 的两阶段方案。
1. **Superposition 阶段**：把多个相邻 token **合成一个 bag**，使用 multi-hot cross-entropy (MCE) 训练，单步处理的有效 token 数大幅提升；
2. **Recovery 阶段**：恢复标准训练。
完全不改并行、tokenizer、数据或架构。

### 关键实验结果
跨 270M / 600M / 3B / 10B A1B MoE 验证：
- TST **一致优于 baseline 的 loss 与下游评测**；
- 在 equal-loss 下，**10B A1B 总预训练时间最多减少 2.5×**；
- 跨规模与设置高度鲁棒。

### 局限性与开放问题
"bag" 的最优大小、bag 内 token 顺序是否影响 long-range 依赖学习需更多 ablation；MCE 与传统 CE 的统计性质差异理论尚浅；视觉 token / 多模态 token 是否同样受益未知。

### 启发与应用前景
TST 的"无侵入"特性让其几乎适合所有现有训练管线——这是预训练加速里最重要的属性。值得 follow：把 TST 用于视觉 token、speech token 预训练；与 MoE 路由的协同（10B A1B 已验证 1 步）；探索"bag size 自动调度"。

---

## 39. ROPD：Rubric-based 黑盒在线策略蒸馏
**👍 40** · [arXiv:2605.07396](https://arxiv.org/abs/2605.07396) · [GitHub](https://github.com/Peregrine123/ROPD_official)

### 问题与动机
On-Policy Distillation (OPD) 在对齐中表现强劲，但依赖**教师 logits** 把它锁死在 white-box 场景；闭源教师（GPT、Claude、Gemini）的 logits 拿不到，OPD 就没法做。

### 方法与核心创新
**ROPD**：把"结构化语义 rubric"作为可扩展替代物。先从 teacher-student 对比中诱导出 **prompt-specific rubric**，再用这些 rubric 对学生 rollout 打分做在线优化——**全程只用教师的生成回复，不需要 logits**。

### 关键实验结果
- 在多数场景下 **超过先进 logit-based OPD 方法**；
- **样本效率最高 10×**；
- 把 OPD 从 white-box 局限释放到黑盒。

### 局限性与开放问题
rubric 诱导本身依赖模型能力，弱学生上诱导出的 rubric 可能过粗；与 [[17-rubricem]] 的元 rubric 是否能整合未探索；闭源教师的合法蒸馏边界（TOS 风险）需注意。

### 启发与应用前景
**所有用闭源前沿 LLM 做"老师"的对齐研究都会受益**——这是 ROPD 的最大价值。值得 follow：把 ROPD 与 [[10-flow-opd]] 视觉 OPD 整合做"黑盒视觉教师蒸馏"；和 [[39-ropd]] 自身的 rubric 复用机制（可与 [[17-rubricem]] 蒸馏出共享 rubric library）整合做"统一对齐协议"。

---

## 40. Warp-as-History：单视频训练的相机可控视频生成
**👍 38** · [arXiv:2605.15182](https://arxiv.org/abs/2605.15182) · [GitHub](https://github.com/yyfz/Warp-as-History) · [Project](https://yyfz.github.io/warp-as-history/)

### 问题与动机
现有 camera-controlled video generation 通常学 camera 专属 conditioning（camera encoder / control branch / 位置编码改造），需要大规模相机标注数据做 post-train；训练-free 方案把成本搬到 test-time 优化。两端都有大量额外开销。

### 方法与核心创新
**Warp-as-History**：把"相机引起的画面 warp"转成 **camera-warped 伪历史**送进模型的视觉历史路径——同时做 (i) target-frame 位置编码对齐 (ii) 选取仅含有效源观测的可见 token。本质是用模型已有的"参考过去帧"通道完成相机条件，**无训练、无架构改造、无测试时优化**即揭示出 frozen 视频模型的零样本相机跟随能力。
进一步：仅用 **一条相机标注视频** 做轻量 offline LoRA 微调，能力进一步增强并泛化到未见视频。

### 关键实验结果
- frozen 模型即刻具备非平凡的零样本相机轨迹跟随；
- 单视频 LoRA 微调显著提升 camera 一致性、视觉质量与运动动态；
- 跨多种数据集验证通用性。

### 局限性与开放问题
单视频 LoRA 的过拟合风险在更复杂相机轨迹（高速、旋转）下需评估；缺乏对极端 viewpoint 跳变的失败案例分析；与 [[16-sana-wm]] 双分支 camera control 的优劣对比缺位。

### 启发与应用前景
"用已有视觉历史路径塞 warp 假象"是种极简但强大的工程技巧——可推广到 3D 控制、深度控制等其他几何条件。值得 follow：把 Warp-as-History 与 [[41-trackcraft3r]] 的几何先验联立；探索"单视频微调"在动作控制（不仅相机）上的潜力。

---

## 41. TrackCraft3R：复用视频扩散 Transformer 做密集 3D 追踪
**👍 36** · [arXiv:2605.12587](https://arxiv.org/abs/2605.12587) · [GitHub](https://github.com/cvlab-kaist/TrackCraft3r) · [Project](https://cvlab-kaist.github.io/TrackCraft3r)

### 问题与动机
单目视频密集 3D 追踪是动态场景理解的基础。已有 3D 追踪器要么从合成数据从零训练、要么微调静态多视图重建模型，**都缺少真实世界运动先验**。预训练 Video DiT 含有丰富时空先验，但其 **frame-anchored** 公式（生成每帧内容）与"reference-anchored"密集 3D 追踪需求**根本不匹配**。

### 方法与核心创新
**TrackCraft3R**：首个把 Video DiT 改造成 feed-forward 密集 3D 追踪器的方法。两个关键设计：
1. **Dual-latent representation**：per-frame geometry latents + reference-anchored track latents 作为密集 query；
2. **Temporal RoPE alignment**：为每个 track latent 指定目标时间戳。
通过 LoRA 微调把"每帧生成"范式转换为"参考锚定追踪"。

### 关键实验结果
- 在标准稀疏 / 密集 3D tracking benchmark 上 **SOTA**；
- **比最强先前方法快 1.3×、峰值显存少 4.6×**；
- 对大幅度运动和长视频鲁棒。

### 局限性与开放问题
依赖 frame-anchored pointmap 输入，意味着需要前置 3D 重建模型；track latent 在极长视频下的内存增长曲线披露不足；遮挡建模的细节较少。

### 启发与应用前景
"把生成模型改造成几何感知 feed-forward 模型"的思路对 3D 视觉很有启发——视频生成预训练的红利可被几何任务"借用"。值得 follow：把同思路用到光流、深度、segmentation；与 [[48-pixal3d]] 像素对齐 3D 一起读做对照。

---

## 42. AlphaGRPO：UMM 的自反思多模态生成
**👍 35** · [arXiv:2605.12495](https://arxiv.org/abs/2605.12495) · [GitHub](https://github.com/huangrh99/AlphaGRPO) · [Project](https://huangrh99.github.io/AlphaGRPO/)

### 问题与动机
AR-Diffusion 统一多模态模型 (UMM) 越来越强，但缺少"激发其内在推理能力"的训练范式——尤其在"推理性 text-to-image 生成"（推用户隐含意图）与"自反思修正"两个新颖能力上。

### 方法与核心创新
**AlphaGRPO**：把 **GRPO** 用到 AR-Diffusion UMM，免去额外冷启动阶段。关键创新是 **Decompositional Verifiable Reward (DVReward)**：让 LLM 把复杂用户请求**分解为可验证的原子语义/质量问题**，再由通用 MLLM 判定——给出可靠且可解释的反馈，替代之前 monolithic scalar reward。

### 关键实验结果
- 在 **GenEval / TIIF-Bench / DPG-Bench / WISE** 全面提升；
- 在 **GEdit 编辑任务上未训练就大幅改进**——表明 unified backbone 的能力被自反思机制激活；
- 项目页展示自反思修复的定性案例。

### 局限性与开放问题
DVReward 依赖 MLLM judge 自身的可靠性，对超出 judge 训练分布的复杂语义场景可能失效；GRPO 的 sample efficiency 与显存代价对 UMM 来说仍偏重；自反思的"出错率"上界缺乏量化讨论。

### 启发与应用前景
"分解可验证奖励"在 [[17-rubricem]] / [[39-ropd]] 之外提供了第三种 rubric-like 范式，且专攻多模态生成。值得 follow：把 DVReward 推到视频生成（[[10-flow-opd]]）；探索"reasoning text-to-image"作为新评测基准。

---

## 43. DRoRAE：多层视觉特征融合的 Tokenizer
**👍 33** · [arXiv:2605.10780](https://arxiv.org/abs/2605.10780) · [GitHub](https://github.com/zhuzil/DRoRAE)

### 问题与动机
代表性 autoencoder 把冻结预训练视觉 encoder 当 tokenizer，但**普遍只取最后一层**特征，丢掉中间层的细粒度信息。低层细节虽在最后层以"残差"幸存，却被多层语义抽象层层削弱。

### 方法与核心创新
**DRoRAE (Depth-Routed Representation AutoEncoder)**：
- 通过 **能量约束 routing + 增量校正** 自适应聚合 **所有 encoder 层**；
- 产出兼容冻结 decoder 的增强 latent；
- 三阶段解耦训练：先学融合、再微调 decoder 充分利用增强表征。

### 关键实验结果
ImageNet-256：
- **rFID 从 0.57 降到 0.29**；
- **生成 FID 1.74 → 1.65**（用 AutoGuidance）；
- 收益迁移到 text-to-image 合成；
- 揭示**融合容量与重建质量的 log-linear scaling law（R²=0.86）**，识别"表征丰富度"为类比 NLP vocab size 的可规划新维度。

### 局限性与开放问题
新增 routing 模块带来的推理开销未充分披露；与高压缩 VAE（[[27-qwen-image-vae]]）思路是否互补尚未探索；视频与 3D tokenizer 的拓展性需要验证。

### 启发与应用前景
"表征丰富度作为可规划维度"是个被低估的视角——以前社区只在分辨率、词表、参数量上找 scaling，DRoRAE 提供了第四条曲线。值得 follow：把 DRoRAE 与高压缩 VAE 串联；探索视频 tokenizer 的多层融合；与 [[2-mv-split]] 类深度 DiT 共同推 Tokenizer + Backbone 协同 scaling。

---

## 44. PaperFit：视觉闭环的论文排版优化
**👍 32** · [arXiv:2605.10341](https://arxiv.org/abs/2605.10341) · [GitHub](https://github.com/OpenRaiser/PaperFit)

### 问题与动机
LaTeX 编译过的论文 ≠ 出版就绪——浮动错位、公式溢出、表格缩放不一、寡行孤行、页面失衡等问题依然普遍。规则工具看不到渲染、纯文本 LLM 又**预测不了二维布局后果**。

### 方法与核心创新
形式化 **Visual Typesetting Optimization (VTO)** 任务：把可编译 LaTeX 转成视觉精修、符合页面预算的 PDF。提出 **PaperFit**——vision-in-the-loop Agent：
1. 渲染页面 → 诊断 5 类缺陷 → 在 source 上做约束修复；
2. 闭环迭代直到收敛；
3. 同时构建 **PaperFit-Bench**：200 篇论文 × 10 个 venue 模板 × 13 种缺陷类型。

### 关键实验结果
- **大幅领先所有 baseline**——文本 only 工具与规则工具均无法对照视觉缺陷做精修；
- 论证 VTO 是"document automation pipeline 中关键的缺失阶段"。

### 局限性与开放问题
对中文 / 多语种排版（CJK 字体、双向文本）的支持尚不明；与 Overleaf 等平台集成的工程接口未涉及；agent 修复步骤的失败回退策略可更清晰。

### 启发与应用前景
PaperFit 是首个把"视觉闭环"形式化用到学术写作的工作，可推广到 slides / 海报 / 杂志排版。值得 follow：把视觉闭环与 [[19-collabvr]] 类 VLM-VGM 协作整合；商业化方向考虑 publication-as-a-service。

---

## 45. Edit-Compass：图像编辑与奖励建模统一基准
**👍 31** · [arXiv:2605.13062](https://arxiv.org/abs/2605.13062) · [GitHub](https://github.com/bxhsort/Edit-Compass-and-EditReward-Compass)

### 问题与动机
图像编辑模型在指令跟随、多模态理解、复杂视觉编辑上突飞猛进，但既有评测**反映不了前沿模型与人类判断的差距**——任务难度低、评测粒度粗。奖励模型也面临类似问题：评估场景不真实，与实际 RL 训练脱节。

### 方法与核心创新
**Edit-Compass**：2388 条精标实例 × 六个递进难度任务类别（覆盖 world knowledge reasoning、visual reasoning、multi-image editing），配套基于结构化推理的精细多维评测和评分 rubric。
**EditReward-Compass**：2251 个偏好对，模拟真实 RL optimization 中的奖励建模场景。

### 关键实验结果
作为基准论文：
- 提供前沿模型在六类任务上的细粒度评分，揭示既有评测无法暴露的差距；
- 奖励模型基准在更真实的 RL-like 配置下做评估。

### 局限性与开放问题
人工标注的偏差与一致性需独立审计；任务覆盖以英语 prompts 为主；动态编辑（视频）暂不涉及；EditReward 与实际 RL 收益的相关性需更多下游证据。

### 启发与应用前景
Edit-Compass 是面向 [[42-alphagrpo]] / [[10-flow-opd]] 这类生成 RL 工作的"标准考题"，对 reward model 开发者尤其关键。值得 follow：与 [[47-worldreasonbench]] 一起做"静态编辑 + 视频世界推理"双维评测。

---

## 46. Many-Shot CoT-ICL：上下文测试时学习的扩展规律
**👍 30** · [arXiv:2605.13511](https://arxiv.org/abs/2605.13511)

### 问题与动机
ICL 在 long-context 下可使用几十到几百示例，效果有时接近 fine-tuning。但当前 scaling 研究主要在非推理任务上——many-shot CoT-ICL 在推理任务上的行为完全不同。

### 方法与核心创新
作者系统研究 many-shot CoT-ICL，三个核心发现：
1. **设置依赖的 scaling**：增加 CoT demonstrations 在非推理 LLM 上**不稳定**，主要利好推理型 LLM；
2. **基于相似度的检索在推理任务上失败**——语义相似度无法预测"procedural（CoT）兼容性"；
3. **顺序扩展效应**：示例越多，**variance 越大**。
据此提出两条原则：示例需易于目标模型理解；要按"顺滑的概念递进"排序。基于此设计 **Curvilinear Demonstration Selection (CDS)**。

### 关键实验结果
- CDS 在 **64 个 demonstration、几何题** 上 **+5.42 个百分点**；
- 把"长上下文 = 检索缓冲区"重新定义为"上下文测试时学习的结构化课程"。

### 局限性与开放问题
方法主要在数学推理上验证，跨学科泛化待补；"概念递进"如何自动评估仍依赖启发式；与人工 CoT 数据质量耦合较紧。

### 启发与应用前景
"长上下文 = 课程而非缓冲"是非常重要的概念转变，对 RAG 与 prompt 工程有直接影响。值得 follow：把 CDS 整合到 [[20-autotts]] 的自动发现循环；与 [[17-rubricem]] 的 rubric-as-curriculum 做对照。

---

## 47. WorldReasonBench：视频生成的世界推理压力测试
**👍 30** · [arXiv:2605.10434](https://arxiv.org/abs/2605.10434) · [GitHub](https://github.com/UniX-AI-Lab/WorldReasonBench) · [Project](https://unix-ai-lab.github.io/WorldReasonBench/)

### 问题与动机
Seedance2.0、Veo3.1 等商业视频生成系统让"视频模型 = 世界模拟器"的提法升温，但社区缺一个评测：模型能否**根据初始状态 + 动作**生成物理、社交、逻辑、信息上一致的未来视频？

### 方法与核心创新
**WorldReasonBench** 把视频生成评测重构为 **world-state prediction**：
- **436 个 curated 测试**，结构化 QA 标注覆盖 4 个推理维度 / 22 个子类；
- 评测分两层：
  1. **Process-aware Reasoning Verification**：结构化 QA + 推理阶段诊断，检测时序 / 因果失败；
  2. **Multi-dimensional Quality Assessment**：评 reasoning 质量、temporal 一致性、视觉美学；
- 另引入 **WorldRewardBench**：~6K 专家标注偏好对、覆盖 1.4K 视频，用于 reward model 评估。

### 关键实验结果
- 跨现代视频生成器一致暴露 **视觉合理性 ≠ 世界推理**：视频可以好看但 dynamics / 因果 / 信息保持出错；
- 不同生成器在不同维度上有不同失败 pattern；
- 评测工具包将开源。

### 局限性与开放问题
436 题虽然 curated 但样本量对统计稳健性挑战仍在；reward 模型部分对 video preference 数据本身的噪声敏感；评估方法依赖人工 QA 设计，可扩展性需关注。

### 启发与应用前景
和 [[22-wam-survey]] / [[16-sana-wm]] 联读，WorldReasonBench 给"世界模型"评测立柱。值得 follow：把同思路扩到游戏世界模型评测；与 [[19-collabvr]] 的 VLM verifier 协同自动化打分。

---

## 48. Pixal3D：图像到 3D 的像素对齐生成
**👍 30** · [arXiv:2605.10922](https://arxiv.org/abs/2605.10922) · [GitHub](https://github.com/TencentARC/Pixal3D) · [Project](https://ldyang694.github.io/projects/pixal3d/)

### 问题与动机
图像到 3D 已经在几何分辨率与外观真实度上飞跃，但**保真度**（生成 3D 资产与输入图的 pixel-level 忠实度）仍是核心瓶颈。多数 3D 生成器在 canonical space 合成，再通过 attention 注入图像线索，留下模糊的 2D-3D 对应。

### 方法与核心创新
**Pixal3D**：抛弃 canonical pose，直接在 **pixel-aligned** 空间生成 3D，与输入视图一致。引入 **像素 back-projection conditioning**：把多尺度图像特征显式投影到 3D 特征体积，建立直接、无歧义的 pixel-to-3D 对应。自然扩展到 multi-view：聚合不同视角的 back-projected feature volume。配套提供 modular pipeline 生成对象分离的 3D 场景。

### 关键实验结果
- 大幅提升保真度，**接近"重建级"水平**——这是 3D 生成长期不可企及的目标；
- 在 single-view / multi-view / 场景合成上均工作；
- 首次在大规模上展示 3D-native 像素对齐生成。

### 局限性与开放问题
对输入图像质量与遮挡的鲁棒性需更全面评估；"像素对齐 ↔ 视角灵活"的 trade-off 不可避免；与基于 SDS / NeRF 的方法的成本对比披露较少。

### 启发与应用前景
Pixal3D 解决了 3D 生成最迫切的"可控保真"问题——对游戏资产、电商、AR/VR 直接落地。值得 follow：与 [[41-trackcraft3r]] 的几何先验联立做"动态 + 像素对齐"3D；探索把 Pixal3D 范式推到视频。

---

## 49. RouteProfile：LLM 路由的画像设计空间
**👍 30** · [arXiv:2605.00180](https://arxiv.org/abs/2605.00180) · [GitHub](https://github.com/ulab-uiuc/RouteProfile) · [Project](https://ulab-uiuc.github.io/RouteProfile/)

### 问题与动机
LLM 路由（按 query 把请求分发给最合适模型）研究多数聚焦"路由器机制设计"，**"模型画像"如何影响路由表现却被忽视**——而画像可能比 router 本身更关键。

### 方法与核心创新
把 LLM 画像视作"基于异构交互历史的结构化信息整合"问题，提出 **RouteProfile** 设计空间，沿四个维度展开：
1. **organizational form**：画像如何组织；
2. **representation type**：表示形式；
3. **aggregation depth**：聚合深度；
4. **learning configuration**：学习配置。
跨三个代表性 router、在标准 + new-LLM 泛化两类场景做系统评估。

### 关键实验结果
- **结构化画像稳定优于扁平画像**；
- **query-level 信号比 coarse domain-level 信号更可靠**；
- 在**新接入 LLM** 的泛化场景中，**结构化 + 可训画像**收益最大。

### 局限性与开放问题
画像构建本身的成本与时效（模型版本迭代后如何更新）未充分讨论；与商业路由方案（如 Martian、OpenRouter）的对照缺位；adversarial query 下 router 稳健性未涉及。

### 启发与应用前景
对所有 LLM 路由产品 / 多模型聚合平台都有直接指导：**先把模型画像做好，再谈路由器**。值得 follow：把 RouteProfile 与 [[35-merging-scaling-law]] 联合研究"路由 vs 合并"的资源 trade-off；构建画像 + 反馈循环做"实时演化路由"。

---

## 🗺️ 趋势洞察

### 1. 记忆是 2026-W20 最密集的研究主题
**涉及论文**：[#5 MemPrivacy](#5-memprivacy)、[#7 δ-mem](#7-delta-mem)、[#18 MemLens](#18-memlens)、[#26 MemEye](#26-memeye)、[#37 STALE](#37-stale)
**核心观点**：一周内有 5 篇高分论文同时围绕"长程 Agent 记忆"展开，且**每一篇切入角度都不同**——隐私保护型记忆（5）、轻量在线状态记忆（7）、跨模态记忆评测（18, 26）、记忆是否过期（37）。这意味着"记忆"已经从单一系统问题分化成一组子领域，且急需统一的评测与协议——目前各篇 benchmark 互不兼容。短期内不会出现"the memory paper"，更可能形成 RAG 当年那种"组件分工的生态"。

### 2. RL/蒸馏方法论的统一化与黑盒化
**涉及论文**：[#9 SDAR](#9-sdar)、[#10 Flow-OPD](#10-flow-opd)、[#17 RubricEM](#17-rubricem)、[#21 LPO](#21-lpo)、[#23 HyperEyes](#23-hypereyes)、[#39 ROPD](#39-ropd)、[#42 AlphaGRPO](#42-alphagrpo)
**核心观点**：GRPO 类 RL 在过去半年成为 LLM 后训练默认工具，本周则呈现两条交汇趋势：(a) **理论统一**（[#21](#21-lpo) 揭示所有 group-based 方法都在 simplex 上做投影），(b) **奖励替代**（[#17 rubric](#17-rubricem) / [#39 rubric](#39-ropd) / [#42 分解可验证奖励](#42-alphagrpo) 取代标量奖励 + 让黑盒教师可被蒸馏）。两条线合起来意味着：**未来一年 RL post-training 会从"哪个变种好"转向"哪个奖励信号更结构化"**。

### 3. 视频生成的 any-step / few-step 蒸馏走向实时化
**涉及论文**：[#10 Flow-OPD](#10-flow-opd)、[#11 AnyFlow](#11-anyflow)、[#12 Causal Forcing++](#12-causal-forcing-pp)、[#16 SANA-WM](#16-sana-wm)、[#40 Warp-as-History](#40-warp-as-history)、[#41 TrackCraft3R](#41-trackcraft3r)
**核心观点**：视频扩散正在同时压低"步数"（[#11 任意步](#11-anyflow) / [#12 1-2 步 frame-wise](#12-causal-forcing-pp)）、扩展"长度"（[#16 1 分钟世界模型](#16-sana-wm)）、增强"控制"（[#40 相机](#40-warp-as-history)）、并把生成 backbone 反用于"几何理解"（[#41 3D 追踪](#41-trackcraft3r)）。综合趋势是：**视频扩散模型正在从"离线生成器"变成"实时交互世界引擎 + 通用视觉先验"**——这一变化的产业影响很可能不亚于 GPT-3.5。

### 4. 多模态走向"原生统一"与"评测对齐"
**涉及论文**：[#3 SenseNova-U1](#3-sensenova-u1)、[#6 MulTaBench](#6-multabench)、[#8 Qwen-Image-2.0](#8-qwen-image-20)、[#13 MMProLong](#13-mmprolong)、[#27 Qwen-Image-VAE-2.0](#27-qwen-image-vae)、[#42 AlphaGRPO](#42-alphagrpo)、[#43 DRoRAE](#43-drorae)、[#45 Edit-Compass](#45-edit-compass)、[#47 WorldReasonBench](#47-worldreasonbench)
**核心观点**：本周多模态的两个明显信号是：(a) 架构层"原生统一"逐步从概念走向产品 ([#3](#3-sensenova-u1) NEO-unify、[#8](#8-qwen-image-20)/[#27](#27-qwen-image-vae) 阿里影像栈)，(b) 评测从 "co-occurrence" 升级到 "task-aware" / "world reasoning" ([#6](#6-multabench) / [#45](#45-edit-compass) / [#47](#47-worldreasonbench))。模型与基准在同步推进。

### 5. 具身智能与世界模型形成可复现的开源底座
**涉及论文**：[#16 SANA-WM](#16-sana-wm)、[#22 WAM 综述](#22-wam-survey)、[#25 CascadeBench](#25-cascadebench)、[#29 HumanNet](#29-humannet)、[#36 MCP-Cosmos](#36-mcp-cosmos)、[#47 WorldReasonBench](#47-worldreasonbench)
**核心观点**：本周一次出现了**世界模型自身**（[#16](#16-sana-wm)）、**领域综述**（[#22](#22-wam-survey)）、**数据底座**（[#29](#29-humannet)）、**接口范式**（[#36](#36-mcp-cosmos)）、**评测协议**（[#47](#47-worldreasonbench)）和**企业 vs 学术对比**（[#25](#25-cascadebench)）——这是世界模型领域罕见的"全栈出场"。意味着从下半年起，社区可以**复制粘贴**一套世界模型工作流而非各自手工搭。

### 对比与张力

- **离线学习 vs 运行时发现**：[#25 CascadeBench](#25-cascadebench) 与 [#22 WAM 综述](#22-wam-survey) 的张力——在可读规则环境（企业）下，把规则学进权重未必胜过 runtime read。这意味着具身 Agent 的"内化 vs 外化"边界要被重新定义。
- **长上下文 vs 记忆 Agent**：[#13 MMProLong](#13-mmprolong) 与 [#18 MemLens](#18-memlens) / [#26 MemEye](#26-memeye) 表明二者各擅胜场，目前**没有任一路线独立解决多模态长程问题**，未来一段时间应该混合架构占主导。
- **训练 vs 合并 vs 路由**：[#28 Darwin](#28-darwin) + [#35 Merging Scaling Law](#35-merging-scaling-law) 主张合并取代多任务训练；[#49 RouteProfile](#49-routeprofile) 主张多模型路由；两条路径在"特化能力来自哪里"上互相替代——这是 2026 下半场最关键的资源分配辩论。
- **步数蒸馏的两条路线**：[#11 AnyFlow](#11-anyflow) 推任意步数 + 越多越好，[#12 Causal Forcing++](#12-causal-forcing-pp) 推极少步 + 极低时延——两者分别面向"质量"和"实时"端，应被视作互补而非竞争。

### 值得关注的研究方向

1. **统一多模态记忆协议**：MemLens / MemEye / STALE / MemPrivacy 四篇都在挖记忆的不同断面，但 schema、评测、隐私模型各自为政。把它们统一成"多模态记忆 v1 协议"是高 ROI 的工作，且具备产品落地价值（数字员工、长程助手）。
2. **黑盒教师蒸馏的工业管线**：[#39 ROPD](#39-ropd) 解锁了用 GPT/Claude/Gemini 当教师的 OPD 可行性，配合 [#17 RubricEM](#17-rubricem) / [#42 DVReward](#42-alphagrpo) 的奖励替代，2026 下半年很可能涌现"黑盒蒸馏 + 自动 rubric 生成"的开源管线，要尽早布局基础设施（如 MinT 类基础设施 [#1](#1-mint)）。
3. **训练-free 模型组装**：[#28 Darwin](#28-darwin) + [#31 GCWM](#31-gcwm) + [#35 Merging Scaling Law](#35-merging-scaling-law) 三篇组合，能否催生"训练-free 模型工厂"？这种范式对算力受限团队意义巨大，是绕开大厂军备竞赛的差异化路径。
4. **AR + Frame-wise 视频实时化**：[#12 Causal Forcing++](#12-causal-forcing-pp) + [#16 SANA-WM](#16-sana-wm) + [#40 Warp-as-History](#40-warp-as-history) 三篇组合后已经离"游戏级实时世界模型"非常近，谁能整合出第一款消费级实时世界引擎 demo，谁就拿下 2026-2027 年视频生成的话语权。
5. **企业 Agent 的可读规则范式**：[#25 CascadeBench](#25-cascadebench) + [#36 MCP-Cosmos](#36-mcp-cosmos) + [#34 WildClawBench](#34-wildclaw) 三篇组合，把"企业 Agent = LLM + 工具 + 可读配置 + 世界模型"形式化。对企业 SaaS 与 B2B AI 公司是必读组合。



