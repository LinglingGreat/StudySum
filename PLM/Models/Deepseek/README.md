## DeepSeek-LLM（67B / 7B等）

DeepSeek 在2023年11月底开源了名为 “DeepSeek-LLM” 的模型系列（包括 7B / 67B 基础模型 + Chat 版本）

论文地址：https://arxiv.org/abs/2401.02954

模型地址：https://huggingface.co/collections/deepseek-ai/deepseek-llm-65f2964ad8a0a29fe39b71d8

Github地址：https://github.com/deepseek-ai/DeepSeek-LLM

**架构 / 能力 /定位**

- 这是一个较为“传统”的密集 decoder-only Transformer 架构（即标准的自注意力 + 前馈网络堆叠）

- 7B模型采取Multi-Head Attention，67B模型采取Grouped-Query Attention。

- 2个模型都使用2T tokens的英文 + 中文双语训练，倾向成为通用语言模型（不是刻意偏工具 / 推理方向）
    
- 在很多常见任务（写作、摘要、对话、翻译、常识问答）上，就开源模型而言，表现通常优于当时同规模的 LLaMA /其他开源模型。[GitHub](https://github.com/deepseek-ai/DeepSeek-LLM)
    
- 作为早期版本，它更多扮演“基础模型 / 研究基座”的角色，而不是高精尖推理模型
    

**局限 /挑战**

- 在大规模推理、复杂数学 / 逻辑题、链式思维 / 工具调用等方面能力不足
    
- 模型规模和资源消耗限制其在生产环境中的应用
    
- 对齐 / 输出一致性 /安全性 /控制能力尚未成熟


## DeepSeek-V3

**发布时间 / 背景**

- DeepSeek 团队于 2024 年底发布 V3 版本，替代之前的 DeepSeek-V2.5。([维基百科](https://zh.wikipedia.org/wiki/DeepSeek-V3?utm_source=chatgpt.com "DeepSeek-V3"))
    
- 官方资料中称 V3 是一个 Mixture-of-Experts (MoE) 结构的语言模型，具有 671B 总参数规模，每个 token 激活约 37B 规模的子网络。([GitHub](https://github.com/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3 - GitHub"))
    
- 架构上，V3 使用 Multi-head Latent Attention (MLA) 与 DeepSeekMoE 架构以控制计算和效率。([GitHub](https://github.com/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3 - GitHub"))
    

**主要特点 / 局限**

- 高参数但稀疏激活（MoE 设计）以平衡性能与计算成本。([GitHub](https://github.com/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3 - GitHub"))
    
- 强调在数学、编码、中文等任务上的能力，并与当时的开源与闭源模型竞争。([维基百科](https://zh.wikipedia.org/wiki/DeepSeek-V3?utm_source=chatgpt.com "DeepSeek-V3"))
    
- 在推理、复杂逻辑任务、工具调用 (tool use / agent) 等方面仍有提升空间。即在复杂链式思考、工具整合、多步 agent 任务等场景，性能略受局限。
    

因此，后续版本的主要方向通常围绕：**更强推理能力 / 混合思考能力 / 长上下文 / 工具调用 / 效率优化** 做增强。

---

## DeepSeek-V3-0324

这是 DeepSeek 在 2025 年 3 月 24 日发布的 V3 系列更新版本，是在原 V3 架构基础上的改进「checkpoint」版本。

**公告 / 架构关系**

- 官方明确指出：V3-0324 的模型结构与 DeepSeek-V3 **完全相同**，只是一个新的 checkpoint（权重更新、训练调优）。([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3-0324 - Hugging Face"))
    
- 在 API 文档中称，该版本在推理性能、前端工具能力、推理 & 逻辑性能等方面都有“重大提升（Major boost）”。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250325?utm_source=chatgpt.com "DeepSeek-V3-0324 Release"))
    
- 它继续支持函数调用 (function calling)、JSON 输出、FIM 完成等接口功能。([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3-0324 - Hugging Face"))
    

**改进 / 优势**

- 推理能力有显著提升：官方在更新日志中说 “Major boost in reasoning performance” ([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250325?utm_source=chatgpt.com "DeepSeek-V3-0324 Release"))
    
- 工具使用、前端开发能力更强：公告中提 “Stronger front-end development skills / Smarter tool-use capabilities” ([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250325?utm_source=chatgpt.com "DeepSeek-V3-0324 Release"))
    
- API 使用方式保持不变，对调用接口兼容性友好。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250325?utm_source=chatgpt.com "DeepSeek-V3-0324 Release"))
    
- 许可方面：V3-0324 使用 MIT License 发布（更开放）([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250325?utm_source=chatgpt.com "DeepSeek-V3-0324 Release"))
    
- 在 benchmark 上，对比原 V3 的多个维度有上升（例如 MMLU、GPQA、AIME、LiveCodeBench 等）([DeepSeek API Docs](https://api-docs.deepseek.com/updates?utm_source=chatgpt.com "Change Log | DeepSeek API Docs"))
    

**用户 /社区反馈**

- 有用户在 Reddit 上表示 “V3 0324 显著优于 V3 OG（原版）” ([Reddit](https://www.reddit.com/r/SillyTavernAI/comments/1jk8ppf/deepseek_v3_0324_is_incredible/?utm_source=chatgpt.com "DeepSeek V3 0324 is incredible : r/SillyTavernAI - Reddit"))
    
- 有文章评价：在逻辑、编码、结构化问题解答方面有明显提升，有时甚至超过 Claude 3.7 在某些维度的表现。([Medium](https://medium.com/data-science-in-your-pocket/deepseek-v3-0324-vs-deepseek-v3-b4bd73e39bec?utm_source=chatgpt.com "DeepSeek V3–0324 vs DeepSeek-V3 - Medium"))
    
- 部分社区文章把它称为“V3 的重大升级版”([Milvus](https://milvus.io/blog/deepseek-v3-0324-minor-update-thats-crushing-top-ai-models.md?utm_source=chatgpt.com "DeepSeek V3-0324: The \"Minor Update\" That's Crushing Top AI ..."))
    

**总结**  
V3-0324 可视为对 V3 的中期强化补丁（checkpoint upgrade），在推理能力、工具调用、稳定性方面有较为显著提升，但不是架构上的重大革新。架构仍为原来的 V3，参数规模与设计保持一致。

---

## DeepSeek-V3.1

这是 DeepSeek 在 2025 年 8 月发布的 “新一代 V3 系列” 版本，是一个在架构与训练策略上有实质性改动的版本。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250821?utm_source=chatgpt.com "DeepSeek-V3.1 Release"))

**主要新特性 / 架构设计**

1. **Hybrid 推理 / 模式切换（Thinking / Non-Thinking）**
    
    - DeepSeek-V3.1 引入一种混合推理架构：一个模型同时支持 **Thinking 模式**（链式思考 / 中间过程推理）与 **Non-Thinking 模式**（直接输出答案）两种推理方式。用户/系统可以通过 chat 模板或切换标识来选择。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250821?utm_source=chatgpt.com "DeepSeek-V3.1 Release"))
        
    - 在内部评测中，V3.1-Think 在推理效率上比 R1 更快，减少推理时的中间 token 耗费。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250821?utm_source=chatgpt.com "DeepSeek-V3.1 Release"))
        
2. **更强的工具调用 / Agent 能力**
    
    - V3.1 在 post-training（后训练 / fine-tuning）阶段加强了工具调度、函数调用、agent 多步骤任务处理能力。公告中提“Stronger agent skills” ([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250821?utm_source=chatgpt.com "DeepSeek-V3.1 Release"))
        
    - 在 API 更新日志中，V3.1 的 `deepseek-chat` 与 `deepseek-reasoner` 分别对应 Non-Thinking / Thinking 模式。([DeepSeek API Docs](https://api-docs.deepseek.com/updates?utm_source=chatgpt.com "Change Log | DeepSeek API Docs"))
        
3. **上下文长度 / 训练扩展**
    
    - V3.1 支持更长上下文，目前主流版本支持 128K tokens 上下文长度（或已经扩展）([bdtechtalks.substack.com](https://bdtechtalks.substack.com/p/deepseek-v31-is-here-heres-what-you?utm_source=chatgpt.com "DeepSeek-V3.1 is here. Here's what you should know. - TechTalks"))
        
    - 在训练数据量上，V3.1 在 V3 基础上进行了更多训练（例如扩展上下文训练阶段）([bentoml.com](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond?utm_source=chatgpt.com "The Complete Guide to DeepSeek Models: V3, R1, V3.1 and Beyond"))
        
4. **优化精度 / 兼容更多数值格式**
    
    - V3.1 在内部支持多种精度格式（如 BF16, FP8, F32 等），以更好适配不同硬件 / 芯片平台。([Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/08/deepseek-v3-1-quiet-release-big-statement/?utm_source=chatgpt.com "DeepSeek V3.1: Quiet Release, Big Statement - Analytics Vidhya"))
        
    - 对于国内芯片生态进行了优化支持（如在公告中提到对国产芯片 / 生态的适配）([eWeek](https://www.eweek.com/news/deepseek-introduces-deep-thinking-mode/?utm_source=chatgpt.com "DeepSeek V3.1 Outperforms Popular R1 in Benchmarks - eWeek"))
        
5. **兼容 / 接口更新**
    
    - V3.1 在 API 文档中宣称支持函数调用、长上下文、Anthropic API 兼容等。([eWeek](https://www.eweek.com/news/deepseek-introduces-deep-thinking-mode/?utm_source=chatgpt.com "DeepSeek V3.1 Outperforms Popular R1 in Benchmarks - eWeek"))
        
    - 企业 / 开发者可通过同一模型实现 Non-Thinking 与 Thinking 模式，无需切换模型版本。([bentoml.com](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond?utm_source=chatgpt.com "The Complete Guide to DeepSeek Models: V3, R1, V3.1 and Beyond"))
        

**版本稳定性 / 性能修正（Terminus）**

- 在 2025 年 9 月 22 日，DeepSeek 发布 **V3.1-Terminus** 版本，作为对 V3.1 的稳定性、输出一致性、语言混用（中英文混杂）、工具调用错误等问题的修正升级。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250922?utm_source=chatgpt.com "DeepSeek-V3.1-Terminus"))
    
- V3.1-Terminus 主要改进包括语言一致性（减少中英文混杂、奇怪字符现象）、更稳定输出、更优化 agent / tool 使用性能。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250922?utm_source=chatgpt.com "DeepSeek-V3.1-Terminus"))
    
- 在性能基准测试方面，V3.1-Terminus 相比 V3.1 在多个评测项目上略有提升（如 MMLU-Pro、GPQA、工具调用基准等）([Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF?utm_source=chatgpt.com "unsloth/DeepSeek-V3.1-Terminus-GGUF - Hugging Face"))
    
- 在 API 变更日志中，`deepseek-chat`（Non-Thinking）与 `deepseek-reasoner`（Thinking）均升级到 Terminus 版本。([DeepSeek API Docs](https://api-docs.deepseek.com/updates?utm_source=chatgpt.com "Change Log | DeepSeek API Docs"))
    

---

## DeepSeek-V3.2-Exp

这是 DeepSeek 在 2025 年 9 月 29 日发布的 “实验性” 模型，作为 V3 系列向下一个大架构演进的过渡版本。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250929?utm_source=chatgpt.com "Introducing DeepSeek-V3.2-Exp"))

**核心新机制 / 架构创新**

- **DeepSeek Sparse Attention (DSA)**：V3.2-Exp 引入了一个稀疏注意力机制（DeepSeek Sparse Attention），用于在长上下文场景下减少计算和内存开销，同时保持输出质量几乎不变。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250929?utm_source=chatgpt.com "Introducing DeepSeek-V3.2-Exp"))
    
- 架构上，V3.2-Exp 是在 V3.1-Terminus 基础上继续训练并注入 DSA 的版本。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250929?utm_source=chatgpt.com "Introducing DeepSeek-V3.2-Exp"))
    
- 在 benchmark 对比中，V3.2-Exp 在多个公开基准上的性能几乎与 V3.1-Terminus 相当。([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3.2-Exp"))
    

**效率 / 成本优化**

- 使用稀疏注意力机制后，在长文本 / 大上下文推理时，计算成本/内存开销大幅下降。公开资料称在长上下文场景下每百万 token 的推理成本低于 V3.1-Terminus 的一半。([Venturebeat](https://venturebeat.com/ai/deepseeks-new-v3-2-exp-model-cuts-api-pricing-in-half-to-less-than-3-cents?utm_source=chatgpt.com "DeepSeek's new V3.2-Exp model cuts API pricing in half to ..."))
    
- 在新闻报道中指出，DeepSeek 将相应地将 API 价格下调 50%+。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250929?utm_source=chatgpt.com "Introducing DeepSeek-V3.2-Exp"))
    

**局限 / 注意事项**

- 虽然在公开基准上表现相当，但 DeepSeek 官方和社区警告说，在真实用户场景的“长尾用例 / 异常输入”上，V3.2-Exp 仍需更广泛测试。([DeepSeek API Docs](https://api-docs.deepseek.com/guides/comparison_testing?utm_source=chatgpt.com "V3.1-Terminus Comparison Testing"))
    
- 因为是较新的实验版本，某些边缘情况、稳定性、输出一致性、兼容性可能尚未完全打磨。([DeepSeek API Docs](https://api-docs.deepseek.com/guides/comparison_testing?utm_source=chatgpt.com "V3.1-Terminus Comparison Testing"))
    

---

## 各版本的总结对比与演进趋势

| 版本            | 推理能力 / 逻辑 / 工具调用       | 上下文 / 长文本处理      | 效率 / 成本              |
| ------------- | ---------------------- | ---------------- | -------------------- |
| V3（原版）        | 基本的推理与逻辑能力，对复杂链式思考弱    | 支持较长上下文（128K）    | 作为基线成本与效率已有优化        |
| V3-0324       | 推理、工具调用、性能整体提升         | 同原架构支持           | 成本略有优化 / 性能提升        |
| V3.1          | 更强工具 / agent 能力，更高效推理  | 支持 128K 或更长上下文   | 支持更多精度格式 / 硬件适配优化    |
| V3.1-Terminus | 在工具 / agent、推理一致性上略优   | 同 V3.1 支持        | 稳定性 / 一致性更好          |
| V3.2-Exp      | 推理能力维持 ≈ V3.1-Terminus | 更适合极长上下文 / 长文档处理 | 推理 / 训练效率 / 内存开销显著下降 |


## DeepSeek 系列：各版本的总结对比


| 版本 / 模型                            | 时间 / 发布时期                  | 架构 / 模型规模 / 激活方式等                                                                                          | 主要特性 / 所在定位                                                        | 与前代相比的改进 / 差异 /不足                                                                                                          |
| ---------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **DeepSeek-LLM（或称 DeepSeek 基础系列）** | 2023 年底 / 11 月             | 公布有 7B / 67B 参数版本（Base 与 Chat）                                                                             | 作为 DeepSeek 的最早阶段模型，用来做通用语言理解 / 生成 / 对话                            | 架构相对传统（decoder-only transformer + RoPE、分组 query attention 等），上下文长度为 4096；在能力上与当时主流开源 LLM 比较接近，但在大规模推理 / 复杂推理 / 工具调用等方面还受限。 |
| **DeepSeek-V2**                    | 2024 年 5 月（或中期）            | 混合稀疏架构（Mixture-of-Experts, MoE），采用 Multi-head Latent Attention (MLA)，总参数 236B，每 token 激活 ~21B。支持长上下文（128K） | 目标是在提升模型能力的同时显著降低资源 / 内存开销 / KV 缓存占用 / 推理成本                        | 与前代 DeepSeek-LLM 相比，在规模、稀疏计算、上下文长度、性能效率等多方面都有跳跃；训练成本比起全激活模型显著下降（节省 ~42.5%）；KV 缓存压缩（减少 ~93.3%）；推理吞吐率提升（5.76×）               |
| **DeepSeek-V2.5**                  | 2024 年 / 后期                | 在 V2 基础上做进一步融合 / 优化，包括将通用模型 (V2-Chat) 与 Coder 模型融合 / 协调能力                                                  | 更好地兼顾 general 和 编码 (code) 能力；提升 instruction following、写作 / 推理能力等指标 | 相比 V2，在 instruction 对齐、编程能力、对任务的多样性适应上更强；但相比后续 V3 / R1 版本，其在复杂推理 / reasoning 任务上仍是下游补强对象                                   |
| **DeepSeek-V3（基线版）**               | 2024 年底 / 2024 年底至 2025 年初 | 混合稀疏 / MoE 设计（671B 模型规模，稀疏激活）                                                                              | 作为 DeepSeek 的“下一代”旗舰版本，是 V3 系列的基础版本                                | 相比 V2 / V2.5，有到更大规模、更复杂能力与潜在能力空间；但在逻辑推理 / agent / 工具调用等场景仍需增强（这正是后续版本努力方向）                                                 |
| **DeepSeek-V3-0324**               | 2025 年 3 月 24 日            | 与 V3 基线模型架构相同，但用新的 checkpoint / 权重训练 / 调优                                                                  | 更好的推理、工具调用、稳定性表现                                                   | 相较于 V3 基线版，在推理能力、工具能力、稳定性等方面有“重大提升”但不是结构性变更                                                                                |
| **DeepSeek-V3.1**                  | 2025 年 8 月                 | 架构升级：支持 **混合推理模式**（Thinking / Non-Thinking），更深入强化的工具 / agent 能力                                            | 旨在兼顾日常对话生成与复杂推理 / agent 调度能力                                       | 相比 V3 / V3-0324，在推理效率、工具调用、混合模式支持等方面有结构性改进                                                                                 |
| **DeepSeek-V3.1-Terminus**         | 2025 年 9 月（版本稳定包）          | 基于 V3.1 的稳定版本，对输出一致性、语言混杂、不稳定行为等做修正                                                                        | 更加适合生产环境 / 应用部署                                                    | 相比 V3.1，在稳定性、语言一致性、边界行为（异常输入）等方面改善                                                                                         |
| **DeepSeek-V3.2-Exp**              | 2025 年 9 月末                | 在 V3.1 基础上引入 **DeepSeek Sparse Attention (DSA)** 稀疏注意力机制，以降低计算 / 内存成本                                      | 面向长上下文 / 高效推理场景                                                    | 在长文本场景下效率 / 成本有显著改进；但作为实验版本，在边界稳定性 / 稳妥性上尚需更多验证                                                                            |

此外，还有 **DeepSeek-R1** 这一支线 / 推理 / reasoning 专用方向的模型：

- **DeepSeek-R1 / R1-Zero**：作为 DeepSeek 的 reasoning 方向探索版本（即更偏向让模型“思考”能力 / 逻辑推理能力）([arXiv](https://arxiv.org/abs/2501.12948?utm_source=chatgpt.com "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via ..."))
    
    - R1-Zero 是用纯强化学习 (RL)（无监督微调 / SFT）来训练出推理能力，但其初版存在可读性 / 语言混杂问题。([arXiv](https://arxiv.org/abs/2501.12948?utm_source=chatgpt.com "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via ..."))
        
    - R1 在 R1-Zero 基础上加入 cold-start 数据 + SFT + RL，使输出可读性、语言一致性、推理链等更稳定。([DeepSeek API Docs](https://api-docs.deepseek.com/news/news250120?utm_source=chatgpt.com "DeepSeek-R1 Release"))
        
    - R1 在一些 benchmark 上能与 OpenAI 的 o1 模型匹敌（甚至在某些任务上优于）([InfoQ](https://www.infoq.com/news/2025/02/deepseek-r1-release/?utm_source=chatgpt.com "DeepSeek Open-Sources DeepSeek-R1 LLM with Performance ..."))
        
    - R1 的推理链 / 多步思考过程被公开为 “thought chains” / 可观察思考过程（即模型自身的思考轨迹）([arXiv](https://arxiv.org/abs/2504.07128?utm_source=chatgpt.com "DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning - arXiv"))
        
    - 有研究指出 R1 输出在部分政治敏感 / 本地化语境下可能带有过滤 / 审查倾向。([arXiv](https://arxiv.org/abs/2505.12625?utm_source=chatgpt.com "R1dacted: Investigating Local Censorship in DeepSeek's R1 Language Model"))
        

综合来看，DeepSeek 的演进路径可以概括为：

1. 从相对密集 / 传统 transformer 模型（DeepSeek-LLM）到稀疏 / MoE 架构（V2）。
    
2. 在稀疏架构基础上不断在效率、上下文长度、KV 缓存、路由优化等方面打磨（V2 → V2.5 → V3）。
    
3. 在 V3 架构基础上探索混合推理模式、工具 / agent 加强、输出一致性、思考链可解释性等方向（V3.1 / Terminus / V3.2 Exp）。
    
4. 从通用能力过渡到 **推理 / 思考能力导向**（R1 系列），即不只是生成 / 模拟语言，而是让模型“思考 / 推理 / 算法式解决问题”更强。
    
**演进趋势解读**：

1. **从架构稳定 + checkpoint 优化 → 架构演进**：  
    V3 → V3-0324 是在同构架构上的训练优化；而 V3.1 是引入混合推理的新架构；V3.2-Exp 则尝试引入稀疏注意力做机制创新。
    
2. **能力向“推理 + 工具调用 + Agent 任务”拓展**：  
    随着版本推进，DeepSeek 越来越重视工具调用 (function / API)、多步骤 agent 任务、插件调用等复杂工作流场景。
    
3. **效率 / 成本压缩 / 硬件适配**：  
    V3.2-Exp 的稀疏注意力是为了解决在极长上下文场景下的计算 / 内存瓶颈；同时 V3.1 在支持多精度格式、适配本地 / 国产硬件方面下功夫。
    
4. **稳定性 / 一致性优化**：  
    V3.1-Terminus 的推出正是为了解决 V3.1 在输出一致性、语言混用、奇怪字符、模糊边缘行为等方面的问题，从而提升在应用/部署中的鲁棒性。
    
5. **“兼顾通用 + 专注推理”的折衷**：  
    V3.1 的混合推理模式使得一个模型能同时兼顾日常对话/生成和复杂推理场景，是 DeepSeek 在宽泛能力与专精任务间的折中路径。