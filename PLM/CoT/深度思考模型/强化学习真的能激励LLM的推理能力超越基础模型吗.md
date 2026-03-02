---
title: 强化学习真的能激励LLM的推理能力超越基础模型吗
created: 2026-02-28
tags:
  - rlhf
type: 论文
papername:
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - 清华
  - 上海交通大学
---

## 论文基本信息

标题：Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?


作者：

链接：

代码：

框架图：


## 简介

大型语言模型（LLMs）通过应用可验证奖励强化学习（RLVR）在复杂推理任务，尤其是数学和编程方面取得了显著成功。这一范式以OpenAI-o1和DeepSeek-R1等模型为代表，利用可计算的奖励对预训练模型进行微调，绕过了人工策划指令调优的局限。该领域的主流观点是，RLVR使LLM能够自主发展新的推理能力，类似于传统强化学习让智能体在围棋或国际象棋等游戏中发现新策略。

![图1：概念性示意和实证结果，展示RLVR对推理能力的影响](https://paper-assets.alphaxiv.org/figures/2504.13837v5/img-0.jpeg "Figure 1: Conceptual illustration and empirical results showing RLVR's effect on reasoning capacity") _图1：RLVR如何影响推理能力的概念性示意。左侧面板显示，虽然RLVR提高了现有解的抽样效率，但可能缩小了整体推理能力的范围。右侧面板在Omni-MATH-Train数据集上实证演示了这一点，展示了不同训练步骤如何影响pass@k曲线。_

然而，本研究通过系统调查挑战了这一基本假设。利用稳健评估指标，衡量模型在k次尝试内解决问题的潜力，而不仅仅是平均表现，作者揭示了一个显著发现：当前的RLVR方法主要优化基础模型中已有推理模式的抽样效率，而非真正创造新颖的推理能力。`pass@k`

## 问题陈述与方法论

本研究的核心问题是，当前的RLVR是否真正使LLM获得新的推理能力，还是仅仅优化了现有能力。传统的评估指标，如贪婪解码准确率，可能低估模型的真实潜力，因为它们未能考虑后续抽样尝试中可能出现的正确解。

为解决这一限制，作者采用了**`pass@k`公制**，从代码生成任务被改编到所有可验证的奖励场景。对于给定问题，如果采样输出中至少有一个正确，则为1，否则为0。该指标揭示了模型在k次试验中潜在解决问题的比例，提供了更全面的推理能力边界衡量。`pass@k`

实验设计包括：

- **模型系列**：Qwen2.5（7B/14B/32B）、LLaMA-3.1-8B，以及Mistral-Medium
- **强化学习算法**：PPO、GRPO、Reinforce++、RLOO、ReMax和DAPO
- **任务领域**：数学推理（GSM8K、MATH500、AIME24）、代码生成（HumanEval+、MBPP+）和视觉推理（MathVista、MathVision）
- **训练方案**：数学任务使用零强化环境，编码/视觉任务则采用指令调整起点

该方法包括超越简单曲线的复杂分析技术，如准确率分布分析、可解问题覆盖分析和困惑度分析，以确定RLVR生成的解是否已在基模型分布内。`pass@k`

## 主要发现

系统性调查揭示了若干反直觉的发现，挑战了关于RLVR疗效的传统看法：

**RLVR提高了抽样效率，但缩小了推理覆盖范围**

![图2：不同模型族和数学推理基准之间的Pass@k曲线](https://paper-assets.alphaxiv.org/figures/2504.13837v5/img-1.jpeg "Figure 2: Pass@k curves across different model families and mathematical reasoning benchmarks") _图2：跨模型家族（Qwen-2.5-7B、14B、32B和LLaMA-3.1-8B）及数学基准（AIME24、MATH500、Minerva、Olympiad）的综合pass@k分析。一致的模式显示，强化学习模型在低k值时优于基础模型，但在高k值时表现逊色。_

虽然RLVR训练模型在k值较小（例如）时持续优于基础模型，但随着k值增加，这一优势会相反。在k值较大（k=128，k=256）时，基础模型在所有基准和模型族中得分更高。这一模式表明RLVR优化了正确响应的高效采样，但实际上缩小了整体可解问题的范围。`pass@1``pass@k`

**没有新颖推理模式的生成**

研究表明，RLVR模型生成的推理路径在基础模型分布下具有较低的困惑度，表明这些路径已在基础模型的能力范围内存在。准确率分布分析表明，性能提升来自于更高效地采样已可解问题的解，而非使模型能够解决根本性新问题。

![图3：基础模型与强化模型的准确性分布比较](https://paper-assets.alphaxiv.org/figures/2504.13837v5/img-4.jpeg "Figure 3: Accuracy distribution comparison between base and RL models") _图3：MATH500的准确率分布分析显示RLVR提高了高精度（接近1.0）和零精度问题的频率，表明是对现有能力的优化，而非问题解决范围的扩展。_

**强化学习算法间的一致性性能**

“抽样效率差距”（∆_SE = RL模型的pass@1 - 基础模型的pass@256）在不同RL算法中始终保持较大，仅有细微变化。这表明当前的RLVR方法远未充分发挥基础模型固有的能力。

![图4：算法比较显示采样效率差距一致](https://paper-assets.alphaxiv.org/figures/2504.13837v5/img-7.jpeg "Figure 4: Algorithm comparison showing consistent sampling efficiency gaps") _图4：跨数学基准测试的不同强化学习算法比较，显示采样效率差距（∆_SE）一致且在所有方法中依然存在显著，显示基础模型能力的利用仍有显著改进空间。_

## 对比RLVR与蒸馏

与RLVR的局限形成鲜明对比的是，研究表明，从更强的教师模型中提炼出来确实拓宽了推理的边界。在比较RLVR训练模型与蒸馏模型（如DeepSeek-R1-Distill-Qwen-7B）时，蒸馏持续使整个曲线高于基础模型，表明引入了新的推理模式。`pass@k`

![图5：RLVR与蒸馏对推理边界影响的比较](https://paper-assets.alphaxiv.org/figures/2504.13837v5/img-6.jpeg "Figure 5: Comparison of RLVR and distillation effects on reasoning boundaries") _图5：在Minerva基准测试下RLVR与蒸馏的比较。提炼模型（黑线）在所有k值下均有持续改善，而强化模型则表现出早期增长，k值高时覆盖率下降的典型模式。_

这一比较凸显了一个根本区别：提炼可以从更强的教师模型中引入真正的新能力，而当前的RLVR方法主要在基础模型分布内重组现有知识。

## 影响与未来方向

这项研究对该领域对RLVR的理解以及未来推理能力LLMs的发展具有深远意义。通过证明当前RLVR主要提升采样效率而非产生新能力，该研究挑战了这些方法实现持续自我进化的乐观观念。

这些发现为未来研究提出了几个有前景的方向：

1. **有效的探索机制**：超越代币级采样，探索更高级的抽象空间（例如程序级演进）
    
2. **课程学习**：实施结构化课程，通过精心设计的问题进展逐步扩展模型能力
    
3. **流程层级奖励**：包含细致信号，奖励正确的中间推理步骤，而不仅仅是最终结果
    
4. **多回合互动**：开发能够通过环境反馈和工具使用实现迭代优化的agentic强化学习方法
    

研究还确立了评估大型语言模型推理能力的更高标准，强调不仅要考虑平均表现，还要考虑模型解决问题能力的广度和来源。从“有多容易”转变为“模型能解决哪些问题”，为真实的能力评估提供了关键洞察。

在承认该领域快速发展及专有前沿模型访问的局限性下，这项工作为重新思考强化学习在开发真正有能力的推理系统中的作用提供了坚实基础。这些发现既是对现有方法的警醒评估，也是为释放大型语言模型推理潜力的更有效方法提供了路线图。

## 主要收获


## 参考资料
