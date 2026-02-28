---
title: GLM5
created: 2026-02-28
tags:
  - 大模型
type: 论文
papername:
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2026
institution:
  - 智谱
---

# 论文基本信息

标题：

作者：

链接：

代码：

框架图：


# 模型架构
首先是**尺寸扩容**，GLM-5选择了往更大模型训这条路线，提升整体模型能力。专家数量从160个扩展至256个，总参数量从 355B 提升至 744B，激活参数从 32B 增至 40B。同时减少层数，降低专家并行通信overhead。

其次是**200K长文**支持，GLM-5使用**DSA稀疏注意力**来提升长文效率，在长序列下attention计算消耗可降低1.5-2倍。值得注意的是，GLM-5放弃了GLM-4.5中的 GQA (Grouped-Query Attention)，转而使用 MLA (Multi-latent Attention)，因为两者效果一样的情况下MLA更省显存、长文速度更快、且对国产芯片兼容性更好。这里作者做了两个改动：

1. 最开始576维的MLA的表现不如GQA，这是由于MLA将KV缓存维度从GQA的2048维缩减至576维，导致其投影矩阵的参数表达空间被压缩。若采用传统Muon的全局正交化，会进一步限制不同注意力头的功能分化，部分头可能因全局优化丢失关键特征。因此作者使用Muon Split，将三个全局投影矩阵，按注意力头拆分为多个独立的小型投影矩阵（每个注意力头对应一个子矩阵），让每个注意力头的子矩阵在有限维度内依然能保持足够的表达多样性，追平了GQA。
    
2. MLA虽然省了KV cache的显存，但在解码时计算成本更高，因此作者将MLA头维度从192提升至256（+33%），头数量减少1/3，即保持训练阶段的计算量和参数总量不变，但头数量减少直接让解码阶段的总计算量降低 1/3，提升了解码速度。 值得注意的是，**GLM-5仅在midtrain的200K阶段通过20B tokens完成DSA适配**，显著少于DeepSeek V3.2中的943B，这一发现可以显著降低新结构适配成本。


# 训练数据

  

![GLM-4.5 vs GLM-5 训练数据对比](https://mmbiz.qpic.cn/mmbiz_png/2oicuz5vRaSviab8P3wticUCY5FTbMhZnz8OdUxKjD8zPN9via0kajK3DbNloQg8vfAne7ibiaZqibNhWbrI1DQj2bWltag1I2T5ibXESLQUnzsUJhI/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=0)数据对比

## 预训练

如图所示，GLM-5在预训练阶段增加了5T数据，除了web语料的筛选优化外，专门增加了很多推理类数据，同时也进一步提升了质量和采样策略：

1. **扩增code网页数据，新增28% code语料**；修复 Software Heritage 代码文件的元数据对齐问题，减少数据噪声，并且提升Scala、Swift、Lua 等低资源编程语言的采样质量。
    
2. **从网页、书籍、论文中采集高质量相关数据，同时改进网页内容提取流水线与书籍/论文的PDF解析机制，从源头提升数据质量**；利用大型语言模型对候选文档进行评分，仅保留教育价值最高的内容，通过“分块-聚合”打分，提升长文本数据的评分准确性；优化过滤pipeline，明确排除合成数据及模板化数据，确保语料的真实性与可靠性，避免无效噪声对模型推理能力的干扰。
    

## Midtrain

在midtrain阶段，从上面的对比图可以看到32K数据量没变，但128K数据翻了5倍，扩充了很多长文的推理和agent数据，核心优化如下：

1. 在128K和200K的后期训练阶段，重点**上采样长文档和合成智能体轨迹数据**，强化模型对长序列内容的理解与 agentic任务适配能力。
    
2. 筛选出约1000万条issue–PR对，**总计160B软件工程场景数据**。在拓宽数据来源的同时，强化单条issue级别的质量过滤，减少噪声；为每条issue–PR对检索更多相关文件，丰富开发上下文，提升模型对真实软件开发流程（如漏洞修复、功能开发）的适配性。
    
3. 补充更多长文数据：从书籍、学术论文、通用预训练语料中筛选，经多阶段过滤（PPL 评分、去重、长度筛选）后，上采样知识密集型领域数据，**保障基础长文本质量**；借鉴 NextLong 和 EntropyLong 技术，通过交错打包高度相似文本构建长序列，缓解 “中间信息丢失” 问题；在200K阶段新增MRCR类多轮对话数据变体，强化超长上下文下的多轮交互召回能力。实验显示200K阶段训练不仅优化了超长文本处理能力，还进一步提升了模型在128K上下文窗口内的性能。
    

## Post-train

在冷启动SFT阶段，GLM-5显著扩充了Agent和Coding数据的量级，同时也包含200K长文数据。具体的数据优化如下：

1. **对于通用数据，优化回复风格，相比 GLM-4.5 更具逻辑性和简洁性**，提升实际交互体验。可能是考虑到open-router上的角色扮演类调用，GLM-5也专门优化了角色扮演类数据，通过构建覆盖多语言、多角色配置的多样化数据集，丰富场景适配性。同时他们给出了**内部的角色扮演任务五大评估维度：指令遵循度、语言表达力、创造性、逻辑连贯性、长对话一致性**。
    
2. 对于逻辑推理能力，合成可验证的逻辑推理问题，采用拒绝采样技术合成高质量数据，确保推理过程的严谨性；基于难度对数学和科学类数据进行筛选，**仅保留对 GLM-4.7 模型具有挑战性的题目**，强化模型对复杂推理任务的适配能力。
    
3. **对于Agentic Coding任务，GLM-5搭建了大量执行环境**，重点采集真实世界场景和长周期任务的训练轨迹，提升数据的实用性与场景贴合度，并且结合专家强化学习与拒绝采样技术优化SFT数据，剔除低质量样本。对于错误数据，保留轨迹中的错误片段，但在损失函数中对其进行掩码处理，既让模型学习错误修正逻辑，又避免强化错误行为。
    

# 后训练策略

![图片](https://mmbiz.qpic.cn/mmbiz_png/2oicuz5vRaStYkZCG2zuSbCj2cNjfQ8QgaBVJ7Sm0j0c6PHnCqrEfAtiaT0uEHE3WzbpZshTsVicsaKCicRF9rTOgxoYe98R4VvJpuw11XPq7QY/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=1)

## RL

如图所示，GLM-4.5和GLM-5在后训练的stage-1相同，都是冷启动SFT，再进行3个专项RL（Reasoning、Agentic、General）。GLM-5对3个专项RL的优化如下：

1. Reasoning：基于GRPO框架融入IcePop技术，通过区分训练策略与推理策略、移除KL正则化，缓解训练-推理偏差；针对 DSA架构优化训练稳定性，采用确定性torch.topk算子并冻结索引器参数，避免非确定性算子导致的性能下降；混合训练模式，融合数学、科学、代码、工具集成推理（TIR）四类任务，统一优化推理链条，同时对数据进行难度过滤，聚焦GLM-4.7难以解决的高价值问题，提升复杂推理能力。
    
2. Agentic：GLM-5颠覆了GLM-4.5的同步RL框架，构建全异步decoupled训练体系：通过Multi-Task Rollout Orchestrator decouple推理与训练引擎，结合Token-in-Token-out（TITO）网关和令牌级裁剪的Direct Double-sided Importance Sampling，解决异步训练的离策略偏差与重 token 化匹配问题；拓展 10K + 可验证环境，覆盖真实SWE任务、终端任务和高难度多跳搜索任务，提升场景适配性；新增DP-aware路由优化，最大化 KV 缓存复用，加速长上下文推理；采用迭代自蒸馏与动态采样策略，强化模型在长周期、复杂环境中的自主规划与自我修正能力。
    
3. 相较于GLM-4.5的多源反馈体系，GLM-5构建三维优化目标与混合奖励系统：以基础正确性、情绪智能、任务特定质量为核心目标，整合规则型奖励、结果奖励模型（ORM）、生成奖励模型（GRM），平衡精准性、效率与鲁棒性，减少reward hacking；引入人类撰写的响应作为风格与质量锚点，避免模型生成机械化的表达，提升交互自然度；细化任务覆盖，针对指令遵循、函数调用、病理修正等场景设计专属训练方案，其中函数调用RL分为分步规则型与端到端多轮两种模式，病理RL主要优化语言混合、重复等问题，全面提升模型通用性与可靠性。
    

## 模型合版

相比GLM-4.5采用多个专家采样的数据进行大SFT蒸馏，GLM-5采用On-Policy Cross-Stage Distillation， 解决GLM-4.5多阶段训练的能力遗忘问题。具体的做法是：

1. 以 SFT、三个RL阶段的最终checkpoint为教师模型，从对应训练集中采样query，按比例混合后训练；
    
2. 通过替换 GRPO 算法中的优势项，基于教师模型推理分布与学生模型训练分布的差距计算损失，无需依赖样本优势估计；
    
3. 配置小组大小为 1、批量大小为1024，提升数据吞吐量，快速恢复并融合各阶段核心能力，确保模型在迭代中既强化专项性能，又不丢失已有优势。
    

GLM旗舰模型在去年展示出了极快的迭代速度，保持2-3个月更新一版：

- 2025.07.29：GLM-4.5
    
- 2025.09.30：GLM-4.6
    
- 2025.12.23：GLM-4.7
    
- 2026.02.12：GLM-5
    



## 主要收获


## 参考资料

[mp.weixin.qq.com/mp/wappoc\_appmsgcaptcha?poc\_token=HBGIommjlPcbZ1PtIAW9VfuhTtDZu2HkrlgQZVL1&target\_url=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2Fp3KETk4XTMXancZypGfaiQ](https://mp.weixin.qq.com/s/p3KETk4XTMXancZypGfaiQ)

