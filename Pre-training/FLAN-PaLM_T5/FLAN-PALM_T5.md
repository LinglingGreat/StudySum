---
title: FLAN-PALM_T5
created: 2023-02-19
tags: instruction-tuning 
type: 论文
papername: Scaling Instruction-Finetuned Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: 谷歌
---

## 论文基本信息

标题：Scaling Instruction-Finetuned Language Models

作者：

链接： 

代码： https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints

框架图：


在吸收FLAN的精华的基础上，加入了CoT的数据来做finetune(9个CoT数据集，在所有evaluations上都有更好的表现)。还有T0, Natural Instructions, 对话，程序合成数据

CoT 混合中共有 74,730 个示例。以下是每个数据集的训练示例数量：AQuA (2,715)、CREAK (6,910)、ECQA (7,110)、ESNLI (36,170)、GSM8K (7,470)、QASC (1,080)、QED (5,145)、Sensemaking (6,070)、 StrategyQA (2,060)。

![](img/Pasted%20image%2020230219165016.png)

![](img/Pasted%20image%2020230301102204.png)

![](img/Pasted%20image%2020230301103427.png)

### 训练
- 学习率、批量大小和 dropout 是最重要的超参数
- 固定学习率，adafactor优化器
- 多个训练示例组合成一个，使用结束标记将输入和目标分开。过程中用了mask，防止示例间的跨越（packing）
- 每个任务的采样比例，根据任务的样本数作为权重，但是有上限

![](img/Pasted%20image%2020230306202207.png)

![](img/Pasted%20image%2020230306202310.png)

### 实验结果

评估：MMLU，BBH（BIG-Bench中的23个任务），TyDiQA（问答，跨越8个语种），MGSM（多语言的数学应用问题，手动翻译成了10种语言）。
- MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT.
- five-shot for MMLU, three-shot for BBH, one-shot for TyDiQA, and 8-shot for MGSM.

这么finetune过后的模型，其实不论在CoT任务和非CoT任务上其实都表现得最好，而且在BBH上做zeroshot优势更是巨大。这也进一步证明了CoT是可以和当前流行的instruction tuning无缝衔接的。

![](img/Pasted%20image%2020230306200538.png)

![](img/Pasted%20image%2020230219170727.png)

![](img/Pasted%20image%2020230219170809.png)

![](img/Pasted%20image%2020230306200720.png)


主要结论
- 比起没有微调的模型，多任务指令微调可以大幅提升性能，从9.4%到15.5%。
- 增加微调任务的数量可以提升性能，虽然任务达到282个后提升就很小了。原因可能是：1.更多的任务diverse不够，没有提供新的知识；2.多任务指令微调的大部分收益来自于模型学会了更好地表达预训练中学到的知识，超过282个任务没有多大帮助。预训练数据包括了780B token，指令微调只有1.4B token
- 将模型比例增加一个数量级（即 8B → 62B 或 62B → 540B）显着提高微调和非微调模型的性能。确定指令微调是改进小型模型更多还是改进大型模型更多可能会很复杂。尽管 8B 模型的绝对增益大于 540B 模型（8B 为 15.5%，540B 为 9.4%），但 540B 模型的错误率相对降低幅度更大（540B 为 18.4%，8B 为 16.6%） ).
- CoT可以跟self-consistency（SC）结合，在一些任务上可以得到sota的效果。
- 结合非 CoT 和 CoT 微调，在held-out的 CoT 基准上的性能比单独的 CoT 微调更强。这种联合微调可以显着提高 CoT 性能，同时保持非 CoT 任务的性能，从而允许单个模型在所有评估中表现出色。只在non-CoT数据上微调会降低CoT任务上的效果。可能的解释是，当未见过的任务与微调任务处于相同的提示范式中时，指令微调会改进未见过的任务。因此，非 CoT 和 CoT数据都是需要的。
- 在有和没有范例的情况下对 CoT 数据进行指令微调的最后一个好处是，生成的模型能够在零样本设置中执行 CoT 推理
- 指令微调与其他模型自适应技术（如 UL2R）结合得很好。指令微调和 UL2 持续预训练是互补的计算高效方法，可以在不增加模型规模的情况下提高语言模型的性能。模型Flan-U-PaLM
- 使用指令微调的较小模型有时可以胜过没有指令微调的较大模型
- 当任务需要多个推理步骤并且与足够大的语言模型一起使用时，CoT 相对直接回答的提升最大

附录C是有毒性判断的实验和性别、职业偏见的实验。

有毒性判断：用的是Perspective API，其AUC 达到 97.0%
- Real Toxicity Prompts dataset


## 核心亮点

## 主要收获

