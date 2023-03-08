## 大模型

### Motivation

**scaling law**

其它条件保持不变且合适（比较好的数据上训练，学习率调整好）的时候，模型的效果（纵坐标）会随着参数量（横坐标）的增大线性增长（Log-linear）。
- 最初openai研究出了scaling law，预测出如果有一个很大的模型，一定会带来效果的巨大提升，所以才做出了gpt3。而不是为了大而大
- 横坐标除了模型参数量，还可以是预训练的token数，微调的token数，输入框窗口的大小，instruction指令的种类——泛化能力up，外部知识库outside memory的大小）

- 纵坐标还可以是in-context pref., zero-shot perf., fine-tuning perf., in-dist. perf., OOD perf.（in-distribution, out of distribution）

模型能力会被锁死在模型的大小上，比如MOSS的模型是16B，一些能力只能这样了


**Emergent abilities** 涌现能力

有些能力是当你的模型足够大的时候才会出现。

比如模型举一反三、跨域迁移的能力。明显大模型比小模型好得多。cot的能力目前被证明小模型大模型都有。


所以一定要做大模型

小模型的研究分为2类
- 修改模型结构，可能被模型的大小冲掉。模型的大小会补偿模型结构的改动
- instruction finetune，大小模型上做都能提升效果，会把两条曲线的纵截距往上提。

instruction tuning能够让模型的效果提升，不论模型大小如何。把scaling的曲线往上提（提升斜率）

数据的干净程度影响的是两条曲线的斜率

有些模型的能力是可预测的，随着模型增大就会有（第一条曲线）

有些能力是不可预测的，到某个数量级之后突然出现（第二条曲线）



### LLM Model Families

模型演化的历程。从模型演化的视角分析，而不是单个模型

![](img/Pasted%20image%2020230218181401.png)

都分为3个阶段

![](img/Pasted%20image%2020230218181512.png)

#### 预训练
预训练阶段希望得到一个很强的基础模型。有些能力在这个阶段就能观测到。

强的预训练模型
- Code-davince-002
- PaLM
- Chinchilla
- Gopher
- Calactica(没有经过alignment，值得试试)
- GLM-130B
- BLOOM
- OPT

Leaderboard-MMLU数据集有很多维度，可以测试模型效果，主要测知识。

BBchat, BBH数据集

斯坦福——模型评估更快的方法，最近


能力
- Language generation
- World knowledge（模型参数越大储存的越多）
- In-context learning
- Code understanding/generation
- Complex reasoning/chain-of-thought

代码数据可以跟其它数据放在一起，进行预训练（谷歌的做法）。openai是用代码数据在gpt3上做continual training。代码数据的加入可以增强模型的推理能力+cot能力——是一个假设，不是结论。（面向对象编程C++：很大问题拆分成一个个小问题；面向过程编程的C：一步一步解决）（代码注释——高质量的cot）


#### Instruction Tuning

目标：unlock model abilities

Instruction Tuning（任务描述+任务例子）可以解锁模型的一些能力。（数据量很少，所以是解锁,万量级即可，大模型需要的数据比小模型少），重点是增加指令的种类的数量，而不是某个指令的数据量。指令种类的数量和模型的效果符合scaling law。指令种类指数增长，导致模型零样本迁移能力的线性增长。思维链等能力有时会在预训练后直接出现（例如PaLM）。如果这种能力没有出现，我们可以将其作为特殊的指令，进行指令微调。经过预训练后，大模型相较于小模型具有更大的能力边界。如果预训练模型具备某项能力，指令微调可以继续提升该能力；若预训练模型不具备某项能力，指令微调有望开发出该能力。因此，指令微调后的小模型也有可能获取强于大模型的能力。同时，指令微调的效果也和基础模型息息相关。
- 能够让小六边形——大六边形

Leaderboard-MMLU
- Text-davinci-002/003
- Flan-PaLM
- OPT-IML(取决于基础模型的效果，175B的OPT-IML还不如11B的Flan-T5)
- LM self-instruct(它模拟了未经过指令微调的初代 GPT 175B 到经过指令微调的 Text-davinci 001 之间的演化。指令微调的种类要多，单个指令的数据量不需要很多，但是如果数量增多的话可以提高这单个指令的效果，同时又有平衡性的问题，影响其它指令的效果。)
- MOSS

能力
- Follow instructions
- Zero-shot generation, no in-context
- Generalize to unseen instructions/tasks
- Compositional generalization（指令和指令的组合，比如把问答和摘要和生成代码能力合在一起，提一个问题，让模型对一段代码做摘要。我们可以将指令视为线性代数中的一组基，将不同能力混合在一起实际上就是对线性空间中的基做线性组合或凸组合。模型在没有见过指令时，只能在学到的空间内做内插，而很难外推到没有学习到的基上。）
- Complex reasoning/Chain-of-thought

#### Alignment
目标：align with human value system

Alignment：塑造模型的价值观。拿能力换安全。openai叫它对齐税，符合人类期望，那么就要牺牲一部分能力。安全度是可以调的。alignment是牺牲能力换安全。
- 六边形的某个维度往外扩，其它能力往内缩
- 可以融入instruction tuning中，instruction对齐人类期望
- 也可以融入预训练（Anthropic最近的文章），塑造模型信念，一旦塑造了后面改动成本可能比较高甚至改不了。越早发生越好。

人做的事情
- 标数据（openai做到了极致，找藤校的硕士生标，据说是50美元/条，我们是3元/条）
- 人类选择偏好的数据（模型足够好的时候）
- RL是让模型去选答案

Models
- OpenAI-ChatGPT（希望更加有用，尽可能提供有用信息，拒绝的没有那么彻底）
- DeepMind-Sparrow
- Anthropic-Claude（更加安全，保守的模型，拒绝的更加彻底）

Training can either be supervised/RLHF
- RLHF也需要先做supervised，不然怎么都选不出好的

Abilities
- Informative and userful responses
- Impartial responses
- Reject improper queries
- Reject unknown knowledge

Pretraining+instruction tuning: to bulid powerful model

Alignment: to shape model's personality

specialization：把模型能力重新分布到某个目标领域


Q: 如果我的计算资源不够，我该怎么办？——lower bound

Q：如果拿到了ChatGPT的基础模型，你希望塑造成什么样子——upper bound

符尧等人的尝试：Model specialization：specializing Smaller Language Models towards Multi-Step Reasoning
- 大模型的能力不是在所有方面都拉满，比如数学只能高中数学。
- chatgpt增强了对话的能力，减弱了其它方向的能力
- 大模型的技能多，增加一个方向的能力，其它方向的能力不会掉很多；而小模型想要增加一个方向的能力，其它方向的能力就会掉很多。专业化的小模型可能比没有专业化的中等模型的能力强
- 例如copilot

在什么阶段做专业化？实验表明，对指令微调后的模型进行专门化处理的效果要远远优于对原始预训练模型进行专门化处理的效果。

针对数学问题，我们考虑测试模型的思维链推理能力。经过专门化后，模型在 BigBench-Hard 这种通用测试上的能力有所下降，而在数学问题上的思维链推理能力有所增强。通用能力下降的程度预模型大小相关，模型越大，通用能力下降得越少。为了测试模型的分布外泛化能力，我们使用 GSM8K 作为训练数据集，使用 MultiArith、ASDiv、SVAMP 作为测试集。


semi-自回归，非自回归，可能会被大模型的scale弥补效果，可以试试

模型评估：推理+结论（结论对了就行），给选项做多选，比较哪个生成的好，人工评估

sample efficient：如果算法可以从每个样本中获得最大收益，那么该算法就是样本高效的。

## 现有大模型

![](img/Pasted%20image%2020230227103950.png)

[LLaMA](LLaMA/LLaMA.md)



## Prompt
- [ ] todo


## Instruction tuning

[Flan-PaLM_T5](Flan-PaLM_T5/Flan-PaLM_T5.md)


## CoT

总结：[CoT](CoT/CoT.md)



[Why did all of the public reproduction of GPT-3 fail? In which tasks should we use GPT-3.5/ChatGPT?](https://jingfengyang.github.io/gpt)

## Benchmark

| 评测系统                            | 语言   | 任务范围       | 能力/任务/数据集 |
| ----------------------------------- | ------ | -------------- | ---------------- |
| [GLUE](GLUE/GLUE.md)                | 英语   | 理解           | 1/7/9            |
| [SuperGLUE](SuperGLUE/SuperGLUE.md) | 英语   | 理解           | 2/4/8            |
| [CLUE](CLUE/CLUE.md)                | 中文   | 理解           | 2/6/9            |
| [CUGE](CUGE/CUGE.md)                | 中文   | 理解+生成      | 7/17/19          |
| [MMLU](MMLU/MMLU.md)                | 英文   | 理解           |                  |
| [BIGBench](BigBench/BIGBench.md)    | 多语言 | 理解+生成      | x/200+/x         |
| [XTREME](XTREME/XTREME.md)          | 多语言 | 跨语言迁移能力 | x/9/x            |
| [HELM](HELM/HELM.md)                | 英文   | 理解+生成      |                  |
| [LAMBADA](LAMBADA/LAMBADA.md)                                    |  英文      |     上下文理解生成           |                  |


HELM包括了基本上每个评测方向的数据集，可以在此基础上评测，补充其他评测任务。
- 问答、信息抽取、摘要、情感分析、有毒性检测、文本分类、aspirational场景（文本生成、故事生成等）、语言、知识、推理、危害、效率、校准、鲁棒性

[RACE数据集](https://www.cs.cmu.edu/~glai1/data/race/)来自中国12-18岁之间的初中和高中英语考试阅读理解，包含28,000个短文、接近100,000个问题。包含用于评估学生理解能力的多种多样的主题。

[TyDiQA](https://github.com/google-research-datasets/tydiqa)（问答，跨越多个语种），包括以下任务
- **Passage selection task (SelectP)**：选择答案所在段落
- **Minimal answer span task (MinSpan)**：选择答案的span或者回答yes/no
- **Gold passage task (GoldP)**：给定包含答案的段落，预测答案的连续span

[MGSM](https://github.com/google-research/url-nlp)（多语言的数学应用问题，GSM8K数据集手动翻译成了10种语言）
- 可用来做CoT推理评测

CoT效果的验证，也可以参考[Flan-PaLM_T5](Flan-PaLM_T5/Flan-PaLM_T5.md)执行MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT的对比

MMLU（包含在HELM中）
- 偏向语言理解的知识密集型任务（比如计算机、数学、法律、哲学、医学、经济学、社会学）
- 提供选项，选择答案，衡量指标是准确率
- 可用来做zero-shot和few-shot

BBH
- BIG-Bench的一部分

[XCOPA](https://github.com/cambridgeltl/xcopa)
- 跨语言常识推理
- 任务目标是根据一个问题确定前提和两个选项之间的因果关系。因此，一个成功的模型不仅要执行常识推理，还要将其推理能力推广到新语言。

[XL-WiC](https://pilehvar.github.io/xlwic/)
- 多语言单词上下文语义判断
- 给定同一种语言的两个句子和一个出现在两个句子中的感兴趣的词，模型被询问这个词在句子中是否具有相同的意义。

NaturalInstructions



## dataset

- [ ] SuperNaturalInstructions


## 要点

预训练数据
- 通过数据去重避免记忆和过拟合
- 通过数据筛选以得到高质量数据
- 保证数据多样性以确保 LLM 的泛化性

训练策略
- 训练框架，bf16可以表示更大范围的浮点数，能够处理在损失尖峰时出现的大数值。PaLM用了它。
- 训练过程中的修改，PaLM 几乎没有做任何中途调整。它只是当损失尖峰出现时，从尖峰开始前大约 100 步的 checkpoint 重新开始训练，并跳过了大约 200-500 个 batch 的数据。仅仅依靠这种简单的重启，PaLM 就取得神奇的成功。这是由于它在预训练数据构建期间就已经完成采样，因此模型具有在 Bit 意义上的确定性，以及它对模型架构和训练设置进行了许多修改以获得更好的稳定性。
- 模型架构/训练设置，为了使训练更稳定，PaLM 对模型架构和训练设置进行了多项调整，包括使用 Adafactor 的修改版本作为优化器，缩放在 softmax 之前的输出 logit，使用辅助损失来鼓励 softmax 归一化器接近 0，对词向量和其他层权重使用不同的初始化，在前馈层和层归一化中不使用偏差项，并且在预训练期间不使用 dropout。
- 训练过程。原始的 GPT-3 预训练过程见过的 token 数与 OPT 和 BLOOM 接近，而 PaLM 则远远超过了它们。同样，PaLM 和 GPT-3 预训练语料库都大于 BLOOM 和 OPT。

![](img/Pasted%20image%2020230228100643.png)

- PaLM 和 GPT-3 都使用了在训练过程中从小到大逐渐增加的 batch size，这已经被展示对于训练一个更好的 LLM 是有效的，然而 OPT 和 BLOOM 都使用了恒定的 batch size。

- OPT 使用了 ReLU 激活函数，而 PaLM 使用 SwiGLU 激活函数，GPT-3 和 BLOOM 使用 GeLU，它通常使得训练的 LLM 的性能更好。

- 为了更好的建模更长的序列，PaLM 使用 RoPE 词向量，BLOOM 使用 ALiBi 词向量，而原始的 GPT-3 和 OPT 使用学习得到的词向量，这可能影响在长序列上的性能。

参考：[万字长文解析！复现和使用GPT-3/ChatGPT，你所应该知道的](https://mp.weixin.qq.com/s/ILpbRRNP10Ef1z3lb2CqmA)，推特原文： https://twitter.com/JingfengY/status/1625003999387881472

