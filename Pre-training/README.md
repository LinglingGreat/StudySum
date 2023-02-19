## 大模型

**scaling law**

其它条件保持不变且合适的时候，模型的效果（纵坐标）会随着参数量（横坐标）的增大线性增长（Log-linear）。
- 横坐标除了模型参数量，还可以是预训练的token数，微调的token数，输入框窗口的大小，instruction指令的种类，外部知识库）

- 纵坐标还可以是in-context pref., zero-shot perf., fine-tuning perf., in-dist. perf., OOD perf.


**Emergent abilities** 涌现能力

有些能力是当你的模型足够大的时候才会出现。

比如模型举一反三、跨域迁移的能力。



**LLM Model Families**

模型演化的历程。

![](img/Pasted%20image%2020230218181401.png)

![](img/Pasted%20image%2020230218181512.png)

预训练阶段希望得到一个很强的基础模型。

instruction tuning能够让模型的效果提升，不论模型大小如何。把scaling的曲线往上提（提升斜率）

alignment是牺牲能力换安全。


强的预训练模型
- Code-davince-002
- PaLM
- Chinchilla
- Gopher
- Calactica(没有经过alignment，值得试试)
- GLM-130B
- BLOOM
- OPT

能力
- Language generation
- World knowledge
- In-context learning
- Code understanding/generation
- Complex reasoning/chain-of-thought

代码数据可以跟其它数据放在一起，进行预训练。openai是用代码数据在gpt3上做continual training。代码数据的加入可以增强模型的推理能力——是一个假设，不是结论。（面向对象编程和面向过程编程）

Instruction Tuning可以解锁模型的一些能力。（数据量很少，所以是解锁），重点是增加指令的种类的数量，而不是某个指令的数据量。指令种类的数量和模型的效果符合scaling law。指令种类指数增长，导致模型零样本迁移能力的线性增长。思维链等能力有时会在预训练后直接出现（例如PaLM）。如果这种能力没有出现，我们可以将其作为特殊的指令，进行指令微调。经过预训练后，大模型相较于小模型具有更大的能力边界。如果预训练模型具备某项能力，指令微调可以继续提升该能力；若预训练模型不具备某项能力，指令微调有望开发出该能力。因此，指令微调后的小模型也有可能获取强于大模型的能力。同时，指令微调的效果也和基础模型息息相关。

Leaderboard-MMLU
- Text-davinci-002/003
- Flan-PaLM
- OPT-IML(取决于基础模型的效果，175B的OPT-IML还不如11B的Flan-T5)
- LM self-instruct(它模拟了未经过指令微调的初代 GPT 175B 到经过指令微调的 Text-davinci 001 之间的演化。指令微调的种类要多，单个指令的数据量不需要很多，但是如果数量增多的话可以提高这单个指令的效果，同时又有平衡性的问题，影响其它指令的效果。)

能力
- Follow instructions
- Zero-shot generation, no in-context
- Generalize to unseen instructions/tasks
- Compositional generalization（指令和指令的组合，比如把问答和摘要和生成代码能力合在一起，提一个问题，让模型对一段代码做摘要。我们可以将指令视为线性代数中的一组基，将不同能力混合在一起实际上就是对线性空间中的基做线性组合或凸组合。模型在没有见过指令时，只能在学到的空间内做内插，而很难外推到没有学习到的基上。）
- Complex reasoning/Chain-of-thought

Alignment：塑造模型的价值观。拿能力换安全。openai叫它对齐税，安全度是可以调的

Models
- OpenAI-ChatGPT
- DeepMind-Sparrow
- Anthropic-Claude（更加安全，保守的模型）

Training can either be supervised/RLHF
- RLHF也需要先做supervised，不然怎么都选不出好的

Abilities
- Informative and userful responses
- Impartial responses
- Reject improper queries
- Reject unknown knowledge

Pretraining+instruction tuning: to bulid powerful model

Alignment: to shape model's personality

Q: 如果我的计算资源不够，我该怎么办？——lower bound

Q：如果拿到了ChatGPT的基础模型，你希望塑造成什么样子——upper bound

符尧等人的尝试：Model specialization：specializing Smaller Language Models towards Multi-Step Reasoning
- 大模型的能力不是在所有方面都拉满，比如数学只能高中数学。
- 大模型的技能多，增加一个方向的能力，其它方向的能力不会掉很多；而小模型想要增加一个方向的能力，其它方向的能力就会掉很多。专业化的小模型可能比没有专业化的中等模型的能力强

在什么阶段做专业化？实验表明，对指令微调后的模型进行专门化处理的效果要远远优于对原始预训练模型进行专门化处理的效果。

针对数学问题，我们考虑测试模型的思维链推理能力。经过专门化后，模型在 BigBench-Hard 这种通用测试上的能力有所下降，而在数学问题上的思维链推理能力有所增强。通用能力下降的程度预模型大小相关，模型越大，通用能力下降得越少。为了测试模型的分布外泛化能力，我们使用 GSM8K 作为训练数据集，使用 MultiArith、ASDiv、SVAMP 作为测试集。


## Prompt
- [ ] todo


## Instruction tuning

- [ ] todo



## CoT

[CoT](CoT/CoT.md)



[Why did all of the public reproduction of GPT-3 fail? In which tasks should we use GPT-3.5/ChatGPT?](https://jingfengyang.github.io/gpt)






