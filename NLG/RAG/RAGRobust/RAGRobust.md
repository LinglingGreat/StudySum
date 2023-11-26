
## 何为检索增强

纯参数化的模型存在诸多不足： 

**1. 长尾记忆困难：**不能记住所有训练语料中的所有知识，尤其是对低频的长尾知识记忆困难； 

**2. 容易过时：**参数中的知识容易过时（ChatGPT 和 LLaMa肯定不知道周二国足的比分，硬预测的话应该会预测个比三比零更大的数 x），更新起来很困难（训练代价且容易造成灾难性遗忘）； 

**3. 参数太多导致计算代价大：**训练和推理代价高昂（虽然有 Scaling Law，但参数量上去之后就没什么人训练甚至部署得起了→_→）。

语言模型可以是半参数化的，也就是（参数化的）模型可以外挂一个（非参数化的）语料数据库，推理时以从语料库召回的部分数据为参考组织答案（具体的形式可以是作为额外的上文输入，也可以插在中间的 cross attention 或者输出中），这一范式被称为**检索增强生成（Retrieval-Augmented Generation，RAG）**。

**检索增强的语言模型（Retrieval-Augmented Language Model，RALM）的正式定义是：** 

A language model (LM) that uses an external datastore at test time.

RAG除了缓解以上三个问题（长尾记忆困难、容易过时、参数太多导致计算代价大）之外，还可以起到给模型的回答提供可靠的消息来源、防止模型权重泄露隐私信息等作用，具体的机制和代表性工作可以参见今年 ACL 上陈丹琦老师领衔的 Tutorial [1]

## 检索增强是否可靠

如果检索增强的时候召回的是和输入问题**无关的内容（噪声干扰）**，甚至是**反事实**的 fake news 或者被篡改的百科，模型就会像吃了毒蘑菇一样胡言乱语。

以下是来自论文 [2] 的一个检索回无关内容后输出被影响的例子，原本对“德牧能不能进机场”这样的简单的问题，ChatGPT是高度认可小狗同志作为导盲犬的价值的，果断说 yes，但是检索模块召回了一段“老德牧是一类 balabala 某种狗的争议性名称”的百科介绍作为额外上文输入后，模型突然对小狗变凶了，说出了“机场不许你入内”这样的负心话。

![](img/Pasted%20image%2020231126155150.png)

以下是来自论文 [3] 的检索到反事实信息造成模型错误输出的例子。对博闻强识的大模型来说，17-18 赛季的欧冠冠军是道简单题，不用检索增强就知道是皇马，但如果有恶意用户某一天编辑了相关的维基百科把答案改成巴萨，模型通过检索模块吃到这样与自身知识冲突的辅助输入就会被忽悠住，人云亦云，复读出错误的答案。

![](img/Pasted%20image%2020231126155212.png)

## 如何提高检索增强的可靠性

[SKR](../SKR/SKR.md)

[RECALL](../RECALL/RECALL.md)

[TrainRobustRALMs](../TrainRobustRALMs/TrainRobustRALMs.md)

[ChainofNote](../ChainofNote/ChainofNote.md)

[SelfRAG](../SelfRAG/SelfRAG.md)
## 总结

整体来看，方法可以分为两类： 

**1. 自适应检索和过滤：**即在检索前加一个模块判断该问题是不是需要检索增强才能回答或判断检索回的内容是否有用，以避免不必要的检索召回内容被输入模型产生干扰，如 SKR用模型自身的信号在训练数据上额外构建一个分类器，TrainRobustRALMs直接使用 NLI 模型，Self-RAG从 GPT-4 蒸馏能力，让语言模型自己以预测 Retrieve token 的形式判断。

实验已证实这类方法能有效地避免无用的召回内容的干扰，坏处是直接删除被判断为无用的内容，可能误伤有用的检索召回内容。 

**2. 生成时干预：**希望即使无用甚至错误的内容被检索回来、输入模型，模型对这样的增强输入依然能凭借自身知识保持鲁棒，如 RECALL 的 prompt engineering 或者 Dola 干预，TrainRobustRALMs的直接构造相应的训练数据进行训练，Chain of Note 的思维链蒸馏，Self-RAG  的让模型自身判断召回的内容是否有用。其中只有 RECALL 是不需要训练的，但未取得明显收益，另外三类都需要依赖 ChatGPT 或 GPT4 这些强大的闭源模型构造训练信号。 

最后，笔者想讨论的一点零碎思考是，以上的各工作基本假定检索模型是固定的（Google API 或者冻结的预训练召回模型），如果把检索模型和 index 的更新也考虑进来，是否能进一步提升整个 RAG 系统的鲁棒性？



## 参考资料

[开卷翻到毒蘑菇？浅谈大模型检索增强（RAG）的鲁棒性](https://mp.weixin.qq.com/s/YsP5gYxCYobnf8x8nTMxug)

## 参考文献

[1] Asai, Akari, et al. "Retrieval-based language models and applications." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts). 2023.   






