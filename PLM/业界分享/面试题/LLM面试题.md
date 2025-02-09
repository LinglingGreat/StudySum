
**提问**：DPO的第0步loss是固定的么？如果固定，是多少？

**回答**：loss=−logsigmoid(β−β) 。这个数应该=0.693。



**提问**：如果预训练[阶段模型](https://zhida.zhihu.com/search?content_id=240476043&content_type=Article&match_order=1&q=%E9%98%B6%E6%AE%B5%E6%A8%A1%E5%9E%8B&zhida_source=entity)训练句子的时候没有加BOS Token，但是预测的时候增加了BOS Token，或者训练的时候加了BOS token, 但是预测的时候没有加BOS Token, benchmark预测会有啥问题？

**回答**：benchmark会直接崩溃，之前gemma-2b是训的时候带BOS，预测忘加了，benchmark全崩了。那聚焦背后的真正原因是，一个句子的第一个Token在模型中会吸收大量attention，那么当你预测的时候改变了第一个Token，句子的预测会改变比较大，因为第一个Token改变了，而预测中大量attention来自第一个Token，所以预测的时候大量benchmark效果会不好。


**提问**：DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？

**回答：** DPO是一个off-policy的算法，因为训练DPO的pair数据不一定来自ref policy或者sft policy。优势是不需要对模型进行采样，然后标注，直接可以拿已有的数据集进行训练，这样的情况下包括采样的成本和标注的成本都可以节约。劣势是效果很难保证，尤其是你的模型本身能力和发布的pair数据不匹配的时候。相比而言，PPO是一个on-policy的算法，整体效果会比DPO要好。


**提问**：DPO公式是由PPO的objective公式推导过来的，为什么DPO是[off-policy算法](https://zhida.zhihu.com/search?content_id=240562968&content_type=Article&match_order=1&q=off-policy%E7%AE%97%E6%B3%95&zhida_source=entity)，而PPO是on-policy算法，到底哪一步推导出了问题？

**回答**：[Site Unreachable](https://zhuanlan.zhihu.com/p/685948009)



**提问**：DPO为什么会在学习过程中training positive的概率和training negative的概率都同时下降？

**回答**：[Site Unreachable](https://zhuanlan.zhihu.com/p/686122806)


**提问**：在PPO过程中，[reward model](https://zhida.zhihu.com/search?content_id=240631896&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)的效果上会有什么问题？

**回答**：在模型PPO过程中，reward model的准确率逐渐下降，这就是俗称的reward model的OOD问题，因为reward model的训练样本一般来自[sft模型](https://zhida.zhihu.com/search?content_id=240631896&content_type=Article&match_order=1&q=sft%E6%A8%A1%E5%9E%8B&zhida_source=entity)的responses，那么在PPO过程中，policy model刚开始和sft生成的response很相似，所以reward model准确率较高，但是在逐渐偏离sft的时候，reward model的准确率会持续下降，这基本就是现阶段reward model的主要问题。我个人认为AGI过程中，一定需要一个generalize 很强的reward model，就是所谓的global reward model or world model. 这里我也参考了一篇OpenAI blog的愿景：

[Our approach to alignment research​openai.com/blog/our-approach-to-alignment-research](https://link.zhihu.com/?target=https%3A//openai.com/blog/our-approach-to-alignment-research)

1. Training AI systems using human feedback
2. **Training AI systems to assist human evaluation**
3. Training AI systems to do alignment research

其中2就是OpenAI对reward model的一个设想。


**提问**：SFT [packing](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=1&q=packing&zhida_source=entity)对SFT训练的影响是什么？

**回答：** SFT packing以后其实是削弱了模型对难的短query和短答案的拟合。在无sft packing得情况下，假设[batch_size](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=1&q=batch_size&zhida_source=entity) = 1，那么如果有个短query和短答案在这个batch里，其余补充padding，那么这个batch的gradient全是这个短文本的gradient，模型对这个query的拟合能力会变强。但是如果SFT packing以后，多个短文本在一个样本中，这个batch的gradient会被稀释，短文本的拟合就不会特别强。但拟合能力似乎和泛化不可以挂钩，我们初步观察[sft packing](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=2&q=sft+packing&zhida_source=entity)和non sft packing的效果差不了很多。

补充, 今天review了一下之前实验，在数据量小，或者特定困难的数据上，sft packing是有损泛化效果的，但在大批量数据上是无损泛化效果的。除此之外经群友补充，non-packing的方式会影响模型续写的效果，因此会影响一些benchmark效果。


**提问**：如何解决[reward model](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)的OOD的问题？

**回答：** 现阶段解决reward model的OOD普遍解决方法，就是Llama2 [1]的做法，也就是在训练过一段时间RLHF以后，重新对policy采样pair对，人标数据然后继续训练reward model。但这种方式就是太费人力，感觉并不是持久之道。

除此之外也有一些paper试图解决这个问题：

- 比如Secrets of RLHF in Large Language Models Part II: Reward Modeling [2]中，通过[meta learning](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=meta+learning&zhida_source=entity)的方式解决这个问题，整体思想就是由于policy model在reward model训练情况下会向reward 高的方向更新，所以reward model应该对reward高的response pair更有区分度，所以设置gradient更新逐渐倾向于对reward高分training response pair倾斜。这种方法比较make sense，但实际中，由于缺少对模型on policy的采样，效果不太好。
- West-of-N: Synthetic Preference Generation for Improved Reward Modeling [3] 这篇文章跟Llama2的方式相似，区别就是不再用人进行标记，而是通过reward model本身对新的模型on policy pair进行打分，取一个query的response set中最高的分数和最低的分数数据组pair，加入到reward model的训练中。个人感觉这种方式的采样，虽然通过on policy采样加强rm的[泛化能力](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)，但实际上上限受原先rm model的能力影响。

个人觉得如何做出泛化能力比较强的rm会是一个比较难，也是比较限制模型发展的问题。


**提问**：[RLHF](https://zhida.zhihu.com/search?content_id=240791813&content_type=Article&match_order=1&q=RLHF&zhida_source=entity)中PPO有什么问题，为什么大家都设计很多方法去替代它。

**回答：**

1. Notable Complexity: 由于PPO中需要4个模型同时加载在GPU中，policy model，ref policy model，value model，reward model。所以会占用很多GPU机器。
2. Online learning problem, 除此之外，由于模型是online 采样，在policy过batch samples的时候，reward model会空置，在reward model给pair打分的时候，policy model也会空置，那么GPU利用率会不高。
3. PPO的调超参数会比较困难，需要一些炼丹高手和经验去做。


**提问**：SFT[阶段模型](https://zhida.zhihu.com/search?content_id=240873941&content_type=Article&match_order=1&q=%E9%98%B6%E6%AE%B5%E6%A8%A1%E5%9E%8B&zhida_source=entity)可以学习新知识么？

**回答：**虽然理论上可以，但很少，且不推荐sft阶段去学习知识。在LIMA原文中就表述过同样一个假设：

> **Superficial alignment hypothesis**  
> A model’s knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this hypothesis is correct, and alignment is largely about learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples.

我个人认为这个假设是合理的，sft阶段更多是将模型的能力和人类对齐，而不过多学习新的知识。原因如下：

- sft相对于pretrain过的数据量实在太小，模型的知识学习的概率就很低。
- 如果加大sft的数据量和pretrain数据相当，那么sft有一些特定的格式，以及一些system prompt需要重复当作context进行attention，这些重复的context势必会影响模型原始的attention模式，从而影响模型的效果。

最后，如果希望sft学习新知识，不如把这部分sft的新知识组织好放入pre-train or post-train阶段更为合适。


**提问**：建立sft数据主要需要关注什么方面？

**回答：**

在Lima中给予了两个重要的点：

1. Prompt的[diversity](https://zhida.zhihu.com/search?content_id=240902032&content_type=Article&match_order=1&q=diversity&zhida_source=entity)：丰富多样的prompt数据，可以让模型更多的了解人类的指令，包括指令的意思，复杂指令中每一步的含义。Prompt的丰富程度决定了模型指令遵循的能力。
2. Answer的质量：Answer的质量包括内容和格式两方面，一方面内容的正确性需要得到保证，一方面内容的格式也很重要，细节丰富，逻辑缜密的answer可以激发模型更多的回答能力。

补充：

1. SFT阶段不能太多的知识注入：过多的知识注入，或者超过模型能力本身的回答过多会导致[对齐税](https://zhida.zhihu.com/search?content_id=240902032&content_type=Article&match_order=1&q=%E5%AF%B9%E9%BD%90%E7%A8%8E&zhida_source=entity)，这是OpenAI的blog也曾经提到的，这就是我为什么不建议模型过多在SFT阶段学习知识，会影响其学习指令遵循的能力。


**提问：**提升sft的prompt的多样性有什么好的方法？

**回答：**

- 明文TAG法：也就是对SFT的prompt进行打tag，对其中的名词和动词进行分类打标，最后通过tag对prompt的分布进行调整，保证tag的分布是均匀的。著名的就是InsTag [1] 这个方法。

![](https://pic1.zhimg.com/v2-6866bd7d362e6597ac0779198d725b90_1440w.jpg)

- 模型embedding聚类方法：通过模型最后一层的embedding对prompt进行表示，那么通过prompt embedding的距离表示prompt的相似度，对于过于相似的prompt进行删除。著名的有Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning [2]。

![](https://pic3.zhimg.com/v2-3680b3c9e49fcfb4ea2cc6fdfe0939e0_1440w.jpg)

Figure: Blue points: training data point ; Red points: novel data points to be seleted.

- 从complexity角度，对于prompt直接进行难度的升级，所以即使在同一个语意空间的prompt也会变得diverse。比较著名的是Wizard 方法 [3]，通过GPT4进行prompt难度升级，然后构成complexity丰富的prompt。

![](https://pic3.zhimg.com/v2-5035db54296d19455f86a80951477c18_1440w.jpg)


