
## **提问**：DPO的第0步loss是固定的么？如果固定，是多少？

**回答**：loss=−logsigmoid(β−β) 。这个数应该=0.693。



## **提问**：如果预训练[阶段模型](https://zhida.zhihu.com/search?content_id=240476043&content_type=Article&match_order=1&q=%E9%98%B6%E6%AE%B5%E6%A8%A1%E5%9E%8B&zhida_source=entity)训练句子的时候没有加BOS Token，但是预测的时候增加了BOS Token，或者训练的时候加了BOS token, 但是预测的时候没有加BOS Token, benchmark预测会有啥问题？

**回答**：benchmark会直接崩溃，之前gemma-2b是训的时候带BOS，预测忘加了，benchmark全崩了。那聚焦背后的真正原因是，一个句子的第一个Token在模型中会吸收大量attention，那么当你预测的时候改变了第一个Token，句子的预测会改变比较大，因为第一个Token改变了，而预测中大量attention来自第一个Token，所以预测的时候大量benchmark效果会不好。


## **提问**：DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？

**回答：** DPO是一个off-policy的算法，因为训练DPO的pair数据不一定来自ref policy或者sft policy。优势是不需要对模型进行采样，然后标注，直接可以拿已有的数据集进行训练，这样的情况下包括采样的成本和标注的成本都可以节约。劣势是效果很难保证，尤其是你的模型本身能力和发布的pair数据不匹配的时候。相比而言，PPO是一个on-policy的算法，整体效果会比DPO要好。


## **提问**：DPO公式是由PPO的objective公式推导过来的，为什么DPO是[off-policy算法](https://zhida.zhihu.com/search?content_id=240562968&content_type=Article&match_order=1&q=off-policy%E7%AE%97%E6%B3%95&zhida_source=entity)，而PPO是on-policy算法，到底哪一步推导出了问题？

**回答**：[Site Unreachable](https://zhuanlan.zhihu.com/p/685948009)



## **提问**：DPO为什么会在学习过程中training positive的概率和training negative的概率都同时下降？

**回答**：[Site Unreachable](https://zhuanlan.zhihu.com/p/686122806)


## **提问**：在PPO过程中，[reward model](https://zhida.zhihu.com/search?content_id=240631896&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)的效果上会有什么问题？

**回答**：在模型PPO过程中，reward model的准确率逐渐下降，这就是俗称的reward model的OOD问题，因为reward model的训练样本一般来自[sft模型](https://zhida.zhihu.com/search?content_id=240631896&content_type=Article&match_order=1&q=sft%E6%A8%A1%E5%9E%8B&zhida_source=entity)的responses，那么在PPO过程中，policy model刚开始和sft生成的response很相似，所以reward model准确率较高，但是在逐渐偏离sft的时候，reward model的准确率会持续下降，这基本就是现阶段reward model的主要问题。我个人认为AGI过程中，一定需要一个generalize 很强的reward model，就是所谓的global reward model or world model. 这里我也参考了一篇OpenAI blog的愿景：

[Our approach to alignment research​openai.com/blog/our-approach-to-alignment-research](https://link.zhihu.com/?target=https%3A//openai.com/blog/our-approach-to-alignment-research)

1. Training AI systems using human feedback
2. **Training AI systems to assist human evaluation**
3. Training AI systems to do alignment research

其中2就是OpenAI对reward model的一个设想。


## **提问**：SFT [packing](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=1&q=packing&zhida_source=entity)对SFT训练的影响是什么？

**回答：** SFT packing以后其实是削弱了模型对难的短query和短答案的拟合。在无sft packing得情况下，假设[batch_size](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=1&q=batch_size&zhida_source=entity) = 1，那么如果有个短query和短答案在这个batch里，其余补充padding，那么这个batch的gradient全是这个短文本的gradient，模型对这个query的拟合能力会变强。但是如果SFT packing以后，多个短文本在一个样本中，这个batch的gradient会被稀释，短文本的拟合就不会特别强。但拟合能力似乎和泛化不可以挂钩，我们初步观察[sft packing](https://zhida.zhihu.com/search?content_id=240699804&content_type=Article&match_order=2&q=sft+packing&zhida_source=entity)和non sft packing的效果差不了很多。

补充, 今天review了一下之前实验，在数据量小，或者特定困难的数据上，sft packing是有损泛化效果的，但在大批量数据上是无损泛化效果的。除此之外经群友补充，non-packing的方式会影响模型续写的效果，因此会影响一些benchmark效果。


## **提问**：如何解决[reward model](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)的OOD的问题？

**回答：** 现阶段解决reward model的OOD普遍解决方法，就是Llama2 [1]的做法，也就是在训练过一段时间RLHF以后，重新对policy采样pair对，人标数据然后继续训练reward model。但这种方式就是太费人力，感觉并不是持久之道。

除此之外也有一些paper试图解决这个问题：

- 比如Secrets of RLHF in Large Language Models Part II: Reward Modeling [2]中，通过[meta learning](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=meta+learning&zhida_source=entity)的方式解决这个问题，整体思想就是由于policy model在reward model训练情况下会向reward 高的方向更新，所以reward model应该对reward高的response pair更有区分度，所以设置gradient更新逐渐倾向于对reward高分training response pair倾斜。这种方法比较make sense，但实际中，由于缺少对模型on policy的采样，效果不太好。
- West-of-N: Synthetic Preference Generation for Improved Reward Modeling [3] 这篇文章跟Llama2的方式相似，区别就是不再用人进行标记，而是通过reward model本身对新的模型on policy pair进行打分，取一个query的response set中最高的分数和最低的分数数据组pair，加入到reward model的训练中。个人感觉这种方式的采样，虽然通过on policy采样加强rm的[泛化能力](https://zhida.zhihu.com/search?content_id=240746384&content_type=Article&match_order=1&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)，但实际上上限受原先rm model的能力影响。

个人觉得如何做出泛化能力比较强的rm会是一个比较难，也是比较限制模型发展的问题。


## **提问**：[RLHF](https://zhida.zhihu.com/search?content_id=240791813&content_type=Article&match_order=1&q=RLHF&zhida_source=entity)中PPO有什么问题，为什么大家都设计很多方法去替代它。

**回答：**

1. Notable Complexity: 由于PPO中需要4个模型同时加载在GPU中，policy model，ref policy model，value model，reward model。所以会占用很多GPU机器。
2. Online learning problem, 除此之外，由于模型是online 采样，在policy过batch samples的时候，reward model会空置，在reward model给pair打分的时候，policy model也会空置，那么GPU利用率会不高。
3. PPO的调超参数会比较困难，需要一些炼丹高手和经验去做。


## **提问**：SFT[阶段模型](https://zhida.zhihu.com/search?content_id=240873941&content_type=Article&match_order=1&q=%E9%98%B6%E6%AE%B5%E6%A8%A1%E5%9E%8B&zhida_source=entity)可以学习新知识么？

**回答：**虽然理论上可以，但很少，且不推荐sft阶段去学习知识。在LIMA原文中就表述过同样一个假设：

> **Superficial alignment hypothesis**  
> A model’s knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this hypothesis is correct, and alignment is largely about learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples.

我个人认为这个假设是合理的，sft阶段更多是将模型的能力和人类对齐，而不过多学习新的知识。原因如下：

- sft相对于pretrain过的数据量实在太小，模型的知识学习的概率就很低。
- 如果加大sft的数据量和pretrain数据相当，那么sft有一些特定的格式，以及一些system prompt需要重复当作context进行attention，这些重复的context势必会影响模型原始的attention模式，从而影响模型的效果。

最后，如果希望sft学习新知识，不如把这部分sft的新知识组织好放入pre-train or post-train阶段更为合适。


## **提问**：建立sft数据主要需要关注什么方面？

**回答：**

在Lima中给予了两个重要的点：

1. Prompt的[diversity](https://zhida.zhihu.com/search?content_id=240902032&content_type=Article&match_order=1&q=diversity&zhida_source=entity)：丰富多样的prompt数据，可以让模型更多的了解人类的指令，包括指令的意思，复杂指令中每一步的含义。Prompt的丰富程度决定了模型指令遵循的能力。
2. Answer的质量：Answer的质量包括内容和格式两方面，一方面内容的正确性需要得到保证，一方面内容的格式也很重要，细节丰富，逻辑缜密的answer可以激发模型更多的回答能力。

补充：

1. SFT阶段不能太多的知识注入：过多的知识注入，或者超过模型能力本身的回答过多会导致[对齐税](https://zhida.zhihu.com/search?content_id=240902032&content_type=Article&match_order=1&q=%E5%AF%B9%E9%BD%90%E7%A8%8E&zhida_source=entity)，这是OpenAI的blog也曾经提到的，这就是我为什么不建议模型过多在SFT阶段学习知识，会影响其学习指令遵循的能力。


## **提问：** 提升sft的prompt的多样性有什么好的方法？

**回答：**

- 明文TAG法：也就是对SFT的prompt进行打tag，对其中的名词和动词进行分类打标，最后通过tag对prompt的分布进行调整，保证tag的分布是均匀的。著名的就是InsTag [1] 这个方法。

![](https://pic1.zhimg.com/v2-6866bd7d362e6597ac0779198d725b90_1440w.jpg)

- 模型embedding聚类方法：通过模型最后一层的embedding对prompt进行表示，那么通过prompt embedding的距离表示prompt的相似度，对于过于相似的prompt进行删除。著名的有Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning [2]。

![](https://pic3.zhimg.com/v2-3680b3c9e49fcfb4ea2cc6fdfe0939e0_1440w.jpg)

Figure: Blue points: training data point ; Red points: novel data points to be seleted.

- 从complexity角度，对于prompt直接进行难度的升级，所以即使在同一个语意空间的prompt也会变得diverse。比较著名的是Wizard 方法 [3]，通过GPT4进行prompt难度升级，然后构成complexity丰富的prompt。

![](https://pic3.zhimg.com/v2-5035db54296d19455f86a80951477c18_1440w.jpg)

## **提问**：在什么情况下DPO exactly 数学上等同于 PPO。

![](img/Pasted%20image%2020250210154618.png)

[Site Unreachable](https://zhuanlan.zhihu.com/p/687067338)

## **提问**：DPO的变体有哪些，主要解决DPO的什么问题？

**回答：**

- RSO [1]：由于DPO的[蒙特卡洛采样](https://zhida.zhihu.com/search?content_id=240837368&content_type=Article&match_order=1&q=%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E9%87%87%E6%A0%B7&zhida_source=entity)很难达到，所以其实DPO几乎是off-policy的[采样方式](https://zhida.zhihu.com/search?content_id=240837368&content_type=Article&match_order=1&q=%E9%87%87%E6%A0%B7%E6%96%B9%E5%BC%8F&zhida_source=entity)，RSO主要从DPO的采样方式来解决DPO的问题。
- Iterative DPO [2]：同样由于DPO的蒙特卡洛采样很难达到，所以通过[on-policy](https://zhida.zhihu.com/search?content_id=240837368&content_type=Article&match_order=1&q=on-policy&zhida_source=entity)的方式采样来替代off-policy的采样。它通过多次迭代来逐步优化模型，每次迭代都基于上一次的结果进行调整。
- IPO [3]：由于BT model的目标是最大化正负response的reward gap，但其实其中忽略了真实情况下我们组的pair可能会有噪音，那么无限去扩大reward gap其实是不准确的，也就是overfit了preference的pair数据，那么解决方案是需要限制这个gap的范围。
- DPOP [4]：由于LLM model很难区分[编辑距离](https://zhida.zhihu.com/search?content_id=240837368&content_type=Article&match_order=1&q=%E7%BC%96%E8%BE%91%E8%B7%9D%E7%A6%BB&zhida_source=entity)较小的pair，那么当持续去区分这批case的时候，模型效果会崩塌，现象是正例子和负例子的概率都往下掉。那么DPOP用了一个新项来惩罚正例往下掉的pair，使得正例概率继续提升。

[1] Liu T, Zhao Y, Joshi R, et al. Statistical rejection sampling improves preference optimization[J]. arXiv preprint arXiv:2309.06657, 2023.

[2] Yuan W, Pang R Y, Cho K, et al. Self-rewarding language models[J]. arXiv preprint arXiv:2401.10020, 2024.

[3] Azar M G, Rowland M, Piot B, et al. A general theoretical paradigm to understand learning from human preferences[J]. arXiv preprint arXiv:2310.12036, 2023.

[4] Pal A, Karkhanis D, Dooley S, et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive[J]. arXiv preprint arXiv:2402.13228, 2024.

[一些RLHF的平替汇总](https://mp.weixin.qq.com/s/Gjng5jKNc7igblOfHNMYZQ)

## **提问：**如何在公开数据集中筛选合适自己模型的sft数据？

- 利用sft model和pretrain model的关系筛选模型的sft数据：

- IFD方法 [1]：利用以下公式进行数据选择： rθ(Q,A)=Pθ(A|Q)/Pθ(A) 。这个公式其实是计算pretrain model生成对齐后模型的answer的难度（在 prompt的condition 下生成A的概率）。这个概率越低，越说明生成难度高，那么[sft模型](https://zhida.zhihu.com/search?content_id=241026411&content_type=Article&match_order=1&q=sft%E6%A8%A1%E5%9E%8B&zhida_source=entity)学习到的对齐规律越多，那么我们更应该选择这个sft数据。

![](https://pic3.zhimg.com/v2-5f827469a56fe1828f65da3fafc47eb8_1440w.jpg)

- Hybrid Method （混合了多种之前列举的指标和方法。）：例如 What MakeGood Data for Alignment? A Comprehensive Study of Automatic Data Selectionin Instruction Tuning [2] 文章，从complexity，[diversity](https://zhida.zhihu.com/search?content_id=241026411&content_type=Article&match_order=1&q=diversity&zhida_source=entity)和quality三个方向对sft数据建模，训练了多个模型对各个指标维度进行分别衡量。

![](https://pic3.zhimg.com/v2-0a8d75e21061628ef917fc3a2d89a730_1440w.jpg)

[1] Li M, Zhang Y, Li Z, et al. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning[J]. arXiv preprint arXiv:2308.12032, 2023.

[2] Liu W, Zeng W, He K, et al. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning[J]. arXiv preprint arXiv:2312.15685, 2023.

## **提问：**如果我们把推理结果相关的多个关键段落按照下列规则藏在一段文章里：

- A：关键段落藏在头部
- B：关键段落藏在尾部
- C：关键段落藏在中间
- D：关键段落随机分布在文章中

然后，把文章交给大模型，并要求它生成正确的推理过程和结果。请按照模型能预测正确推理过程和结果的概率对这几个规则排序？

**回答：**B > A > C > D. 这是一个经典的长文本的大海捞针的问题，整体而言，模型的attention分布集中在头部和尾部，中间的attention较少，那么B > A > C。最后还有如果分散在文章中，这个结论就更难获得。Mistral 70B的研究结果如下：

![](https://picx.zhimg.com/v2-3e3df2b2c71fb5592393a6023ad53fdb_1440w.jpg)

更多结论可以参考论文 Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models [1]。

## **提问：** Pair RM是什么形式的RM，相比于原RM形式有什么好处？

**回答：** 原RM是BT model形式的RM，每个sample组成形式是（prompt，answer)，通过maximize positive sample和negative sample的gap来完成pointwise的rank。Pair RM是pairwise rank，数据组成形式是（prompt，pos_answer, neg_answer）. Pair RM的好处是pos answer和neg answer可以互相在context下看到两者，那么可以通过字面的比较找到两者的diff，整体解释性和[泛化能力](https://zhida.zhihu.com/search?content_id=241149187&content_type=Article&match_order=1&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)都会比普通RM好。**因为普通RM很容易overfit原数据，很难找到真正diff地pattern。**现在[Alpaca-Eval](https://zhida.zhihu.com/search?content_id=241149187&content_type=Article&match_order=1&q=Alpaca-Eval&zhida_source=entity) [1]榜单上就有Pair RM的身影，而且Pair RM整体很小 [2]，效果很好。



## 参考资料

百面LLM： https://www.zhihu.com/column/c_1747590116120698880 

LLM常见面试问题- SFT篇 - 技术微佬的文章 - 知乎
https://zhuanlan.zhihu.com/p/714687583

[什么是Cosine优化器？在大模型中应该怎么设置cosine优化器的周期比较好？](https://zhuanlan.zhihu.com/p/685354437)

