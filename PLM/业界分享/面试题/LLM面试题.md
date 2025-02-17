
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

## **提问：** 为什么G4-turbo在topp=0.0的情况下，还不能保证预测稳定性？（社区最近广泛讨论的问题）

附社区具体问题：

> 我构造了一个测试场景来测试该问题：  
> 使用英文翻译为中文的任务，输入token量大于1k，输出token量也大于1k。总体控制在4k context范围内，方便兼顾到老版本模型和其他供应商的模型。  
> 所有测试中输入prompt完全一致。  
> 设置temperature=0，固定seed  
> 对于每个模型请求至少20次，计算输出token序列结果之间的最大相同前缀的平均长度，越大说明第一个分歧token出现的越晚。  
> 备注：0613版本的模型并不支持seed，也不会输出 system_fingerprint 。  
> gpt-3.5-turbo-0613模型的结果是最符合一般人预期的，在这种[贪心解码](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=1&q=%E8%B4%AA%E5%BF%83%E8%A7%A3%E7%A0%81&zhida_source=entity)的策略下，平均相同长度超过500。  
> gpt-4-0613与从1106开始的所有后续模型的平均相同长度大多在60-90的范围内。**对于我的测试案例，超过100token的输出时，大概率结果不会完全一致。**  
> 作者：孔某人  
> 链接：[https://zhuanlan.zhihu.com/p/688676344](https://zhuanlan.zhihu.com/p/688676344)

**回答：**预测稳定性在[LLM](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=1&q=LLM&zhida_source=entity)中一直是一个问题，我通过现在的现象分析一下可能存在的问题：

1. 如[swtheking：大模型的面试题系列-1](https://zhuanlan.zhihu.com/p/684958325)所讲的一种原因，由于模型预测时候，Open-AI会收集当时多个请求同时组batch，那么预测由于padding的影响，本身就会产出token level的不一致，当然这个不一致影响比较大，因为第一个token不一致后续都会不一致。
2. 更极端的组batch的方式是，把多个请求pack在一起，然后使用block diagonal attention的方式进行预测，那么这个预测会比1方式更严重影响预测稳定性，因为pad更多个token。（ 
    
    [@王焱](https://www.zhihu.com/people/7c894b915042fe363aed838b276951eb)
    
     提供）
3. 预测中融合算子的加速，以及算子计算顺序的问题影响的预测结果。因为对于不同的硬件适配，甚至在同一硬件的算子计算顺序的随机性也能带来token level的不一致，但这种不一致概率较小。
4. 除此之外，[tensor parallel](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=1&q=tensor+parallel&zhida_source=entity)，data parallel以及batchsize变化会导致内部选择kennel不一致（计算算子不一致），那么导致效果的diff（比如vllm框架下）。
5. 可能存在类似投机解码类更高效的解码方式带来的预测不稳定。当然[投机解码](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=2&q=%E6%8A%95%E6%9C%BA%E8%A7%A3%E7%A0%81&zhida_source=entity)理论上不会带来预测不稳定。
6. Sparse MOE的一些问题存在，Continuous batching 的时候因为每个 batch 的 case 不一样导致超 capacity factor 的时候会丢不同 token 导致不 [deterministic](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=1&q=deterministic&zhida_source=entity)，参考[Non-determinism in GPT-4 is caused by Sparse MoE](https://link.zhihu.com/?target=https%3A//152334h.github.io/blog/non-determinism-in-gpt-4/%23are-you-really-sure-it-isnt-hardware)。（[Yao Fu](https://link.zhihu.com/?target=https%3A//yaofu.notion.site/Yao-Fu-s-Blog-b536c3d6912149a395931f1e871370db) 提供）

从 

[@孔某人](https://www.zhihu.com/people/4e46fd4005e7434340ea1e82ac641a6d)

 观测到的现象可以推测，

- GPT3.5-turbo由于模型小，可能没有进行组batch的操作，仅仅是某些[融合算子](https://zhida.zhihu.com/search?content_id=241175789&content_type=Article&match_order=2&q=%E8%9E%8D%E5%90%88%E7%AE%97%E5%AD%90&zhida_source=entity)加速导致了算子计算顺序不一致带来的不稳定。
- G4-turbo和G4也许使用了组batch的形式，或者pack多个user query的形式进行预测推理，导致了很多预测不稳定。
- 也许在大模型下有类似投机解码类更高效的解码方式带来的预测不稳定。

## **提问：** BT model （DPO，RM的训练形式）的问题在哪？

**回答：**

BT model loss形式如下：

loss=−logsigmoid(pos−neg)

- 最大化正负例子的差距得到的模型会塌缩成只有正例子的空间，失去所有负例子的概率。在DPO中就是只会生成正例，负例子输出概率为0。在RM中正例子会无限接近于1，负例子会无限接近于0。那么这样的模型是没有entropy的，抗噪声能力会减弱。如果正负pair标错了，会导致严重后果。
- 忽略语意或字面上差别较小的pos sample和neg sample，过度关注语意或字面上差别较大的pos sample和neg sample，也就是比较容易学的case，并overfit。这是logsigmoid函数的问题，用[hinge loss](https://zhida.zhihu.com/search?content_id=241198497&content_type=Article&match_order=1&q=hinge+loss&zhida_source=entity)这类loss可以缓解这一问题。
- 不能找出[全序关系](https://zhida.zhihu.com/search?content_id=241198497&content_type=Article&match_order=1&q=%E5%85%A8%E5%BA%8F%E5%85%B3%E7%B3%BB&zhida_source=entity)，如果数据集里有A > B, B > C, C > A这种[偏序关系](https://zhida.zhihu.com/search?content_id=241198497&content_type=Article&match_order=1&q=%E5%81%8F%E5%BA%8F%E5%85%B3%E7%B3%BB&zhida_source=entity)，并不能找到它的nash equivalence的点，只会学乱。

## **提问：** 在LLM中，假设我们可以在不同层中，交换两个位置token的[embedding](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=1&q=embedding&zhida_source=entity)，那么是偏顶层对最后预测的影响大，还是底层对最后预测的影响大？

**回答：**

有两个观点：

顶层：

- 因为第n层的第m个token会看到下面n-1层前m-1个token的所有attention模式(对所有前面token的attention)，那么如果你交换的是第k层（k << n）的(m-1)个token和前面任一token，那么在第k层可以兼容这个模式，所以预测影响不会特别大。相反如果在最顶层，那么第(m-1)个token和第m - j (j > 1)个token交换，那么第m个token只有一个错误的[attention模式](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=2&q=attention%E6%A8%A1%E5%BC%8F&zhida_source=entity)可以看到（最后的错误模式），那么预测影响很大。
- 底层的typo错误会在模型中间被减弱，被LLM兼容。还有之前有对bert模型进行分析，BERT Rediscovers the Classical NLP Pipeline [1], BERT分析底层是[词法分析](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=1&q=%E8%AF%8D%E6%B3%95%E5%88%86%E6%9E%90&zhida_source=entity)，高层是语义分析, 因此模型可能在高层影响更大。
- 第n层改变某个token等于前n-1层这个位置的token的作用都被修改了。

底层：

- 底层是等于是从句子角度改了两个词的位置，从底层进行修改了句子的顺序。

我个人觉得是**顶层比较大**，**因为改顶层的embedding相当于底层把顺序换了，而且删了下面很多层的作用。**除此之外[GPT4](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=1&q=GPT4&zhida_source=entity)也是可以兼容顺序错乱的情况的。

最后附几个insightful的实验

1. In-Context Learning Creates Task Vectors [1]发现，[大模型](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=1&q=%E5%A4%A7%E6%A8%A1%E5%9E%8B&zhida_source=entity)在模型的中下层主要是做task classification任务，在上层才是做预测任务，它也曾经做过类似实验，只是选择的是特定的token交换（[ICL](https://zhida.zhihu.com/search?content_id=241245066&content_type=Article&match_order=1&q=ICL&zhida_source=entity)中demonstrations的最后一个token和classification任务的最后一个预测token对换），发现是底层影响没有上层高。具体也可以看我的blog [2]。

![](https://pica.zhimg.com/v2-11c5aad0dc052afa5332468e26e6c66a_1440w.jpg)

![](https://pica.zhimg.com/v2-72fa0b3f3e56e543f57a5ee4c5cf5278_1440w.jpg)

2. GPT4对打乱文字顺序的兼容性很强，这可能是训过类似数据的原因。[4]

![](https://pica.zhimg.com/v2-2d4eea847807f7b7bce78a3f2d9a92b8_1440w.jpg)

[1] Tenney I, Das D, Pavlick E. BERT rediscovers the classical NLP pipeline[J]. arXiv preprint arXiv:1905.05950, 2019.

[2] Hendel R, Geva M, Globerson A. In-context learning creates task vectors[J]. arXiv preprint arXiv:2310.15916, 2023.

[3] [https://difficult-link-dd7.notion.site/c31d141411be4d0eb50473fe6abae1db?v=50264a9824494b6c836ba0c6f3bebd2f](https://link.zhihu.com/?target=https%3A//difficult-link-dd7.notion.site/c31d141411be4d0eb50473fe6abae1db%3Fv%3D50264a9824494b6c836ba0c6f3bebd2f)

[4] Cao Q, Kojima T, Matsuo Y, et al. Unnatural error correction: Gpt-4 can almost perfectly handle unnatural scrambled text[C]//Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023: 8898-8913.

## **提问：** 在预训练中的学到的代码数据和文本数据，在SFT中哪种数据的模式越难被改变，或者说知识注入越难？

**回答：** 是代码数据，因为在预训练中代码数据的确定性更高，ppl更低，记忆越深刻，而文本数据变化更大，ppl更高，熵更高。在SFT过程中，改变文本数据比较容易，因为本身ppl就会高，但代码数据会比较难，因为本身ppl会比较低，或者说代码数据的生成确定性更高，少量样本很难对其内部改变，只能大段替换。

## **提问：** 什么样的数据格式在SFT或者ICL阶段可以提升模型的reasoning的能力？

**回答：**

数学[reasoning](https://zhida.zhihu.com/search?content_id=241354744&content_type=Article&match_order=2&q=reasoning&zhida_source=entity)上是有三种形式可以显著提高效果模型的reasoning的能力 [1]

- Reverse ： 128 + 367 = 495 -> 128 + 367 = ^594, 因为人就是反着计算的，从个位数到百位数。
- COT or POT (Simplified Scratchpad): 把这个计算过程列举下来，用自然语言，符号或者代码形式呈现。
- Detailed Scratchpad：把整个思考过程详细地用自然语言和符号表达出来。

![](https://pic2.zhimg.com/v2-6142d053cde90342c01315e9fc668ccb_1440w.jpg)

整体上Detailed Scratchpad需要的总条数最少就能达到100%在加法上的效果，但是其实总token数和plain需要差不多数量达到最好的效果。

[1] Lee N, Sreenivasan K, Lee J D, et al. Teaching arithmetic to small transformers[J]. arXiv preprint arXiv:2307.03381, 2023.

## **提问：** In context learning中下面哪一项比较重要：

1）input distribution

2）output distribution

3）input-output mapping

4）the sample order

5）the formatting of the demonstrations

**回答：**1）2）4）5）比较重要，3）有的说比较重要，有的说不是特别重要。

1）2）重要来自下面实验 [1]：在使用gold label（正确的label）并不比random label的效果高多少，但是比无ICL高很多（无ICL就没有了input distribution和output distribution了）。

![](https://pic3.zhimg.com/v2-567bc490afffd326d55d20c62e1eff7e_1440w.jpg)

Figure: Min et al compare three different methods: 1) **No-examples**: the LM conditions on the test input only, with no examples. This is typical zero-shot inference, first done by GPT-2/GPT-3. 2) **Examples with ground truth outputs**: the LM conditions on the concatenation of a few in-context examples and the test input. This is a typical in-context learning method, and by default, all outputs in the prompt are ground truth. 3) **Examples with random outputs**: the LM conditions on in-context examples and the test input, but now, each output in the prompt is randomly sampled from the output set (labels in the classification tasks; a set of answer options in the multi-choice tasks).

4)实验来源于论文 [2]: 无论few shot有多少个shot，不同模型只要重新排列demonstrations，分类准确率效果variance很大。

![](https://pic1.zhimg.com/v2-abf797c943f735f7442fb35e9f4997fe_1440w.jpg)

5）的细节可以看[swtheking：大模型的面试题系列-27](https://zhuanlan.zhihu.com/p/689508595)：因为ICL和SFT同源，COT数据在ICL中可以prompt更好的reasoning数据 。

4）在[1]论文中看上去不是很重要，但是在[openai](https://zhida.zhihu.com/search?content_id=241400305&content_type=Article&match_order=1&q=openai&zhida_source=entity)的论文中有一些不一样的结论 [3],

![](https://pic3.zhimg.com/v2-d90ac756918aa251fc1df572483c53a2_1440w.jpg)

Figure : Few-shot prompting becomes competitive with finetuning for large models; weak-to- strong learning is qualitatively similar in the prompting setting. (a) Average [zero-shot](https://zhida.zhihu.com/search?content_id=241400305&content_type=Article&match_order=2&q=zero-shot&zhida_source=entity) (single dashed), 5-shot (double dashed) and finetuning (solid) accuracy with ground truth labels as a function of strong student size. (b) Average 5-shot with weak labels (colored dashed) accuracy as a function of student model size. Hue of line indicates size of weak supervisor. Zero-shot and 5-shot same as in panel a. (c) Average weak-to-strong performance for 5-shot prompting (dashed with crosses), naive finetuning (dashed thin) and finetuning with the confidence loss (solid with triangle) as a function of student model compute. Results are averaged across 7 NLP tasks. Few-shot weak- to-strong performance becomes competitive with or outperforms finetuning for the largest strong students, though finetuning with the confidence loss does better.

这里的中间图片发现，黑色的虚线是ICL中用了gold label，而蓝色的虚线是小模型用gold label finetune以后生成的weak label（准确率是90左右），但效果却差了很多。和[1]可能的区别是任务的难度，整体这里的任务难度会高一些（虽然都是[classification](https://zhida.zhihu.com/search?content_id=241400305&content_type=Article&match_order=2&q=classification&zhida_source=entity)，但是这里任务是 22 popular NLP classification datasets covering ethics, commonsense reasoning, natural language inference, sentiment analysis, and other domains. ）。

最后还有更多细节可以看我的blog：

[Exploring the Potential of In-Context Learning: New Pathways for Enhancing Chat-Based Large Language Model Performance](https://link.zhihu.com/?target=https%3A//www.notion.so/c31d141411be4d0eb50473fe6abae1db%3Fv%3D50264a9824494b6c836ba0c6f3bebd2f)

[1]Sang Michael Xie and Sewon Min. "How does in-context learning work? A framework for understanding the differences from traditional supervised learning ".

[2]Lu Y, Bartolo M, Moore A, et al. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order [sensitivity](https://zhida.zhihu.com/search?content_id=241400305&content_type=Article&match_order=1&q=sensitivity&zhida_source=entity)[J]. arXiv preprint arXiv:2104.08786, 2021.

[3]Burns C, Izmailov P, Kirchner J H, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision[J]. arXiv preprint arXiv:2312.09390, 2023.

## **提问：** 如何在predict阶段提升一个大模型的回答质量？

**回答：**有以下三种主流的做法：

- 促使模型给予COT形式的回答，会提升模型的输出质量。其中包括提示词"think step by step"，以及prompt中加入COT形式回答的例子，都会促使模型有COT的回答，来提升模型输出质量。
- Self consistency [1] 或者 universal self-consistency [2]的方法。这一系列的方法主要是利用模型多个答案[ensemble](https://zhida.zhihu.com/search?content_id=241430254&content_type=Article&match_order=1&q=ensemble&zhida_source=entity)的思想提升模型的效果。Self consistency使用在数学领域，在模型生成多个答案的时候，利用答案结果的一致性判断哪个答案是正确的：一般模型解数学题的时候会生成COT过程和答案，最终多个结果用投票方式决定使用哪个答案作为最终答案。对于universal self-consistency是把self consistency方法推广到所有领域，最终是把多个答案再次输入模型判断哪几个答案是一致的，并输出最终结果。

![](https://pic3.zhimg.com/v2-a6b37e4ae2dbe039a1dc6fd11e7a634c_1440w.jpg)

- Self-debug或者反思[3]，把模型结果和**现实交互反馈**再次喂给模型，让其反思和debug，那么会提升模型最终效果。

![](https://pic3.zhimg.com/v2-0eb271c697bd7a45cdf0838e22cb48c6_1440w.jpg)

这三种做法后续的大融合就是类似于lang chain或者agent的思路。本质是让大模型这种[概率模型](https://zhida.zhihu.com/search?content_id=241430254&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B&zhida_source=entity)一次生成对的答案的概率不是特别高（> 90%），需要在多轮对话反馈(这种反馈可以来源于工具或者人的监督)以后修正其答案，提升答案质量才能完全为人所用。

[1]Wang X, Wei J, Schuurmans D, et al. Self-consistency improves chain of thought reasoning in language models[J]. arXiv preprint arXiv:2203.11171, 2022.

[2]Chen X, Aksitov R, Alon U, et al. Universal self-consistency for large language model generation[J]. arXiv preprint arXiv:2311.17311, 2023.

[3]Chen X, Lin M, Schärli N, et al. Teaching large language models to self-debug[J]. arXiv preprint arXiv:2304.05128, 2023.

## **提问：** In Context Learning和SFT的关系是什么？

**回答：** ICL是一种特殊的SFT。在论文 EXPLORING THE RELATIONSHIP BETWEEN IN- CONTEXT LEARNING AND INSTRUCTION TUNING [1] 中，用很多实验证明了ICL和SFT在改变LLM内部embedding维度有诸多相似。

![](https://pic3.zhimg.com/v2-f28f11dea02d773da1987c99bdb6b31e_1440w.jpg)

- ICL和SFT（IT）在最终模型state上是相似的：其中用 hanchor 是普通输入一个query得到的最后一个词最后一层的表示，而 hICL 是ICL+一个query的最后一个词最后一层的表示，最后 hIT 是SFT后的一个query预测阶段的query最后一个词最后一层的表示。

![](https://pic4.zhimg.com/v2-103e13e34539e2d55a89e1c75b4587d9_1440w.jpg)

图a）中显示了ICL和SFT的最后表示相似度很高，但ICL和Anchor，以及SFT和Anchor的最后表示相似度很低。  
  
[1] Duan H, Tang Y, Yang Y, et al. Exploring the relationship between in-context learning and instruction tuning[J]. arXiv preprint arXiv:2311.10367, 2023.

## **提问：** 现阶段利用SFT阶段在做RLHF的方法有哪些？优点和缺点是什么？

**回答：** 总结的方法有：

	- CFT[1], 通过在SFT中放入两个response，一个GPT4的，一个GPT3.5的，然后加一个后缀prompt进行区分，比如<GPT4>，<GPT3>。sft data会变成 Q <GPT4>：GPT_ans_4, Q<GPT3.5>: GPT_ans_3.5。然后模型内部会通过prompt区分何时生成GPT4这种高质量答案，何时生成GPT3.5这种低质量答案。

![](https://pica.zhimg.com/v2-83f96e2a4a33a1daa2951735e1fef284_1440w.jpg)

![](https://pic3.zhimg.com/v2-31d408fe6206096f8303c4973fd64010_1440w.jpg)

- CUT 方法[2]：如CFT一样加入后缀，不过是用reward表示，而且会把Judgement也加入到后缀中，增加整体的可解释性。除此之外，他们还引入了一种Contrastive Unlikelihood Training的方式，这个留到后续再细讲。

![](https://pic1.zhimg.com/v2-2974bf051c85a175d2ce0c7973357a92_1440w.jpg)

- ORPO[3]利用SFT CE loss后面加了一项类似DPO的loss，同时做SFT和DPO，进行对比区分。这个方法相比于前两个方法的好处，是没有特定解释的prompt。

![](https://pic2.zhimg.com/v2-e0b2a88feeaafa154305c699fa052beb_1440w.jpg)

![](https://pic1.zhimg.com/v2-0e2e409a7798cd6efaea891e278c33f0_1440w.jpg)

最后给我自己的见解：

SFT阶段做RLHF本质是强化学习的一个分支Hindsight relabeling [4] 的方法的演变。通过对比好坏case的方式找到自动relabeling的线索。但整体和RLHF相比差距比较远，因为这种relabeling的方式非常隐式的表达RLHF，目标不够强（没有强reward指导，和gradient回传）。总结而言这是一个简单，高效且解释性高的RL方法，适合在不做RLHF的情况下达到超过SFT的效果。

[1] Wang G, Cheng S, Zhan X, et al. Openchat: Advancing open-source language models with mixed-quality data[J]. arXiv preprint arXiv:2309.11235, 2023.

[2] Xu W, Cai D, Zhang Z, et al. Reasons to reject? aligning language models with judgments[J]. arXiv preprint arXiv:2312.14591, 2023.

[3] Hong J, Lee N, Thorne J. Reference-free Monolithic Preference Optimization with Odds Ratio[J]. arXiv preprint arXiv:2403.07691, 2024.

[4] Andrychowicz M, Wolski F, Ray A, et al. Hindsight experience replay[J]. Advances in neural information processing systems, 2017, 30.


## **提问：** 如何看待各种ppo [rlhf](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=1&q=rlhf&zhida_source=entity)的平替算法dpo/kto/rrhf/slic/orpo/samug/remax等算法号称性能等能超过[ppo](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=2&q=ppo&zhida_source=entity)？

**回答：**

那么我把[PPO算法](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=1&q=PPO%E7%AE%97%E6%B3%95&zhida_source=entity)的优点列为以下几点，如果后续有算法可以做到，maybe可以平替：

1. On policy采样：on policy采样目前看来是最高效的拟合[蒙特卡洛采样](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=1&q=%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E9%87%87%E6%A0%B7&zhida_source=entity)方式。举个例子，如果不使用on policy采样，你[随机采样](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=1&q=%E9%9A%8F%E6%9C%BA%E9%87%87%E6%A0%B7&zhida_source=entity)到一个模型generate概率差值很大的两个response，如果符合人类preference，那么本身就不需要排序，如果不符合，你也很难通过RLHF纠正它。如果强行纠正，会破坏模型本来的平衡。
2. Credit Assign: 由于value model的存在，其实PPO会很好的把reward分配给不同的token，那么一些关键的token会合理地分配一个高reward，一些不关键的token会分配一个低reward。
3. Rank Model：PPO内部其实是一种内置的rank model，比较的是高reward和低reward的response，只是高和低一直是动态的变化的。为什么[rejection sampling](https://zhida.zhihu.com/search?content_id=241624841&content_type=Article&match_order=1&q=rejection+sampling&zhida_source=entity)这类的算法无法work，因为preference data中的噪声，你选出的Top 1大概率不是Top 1。

更多内容：[Site Unreachable](https://zhuanlan.zhihu.com/p/690724347)

## **提问：** 如何处理[reward model](https://zhida.zhihu.com/search?content_id=241682255&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)中的噪声数据？

**回答：**这个问题首先需要回答reward model的噪声来自哪几个方面：

- 如果reward model的pair数据来自人标注的，那么人类的preference的倾向性以及标注人员的专业性会带来一定的[bias](https://zhida.zhihu.com/search?content_id=241682255&content_type=Article&match_order=1&q=bias&zhida_source=entity)，也就是之前广泛研究的众包系统的Noise。
- 如果reward model的pair数据来自AI，例如[GPT4](https://zhida.zhihu.com/search?content_id=241682255&content_type=Article&match_order=1&q=GPT4&zhida_source=entity)，那么这种倾向性也很严重，比如length bias。（严格来说，这属于bias，不能算噪声。）

那么如何去噪，这里可以使用一些古早的方式：

预测阶段去噪声：

1. Ensumble model去噪声，也就是ensemble多个rm model的checkpoint进行预测减少噪声的影响（model merge）。
2. Margin 去噪声，只有预测的pair的分数大于一定阈值的时候，进行预测减少噪声。

数据阶段去噪声：

1. Multiview去噪声，用多个模型进行训练，然后预测训练集合，全部可以预测正确pair保留下来，有对有错的可以丢弃或者交给人标注。
2. Active Learning思路去噪声，训练一个模型，然后把margin小于一定阈值的送给标注人员去噪声。

最后这些思路我没有真正实践过，也没有刻意比较过哪种方法好坏，但基本这些方法在之前的对话系统工作中和 

[@王焱](https://www.zhihu.com/people/7c894b915042fe363aed838b276951eb)

 一起实践过，都比较有效。

## **提问：** 现在主流实现RM有几种，是怎么做的？

**回答：**

1）是主流Instruct GPT [1] 提出的，就是在整个句子之后插入一个新token位置，这个token是只有0-1两个选择，在实现上trl库其实是利用最后一个token后加一层MLP（或两层MLP）然后进行BT model的loss，进行pair-wise的训练。当然我也见过有一种改进就是0-1的二分类预测 [2]，这种做法比较适合有明确正误标准的方向，比如数学，代码和推理。

2）是phi-2-math [3] 使用的，使用全序列token进行预测，也就是在全序列的token的logit后都加入MLP层进行二分类，然后对这些所有token进行BT model的loss，进行[pair-wise](https://zhida.zhihu.com/search?content_id=241822855&content_type=Article&match_order=2&q=pair-wise&zhida_source=entity)的训练。这种做法本质是类似RL中[Reinforce算法](https://zhida.zhihu.com/search?content_id=241822855&content_type=Article&match_order=1&q=Reinforce%E7%AE%97%E6%B3%95&zhida_source=entity)的做法，然后把所有的token赋予最后的[reward值](https://zhida.zhihu.com/search?content_id=241822855&content_type=Article&match_order=1&q=reward%E5%80%BC&zhida_source=entity)，这样的做法好处其实是变相地增加了样本量，坏处是增加了大量噪声。和Reinforce算法一样是无bias，但高variance的做法。最终效果取决于变相增加的样本量是否能抵抗住高variance。在数学上使用合理地原因是，数学是过程式学习，过程中每一个token都很重要。**这种做法类似于没钱版本的verify step by step** [4]。

[1] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in neural information [processing systems](https://zhida.zhihu.com/search?content_id=241822855&content_type=Article&match_order=1&q=processing+systems&zhida_source=entity), 2022, 35: 27730-27744.

[2] Dubois Y, Li C X, Taori R, et al. Alpacafarm: A simulation framework for methods that learn from human feedback[J]. Advances in Neural Information Processing Systems, 2024, 36.

[3] Liu B, Bubeck S, Eldan R, et al. Tinygsm: achieving> 80% on gsm8k with small language models[J]. arXiv preprint arXiv:2312.09241, 2023.

[4] Lightman H, Kosaraju V, Burda Y, et al. Let's Verify Step by Step[J]. arXiv preprint arXiv:2305.20050, 2023.

## **提问：** 现有的rm模型泛化能力怎么样？

**回答：** 现有rm模型似乎泛化能力非常有限，原因是很多论文提出很难看到rm模型在scale model size以后有更多的收益[1]。如果把rm当做一种instruct任务，如果你用同样的数据进行训练LLM，理论上[泛化能力](https://zhida.zhihu.com/search?content_id=242007524&content_type=Article&match_order=3&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)应该可以随着模型变大而获得更强的能力，因为你的[底座模型](https://zhida.zhihu.com/search?content_id=242007524&content_type=Article&match_order=1&q=%E5%BA%95%E5%BA%A7%E6%A8%A1%E5%9E%8B&zhida_source=entity)的能力变大了。但现阶段，甚至在alpaca eval榜单上能看到[bert](https://zhida.zhihu.com/search?content_id=242007524&content_type=Article&match_order=1&q=bert&zhida_source=entity)训练出的rm效果很好，那么显然rm的整体scale是没有特别的收益的。那么后续怎么提升rm模型能更好地利用大模型[基座](https://zhida.zhihu.com/search?content_id=242007524&content_type=Article&match_order=1&q=%E5%9F%BA%E5%BA%A7&zhida_source=entity)的能力将是rm研究的重要方向。

[1] Huang S, Noukhovitch M, Hosseini A, et al. The N+ Implementation Details of RLHF with PPO: A Case Study on TL; DR Summarization[J]. arXiv preprint arXiv:2403.17031, 2024.

## **提问：** 请问模型在SFT后会出现“复读机”情况该如何debug（可以是各种形式上的复读，比如复读最后1-N个token，复读训练数据很少出现的token，复读大段有逻辑的文字），以及出现的原因是什么？

**回答：**复读机问题是一个偏向[LLM](https://zhida.zhihu.com/search?content_id=242122203&content_type=Article&match_order=1&q=LLM&zhida_source=entity)早期的问题，也就是pretrain模型能力不强的时候才会发生的问题。

如果debug会发现，复读机的本质是，复读的那部分数据不能给予更多的信息，所以模型attention时候会跳过这部分信息依然从之前的context后进行预测。也就是`<context> -> 复读数据 & <context, 复读数据> -> 复读数据`。之前的做法一般会搞一个复读的penalty阻止这一现象，现在几乎没用了。

那为什么sft以后会发生这种情况？

**因为当sft数据的能力远大于pre-train model的本身能力，尤其你试图想overfit这部分数据的时候。**

在overfit的过程中，模型会为了记住这部分数据，而过多修改原先的attention模式，且会打乱原始pretrain模型的原始分布。之前国内经常发现这种问题，就是总是想用比较低水平的模型硬distill G4的效果，那么最终结局就是复读机，但现在大家pre-train做起来以后，这个问题几乎不存在了。

---

最近好像大家又开始聊复读机的问题，在pretrain完模型更严重（我倒是没遇到过）。猜测模型在这个context下大概率只能输出这个token，那就是在这个位置塌缩了，因此模型遇到数据多样性不够的情况下，某些位置倾向性输出固定内容。但这种位置确定性塌缩会泛化到新数据上，感觉是模型承载力不足。所以导致“复读机”的问题应该主要有两个因素：1）模型承载能力（模型大小，[模型结构](https://zhida.zhihu.com/search?content_id=242122203&content_type=Article&match_order=1&q=%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84&zhida_source=entity)），2）[数据多样性](https://zhida.zhihu.com/search?content_id=242122203&content_type=Article&match_order=2&q=%E6%95%B0%E6%8D%AE%E5%A4%9A%E6%A0%B7%E6%80%A7&zhida_source=entity)。

补充：确实sft数据训练过少的情况也会导致复读，本质还是模型一般学不会何时停止输出结果，因为pre-train的时候一般是packing，那么基本不太会学会输出`<eos>`，所以[pre-train模型](https://zhida.zhihu.com/search?content_id=242122203&content_type=Article&match_order=1&q=pre-train%E6%A8%A1%E5%9E%8B&zhida_source=entity)一般都是会复读的。

## 大模型lr退火阶段的模型变化和启示

**minicpm[1]的实验结果：**

- 在lr退火阶段，模型的loss会迅速下降。

![](https://pic4.zhimg.com/v2-d36d83ddf64bfb00bae788a3e12c4815_1440w.jpg)

- 在lr退火阶段，引入高质量数据，模型效果会提升。

![](https://picx.zhimg.com/v2-1a1073199d1694d95ac62ca0ec447f69_1440w.jpg)

1. A-1: 2.4B model, decay using only [pre-training](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=pre-training&zhida_source=entity) data, followed by 4B token SFT.
2. A-2: 2.4B model, decay using the aforementioned high-quality data unlabeled data and SFT data mixed into pre-training data, also followed by 4B token SFT.
3. B-1: 1.2B model, decay using only pre-training data, followed by 6B token SFT.
4. B-2: 1.2B model, decay using only pre-training data, followed by 12B token SFT.
5. B-3: 1.2B model, annealing using the aforementioned high-quality data + SFT data mixed into pre-training data, also followed by 6B token SFT.

以上现象说明：大模型的lr退火阶段非常特殊（区别于模型lr未退火阶段），模型内部参数此时的变化会影响最终模型的效果。那我们想问一个问题：

> 模型退火阶段，模型参数的变化到底是如何的？和退火前的[模型参数](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=2&q=%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0&zhida_source=entity)变化有何不同？

---

**分析现象：**

- 在lr比较小阶段整体loss迅速下降，反而在lr大的时候loss下降速度不快。这个现象应该和loss的landscape相关，我们可以假设loss的landscape是下面的形式：

![](https://pic3.zhimg.com/v2-6f5dbe2c7e8dd27216ce36ec4aae403e_1440w.jpg)

loss，纵轴是loss，横轴是step（有点丑，用chatgpt画了两次表达不了我的意思）

那么这个loss图片里存在两个[sharp minimum](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=sharp+minimum&zhida_source=entity)，当lr比较大的时候，会跳过这些sharp minimum，所以整体下降速度不快，但当lr比较小，退火的时候会进入sharp minimum，下降速度比较快。当然真实的[loss landscape](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=loss+landscape&zhida_source=entity)是多维度且更加复杂。（打个比喻，可能退火的时候类似进行了一种洞形式的空间，而loss landscape大体上看是个平原）。

- 在退火阶段加入更多高质量数据能获得loss更低点。

在这个“洞空间”内，由于minimum更加sharp，需要的gradient需要更加准确，配合着小lr才能获得红色的最低点。而更加高质量的数据可以提升gradient的准确性。

**问题回答：**

模型退火阶段，模型参数的变化到底是如何的？和退火前的模型参数变化有何不同？

猜测：**模型预测中某些特定context的特定位置塌缩成[长尾分布](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=%E9%95%BF%E5%B0%BE%E5%88%86%E5%B8%83&zhida_source=entity)。**

Loss迅速下降代表Cross Entropy迅速下降，压缩过程剧烈。那么猜测在这段lr下降过程中，说明**有一些位置**的token快速overfit了训练集合里的token分布，而形成长尾分布。也就是说对于下一个词预测更加确定，**某些context**后的next word prediction预测空间塌缩成只有_几个词_占据90%以上，其余词占据10%左右的[概率空间](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E7%A9%BA%E9%97%B4&zhida_source=entity)。**对于lr退火前，我个人猜测这些特定位置的词仍然保持着和别的位置一样的非长尾分布**，具体而言，就是可能_100-200个词_占据90%以上。（斜体的几个和100-200这些数字都是猜测，需要真实实验观测）。

**进一步的思考和猜测：**

- 这些特定位置大概是表达什么的位置？猜测：大概是一些事实型的答案和知识类的答案。

相比于无意义的形容词和语言结构的变化，更大概率塌缩的是事实型的答案，比如2024年的美国总统**拜登**，那么预计**拜登**这个词会迅速塌缩，然后成为事实型答案。

- 用这个理论解释为什么[sft模型](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=sft%E6%A8%A1%E5%9E%8B&zhida_source=entity)无法学习新知识？

相比于pre-train未退火阶段，大量位置的token还未塌缩，退火后的模型想在sft阶段学习新知识比较困难，因为一般sft阶段设置lr = 退火后的lr，那么这些token很难被修改，如果多次训练，强行修改容易把整个预训练学到的知识打乱。（这里的知识特指常识类别的知识。对于一些新的领域的知识或许能看到泛化性，但记忆和泛化效果应该不如，re-warmup然后退火这种[post-train](https://zhida.zhihu.com/search?content_id=242147517&content_type=Article&match_order=1&q=post-train&zhida_source=entity)。）

- 那么想压缩新知识进入模型应该怎么办？

借用之前一篇post-train论文[2]的方法, 需要我们混合一部分新数据加上老数据训练模型（老数据防止遗忘）。

6. 要经历re-warm up，让模型从之前的“洞穴”内出来，也就是图上红色的sharp minimum出来。
7. 然后经历高lr，寻找新的洞穴。
8. 在新的洞穴开始重新塌缩某些位置的概率空间。

[1] Hu S, Tu Y, Han X, et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies[J]. arXiv preprint arXiv:2404.06395, 2024.

[2] Ibrahim A, Thérien B, Gupta K, et al. Simple and Scalable Strategies to Continually Pre-train Large Language Models[J]. arXiv preprint arXiv:2403.08763, 2024.

## 对于两个100B的数据集1 & 2（假设分布差距比较大），那么如果需要顺序训练数据集，也就是在数据集1训完以后post-train数据集2，我们应该怎么做能达到几乎合并训（数据集1&2）的效果：

![](img/Pasted%20image%2020250211133440.png)

结论：当replay的数据量和原始数据量成正比时，几乎等同于混合训练。

这个结论在论文 Simple and Scalable Strategies to Continually Pre-train Large Language Models [1]证实，当然它列举了三要素：

- rewarm-up (但好像实验结论反而是是否warm up不重要)。
- re-decay。
- 当[分布差异](https://zhida.zhihu.com/search?content_id=242411567&content_type=Article&match_order=1&q=%E5%88%86%E5%B8%83%E5%B7%AE%E5%BC%82&zhida_source=entity)不是特别大的时候replay 5%原始数据，当分布差异特别大的时候10%-20%原始数据。

最终实验结果如下：

![](https://pic4.zhimg.com/v2-1608ed307a7558678360f8dbb2852641_1440w.jpg)

405M模型对比,左图是分布差距不大的两个数据集，右图分布差距较大

![](https://pic3.zhimg.com/v2-b98a9efe6fef1c08b9694883449cd190_1440w.jpg)

405M v.s. 10B模型对比

![](https://picx.zhimg.com/v2-c4ba2af43e8f5f7685fe840d03aa8995_1440w.jpg)

Bench mark效果，差距不大

[1] Ibrahim A, Thérien B, Gupta K, et al. Simple and Scalable Strategies to Continually Pre-train Large Language Models[J]. arXiv preprint arXiv:2403.08763, 2024.

## DPOP，也就是Smaug: Fixing Failure Modes of Preference Optimization with DPO-Positive论文中发现了DPO一个缺点，也就是positive和negative的probability同时下降，这里他给的[数学证明](https://zhida.zhihu.com/search?content_id=242565859&content_type=Article&match_order=1&q=%E6%95%B0%E5%AD%A6%E8%AF%81%E6%98%8E&zhida_source=entity)是对的嘛？如果是错的，请指出错误？（这里可以先看论文的数学证明）

[Site Unreachable](https://zhuanlan.zhihu.com/p/694960319)

## **提问：** 在PPO阶段为什么需要在把actor freeze 50步？模型这里在做什么？

**回答：**在internlm2 [1]和secret of RLHF [2]中都有提及，可以稳定value network的学习。

本质这个过程是在学习一个Dense reward model，或者说 token-wise reward model。在actor freeze 50步中，只有value network在被训练，且由于ref model和actor model一致，那么[KL散度](https://zhida.zhihu.com/search?content_id=242835782&content_type=Article&match_order=1&q=KL%E6%95%A3%E5%BA%A6&zhida_source=entity)的那部分loss为0。因此模型只做了图中从actor model采集experience，然后用reward model计算sentence reward，然后通过GAE来分别计算A advantage函数，最后通过采样中的P(s_t, a_t, s_(t+1))来聚合v_t,计算出token wise的reward，最后训练value network。

![](https://pic1.zhimg.com/v2-51e7c65b90d7310e02df1ff8e3d16c62_1440w.jpg)

其中GAE的计算如下, 参考[3]：

![](https://pic3.zhimg.com/v2-ed9c5622e52bf32cf9aa88d8e80b7a58_1440w.jpg)

[1] Cai Z, Cao M, Chen H, et al. Internlm2 technical report[J]. arXiv preprint arXiv:2403.17297, 2024.

[2] Zheng R, Dou S, Gao S, et al. Secrets of rlhf in large language models part i: Ppo[J]. arXiv preprint arXiv:2307.04964, 2023.

[3] [https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html](https://link.zhihu.com/?target=https%3A//newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)

## 为什么我们一直要追求更大的模型，更大的模型到底会带来什么？

**回答：**

1. 在经典论文scaling law [1]中，提出更大的模型可以得到更低的test loss，也就是更好的[泛化能力](https://zhida.zhihu.com/search?content_id=242942485&content_type=Article&match_order=1&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)，或者说是更高效的压缩能力。

![](https://pic3.zhimg.com/v2-ff7f297a985edb19bbddff96740e32d6_1440w.jpg)

2. 在Codex [2]中提出更大的模型可以得到更低的systax error. 这个应该算是[test loss](https://zhida.zhihu.com/search?content_id=242942485&content_type=Article&match_order=2&q=test+loss&zhida_source=entity)更低的副产物。以及更受instruction中error影响的问题，这可能是由于更大的模型中transformer predictor的拟合能力更强，更能受context的影响。

![](https://picx.zhimg.com/v2-0506a2fde9468cd6f4d2e6c4ca76f731_1440w.jpg)

1. 自我评估能力（self-reward）随着模型变大涌现出来的能力 [3].

![](https://pic1.zhimg.com/v2-b6dab0cb259ff0fd0f44c90e2e7cfc78_1440w.jpg)

在小模型中几乎等于随机猜测，等模型接近Large水平（62B），自我评估达到60以上。

[1] Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models[J]. arXiv preprint arXiv:2001.08361, 2020.

[2] Chen M, Tworek J, Jun H, et al. Evaluating large language models trained on code[J]. arXiv preprint arXiv:2107.03374, 2021.

[3] Luo L, Lin Z, Liu Y, et al. Critique ability of large language models[J]. arXiv preprint arXiv:2310.04815, 2023.

## ROPE公式中的base是越大越能适应长文本还是越小越能适应长文本？

[Site Unreachable](https://zhuanlan.zhihu.com/p/698397614)

## 在[LLM](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=LLM&zhida_source=entity)中选择像传统RL中value network和[policy network](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=policy+network&zhida_source=entity)共享底座会有问题吗？如果有解释一下为什么？

**回答：这种做法是有问题的。**但甚至在主流的[TRL库](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=TRL%E5%BA%93&zhida_source=entity)中使用的就是value network和policy network共享底座的方式，其motivation是为了降低显存。具体而言，这种方式仅仅在policy network后加了一层[MLP](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=MLP&zhida_source=entity), 也就是ValueHead,代表value network， 细节可以参考我们写的一个TRL库的[RLHF](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=RLHF&zhida_source=entity)的介绍[1]。

这种做法的问题是共享底座，两个network会互相影响学习。当reward normalize做的不好的时候（比如过度稀疏，比如variance较大），value network学习会占主导，影响policy的学习。当降低value network的loss占比的时候，value network又很难学好，那么token-wise reward学得很差。根据[PPO](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=PPO&zhida_source=entity)的GAE:

Advantage(T)=∑t≥Tγt−Trt−Value(T)

那么Advantage函数会学的很不稳定，那么会影响整个policy的学习，因此也学习不到好的policy。这样可能在某些简单场景下效果还行，但复杂场景下是不行的。

附：为什么在传统RL中这一套共享参数是make sense的，但在LLM领域却是不好的呢？

答：关键传统RL中一般是learn from zero，那么value network和policy network都是从头学的，所以影响不太大。但LLM是先模仿学习出policy，然后试图用rm来纠正这个policy。那么这个policy初始化的value network带来的bias太大，让rm很难纠正。（ps，还有就是底座模型太大，但value头太小了，就一层[mlp](https://zhida.zhihu.com/search?content_id=243647663&content_type=Article&match_order=1&q=mlp&zhida_source=entity)，这个bias也是确实太大）。

附：改进方案？

实在想共享参数，那就多加几层MLP在policy network上构造value network。

[1] [Reinforcement Learning From Human Feedback](https://link.zhihu.com/?target=https%3A//newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)

## 为什么Position Embedding和ROPE中不同维度需要设置不同的三角函数？

[Site Unreachable](https://zhuanlan.zhihu.com/p/699926500)

## Deepseek math中GRPO [1]和[DPO](https://zhida.zhihu.com/search?content_id=243921181&content_type=Article&match_order=1&q=DPO&zhida_source=entity)，PPO的关系？

**回答：**GRPO执行图如下：

![](https://picx.zhimg.com/v2-a821614abbb7887dee42e5010cf240ad_1440w.jpg)

他们相对于PPO有两个比较重要的改变

1. 抛弃了PPO的[GAE](https://zhida.zhihu.com/search?content_id=243921181&content_type=Article&match_order=1&q=GAE&zhida_source=entity)和value-network
2. 选择对于一个Q采样多次，且在这批次内进行reward normalize

对于第一次操作主要目的是为了降低显存占用，这个目的很充分，尤其在很大的模型下，省下不少资源。但带来的弊端就是缺少了token-wise reward的预估，且抛弃了GAE这种TD & MC learning的方式，采取了纯[蒙特卡洛](https://zhida.zhihu.com/search?content_id=243921181&content_type=Article&match_order=1&q=%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B&zhida_source=entity)采样的方式。那么势必会一定带来很大的gradient预估的variance。那他们后续采用的方式，是在同一个query state s_0下多次采样来降低预估的variance。(这样有点像DPO的方式，DPO降低variance的方式本质也是在同一个query state下进行两次采样进行对比学习)。

从传统RL方向看：GRPO可以对应于Reinforce-baseline-meta learning算法。特殊性在于它把游戏按照query state划分成了多个子游戏，用一个policy分别在不同子游戏内做Reinforce-baseline算法，相当于[meta learning](https://zhida.zhihu.com/search?content_id=243921181&content_type=Article&match_order=2&q=meta+learning&zhida_source=entity)版本的Reinforce-baseline算法。

整体在reward model打的reward比较准确下，比DPO采样效率更高，variance更低。比PPO采样效率略低，但节省内存。

附录：给我们一个PPO的改进的方向，**也就是在不同query-state下做PPO-[meta-learning](https://zhida.zhihu.com/search?content_id=243921181&content_type=Article&match_order=1&q=meta-learning&zhida_source=entity)**。

[1] Shao Z, Wang P, Zhu Q, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models[J]. arXiv preprint arXiv:2402.03300, 2024.

## 为什么长文本训练中需要[高频外推](https://zhida.zhihu.com/search?content_id=243964032&content_type=Article&match_order=1&q=%E9%AB%98%E9%A2%91%E5%A4%96%E6%8E%A8&zhida_source=entity)，低频内插？

[Site Unreachable](https://zhuanlan.zhihu.com/p/701251673)

## 稳定PPO训练的trick有哪些？

1. reward normalize：使用历史获得过的所有 reward 的均值和[方差](https://zhida.zhihu.com/search?content_id=244413677&content_type=Article&match_order=1&q=%E6%96%B9%E5%B7%AE&zhida_source=entity)进行标准化 [1]。
2. token KL penalty**：**限制模型更新方向 [1]。
3. Critic Model：使用 RM 初始化 Critic，并在 PPO 正式训练之前先进行 Critic 预训练 [1]。
4. Global Gradient Clipping [1]。
5. 使用相对较小的 Experience Buffer [1]。
6. **Pretrain Loss：**在 PPO 训练 loss 中加入 Pretrain Language Model Loss [1]。
7. 按照各个task对不同reward 进行normalize [2]。
8. 训练[reward model](https://zhida.zhihu.com/search?content_id=244413677&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)的时候，加上L2 normalize [2]。

  
[1] [何枝：【RLHF】怎样让 PPO 训练更稳定？早期人类征服 RLHF 的驯化经验](https://zhuanlan.zhihu.com/p/666455333)

[2] ChatGLM-RLHF: Practices of Aligning Large Language Models with Human Feedback

## 如何在RLHF中做state探索？

在传统RL中，state探索是一个很重要的方向，因为模型永远无法预估好完全没见过的state的value。当然对于和原始state分布相近的state，模型可以泛化出来。但当遇到的state分布和曾经见过的state分布相差越来越远的时候，这个评估会越来越不准确，具体理论可以看 [1]。

在RLHF中，我们首先收集pair数据进行[reward model](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=reward+model&zhida_source=entity)训练，然后用reward model对[LLM](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=LLM&zhida_source=entity)进行RL训练。当LLM刚开始训练的时候，生成的response（state），reward model是见过的。但当训练过久以后，生成的response逐渐偏离原始policy的response分布。那么这里面有一部分是准确的（离原始response分布相近），另一部分则是不准确的（离原始response分布相远）。如下图：

![](https://picx.zhimg.com/v2-e1feebbb4e57f07274bceb0b1ebbd32f_1440w.jpg)

在[LLama2](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=LLama2&zhida_source=entity) [2]中, 使用边训边标的方式来解决这个问题，也就是训一段时间，收集数据进行标注。但这种标注方法太过于耗费人力。尤其是标注的某些部分（离原始response分布相近的state）本身reward是依然准确的，标注所以是被浪费了。

因此，为了减少人力浪费，用最节省的方式进行标注，论文 [3]提出一个[active learning](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=active+learning&zhida_source=entity)的方式完成探索潜在高分state。具体来说，他们在[DPO loss](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=DPO+loss&zhida_source=entity)后新加了一项探索loss：

![](https://pic1.zhimg.com/v2-313412c74f2b11f22755bceb52f389a6_1440w.jpg)

这里的第二项的目的是让模型生成更多sft model中生成不了的response。

Tips：这里和降低[KL散度](https://zhida.zhihu.com/search?content_id=244628532&content_type=Article&match_order=1&q=KL%E6%95%A3%E5%BA%A6&zhida_source=entity)的weight区别是，降低KL散度weight能让sft model生成很多高得分却偏离sft model的response，但是不能保留那些低得分但偏离sft model的response（也就是上图中y_u右边部分）。

对于新的loss，模型可以采样出更多偏离reward response原始分布的response，提高标注的sample efficiency，降低成本。

[1] Liu Z, Lu M, Xiong W, et al. Maximize to explore: One objective function fusing estimation, planning, and exploration[J]. Advances in Neural Information Processing Systems, 2024, 36.

[2] Touvron H, Martin L, Stone K, et al. Llama 2: Open foundation and fine-tuned chat models[J]. arXiv preprint arXiv:2307.09288, 2023.

[3] Zhang S, Yu D, Sharma H, et al. Self-Exploring Language Models: Active Preference Elicitation for Online Alignment[J]. arXiv preprint arXiv:2405.19332, 2024.

## 为什么[SAC](https://zhida.zhihu.com/search?content_id=245118822&content_type=Article&match_order=1&q=SAC&zhida_source=entity) [1]算法在RL届和[PPO](https://zhida.zhihu.com/search?content_id=245118822&content_type=Article&match_order=1&q=PPO&zhida_source=entity)平分秋色，甚至于略胜一筹，而在[LLM](https://zhida.zhihu.com/search?content_id=245118822&content_type=Article&match_order=1&q=LLM&zhida_source=entity)届却无人问津？

[Site Unreachable](https://zhuanlan.zhihu.com/p/706444920)

## 同等[MOE模型](https://zhida.zhihu.com/search?content_id=245318723&content_type=Article&match_order=1&q=MOE%E6%A8%A1%E5%9E%8B&zhida_source=entity)的loss能下降到和同等规模Dense模型的水准吗？

这是显然不能的，因为MOE在训练中每个token forward和backward的实际的激活参数是远少于同等规模的Dense 模型的（Btw，尽管Dense模型训练完也是个偏向sparse的模型，也就是有少量神经元被激活，但是在训练中，Dense模型是可以自由选择激活哪部分神经元的。而Sparse Moe，通过训练路由来控制哪个token激活哪部分的expert，本质差距还蛮远的。）。那么从DeepseekV2-MOE-236B [1]来看，激活21B，总参 236B，等效一个 90B 的Dense，从Deepseek-Coder-MOE-16B [1]，激活2.4B，总参数16B，等效于一个7B模型。（等效计算是和激活参数，总参数都挂钩的函数计算出来的。）

最后推荐一下[deepmind](https://zhida.zhihu.com/search?content_id=245318723&content_type=Article&match_order=1&q=deepmind&zhida_source=entity)的moe scaling law，这个是群内skywork的小伙伴推荐的。

[1] Zhu Q, Guo D, Shao Z, et al. [DeepSeek-Coder-V2](https://zhida.zhihu.com/search?content_id=245318723&content_type=Article&match_order=1&q=DeepSeek-Coder-V2&zhida_source=entity): Breaking the Barrier of Closed-Source Models in Code Intelligence[J]. arXiv preprint arXiv:2406.11931, 2024.

[2] Clark A, de Las Casas D, Guy A, et al. Unified scaling laws for routed language models[C]//International conference on machine learning. PMLR, 2022: 4057-4086.

## 以下是两个[RLHF算法](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=RLHF%E7%AE%97%E6%B3%95&zhida_source=entity)不同种decoding的结果：

RLHF算法一：

1）抽样10次，top_p = 0.7, temperature = 0.95, [pass@1](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=pass%401&zhida_source=entity) = 0.7

1) Greedy decoding, pass@1 = 0.78

RLHF算法二：

1）抽样10次，top_p = 0.7, temperature = 0.95, pass@1 = 0.72

2) Greedy decoding, pass@1 = 0.75

请问算法一和算法二哪个是[PPO算法](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=PPO%E7%AE%97%E6%B3%95&zhida_source=entity)，哪个是[DPO算法](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=DPO%E7%AE%97%E6%B3%95&zhida_source=entity)？假设RM噪声较小。

**回答：**这是我真实训练发现的一个现象，这里实际中算法一是PPO，算法二是DPO算法。其中原因是：

1）当使用DPO算法时，正确的response在被maximize，错误的responses在被minimize，但是由于DPO不能分辩token-wise的reward，那么虽然整体正确的response概率在增大，但某几个在正确response中关键的token未必能增大，甚至到达top 1。所以greedy的效果可能没有那么好。但随机pass@1的效果会还不错（因为整体的概率提升了）。

2）当使用PPO算法的时候，由于使用的是[weighted logistics regression](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=weighted+logistics+regression&zhida_source=entity)，那么在token维度，如果这个token可以获得的未来reward大于现在的state value，那么这个token就会被增强，而reward 得分最高的token会被最大化增强。因此Greedy的效果一定会很快的提升，但依然有部分和Greedy很像的response也被增强了，因为reward model使用的是[BT model](https://zhida.zhihu.com/search?content_id=245473544&content_type=Article&match_order=1&q=BT+model&zhida_source=entity)，很多错误但和正确很像的response的得分也很高。所以pass@1反而没那么高。

评论区有不同见解，[Site Unreachable](https://zhuanlan.zhihu.com/p/708037980)



## 参考资料

百面LLM： https://www.zhihu.com/column/c_1747590116120698880 

LLM常见面试问题- SFT篇 - 技术微佬的文章 - 知乎
https://zhuanlan.zhihu.com/p/714687583

[什么是Cosine优化器？在大模型中应该怎么设置cosine优化器的周期比较好？](https://zhuanlan.zhihu.com/p/685354437)



