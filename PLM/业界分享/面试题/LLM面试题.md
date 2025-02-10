
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



## 参考资料

百面LLM： https://www.zhihu.com/column/c_1747590116120698880 

LLM常见面试问题- SFT篇 - 技术微佬的文章 - 知乎
https://zhuanlan.zhihu.com/p/714687583

[什么是Cosine优化器？在大模型中应该怎么设置cosine优化器的周期比较好？](https://zhuanlan.zhihu.com/p/685354437)

