---
title: In-Context Learning
created: 2023-01-09
tags: In-Context 大模型
type: 论文
papername: A Survey for In-context Learning
---

## 论文基本信息

## 核心亮点

## 主要收获

## 个人评价


**In-Context Learning（ICL）** **指在不进行参数更新的情况下，只在输入中加入几个示例就能让模型进行学习**

综述：

题目：A Survey for In-context Learning  
链接：https://arxiv.org/abs/2301.00234

![](img/Pasted%20image%2020230109233302.png)

# 通过精调优化ICL效果

## 有监督ICL训练

MetaICL[1]直接把很多任务整合成了ICL的形式精调模型，在52个数据集上取得了比肩直接精调的效果。

另外还有部分研究专注于Instruction tuning，构建更好的任务描述让模型去理解，而不是只给几个例子（demonstration），比如LaMDA-PT[2]、FLAN[3]。

## 自监督ICL训练

有监督的数据毕竟是有限的.

MetaAI的一篇工作[4]就很巧妙地把语言模型的一些任务转化成了ICL形式：

![](img/Pasted%20image%2020230109233510.png)

从上述两种训练方法来看，**语言模型的ICL能力还有不少提升空间，因此建议大家使用ICL前最好先进行模型预热**。不过也不用太多数据，因为上述的研究显示，**随着预热的数据增多，模型的提升会到达一个瓶颈**。

# 在推理阶段优化ICL效果

## Prompt设计

可以从**组织方式和格式来进行Prompt的设计。组织方式是指如何选择数据样本并排序，格式是指怎么去写Prompt**。

对于数据样本的选取，可以有以下方法：

-   无监督：比如直接通过文本表示、互信息选取相近的结果；也有研究通过perplexity或者其他指标进行选取；甚至可以直接让语言模型自己生成[5]。
    
-   有监督：既然选取不同的样本能得到不同的效果，那可以直接构造监督模型，去判别效果更好的样本；甚至有研究把样本选择建模成序列决策任务，把最终效果当作reward，用强化学习去做[6]。
    

对于数据样本的排序，目前的研究并不多，有两个思路：

1.  基于一些距离度量，把跟输入相近的排在后面（靠近输入）。
    
2.  在Lu等人[7]的研究中，他们找到了信息熵和ICL效果的联系，因此根据熵来决定最佳排序。
    

对于Prompt的格式，常见有两种：指令（Instruction）和推理步骤（Reasoning Steps）说明。

-   Instruction：任务的指令描述非常依赖人工，不过也可以尝试让语言模型自动生成描述并选择。
    
-   Reasoning Steps：对于更复杂的任务，可以人工显示地把推理步骤写出来，比如Chain-of-thought（CoT），来启发模型的推理能力。除了纯人工撰写外，还有以下方法：
	-   让模型自己生成推理步骤
	-   Multi-stage ICL：分多个步骤来完成任务，每一步都设计不同的子问题，让模型一步步解答。比如Self-Ask[8]这篇工作甚至让模型自己问自己。再比如Least-to-Most Prompting这篇工作先让模型把大问题拆成多个子问题，再挨个回答。
    

从上述Prompt设计的工作来看，作者认为有以下可以探索或注意的点：

1.  目前大多数样例选择的策略都是围绕单个样本展开的，但语料级别的选择还没有人研究过。
    
2.  对于`k`个样本，搜索空间是`k!`，怎样找到最优解还是一个很有挑战的问题
    
3.  虽然增加CoT可以提升推理效果，但怎样优化CoT还有待研究
    
4.  人工去写prompt的消耗还是很大的，可以尽量依靠语言模型去生成

## 打分函数（Scoring Function）

目前有几种方法：

1.  Direct[9]：直接取答案的条件概率，这种方法的缺点是只能衡量固定模式的答案（答案y在输入x后面）
    
2.  Perplexity：再用语言模型过一遍句子，这种方法可以解决上述固定模式的问题，但计算量增加了
    
3.  Channel[10]：评估`P(x|y)`的条件概率（用贝叶斯推一下），这种方法在不平衡数据下表现较好
    

这三种方法的对比如下：

![](img/Pasted%20image%2020230109233919.png)

目前关于如何用打分策略来校准偏差、降低模型敏感度的研究还是比较少。

# 还有什么会影响ICL表现？

除了上述提到的方法外，作者还调研到一些LM预训练阶段影响ICL效果的因素，比如：

1.  预训练语料的多样性比数量更重要，增加多种来源的数据可能会提升ICL表现[11]
    
2.  用下游任务的数据预训练不一定能提升ICL表现，并且PPL更低的模型也不一定表现更好[12]
    
3.  当LM到达一定规模的预训练步数、尺寸后，会涌现出ICL能力[13]，且ICL效果跟参数量正相关[14]
    

# 为什么ICL有效果？

目前的一些研究猜测有以下原因：

1.  **跟训练数据的分布相关**：比如训练数据有很多样例[15]，也有学者认为ICL可能是隐式的Bayesian inference[16]
    
2.  **跟学习机制相关**：有学者猜测LM可能自己就具备学习的能力，在做ICL的时候学到了这些知识[17]，或者隐式直接精调了自己[18]
    
3.  **跟Transformer中的模块相关**：有学者发现Transformer里的某些注意力头会通过拷贝固定的模式来预测下一个token[19]
    

上述大部分研究虽然都有数据来证实自己的猜想，但还是停留在很简单的任务或者小模型上，对于为什么ICL会有效还有待进一步的解释。

# 之后的研究方向

1.  新的预训练策略：毕竟目前LM的预训练目前跟ICL并不完全一致
    
2.  把ICL能力蒸馏到更小的模型
    
3.  在语言模型上进行知识增强和更新
    
4.  对于样例的鲁棒性

# 参考资料

[In-Context Learning玩法大全](https://mp.weixin.qq.com/s/NLWCuzcCdwljQfzu-Jd9lQ)

[1]

MetaICL: Learning to Learn In Context: _https://aclanthology.org/2022.naacl-main.201/_

[2]

LaMDA: Language Models for Dialog Applications: _https://arxiv.org/abs/2201.08239_

[3]

Finetuned Language Models are Zero-Shot Learners: _https://openreview.net/forum?id=gEZrGCozdqR_

[4]

Improving In-Context Few-Shot Learning via Self-Supervised Training: _https://aclanthology.org/2022.naacl-main.260/_

[5]

Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator: _https://arxiv.org/abs/2206.08082_

[6]

Active Example Selection for In-Context Learning: _https://arxiv.org/abs/2211.04486_

[7]

Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity: _https://aclanthology.org/2022.acl-long.556/_

[8]

Measuring and Narrowing the Compositionality Gap in Language Models: _https://doi.org/10.48550/arXiv.2210.03350_

[9]

Language models are few-shot learners.: _https://arxiv.org/abs/2005.14165_

[10]

Noisy Channel Language Model Prompting for Few-Shot Text Classification: _https://aclanthology.org/2022.acl-long.365/_

[11]

On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model: _https://aclanthology.org/2022.naacl-main.380/_

[12]

On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model: _https://aclanthology.org/2022.naacl-main.380/_

[13]

Emergent abilities of large language models.: _https://arxiv.org/abs/2206.07682_

[14]

Language models are few-shot learners.: _https://arxiv.org/abs/2005.14165_

[15]

Data distributional properties drive emergent in-context learning in transformers.: _https://arxiv.org/abs/2205.05055_

[16]

An explanation of in-context learning as implicit bayesian inference: _https://arxiv.org/abs/2111.02080_

[17]

What can transformers learn in-context? A case study of simple function classes.: _https://arxiv.org/abs/2208.01066_

[18]

Transformers learn in-context by gradient descent.: _https://arxiv.org/abs/2212.07677_

[19]

In-context learning and induction heads: _https://arxiv.org/abs/2209.11895_