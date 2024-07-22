---
title: AutoCoT
created: 2023-06-04
tags: cot
type: 论文
papername: Automatic Chain of Thought Prompting in Large Language Models
conference: ICLR
year: 2023
institution: Amazon, SJTU
---

## 论文基本信息

标题： **Automatic Chain of Thought Prompting in Large Language Models，ICLR2023**

作者：Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola（上交，亚马逊）

链接：

代码：

框架图：


## 背景

我们有**一大堆的待测试的问题**（没有标注，不知道正确答案和推理过程），我们要**怎么利用 LLM 和这么一个无标注问题集合，在不进行手工编写CoT的情况下，提升LLM回答这些模型的质量。**

能不能**利用 Zero-shot CoT 来让 LLM 产生很多带有推理的QA pair，然后把这些QA pair加入到prompt中，构成ICL的上文，再让LLM进行推理。**


作者的基本思路是这样的：

- 给定待测试的问题q，从无标注问题集合中，**采样**一批问题；
    
- 使用 GPT-3 作为产生推理过程的工具，即直接使用 “Let's think step by step.” 咒语，来对这一批采样的问题产生推理过程；
    
- 把产生的这些问题和推理过程，构成In-Context-Learning的上文加入到prompt中，再让LLM对问题q进行回答。
    

关键就在于这个**采样**过程，作者分别先测试了两种简单的采样过程：

1. 随机采样，Random-Q-CoT
    
2. 基于跟待测试q的相似度进行采样，Retrieval-Q-CoT
    

实验发现，居然随机采样还要更好一些。经过探究，作者发现GPT-3自动产生推理过程是有一定比例出错的，而**出错的问题也容易聚集**

因此基于相似度搜索的时候，容易导致采样出一批错误的示范，而随机采样的方法，则可能避免聚集性地出错。

基于这样的考虑，作者设计了基于多样性的采样方法，先试用SentenceBERT对所有问题进行聚类，然后从每个cluster中进行采样。

具体采样过程则是：

- 假设需要再In-Context-Learning的时候加入k个示例，则对问题集合聚k类
    
- 对于每个cluster中的问题，按照每个问题跟cluster中心点的相似度来排序
    
- 每个cluster采样一个问题，距离中心点越近的越优先采样，但是得符合一些规则：
    
	- 问题不超过60个token，通过Zero-shot-CoT产生的推理步骤不超过5步（也是前人的经验，简单原则）

利用现有的有zero-shot CoT能力的模型生成k个A。然后直接用这个合成QA对去做few-shot。

需要注意的是，哪怕合成的QA对是不对的（比如模型回答错了数学问题），实测也可以涨点，作者认为1.加长了模型的思考过程2.模型推理的形式更有助于它自己理解，从而生成更有用的fewshot形式。

![](img/Pasted%20image%2020230604151605.png)

Auto的方法居然可以比Manual更好。其实有一种解释，Manual方法其实给多个任务都使用的是同一套模板，比方6个数学任务里面5个都使用的同一套示例（为了省力，同时Manual-CoT的论文也不是为了刷榜，而是为了揭示这么一个现象，所以CoT没有进行仔细调优），而Auto-CoT则是每个任务都会有自己的一套示例产生，毕竟问题集合不一样，聚类的结果也会不一样。

### 应用

![](img/Pasted%20image%2020230227225711.png)

![](img/Pasted%20image%2020230227225952.png)

