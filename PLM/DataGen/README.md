---
title: README
created: 2024-08-15
tags:
  - 数据合成
---
## 指令数据

Self-Instruct：**人工制定一个类目体系**，有175个种子任务，每个任务配一个指令和例子让模型进行合成：

![](img/Pasted%20image%2020240815153036.png)

Nemotron-4也采用了类似的做法，分了4个大任务：OpenQA、Writing、Closed QA、Math&Coding。对于每类任务，以多个主题词和关键词作为种子。

**但人工指定 or 统计出的任务仍会有一定局限性，因此更好的方法是让模型生成任务**。近期腾讯一篇工作Persona-Hub就非常巧妙，从Web文本中提取出10亿的人物描述，再让模型以不同人物的口吻去生成问题，多样性直接拉满。

**还有一种比较tricky的，直接诈出某个对齐后模型的训练数据**，比如Magpie，只输入前缀USER，让模型自己把问题生成出来：

![](img/Pasted%20image%2020240815153147.png)

**最后，Doc2Query也能生成多样性较高的问题**，但有一定局限性，更多用于知识类Query的合成。

**对于复杂度，做法跟多样性差不多，需要定义出多种约束标签，再不断去改写Prompt增加其复杂度**。比如很经典的WizardLM，定义了几个不同的改写操作：add constraints、deepening、concretizing、increase reasoning steps、 complicate input，让指令的复杂度不断增加。

![](img/Pasted%20image%2020240815153208.png)

**还有一种方法是量化模型生成结果复杂度，从而训一个能生成复杂Query的模型**。例如在安全领域的工作SEAS，就构造了一个对抗攻击的场景：

![](img/Pasted%20image%2020240815153234.png)

如上图，该框架分为三个阶段：

1. 初始化：用少量的数据，精调一个攻击模型R（负责生成有害问题）和一个目标模型T（最终要被优化安全能力的模型）
    
2. 攻击阶段：模型R生成多条有害样本，T针对每个问题生成多个答案，利用Llama guard作为打分器，如果攻击成功，则该Query作为模型R的chosen，否则为rejected。如果攻击失败，则该Response作为模型T的chosen，否做为rejected
    
3. 优化阶段：利用第二步构造的DPO数据迭代模型R和模型T
    

通过上面的2、3步骤多轮迭代，模型R生成的攻击问题越来越强，模型T的安全能力也随之提升

## 监督信号合成

相比Prompt合成，Pair的合成会更难一些，Prompt即使质量一般，只要用较好的模型生成Response和较强的基座，也能学得还行，但如果Pair不准确，RM/PPO或者DPO直接就学偏了。

**最直接的方法就是LLM-as-Judge，用模型打分或排序**。Meta的Self-Rewarding的工作非常赞且优雅：

![](img/Pasted%20image%2020240815153313.png)

直接用当前模型T生成答案->给自己打分->做成DPO数据训练模型T+1，形成了一个self-play的闭环。在模型生成能力不断提升的同时，判别能力（Pairwise acc）也在提升

但可以看到后期的acc增长放缓，不知道模型最远能跑多少轮。

**如果担心LLM-as-Judge的准确率不够高，可以使用约束更强的Instruction来提升模型判别的准确率**，比如阿里的工作AutoIF，就让模型自己写代码来验证结果的正确性：

![](img/Pasted%20image%2020240815153354.png)

**还有一种方法是想办法让模型做答案固定的任务**，接近传统游戏RL，比如腾讯的工作SPAG中，就制定了一个游戏规则：

1. 攻方需要让守方无意识地说出某个词
    
2. 守方需要推理出攻方的词

![](img/Pasted%20image%2020240815153426.png)

在这样的框架下，判断最终结果的难度极大降低，作者也利用这类数据提升了模型的逻辑推理能力。


## 参考资料

[Alignment下一站：合成数据](https://mp.weixin.qq.com/s/k_lpWa1FCnE6gj6qz4r7Jg)

