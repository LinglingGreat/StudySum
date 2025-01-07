---
title: Tulu
created: 2025-01-06
tags:
  - 论文
  - alignment
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## 背景



## 核心亮点

主要有四个步骤:

1. 构建多样，高质量的Prompt
2. 有监督微调
3. 偏好优化
4. 强化学习

整体来看还是在狠狠搞数据，把已有的公开数据，模型都利用了起来。第三四步对[DPO](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=DPO&zhida_source=entity)和RL的技术创新，更多还是资源不够的一种妥协。

### Prompt构建：

选取了几个公开的数据集，包括:WildChat, OpenAssistant, NoRobots, FLAN v2等，用以确保多样性。除此之外，为了确保一些特定能力，还用了OpenMathInstrct, Evol-CodeAlpaca等[数学代码](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E6%95%B0%E5%AD%A6%E4%BB%A3%E7%A0%81&zhida_source=entity)相关的数据集。 基本上这一步就是一个大大的[数据收集](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E6%94%B6%E9%9B%86&zhida_source=entity)工。质量好点的[开源数据](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE&zhida_source=entity)都给用上了。

除此之外还做了一些数据合成，借助的是所谓的persona-driven的方法，通过让[LLM](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=LLM&zhida_source=entity)来扮演不同角色从而生成不同的指令。

最后还做了Prompt的去污染，防止污染测试集。

### 有监督微调：

主要也是对已有公开数据的过滤清洗，以及对于没有response的数据，利用[GPT4o](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=GPT4o&zhida_source=entity)来进行合成。

![](https://picx.zhimg.com/v2-bb23336bd2d80587b167ad78901e3d29_1440w.jpg)

对SFT数据的消融，比较有趣的是Persona数据确实改善了对应技能如Math/Coding上的能力。(不过其实也没有很显著)

![](https://pic1.zhimg.com/v2-5b069139e246486b2eb4d1f2f51a9bea_1440w.jpg)

一个比较抽象的发现是: 不同的prompt template看起来对结果的影响还不小....

此阶段学习率5e-6，只需要训两轮，更多轮数不能带来提升。

### 偏好优化：

这块在技术上无甚可说，主要也是讲讲数据: 通过从一个model pool中选择模型来生成多样的回答(模仿Ultrafeedback)，并且也做了[on-policy](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=on-policy&zhida_source=entity)的数据采样，大致保持了on-policy和[off-policy](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=off-policy&zhida_source=entity)的数据量1:1。最后使用GPT4o来进行标注。(本GPT4o企业级用户表示: 这看起来感觉并不是一个很强的做法。。)

最后得出了几个结论:

1. [Prompts](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=Prompts&zhida_source=entity)的多样性大大影响了DPO的效果。(SFT，DPO的Scaling Law)
2. 只增加数量，但是Prompt的多样性不增加，其实模型效果是会退化的。
3. DPO阶段，复用SFT阶段的Prompt，会带来一定收益，但还是采用新的Prompt效果更佳。
4. PT-4o-2024-08-06是标注能力最强的模型 (用它标注的结果做DPO效果最佳，和Llama405b打平)。这里还把Llama 3.1 405b拿出来试了下，看来4o的参数效率还是很领先的。

本阶段最后选择的学习率是5e-7，并且只需要训练一轮。另外作者用的是Length-normalized DPO，顾名思义，给[对数概率](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E5%AF%B9%E6%95%B0%E6%A6%82%E7%8E%87&zhida_source=entity)和除了个权重。

### 强化学习：

这个地方的方法比较粗暴

![](https://pic4.zhimg.com/v2-420ceb691fc2ee3e98c252d8469b3bdd_1440w.jpg)

直接基于GroundTruth来判断答案是否正确，然后应用PPO来进行训练。

发现:

1. 这样可以直接在目标领域(比如数学)改善效果，其实这也是一个趋势了，[代码生成](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90&zhida_source=entity)，数学推理这些可自动验证的领域后面应该都会跟上。
2. Value Model最好从一个通用的[RM](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=RM&zhida_source=entity)上去初始化。
3. 用RM产生的分数反而会产生噪音。(所以在能用规则验证的地方，还是不要用模型了吧)

### 其它发现:

1. 在线的DPO没生效。
2. 拒绝采样也没怎么生效。

### 尾声:

本文看起来朴实无华，但对于整个后训练链路的探索，还是很有价值的。不管是RLVR，还是在线DPO的失败，其实应该都体现的是在本文中RM的失败，RM的[过拟合](https://zhida.zhihu.com/search?content_id=250740756&content_type=Article&match_order=1&q=%E8%BF%87%E6%8B%9F%E5%90%88&zhida_source=entity)，噪音，Hacking等问题，依然是值得大家去警惕的。

## 实验



## 未来方向



## 主要收获


## 参考资料

[TÜLU 3: 拒绝RM的后训练技术总结](https://zhuanlan.zhihu.com/p/8589852586)

