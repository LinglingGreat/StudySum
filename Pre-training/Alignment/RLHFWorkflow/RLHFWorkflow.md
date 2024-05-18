---
title: RLHFWorkflow
created: 2024-05-18
tags:
  - rlhf
type: 论文
papername: RLHF Workflow- From Reward Modeling to Online RLHF
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - Salesforce Research
---

## 论文基本信息

标题：RLHF Workflow: From Reward Modeling to Online RLHF

作者：

链接：

代码：

框架图：

训练代码也开源了，大家可以用自己收集的pairwise preference data训练preference model：https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/pair-pm

我们Iterative DPO的代码在：https://github.com/RLHFlow/Online-RLHF

我们训练用的所有数据都放在了：https://huggingface.co/RLHFlow

## 背景



## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点
一个overview+reproduction paper

reward modeling部分是第一个开源复现了Google去年Slic (https://arxiv.org/abs/2305.10425)里面提出的preference model，然后发现它还真的比大家最常用的Bradley-Terry reward model (InstructGPT / Claude / Llama2都用的这种）要有一些优势，主要在对于Reasoning (Math/Coding)数据的判断上。

online RLHF部分是展示了一种比较简洁有效的Iterative DPO recipe。从一月份Snornel那边放出Iterative DPO训练的模型以来（但没有开源代码），这几个月RLHF领域对Iteartive DPO非常感兴趣。论文做了很多实验，找到了一个很work的recipe，也就是下图展现的这个

![](img/Pasted%20image%2020240518114130.png)

iterative DPO没有用这个Slic paiwise preference model，是因为它只能做pairwise comparison；而我们Iteartive DPO里面对于每个prompt，需要对8个responses做reward ranking -- 用Bradley-Terry reward model的话比这个pairwise preference model要快很多，所以我们选择了前者。但大家如果只需要做pairwise comparison的话，不妨试试这个preference model。

这个Pairwise Preference Model模型在这里： https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B
现在是RewardBench榜单第二名（开源模型里第一名）


为什么Pairwise Preference Model对于Math/Coding的判断更准确。我们的直观理解是，对于math/coding，你给模型一个response A，模型可能觉得还不错（比如打个90分），但你再给模型一个response B，它可能立马能看出来原来A是有bug的（不应该给高分）。
如果用Bradley-Terry reward model的话，它每次只能看到一个response然后就要打分，很难打分打的精准。但是Pairwise preference model 的话能同时看到一对回答，就比较容易判断出哪个更好。请看下图

![](img/Pasted%20image%2020240518114335.png)



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
[仅靠开源数据复刻出LLaMA3指令学习效果，在线迭代RLHF全流程解决方案来了](https://mp.weixin.qq.com/s/bRxdSCCPIrgNBgtDfyzhAA)

LLM微信群里作者的分享讨论

