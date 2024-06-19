---
title: Nemotron-4
created: 2024-06-18
tags:
  - 大模型
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

具体来说，Nemotron-4 340B包括基础模型Base、指令模型Instruct和奖励模型Reward，并构建了一个高质量合成数据生成的完整流程。SFT+RLHF+DPO

模型支持4K上下文窗口、50多种自然语言和40多种编程语言，训练数据截止到2023年6月。

训练数据方面，英伟达采用了高达9万亿个token。其中，8万亿用于预训练，1万亿用于继续训练以提高质量。

值得一提的是，指令模型的训练是在98%的合成数据上完成的。

![](img/Pasted%20image%2020240618200812.png)

结果显示，Nemotron-4-340B-Base在常识推理任务，如ARC-Challenge、MMLU和BigBench Hard基准测试中，可以和Llama-3 70B、Mixtral 8x22B和Qwen-2 72B模型媲美。

而Nemotron-4-340B-Instruct，在指令跟随和聊天能力方面也超越了相应的指令模型。

Nemotron-4-340B-Reward在发表时，在RewardBench上实现了最高准确性，甚至超过了GPT-4o-0513和Gemini 1.5 Pro-0514这样的专有模型。




## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料

[英伟达开源3400亿巨兽，98%合成数据训出最强开源通用模型！性能对标GPT-4o](https://mp.weixin.qq.com/s/Q3CxTWPR1-_GEBbuOTcTjA)

