---
title: FollowComplexInstruction
created: 2024-07-20
tags:
  - 指令遵循
type: 论文
papername: From Complex to Simple - Enhancing Multi-Constraint Complex Instruction Following Ability of Large Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 复旦
---

## 论文基本信息

标题：From Complex to Simple: Enhancing Multi-Constraint Complex Instruction Following Ability of Large Language Models

作者：Qianyu He, Jie Zeng, Qianxi He, Jiaqing Liang, Yanghua Xiao

链接：

代码： https://github.com/meowpass/FollowComplexInstruction  （数据和代码）（看了一条数据，chosen的回答也不对。。这里面的指令遵循都是长度、标点符号、markdown等规则式的。）

框架图：

![](img/Pasted%20image%2020240720160256.png)

![](img/Pasted%20image%2020240720160339.png)


## 背景

本文提出了三个问题： 1：什么样的数据能增强复杂指令约束的能力？ 2：如何得到这类数据？ 3：如何利用上述数据进行有效微调？


## 相关研究




## 核心亮点

1：什么样的数据能增强复杂指令约束的能力？

复杂约束的数据相比单一约束具有更好的效果。也就是数据的描述越详细、准确，效果越好。效果和约束的数量、模型大小都有关系，甚至可以推广到域外约束的组合。

![](img/Pasted%20image%2020240720161515.png)

其中Backbone是指原始预训练模型，Atom指使用单约束数据微调的模型，Comp是指使用多条合成约束微调的模型。可以看出多约束数据模型结果更好。（微调时候加入了shareGPT数据防止遗忘）

与Backbone相比，使用Atom数据（主要具有 1 个约束）进行训练通常会降低超过 1 个约束的指令的性能。此外，使用Comp数据（通常有 3 到 5 个约束）进行训练可以显着提高具有 1 到 3 个约束的指令的性能，但对于具有 4 到 5 个约束的指令，表现出较小的增强甚至下降。


2：如何得到这类数据？ 

![](img/Pasted%20image%2020240720161937.png)

Vanilla: 用backbone模型生成；Generation: GPT-3.5-turbo；Discrimination: 识别出Vanilla没有遵循指令的输出，然后我们通过使用 GPT-3.5-turbo 的约束来纠正 Vanilla 输出约束

高级LLM（Generation）的输出质量高于较弱LLM（Vanilla）的输出质量。然而，来自较弱LLM的输出然后由高级LLM细化（判别）显著优于由高级LLM直接生成的输出（生成）。我们认为这是因为指令（即约束）的微小变化会导致实质性的输出差异，基于判别的方法比基于生成的方法更好地捕捉到这一点。

> 大模型做判别的效果比生成效果更好！！！



3：如何利用上述数据进行有效微调？

引入了一种基于强化学习微调（RLFT）的方法，该方法利用正样本和负样本来改善复杂的指令跟踪。

方法：

1、为了获得组合数据，首先从三个广泛使用的预调优数据集收集种子指令。然后重写指令以合并多个约束。

2、提出了一种基于判别的方法来获得输出，首先利用LLaMA 2-13B-Chat（学生模型）来生成我们合成的复杂指令的果。然后利用Zhou等人（2023 a）的测试脚本来识别模型未能遵循的约束，因为这些约束是客观的，并且可以自动验证。最后采用先进的LLM（教师模型）GPT-3.5-turbo，逐一纠正失败的约束。

3、对于每个指令Ic，可以收集正样本集和负样本集。监督微调（SFT）仅利用成功满足复杂指令中指定的约束的正样本。直接偏好优化（DPO）可以应用于对偏好信息进行建模，我们另外整合SFT损失以约束πθ偏离优选的数据分布。

![](img/Pasted%20image%2020240720163059.png)

![](img/Pasted%20image%2020240720163046.png)

## 实验

![](img/Pasted%20image%2020240720163149.png)

其中**Ours-13B-Generation** 直接使用 GPT-3.5-turbo 生成输出，并通过监督微调 (SFT) 训练backbone模型。**Ours-13B-Discrimination** 通过backbone模型生成输出，然后使用 GPT-3.5-turbo 进行细化，并通过 SFT 训练backbone模型。 **Ours-13B-Contrastive** 利用 DPO 进行训练，对正样本和负样本进行建模。三种方法的backbone模型均为LLaMA2-13B-Chat，训练数据的说明相同；只有训练数据和训练范式的输出不同。

泛化

![](img/Pasted%20image%2020240720163309.png)

![](img/Pasted%20image%2020240720163336.png)



## 未来方向



## 主要收获


## 参考资料
