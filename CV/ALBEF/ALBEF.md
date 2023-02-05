---
title: ALBEF
created: 2023-02-05
tags: 关键词
type: 论文
papername: Align before Fuse Vision and Language Representation Learning with Momentum Distillation
---

## 论文基本信息

标题：Align before Fuse: Vision and Language Representation Learning with Momentum Distillation

链接： https://arxiv.org/abs/2107.07651

代码：

框架图：


salesforce团队做的，他们还做了很多很好的工作，比如BLIP

以前的工作中，视觉特征和文本特征没有做对齐，这样使得多模态融合的时候很难学习他们之间的交互。

这篇论文就用了对比学习来对齐视觉特征和文本特征。

为了从网上的噪声数据（具备搜索属性，但不是很好的描述图像，可能只是一些关键词）学习，提出了momentum distillation。从伪标签中学习的self-learning自训练方法。

图文检索上，ALBEF超过了其它模型。在VQA和NLVR任务上也比很多模型强。

## 核心亮点

### 模型结构

![](img/Pasted%20image%2020230205153825.png)

图像部分就是标准的vision transformer，文本部分分成2部分，BERT前6层作为文本编码器，另外6层作为multimodal encoder

momentum model是基础模型（编码部分的ViT和BERT）的moving average，跟MoCo模型的做法是一样的。

### 损失函数

ITM loss：Image Text Matching。选择同一个batch中跟图像最相似的文本作为负样本（hard negatives）。

MLM loss：完形填空，也借助了图像信息去还原mask

模型做了两次forward，因为MLM loss需要用到mask后的文本，ITM则是用到原始文本。

### Momentum Distillation

数据是noisy的。导致：

- ITC学习中，负样本可能也符合图片内容
- MLM学习中，其它文本也可能很好的描述图像

构建一个momentum model来生成伪标签。momentum model是一个持续进化的教师模型，包括单模态和多模态编码器的EMA（exponential-moving-average）版本。在训练中，基础模型不仅要跟ground truth尽可能match，还要跟momentum model生成的伪标签尽可能match。希望momentum model可以在noisy的one-hot label上做一些改进。

![](img/Pasted%20image%2020230205160110.png)

![](img/Pasted%20image%2020230205160125.png)

ITM本身就是二分类，没有动量版本。

![](img/Pasted%20image%2020230205160322.png)

### 实验
实验中两类数据集：4M和14.1M的

任务：图文检索（TR），视觉蕴含（VE），视觉问答（VQA），视觉推理（NLVR），Visual Grounding

![](img/Pasted%20image%2020230205161520.png)

加了ITC的提升非常大。Momentum Distillation的提升没有那么大。




## 主要收获

对比学习YYDS！



## 个人评价





