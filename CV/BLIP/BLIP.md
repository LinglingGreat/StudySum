---
title: BLIP
created: 2023-02-05
tags: 多模态
type: 论文
papername: BLIP Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
---

## 论文基本信息

标题：BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

链接： https://arxiv.org/abs/2201.12086

代码：

核心一是使用了Unified方式同时建模了多个Vision-Language任务，二是使用Bootstrapping策略优化了网上爬取的Noisy的图文pair数据

### 研究动机

- 模型角度：encoder结构不能直接做文本生成；encoder decoder结构不能很好地应用在图文检索中
- 数据角度：目前的模型都是在大量的图文数据集上训练的。这种noisy的数据用来训练是suboptimal的。

![](img/Pasted%20image%2020230205170415.png)

### 模型结构

![](img/Pasted%20image%2020230205170650.png)

除了Decoder部分的其它部分其实跟ALBEF的结构是一样的，但是借鉴了VLMo的共享参数有一些共享层。（同样颜色的是共享参数）

Decoder部分只增加了Causal Self-Attention层，其它层是共享的，所以增加的参数不多。


![](img/Pasted%20image%2020230205171417.png)

### 实验

![](img/Pasted%20image%2020230205171801.png)


LAION COCO数据集用了BLIP

## 核心亮点

## 主要收获


