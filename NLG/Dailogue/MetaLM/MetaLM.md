---
title: MetaLM
created: 2022-11-08
tags: 对话 多模态
type: 论文
---

## 论文基本信息

## 核心亮点

## 主要收获

## 个人评价

微软的MetaLM是一篇主打交互的工作，支持用语言模型作为交互接口，去调动其他模型执行各种任务

![](img/Pasted%20image%2020221108223744.png)

考虑到单向LM更通用、双向LM效果更好，作者把两个做了结合：

1.  最上层的绿色模型是单向，更general，支持多种任务的执行
    
2.  下面可以接多个蓝色的双向模型，给图片、语音等数据编码

对于文本预训练，主要做单向LM，同时随机选择一些span进行双向编码

![](img/Pasted%20image%2020221108223830.png)

对于图像预训练，直接选用了一些text-image数据进行预训练，这里其实也可以参考Flamingo的做法

![](img/Pasted%20image%2020221108223902.png)
