---
title: ChatHome
created: 2023-09-03
tags:
  - 关键词
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
论文试图解决什么问题？这是否是一个新的问题？

这篇文章要验证一个什么科学假设？

论文中提到的解决方案之关键是什么？


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点



## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？


1. 基于baichuan-13B base 预训练模型做fine-tune时， 领域数据：通用数据配比是1:10时在领域指标上最好
2. 基于baichuan-13B base 上继续做预训练（不用通用领域数据）时，领域数据：通用数据配比是1:5时在领域指标上最好
3. 基于baichuan-13B base 上继续做预训练（领域数据：通用数据配比是1:5）时，领域数据：通用数据配比是1:5时在领域指标上最好
4. 基于baichuan-13B chat（做过多轮对话和指令微调），领域数据：通用数据配比是1:5时在领域指标上最好
5. 最后一行，基于baichuan-13B base 上做预训练和SFT finetune时（无通用数据），效果最好。【缺少了基于baichuan-13B base 上做预训练和SFT finetune时（领域数据：通用数据配比是1:5）的对比】

## 未来方向



## 主要收获


## 参考资料