---
title: DPOP
created: 2024-06-15
tags:
  - 关键词
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

## 论文基本信息

标题：Fixing Failure Modes of Preference Optimisation with DPO-Positive

作者：

链接：

代码：

框架图：


## 背景

DPO 有一个非常致命的问题，

由于 DPO 的训练 loss 目标是「尽可能最大化好答案和坏答案之间的采样概率差」，

一种常见的情况是：**好答案 & 坏答案被采样的概率同时在变低，只不过坏答案降低的比好答案更多**。

这样一来，虽然好坏答案之间的概率差变大了，但这个过程中「好答案」被采样的概率也降低了，

这并不是我们想要的！

这种情况在 **chosen 和 rejected 答案有大部分内容相同，仅有少部分内容不同时较为常见**。

为此，[[DPOP](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.13228)] 在 DPO loss 的基础上加入了一个正则项：

- 若当前 chosen 答案在 SFT 模型中采样概率 > 当前 Policy 模型的采样概率，则减去一个正则化系数（当前的 chosen 答案 policy 还没有拟好，别再更新那么猛了）；
- 若当前 chosen 答案在 Policy 模型中采样概率更高，证明 Policy 已经对这个 chosen 答案拟合的比较充分了，此时着重降低一下坏答案的采样概率。

![](img/Pasted%20image%2020240615170704.png)

使用这种方法，相当于在「好答案」和「坏答案」中添加了一个截断式的 “attention”，让模型优先学会 chosen 答案，当对好答案学的足够好时再着重考虑惩罚坏答案，从而降低 DPO 模型 “训崩” 的可能性，最起码也要不弱于单拿 chosen 数据出来做 SFT 的效果。

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
