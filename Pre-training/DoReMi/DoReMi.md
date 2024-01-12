---
title: DoReMi
created: 2024-01-12
tags:
  - 数据配比
type: 论文
papername: DoReMi-Optimizing Data Mixtures Speeds Up Language Model Pretraining
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution:
  - 斯坦福
  - DeepMind
---

## 论文基本信息

标题：DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

作者：Sang Michael Xie∗1,2, Hieu Pham1, Xuanyi Dong1, Nan Du1, Hanxiao Liu1, Yifeng Lu1, Percy Liang2, Quoc V. Le1, Tengyu Ma2, and Adams Wei Yu

链接： http://arxiv.org/abs/2305.10429

代码：

框架图：


## 背景

预训练数据域（例如维基百科、书籍、网络文本）的混合比例极大地影响语言模型（LM）的性能。在本文中，我们提出了采用最小最大优化的域重新加权（DoReMi），它首先使用域上的组分布鲁棒优化（Group DRO）来训练一个小型代理模型，以在不了解下游任务的情况下产生域权重（混合比例）。然后，我们使用这些域权重对数据集进行重新采样，并训练一个更大的全尺寸模型。在我们的实验中，我们在 280M 参数代理模型上使用 DoReMi 来查找域权重，以更有效地训练 8B 参数模型（大 30 倍）。在 The Pile 上，DoReMi 改善了所有领域的复杂性，即使它降低了某个领域的权重。与使用 The Pile 默认域权重训练的基线模型相比，DoReMi 将平均少样本下游准确率提高了 6.5%，并以减少 2.6 倍的训练步骤达到基线准确率。在 GLaM 数据集上，不了解下游任务的 DoReMi 甚至可以与使用针对下游任务调整的域权重的性能相匹配。


## 相关研究

Pile使用启发式选择的域权重

PaLM和 GLaM根据一组下游任务调整域权重，但需要在不同域上训练潜在的数千个 LM权重和风险过度拟合特定的下游任务集。

Example级过滤也为 LM 训练带来好处。 C4 数据集显示，通过启发式数据清理方法比 CommonCrawl 取得了进步。GLaM(2021)和Data selection for language models via importance resampling（2023）表明，在Example级别过滤数据以获得类似于维基百科和书籍的高质量文本可以显着提高 LM 的下游性能。与这些工作相比，DoReMi 仅通过 2 次小型 LM 训练运行自动设置域权重，并且不对首选数据类型做出假设（类似维基百科等）。

## 核心亮点

![](img/Pasted%20image%2020240112165050.png)

首先，DoReMi以标准方式（比如均匀采样所有领域数据）训练一个小型参考模型（例如280M参数）。

其次，我们训练一个小型分布式鲁棒语言模型（DRO-LM）（Oren et al., 2019），它可以最大限度地减少所有领域最坏情况的额外损失（相对于参考模型的损失）。损失越大，对应的领域权重就越高。

![](img/Pasted%20image%2020240112171502.png)

![](img/Pasted%20image%2020240112172106.png)

最后，我们没有使用鲁棒的 LM，而是采用 DRO 训练生成的域权重。我们在由这些域权重定义的新数据集上训练大型 (8B) LM。

![](img/Pasted%20image%2020240112172300.png)

![](img/Pasted%20image%2020240112172805.png)


## 实验

![](img/Pasted%20image%2020240112174010.png)

这里GLaM根据domain数据在小模型上训练后下游任务的效果决定领域的权重

域重新加权对 GLaM 的影响较小，可能是因为与 The Pile 中的 22 个域相比，只有 8 个域。

![](img/Pasted%20image%2020240112174731.png)



![](img/Pasted%20image%2020240112174120.png)

![](img/Pasted%20image%2020240112174144.png)

直观上，熵最低和最高的域可以被降低权重，而不会太大影响困惑度。从统计上看，最低熵域需要很少的样本来学习。最高熵域的令牌分布接近常见的统一先验——例如，随机初始化的模型倾向于输出统一的下一个令牌分布。因此，我们需要更少的样本来适应这些领域。

![](img/Pasted%20image%2020240112174952.png)


proxy模型和main model是相同size的时候

![](img/Pasted%20image%2020240112175356.png)

使用不同的proxy模型，对main model的效果的影响。一般来说，proxy越大效果越好，但是1B大小的proxy模型表现不佳，可能是因为Group DRO优化器在大的proxy模型下表现较差。


![](img/Pasted%20image%2020240112175730.png)

proxy模型和main model是相同size的时候，proxy模型和main model哪个表现更好？main model（图中(b)的DoReMi比Proxy好）。

![](img/Pasted%20image%2020240112175956.png)


## 未来方向



## 主要收获


## 参考资料
