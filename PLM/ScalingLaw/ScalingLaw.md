---
title: ScalingLaw
created: 2024-06-03
tags:
  - ScalingLaw
---
在大模型的研发过程中，通常会有下面一些需求

1. 计划训练一个10B的模型，想知道至少需要多大的数据？
2. 收集到了1T的数据，想知道能训练一个多大的模型？
3. 老板准备1个月后开发布会，给的资源是100张A100， 应该用多少数据训多大的模型效果最好？
4. 老板对现在的10B 的模型不满意，想知道扩大到100B模型的效果能提升多少？

以上这些问题都可以基于Scaling Law的理论进行回答。

**核心结论：**

大模型的Scaling Law是OpenAI在2020年提出的概念[1]，具体如下:

1. 对于Decoder-only的模型，计算量C(Flops), 模型参数量N, 数据大小(token数)D，三者满足:C=6N\*D。
2. 模型的最终性能主要与计算量C，模型参数量N和数据大小D三者相关，而与模型的具体结构(层数/深度/宽度)基本无关。
> 固定模型的总参数量，调整层数/深度/宽度，不同模型的性能差距很小，大部分在2%以内

3. 对于计算量C，模型参数量N和数据大小D，当不受其他两个因素制约时，模型性能与每个因素都呈现幂律关系
4. 为了提升模型性能，模型参数量和数据大小需要同步放大，但模型和数据分别放大的比例还存在争议。
5. Scaling Law不仅适用于语言模型，还适用于其他模态以及跨模态的任务

  
大模型中的scaling law

下图是GPT4报告[5]中的Scaling Law曲线，计算量和模型性能满足幂律关系

![](https://pic4.zhimg.com/v2-0c464866ad2b43915f0a16fc983da61f_1440w.jpg)

- 横轴是归一化之后的计算量，假设GPT4的计算量为1。基于10,000倍小的计算规模，就能预测最终GPT4的性能。  
    
- 纵轴是"Bits for words", 这也是交叉熵的一个单位。在计算交叉熵时，如果使用以 2 为底的对数，交叉熵的单位就是 "bits per word"，与信息论中的比特（bit）概念相符。所以这个值越低，说明模型的性能越好。

**Baichuan2**

![](https://pica.zhimg.com/v2-1f47ba9b875ed6fe142c6f46316babbc_1440w.jpg)

**MindLLM**

![](https://picx.zhimg.com/v2-3954de9ab4987844b05f0d02b60ad84f_1440w.jpg)

**Scaling Law实操： 计算效率最优**

根据幂律定律，模型的参数固定，无限堆数据并不能无限提升模型的性能，模型最终性能会慢慢趋向一个固定的值

![](https://pic4.zhimg.com/v2-a07c381c5776224b9f6121f5621a0c53_1440w.jpg)

![](https://pic3.zhimg.com/v2-f8d8584c4e03066f31849e5ba0b951a4_1440w.jpg)

![](https://pica.zhimg.com/v2-e48c1f766f8686857374a7d261261186_1440w.jpg)

所以LLaMA工作的重点是训练一系列语言模型，通过使用更多的数据，让模型在有限推理资源下有最佳的性能。

具体而言，确定模型尺寸后，Scaling Law给到的只是最优的数据量，或者说是一个至少的数据量，实际在训练中观察在各个指标上的性能表现，只要还在继续增长，就可以持续增加训练数据。

**计算量、模型和数据大小的关系推导**

![](https://picx.zhimg.com/v2-e50d6038a3a99b2929cba8458ae0de6b_1440w.jpg)

## 文章

#ScalingLaw [训练10B的模型需要多大的数据？详解大模型中的Scaling Law](https://mp.weixin.qq.com/s/lSLJhyT5LKuKtZMD3EaR_A)

#ScalingLaw [A Mechanistic Interpretability Analysis of Grokking — AI Alignment Forum](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking)

#ScalingLaw [Scaling 能通往 AGI 吗？万字科普 Scaling Law 的业内争议与讨论](https://mp.weixin.qq.com/s/Pn-vR-cRKNN5w5tlb2O6UA)

#alignment #ScalingLaw[大模型对齐阶段的Scaling Laws](https://mp.weixin.qq.com/s/PfWMa7qflwPYcJVG2LAtHg)

#ScalingLaw [揭开OpenAI Scaling Laws面纱](https://mp.weixin.qq.com/s/4y3io0JUNOjVOqLSMkOS6A)

[大模型应用面试准备2](https://zhuanlan.zhihu.com/p/687470762)

