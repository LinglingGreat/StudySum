---
title: SelfExtend
created: 2024-01-15
tags:
  - 关键词
type: 论文
papername: LLM Maybe LongLM Self-Extend LLM Context Window Without Tuning
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - Texas
  - Amazon
  - RiceUniversity
---

## 论文基本信息

标题：LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning

作者：

链接： http://arxiv.org/abs/2401.01325

代码： https://github.com/datamllab/LongLM

框架图：


## 背景
假设：现有的LLMs有处理长文本的能力，本论文提出的方法是为了激活LLMs的能力。

长文本处理的主要挑战在于位置的OOD问题，即遇到的文本长度超过预训练期间见到的长度。方法一般是把没有见过的位置映射到见过的位置上。
- 文本中距离较长的单词之间的位置信息不需要很精确。
- 大多数时候，虽然一个词袋（ngram）一起出现在一个区域中，但由于语言语法的约定，该袋中的所有token只有一个可能的顺序。

![](img/Pasted%20image%2020240115145539.png)

通过FLOOR操作将0-7范围的位置映射到0-3.

![](img/Pasted%20image%2020240115150044.png)

1. FLOOR操作后仍然能保持一个比较好的PPL
2. 在小的group size下，PPL会略高于原本的模型（红色dot线和蓝色线）

在生成某个token时，邻居token是对该token最重要的，因此我们仍然需要保留邻域内的attention机制。

## 相关研究


## 核心亮点

![](img/Pasted%20image%2020240115151916.png)

![](img/Pasted%20image%2020240115152041.png)

扩充的最大长度是：$(L-w_n)*G+w_n$


## 实验
任务：语言建模、合成长上下文任务和真实长上下文任务。以及短文本任务。

较低的 PPL 并不能保证在实际任务中具有良好的性能，而过高的 PPL 则表明LLM的性能严重下降。

![](img/Pasted%20image%2020240115153317.png)

由于算力限制，设置了sliding window=256来计算PPL。mistral的SWA和selfExtend方法表现都可以。

在很长的文本里进行密钥检索任务。虽然SWA的mistral的PPL比较低，但是在这个任务里，也只能在原生的4k window里表现好，超过就表现变差了。

![](img/Pasted%20image%2020240115153815.png)


这可能是因为PPL计算的是所有token的均值，只要大部分token建模好，PPL就不会高。

Longbench和L-EVAL

![](img/Pasted%20image%2020240115154142.png)

与许多微调模型相比，Self-Extend 具有可比甚至更好的性能。
MultiNews的平均长度只有2k。或者像 PassageCount 这样的一些任务不适合测试这种大小的模型（即太具有挑战性）。

Llama2-7B: 在多个数据集（例如 HotpotQA）上比所有经过微调的同类产品具有更好的性能, 其他数据集具有可比性。

Vicuna: 比finetune的版本好。部分数据集上16k比25k的好是因为更大的context window和位置精确性之间的权衡。window变大了，位置精确度变粗糙了。

Mistral-7B：Lite版本finetune的时候用了大量的NarrativeQA等数据集，评估结论不可靠。

在 LEval 上，观察到类似的结果。除了使用 Mistral 作为基础模型外，与一些微调免费基线（例如 NTK）或进一步训练的基线（例如 Longchat1.5-7b-32k 和 Vicuna1.5-7b-32k）相比，Self-Extend 几乎在所有数据集上都实现了卓越的性能。对于Mistral来说，我们怀疑其较差的性能主要来自于prompt工程。与原生 Mistral 相比，MistralLite 的性能要差得多，这表明了这一点。我们没有为Mistral进行prompt工程。LEVAL在13B以下的模型里受prompt的影响较大，甚至出现了vicuna-13b比vicuna-7b更差的效果。

![](img/Pasted%20image%2020240115155811.png)

![](img/Pasted%20image%2020240115155902.png)

短文本任务上几乎没有负面影响。而且这个方法可插拔，在短文本场景下可以保持原有能力。

![](img/Pasted%20image%2020240115160155.png)




## 未来方向



## 主要收获


## 参考资料
