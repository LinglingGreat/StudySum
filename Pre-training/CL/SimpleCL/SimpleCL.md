---
title: SimpleCL
created: 2024-05-04
tags:
  - ContinualLearning
type: 论文
papername: Simple and Scalable Strategies to Continually Pre-train Large Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - EleutherAI
---

## 论文基本信息

标题：Simple and Scalable Strategies to Continually Pre-train Large Language Models

作者：

链接：

代码：

框架图：


## 背景

当应用简单且可扩展的持续学习技术时，持续预训练的 LLM 与基于随机初始化通过所有数据的联合预训练的 LLM 之间的性能差异是什么？

Our empirical evaluation spans large (10B parameters) and small (405M parameters) models as well as weak (English → English) and stronger (English → German) distribution shifts.

**Rules of thumb for continual pretraining**

![](img/Pasted%20image%2020240504153817.png)


## 相关研究



## 核心亮点

a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is sufficient to match the performance of fully re-training from scratch on all available data, as measured by final loss and language model (LM) evaluation benchmarks.总结起来就是三个点：re-warmup, re-decay, replay data。


## 实验

### 实验设置

预训练的best practices：建议从线性预热阶段开始，并将学习率衰减到其最大值的10倍，以便将余弦周期的结束设置为与令牌数量相匹配。本论文中的warmup默认比例是1%，最小学习率为最大学习率的1/10。

论文采取的replay方法是compute-equivalent replay。保持总的训练tokens不变，当增加replay数据的时候，会相应地减少新数据的token数量。每个训练batch中，x%的来自D0的replay数据，100%-x%的来自D1的新数据。

数据集：SlimPajama, German CommonCrawl, Pile。

SlimPajama训练集如下表所示，验证集采取原始的验证集。

![](img/Pasted%20image%2020240504154335.png)

German CommonCrawl（Oscar数据的一部分）分成195.43B的训练集和982.6M的验证集

Pile采用原始的训练和验证集。

Baseline

![](img/Pasted%20image%2020240504155027.png)

N=2的时候两种数据集设置
- Two datasets, weak shift: In this variation, we consider D0 to be the Pile (Gao et al., 2020) and D1 to be pre-training on SlimPajama
- Two datasets, stronger shift: In this variation, we consider D0 to be pre-training on the Pile (Gao et al., 2020) and D1 to be pre-training on German Common Crawl.
N>2的时候
- Three datasets, no shift : We consider an N = 3 setting, where D0, D1, D2 are each district 100B token splits of SlimPajama
- Domain incremental continual pre-training: This setting considers consuming the tokens of SlimPajama sequentially ordered by domain.

模型：GPT-NeoX，batch size of 1104 and a sequence length of 2048。two model sizes 405M and 9.6B parameters (referred to as 10B in this work) including embeddings.

### Learning Rate Schedule

![](img/Pasted%20image%2020240504160141.png)

在这两种分布变化中，我们最初观察到使用较短线性预热的模型最初忘记并适应的速度比较长预热的模型更快。发生这种情况是因为它们更快地增加了 LR，从而导致更快的遗忘和适应。然而，在所有场景中，这些初始差异在整个训练过程中都会减少，使得所有模型在 50B 个令牌之后都具有相对相似的遗忘和适应能力。因此，在探索的设置中，线性热身阶段的持续时间似乎对继续预训练时的遗忘或适应没有影响。

我们为所有后续实验设置了 1% 训练迭代的线性预热持续时间。

![](img/Pasted%20image%2020240504160545.png)

re-warm和re-decay的策略

![](img/Pasted%20image%2020240504160940.png)

恒定学习率的遗忘最小，re-warm and re-decay在新数据上的适应性最好。

在重新加热和重新衰减 LR 的模型中，我们观察到改变学习率会导致适应和遗忘方面的微小差异：较高的 ηmax 值会导致更多的遗忘和更多的适应，而对于较低的值则相反。将基线与联合训练基线进行比较时，我们观察到 D1 的最终验证损失在两个分布变化上均显着高于联合训练模型。对于弱分布偏移的 D1 也是如此，但有趣的是，对于强分布偏移，恒定学习率基线比联合训练模型实现了更低的 D1 验证损失。我们假设这是由于法学硕士背景下更强的分布变化通常会增强适应并加剧遗忘。当将不断预训练并重新加热和重新衰减的模型与联合基线进行比较时，我们注意到这些模型比联合基线更好地适应 D1（最终验证损失更低）。然而，这些模型在 D0 上经历了严重的遗忘，这表明需要replay才能使这些模型与联合基线竞争。

![](img/Pasted%20image%2020240504161501.png)

### The Effect of Replay

![](img/Pasted%20image%2020240504161718.png)

与 0% 基线相比，观察到 1%、5% 和 10% 重放对下游性能的影响很小，这表明在我们的设置中重放的遗忘好处几乎没有成本。

50% 重放模型达到或超过 D1 ∪ D0 基线训练的最终平均验证性能。这很奇怪，因为这些模型的 D1 代币比各自的基线少了 150B 和 100B.

![](img/Pasted%20image%2020240504162045.png)

![](img/Pasted%20image%2020240504161945.png)

### Continual Pre-training Final Performance for Weak and Strong Distribution Shifts.

![](img/Pasted%20image%2020240504162215.png)


对于没有重放的持续预训练模型，德语训练会导致 Pile (D0) 上的遗忘显着增加（弱转变和强转变分别为 0.27 和 1.39）

![](img/Pasted%20image%2020240504162413.png)

![](img/Pasted%20image%2020240504162448.png)

![](img/Pasted%20image%2020240504162715.png)

### Continual Pre-training Final Performance at Different Model Scales

![](img/Pasted%20image%2020240504162815.png)

![](img/Pasted%20image%2020240504163001.png)

![](img/Pasted%20image%2020240504163029.png)

![](img/Pasted%20image%2020240504163052.png)

### Understanding and Circumventing the Pathologies of Re-warming

**Re-warming on the Same Data**

之前的实验中，当在新的数据上持续预训练的时候，旧数据上的Loss会有个快速增加，随着学习率增大，这个增加会越多。我们猜想这主要是因为预训练数据集的分布迁移导致的负迁移。we follow a similar methodology as in our experiments from Fig. 3 but continue to pre-train on Pile as D1.

![](img/Pasted%20image%2020240504172045.png)

re-warmup学习率似乎是损失增加的一个重要原因。

**Infinite Learning Rate Schedules**

探索一种不需要warm-up的学习率机制，在所有新任务上保持恒定学习率。一方面，余弦衰减时间表要求我们提前知道要预训练的令牌总数。这限制了继续预训练融合检查点的能力。另一方面，我们在上一节中看到，当在以小学习率结束的余弦衰减机制预训练的模型上继续预训练时，需要将学习率从最小值warmup以最好地适应新的数据集。然而，正如上一小节所见，我们观察到重新加热学习率会加剧遗忘。

![](img/Pasted%20image%2020240504172856.png)

论文中考虑的cooldown phase包括2种

![](img/Pasted%20image%2020240504173136.png)

**Comparing Cosine Decay to Variants of our Infinite Schedules**

![](img/Pasted%20image%2020240504173322.png)

**Infinite Learning Rate Schedules: Scaling to Infinite Future Updates**

![](img/Pasted%20image%2020240504173501.png)

![](img/Pasted%20image%2020240504173636.png)



## 未来方向



## 主要收获


## 参考资料
