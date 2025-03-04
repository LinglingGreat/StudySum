---
title: ReFT
created: 2025-02-19
tags:
  - o1-related
  - reasoning模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 字节
---

## 论文基本信息

标题：REFT: Reasoning with REinforced Fine-Tuning

作者：

链接：https://arxiv.org/pdf/2401.08967

代码：

框架图：


## 背景



## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

ReFT方法包括两个阶段：SFT冷启阶段 和 强化学习训练阶段。

### **ReFT SFT冷启阶段（warm-up stage）**

SFT阶段通过构造一批带推理过程的数据，来精调Base LLM模型，这个阶段主要是让模型有基本的CoT推理能力。ReFT的做法也非常简单，就是用一批开源的数据，通过Prompt工程来发压GPT-3.5t来收集样本，再SFT微调自己的小模型。

具体实现细节上，**样本数据**主要来源于GSM8K，SVAMP和MathQA 三个数据集，通过GPT-3.5-turbo few-shot prompting的方法收集的训练数据。数据集有两种推理格式： N-CoT， P-CoT。

> 作者在实验中并没有人工标注训练数据，而是完全通过self-instruct方式，基于GPT-3.5dump的训练样本。

![](https://pic3.zhimg.com/v2-03e1b117870487e70af5b67e76f525c2_1440w.jpg)



通过上述方法，收集了SFT数据集： 62K的指令集样本，分布如下，

![](https://pic2.zhimg.com/v2-bb70bdec82eceab863f420c03e8a995d_1440w.jpg)

图4、ReFT的SFT阶段样本分布

**模型训练：** 这个阶段模型训练就是常规的SFT， 训练了40个Epoch（作者说这个训练步数是保证模型收敛的一个比较大的设置了）

### **强化学习训练阶段（Reinforcement Learning Stage）**

ReFT使用PPO的方法做强化学习，我们先回顾下PPO的具体方法，如下图5所示 （参考： [姜富春：OpenRLHF源码解读：1.理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674)）

![](https://pic1.zhimg.com/v2-8e77bd9d167d2ecf70434ddfaa4ab9f0_1440w.jpg)

图5、PPO的训练流程图

**3.2.1. PPO训练四阶段**

- **阶段1：** 先基于Pretrain model，训练一个**精调模型（SFT Model）** 和 一个**奖励模型（Reward Model）**。Reward model 一般可以基于SFT model 热启 或 基于 Pretrain model 热启训练
- **阶段2：** 模型初始化，PPO过程，在线同时有 **四个模型，分别为**

	- **Actor Model ：** 是我们要优化学习的策略模型，同时用于做数据采样，用SFT Model热启
	- **Reference Model ：** 是为了控制Actor模型学习的分布与原始模型的分布相差不会太远的参考模型，通过loss中增加KL项，来达到这个效果。训练过程中该模型不更新
	- **Critic Model：** 是对每个状态做打分的价值模型，衡量当前token到生成结束的整体价值打分，一般可用Reward Model热启
	- **Reward Model ：** 这里是ORM（Outcome Reward Model），对整个生成的结果打分，是事先训练好的Reward Model。训练过程中该模型不更新

- **阶段3：** 采样Experience数据，流程为：

	- 首先采样一批随机指令集（Prompt）
	- 调用Actor模型的generate()方法，采样1条或多条结果（sequences）
	- 四个模型一起参与组装Experience的多个Tensor域，用于后续模型训练

- **阶段4:** 用Experience样本，训练 Actor Model 和 Critic Model

重复3-4阶段，循环采样Experience数据-> 模型训练 ，直到loss收敛

由上述过程可知，做PPO训练，我们需要预先准备两个训练好的模型： Base LLM Generator 和 Reward Model。ReFT是如何准备这两个模型的？首先Base LLM Generator就是warm-up阶段SFT的模型，对于Reward Model本文作者并没有训练一个模型，而是通过定义一个规则函数，设置了一个Rule-Base RM，下面我们来看看细节。

**3.2.2. Reward Model的设计：Rule-Base RM**

ReFT主要是针对数学场景富集样本来训练Reasoning Model，数学计算的问题是可简单判断答案正确与否的。所以作者设置了一个判别函数作为奖励。具体如下

![](https://pic2.zhimg.com/v2-c6b55c713ddaa40c5f19a929b66bed29_1440w.jpg)

这里的Reward Model是个ORM，对于模型生成的中间状态，奖励都为0； 对于终止状态判别有3种情况：

1. 如果通过规则能抽取出正确答案则奖励为1；
2. 如果抽取不出正确答案，但能解析出一个结果，奖励值设置为0.1；
3. 如果最终结果无法解析，奖励制设置为0。

> 这里对错误答案的推理路径，设置了一个弱奖励的机制（赋值0.1），主要是为了减少奖励反馈稀疏的问题。如果能解析出一个答案，证明生成过程是在做一个推理的过程，虽然答案错了，但推理的执行过程对模型是有帮助的，所以设置个小的奖励值，激励模型按推理逻辑输出结果。

**3.2.3. Critic Model的设计**

Critic Model是对每个状态做打分的价值模型，衡量当前token到生成结束的整体价值打分，模型的结构一般跟Reward Model一致，通常也会用Reward热启。但本文中，并没有Reward Model，那么Critic Model如何设计的呢？

作者对Critic Model的设计还是遵从Reward Model的设计方式，在Base Model之上，增加一个回归头（regresion head）对每个生成的状态进行打分。ReFT也做了些优化，为了减少训练时模型的计算量和显存占用， Critic Model的参数与Actor Model(Policy Model)的参数共享。如下图所示：

![](https://picx.zhimg.com/v2-9e5702e7fe1ac927fe7adc98b263c119_1440w.jpg)

**3.2.4. self-training 实验设置**

为了证明ReFT的有效性，作者也实现两个self-training的方法。这两个方法虽然被证明没有ReFT效果好，但这两种方法实际工作，是经常被使用的方法。所以也详细展开介绍下。有助于在实际工作中，根据自己的业务特点，选择合适的优化方案。

所谓self-training 就是不采用额外的人工标注的数据集，而是通过模型自己产出的数据再来迭代优化模型的方法。作者设置的两个self-training的实验如下：

4. Offline Self-Training (Offline- ST) ： 用初始的SFT的模型作为generator，对于训练集的每个问题通过生成多次来采集多个样本。然后将采集的样本的答案跟ground Truth对比，筛选答案正确的样本，然后跟原始样本混合，再训练初始模型。**这就是我们通常使用的拒绝采样训练模型的方法。**
5. Onlline Self-Training (Online-ST)：跟ReFT类似，用SFT的模型warm-up，之后在每个训练步，我们通过当前版本的模型做即时采样，然后保留答案正确的Sample，再通过SFT训练当前版本模型，得到下一个版本的模型。重复上面的迭代过程，直到达到预期的训练步数。

为了清晰对比 Offline-ST， Online-ST和ReFT，如下图所示：

![](https://pica.zhimg.com/v2-371518a497eafa61a18c43ec6a82db92_1440w.jpg)

相较于上面两种Self-Training，**ReFT优势主要有如下两方面**：

6. **样本充分利用**：在ReST中是基于RL的优化过程，对于采样的正负样本都参与模型训练。而上述两种Offline-ST和Online-ST两种方法都是基于SFT训练模型，SFT是只能使用正样本做模型训练的，样本使用上是不够充分的。
7. **模型训练稳定**： Offline-ST采样模型直接使用初始模型，而Online-ST采样模型是随着Policy模型的更新实时更新的，导致Online的方式可能使模型的分布大大偏离原始模型的分布。导致模型在采样的数据集上overfitting，最终实际应用中效果达不到预期。而ReST采用PPO的方法训练，模型更新过程通过KL限制Policy模型与初始Model分布不会偏离太远。保证了模型的效果稳定。

> 上述的Self-Training 方法，在实际的工作中，都是值得借鉴的，虽然ReFT理论上效果更好，但基于PPO的训练更复杂。可以结合自己业务的特点，考虑合适的方法驱动提升模型效果。

### **总结ReFT**

ReFT核心使用PPO算法来提升Reasoning的能力，相对于传统的PPO算法，主要做了两方面优化： 1) 简化Reward Model ，使用的是Rule-Base Reward而非训练一个模型。2) Critic Model参数与Policy Model共享，压缩训练阶段模型的参数的存储空间，也进一步降低模型训练的复杂度。


## 实验




## 未来方向



## 主要收获


## 参考资料

[聊聊Reasoning Model的精巧实现（ReFT, Kimi K1.5, DeepSeek R1）](https://zhuanlan.zhihu.com/p/20356958978)

