---
title: AFM
created: 2024-08-22
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - Apple
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：

论文描述了 AFM 的两个版本：一个用于在手机、平板电脑或笔记本电脑上部署的30亿参数的设备模型，以及一个更强大的30亿参数的服务器模型。

这些模型是为聊天、数学和编码任务开发的，尽管论文没有讨论任何与编码相关的特定训练和能力。

与 Qwen 2 类似，AFM 是密集的 LLMs，并没有使用专家混合方法。

![](img/Pasted%20image%2020240824165058.png)

几个细节：

- 共享了输入输入的embedding，减少参数量
    
- 参考《Small-scale proxies for large-scale transformer training instabilities》，使用Query/key normalization，提升训练稳定性
    
- RoPE的base frequency为500k

tokenizer是基于SentencePiece用BPE训的，所有数字都切分为单个数字。AFM-server模型的词表大小为100k，AFM-on-device则小一些，只有49k。

## 预训练

### 数据

首先，除了使用公开可用的数据和出版商授权的数据，他们还尊重网站上的 robots.txt 文件，并且没有爬取这些网站。其次，他们还提到使用基准数据进行了去污染。

为了强调 Qwen 2 论文中的一个要点，研究人员提到质量比数量更重要。（设备模型的词汇量为49k个词元，服务器模型的词汇量为100k个词元，明显小于 Qwen 2 模型使用的150k个词元词汇量。）

另外苹果（自称）特别看重隐私和安全性，因此所有数据的几乎全部流程都有大量移除有害数据、personally identifiable information（PII）、成人内容的处理工作。

下面罗列一些预训练数据的处理细节。

1、网页数据

处理pipeline包括：

- 结合Safari的reader mode和Boilerpipe算法提取网页的主体内容
    
- 规则+model based的安全过滤
    
- 基于locality-sensitive n-gram hashing的模糊去重
    
- 质量过滤（《Large language model-guided document selection》，《Datacomp-lm: In search of the next generation of training sets for language models》）
    
- Decontamination：从预训练数据按n-gram删除和811个benchmark过度相关的数据，避免测试集污染
    

2、授权数据

从出版社获取的高质量长文本数据，主要用在二阶段和三阶段的预训练（各阶段方案在后面）。同样做了避免测试集污染的操作。

3、代码

来自github的开源仓库，包含14种语言，经过去重、PII过滤、质量过滤和Decontamination处理。

4、数学

包括3B数学QA内容，和14B数学相关的文档，来自数学论坛、博客、tutorial和seminar等。为了提取这些数据，苹果专门开发了对应的模板、数学符号filter、数学相关的quality filter以及领域filter。

5、公开数据

从公开数据里挑了一部分高质量数据。




有趣的是，预训练不是在2个阶段而是在3个阶段完成的！

1. 核心（常规）预训练，大部分训练预算都在这一个阶段消耗
    
2. 持续预训练，其中网络抓取（质量较低）数据的权重被降低；数学和代码的权重被提高
    
3. 使用较长序列数据和合成数据进行上下文扩展

三个stage在调参的时候，用了和《Small-scale proxies for large-scale transformer training instabilities》中的“μParam (simple)”类似的方法。

### 预训练 I:核心预训练

核心预训练描述了苹果预训练流水线中的第一个预训练阶段。这类似于常规预训练，其中AFM服务器模型在6.3万亿个标记、4096个批次大小和4096个标记序列长度上进行训练。这与Qwen 2模型非常相似，后者在7万亿个标记上进行训练。

然而，AFM设备上的模型更有趣，它是从一个更大的64亿参数模型中蒸馏和修剪而来的（从头开始训练，就像前面描述的AFM服务器模型一样。请注意，AFM服务器和AFM设备都是30亿参数模型。）

关于蒸馏过程的细节不多，除了"通过将目标标签替换为真实标签和教师模型的top-1预测的凸组合（以0.9的权重分配给教师标签）来使用蒸馏损失。"

_知识蒸馏概述，其中一个小模型（这里是AFM设备3B模型）在原始训练标记加上来自更大教师模型（这里是64亿模型）的输出上进行训练。请注意，a）中的交叉熵损失是用于预训练LLM的常规训练损失_

![](img/Pasted%20image%2020240822201835.png)

知识蒸馏，如上所述，仍然涉及在原始数据集上进行训练。然而，除了数据集中的训练标记外，被训练的模型（称为学生）还从较大的（教师）模型接收信息，与没有知识蒸馏的训练相比，提供了更丰富的信号。不利的一面是，你必须：1）首先训练较大的教师模型，2）使用较大的教师模型计算所有训练标记的预测。这些预测可以提前计算（这需要大量的存储空间）或在训练过程中计算（这可能会减慢训练过程）。

1、AFM-server

- 使用6.3T数据
    
- sequence length = 4096
    
- batch size = 4096
    
- weight decay = 3.16e-4
    
- cosine lr schedule, max lr = 0.01, min lr = 0.5% max lr
    
- warmup step = 5000
    

batch size是通过scaling law的实验决定的，不过实践中发现，下游任务的效果对预训练的batch size并不敏感：batch size增大一倍或者缩小一半下游任务效果没有影响，因此虽然scaling law给出的预测最佳batch size是3072，实际训练的时候，为了效率还是使用了4096。

通过proxy model的lr扫描，定了最佳lr在0.01~0.02，最终选择了0.01。（这里使用类似μParam的方法，各参数初始化和前向计算的时候应该都有缩放，所以这个lr会相对大一些）

苹果训练的时候选择的优化器是RMSProp with momentum，而其他大部分大模型基本都是使用AdamW。

对于训练设置的问题，苹果做了消融实验，把上面的core training和以下配置的训练（baseline）进行对比：

- 使用AdamW，beta_1 = 0.9，beta_2 = 0.95
    
- weight decay = 1e-4
    
- lr 最小decay到0.0001
    
- batch size = 1024
    

其他设置保持一致，用AFM-on-device模型结构训练3.1T数据。

二者的对比如下：

![](img/Pasted%20image%2020240824165601.png)

整体上AFM的core training比baseline略略好一点，基本上可以认为是持平。

2、AFM-on-device

AFM-on-device模型不是从零训练的，而是基于一个6.4B的模型（使用和AFM-server一样的训练方法得到的），使用了structural pruning和distillation得到的。

所用的structural pruning和《Structured pruning of large language models》、《Sheared llama: Accelerating language model pre-training via structured pruning》相似，除了几点变化：

- 只对FFN层做prune
    
- 使用Soft-Top-K masking（《Conditional adapters: Parameter-efficient transfer learning with fast inference》）
    
- 用了和core training一样的data mix训练了188B得到pruning mask
    

以得到的模型的为初始化，进行知识蒸馏：把原来core训练的target label替换成：0.9 * teacher top-1 prediction + 0.1 * true label。

同样进行了6.3T的蒸馏训练。

相比直接从零训练，pruning和distillation在数据效率和最终结果上都有收益。使用不同方法训练出来的模型效果对比如下：

![](img/Pasted%20image%2020240824165637.png)

整体来看，prune + distill能比多5倍training cost的从零训练baseline更好一点，训练效率更高。

### 预训练 II:持续预训练

这一stage提高了math和code的比例，而降低了低质量的爬虫数据比例，进行了1T token的训练。

训练设置：

- sequence length = 8192（从4096增加到8192
    
- max lr = 3e-4，min lr = 0.1% max lr
    
- weight decay = 1e-5
    
- warmup step = 1000
    

其他和core training保持一致。

研究人员发现蒸馏损失在这一阶段并没有带来好处，所以AFM-on-device和AFM-server一样，采用直接训练的方式。


### 预训练 III:上下文扩展


最后这一阶段使用100B的长窗口训练来提升模型的长文本能力：

- sequence length = 32768
    
- RoPE base frequency 500k --> 6315089（《Scaling laws of rope-based extrapolation》）
    
- 在二阶段数据的基础上，增加长的QA合成数据

### 评测

三个阶段后，AFM-on-device和AFM-server的评测效果如下（报告提到，使用了internal的formulation，所以没法和其他模型直接比较）

![](img/Pasted%20image%2020240824165956.png)

![](img/Pasted%20image%2020240824170003.png)

continued pre-training和预期的一样，对math和code的能力有比较大的提升。

## 后训练

AFM的post-training包括SFT和RLHF两个阶段，并使用了两个新方法iTec和MDLOO。

### 数据

post-training的数据包括人类真实数据和合成数据。

一个好的reward model是合成高质量数据的关键，同时扩展prompt set提高多样化和覆盖范围也很重要。

苹果介绍了数学、工具使用和代码这3个领域的数据合成。

1、Mathematics

数学数据的合成包括两个stage：

- 生成数学问题
    
- 生成对应答案
    

基于一些种子prompt，通过以下方法获取数量更大、更多样化的prompt：

- Problem rephrase and reversion：参考《Metamath: Bootstrap your own mathematical questions for large language models》，进行问题重述
    
- Problem evolution：和指令进化类似（《WizardLM: Empowering large language models to follow complex instructions》），深度进化提升指令的复杂度，而广度进化提升话题的覆盖范围

2、Tool use

先从简单的single-tool数据开始，训练模型。然后逐步包含multi-tool和multi-step的问题，提升模型能力。此外，还会在数据里混入oracle tool和其他相似同居，增加工具选择的难度。

另外还增加了tool intent detection数据，以减少过度使用工具的问题。

3、Coding

从71个话题的种子数据开始，通过self-instruct和rejection sampling让模型自动化学习。

对于每个问题，模型会生成单元测试和多个solution，通过执行这些solution能够检验结果的正确性，组中选择通过测试最多的solution。

另外还会给通过的单元测试设定一个阈值，只要高于这个阈值才会被使用。最终得到了12k的高质量代码数据。

### SFT

1、数据选择

在质量过滤、去重之外，通过数据合成 + rejection sampling来提供大量合成数据，提升SFT训练数据规模。

2、比例调整

对不同数据组成部分的权重进行训练，然后调整比例。对此进行了大量实验，移除掉一些影响较小的数据。

3、训练超参

模型使用constant lr训练，AFM-server和AFM-on-device的lr分别为5e−6和2e−5。

和其他家做法比较不同的，苹果使用的0.1的dropout rate。

由于不同checkpoint的eval指标会有波动，因此使用RM选择best-of-N的方式来挑选最佳checkpoint。

### RLHF

苹果的RLHF有多轮，迭代提升模型。

用前面收集的偏好数据训练RM：

- 每条prompt有两个response（对比一下，Llama-3可能有3条）
    
- 偏好数据分为significantly better, better, slightly better, negligibly better四个等级
    
- 除了综合的对比之外，每条response还有细粒度的打分，维度包括指令跟随、真实性、有害性、简明程度，每个维度的打分有3个等级

RM取最后一层的最后一个non-padding token的embedding，再加上一个linear层和4个MLPhead来输出打分。linear层输出偏好奖励，而4个MLP层分别输出4个细粒度打分的分类结果。

RM训练时，使用soft label loss function，这样可以把偏好的程度也纳入考虑。同时细粒度的打分也作为regularization term加入训练，实验发现这些细粒度打分能提升RM的准确性。

损失函数

![](img/Pasted%20image%2020240824170718.png)

### Iterative teaching committee（iTeC）

苹果提出一个iterative RLHF框架来优化模型。

苹果在AFM的RLHF中，学到的最重要的事情之一就是“refresh online human preference data collection using a diverse set of the best performing models”。

具体来说，构建一个由SFT、拒绝采样、DPO/IPO和RL训练出来的最佳模型，以及前几轮的最佳模型组成的集合，称之为“model committee”，并从这个model committee收集最新的偏好数据。

在获取最新的偏好数据之后，会更新RM，让后训练一组新的最佳模型，这些新的模型会加入model comittee，继续下一轮迭代。

不同的优化算法训练出来的模型有不同的特点，比如使用负例的算法，online RLHF、DPO、IPO等，在数学推理方面的能力较好，而rejection sampling在指令遵循和写作方面更有效。通过在model comittee进行采样，并用最新的RM进行排序，可以结合多个模型的强项。

### Online RLHF algorithm: MDLOO

![](img/Pasted%20image%2020240824170913.png)

2、Mirror descent policy optimization (MDPO)

和常用的clipping-based PPO不同，使用KL divergence作为regularization。

## 赋能Apple Intelligence

AFM是给Apple Intelligence使用的，而Apple Intelligence主要是支持iPhone、iPad和Mac等设备的，因此「计算效率」和针对这些设备场景下的「专用能力」是重点。

虽然经过post-training之后，模型的通用能力已经不错，但是针对设别上的任务进行专门的微调，还能获得进一步的提升。苹果通过使用多个任务相关的adapter，在提升多个任务效果的同时，保持了参数和计算的高效。这些adapter很小，运行时可以在内存中随意切换。

![](img/Pasted%20image%2020240824171000.png)

### accuracy-recovery adapter

1、效果恢复

端侧设备的空间比较小，所以量化是必须要做的。首先，post-training后的模型会用4-bit的精度进行量化。

但是由于量化模型会带来一定的效果损失，所以这个量化模型并不是直接使用，而是会在固定量化模型的基础上，用16-bit的LoRA进行训练，以尽量恢复因为量化带来的效果损失。这个LoRA就叫accuracy-recovery adapter。

accuracy-recovery adapter的训练过程和主干模型的训练保持一致，也进行了pre-training和post-training的训练。不过由于参数量很小（只有几十MB），所以整个预训练大概只用了10B的数据，并且基本可以恢复大部分由于量化带来的效果损失。

实践上，rank 16基本上可以获得比较好的效果，不过出于灵活性的考虑，还是提供了不同rank的LoRA参数给下游使用：8、16、32。

模型量化前后，以及使用accuracy-recovery adapter之后的效果对比如下：

![](img/Pasted%20image%2020240824171051.png)

rank 16的adapter基本可以恢复大部分量化带来的效果损失，并且量化的损失越多，adapter恢复的比例越大。也就是使用了accuracy-recovery adapter之后，基本可以不用太在意量化的损失，可以进一步提高模型压缩的程度。

2、Quantization schemes

以往量化的时候，因为要兼顾效率和效果损失，一般把block size设成32或者64这样比较小的规模。现在有了accuracy-recovery adapter，反正损失掉的基本都可以恢复，那block size就可以设得更大了，甚至可以达到100k。

另外，由于AFM的输入输出embedding是shared的，为了有更好的效率，embedding部分使用8-bit的per-channel quantization。

3、混合精度量化

模型中明显每层对效果的影响是不同的，对于对最终效果影响较小的层，苹果进一步用2-bit的量化精度，最终整体可以达到3.5~3.7的bpw。

### task-specific adapter

针对不同的下游任务，可以在accuracy-recovery adapter的基础上再进一步微调。这样在保持主干网络为4-bit模型的情况下，下游任务就能有很好的效果。

以summarization为例，具体任务是对设备上的email、message和notification进行摘要。使用AFM-server模型，用设备上真实信息的格式构造训练数据，然后用这些训练数据训练adapter。


## 主要收获


## 参考资料

[LLM预训练和后训练新范式](https://mp.weixin.qq.com/s/sFdnWwWXouU2ZPbf5GoXhA)

[苹果智能系统模型--AFM](https://mp.weixin.qq.com/s/itou554WdJrpHK4TMZ9kow)

