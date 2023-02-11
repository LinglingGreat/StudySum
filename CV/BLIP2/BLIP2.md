---
title: BLIP2
created: 2023-02-10
tags: 多模态
type: 论文
papername: BLIP-2-Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
conference: 
year: 2023
institution: Salesforce Research
---

## 论文基本信息

标题：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

作者：Junnan Li Dongxu Li Silvio Savarese Steven Hoi

链接： https://arxiv.org/abs/2301.12597

代码： https://github.com/salesforce/LAVIS/tree/main/projects/blip2

框架图：

![](img/Pasted%20image%2020230210225534.png)


![](img/Pasted%20image%2020230210231142.png)

模态之间的对齐非常重要。Flamingo是通过图像到文本的生成损失达到这一点的，我们证明了这不足以弥合模态的差距。

**Q-Former**是一个弥补冻结图像编码器和冻结语言模型之间的差距的可训练模块。它从图像编码器中抽取固定数量的输出特征，与输入图像分辨率无关。

它包括2个共享自注意力层的transformer子模块：一个image transformer，和冻结的图像编码器交互，提取视觉特征；一个text transformer，既能作为文本编码器，也能作为文本解码器。

我们创建了一组可学习的query embeddings作为图像transformer的输入，这些queries通过自注意力层进行彼此交互，通过cross-attention层进行与冻结的图像特征的交互。它们还可以通过同样的自注意力层和文本交互。根据预训练任务，我们应用不同的自注意力掩码来控制query-text的交互。

我们用BERT- base的预训练权重初始化Q-Former，cross-attention层则随机初始化。Q-Former总共包含188M的参数，注意queries视为模型参数。

实验中，我们用了32个queries，每个query的维度是768（和Q-Former的隐藏层维度一致）。Z代表输出query的表征，（32x768）的维度远远小于冻结的图像特征。


**预训练分为2个阶段**

第一阶段：视觉语言表示学习，冻结图像编码器，使用图像文本对进行训练，使得Q-Former能学习提取和文本最相关的视觉表征。

损失函数：Image-Text Contrastive Learning (ITC)，Image-grounded Text Generation (ITG)，Image-Text Matching (ITM)
	- ITC：Z和[CLS] token的embedding t之间计算。计算Z中的每个query output和t的pairwise相似度，选择最大的一个作为image-text相似度。为了避免信息泄漏，我们用了单模态自注意力掩码（queries和text不能相互看到）
	- ITG：计算给定图像的情况下生成文本的损失。Q-Former不允许图像编码器和文本tokens之间的交互，因此生成文本的所需信息必须通过queries抽取出来，然后通过自注意力传递到text tokens去。这样，queries就被迫提取能够捕获文本所有信息的视觉特征。我们使用了一个多模态causual self-attention mask来控制query-text的交互，和UniLM中所用的类似。queries可以互相attend，但是不能attend to 文本tokens。每个文本token可以attend to所有的queries以及它之前的文本tokens。我们使用[DEC]替代[CLS]来指示解码任务。
	-  ITM：预测图像文本对是否匹配的二分类问题。使用queries和text能够互相attend的双向自注意力层，输出的query embeddings Z能够捕获到多模态信息。每个embedding输出到二分类的线性分类器中，然后平均所有的logits值作为matching score。

![](img/Pasted%20image%2020230210234726.png)

第二阶段：将Q-Former的输出连接到冻结的LLM中，执行视觉到语言的生成学习，并且训练Q-Former使其输出的视觉表征可以被LLM解释

我们使用全连接 (FC) 层将输出查询嵌入 Z 线性投影到与LLM 的文本嵌入相同的维度。然后将投影查询嵌入添加到输入文本嵌入中。它们充当软视觉提示，根据 Q-Former 提取的视觉表示来调节 LLM。由于 Q-Former 已经过预训练以提取语言信息视觉表示，因此它有效地充当信息瓶颈，将最有用的信息提供给 LLM，同时删除不相关的视觉信息。这减轻了 LLM 学习视觉语言对齐的负担，从而减轻了灾难性遗忘问题。

### 实验

实验了两种LLM：decoder-based的OPT和encoder-decoder-based的FlanT5。

图像编码器：ViT-L/14和ViT-G/14。删除ViT的最后一层，使用倒数第二层的输出特征，会带来更好的表现。

![](img/Pasted%20image%2020230210235653.png)

![](img/Pasted%20image%2020230210235711.png)

更强的图像编码器或更强的 LLM 都会带来更好的性能。FlanT5，一种指令调整的 LLM，在 VQA 上优于无监督训练的 OPT。

![](img/Pasted%20image%2020230211000114.png)

图5表明了视觉-语言表征学习的重要性。

表3展示了image caption的结果。我们为图像字幕任务微调 BLIP-2 模型，该任务要求模型为图像的视觉内容生成文本描述。我们使用提示“a photo of”作为 LLM 的初始输入，并训练模型生成具有语言建模损失的说明。我们在微调期间保持 LLM 冻结，并与图像编码器一起更新 Q-Former 的参数。我们用 ViT-G 和各种 LLM 进行实验。我们对 COCO 进行微调，并对 COCO 测试集和零样本迁移到 NoCaps验证集进行评估。

表4视觉问答。给定带注释的 VQA 数据，我们微调 Q-Former 和图像编码器的参数，同时保持 LLM 冻结。我们对开放式答案生成损失进行微调，其中 LLM 接收 Q-Former 的输出和问题作为输入，并被要求生成答案。为了提取与问题更相关的图像特征，我们在问题上额外设置了 Q-Former。具体来说，问题标记作为 Q-Former 的输入，并通过自注意力层与查询交互，这可以引导 Q-Former 的交叉注意力层专注于信息量更大的图像区域。

![](img/Pasted%20image%2020230211000321.png)

表5图文检索。由于图像文本检索不涉及语言生成，我们直接微调第一阶段预训练模型而不使用 LLM。具体来说，我们使用与预训练相同的目标（即 ITC、ITM 和 ITG）在 COCO 上与 Q-Former 一起微调图像编码器。然后，我们在 COCO 和 Flickr30K (Plummer et al., 2015) 数据集上评估图像到文本检索和文本到图像检索的模型。在推理过程中，我们遵循 Li 等人。 (2021; 2022) 首先根据图像文本特征相似性选择 k = 128 个候选者，然后根据成对的 ITM 分数重新排序。我们尝试使用 ViT-L 和 ViT-G 作为图像编码器。

![](img/Pasted%20image%2020230211000330.png)

图文检索任务中，ITC和ITM损失是至关重要的。ITG损失也是有效的。

### 限制

最近的 LLM 可以在给定少量示例的情况下执行上下文学习。然而，我们使用 BLIP-2 的实验在为 LLM 提供上下文 VQA 示例时没有观察到 VQA 性能的改进。我们将缺乏上下文学习能力归因于我们的预训练数据集，每个样本只包含一个图像-文本对。 LLM 无法从中学习单个序列中多个图像-文本对之间的相关性。 Flamingo 论文中也报道了相同的观察结果，该论文使用了一个闭源交错图像和文本数据集 (M3W)，每个序列具有多个图像-文本对。我们的目标是在未来的工作中创建一个类似的数据集。

![](img/Pasted%20image%2020230211001417.png)

由于各种原因，包括来自 LLM 的知识不准确、激活不正确的推理路径，或者没有关于新图像内容的最新信息，BLIP-2 的图像到文本生成可能会产生不理想的结果（见图 7）。此外，由于使用了冻结模型，BLIP-2 继承了 LLM 的风险，例如输出攻击性语言、传播社会偏见或泄露私人信息。补救方法包括使用指令来指导模型的生成或在已删除有害内容的过滤数据集上进行训练。

## 核心亮点

## 主要收获

