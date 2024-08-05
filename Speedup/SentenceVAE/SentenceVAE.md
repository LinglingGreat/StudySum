---
title: SentenceVAE
created: 2024-08-04
tags:
  - 推理加速
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
---

## 论文基本信息

标题：SentenceVAE: Faster, Longer and More Accurate Inference with Next-sentence Prediction for Large Language Models

作者：[Hongjun An](https://arxiv.org/search/?searchtype=author&query=Hongjun%20An) ; [Yifan Chen](https://arxiv.org/search/?searchtype=author&query=Yifan%20Chen) ; [Xiaozhen Qiao](https://arxiv.org/search/?searchtype=author&query=Xiaozhen%20Qiao) ; [Zhe Sun](https://arxiv.org/search/?searchtype=author&query=Zhe%20Sun) ; [Xuelong Li](https://arxiv.org/search/?searchtype=author&query=Xuelong%20Li)

![](img/Pasted%20image%2020240804164916.png)

链接：http://arxiv.org/abs/2408.00655

代码：无

框架图：

![](img/Pasted%20image%2020240804170414.png)

![](img/Pasted%20image%2020240804170402.png)

## 摘要

目前LLM的推理方式都是next token prediction，限制了推理速度。这篇文章提出next sentence prediction的推理方法，可以提高LLM的推理速度。

具体怎么做的呢？论文通过提出包含一个编码器(encoder)和一个解码器(decoder)的SentenceVAE来实现next sentence prediction。编码器有效地将句子中的信息压缩为单个token，而解码器则将压缩的数据重建回其原始句子形式。

通过将SentenceVAE集成到LLM的输入和输出层，论文作者开发了Sentence-level LLMs（SLLMs），它采用逐句推理方法，显著加快了推理速度。 SentenceVAE 还通过将文本分割成句子来保持原始语义内容的完整性，从而在提高推理速度的同时保持准确性。与传统的 LLMs相比，SLLMs 在同等上下文长度上处理更少的tokens，显著减少了 Self-Attention 计算的内存需求，并有助于处理更长的上下文。论文的实验结果表明，该方法可以将推理速度提高 204∼365%，将困惑度 (PPL) 降低至原始指标的 46∼75%，并在相同上下文长度下将内存开销降低 86∼91%。随着模型参数的增加，这种方法的优点进一步放大。

## 相关研究

Multi token prediction

## 核心亮点

具体方法其实看Figure 1和2已经非常清晰了。

### Sentence Variational Autoencoder (SentenceVAE)

第一步，分句机制(Sentence Segmentation Mechanism)。在分词之前，使用正则表达式将原文本分成多个句子。

第二步，句子编码器(Sentence Encoder)。将每个句子进行分词得到$D = [d1, d2, d3, ..., dL]$（L个token），并将每个句子单独经过一个Embedding层，得到这个句子的token-level的embedding表示E。再将E输入基于自注意力的编码器块，得到hidden features H（维度是L×hidden size）。

为了得到句子embedding，论文将这 L 个特征融合成单个向量（维度是1×hidden size）。然后将该句子embedding向量输入仅解码器结构的 LLM，该模型在每个时间步会生成一个新的句子embedding向量。

上述的编码器得到H后是怎么融合成单个向量的呢？这就涉及到特征融合机制(Feature Fusion Mechanism)。通过将L个向量累加，然后经过一个Layer Normalization归一化后就得到了单个向量。（这里的t代表的是第t个句子）

![](img/Pasted%20image%2020240804172745.png)

句子解码器(Sentence Decoder)。句子解码器包含掩码自注意力(masked self-attention)和交叉注意力(cross-attention)模块。交叉注意力将上述说到的单个句子向量作为key和value。

预测的时候，给定$\Omega_{t+1}$，初始化输入token d0，decoder输出d1，然后把d0, d1和$\Omega_{t+1}$输入decoder，重复这个过程直到生成eos token。

![](img/Pasted%20image%2020240804172804.png)

训练时候的损失函数

![](img/Pasted%20image%2020240804174420.png)

### Sentence-level Large Language Models (SLLMs)

传统的LLM预测路径（这里的EMB是embedding layer，DB是decoder-only blocks，FC是fully connected layer）

![](img/Pasted%20image%2020240804173629.png)

在SLLM中，会变成这样：

![](img/Pasted%20image%2020240804173821.png)

N个decoder-only模块不再直接接收word-level tokens。相反，它接收由句子编码器编码的句子embedding向量。因此，LLM 架构中的传统embedding层被删除。

![](img/Pasted%20image%2020240804174117.png)

在 SLLM 中，DB 的输出是句子级隐藏状态向量。因此，论文中使用一个新的全连接层，称为终止判断层，它将 H转换为二维布尔向量标志。该向量有助于确定当前句子级隐藏状态向量是否表示句子的结束（停止标志）或需要句子解码器进一步解码。如果向量指示结束标志，则迭代终止。否则，句子解码器处理嵌入以生成相应的token。训练的时候，会计算这个部分的focal loss。



## 实验
为了验证想法，论文作者首先使用自监督方法训练单个 SentenceVAE，证明句子中的多个token可以由编码器压缩为单个句子向量并由解码器恢复。随后，论文作者在开源LLM的开头和结尾嫁接了编码器和解码器，证明通过简单的修改，LLM可以升级为SLLM并在句子嵌入空间中工作，同时保留PPL并提高推理速度。同时，通过观察损失曲线，论文作者发现SLLM仍然遵循缩放定律。

实验设置
- 单个 4 卡 RTX 4090 (24G) 或 4 卡 A100 (40G)（ SLLM-1.3B）
- 基于opt系列的125m, 350m, and 1.3B模型
- Wanjuan-1.0数据集的英文子集(EN/WebText)作为训练集
- 1,000 个与训练集互斥的随机句子或段落作为验证集。

实验一：证明句子中的多个token可以由编码器压缩为单个句子向量并由解码器恢复。

对于每个hidden size, 实验中调整了block layers的数量为1, 2和4。

![](img/Pasted%20image%2020240804175530.png)

![](img/Pasted%20image%2020240804175807.png)

实验二：LLM 可以在句子级嵌入空间中工作，具有更快的推理速度、更准确的 PPL 和更长的上下文，从而产生 SLLMs

![](img/Pasted%20image%2020240804180037.png)

![](img/Pasted%20image%2020240804180053.png)

实验三：SLLM 仍然遵循缩放法则

![](img/Pasted%20image%2020240804180125.png)


