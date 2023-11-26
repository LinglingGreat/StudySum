---
title: SelfRAG
created: 2023-11-26
tags: 
type: 论文
papername: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
  - 华盛顿大学
  - AllenAI
  - IBM
---

## 论文基本信息

标题：Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

作者：

链接： https://arxiv.org/pdf/2310.11511.pdf

代码： https://selfrag.github.io/

框架图：


## 背景

本文提出了一个叫 Self-RAG 的框架，方法如其名，希望 LM 自主决定对当前输入是否需要召回（而不是像[SKR](../SKR/SKR.md)那样训练一个额外的分类器或像[TrainRobustRALMs](../TrainRobustRALMs/TrainRobustRALMs.md)那样借助一个 NLI 模型判断），把召回内容拼接近输入，再生成一段下文，自主判断召回文档是否与输入问题相关、自己借此生成的一段下文是否合理、是否有用，对 topk 召回内容进行排序，把 top-1 加进最后的输入以尽量生成正确答案。框架如下图右栏所示。

![](img/Pasted%20image%2020231126161335.png)

Self-RAG是一个新的框架，通过自我反思令牌（Self-reflection tokens）来训练和控制任意LM。它主要分为三个步骤：检索、生成和批评。

1. **检索**：首先，Self-RAG解码检索令牌（retrieval token）以评估是否需要检索，并控制检索组件。如果需要检索，LM将调用外部检索模块查找相关文档。
    
2. **生成**：如果不需要检索，模型会预测下一个输出段。如果需要检索，模型首先生成批评令牌（critique token）来评估检索到的文档是否相关，然后根据检索到的段落生成后续内容。
    
3. **批评**：如果需要检索，模型进一步评估段落是否支持生成。最后，一个新的批评令牌（critique token）评估响应的整体效用。


Retrieve 决定是否进行检索，如果进行检索的话，各段内容对应的相关程度 IsREL、自我支持程度 IsSUP、有用程度 IsUSE 共同组成排序的分数标准。作者把各个维度分别分了几档做离散的预测，如下表所示：

![](img/Pasted%20image%2020231126161408.png)

以上是方法的骨架，接下来的关键在于如何构造包含reflection tokens 的训练数据来训练 Self-RAG。 Self-RAG 的训练包括三个模型：检索器（Retriever）、评论家（Critic）和生成器（Generator）。

- GPT-4 收集种子数据：对四种类型的 reflection tokens，各用 GPT-4 标注 4k-20k 个从开源的 QA 和知识问答数据中收集的样本； 
    
- 知识蒸馏，训练 critric model: 在第 1 步的训练数据上微调开源大模型，如LLaMa2-7B，称为 critic model； 
    
- 为生成模型生成训练数据：使用上述的 critic model联合检索模块，为最后的生成模型构造模拟整个 Self-RAG 推理过程的训练集（两个例子如下图所示），约 150k 大小； 
    
- 训练生成模型：在第 3 步生成的训练数据上使用标准的下一个 token 预测目标来训练生成模型，文中为 LLaMa2-7B和 13B，以学习生成 自然延续(continuations)以及特殊 tokens (用来检索或批评其自己的生成内容). 最后推理时只需要该模型，不需要 critic model。

![](img/Pasted%20image%2020231126161440.png)

疑问：是否相关、是否自我支持、是否有用这几个客观标准，用 GPT-4 标注是合理的，但是否需要检索增强，也就是上面的 Retreive 这个 reflection token，是和生成模型本身的能力相关的，GPT-4 不需要检索就能回答的问题，可能 LLaMa2 就需要检索，这里这样蒸馏是否合理有待讨论。

推理：

![](img/Pasted%20image%2020231126161353.png)

## 相关研究


## 核心亮点

Self-RAG 通过学习生成反思令牌，使得在不需要训练LMs的情况下为各种下游任务或偏好量身定制模型行为。特别是：

1. 它可以适应性地使用检索令牌进行检索，因此模型可以自发判断是不是有必要进行检索。
    
2. 它引入了多种细粒度的批评令牌，这些令牌用于评估生成内容的各个方面的质量。在生成过程中，作者使用期望的批评令牌概率的线性插值进行segment级的beam search，以在每一个时间步骤中确定最佳的K个续写方案。

## 实验

Self-RAG 在一系列开放域 QA 和生成任务上都能比普通的检索增强 LLaMa2 取得明显提升。

![](img/Pasted%20image%2020231126161512.png)

消融实验，可以看到：每一个组件和技术在Self-RAG中都起到了至关重要的作用。调整这些组件可以显著影响模型的输出性质和质量，这证明了它们在模型中的重要性。

![](img/Pasted%20image%2020231126163456.png)

## 未来方向



## 主要收获


## 参考资料
