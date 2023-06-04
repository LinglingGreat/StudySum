---
title: RecurrentGPT
created: 2023-06-04
tags: 长文本
type: 论文
papername: RECURRENTGPT Interactive Generation of (Arbitrarily) Long Text
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution: ETH
---

## 论文基本信息

标题：RECURRENTGPT: Interactive Generation of (Arbitrarily) Long Text

作者：Wangchunshu Zhou,  Yuchen Eleanor Jiang,  Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, Mrinmaya Sachan

链接： https://arxiv.org/abs/2305.13304

代码： https://github.com/aiwaves-cn/RecurrentGPT

在线 Demo：

https://www.aiwaves.org/recurrentgpt (长篇小说写作)

https://www.aiwaves.org/interactivefiction (交互式小说)

框架图：


## 背景
基于 Transformer 的大语言模型最明显的限制之一就是输入和输出的长度限制。虽然输入端的长度限制可以通过 VectorDB 等方式缓解，输出内容的长度限制始终是限制 ChatGPT 等大语言模型广泛应用于长内容生成的关键障碍。为解决这一问题，过去很多研究试图使用基于向量化的 State 或 Memory 来让 Transformer 可以进行循环计算。这样的方法虽然在长文本建模上展现了一定的优势，但是却要求使用者拥有并可以修改模型的结构和参数，这在目前闭源模型遥遥领先的大语言模型时代中是不符合实际的。

RecurrentGPT 则另辟蹊径，是利用大语言模型进行交互式长文本生成的首个成功实践。它利用 ChatGPT 等大语言模型理解自然语言指令的能力，通过自然语言模拟了循环神经网络（RNNs）的循环计算机制。

## 核心亮点

如图 2 所示，在每一个时间步中，RecurrentGPT 会接收上一个时间步生成的内容、最近生成内容的摘要（短期记忆），历史生成内容中和当前时间步最相关的内容 (长期记忆)，以及一个对下一步生成内容的梗概。RecurrentGPT 根据这些内容生成一段内容，更新其长短时记忆，并最后生成几个对下一个时间步中生成内容的规划，并将当前时间步的输出作为下一个时间步的输入。这样的循环计算机制打破了常规Transformer 模型在生成长篇文本方面的限制，从而实现任意长度文本的生成，而不遗忘过去的信息。

![](img/Pasted%20image%2020230604175146.png)

具体来讲。作者们设计了如图 2 所示的 prompt 去指导和规范循环的生成：

![](img/Pasted%20image%2020230604175205.png)

首先指明任务，比如写小说，并说明在输入部分会给出的内容：上一步生成的段落（图中 Ot-1）、当前维持的近期生成内容的摘要，即短期记忆（图中 ht-1），所有生成内容中和当前时间步相关程度最高的几个段落，即短期记忆（图中 ct-1），以及对接下来生成内容的规划（图中 xt-1）。

接着在 prompt 中给 ChatGPT 提出要求：首先基于当前的输入生成一个新的段落，接着对维护的短期记忆进行修改，同时在对短期记忆修改时作者们指示大语言模型首先分析短期记忆中哪些内容对于后续创作不再重要以及新生成的内容中哪些会对后续生成有所影响，之后相应地在地短期记忆库中去去除无用的信息并增添新的信息，从而保持短期记忆不会因为迭代的轮数增加而变得过长。最后要求 ChatGPT 基于当前的情节铺设，给出三个逻辑顺承又有趣的新的情节的规划。

在提出要求后，作者在结尾再次精心设计了 prompt 来规范 ChatGPT 的输出，并重申了当前小说写作的情景。这个好处是让 ChatGPT 生成的内容更具备像小说那样的细节，而不是在每一轮的迭代中，快速地完成情节的叙述。

![](img/Pasted%20image%2020230604175253.png)

在实际使用中，内容创作者只需先选择一个主题，然后简单地描述一下要生成的内容的背景设定和大纲，剩下的工作就可以交给 RecurrentGPT。每一个它将自动生成第一段，并提供几个可能的选项（plan）供创作者继续写故事。创作者可以选择一个选项、对某个选项进行修改或者自己编辑一个新的选项。这个流程能显著提高内容创作者的效率。

这个新的长文本生成范式将带给所有内容创作者和读者一种全新的体验。首先，相比现有的方法，RecurrentGPT 有更强的可解释性，因为用户可以观察和编辑自然语言记忆，这使得用户可以更清晰地理解这个框架是如何工作的。其次，用户可以直接影响生成内容的方向，让整个写作过程变得更加有趣。

除了作为 AI 内容生成 (AIGC) 的工具以外，RecurrentGPT 可以直接作为交互式小说，直接与消费者互动，跳过了内容创作者使用 AI 进行内容创作的步骤。这让消费者的体验更直接有趣，并且带来更丰富的可能性。作者们将这样的生成式 AI 的使用范式称之为 (AI as Content, AIAC), 也就是 “AI 即内容”。而 RecurrentGPT 则是通往这个范式的第一步。

## 实验
在实验中，作者们将 RecurrentGPT 与之前的 SoTA 长文本生成方法，在统一使用 ChatGPT 作为基座模型的情况下，在长文本（6000 单词）和较长文本（3000 单词）的设定下进行 pair-wise 的人工比较。

![](img/Pasted%20image%2020230604175339.png)

在上述一系列测试中，RecurrentGPT 无论是在科幻、浪漫、幻想、恐怖、神秘还是惊悚小说的生成上，都被人类读者认为更有趣和连贯。



## 参考资料

[ChatGPT能写长篇小说了，ETH提出RecurrentGPT实现交互式超长文本生成](https://mp.weixin.qq.com/s/9zDyyqaHA8Ghnh96f2IOLg)