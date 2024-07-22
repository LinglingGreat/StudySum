---
title: Branch-Train-MiX
created: 2024-03-18
tags:
  - 专家模型
type: 论文
papername: Branch-Train-MiX- Mixing Expert LLMs into a Mixture-of-Experts LLM
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - Meta-FAIR
---

## 论文基本信息

标题： [Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM](https://papers.cool/arxiv/2403.07816)

作者：[Sainbayar Sukhbaatar](https://arxiv.org/search/?searchtype=author&query=Sainbayar%20Sukhbaatar) ; [Olga Golovneva](https://arxiv.org/search/?searchtype=author&query=Olga%20Golovneva) ; [Vasu Sharma](https://arxiv.org/search/?searchtype=author&query=Vasu%20Sharma) ; [Hu Xu](https://arxiv.org/search/?searchtype=author&query=Hu%20Xu) ; [Xi Victoria Lin](https://arxiv.org/search/?searchtype=author&query=Xi%20Victoria%20Lin) ; [Baptiste Rozière](https://arxiv.org/search/?searchtype=author&query=Baptiste%20Rozi%C3%A8re) ; [Jacob Kahn](https://arxiv.org/search/?searchtype=author&query=Jacob%20Kahn) ; [Daniel Li](https://arxiv.org/search/?searchtype=author&query=Daniel%20Li) ; [Wen-tau Yih](https://arxiv.org/search/?searchtype=author&query=Wen-tau%20Yih) ; [Jason Weston](https://arxiv.org/search/?searchtype=author&query=Jason%20Weston) ; [Xian Li](https://arxiv.org/search/?searchtype=author&query=Xian%20Li

链接：

代码：

框架图：


## 背景

这篇论文提出了一种名为Branch-Train-MiX (BTX) 的方法，旨在解决大型语言模型（LLMs）在多个专业领域（如编程、数学推理和世界知识等）中提高能力的训练效率问题。具体来说，它试图解决以下几个问题：

1. **训练成本高**：训练这样的LLMs需要大量的计算资源和数据，通常涉及数千个GPU和数万亿个令牌。
    
2. **通信成本**：在分布式训练中，保持多个模型副本同步的通信成本是扩展训练的主要瓶颈。
    
3. **同步训练的脆弱性**：同步训练更容易受到硬件故障的影响，单个GPU故障可能导致整个训练停止。
    
4. **专家模型的局限性**：先前的方法（如Branch-Train-Merge）虽然通过并行训练提高了效率，但最终得到的是多个独立的模型，缺乏统一的单一模型，这限制了进一步的监督微调（SFT）或基于人类反馈的强化学习（RLHF）微调，这些步骤对于提升性能和构建与人类对齐的LLMs至关重要。
    
5. **Mixture-of-Experts（MoE）方法的局限性**：MoE方法通过只激活一部分参数来减少LLMs的计算足迹，但MoE通常在完全同步的方式下训练，并且随着专家数量的增加，通信成本也会增加。
    

BTX方法通过结合Branch-Train-Merge和Mixture-of-Experts的优势，同时减少它们的不足，提供了一种更高效的训练方法。具体来说，BTX首先并行异步地训练多个专家模型，然后将这些专家的前馈参数混合到MoE层中，并对剩余参数进行平均，接着通过MoE微调阶段学习令牌级别的路由。这种方法提高了训练效率，同时保持了模型的统一性，允许进行进一步的微调和使用。

## 相关研究

这篇论文提到了几项与其研究相关的工作领域和具体研究，包括：

1. **异步并行训练**：为了减少训练深度学习系统时的通信成本，研究者们探索了减少训练工作之间通信的方法。例如，Zhang等人（2015年）介绍了一种允许模型实例在不同工作之间发散的方法，从而消除了持续同步的需要。Douillard等人（2023年）展示了通过平均权重变化来进行较少频繁的同步，结合Nesterov动量在实践中对LLMs训练效果很好。
    
2. **Branch-Train-Merge方法**：Li等人（2022a）提出了这种方法，通过完全独立的多个训练过程来进行LLMs的并行训练。每个训练过程使用特定领域的数据，因此相应的模型成为该领域的专家。最终，通过平均这些专家模型的输出分布来进行下一个令牌的预测。
    
3. **Mixture-of-Experts（MoE）方法**：MoE用于扩展深度网络，其中只有一部分参数在任何给定时间处于活动状态。例如，Shazeer等人（2017年）使用简单的Top-K路由方案来实现MoE。Fedus等人（2022年）和Lewis等人（2021年）探索了Transformer架构的MoE训练方法。Roller等人（2021年）展示了即使通过基于输入令牌的随机映射进行固定路由，没有任何学习的路由也能很好地工作。
    
4. **持续学习（Continual Learning）**：BTX方法与持续学习有关，因为它在不同的数据分布上训练领域专家，这些数据分布与用于训练种子模型的初始数据不同。这种方法通过在分支后继续训练来实现。BTX特别与参数隔离方法相关，因为我们为不同领域使用了不同的参数。
    
5. **其他相关工作**：包括Gururangan等人（2021年）的工作，他们使用领域条件的固定路由来训练特定领域的专家模型，但没有采用BTX的异步训练方法。
    

这些相关工作为BTX方法的开发提供了背景和基础，BTX在此基础上进行了改进和创新。


## 核心亮点

论文提出了一种名为Branch-Train-MiX (BTX) 的方法来解决上述问题，具体步骤如下：

1. **Branch（分支）**：从一个预训练的种子模型开始，创建多个副本，每个副本都是一个专家模型（expert LLM）。
    
2. **Train（训练）**：这些副本（即专家模型）在不同的数据子集上并行且异步地进行训练，每个数据子集对应于一个特定的知识领域，如数学、编程或维基百科。这种训练方式是高度并行的，减少了通信成本，并提高了训练吞吐量。
    
3. **MiX（混合）**：将训练好的专家模型的前馈子层（feedforward sublayers）合并到一个单一的混合专家（Mixture-of-Expert, MoE）模块中，同时使用一个路由器网络来选择每个令牌应该使用哪个专家的前馈子层。对于自注意力层（self-attention layers）和其他剩余的模块，通过简单地平均它们的权重来合并。
    
4. **MoE-finetuning（MoE微调）**：将合并后的模型在所有组合数据上进行微调，以便路由器学习如何混合专家前馈模块。这个过程称为MoE微调阶段，它使得最终的BTX模型能够像任何其他标准LLM一样进行微调或使用。
    

通过这种方法，BTX结合了Branch-Train-Merge的高效训练和Mixture-of-Experts的灵活性，同时避免了它们的缺点。BTX模型在保持较低的推理计算成本的同时，实现了在多个专业领域内的性能提升，并且相比原始模型和其他基线模型，展现了更好的准确性和效率平衡。

## 实验

论文中进行了一系列实验来验证Branch-Train-MiX (BTX) 方法的有效性，具体包括：

1. **基于Llama-2 7B模型的BTX训练**：使用Llama-2 7B模型作为种子模型，创建了三个副本，并在对应的领域数据集上继续训练以获得三个领域专家模型：数学、编程和维基百科。此外，还包括原始的Llama-2 7B模型作为一个“通才”专家，将其与领域专家模型混合成一个单一的MoE模型，并在所有用于训练四个专家的数据源上对这个MoE模型进行微调。
    
2. **基线比较**：将BTX模型与多个基线模型进行比较，包括原始的Llama-2 7B模型、在所有数据上继续训练的密集模型（Dense）、稀疏上循环（Sparse upcycling）、Branch-Train-Merge（BTM）方法以及专门化的模型（如CodeLlama 7B和Llemma 7B）。
    
3. **不同任务的性能评估**：使用多个基准测试来评估模型在不同任务上的性能，包括数学推理、代码生成、世界知识和常识推理等。这些任务覆盖了零样本和少样本的设置。
    
4. **路由方法的消融实验**：研究了不同的路由方法（如Top-k、Switch、Soft routing和Sample Top-1）对模型性能的影响，并比较了使用和不使用负载平衡（load balancing）的Top-2路由方法。
    
5. **训练策略的消融实验**：测试了不同的BTX训练策略，例如不使用负载平衡、冻结专家模型的前馈模块、混合和分割专家等。
    
6. **路由分析**：分析了不同领域任务的令牌路由决策，以了解不同专家在处理特定任务时的参与情况。
    

通过这些实验，论文展示了BTX方法在多个专业领域内提高性能的能力，同时保持了良好的计算效率和任务间的平衡性能。实验结果表明，BTX方法在大多数任务上都优于比较的基线模型，并且在某些领域任务中达到了或超过了专门化模型的性能。


## 未来方向

论文中提到了一些可以进一步探索的点，包括但不限于：

1. **更多领域的训练**：由于计算限制，论文中仅在三个领域上进行了实验。探索更多领域的训练可能会进一步放大并行专家训练的好处。
    
2. **更复杂的MoE实现**：论文中使用了简单的MoE实现，没有利用如将不同专家放置在不同GPU上并行运行等复杂技术。研究更高效的MoE实现可能缩短BTX的训练时间。
    
3. **计算分配比例的全面探索**：论文中仅比较了BTX与两种特殊变体，未来工作可以全面探索专家训练与MoE训练之间的计算分配比例。
    
4. **不同数据混合的MoE微调**：论文没有尝试MoE微调使用不同数据混合的其他实验。探索不同的数据混合策略可能会影响模型性能。
    
5. **MoE专家的领域专业化**：值得进一步研究的是，MoE专家是否应该专门化于特定领域。论文中的方法明确将专家与某些领域绑定，但这种专业化在MoE微调后是否会自然出现还有待观察。
    
6. **MoE finetuning的应用**：BTX提供了一个可以直接应用于指令微调或RLHF过程的微调组合专家的方法。论文中没有深入探讨这一点，这可以作为未来工作的方向。
    
7. **路由方法的改进**：研究更先进的路由方法，例如基于任务的路由或自适应路由，可能会提高模型在特定任务上的性能。
    
8. **模型的可解释性**：提高模型的可解释性，理解为什么和何时选择特定的专家，可以帮助我们更好地理解和信任模型的决策。
    

这些方向可以为未来的研究提供指导，并可能进一步提高大型语言模型在多个专业领域内的性能和效率。

## 主要收获


## 参考资料
