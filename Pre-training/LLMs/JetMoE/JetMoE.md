---
title: JetMoE
created: 2024-04-13
tags:
  - 专家模型
type: 论文
papername: JetMoE Reaching Llama2 Performance with 0.1M Dollars
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - MIT
---

## 论文基本信息

标题：JetMoE: Reaching Llama2 Performance with 0.1M Dollars

作者： [Yikang Shen](https://arxiv.org/search/?searchtype=author&query=Yikang%20Shen) ; [Zhen Guo](https://arxiv.org/search/?searchtype=author&query=Zhen%20Guo) ; [Tianle Cai](https://arxiv.org/search/?searchtype=author&query=Tianle%20Cai) ; [Zengyi Qin](https://arxiv.org/search/?searchtype=author&query=Zengyi%20Qin)

链接：

代码：

框架图：

![](img/Pasted%20image%2020240413172754.png)

![](img/Pasted%20image%2020240413173525.png)
## 背景
这篇论文介绍了JetMoE-8B，这是一种基于Mixture-of-Experts (MoE) 架构的大型语言模型（LLM），旨在解决大型语言模型（LLMs）日益增长的资源需求问题。这种资源需求成为发展强大且易于访问的超人类智能的主要障碍。具体来说，论文试图解决以下问题：

1. **成本效益**：展示如何在有限的预算（小于10万美元）下训练出一个性能卓越的LLM，从而提高LLM训练的成本效益。
    
2. **资源优化**：通过稀疏激活机制，JetMoE-8B在注意力和前馈层都实现了稀疏激活，减少了每个输入标记的激活参数数量，从而降低了推理计算量。
    
3. **开放性和可访问性**：使用公开数据集和开源训练代码，使模型对学术界友好，便于在资源有限的研究环境中进行微调和使用。
    
4. **性能提升**：尽管成本低廉，JetMoE-8B在多个基准测试中表现出色，超越了其他更大的模型，如Llama2-7B和Llama2-13B-Chat。
    

论文通过这些方法，旨在推动大型语言模型的发展，使其更加高效、易于访问，并促进AI研究社区的合作与进步。


## 相关研究
这篇论文提到了多个与JetMoE-8B相关的研究领域和具体工作，包括：

1. **大型语言模型（LLMs）**：论文提到了其他几个大型语言模型，如Llama、Pythia、GPT-3和Mistral，这些模型在某些任务上超越了人类性能，但仍然存在效率和灵活性的问题。
    
2. **Mixture-of-Experts (MoE) 架构**：MoE架构是一种流行的解决方案，它通过在Transformers中应用MoE来实现参数扩展，同时保持计算成本适中。论文中提到的相关工作包括Deepseek MoE、Mixtral 8x7B、Grok-1和DBRX等。
    
3. **开源MoE模型**：论文提到了开源社区尝试训练MoE模型的努力，例如OpenMoE，但其性能仅与具有类似激活参数的弱密集模型相当。
    
4. **稀疏激活机制**：论文中提到了Sparsely-gated Mixture-of-Experts (SMoE) 架构，这是一种在注意力和前馈层都应用稀疏激活的机制，以减少计算成本。
    
5. **数据集和训练策略**：论文讨论了使用公开数据集和训练代码的重要性，并提到了如RefinedWeb、StarCoder、Dolma和The Pile等数据集，以及如何通过精心混合这些数据集来训练JetMoE-8B。
    
6. **模型对齐和微调**：论文中还提到了使用Distilled Supervised Fine-Tuning (dSFT) 和 Distilled Direct Preference Optimization (dDPO) 等技术来对齐和微调模型，以提高其性能。
    
7. **评估和基准测试**：论文中提到了在OpenLLM Leaderboard和其他领域的任务中评估JetMoE-8B的性能，包括物理推理、社会推理、问答、数学、常识推理、语言建模、阅读理解等。
    

这些相关研究表明，JetMoE-8B是在现有研究的基础上，通过结合多种技术和方法，实现高效、低成本的大型语言模型训练的尝试。



## 核心亮点

1. **高效的MoE架构**：JetMoE-8B采用了Sparsely-gated Mixture-of-Experts (SMoE) 架构，该架构在注意力和前馈层都实现了稀疏激活。这意味着模型在处理输入时，只有一部分专家被激活，从而大幅减少了计算成本。
    
2. **低成本训练**：使用混合的开源数据集和有限的计算资源（96个H100，30,000 H100 GPU小时和1.25T tokens），以及不到10万美元的预算，实现了模型的训练。
    
3. **开放和透明**：论文详细公开了所有的训练参数和数据混合比例，使用公开数据集和开源训练代码，使得模型对学术界友好，并便于在资源有限的环境中复现和微调。

使用多种真实世界和合成数据集进行预训练，包括RefinedWeb、StarCoder、Dolma、The Pile、OpenWebMath等，以及一些特定领域的数据集，如数学和编程问题。1.25T tokens of primarily English data。

预训练数据集包括：
- 真实数据RefinedWeb、StarCoder、Dolma、The Pile、Proof-Pile-2、OpenWebMath、StackMathQA、OpenAssistant、xP3x、CommitPackFT
- 合成数据OpenHermes 2.5、UltraTextbooks、UltraChat 200k、TemplateGSM、Magicoder-Evol-110K and Magicoder-OSS-75K、Evol-Code Alpaca、Code-290k-ShareGPT

训练代码采用Megatron框架，集成了Megablock，还支持MoA（Mixture of Attention heads）和z-loss（为了平衡MOE训练）。在训练中的模型并行中，使用管道并行而不是专家并行。根据1B transformer语言模型的常见实践，选择了一系列的超参数，包括模型层数、激活参数数量、专家数量、头部数量等，并进行了相应的调整以适应JetMoE-8B的架构。训练参数如下

![](img/Pasted%20image%2020240413173901.png)

    
4. **两阶段训练过程**：使用AdamW优化器，并采用Warmup-Stable-Decay (WSD) 学习率调度策略，以在训练过程中平衡模型的收敛速度和最终性能。JetMoE-8B采用了两阶段训练策略，第一阶段使用高质量的数据集进行模型预热和稳定学习率训练，第二阶段在降低学习率的同时加入更多高质量数据，以进一步提升模型性能。

AdamW优化器，最大学习率5e-4，batch size 4M tokens，sequence length=4096

Phase 1 (warmup and stable learning rate): The dataset includes RefinedWeb, Starcoder, The Pile, peS2o from Dolma, and OpenWebMath.

Phase 2 (decay learning rate): We include additional high-quality data to further improve the model’s performance.

![](img/Pasted%20image%2020240413174735.png)

![](img/Pasted%20image%2020240413174746.png)
    
5. **模型对齐技术**：通过Distilled Supervised Fine-Tuning (dSFT) 和 Distilled Direct Preference Optimization (dDPO) 技术对模型进行对齐和微调，以提高模型在特定任务上的表现和输出的一致性。整个对齐过程花费60 H100 GPU hours.

SFT数据集：UltraChat 200k、Airoboros-3.2、Code-Feedback、Orca-math-word-problems-200k、SystemChat、Capybara。学习率2e-5，batch size=128，epoch=3, AdamW

DPO数据集：UltraFeedback。学习率5e-7，batch size=128，epoch=1,AdamW


6. **综合评估**：在多个基准测试中评估JetMoE-8B的性能，包括OpenLLM Leaderboard和其他领域的任务，如物理推理、社会推理、问答、数学、常识推理、语言建模、阅读理解等，确保模型的广泛适用性和有效性。

![](img/Pasted%20image%2020240413175933.png)

![](img/Pasted%20image%2020240413180134.png)

MT-Bench的temperature遵循FastChat的官方实现。


    
7. **效率分析**：分析JetMoE-8B在推理计算上相比于其他模型的效率提升，特别是在激活参数数量和计算成本方面的减少。
    



## 未来方向
论文中提到了一些可以进一步探索的点，包括但不限于：

1. **模型架构的消融研究**：由于预算限制，论文中没有进行关于模型架构的消融研究。未来的工作可以通过系统地改变模型的各个组成部分来评估它们对最终性能的实际贡献。
    
2. **数据集和训练策略的优化**：尽管使用了精心挑选的数据集和训练策略，但仍有可能通过进一步优化数据混合和训练过程来提高模型性能。例如，可以尝试不同的数据清洗和预处理技术，或者探索更先进的训练策略。
    
3. **多语言和跨领域能力**：虽然JetMoE-8B主要针对英语数据进行了训练，但未来的研究可以探索如何将其扩展到多语言和跨领域的场景中。
    
4. **模型压缩和加速**：研究如何通过模型压缩技术（如知识蒸馏、模型剪枝等）来减小模型大小和提高推理速度，同时保持或仅轻微牺牲性能。
    
5. **更复杂的对齐技术**：进一步探索和开发新的模型对齐技术，以提高模型的安全性、公平性和透明度。
    
6. **更广泛的应用场景**：将JetMoE-8B应用于更多样化的任务和领域，如医疗、法律、教育等，以评估其在专业领域的适用性和有效性。
    
7. **长期影响和伦理考量**：研究大型语言模型对社会、文化和经济的长期影响，以及如何确保这些模型的负责任和伦理使用。
    
8. **开源社区的合作**：鼓励开源社区的合作和知识共享，以共同推动大型语言模型的发展和应用。
    

这些探索点不仅可以帮助提高JetMoE-8B模型的性能和可用性，还可以为整个人工智能领域的进步做出贡献。


## 主要收获


## 参考资料
