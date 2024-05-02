---
title: MoI
created: 2024-05-02
tags:
  - 多任务
  - SFT
type: 论文
papername: Mixture-of-Instructions - Comprehensive Alignment of a Large Language Model through the Mixture of Diverse System Prompting Instructions
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 阿里
---

## 论文基本信息

标题：[Mixture-of-Instructions: Comprehensive Alignment of a Large Language Model through the Mixture of Diverse System Prompting Instructions](https://papers.cool/arxiv/2404.18410)

作者：[Bowen Xu](https://arxiv.org/search/?searchtype=author&query=Bowen%20Xu) ; [Shaoyu Wu](https://arxiv.org/search/?searchtype=author&query=Shaoyu%20Wu) ; [Kai Liu](https://arxiv.org/search/?searchtype=author&query=Kai%20Liu) ; [Lulu Hu](https://arxiv.org/search/?searchtype=author&query=Lulu%20Hu)

链接：

代码：

框架图：


## 背景

这篇论文试图解决大型语言模型（LLMs）在多任务对齐方面的挑战。具体来说，它关注以下几个关键问题：

1. **多任务对齐**：现有的对齐方法主要针对单一任务，如多轮对话、编程、数学问题解决和工具使用。然而，实际应用中，AI产品通常需要融合这些能力才能在现实世界场景中有效工作。
    
2. **计算资源需求**：对LLMs进行适当的对齐需要大量的计算资源，这突显了对一种更健壮、高效且全面的多任务对齐方法的需求，以确保改进的生成性能。
    
3. **知识冲突**：在特定任务中，如编程、数学和推理，即使在访问高质量、特定领域的数据集后，LLMs也经常表现不佳。这是因为在监督式微调（SFT）期间，新旧知识之间可能存在冲突。
    
4. **系统提示的重要性**：论文发现，改变系统提示可以解决知识冲突，提高任务性能。
    

为了应对这些挑战，论文提出了一种名为“Mixture-of-Instructions (MoI)”的新技术，它通过指令串联结合多样化的系统提示来提高语言模型的对齐效率。此外，论文还编译了一组包含七个基准数据集的多样化集合，以严格评估MoI增强语言模型的对齐效果。

通过MoI方法的应用，论文开发了Qwen-SFT-MoI模型，该模型在编程、数学和工具使用任务中的生成能力显著提高，证明了MoI策略在进一步对齐已经进行SFT的模型方面的有效性。

## 相关研究

论文中提到了与大型语言模型（LLMs）相关的几个研究领域，包括：

1. **数学推理**：研究如何通过微调开源模型或从高级专有LLMs中获取洞见来解决复杂的数学任务。例如，WizardMath和MAmmoTH项目。
    
2. **代码生成**：特别为代码生成任务设计的LLMs，如StarCoder和DeepSeek-Coder，或者通过对通用模型进行微调来派生出的代码生成LLMs，如CodeLlama和WizardCoder。
    
3. **工具使用**：LLMs在现实世界场景中必须能够熟练选择和应用来自众多API的工具。例如，Gorilla项目将LLMs与广泛的API集配对，而ToolAlpaca和ToolLLM则提供了大量API工具的使用文档。
    
4. **多任务学习**：在多任务学习中，数据集偏差是一个现象，其中一个或多个任务可能会将模型参数拉向不同方向，从而影响多任务学习性能。Mixture-of-Experts (MoE)框架提供了一种处理这种机制的方法。
    

这些研究领域为论文提出的Mixture-of-Instructions (MoI)方法提供了背景和动机，MoI方法旨在通过多任务学习来提高LLMs在多个领域的性能，同时解决数据集偏差问题。论文通过这些相关研究展示了MoI方法的有效性，并与现有的一些方法进行了比较。


## 核心亮点

论文通过提出一种名为“Mixture-of-Instructions (MoI)”的方法来解决大型语言模型（LLMs）在多任务对齐方面的挑战。MoI方法包括以下几个关键步骤：

1. **系统提示的重要性**：论文首先发现，改变系统提示可以解决知识冲突，提高特定任务的性能。例如，在编程任务中，将系统提示从默认的“你是一个有帮助的助手”改为“你是一个程序员”，可以提高模型在编程任务上的表现。

![](img/Pasted%20image%2020240502174440.png)
    
2. **平衡采样多样化的系统提示指令**：为了使模型能够同时获得不同领域的能力，论文提出了一种平衡采样方案，该方案将不同任务的数据集按照它们的系统提示指令进行组合，以实现多任务学习。

3. **指令重排序与默认系统提示**：为了使语言模型在只有一个默认系统提示的情况下也能擅长跨任务，论文提出了一种重排序混合指令的方法，将默认系统提示放在混合指令的开始部分。通过注意力机制的分析，论文证明了这种方法可以有效地将不同任务中学到的知识转移到默认提示下。

![](img/Pasted%20image%2020240502175110.png)

![](img/Pasted%20image%2020240502174928.png)
    
4. **MoI方法**：最终，论文提出了一种新的SFT（Supervised Fine-Tuning）方案，即MoI，它结合了平衡采样和指令重排序策略。MoI方法不仅减少了多任务学习中的数据集偏差，还有效地将从各种系统提示中学到的知识转移到默认提示下，显著提高了模型的对齐和性能。
    
5. **实验验证**：论文通过在多个基准数据集上的实验，验证了MoI方法在多任务学习中的有效性。通过与Qwen-7B-chat模型的比较，展示了MoI增强模型在编程、数学和工具使用任务上的显著改进。
    
6. **开发Qwen-SFT-MoI模型**：通过在高质量的数据集上应用MoI方法，论文开发了Qwen-SFT-MoI模型，该模型在多个基准测试中表现出色，证明了MoI方法在提高SFT模型性能方面的潜力。
    

通过这些方法，论文成功地展示了如何通过多任务学习和系统提示的策略来提高LLMs在多个任务上的性能，并解决了多任务对齐的挑战。



## 实验

数据集

![](img/Pasted%20image%2020240502175336.png)


![](img/Pasted%20image%2020240502175430.png)

这里seq是非pack方式的SFT，concat是pack方式的SFT。balanced是从所有数据中平衡采样.

**Ablation study**

System prompt的影响

![](img/Pasted%20image%2020240502175723.png)

Dataset bias

![](img/Pasted%20image%2020240502175916.png)

和开源模型比较

![](img/Pasted%20image%2020240502180042.png)



## 未来方向



## 主要收获


## 参考资料
