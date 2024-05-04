---
title: ContinualLearning
created: 2024-03-01
tags:
  - ContinualLearning
type: 论文
papername: Investigating Continual Pretraining in Large Language Models Insights and Implications
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - CohereAI
---

## 论文基本信息

标题：[Investigating Continual Pretraining in Large Language Models: Insights and Implications](https://papers.cool/arxiv/2402.17400)

作者：[Çağatay Yıldız](https://arxiv.org/search/?searchtype=author&query=%C3%87a%C4%9Fatay%20Y%C4%B1ld%C4%B1z) ; [Nishaanth Kanna Ravichandran](https://arxiv.org/search/?searchtype=author&query=Nishaanth%20Kanna%20Ravichandran) ; [Prishruit Punia](https://arxiv.org/search/?searchtype=author&query=Prishruit%20Punia) ; [Matthias Bethge](https://arxiv.org/search/?searchtype=author&query=Matthias%20Bethge) ; [Beyza Ermis](https://arxiv.org/search/?searchtype=author&query=Beyza%20Ermis)

链接：

代码：

框架图：


## 背景

key insights：(i) 当领域序列显示语义相似性时，与独立微调某个领域相比，持续预训练使法学硕士能够更好地专注于当前领域，(ii) 跨不同领域的培训可以增强向后和向前的知识迁移， (iii) 较小的模型对持续预训练特别敏感，表现出最显着的遗忘率和学习率。(iv) CL 的后期阶段更容易忘记, (iv) 持续的预训练可以增强下游性能

这篇论文研究了大型语言模型（LLMs）在持续学习（Continual Learning, CL）领域的发展，特别是关注于开发高效且可持续的训练策略。论文的主要目标是解决以下几个问题：

1. **持续域适应性预训练（Continual Domain-Adaptive Pretraining）**：这是一个旨在让LLMs能够整合来自不同领域的新信息，同时保留之前学习的知识，并在不依赖于特定领域识别的情况下增强跨领域知识转移的过程。
    
2. **适应变化的数据环境**：研究评估了LLMs在实际场景中适应不断变化的数据环境的能力，这与以往研究主要集中在有限的任务或领域选择上，主要解决遗忘问题不同。
    
3. **模型大小对学习效果的影响**：论文探讨了模型大小对学习效率和遗忘的影响，以及新出现的领域进展和相似性如何影响模型内的知识转移。
    
4. **建立更现实的基准**：研究提出了一个新的基准测试，用于衡量LLMs适应这些不断变化的数据环境的能力，并提供了一个全面的评估框架。
    
5. **指导未来研究方向**：通过这些发现，论文旨在为LLMs中的CL研究建立一个更现实的基准，并可能在未来的研究方向上发挥关键作用。
    

总的来说，这篇论文试图通过持续域适应性预训练来提高LLMs的适应性和知识转移能力，同时减少训练成本，以应对LLMs规模和复杂性的增加所带来的财务和生态成本问题。


## 相关研究
**A**: 论文中提到了与持续学习（CL）在大型语言模型（LLMs）相关的两个主要研究方向：

1. **持续微调（Continual Fine-Tuning）**：这一领域的研究主要关注于在一系列下游任务上微调LLMs，以减轻遗忘问题。这些方法通常涉及在新的数据流到来时，对模型进行增量式的微调，而不是完全重新训练。相关研究包括：
    
    - Luo et al. (2023a, 2023b) 研究了在一系列下游分类任务上持续学习过程中的遗忘和分布漂移。
        
    - Scialom et al. (2022) 对LLM进行了指令式微调。
        
    - Khan et al. (2022) 提出了基于适配器的微调策略。
        
    - Zhang et al. (2022) 提出了为序列生成器（如LLM）添加新模块的方法。
        
    - Razdaibiedina et al. (2023) 提出了渐进式提示，在学习过程中逐渐增加提示的数量。
        
    - Wang et al. (2023) 提出了学习正交适配器以最小化干扰的方法。
        
    - Qin et al. (2022) 提出了一种名为ELLE的高效终身预训练方法。
        
2. **持续域适应性预训练（Continual Domain-Adaptive Pretraining）**：这一研究方向与本文的工作更为接近，旨在通过持续预训练来适应新领域，而不需要从头开始进行全面的重新训练。相关研究包括：
    
    - Gururangan et al. (2020) 提出了一种用于领域适应性持续预训练的增长混合专家架构。
        
    - Chen et al. (2023) 研究了基于逐渐增长的混合专家（MoE）架构的在线预训练语料库序列的终身学习。
        
    - Ke et al. (2023a) 展示了RoBERTa模型的软掩蔽梯度机制在领域适应性预训练中的有用性。
        
    - Cossu et al. (2022) 研究了在十个领域上持续预训练的特性。
        
    - Jin et al. (2021) 在领域递增的研究论文流和按时间顺序排列的推文流上持续预训练RoBERTa-base。
        
    - Gupta et al. (2023) 研究了持续预训练的不同热身策略。
        
    - Fisch et al. (2023) 提出了一个任务序列的基准，这些序列可能导致正向和负向转移，并提出了一种简单的策略以实现稳健的正向转移。
        

这些相关研究为本文提供了背景和对比，展示了在LLMs中实现持续学习的不同方法和策略。本文的研究通过在更广泛的领域集上进行持续预训练，进一步探索了知识保留、新信息保留和知识转移的动态。

## 实验

### 数据集

基于Massively Multi-Domain Language Modeling Dataset (M2D2)，它包含了来自Wikipedia和Semantic Scholar(S2ORC) database的一共236个domain，8.5B tokens。

语料库分为两个级别：L1 域和 L2 域。在 S2ORC 语料库中，L1 域指的是广泛的学术研究领域，例如计算机科学和物理学，而 L2 域对应于这些领域内的特定 arXiv 类别，例如计算机科学下的“计算和语言”。对于维基百科，L1 域代表主要类别，L2 域包含每个 L1 域内的类别页面。为了保持实验中的平衡和计算效率，我们排除了超过 5GB 数据的领域，例如 Medicine。最终，我们在研究中使用了 159 个领域。

![](img/Pasted%20image%2020240502225712.png)

使用sentence-bert，每个domain 10k samples, OpenWebText 50k samples，计算task embedding, 然后得到cross-domain similarity.

![](img/Pasted%20image%2020240502225727.png)


在M2D2数据集上，对不同大小的GPT模型（GPT2-small, GPT2-medium, GPT2-large, GPT2-xlarge）进行了持续预训练。这些模型在一系列领域特定的语料库上进行训练，以适应新的领域。
### 评估

数据集的每个domain被划分成train, validation, test. validation和test集合包括超过1M tokens，能够准确评估该领域。计算困惑度。

使用了多种评估指标来衡量模型的性能，包括
- 零样本（Zero-Shot, ZS）：原始的未进行任何领域适应的预测结果
- 微调（Fine-Tuned, FT）基线：针对每个domain进行微调后的模型
- ZS 充当基本基线，确保我们的模型具有基本的能力水平，而 FT 为我们的持续学习方法设定了目标性能标准。实现比 FT 基线更低的困惑度是持续预训练的主要目标，这意味着它可以有效地适应新领域，而不会丢失以前的知识。
- 持续预训练困惑度（Continual Pretraining Perplexity, CPT）：评估模型在最新训练领域的表现，可以帮助我们了解模型随着时间的推移适应新信息的情况
- 最后检查点（Last Checkpoint, LC）：针对所有训练领域的最后一个检查点，以检查最终模型在广泛的学科中保留和转移知识的能力
- 对之前见过的领域（backward transfer）和未来没见过的领域（forward transfer）的评估。

![](img/Pasted%20image%2020240502231416.png)

### 训练

 **任务顺序的影响**：实验中考虑了两种任务（领域）顺序：相似顺序（similar-order）和随机顺序（random-order）。相似顺序是根据领域之间的相似性进行排序，从culture domain开始（和OpenWebText最相似），然后到下一个最相似的domain。而随机顺序则是随机排列训练领域。

**Analysis of continual pretraining and the final model**

- 最终模型比零样本实现了更好的困惑度（这里是所有domain的平均困惑度）。平均而言，整个 CL 中积累的知识不会损害对所学领域的预测。值得注意的是，随机化训练序列会比相似阶域序列产生更有利的平均困惑度。Table 4强调了 GPT 系列倾向于忘记前面的部分，而对后面的部分却感到越来越困惑
- 当域按语义排序时，CPT 比标准 FT 更有利。模型在当前任务上的表现与其起始检查点有着内在的联系。
- 最终性能与模型大小相关。我们观察到，在四分之三的评估场景中，GPT2-small 从持续预训练中获益最多。


![](img/Pasted%20image%2020240502231634.png)


**Backward transfer**

![](img/Pasted%20image%2020240502232530.png)

平均backward transfer性能在很大程度上取决于域顺序。一方面，我们从未观察到顺序训练的正向后迁移，并且当我们切换训练portion时，测试困惑度显着降低。另一方面，与零样本基线相比，随机顺序的训练通常会增强测试的困惑度。与初始模型 M0 相比，最显着的改进是在训练早期观察到的，并在大约 25 个任务后达到饱和。

![](img/Pasted%20image%2020240502232852.png)

Similar training order facilitates backward transfer to recent past。当后续领域具有高度概念重叠时，我们观察到正向后迁移最多 30 个领域。当然，随着最近的训练领域与测试领域变得显着不同，随着时间的推移，这种改进会变得更糟。值得注意的是，最小的 GPT 模型显示出最显着的性能波动，既经历了最高的增益，也经历了最明显的下降。

Longer CL improves backward transfer if domain order is randomized。

![](img/Pasted%20image%2020240502233225.png)

LLMs forget more in the later stages of continual learning。


**Forward transfer**

![](img/Pasted%20image%2020240502233258.png)

Positive forward transfer is rarely possible in similar training order。

Random-order training enables positive transfer to S2ORC。

![](img/Pasted%20image%2020240502233429.png)

    
 **下游任务性能**：通过在BIG-Bench基准测试上的一系列任务（如算术、一般知识、物理、计算机科学算法和少量样本自然语言生成）来评估模型在持续预训练后在不同任务上的性能。

结果表明，在与这些任务相关的领域进行持续预训练通常会提高模型性能，而在不相关领域进行预训练通常会导致遗忘，从而对模型的初始任务熟练程度产生负面影响。如图 7 所示，当模型在 Wiki 域上持续训练时，算术任务性能持续下降，然后在切换到 S2ORC 域后得到改进（非线性科学和天体物理学域除外）。相比之下，一般知识任务的性能在 Wiki 领域训练中有所提高，但在 S2ORC 训练中有所下降，除了 CS 和统计领域略有增加。

## 消融实验

由于不断学习，泛化能力可能会恶化

![](img/Pasted%20image%2020240502233803.png)
    
- **批量大小的影响**：研究了批量大小（16到64）对模型学习动态的影响。当以随机顺序训练时，尽管批量大小不同，连续预训练和最后检查点的性能实际上保持不变。在类似的顺序中，较小的批量大小有助于改善持续预训练的困惑度，但会恶化最后一个检查点的性能。我们假设采取更多的梯度步骤有助于模型更好地适应当前任务，同时促进忘记旧任务。
	
- **数据大小平衡**：探讨了在L2领域中平衡训练数据大小对性能的影响。建议使用手头的所有数据，而不是为了平衡训练而遗漏一些数据。
	
- **Wiki和S2ORC部分的交换**：交换了Wiki和S2ORC部分的训练顺序，以验证先前发现的一致性。我们将这些部分交换为类似顺序的训练，即首先在 S2ORC 上训练，然后在 Wiki 部分上训练。可以说，这种训练顺序仍然遵循概念相似性；因此，它使我们能够看看我们之前的发现是否仍然成立。图 15 的左图显示持续预训练的困惑度几乎保持不变。然而，最后一个检查点的困惑度显着变化：虽然 S2ORC 部分的性能大幅下降，但我们观察到 Wiki 部分的相反效果。与我们之前的发现一致，我们得出的结论是，在较旧的域/部分上进行测试时，检查点的表现更差
	
- **编码器-解码器模型的实验**：使用RoBERTa模型重复了所有实验，以深入了解不同架构的行为。首先也是最重要的是，即使在比较零样本性能时，RoBERTa-large 总是比 RoBERTa-base 获得更差的困惑度。其次，两种模型大小都几乎没有遗忘，甚至在概念上不相关的领域进行训练似乎也能提高性能。最后，仅保留第一个领域（文化和艺术）可以促进向所有领域的前向转移

## Discussion

![](img/Pasted%20image%2020240502234828.png)

![](img/Pasted%20image%2020240502234842.png)
## 主要收获


## 参考资料
