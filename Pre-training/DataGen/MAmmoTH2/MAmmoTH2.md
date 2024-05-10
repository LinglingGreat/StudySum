---
title: MAmmoTH2
created: 2024-05-10
tags:
  - 数据
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - CMU
  - Waterloo
---

## 论文基本信息

标题：## [MAmmoTH2: Scaling Instructions from the Web](https://papers.cool/arxiv/2405.03548)

作者：[Xiang Yue](https://arxiv.org/search/?searchtype=author&query=Xiang%20Yue) ; [Tuney Zheng](https://arxiv.org/search/?searchtype=author&query=Tuney%20Zheng) ; [Ge Zhang](https://arxiv.org/search/?searchtype=author&query=Ge%20Zhang) ; [Wenhu Chen](https://arxiv.org/search/?searchtype=author&query=Wenhu%20Chen)

链接：http://arxiv.org/abs/2405.03548

代码：https://tiger-ai-lab.github.io/MAmmoTH2/

框架图：

![](img/Pasted%20image%2020240510135459.png)

![](img/Pasted%20image%2020240510135040.png)

## 背景
这篇论文提出了一种新的范式，旨在从预训练的网络语料库中高效地收集自然存在的指令数据，以增强大型语言模型（LLMs）的推理能力。具体来说，论文试图解决的问题包括：

1. **数据质量和可扩展性**：指令调整（instruction tuning）是提升大型语言模型推理能力的关键方法，但现有的指令调整数据主要来自人工众包或GPT-4的蒸馏，这些数据在质量和规模上存在限制。
    
2. **成本和偏差**：人工标注的指令数据集通常规模有限，成本高昂，而且容易受到偏见和幻觉的影响。而通过GPT-4合成的指令数据虽然可以扩展规模，但可能会缺乏多样性并且容易引入幻觉。
    
3. **自然存在的指令数据的发现**：论文认为网络语料库（如Common Crawl）中已经包含了大量的、高质量的指令数据，但这些数据分散在庞大的语料库中，发现它们是一个挑战。
    
4. **自动化数据收集和优化**：为了解决上述问题，论文提出了一个三步流水线方法，包括（1）从网络语料库中召回相关文档，（2）从召回的文档中提取指令-响应对，以及（3）使用开源的大型语言模型对提取的对进行优化，以提高数据质量。
    
5. **模型性能提升**：通过在收集到的数据集上进行微调，构建了MAmmoTH2模型，并在多个推理基准测试上显著提高了性能，证明了该方法的有效性。
    

总的来说，这篇论文试图通过自动化地从网络中收集和优化指令数据，来解决现有指令调整数据集成本高、规模有限、容易受到偏见和幻觉影响的问题，并展示了如何利用这些数据来提升大型语言模型在多个任务上的推理能力。


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

![](img/Pasted%20image%2020240510140525.png)

论文通过以下步骤解决从网络语料库中高效收集高质量指令数据的问题：

1. **召回相关文档（Recall from Common Crawl）**：使用fastText模型从Common Crawl中召回相关文档。首先，通过爬取教育网站的考试数据建立种子数据集，100K作为正样本，从CC中随机抽取100K文档作为负样本，然后训练fastText模型，用于从Common Crawl中召回文档。接着，使用GPT-4进一步筛选召回的文档，最终获得约1800万份文档。
	- We employ the open-source fastText library with a vector dimension of 256 to train the model for 3 epochs, with a learning rate of 0.1, a maximum n-gram length of 3, and a maximum number of word occurrences of 3.
	- 在第一阶段，经过训练的 fastText 模型召回来自 CC 的前 100B 个文档。文档按其域（根 URL）分组，仅保留文档超过 1000 个的域。然后，我们提示 GPT-4 扫描域并自动选择可能包含指令数据的域。接下来，我们从选定域中采样文档作为正例，从非选定域和一般 Common Crawl 中采样文档作为负例，以重新训练改进的 fastText 分类器。新训练的 fastText 分类器用于召回文档。用 GPT-4 再次筛选域，最终生成 1800 万份原始文档，主要来自论坛、作业、测验和考试网站等所需领域。

![](img/Pasted%20image%2020240510140556.png)

2. **提取Q-A对（Q-A Pair Extraction）**：从召回的文档中提取问题-答案（Q-A）对。使用开源的大型语言模型（如Mixtral和Qwen）来识别和提取Q-A对，产生大约500万个候选Q-A对。
- 18M 文档中存在大量自然存在的 Q-A 对。然而，这些问答对中散布着大量的噪音，例如广告、标记、样板文件等。我们对这些原始文档的初步训练仅产生有限的收益。
- 首先，我们仔细地预处理 HTML，从召回的文档中预先提取有用的内容。这主要是基于规则的过滤，以清理站点信息、广告和 HTML 样板等。此步骤显着减少了下一阶段的文档长度。然后，我们提示 Qwen-72B (Bai et al., 2023) 从预处理的文档中识别问题和答案对。具体来说，我们提供了一些上下文示例来帮助模型理解要提取的内容。如果不存在自然问答对，我们还允许模型返回 void。在此阶段，只有 30% 的召回文档被识别为包含自然存在的 Q-A 对，从而产生大约 500 万个 Q-A 对作为下一步的候选。然而，这些候选者仍然包含大量不相关的内容和形式问题。除此之外，大部分提取的问答对也缺乏对如何得出答案的解释。
- 为了避免数据污染，过滤掉包含我们所有评估基准的问题或答案的网页。

3. **Q-A对优化（Q-A Pair Refinement）**：对提取的Q-A对进行优化，以提高数据质量。使用Mixtral-8×7B和Qwen-72B模型来重新格式化提取的Q-A对，并补充缺失的解释步骤。我们采用两种模型来增加数据集的多样性。这一步骤至关重要，以确保所收集的Q-A对的质量。


4. **构建WEBINSTRUCT数据集**：通过上述三个步骤，最终收集到1000万个指令-响应对，构建了WEBINSTRUCT数据集。

![](img/Pasted%20image%2020240510141607.png)

5. **训练MAmmoTH2模型**：使用WEBINSTRUCT数据集对不同的基础大型语言模型进行微调，构建了MAmmoTH2模型，并在多个推理基准测试上验证了其性能。
- We fine-tune these models to validate our WEBINSTRUCT at multiple scales using the LLaMAFactory (Zheng et al., 2024d) library. We use a learning rate of 5e-6 for Mistral 7B and 1e-5 for Mixtral, Llama-3 8B, and Yi 34B. The global batch size is set to 512 with a maximum sequence length of 4096. We employ a cosine scheduler with a 3% warm-up period for 2 epochs. To efficiently train the models, we utilize DeepSpeed (Rasley et al., 2020) with the ZeRO-3 stage.

6. **进一步增强性能**：为了进一步提升MAmmoTH2模型在代码生成、数学推理和指令跟随任务上的性能，论文还将其在多个开源指令数据集上进行了微调，得到了MAmmoTH2-Plus模型，并在多个推理基准测试和其他通用任务上取得了优异的性能。
- 这些数据集是根据它们与不同推理主题的相关性精心选择的。此外，我们考虑一些聊天数据集来平衡推理能力和一般聊天能力。包括OpenHermes 2.5（1M）、Code-Feedback（68000，a multi-turn code generation and refinement dataset）、Math-Plus（894K，包括MetaMathQA，Orca-Math，GPT4重写QA对等）

通过这种方法，论文成功地从网络中收集了大规模的、高质量的指令数据，并展示了如何利用这些数据来提升大型语言模型在多个任务上的推理能力。



## 实验
论文中进行了多项实验来验证MAmmoTH2模型的性能，这些实验包括：

1. **基础模型微调**：在多个不同规模的基线模型上进行微调，包括Mistral-7B、Llama3-8B、Mixtral-8×7B和Yi-34B，以验证WEBINSTRUCT数据集的有效性。
    
2. **推理基准测试**：使用多个广泛使用的推理数据集来评估模型在不同领域的推理能力，包括TheoremQA、GSM8K、MATH、ARC-C、MMLU-STEM、GPQA和BBH。
    
3. **代码生成任务**：评估模型在代码生成任务上的性能，使用HumanEval和MBPP数据集。
    
4. **通用语言理解**：在MMLU基准测试上评估模型的通用语言理解和指令跟随能力。
    
5. **聊天基准测试**：在MT-Bench、AlpacaEval 2.0和Arena Hard等聊天基准测试上评估模型的对话能力。
    
6. **扩展训练**：在公开的指令数据集上对MAmmoTH2进行进一步训练，以增强模型在代码生成、数学推理和指令跟随任务上的性能。
    
7. **模型扩展效应**：研究模型规模和损失函数对性能的影响，使用不同数量的训练样本（1M到10M）和不同的训练损失函数（LM Loss和SFT Loss）。
    
8. **案例研究**：对从数据集中提取和优化的Q-A对进行案例研究，以评估数据集的质量。
    

这些实验全面评估了MAmmoTH2模型在各种任务上的性能，并与现有的基线模型进行了比较，展示了其在推理、代码生成、语言理解和对话任务上的优势。

![](img/Pasted%20image%2020240510142503.png)

![](img/Pasted%20image%2020240510142515.png)

MAmmoTH2-8B-Plus 和 Llama3-Instruct 之间有一个有趣的比较，因为这两个模型都是从 Llama3 基础上训练的。 Llama-3-instruct 在 10M 人工注释指令数据集和公共数据集上进行训练，类似于 WEBINSTRUCT 与其他公共数据集的结合。因此，这两个模型具有很强的可比性。我们的实验表明，在基准测试中，MAmmoTH2-8B-Plus 的性能平均优于 Llama3-Instruct 6%。这一可观的收益表明WEBINSTRUCT具有极高的性价比。对于较大的模型，我们发现 MAmmoTH2-8x7B-Plus 甚至可以与仅 13B 有效参数的 Qwen-1.5-110B 的性能相媲美。这些结果证明了我们可扩展指令调整方法的有效性。

![](img/Pasted%20image%2020240510142555.png)

![](img/Pasted%20image%2020240510142939.png)

在本节中，我们研究模型缩放和损失函数对三个代表性任务的语言模型性能的影响：MATH、TheoremQA 和 ARC-C。我们使用提取的 QA 和精炼的 QA 数据来训练具有不同训练样本（1M 到 10M）的模型，并比较两种训练损失的有效性：LM 损失和 SFT 损失函数。图 6 显示，增加模型大小以及将 SFT 损失与合成数据结合使用，可以持续提高所有任务的准确性。这些发现证明了模型扩展和利用合成数据进行监督微调对于增强各个领域的语言模型性能的重要性。

![](img/Pasted%20image%2020240510143202.png)

我们进一步进行了案例研究，检查从数据集中提取和细化的 QA 对的质量。我们在附录 A 中展示了一些好的和坏的例子。我们观察到，从格式良好的考试和作业网站中提取的问题/答案对是高质量的。常见的问题是，大部分提取的答案不包含中间原理（思想链）。这个问题可能会导致更糟糕的泛化。因此，我们提示Mixtral和Qwen-72B完成中间步骤。我们观察到，此类完成的成功率相对较高。然而，在某些情况下，提取的问题/答案对包含严重的格式问题，这给后续的细化步骤带来了挑战。除了这些问题之外，我们还观察到法学硕士有时会修改最初提取内容的意图，从而引起幻觉。总的来说，我们的案例研究表明，收获的指令调整数据集通常是准确的，幻觉率较低


## 未来方向



## 主要收获


## 参考资料
