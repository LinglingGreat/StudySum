---
title: Deepseek
created: 2024-01-18
tags:
  - 大模型
type: 论文
papername: DeepSeek LLM Scaling Open-Source Language Models with Longtermism
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution:
  - DeepSeek
---

## 论文基本信息

标题：DeepSeek LLM Scaling Open-Source Language Models with Longtermism

作者：

链接：

代码：

框架图：


## 核心亮点



### 预训练

#### 数据

核心阶段：deduplication, filtering, and remixing. 重复数据删除和重新混合阶段通过对唯一实例进行采样来确保数据的多样化表示。过滤阶段增强了信息密度，从而实现更高效、更有效的模型训练。

我们采取了积极的重复数据删除策略，扩大了重复数据删除的范围。我们的分析表明，与在单个转储中进行重复数据删除相比，对整个 Common Crawl 语料库进行重复数据删除可以更好地删除重复实例。

![](img/Pasted%20image%2020240118184001.png)

tokenizer：BBPE，。根据我们之前的经验，我们将词汇表中的常规标记数量设置为 100000。标记器在大约 24 GB 的多语言语料库上进行训练，我们用 15 个特殊标记扩充了最终词汇表，使总大小达到 100015。为了保证训练期间的计算效率，并为将来可能需要的任何其他特殊标记预留空间，我们将模型的词汇量大小配置为 102400 进行训练

人们普遍认为，在预训练阶段的后半部分合并指令数据可以增强基础模型在基准任务上的性能。在我们的研究中，我们在预训练阶段的最后 10% 整合了 500 万个指令数据，主要包括多项选择题。我们观察到基本模型确实在基准测试中表现出了改进的性能。然而，最终结果与在 SFT 阶段添加相同数据所获得的结果几乎相同。我们的结论是，虽然这种方法增强了基本模型在基准测试中的性能，但其总体潜力相当于不合并这些指令数据。如果指令数据量很大，则可以将其合并到预训练过程中。由于我们倾向于排除多项选择题，并且我们拥有的非多项选择题的可用性有限，因此我们决定不在预训练过程中包含指令数据

#### 模型结构

![](img/Pasted%20image%2020240118184142.png)

#### 超参数
DeepSeek LLM is initialized with a standard deviation of 0.006 and trained using the AdamW optimizer, with the following hyperparameters: 𝛽1 = 0.9, 𝛽2 = 0.95, and weight_decay = 0.1

A multi-step learning rate scheduler is employed during pre-training instead of the typical cosine scheduler. Specifically, the learning rate of the model reaches its maximum value after 2000 warmup steps, and then decreases to 31.6% of the maximum value after processing 80% of the training tokens. It further reduces to 10% of the maximum value after 90% of the tokens. The gradient clipping during the training phase is set to 1.0.

![](img/Pasted%20image%2020240118184331.png)

从训练过程中的趋势来看，使用多步学习率调度器的最终性能与余弦调度器的最终性能基本一致。
当在保持模型大小固定的情况下调整训练规模时，多步学习率调度器允许重复使用第一阶段的训练，为持续训练提供了独特的便利。因此，我们选择多步学习率调度器作为默认设置。我们还在图 1(b) 中证明，调整多步学习率调度器中不同阶段的比例可以产生稍微更好的性能。然而，为了平衡持续训练的重用率和模型性能，我们选择了上述三个阶段分别为 80%、10% 和 10% 的分配。

### Scaling Laws

#### Scaling Laws for Hyperparameters

![](img/Pasted%20image%2020240118184626.png)

![](img/Pasted%20image%2020240118184703.png)

#### Estimating Optimal Model and Data Scaling

![](img/Pasted%20image%2020240118184754.png)

![](img/Pasted%20image%2020240118184808.png)

![](img/Pasted%20image%2020240118184825.png)

![](img/Pasted%20image%2020240118184836.png)

![](img/Pasted%20image%2020240118184855.png)

#### Scaling Laws with Different Data

我们的内部数据评估显示，当前的内部数据比早期的内部数据具有更高的数据质量。此外，OpenWebText2 的质量甚至超过了当前的内部数据，因为其规模较小，可以进行更细致的处理。

![](img/Pasted%20image%2020240118185031.png)

分析中一个有趣的观察结果是，这三个数据集的最佳模型/数据扩展分配策略显示出与数据质量的一致性。如表4所示，随着数据质量的提高，模型缩放指数a逐渐增加，而数据缩放指数b减小，这表明增加的计算预算应该更多地分配给模型而不是数据。这一发现也可以解释早期缩放定律研究中观察到的最佳模型/数据扩展分配的显着差异。

对于这一发现的直观推测是，高质量的数据通常意味着逻辑清晰，并且经过充分训练后预测难度较小。因此，在增加计算预算时扩大模型大小更有利。

### Alignment

小模型需要在数学和代码数据集上进行更长的微调，但这会损害模型的对话能力，例如增加重复行为。为了解决这个问题，我们实施了分阶段的微调过程。在这种方法中，第一阶段涉及对所有可用数据进行微调，而第二阶段特别关注对对话数据进行微调。第二阶段不会影响模型对代码和数学的熟练程度，同时减少重复行为并增强指令跟踪能力。

![](img/Pasted%20image%2020240118185721.png)

使用多选择风格评估数据（例如 MMLU、AGI Eval 和 C-Eval）来测试模型是一种常见的做法。选择题要求模型不仅要有相应的知识，还要理解选项指的是什么。在对齐阶段，我们测试了添加2000万道中文选择题，得到的性能如表13所示。需要注意的是，我们对C-Eval验证集和CMMLU测试集进行了重复数据删除，以防止数据污染。

![](img/Pasted%20image%2020240118185753.png)

事实证明，纳入额外的 20M MC（多项选择）数据不仅有利于中文多项选择基准，而且也有利于提高英语基准。这表明模型解决MC问题的能力得到了增强。然而，我们观察到这种改进并没有扩展到模型在不使用多项选择格式的其他评估中的表现，例如 TriviaQA 和我们的内部评估。这表明用户可能不会认为模型在对话交互过程中变得更加智能，因为这些交互涉及生成响应而不是解决多项选择问题。


## 评估

我们将基于困惑度的评估应用于需要从多个选项中选择答案的数据集。这些数据集包括 HellaSwag、PIQA、WinoGrande、RACE-Middle、RACEHigh、MMLU、ARC-Easy、ARC-Challenge、OpenBookQA、CHID、C-Eval、CMMLU、C3 和 CCPM。这里的基于困惑度的评估是指计算每个选项的困惑度，并选择最低的作为模型预测。对于 ARC 和 OpenBookQA，我们使用无条件归一化计算困惑度，对于其他数据集，我们使用长度归一化。

一个有趣的观察是，DeepSeek 67B 相对于 LLaMA2 70B 的优势大于 DeepSeek 7B 相对于 LLaMA2 7B 的优势。这一现象凸显了语言冲突对较小模型的影响更大。此外，LLaMA2 在某些中文任务（例如 CMath）上表现出了令人印象深刻的性能，尽管没有经过中文数据的专门训练。这表明某些基本能力，例如数学推理，可以有效地跨语言迁移。然而，像 CHID 这样涉及评估中文成语使用情况的任务，要求模型在预训练期间消耗大量中文标记。在这种情况下，LLaMA2 的表现明显低于 DeepSeek LLM。

我们观察到知识相关任务中基础模型和聊天模型的波动，例如 TriviaQA、MMLU 和 C-Eval。然而，我们并不认为这种微小的波动表明 SFT 后知识的获取或丢失。 SFT 的价值在于能够学习在聊天模型的零样本设置中获得与基本模型的少样本设置相当的分数，这与真实场景相符。例如，聊天模型的 0-shot MMLU 性能与基础模型的 5-shot MMLU 性能相当。

经过微调后，我们的模型在数学和编码任务方面表现出了显着的改进。例如，HumanEval和GSM8K分数提高了20多分。我们对此的解释是，基础模型最初不适合这些任务，而 SFT 阶段通过大量的 SFT 数据学习了编码和数学方面的额外知识。然而，值得注意的是，该模型的功能可能主要集中在代码完成和代数问题上。为了全面理解数学和编码，在预训练阶段整合各种数据至关重要，这将留作未来的工作。

DPO 模型在几乎所有开放域评估指标上都显示出改进，这证明了 DPO 培训过程对模型对齐的积极影响。

DPO不会显著影响LLM的基础能力。

![](img/Pasted%20image%2020240118183646.png)


我们观察到一个有趣的现象，即当引入系统提示时，7B LLM 的性能会略有下降。然而，当使用 67B LLM 时，添加提示会显着改善结果，如表 14 所示。我们对这种差异的解释是，较大的模型能够更好地理解系统提示背后的预期含义，使它们能够更有效地遵循指示并产生更好的响应。另一方面，较小的模型很难充分掌握系统提示，并且训练和测试之间的不一致可能会对它们的性能产生负面影响。

![](img/Pasted%20image%2020240118185941.png)


## 未来方向



## 主要收获


## 参考资料
