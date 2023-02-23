---
title: BIGBench
created: 2023-02-23
tags: Benchmark
type: 论文
papername: Beyond the Imitation Game collaborative benchmark for measuring and extrapolating the capabilities of language models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: 谷歌
---

## 论文基本信息

标题：Beyond the Imitation Game collaborative benchmark for measuring and extrapolating the capabilities of language models

作者：

链接： https://arxiv.org/abs/2206.04615

代码： https://github.com/google/BIG-bench

框架图：


BIG-bench 目前由 204 个任务组成，获得了来自 132 个研究机构的 442 位作者贡献。该基准的任务主题多种多样，涉及语言学、儿童发展、数学、常识推理、生物学、物理学、社会偏见、软件开发等领域的问题。BIG-bench 专注于被认为超出当前语言模型能力的任务。谷歌在 BIG-bench 上评估了 OpenAI 的 GPT 系列模型、谷歌内部的密集 transformer 架构和 Switch 式稀疏 transformer 的行为，模型规模跨越数百万到数千亿个参数。

BIG-bench Lite (BBL) 是来自 BIG-bench 的 24 个不同 JSON 任务的一小部分，旨在提供模型性能的规范度量，同时比 BIG-bench 中的 200 多个编程和 JSON 任务的全套评估轻便得多。

Beyond the Imitation Game 基准（BIG-bench）的GitHub 资源库包括：

-   超过 204 个语言任务。如 BIG-bench 审查标准那样，基准任务涵盖了不同的主题和语言，并且是目前的模型所不能完全解决的。
    

-   BIG-bench Lite：一个小型、且具有代表性的任务子集，比在整个基准上进行更快的评估。
    

-   实现基准 API 的代码：支持在公开可用的模型上进行任务评估，并实现新任务的轻量级创建。
    

-   对规模横跨六个数量级的密集和稀疏语言模型的详细评估结果，以及由人类评估员建立的基线结果。


BIG-bench支持两种类型的任务：JSON和编程任务，其中大约80%的基准任务是JSON任务。

JSON任务由JSON文件定义，该文件包含由输入和目标组成的示例列表。通过使用标准指标（如ROUGE）或基于模型分配的概率（如回答多项选择题），将生成的模型输出与目标进行比较来评估性能。基于示例的JSON任务规范还允许进行简单的少样本评估。

另外大约20%的基准任务是程序化的，它们用Python编写，能够在多轮查询中直接与模型交互，并且能够使用自定义度量来衡量性能。

### 评估发现

作者团队在 BIG-bench 上评估了多个语言模型的能力，模型大小从数百万到数千亿个参数，包括 OpenAI 的 GPT 模型、Google 内部密集 transformer 架构和 Switch 式稀疏transformer的性能等等。

尽管语言模型因其大规模而具有良好的性能，但相比于人类，它们在BIG-bench上的表现仍然很差。

他们还评估了谷歌自家的PaLM模型，结果表明其性能击败了在PaLM之前的其他模型（狗头），尽管PaLM仍然低于最好的人类评分者，但它已经超过了BIG-bench Lite分区上平均人类评分者。

在一些任务上，语言模型的性能随规模的增大而平稳提升；而在另一些任务上，语言模型会在某个特定规模上突然产生突破性的表现。

经过评估，他们还发现，随着模型规模的扩大，它们的社会偏见性越来越突出。对此，一个可能解释是较大的模型在匹配其训练集中的偏差方面做得更好。不过，当上下文清楚表明偏见不可取时，偏见就会随着规模的扩大而减少。

这一结果强调了针对机器学习系统公平性的研究、工程和政策努力的重要性。

要解决模型中的社会偏见问题，作者团队给出三个发现：1）在上下文广泛或模棱两可的情况下，偏见通常会随着规模的扩大而增加；2）在狭窄、明确的上下文中，偏差会随着规模的增大而减小；3）可以通过选择适当的提示来引导偏见。

他们还发现，模型在英语任务上的表现优于非英语任务，在涉及低资源语言的任务上表现尤其糟糕。在一些情况下，低资源语言任务的性能没有随着模型规模的增大而提高，而相应的英语任务的性能则会随着规模的增大而提高。

总体上，稀疏模型的性能与使用多 2 倍推理成本的密集模型一样好，它们的校准效果与使用多出约 10 倍推理计算的密集模型一样好。

当手动检查模型输出时，团队发现，模型在一定规模后开始生成电影标题，在更大的规模下会开始识别表情符号的语义，并且在某些情况下以最大的规模输出正确的答案。

此外，他们发现，模型的编程能力十分主观。即使是通过具体的任务进行量化，语言模型的能力和跨规模的轨迹也比我们所想的要主观得多。

## 核心亮点

## 主要收获


## 参考资料

[又一篇超百名作者的 AI 论文问世！442位作者耗时两年发布大模型新基准 BIG-bench……](https://www.leiphone.com/category/academic/q9oHlSSWWmdbJ46L.html)


