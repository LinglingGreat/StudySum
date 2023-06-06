---
title: XTREME
created: 2023-02-24
tags: benchmark
type: 论文
papername: XTREME A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2020
institution: CMU 谷歌 DeepMind
---

## 论文基本信息

标题：XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization

作者：Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, Melvin Johnson

链接：arXiv:2003.11080

代码： https://github.com/google-research/xtreme

框架图：

**覆盖四十种语言的大规模多语言多任务基准 XTREME**. 覆盖了 40 种类型不同的语言（跨 12 个语系），并包含了 9 项需要对不同句法或语义层面进行推理的任务。

在 XTREME 大规模多语言多任务基准上选择 40 种不同类型的语言，这是为了实现语言多样性、现有任务覆盖以及训练数据可用性的最大化。其中一些是 under-studied 的语言，如达罗毗荼语系中的泰米尔语（印度南部、斯里兰卡和新加坡）、泰卢固语和马拉雅拉姆语（主要集中在印度南部）以及尼日尔-刚果语系中的斯瓦希里语和约鲁巴语（非洲）。

**XTREME 论文的并列一作是 CMU 语言技术研究所的在读博士胡俊杰，和 DeepMind 著名的研究科学家 Sebastian Ruder**。

XTREME 中的任务涵盖了句子分类、结构化预测、句子检索和问答等一系列样式，因此，为了**使模型在 XTREME 上取得好的表现，就必须学习可以泛化至多标准跨语种迁移设置的表征**。

每种任务都涵盖 40 种语言的子集，为了获得 XTREME 分析所用的低资源语言的附加数据，自然语言推理（XNLI）和问答（XQuAD）这两个代表性任务的测试集会自动从英语翻译为其他语言。**模型在使用这些翻译过来的测试集执行任务时的性能表现，可与使用人工标注测试集的表现相媲美**。

在使用 XTREME 评估模型的性能之前，首先要用支持跨语言学习的多语言文本进行模型预训练。然后根据任务特定的英语数据对模型进行微调，因为英语是最容易获得标签化数据的语言。之后，XTREME 会评估这些模型的 zero-shot 跨语言迁移性能，包括在其他没有任务特定数据的语言中。




## 核心亮点

## 主要收获

## 参考资料

[覆盖40种语言：谷歌发布多语言、多任务NLP新基准XTREME](https://www.linkresearcher.com/theses/74a45684-ff8e-4a16-a94c-97d780e2e8a1)

