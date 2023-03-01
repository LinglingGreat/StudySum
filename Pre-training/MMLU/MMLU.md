---
title: MMLU
created: 2023-02-23
tags: Benchmark
type: 论文
papername: Measuring Massive Multitask Language Understanding
conference: ICLR
year: 2021
institution: UC_Berkeley
---

## 论文基本信息

标题：Measuring Massive Multitask Language Understanding

作者：

链接： https://arxiv.org/abs/2009.03300

代码： https://github.com/hendrycks/test

框架图：

衡量语言模型在57个知识密集型任务上的表现

**MMLU**（**大规模多任务语言理解**）是一种新的基准测试，旨在通过专门在零样本和少样本设置中评估模型来衡量预训练期间获得的知识。这使得基准测试更具挑战性，也更类似于我们评估人类的方式。该基准涵盖 STEM、人文学科、社会科学等领域的 57 个学科。难度从初级到专业高级，既考验世界知识，又考验解决问题的能力。学科范围从数学和历史等传统领域到法律和伦理等更专业的领域。主题的粒度和广度使基准成为识别模型盲点的理想选择。

多项选择题的问题形式，准确率为衡量指标

我们总共收集了 15908 个问题，我们将其分为few-shot开发集、验证集和测试集。 few-shot开发集每个主题有5个问题，验证集可用于选择超参数，由1540个问题组成，测试集有14079个问题。每个科目至少包含 100 个测试示例，这比大多数旨在评估人的考试都要长。

人文学科：法律，哲学，历史

社会科学：经济学，政治学，社会学，地理学，心理学

STEM：物理，计算机，数学

其它：专业医学，财务、会计和营销，以及全球事实知识（包括不同国家随时间推移的贫困统计数据）

![](img/Pasted%20image%2020230301141201.png)


![](img/Pasted%20image%2020230301151209.png)

## 核心亮点

## 主要收获

