---
title: Lion
created: 2023-06-04
tags: 蒸馏
type: 论文
papername: Lion Adversarial Distillation of Closed-Source Large Language Model
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution: 香港科技大学
---

## 论文基本信息

标题：Lion: Adversarial Distillation of Closed-Source Large Language Model

作者：

链接： https://arxiv.org/abs/2305.12870

代码： https://github.com/YJiangcm/Lion

框架图：


## 背景
由香港科技大学提出的针对闭源大语言模型的对抗蒸馏框架，成功将 ChatGPT 的知识转移到了参数量 **7B** 的 LLaMA 模型（命名为 Lion），在只有 **70k** 训练数据的情况下，实现了近 **95%** 的 ChatGPT 能力近似。此外，框架的普适性使它不仅可以用于蒸馏 ChatGPT，还可方便地适用于其他闭源 LLMs。

具体来说，作者设计 prompt 让闭源 LLM 充当一个“裁判” **Referee** 来判别出教师的回答和学生的回答存在显著性能差距的难指令。并且，作者设计 prompt 让闭源 LLM 充当一个“生成器” **Generator** 来生成新的指令，这些生成的指令模拟了对应于被判别出的难指令的数据分布。提出的对抗蒸馏框架如下图所示，每一轮迭代包括三个阶段：

1）模仿阶段，对于一组指令，将学生的响应与老师的响应对齐；

2）区分阶段，识别出难指令；

3）生成阶段，根据识别出的难指令，产生新的难指令以增加对学生模型的挑战。

考虑到学生模型在学习过程中可能会出现灾难性遗忘的问题，作者也生成了同等数量的新的简单指令，来增加训练数据的多样性。



## 贡献


## 相关研究



## 实验



## 未来方向


## 核心亮点


## 主要收获


## 参考资料

[7B LLaMA模型接近ChatGPT 95%的能力！港科大提出全新对抗蒸馏框架Lion](https://mp.weixin.qq.com/s/UPPkMbnkG1BZhE0nctNvhQ)



