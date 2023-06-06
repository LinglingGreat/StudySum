---
title: CEval
created: 2023-06-04
tags: Benchmark
type: 论文
papername: C-Eval - A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models
conference: 
year: 2023
institution: SJTU
---

## 论文基本信息

标题：C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models

作者：Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian

链接： arXiv:2305.08322

代码： https://github.com/SJTU-LIT/ceval

框架图：


## 背景
考虑**知识**和**推理**这两项核心。

C-Eval 希望可以在整体上对标 MMLU (这个数据集被用于 GPT-3.5, GPT-4, PaLM, PaLM-2, Gopher, Chinchilla 的研发)，希望在 Hard 的部分对标 MATH (这个数据集被用于 GPT-4, PaLM-2, Minerva, Galactica 的研发)。

在实际研发的过程中，很多时候我们需要知道某种方案的好坏或者某种模型的好坏，这个时候我们需要一个数据集帮助我们测试。以下是两个经典场景：

**• 场景 1 ，辅助超参数搜索：**我们有多种预训练数据混合方案，不确定哪种更好，于是我们在 C-Eval 上相互比较一下，来确定最优预训练数据混合方案；

**• 场景 2 ，比较模型的训练阶段：**我有一个预训练的 checkpoint ，也有一个 instruction-tuned checkpoint，然后我想要衡量我的 instruction-tuning 的效果如何，这样可以把两个 checkpoint 在 C-Eval 上相互比较，来衡量预训练和 instruction-tuning 的相对质量。


具体来说，需要重点关注以下机构的论文  

• OpenAI - 这个毋庸置疑，所有文章都要全文背诵；

• Anthropic - OpenAI 不告诉你的东西，Anthropic 会告诉你；

• Google DeepMind - Google 比较冤大头，什么技术都老实告诉你，不像 OpenAI 藏着掖着。

如果读者在里经验不足，那么可以先不要看其他的地方的文章。先培养判断力，再去读其他地方的文章，这样才能分清好坏。在学术上，要分清好坏，而不是不加判断一味接受。

在研发的过程中，建议关注以下内容：

• 如何组 pretraining 的数据，比如 DoReMi 这个方法；

• 如何增加 pretraining 的稳定性，比如 BLOOM 的方法；

• 如何组 instruction tuning 的数据，比如 The Flan Collection；

• 如何做 instruction tuning ，比如 Self-instruct；

• 如何做 RL，比如 Constitutional AI；

• 如何增加 reasoning 的能力，比如我们先前的博客；

• 如何增加 coding 能力，比如 StarCoder。

## 参考资料

[大模型知识&推理评估基准](https://mp.weixin.qq.com/s/P0ohd5DpwJOkL8DFVC4qoA)
