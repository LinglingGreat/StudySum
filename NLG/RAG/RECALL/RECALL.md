---
title: RECALL
created: 2023-11-26
tags:
  - 检索增强
type: 论文
papername: "RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge"
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
  - 北京大学
  - 微信
---

## 论文基本信息

标题：RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge

作者：

链接： https://arxiv.org/pdf/2311.08147.pdf

代码：

框架图：


## 背景

构建了一套名为 RECALL 的 benchmark 来分析大模型对反事实信息输入的鲁棒性，发现现有开源大模型非常容易被反事实的输入误导，prompt engineering 和幻觉缓解领域中的现存方法难以有效解决该问题。

本文聚焦是一种更极端的干扰现象，即反事实信息（counterfactual information），也就是检索召回的内容是与事实恰好相反的假消息。理想情况下，一个明辨是非的模型对不同类输入问题和检索召回内容的处理能力应该是这样的： 

1. 对自己的参数中有明确记忆的问题，即使检索模块的召回的内容与之冲突，也应该坚持原有的正确答案； 

2. 对自己不知道答案的问题，有正确的召回内容时可以以其为参考正确回答，如果召回的内容是错就随缘了 x。 

本文首先提出了量化这一能力的一套 benchmark（名为 RECALL），向 EventKG（常识性知识问答）和 UJ（科学性知识问答）这两个阅读理解数据集中注入了反事实信息，在二选一的 QA 和生成式的问答任务上测试了 ChatGLM2、LLaMa2、Vicuna、Baichuan2 等四个 6B-13B 规模的开源大模型，其中 QA 任务分为两个子集呈现指标，即扰动的时候答案部分被修改（QA-A）和未被修改（QA-NA）。 

结果显示，选择题形式的 EventKG QA 任务上，一旦对 context 的反事实扰动涉及到了答案本身（即答案被篡改为错误选项），模型的 accuracy （下图中的 QA-A Acc）会从 90%+ （图中的第一行"original"）下降到 20% 以下（第二行"edited"），远低于没有检索机制，模型直接回答时的 60%左右（第三行“no”）。相比之下，QA-NA 和文本生成的指标下降幅度较小。

![](img/Pasted%20image%2020231126160331.png)

为了更精细地量化反事实信息带来的影响，作者额外定义了两个指标： 

- **误导率 M-Rate:** 选择式 QA 中，模型在无上下文输入时原本能答对的问题（即模型预训练阶段记忆住的问题），在接收反事实上下文后回答出错的比例； 
    
- **错误重复率 R-Rate:** 生成任务中，反事实扰动对应的 tokens 在模型的答案中出现的比例。 
    

结果显示，EventKG 数据集上，四个大模型在 QA-A 设定下的误导率 M-Rate 高达 80% 以上，生成设定下反事实信息被复读的比例 R-Rate 也高达 85-91%，可见 RAG 模块如果召回了包含反事实信息的参考文档，将对模型的可信度造成巨大的危害。

![](img/Pasted%20image%2020231126160417.png)

本文比较侧重前面的 benchmark 构建与分析部分，本身没有直接提出新方法来增强模型的鲁棒性，而且测试了两种现存方法： 

- **Prompt Engineering:** 简单粗暴，直接在 prompt 中告诉模型“忽略上下文中的反事实信息”； 
    
- **DOLA [7]:** 最近受到关注的一种针对模型幻觉的推理时干预方法，概括地讲是用模型（最后一层）输出的分布减去浅层 hidden states 对应的输出分布做解码。

结论
- Prompt Engineering：虽然能提升 QA-A 设定下被扰动时的 accuracy，但对 QA-NA 设定下的 accuracy 反而有损害，有时对生成的质量也有损害； 
    
- **DOLA:** 虽然能小幅提升大部分指标，但会导致生成任务中错误复现率 R-Rate 显著上升。 
    
- 结论：以上两类方法都不能稳定地提升模型对反事实输入的鲁棒性，亟需有可靠的新方法解决这一问题。


## 相关研究




## 核心亮点




## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
