---
title: GLM
created: 2023-02-10
tags: 预训练
type: 论文
papername: GLM-General Language Model Pretraining with Autoregressive Blank Infilling
conference: ACL
year: 2022
institution: 清华
---

## 论文基本信息

标题：GLM: General Language Model Pretraining with Autoregressive Blank Infilling (ACL 2022)

作者：`Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)`

链接： https://arxiv.org/abs/2103.10360

代码： https://github.com/THUDM/GLM

框架图：

预训练任务一：Autoregressive Blank Infilling

![](img/Pasted%20image%2020230211170632.png)

mask掉15%的token

预训练任务二：Multi-Task Pretraining

- 文件级。我们对单个span进行采样，其长度是从原始长度的 50%–100% 的均匀分布中采样的。该目标旨在生成长文本。
- 句子级。我们限制mask的span必须是完整的句子。对多个跨度（句子）进行采样以覆盖原始标记的 15%。该目标针对的是预测通常是完整句子或段落的 seq2seq 任务。

### 模型结构

采用transformer结构，做了一些改动：(1) 我们重新排列了层归一化和残差连接的顺序，这已被证明对于大规模语言模型避免数值错误至关重要； (2) 我们使用单个线性层进行输出标记预测； (3) 我们用 GeLU 替换 ReLU 激活函数。

我们提出 2D 位置编码。具体来说，每个令牌都使用两个位置 id 进行编码。第一个位置 id 表示损坏的文本 xcorrupt 中的位置。对于masked span，它是相应 [MASK] 令牌的位置。第二个位置 id 表示跨度内位置。对于 A 部分中的标记，它们的第二个位置 id 为 0。对于 B 部分中的标记，它们的范围从 1 到跨度的长度。这两个位置 id 通过可学习的嵌入表投影到两个向量中，这两个向量都被添加到输入标记嵌入中。

### 实验

![](img/Pasted%20image%2020230211172516.png)

![](img/Pasted%20image%2020230211172628.png)

![](img/Pasted%20image%2020230211172638.png)



## 核心亮点

## 主要收获


## 参考资料

[GLM-130B: An Open Bilingual Pre-Trained Model | GLM-130B](https://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/)


