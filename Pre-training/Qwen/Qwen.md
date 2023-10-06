---
title: Qwen
created: 2023-10-06
tags:
  - 关键词
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

# 论文基本信息

标题：

作者：

链接： https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf

代码： https://github.com/QwenLM/Qwen

框架图：

![](img/Pasted%20image%2020231006165627.png)

表现

![](img/Pasted%20image%2020231006165813.png)
# 预训练

## 数据

- 过了3T token，超过了Baichuan2的2.6T，（大概率）是目前中文社区过了最多语料的开源模型
- 数据：主要涉及公共网络文档、百科全书、书籍、代码等，数据涉及多语言，但以中文和英文为主。不仅仅包含语数英等基础学科，还包括了理化生政史地等多个其他学科的知识、以及代码知识。
- 提升多样性：数据归一化、MinHash和LSH去重
- 提升质量：通过规则和分类器，给样本打标，包括语言、质量分、有害内容等；随机抽样再进行review；对高质量数据源进行上采样

- Web数据需要从HTML中提取文本内容，并采用语言识别工具确定语种；
- 通过重复数据删除技术增加数据的多样性，包括规范化后的精确匹配重复数据删除方法和使用MinHash和LSH算法的模糊重复数据删除方法；
- 结合规则和机器学习的方法过滤低质量数据，即通过多个模型对内容进行评分，包括语言模型、文本质量评分模型以及用于识别潜在冒犯性模型；
- 从各种来源数据中手动采样并进行审查，以确保其质量；
- 有选择地对来自某些来源的数据进行采样，以确保模型在各种高质量内容上进行训练。

## Tokenization

- BPE，开源tiktoken的实现，以cl100k为基础词库，增加了常用的中文字词以及其他语言的词汇
    
- 把数字切成digit
    
- 最终词表152k，压缩比优于llama、Baichuan、ChatGLM等，但未跟llama2、Baichuan2对比
    

## 模型结构
模型采用Transformer框架，主要做了以下修改：

- 本来LM里为了节省内存，词表大小的embedding层和输出的预测层是权重共享的，千问为提升效果取消了embedding和output的权重共享
    
- 采用RoPE[1]，为了提升精度和表现，inverse frequency矩阵采用FP32
    
- 参考PaLM，去掉了大部分层的bias计算，但为了提升外推能力，保留了QKV计算时的bias
    
- 把Pre-Norm换成了RMSNorm，免去了均值的计算，主要是提升效率，效果差不多
    
- 激活函数用SwiGLU，为了保证参数量不变，缩小了FFN的维度（不同于传统FFN的2个矩阵，SwiGLU有三个矩阵，因此缩小了隐藏层维度，由原来的4倍变成8/3倍。）

Qwen模型利用了简单地非训练计算，在推理过程中扩展上下文长度。
    
- 对于外推，提出了一种dynamic NTK-aware[2]的插值方法（即对序列长度的增加动态缩放位置信息。），可以避免效果下降
    
- 在attention计算时使用LogN-Scaling，根据上下文长度调整点乘，保证注意力的熵在上下文长度增加时也保持稳定，同时能提升外推表现。公式如下，完整的讲解请移步苏神博客[3]

![](img/Pasted%20image%2020231006170027.png)

- 采用window attention，只在一段窗口内做注意力计算，减少计算量。同时发现较低的层对上下文长度更敏感，因此用更短的窗口

## 训练

- 遵循自回归语言建模的标准方法，通过前面Token的内容预测下一个Token；
    
- 模型预训练时最大长度为2048，为了构建批次数据，对文本内容进行随机打乱及合并，再讲其截断到指定长度。
    
- 注意力模块采用Flash Attention技术，提高训练速度；
    
- 优化器采用AdamW，超参数β1、β2和ϵ为别为0.9、0.95和10−8；
    
- 采用余弦学习率计划，学习率会衰减到峰值的10%；
    
- 采用BFloat16进行混合精度训练。

# 微调

- 数据质量上，去除了只用prompt模版构造的数据，在人类风格的对话上精调
    
- 采用了ChatML的格式，让模型可以区分角色和多轮
    

```
[ {"token": "<|im_start|>"}, "system\nYou are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-01", {"token": "<|im_end|>"}, "\n", {"token": "<|im_start|>"}, "user\nHow are you", {"token": "<|im_end|>"}, "\n", {"token": "<|im_start|>"}, "assistant\nI am doing well!", {"token": "<|im_end|>"}, "\n", {"token": "<|im_start|>"}, "user\nHow are you now?", {"token": "<|im_end|>"}, "\n"]
```

- 过了128\*4000step数据，但没说过了多少epoch，这样算最多51万精调数据
- 优化器采用AdamW，超参数β1、β2和ϵ为别为0.9、0.95和1e−8；
    
- 模型最大输入长度2048；
    
- 训练批次大小为128；
    
- 模型共训练4000步，在前1430步中，学习率逐渐增加，达到2e−6的峰值。
    
- 为了防止过拟合，权重衰减的值设置为0.1，dropout设置为0.1，梯度裁剪的限制为1.0。

# 强化对齐

## RM

- 参考Anthropic[4]，先在较糙的数据上预训练RM（StackExchange、Reddit等），再用质量好的数据精调
    
- 训练数据的prompt体系做的很全，6600个标签，确保多样性和复杂度
    
- 回复的多样性提升可以降低标注难度、提升RM表现

- 奖励模型时由同等大小Qwen模型+池化层得来，用特殊的句子结束标记映射值作为模型奖励值。
    
- 获取句子打分时加了一个pooling层，正常都是直接取最后一个token的表示，直接影射到scalar，这里千问并没说是加的怎样的pooling
    
- 学习率恒为3e−6，批次大小为64，最大长度为2048，训练一个epoch。
    

## RL

- critic model warmup 50，百川也是相同的做法
    
- RL训练阶段每个query采样两个答案，作者说这样效率会更高（意思是这两个答案都会计算奖励值然后强化？）
    
- 用running mean进行奖励归一化
    
- value loss clipping，提升RL稳定性
    
- actor 采样top-p=0.9，发现可以提升评估效果
    
- 用ptx loss来缓解对齐税，用的预训练数据需要比RL数据多很多，但不好调节，系数大了影响对齐，小了又没效果
    

最终，在300条评估集上，RLHF后的模型在知识、理解、写作、Math、Coding都有提升，有的能力提升还挺大（颜色由深到浅分别是wins、ties、losses）：

![](img/Pasted%20image%2020231006170259.png)

# 工具使用

Qwen模型具有工具使用能力：

- 可以通过ReAct提示进行使用未见的工具；
    
- 使用Python解释器增强数学推理、数据分析等能力；
    
- 作为代理，与人类交互过程中，可以访问HuggingFace中大量多模态模型集合。
    

PS：高质量数据2000条-React格式数据。

# Code模型

- 为了保证作为助理的能力，选择以文本预训练模型为基座，用代码和文本联合继续训练
    
- 提升数据来源多样性很重要
    
- 窗口扩到8192
    
- 又训了90b的数据，得到CODE-QWEN
    

# Math模型

- 数学题目一般较短，用1024长度训，提升训练速度
    
- 在SFT精调时，mask掉题目中的条件和数字等无意义的词，可以加速收敛



# 主要收获


# 参考资料

[1]Roformer: Enhanced transformer with rotary position embedding: _https://arxiv.org/abs/2104.09864_

[2]YaRN: Efficient context window extension of large language models: _https://arxiv.org/abs/2309.00071_

[3]从熵不变性看Attention的Scale操作: _https://kexue.fm/archives/8823_

[4]Training a helpful and harmless assistant with reinforcement learning from human feedback: _https://arxiv.org/pdf/2204.05862.pdf_

[千问的大模型KnowHow](https://mp.weixin.qq.com/s/qtkmZreX3IKTtlrSUIKz7A)

[通义千问-Qwen技术报告细节分享](https://mp.weixin.qq.com/s/HiukJ3V44epvy3xgTVCNMQ)
