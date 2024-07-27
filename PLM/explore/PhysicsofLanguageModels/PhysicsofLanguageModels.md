---
title: PhysicsofLanguageModels
created: 2024-07-27
tags:
  - 模型本质
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution:
  - MetaAI
---

## 论文基本信息

标题：Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

作者：

链接：

代码：

框架图：



concern 1: Studying models pretrained using internet data is not "scientific enough"

- need full control of the data!

concern 2: Studying individual models is not "scientific enough"

- want universal laws that holds for all LLMs, not just the July version of GPT4-o, regardless of the pretrain/finetune parameters, models sizes.

concern 3: Studying benchmarks may not be "scientific enough"

- e.g., GSM8k only has 8k grade-school math problems...

concern 4: tell us little about the internals of LLMs / how things work / why things fail

  

decompose "intelligence" into building blocks(structures, knowledge, reasonging, etc.)

studying in a controlled, idealized environment(control the data, tweak the params)

highly repetable experiments(use 100M-size models, derive universal laws)

probing techniques to see the inner workings

## Knowledge

![](img/Pasted%20image%2020240727185025.png)

![](img/Pasted%20image%2020240727185042.png)


### Knowledge Storage and Extraction

![](img/Pasted%20image%2020240727191752.png)

![](img/Pasted%20image%2020240727185610.png)

知识抽取：在剩余N/2个人的QA对上做测试，如果正确，说明模型有知识抽取的能力。

现在一般的做法是这样：

![](img/Pasted%20image%2020240727190109.png)

pretrain+finetune并不能得到知识抽取的能力（注意这里每个individual只有1个biography）

要想提高知识抽取的能力，必须做knowledge augmented！（句子的多样化表示，翻译，重复，改写等）

下图展示了用prob去探究知识是存储在哪里，如何存储的。


![](img/Pasted%20image%2020240727190139.png)

pretrained with knowledge augmentation 
=> changes how knowledge is stored inside LLM 
=> affects whether knowledge can be extracted(via instruction finetune)



![](img/Pasted%20image%2020240727190150.png)

knowledge augmentation on celebrity => knowledge extraction for minority

![](img/Pasted%20image%2020240727190201.png)

![](img/Pasted%20image%2020240727191658.png)

### Knowledge Manipulation

![](img/Pasted%20image%2020240727200837.png)



![](img/Pasted%20image%2020240727200857.png)

训练中用了cot, 推理的时候也要用cot才行。在knowledge manipulation中（和reasoning中的cot不同，不需要写出中间过程）

![](img/Pasted%20image%2020240727200908.png)

reverse knowledge search

![](img/Pasted%20image%2020240727200920.png)


![](img/Pasted%20image%2020240727200936.png)


![](img/Pasted%20image%2020240727200948.png)

![](img/Pasted%20image%2020240727203521.png)

### Knowledge Capacity Scaling Laws
![](img/Pasted%20image%2020240727203604.png)

如何计算数据的bit

![](img/Pasted%20image%2020240727203902.png)

 LLMs can "consistently" achieve 2bit/param in storing knowledge after sufficient training

![](img/Pasted%20image%2020240727203621.png)

充分训练的情况下，各个架构没什么区别

不充分训练的情况下，GPT2比Llama和Mistral好
 
![](img/Pasted%20image%2020240727203632.png)


![](img/Pasted%20image%2020240727203643.png)



![](img/Pasted%20image%2020240727203654.png)

 

![](img/Pasted%20image%2020240727203707.png)

![](img/Pasted%20image%2020240727205429.png)

![](img/Pasted%20image%2020240727205609.png)

![](img/Pasted%20image%2020240727205638.png)

## Grade-School Math(Reasoning)

### Hidden Reasoning Process 

![](img/Pasted%20image%2020240727205759.png)



![](img/Pasted%20image%2020240727210405.png)

![](img/Pasted%20image%2020240727210452.png)

2个重要的事情

![](img/Pasted%20image%2020240727210836.png)


![](img/Pasted%20image%2020240727210938.png)

![](img/Pasted%20image%2020240727211119.png)

![](img/Pasted%20image%2020240727211426.png)

![](img/Pasted%20image%2020240727211626.png)

![](img/Pasted%20image%2020240727211858.png)

### How to Learn From Mistakes

![](img/Pasted%20image%2020240727205836.png)



## 核心亮点






## 未来方向



## 主要收获


## 参考资料

【ICML 2024 tutorial: 语言模型物理学】 https://www.bilibili.com/video/BV1Yw4m1k7nH

[Physics of Language Models](https://physics.allen-zhu.com/home)

