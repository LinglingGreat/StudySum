---
title: ProgramofThoughts
created: 2023-02-19
tags: prompt
type: 论文
papername: Program of Thoughts Prompting Disentangling Computation from Reasoning for Numerical Reasoning Tasks
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: 谷歌
---

## 论文基本信息

标题：**Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.**

作者：

链接：

代码：

框架图：

CoT是说我如何激发LLM的推理能力，以数学为例，但PoT谈的是LLM其实就不太适合做这事。毕竟它一股脑学一大堆文字的数据等等，其实本身很有可能学的东西就有错，自己做推理也很容易出错，此外数学问题往往是复杂的可能包含很多表达式，LLM其实用起来很不方便。PoT的策略就是让模型模仿程序来实现推理。作者给的例子谈到循环和递归的问题，如果LLM配合传统ICL（in-context learning）或者CoT，序列就会无比长，而模拟程序的PoT在形式上就看起来一步到位：

![](img/Pasted%20image%2020230219171047.png)

zero-shot，模仿“let‘s think step by step”替换成代码的开头和注释，似乎更能激发模型做出高质量的回答：

![](img/Pasted%20image%2020230219171352.png)

实验层面，背后的大模型用了GPT-3和Codex，但其实这里用的GPT-3是text-davinci-002，应该训练本身就包含了大量代码数据（不过它确实不是只针对代码），如果能看到davinci的结果可能会更有趣。结果非常优秀，配合self-consistency服用效果更佳：

![](img/Pasted%20image%2020230219171431.png)


## 核心亮点

## 主要收获

