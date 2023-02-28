---
title: LLM-AUGMENTER
created: 2023-02-28
tags: 知识注入
type: 论文
papername: Check Your Facts and Try Again_Improving Large Language Models with External Knowledge and Automated Feedback
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution: 微软 哥伦比亚
---

## 论文基本信息

标题：**Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback**

作者：_Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, Jianfeng Gao_

链接：

代码： https://github.com/pengbaolin/LLM-Augmenter

框架图：

![](img/Pasted%20image%2020230228145202.png)

![](img/Pasted%20image%2020230228145235.png)

马尔可夫链

![](img/Pasted%20image%2020230228145640.png)

Working Memory：追踪对话过程中的所有必要信息（dialog state）

![](img/Pasted%20image%2020230228150036.png)

Policy：选择下一步行动，最大化reward。下一步行动包括从外部知识库中为q检索证据e，请求LLM模型生成候选回复，如果回复通过了utility模块的验证则发送给用户

Action Executor：执行policy选择的行动。包括Knowledge Consolidator和Prompt Engine

Knowledge Consolidator包括a knowledge retriever, an entity linker and, an evidence chainer.
- retriever根据query和对话历史生成搜索query，然后调用一系列外部知识API获取raw evidence
- entity linker用相关的文本丰富raw evidence形成evidence graph，也就是链接raw evidence中的每个实体到wiki的描述上
- chainer裁剪graph中不相关的evidence，只保留和query最相关的证据链短候选集合
- 然后将证据发送给working memory

Prompt Engine生成一个prompt，去让LLM生成回复

Utility：给定response，生成utility分数以及feedback f
- 基于模型的方法：流畅度，信息度，事实性
- 基于规则的方法：设置某些需要符合的规则



## 核心亮点

## 主要收获

