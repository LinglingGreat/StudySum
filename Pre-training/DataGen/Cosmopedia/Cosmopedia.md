---
title: Cosmopedia
created: 2024-07-04
tags:
  - 合成数据
---

本文专注点是如何将样本从 **几千** 扩展到 **数百万**，从而使其可用于 **从头开始预训练 LLM**。

Cosmopedia是由Mixtral-8x7B-Instruct-v0.1生成的包含教科书、博文、故事、帖子以及 WikiHow 文章等各种体裁的合成数据集。其中有超过 3000 万个文件、250 亿个词元，是迄今为止最大的开放合成数据集。

Cosmopedia 完全开放: 我们发布了端到端流水线代码，数据集，以及一个在其上训练的 1B 模型，即cosmo-1b。因此，社区可以重现我们的结果并在此基础上继续研究。

- Cosmopedia：https://hf.co/datasets/HuggingFaceTB/cosmopedia
    
- 代码：https://github.com/huggingface/cosmopedia
    
- 数据集：https://hf.co/datasets/HuggingFaceTB/cosmopedia
    
- cosmo-1b：https://hf.co/HuggingFaceTB/cosmo-1b





## 参考资料

合成数据：

- Phi-1.5：https://arxiv.org/abs/2309.05463
    
- Cosmopedia：https://hf.co/datasets/HuggingFaceTB/cosmopedia

合成微调数据：

- [1] Enhancing Chat Language Models by Scaling High-quality Instructional Conversationshttps://arxiv.org/abs/2305.14233
    
- [2] Magicoder: Empowering Code Generation with OSS-Instructhttps://arxiv.org/abs/2312.02120
    
- [3] OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Datasethttps://arxiv.org/abs/2402.10176
    
- [4] WizardLM: Empowering Large Language Models to Follow Complex Instructionshttps://arxiv.org/abs/2304.12244
    
- [5] Synthetic data: save money, time and carbon with open sourcehttps://hf.co/blog/synthetic-data-save-costs


