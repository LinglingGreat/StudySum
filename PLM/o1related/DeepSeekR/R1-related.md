---
title: R1-related
created: 2025-01-26
tags:
  - o1-related
---
## qwen2.5-Math-7B+RL

我们从qwen2.5-Math-7B的base model开始直接做RL。没有SFT和reward model ，RL只用了8000条MATH数据做verification，模型最后pass@1 acc 33.3% on AIME, 62.5% on AMC, and 77.2% on MATH，超过了qwen2.5-math-7B-instruct，也comparable to 一些很强的7B baselines，但是这些方法用的数据量都比我们多至少50倍，也相对更复杂。我们也看到了long cot和self reflection的涌现

我们写了一个博客有更多的细节 https://hkust-nlp.notion.site/simplerl-reason
我们也完全开源了训练代码 https://github.com/hkust-nlp/simpleRL-reason

## TinyZero-3B+RL

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own

You can experience the Ahah moment yourself for < $30

Twitter thread: [https://x.com/jiayi_pirate/status/1882839370505621655](https://x.com/jiayi_pirate/status/1882839370505621655)

Full experiment log: [https://wandb.ai/jiayipan/TinyZero](https://wandb.ai/jiayipan/TinyZero)

Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

**3B+ model** In this case, the base model is able to develop sophisticated reasoning skills.

