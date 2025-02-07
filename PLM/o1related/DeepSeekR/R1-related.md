---
title: R1-related
created: 2025-01-26
tags:
  - o1-related
---
## qwen2.5-Math-7B+RL

![](img/Pasted%20image%2020250126142001.png)

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|AIME 2024|MATH 500|AMC|Minerva Math|OlympiadBench|Avg.|
|Qwen2.5-Math-7B-Base|16.7|52.4|52.5|12.9|16.4|30.2|
|Qwen2.5-Math-7B-Base + 8K MATH SFT|3.3|54.6|22.5|32.7|19.6|26.5|
|Qwen-2.5-Math-7B-Instruct|13.3|79.8|50.6|34.6|40.7|43.8|
|Llama-3.1-70B-Instruct|16.7|64.6|30.1|35.3|31.9|35.7|
|rStar-Math-7B|26.7|78.4|47.5|-|47.1|-|
|Eurus-2-7B-PRIME|26.7|79.2|57.8|38.6|42.1|48.9|
|Qwen2.5-7B-SimpleRL-Zero|33.3|77.2|62.5|33.5|37.6|48.8|
|Qwen2.5-7B-SimpleRL|26.7|82.4|62.5|39.7|43.3|50.9|

|                          |                                 |                                 |                          |                        |
| ------------------------ | ------------------------------- | ------------------------------- | ------------------------ | ---------------------- |
| Qwen2.5-Math-7B-Instruct | rStar-Math-7B                   | Eurus-2-7B-PRIME                | Qwen2.5-7B-SimpleRL-Zero |                        |
| Base Model               | Qwen2.5-Math-7B                 | Qwen2.5-Math-7B                 | Qwen2.5-Math-7B          | Qwen2.5-Math-7B        |
| SFT Data                 | 2.5M (open-source and in-house) | ～7.3 M (MATH, NuminaMath, etc.) | 230K                     | 0                      |
| RM Data                  | 618K (in-house)                 | ～7 k (in-house)                 | 0                        | 0                      |
| RM                       | Qwen2.5-Math-RM (72B)           | None                            | Eurus-2-7B-SFT           | None                   |
| RL Data                  | 66K queries × 32 samples        | ～3.647 M × 16                   | 150K queries × 4 samples | 8K queries × 8 samples |
|                          |                                 |                                 |                          |                        |
我们从qwen2.5-Math-7B的base model开始直接做RL。没有SFT和reward model ，RL只用了8000条MATH数据做verification，模型最后pass@1 acc 33.3% on AIME, 62.5% on AMC, and 77.2% on MATH，超过了qwen2.5-math-7B-instruct，也comparable to 一些很强的7B baselines，但是这些方法用的数据量都比我们多至少50倍，也相对更复杂。我们也看到了long cot和self reflection的涌现

我们写了一个博客有更多的细节 https://hkust-nlp.notion.site/simplerl-reason
我们也完全开源了训练代码 https://github.com/hkust-nlp/simpleRL-reason

## TinyZero-3B+RL

![](img/Pasted%20image%2020250126142023.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own

You can experience the Ahah moment yourself for < $30

Twitter thread: [https://x.com/jiayi_pirate/status/1882839370505621655](https://x.com/jiayi_pirate/status/1882839370505621655)

Full experiment log: [https://wandb.ai/jiayipan/TinyZero](https://wandb.ai/jiayipan/TinyZero)

code: [GitHub - Jiayi-Pan/TinyZero: Clean, minimal, accessible reproduction of DeepSeek R1-Zero](https://github.com/Jiayi-Pan/TinyZero)

Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

**3B+ model** In this case, the base model is able to develop sophisticated reasoning skills.

## openr1

[GitHub - huggingface/open-r1: Fully open reproduction of DeepSeek-R1](https://github.com/huggingface/open-r1)

The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:

- `src/open_r1` contains the scripts to train and evaluate models as well generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset.
    - `sft.py`: simple SFT of a model on a dataset.
    - `evaluate.py`: evaluates a model on the R1 benchmarks.
    - `generate.py`: generate synthetic data from a model using [Distilabel](https://github.com/argilla-io/distilabel).
- `Makefile` contains an easy to run command for each step in the R1 pipeline leveraging the scipts above.

## ragen

https://github.com/ZihanWang314/ragen

**RAGEN** is the first reproduction of the **DeepSeek-R1(-Zero)** methods for _training agentic models_.  
_We strongly believe in the future of RL + LLM + Agents. The release is a minimally viable leap forward._

## Deepseek-R1-Zero复现

[Deepseek R1 Zero成功复现, 三阶段RL，Response长度涨幅超50%，涌现语言混杂，double-check, Verify, Let's Summarize！](https://zhuanlan.zhihu.com/p/21290410831)

代码：[GitHub - Unakar/Logic-RL](https://github.com/Unakar/Logic-RL)


[DeepSeek R1 Zero中文复现教程来了！](https://mp.weixin.qq.com/s/Z7P61IV3n4XYeC0Et_fvwg)



