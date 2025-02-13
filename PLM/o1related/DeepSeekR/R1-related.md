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

基于OpenRLHF框架
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

[Open-R1: Update #1](https://huggingface.co/blog/open-r1/update-1)
- 包括了R1相关的项目资源、数据集资源

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

[我们在Gym-Sokoban](https://github.com/mpSchrader/gym-sokoban)任务中在 Qwen-2.5-{0.5B, 3B}-{Instruct, None} 和 DeepSeek-R1-Distill-Qwen-1.5B 上运行 RAGEN 。

关于推箱子任务（来自官方仓库）：推箱子是日语中“仓库管理员”的意思，也是一款传统视频游戏。这款游戏是一款运输拼图游戏，玩家必须将房间内的所有箱子推到存储位置/目标上。犯下不可逆转的错误的可能性使得这些拼图游戏极具挑战性，尤其是对于强化学习算法而言，因为强化学习算法大多缺乏提前思考的能力。

损失曲线尚未收敛（因为我们的计算能力目前有限……）。但我们已经看到了一些趋势：

- 尽管在开始时指导微调模型表现更好，但它们并没有明显优于仅预训练的模型。
- 3B模型的表现也优于0.5B模型，但在40步左右时优势也不是那么明显。
- 有趣的是，目前 R1 蒸馏的 1.5B 模型表现不如 0.5B 模型。

致谢：veRL和TinyZero

## Logic-RL

使用veRL框架

[Deepseek R1 Zero成功复现, 三阶段RL，Response长度涨幅超50%，涌现语言混杂，double-check, Verify, Let's Summarize！](https://zhuanlan.zhihu.com/p/21290410831)
- 代码：[GitHub - Unakar/Logic-RL](https://github.com/Unakar/Logic-RL)
- 飞书：[逻辑Puzzle上Deepseek R1 Zero成功复现, 三阶段RL，Response长度涨幅超50%，涌现语言混杂，double-check, Verify, Let's Summarize！ - 飞书云文档](https://evxpwrsfkdb.feishu.cn/docx/NokEdaMBmo6aqZxVdxkcSm2cnab)
- 

## unlock-deepseek

[DeepSeek R1 Zero中文复现教程来了！](https://mp.weixin.qq.com/s/Z7P61IV3n4XYeC0Et_fvwg)
- [GitHub - datawhalechina/unlock-deepseek: DeepSeek 系列工作解读、扩展和复现。](https://github.com/datawhalechina/unlock-deepseek)

## Deepseek-R1-Zero复现心得


[关于zero-rl的碎碎念和想法](https://zhuanlan.zhihu.com/p/22288441283)
1. 不同的[rl算法](https://zhida.zhihu.com/search?content_id=253479911&content_type=Article&match_order=3&q=rl%E7%AE%97%E6%B3%95&zhida_source=entity)，在base-rl上的差异性不显著。lr、warmup等等也没特别大的影响。
	1. 调整这些参数，reward/response-length 不会同步增长（response-length会和任务特性相关，有些任务容易涨比如text-game，有些不容易涨比如math）。
	2. 容易饱和（比如跑不到100个step，效果就不涨了）。
	3. 最朴素的方法可能是最有效的。比如 reinforce以及使用ppo的loss-objective就足够用了。
2. 是否加入kl约束会有比较大的影响。
	1. 加入kl会限制模型的exploration。而base上的rl，前期的exploration更重要。

3. 使用的prompt-template影响也较大。
	1. 如果使用的template不恰当，可能最后会训出来一个 类instruct风格的模型（也侧面证明 base-model大概率刷了类似的数据，否则，rl不太可能探索出这种风格）。


[R1-ZERO 尝试复现的一些现象分享](https://zhuanlan.zhihu.com/p/22517127574)
- 格式奖励很好学
- 难的query上 格式奖励更容易hack
- **难的query上 似乎更容易出现accuracy 与 response 同增的情况**
- 