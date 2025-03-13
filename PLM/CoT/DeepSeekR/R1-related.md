---
title: R1-related
created: 2025-01-26
tags:
  - o1-related
---

# 复现项目
## simplerl-reason

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

我们的许多实验是在发布DeepSeek-R1之前完成的。主要区别在于我们使用PPO而不是GRPO。

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
- 由huggingface组建，在MATH-500任务上接近deepseek的指标，可以在open-r1/open-r1-eval-leaderboard查看指标的排行榜。

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

RAGEN 是用于训练智能体模型的 DeepSeek-R1 (-Zero) 方法的首次复现，主要在gym-sokoban（传统的推箱子游戏）任务上进行训练。

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
- 飞书：[逻辑Puzzle上Deepseek R1 Zero成功复现, 三阶段RL，Response长度涨幅超50%，涌现语言混杂，double-check, Verify, Let's Summarize！ - 飞书云文档](http s://evxpwrsfkdb.feishu.cn/docx/NokEdaMBmo6aqZxVdxkcSm2cnab)

[摸着Logic-RL，复现7B - R1 zero](https://zhuanlan.zhihu.com/p/25982514066)

7b模型+数学和逻辑推理，bs=8,rollout=8,kl=0.001,len=4096

更长的回答不一定是更好的推理过程；语言混合现象会阻碍推理；thinking token的频率提高并不一定有帮助。sft倾向于记忆，而rl更容易泛化。cold start可以做的稍微好一点，但不一定有必要；课程学习依然是有用的。

## unlock-deepseek

[DeepSeek R1 Zero中文复现教程来了！](https://mp.weixin.qq.com/s/Z7P61IV3n4XYeC0Et_fvwg)
- [GitHub - datawhalechina/unlock-deepseek: DeepSeek 系列工作解读、扩展和复现。](https://github.com/datawhalechina/unlock-deepseek)

## mini-deepseek-r1

[deep-learning-pytorch-huggingface/training/mini-deepseek-r1-aha-grpo.ipynb at main · philschmid/deep-learning-pytorch-huggingface · GitHub](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)

用 GRPO 和倒计时游戏复制出一个简单版本的 R1。

在大约 50 步时，模型学会了正确的格式，即...\n...;在 100 步时，解方程的成功率约为 25%，并且模型开始用文字进行 “推理”;在 200 步时，收敛变慢，成功率约为 40%。模型开始学习一种新的“格式”，它通过尝试不同的组合并检查结果来解方程，这种方式类似于编程解决问题的方式；在 450 步时，解方程的成功率为 50%，性能仍然在缓慢提升，并且模型保持了从 200 步开始的新格式。

## open-thoughts

[GitHub - open-thoughts/open-thoughts: Open Thoughts: Fully Open Data Curation for Thinking Models](https://github.com/open-thoughts/open-thoughts)

该项目目标是策划一个**推理数据集**来训练最先进的小型推理模型，该模型在数学和代码推理基准上超越DeepSeek-R1-Distill-Qwen-32B和DeepSeek-R1-Distill-Qwen-7B 。

## unsloth-Reasoning - GRPO

项目地址： https://docs.unsloth.ai/basics/reasoning-grpo 

使用 GRPO（强化学习微调的一部分）通过 Unsloth 训练自己的 DeepSeek-R1 推理模型。

DeepSeek 的 GRPO（组相对策略优化）是一种无需价值函数模型的强化学习技术，能够高效优化响应并降低内存和计算成本。借助 Unsloth，仅需 7GB VRAM 即可在本地训练高达 15B 参数的推理模型（如 Llama 3.1、Phi-4、Mistral 或 Qwen2.5），而此前类似任务需要 2xA100 GPU（160GB VRAM）。GRPO 现已支持 QLoRA 和 LoRA，可将标准模型转化为成熟的推理模型。测试显示，仅训练 Phi-4 100 步，GRPO 模型已能生成思考 token 并给出正确答案，显著优于未使用 GRPO 的模型。

## oat-zero

项目地址： https://github.com/sail-sg/oat-zero

DeepSeek-R1-Zero 的轻量级复制品，对自我反思行为进行了深入分析。

DeepSeek-R1-Zero 最鼓舞人心的结果之一是通过纯强化学习 (RL) 实现**“顿悟时刻”**。在顿悟时刻，模型会学习自我反思等新兴技能，这有助于它进行情境搜索来解决复杂的推理问题。

在 R1-Zero 发布后的短短几天内，多个项目在较小规模（例如 1B 到 7B）上独立“复现”了类似 R1-Zero 的训练，并且都观察到了 Aha 时刻，这通常通过模型响应长度的突然增加来衡量。按照他们的设置仔细检查了类似 R1-Zero 的训练过程，并分享了以下发现：

- 在类似 R1-Zero 的训练中，可能不存在顿悟时刻。相反，发现顿悟时刻（例如自我反思模式）出现在第 0 个时期，即基础模型中。
    
- 从基础模型的反应中发现了**肤浅**的自我反思（SSR），在这种情况下自我反思并不一定会导致正确的最终答案。
    
- 通过 RL 仔细研究了类似 R1-Zero 的训练，发现响应长度增加的现象不是由于自我反思的出现，而是 RL 优化精心设计的基于规则的奖励函数的结果。

## deepscaler

> 项目地址：**https://github.com/agentica-project/deepscaler**

> 只用4500美元成本，就能成功复现DeepSeek？就在刚刚，UC伯克利团队只用简单的RL微调，就训出了DeepScaleR-1.5B-Preview，15亿参数模型直接吊打o1-preview，震撼业内。

第一步，研究人员会训练模来型进行短思考。他们使用DeepSeek的GRPO方法，设定了8k的上下文长度来训练模型，以鼓励高效思考。经过1000步训练后，模型的token使用量减少了3倍，并比基础模型提升了5%。接下来，模型被训练进行长思考。强化学习训练扩展到16K和24K token，以解决更具挑战性、以前未解决的问题。随着响应长度增加，平均奖励也随之提高，24K的魔力，就让模型最终超越了o1-preview！

![](img/Pasted%20image%2020250217115706.png)

近日，来自UC伯克利的研究团队基于Deepseek-R1-Distilled-Qwen-1.5B，通过简单的强化学习（RL）微调，得到了全新的DeepScaleR-1.5B-Preview。在AIME2024基准中，模型的Pass@1准确率达高达43.1% ——不仅比基础模型提高了14.3%，而且在只有1.5B参数的情况下超越了OpenAI o1-preview！

![图片](https://mmbiz.qpic.cn/mmbiz_png/1FD1x61uYVcxXAZapw1KzmphgKz8PDsaq8Ccsicjrw30s6LgRQ992cicZqsVWGIZ61TucnglW1hWRuR2lFB35D5Q/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

## grpo_demo

项目地址：https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb 原始的 grpo_demo.py 帖子

# 数据集

- **open-r1/OpenR1-Math-220k**:OpenR1-Math-220k 是一个大规模数学推理数据集，包含 220k 道数学题，每道题都有DeepSeek R1针对 NuminaMath 1.5 中的问题生成的 2 到 4 条推理痕迹。
    
- **OpenThoughts-114k**：拥有 114,000 个高质量示例，涵盖数学、科学、代码和谜题等。
    
- **bespokelabs/Bespoke-Stratos-17k**：对伯克利 Sky-T1 数据的复制，使用 DeepSeek-R1 创建了一个包含问题、推理过程和答案的数据集。
    
- **R1-Distill-SFT**：目前有 17000 个样本，目的是创建数据以支持 Open-R1 项目。
    
- **cognitivecomputations/dolphin-r1**：包含 80 万个样本的数据集，其中的数据来自 DeepSeek-R1 和 Gemini flash 的生成结果，同时还有来自 Dolphin chat 的 20 万个样本。
    
- **GSM8K**:GSM8K（小学数学 8K）是一个包含 8.5K 道高质量、语言多样化的小学数学应用题的数据集。该数据集的创建是为了支持需要多步推理的基本数学问题的问答任务。

- [General Reasoning](https://link.zhihu.com/?target=https%3A//gr.inc/): 160余万（截止目前）包含数学、coding、stem、医疗和社科等各类领域的问题和不同模型的回复。

- [Natural Questions](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/facebook/natural_reasoning): Meta整理的从[DCLM](https://link.zhihu.com/?target=https%3A//github.com/mlfoundations/dclm)和[FineMath](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/HuggingFaceTB/finemath)等预训练预料中挖掘出来的280余万条通用推理问题，部分提供答案。

- [Chinese-DeepSeek-R1-Distill-data-110k](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k): 聪哥整理的110k中文数学、考试、stem以及通用推理问题及R1的参考回复。

# Deepseek-R1-Zero复现心得


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


[Courage：Deepseek-R1-Zero复现实验](https://zhuanlan.zhihu.com/p/27100972384)

- 目前的RL微调模型相比DeepSeek-R1蒸馏的模型依然差距显著；也许就像deepseek-r1论文中提到的，用很强的大模型对小模型蒸馏比直接对小模型做RL效果更好；

- aha moment在很早的step中就已经出现，说明base模型中已经存在拟人化的反思行为，并不是强化学习凭空激发出来的；即便反思，最后结果也可能是错的；

- 由于base模型输出一般比较差，kl在前期可能确实没必要，即不需要让模型和base距离比较近；当模型能力增长到一定程度后，再增加kl，防止模型泛化能力变差；

- 简单题（GSM8K&Math）并不会出现response、reward同时增长的现象；可能因为GSM8K&Math对于模型很简单，不需要长思维链的训练准确率就可以达到90%；简单题难以提升困难Benchmark的分数，难题可以显著提升困难Benchmark的分数；简单题训练时依然存在Aha moment，说明base模型本身具有拟人化的反思能力；

- RL scaling law：（采样）数据越多效果越好；另外一个好处是：采样越多，advantage估计的越准；n_samples_per_prompt=1时不会出现response、reward同时增长的现象；可能因为n_samples_per_prompt=1时模型探索太少，没有发现长思维链的好处；

- 即便不加think step by step，模型也可以出现“思考”行为；这说明强化学习不仅仅是通过“prompt内化”来提高模型思考能力，而是自我探索出提高模型思考能力的思维方式；

- reward设计非常影响RL微调效果，需要一定的实验探索；对于数学题来说，只要规定了回答正确得1分，格式是否惩罚结果都差不多；

- 32B比14B具有更高的训练和测试精度；14B比32B的最终回复长度更长，可能因为14B基座能力差，所以需要更多的推理时间/长度才能效果好；

[32b R1-Zero复现，聚焦scaling](https://mp.weixin.qq.com/s/yksjeB41XFBYbS-bWYhbXw)
7B模型：
- 不使用fomat reward的话，eval的效果会低一些，response lens也不能观测到增长的趋势，这个可能和模型大小有关，也和探索空间过大有关
- 一个ppo epoch的mini-batch切分不能太多次更新，太offpolicy的话会导致训练缓慢甚至训崩，ppo对这个很敏感，我们尝试用simpleRL的超参发现并不好用
- GRPO在效果和效率上与PPO差别不大，而且超参更好调，scaling的话grpo会更好
- qwen-7bmath的反思行为在初始阶段就会多于qwen-7b，这和math的cpt应该是有比较大的关系
- qwen-7bmath并没观测到resps lens明显变长，可能是和训练集数量比较少有关，但是观察输出发现7bmath会吐很多代码等无关信息，所以后续大size不选用math系列
- 可以用pass8或者16描述模型上限，这对后续指标观察意义比较大，pass1应该随着训练逼近pass8，如果pass1，pass8，train acc都不涨了那就说明算法应该出问题了

32B模型
- 更大size的模型更容易涌现long cot
- 反思与正确反思是逐步增加的（这个在7b模型上的增速缓慢），lens也随之增加，eval的效果也在逐渐变好
- 大temperature更有助于反思出现，与反思行为的进一步增加，模型的行为是先出现大量无效反思的探索，然后逐渐通过rl学习到有效反思
- gbz调大是必须的
- 课程学习会有较大收益


# An Empirical Study on Eliciting and Improving R1-like Reasoning Models

**论文题目**：An Empirical Study on Eliciting and Improving R1-like Reasoning Models

**论文链接**：https://arxiv.org/pdf/2503.04548

**开源链接**：https://github.com/RUCAIBox/Slow_Thinking_with_LLMs

🌟 **on-policy 学习策略被证明是一个关键因素，在优化 RL 训练方面具有显著作用。** 大型推理模型的性能受强化学习设置的影响极大，我们对此进行了深入研究。

🌟 **回复长度是衡量强化学习成功的重要指标，但它是性能提升的结果，而非原因。** 若通过设计专门的奖励函数来显式鼓励模型生成更长的回复，并不能从根本上提升模型的推理能力。

🚀【STILL-3-Zero-32B】**Base Model在预训练后已展现出执行复杂推理操作的潜力，且能通过强化学习进一步激活。** 我们的 RL 训练方法能够持续提升 Qwen2.5-32B 的性能，提高回复长度和测试准确性。

🚀【STILL-3-1.5B】**强化学习能够持续提升微调模型的性能。** 在已接近性能巅峰的模型上，通过 RL 训练，我们成功将 AIME 2024 的准确率大幅提升。

🚀【STILL-3-Tool-32B】**通过监督微调，大型推理模型能够掌握操作外部工具的能力，从而实现了性能的飞跃。** STILL-3-Tool-32B 在 AIME 2024 基准上取得了 86.67% 的高准确率，远远超越了 DeepSeek-R1 满血版！

![](img/Pasted%20image%2020250309150413.png)

dynamic KL annealing (cosine decaying from KL = 1 × 10−3 to KL = 0).



# 参考资料

[DeepSeek-R1复现方案梳理](https://mp.weixin.qq.com/s/3LzuD1yWuGiHnP3xGYls0w)

[llm+rl训练项目的一些takeaway](https://zhuanlan.zhihu.com/p/27973092256)(持续更新中)

[复现和改进deepseek-r1的一些tips](https://zhuanlan.zhihu.com/p/25579111309)

