## 预训练

在预训练方面，一个共同的特征似乎是所有方法都使用多阶段的预训练管道，其中一般的核心预训练之后是上下文延长，有时还有高质量的退火步骤。

预训练中使用的技术概述

![](img/Pasted%20image%2020240822203322.png)

[LLM训练-pretrain- from ybq](https://zhuanlan.zhihu.com/p/718354385)

[LLM预训练数据策略(一)](https://mp.weixin.qq.com/s/o03ftrLz8B1qSPpEStfYGg)

[LLM高效预训练(二)](https://mp.weixin.qq.com/s/S_Nx5I4C2gxwumG4GMxkDg)



## 后训练

似乎拒绝采样现在是后训练过程中常见的做法。然而，当谈到DPO或RLHF时，目前还没有一致的意见或偏好。

后训练技术概述

![](img/Pasted%20image%2020240822203347.png)



## 任务特定的训练/精调方式

新加坡国立大学的一篇文章[4]提出，基于7B的LLaMA，用LoRA+24GB显存，结合一个人造数据集精调，就可以在BIG-bench算数任务上取得和GPT-4相当的表现。

在7个写作辅助任务上，Writing-Alpaca-7B[5]经过特定的指令精调，也可以取得超越ChatGPT的表现。

浙江大学提出[6]，以Galactica-1.3b为基础，针对自然语言推断（NLI）相关的5个任务，从P3中筛选0.5%的指令精调数据，就可以取得比用全部数据精调高2%的平均表现。

## 任务特定的prompt方法

5月港中文和哈工深的一篇文章[7]提出elicit CoT prompt，在对话生成任务上用一组辅助的prompt让大模型生成一些与用户的personality, emotions, psychology相关的内容，进而辅助对话生成，提升helpfulness等主观指标。

清华大学和UIUC[8]提出交互式地结合外部工具，可以让ChatGPT更好地解决数学任务。

谷歌和普林斯顿提出[9]，针对需要探索或初始决策很重要的任务，设计Tree of Thoughts以取代CoT，在24点、创意写作、crosswords等任务上取得了明显的提升。

南京大学提出头脑风暴法[10]，在CoT的基础上，通过一个过生成+排序筛选+生成的过程，在APPS和CodeContests上的复杂编程题中取得明显提升。

西湖大学和港中文提出Chain-of-Symbol方法[11]，在给定一个文字表述的和地理位置信息相关的内容，生成回复的任务中，用简练的符号而非自然语言在CoT中阐述位置关系，相较ChatGPT与InstructGPT取得提升。

浙江大学与香侬科技针对文本分类任务，提出了更好的prompt: Clue And Reasoning Prompting[12] (CARP)。

浙江大学和阿里提出，通过反刍式思考[13]，反思生成内容，以提高大模型的推理能力。

阿里达摩院提出通过可执行的代码[14]来解锁InstructGPT与GPT-4回答时序推理相关问题的能力。


 [4] Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks, https://arxiv.org/pdf/2305.14201.pdf  
 [5] Multi-Task Instruction Tuning of LLaMa for Specific Scenarios: A Preliminary Study on Writing Assistance, https://arxiv.org/pdf/2305.13225.pdf  
 [6] MAYBE ONLY 0.5% DATA IS NEEDED: A PRELIMINARY EXPLORATION OF LOW TRAINING DATA INSTRUCTION TUNING, https://arxiv.org/pdf/2305.09246.pdf  
 [7] Chain-of-thought prompting for responding to in-depth dialogue questions with LLM, https://arxiv.org/pdf/2305.11792.pdf  
 [8] CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation, https://arxiv.org/pdf/2305.14318.pdf  
 [9] Tree of Thoughts: Deliberate Problem Solving with Large Language Models, https://arxiv.org/pdf/2305.10601.pdf  
 [10] Think Outside the Code: Brainstorming Boosts Large Language Models in Code Generation, https://arxiv.org/pdf/2305.10679.pdf  
 [11] Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models, https://arxiv.org/pdf/2305.10276.pdf  
 [12] Text Classification via Large Language Models, https://arxiv.org/pdf/2305.08377.pdf  
 [13] Knowledge Rumination for Pre-trained Language Models, https://arxiv.org/pdf/2305.08732.pdf  
 [14] Unlocking Temporal Question Answering for Large Language Models Using Code Execution, https://arxiv.org/pdf/2305.15014.pdf



[大模型表现好是源于对训练数据的记忆吗？请不要迷信大模型](https://mp.weixin.qq.com/s/OmAdeQgyfTimJKg1wAjs8g)

[InterLM系列模型](https://mp.weixin.qq.com/s/D8Iv-PKS0O6YyaSsU4D2Dw)
- 对齐阶段的一些trick还挺值得看看的


[长文详解--LLM高效预训练(一)](https://mp.weixin.qq.com/s/_3-hX3e1TqGEOo6O41JJ_A)





