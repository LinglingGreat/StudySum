---
title: CoT
created: 2023-02-19
tags: 综述

---


## CoT

**Chain of Thought Prompting Elicits Reasoning in Large Language Models.**

Chain-of-thought prompting有以下几个优势

      a) 通过产生chain-of-thought，允许模型将复杂问题分解为多个中间步骤，为需要更多推理步骤的问题分配更多计算资源。

	b) 思维链提供了一个语言模型可解释性的窗口，显示语言模型得出答案的过程，提供debug的机会去定位推理路径中错误的位置。
    
	c) 思维链适用的任务宽泛，包扩数学推理，符号操作，常识推理等问题，在原则上适用于所有人类可以通过语言解决的问题。
    
	d) 很容易从目前的大规模语言模型中引导出思维链推理的能力，只需要在few-shot prompting中插入对应思维链的示例即可。

![](img/Pasted%20image%2020230219164019.png)

当遇到涉及推理的问题时，standard prompting配合模型的scaling就显得作用不是很大，但换成CoT，提升就明显不少。而且在作者提出的OOD的setup里面（这里OOD意思是，比如Last letter concatenation，例子给的都是2个词的，问题问的是4个词的，或者coin flip里，例子都是flip了2次，但问题flip了4次），scaling在CoT的帮助下效果相当显著。这也可以看到，LLM真的强，越大越强，如果不强，可能是你不会用:)

模型参数增大后的涌现能力

![](img/Pasted%20image%2020230219170121.png)

## CoT  family

![](img/Pasted%20image%2020230227220518.png)

## Manual CoT

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models，NeurIPS2022**
    
- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou（谷歌）

![](img/Pasted%20image%2020230227220808.png)

 
## Zeroshot CoT

- **Large language models are zero-shot reasoners. NeurIPS2022**
    
- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa（东京大学，谷歌）

![](img/Pasted%20image%2020230219164254.png)

![](img/Pasted%20image%2020230219170231.png)

MultiArith直接能让GPT-3从17提升到78，换到PaLM上稍微小点，25到66，然而配上self consistency（一句话说，让模型通过sampling生成多条路径和答案，用投票的方式选择概率最高的一条），直接干到88了。GSM8K也是类似，提升相当巨大。Scaling的表现也是看出来这方法尤其在超大模型才好使。

![](img/Pasted%20image%2020230219164420.png)

- Zero-shot CoT和Few-shot CoT在常识推理问题（CommonsenseQA）上，并没有太大的提升（相比于数学推理）。很多时候CoT给不出正确的答案，但是推理过程却是合理且灵活的。Zero-shot CoT在多项选择时，倾向于给出多个答案，很难只给出一个答案。
    
- 在数学推理问题上，CoT能有显著的提升，但是Zero-shot CoT和Few-shot CoT犯错误时的特点很不一样：Zero-shot方法在推出正确答案后，可能会继续“画蛇添足”，导致最终错误；另外，Zero-shot有时候干脆不推理，直接重复题目。Few-shot方法则是在生成的推理过程中包含三元运算的时候很容易出错，例如(3+2)*4
    

总体上，Few-shot CoT（又可以称之为Manual-CoT）的效果还是比Zero-shot CoT更好的。


## Self-consistency
**通过chain-of-thought prompt让语言模型生成多个不同推理路径，然后通过投票机制采纳全部推理结果中的多数作为最终结果返回。**

## Self-ask
**self-ask会显示说明接下来的问题并依赖于某些记号（例如“Follow-up question:”或者”So the final answer is:”，具体情形见下图）,因此它的答案更容易解析。字面上理解，self-ask，就是在推理出最终答案前，模型会显示询问自身或者回答自身的下一个问题，步步诱导。这种prompt技巧也容易通过插入搜索引擎去回复下一个问题的答案，从而提高准确率。**

![](img/Pasted%20image%2020230325001031.png)


## Auto CoT
[AutoCoT](AutoCoT/AutoCoT.md)

## Least-to-Most Prompting，把大问题分解成一个个小问题逐个击破

**Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.**

它针对的问题是，单纯的CoT不足以解决复杂问题，比如组合泛化等，但是我们可以把它分解成一个个小问题，然后再使用CoT，这样模型就能把问题解出来。所以从这个角度看，Least-to-Most和CoT不是选择关系，而是可以互相打配合的。

![](img/Pasted%20image%2020230219170442.png)

在SCAN这个数据集上，Least-to-Most的表现可以说是让人惊诧，如果使用GPT-3的code-davinci-002，准确率能从16提到接近100%。

另外还有数学题方面也是在原版CoT上很有进一步明显提升.

## CoT+Finetuning一样效果很好

[Flan-PaLM_T5](../Alignment/Flan-PaLM_T5/Flan-PaLM_T5.md)

## Program of Thoughts

[ProgramofThoughts](ProgramofThoughts/ProgramofThoughts.md)


## React
另一种few-shot prompting的方式，让语言模型通过相互交错的形式同时生成推理路径跟特定任务动作，实现两者之间更大的协同，推理帮助模型去推断，追踪，更新动作规划以及处理异常，而动作允许跟外部环境进行交互跟获取更多信息。可以理解为了通过动作为整个推理过程提供更多信息，避免模型幻视或者生成不合理的结果。

![](img/Pasted%20image%2020230325001127.png)


## UL2

思维链的成功要求语言模型参数超过千亿，比如几个千亿语言模型，LaMDA，PaLM，GPT3等。而Google提出的UL2，则在200亿模型参数量的条件下具备了Chain-of-thought的能力。目前的预训练模型通常适用于特定类型的问题，对于正确的模型架构跟设置似乎还没达成共识，于是Google提出了一个统一的预训练语言模型框架UL2，该框架普遍适用于众多数据集跟设置。它的预训练语言目标Mixture-of-Denoisers联合多种预训练任务，包括单向语言建模的自回归生成任务，双向语言建模的span corruption任务，前缀文本建模的任务等。UL2包括三种范式X-denoiser，R-denoiser跟S-denoiser，对应的训练任务见下图。其中R-denoiser用的是T5的span corruption任务，S-Denoiser用的是类GPT的因果语言模型的建模任务，X-denoiser用的是T5跟前缀语言模型训练目标的结合。在预训练阶段，通过为模型提供对应的标记来帮助模型切换到适合给定任务下的模型。同时，UL2采用跟架构无关的理念，无论是纯decoder还是encoder-decoder都可，架构的选择不应跟预训练目标混为一谈，具体选择何种架构更多的是效率之间的权衡。


## 参考资料

[有了Chain of Thought Prompting，大模型能做逻辑推理吗？](https://zhuanlan.zhihu.com/p/589087074)

[增强语言模型之Eliciting reasoning with prompting](https://mp.weixin.qq.com/s/1KkAmqTVlwxwtpEnLUPDWQ)

文献列表： github.com/zchuz/CoT-Reasoning-Survey



