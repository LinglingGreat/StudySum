---
title: CoT
created: 2023-02-19
tags: 综述

---


## CoT

**Chain of Thought Prompting Elicits Reasoning in Large Language Models.**

![](img/Pasted%20image%2020230219164019.png)

当遇到涉及推理的问题时，standard prompting配合模型的scaling就显得作用不是很大，但换成CoT，提升就明显不少。而且在作者提出的OOD的setup里面（这里OOD意思是，比如Last letter concatenation，例子给的都是2个词的，问题问的是4个词的，或者coin flip里，例子都是flip了2次，但问题flip了4次），scaling在CoT的帮助下效果相当显著。这也可以看到，LLM真的强，越大越强，如果不强，可能是你不会用:)

![](img/Pasted%20image%2020230219170121.png)

## Zeroshot CoT

**Large Language Models are Zero-Shot Reasoners.**

![](img/Pasted%20image%2020230219164254.png)

![](img/Pasted%20image%2020230219170231.png)

MultiArith直接能让GPT-3从17提升到78，换到PaLM上稍微小点，25到66，然而配上self consistency（一句话说，让模型通过sampling生成多条路径和答案，用投票的方式选择概率最高的一条），直接干到88了。GSM8K也是类似，提升相当巨大。Scaling的表现也是看出来这方法尤其在超大模型才好使。

![](img/Pasted%20image%2020230219164420.png)

## Least-to-Most Prompting，把大问题分解成一个个小问题逐个击破

**Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.**

它针对的问题是，单纯的CoT不足以解决复杂问题，比如组合泛化等，但是我们可以把它分解成一个个小问题，然后再使用CoT，这样模型就能把问题解出来。所以从这个角度看，Least-to-Most和CoT不是选择关系，而是可以互相打配合的。

![](img/Pasted%20image%2020230219170442.png)

在SCAN这个数据集上，Least-to-Most的表现可以说是让人惊诧，如果使用GPT-3的code-davinci-002，准确率能从16提到接近100%。

另外还有数学题方面也是在原版CoT上很有进一步明显提升.

## CoT+Finetuning一样效果很好

[FLAN-PALM_T5](../FLAN-PaLM_T5/FLAN-PALM_T5.md)

## Program of Thoughts

[ProgramofThoughts](../ProgramofThoughts/ProgramofThoughts.md)



## 参考资料

[有了Chain of Thought Prompting，大模型能做逻辑推理吗？](https://zhuanlan.zhihu.com/p/589087074)




