---
title: Forgetting_李宏毅
created: 2025-04-21
tags:
  - 灾难性遗忘
---
## 什么是灾难性遗忘

灾难性遗忘是指，我们在对Fundation模型进行post-training（可以是pretrain、SFT、RLHF）之后，模型获得了post-training针对性训练的能力，但是失去了Fundation模型的某些其他能力。

比如下面这个教LLaMA-2-Chat学中文的例子中，原始的LLaMA-2-Chat通常都是用英文回复，经过中文数据Post-training之后，新的LLaMA-2-Chat能够用中文回复，但是失去了部分Alignment能力：在面对攻击性prompt的时候，不再像原始模型那样拒绝，而是直接给出答案。也就是失去了安全防御的能力。

![](img/Pasted%20image%2020250421142939.png)

![](img/Pasted%20image%2020250421143337.png)

另一个例子是Post-training之后的模型重复输出同样的一句话。

![](img/Pasted%20image%2020250421143617.png)

经过对模型的有害输出的测试，可以发现原始的Llama-2-7b-chat有害输出概率很低，经过post-training后，有害输出概率升高，尝试冻结模型中的某些层可能有一点点缓解，但仍然存在遗忘问题。相比之下LORA训练方式导致的灾难性遗忘程度最低。

![](img/Pasted%20image%2020250421143701.png)

前面的例子都是关注模型安全方面的能力的遗忘，下面这个例子测试了模型其他方面能力的遗忘情况。分别在函数使用、数学、代码的数据集上训练模型，相应的能力确实都增强了，但是其他能力都下降了。

![](img/Pasted%20image%2020250421145521.png)

再一个不同模态的例子：教LLaMA听声音。首先把声音输入给一个Speech Encoder得到向量，再把向量输入给文本LLM，因为文本LLM识别不了这些向量，又不希望对LLM参数改太多，所以通常会对文本LLM加一些Adapter，用于微调学会识别这些向量。

![](img/Pasted%20image%2020250421150404.png)

然后就是组织一些任务相关的数据训练模型，需要有（语音，文本指令，正确输出）。

当训练到1个Epoch的时候，模型能够根据指令输出json格式，并给出回答，虽然这个答案是不正确的，说明模型还不能识别声音中的情绪。

![](img/Pasted%20image%2020250421150831.png)

当训练增加到3个Epoch的时候，模型能正确输出声音的情绪了，但是无法输出json格式了。模型也遗忘了自己输出json格式的能力。

![](img/Pasted%20image%2020250421151007.png)

以上就是灾难性遗忘的例子。

## Post-training的挑战：灾难性遗忘

为什么会出现灾难性遗忘呢？因为在Post-training的时候，只让他学习目标任务，其他任务没有学习，就过拟合到目标任务上了。

那么比较大的模型会不会好一点呢？也有论文研究不同模型的遗忘问题，似乎并没有好一些。不过这篇论文只研究了1B-7B大小的模型，没有研究更大的模型。而且模型在目标任务上学习得越好，遗忘的越严重。不同颜色代表LoRA rank的不同值。

![](img/Pasted%20image%2020250421151507.png)

LoRA训练导致的遗忘问题更少，但是也有相应的代价：模型在目标任务上学习得也更少了。横轴是3个任务的分数均值，分数越高，代表遗忘的越少。

![](img/Pasted%20image%2020250421151637.png)

这篇论文也试了其他方法，比如Dropout，weight decay，但不如LoRA防止遗忘的效果好。

![](img/Pasted%20image%2020250421152018.png)

## 如何缓解灾难性遗忘

### Experience Replay

GPT-2时代就有相关研究了。

看下图，纵轴是在SQuAD（阅读理解）任务上正确率，GPT-2学了SQuAD后分数上升，接着学了WikiSQL，SST都会导致分数下降。然后又学习SRL，SRL有点像是阅读理解任务，居然SQuAD的分数又回来了！说明模型并没有把能力丢失，只是没想起来，可以通过某种方式把这种能力唤回来。

![](img/Pasted%20image%2020250421153634.png)

那么怎么解决呢？只需要在训练任务2的时候，混入一点点（5%即可）任务1的数据就可以大大缓解灾难性遗忘问题！这个方法叫做Experience Replay。上图中其他颜色的线就是Replay实验的效果。

![](img/Pasted%20image%2020250421154046.png)

那么如何没有任务1的训练数据怎么办呢？比如现在开源的模型，我们都不知道它的训练数据都有哪些。可以输入一个起始token，让LLM自说自话，产生一些数据，把这些数据作为replay数据训练进去。

![](img/Pasted%20image%2020250421154502.png)

语言模型自己生成的数据：

![](img/Pasted%20image%2020250421154535.png)

图中上方的多种颜色的线，其中2条是真实的replay数据，后面几条线是GPT-2自己生成的数据。

![](img/Pasted%20image%2020250421154734.png)

有一篇Magpie的论文给出了让Llama-3-Instruct自己生成数据的方法。第一步输入一个代表`user`的 token，让Llama输出一个query，然后把代表`user`的 token、query、代表`assistant`的 token拼接起来输入给Llama，让他输出回复。这里的问题、答案都是Llama自己生成的，当然你也自己准备好问题部分，让模型只输出回答部分。这样可以针对性收集你想要的任务类型的数据。

![](img/Pasted%20image%2020250421155006.png)

### Paraphrase & self-output

类似的方法还有
- Paraphrase：不直接用原始的output，而是让模型用自己的话换句话说（也就是改写）
- self-output：让模型直接输出回答，答对了就用它输出的。可以多次采样。有点类似于RL训练的方法，都是用模型自己输出的回答，从这个角度看RL-based的方法可能能够防止灾难性遗忘的问题。

![](img/Pasted%20image%2020250421161351.png)

SSR（self-output的方法）可以让灾难性遗忘的问题减少。

![](img/Pasted%20image%2020250421161502.png)

self-output中采取的是Foundation Model生成的回复，那么可不可以用其他语言模型生成的回复呢？

有个研究做了这个实验。分别对比人类答案、GPT-4、Claude答案数据训练的效果，图中标红色的代表效果特别差的。这个研究表明，用人类的答案训练模型，通常效果是最差的，用其他语言模型的输出来训练的效果要比人类答案好。

但是某些情况下（比如HumanEval），用GPT-4的答案训练，效果也不好。因此又尝试了另一种方法Minimum Change：先用Foundation Model生成回复，再用GPT-4修改，只改掉回复中错误的地方，内容越像越好。这样做比直接用GPT-4的答案效果更好。

![](img/Pasted%20image%2020250421162011.png)

再来看看之前教LLM听声音的例子。我们能不能把self-output的方法用在这里，尽量用模型自己的语言作为答案呢？可以针对语音做标注，把声音讯号的各种语音特征用文字描述出来（语音长度、说话人性别，情绪，口音等）。把这些特征和你的query输入给LLM，虽然它不能识别语音，但是把语音特征用文字输入给它，它就仿佛能听到语音一样，输出一个回复。在训练听声音的LLM的时候，就可以用这个输出作为它的Target去训练。这样就能防止模型遗忘它作为LLM的一些能力。

目前有很多模型都是这样训练的，比如BLSP、DeSTA2、DiVA。

![](img/Pasted%20image%2020250421162929.png)

DeSTA2在训练的时候只用了一个instruction（what can you hear？），但是它能够回答其他任何问题。

![](img/Pasted%20image%2020250421163639.png)

使用的Benchmark涉及声音相关的各种任务。

![](img/Pasted%20image%2020250421163832.png)

DeSTA2在Benchmark上总体而言是最好的，而且用了很少的数据。

![](img/Pasted%20image%2020250421164008.png)

### Selective Token Masking (STM)

还有没有其他方法能够防止灾难性遗忘呢？有个研究观察了self-output的结果和Ground Truth之间的差异，计算Foundation Model产生token得概率，红色代表概率比较低的token。发现Ground Truth中的低概率token更多。

![](img/Pasted%20image%2020250421164201.png)

计算整个回复的perplexity也发现，Ground Truth是最高的。

![](img/Pasted%20image%2020250421164444.png)

那么就可以在训练的时候，过滤掉那些很难预测的token，不计算这些token的loss。

![](img/Pasted%20image%2020250421164614.png)

效果怎么样呢？是有效的。横轴代表token被过滤的比例（从最难预测的开始过滤）。过滤掉一部分token后，模型能够表现得更好，不管是in-domain还是out-of-domain（表示遗忘程度）。

![](img/Pasted%20image%2020250421164722.png)

## 参考资料

李宏毅老师的[【生成式AI時代下的機器學習(2025)】第六講：生成式人工智慧的後訓練(Post-Training)與遺忘問題 - YouTube](https://youtu.be/Z6b5-77EfGk?si=NDOc2orRnr5nzQfI)

