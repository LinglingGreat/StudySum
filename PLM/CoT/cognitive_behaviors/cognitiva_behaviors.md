---
title: cognitiva_behaviors
created: 2025-03-05
tags: 
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - 斯坦福
---

## 论文基本信息

标题：Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs

作者：

链接：https://arxiv.org/abs/2503.01307

代码： https://github.com/kanishkg/cognitive-behaviors

框架图：


## 背景
在同样的强化学习训练下，不同模型自我改进的能力却存在很大差异。比如在一个游戏中，Qwen-2.5-3B 的自我改进能力远远超过 Llama-3.2-3B（两个模型初始都很差，但强化学习训练结束后，Qwen 达到约 60% 的准确率，Llama 只有 30%）。这是什么原因？


## 相关研究




## 核心亮点

### 认知行为分析框架

为了系统地研究这个问题，作者开发了一个框架来分析对解决问题有用的认知行为，其中描述了四种关键的认知行为：验证（系统错误检查）、回溯（放弃失败的方法）、子目标设定（将问题分解为可管理的步骤）和逆向思考（从期望结果推理到初始输入）。这些行为反映了专家级问题解决者处理困难任务的方式 —— 数学家会验证证明的每个步骤、遇到矛盾时回溯以及将复杂定理分解为更简单的引理。


![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrib1UevN2u7PhFWQjSYz5HakwQy3fmO8ibjqgibUm3hoiaf4UuzYIZueWew/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

初步分析表明，Qwen 自然地表现出了这些推理行为，特别是验证和回溯，而 Llama 则缺乏这些行为。从这些观察中作者得出了核心假设：初始策略中的某些推理行为对于通过扩展推理序列有效利用增加的测试时间计算（test-time compute）是必不可少的。也就是说，**AI 模型要想在有更多时间思考时真正变得更聪明，必须先具备一些基本的思考能力（比如检查错误、验证结果的习惯）。如果模型一开始就不会这些基本思考方法，即使给它再多的思考时间和计算资源，它也无法有效利用这些资源来提高自己的表现。**这就像人类学习一样 —— 如果一个学生不具备基本的自我检查和纠错能力，单纯给他更多的考试时间也不会让他的成绩有显著提高。

  
研究人员又通过对初始模型进行干预来检验这一假设。

首先，他们发现，通过用包含这些行为（尤其是回溯）的人工合成推理轨迹对 Llama 进行引导，可以使其在强化学习过程中表现大幅改善，甚至能达到与 Qwen 相当的性能提升。其次，即使这些引导用的推理轨迹包含错误答案，只要它们展现出正确的推理模式，Llama 依然能取得进步。这表明，推理行为的存在，而不是正确答案本身，才是实现成功自我改进的关键因素。最后，他们从 OpenWebMath 数据集中筛选出强调这些推理行为的内容，用于对 Llama 进行预训练。结果表明，这种有针对性的预训练数据调整能够成功诱导出高效利用计算资源所需的推理行为模式 ——Llama 的性能提升轨迹与 Qwen 一致。

**这项研究揭示了模型的初始推理行为与其自我改进能力之间存在紧密联系。这种联系有助于解释为什么有些语言模型能够找到有效利用额外计算资源的方法，而另一些模型则停滞不前。**理解这些动态变化可能是开发能够显著提升问题解决能力的 AI 系统的关键。

  

### 如何让 AI 学会自我改进？

**参与对比的模型：Qwen-2.5-3B 和 Llama-3.2-3B**

研究开始于一个令人惊讶的观察结果：规模相当但来自不同家族的语言模型通过强化学习表现出差异巨大的提升能力。

Countdown 游戏作为主要测试平台 —— 这是一个数学难题，玩家必须使用四种基本算术运算（+、−、×、÷）组合一组输入数字以达到目标数字。例如，给定数字 25、30、3、4 和目标 32，玩家需要通过一系列操作将这些数字组合起来，得到精确的 32：(30 − 25 + 3) × 4。 

选择 Countdown 进行分析是因为它需要数学推理、规划和搜索策略。与更复杂的领域不同，Countdown 提供了一个受限的搜索空间，使得可行的分析成为可能，同时仍然需要复杂的推理。此外，与其他数学任务相比，Countdown 游戏中的成功更依赖于问题解决能力而非数学知识。 

研究者使用两个基础模型来对比不同模型家族之间的学习差异：Qwen-2.5-3B 和 Llama-3.2-3B。强化学习实验基于 VERL 库，利用 TinyZero 实现。他们使用 PPO 方法训练模型 250 步，每个提示采样 4 个轨迹。选择 PPO 而非 GRPO 和 REINFORCE 等替代方案，是因为它在各种超参数设置下表现出更优的稳定性，尽管各算法的性能非常相似。 

PPO训练的超参数包括Actor学习率1e-6，Critic学习率1e-5，KL系数0.001，Mini-batch大小128，回滚次数4，回滚温度1.0，总训练轮次15。

结果揭示了截然不同的学习轨迹。尽管这两种模型在任务开始时表现相似，得分都很低，但 Qwen 在第 30 步左右表现出质的飞跃，特点是响应明显变长且准确性提高，如下图所示。到训练结束时，Qwen 达到了约 60% 的准确率，大大超过 Llama 的 30%。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWr8ATj9nD1JlNC9EVoPE0Mydib3ydyqu8QdhQ0QU9rcnlqfU19vqBWeHQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

在训练后期，可以观察到 Qwen 行为的一个有趣变化：模型从语言中的显式验证语句「8\*35 是 280，太高了」过渡到隐式解决方案检查，模型依次尝试不同的解，直到找到正确的答案，而不使用文字来评估自己的工作。

这种对比引出了一个基本问题：哪些潜在的能力能够成功地实现基于推理的改进？要回答这个问题，需要一个系统的框架来分析认知行为。 

### 分析认知行为的框架

为了理解这些不同的学习轨迹，研究者开发了一个框架来识别和分析模型输出中的关键行为。他们重点关注四种基本行为： 

1、回溯或在检测到错误时显式修改方法（例如，「这种方法行不通，因为...」）；

2、验证或系统地检查中间结果（例如，「让我们通过... 来验证这个结果」）；

3、子目标设定，即将复杂问题分解为可管理的步骤（例如，「要解决这个问题，我们首先需要...」）；

4、逆向思考，即在目标导向的推理问题中，从期望的结果出发，逐步向后推导，找到解决问题的路径。（例如，「要达到 75 的目标，我们需要一个能被... 整除的数字」）。 

选择这些行为是因为它们代表了与语言模型中常见的线性、单调推理模式不同的问题解决策略。这些行为使更加动态、类似搜索的推理轨迹成为可能，解决方案可以非线性地演变。虽然这组行为并非详尽无遗，但选择这些行为是因为它们容易识别，并且自然地与 Countdown 游戏和更广泛的数学推理任务（如证明构建）中的人类问题解决策略相一致。

每种行为都可以通过其在推理 token 中的模式来识别。回溯被视为显式否定并替换先前步骤的 token 序列，验证产生将结果与解决方案标准进行比较的 token，逆向思考从目标出发，逐步构建通往初始状态的解决方案路径的 token，而子目标设定则显式提出在通往最终目标的路径上要瞄准的中间步骤。研究者开发了一个使用 GPT-4o-mini 的分类 pipeline，可靠地识别模型输出中的这些模式。

**初始行为在自我提升中的作用**

将这个框架应用于初始实验揭示了一个关键洞察：**Qwen 的显著性能改进与认知行为的出现相吻合，特别是验证和回溯（图 1（中））。相比之下，Llama 在整个训练过程中几乎没有表现出这些行为的证据。**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrPfibvxA7GJrkmq5icheiaDIUPza8Ln7zx4RjicibrRXIeEU8dTkB3qvskHg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

为了更好地理解这种差异，研究者分析了三个模型的基线推理模式：Qwen-2.5-3B、Llama-3.2-3B 和 Llama-3.1-70B。分析揭示，与两种 Llama 变体相比，Qwen-2.5-3B 产生各种行为的比例都要更高（图 4）。尽管较大的 Llama-3.1-70B 在这些行为的激活频率上普遍高于 Llama-3.2-3B，但这种提升并不均衡 —— 特别是回溯行为，即便在更大的模型中，其表现仍然有限。

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrpSSbhYyvKMUBuy5L9BjwnLHqVX9eK7SG6CbLvtmZ0icZBX8WHUFqDJA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

这些观察结果表明两个洞察：

  

1、初始策略中的某些认知行为可能是模型通过扩展推理序列有效利用增加的测试时间计算所必需的；

2、增加模型规模可以改善这些行为的上下文激活。这种模式尤为重要，因为强化学习只能放大成功轨迹中出现的行为 —— 使这些初始行为能力成为有效学习的先决条件。 

  

### 干预初始行为 

  

在确立了基础模型中认知行为的重要性之后，接下来研究是否可以通过有针对性的干预人为诱导这些行为。

研究者提出的假设是，通过在 RL 训练前创建选择性表现特定认知行为的基础模型变体，可以更好地理解哪些行为模式对于实现有效学习至关重要。

他们首先使用 Countdown 问题策划了七个不同的启动数据集。其中五个数据集强调不同的行为组合：所有策略组合、仅回溯、回溯与验证、回溯与子目标设定以及回溯与逆向思考。他们使用 Claude-3.5-Sonnet 生成这些数据集，利用其能够产生具有精确指定行为特征的推理轨迹的能力。

为了验证改进源于特定的认知行为而非简单的计算时间增加，研究引入了两个控制条件：一个空的思维链和一个与所有策略数据集的数据点长度匹配的填充占位符 token 的链。这些控制数据集帮助作者验证观察到的任何改进是否源于特定的认知行为，而非简单的计算时间增加。作者还创建了全策略数据集的变体，其中仅包含不正确的解决方案，同时保持所需的推理模式。此变体使作者能够将认知行为的重要性与解决方案的准确性区分开来。

当使用包含回溯行为的数据集进行初始化时，Llama 和 Qwen 都通过 RL 训练表现出明显的改进（图 2）。


![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrOrscYo5xaMzBmhnWltuUZiaRsbm45mgKx16QibREvDYqJbEmXKXozzrw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

行为分析表明，**RL 会选择性地放大经验上有用的行为，同时抑制其他行为**（图 3）。

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrAKwym2fYsrhwwM1icW69rL0Ld1zFzRH7G8QWBEWYWf2fqicdUwy98n9A/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

例如，在全策略条件下（图 1（左下）），模型保留并加强回溯和验证，同时减少逆向思考和子目标设定。然而，当仅与回溯配对时，被抑制的行为（逆向思考和子目标设定）会在整个训练过程中持续存在。

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrMhg1xWnF51YbicrgAH67AyaFwxrgVH35qNYnLH8LlrPWNGeWwnXbxfQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

当用空的思维链控制进行启动时，在两种情况下，模型的性能都与基本 Llama 模型相当（≈30-35%；见图 5），这表明仅仅分配额外的 token 而不包含认知行为无法有效利用测试时间计算。此外，使用空的思维链进行训练会产生不利影响，Qwen 模型会停止探索行为。这表明**这些认知行为对于模型通过更长的推理序列有效利用扩展计算是特别必要的。**

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrIROTIMFFTzdMvJrXorlxMuw3eRibmibn7ZFYZ2ia4Z6dH0b1ib2ANHO4ibg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

令人惊讶的是，用不正确的解决方案启动但具有正确行为的模型，与在具有正确解决方案的数据集上训练的模型具有相同的性能（图 6）。这表明认知行为存在（而不是获得正确的解决方案）是通过强化学习成功实现自我改进的关键因素。因此，来自较弱模型的推理模式可以有效地引导学习过程以构建更强大的模型，这表明**认知行为的存在比结果的正确性更重要。**

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrz7x5sA2T1CHHlPh2chCia4svmKE8ypowOEWqKFVNvwFKJMtduTNS6zA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

### 在预训练数据中选择性地放大行为

  

上述结果表明，某些认知行为对于自我完善是必要的。然而，作者在初始模型中诱导行为的启动方法是领域特定的，依赖于 Countdown 游戏。这可能会对最终推理的泛化产生不利影响。我们能否通过修改模型的预训练分布来增加有益推理行为的频率，从而实现自我完善？

  

为了探究预训练数据中的行为频率，作者首先分析了预训练数据中认知行为的自然频率，重点关注 OpenWebMath 和 FineMath，它们是专门为数学推理而构建的。使用 Qwen-2.5-32B 作为分类器，研究分析了 20 万份随机抽样的文档，以查找目标行为的存在。即使在这个以数学为重点的语料库中，回溯和验证等认知行为也很少出现，这表明标准预训练对这些关键模式的接触有限（见图 7）。

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrQRcjHQkSRqiafNyEC7vaTI09micJ0Bib82voWGztyIYEib81mIQtRUz5VA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

为了测试人为增加认知行为的接触是否会增强自我提升的潜力，作者从 OpenWebMath 开发了一个有针对性的持续预训练数据集。首先使用 Qwen-2.5-32B 作为分类器，分析来自预训练语料库的数学文档，以了解目标推理行为的存在。以此为基础人们创建了两个对比集：一个具有认知行为，另一个极少认知内容的控制集。

  

然后，他们使用 Qwen-2.5-32B 将集合中的每个文档重写为结构化的问答格式，保留源文档中认知行为的自然存在或缺失。最终的预训练数据集每个都包含总共 830 万 token。这种方法使作者能够隔离推理行为的影响，同时控制预训练期间数学内容的格式和数量。

  

在这些数据集上对 Llama-3.2-3B 进行预训练并应用强化学习后，作者能观察到：1）行为丰富模型实现了与 Qwen 相当的性能，而控制模型的改进有限（图 8a）；2）对训练模型的行为分析表明，行为丰富变体在整个训练过程中保持推理行为的高激活度，而控制模型表现出与基本 Llama 模型类似的行为（图 8c）。

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW9eCFdbpDojp6rLmbYGfEWrOUbezc8JsfxYQJFI5W07ibeXQDXu4uhJQR0R5oBgXfVnYwV7V2aK01g/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

  

这些结果表明，**有针对性地修改预训练数据，可以通过强化学习成功生成有效自我改进所必需的认知行为。**

## 实验



## 未来方向



## 主要收获


## 参考资料

[为什么Qwen能自我改进推理，Llama却不行？斯坦福找到了原理](https://mp.weixin.qq.com/s/OvS61OrDp6rB-R5ELg48Aw)