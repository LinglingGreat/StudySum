## 概述
![](img/Pasted%20image%2020210718142519.png)

对话系统分为

- 任务型对话，用于帮助用户完成某领域的特定任务，例如订餐、查天气、订票等
- 闲聊型对话（非任务型对话，又称聊天机器人），也称作开放域对话系统，目标是让用户持续的参与到交互过程，提供情感陪伴
- 问答型，提供知识满足，具体类型比较多，如图谱问答、表格问答、问答问答等

从大的架构上来看，所有的对话系统都有一些基本共同的组件。

- 首先，一个对话系统需要有一个模块对人的语音进行识别，转换成计算机能理解的信号。这个模块常常叫作“自动语音识别器”（Automatic Speech Recognition），简称 ASR。比如，现在很多手机终端、或者是智能家居都有一些简单的对话系统可以根据你的指令来进行回应。

- 第二，在通过了语音识别之后，就是一个“自然语言理解器”，也简称为 NLU。在这个组件里，我们主要是针对已经文字化了的输入进行理解，比如提取文字中的关键字，对文字中的信息，例如数字、比较词等进行理解。

- 第三，对话系统往往有一个叫“对话管理器”，简称是 DM 的组件。这个组件的意义是能够管理对话的上下文，从而能够对指代信息，上下文的简称，以及上下文的内容进行跟踪和监测。从整个对话的角度来看，DM 的主要职责就是监控整个对话的状态目前到达了一个什么情况，下一步还需要进行什么样的操作。

- 第四，在任务型的对话系统中，我们还需要一个叫“任务管理器”，简称是 TM 的模块，用于管理我们需要完成的任务的状态，比如预定的机票信息是否完备，酒店的房间预定是不是已经完成等等。

- 第五，我们需要从管理器的这些中间状态中产生输出的文本，也简称是 NLG。这个部分是需要我们能够产生语言连贯、符合语法的有意义的自然语言。

- 最后，在一些产品中，我们还需要把自然语言能够用语音的方法回馈给用户，这个组件往往简称为 TTS。

技术手段

![](img/Pasted%20image%2020210920113811.png)

## 任务型对话系统

NLU

- 意图（Intent）分类
- 填空（Slot Filling）

DM

- 如果还有一些核心信息缺失，需要填空，就需要DM来对所有的空进行管理，并且决定下面还有哪些空需要填写。

TM

- 执行任务

在很多现在的系统中，DM 和 TM 都是结合在一起进行构建的。在此之上往往有一个叫作“协议学习”（Policy Learning）的步骤。总体来说，协议学习的目的是让对话系统能够更加巧妙和智能地学习到如何补全所有的“空”并且能够完成模块动作。比如，有没有最简化的对话方法能够让用户更加快捷地回答各种信息，这都是协议学习需要考虑的方面。目前来说，在协议学习方面比较热门的方法是利用深度强化学习来对 DM 和 TM 进行统一管理。

NLG

- 填写模板，事先生成一些语句的半成品
- RNN，LSTM 来对 NLG 进行建模

## 非任务型对话系统

非任务型的对话系统有时候又会被称作是“聊天机器人”

- 在一个知识库的基础上和用户进行对话。这个知识库可以是海量的已经存在的人机对话，也可以是某种形式的知识信息。

### 检索式

例如DSSM及其变种

针对当前的输入，利用之前已经有过的对话进行回馈，这就是基于信息检索技术的对话系统的核心假设。一种最基本的做法就是，找到和当前输入最相近的已有对话中的某一个语句，然后回复之前已经回复过的内容。

当然，上面这种对话系统可能会显得比较原始。但是，一旦我们把整个问题抽象成广义的搜索问题，其实就可以建立非常复杂的检索系统，来对我们究竟需要回复什么样的内容进行建模。

从理论上来讲，基于检索的对话系统有很多先天的问题。比如，从根本上，搜索系统就是一个“无状态”（Stateless）的系统。特别是传统意义上的搜索系统，一般没有办法对上下文进行跟踪，其实从整个流程上讲，这并不是真正意义上的对话，当然也就谈不上是“智能”系统。

开源文本匹配工具AnyQ

在多轮对话匹配这一方面百度的对话团队做了大量的工作，属于长期霸榜的状态，比如Multi-view、DAM以及DGU。



### S2S

基于深度学习的对话系统逐渐成为了对话系统建模的主流，就是因为这些模型都能够比较有效地对状态进行管理。

那么，在这么多的深度对话系统中，首当其冲的一个经典模型就是“序列到序列”（Sequence To Sequence）模型，简称 S2S 模型。S2S 模型认为，从本质上对话系统是某种程度上的“翻译”问题，也就是说，我们需要把回应输入的句子这个问题看作是把某种语言的语句翻译成目标语言语句的一个过程。S2S 模型也广泛应用在机器翻译的场景中。

具体来说，S2S 把一串输入语句的字符，通过学习转换成为一个中间的状态。这其实就是一个“编码”（Encode）的过程。这个中间状态，可以结合之前字句的中间状态，从而实现对上下文进行跟踪的目的。这个部分，其实就成为很多具体模型各具特色的地方。总的来说，中间状态需要随着对话的演变而产生变化。然后，我们需要一个“解码”（Decode）的过程，把中间的状态转换成为最后输出的字句。

相比于基于信息检索的系统来说，S2S 模型并没有一个“显式”的搜索过去信息的步骤，因此可以更加灵活地处理语言上的多样性，以及不是完全匹配的问题。因此，从实际的效果中来看，S2S 模型在对话系统中取得了不小的成功。

### 存在的问题

在实际的开发中，非任务型对话系统会有一系列的实际问题需要解决。

- 首先，因为是开放性的对话系统，其实并没有一个标准来衡量这些聊天机器人式的系统的好坏。究竟什么样的系统是一个好的聊天系统，依旧是一个非常具有争议的领域。

- 其次，人们在实际的应用中发现，基于深度学习的序列模型，虽然常常能够给出比较“人性化”的语句回答，但是很多回答都没有过多的“意义”，更像是已经出现过的语句的“深层次”的“翻译”。因此在最近的一些系统中，人们又开始尝试把信息检索系统和 S2S 模型结合起来使用。

- 最后，我们需要提出的就是，非任务型对话系统和任务型对话系统有时候常常需要混合使用。比如，在一个订票系统中，可能也需要掺杂一般性的对话。如何能够有效地进行两种系统的混合，肯定又是一种新的挑战。


## 解决方案

https://github.com/deepmipt/DeepPavlov

[DeepPavlov：一个面向端到端对话系统和聊天机器人的开源库](https://www.infoq.cn/article/zgn7hquufriwmmg1opyv)

https://github.com/chatopera

https://bot.chatopera.com/

百度UNIT

https://github.com/shibing624/dialogbot

## 聊天机器人

https://github.com/PaddlePaddle/Knover/tree/master/plato-2

https://github.com/thu-coai/CDial-GPT

https://github.com/bojone/nezha_gpt_dialog

https://github.com/yangjianxin1/GPT2-chitchat

https://github.com/liucongg/UnilmChatchitRobot

## 进一步学习

[任务型对话概述](任务型对话/任务型对话概述.md)
[闲聊型对话系统概述](闲聊型对话/闲聊型对话系统概述.md)
[问答型对话系统概述](问答型对话/问答型对话系统概述.md)

chatbot相关的论文比较少，可以去搜相关的专利，比如google patent，会写的详细一些

## 业界分享


## 论文
ConceptNet infused DialoGPT for Underlying Commonsense Understanding and Reasoning in Dialogue Response Generation，对话中融入常识理解

## 资源汇总

https://github.com/cingtiye/Awesome-Open-domain-Dialogue-Models


## 参考资料

《AI内参》课程

https://github.com/km1994/NLP-Interview-Notes/tree/main/NLPinterview/DialogueSystem

[上篇 | 如何设计一个多轮对话机器人](https://www.jiqizhixin.com/articles/2019-07-11-7)

[做了20+个AI多轮对话项目后的总结](https://www.163.com/dy/article/G3M9899R0511805E.html)

https://github.com/qhduan/ConversationalRobotDesign

阿里的分享：[达摩院Conversational AI研究进展及应用](https://mp.weixin.qq.com/s/-PFnaczT8fTlOfhAefybgw)（包括小样本学习，对话管理具备学习能力，Text2SQL）

[生成式对话seq2seq：从rnn到transformer](https://zhuanlan.zhihu.com/p/97536876)（介绍了一系列模型rnn,transformer,bert等）

[如何做一个完全端到端的任务型对话系统？](https://zhuanlan.zhihu.com/p/108095526)（介绍了几篇论文）

[开源对话系统架构](https://cloud.tencent.com/developer/article/1796656)（介绍rasa和DeepPavlov）

[《小哥哥，检索式chatbot了解一下》](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzIwNzc2NTk0NQ%3D%3D%26mid%3D2247484934%26idx%3D1%26sn%3D40332a00a0a8f4b3943ec0dae35d5c5a%26chksm%3D970c2ed0a07ba7c67248524c08b1cb49217598c93a3b4ba2a8eda053a443136a3a8c578c4121%26scene%3D21%23wechat_redirect)详解了Multi-view、SMN、DUA以及当时最好的DAM模型。

[如何打造高质量的NLP数据集](https://mp.weixin.qq.com/s/r4ycLnjOl5hSPBMwKpnmsQ)

