> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [mp.weixin.qq.com](https://mp.weixin.qq.com/s/b3JSE1o9dr7loafwhEWomA)

![](https://mmbiz.qpic.cn/mmbiz_gif/Psho9dm7oDHKVtfYDubjKdZRUjAfBQQicXjoZWJ3qnK42ooD4eeJUfJBM4SSZVa2RE5lO0j6rWwzliby0j9u4bDg/640?wx_fmt=gif)

**©PaperWeekly 原创 · 作者 |** 褚维芜  

**单位 |** 北京邮电大学硕士生

**研究方向 |** 自然语言处理

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wulOVRfC18yCkd6xXqGq22h6QUk8chptF0fnQ4uXeZtAktYMrWwG2SyQ/640?wx_fmt=png)

**引言**

近年来，随着预训练模型的发展，对话领域的研究也逐渐开始关注基于预训练的端到端对话系统，2019-2021 这三年的时间涌现出很多关于开放域对话系统预训练的相关研究，基于英文的包括 google 在 2020 年 1 月发表的 Meena、Facebook 在 4 月发表的 Blender，基于中文的主要以百度 PLATO 系列模型为代表 [1]。这些模型的成功一定程度上表明海量数据和更大的模型能为对话系统带来很好的性能收益。

然而，这种依靠参数量、数据量来提升系统性能的方式对于任务型对话而言并不完全适用。一方面，任务型对话数据集本身比闲聊型对话更难收集，想要获取一个非常大的数据集来对任务型对话系统进行预训练是非常困难的；另一方面，预训练模型参数过大，训练和运行需要很高的计算成本，会存在无法快速部署的问题。由于以上问题的存在，任务型对话预训练的发展速度明显不如开放域对话，但最近两年也逐渐有一些针对任务型对话进行预训练的相关工作，本文将对这些工作做一个梳理总结，供大家参考。

本文主要介绍的工作有：

*   2020EMNLP：**TOD-BERT**: Pre-trained Natural Language Understanding for Task-Oriented Dialogue [2]
    
*   2021TACL：**Soloist**: Building task bots at scale with transfer learning and machine teaching [3]
    
*   2021arXiv：**PPTOD**：Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System（PPTOD）[4]
    
*   2022AAAI：**GALAXY**: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised[5] Learning and Explicit Policy Injection
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wuhfgUpIfdPSqH8YjjHbCUiaaKsMA36bIMsMtGNKoBcus5py06M0fvx3A/640?wx_fmt=png)

**TOD-BERT：面向任务型对话理解的预训练模型**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xiao5hIBVfu8oicQ0NJrL89QmExVB2J3Nicl8YueiaOuAaFOa0ficUbw4gMeQ/640?wx_fmt=png)

**2.1 Background & Motivation**  

使用现有的预训练语言模型直接在任务型对话数据上进行 fine-tune 无法取得很好的性能，主要原因包括以下两点：一是，对话数据和文本数据的语言模式不同，所以两者的数据分布有很大差异，因此普通文本预训练的语言模型在对话数据上表现不佳；二是，闲聊型对话数据广泛且易于获得，但是它们通常很短，没有明确的对话目标。而任务型对话通常有明确的目标，多轮对话交互，因此闲聊型对话数据预训练模型也很难在任务型对话数据上取得很好的表现。

任务型对话数据集通常小而稀疏，标注成本很高，本文通过联合多个数据集在一定程度上缓解了任务型对话预训练数据量不足的问题，并针对对话数据的特点对原始 BERT 模型的输入、预训练任务进行修改使得模型可以更好地捕捉对话特有的任务信息。

**2.2 Method**

**数据集**：本文联合了九个不同的多轮任务型对话数据集，如下表所示。最终，本文预训练所采用的数据集包含 60 多个领域的 100,707 段对话，1.3M 句话语。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xiag8pg8JgG8hC3dEY1oYEPcxU8ib6k5rrwvYIITNUqFamy6ZS7OTksP9Q/640?wx_fmt=png)

**TOD-BERT 模型**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xiaib4bqld4vG9icpYicCexyQz0vgyOmYUlWNFZEtPEI7WuOPkIXp3iblaicbg/640?wx_fmt=png)

TOD-BERT 在 BERT 模型原有的 MLM 损失函数上，添加了一个 Response contrastive loss（RCL）用于模拟回复选择任务。原始 BERT 中对两段话语进行拼接并对他们的连续性进行 0-1 预测，而 RCL 损失则是采用了类似 ConveRT 中的双编码器结构，同一个 batch 中的其他回复为负样本，如下图所示，优化目标为最大化正样例的概率。RCL 损失一方面可以学习更好的【CLS】位置的表示，另一方面可以捕获潜在的对话顺序、结构信息和回复的相似性。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaoicM4jFOFtrgfqXRs37OBcpp0PapNGA7ZTETze1loZsSGgtL35zskmQ/640?wx_fmt=png)

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaLfK5xqKmkWlYkctVmFyviafbag4zSILreLJkKXWh11QQkliadynJOWEg/640?wx_fmt=png)

另外，TOD-BERT 的输入中加入对说话人角色的编码。对于对话，TOD-BERT 的输入在每一个话语前面添加角色信息的 token：。TOD-BERT 在意图识别、对话状态追踪、对话动作预测、回复选择这四个下游任务上进行了评测，性能均超越了 BERT。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wukOjHSmSsEuRCB0fJu69CtdNgLnvFPDUCgeicOppBKuDvniaD3q8XWQ0Q/640?wx_fmt=png)

**SOLOIST：预训练对话系统迁移到新的对话任务**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaNCOCRmDPlZ73HQSFuyhMACLTAps3PoqwCQCaibCqhGY9JJucc28z0cA/640?wx_fmt=png)

**3.1 Background & Motivation**  

构建对话系统需要大量的标注、领域知识以及专家经验，人工标注数据费时费力。即便已经对某个领域的数据进行了大量标注，现实情况下遇到新的任务时，对话系统依然难以完成。

针对以上问题，本文提出了一种新的大规模构建任务型对话系统的方法，并通过迁移学习和机器教学使其适应新的任务。现有的模型都是通过收集、标记新的数据，并为每个任务构建一个系统，而 SOLOIST 则不需要，这极大简化了对话系统遇到新的任务时模型的训练和部署的工作流程。

**3.2 Method**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiavNhtPzaKvw4vFTfbibbUN2GibicgD3QyibR9W1rjjTPpsvd0miaQDtJQrRw/640?wx_fmt=png)

**数据集**：本文使用 Schema、Taskmaster 这两个任务型对话数据集对模型进行预训练，数据集统计数据如下表所示。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaD9hukfRrJNMibQAgiaomTQcQRf9FpLSxVjjTkFJQ9z8ymD9Q8qvvIp8g/640?wx_fmt=png)

对于数据集中的每一段对话进行预处理如下：定义  为对话历史、 为 belief state、 为 DB state、 为 delexicalize 的对话回复，模型输入是将以上信息进行拼接，因此，训练数据集中的每轮对话可以表示为：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XianWic5k2xdDlxaUBDXoNaNYcE6jFYXnpZkDrBTMRfbiaWqRyic525RhpaA/640?wx_fmt=png)

**SOLOIST 模型**

结构说明：使用 GPT-2 直接生成对话状态、对话动作、系统回复

预训练任务：

*   Belief Prediction：生成任务
    
    ![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xiaf9ETHyf1rJMBiaWia1pyKBFiaQSB4g3hjOdVrfRNJgzgccpUmljXLWmDw/640?wx_fmt=png)
    
*   Grounded Response Generation：生成任务
    
    ![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaNe5WTjG3zqOia7HOib3uPW3Y0YfYeRiaPibBsT9BpRQnvg633F2mEm92tg/640?wx_fmt=png)
    
*   Contrastive Objective：对比学习
    
    SOLOIST 在输入的 [EOS] 位置，引入了一个对比损失函数，该函数用于预测输入是正样例 还是负样例 ，对比损失函数计算如下所示：
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaolkTtk1Ra1991X0Rl6HZCiaI64Nva2LAiaHZSH4XhPSl9hPcGPw1LNicA/640?wx_fmt=png)

模型的损失函数为三个预训练任务的损失函数相加：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaHmaI73Wl2mr8PprLmaU7C4oyXv19AiahyCzwpRMpnEhib9cGMAWDVHbA/640?wx_fmt=png)

在预训练阶段，本文使用 GPT-2 初始化，利用大型任务型对话标注数据训练一个面向任务型对话的回复生成模型。该模型学习主要的任务完成技能，如对话状态追踪和对话策略学习，并可以根据用户目标和外部知识生成可以完成对话任务的回复。在微调阶段，本文通过机器教学将预训练的 SOLOIST 模型微调为完成特定（新的）任务的系统，其中训练样本是由真人教师与系统交互生成。

实验表明，SOLOIST 成功地将两种能力从预训练模型转移到一个新的任务型对话系统：一是，预训练过程中学习到的自然语言理解（NLU）和自然语言生成（NLG）的能力；二是，在域外对话语料库上根据用户目标和外部知识生成可以完成对话任务的回复的能力。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wuiaLfO9V4lkD8cXK7ImEicqib5bPGH6syOrWzicR2KaqPyAicMccs8icC03Gw/640?wx_fmt=png)

**PPTOD：基于 prompt 方法的任务型对话预训练**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xia6dCoiaicjQ2CGRYWmPdJNeR0K9lyNrx9sW3PdqT9fozGuOT3rOb7ojTg/640?wx_fmt=png)

**4.1 Background & Motivation**  

现有基于预训练模型的任务型对话系统（SimpleTOD、SOLOIST 等）存在一定的局限性，一方面它们将各个子任务级联来进行对话生成，有误差累积，且系统推理会有延迟；另一方面，这些模型的训练需要提前标注大量的数据，且标注必须是完整的，因此大量的只有部分标注的数据无法使用（eg. 只标注了对话状态或者对话动作任务的数据）。

因此，针对以上两个问题，本文以 T5 模型为基础，通过 prompt 方法使得预训练模型不仅可以使用标注完整的对话数据，还可以使用部分标注的数据。这在一定程度上缓解了任务型对话预训练所面临的数据量不足的问题。

**4.2 Method**

**数据集**：本文使用 11 个部分标注的任务型对话数据集对模型进行预训练，总共 2.3M 句话语，包含 80 个领域。数据集及其标注信息如下表所示。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xia5DE7qiaib3VURcygJmT1J4eNianiau7LEr5p9wGsOqnjqI3h1Luia9icpFjw/640?wx_fmt=png)

**PPTOD 模型**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaORBm9Cj4qdicthrxxzof4ia7bL6fnF9EabYP0jopvZbo0Xs3gTrXLHcQ/640?wx_fmt=png)

从图中可以看出每一个训练样例之前都添加了一个任务提示，共有四种不同的任务：NLU、DST、POL、NLG，这四种任务是通过多任务学习的方式一起训练的，任何一个包含上述四种标注之一的数据集都可以用于训练 PPTOD。预训练和微调阶段的损失函数如下：

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaqtAiaoRSA5duFVpaeoN1kUdO6HrrZicGKQQ5CqTibo5k3DMiaQ6z8Fphow/640?wx_fmt=png)

本文在端到端对话生成、对话状态追踪、用户意图识别三个下游任务上对模型进行了实验，结果表明 PPTOD 在各种评估指标上均优于当前的 SOTA 系统。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wukGHdevfTibLOpic6945Lrhqmt43pKicyIhGs4m7ANzKOfY9RJgmTicZGdg/640?wx_fmt=png)

**GALAXY：基于半监督学习的任务型对话预训练**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xia9FtIaNIsBKxyjY8DFVvpMn97nbMTaDxkuY9Ummh1AdbzFwuG7U5IJg/640?wx_fmt=png)

**5.1 Background & Motivation**  

现有的任务型对话预训练的相关研究并没有在预训练阶段丰富有关对话策略的知识，作者假设在预训练阶段直接学习进行对话策略的学习（DA prediction）可以使模型学习到更好地表示，并进一步提高端到端地性能。因此，本文主要关注于怎样在预训练阶段来对对话策略进行更好地建模。

一个简单的方式是将有监督对话动作分类损失和预训练的无监督 MLM 损失一起进行多任务训练，但这种方式存在三个问题：

1.  目前各个任务型对话的 DA 标注不一致，收集一个大规模的有 DA 标注的数据集比较困难
    
2.  大量的对话是没有 DA 标注的，因为在联合训练过程中，模型可能会对这些少量的标注数据过拟合
    
3.  对于无标注的对话数据，模型只能提取到一般的语言知识，不能有效地挖掘对话策略相关知识
    

### **5.2 Method**

针对以上问题，本文所设计的解决方案如下：

**数据集**：本文为任务型对话系统构建了一个统一的 DA 标注方法，并整合八个任务型对话数据集构建了一个新的有 DA 标注的数据集——UniDA；收集并处理了一个大规模无标注闲聊对话数据集——UnDial。

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaTIHfib06hvibanvxePftTHO8d7YcjBq2S4AvHKqicxfjKRo19nxJWicEFA/640?wx_fmt=png)

**GALAXY 模型**

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaVG7wJeBasiarYIgiaia2gjVU7aC7zZvEm443PMac8LzxJ65sSHV9zp0fA/640?wx_fmt=png)

结构说明：

*   UniLM 为 backbone，它包含一个用于理解的双向编码器和一个用于生成的单向解码器，编码器和解码器是权重共享的
    
*   输入表示采用 PLATO 中的方式，包括四个部分：位置编码、轮次编码、角色编码、token 编码
    

预训练任务：

*   回复选择：构造正负样例进行 0-1 分类
    
*   回复生成：解码器逐个 token 进行解码
    
*   对话动作预测：多分类任务，仅对有标注数据有用
    
*   一致性正则化：将一段对话历史两次输入编码器，由于 dropout 扰动会得到两个不同的分布，采用 KL loss 来最小化这两个分布之间的距离，如下图所示。
    
    ![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaiaZtAOcFS4G4I4sPCrPMvWCbfcKAIkd04BkrBz9QVVwpP4G2e8OqJyQ/640?wx_fmt=png)  
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaO6DGRbicZWPzOzgQ2tibxOHEHGdIA8G8elvLujpQDchAZIk5v7WzwoqA/640?wx_fmt=png)

半监督预训练范式  

*   有标注数据的损失函数
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5Xia7yz2NjDmW70W0O6vPDl73NQTaIeiaslDTvShA5eqVLiaiaoxGZLdib115w/640?wx_fmt=png)  

*   无标注数据的损失函数
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaOJW6t5NuCX8E3c3U8Z86YJCgPwZgM65eocZAftXMtKibt2eDWS0OpVQ/640?wx_fmt=png)  

*   总的损失函数（有标和无标数据混合训练）
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaGyy1SRMkQZ32ib5ibd4N6Kbn4rvM8S59j1y6LCIcs4F2fYef8BCkXneQ/640?wx_fmt=png)  

微调及推理

*   数据集：MultiWOZ
    
*   对于有语义标注信息的对话数据，将标注信息与系统回复拼接作为新的生成，并保留对话动作预测任务
    
*   损失函数
    

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDH9pxFfIwX0IoKW5d1ya5XiaTMoM9eBkIibQ1VbaVIfxq8HlKCiceicwmYFBLzcdY9TibTa4QyQ85FJ46w/640?wx_fmt=png)

GALAXY 的实验结果表明大规模任务型对话数据进行预训练可以带了很好的收益，且有监督对话动作预测任务对学习对话策略是有效的。  

![](https://mmbiz.qpic.cn/mmbiz_png/Psho9dm7oDGhKg9nnSz5qQrwKvXibt3wuCkR96mP8kh7KicSzPQiaIQa3ft5MLn54FNK0UD2MI99iaHjT9m9NjLl7A/640?wx_fmt=png)

**总结**

从预训练数据来看，除了 SOLOIST 外，其他三个模型都是尽量使模型可以使用更多的预训练数据。TOD-BERT 联合了九个任务型对话数据集进行预训练，PPTOD 设计了一种可以利用部分标注数据集的模型，GALAXY 则更近一步，有标注和无标注的数据都可以用于训练。

从预训练任务上来看，四个模型都针对对话的特点调整了传统的预训练任务。TOD-BERT 采用了可以模拟下游回复选择任务的 Response contrastive loss，SOLOIST 将 DST、NLG 均建模为生成任务，PPTOD 基于 prompt 将下游任务均建模为生成任务，GALAXY 则采用了对话动作预测、回复生成 、回复选择、一致性正则化作为预训练任务。

从以上四个模型可以看出，目前任务型对话系统预训练领域的研究主要集中在：如何解决任务型对话数据量不足的问题；以及怎样设计更适用于对话系统的预训练任务来捕捉对话中的任务相关的信息。本文所介绍的模型虽然一定程度上缓解了上述问题，但是任务型对话预训练相比于 PLATO-XL 这种通用的对话预训练模型还有很大的差距。

![图片](https://mmbiz.qpic.cn/mmbiz_svg/lpHDr05YrIRWFnyDtmhYYNLAicC4jAr48J0MLvlaiblAmGwp6XOJ0vZ3ib5zhraopw6gFNtyBcl9Cz2euDMyGBg9SFkAicVbYiac3/640?wx_fmt=svg)

**参考文献**

![图片](https://mmbiz.qpic.cn/mmbiz_svg/lpHDr05YrIRWFnyDtmhYYNLAicC4jAr48J0MLvlaiblAmGwp6XOJ0vZ3ib5zhraopw6gFNtyBcl9Cz2euDMyGBg9SFkAicVbYiac3/640?wx_fmt=svg)

[1] Ni J, Young T, Pandelea V, et al. Recent advances in deep learning based dialogue systems: A systematic survey[J]. arXiv preprint arXiv:2105.04387, 2021.  

[2] Wu C S, Hoi S C H, Socher R, et al. TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue[C]//Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020: 917-929.

[3] Peng B, Li C, Li J, et al. Soloist: Building task bots at scale with transfer learning and machine teaching[J]. Transactions of the Association for Computational Linguistics, 2021, 9: 807-824.

[4] Su Y, Shu L, Mansimov E, et al. Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System[J]. arXiv preprint arXiv:2109.14739, 2021.

[5] He W, Dai Y, Zheng Y, et al. GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection[J]. arXiv preprint arXiv:2111.14592, 2021.

