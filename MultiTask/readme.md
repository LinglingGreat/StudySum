
举例：

1. 推荐系统排序模型

腾讯PCG在推荐系统顶会RecSys 2020的最佳长文：

Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations；

2019年RecSys中Youtube排序模块论文：

Recommending what video to watch next: a multitask ranking system；

以及广告推荐、新闻推荐、短视频推荐、问答推荐算法工程师们都经常使用/尝试过的MMOE模型Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts。

2. NLP中著名的多跳（multi-hop）问答数据集榜单HotpotQA上出现了一个多任务问答框架（2020年11月是榜1模型，名字叫IRRR），把open domain question answering中的信息抽取，排序，问答统一用一个模型来解决，既简洁又拿了高分！

## 概念
单任务学习（single task learning）：**一个loss，一个任务，例如NLP里的情感分类、NER任务一般都是可以叫单任务学习**。

多任务学习（multi task learning）：**简单来说有多个目标函数loss同时学习的就算多任务学习。例如现在大火的短视频，短视频APP在向你展示一个大长腿/帅哥视频之前，通常既要预测你对这个视频**感兴趣/不感兴趣，看多久，点赞/不点赞，转发/不转发**等多个维度的信息。这么多任务既可以每个任务都搞一个模型来学，也可以一个模型多任务学习来一次全搞定的。

为什么需要

1. 方便，一次搞定多个任务，这点对工业界来说十分友好。

假设要用k个模型预测k个任务，那么k个模型预测的时间成本、计算成本、存储成本、甚至还有模型的维护成本都是大于一个模型的。


2. 多任务学习不仅方便，还可能**效果更好**！针对很多数据集比稀疏的任务，比如短视频转发，大部分人看了一个短视频是不会进行转发这个操作的，这么稀疏的行为，模型是很难学好的（过拟合问题严重），那我们把预测用户是否转发这个稀疏的事情和用户是否点击观看这个经常发生事情放在一起学，一定程度上会**缓解模型的过拟合，提高了模型的泛化能力**。

3.  多任务学习能提高泛化能力，从另一个角度来看，对于数据很少的新任务，也解决了所谓的“冷启动问题”。
    
4.  多个任务放一起学的另一个好处：
    
    数据增强，不同任务有不同的噪声，假设不同任务噪声趋向于不同的方向，放一起学习一定程度上会抵消部分噪声，使得学习效果更好，模型也能更鲁棒。
    
    NLP和CV中经常通过数据增强的方式来提升单个模型效果，多任务学习通过引入不同任务的数据，自然而言有类似的效果。
    
5.  任务互助，某些任务所需的参数可以被其他任务辅助训练的更好，比如任务A由于各种限制始终学不好W1，但是任务B却可以轻松将W1拟合到适合任务A所需的状态，**A和B搭配，干活儿不累**～。

另一种机器学习角度的理解：**多任务学习通过提供某种先验假设（inductive knowledge）来提升模型效果，这种先验假设通过增加辅助任务（具体表现为增加一个loss）来提供，相比于L1正则更方便（L1正则的先验假设是：模型参数更少更好）**。

## 基本模型框架

通常将多任务学习方法分为：**hard parameter sharing**和**soft parameter sharing**。

**一个老当益壮的方法：hard parameter sharing**

即便是2021年，hard parameter sharing依旧是很好用的baseline系统。无论最后有多少个任务，底层参数统一共享，顶层参数各个模型各自独立。由于对于大部分参数进行了共享，模型的过拟合概率会降低，共享的参数越多，过拟合几率越小，共享的参数越少，越趋近于单个任务学习分别学习。

**现代研究重点倾向的方法：soft parameter sharing**

![](img/Pasted%20image%2020220123143905.png)

底层共享一部分参数，自己还有独特的一部分参数不共享；顶层有自己的参数。底层共享的、不共享的参数如何融合到一起送到顶层，也就是研究人员们关注的重点啦。这里可以放上咱们经典的MMOE模型结构，大家也就一目了然了。和最左边（a）的hard sharing相比，（b）和（c）都是先对Expert0-2（每个expert理解为一个隐层神经网络就可以了）进行加权求和之后再送入Tower A和B（还是一个隐层神经网络），通过Gate（还是一个隐藏层）来决定到底加权是多少。

看到这里，相信你已经对MTL活跃在各大AI领域的原因有一定的感觉啦：把多个/单个输入送到一个大模型里（参数如何共享根据场景进行设计），预测输出送个多个不同的目标，最后放一起（比如直接相加）进行统一优化。

## 改进方向

### 模型结构设计

**模型结构设计：哪些参数共享，哪些参数不共享？***想象一下我们是竖着切了吃，还是横着一层侧概念拨开了吃。*

A . 竖着切了吃：对共享层进行区分，也就是想办法给每个任务一个独特的共享层融合方式。MOE和MMOE模型就是竖着切了吃的例子。另外MMOE在MOE的基础上，多了一个GATE，意味着：多个任务既有共性（关联），也必须有自己的独特性（Task specific）。共性和独特性如何权衡：每个任务搞一个专门的权重学习网络（GATE），让模型自己去学，学好了之后对expert进行融合送给各自任务的tower，最后给到输出，2019年的SNR模型依旧是竖着切开了吃，只不过竖着切开了之后还是要每一小块都分一点放一起送给不同的流口水的人。（可能某块里面榴莲更多？）

![](img/Pasted%20image%2020220123144230.png)

**B：一层层拿来吃**：对不同任务，不同共享层级的融合方式进行设计。如果共享网络有多层，那么通常我们说的高层神经网络更具备语义信息，那我们是一开始就把所有共享网络考虑进来，还是从更高层再开始融合呢？如图最右边的PLE所示，Input上的第1层左边2个给了粉色G，右边2个给了绿色G，3个都给了蓝色G，然后第2层左边2块给左边的粉色G和Tower，右边两块输出到绿色G和Tower。

NLP的同学可以想象一下phrase embedding， sentence embedding。先用这些embedding按层级抽取句子信息之后，再进行融合的思路。图最右边PLE第2层中间的蓝色方块可以对应sentence embedding，第2层的粉色方块、绿色方块可以对应两个phrase embedding。虽然一句话里的多个短语有自己的含义，但是毕竟还是在一句话里，他们在自己独特的语义表示上，当然也会包含了这句话要表达核心思想。

![](img/Pasted%20image%2020220123144416.png)

### **MTL的目标loss设计和优化改进**

既然多个任务放在一起，往往不同任务的数据分布、重要性也都不一样，大多数情况下，直接把所有任务的loss直接求和然后反响梯度传播进行优化，是不是不合适呢？

**优化思路**

1.  loss的权重进行设计，最简单的权重设计方式是人为给每一个任务一个权重；
    
2.  根据任务的Uncertainty对权重进行计算，可参考经典的：
    
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics。
    
3.  由于不同loss取值范围不一致，那么是否可以尝试通过调整loss的权重w让每个loss对共享Wsh 参数贡献平等呢？
    
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks，另外一篇相似思路的文章End-to-end multi-task
    
4.  learning with attention 提出一种Dynamic Weight Averaging的方法来平衡不同的学习任务。
    
5.  Multi-Task Learning as Multi-Objective Optimization对于MTL的多目标优化理论分析十分有趣，对于MTL优化理论推导感兴趣的同学值得一看。


### **直接设计更合理的辅助任务**

辅助任务设计的常规思路：

1.  找相关的辅助任务！
    
    不想关的任务放一起反而会损害效果的！
    
    如何判断任务是否想关呢？
    
    当然对特定领域需要有一定的了解，比如视频推荐里的：
    
    是否点击+观看时长+是否点赞+是否转发+是否收藏等等。
    
    。
    
2.  对于相关任务不太好找的场景可以尝试一下对抗任务，比如学习下如何区分不同的domain的内容。
    
3.  预测数据分布，如果对抗任务，相关任务都不好找，用模型预测一下输入数据的分布呢？
    
    比如NLP里某个词出现的频率？
    
    推荐系统里某个用户对某一类iterm的点击频率。
    
4.  正向+反向。
    
    以机器机器翻译为例，比如英语翻译法语+法语翻英语，文本纠错任务中也有类似的思想和手段。
    
5.  NLP中常见的手段：
    
    language model作为辅助任务，万精油的辅助任务。
    
    那推荐系统里的点击序列是不是也可以看作文本、每一个点击iterm就是单词，也可以来试一试language model呢？
    
    （纯类比想象，虽然个人感觉有点难，毕竟推荐系统的点击序列iterm不断变化，是个非闭集，而NLP中的单词基本就那些，相对来说是个闭集）。
    
6.  Pretrain，某正程度上这属于transfer learning，但是放这里应该也是可以的。
    
    预训练本质上是在更好的初始化模型参数，随意想办法加一个帮助初始化模型参数的辅助任务也是可以的～～。
    
7.  预测要做的任务该不该做，句子中的词位置对不对，该不该放这里，点击序列中该不该出现这个iterm？
    
    这也是一个有趣的方向。
    
    比如文本纠错任务，可不可以先预测一下这个文本是不是错误的呢？
  

对推荐系统感兴趣的同学可以阅读阿里的DIN：Deep Interest Evolution Network for Click-Through Rate Prediction，看看阿里爸爸如何科学的加入辅助loss。另一篇ESSM也可以从辅助任务设计这个角度学习学习：Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate。
  

对NLP感兴趣的同学呢，也可以从如下角度入手看看是否可行：

比如对于问答（QA）任务而言，除了预测答案的起始位置，结束位置之外，预测一下哪一句话，哪个段落是否包含答案？其实个人实践中，**先过滤掉一些噪声段落，再进行QA大概率都是会提升QA效果的**～～特别是有了预训练BERT、Roberta、XLNET这些大规模语言模型之后，把句子、段落是否包含答案作为二分类任务，同时预测答案的位置，模型的Capacity是完全够的！

对于信息抽取任务，是否可以把词/短语级别的sequence labeling 任务和关系抽取任务放一起呢？比如SRL和关系抽取，entity识别和entity关系抽取。不过在那之前，建议看看danqi chen这个令人绝望的模型：A Frustratingly Easy Approach for Joint Entity and Relation Extraction。

对于multilingual task和MT（机器翻译），其实有个比较简单的思路就是多个语言、多模态一起学，天然的辅助任务/多任务。另外在以前的机器翻译任务里）大家还喜欢加上POS/NER这种任务来辅助翻译。

比较有资源的大厂呢，比如微软，Multi-Task Deep Neural Networks for Natural Language Understanding，百度的ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding，好家伙，相似的任务都放一起吧哈哈。

## 技巧和注意事项

**数据！洗掉你的脏数据！**

NLP领域大家都专注于几个数据集，很大程度上这个问题比较小，但推荐系统的同学们，这个问题就更常见了，**静下心来理解你的数据、特征的含义、监督信号是不是对的，是不是符合物理含义的（比如你去预测视频点击，结果你的APP有自动播放，自动播放算点击吗？播放多久算点击？）。**

对于想快速提升效果的同学，也可以关注以下几点：

1.  如果MTL中有个别任务数据十分稀疏，可以直接尝试一下何凯明大神的Focal loss！
    
    笔者在短视频推荐方向尝试过这个loss，**对于点赞/分享/转发这种特别稀疏的信号，加上它说不定你会啪一下站起来的**。
    
2.  仔细分析和观察数据分布，如果某个任务数据不稀疏，**但负例特别多，或者简单负例特别多，对负例进行降权/找更难的负例也可能有奇效果哦。正所谓：负例为王。**
    
3.  另外一个其实算trick吧？
    
    将任务A的预测作为任务B的输入。
    
    实现的时候需要注意：
    
    任务B的梯度在直接传给A的预测了。
    

最后是：读一读机器学习/深度学习训练技巧，MTL终归是机器学习，不会逃离机器学习的范围的。该搜的超参数也得搜，Dropout，BN都得上。无论是MTL，还是single task，**先把baseline做高了做对了，走在正确的道路上，你设计的模型、改正的loss，才有置信的效果～～**。

## 学习指南

几个经典的代码库，方便大家操作和学习：

1.  multi task example：
    
    https://github.com/yaringal/multi-task-learning-example.git
    
2.  MMOE https://github.com/drawbridge/keras-mmoe.git
    
3.  NLP福利库包含各大SOTA的BERT类模型：
    

https://github.com/huggingface/transformers.git

1.  百度ERNIE https://github.com/PaddlePaddle/ERNIE.git
    
2.  微软MT-DNN https://github.com/namisan/mt-dnn.git

## 参考资料

[收藏｜2021年浅谈多任务学习](https://mp.weixin.qq.com/s/hbtrijHy2E177fA7oe7SSA)

