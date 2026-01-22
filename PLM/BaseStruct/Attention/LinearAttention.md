---
title: LinearAttention
created: 2026-01-07
tags:
  - attention
---
[119. Kimi Linear、Minimax M2？和杨松琳考古算法变种史，并预演未来架构改进方案 - YouTube](https://www.youtube.com/watch?v=858HR43pegk)

[微信公众平台](https://mp.weixin.qq.com/s/USj6wbOI8CD0MTQCQ01hXQ)

今天这集节目，我们将讨论一个在当下非常关键的话题：人工智能的算法与架构创新。

嘉宾是我们的往期嘉宾返场，她是MIT在读博士杨松琳，研究方向是线性注意力机制。

我们将从最新发布的几个模型Kimi Linear、Minimax M2、Qwen3-Next切入。松琳参与讨论Kimi Linear和Qwen3-Next的部分工作，是Kimi Linear论文的作者之一。

算法创新为什么在2025年变得尤为重要？

它的背后原因是，数据、算力和算法是驱动人工智能的三驾火车，在高质量数据获取难度增大的无奈前提下，各个模型公司不得不重新开始“雕模型架构”，以期Scaling Law的魔法继续。而由于中国的算力相对美国有限，这反而让中国的AI算法创新走在了世界前沿。

这集节目你将听到，近几年架构的最大突破是MoE（混合专家模型）；而下一个突破的重要方向可能就是Attention（注意力机制）。

中国公司在Attention展开了不同技术bet（押注）：

- 截至目前已发布模型，DeepSeek正在探索Sparse Attention（稀疏注意力机制）；
    
- Kimi正在探索Linear Attention（线性注意力机制）；

- Minimax在年初的M1版本中探索Linear Attention，而在刚发布的M2版本中又回退到 Full Attention（全局注意力机制）。
    

节目中，松琳将讲解她参与讨论的这篇《Kimi Linear: An Expressive, Efficient Attention Architecture》的工作，并分析以上这些公司在Attention上的不同抉择；

与此同时，她也将带领大家考古人工智能算法变种史，并预演未来算法与架构的改进方案。

  

# 01   最近更热门的是Hybrid架构

---

张小珺：Hello，松琳，先给听众朋友们打个招呼，并做一个简单的自我介绍。

杨松琳：Hello，大家好，我叫杨松琳，我现在是MIT CSAIL的Ph.D.在读。我的主要研究方向是大型语言模型（Large Language Model）的架构，主要是研究比较高效的注意力机制（Attention Mechanism）。更具体地说，我正在深入研究一类注意力模型，我们称之为线性注意力（Linear Attention）。

张小珺：能不能给大家讲讲你的研究主线是如何递进的？你是怎么走向Linear Attention研究的？

杨松琳：关于Linear Attention，最开始，我应该是当时阅读了斯坦福大学Hazy Research团队的很多博客，觉得序列建模（Sequence Modeling）是一个非常有意思的问题，因此决定做一些序列建模相关的研究。

我读博之初，是受到微软亚洲研究院一篇名为RetNet工作的启发，最开始是想办法来提高RetNet的效率和性能（Performance）。之后我发现，提高效率的这套硬件优化算法可以扩展到很多其他类似的架构中。随后的工作，主要是进一步在能够实现硬件高效训练的同时，提高线性注意力架构的性能。

举例来说，从门控机制，到一种叫做Delta Rule的机制。再往后，我将这两个机制结合在一起，合成了一个统一的Rule，并将其转化为RNN的更新规则。同时，这套规则还可以配合一些可以实现硬件高效的算法来进行训练。

张小珺：我看到上次我们节目播出后，有人叫你“Linear Attention 之母”，这是为什么？

杨松琳：可能是我在这个领域确实做了不少工作，尤其是有一个开源库叫做Flash Linear Attention。这个领域内很多人在使用这个库，包括业界也有很多人使用这个库进行Linear Attention的探索。另外，我的那几篇工作也比较有影响力，所以大家可能会这么叫我。

张小珺：能否用更通俗的方式，向大家解释一下Linear Attention？

杨松琳：“线性”在这里主要指的是线性复杂度（Linear Complexity）。线性复杂度对应的是平方复杂度（Quadratic Complexity），也就是我们平常使用的 Softmax Attention 是平方复杂度。

我们都知道Softmax Attention有三个矩阵：Q (Query)、K (Key)、V (Value)。通常的做法是Q和K先进行矩阵相乘，得到一个$L \times L$的矩阵，这里的$L$是序列长度（Sequence Length）。接着，我们会对这个$L \times L$的矩阵做一个Masking（遮蔽），因为它基本上都是自回归的语言建模，所以我们需要把未来的信息遮蔽掉，从而得到一个下三角（Lower Triangular）的$L \times L$矩阵。然后我们再施加一个Softmax，这样我们就得到了一个注意力分数矩阵（Attention Score Matrix）。最后，再用这个注意力分数矩阵和Value矩阵相乘，得到最终的输出（Output）。这就是Softmax Attention在自回归建模中的粗略介绍。

因为它会生成一个$L \times L$的矩阵，所以它的计算复杂度是平方的。而线性注意力通常会把这个Softmax Operator去掉。这样，去掉了这个非线性的Softmax后，我们可以通过一些等式变化，把它写成一个类似于RNN的推理形式。

如此一来，它每一步的计算成本（Cost）就是 $O(1)$。处理一个长度为$L$的序列，它的整体复杂度就是$O(L)$。所以，它是跟长度$L$呈线性关系，因此大家称之为线性注意力。

张小珺：如果把现在大语言模型的算法做一个整体框架图，让大家有一个背景，Linear Attention应该被放在哪个位置？

杨松琳：都是在Transformer这个基础架构里面进行一些魔改。

LLM的技术栈可能分为Pre-training、Post-training等阶段。而像架构的研究，肯定是在Pre-training这个阶段。Pre-training还有很多其他类别的研究，比如优化器（Optimizer）、基础架构，以及Pre-training Data等。线性注意力就属于基础架构的研究范畴。

当前的基础架构，整体框架仍然是Transformer，它会有一个注意力机制和一个前馈网络（Feedforward Network），也就是FFN。它会在这两个模块里面反复叠加多次，得到我们最经典的Transformer Architecture。一般而言，大家都是在这个框架下进行修改。

近几年的趋势是，大家会把传统的MLP或者FFN换成混合专家（Mixture-of-Experts, MoE）模块。而线性注意力就是把传统的Softmax Attention换成一些线性复杂度的Attention。

当然，最近更热门的是一类叫做Hybrid的架构，即有一些层仍然是Softmax Attention，但大部分的层被换成了线性注意力层。

  

# 02  如果每一层都使用平方注意力架构，在解码时太昂贵了

---

  

张小珺：我们来聊聊你最近参与的一个新工作Kimi Linear。你是怎么参与到Kimi Linear的工作中的？这个工作应该是10月底刚发布。

杨松琳：这个工作他们应该是年初就开始着手了。当时Flash Linear Attention这个库的另一位主要作者叫张宇，他正好是今年博士毕业，在国外读博。当时他正好在Kimi，而Kimi正好想做这个混合注意力（Hybrid Attention）。张宇就是在负责这个项目。

因为他也是FLA开源库的合作者（Collaborator），我们很熟，所以我也会帮他们看一下，比如对于一些线性注意力的变种，它们的并行算法该如何设计等等。

张小珺：当时他们团队遇到的核心问题是什么？为什么决定要重新设计注意力机制？

杨松琳：年初，大背景是像DeepSeek R1和Kimi 1.5刚刚发布。它们的核心是会做一些RL（Reinforcement Learning），并且会得到一些非常长的思维链（Chain-of-Thought）。它们会用这个非常长的思维链来做Test Time Scaling，从而解决一些比较复杂的问题。这个思维链的长度往往能够达到几万个Token。

Kimi团队认为，如果每一层都使用平方注意力的架构，那么在解码（Decoding）时就太昂贵了。

首先，每一层都需要存储大量的KV Cache；其次，每一步的解码时间复杂度是线性的，所以如果解码$L$个Token，它的时间复杂度也是平方的。

因此，在这种长思维链生成的背景下，以及今年整体Agentic AI的背景下，Kimi认为需要投入资源来探索这种混合注意力，因为它能够大幅降低推理（Inference）的成本。这一点在这种长思维链（Long Context）和Agentic AI的背景下是非常有用的。大概背景就是这样。

张小珺：魔改的核心目标是什么？

杨松琳：当时的核心目标可能主要是张宇在那里做，他们的目标应该是：相比于之前的Full Attention，性能（Performance）不能下降（不掉点），同时推理速度要快很多倍。

张小珺：如果用Full Attention，缺陷会是什么？

杨松琳：使用Full Attention在进行长文本解码时，成本是非常昂贵的。

张小珺：能不能从你的视角给大家讲解一下这篇论文《Kimi Linear: An Expressive, Efficient Attention Architecture》，划一下重点。

杨松琳：像这篇文章，他们最终选定的线性注意力模块，叫做KDA，即Kimi Delta Attention。这个名字我觉得挺有梗的，他们应该是想对标DeepSeek Sparse Attention，所以特意取了一个Kimi开头、非常对仗的名字。

这个线性注意力模块基本上是基于我去年的一项工作叫做Gated Delta Net，在这个基础上进行了一些改进，最终形成了KDA模块。

总的来说，首先我们有一个叫做Delta Rule的机制，之后可以再具体讲解。

在Gated Delta Net的工作中，当时受限于效率（Efficiency）问题，我用到了一个类似于Mamba-2的标量门控（Scalar-valued Gating）。这意味着，对于一个Attention Head来说，它下面的所有维度（Dimension）都要共享一个衰减率（Decay Rate）。这样可以在计算上带来一些简化。所以当时的考虑是，我先在Mamba-2的基础上加上Delta Rule，从而确保它的效率。因此，当时只用到了那种粒度比较粗的门控机制。

而张宇玩的这个KDA，就是把这个粒度比较粗的衰减率，换成了一个粒度比较细的衰减率（Fine-grained Decay）。之前是一个Attention Head下面，不同维度需要共享同一个衰减率；现在是不同的维度，每个维度都有自己独立的衰减率。这样，每个维度对应的RNN隐藏状态（Hidden State）的更新频率就是独立的。从直觉上来看，它能更好地利用RNN有限的Hidden State，从而提高性能。

张小珺：你们的设计逻辑和灵感来源于？

杨松琳：我觉得KDA的设计是把我前两个工作的Idea合并在一起。我之前还有一个工作叫做Gated Neural Attention，它就是使用这种粒度比较细的衰减率。后来到Gated Delta Net的时候，当时之所以没有用到这种粒度比较细的衰减率，是因为当时算法本身和核优化（Kernel Optimization）都没有优化到一个比较好的状态。考虑到效率问题，我们“被迫”只能使用Mamba-2那种粒度更粗的衰减率。

但后来，在算法层面和核优化层面都有了很大的进步。到今年年初，大家就觉得是不是可以重新来研究一下，能不能把这个Fine-grained Decay（粒度比较细的衰减率）重新引入到Gated Delta Net里面。

张小珺：你们设计完最初的效果怎么样？

杨松琳：我记得张宇应该是先尝试了一大堆这种混合注意力的混法。他最开始发现混Gated Delta Net比混其他的要好。后面，因为Kimi内部有一个叫Scaling Ladder的机制，就是说你在一个规模下表现好，那你就要到下一个规模去继续Scale。这有点像通关一样，有很多关卡，过了一关之后，他可能要到下一关去继续跟Full Attention比较。

最开始，他可能发现Hybrid Gated Attention在某些地方还是不如那种Full Softmax Attention。后面他开始尝试把那个Decay换成这种更加细粒度的Decay，他发现在他的一些实验下面，提升还是挺大的。

  

# 03  DeepSeek vs Kimi vs MiniMax

---

  

张小珺：Kimi Linear Attention和DeepSeek Sparse Attention，在你看来它们的表现哪个更好？分别适合什么样的任务？

杨松琳：这两种Attention实际是想解决同一个问题：在长文本解码下面，如何解决效率问题。Kimi走的是混合注意力的路线。千问（Qwen）也走的是这一条，现在主要是投入混合注意力的路线。而DeepSeek则主要喜欢走稀疏注意力（Sparse Attention）的路线。

他们可能觉得稀疏是一种更好的方式来降低解码成本（Decoding Cost）。像DeepSeek Sparse Attention，它应该是没有Full Attention层，所以它应该是每一层都是DeepSeek Sparse Attention。但是，它每一层都要把所有的KV Cache都存下来。它只能通过一个Check Point，经过一些蒸馏，得到一个叫做Indexer的东西，来选择那些Top K的Token。这就是DeepSeek Attention。

而混合注意力这条路线，它还是保留了一些全局注意力层，但它那些比较快的层是线性注意力层。这个好处就是它可以节省很多KV Cache。

混合注意力不仅能减少KV Cache，因为它绝大多数层都是那种类似于RNN的层，所以能减少很多KV Cache的体积。同时，它也能提高解码效率。因为它减少了KV Cache的大小，所以在做解码时，可能就可以使用更大的Batch Size。因为之前可能放不下，现在KV Cache减少了很多，就可以加大Batch Size。

而DeepSeek Sparse Attention没有减少KV Cache的作用，但它可以通过Sparse的激活（Activation）来减少每个Token生成的花费。

张小珺：还有一家Minimax，他们最新也做了一个算法的选择。

杨松琳：对，它应该算是这种混合线性和平方注意力的先驱，因为它年初发布的M1那个版本是一个非常大规模的混合注意力实践。而他们前几天发布了一个叫做M2的模型，这个模型现在就变成了一个Full Attention。它既不是混合注意力，也不用Sparse Attention，它干脆退回到了Full Attention。

张小珺：这是为什么？

杨松琳：我觉得他们的负责团队非常开放（Open），他们分享了很多宝贵的经验。我记得他们提到，M1版本他们监控的一些指标显示，他们用到的那个Lightning Attention模块在这些指标上表现得都很好，而且Lightning Attention效率更高一点，所以他们最终就采用了这个Lightning Attention。

但是后来他们发现，如果在一些例如叫做Multi-Hop Reasoning（多跳推理）这样的任务（Task）上，掉点会非常大。当时之所以选用那个方案，是因为他们最开始没有去检测这种多跳推理的能力，他们主要只看MMLU之类的能力。

他们选择的那个Lightning Attention，就我来说，我觉得它是一个比较弱的线性注意力，因为它那个机制给人的感觉就像是两年前的一个Linear Attention。那个技术还停留在两年前。很有可能就是因为他们第一版做评价的Pipeline不够详尽，导致他们选择了一套比较略显落后的方案。

最近他们可能是想做Agent Task、Coding，那么多跳推理这个能力就会在这种场景下变得非常重要。他们就发现Lightning Attention和Full Attention之间的性能差距还挺大的，所以他们暂时退回到了全部都是Self Attention的Full Attention架构。

但他们说他们还在继续探索混合注意力架构，说不定下一版M3又会变回混合注意力架构。

张小珺：你怎么看待大家在算法上的不同选择或者反复？

杨松琳：历史就是螺旋上升的。一套技术方案肯定要经过很多验证才能最终定下来。像M1可能当时验证得不够充分，当时比较草率地就上了。后来发现它在多跳推理上效果不好，就暂时退回来，这个也是很正常的。

张小珺：硅谷公司现在对于混合注意力机制，探索方向是什么样的？各家有什么不一样？

杨松琳：这个我感觉我不能讲。

张小珺：OpenAI可以讲吗？

杨松琳：OpenAI 我只能讲一些有论文（Paper）的方案，没有论文的方案我是不会讲的。

OpenAI比如像GPT-3，它在Technical Report里就讲了，它会用到一个混合的全局注意力和一个Local的Sliding Window Attention这么一个混合方案。这个在GPT-3的报告里已经明确写出来了，所以是可以讲的。然后像他们最近发布的OSS开源模型，也用到了滑动注意力（Sliding Window Attention）的方案。

所以他们应该是一直在使用这一套滑动注意力的方案。

  

# 04  每次大家关心Linear Attention，肯定是大家碰到了Context Wall

---

  

张小珺：我们倒回去讲一下，你刚才说Linear Attention这两年发展很多，你能给大家讲一下它的这个发展线索吗？

杨松琳：像Linear Attention，它最开始的时候我觉得非常不Work。它就算在短文本下面也不Work。最早的Linear Attention是2020年发明的，我觉得它可能中间这几年，在语言建模（Language Modeling）上面都没有取得很好的效果。

后来一个比较有代表性的工作是RetNet。它通过加入一个遗忘衰减（Forgetting Decay）机制，发现Linear Attention Scale上去后，在语言建模上还是可以取得一个比较好的效果的。

RetNet往模型中加入了一个输入无关（Input-agnostic）的Decay。输入无关的Decay就是说，它的遗忘率是跟输入没有关系的。比如说它的遗忘率是0.99，那么它每过一个Token，前面的Hidden State就要乘上0.99，这样它就要遗忘掉它1%的内容，然后再把新的内容写进去。这就是一个叫做输入无关的衰减，是RetNet用到的技术。

这种输入无关的遗忘，在之后逐渐被替换成了输入相关（Input-dependent）的衰减。

比如我之前的一个工作叫Gated Linear Attention，前面也提到了，就是加了一个门控机制。像Mamba和Mamba-2，它们也和线性注意力有很多联系，尤其是Mamba-2。Mamba-2基本上就可以看成是线性注意力，它加了一个衰减，但是这个衰减跟RetNet非常像，它跟RetNet的区别是：那个衰减是由输入来决定的。也就是说，每一个Token的衰减率可能不一样。

例如，它遇到有些Token，觉得前面的内容没有必要忘记，它可以把衰减率设为1，这样前面的State就根本不会做衰减。如果它遇到一些Token，觉得前面这些信息已经没有用了，那它可以在那个位置上把衰减率设为0，这样前面的State就会被完全忘记。这种输入相关的Decay比较灵活，能够通过数据动态地学习什么时候该遗忘，什么时候该记忆前面的State。这是第一个比较大的改进，就是把衰减从输入无关变成输入相关。

第二个改进就是Delta Net这一条路线。它将更新的公式，从最开始那个简单的，例如Linear Attention用到的一个叫做Hebbian Rule（赫布规则）。这个Rule只是简单地把Key和Value它们的外积（Outer Product）加到Hidden State上面。

而像Delta Net这一套模型，它用到的是一个叫做Delta Rule的东西。Delta Rule意味着每一步时，它先用这个Key去取出那个Memory里面的值，这就是这个Key在Memory里本来对应的Value，我们称之为Old Value。然后这个Key又会有一个输入的Value，我们称之为Input Value。

因为这是一个关联记忆网络视角，我们想让每一个Key只对应一个Value。模型也不知道它应该对应前面的Old Value还是输入的Value。这样的话，我们又有一个可以学习的系数叫做Beta，我们可以看成它是一个在$0$到$1$之间的系数，用来决定我们要用多少前面的Old Value，以及要用多少输入的Value。

我们会通过这个系数来做一个线性组合，得到它最终新Value。然后，将这个旧Value和Key的外积从Memory里面减去，再把这个新Value和Key的外积加到这个关联网络里面。这就是Delta Net那个更新公式的一个High Level的Idea。

相比于Linear Attention，它是有一个减法操作在里面的。加法可以想象成是往这个记忆网络里面去记忆东西；那么减法就可以理解成从这个记忆网络里面删除一些东西。这种方式会比较更有针对性地来删东西。像之前的Decay可能就是很多维度一起在做Decay；像现在就是只取某一个分量，然后它有一些非常有目标性的删除操作在里面。

所以，以Delta Rule为代表的第二个改进，应该是Linear Attention这个领域里面最近的第二个改进了，包括像Delta Net、Gated Delta Net、像Raku-7他们都用到了这个Delta Rule。

张小珺：为什么Linear Attention从一开始效果不好到慢慢改进，大家相信它还是Promising的？

杨松琳：我觉得每一次大家关心Linear Attention，肯定是因为大家碰到了一些上下文墙（Context Wall）。最开始大家去研究Linear Attention，例如在2020年左右，是因为那个时候大家遇到了第一个Context Wall，它撞到了这堵墙：如果想继续提高上下文长度（Context Length），就只能找一些复杂度小于平方的东西来了。

当时像BERT那个年代，它的训练长度就是512。当时可能觉得2048、8192就算长文本了，因为那个地方就会变得非常慢。后来，随着Flash Attention这个技术诞生，打破了这堵墙。现在看来8192已经是一个非常短的文本了，在上面做训练没有任何压力。

但在之前没有Flash Attention的时候，计算需要把这个平方的Attention矩阵materialize在Global Memory上面，再把它从Global Memory搬回Flash Memory里面。这样它的Memory读写整体开销是非常大。同时，因为Attention矩阵会被实例化在Global Memory，它可能还会带来Out of Memory的问题。这就是最开始大家研究Linear Attention的动机（Motivation）。

随着Flash Attention出现，大家发现这堵墙已经被打破了。既然我们能用这种Exact的方式来直接计算Softmax Attention，那我们就没必要找一些Linear Attention去逼近（Approximate）它了。所以，大家对Linear Attention的研究就开始没有那么受关注了。

直到最近，例如长文本解码又重新成为了一个需求量非常大的东西。大模型需要吐出非常多的Token，要做这种Decoding。这个花费又会让人不由自主地重新来审视这一套技术。

同时，这一套技术本身在学界又有了这么久的发展，尤其是在Flash Attention之后，学界也意识到，如果像Linear Attention这一套模型想要被大家接受，那么它在硬件上面的效率是非常关键的。

这也就是为什么我最开始搞了一个叫做Flash Linear Attention的Project，致力于将这些Linear Attention的变种，用Triton写成一个库和很多Kernel，让它能够在当代硬件上，主要是GPU上面，能够来快速运行。

张小珺：所以它的核心是效率更高，价格更低。

杨松琳：对，每当Softmax的效率变成一个瓶颈的时候，大家就会回来看Linear Attention。大概就是这样的一个历史。

  

# 05 非共识与共识

---

  

张小珺：Linear Attention现在是业界共识了吗？

杨松琳：我觉得关于Linear Attention，现在的共识是：纯Linear Attention是不Work的。它在这种长文本下面，有一些比较基础（Fundamental）的缺陷。所以现在大家一般都不会去尝试这种纯线性的模型。

像一些比较折中的方案，例如这种混合注意力，它还是有很多很多的线性注意力层，但它还是有一定数目的全局注意力层。

这样，这个模型的下限是有保证的，它处理长文本也是有一定保证的。因为它终归还是有很多全局注意力层。像全线性的网络，它可能从理论上就没有办法做那种长文本的Task。因为它的RNN状态数目是恒定的，随着Context长度增加，它迟早会存不下，迟早会损失很多那种精度在里面。

但是像混合注意力，它有很多全局注意力在里面，所以还是可以通过这些全局注意力来完成这些长文本的Task。像Kimi Linear的这篇论文，以及像之前千问-3 Next，他们的长文本，比如像Ruler、以及其他Task的表现没有掉点。所以它在长文本上面还是有一定能力。

所以，混合注意力会受到很多地方的关注，但我也不知道它算不算共识。因为不同的地方还是在尝试不同的方案，比如说像DeepSeek，它就在尝试Sparse Attention的方案。

张小珺：在Kimi的论文里，提出的是每三层的KDA，也就是Kimi Delta Attention（增量注意力机制），插入一层Full Attention（全注意力机制）。这个比例是怎么确定的？

杨松琳：我觉得3:1现在也快变成一个共识了吧。像Minimax之前是一个7:1的比例。7:1可能Softmax Attention的层数不够，长文本的那个保证可能没有那么好。

我记得之前就是字节也发了一篇Paper，来研究这个Hybrid架构，它需要百分之多少的Softmax Attention。他们的结论也是说，他们做了很多Pre-train from Scratch的实验，通过改变不同的Linear Attention模块，也改变混合比例，他们的结论大概是说3:1的比例是最好的，而且Gated Delta Net这个模块比其他的Candidate要好。后面千问-3 Next也用到了3:1，换成了Gated Delta Net这个方案。

这个方案应该是不同的厂商探索出来都觉得这个比例是更好的。这可能是最开始Minimax没有验证充分，他们可能最开始的评测还是有一些不足，所以他们用到了一个更激进（Aggressive）的方案，就是7:1。

现在的话基本上都回到了3:1这个上面来了。我觉得3:1应该就是在不共识的Hybrid Linear里面的一个共识了，大家用3:1的比例来混合这个模型。

张小珺：是不是你们在算法设计的时候，始终要平衡表达能力和计算效率？这两者是它的核心北极星指标吗？

杨松琳：确实，我觉得还是有一些Trade-off（权衡）的。像全局注意力如果太少，我觉得像这种Reasoning Task，然后像长文本Task，它肯定会受影响比较大。它可能一些Short Context的Task没有什么影响，比如MMLU，但是那些长文本和推理的Task能够看到比较大的现象。

但从另外一方面来说，也不是说Attention层越多越好。因为大家如果训练完之后会发现，绝大多数Attention层可能就是没有用的。它只有一些关键的层的Attention是有用的，但它不是每一层的Attention都有用。

这个网络本身自己它就是有一个冗余度在里面的。这样就给我们带来一些机会，比如我们可以把一些层换成一些线性层。

所以，混合的架构不一定就代表它比全局的要差。它很有可能就是说，它可能是一个全面更好的替代方案。

像Minimax之前也承认了一点，我觉得非常棒，他们发现Hybrid Linear Attention或者Hybrid这种滑窗注意力在长文本的Multi-Hop（多跳推理）上面会有缺陷。这应该是现在Hybrid唯一的一个问题。因为就我所知，它在其他Task上面基本上是不会比全部都是Softmax Attention要差的。它只会在多跳推理这个任务上表现不佳。这个也比较好理解，像这种多跳推理的话，它就比较吃这种Token和Token之间的关系，所以它可能就比较吃Softmax Attention的层数。

我觉得这种吃全局注意力层数的任务不是很多，可能就只有这种多跳推理，然后这种长文本做Reasoning这种会稍微吃一点。其他很多Task基本上不吃的话，那它就是完全不会受影响的。

而像这种多跳推理的Task，就是如果我们去开发一些硬件高效，但是它表达能力更好的RNN，这个Gap是有可能被直接缩小，甚至会反超这个Gap的。比如说像Kimi最近这个Leader，张宇之前玩就发现，把这个粒度粗的Decay换成粒度细的Decay之后，它在这些Multi-Hop Reasoning、Coding和Math这些Task上面，那个提升还是比较可观的。就说明这些Task Hybrid可以做得更好。

现在我觉得混合线性注意力只是一个开始。我觉得整体还是很有可能做出更好的混合注意力机制的，就是可以“雕”一下线性注意力的模块。

张小珺：你在过程中有给Kimi什么算法建议？

杨松琳：张宇想玩那个细粒度Decay，我就帮他想了一个分块（Chunk）的并行算法。这个可能就是我对这个工作的唯一贡献了吧。因为这个基本上都是张宇在Kimi做了很多很多的Ablation Study（消融实验），基本上都他做的，所以Credit（功劳）基本上都在他那里，不在我这里。

像这个算法的话，也是之前有一篇文章叫做Comba，它设计了一个新的算法，能够把Gated Delta Net那个求逆（Inversion）的算法减少一次。我看完那个算法之后我就发现，我可以把Gated Delta Net的那个求逆减少一次。我紧接着又推了一个能够适用于KDA的这个算法。

我就把这个算法告诉张宇了。张宇他就去写Kernel去实现这个算法，就发现这个算法对于它的Scalability来说，比之前的那个算法要好一点的。

  

# 06 最好的结合是，把混合注意力里的全局注意力换成稀疏注意力

---

  

张小珺：问一个很General的问题：Attention到底应该怎么设计？

杨松琳：这个问题的话，现在可能就只有两条比较主流的路线：一种就是Hybrid Linear，一种就是Sparse。这两种它其实都是非常Promising。

另外可能有一些比较非主流的一些Attention设计，比如说我上次看到Meta还放了一篇论文，搞了一个三次方的Attention，就是嫌平方复杂度还不够，它还要搞一个三次方的。

还有些地方有一些比较有意思的一些平方复杂度的Attention变种，比如说ByteDance之前有个叫做DeltaFormer，它相当于就是把Delta Rule的思想引入到Softmax Attention，能够让它表达能力更强。这个工作我觉得也非常有意思。

改进注意力的话，它要么就是把Softmax让它做得更好，要不然的话就是做一些更加高效的一些Variant，比如Sparse Attention，或者这种混合线性的Attention。这两种我觉得它也是可以结合的，它们有各自的优点和各自的缺点。

像Sparse Attention的话，它做Retrieval肯定要更强一点，但它的缺点就是说它KV Cache不能省。而像线性的话，它可以省很多KV Cache。

我之前写了一个知乎的回答，就说这两种方案为什么我们不能把它结合到一起呢？

比如我们可以让Sparse Attention去取代这种混合注意力里面的那个全局的注意力层。这样，我们就不需要有一个全局注意力的那个复杂度在了，但我们还是要存那个KV Cache。但剩下很多层的KV Cache就可以通过这个线性注意力把KV Cache的Size把它打下来。这样子的话，可能就是我目前心中比较理想的一个高效的架构了，即在高效不掉点方面。

张小珺：Linear Attention和Sparse Attention的未来关系可能是融合到一个统一框架里？

杨松琳：对，因为Linear Attention和Sparse Attention没有什么竞争关系。Linear Attention的竞争对手可能更多的是Sliding Window Attention。比如像GPT-3那个论文里面提到的那个全局混Sliding Window，如果让线性去取代这个Sliding Window能够让它更好的话，那也未尝不可。

张小珺：怎么把Linear Attention和Sparse Attention做更好的结合？现在有人在探索这件事吗？

杨松琳：工业界的话，就我所知，我应该没有看到有人在同时去结合Sparse Attention和Linear Attention。但学界有一些工作还是有一些这方面的探索的，就是有些层用Sparse，有些层用Linear Attention。

张小珺：DeepSeek选择了Sparse Attention，Kimi选择了Linear Attention，这可能是阶段性的，也许未来大家会探索一条新的路，把两者结合。

杨松琳：对，我觉得混合注意力的话，它解码长度上去之后，问题就是说它还是会被全局注意力的效率把它限制住（Bound）。后面的瓶颈就主要在这个全局注意力的效率上面了。而像全部都用那种Sparse Attention，它的瓶颈可能是在KV Cache的管理上面。因为它还是不省KV Cache。所以等长度上去了，可能要做很多各种各样的KV Cache压缩之类的功能。

两者都是还是有各自的问题的。

张小珺：它的结合是，比如说可能是不同的层用不同的Attention吗？

杨松琳：最好的结合就是，把混合注意力里面的全局注意力换成Sparse Attention。我觉得理论上只要Sparse Attention能选得准的话，它是完全可以取代Full Attention这个层的。但它现在问题可能是选不准。这是一个很大的问题。

这也是为什么可能是为什么DeepSeek它最近放的那个DSA（DeepSeek Sparse Attention），它要用蒸馏（Distillation）的方式来尽可能地让他那个Indexer，就是来选Token，选得准一点。这也可能是一个原因。

张小珺：选得准or选不准的核心瓶颈在哪？

杨松琳：我觉得就是学习难度吧。像Sparse Attention的话，如果你从头开始训练，它可能那个梯度（Gradient）不太准，然后它可能学着学着它就选不准那个Block了。它学会选Block还是挺难的，它有各种那种稀疏梯度的问题吧。

像Sparse的话，它经常就会有这种问题。而像蒸馏的方式的话，它其实就是已经让一个训练好的，全部都是Softmax Attention的一个Teacher Model来蒸馏它那个Token的选法，那这个就可以选得非常好了，这个从直觉上面来说也是Make Sense的。

张小珺：Kimi这个工作，相比年初Minimax M1的工作，进步在哪里？

杨松琳：它主要就是在于线性注意力它那个模块它还是会好很多的。就像我之前说，Lightning Attention给人的感觉就像一个两年前的工作，就还停留在RetNet那个版本。像这两年的话，线性注意力还是有很多发展的。这些发展我觉得都是Work的。

千问和Kimi都发现，这两年有一些进步，例如那个门控，比如那个Delta Rule都是有用的。所以把这些最新的进展把它融合进来肯定是更好的。

然后像Kimi甚至在之前的工作的基础上，还新开发了一个KDA，让它的那个模型能力会更强。另外，可能它还有一些其他不同，比如MoE的话，像Kimi它应该用的是细粒度MoE，然后M1我记得它那个MoE好像还比较粗，它还没有用到这么细粒度的MoE。所以就是有很多种可能性。

张小珺：Kimi Linear Attention的效果跟去年DeepSeek Sparse Attention的效果比哪个更强？

杨松琳：我觉得效果对比的话，需要有个地方来做一个Apple-to-Apple的比较。因为这个东西就是非常Tricky，不太好比。

我觉得不同的地方训练出来不同的模型，它可能就是完全不能比了。因为它那个训练架构、那个Data Recipe、那个优化方案完全都不一样。它就没有一个Apple-to-Apple的比较。

像Kimi Linear最近这个Report，他还有一点就是说他有一个Apple-to-Apple的跟Full Attention的比较。但他没有Apple-to-Sparse Attention的比较。

要是有一个地方能做慈善，来Apple-to-Apple来比一下，让大家能更好地知道就更好了。但现在因为没有人再做一个Apple-to-Apple的比较，这个问题我也不知道哪个会更好。

张小珺：为什么Kimi不做这个比较？

杨松琳：可能还是资源有限吧。如果就那么多卡的话，那可能先投入一个路线去验证。如果验证出来了，再去投入另外一个路线，看看有没有可能，比如把全局注意力再把它替换掉。感觉就是没有这么多卡来同时来跑一些不同方案的对比。

像硅谷的话就很多东西都闭源，你也不知道他们有没有跑一些Apple-to-Apple的比较。

  

# 07 下一个突破点可能就在Attention

---

  

张小珺：如何做一个公平的比较，比较一下Linear Attention和Sliding Window Attention？

杨松琳：Sliding Window Attention和Linear Attention做公平的比较的话，我觉得可以有两种。

一种就比如说控制它的状态大小（State Size）。Sliding Window它有KV Cache，这个KV Cache因为它是滑窗，所以它那个KV Cache的上限是被限制住的。这样的话，我们就可以把它这个KV Cache的上限它的Size当成Sliding Window Attention的一个State Size。而RNN它有RNN的那个 State Size，它有那个状态数。如果这两个东西大概在一个Level的话，我觉得就是一个公平的比较。

因为像解码（Decoding）的时候，Sliding Window和RNN，因为解码的话它基本上都是一个Memory-Bound的一个过程，所以只要它的这个State Size差不多，那它解码的效率就不会差太多了。因为Memory-Bound它主要就是看它读多少State，只要它们这个State差不多大，那它们这个解码的效率基本上就会差不多大，因为解码还是主要是Memory-Bounded。

张小珺：说到算法的演进，它最早从Transformer到MoE，到现在大家探索Linear Attention或者Sparse Attention，这种渐进式的创新，它优化的最终目标可能是什么？最终可能形成的一个算法的共识会是什么样的？

杨松琳：我觉得这些优化基本上都是体现在：给定你相同的FLOPs（浮点运算次数），你怎么去更好地利用这些FLOPs，然后取得更低的损失函数（Loss Function）。

像MoE这个技术，在前两年，可能比如2023年的时候都在传GPT-4用MoE，但也有很多地方不太敢跟。而像现在的话，MoE基本上都已经变成一个显学了，每一家都会做这种细粒度的MoE。

像MoE的话，它其实也是一种可以想象成更高效的一个FFN的替代品。它可以更好地去扩大FFN的这个参数量，同时它又保证它那个FLOPs不变。这样的话，它付出相同的FLOPs，它能在预训练里面取得的那个训练Loss就会越低。这就是一个点。我觉得MoE它可能是近几年在架构方面突破最大的一个方案。

下一个突破点可能就在Attention。因为Transformer就两个模块，一个FFN，一个Attention。现在FFN基本上已经“雕”成了这种细粒度MoE的形状。我觉得Attention大家也是可以来“雕”一下的。Why Not？

这样的话，比如在长文本下面，它付出相同的FLOPs，它可能取得的那个Loss也会更低。我觉得这两套思路都是一样的，就是减少FLOPs，然后能够让它…像FFN的话，它减少FLOPs，它就可以去用更大的参数量，更大规模的一个模型。比如你总参数量就可以堆高了，因为你这个FFN的这个算力减少了。大家都知道在大规模训练下面，FFN的那个计算是主导的。把它换成这种细粒度MoE的话，它其实是能降低很多很多这种成本（Cost）。

而Attention它Scale的就主要不是参数量，它Scale的就是那个Context Window Size。如果这个Attention的这个FLOPs在长文本下面能够把它打下来的话，那我们就是做那种长文本的生成，比如你有很多Agent让他去处理很多很多Workflow，然后喂很多很多Context给他做，这样的话它也会Benefit from这个更大的Context Window的。

张小珺：如果把模型的架构比作比如说大脑的结构，你觉得MoE和Attention它们分别代表的是大脑的什么组件？能这样去形象化地去理解吗？

杨松琳：像Attention的话，它应该就相当于Working Memory吧，就是那种工作记忆。像FFN的话就有点像海马体，是来存储过去信息的。像FFN它基本上会被看成是一个键值对（Key-Value Pair）的一个关联网络，它可以记下很多很多这种Knowledge。像这种World Knowledge都会被它记到这个FFN里面。这就是一些World Knowledge会存下来。而Attention就是比如你在一个新的场景，然后你遇到新的这种信息，你会读到新的Context，它会在这个Context Window里面就是动态地来做这个处理这些信息，那就有点很像我们人的大脑那个工作记忆（Working Memory）。

张小珺：它更偏即时性一些。

杨松琳：对。

  

# 08 国内算法创新肯定是更强的

---

  

张小珺：当现在遇到数据瓶颈的时候，是不是算法的创新变得更重要了？

杨松琳：我觉得是的。

张小珺：你需要在有限的数据里面去压缩更多的智能。

杨松琳：对，我觉得之前的话，比如你数据一直能Scale的话，谈这个Data Efficiency就是没有什么特别大的用途。因为大家闭着眼睛加这个数据就行了，让它模型继续Scale Up，然后继续加数据，所以大家都不需要去动算法了，大家就只需要买卡就行了。

现在如果有这种数据墙，还有这种算力墙的话，可能到最终还是要回到这个算法这种本质的东西上面来。

这些东西都是缺一不可的，比如像Data，像算法，像算力——三匹马车来驱动整个人工智能的发展。

我记得之前像OpenAI的CTO，她也说过就在这个节点上面，算法的研究的重要性可能会被重新抬高。

张小珺：你觉得现在的架构Transformer架构的天花板是什么？

杨松琳：它的天花板还是先把Efficiency的问题解决掉吧。

因为现在还没有解决掉Efficiency的问题，它处理一个很长的一个Context Window还是有一些局限性。大家会做很多上下文工程（Context Engineering），做一些RAG来通过一些其他的方式来解决这些问题。但如果你这个Context的问题把它解决掉，那你RAG这一套技术都不需要了，你直接把它放到Context里面做In-Context RAG就行了。

我觉得天花板就先看看能不能就是把全局注意力把它干掉吧。这是第一点，因为它确实是阻止这个Context Window继续Scale Up上去的一个主要瓶颈，所以这个瓶颈我觉得是早晚都要把它弄掉的。

第二点的话可能就是Continual Learning。像现在这种Transformer架构还是没法做Continual Learning的。之后Continual Learning让AI自己学习这种，甚至大家不都想把Pre-training这个地方变成直接从RL开始，让这个模型直接从零开始学，不给他这种Pre-train Data。像这种新的范式可能就是之后的这种探索。

张小珺：如何把Linear Attention的Transformer Scale Up？

杨松琳：我觉得Scale Up应该是没有什么特别大问题吧。

可能还有一点的话，就是说那些配套的这种Infra（基础设施）还是需要继续搭建的。像Flash Linear Attention只是提供了一些Triton的Kernel，基本上就是可以凑合用，但是它的那个效率肯定不是最优的，因为它是Triton写的。

如果有志向投入这个领域的，比如一些公司，或许可以花一些精力去优化这些Kernel。这个是对继续Scale Up上去有好处的。

然后像Infra那一边的支持，现在已经在逐渐变多了。像半年前我参加Minimax，它有一个那个圆桌讨论，当时主持人是俊贤老师。俊贤老师问我这个领域它主要的瓶颈是什么？我当时说是Infra的那个配套没有跟上。当时俊贤老师还觉得挺意外的，以为我会回答一些别的东西。

事实上就是这样子。我觉得算法层面可以，比如像近两年的这个发展就已经可以去大规模地来试了，后面的部署（Deploy）的瓶颈可能就是更多是在这种配套设施。

张小珺：现在中国的算法创新相对于硅谷来说是差不多、更强还是落后的？

杨松琳：我觉得国内算法创新肯定是更强的。主要是in terms of架构的话，那肯定是国内更强的。

这也是有一些生态地位不同。比如国内没有那么多卡，他们其实对这个Efficiency的要求是更高的，所以他们更有动力来尝试这一些更高效的一些Linear Attention的变种。而像硅谷有些公司基本上就是卡太多了，他们就懒得搞。

张小珺：反正三驾马车你总得有一辆跑得快一点。

杨松琳：对，他们有那个算力，那也能凑合跑。

张小珺：这脑子长得不怎么样无所谓，反正我先把算力堆上去。

杨松琳：对，我觉得硅谷这边感觉美国的公司会更注重优化一点。像Optimization，比如优化器（Optimizer）。国内公司也感觉在逐渐在用，比如像Kimi，它也是最早吃Muon这个优化器的螃蟹的。

给我的感觉是，美国他们对优化器的投入明显是比国内对优化器的投入要大一些的。

张小珺：Kimi Linear的论文，你觉得还有哪些是值得大家关注的？

杨松琳：前面说了就是这个线性注意力的模块。还有可能就全局注意力，它的那个用RoPE还是用NoPE的比较。

像Kimi选的是用NoPE，像千问-3 Next选到是一个Partial RoPE，它就是25%是RoPE，75%是NoPE。我觉得在这种混合注意力里面大家都在砍RoPE，但是看大家砍多少。像千问-3 Next 砍了75%，像Kimi砍了 100%。

像这种长度外推（Length Extrapolation），就感觉现在看起来的话就是RoPE在这种Hybrid架构里面可能会比较阻碍这种长度外推。这个地方其实也没有共识，就是大家也不知道是用有些还是用RoPE，有些还是用NoPE。我觉得这个地方还是没有共识的，然后有些地方还用Partial RoPE。

张小珺：千问的工作你有参与没有？

杨松琳：就千问-3 Next的话，我就基本上类似，就是他们要是碰到什么问题我就可以帮忙答一下。就是不参与他们训模型什么的，如果他们有一些学术上的讨论的话，我是会跟他们讨论的。我跟千问-3 Next训练的那几个同学都还挺熟的。

张小珺：Minimax参与没有？

杨松琳：我参与了，他们应该不会用这个方案，我会觉得这个方案在开倒车。

  

# 09 考古

---

  

张小珺：我觉得你用词很好玩，你说把这个架构“玩一下”或者“雕一下”。这是一种研究员之间的文化吗？

杨松琳：“雕”这个字好像还挺常见的，就是有种“雕花”的自嘲的那种说法吧。

张小珺：现在没办法，算力不够，数据也有限了，所以只能雕。

杨松琳：对，但我觉得“雕”架构还是挺有用的。

像DeepSeek MoE那个雕出来之后，大家都已经成为一个共识了。就很多地方会用DeepSeek的那个MoE方案。如果在他之前，他在做那个，可能大家也会说可能在“雕MoE”。感觉“雕”已经变成一个常见的形容词了。

我觉得它不是一个贬义词了，它是一个就是把一个模块把它打磨到更好。

张小珺：如果数据的Scale非常突出的话，其实没有必要雕，怼数据就好了。当数据还很少的时候，比如说机器人领域，现在就是没什么数据，只要加数据就能够显著的效果提升，这时候没有必要去做模型算法的创新。

杨松琳：对，这是一点。所以Robotics最应该做的还是先把数据这个问题搞定吧。数据搞定之后再回来看这种Efficiency的问题也不迟。

张小珺：你是怎么进入AI这个行业的？

杨松琳：AI行业就是本科的时候对Machine Learning、Deep Learning挺有兴趣的，当时Master在上科大念NLP，那个时候就已经进入AI了。22年、23年就是ChatGPT这一波，Large Language Model风靡开来。做NLP的人就基本上都来做Large Language Model。

现在做AI更有意思一点。因为之前大家还是在分Task做，现在就是比较Unify了，会比较Focus on更加通用的问题来了。不需要去操心某些特定Task，因为你只要训一个很好的基础模型，你对不同的Task都可以用。你无非是Post-train的时候要注意的地方不一样。

现在感觉自己做的东西能看到更多影响力，还是挺开心的。

张小珺：你过程中有遇到过什么样的挫折没有？

杨松琳：我感觉我读Ph.D.好像这些工作都还挺顺的，这些工作都还挺连贯的。

还是因为可能是读Ph.D.之前，花了半年的时间来调研这些东西。可能对这些这个领域的理解会深很多，就深耕这个领域来做。其实问题也不是很多，因为对这个领域非常熟，然后碰到什么问题大概也知道怎么去解决。

张小珺：读Ph.D.前花半年去调研？

杨松琳：申请完之后有半年可以自由的时光，当时就基本上就是在调研这种架构的论文（Paper）。当时读了很多比较老的Paper，就比如说像Delta Net，它最早是2021年，就是那个LSTM之父的Paper。

当时我就对这个工作有印象，后来，那年年底就做完那个Gated Linear Attention，发现这个领域的话，大家会对那个In-Context Recall，就是从那个前面的文章里面去做一个Retrieval，这个Task会感兴趣。

这个就让我一下子联想到了那个2021年那一篇工作了。因为之前的这个整个领域他把握得非常的通畅，所以我知道就是如果领域大家其他人关心这个问题的话，我应该从什么角度去切入？我也知道它前面工作有什么问题，比如2021年Delta Net的话，它是没有硬件效率（Hardware Efficiency）的一个保证。

我后面就觉得交了这个工作之后做Delta Net的话，我就知道Delta Net是一个很好的模型，它的缺点就是现在大家还不能大规模用起来，如果我能开发出一个算法，能把它Scale Up，那就是一个非常有意义的工作。我大概就是这一套逻辑链，可能也是运气好吧。

后面，可能就是像Gated Attention是沿着这个工作做，因为当时发现它还是在很多Task上面是打不过Mamba-2的。我当时就觉得打不过就加入嘛，我就把Mamba-2的那个Gating把它拿过来，把Delta Rule再加回来。这样子就把它A加B，变成一个Gated Delta Net。

我感觉我做的东西就是会看这个领域它需要什么样的工作，然后哪些做什么样的东西会带来更多的这种领域的影响力，还有业界的影响力。

如果当你很清楚你要做什么的时候，你其实是不会遇到什么挫折的。技术的那种Challenge，我觉得都是有办法把它搞定的。更大的Challenge就是你不知道你要做什么东西，你不知道做什么东西是有用的，我觉得这个才是最大的Challenge。

张小珺：你核心是从历史中学习了很多。

杨松琳：我还是挺喜欢看最早的那些Paper，我觉得那些Paper写得都挺好的，我管这个叫做“考古”。因为我就喜欢考那些古代的Paper。

古代的话可能2021年也算古代，因为可能现在一年前的Paper叫老Paper，那五年前的Paper肯定叫做古代的Paper了。

张小珺：那半年你读的最老的Paper到什么时候？

杨松琳：可能就是读到，比如说二零一几年的文章吧。

不同的人有不同的Research Philosophy，我觉得就一定要把这个领域里面值得看的文章全部都看一遍。

张小珺：为什么在AI的众多领域分支里面你喜欢的是架构？

杨松琳：因为我比较喜欢做算法。然后就想做一些比较通用的，整体都是对这种LLM有用的一些Work。结合一下自己兴趣，正如最开始说的就是像Hazy Research，他们有很多博客，主要还是自己喜欢做算法，就发现这个领域很适合我来做。

张小珺：你提到你读博士前半年做了很多算法的考古，能不能给大家讲讲，就是算法是怎么一步步演进到今天的这段算法历史？

杨松琳：那我从 Transformer 开始讲。

Transformer 的话，它感觉可能就三个主要模块，一个是注意力机制，另外一个是位置编码（Positional Encoding），最后就是FFN。

最开始那几年我感觉可能架构Research非常多，有一些架构的改进也确实被用到了今天。比如说像相对位置编码（Relative Positional Encoding），比如说像RoPE。它最开始Transformer的话它是绝对位置编码（Absolute Positional Encoding），像今天基本上都改成了这种相对位置编码了。

像MoE的话，可能也是从2021年左右就开始发展了，中间有段时间大家可能就不怎么信MoE，后面又发现像DeepSeek把MoE做通了，大家又回来重新做MoE。现在MoE应该就是大家都会用的东西。

像Attention的话，Attention的这种变种可能就更多了。像前面也说到，2020年前后，可能Attention的变种就非常非常多。也主要就是两种变种：第一种就是线性注意力，第二种就是稀疏注意力。

他们线性注意力就会搞很多那种Kernel Method来近似Softmax Attention。在今天来看，我觉得这是一个非常错误的方向，我觉得就不应该去用Kernel Method去估计这些Softmax Attention。

有一些好工作的话可能就会因为没有Follow Up被埋没在文献海里面。比如像Delta Net这个工作我前面也说他是2021年就有了，可能后面几年就根本没有人Take It Seriously，没有什么Follow Up Work。

从时间演进来看，像这种细粒度的遗忘（细粒度Decay），很多年前就已经有了。至少EMNLP 2022年就有相关工作，而最早我可以考古到2016 年。不过后来，比如RetNet 2023年，它反而用了一个更粗粒度的遗忘速率（粗粒度Decay）。所以可能是之前的技术没有很好地传承下来。

我又比较喜欢把所有之前所有的技术全部重新审视一遍，挑选一些我觉得最合理（Make Sense）的技术来使用。比如Delta Rule这个技术，就可能重新发挥作用。如果没有我来跟进，这套技术路线可能就会淹没在文献海里。

像Sparse Attention的话，他们最早可能就做一些Static的Sparse Attention，好像后面就逐渐收敛到用Sliding Window 了。可能近几年它会有一些不一样的东西出来。这就是早几年比较少，但是最近又比较多的。

比如说像动态稀疏（Dynamic Sparsity），像Kimi的MoE、DeepSeek的Sparse Attention都属于动态稀疏。总的来说我感觉整体还是算法在不断演进，可能它整个发展就是需要有一些技术，可能需要Rethink几次，多多少少感觉这个发展还是会有点螺旋上升的味道在里面吧。

张小珺：历史中已经有很多工具，但是今天我们需要拿哪些工具来运用推动今天的算法演进，是很关键的。

杨松琳：对，很多历史的算法很先进的，但当时的同行没有意识到这个工作的价值。有可能那个工作就被埋没了，也有可能就是那个工作的配套，比如说那些代码、开源代码做得太烂了，其他人想Follow也没法Follow。

总的来说，如果今天做工作的话，可能就是比如像我就会把这种代码做得好，让大家好用，所以这一套技术肯定能把它让它流传下去。

张小珺：Delta Rule是什么给你带来的灵感？

杨松琳：就是2021年那个工作，是他们提出来的。我就想了一个并行算法，就挺类似于Flash Attention之于Softmax Attention。其实就是一个算法能够让它硬件高效地来实现的。

如果没有Flash Attention，那Softmax Attention也走不到今天。没有那个并行算法的话，那Delta Net肯定也不能走到今天的，大概是一个这么样子的关系。

我做Research可能就是比较喜欢从实际上的硬件亲和力来研究，因为我看一个算法有没有潜力，我会来分析这个算法它的并行潜力有多大，然后它的Scalability会有多大？我会在历史的文献海里面找出一些Machine Learning上面Make Sense，同时我又能想办法把它并行的一些算法来玩。

这是我的做Research的思路，总的来说还是就是Machine Learning上面Make Sense，然后它这个算法又可以有并行的算法，这样的算法才能在这个年代被用到。因为你肯定需要有一些能够Scalable的算法。

如果一个算法它就更Make Sense，比如说像Delta Rule这个算法，我觉得这个算法就非常的Make Sense，同时又能Scalability比较好的话，就完全有可能在今天这个时代上面，带来一些不一样的一些架构吧。

就比如像千问-3 Next和Kimi Linear就已经让我们带来了一些新气象。就这个新架构领域。

张小珺：我前几天做了一个论文播客提到，Transformer是这一代硬件的天选架构。

杨松琳：Transformer肯定是天选。当时设计Transformer就是为了让它硬件亲和。像FFN那肯定不用说，都是大矩阵乘法，自然很快。Attention也是之前Attention的演进。以前大家用L-LSTM这种RNN模块，不能并行，硬件加速很难。而Attention虽然理论复杂度是平方级，但它可以通过矩阵乘法直接计算输出，硬件亲和比RNN好很多。

所以大家宁愿用理论复杂度更高的Transformer，也不会用理论复杂度更低的LSTM，因为硬件亲和度完全不一样。

我觉得算法的发展就是要找到这些硬件亲和度高、算法本身也优秀的方案。Transformer不仅硬件亲和，还解决了长程依赖（Long-term Dependency）问题，所以流行开来。

今天Linear Attention又重新登上舞台，也离不开这一系列发展。比如把它分成Chunk的并行算法，这些设计能从Machine Learning Performance（机器学习性能）的角度更合理（Make Sense），这是推动发展的原动力。

我主张做一些非常有原则（Principle）的设计，从Machine Learning角度来说，它要数学上成立（Mathematically Grounded）。比如Delta Rule从数学上就Make Sense，同时还要Hardware Aligned（与硬件契合）。做模型一定要结合当前硬件，否则就不现实。有人可能会说，我算法够好，硬件公司会帮我优化，但这不可能。算法本身必须先满足通用原则。

像Memory Hierarchy（存储层级）、矩阵乘法优化这些硬件原则是通用的，不论什么类型的硬件基本都遵循。设计算法时不必专门针对H100优化，但至少要满足这些通用硬件原则，否则算法在当前可扩展性（Scalability）和场景下基本没实际价值，只能自娱自乐。

张小珺：Kimi Linear对于硬件亲和有做什么样的优化没有？

杨松琳：Kimi Linear我觉得它的算法还是硬件亲和的，然后Kernel的话，它现在应该还是张宇写的那个Triton的算法。就凑合用吧。

我相信大家都没有那么多…就是算子优化，它是一个非常耗时的一个工种。它就非常的需要时间。要慢慢磨，就要老师傅核优化慢慢来打磨。

张小珺：从硬件亲和的角度，你觉得下一代的算法会怎么演进？

杨松琳：现在我觉得这硬件演进的话，它跟Transformer是有一点协同演进了，就是硬件会变成Transformer更喜欢的模样。

所以对于一些Alternative来说，是有一些不好的因素在里面的。因为现在要架构这样硬件，大家会发现它就是为了去优化矩阵乘，然后让它矩阵乘越快越好。因为Transformer里面有大量的矩阵乘，它就想硬件就想搞一些快速的矩阵乘的东西，比如说像Tensor Core，然后像这种TMA这种东西。然后像最近的Blackwell上面它有一些专门针对这种矩阵乘，它有一些单独的那种内存上面单独的那种Memory。这都是来优化矩阵乘的。

可能大家会看到Flash Attention会越来越快，FA-4它会在Blackwell上面会越来越快。我觉得既然这个硬件是这么Evolve 的，那从设计算法的角度来看，你就必须要设计一些能有矩阵乘法的算法，要不然你这个硬件效率肯定是跟不上的。

像Linear Attention它创个算法有个好处，就是它基本上都是一些矩阵乘，当然它还会有一些其他的Overhead。那这个的话那可能就是得克服一下。它可能比如说在Training的时候可能还是不如Flash Attention-4这种在Blackwell上面高效，但其实也无所谓。

就很多地方也不Care训练效率，它只Care那种Inference效率。所以我觉得只要训练的时候就是能以Reasonable的速度来训，然后Reasonable的速度来prefill，然后解码快的话这种架构其实也是有市场的。

另外，就比如说像细粒度MoE，然后Sparse Attention这种降FLOPs，然后又能用矩阵乘法。他们都是属于这种类型，他们肯定还是要用矩阵乘法的，就是想办法把FLOPs打下去，通过一些算法的这种创新来把FLOPs打下去。同时保证，这里面有大量的矩阵乘。就一旦一个算法里面基本上都矩阵乘，那基本上这个算法也Hardware也挺相对而言还是挺好优化的。

因为我认为当前的硬件发展方向，正是朝着加速矩阵乘法的方向不断迈进。甚至像FlashAttention 2 (FA2) 的优化，由于矩阵乘法速度极快，反而导致其中的Softmax归一化以及指数运算（Exponential）模块成为了新的性能瓶颈。

因此，在FlashAttention中，他们会采用一些近似（Approximate）方法来处理指数运算。这听起来也挺有意思——当矩阵乘法速度太快时，我们反而需要尽量利用它快速的特性，并以此为基础来迭代和优化我们的算法。

我认为像DeepSeek的稀疏注意力（Sparse Attention）就很好地利用了这一特性。DeepSeek是一家非常注重硬件和算法协同设计的公司。以DeepSeek的Sparse Tensor优化为例，它会使用一个被称为Indexer的模块，这个Indexer使用FP8 (8位浮点数) 来计算Tensor score，因为它不需要Softmax归一化，只需要计算Logits，然后通过Top-K选择来挑选出重要的score。

首先，它使用了FP8，其次，它成功地去除了昂贵的指数运算操作。这样一来，整个计算就基本等同于一系列矩阵乘法。因此，它的Indexer模块计算速度会非常快，这就有可能被整合到DeepSeek的下一代架构中。虽然我们不知道他们的下一代架构具体是什么，但这些优化特性无疑使其成为下一代架构的潜在候选（Candidate）。

张小珺：相对来说，在硬件亲和性方面，DeepSeek和Kimi哪一个做得更好？听起来像是DeepSeek。

杨松琳：毫无疑问是DeepSeek。

我认为Kimi肯定也在进行硬件扩展（Scale Up）方面的投入，但可能没有DeepSeek这样极致的追求。DeepSeek非常追求这种协同设计，例如：某个算法能否在FP8上高效运行等等。我推测，在他们的算法迭代过程中，基础设施（Infra）团队的话语权会比较高。

我认为这因公司而异。有些公司的Infra团队话语权更高，而另一些公司则是算法团队的话语权更高。通常感觉算法团队经常会提出一些让Infra团队头疼或难以优化的东西。

张小珺：最后，你对想要进入注意力机制、架构设计或算法等领域的年轻研究者有什么建议？他们应该从哪些方面着手？

杨松琳：当前阶段，最好的方式是找一家公司去实习。

我认为架构研究必须要有算力作为支撑，没有算力就无法进行架构层面的工作。

所以，我建议先找一个实验室（Lab）或公司去实习。