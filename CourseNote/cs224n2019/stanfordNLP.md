## 资料

课程主页： https://web.stanford.edu/class/cs224n /

中文笔记： http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html

视频：https://www.bilibili.com/video/av30326868/?spm_id_from=333

http://www.mooc.ai/course/494

学习笔记：http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html

实验环境推荐使用Linux或者Mac系统，以下环境搭建方法皆适用:

· Docker环境配置： https://github.com/ufoym/deepo
· 本地环境配置： https://github.com/learning511/cs224n-learning-camp/blob/master/environment.md

训练营： https://github.com/learning511/cs224n-learning-camp

清华大学NLP实验室总结的机器阅读论文和数据：

https://github.com/thunlp/RCPapers?utm_source=wechat_session&utm_medium=social&utm_oi=804719261191909376  

**重要的一些资源：**

深度学习斯坦福教程： 

<http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B>

廖雪峰python3教程： 

<https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000>

github教程： 

<https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000>

莫烦机器学习教程： <http://morvanzhou.github.io/tutorials> /

深度学习经典论文： <https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap>

斯坦福cs229代码(机器学习算法python徒手实现)： <https://github.com/nsoojin/coursera-ml-py>

博客： <https://blog.csdn.net/dukuku5038/article/details/82253966>

==数学工具==

**斯坦福资料**：

- 线性代数（链接地址： <http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf> ）
- 概率论（链接地址： <http://101.96.10.44/web.stanford.edu/class/cs224n/readings/cs229-prob.pdf> ）
- 凸函数优化（<http://101.96.10.43/web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf> ）
- 随机梯度下降算法（链接地址： <http://cs231n.github.io/optimization-1> /）

**中文资料**：

- 机器学习中的数学基本知识（链接地址： <https://www.cnblogs.com/steven-yang/p/6348112.html> ）
- 统计学习方法（链接地址： <http://vdisk.weibo.com/s/vfFpMc1YgPOr> ）
- 大学数学课本（从故纸堆里翻出来^_^）

==编程工具==

**斯坦福资料**：

- Python复习（链接地址： <http://web.stanford.edu/class/cs224n/lectures/python-review.pdf> ）
- TensorFlow教程（链接地址： <https://github.com/open-source-for-science/TensorFlow-Course#why-use-tensorflow> ）

**中文资料**：

- 廖雪峰python3教程（链接地址： 
- <https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000> ）
- 莫烦TensorFlow教程（链接地址： <https://morvanzhou.github.io/tutorials/machine-learning/tensorflow> /）

## paper

1.A Simple but Tough-to-beat Baseline for Sentence Embeddings

Sanjeev Arora, Yingyu Liang, Tengyu Ma
Princeton University
In submission to ICLR 2017

2.Linear Algebraic Structure of Word Senses, with Applications to Polysemy

Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, Andrej Risteski

3.Distributed Representations of Words and Phrases and their ComposiRonality (Mikolov et al. 2013)

4..GloVe: Global Vectors for Word Representation (Pennington et al. (2014)

Word Vector Analogies: SyntacRc and Semantic examples from
http://code.google.com/p/word2vec/source/browse/trunk/questionswords.txt

Word vector distances and their correlation with human judgments
Example dataset: WordSim353
http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

5.Improving Word Representations Via Global Context And Multiple Word Prototypes (Huang et al.
2012)

1. Gather fixed size context windows of all occurrences of the word
    (for instance, 5 before and 5 after)
2. Each context is represented by a weighted average of the context
    words’ vectors (using idf-weighting)
3. Apply spherical k-means to cluster these context representations.
4. Finally, each word occurrence is re-labeled to its associated cluster
    and is used to train the word representation for that cluster.

6.Bag of Tricks for Efficient Text Classification
Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
Facebook AI Research

● fastText is often on par with deep learning classifiers
● fastText takes seconds, instead of days
● Can learn vector representations of words in different languages (with performance better than word2vec!)

## 概述

NLP levels

![1540300272862](img/nlplevels.png)

作为输入一共有两个来源，语音与文本。所以第一级是语音识别和OCR或分词（事实上，跳过分词虽然理所当然地不能做句法分析，但字符级也可以直接做不少应用）。接下来是形态学，援引《统计自然语言处理》中的定义：

> 形态学（morphology）：形态学（又称“词汇形态学”或“词法”）是语言学的一个分支，研究词的内部结构，包括屈折变化和构词法两个部分。由于词具有语音特征、句法特征和语义特征，形态学处于音位学、句法学和语义学的结合部位，所以形态学是每个语言学家都要关注的一门学科［Matthews,2000］。

下面的是句法分析和语义分析，最后面的在中文中似乎翻译做“对话分析”，需要根据上文语境理解下文。

**自然语言处理应用**

一个小子集，从简单到复杂有：

- 拼写检查、关键词检索……
- 文本挖掘（产品价格、日期、时间、地点、人名、公司名）
- 文本分类
- 机器翻译
- 客服系统
- 复杂对话系统

在工业界从搜索到广告投放、自动\辅助翻译、情感舆情分析、语音识别、聊天机器人\管家等等五花八门。

**深度学习是表示学习的一部分，用来学习原始输入的多层特征表示**

## Iteration Based Methods - 词向量表示Word2vec

**Two algorithms**

Skip-grams (SG)
Predict context words given target (position independent)

Continuous Bag of Words (CBOW)
Predict target word from bag-of-words context

**Two (moderately efficient) training methods**

Hierarchical softmax

Negative sampling

**hierarchical softmax tends to be better for infrequent words, while negative sampling**
**works better for frequent words and lower dimensional vectors.**

主要思路：

遍历整个语料库中的每个词

预测每个词的上下文：$p(o|c)=\frac{exp(u_o^Tv_c)}{\sum_{w=1}^v exp(u_w^Tv_c)}$

然后在每个窗口中计算梯度做SGD

**word2vec存在的问题**

1. word2vec在计算梯度时会遇到梯度稀疏的问题

![1540898521791](img/1540898521791.png)

可以考虑每次只更新那些真实出现的单词的向量。

解决方案有两种：一种是需要有一个稀疏矩阵更新操作，每次只更新向量矩阵的某些列。第二种方法是为单词建立到词向量的hash映射。

If you have millions of word vectors and do distributed computing, it is important to not have to send gigantic updates around!

2. word2vec的归一化因子部分计算复杂度太高

解决方法：使用负采样（negative sampling）实现skip-gram。具体做法是，对每个正例（中央词语及上下文中的一个词语）采样几个负例（中央词语和其他随机词语），训练binary logistic regression（也就是二分类器）。

目标函数：

$J_t(θ)=logσ(u^T_ov_c)+∑_{i=1}^kE_{j∼P(w)}[logσ(−u^T_jv_c)]$

这里t是某个窗口，k是采样个数，P(w)是一个unigram分布

这个目标函数就是要最大化中央词与上下文的相关概率，最小化与其他词语的概率。

$P(w)=U(w)^{3/4}/Z$

这样使得不那么常见的单词被采样的次数更多。



word2vec将窗口视作训练单位，每个窗口或者几个窗口都要进行一次参数更新。要知道，很多词串出现的频次是很高的。能不能遍历一遍语料，迅速得到结果呢？

早在word2vec之前，就已经出现了很多得到词向量的方法，这些方法是基于统计共现矩阵的方法。如果在窗口级别上统计词性和语义共现，可以得到相似的词。如果在文档级别上统计，则会得到相似的文档（潜在语义分析LSA）。

**基于窗口的共现矩阵**

在某个窗口范围内，两个词共同出现的次数组成的矩阵。

根据这个矩阵，的确可以得到简单的共现向量。但是它存在非常多的局限性：

- 当出现新词的时候，以前的旧向量连维度都得改变
- 高纬度（词表大小）
- 高稀疏性

**解决办法：低维向量**

用25到1000的低维稠密向量来储存重要信息。如何降维呢？

方法：SVD，但是类似于the,he,has这样的词频次太高

改进

- 限制高频词的频次，或者干脆停用词
- 根据与中央词的距离衰减词频权重
- 用皮尔逊相关系数代替词频

SVD的问题

- 计算复杂度高：对n×m的矩阵是$O(mn^2)$
- 不方便处理新词或新文档
- 与其他DL模型训练套路不同

**Count based vs direct prediction**

这些基于计数的方法在中小规模语料训练很快，有效地利用了统计信息。但用途受限于捕捉词语相似度，也无法拓展到大规模语料。

而NNLM, HLBL, RNN, Skip-gram/CBOW这类进行预测的模型必须遍历所有的窗口训练，也无法有效利用单词的全局统计信息。但它们显著地提高了上级NLP任务，其捕捉的不仅限于词语相似度。

![1540906029954](img/1540906029954.png)

综合两者优势：GloVe

##高级词向量表示Global Vectors for Word Representation (GloVe)

这种模型的目标函数是：

$J =\frac12\sum_{i,j=1}^W f(P_{ij})(\mu_i^Tv_j - logP_{ij})^2$

这里的Pij是两个词共现的频次，f是一个max函数：

![1540906170932](img/1540906170932.png)

优点是训练快，可以拓展到大规模语料，也适用于小规模语料和小向量。

这里面有两个向量u和v，它们都捕捉了共现信息，怎么处理呢？试验证明，最佳方案是简单地加起来：

$X_{final}=U+V$

相对于word2vec只关注窗口内的共现，GloVe这个命名也说明这是全局的（我觉得word2vec在全部语料上取窗口，也不是那么地local，特别是负采样）。

**评测方法**

有两种方法：Intrinsic（内部） vs extrinsic（外部）

Intrinsic：专门设计单独的试验，由人工标注词语或句子相似度，与模型结果对比。好处是计算速度快，但不知道对实际应用有无帮助。有人花了几年时间提高了在某个数据集上的分数，当将其词向量用于真实任务时并没有多少提高效果。

Extrinsic：通过对外部实际应用的效果提升来体现。耗时较长，不能排除是否是新的词向量与旧系统的某种契合度产生。需要至少两个subsystems同时证明。这类评测中，往往会用pre-train的向量在外部任务的语料上retrain。

**Intrinsic word vector evaluation**

也就是词向量类推，或说“A对于B来讲就相当于C对于哪个词？”。这可以通过余弦夹角得到：

a:b  ::  c:?                $d=argmax_i\frac{(x_b-x_a+x_c)^Tx_i}{||x_b-x_a+x_c||}$

这种方法可视化出来，会发现这些类推的向量都是近似平行的

word2vec还可以做语法上的类比，比如slow——slower——slowest这种比较级形式

实验中，GloVe的效果显著地更好。另外，高纬度并不一定好。而数据量越多越好。

**调参**

窗口是否对称（还是只考虑前面的单词），向量维度，窗口大小

大约300维，窗口大小8的对称窗口效果挺好的，考虑到成本。

对GloVe来讲，迭代次数越多越小，效果很稳定

适合word vector的任务，比如单词分类。有些不太适合的任务，比如情感分析。

消歧，中心思想是通过对上下文的聚类分门别类地重新训练。

we have looked at two main classes of methods to find word embeddings. 

The first set are count-based and rely on matrix factorization (e.g. LSA, HAL). While these methods effectively leverage global statistical information, they are primarily used to capture word similarities and do poorly on tasks such as word analogy, indicating a sub-optimal vector space structure. 

The other set of methods are shallow window-based (e.g. the skip-gram and the CBOW models), which learn word embeddings by making predictions in local context windows. These models demonstrate the capacity to capture
complex linguistic patterns beyond word similarity, but fail to make use of the global co-occurrence statistics.

Glove：Using global statistics to predict the probability of word j appearing in the context of word i with a least squares objective

1.Co-occurrence Matrix

$X_{ij}$ ：the number of times word j occur in the context of word i

2.Least Squares Objective

$J = \sum_{i=1}^W\sum_{j=1}^W f(X_{ij})(\mu_j^Tv_i - logX_{ij})^2$

In conclusion, the GloVe model efficiently leverages global statistical information by training only on the nonzero elements in a wordword co-occurrence matrix, and produces a vector space with meaningful sub-structure. It consistently outperforms word2vec on the word analogy task, given the same corpus, vocabulary, window size, and training time. It achieves better results faster, and also obtains the best results irrespective of speed.

## Evaluation of Word Vectors

### Intrinsic Evaluation

Intrinsic evaluation of word vectors is the evaluation of a set of word vectors generated by an embedding technique (such as Word2Vec or GloVe) on specific intermediate subtasks (such as analogy completion).

Intrinsic evaluation:
• Evaluation on a specific, intermediate task
• Fast to compute performance
• Helps understand subsystem
• Needs positive correlation with real task to determine usefulness

A popular choice for intrinsic evaluation of word vectors is its performance in completing word vector analogies.

### Extrinsic Evaluation

Extrinsic evaluation of word vectors is the evaluation of a set of word vectors generated by an embedding technique on the real task at hand.

Extrinsic evaluation:
• Is the evaluation on a real task
• Can be slow to compute performance
• Unclear if subsystem is the problem, other subsystems, or internal interactions
• If replacing subsystem improves performance, the change is likely good

Most NLP extrinsic tasks can be formulated as classification tasks.



## About Project

project types:

1. Apply existing neural network model to a new task
2. Implement a complex neural architecture
3. Come up with a new neural network model
4. Theory of deep learning, e.g. optimization

**Apply Existing NNets to Tasks**

1. Define Task:
    • Example: Summarization
2. Define Dataset
   1. Search for academic datasets
       • They already have baselines
       • E.g.: Document Understanding Conference (DUC)
   2. Define your own (harder, need more new baselines)
       • If you’re a graduate student: connect to your research
       • Summarization, Wikipedia: Intro paragraph and rest of large article
       • Be creative: Twitter, Blogs, News

3. Define your metric
    • Search online for well established metrics on this task
    • Summarization: Rouge (Recall-Oriented Understudy for
    Gisting Evaluation) which defines n-gram overlap to human
    summaries
4. Split your dataset!
    • Train/Dev/Test
    • Academic dataset often come pre-split
    • Don’t look at the test split until ~1 week before deadline!
    (or at most once a week)

5. Establish a baseline
    • Implement the simplest model (often logistic regression on
    unigrams and bigrams) first
    • Compute metrics on train AND dev
    • Analyze errors
    • If metrics are amazing and no errors:
    done, problem was too easy, restart :)
6. Implement existing neural net model
    • Compute metric on train and dev
    • Analyze output and errors
    • Minimum bar for this class

7. Always be close to your data!
    • Visualize the dataset
    • Collect summary statistics
    • Look at errors
    • Analyze how different hyperparameters affect performance
8. Try out different model variants
    • Soon you will have more options
    • Word vector averaging model (neural bag of words)
    • Fixed window neural model
    • Recurrent neural network
    • Recursive neural network
    • Convolutional neural network

**A New Model -- Advanced Option**

• Do all other steps first (Start early!)
• Gain intuition of why existing models are flawed
• Talk to researcher/mentor, come to project office hours a lot
• Implement new models and iterate quickly over ideas
• Set up efficient experimental framework
• Build simpler new models first
• Example Summarization:
• Average word vectors per paragraph, then greedy search
• Implement language model (introduced later)
• Stretch goal: Generate summary with seq2seq!

**Project Ideas**
• Summarization
• NER, like PSet 2 but with larger data
Natural Language Processing (almost) from Scratch, Ronan Collobert, Jason Weston, Leon Bottou, Michael
Karlen, Koray Kavukcuoglu, Pavel Kuksa, http://arxiv.org/abs/1103.0398
• Simple question answering, A Neural Network for Factoid Question Answering over
Paragraphs, Mohit Iyyer, Jordan Boyd-Graber, Leonardo Claudino, Richard Socher and Hal Daumé III (EMNLP
2014)
• Image to text mapping or generation,
Grounded Compositional Semantics for Finding and Describing Images with Sentences, Richard Socher, Andrej
Karpathy, Quoc V. Le, Christopher D. Manning, Andrew Y. Ng. (TACL 2014)
or
Deep Visual-Semantic Alignments for Generating Image Descriptions, Andrej Karpathy, Li Fei-Fei
• Entity level sentiment
• Use DL to solve an NLP challenge on kaggle,
Develop a scoring algorithm for student-written short-answer responses, https://www.kaggle.com/c/asap-sas

**Another example project: Sentiment**
• Sentiment on movie reviews: http://nlp.stanford.edu/sentiment/
• Lots of deep learning baselines and methods have been tried

**And here are some NLP datasets:**

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- Sequence Tagging: [Named Entity Recognition](https://www.clips.uantwerpen.be/conll2003/ner/) and [Chunking](https://www.clips.uantwerpen.be/conll2000/chunking/)
- [Dependency Parsing](https://github.com/UniversalDependencies/UD_English)
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- [Sentence-Level Sentiment Analysis](https://nlp.stanford.edu/sentiment/treebank.html) and [Document-Level Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Textual Entailment](https://nlp.stanford.edu/projects/snli/)
- [Machine Translation (Ambitious)](https://wit3.fbk.eu/mt.php?release=2016-01)
- [Yelp Reviews](https://www.yelp.com/dataset/challenge)
- [WikiText Language Modeling](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
- [Fake News Challenge](https://github.com/FakeNewsChallenge/fnc-1)
- [Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

https://web.stanford.edu/class/cs224n/project.html

## Word Window分类与神经网络



## Dependency Parsing

### Dependency Grammar and Dependency Structure

Parse trees in NLP, analogous to those in compilers, are used to analyze the syntactic structure of sentences.

There are two main types of structures used - constituency structures and dependency structures.

Constituency Grammar uses phrase structure grammar to organize words into nested constituents.

Dependency structure of sentences shows which words depend on (modify or are arguments of) which other words.

![1529482908860](img/dependencytree)

Figure 1: Dependency tree for the sentence "Bills on ports and immigration were submitted by Senator Brownback,
Republican of Kansas"

**1.1 Dependency Parsing**

Dependency parsing is the task of analyzing the syntactic dependency structure of a given input sentence S. The output of a dependency parser is a dependency tree where the words of the input sentence are connected by typed dependency relations.

there are two subproblems in dependency parsing：

1. Learning: Given a training set D of sentences annotated with dependency graphs, induce a parsing model M that can be used to parse new sentences.
2. Parsing: Given a parsing model M and a sentence S, derive the optimal dependency graph D for S according to M.

**1.2 Transition-Based Dependency Parsing**

Transition-based dependency parsing relies on a state machine which defines the possible transitions to create the mapping from the input sentence to the dependency tree.

**1.3 Greedy Deterministic Transition-Based Parsing**

**1.4 Neural Dependency Parsing**

