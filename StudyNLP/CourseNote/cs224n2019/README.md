

## cs224nSum

winter-2019

课程资料：

- [Course page](https://web.stanford.edu/class/cs224n)
- [Video page](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
- Video page (Chinese): 
  - [可选字幕版](https://www.bilibili.com/video/av61620135)
  - [纯中文字幕版](https://www.bilibili.com/video/av46216519)

学习笔记参考：

[CS224n-2019 学习笔记](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)

[斯坦福CS224N深度学习自然语言处理2019冬学习笔记目录](https://zhuanlan.zhihu.com/p/59011576)

参考书：

- Dan Jurafsky and James H. Martin. [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
- Jacob Eisenstein. [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Yoav Goldberg. [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. [Deep Learning](http://www.deeplearningbook.org/)

神经网络相关的基础:

- Michael A. Nielsen. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- Eugene Charniak. [Introduction to Deep Learning](https://mitpress.mit.edu/books/introduction-deep-learning)



### Lecture 01: Introduction and Word Vectors

1. The course (10 mins)
2. Human language and word meaning (15 mins)
3. Word2vec introduction (15 mins)
4. Word2vec objective function gradients (25 mins)
5. Optimization basics (5 mins)
6. Looking at word vectors (10 mins or less)

**课件**

- [x] [cs224n-2019-lecture01-wordvecs1](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture01-wordvecs1.pdf)
  - WordNet, 一个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms** 的列表的辞典
  - 在传统的自然语言处理中，我们把词语看作离散的符号，单词通过one-hot向量表示
  - 在Distributional semantics中，一个单词的意思是由经常出现在该单词附近的词(上下文)给出的，单词通过一个向量表示，称为word embeddings或者word representations，它们是分布式表示(distributed representation)
  - Word2vec的思想
- [x] [cs224n-2019-notes01-wordvecs1](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
  - Natural Language Processing. 
  - Word Vectors. 
  - Singular Value Decomposition(SVD). (对共现计数矩阵进行SVD分解，得到词向量)
  - Word2Vec.
  - Skip-gram. (根据中心词预测上下文)
  - Continuous Bag of Words(CBOW). (根据上下文预测中心词)
  - Negative Sampling. 
  - Hierarchical Softmax. 

**Suggested Readings**

- [x] [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) (该博客分为2个部分，skipgram思想，以及改进训练方法：下采样和负采样)
- [x] [理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078)(上述文章的翻译)
- [x] [Applying word2vec to Recommenders and Advertising](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/) (word2vec用于推荐和广告)
- [x] [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)(没太看懂，之后再看一遍)
- [x] [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

参考阅读

- [x] [[NLP] 秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)(推荐了一些很好的资料)
- [ ] word2vec Parameter Learning Explained
- [ ] 基于神经网络的词和文档语义向量表示方法研究
- [ ] word2vec中的数学原理详解
- [x] 网易有道word2vec(词向量相关模型，word2vec部分代码解析与tricks)

**Assignment 1：Exploring Word Vectors**

[[code](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a1.zip)] [[preview](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a1_preview/exploring_word_vectors.html)]

- [x] Count-Based Word Vectors(共现矩阵的搭建, SVD降维, 可视化展示)

- [x] Prediction-Based Word Vectors(Word2Vec, 与SVD的对比, 使用gensim, 同义词,反义词,类比,Bias)

**笔记整理**

- [ ] word2vec的思想、算法步骤分解、代码



### Lecture 02: Word Vectors 2 and Word Senses

1. Finish looking at word vectors and word2vec (12 mins)
2. Optimization basics (8 mins)
3. Can we capture this essence more effectively by counting? (15m)
4. The GloVe model of word vectors (10 min)
5. Evaluating word vectors (15 mins)
6. Word senses (5 mins)

**课件**

- [x] Gensim word vector visualization[[code](https://web.stanford.edu/class/cs224n/materials/Gensim.zip)] [[preview](https://web.stanford.edu/class/cs224n/materials/Gensim word vector visualization.html)]
- [x] [cs224n-2019-lecture02-wordvecs2](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture02-wordvecs2.pdf)
  - 复习word2vec(一个单词的向量是一行；得到的概率分布不区分上下文的相对位置；每个词和and, of等词共同出现的概率都很高)
  - optimization: 梯度下降，随机梯度下降SGD，mini-batch(32或64,减少噪声，提高计算速度)，每次只更新出现的词的向量(特定行)
  - 为什么需要两个向量？——数学上更简单(中心词和上下文词分开考虑),最终是把2个向量平均。也可以每个词只用一个向量。
  - word2vec的两个模型：Skip-grams(SG), Continuous Bag of Words(CBOW), 还有negative sampling技巧，抽样分布技巧(unigram分布的3/4次方)
  - 为什么不直接用共现计数矩阵？随着词语的变多会变得很大；维度很高，需要大量空间存储；后续的分类问题会遇到稀疏问题。解决方法：降维，只存储一些重要信息，固定维度。即做SVD。很少起作用，但在某些领域内被用的比较多，举例：Hacks to X(several used in Rohde et al. 2005)
  - Count based vs. direct prediction
  - Glove-结合两个流派的想法，在神经网络中使用计数矩阵，共现概率的比值可以编码成meaning component
  - 评估词向量的方法（内在—同义词、类比等，外在—在真实任务中测试，eg命名实体识别）
  - 词语多义性问题-1.聚类该词的所有上下文，得到不同的簇，将该词分解为不同的场景下的词。2.直接加权平均各个场景下的向量，奇迹般地有很好的效果
- [x] [cs224n-2019-notes02-wordvecs2](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)
  - Glove
  - 评估词向量效果的方法

**Suggested Readings**

- [x] [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
- [ ] [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
- [ ] [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)

Additional Readings:

- [ ] [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
- [ ] [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
- [ ] [On the Dimensionality of Word Embedding.](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

参考阅读

- [x] [理解GloVe模型（+总结）](https://blog.csdn.net/u014665013/article/details/79642083)(很详细易懂，讲解了GloVe模型的思想)



Python review[[slides](https://web.stanford.edu/class/cs224n/readings/python-review.pdf)]

**review**

glove的思想、算法步骤分解、代码

评估词向量的方法



### Lecture 03: Word Window Classification, Neural Networks, and Matrix Calculus

1. Course information update (5 mins)
2. Classification review/introduction (10 mins)
3. Neural networks introduction (15 mins)
4. Named Entity Recognition (5 mins)
5. Binary true vs. corrupted word window classification (15 mins)
6. Matrix calculus introduction (20 mins)

课件

- [ ] [cs224n-2019-lecture03-neuralnets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture03-neuralnets.pdf) 
  - 分类：情感分类，命名实体识别，买卖决策等，softmax分类器，cross-entropy损失函数(线性分类器)
  - 神经网络分类器，词向量分类的不同(同时学习权重矩阵和词向量，因此参数也更多)，神经网络简介
  - 命名实体识别(NER)：找到文本中的"名字"并且进行分类
  - 在上下文语境中给单词分类，怎么用上下文？将词及其上下文词的向量连接起来
  - 比如如果这个词在上下文中是表示位置，给高分，否则给低分
  - 梯度
- [x] [matrix calculus notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/gradient-notes.pdf)
- [ ] [cs224n-2019-notes03-neuralnets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes03-neuralnets.pdf)
  - 神经网络，最大边缘目标函数，反向传播
  - 技巧：梯度检验，正则，Dropout，激活函数，数据预处理(减去均值，标准化，白化Whitening)，参数初始化，学习策略，优化策略(momentum, adaptive)

Suggested Readings:

- [ ] [CS231n notes on backprop](http://cs231n.github.io/optimization-2/)
- [ ] [Review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/review-differential-calculus.pdf)

Additional Readings:

- [ ] [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

Assignment 2

[[code](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a2.zip)] [[handout](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a2.pdf)]

review

NER

梯度



### Lecture 04: Backpropagation and Computation Graphs

1. Matrix gradients for our simple neural net and some tips [15 mins]
2. Computation graphs and backpropagation [40 mins]
3. Stuff you should know [15 mins]
a. Regularization to prevent overfitting
b. Vectorization
c. Nonlinearities
d. Initialization
e. Optimizers
f. Learning rates

课件

- [x] [cs224n-2019-lecture04-backprop](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture04-backprop.pdf)
  - 梯度计算分解，一些tips，使用预训练的词向量的问题
  - 计算图表示前向传播和反向传播，用上游的梯度和链式法则来得到下游的梯度
  - 正则，矢量化，非线性，初始化，优化器，学习率

- [ ] [cs224n-2019-notes03-neuralnets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes03-neuralnets.pdf)



Suggested Readings:

1. [CS231n notes on network architectures](http://cs231n.github.io/neural-networks-1/)
2. [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)
3. [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)
4. [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)





### Lecture 05: Linguistic Structure: Dependency Parsing

1. Syntactic Structure: Consistency and Dependency (25 mins)
2. Dependency Grammar and Treebanks (15 mins)
3. Transition-based dependency parsing (15 mins)
4. Neural dependency parsing (15 mins)

[cs224n-2019-lecture05-dep-parsing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture05-dep-parsing.pdf) [[scrawled-on slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture05-dep-parsing-scrawls.pdf)]

- 短语结构，依赖结构

[cs224n-2019-notes04-dependencyparsing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes04-dependencyparsing.pdf)



Suggested Readings:

1. [Incrementality in Deterministic Dependency Parsing](https://www.aclweb.org/anthology/W/W04/W04-0308.pdf)
2. [A Fast and Accurate Dependency Parser using Neural Networks](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)
3. [Dependency Parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002)
4. [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf)
5. [Universal Stanford Dependencies: A cross-linguistic typology](http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf)
6. [Universal Dependencies website](http://universaldependencies.org/)

Assignment 3

[[code](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a3.zip)] [[handout](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a3.pdf)]



### Lecture 06: The probability of a sentence? Recurrent Neural Networks and Language Models

Recurrent Neural Networks (RNNs) and why they’re great for Language
Modeling (LM).

[cs224n-2019-lecture06-rnnlm](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture06-rnnlm.pdf)

- 语言模型
- RNN

[cs224n-2019-notes05-LM_RNN](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf)

Suggested Readings:

1. [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (textbook chapter)
2. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview)
3. [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2)
4. [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html)





### Lecture 07: Vanishing Gradients and Fancy RNNs

- Problems with RNNs and how to fix them

- More complex RNN variants

[cs224n-2019-lecture07-fancy-rnn](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture07-fancy-rnn.pdf)

- 梯度消失
- LSTM和GRU

[cs224n-2019-notes05-LM_RNN](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf)



Suggested Readings:

1. [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.3, 10.5, 10.7-10.12)
2. [Learning long-term dependencies with gradient descent is difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) (one of the original vanishing gradient papers)
3. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf) (proof of vanishing gradient problem)
4. [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html) (demo for feedforward networks)
5. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (blog post overview)

Assignment 4

[[code](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.zip)] [[handout](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.pdf)] [[Azure Guide](https://docs.google.com/document/d/1MHaQvbtPkfEGc93hxZpVhkKum1j_F1qsyJ4X0vktUDI/edit)] [[Practical Guide to VMs](https://docs.google.com/document/d/1z9ST0IvxHQ3HXSAOmpcVbFU5zesMeTtAc9km6LAPJxk/edit)]



### Lecture 08: Machine Translation, Seq2Seq and Attention

How we can do Neural Machine Translation (NMT) using an RNN based
architecture called sequence to sequence with attention

[cs224n-2019-lecture08-nmt](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf)

- 机器翻译：
  - 1.1950s，早期是基于规则的，利用词典翻译；
  - 2.1990s-2010s，基于统计的机器翻译(SMT)，从数据中学习统计模型，贝叶斯规则，考虑翻译和句子语法流畅。对齐：一对多，多对一，多对多。
  - 3.2014-，基于神经网络的机器翻译(NMT)，seq2seq，两个RNNs。seq2seq任务有：总结(长文本到短文本)，对话，解析，代码生成(自然语言到代码)。贪心解码。束搜索解码
  - 评估方式：BLEU(Bilingual Evaluation Understudy)
  - 未解决的问题：词汇表之外的词，领域不匹配，保持较长文本的上下文，低资源语料少，没有加入常识，从训练数据中学到了偏见，无法解释的翻译，
  - Attention。

[cs224n-2019-notes06-NMT_seq2seq_attention](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf)



Suggested Readings:

1. [Statistical Machine Translation slides, CS224n 2015](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml) (lectures 2/3/4)
2. [Statistical Machine Translation](https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5) (book by Philipp Koehn)
3. [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) (original paper)
4. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) (original seq2seq NMT paper)
5. [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf) (early seq2seq speech recognition paper)
6. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (original seq2seq+attention paper)
7. [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) (blog post overview)
8. [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) (practical advice for hyperparameter choices)



### Lecture 09: Practical Tips for Final Projects

1. Final project types and details; assessment revisited
2. Finding research topics; a couple of examples
3. Finding data
4. Review of gated neural sequence models
5. A couple of MT topics
6. Doing your research
7. Presenting your results and evaluation

[cs224n-2019-lecture09-final-projects](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture09-final-projects.pdf)

- 默认的项目是问答系统SQuAD
- Look at ACL anthology for NLP papers: https://aclanthology.info
- https://paperswithcode.com/sota
- 数据：
  - https://catalog.ldc.upenn.edu/
  - http://statmt.org
  - https://universaldependencies.org
  - Look at Kaggle，research papers，lists of datasets
  - https://machinelearningmastery.com/datasets-natural-languageprocessing/
  - https://github.com/niderhoff/nlp-datasets
- 

[final-project-practical-tips](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/final-project-practical-tips.pdf)



Suggested Readings:

1. [Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html) (*Deep Learning* book chapter)



### Lecture 10: Question Answering and the Default Final Project

1. Final final project notes, etc.
2. Motivation/History
3. The SQuAD dataset
4. The Stanford Attentive Reader model
5. BiDAF
6. Recent, more advanced architectures
7. ELMo and BERT preview

[cs224n-2019-lecture10-QA](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture10-QA.pdf)

- 两个部分：寻找那些可能包含答案的文档(信息检索)，从文档或段落中找答案(阅读理解)
- 阅读理解的历史，2013年MCTest：P+Q——>A，2015/16：CNN/DM、SQuAD数据集
- 开放领域问答的历史：1964年是依赖解析和匹配，1993年线上百科全书，1999年设立TREC问答，2011年IBM的DeepQA系统，2016年用神经网络和信息检索IR
- SQuAD数据集，评估方法
- 斯坦福的简单模型：Attentive Reader model，预测回答文本的起始位置和结束位置
- BiDAF

 [cs224n-2019-notes07-QA](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes07-QA.pdf)



Project Proposal

[[instructions](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/project/project-proposal-instructions.pdf)]

Default Final Project

[[handout](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/project/default-final-project-handout.pdf)] [[code](https://github.com/chrischute/squad)]



### Lecture 11: ConvNets for NLP

1. Announcements (5 mins)
2. Intro to CNNs (20 mins)
3. Simple CNN for Sentence Classification: Yoon (2014) (20 mins)
4. CNN potpourri (5 mins)
5. Deep CNN for Sentence Classification: Conneau et al. (2017)
(10 mins)
6. Quasi-recurrent Neural Networks (10 mins)

[cs224n-2019-lecture11-convnets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture11-convnets.pdf)

- CNN
- 句子分类

[cs224n-2019-notes08-CNN](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes08-CNN.pdf)



Suggested Readings:

1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf)



### Lecture 12: Information from parts of words: Subword Models

1. A tiny bit of linguistics (10 mins)
2. Purely character-level models (10 mins)
3. Subword-models: Byte Pair Encoding and friends (20 mins)
4. Hybrid character and word level models (30 mins)
5. fastText (5 mins)

[cs224n-2019-lecture12-subwords](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture12-subwords.pdf)

- 

Suggested readings:

1. Minh-Thang Luong and Christopher Manning. [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788)

Assignment 5

[[original code (requires Stanford login)](https://stanford.box.com/s/t4nlmcc08t9k6mflz6sthjlmjs7lip6p) / [public version](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5_public.zip)] [[handout](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5.pdf)]



### Lecture 13: Modeling contexts of use: Contextual Representations and Pretraining

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture13-contextual-representations.pdf)] [[video](https://youtu.be/S-CspeZ8FHc)]

Suggested readings:

1. Smith, Noah A. [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006). (Published just in time for this lecture!)
2. [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)





### Lecture 14: Transformers and Self-Attention For Generative Models(guest lecture by Ashish Vaswani and Anna Huang)

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture14-transformers.pdf)] [[video](https://youtu.be/5vcj8kSwBCY)]

Suggested readings:

1. [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
2. [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf)
3. [Music Transformer: Generating music with long-term structure](https://arxiv.org/pdf/1809.04281.pdf)

Project Milestone

[[instructions](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/project/project-milestone-instructions.pdf)]



### Lecture 15: Natural Language Generation

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture15-nlg.pdf)] [[video](https://youtu.be/4uG1NMKNWCU)]



### Lecture 16: Reference in Language and Coreference Resolution

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture16-coref.pdf)] [[video](https://youtu.be/i19m4GzBhfc)]



### Lecture 17: Multitask Learning: A general model for NLP? (guest lecture by Richard Socher)

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture17-multitask.pdf)] [[video](https://youtu.be/M8dsZsEtEsg)]



### Lecture 18: Constituency Parsing and Tree Recursive Neural Networks

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture18-TreeRNNs.pdf)] [[video](https://youtu.be/6Z4A3RSf-HY)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes09-RecursiveNN_constituencyparsing.pdf)]

Suggested Readings:

1. [Parsing with Compositional Vector Grammars.](http://www.aclweb.org/anthology/P13-1045)
2. [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/pdf/1805.01052.pdf)



### Lecture 19: Safety, Bias, and Fairness (guest lecture by Margaret Mitchell)

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture19-bias.pdf)] [[video](https://youtu.be/XR8YSRcuVLE)]



### Lecture 20: Future of NLP + Deep Learning

[[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture20-future.pdf)] [[video](https://youtu.be/3wWZBGN-iX8)]

**Final project poster session**
[[details](https://www.facebook.com/events/1218481914969541)]

**Final Project Report due** [[instructions](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/project/project-report-instructions.pdf)]

**Project Poster/Video due** [[instructions](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/project/project-postervideo-instructions.pdf)]



