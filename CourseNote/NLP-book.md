

**Neural Network Methods for Natural Language Processing**

the current day dominant approaches to language processing are all based on **statistical machine learning**

## Introduction

### The challenges of NLP

**ambiguous** 语义模糊：consider the sentence *I ate pizza with friends*, and compare it to *I ate pizza with olives*

**variable inputs in a system with ill defined**

**unspecified set of rules**：比如文档分类，文档中哪些词对该任务有帮助？如何写出这个任务的规则？

computational approaches, including machine learning:
**discrete**：语言的基本元素是字符，字符组成单词，又用来表示物体、概念、事件。字符和单词都是离散的符号，hamburger和pizza从symbols或letters来推测都没有任何内在关系。

**compositional**：字母组成单词，单词组成短语和句子，短语的含义要比组成它的单词的含义大得多。为了解释一段文本，必须不仅仅看单词和字母，还要看单词的长序列（句子甚至完整的文档）。

**sparse**：可能有效的句子数量是巨大的，不可能穷举，很多单词对我们来说都是新的，比如十年前或十年后的品牌名、俗语、术语。

### 神经网络和深度学习

While all of machine learning can be characterized as learning to make predictions based on past observations, deep learning approaches work by learning to not only predict but also to **correctly represent** the data, such that it is suitable for prediction.

### Deep Learning in NLP

A major component in neural networks for language is the use of an **embedding layer**, a mapping of discrete symbols to continuous vectors in a relatively low dimensional space.

Feed-forward networks, in particular multi-layer perceptrons (MLPs), allow to work with fixed sized inputs, or with variable length inputs in which we can disregard the order of the elements. When feeding the network with a set of input components, it learns to combine them in a meaningful way. MLPs can be used whenever a linear model was previously used. The nonlinearity of the network, as well as the ability to easily integrate pre-trained word embeddings, often lead to superior classification accuracy.

Convolutional feed-forward networks are specialized architectures that excel at extracting local patterns in the data: they are fed arbitrarily sized inputs, and are capable of extracting meaningful local patterns that are sensitive to word order, regardless of where they appear in the input. These work very well for identifying indicative phrases or idioms of up to a fixed length in long sentences or documents.

Recurrent neural networks (RNNs) are specialized models for sequential data. These are network components that take as input a sequence of items, and produce a fixed size vector that summarizes that sequence. As “summarizing a sequence” means different things for different tasks, recurrent networks are rarely used as standalone component, and their power is in being trainable components that can be fed into other network components, and trained to work in tandem with them. For example, the output of a recurrent network can be fed into a feed-forward network that will try
to predict some value. The recurrent network is used as an input-transformer that is trained to produce informative representations for the feed-forward network that will operate on top of it.They allow abandoning the markov assumption that was prevalent in NLP for decades, and designing models that can condition on entire sentences, while taking word order into account when it is needed, and not suffering much from statistical estimation problems stemming from data sparsity.

### Success stories



##Learning Basics and Linear Models

### Loss Functions

**Hinge (binary)**

For binary classification problems, the classifier’s output is a single scalar $\tilde y$ and the intended output y is in {-1, +1}.The classification rule is $\hat y =sign(\tilde y)$, and a classification is considered correct if $y·\tilde y > 0$

The hinge loss, also known as margin loss or SVM loss, is defined as:

$L_{hinge(binary)}(\tilde y,y)=max(0, 1-y·\tilde y)$

the binary hinge loss attempts to achieve a correct classification, with a margin of at least 1.

**Hinge (multi-class)**

The classification rule is defined as selecting the class with the highest score:

prediction = $argmax_i \hat y_{[i]}$

Denote by t =$argmax_i \hat y_{[i]}$ the correct class, and by k = $argmax_{i\ne t} \hat y_{[i]}$ the highest scoring class such that $k \neq t$ . The multi-class hinge loss is defined as:

$L_{hinge(binary)}(\hat y,y)=max(0, 1-(\hat y_{[t]}-\hat y_{[k]}))$

The multi-class hinge loss attempts to score the correct class above all other classes with a margin of at least 1.

Both the binary and multi-class hinge losses are intended to be used with linear outputs.The hinge losses are useful whenever we require a hard decision rule, and do not attempt to model class membership probability.

**Log loss**

The log loss is a common variation of the hinge loss, which can be seen as a “soft” version of the hinge loss with an infinite margin [LeCun et al., 2006]:

$L_{log}(\hat y,y)=log(1+exp((\hat y_{[t]}-\hat y_{[k]}))$

**Binary cross entropy**

![1529846154128](../summary_notes/NLP/img/bceloss.png)

**Categorical cross-entropy loss**

The categorical cross-entropy loss (also referred to as negative log likelihood) is used when a probabilistic interpretation of the scores is desired.

![1529846290722](../summary_notes/NLP/img/cceloss.png)

For hard-classification problems in which each training example has a single correct class assignment, y is a one-hot vector representing the true class. In such cases, the cross entropy can be simplified to:

![1529846394068](../summary_notes/NLP/img/cceloss2.png)

The cross-entropy loss is very common in the log-linear models and the neural networks literature, and produces a multi-class classifier which does not only predict the one-best class label but also predicts a distribution over the possible labels. When using the cross-entropy loss, it is assumed that the classifier’s output is transformed using the softmax transformation.

**Ranking losses**

![1529846545785](../summary_notes/NLP/img/rankloss.png)

###Regularization

![1529846855259](../summary_notes/NLP/img/L2.png)

The L2 regularizer is also called a gaussian prior or weight decay.

Note that L2 regularized models are severely punished for high parameter weights, but once the value is close enough to zero, their effect becomes negligible. The model will prefer to decrease the value of one parameter with high weight by 1 than to decrease the value of ten parameters that already have relatively low weights by 0.1 each.

![1529846954984](../summary_notes/NLP/img/L1.png)

![1529846988137](../summary_notes/NLP/img/elasticnet.png)

##From Linear Models to Multi-layer Perceptrons

## Feed-forward Neural Networks

### Common nonlinearities

**Sigmoid**:	$\sigma (x) = 1 / (1+e^{-x})$ also called the logistic function, is an S-shaped function, transforming each value x into the range [0,1].

**Hyperbolic tangent (tanh)**: $tanh(x)=\frac{e^{2x}-1}{e^{2x}+1}$, is an S-shaped function, transforming each value x into the range [0,1].

**Hard tanh**: an approximation of the tanh function which is faster to compute and to find derivatives thereof

$hardtanh(x)=\begin{cases} -1, & \text {x<-1} \\ 1, & \text{x>1} \\x, &\text{otherwise} \end{cases} $

**Rectifier (ReLU)**: Despite its simplicity, it performs well for many tasks, especially when combined with the dropout regularization technique

$ReLu(x)=max(0,x)=\begin{cases} 0, & \text {x<0} \\x, &\text{otherwise} \end{cases} $

As a rule of thumb, both ReLU and tanh units work well, and significantly outperform the sigmoid.

![1530166011941](../summary_notes/NLP/img/activation.png)

## Neural Network Training

if your computational resources allow, it is advisable to **run the training process several times, each with a different random initialization**, and choose the best one on the development set. This technique is called random restarts.The average model accuracy across random seeds is also interesting, as it gives a hint as to the stability of the process.

**Saturated** neurons are caused by too large values entering the layer. This may be controlled for by changing the initialization, scaling the range of the input values, or changing the learning rate. **Dead neurons** are caused by all signals entering the layer being negative (for example this can happen after a large gradient update). Reducing the learning rate will help in this situation. For saturated layers, another option is to normalize the values in the saturated layer after the activation.

Layer normalization is an effective measure for countering saturation, but is also expensive in terms of gradient computation. A related technique is batch normalization, due to Ioffe and Szegedy [2015], in which the activations at each layer are normalized so that they have mean 0 and variance 1 across each mini-batch. The batch-normalization techniques became a key component for effective training of deep networks in computer vision. As of this writing, it is less popular in natural language applications.

## Features for Textual Data

Word，Texts，Paired Texts，Word in Context，Relation between two words







