# Why ML Strategy

Ideas:

* Collect more data

* Collect more diverse training set

* Train algorithm longer with gradient descent

* Try Adam instead of gradient descent

* Try bigger network

* Try smaller network

* Try dropout

* Add 𝐿2regularization

* Network architecture

  * Activation functions

  * num of hidden units

  * ...

    ​

# 正交化

Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects to other components of the system. It becomes easier to verify the algorithms independently from one another, it reduces testing and development time.
When a supervised learning system is design, these are the 4 assumptions that needs to be true and orthogonal.

1. Fit training set well in cost function
   - If it doesn’t fit well, the use of a bigger neural network or switching to a better optimization algorithm might help.
2. Fit development set well on cost function
   - If it doesn’t fit well, regularization or using bigger training set might help.
3. Fit test set well on cost function
   - If it doesn’t fit well, the use of a bigger development set might help
4. Performs well in real world
   - If it doesn’t perform well, the development test set is not set correctly or the cost function is not evaluating the right thing.

# 单一数字评估指标

Single number evaluation metric

To choose a classifier, a well-defined development set and an evaluation metric speed up the iteration process.

比如可以用F1-score综合precision和recall，这样只用看一个指标，在precision和recall无法判断的情形下：如

| classifier | Precision(p) | Recall(r) | F1-Score |
| ---------- | ------------ | --------- | -------- |
| A          | 95%          | 90%       | 92.4%    |
| B          | 98%          | 85%       | 91.0%    |

这样从F1-Score上判断，分类器A更好

# 满足和优化指标

Satisficing and optimizing metric

There are different metrics to evaluate the performance of a classifier, they are called evaluation matrices. They can be categorized as satisficing and optimizing matrices. It is important to note that these evaluation matrices must be evaluated on a training set, a development set or on the test set.

比如在某些任务中，accuracy是我们想要最大化的目标，它是优化指标，但我们也不想运行时间太长，可以设置运行时间小于100ms，它是满足指标。

# 训练/开发/测试集分布

Setting up the training, development and test sets have a huge impact on productivity. It is important to choose the development and test sets from the same distribution and it must be taken randomly from all the data.
Guideline
Choose a development set and test set to reflect data you expect to get in the future and consider important to do well.

# 开发机和测试集的大小

Old way of splitting data
We had smaller data set therefore we had to use a greater percentage of data to develop and test ideas and models.

70/30或60/20/20

Modern era – Big data
Now, because a large amount of data is available, we don’t have to compromised as much and can use a greater portion to train the model.

98/1/1

Guidelines

* Set up the size of the test set to give a high confidence in the overall performance of the system.
* Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data set.
* The development set has to be big enough to evaluate different ideas.

# 什么时候该改变开发/测试集和指标

The evaluation metric fails to correctly rank order preferences between algorithms. The evaluation metric or the development set or test set should be changed.

If doing well on your metric + dev/test set does not correspond to doing well on your application, change your metric and/or dev/test set.

Guideline 

1. Define correctly an evaluation metric that helps better rank order classifiers 
2. Optimize the evaluation metric

# 和人类水平比较

Today, machine learning algorithms can compete with human-level performance since they are more productive and more feasible in a lot of application. Also, the workflow of designing and building a machine learning system, is much more efficient than before.
Moreover, some of the tasks that humans do are close to ‘’perfection’’, which is why machine learning tries to mimic human-level performance.

Machine learning progresses slowly when it surpasses human-level performance. One of the reason is that human-level performance can be close to Bayes optimal error, especially for natural perception problem.
Bayes optimal error is defined as the best possible error. In other words, it means that any functions mapping from x to y can’t surpass a certain level of accuracy.
Also, when the performance of machine learning is worse than the performance of humans, you can improve it with different tools. They are harder to use once its surpasses human-level performance.
These tools are:

- Get labeled data from humans
- Gain insight from manual error analysis: Why did a person get this right?
- Better analysis of bias/variance.

# 可避免误差

Avoidable bias

By knowing what the human-level performance is, it is possible to tell when a training set is performing well or not.

在某些任务中，比如计算机视觉，可以用human leval error代替Bayes error,因为人类在这方便的误差很小，接近最优水平。

Scenario A
There is a 7% gap between the performance of the training set and the human level error. It means that the algorithm isn’t fitting well with the training set since the target is around 1%. To resolve the issue, we use bias reduction technique such as training a bigger neural network or running the training set longer.
Scenario B
The training set is doing good since there is only a 0.5% difference with the human level error. The difference between the training set and the human level error is called avoidable bias. The focus here is to reduce the variance since the difference between the training error and the development error is 2%. To resolve the issue, we use variance reduction technique such as regularization or have a bigger training set.

# 理解人类水平表现

The definition of human-level error depends on the purpose of the analysis.比如，如果目标是替代贝叶斯误差，就应该选择最小的误差作为人类误差。

Summary of bias/variance with human-level performance

* Human - level error – proxy for Bayes error
* If the difference between human-level error and the training error is bigger than the difference between the training error and the development error. The focus should be on bias reduction technique
* If the difference between training error and the development error is bigger than the difference between the human-level error and the training error. The focus should be on variance reduction technique

# 超过人的表现

Surpassing human-level performance

There are many problems where machine learning significantly surpasses human-level performance, especially with structured data:

* Online advertising
* Product recommendations
*  Logistics (predicting transit time)
* Loan approvals

# 改善你的模型表现

The two fundamental assumptions of supervised learning

The first one is to have a low avoidable bias which means that the training set fits well. The second one is to have a low or acceptable variance which means that the training set performance generalizes well to the development set and test set.

Avoidable bias   (Human-level <——> Training error)

* Train bigger model
* Train longer, better optimization algorithms
* Neural Networks architecture / hyperparameters search

Variance   (Training error <——> Development error)

* More data
* Regularization
* Neural Networks architecture / hyperparameters search



# 进行误差分析

Error analysis:

* Get ~100 mislabeled dev set examples.
* Count up how many are dogs.统计不同类型错误的比例

Ideas for cat detection:

* Fix pictures of dogs being recognized as cats
* Fix great cats (lions, panthers, etc..) being misrecognized
* Improve performance on blurry images

# 清除标注错误的数据

DL algorithms are quite robust to random errors in the training set.(but not systematic errors比如所有白色的小狗都被标记成猫)

误差分析：

Overall dev set error

Errors due incorrect labels

Errors due to other causes

根据误差来源决定应该侧重于修正哪方面的误差

Correcting incorrect dev/test set examples

* Apply same process to your dev and test sets to make sure they continue to come from the same distribution
* Consider examining examples your algorithm got right as well as ones it got wrong.（通常不这么做，因为分类正确的样本太多）
* Train and dev/test data may now come from slightly different distributions.

# 快速搭建第一个系统

* Set up dev/test set and metric
* Build initial system quickly
* Use Bias/Variance analysis & Error analysis to prioritize next steps

Guideline:

Build your first system quickly, then iterate

# 训练集和测试集来自不同分布

可以把比较好的数据（如清晰的图片）作为训练集，不太好的数据（比如模糊的图片）分成三部分：训练集、开发集和测试集

# 偏差和方差(不匹配数据分布中)

Training-dev set: Same distribution as training set, but not used for training

Human level——(avoidable bias)——Training error——(variance)——Training-dev error——(data mismatch)——Dev error——(degree of overfitting to dev set)——Test error

# 定位数据不匹配

如果认为存在数据不匹配问题，可以做误差分析，或者看看训练集或开发集来找出这两个数据分布到底有什么不同，然后看看是否能够收集更多看起来像开发集的数据做训练。其中一种方法是人工数据合成，在语音识别中有用到。但是要注意在合成时是不是从所有可能性的空间只选了很小一部分去模拟数据。

# 迁移学习

神经网络中把从一个任务中学到的知识应用到另一个独立的任务中。

如，把图像识别中学到的知识应用或迁移到放射科诊断上。

假设已经训练好一个图像识别神经网络。应用到放射科诊断上时，可以初始化最后一层的权重，重新训练。

如果数据量小，可以只训练后面一道两层；如果数据量很大，可以对所有参数重新训练。

用图像识别的数据对神经网络的参数进行训练——预训练pre-training

用放射科数据重新训练权重——微调fine tuning

用于迁移目标任务数据很少，而相关知识任务数据量很大的情况下。

When transfer learning makes sense?

* Task A and B have the same input x
* You have a lot more data for Task A than Task B
* Low level features from A could be helpful for learning B

# 多任务学习

让单个神经网络同时做多件事，每个任务都能帮到其他所有任务。

使用频率低于迁移学习。计算机视觉，物体检测中常用多任务学习。

When multi-task learning makes sense

* Training on a set of tasks that could benefit from having shared lower-level features.
* Usually: Amount of data you have for each task is quite similar.至少一个任务的数据要低于其它任务的数据之和
* Can train a big enough neural network to do well on all the tasks.

# 端到端深度学习

end-to-end learning, 学习从输入到输出的直接映射，省略了中间步骤

需要数据集很大

端到端学习不一定是最好的，有时候多步学习会表现更好

# 是否要用端到端学习

Pros:

* Let the data speak
* Less hand-designing of components needed

Cons:

* May need large amount of data
* Excludes potentially useful hand-designed components

Key question: Do you have sufficient data to learn a function of the complexity needed to map x to y?

