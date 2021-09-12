预训练语言模型(PLMs)通常有着极大的参数量，并且需要较长的推断时间，以致于很难应用到边界设备比如手机上。而且研究表明PLMs中存在冗余。因此，有必要并且有这个可能在保证效果的情况下去降低PLMs的计算量和模型存储。

压缩技术有：量化(quantization)，权重裁剪(weights pruning)，知识蒸馏(knowledge distillation, KD)。TINYBERT专注于知识蒸馏。KD旨在将大型的teacher网络中的知识迁移(transfer)到小型的student网络中，训练一个reproduce teacher网络的行为的student网络。

本文提出的方法中，Transformer distillation和two-stage learning framework是两个关键的idea。

Transformer distillation去蒸馏teacher BERT中的知识，有几个损失函数（1）embedding层的输出；（2）Transformer层得到的hidden states和attention matrices；（3）prediction层的logits输出。BERT学到的attention weights能够捕捉到实质性的语言知识，现有的BERT的知识蒸馏方法都忽略了这一点。

two-stage learning framework包括general distillation和task-specific distillation。在general阶段，没有fine-tuning的原始BERT作为teacher model，student TinyBERT通过在general domain的大量语料中使用Transformer distillation来学习模仿teacher的行为。task-specific阶段，使用数据增强和Transformer distillation来学习。

标准的Transformer层包括两个子层：多头注意力层(multi-head attention, MHA), 前向全连接层(fully connected feed-forward network, FFN).

KD就是最小化tearcher网络和student网络的行为函数之间的loss。关键就在于如何定义行为函数和损失函数。KD中的行为函数(behavior function)旨在将网络的输入转变成一些富有信息量的表示，可以是网络的任何一层的输出。

问题1：BERT 结构

BERT是多个Transformer Encoder堆叠起来的网络。

输入是：CLS, 句子1, SEP, 句子2, SEP.

然后每个token都会有3个embedding：token embedding，position embedding, segment embedding（属于第一个句子还是第二个句子）

预训练包括两个任务：预测被mask的词，预测两个句子是否是关联的（句子2是否是1的下一个句子）

输出则是每个token对应的向量表示。

问题2：什么是knowledge distillation （KD）

KD旨在将大型的teacher网络中的知识迁移(transfer)到小型的student网络中。student网络计算量更小，同时能够保留teacher网络中关键的信息特征。

问题3：常用的KD模型

Output Transfer——将网络的输出（Soft-target，后面会介绍其含义）作为知识；

Feature Transfer——将网络学习的特征作为知识；

Relation Transfer——将网络或者样本的关系作为知识；



DistillBert FastBert







