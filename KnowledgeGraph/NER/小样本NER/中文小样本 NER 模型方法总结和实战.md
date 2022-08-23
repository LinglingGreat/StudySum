
一、简介
----

在 UIE 出来以前，小样本 NER 主要针对的是英文数据集，目前主流的小样本 NER 方法大多是基于 prompt，在英文上效果好的方法，在中文上不一定适用，其主要原因可能是：

1.  中文长实体相对英文较多，英文是按 word 进行切割，很多实体就是一个词；边界相对来说更清晰；
    
2.  生成方法对于长实体来说更加困难。但是随着 UIE 的出现，中文小样本 NER 的效果得到了突破。
    

二、主流小样本 NER 方法
--------------

### 2.1、EntLM

EntLM 该方法核心思想：抛弃模板，把 NER 作为语言模型任务，实体的位置预测为 label word, 非实体位置预测为原来的词，该方法速度较快。模型结果图如图 2-1 所示：

![](img/Pasted%20image%2020220822103533.png)
图 2-1 EntLM 模型

论文重点在于如何构造 label word：在中文数据上本实验做法与论文稍有区别，但整体沿用论文思想：下面介绍了基于中文数据的标签词构造过程；

1.  采用领域数据构造实体词典；
    
2.  基于实体词典和已有的实体识别模型对中文数据 (100 000) 进行远程监督，构造伪标签数据；
    
3.  采用预训练的语言模型对计算 LM 的输出，取实体部分概率较高的 top3 个词；
    
4.  根据伪标签数据和 LM 的输出结果，计算词频；由于可能出现在很多类中都出现的高频标签词，因此需要去除冲突，该做法沿用论文思想；
    
5.  使用均值向量作为类别的原型，选择 top6 高频词的进行求平均得到均值向量；
    

### 2.2、TemplateNER

TemplateNER 的核心思想就是**采用生成模型的方法来解决 NER 问题**，训练阶段通过构造模板，让模型学习哪些 span 是实体, 哪些 span 不是实体，模板集合为：$T=[T+,T+ ...T+,T-]$,T + 为 xx is aentity，T - 为 xx is not aentity, 训练时采用目标实体作为正样本，负样本采用随机非实体进行构造，负样本的个数是正样本的 1.5 倍。推理阶段，原始论文中是 n-gram 的数量限制在 1 到 8 之间，作为实体候选，但是中文的实体往往过长，所以实验的时候是将，n-gram 的长度限制在 15 以内，推理阶段就是对每个模板进行打分，选择得分最大的作为最终实体。

这篇论文在应用中的需要注意的主要有二个方面：

1.  模板有差异，对结果影响很大，模板语言越复杂，准确率越低；
    
2.  随着实体类型的增加，会导致候选实体量特别多，训练，推理时间更，尤其在句子较长的时候，可能存在效率问题，在中文数据中，某些实体可能涉及到 15 个字符（公司名），导致每个句子的候选 span 增加，线上使用困难, 一条样本推理时间大概 42s
    

![](img/Pasted%20image%2020220822103620.png)
 图 2-2 TemplateNER 抽取模型

### 2.3、LightNER

LightNER 的核心思想**采用生成模型进行实体识别**，预训练模型采用 BART 通过 **prompt** 指导**注意力层**来重新调整注意力并适应预先训练的权重， 输入一个句子，输出是：实体的序列，每个实体包括：实体 span 在输入句子中的 start index，end index ，以及实体类型 ，该方法的思想具有一定的通用性，可以用于其他信息抽取任务。
![](img/Pasted%20image%2020220822103651.png)

图 2-3 LightNER 抽取模型

### 2.4、UIE

UIE(通用信息抽取框架) 真正的实现其实是存在两个版本，最初是中科院联合百度发的 ACL2022 的一篇论文，Unified Structure Generation for Universal Information Extraction，这个版本采用的是 T5 模型来进行抽取，采用的是生成模型，后来百度推出的 UIE 信息抽取框架，采用的是 span 抽取方式，直接抽取实体的开始位置和结束位置，其方法和原始论文并不相同，但是大方向相同。

1.  输入形同：UIE 采用的是前缀 prompt 的形式，采用的是 Schema+Text 的形式作为输入，文本是 NER 任务，所以 Schema 为实体类别，比如：人名、地名等。
    
2.  采用的训练形式相同，都是采用预训练加微调的形式
    

不同点：

1.  百度 UIE 是把 NER 作为抽取任务，分别预测实体开始和结束的位置，要针对 schema 进行多次解码，比如人名进行一次抽取，地名要进行一次抽取，以次类推，也就是一条文本要进行 n 次，n 为 schema 的个数，原始 UIE 是生成任务，一次可以生成多个 schema 对应的结果
    
2.  百度 UIE 是在 ernie 基础上进行预训练的，原始的 UIE 是基于 T5 模型。
    

![](img/Pasted%20image%2020220822103714.png)
图 2-4 UIE 抽取模型

三、实验结果
------

该部分主要采用**主流小样本 NER 模型在中文数据**上的实验效果。

通用数据 1 测试效果：

<table><thead><tr><th>Method</th><th>5-shot</th><th>10-shot</th><th>20-shot</th><th>50-shot</th></tr></thead><tbody><tr><td>BERT-CRF</td><td>-</td><td>0.56</td><td>0.66</td><td>0.74</td></tr><tr><td>LightNER</td><td>0.21</td><td>0.42</td><td>0.57</td><td>0.73</td></tr><tr><td>TemplateNER</td><td>0.24</td><td>0.44</td><td>0.51</td><td>0.61</td></tr><tr><td>EntLM</td><td>0.46</td><td>0.54</td><td>0.56</td><td>-</td></tr></tbody></table>

从实验结果来看，其**小样本 NER 模型在中文上的效果都不是特别理想**，没有达到 Bert-CRF 的效果，一开始怀疑结果过拟了，重新换了测试集，发现 BERT-CRF 效果依旧变化不大，就是比其他的小样本学习方法好。

### 3.1、UIE 实验结果

UIE 部分做的实验相对较多，首先是消融实验，明确 UIE 通用信息抽取的能力是因为预训练模型的原因，还是因为模型本身的建模方式让其效果好，其中，BERTUIE，采用 BERT 作为预训练语言模型，pytorch 实现，抽取方式采用 UIE 的方式，抽取实体的开始和结束位置。

领域数据 1 测试结果（实体类型 7 类）：

<table><thead><tr><th>预训练模型</th><th>框架</th><th>F1</th><th>Epoch</th></tr></thead><tbody><tr><td>Ernie3.0</td><td>Paddle</td><td>0.71</td><td>200</td></tr><tr><td>Uie-base</td><td>paddle</td><td>0.72</td><td>100</td></tr><tr><td>BERT</td><td>pytorch</td><td>0.705</td><td>30</td></tr></tbody></table>

从本部分实验可以确定的是，预训练模型其实就是一个锦上添花的作用， UIE 的本身建模方式更重要也更有效。

领域数据 1 测试结果（实体类型 7 类）：

<table><thead><tr><th><br></th><th>5-shot</th><th>10-shot</th><th>20-shot</th><th>50-shot</th></tr></thead><tbody><tr><td>BERT-CRF</td><td>0.697</td><td>0.75</td><td>0.82</td><td>0.85</td></tr><tr><td>百度 UIE</td><td>0.76</td><td>0.81</td><td>0.84</td><td>0.87</td></tr><tr><td>BERTUIE</td><td>0.73</td><td>0.79</td><td>0.82</td><td>0.87</td></tr><tr><td>T5（放宽后评价）</td><td>0.71</td><td>0.75</td><td>0.79</td><td>0.81</td></tr></tbody></table>

领域数据 3 测试效果（实体类型 6 类），20-shot 实验结果：

<table><thead><tr><th width="38"><br></th><th width="47">BERT-CRF</th><th>LightNER</th><th>EntLM</th><th>百度 UIE<br></th><th>BERTUIE</th></tr></thead><tbody><tr><td width="38">F1</td><td width="47">0.69</td><td>0.57</td><td>0.58</td><td>0.72</td><td>0.69</td></tr></tbody></table>

**UIE 在小样本下的效果相较于 BERT-CRF 之类的抽取模型要好，但是 UIE 的速度较于 BERT-CRF 慢很多**，大家可以根据需求决定用哪个模型。如果想进一步提高效果，可以针对领域数据做预训练，本人也做了预训练，效果确实有提高。


## 参考资料

https://mp.weixin.qq.com/s/81Ef0hhRoEeOrVyTge61FA
