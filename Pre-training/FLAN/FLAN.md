Finetuned Language Models Are Zero-Shot Learners 2021.9.3

Motivation：

本文提出一种基于instruction-tuning的方法叫做FLAN，一种通过提升语言模型对instructions的理解能力从而提高语言模型零样本学习能力的简单方法。

Method：

a.训练模型：137B规模的decoder-only LM---LaMDA-PT

b.数据集：一共62个的NLP task,根据任务类别进行分类，例如Natural language inference、Machine translation和Sentiment等，为了增加多样性对于每个任务10个手动构建10个template。

c.评估方法：

例如当评估Natural language inference时候，把所有属于Natural language inference从训练集中剔除。当做生成任务时候直接用语言模型生成目标结果，当做分类时候，设置了一个_options suffix_选项后缀，将所有的分类结果通过OPTIONS:拼在样本后面。fews-shot场景使用packing将多个训练示例组合成一个序列，使用特殊的EOS将输入与目标分离。

d:训练方法：

混合所有数据集，并从每个数据集中随机抽取样本，为了平衡不同大小的数据集，将每个数据集的训练示例数限制为30k，并遵循示例比例混合方案，最大混合率为3k。一共训练30k steps，输入最大长度为1024，输出最大长度为256。

Contribution：

在25个数据集中的20个上超过了零样本学习的175B GPT-3，甚至在ANLI、RTE、BoolQ、AI2-ARC、OpenbookQA和StoryCloze上都远远优于few-shot小样本学习的GPT-3。实验表明任务的聚类数量增加、模型规模增大和instruction都对模型的zero-shot能力有促进作用，这些因素也比较符合直觉，并且激发了对通用模型的进一步研究。但是instruction tuning虽然对于自然表述为指令的任务（例如NLI、QA、翻译等）非常有效，而对于直接表述为语言建模的任务(Natural language inference等）则不太有效,可能和NLI样例不太可能在无监督的训练集中自然出现有关系。


链接：https://zhuanlan.zhihu.com/p/558286175  