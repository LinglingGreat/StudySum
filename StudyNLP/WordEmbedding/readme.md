### 评估词向量的方法

内在—同义词、类比等，计算速度快，有助于理解这个系统，但是不清楚是否真的有用，除非与实际任务建立了相关性

外在—在真实任务中测试，eg命名实体识别；计算精确度可能需要很长时间；不清楚子系统是问题所在，是交互问题，还是其他子系统；如果用另一个子系统替换一个子系统可以提高精确度



### 词语多义性问题

1.聚类该词的所有上下文，得到不同的簇，将该词分解为不同的场景下的词。

2.直接加权平均各个场景下的向量，奇迹般地有很好的效果
