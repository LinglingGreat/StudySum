## 训练

![](img/Pasted%20image%2020231121101957.png)

基于LLaMA2-13B扩充词表，训练了700B tokens

第一阶段：大量高质量中英文数据

第二阶段：加入了有监督数据

第三阶段：focus在数学数据

## 数据预处理

![](img/Pasted%20image%2020231121112430.png)

处理速度：15GB/per hour，通过这个积累了4.5T的数据

DP：语言检测，只选择中英文数据；标准化语料的编码格式，将所有中文转成简体；删除无用的tokens比如不可见的控制字符，特殊符号，表情，不正确的标点符号。

AS：用KenLM，中英文维基百科数据训练了2个语言模型，在输入数据上执行ppl评估。ppl分数从低到高排序，选择top 30%的作为高质量数据，30%-60%的作为中等质量数据。

RF：在文档、段落、句子这三个粒度设计了30多种过滤规则，从大粒度到小粒度依次过滤。在文档粒度，规则主要围绕内容长度和格式设计，在段落和句子粒度，规则围绕内容的有毒性。规则设计过程会随机采样一些数据进行人工评估，然后迭代优化规则。

CD：布隆过滤器和Simhash去重。
- 发现CC等开源数据中存在大量重复的网页，用布隆过滤器去重URL，减少后续内容去重的复杂度。
- 发现很多网页内容相似，不同之处在于特殊符号（比如标点和表情），针对这些网页进行一轮精准去重。
- simhash进行粗略去重
- 采样去重数据评估
- 用了cache和bucket技术，使得新数据不需要对所有老数据进行冗余检查。

DE：机器（抽样1%）和人工评估（抽样1000个），满足质量要求的数据占比需要达到一定阈值，否则再过一遍清洗。

![](img/Pasted%20image%2020231121133250.png)

从13TB的数据中过滤出4.5TB的数据。自己的数据包括代码和书籍。

![](img/Pasted%20image%2020231121133736.png)

Pile-Pajama is a de-duplicated fusion of the Pile and Redpajama datasets after removing Common Crawl data. CC is a de-duplicated fusion of data from Pile and Redpajama that originated from Common Crawl. Wudao-Ziya is a dataset that combines our collected data with the Wudao dataset. 
Yuan1.0 is an open-source dataset provided by Inspur Technology, and we filter the raw data using our cleaning rules. Translate is the multilingual translation dataset we collect. Code is the code data we collect from GitHub, which includes multiple programming languages such as C, C++, and Python. We add the program language type before the code and change it to a format that the Markdown syntax is able to recognize. In this way, the model we train is able to generate formatted code. Instruct is a dataset constructed from instructions that we collect. Wanjuan-Ziya is a dataset that combines high-quality data from the Wanjuan dataset, as well as math-related data we collect ourselves. MetaMath-Ziya is a dataset derived from the Huawei’s open-source MetaMath dataset after data augmentation. We construct some Chinese and English prompts for Instruct, WanjuanZiya, and MetaMath-Ziya datasets, such as “QA”, “question-answer”, “problem-solution”, etc



## 评估

![](img/Pasted%20image%2020231121102555.png)

