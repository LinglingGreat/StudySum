
平台：千言，paperwithcode


文本生成： https://dqwang122.github.io/projects/CNewSum/

多任务： https://github.com/CLUEbenchmark/FewCLUE

阅读理解： https://github.com/nlpdata/c3

自然语言推理： https://github.com/CLUEbenchmark/OCNLI

其它： https://github.com/THUNLP-AIPoet/CCPM

[大模型\_人工智能高质量数据集\_人工智能数据标注平台-北京人工智能高质量数据集服务平台](http://dataset.baiia.org.cn/datasets/list-5.html)

搜狗实验室数据：[%E5%B8%B8%E7%94%A8%E6%95%B0%E6%8D%AE%E9%9B%86%E7%AE%80%E4%BB%8B.md](https://github.com/duoergun0729/nlp/blob/master/%E5%B8%B8%E7%94%A8%E6%95%B0%E6%8D%AE%E9%9B%86%E7%AE%80%E4%BB%8B.md)

中文图书语料：[GitHub - FudanNLPLAB/CBook-150K: 中文图书语料MD5链接](https://github.com/FudanNLPLAB/CBook-150K)


[**Pre-training Corpora, Fine-tuning Instruction Datasets, Preference Datasets, Evaluation Datasets, and Traditional NLP Datasets**..](https://github.com/lmmlzn/Awesome-LLMs-Datasets)

[LLMDataHub: Awesome Datasets for LLM Training](https://github.com/Zjh-819/LLMDataHub)

幽默数据集：[TuringsSolutions (Richard A Aragon)](https://huggingface.co/TuringsSolutions)

c.ai的bot：[New Arrivals](https://rentry.org/cai-list)

角色扮演数据集： [ERP/RP and erotica raw data collection](https://rentry.org/qib8f)

[Dampf's list of good datasets for LLM fine-tuning](https://rentry.org/datasets-llm)

[Dataset Collection](https://rentry.co/cvmbb)

[CIS 700 - Resources](https://interactive-fiction-class.org/resources.html)

[Releasing Common Corpus: the largest public domain dataset for training LLMs](https://huggingface.co/blog/Pclanglais/common-corpus)

[大模型预训练中的数据处理及思考](https://mp.weixin.qq.com/s/oKMLhw5hk0LP85dtRAzBDg)

[大模型时代，数据为王，在哪里寻找开源数据集？](https://mp.weixin.qq.com/s/ADGg6OCqjFQ-bLE-X-Q9DA)


## 微调数据集

[H-D-T/Buzz · Datasets at Hugging Face](https://huggingface.co/datasets/H-D-T/Buzz) 一个3千万的数据集

[Eurus - a openbmb Collection](https://huggingface.co/collections/openbmb/eurus-660bc40bec5376b3adc9d1c5) Eurus模型用到的UltraInteract数据集，据他们说很好。

有关COIG系列，我们开源了以下工作用于大家做中文SFT使用，其中每个集合都有一点点各自的问题，但是设计上因为成本等问题无法避免，我们列出了每个子集的设计目标和使用前可能需要进行的处理：
1. https://huggingface.co/datasets/BAAI/COIG ，COIG第一版，其中有几个子集比较老，是当时数据来源比较稀少的情况下，选择了较为粗暴的方法。但是其中Leetcode，Human Value Alignment 3000，以及Counterfactual Correction Multi-round这三个集合质量较高，可以考虑采样使用。
2. https://huggingface.co/datasets/BAAI/COIG-PC ，COIG-PC，target中文版的FLAN，但是由于原始中文互联网数据质量参差不齐，部分任务子集可能存在一些噪音数据，建议使用前自行清洗，同时存在着分布不均的情况。出于作为中文可被处理的任务集数据备份的目的，我们尽量全量备份和处理了所有可用数据。
3. https://huggingface.co/datasets/BAAI/COIG-PC-core ，COIG-PC-core，在COIG-PC基础上使用GPT-4，人力清洗等方式筛选出来的干净的核心数据，可以采样或者全量使用提升模型的传统中文NLP任务理解和执行能力。(GPT-4只用于判分并未用于生成，所以也可以随便商用)
4. https://huggingface.co/datasets/m-a-p/COIG-Kun ，https://huggingface.co/m-a-p/Kun-LabelModel，COIG-Kun，中文版的humpback，可以从预训练数据转译生成SFT数据，目前存在生成数据指令逻辑比较简单等问题，但是可以有效扩充SFT数据量尤其是垂域的SFT数据量。
5. https://huggingface.co/datasets/m-a-p/COIG-CQIA ，中文版的LIMA，全人工爬取，精心挑选和修改后的数据集，可以直接作为提升中文基础能力的SFT集合。

