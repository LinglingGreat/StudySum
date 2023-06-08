---
title: Linly
created: 2023-06-08
tags: LLM, 预训练, 增量预训练, SFT

---

https://github.com/CVI-SZU/Linly

# Linly OpenLLaMA

https://github.com/CVI-SZU/Linly/wiki/Linly-OpenLLaMA

Linly-OpenLLaMA 使用和 Meta 相同的模型结构和训练参数从头预训练。使用的数据包含中、英文无监督数据和平行语料，在语料上重新训练 spm tokenizer，在中文上获得字词结合的分词效果。

## 构建tokenizer

首先在大规模中英文语料上训练 SPM，词表大小为 50000

根据[结巴分词](https://github.com/fxsjy/jieba)词频前20000的词表扩充中文词，并扩充简繁体汉字。扩充后词表大小为 66242。

## 预训练数据

在第一阶段预训练，共使用100GB语料，其中20G中文语料、10G平行语料、70G英文语料。

### 中文数据集

|数据集|Disk Size|Link|
|---|---|---|
|ClueCorpusSmall|13G|[https://github.com/CLUEbenchmark/CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)|
|中文维基百科 2023|2.5G|[https://download.wikipedia.com/zhwiki/](https://download.wikipedia.com/zhwiki/)|
|CSL|1.5G|[https://github.com/ydli-ai/CSL](https://github.com/ydli-ai/CSL)|
|news-crawl|2.3G|[https://data.statmt.org/news-crawl/zh/](https://data.statmt.org/news-crawl/zh/)|

### 平行语料

|数据集|Disk Size|Link|
|---|---|---|
|UNCorpus|4.3G|[https://conferences.unite.un.org/UNCorpus](https://conferences.unite.un.org/UNCorpus)|
|translation2019zh|1.3G|[https://github.com/brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)|
|WikiMatri|0.6G|[http://data.statmt.org/wmt21/translation-task/WikiMatrix/](http://data.statmt.org/wmt21/translation-task/WikiMatrix/)|
|news-commentry|67M|[http://data.statmt.org/wmt20/translation-task/back-translation/](http://data.statmt.org/wmt20/translation-task/back-translation/)|
|ParaCrawl v9|2.6G|[https://paracrawl.eu/](https://paracrawl.eu/)|

### 英文数据集

|数据集|Disk Size|Link|
|---|---|---|
|英文维基百科 2023|20G|[https://download.wikipedia.com/enwiki/](https://download.wikipedia.com/enwiki/)|
|arxiv|10G|[https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data)|
|GitHub|10G|同上|
|Book|18G|同上|
|stackexchange|13G|同上|

## 预训练参数

我们首先训练 13B 模型，基本上采用 LLaMA 的训练参数，其中 Batch Size 通过梯度累积实现。

- Sequence Length: 2048
- Batch Size: 4096
- Learning Rate: 3e-4
- cosine schedule, 0.1 weight decay

# Linly-Chinese-LLaMA

语料包括 [CLUECorpusSmall、中英文翻译数据、News Commentary v13](https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE) 和[中文科学文献数据 CSL](https://github.com/ydli-ai/CSL)。

基于[TencentPretrain](https://github.com/Tencent/TencentPretrain)框架

```bash
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json --enable_zero3 \
                      --pretrained_model_path models/llama-7b.bin \
                      --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path models/llama_zh_7b \
                      --world_size 8 --data_processor lm  --deepspeed_checkpoint_activations \
                      --total_steps 300000 --save_checkpoint_steps 5000 --batch_size 24
```



# SFT

数据集：

1.BELLE: 150万数据，175个指令seed
2.pCLUE: 120万训练数据，73个Prompt
3.CSL: 40万中文论文元数据，26个Prompt
5.GuanacoDataset: 多语言指令数据集
6.Chain-of-Thought: 中英文思维链数据
7.news_commentary: 中英文翻译数据
8.firefly: 23个中文NLP任务集合
9.[Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT/tree/main)

