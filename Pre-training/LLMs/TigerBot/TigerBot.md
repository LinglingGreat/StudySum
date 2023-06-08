---
title: TigerBot
created: 2023-06-08
tags: LLM, 增量预训练, SFT, 数据

---

https://github.com/TigerResearch/TigerBot

## 基本情况

- 模型：TigerBot-7B, TigerBot-7B-base，TigerBot-180B (research version)，
- 代码：基本训练和推理代码，包括双卡推理 180B 模型的量化和推理代码，
- 数据：预训练 100G，从 2TB 过滤后的数据中经过去噪去重清洗而得；监督微调 1G 或 100 万条数据，按比例涵盖用户指令常见的 10 大类 120 小类任务，
- API: chat, plugin, finetune, 让用户能在半小时内无代码的训练和使用专属于自己的大模型和数据，
- 领域数据：涵盖金融，法律，百科，广邀大模型应用开发者，一起打造中国的世界级的应用。

我们在 BLOOM 基础上，在模型架构和算法上做了如下优化：

- 指令完成监督微调的创新算法以获得更好的可学习型(learnability)，
- 运用 ensemble 和 probabilistic modeling 的方法实现更可控的事实性(factuality)和创造性(generativeness)，
- 在并行训练上，我们突破了 deep-speed 等主流框架中若干内存和通信问题，使得在千卡环境下数月无间断，
- 对中文语言的更不规则的分布，从 tokenizer 到训练算法上做了更适合的算法优化。

## 预训练

有验证集，正常的clm训练方式, bf16

```bash
deepspeed \
--include="localhost:0,1,2,3" \
./train_clm.py \
--deepspeed ./ds_config/ds_config_zero3.json \
--model_name_or_path TigerResearch/tigerbot-7b-base \
--dataset_name TigerResearch/dev_pretrain \
--do_train \
--output_dir ./ckpt-clm \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 5 \
--learning_rate 1e-5 \
--evaluation_strategy steps \
--eval_steps 10 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2
```


## 开源数据集

### 预训练数据

基于 GPT3 的 pretrain 的数据分布，采集中文书籍，互联网，和百科类数据，并通过数据源质量分过滤和 tf-idf soft deduping，从 20TB 数据过滤到 2TB，保持语言和类目的比例，并在此基础上随机抽样 100G 数据开源：

- [中文开源预训练集 - 55G，包含中文书籍、中文互联网、中文百科 [hugging face]](https://huggingface.co/datasets/TigerResearch/pretrain_zh)
    
- [英文开源预训练集 - 51G，包含英文书籍、英文互联网、英文百科 [hugging face]](https://huggingface.co/datasets/TigerResearch/pretrain_en)
    
      
    
    | 类型       | 磁盘占用 | 来源 |
    | ---------- | -------- | ---- |
    | 中文书籍   | 12G      | 自研 |
    | 中文互联网 | 25G      | 自研 |
    | 中文百科   | 19G      | 自研 |
    | 英文书籍   | 22G      | 开源 |
    | 英文互联网 | 6.9G     | 开源 |
    | 英文百科   | 22G      | 开源 |
    | **总量**   | **106G** |      |


### 微调数据

#### 数据搜集

- 模型中使用的微调数据的搜集思想如下：
    
    a. 从用户指令的自然分布，人工标注总结 10 大类，120 小类任务，例如，事实性问答，开放式创作，语法分析，代码编辑等；
    
    b. self-instruct: 参考 Alpaca self-instruct 方法，扩充中英文 seed_tasks，增加一些中文习惯种子问题，基于此生成 2M 中文(本次开源 0.5M)及 0.1M 英文(本次开源 50k)；
    
    c. human-labeling: 基于人工写题及答案、网络搜集方式，整理加工问答集数据，在开源列表中标识为[自研]部分，本次开放部分数据；
    
    d. open-source data cleaning: 基于各类公开数据集转换清洗，其中[自研*]部分，表示基于原始数据进行二次开发后得到，[开源]部分数据集一般原始数据即为较规整的问答数据，进行简单清洗得到；
    
    e. 总的数据分布符合用户指令自然分布。
    

#### 数据清洗

- 由于各类数据质量存在差异，通过 Alpaca Self-Instruct 生成的数据亦存在各种问题。因此，我们经过细致的人工校验和分类，总结出一套全面且系统化的数据清洗规则与方法。
    
- 整体规则可以划分为**过滤类规则**和**清洗类规则**两大类。其中，命中过滤规则的数据项将被弃用，而清洗规则旨在处理并保留所需的数据。
    
- 同时，在数据梳理与积累的过程中，我们也不断对清洗规则进行迭代和优化。
    
- 通用清洗规则描述如下所示：
    
    a. 过滤类-敏感词规则：基于积累的敏感词库，清洗丢弃涉政、涉黄、涉暴、涉恐等数据项；
    
    b. 过滤类-无效输入输出：此类规则主要针对 Self-Instruct 生成数据缺陷进行专项清理，根据输入输出分别制定规则，以丢弃一些无效的数据项；
    
    > 无效输入如"<一段文本>"，无效输出如"[图画]"；
    
    c. 清洗类-关键词规则：根据整理的关键词/正则列表进行数据的替换，包括：清理特殊标志位字符、清理非可见字符、清理标签、繁简转换等；
    
    d. 清洗类-特殊逻辑规则：此类规则用于清洗一些特殊现象数据，如指令与输入重复等，如下所示：
    
    > `{"instruction": "描述如何做一道红烧肉。请提供食材和详细的步骤。", "input": "请描述如何做一道红烧肉，提供食材和详细步骤。", ...}`
    

#### 数据开源

- 指令数据集, 当前开源 120W 问答对，磁盘空间 1.1G (数据集开放到 huggingface）
    
    |类型|语言|数据集|数量|来源|
    |---|---|---|---|---|
    |alpaca 中文|中文|[tigerbot-alpaca-zh-0.5m](https://huggingface.co/datasets/TigerResearch/tigerbot-alpaca-zh-0.5m)|0.5m|自研|
    |百科问答|中文|[tigerbot-wiki-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-qa-zh-1k)|1k|自研|
    |名著问答|中文|[tigerbot-book-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-book-qa-1k)|1k|自研|
    |猜谜语|中文|[tigerbot-riddle-qa-1k](https://huggingface.co/datasets/TigerResearch/tigerbot-riddle-qa-1k)|1k|自研|
    |阅读理解|中文|[tigerbot-superclue-c3-zh-5k](https://huggingface.co/datasets/TigerResearch/tigerbot-superclue-c3-zh-5k)|5k|自研*|
    |问答|中文|[tigerbot-HC3-zh-12k](https://huggingface.co/datasets/TigerResearch/tigerbot-HC3-zh-12k)|12k|开源|
    |知乎问答|中文|[tigerbot-zhihu-zh-10k](https://huggingface.co/datasets/TigerResearch/tigerbot-zhihu-zh-10k)|10k|开源|
    |alpaca 英文|英文|[tigerbot-alpaca-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-alpaca-en-50k)|50k|自研|
    |头脑风暴|英文|[tigerbot-dolly-Brainstorming-en-1.7k](https://huggingface.co/datasets/TigerResearch/tigerbot-dolly-Brainstorming-en-1.7k)|1.7k|开源|
    |分类|英文|[tigerbot-dolly-Classification-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-dolly-Classification-en-2k)|2k|开源|
    |数学问题|英文|[tigerbot-gsm-8k-en](https://huggingface.co/datasets/TigerResearch/tigerbot-gsm-8k-en)|8k|开源|
    |代码|英文|[tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k)|2k|自研*|
    |食谱生成|英文|[tigerbot-kaggle-recipes-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-recipes-en-2k)|2k|开源|
    |病历生成|英文|[tigerbot-mt-note-generation-en](https://huggingface.co/datasets/TigerResearch/tigerbot-mt-note-generation-en)|450|开源|
    |多轮对话|英文|[tigerbot-OIG-multichat-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-OIG-multichat-en-50k)|50k|自研*|
    |综合问答|英文|[tigerbot-stackexchange-qa-en-0.5m](https://huggingface.co/datasets/TigerResearch/tigerbot-stackexchange-qa-en-0.5m)|0.5m|开源|
    |wiki 问答|英文|[tigerbot-wiki-qa-bart-en-10k](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-qa-bart-en-10k)|10k|开源|
    |如何做类教程|英文|[tigerbot-youtube-howto-en-50k](https://huggingface.co/datasets/TigerResearch/tigerbot-youtube-howto-en-50k)|50k|开源|
    |**总量**|||**120W 条**||
    

### 领域数据

- 开放金融、法律、百科相关领域数据，作为 rethink 外部数据源
    
    |类型|数量|
    |---|---|
    |[金融-研报](https://huggingface.co/datasets/TigerResearch/tigerbot-research-plugin)|2W 篇|
    |[金融-财报](https://huggingface.co/datasets/TigerResearch/tigerbot-earning-plugin)|2500 篇|
    |[法律](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin)|11 类 5.5W 条款|
    |[百科](https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-plugin)|10W 词条|
    

