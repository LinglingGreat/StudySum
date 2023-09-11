  
### [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

- 【扩词表】在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中针对一代LLaMA模型的32K词表扩展了中文字词（LLaMA：49953，Alpaca：49954），二期项目中**重新设计了新词表**（大小：55296），进一步提升了中文字词的覆盖程度，同时统一了LLaMA/Alpaca的词表，避免了因混用词表带来的问题，以期进一步提升模型对中文文本的编解码效率。
    
- 【continue pretraining】120G的文本数据+lora继续训练。
    
- 一期项目的介绍： https://mp.weixin.qq.com/s/-Zei1OsM45BHc41WNGmZQQ
    

### [BELLE](https://github.com/LianjiaTech/BELLE)

- 用self-instruct的方法收集中文数据，在Bloom和llama上微调（开源了数据，代码和模型）
	- BelleGroup/BELLE-LLaMA-13B-2M-enc：在LLaMA基础上用2M的中文数据+50k的英文数据finetune得到的
	- BELLE-LLaMA-EXT-7B：加上中文词表后一共是79458个token的词表，在3.4B的中文词汇上继续预训练。
	- [BELLE-LLaMA-EXT-13B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B)：基于sentencepiece的BPE, 12M行的中文文本上训练，设置为50K tokens。合并词表大小79458，在3.4B的中文词汇上继续预训练(冻结其它参数)。然后在4M的高质量instruction数据上全参数finetune
  

### [LinkSoul-AI/Chinese-Llama-2-7b](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b)

- 【扩词表】没有做
    
- 【指令微调】作者的主要贡献是通过instruction tuning的方法使模型具备了很好的指令跟随的能力，同时通过大量的中文指令数据的训练，模型的中文能力得到的很好的提高。和Alpaca，Vicuna等工作不同，作者使用将近1000w条指令数据进行指令微调，将提高中文能力和指令微调两个任务统一完成。而且1000w条指令数据全部开源！仔细查看开源指令数据集可以看到作者合并了大量开源的指令微调数据集，同时对格式做了很好的处理。而且仔细看代码，会发现作者用了System Message的方法进行微调。微调时中英数据集比例保持了差不多1:1的关系。这些细节上都可以看出作者对指令微调有很深的理解，工作做的非常细致，值得所有其他做指令微调的团队学习。
    

### [Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese)

- 【扩词表】Atom系列模型基于数百G的中文文本，在该模型词表的基础上扩展词库至65,000个单词。扩大了中文字符集的覆盖范围，包括所有emoji符号😊。
    
- 【continue pretraining】原子大模型Atom在Llama2的基础上，采用大规模的中文数据进行持续预训练，包含百科、书籍、博客、新闻、公告、小说、金融数据、法律数据、医疗数据、代码数据、专业论文数据、中文自然语言处理竞赛数据集等。同时对庞大的数据进行了过滤、打分、去重，筛选出超过1T token的高质量中文数据，持续不断加入训练迭代中。
    

### [Linly](https://github.com/CVI-SZU/Linly)

- **中文基础模型 Chinese-LLaMA (1-2)、Chinese-Falcon：**以 LLaMA 和 Falcon 为底座，使用中文和中英平行语料进行增量预训练，将其在英文上的语言能力扩展到中文上。目前训练了5B tokens，还在持续迭代
    
    - 【扩词表】Linly-LLaMA-2 中直接扩充了 8076 个常用汉字和标点符号，在模型 embedding 和 target 层使用这些汉字在原始词表中对应 tokens 位置的均值作为初始化。
        
    - 【continue pretraining】参考https://zhuanlan.zhihu.com/p/645103186
        
        - 无监督语料包括中文百科、科学文献、社区问答、新闻等通用语料，提供中文世界知识；英文语料包含 SlimPajama、RefinedWeb 等数据集，用于平衡训练数据分布，避免模型遗忘已有的知识；以及中英文平行语料，用于对齐中英文模型的表示，将英文模型学到的知识快速迁移到中文上。
            
        - 有监督数据包括基于self-instruction构建的指令数据集，例如 BELLE、Alpaca、Baize、InstructionWild 等；使用人工 prompt 构建的数据例如 FLAN、COIG、Firefly、pCLUE 等。
            
        - 在训练阶段，使用Alpaca格式作为指令的分隔符，将所有数据随机打乱，全参数微调模型。此外，使用课程学习（Curriculum learning）训练策略，在训练对初期使用更多英文语料和平行语料，随着训练步数增加逐步提升中文数据对比例，为模型训练提供平缓的学习路径，有助于收敛稳定性。训练基于 TencentPretrain 预训练框架，使用 5e-5 学习率、cosine scheduler、2048 序列长度、512 batch size、BF16 精度，用 deepspeed zero2 进行训练。
            
    - 【指令微调】
        
- 从头训练的 **[Linly-OpenLLaMA](https://github.com/CVI-SZU/Linly/wiki/Linly-OpenLLaMA)** 模型，包含 **3B、7B、13B** 规模，在 1TB 中英文语料上进行预训练，针对中文优化了字词结合tokenizer。第一阶段100G语料。
    
    - 【词表】1. 首先在大规模中英文语料上训练 SPM，词表大小为 50000。2. 根据[结巴分词](https://github.com/fxsjy/jieba)词频前20000的词表扩充中文词，并扩充简繁体汉字。扩充后词表大小为 66242。
        

### [Yayi](https://github.com/wenge-research/YaYi)

- 没有扩充词表，直接基于llama微调
    
- 雅意大模型基于中科闻歌百万级高质量领域指令微调数据集训练而来，训练数据覆盖媒体宣传、舆情分析、公共安全、金融风控、城市治理等五大领域，上百种自然语言指令任务。开源 5w 条训练数据集，可在 [Huggingface 数据仓库](https://huggingface.co/wenge-research) 下载。
    

### [Panda](https://github.com/dandelionsllm/pandallm)
- 'Panda: 海外中文开源大语言模型，基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练，使用了接近15M条数据（维基百科、新闻语料、百科问答、社区问答、翻译语料、COIG），并针对推理能力在中文benchmark上进行了评测’ dandelionsllm

### [OpenBuddy](https://github.com/OpenBuddy/OpenBuddy)
- 【OpenBuddy：一款强大的开源多语言聊天机器人模型，目标是全球用户，重点是对话AI和流畅的多语言支持，包括英文、中文等多种语言。基于Facebook的LLAMA模型，进行了微调，包括扩展词汇表、增加常用字符和增强的token embeddings。通过这些改进和多轮对话数据集，OpenBuddy提供了一个强大的模型，能回答问题并在各种语言之间进行翻译任务。OpenBuddy的使命是提供一个免费、开放且可离线使用的AI模型，该模型可以在用户的设备上运行，无论他们的语言或文化背景如何。目前，OpenBuddy-13B的演示版本可以在Discord服务器上找到。其关键功能包括多语言对话AI(包括中文、英文、日文、韩文、法文等)、增强的词汇表和对常见CJK字符的支持，以及两种模型版本：7B和13B】
	- 词表37498

### [BiLLa](https://github.com/Neutralzz/BiLLa)
- 【BiLLa: 开源的中英双语LLaMA模型，具有增强的推理能力。通过扩充中文词表和利用任务型数据进行训练，提升了中文理解和推理能力。在评测中，BiLLa在中英语言建模和推理任务上表现出色，优于其他模型，并与ChatGLM-6B相比在解题和代码得分方面更高。开发者可以使用BiLLa-7B-LLM和BiLLa-7B-SFT模型，并可通过提供的工具进行模型权重的还原和使用。评测结果显示，BiLLa在语言建模和各种问题类型上取得了良好的性能】
	- 有扩充词表预训练，用的colossalai。词表大小46943
	- 第一阶段：扩充中文词表，使用中文预训练语料[Wudao](https://www.sciencedirect.com/science/article/pii/S2666651021000152)、英文预训练语料[PILE](https://arxiv.org/abs/2101.00027)、翻译语料[WMT](https://www.statmt.org/wmt22/translation-task.html)的中英数据进行二次预训练。
	- 第二阶段：训练数据在第一阶段基础上增加任务型数据，训练过程中两部分数据保持1:1的比例混合。任务型数据均为NLP各任务的主流开源数据，包含有数学解题、阅读理解、开放域问答、摘要、代码生成等，利用ChatGPT API为数据标签生成解析，用于训练提升模型对任务求解逻辑的理解。
	- 第三阶段：保留第二阶段任务型数据，并转化为对话格式，增加其他指令数据（如[Dolly 2.0](https://github.com/databrickslabs/dolly)、[Alpaca GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)、[COIG](https://huggingface.co/datasets/BAAI/COIG)等），进行对齐阶段的微调。

### [Ziya-LLaMA](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
- 从LLaMA-13B开始重新构建中文词表，进行千亿token量级的已知的最大规模继续预训练，使模型具备原生中文能力。再经过500万条多任务样本的有监督微调（SFT）和综合人类反馈训练（RM+PPO+HFFT+COHFT+RBRS)，进一步激发和加强各种AI任务能力。
	- 增加了7000+个常见中文字。合并词表大小39410。110 B tokens训练。原始数据包含英文和中文，其中英文数据来自openwebtext、Books、Wikipedia和Code，中文数据来自清洗后的悟道数据集、自建的中文数据集。在对原始数据进行去重、模型打分、数据分桶、规则过滤、敏感主题过滤和数据评估后，最终得到125B tokens的有效数据。

### [CaMA](https://github.com/zjunlp/CaMA)

【CaMA: 一种支持中英语言的LLaMA模型，通过全量预训练和指令微调提高了中文理解能力、知识储备和指令理解能力】

### 参考资料

[Llama2 7B中文魔改PK：「雅意」百万指令集微调 VS「伶荔」扩词+增量预训练+指令微调](https://mp.weixin.qq.com/s/t-N9hRm7x1B3joN1WiMabQ)


