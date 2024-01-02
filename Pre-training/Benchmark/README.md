## 现有评测汇总
| 评测系统                            | 语言   | 任务范围       | 能力/任务/数据集 |
| ----------------------------------- | ------ | -------------- | ---------------- |
| [GLUE](GLUE/GLUE.md)                | 英语   | 理解           | 1/7/9            |
| [SuperGLUE](SuperGLUE/SuperGLUE.md) | 英语   | 理解           | 2/4/8            |
| [CLUE](CLUE/CLUE.md)                | 中文   | 理解           | 2/6/9            |
| [CUGE](CUGE/CUGE.md)                | 中文   | 理解+生成      | 7/17/19          |
| [MMLU](MMLU/MMLU.md)                | 英文   | 理解           |                  |
| [BIGBench](BigBench/BIGBench.md)    | 多语言 | 理解+生成      | x/200+/x         |
| [XTREME](XTREME/XTREME.md)          | 多语言 | 跨语言迁移能力 | x/9/x            |
| [HELM](HELM/HELM.md)                | 英文   | 理解+生成      |                  |
| [LAMBADA](LAMBADA/LAMBADA.md)       | 英文   | 上下文理解生成 |                  |
| [SuperNaturalInstructions](NaturalInstructions/SuperNaturalInstructions.md)                                    |   多语言     |                |                  |

HELM包括了基本上每个评测方向的数据集，可以在此基础上评测，补充其他评测任务。
- 问答、信息抽取、摘要、情感分析、有毒性检测、文本分类、aspirational场景（文本生成、故事生成等）、语言、知识、推理、危害、效率、校准、鲁棒性

[RACE数据集](https://www.cs.cmu.edu/~glai1/data/race/)来自中国12-18岁之间的初中和高中英语考试阅读理解，包含28,000个短文、接近100,000个问题。包含用于评估学生理解能力的多种多样的主题。

[TyDiQA](https://github.com/google-research-datasets/tydiqa)（问答，跨越多个语种），包括以下任务
- **Passage selection task (SelectP)**：选择答案所在段落
- **Minimal answer span task (MinSpan)**：选择答案的span或者回答yes/no
- **Gold passage task (GoldP)**：给定包含答案的段落，预测答案的连续span

[MGSM](https://github.com/google-research/url-nlp)（多语言的数学应用问题，GSM8K数据集手动翻译成了10种语言）
- 可用来做CoT推理评测

CoT效果的验证，也可以参考[Flan-PaLM_T5](Flan-PaLM_T5/Flan-PaLM_T5.md)执行MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT的对比

MMLU（包含在HELM中）
- 偏向语言理解的知识密集型任务（比如计算机、数学、法律、哲学、医学、经济学、社会学）
- 提供选项，选择答案，衡量指标是准确率
- 可用来做zero-shot和few-shot

BBH
- BIG-Bench的一部分

[XCOPA](https://github.com/cambridgeltl/xcopa)
- 跨语言常识推理
- 任务目标是根据一个问题确定前提和两个选项之间的因果关系。因此，一个成功的模型不仅要执行常识推理，还要将其推理能力推广到新语言。

[XL-WiC](https://pilehvar.github.io/xlwic/)
- 多语言单词上下文语义判断
- 给定同一种语言的两个句子和一个出现在两个句子中的感兴趣的词，模型被询问这个词在句子中是否具有相同的意义。

中文安全prompts，用于评估和提升大模型的安全性' thu-coai GitHub: github.com/thu-coai/Safety-Prompts

【GAOKAO-bench：以中国高考题目作为数据集，评估大语言模型的语言理解能力和逻辑推理能力的测评框架，包含1781道选择题、218道填空题和812道解答题】: github.com/OpenLMLab/GAOKAO-Bench

C-Eval是全面的中文基础模型评估套件，涵盖了52个不同学科的13948个多项选择题，分为四个难度级别： https://github.com/SJTU-LIT/ceval


https://github.com/EleutherAI/lm-evaluation-harness (还没加入上述表格)

## Ecom(电商)

https://github.com/Alibaba-NLP/EcomGPT （基于BLOOMZ指令微调）

|   |   |   |   |
|---|---|---|---|
|**Dataset**|**Lang.**|**Task**|**Metric**|
|Lenove|EN|Named Entity Recognization(实体识别)|F1, Rouge|
|Lenove|EN|Entity Span Detection(实体检测)|Rouge|
|Reddit|EN|Extractive QA(抽取式问答)|Rouge|
|ABSA|EN|Review Topic Classification(评论的话题分类)|F1, Rouge|
|MEPAVE|ZH|Attribute Value Recognization(属性识别)|F1, Rouge|
|MEPAVE|ZH|Attribute Value Detection(属性检测)|Rouge|
|Multi-CPR|ZH|Product Select(选出doc中和query匹配的)|Rouge|
|Multi-CPR|ZH|Product Align(query和doc是否匹配)|F1, Rouge|
|OpenBG|ZH|Title Attritube Matching(是否匹配)|F1, Rouge|
|OpenBG|ZH|Fine-grain Product Classify(item细粒度分类)|F1, Rouge|
|OpenBG|ZH|Coarse-grain Product Classify(item粗粒度分类)|F1, Rouge|
|OpenBG|ZH|Title Generate(基于属性生成title)|Rouge|
|OpenBG|ZH|Item Align(两个item是否指的是同一个产品)||

金融领域的评估[PIXIU](https://github.com/chancefocus/PIXIU) 也是类似的评估方式，情绪分析、NER、问答、分类等。

## 参考资料

[GitHub - GPT-Fathom/GPT-Fathom: GPT-Fathom is an open-source and reproducible LLM evaluation suite, benchmarking 10+ leading open-source and closed-source LLMs as well as OpenAI's earlier models on 20+ curated benchmarks under aligned settings.](https://github.com/GPT-Fathom/GPT-Fathom)
[字节的GPT-Fathom介绍](https://mp.weixin.qq.com/s/-AWkDzAzoyQNmgYXuC6B4w)


【LLM的评测有关的工具、demo、论文、文档】： [GitHub - onejune2018/Awesome-LLM-Eval: Awesome-LLM-Eval: a curated list of tools, demos, papers, docs for Evaluation on Large Language Models like ChatGPT, LLaMA, GLM](https://github.com/onejune2018/Awesome-LLM-Eval)

[如何快速地设计并评估few shot示例的效果：OpenICL上下文示例学习框架推荐及实现源码](https://mp.weixin.qq.com/s/D2Fbhs13IhpsLyCJoWSwGA)

[Anthropic \\ Challenges in evaluating AI systems](https://www.anthropic.com/index/evaluating-ai-systems)

## 长文本

[GitHub - OpenBMB/InfiniteBench: 100k+ Long-Context Benchmark for Large Language Models (paper upcoming)](https://github.com/OpenBMB/InfiniteBench)

[GitHub - THUDM/LongBench: LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench)

## Agent

[GitHub - karthikv792/LLMs-Planning: An extensible benchmark for evaluating large language models on planning](https://github.com/karthikv792/LLMs-Planning)



