鸭嘴兽（Platypus 2-70B）: [波士顿大学「鸭嘴兽-70B」登顶Hugging Face大模型排行榜！高效数据集+独特LoRA微调是关键](https://mp.weixin.qq.com/s/RED36cGaqrhOOC5SGD9buw)
- **1. 编辑数据集：删除相似和重复的问题**
- **2. 使用LoRA和PEFT对模型进行了优化，重点关注非注意力模块**


[大模型微调技术​报告汇总](www.zhihu.com/question/607397171/answer/3148846973)
- 汇总了10多个在预训练模型上微调的报告，值得一看
- 提供任务描述、统一使用一种语言、任务指令多样化都有助于提升模型泛化性。（from [EcomGPT：指令微调的电商领域大模型](https://mp.weixin.qq.com/s/pT89cpjrRC7nmChEQmTm6A)）

[再看23个医疗领域微调大模型集合：兼看CareLlama医疗模型的一些实践经验与开放医疗数据](https://mp.weixin.qq.com/s/c6aPU2FALAaa4LWKQ8W1uA)
- 医疗领域微调模型有哪些？做了汇总
- CareLlama的一些实验结论
	- 在CareLlama中并未对分词模型进行中文分词的添加和重新训练，但是效果依旧表现可喜；
	- 全流程的LLM训练包括：预训练、监督微调、奖励模型、强化学习，多数情况下监督微调即可满足自身需求；
	- 在算力充足情况下推荐使用医疗数据和通用语料数据进行训练，这样模型既可以有医学上的训练学习，也可以保持通用能力（如指令遵循）；
	- 不要指望一个医疗LLM就可以满足所有需求，合理的做法可能是实时更新的知识库+微调的医疗LLM（如ChatLaw）；
	- BLOOMZ模型系列使用了PILE语料库进行训练，该语料库包含各种医学文本，包括PubMed Central和PubMed Abstracts等。这些宝贵的文本极大地丰富了BLOOMZ模型的医学知识体系，所以很多开源项目都会优先选择BLOOMZ做医学微调的底座模型；

Self-Alignment with Instruction Backtranslation：[有趣的大模型微调指令数据反向增强方法：Instruction Backtranslation原理解读及其在数据分析上的思考](https://mp.weixin.qq.com/s/LbJiDoVHls7Nuwd9jP6wTQ)
- 其大致思想在于：**首先在少量种子数据和给定网络语料库的基础上对语言模型进行微调。种子模型通过生成网络文档的指令提示（自我增强）来构建训练示例，然后从这些候选示例中选择高质量的示例（自我固化），然后利用生成的数据对更强大的模型进行微调。**


[如何自动筛选高质量的指令微调数据喂给大模型？](https://mp.weixin.qq.com/s/YDIEhGdAejy4CSvN11PAXA)

[大模型微调技巧 | 高质量指令数据筛选方法-MoDS](https://mp.weixin.qq.com/s/G4zqS_hOGpLZF4m_aQzxmg)

