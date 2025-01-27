---
title: OpenAI-operator
created: 2025-01-26
tags:
  - agent
---
## operator-吴翼

惊喜点
- 出现广告会点掉
- 计算

思维链相比o1较短，回溯更多发生在动作上。

基座模型和o1/o3应该是不一样的。

如何得到一个operator
1. 需要一个很好的多模态基座模型
2. 需要高质量的数据集，比如如何做到出现广告点掉
3. 强化学习，需要scale up，需要现实环境交互，高效训练

2025年是Agent年，因为现在基座模型已经很好了，各项技术也比较成熟。

OpenAI最早（2016年）就是做web agent的。但是当时没有足够好的基座模型。

2016年是强化学习鼎盛的时代。

如何平衡自主决策和人类指令的优先级：
- 有一些rule-based的，比如点击的时候下单，做个分类。安全团队做。

如何整合语言和动作等不同模态的信息
- 有一个好的多模态基座模型（4o？）
- 动作是文本交互的，特殊格式比如think, action

operator能不能支持和其他agent交互
- 短时间内不会出现多个agent交互的情况
- 完成一件事情，一个有长文本、记忆能力的多模态大模型足够完成
- 被动触发有可能，比如去豆包查个东西，就需要和豆包交互

operator可以进行长期规划
- 好的基座模型，复杂的测试环境、好的人类数据、强化学习激发了模型的长期规划能力

chatbot并不能帮助提升智能，闲聊不太有智能。用户反馈不能提供智能，只能提供产品提升。
operator不一样，用户query比较复杂的。用户反馈是能够提升智能的。

得先有好的智能，才能带动产品。如果希望产品带动智能，是不对的。

Openai的研究团队负责提升智能，产品的posttraining团队负责用户数据对齐。

4o不是一个非常大的模型

## 参考资料

OpenAI应该花了很多精力在post-train这个模型，但pre-train可能做的很不充分或者直接没做（直接拿常规的预训练模型初始化）。visual grounding做的很好，但总体操作水平below college level。文章链接： https://tinyurl.com/4xp5ms5s 






