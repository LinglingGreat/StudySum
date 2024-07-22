**Training language models to follow instructions with human feedback 2022.3.4**

Motivation：

使用一种通过人类反馈的强化学习 (RLHF)技术,根据用户和API的交互结果，对模型的多个输出进行了排名，然后再利用这些数据微调GPT-3，使得InstructGPT模型在遵循指令方面比GPT-3更好。

Method：

a.训练模型：1.3B GPT3

b.数据集：收集人类标注演示数据集用于训练模型，同时收集了一个多模型生成的标注演示数据集,在更大的API提示集上对多个模型的输出进行比较，用于训练强化学习的Reward。

c.评估方法：使用强化学习Reward对模型输出结果进行评估。

d:训练方法：使用Reward作为奖励函数，来微调GPT-3策略，使用PPO算法最大化这个奖励。

Contribution：

虽然参数少了100倍以上，InstructGPT的输出结果比GPT-3以及用监督学习进行微调的模型都要高得多。与GPT-3相比，InstructGPT产生的错误较少，安全性更高。同时还对模型的提示分布进行了人类评估，结果显示，InstructGPT编造事实的情况较少，而且产生的输出结果更合适。此外，InstructGPT的输出比FLAN和T0的输出要好，提供了提供模型泛化能力新思路。

  

链接：https://zhuanlan.zhihu.com/p/558286175  
