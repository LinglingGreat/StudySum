---
title: CharacterGLM
created: 2024-05-08
tags:
  - roleplay
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution:
---

## 论文基本信息

标题：

作者：

链接：https://arxiv.org/abs/2311.16832

代码：https://github.com/thu-coai/CharacterGLM-6B

框架图：


## 背景



## 相关研究




## 核心亮点

将人的语言表达特征的重点落实在属性和行为上：属性主要影响语言表达的内容，行为则影响语言表达的风格和口吻。

- 属性：CharacterGLM的设计主要考虑了七种属性，包括身份、兴趣、观点、经历、成就、社交关系和其他。

![](img/Pasted%20image%2020240509094655.png)

- 行为：行为主要由一些动态的元素组成：语言特征、情感表达和互动模式。例如，老年人更倾向于使用一些更正式的语言，而青少年则更喜欢用网络流行语。CharacterGLM则主要考虑了语言学特征(including a person’s catchphrase, dialect, stylistic features, frequently used words and sentences, etc.) 和性格作为行为(such as gentleness and coldness)方面的设计。


一个对话式的AI角色要想证明自己是一个栩栩如生的角色，需要具备真实的人所具备的表达特质。团队主要关注三个方面的表达特质：一致性、拟人化和吸引力。

- 一致性：角色一致性是角色在交互期间展现稳定的属性和行为的能力。维持一个会话式AI角色在对话中属性和行为的一致对于赢得用户的满足和信任是至关重要的。

- 拟人化：角色拟人化要求角色在与用户的交互中表现自然，类似人与人之间的自然交互。类人的会话式AI角色对于提高用户的接受度以及促进更自然和有吸引力的对话是不可或缺的。

- 吸引力：吸引力是会话式AI角色引起用户兴趣以及促进用户参与的衡量依据。聊天过程中，让对话有趣，让人想聊下去会直接影响用户的体验，这也是对话模型整体性能的一个体现。


## 实验

**收集数据**, 4个类别的人物：celebrities, daily life, games & videos, and virtual love

- 人类角色扮演：雇佣了大量的众包工作者两两配对，一方扮演角色另一方“玩家”，两人自由地选定对话主题进而展开对话。
    
- 大语言模型合成：使用GPT-4生成含有角色描述和对话的合成数据，并人工对合成数据中书面语对话进行了口语化的改写。
    
- 文学作品提取：人工从剧本、小说等文学作品中提取仅包含两方的对话及两方的角色描述。
    
- 人机交互：使用上面三种类型的数据训练完初版的模型后，雇佣深度用户，采用人机交互的方式收集人与CharacterGLM的多轮交互数据。

![](img/Pasted%20image%2020240509101321.png)
  
**训练**

- 角色prompt设计：众包工作者将数据中的角色描述形式化为流畅的自然语言描述作为模型训练的角色prompt，同时考虑总结、复述和风格化改写的角色prompt增广方式，利用Claude-2来合成多样的角色prompt。
    
- 有监督的微调：使用6B到66B参数的ChatGLM作为基座模型，将角色prompt和对话拼接在一起进行有监督的微调。
    
- 自我完善：将上面的“人机交互”数据引入有监督的微调中，促进模型的迭代式自我完善。

除了一致性（Consistency）、拟人化（Human-likeness）和吸引力（Engagement）之外，团队使用：（1）质量（Quality）来评估回复的流畅度和上下文连贯性，（2）安全性（Safety）衡量回复是否符合道德标准，（3）正确性（Correctness）确定回复是否存在幻觉。此外，使用“整体（Overall）”指标来衡量模型回复的整体质量。

团队将 CharacterGLM 与10个中文友好的主流 LLM 进行对比，雇佣了10个标注人员，每个标注人员在11个模型上各创建两个角色，并进行不少于20轮的对话交互。交互完成后，标注人员依据上述6个子维度和整体维度进行1-5分的打分，分值越高表示模型性能越好，最后计算每个模型在各个维度上的平均分。

![](img/Pasted%20image%2020240509101113.png)

![](img/Pasted%20image%2020240509102754.png)

![](img/Pasted%20image%2020240509103020.png)

![](img/Pasted%20image%2020240509103303.png)


## 未来方向



## 主要收获


## 参考资料

[mp.weixin.qq.com/s/pyQUgvaJCWVIMBEzkJPyoA](https://mp.weixin.qq.com/s/pyQUgvaJCWVIMBEzkJPyoA)

