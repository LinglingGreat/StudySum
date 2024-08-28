---
title: LearnFromFeedback
created: 2024-07-20
tags:
  - 数据选择
type: 论文
papername: Learning from Naturally Occurring Feedback
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - MIT
---

## 论文基本信息

标题：Learning from Naturally Occurring Feedback

作者：

链接：

Code and data: 
https://github.com/shachardon/naturally_occurring_feedback, 
https://huggingface.co/datasets/shachardon/naturally_occurring_feedback

框架图：

![](img/Pasted%20image%2020240720164212.png)

## 背景

这篇论文提出了一种可扩展的方法，用于提取用户在与聊天模型互动时自然产生的反馈，并利用这些反馈进行模型训练。论文的主要动机是解决以下问题：

1. **数据收集成本高昂**：传统的模型训练过程中，获取人类反馈数据通常需要昂贵的人力成本，这限制了数据收集的可扩展性。
    
2. **自然反馈的优势**：之前的研究显示，使用自然产生的（而非自动生成的）反馈数据在定性上具有优势，例如减少幻觉（hallucinations）和偏见，并且更容易解释和验证。
    
3. **模型与人类偏好的对齐**：通过使用自然产生的反馈，可以更好地训练语言模型，使其与人类的偏好更加一致。
    

论文通过手动标注对话数据来确认标准语料库中自然产生的反馈的存在，并发现约30%的聊天包含了明确的反馈。作者进一步应用他们的方法在超过100万次的对话中提取了数十万个反馈样本，并通过训练展示了使用这些提取的反馈可以显著提高模型性能，从而证明了他们方法的有效性。

## 相关研究




## 核心亮点
Feedback Taxonomy

![](img/Pasted%20image%2020240720165630.png)

![](img/Pasted%20image%2020240720165638.png)

![](img/Pasted%20image%2020240720165705.png)


作者开发了一种自动化方法，利用大型语言模型（LLM）(Mixtral-8x7B) 来识别和分类对话中包含反馈的文本片段。这通过给定的提示（prompt）来实现，模型被训练识别特定的反馈模式。将自动化提取方法应用于超过100万次的对话中，以获取大量的反馈样本。

使用提取的反馈数据来训练语言模型，并通过与预训练模型的比较来评估性能提升。这包括使用正面样本进行微调，以及使用包括正面和负面样本的KTO偏好训练。

prompt：

```
There are five different patterns in user responses subsequent to errors in assistant utterances:

Repeat or Rephrase (UR1) - The user repeats or  rephrases their concern, e.g., Actually, I wanted...  
Make Aware with Correction (UR2) - The user makes the system aware of the error and provides information to address what is missing or wrong in its utterance, e.g., No. I wanted you to...  
Make Aware without Correction (UR3) - The user makes the system aware of the error without providing additional information, e.g., You’re wrong.  
Ask for Clarification (UR4) - The user asks for  clarification, e.g., Are you sure? Is it really that...  Positive Feedback (UR5) - The user confirms that the assistant did a good job by directly saying so or thanking it, e.g., Thank you

Given these guidelines, please recognize such user responses in the following dialogue. Please use the format: 
{
"User Response Pattern": [Insert User Response Pattern], 
"User Response Text": [Insert User Response Text]
}  
If there is no feedback, use the following format: 
{
"User Response Pattern": "No Feedback", 
"User Response Text": "" 
}
```


## 实验




## 未来方向



## 主要收获


## 参考资料
