---
title: LongtermChat
created: 2022-08-15
tags: [对话/记忆]

---


## LongtermChatExternalSources

来源：[https://community.openai.com/t/solving-goldfish-memory-in-chatbots/19041](https://community.openai.com/t/solving-goldfish-memory-in-chatbots/19041)

找到历史对话中和当前输入最相似的（gpt3的模型得到embedding，计算点积）10句话，和历史对话拼起来（得到convo_block），和prompt一起输入GPT3，获取相关维基百科的title。调用wikipedia包获取页面内容。

再根据title和convo_block和prompt，让GPT3生成一个相关提问，再用问答的prompt让GPT3回答，如果有多个回答的话再让GPT3整合一下。

使用聊天的prompt，提供convo_block和answer（作为HINT），让GPT3生成回复

wikipedia包？

