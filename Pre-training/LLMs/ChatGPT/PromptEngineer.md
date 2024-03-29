
Guidelines  
- 在复杂任务上，给AI多一点时间：把一个复杂问题分解成多个步骤的简单问题，让AI分步骤解决，而不是一次性提出。  
- 如果AI输出有错误，把错误返回给AI，让AI自己反思错误的内容，通常能得到正确的答案。  
  
Iterative  
- 没有最好的prompt，只有根据自己的需求不断完善prompt：当你觉得你的prompt不work时，要分析可能的原因，尤其是有没有给出足够清楚的指示；修改后再次提交，并根据返回的结果再次迭代。  
- 可以对输出进行精确限定：例如长度可以限制到句子、单次和字符数。  
  
Summarizing  
- 让GPT对文字内容进行分析时带有特定关注点，更关注数据还是更关注叙事。  
- 可以输出特定的list项，即只总结你指定的内容。  
- 还可以以html表格或者jason格式输出  
  
Inferring  
- 大语言模型能够很好地代替一些传统NLP模型的功能，例如情绪分析，内容提取，主题判断等。  
- 而且使用起来更为灵活，不需外另外训练，可以用自然语言的形式描述任务。  
  
Transforming  
- 可以指定GPT回复特定信息，然后将这个特定信息组合到预制好的text格式中，这样可以用更稳定的形式输出，而且节省token。  
- 可以方便地将文字在不同表现格式之间进行转换，例如从jason转换成html。  
- 可以比较修改前后的区别 by python redlines  
- 可以指定以某种学术写作格式输出，例如：  
- proofread and correct this review. Make it more compelling.  
- Ensure it follows APA style guide and targets an advanced reader.  
  
Expanding  
- 根据需要回答的情况给出尽可能详细的指导，要达到所需的详细程度通常需要用不断迭代修改的方式来完成。  
- Temperature参数与GPT回复的随机性和多样性成正比。  
- 如果想要确定和稳定的输出用温度0 （同样的输入会总是会得到同样的输出）  
- 如果希望更有发散性、创造性的输出可以用温度0.7  
  
Chatbot  
- 在API调用中可以分配不同的角色：系统，用户，助手（GPT）  
  
- 当你的描述足够详细时，就可以得到一个能够完成特定任务的机器人。（教程中的案例是一个自动接待点餐并生成小票记录的聊天机器人）