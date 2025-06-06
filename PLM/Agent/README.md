
## 参考资料
[使用LLM构建AI Agents的正确姿势！ChatGPT作者博客全面总结](https://mp.weixin.qq.com/s/eCcDG6XCZFP0tUpAi37HCg)

[聊聊LLM Agents的现状，问题和方向](https://zhuanlan.zhihu.com/p/679177488)

[台大李宏毅2025 AI Agent新课来了！](https://mp.weixin.qq.com/s/d5FnSATz3tPfCOu2a53uKQ)

有人研究过使用语言模型要怎么样比较有效，有一个发现就是与其告诉语言模型不要做什么，不如告诉他要做什么。如果你希望它文章写短一点，你要直接跟它说写短一点，不要告诉它不要写太长。让它不要写太长，它不一定听得懂，叫它写短一点，比较直接，它反而比较听得懂。这也符合这个Streambench的发现——负面的例子比较没有效。与其给语言模型告诉他什么做错了，不如告诉他怎么做是对的。

你可以让文字模型使用工具，可以告诉它这边有一堆跟语音相关的工具，有语音辨识的工具，这个语音侦测的工具，有情绪辨识的工具，有各式各样的工具。可能需要写一些描述告诉他每一个工具是做什么用的，把这些资料都丢给ChatGPT，然后他就自己写一段程序，在这些程序里面他想办法去调用这些工具，他调用了语音辨识的工具，调用了语者验证的工具，调用了这个sum classification的工具，调用了emotion recognition的工具，最后还调用了一个语言模型，然后得到最终的答案。这个答案其实是蛮精确的，这个方法其实有非常好的效果。

所以当你有很多工具的时候，你可以采取一个跟我们刚才前一段讲AI agent memory非常类似的做法，你就把工具的说明通通存到AI agent的memory里面，打造一个工具选择的模块。这个工具选择模块跟RAG其实也大差不差，这个工具选择模块就根据现在的状态去memory的工具包选出合适的工具。语言模型真的在决定下一个行为的时候只根据被选择出来的工具的说明跟现在的状况去决定接下来的行为。

符合你的直觉，外部的知识如果跟模型本身的信念差距越大，模型就越不容易相信。如果跟本身的信念差距比较小，模型就比较容易相信。这是个很符合直觉的答案。同一篇文章的另外一个发现就是，模型本身对它目前自己信念的信心，也会影响它会不会被外部的信息所动摇。有一些方法可以计算模型现在给出答案的信心，如果他的信心低，他就容易被动摇。如果他的信心高，他就比较不会被动摇。这都是非常直觉的结果。

如果这两篇文章答案不同，一篇是AI写的，一篇是人类写的，现在这些语言模型都倾向于相信AI的话。而且那个AI不需要是他自己，Claude就可能比较相信ChatGPT的话，ChatGPT比较相信Gemini的话，他们比较相信AI同类的话，比较不相信人类的话。

语言模型比较相信新的文章。当两篇文章的论点有冲突的时候，他相信比较晚发表的文章。那我们也做了一些其他实验，比如说文章的来源，跟他说这是Wikipedia的文章，或跟他说这是某个论坛上面摘取下来的资讯，会不会影响他的判断。我们发现文章的来源对于语言模型是比较没有影响的。还有另外一个有趣的实验，是我们尝试说今天这篇文章呈现的方式会不会影响语言模型的决策。所谓的呈现方式指的是说，你这篇文章放在网页上，做很好不好看。一样的内容，但是如果你只是做一个非常简单的模板和做一个比较好看的模板，会不会影响语言模型的判断呢？我们用的是那种可以直接看图的语音模型，所以要直接看这一个画面去决定他要不要相信这篇文章的内容。直接看这一个画面，决定他要不要相信文章的内容。我们的发现是模型喜欢好看的模板。我们发现Claude 3比较喜欢好看的模板(只比较了两个模板，可能不是好看，可能是颜色之类的原因）.

