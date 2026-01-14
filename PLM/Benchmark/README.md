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

CoT效果的验证，也可以参考[Flan-PaLM_T5](../Alignment/Flan-PaLM_T5/Flan-PaLM_T5.md)执行MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT的对比

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

[GitHub - onejune2018/Awesome-LLM-Eval: Awesome-LLM-Eval: a curated list of tools, datasets/benchmark, demos, learderboard, papers, docs and models, mainly for Evaluation on LLMs. 一个由工具、基准/数据、演示、排行榜和大模型等组成的精选列表，主要面向大型语言模型评测（例如ChatGPT、LLaMA、GLM、Baichuan等）.](https://github.com/onejune2018/Awesome-LLM-Eval/)

## 长文本

[GitHub - OpenBMB/InfiniteBench: 100k+ Long-Context Benchmark for Large Language Models (paper upcoming)](https://github.com/OpenBMB/InfiniteBench)

[GitHub - THUDM/LongBench: LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench)

## Agent

[GitHub - karthikv792/LLMs-Planning: An extensible benchmark for evaluating large language models on planning](https://github.com/karthikv792/LLMs-Planning)

# 关于评估-Anthropic

[Demystifying evals for AI agents \\ Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

为什么要评估？

问题往往出现在用户反馈代理在更改后体验更差时，而团队却只能“盲目摸索”，除了猜测和反复检查之外别无他法。缺乏评估机制，调试只能被动进行：等待用户反馈，手动复现问题，修复错误，然后祈祷没有其他回归问题。团队无法区分真正的回归问题和无关信息，无法在发布前针对数百种场景自动测试更改，也无法衡量改进效果。

我们已经多次见证了这种发展进程。例如，Claude Code 最初是基于 Anthropic 员工和外部用户的反馈进行快速迭代开发的。之后，我们增加了评估环节——最初针对简洁性和文件编辑等具体方面，后来扩展到过度设计等更复杂的行为。这些评估有助于发现问题、指导改进，并聚焦研发与产品之间的合作。结合生产监控、A/B 测试、用户研究等手段，评估结果能够为 Claude Code 的持续改进提供信号，助力其规模化发展。

在代理生命周期的任何阶段，编写评估报告都非常有用。早期阶段，评估报告能促使产品团队明确代理的成功标准；后期阶段，评估报告则有助于维持一致的质量标准。

[Descript](https://www.descript.com/) 的智能体帮助用户编辑视频，因此他们围绕成功的编辑工作流程的三个维度构建了评估体系：不破坏功能、执行指令、高质量地完成任务。他们从人工评分发展到使用 LLM 评分器，评分标准由产品团队定义，并定期进行人工校准。现在， [他们](https://bolt.new/)定期运行两套独立的测试套件，分别用于质量基准测试和回归测试。Bolt AI 团队在拥有一个广泛使用的智能体之后才开始构建评估体系。他们仅用了 3 个月就构建了一个评估系统，该系统运行他们的智能体并使用静态分析对输出进行评分，使用浏览器智能体测试应用程序，并采用 LLM 评判器来评估诸如指令执行等行为。

评估用例在智能体开发的初期尤为重要，它可以明确地编码预期行为。两位工程师阅读同一份初始规范后，可能会对人工智能如何处理极端情况产生不同的理解。评估用例套件可以消除这种歧义。无论何时创建，评估用例都能帮助加速开发。

评估结果还会影响你采用新模型的速度。当更强大的模型出现时，没有评估结果的团队需要花费数周时间进行测试，而拥有评估结果的竞争对手则可以迅速确定模型的优势，调整提示信息，并在几天内完成升级。

一旦评估系统建立起来，您就能免费获得基准测试和回归测试：延迟、令牌使用量、单项任务成本和错误率都可以在一个静态任务库中进行跟踪。评估系统还可以成为产品团队和研究团队之间带宽最高的沟通渠道，定义研究人员可以据此进行优化的指标。显然，评估系统的好处远不止于跟踪回归和改进。由于成本是前期可见的，而收益是后期积累的，因此评估系统的累积价值很容易被忽视。

## 📌 为什么要构建代理评估（Evals）

- 代理系统比传统模型更复杂：它们可能连续多轮调用工具、在环境中修改状态，并根据中间结果调整策略。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 没有评估，团队容易在开发中“盲目飞行”：只能靠上线后用户反馈来修复问题，修复一个又可能引入另一个。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 评估帮助开发团队：
    
    - 明确什么是成功行为；
        
    - 自动化捕获错误；
        
    - 在升级模型时快速判断改进或回退；
        
    - 在早期就定义成功标准，避免产品含糊不清。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
        

---

## 🧱 评估结构与基本定义

评估由多个组件构成：([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

- **任务（Task）**：单个测试，包括输入和明确的成功标准。
    
- **试验（Trial）**：一次任务运行，因为代理表现不确定，通常要重复多次才能稳定评估。
    
- **打分器（Grader）**：具体的逻辑或机制，用来判断代理是否成功完成任务。
    
- **记录（Transcript）**：一次试验的完整交互记录，包含所有请求、工具调用等。
    
- **结果（Outcome）**：试验结束时环境的最终状态（不仅看输出文本）。
    
- **评估框架（Evaluation Harness）**：运行整个评估流程的基础设施。
    
- **代理框架（Agent Harness）**：让模型实际作为一个代理运行的系统（例如工具调用编码等）。
    
- **评估套件（Suite）**：一组具有共同目标的多个任务。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🛠️ 评估的主要用途

1. **发现回归和质量下降**：及时发现模型或代码更改引入的问题。
    
2. **引导改进**：明确哪些能力需要提升。
    
3. **帮助新模型集成**：有评估的团队可以迅速识别新版模型的优势与弱点。
    
4. **定义长期质量标准**：评估成为持续监测品质的工具。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 📊 两类评估：能力 vs 回归

- **能力评估（Capability / Quality Evals）**
    
    - 衡量代理能够做什么；
        
    - 针对模型弱点设计；
        
    - 初期通过较低的通过率来驱动提升。
        
- **回归评估（Regression Evals）**
    
    - 检查代理是否仍然保留之前的能力；
        
    - 目标是 **接近 100% 通过率**，以防止性能倒退。
        
- 随着能力提升，某些通过率高的能力评估可以变成回归评估来持续监控。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🧪 针对不同类型代理的评估方法

### 💻 编码代理（Coding Agents）

- 使用确定性测试（例如运行测试套件、静态分析）衡量正确性。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 可以结合多种评估器：单元测试、代码质量规则等。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 例子包括：SWE-Bench、Terminal-Bench 等。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

对于编码代理来说，确定性评分器是天然的选择，因为软件的评估通常比较直接：代码能否运行，测试能否通过？两个广泛使用的编码代理基准测试工具 [SWE-bench Verified](https://www.swebench.com/SWE-bench/) 和 [Terminal-Bench](https://www.tbench.ai/) 都采用了这种方法。SWE-bench Verified 会向代理提供来自热门 Python 代码库的 GitHub 问题，并通过运行测试套件来评估解决方案；只有当解决方案修复了失败的测试且不破坏现有测试时，才能通过评估。在短短一年内，LLM 在该评估中的得分就从 40% 提升到了 80%。Terminal-Bench 则采用了不同的方法：它测试端到端的技术任务，例如从源代码构建 Linux 内核或训练机器学习模型。

一旦你拥有了一套用于验证编码任务关键_结果_的合格/不合格测试，通常也需要对代码进行评分 _。_ 例如，基于启发式的代码质量规则可以基于除通过测试之外的其他因素来评估生成的代码，而带有清晰评分标准的基于模型的评分器可以评估诸如智能体如何调用工具或如何与用户交互等行为。

### 💬 会话型代理（Conversational Agents）

- 除了功能正确性，还要评估对话质量（例如是否具有同理心）。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 通常结合状态检查和自然语言评分。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

**对话式智能体**在支持、销售或辅导等领域与用户互动。与传统聊天机器人不同，它们会在对话过程中维护状态、使用工具并采取行动。虽然编码和研究型智能体也可能涉及与用户的多次交互，但对话式智能体面临着一个独特的挑战：交互本身的质量也是评估内容的一部分。对对话式智能体进行有效评估通常依赖于可验证的最终状态结果和评估标准，这些标准既能反映任务完成情况，又能反映交互质量。与其他大多数评估方法不同，对话式智能体通常需要第二个逻辑逻辑模型（LLM）来模拟用户。我们在[对齐审计智能体](https://alignment.anthropic.com/2025/automated-auditing/)中使用了这种方法，通过扩展的对抗性对话来对模型进行压力测试。

对话代理的成功可以从多个维度来衡量：问题是否已解决（状态检查）、是否在 30 回合内完成（文本记录限制）以及语气是否恰当（LLM 评分标准）？ [𝜏-Bench](https://arxiv.org/abs/2406.12045) 及其后续版本 [τ2-Bench](https://arxiv.org/abs/2506.07982) 是两个融入多维度的基准测试。它们模拟了零售支持和机票预订等领域的多回合交互，其中一个模型扮演用户角色，而代理则处理真实场景。

### 🔍 研究类代理（Research Agents）

- 评估信息检索、覆盖范围、可靠性等复杂能力。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 难以定义唯一正确输出，因此通常用模型作为评估者并校准人类判断。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

**研究代理**收集、整合和分析信息，然后生成答案或报告等输出。与编码代理的单元测试提供二元通过/失败信号不同，研究质量只能根据具体任务来判断。何为“全面”、“来源可靠”甚至“正确”，取决于具体情况：市场调研、收购尽职调查和科学报告都需要不同的标准。

研究评估面临着独特的挑战：专家们可能对综合分析是否全面存在分歧；随着参考内容的不断变化，真实情况也会随之改变；篇幅更长、更开放的输出结果更容易出错。例如，像 [BrowseComp](http://arxiv.org/abs/2504.12516) 这样的基准测试旨在检验人工智能代理能否在开放的互联网中大海捞针般地找到所需信息——这些问题的设计初衷是易于验证但难以解决。

基础性检查用于验证论断是否得到检索到的资料支持；覆盖面检查用于定义一个好的答案必须包含的关键事实；而来源质量检查则用于确认所参考的资料来源是否权威，而不仅仅是检索到的第一个来源。对于有客观正确答案的任务（例如“X 公司第三季度的收入是多少？”），完全匹配即可。LLM（学习逻辑模型）可以标记出缺乏依据的论断和覆盖面上的不足，还可以验证开放式综合分析的连贯性和完整性。

### 🖥️ 计算机操作代理（Computer Use Agents）

- 在真实或沙箱环境中执行鼠标键盘操作评估结果。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 需要检查外部状态（比如是否下单成功，而不仅是页面显示）。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

**计算机用户代理**通过与人类相同的界面（例如屏幕截图、鼠标点击、键盘输入和滚动）与软件交互，而不是通过 API 或代码执行。它们可以使用任何带有图形用户界面 (GUI) 的应用程序，从设计工具到传统的企业软件。评估需要在真实环境或沙盒环境中运行代理，使其能够使用软件应用程序，并检查其是否达到了预期结果。例如， [WebArena](https://arxiv.org/abs/2307.13854) 测试基于浏览器的任务，使用 URL 和页面状态检查来验证代理是否正确导航，并对修改数据的任务进行后端状态验证（确认订单是否实际已下达，而不仅仅是确认页面是否出现）。OSWorld [则将](https://os-world.github.io/)评估范围扩展到完整的操作系统控制，其评估脚本会在任务完成后检查各种组件：文件系统状态、应用程序配置、数据库内容和 UI 元素属性。

浏览器应用代理需要在令牌效率和延迟之间取得平衡。基于 DOM 的交互执行速度快，但会消耗大量令牌；而基于屏幕截图的交互速度较慢，但​​令牌效率更高。例如，当让 Claude 总结维基百科内容时，从 DOM 中提取文本效率更高。在亚马逊上查找新的笔记本电脑保护套时，截屏效率更高（因为提取整个 DOM 会消耗大量令牌）。在我们的 Claude for Chrome 产品中，我们开发了评估机制来检查代理是否针对每个上下文选择了正确的工具。这使我们能够更快、更准确地完成基于浏览器的任务。


## 📏 非确定性行为与评估指标

代理表现会随试验而变化，因此需要统计方法：

- **pass@k**：在 k 次尝试中至少成功一次的概率。
    
- **pass⁽ᵏ⁾**：在 k 次尝试中全部成功的概率。
    

这两个指标适用于不同场景：

- 客户希望至少一次成功 → 更关心 pass@k；
    
- 需要稳定一致表现 → 更关注 pass⁽ᵏ⁾。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🧠 设计高质量评估的步骤和最佳实践

**0：尽早开始**

我们发现，一些团队因为认为需要数百个任务而推迟构建评估模型。实际上，从真实失败案例中提取 20-50 个简单的任务就是一个很好的开始。毕竟，在智能体开发的早期阶段，对系统的每一次更改通常都会产生清晰可见的影响，而这种较大的影响意味着较小的样本量就足够了。更成熟的智能体可能需要更大、更复杂的评估来检测较小的影响，但在初期最好采用 80/20 法则。评估模型构建的难度会随着等待时间的延长而增加。在早期阶段，产品需求自然而然地转化为测试用例。如果等待时间过长，你就只能从运行中的系统中逆向推导成功标准了。

### 1️⃣ 从现有手动测试开始

将日常开发(每次发布前都要验证的行为以及最终用户经常尝试的任务)或用户反馈中的测试(根据用户影响进行优先级排序)转成自动化评估。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

### 2️⃣ 编写清晰且可通过的任务

任务必须明确无歧义，两位专家看到描述应达成一致。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

确保任务质量远比想象中要难。一个好的任务应该能够让两位领域专家独立地得出相同的通过/失败结论。他们自己能通过这项任务吗？如果不能，那么任务就需要改进。任务规范中的模糊之处会成为评估指标中的噪音。同样的道理也适用于基于模型的评分标准：模糊的评分细则会导致判断结果不一致。

对于前沿模型，多次试验的通过率均为 0%（即 0% pass@100）通常表明任务存在缺陷，而非智能体能力不足，这提示您需要仔细检查任务规范和评分器。对于每个任务，创建一个参考解决方案非常有用：一个已知可行且能够通过所有评分器的输出。这可以证明任务是可解决的，并验证评分器的配置是否正确。
### 3️⃣ 设计平衡的数据集

既测试“应该发生”的也测试“不应该发生”的情况。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

挑战在于既要防止模型在不应该搜索的时候进行搜索，又要保证它在适当的时候能够进行广泛的研究。团队构建的评估涵盖了两个方向：模型应该搜索的查询（例如查找天气）和模型应该根据现有知识回答的查询（例如“谁创立了苹果公司？”）。在触发不足（不应该搜索的时候不搜索）和触发过度（不应该搜索的时候搜索）之间找到合适的平衡点非常困难，需要对提示和评估进行多轮改进。

### 4️⃣ 构建稳健的评估框架

确保环境隔离、不会被先前试验状态干扰。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

### 5️⃣ 设计合适的评估器（Graders）

避免过于僵硬地要求固定调用顺序，关注结果而非路径。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))

对于包含多个步骤的任务，应采用部分计分制 **。**

模型评分通常需要反复迭代以验证其准确性。LLM（逻辑推理模型）评分器应与人类专家进行密切校准，以确保模型评分与人类评分之间的差异很小。为避免模型出现错误，应为 LLM 提供退出机制，例如，当信息不足时返回“未知”。此外，还可以创建清晰、结构化的评分标准，分别对任务的每个维度进行评分，然后使用独立的 LLM 评分器对每个维度进行评分，而不是使用同一个 LLM 评分器对所有维度进行评分。一旦系统稳定可靠，只需偶尔进行人工审核即可。

有些评估存在一些不易察觉的故障模式，即使智能体表现良好，也会导致得分偏低，因为智能体由于评分错误、智能体框架限制或歧义而无法完成任务。即使是经验丰富的团队也可能忽略这些问题。例如， [Opus 4.5 最初在 CORE-Bench 测试中得分仅为 42%](https://x.com/sayashk/status/1996334941832089732?s=46&t=c5pEvnVdVbMkcR_rcCHplg) ，直到 Anthropic 的一位研究人员发现了多个问题：评分机制过于僵化，预期得分为“96.124991…”时却被扣分；任务规范含糊不清；以及随机任务难以精确复现。修复错误并使用限制较少的框架后，Opus 4.5 的得分跃升至 95%。类似地， [METR 在其时间范围基准测试中发现了](https://x.com/metr_evals/status/2001473506442375645?s=46)几个配置错误的任务，这些任务要求智能体优化到预设的分数阈值，但评分却要求超过该阈值。这导致像 Claude 这样遵循指令的模型受到惩罚，而忽略既定目标的模型反而获得了更高的分数。仔细核对作业和评分者可以避免这些问题。

**第六步：查看成绩单**

除非您阅读大量试验的记录和成绩，否则您无法了解评分员的工作是否高效。在 Anthropic，我们投资开发了用于查看评估记录的工具，并且我们定期抽出时间阅读这些记录。当任务失败时，记录会告诉您智能体是犯了真正的错误，还是评分员拒绝了有效的解决方案。记录通常还会揭示智能体和评分员行为的关键细节。

失败结果应该公平合理：清楚地说明智能体错在哪里以及错在哪里。当分数没有提升时，我们需要确信这是智能体表现的问题，而不是评估本身的问题。阅读评估记录是验证评估是否真正衡量了关键指标的方法，也是智能体开发的关键技能。



## 📌 常见问题与解决方法

- **评估任务本身有缺陷**：低通过率往往是任务或评分设置有问题，而不是模型本身不能做。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- **代理“作弊”通过评估**：评估要设计得不易通过漏洞化策略完成。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- **评估饱和（saturation）**：代理达到 100% 后失去改进信号，此时需要更新或增加更难任务。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🛠️ 长期维护与实用建议

- 将评估视为长期资产，需要不断维护和更新。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 让产品专家、研究人员共同参与编写和维护评估。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    
- 定期读记录（transcripts），确认评估是否真正衡量了关键能力。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🧩 与其他性能检测方法的结合

评估只是了解代理表现的一部分，还需要结合：

- **生产监控**：观察真实用户行为；
    
- **用户反馈**：直接来自使用体验；
    
- **A/B 测试**：验证重大更改影响；
    
- **系统人工评估**：用于校准 LLM 评分系统。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))
    

---

## 🧠 结论

评估对于代理的成功开发至关重要，不应是事后补充，而要成为开发流程的一部分。评估帮助团队从“感觉变差”转向有数据支持的改进。构建好的评估体系可以大幅提高开发速度、减少错误并确立明确的质量标准。([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents"))


