---
title: Agent 与 Harness 入门
created: 2026-09-17
tags: [agent, harness, AgentSDK, ClaudeCode, MCP, 学习笔记]
type: 学习笔记
---

# Agent 与 Harness 入门：名词、机制、实现框架

> 2026-09-17 整理的入门材料。目标读者是做后训练的自己：把 agent / harness 这些词说清楚，理清「模型之外还有什么」，再给一张 2026 年 9 月的框架地图和一条动手路线。全文事实以一手来源为准（Anthropic / OpenAI 工程博客、官方文档、原作者博客），二手文章只用来看趋势。库内相关笔记：[ACE（Agentic Context Engineering）](../ACE/ACE.md)、[通义 DeepResearch](../Tongyi-DeepReSearch/Tongyi-DeepSearch.md)、[OpenAI Operator](../Operator/OpenAI-operator.md)、[Claude Code 记忆机制](../../../Tool/ClaudeCode/Claude-Code记忆机制与跨机迁移.md)。

## 0. 一页结论

- **Agent = Model + Harness。** 模型只会「文本进、文本出」，一次调用没有记忆也不会循环；把它变成能干活的 agent 的，是外面那层软件：反复调模型、执行它要的工具、把结果喂回去、决定何时停。这层软件就是 harness。
- **Harness 这个词 2026 年才火，但东西早就有。** 2023 年叫 scaffold（脚手架），SWE-bench 那批评测叫 evaluation harness，Claude Code / Codex CLI 这类产品就是最典型的 harness。2026 年 2 月 Mitchell Hashimoto 和 OpenAI 先后用「harness engineering」命名之后，它成了行业通用词。
- **同一个词有三层用法，读文章先分清是哪层：**
  1. **运行时 harness**（agent loop、工具执行、权限、上下文压缩、子 agent、会话）：Claude Agent SDK、OpenAI Agents SDK、OpenHands 这类框架卖的就是它；
  2. **用户侧 harness engineering**（AGENTS.md / CLAUDE.md、skills、hooks、linter、CI、评测）：用 agent 的团队围绕自己代码库搭的约束和反馈回路；
  3. **评测 / 训练 harness**（SWE-bench、Terminal-Bench 的 runner，RL 里的 environment）：给模型一个标准化的环境跑轨迹、打分。
- **框架选型的判断只有一条：你需要拥有 loop 的多少控制权。** 先直接调 API 手写 loop（20 行），够用就停；要产品级能力（权限、压缩、子 agent、会话）再上厂商 SDK；只有真的需要跨模型、跨厂商编排时才上 LangGraph 这一类通用框架。
- **对后训练工程师，harness 就是 RL 的 environment。** 训练时用什么 harness 采 rollout、评测时用什么 harness 跑 benchmark、线上用什么 harness 服务用户，三者要么一致、要么明确知道差在哪。模型的「agentic 能力」不是模型自己的属性，是「模型 + 特定 harness」这个组合的属性。

## 1. 术语表（先扫一遍，后面都会展开）

| 术语 | 人话解释 | 来源 |
|---|---|---|
| **Agent** | 模型加上让它「能行动而不只是回答」的一切；本质是让文本生成跑在一个循环里 | HF 术语表 [3] |
| **Agentic** | 形容词，「有 agent 味的」：会自己决定下一步、会调工具、会多轮迭代。Anthropic 用 agentic systems 统称 workflow 和 agent | Anthropic [1] |
| **Workflow vs Agent** | Workflow：LLM 和工具按**预先写死的代码路径**编排；Agent：LLM **自己动态决定**流程和用哪个工具 | Anthropic [1] |
| **Augmented LLM** | 加了检索、工具、记忆三件套的 LLM，是所有 agentic 系统的基本积木 | Anthropic [1] |
| **Agent loop（agentic loop）** | 「调模型 → 模型要求调工具 → 执行工具 → 结果喂回 → 再调模型」的循环，直到模型不再要求调工具 | Claude Agent SDK 文档 [9] |
| **Turn** | loop 里的一个来回：模型输出（含工具调用）+ 工具执行 + 结果回填 | 同上 |
| **Tool use / function calling** | 模型输出一段结构化的「我要调 X 工具，参数是 Y」，由 harness 真正执行；模型自己不执行任何东西 | 各家 API 文档 |
| **Harness** | agent 里的执行层：调模型、处理它的工具调用、决定何时停 | HF 术语表 [3] |
| **Scaffold / scaffolding** | 定义模型行为的那层：system prompt、工具描述、输出怎么解析、跨步骤记什么。和 harness 常混用；HF 的区分是 scaffold 管「模型看到什么」，harness 管「怎么执行」 | HF 术语表 [3] |
| **Runtime（运行时）** | 管 agent「怎么被执行」的那层：状态存哪、断了怎么恢复、暂停与续跑。注意 agent 圈里它还常被用来指沙箱（如 OpenHands 的 docker runtime），见 4.5 | LangChain [13]、OpenHands 文档 |
| **Harness engineering** | Hashimoto 的定义：「只要发现 agent 犯了一个错，就花时间做一个工程化的解决方案，让它永远不再犯这个错」 | Hashimoto [5] |
| **Context engineering** | 管理每一步进入模型上下文窗口的全部 token（system prompt、工具、历史、检索内容）的策略集合；prompt engineering 只是其中写指令的那部分 | Anthropic [6] |
| **Context rot** | 上下文越长，模型从中准确召回信息的能力越差；所有模型都有 | Anthropic [6] |
| **Compaction（压缩）** | 上下文快满时，把早期历史总结成摘要替换掉原文 | Anthropic [6][9] |
| **Subagent（子 agent）** | 被另一个 agent 调起来处理子任务的 agent，有自己的上下文和工具，只把结论交回主 agent | HF 术语表 [3]、Anthropic [7] |
| **Orchestrator-workers** | 主 agent 拆任务、派发给多个 worker、汇总结果的多 agent 模式 | Anthropic [1][7] |
| **MCP（Model Context Protocol）** | 把「工具 / 数据源」标准化成服务器，任何 harness 都能接的开放协议 | 见第 5 节 |
| **Skills** | 打包好的可复用任务知识（一个文件夹 + SKILL.md），描述常驻上下文、正文按需加载 | HF 术语表 [3]、Claude 文档 |
| **Hooks** | harness 生命周期上的回调点（工具执行前后、压缩前、agent 结束时），跑在你的进程里不占上下文 | Claude Agent SDK 文档 [9] |
| **Permission / HITL** | 哪些工具自动放行、哪些要人批、哪些禁止；human-in-the-loop 的落点 | 同上 |
| **Sandbox** | 隔离的执行环境（容器 / VM），agent 的 bash 和文件操作在里面跑 | 各家文档 |
| **Session** | 一次可恢复、可分叉的会话，保存完整历史 | 同上 |
| **Trajectory / rollout** | 一次完整的 agent 运行记录：看到了什么、做了什么、得了多少分 | HF 术语表 [3] |
| **Environment（RL 语境）** | 接收动作、更新内部状态、返回观测的有状态对象；agent 训练里 harness 扮演的就是它 | HF 术语表 [3] |
| **Evaluation harness** | 跑 benchmark 的那套代码：起环境、喂任务、收轨迹、判分 | SWE-bench 等 |

## 2. Agent 到底是什么：Anthropic 的光谱定义

最常被引用、也最不含糊的定义来自 Anthropic 2024-12 的《Building effective agents》[1]（Erik Schluntz、Barry Zhang）：

> **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
> **Agents** are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

两者统称 **agentic systems**，区别只在**谁掌握控制流**：代码写死是 workflow，模型自己决定是 agent。这是个光谱不是二分法，大多数生产系统落在中间。

文章给了五种 workflow 模式，值得记住，因为后来所有框架的「预制组件」基本都是它们的封装：

| 模式 | 一句话 | 典型用途 |
|---|---|---|
| Prompt chaining | 任务拆成固定顺序的几步，每步一次 LLM 调用，中间可以加程序化检查 | 先写大纲再写正文 |
| Routing | 先分类，再送到专门的下游分支 | 客服按问题类型分流；难题送大模型、简单题送小模型 |
| Parallelization | 拆分并行（sectioning）或多次投票（voting）再汇总 | 多个 guardrail 并行检查；多次判分取多数 |
| Orchestrator-workers | 主 LLM 动态拆任务、派给 worker、综合结果 | 改动跨多个文件的编码任务、多源检索 |
| Evaluator-optimizer | 一个生成、一个评价，循环迭代 | 翻译润色、多轮搜索 |

真正的 agent 则是：模型在循环里自己规划、调工具、看环境反馈（工具结果）、决定下一步，直到完成或碰到停止条件。文章的三条建议至今没过时：

1. **先用最简单的方案**，只在明确能提升效果时才增加复杂度。Agentic 系统是用延迟和成本换任务表现。
2. **先直接调 API，别急着上框架。** 很多模式几行代码就能写出来；框架会多一层抽象，把底层 prompt 和响应藏起来，更难 debug，也诱使你堆不必要的复杂度。
3. **工具接口（agent-computer interface, ACI）要像给人写的 UI 一样认真设计。**

Anthropic 的 API 文档里另给了「要不要做 agent」的四个判据，很实用：任务够不够复杂到难以事先写死流程（complexity）、收益是否值得更高的成本和延迟（value）、模型在这类任务上是否真的行（viability）、出错能不能被发现和回滚（cost of error）。任何一条是「否」，就退回 workflow 或单次调用。

## 3. Agent loop 解剖：harness 到底在干什么

### 3.1 最小实现：20 行就是一个 harness

下面是用 Anthropic Python SDK 手写的最小 agent loop。它没有任何框架，但已经包含了 harness 的全部核心职责：调模型、识别工具调用、执行、回填、判停。

```python
import anthropic, json

client = anthropic.Anthropic()
TOOLS = [{
    "name": "run_shell",
    "description": "Run a shell command in the sandbox and return stdout/stderr.",
    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
}]

def run_shell(cmd: str) -> str:  # 真正执行工具的是 harness，不是模型
    import subprocess
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (p.stdout + p.stderr)[-4000:]  # 截断：控制回填进上下文的 token

messages = [{"role": "user", "content": "找出当前目录里最大的三个文件"}]
for turn in range(30):  # max_turns：防跑飞的第一道闸
    resp = client.messages.create(model="claude-opus-5", max_tokens=16000,
                                  tools=TOOLS, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason != "tool_use":  # 模型不再要求调工具 → 结束
        break
    results = []
    for block in resp.content:
        if block.type == "tool_use":
            # 这里是权限 / 审批 / 审计 / 沙箱介入的位置
            out = run_shell(**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
    messages.append({"role": "user", "content": results})  # 所有结果放同一条 user 消息
```

看懂这段就理解了 harness 的边界：**模型只产出「我想调什么工具」这个意图，其余全是 harness 的事**。所有框架的差别只在于把注释里那几处（判停、权限、截断、回填）做得多精细。

### 3.2 产品级 loop：以 Claude Code / Agent SDK 为例

Claude Agent SDK 就是把 Claude Code 的 harness 打包成库（Python / TypeScript）。官方文档 [9] 把 loop 描述为五步：

1. **收 prompt**：连同 system prompt、工具定义、历史一起送给模型；
2. **模型评估并回应**：可能是文本、一个或多个工具调用、或两者都有；
3. **执行工具**：SDK 跑每个工具、收集结果，hooks 可以在执行前拦截、改写或阻止；
4. **循环**：2–3 反复，每个完整来回是一个 turn，直到模型给出不含工具调用的回复；
5. **返回结果**：最终文本 + token 用量 + 成本 + session id。

在最小 loop 之上，产品级 harness 加的东西可以按「解决什么问题」归类：

| 问题 | Harness 机制 | Claude Code / Agent SDK 里的对应物 |
|---|---|---|
| 跑飞、烧钱 | 轮数上限、预算上限 | `max_turns`、`max_budget_usd`，子 agent 的花费计入总预算 |
| 危险操作 | 权限系统：白名单、黑名单、审批回调、模式 | `allowed_tools` / `disallowed_tools`；模式 default / acceptEdits / plan / dontAsk / auto / bypassPermissions；规则可细到 `Bash(npm *)` |
| 想在关键点插自己的逻辑 | Hooks | PreToolUse / PostToolUse / UserPromptSubmit / Stop / SubagentStart / SubagentStop / PreCompact；跑在宿主进程，不占上下文 |
| 上下文爆炸 | 自动压缩 + 上下文规则 | 接近上限时自动 compaction，发 `compact_boundary` 事件；CLAUDE.md 每次都重新注入，所以「持久规则放 CLAUDE.md，别放首条 prompt」 |
| 单个上下文装不下大任务 | 子 agent | `Agent` 工具起子 agent，各自干净上下文，只把结论回传；只读工具可并行，写操作串行 |
| 工具太多挤占上下文 | 工具按需加载 | `ToolSearch`：MCP 工具 schema 默认延迟加载 |
| 跨会话延续 | Session | resume / fork；可把 transcript 镜像到自己的存储 |
| 外部系统 | MCP | 数据库、浏览器、SaaS 通过 MCP server 接入 |
| 可复用的任务知识 | Skills / commands / memory | 从 `.claude/` 和 `~/.claude/` 自动加载 |

Anthropic 在《Building agents with the Claude Agent SDK》[8] 里把这一切的设计原则概括成一句话：

> The key design principle behind the Claude Agent SDK is to give your agents a computer, allowing them to work like humans do.

并把 agent 的工作循环抽象成三段：**gather context → take action → verify work**。收集上下文靠 agentic search（用 grep / tail 这类命令在文件系统里翻，精确、透明、慢）而不是先建向量库；行动靠工具、bash 和「写代码来完成任务」；验证靠规则（lint、测试）、视觉反馈（截图）和 LLM 评判。这三段对应的就是第 4 节里 Böckeler 讲的 guides 和 sensors。

### 3.3 多 agent：什么时候值得、代价多大

Anthropic 2025-06 的《How we built our multi-agent research system》[7] 是少数给了数字的一手材料：

- Orchestrator-workers 架构（lead agent 拆问题、并行起子 agent 搜索、子 agent 当「智能过滤器」只回传要点）在内部研究评测上比单个 Claude Opus 4 高 **90.2%**；
- 代价是 token：agent 大约是聊天的 **4 倍**，多 agent 大约是 **15 倍**。所以只适合「价值高到能覆盖成本、可高度并行、信息量超过单个上下文窗口」的任务；
- 生产上的坑：长时运行的有状态错误会叠加、升级要用 rainbow deployment 避免打断运行中的 agent、非确定性行为必须靠全链路 tracing 才能 debug、主 agent 同步等子 agent 是吞吐瓶颈。

这些经验后来被产品化成了 Claude Code 里的子 agent，以及 2026-06 的「dynamic workflows」[10]：让 Claude 现场写一段 JavaScript 来编排一批子 agent（fan-out、pipeline、锦标赛式对比），文章标题就叫「A harness for every task」，把「harness」用作「为某个任务定制的编排结构」。

## 4. Harness：词从哪来、现在指什么

### 4.1 时间线

| 时间 | 事件 | harness 在这里指什么 |
|---|---|---|
| 早于 LLM | 软件测试里的 test harness | 驱动被测对象、收集结果的那套夹具 |
| 2023–2024 | SWE-bench 等 benchmark 的 evaluation harness；同期 agent 论文多用 scaffold / scaffolding | 起容器、喂任务、收补丁、跑测试判分的代码 |
| 2024-12 | Anthropic《Building effective agents》[1] | 尚未用 harness 一词，用的是 agentic systems / framework |
| 2025-09 | Anthropic 发布 Claude Agent SDK [8]，同日发《Effective context engineering》[6] | 「给 agent 一台电脑」的那层基础设施 |
| 2025-10-25 | LangChain《Agent Frameworks, Runtimes, and Harnesses — oh my!》[13] | 试图区分三层：framework 给抽象、runtime 给持久执行、harness 是「开箱即用」的更高层；作者自己承认边界模糊 |
| 2025-11-07 | Terminal-Bench 2.0 发布，运行器拆成 **Harbor**[15] | 「运行器」从 benchmark 里独立出来，harness 指跑 agent 的那套通用基础设施 |
| 2025-11 | Anthropic《Effective harnesses for long-running agents》[4]（Justin Young） | 明确把 Claude Agent SDK 称为「a powerful, general-purpose agent harness」，并讨论在它之上再搭一层跨会话的 harness |
| 2026-02-05 | Hashimoto《My AI Adoption Journey》[5] | 「harness engineering」作为个人工作方法的命名 |
| 2026-02-11 | OpenAI《Harness engineering: leveraging Codex in an agent-first world》[2] | 团队级方法论：代码库本身就是 agent 的 harness |
| 2026-02-17 | LangChain《Improving Deep Agents with harness engineering》[14] | **模型固定不动，只改 harness，Terminal-Bench 2.0 从 52.8 提到 66.5** |
| 2026-03-10 | LangChain《The Anatomy of an Agent Harness》[13] | 给出最宽的定义：「If you're not the model, you're the harness.」 |
| 2026-04-02 | Böckeler（Thoughtworks）《Harness engineering for coding agent users》[11] | 系统化：inner / outer harness，guides / sensors |
| 2026-04-08 | Anthropic《Scaling Managed Agents》[16] | 收敛为最精确的一句：harness 是「the loop that calls Claude and routes Claude's tool calls」，与 session、sandbox 三分 |
| 2026-05-25 | HF 术语表《Harness, Scaffold, and the AI Agent Terms Worth Getting Right》[3] | 给 harness 和 scaffold 划清边界 |
| 2026 年中 | Wikipedia 建立「Agent harness」词条 [12] | 「agent harness, also known as agent scaffolding, is the software infrastructure surrounding a LLM that enables it to operate as an AI agent」 |

Wikipedia 词条说这个词「2026 年初出现，归属在 Hashimoto 和 LangChain 的 Vivek Trivedy 之间有争议」。准确地说，被争的是「harness engineering」这个提法的流行；「harness」本身早在 SWE-bench 时代就是评测圈的日常用语，Anthropic 2025-11 也已经这么叫 Agent SDK。

### 4.2 三层用法，逐一说清

**（a）运行时 harness = agent 的执行层。** HF 术语表 [3] 的定义最干净：

> Harness: The execution layer inside the agent: it calls the model, handles its tool calls, decides when to stop.
> Scaffolding: The behavior-defining layer around the model: system prompt, tool descriptions, how the model's responses get parsed, what it remembers across steps.

也就是 scaffold 决定模型「看到什么、怎么被解读」，harness 决定「怎么执行」。实践中两者绑在一个产品里（Claude Code 既是 scaffold 也是 harness），所以口语里常混用；Wikipedia 干脆写成同义词。第 3 节讲的就是这一层。

**（b）用户侧 harness engineering = 围绕自己的代码库给 agent 搭约束和反馈。** 这是 2026 年 2 月之后「harness」最热的用法，三个来源说的是同一件事：

- Hashimoto [5]：「anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again.」具体手段就是 AGENTS.md 里记常见错误、给 agent 写专用脚本工具（截图、跑过滤后的测试）。
- OpenAI [2]：三个人（后来七个）五个月、零行手写代码、约一百万行、约 1,500 个 PR。方法论要点：
  - AGENTS.md 只做**目录**（约 100 行），知识库放 `docs/` 当 system of record，「一本大 AGENTS.md」的做法会可预期地失败；
  - 代码库首先为 **agent 的可读性**（legibility）优化：Slack 里对齐的架构决策如果 agent 找不到，就等于不存在；
  - 用 linter 和 CI 机械地强制文档新鲜度、架构分层、「golden principles」，靠后台 agent 定期做 doc-gardening 和清理 PR；
  - 「当 agent 卡住，不是让它 try harder，而是问：缺了什么能力，怎么让它对 agent 既可读又可强制」；
  - 吞吐远超人类注意力时，合并哲学也变了：短命 PR、最少阻塞门禁、纠错便宜、等待昂贵。
- Böckeler [11] 给出最完整的框架：**inner harness**（模型厂商做的，即上面的 (a)）和 **outer harness**（用户搭的）；控制手段分 **guides**（前馈，行动前引导：AGENTS.md、skills、bootstrap 脚本、专用工具）和 **sensors**（反馈，行动后自纠：pre-commit、架构测试、linter、覆盖率、AI review）；执行方式分**计算式**（确定、快、CPU：测试、lint、类型检查）和**推理式**（语义分析、LLM 评审）。她还点明：「engineering a user harness for a coding agent is a specific form of context engineering」。

这一层的核心思想用一句话讲：**把「希望 agent 遵守的东西」从概率性的 prompt 提升为确定性的机制**。写在 prompt 里的「遵守编码规范」是概率合规；一个不合规就挡 PR 的 linter 是确定合规。

**（c）评测 / 训练 harness = 给模型的标准化环境。** SWE-bench、Terminal-Bench 这类 benchmark 的 harness 负责起环境、喂任务、跑模型的 agent loop、收轨迹、判分。它和 (a) 的差别是：(a) 追求把任务做好，(c) 追求可复现、可比较。这也是为什么 benchmark 榜单必须注明用的是哪个 agent harness。

**「换 harness 分数差很多」有硬证据**：LangChain 2026-02 的一篇实验 [14] 把模型固定不动（gpt-5.2-codex），只改 harness，Terminal-Bench 2.0 的成绩从 52.8 提到 66.5——**13.7 个百分点，全部来自模型之外**。SWE-bench 主榜上同一个 Claude 4.5 Opus 配不同 agent 相差约 2.4 个百分点（见第 6.1 节），也是同一现象的弱化版。给后训练的含义见第 7 节。

### 4.3 一张分层图

```
┌──────────────────────────────────────────────────────────────┐
│ Outer harness（用户 / 团队搭）                                  │
│   guides:  AGENTS.md / CLAUDE.md、skills、专用脚本工具、模板     │
│   sensors: linter、类型检查、测试、CI 门禁、AI review、评测集    │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Inner harness（框架 / 厂商做）                              │ │
│ │   agent loop、工具执行、权限与审批、hooks、                  │ │
│ │   上下文管理（压缩 / 工具按需加载）、子 agent、会话、沙箱     │ │
│ │ ┌──────────────────────────────────────────────────────┐ │ │
│ │ │ Scaffold：system prompt、工具描述、输出解析、记忆结构    │ │ │
│ │ │ ┌──────────────────────────────────────────────────┐ │ │ │
│ │ │ │ Model：文本进、文本出（含结构化的工具调用意图）     │ │ │ │
│ │ │ └──────────────────────────────────────────────────┘ │ │ │
│ │ └──────────────────────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
        ↑ 评测 / 训练 harness 从外面包住整套，喂任务、收轨迹、打分
```

### 4.4 Anthropic 的「长程 harness」案例：harness 之上再搭 harness

《Effective harnesses for long-running agents》[4] 是理解「用户侧 harness」最好的实操样本。问题：agent 跨多个上下文窗口干活时，每个新 session 对之前的工作零记忆，于是出现三类失败：过早宣布完成、一口气想做完所有功能留下半成品、没做端到端验证就标记通过。解法不是换模型，而是在 Claude Agent SDK 之上再包一层：

| 组件 | 作用 |
|---|---|
| Initializer agent | 首个 session 专职搭环境：`init.sh`、进度文件、把需求展开成约 200 条带 pass/fail 状态的 feature list（JSON）、首个 git commit |
| Coding agent | 之后每个 session 只做**一个**功能，干净交接 |
| 启动仪式 | 每个 session 开头固定动作：看 `pwd`、读 git log 和进度文件、跑 `init.sh`、先跑基线测试再开新活 |
| Git + 进度文件 | 每次改动都 commit；人类可读的 session 摘要 |
| 端到端验证工具 | 用浏览器自动化（Puppeteer MCP）真去点一遍，而不是信 agent 说「完成了」 |

文章明确说：SDK 自带的 compaction 不足以支撑跨多窗口的生产级项目。这正是 (a) 和 (b) 的分工：厂商把通用能力做进 inner harness，任务特有的状态管理和验证仍要用户自己搭。

### 4.5 顺带分清：framework / runtime / harness / scaffold / sandbox

**runtime（运行时）在普通编程里本有两个含义**：一是时间段，即运行期，和编译期相对（"runtime error" 就是这个意思）；二是一层软件，JVM 之于 Java、Node.js 之于 JavaScript、containerd 之于容器。后者的定义性特征是**在程序执行的全过程中一直在场、管着程序怎么被执行**，而不是像库那样被调一次就返回。

到了 agent 圈，这个词有两个互不相同的所指，读文章必须靠上下文分辨：

| 所指 | 管什么 | 代表 |
|---|---|---|
| **执行状态托管层** | 状态存哪、断点恢复、暂停等人审批、回到任意历史步骤重跑 | LangGraph（LangChain 在 [13] 里就是这么定位它的） |
| **执行环境（约等于沙箱）** | agent 的 bash 和文件操作在哪台机器上跑 | OpenHands 的 docker runtime / 远程 runtime |

分辨判据是看它周围出现什么词：出现 checkpoint、persistence、resume、state、durable 的是前者；出现 docker、sandbox、container、image 的是后者；出现 compile-time、error、exception 的则只是「运行期」这个时间概念，和 agent 无关。

按 LangChain [13] 的划分，这一组词的分工是：

| 词 | 管什么 | 代表 |
|---|---|---|
| framework | 给抽象和积木 | LangChain |
| runtime | 管执行：状态、持久化、恢复 | LangGraph |
| harness | 管 loop：调模型、路由工具调用、判停 | Claude Agent SDK |
| scaffold | 管模型看到什么：system prompt、工具描述 | 各家 prompt 设计 |
| sandbox | 管代码在哪跑 | Docker、E2B |

**别把这张表当标准**，各家根本不遵守：Anthropic 直接把 loop 叫 harness、从不叫 runtime；OpenHands 把沙箱叫 runtime；LangChain 自己在那篇文章里也承认边界模糊、这些术语并非其原创。靠上下文判断比背定义可靠。

**为什么偏偏 agent 需要 runtime 这一层**：普通函数调用毫秒级返回、无状态、不用人插手，所以不需要。agent 三条全反过来——跑几分钟到几小时所以中途会断、带着历史和文件所以有状态、常常要人批准所以得能停在任意一点。LangGraph 把 agent 建模成状态图，正是因为状态图天然能序列化、能存 checkpoint、能从任意节点恢复。对训练线而言，runtime 这层直接决定**采 rollout 能不能断点续跑、能不能分片并行**（见第 7 节）。

## 5. 实现框架全景（2026-09）

> 数据口径：stars / 最后提交 / 许可证取自 GitHub API，读取日期 **2026-09-17**；能力与定位取自各项目官方文档或 README。star 数每天在变，看**量级和相对位置**即可。表里的「状态」除非项目自己声明，否则只按最后提交日期描述事实，不替它下「已死」的判断。

### 5.1 选型速判

先回答一个问题：**你要拥有 loop 的多少控制权？**

| 你的处境 | 选什么 |
|---|---|
| 学习、原型、想看清每一层 | 直接调 API 手写 loop（第 3.1 节那 20 行） |
| 要做编码 / 文件系统类 agent，想直接拿到产品级 harness | Claude Agent SDK（Python / TS）；不想自己管沙箱和会话就用 Managed Agents |
| 要做多 agent 协作、handoff、guardrail，且已在 OpenAI 生态 | OpenAI Agents SDK |
| 需要把 agent 编成**确定的图**、要 checkpoint / 断点续跑 / 人审 | LangGraph（要开箱即用就叠 deepagents） |
| 已在 Google Cloud，要部署托管 | Google ADK |
| 要把 agent 嵌进 TypeScript 前端 / 全栈应用 | Vercel AI SDK 或 Mastra |
| 做研究、训练、要一个不污染结论的极简 harness | mini-swe-agent |
| 要跑评测或采 RL 轨迹 | Harbor（Terminal-Bench 的运行器） |

### 5.2 A 组：厂商官方 Agent SDK

| 名称 | 维护方 | 许可 | stars | 最后提交 | 核心抽象 | 内置 harness 能力 | 状态 |
|---|---|---|---|---|---|---|---|
| **Claude Agent SDK** | Anthropic | 用途受 Anthropic 商业条款约束（Python 包 MIT） | 8.1k (py) / 1.8k (ts) | 2026-09-17 | Query / Options / Tool / Subagent / Hook / Session | agent loop、内置工具（Read/Write/Edit/Bash/Glob/Grep/WebSearch）、权限 6 模式、7 类 hooks、自动压缩、子 agent、会话 resume/fork、MCP、Skills、轮数与预算上限 | 活跃 |
| **OpenAI Agents SDK** | OpenAI | MIT | 29.5k (py) / 3.8k (js) | 2026-09-16 | Agent / Handoff / Guardrail / Session / Tracing | agent loop、handoff 委派、输入输出 guardrail、会话记忆（SQLAlchemy/Redis/MongoDB）、内置 tracing、MCP、人工审批 | 活跃 |
| **Google ADK** | Google | Apache-2.0 | 21.6k (py) | 2026-09-17 | LlmAgent / Sequential·Parallel·Loop Agent / Runner / Session·State·Memory / Tools / Callbacks / Events | agent loop、三种预制 workflow agent、会话与状态、记忆、callbacks、A2A、内置评测、图工作流（2.0） | 活跃，ADK 2.0 已 GA |
| **Anthropic Managed Agents** | Anthropic | 托管服务（无公开仓库） | — | — | Agent（持久化、带版本）/ Session / Environment / Vault / Deployment | Anthropic 同时托管 **loop 和每会话沙箱**：工具执行、权限策略、事件流、outcome 判定、定时触发、多 agent、密钥金库 | 公测中 |

**Claude Agent SDK** 就是把 Claude Code 拆出来的库，能力清单见第 3.2 节的表，是目前「inner harness」做得最厚的一家。只有 Python 和 TypeScript；其他语言官方建议把 CLI 当子进程跑（`-p` + `--output-format json`）。

**OpenAI Agents SDK** 走的是另一条路：抽象极少，官方自述是「very few abstractions」。最有特色的是 **handoff**——把「转交给另一个 agent」建模成一个工具调用，于是多 agent 协作不需要额外的编排层。它 provider-agnostic，可通过 LiteLLM 接非 OpenAI 模型。

**Google ADK** 是四家里唯一把 workflow 和 agent 做成**同一套类型**的：`SequentialAgent` / `ParallelAgent` / `LoopAgent` 是「确定性编排」，`LlmAgent` 是「模型自主」，两者能互相嵌套。这正好对应第 2 节 Anthropic 那条光谱。支持 Python / TypeScript / Go / Java / Kotlin 五种语言。

**Managed Agents** 是唯一「harness + 部署」都托管的：Anthropic 自己在工程博客里把它叫 **meta-harness**，并给出了目前最精确的 harness 定义——「the loop that calls Claude and routes Claude's tool calls」，与 session（事件日志）、sandbox（执行环境）三者分开。适合长跑、定时、不想自己维护沙箱的场景。

### 5.3 B 组：通用编排框架

| 名称 | 维护方 | 许可 | stars | 最后提交 | 核心抽象 | 内置 harness 能力 | 状态 |
|---|---|---|---|---|---|---|---|
| **LangGraph** | LangChain | MIT | 41.8k | 2026-09-17 | Graph / Node / Edge / State（把 agent 建模成状态机） | 持久化 checkpoint、断点续跑、HITL 中断、time-travel、流式、子图 | 活跃 |
| **deepagents** | LangChain | MIT | 29.5k | 2026-09-17 | 在 LangGraph 上的一套 middleware | 自述 "The batteries-included agent harness"：规划/todo 工具、文件系统工具、子 agent、上下文管理 middleware（长会话摘要、工具输出卸载到磁盘）；可插拔后端（本地/沙箱/远程） | 活跃 |
| **LangChain** | LangChain | MIT | 146.5k | 2026-09-16 | Model / Tool / Chain 等基础抽象 | 模型与工具的统一封装层，本身不提供 loop 之外的运行时 | 活跃 |
| **CrewAI** | CrewAI Inc. | MIT | 58.7k | 2026-09-17 | Crew / Agent(role·goal·backstory) / Task / Process，外加 Flows | 记忆、知识、工具、checkpoint、异步执行、HITL、tracing；企业版 AMP 另收费 | 活跃 |
| **Microsoft Agent Framework** | Microsoft | MIT | 13.6k | 2026-09-17 | Agent / Workflow（图：顺序·并发·交接·群聊）/ middleware | checkpoint、流式、HITL、time-travel、OpenTelemetry、DevUI、YAML 声明式 agent | 活跃 |
| **AutoGen** | Microsoft + 社区 | 仓库 LICENSE 为 CC-BY-4.0 | 61.0k | **2026-04-15** | Core（事件驱动消息）/ AgentChat / Extensions 三层 | 多 agent 会话、群聊编排、代码执行 | **维护模式**：README 明示「will not receive new features」，官方建议新项目用 Agent Framework |
| **AG2** | AG2 社区（AutoGen 原班分支） | Apache-2.0 | 4.9k | 2026-09-17 | 承袭 AutoGen 的 ConversableAgent / GroupChat | 同上，另加自有路线 | 活跃，自述 "formerly AutoGen" |
| **smolagents** | Hugging Face | Apache-2.0 | 29.4k | 2026-08-25 | CodeAgent / ToolCallingAgent | **让 agent 用 Python 代码表达动作**（官方称比 JSON 方案少约 30% 步骤）；沙箱可选 E2B / Docker / Modal / WASM；模型无关、Hub 共享工具 | 活跃 |
| **Pydantic AI** | Pydantic | MIT | 20.0k | 2026-09-17 | Agent / Tool / 类型化依赖注入 | 结构化输出与校验是卖点；工具执行、流式、多模型 | 活跃 |
| **Mastra** | Mastra | Apache-2.0，`ee/` 目录为商业授权（open-core） | 28.1k | 2026-09-17 | Agent / Workflow / Tool / Memory（TypeScript） | 工作流、记忆、RAG、评测、部署 | 活跃 |
| **Vercel AI SDK** | Vercel | Apache-2.0 | 26.9k | 2026-09-17 | 统一的 `generateText` / `streamText` / tool 接口 | 跨 provider 的统一调用与流式 UI 集成，偏「模型接入层」而非完整 harness | 活跃 |
| **LlamaIndex** | LlamaIndex | MIT | 52.2k | 2026-09-15 | 文档处理 + Workflow / Agent | 索引与检索是主业，agent 是其上的一层 | 活跃 |

**LangGraph 的定位要说清楚**：它不是「更强的 agent」，而是**给 agent 一个可靠的运行时**。把流程建成状态图之后，你就白拿了 checkpoint、断点续跑、随时插人审批、回到任意历史状态重跑。代价是你得先把流程画出来——这恰恰是「越像 workflow 越好用、越像开放式 agent 越难画」。所以 LangChain 又做了 **deepagents**：在 LangGraph 上预置一套 Claude Code 风格的默认配置，仓库描述直接写「The batteries-included agent harness」。

**AutoGen / AG2 / Agent Framework 三者的关系**要特别注意，这是 2026 年最容易踩的信息陷阱：微软的 AutoGen 已在 README 里明示进入**维护模式**、不再加新功能，最后一次提交停在 2026-04-15；社区分支 AG2 独立演进；微软自己的新主线是 **Microsoft Agent Framework**（.NET + Python + Go，融合了 AutoGen 与 Semantic Kernel 的经验并提供迁移路径）。看到 2025 年的「AutoGen 教程」，先查它说的是哪一个。

**smolagents 值得单独看一眼**，因为它体现了一种不同的动作表达方式：不用 JSON 工具调用，而是让模型直接写 Python 代码，工具就是代码里的函数。好处是循环、条件、组合天然可表达（一段代码顶多次工具调用），坏处是必须有沙箱。这个思路和 Anthropic 的「programmatic tool calling」、「让 agent 写代码而不是堆工具调用」是同一个方向。

### 5.4 C 组：开源编码 Agent / CLI（harness 的典型形态）

| 名称 | 维护方 | 许可 | stars | 最后提交 | 特点 | 状态 |
|---|---|---|---|---|---|---|
| **Claude Code** | Anthropic | 仓库未附 LICENSE（主要作为 issue 跟踪与文档） | 145.6k | 2026-09-17 | 终端 + IDE + 桌面；CLAUDE.md、skills、hooks、子 agent、权限模式、MCP、plugins；其 harness 已作为 Agent SDK 开放 | 活跃 |
| **OpenCode** | anomalyco（**已从 sst 迁出**） | MIT | **208.0k** | 2026-09-17 | star 最高的开源编码 agent；provider 无关；内置 build / plan 两个 agent（plan 为只读），另有 general 子 agent | 活跃 |
| **OpenAI Codex CLI** | OpenAI | Apache-2.0 | 124.8k | 2026-09-17 | 本地运行的编码 agent；AGENTS.md 生态的发起者之一 | 活跃 |
| **Gemini CLI** | Google | Apache-2.0 | 107.0k | 2026-09-17 | 免费额度 60 次/分、1000 次/天；GEMINI.md、MCP、extensions、checkpointing、sandbox；仅 Gemini 模型 | 活跃 |
| **Cline** | Cline | Apache-2.0 | 68.5k | 2026-09-17 | 自述为「SDK / IDE 扩展 / CLI」三形态 | 活跃 |
| **OpenHands** | OpenHands（**已从 All-Hands-AI 迁出**） | MIT | 88.2k | 2026-09-17 | 平台型：CLI / GUI / 云；agent 能力拆到独立的 `software-agent-sdk`；Docker 与远程 runtime 双轨；模型无关 | 活跃 |
| **Aider** | Aider-AI | Apache-2.0 | 49.0k | **2026-05-22** | 终端结对编程的开创者之一，git 集成做得最早 | 最后提交距今约 4 个月 |
| **SWE-agent** | 普林斯顿 / 斯坦福 | MIT | 20.3k | 2026-09-14 | 研究型：可实验不同工具集与历史处理器（ACI 概念的来源） | 活跃 |
| **mini-swe-agent** | 同上 | MIT | 7.7k | 2026-09-14 | **100 行、只有 bash 一个工具、不用 tool-calling 接口、完全线性历史、每步 `subprocess.run` 独立执行**；自述 SWE-bench Verified >74% | 活跃 |

**mini-swe-agent 是学习和做训练的人最该读的一份代码。** 它的设计目标就是「让分数反映模型而不是 agent 框架」，官方明确说这样能避免微调和 RL 训练时对特定框架过拟合。SWE-bench 官网的「Bash Only」榜单就建立在它之上。一百行读完，你对第 3.1 节那 20 行的理解会立刻变具体。

**顺带看一眼这批仓库的 star 量级**：OpenCode 20.8 万、Claude Code 14.6 万、Codex 12.5 万、Gemini CLI 10.7 万。2026 年「harness」这个词能火，直接原因就是这类产品的用户量级已经追平甚至超过了模型本身的讨论度。

### 5.5 D 组：协议与标准

| 名称 | 归属 | 许可 | stars | 当前版本 | 解决什么 |
|---|---|---|---|---|---|
| **MCP**（Model Context Protocol） | 社区治理（仓库 modelcontextprotocol） | 正从 MIT 转为 Apache-2.0（文档 CC-BY-4.0） | 9.2k（规范）/ 24.3k（Python SDK） | 规范 **2026-07-28** | agent ↔ 工具/数据源。JSON-RPC 2.0；Host / Client / Server 三角色 |
| **A2A**（Agent2Agent） | Google 捐赠，**Linux Foundation** 治理（TSC 含 AWS、Cisco、Google、IBM、Microsoft、Salesforce、SAP、ServiceNow） | Apache-2.0 | 25.8k | **v1.0** | agent ↔ agent。Agent Card 声明能力、Task 生命周期、流式与异步 |
| **Agent Skills** | Anthropic 发起后开放 | Apache-2.0 | 25.4k | 持续演进 | 把可复用的任务知识打包成文件夹（`SKILL.md` + 脚本 + 参考资料） |
| **AGENTS.md** | 多方共建，现由 **Linux Foundation 下的 Agentic AI Foundation** 托管 | MIT | 24.4k | — | 给编码 agent 的仓库级说明书，自述已被 **6 万多个开源项目**采用 |

**MCP 的能力划分**值得记住，因为它决定了你能把什么东西标准化：服务端提供 **Tools**（模型可执行的函数）、**Resources**（供人或模型使用的数据）、**Prompts**（给用户的模板化工作流）；客户端提供 **Elicitation**（服务端反过来向用户要信息）。2026-07-28 版还定义了可选扩展：**Tasks**（长任务异步执行与轮询）、**Skills over MCP**、**MCP Apps**（在对话里内嵌图表、表单等交互 UI）。

**MCP 与 A2A 不是竞争关系**，A2A 官网自己写得很清楚：「MCP is for agent-to-tool communication; A2A is for agent-to-agent communication. They are highly complementary.」

**Agent Skills 的机制是 progressive disclosure（渐进披露）三级**：启动时只加载每个 skill 的 name 和 description；任务匹配上了才把 `SKILL.md` 正文读进上下文；执行时再按需加载附带的脚本和参考文件。这样几十个 skill 常驻也只占很小的上下文，是「工具太多挤占上下文」这个问题的另一种解法。采用方已覆盖 Claude Code、Codex、Gemini CLI、Cursor、GitHub Copilot、VS Code、OpenHands、OpenCode、Goose 等几十个产品。

**AGENTS.md vs CLAUDE.md**：两者作用相同（给 agent 的仓库说明书），AGENTS.md 是跨厂商的开放格式。注意 OpenAI 那篇 harness 博客的教训——**别把它写成百科全书，要写成目录**，正文放 `docs/` 并用 CI 保证不过期。

## 6. 评测基准现状

> 数据读取日期 **2026-09-17**。所有 agentic 榜单都必须**模型 + harness 成对**引用，单说模型名的数字没有意义——这正是本文第 4 节的核心论点，而榜单自己的表结构就是最好的证据。

### 6.1 SWE-bench：真实 GitHub issue 修复

评什么：给一个真实仓库的 issue，agent 产出补丁，跑仓库自己的测试判定是否解决。指标是 **% Resolved**。变体包括 Verified（人工校验过的 500 题子集，事实上的主榜）、Multilingual（42 个仓库、9 种语言）、Multimodal、Lite、Full。

榜单的表头就是 `Model | Agent | % Resolved | Avg. $`，**模型和 harness 是分开的两列**。当前主榜前列（读取日期 2026-09-17）：

| 排名 | 模型 | Agent（harness） | % Resolved | 提交日期 |
|---|---|---|---|---|
| 1 | Claude 4.5 Opus | Sonar Foundation Agent | 79.20 | 2025-12-05 |
| 2 | Claude 4.5 Opus medium | live-SWE-agent | 79.20 | 2025-12-15 |
| 3 | Doubao-Seed-Code | TRAE | 78.80 | 2025-09-28 |
| 4 | Gemini 3 Pro Preview | live-SWE-agent | 77.40 | 2025-11-20 |
| 7 | Claude 4.5 Opus high | mini-SWE-agent | 76.80（$0.75） | 2026-02 |

注意第 7 名：同一个 Claude 4.5 Opus，换成极简的 mini-SWE-agent 就是 76.80。**同模型换 harness 的差距，和榜单前十名之间的差距是同一个量级。** 这也是 SWE-bench 官方维护「Bash Only」这条对照线的原因——统一用 mini-swe-agent 跑，让分数尽量反映模型本身。这条线上的成本差异同样惊人：MiniMax M2.5 high 拿到 75.80 只花 $0.07/题，而 Claude 4.5 Opus 的 76.80 花 $0.75/题，**十倍价差换一个百分点**。

顺带一提，SWE-bench 自己的代码库里，「harness」一词至今仍指**评测端的 Docker 打分流水线**（`swebench.harness.run_evaluation`），与生成补丁的 agent 无关。同一个站点上两种词义并存，读的时候要靠上下文分辨。

### 6.2 Terminal-Bench：终端里的复杂任务

评什么：在真实终端环境里完成复杂任务（编译、调试、系统配置、数据处理等），容器化执行、脚本判定。当前已迭代到 **4.0**。榜单同样是 `MODEL | AGENT` 两列，并给出 **95% 置信区间**、token 数和总成本。

| 排名 | 模型 | Agent（harness） | 解决率 | 日期 |
|---|---|---|---|---|
| 1 | GPT-6 Astra (max) | Codex | 58.2% ± 2.8% | 2026-09-03 |
| 2 | Fable 5.1 (max) | Claude Code | 57.9% ± 3.8% | 2026-09-01 |
| 3 | Opus 5 (max) | Claude Code | 51.8% ± 3.4% | 2026-07-24 |
| 4 | Fable 5 (max) | Claude Code | 44.5% ± 3.8% | 2026-06-09 |
| 5 | GLM-5.3 (max) | Claude Code | 41.8% ± 3.2% | 2026-08-14 |

三点值得注意。一是**绝对分数比 SWE-bench 低得多**（不到 60% vs 接近 80%），说明开放式终端任务比「改一个仓库让测试通过」难一个层级。二是榜单给了置信区间，第一名和第二名的区间重叠，**严格说来分不出高下**——这正是我们自己做评测时反复强调的事。三是第 5 名 GLM-5.3 用的 harness 是 Claude Code：**harness 和模型可以自由组合**，国产模型挂在别家的 harness 上跑榜是常态。

Terminal-Bench 的运行器已经拆成独立项目 **Harbor**（5.3k★，Apache-2.0，harbor-framework 组织），自述目标是「running any agent with any model on any task in any sandbox」。拆分理由有三条，每条对训练线都直接相关：容器评测慢、需要横向扩到数千容器；不只要评测，还要能采 **SFT / RL 的 rollout**；agent 框架和 benchmark 都太多，需要一个通用接口。它的核心概念是 Task（指令 + 沙箱 + 判定器）、Agent、Sandbox、Verifier、Trial、Job、Trajectory——**这套词汇和 RL 环境是一一对应的**。

### 6.3 其他常见基准（简述）

| 基准 | 评什么 | 现状 |
|---|---|---|
| **τ-bench 系列** | 客服 agent：给定 policy 和工具，与**模拟用户**多轮交互完成任务。领域含航司、零售、电信、银行 | Sierra 维护，已演进到 **τ³-bench**：加了知识检索域和全双工语音评测，并修了 75+ 条任务错误。注意 **v1.0.1 起分数与旧版不可比** |
| **OSWorld** | 真实操作系统桌面里的 GUI 任务（多模态、看屏幕点鼠标） | XLang 维护，2025-07 发布 **OSWorld-Verified** 修正了社区报告的一批问题 |
| **GAIA / GAIA2** | 通用助手能力：需要多步工具使用和检索的问题 | 本次未核实榜单现状 |
| **BrowseComp** | 浏览器上的难检索任务 | 收录在 OpenAI 的 simple-evals 里，本次未核实榜单现状 |

一条通用提醒：**benchmark 修订会让历史分数失效**（τ-bench 的 v1.0.1、OSWorld-Verified、SWE-bench Verified 都是例子）。引用别人的数字前先确认版本，自己报数字时先写清楚版本。

## 7. 对后训练工程师意味着什么

HF 那篇术语表 [3] 之所以值得读，是它把 agent 圈和 RL 圈的词对上了：

| Agent 圈 | RL 圈 | 说明 |
|---|---|---|
| harness + 沙箱 + 工具 | environment | 接收动作、更新状态、返回观测的有状态对象 |
| 一次完整的 agent 运行 | rollout / trajectory / episode | 看到了什么、做了什么、得了多少分 |
| 验证器 / 测试 / LLM 评判 | reward | 告诉训练算法模型是否在变好 |
| 模型 | policy | 给定观测输出动作 |

由此推出几条对训练线直接有用的判断：

1. **Agentic 能力是「模型 × harness」的联合属性。** 换 harness 分数会变，所以：训练采 rollout 用的 harness、评测跑 benchmark 用的 harness、线上服务用户的 harness，三者要么一致，要么明确知道差在哪。这和我们做角色扮演评测时「replay 用的 prompt 拼装 / 兜底 / 注入要和线上一致」是同一条原则，只是 agent 场景里 harness 更厚、差异更大。
2. **设计 harness 就是设计训练环境。** 工具接口、观测截断、错误信息格式、停止条件，这些在 harness 里的「工程细节」，到了训练里就是 observation space 和 reward shaping。极简 harness（bash 唯一工具、无特殊解析）之所以在训练圈受欢迎，是因为它让模型学到的能力更少依赖某个特定 scaffold。
3. **模型和 harness 在共同演化。** 各家前沿模型都是针对自家 harness 做的后训练（Claude 之于 Claude Code、GPT 之于 Codex），所以「哪个模型编码最强」这个问题脱离 harness 没有答案。反过来，harness 里的一些机制（compaction、tool search、子 agent）也在被模型「学会配合」。
4. **数据合成的主战场在 harness 里。** Agentic SFT / RL 的轨迹要靠 harness 大规模生成，harness 的并行度、可恢复性、判分器的可靠性直接决定数据产能。能否「断点续跑」「分片并行」「判分可复现」是评价一个训练用 harness 的硬指标。
5. **Harness 注入会污染归因。** 排查线上模型「出戏」等问题时，先排除 harness 层（兜底文案、安全拦截、system 注入）再怀疑模型，这条我们已经在实践里踩过。

## 8. 学习路线（按动手顺序）

1. **读**（两小时）：Anthropic《Building effective agents》[1] → HF 术语表 [3] → Böckeler [11]。读完就有了词汇和地图。
2. **写**（半天）：把第 3.1 节的 20 行 loop 跑起来，加上：工具输出截断、`max_turns`、一个「删文件前要确认」的权限检查、一个把每轮 messages 落盘的 hook。这时你已经手写了一个 harness 的 60%。
3. **拆**（一天）：读一个极简开源 harness 的源码（见第 5 节推荐），对照第 3.2 节的表格，看它把每个问题落在哪几行。
4. **用**（一天）：用 Claude Agent SDK 重做第 2 步的任务，体会 permission mode、hooks、子 agent、compaction 分别替你省了什么。
5. **测**（半天）：挑一个 benchmark 的 harness（Terminal-Bench 或 SWE-bench Verified 的子集）跑通一个模型，观察「同一模型换 harness 分数怎么变」。
6. **做自己的 outer harness**（持续）：在自己常用的仓库里按 Hashimoto 的规则做：agent 每犯一次错，就把纠正落成 CLAUDE.md 条目、hook 或脚本，而不是下次再提醒一遍。我们知识库里的 memory 系统本质上就是在做这件事。

## 9. 短评

- **「harness」是个好词，但热潮里有一半是重新包装。** 2023 年就有的 scaffold、2024 年的 agentic workflow、2025 年的 context engineering，到 2026 年统一叫 harness engineering。真正新的东西只有两点：一是 OpenAI 那种「代码库就是 harness、人不写代码只搭 harness」的组织形态被证明能跑到百万行；二是模型厂商开始把 inner harness 当产品卖（Agent SDK、Managed Agents），harness 成了锁定用户的新位置。
- **「harness engineering 取代 prompt engineering」这个说法不成立。** Böckeler 说得对：用户侧 harness 是 context engineering 的一种特定形式，prompt 只是搬了地方（从对话框搬进 AGENTS.md、skills 和工具描述），并没有消失。
- **Anthropic 那句「先别上框架」在 2026 年依然对，但边界变了。** 2024 年是「框架帮你省的代码不多，藏的东西不少」；2026 年厂商 SDK 自带的权限、压缩、子 agent、会话已经不是几行代码能复刻的了。所以现在的分界线是：学习和原型阶段手写 loop，生产阶段用厂商 SDK，只有跨模型编排的硬需求才上第三方框架。
- **框架看着几十个，真正的差异只有三条轴。** 谁掌握控制流（写死的图 vs 模型自主）、谁提供运行时（自己托管 vs 厂商托管）、抽象有多厚（薄如 mini-swe-agent vs 厚如 Managed Agents）。任何一个新框架进来，先在这三条轴上定位，剩下的都是 API 风格差异。
- **最值得记住的一个数字是 13.7。** 模型完全不动、只改 harness，Terminal-Bench 2.0 从 52.8 到 66.5。它同时说明两件事：做 harness 的投入回报率可能高于换模型；以及任何脱离 harness 的模型对比都是无效对比。
- **对做训练的人，最该警惕的是 benchmark 数字脱离 harness 被引用。** 报告一个 agentic 分数时，模型名、harness 名、harness 版本三者缺一不可，否则不可比。

## 参考资料

一手来源（本文事实依据）：

- [1] Anthropic, *Building effective agents*, 2024-12-19. https://www.anthropic.com/engineering/building-effective-agents
- [2] OpenAI, *Harness engineering: leveraging Codex in an agent-first world*, 2026-02-11. https://openai.com/index/harness-engineering/
- [3] Hugging Face, *Harness, Scaffold, and the AI Agent Terms Worth Getting Right*, 2026-05-25. https://huggingface.co/blog/agent-glossary
- [4] Anthropic, *Effective harnesses for long-running agents*, 2025-11-26. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- [5] Mitchell Hashimoto, *My AI Adoption Journey*, 2026-02-05. https://mitchellh.com/writing/my-ai-adoption-journey
- [6] Anthropic, *Effective context engineering for AI agents*, 2025-09-29. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- [7] Anthropic, *How we built our multi-agent research system*, 2025-06-13. https://www.anthropic.com/engineering/multi-agent-research-system
- [8] Anthropic, *Building agents with the Claude Agent SDK*, 2025-09-29. https://claude.com/blog/building-agents-with-the-claude-agent-sdk
- [9] Claude Agent SDK 文档：*Overview* 与 *How the agent loop works*. https://code.claude.com/docs/en/agent-sdk/overview 、https://code.claude.com/docs/en/agent-sdk/agent-loop
- [10] Anthropic, *A harness for every task: dynamic workflows in Claude Code*, 2026-06-02. https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code
- [11] Birgitta Böckeler, *Harness engineering for coding agent users*, martinfowler.com, 2026-04-02. https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html
- [12] Wikipedia, *Agent harness*（2026）. https://en.wikipedia.org/wiki/Agent_harness

框架与协议部分的一手来源（读取日期 2026-09-17）：

- [13] LangChain, *Agent Frameworks, Runtimes, and Harnesses — oh my!*（2025-10-25）与 *The Anatomy of an Agent Harness*（2026-03-10）. https://blog.langchain.com/agent-frameworks-runtimes-and-harnesses-oh-my/ 、https://blog.langchain.com/the-anatomy-of-an-agent-harness/
- [14] LangChain, *Improving Deep Agents with harness engineering*, 2026-02-17. https://blog.langchain.com/
- [15] Terminal-Bench / Harbor：https://www.tbench.ai/news/announcement-2-0 、https://docs.harborframework.com/core-concepts 、榜单 https://www.tbench.ai/leaderboard/terminal-bench/4.0
- [16] Anthropic, *Scaling Managed Agents*, 2026-04-08. https://www.anthropic.com/engineering/managed-agents
- 各框架官方文档与 README（逐一读取）：Claude Agent SDK https://code.claude.com/docs/en/agent-sdk/ ；OpenAI Agents SDK https://openai.github.io/openai-agents-python/ ；Google ADK https://adk.dev/ ；deepagents / CrewAI / AutoGen / Microsoft Agent Framework / smolagents / mini-swe-agent / OpenHands / Gemini CLI 的 GitHub README
- 协议标准：MCP 规范 https://modelcontextprotocol.io/specification/latest （2026-07-28 版）；A2A https://a2a-protocol.org/latest/ ；Agent Skills https://agentskills.io/ ；AGENTS.md https://agents.md/
- 评测：SWE-bench 榜单 https://www.swebench.com/ ；τ-bench https://github.com/sierra-research/tau2-bench ；OSWorld https://github.com/xlang-ai/OSWorld
- star / 许可证 / 最后提交：GitHub REST API `https://api.github.com/repos/<owner>/<repo>`，2026-09-17 读取

历史背景（未逐条核对，作延伸阅读）：

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022. https://arxiv.org/abs/2210.03629 （「思考 → 行动 → 观察」循环的论文起点）
- 库内旧材料：[Agent 目录 README](../README.md) 里李宏毅 2025 课程摘录（工具选择模块、记忆、模型对外部信息的信任）
