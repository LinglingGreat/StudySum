---
title: Claude Code 会话存储与命名机制
created: 2026-09-17
tags: [ClaudeCode, session, RemoteControl, 工具]
type: 工具笔记
---

# Claude Code 会话存储与命名机制

> 2026-09-17 排查「终端起的会话为什么在桌面客户端里看不到 / 名字为什么是 `lilingdemacbook-pro-local-valiant-flurry` / 客户端怎么读到远程服务器的会话」时在本机实测摸清的。版本：CLI 2.1.27x，桌面 app 内置 CLI 2.1.260。姊妹篇 [记忆机制](Claude-Code记忆机制与跨机迁移.md)。

## 一、会话在物理上是两套存储

| 入口 | 写什么 | 在哪 |
|---|---|---|
| 终端 `claude` | 只写对话流水 | `~/.claude/projects/<cwd slug>/<sessionId>.jsonl` |
| 桌面 app Code tab | 对话流水 **+ 一份会话元数据** | 流水同上；元数据在 `~/Library/Application Support/Claude/claude-code-sessions/<账号UUID>/<组织UUID>/local_<uuid>.json` |

- 元数据里有 `title` / `titleSource` / `model` / `cwd` / `permissionMode`，用 `cliSessionId` 反指到 jsonl
- **桌面侧边栏的「本地会话列表」只遍历元数据**，终端会话不写元数据 → 列表里没有它。反过来桌面开的会话在终端同目录 `claude --resume` 能看到（jsonl 在同一处），方向是单向的
- 两套存储生命周期独立：jsonl 被清理脚本删了，元数据还留着（本机 4–5 月的 15 条元数据全部找不到对应 jsonl）
- 这两者都是纯本地文件，和 claude.ai 网页端不互通；网页端只显示云端会话

实测对账（2026-09-17）：`my_knowledge_network` 终端 39 个 jsonl vs 桌面 15 条元数据，session id 零重合；`~` 下 30 个终端会话，桌面 0 条。

## 二、三套命名规则

| 来源 | 样子 | 规则 |
|---|---|---|
| 桌面 app 新建 | `gemma模型部署和扩缩容机制` | 模型在开局第一轮后概括生成，`titleSource: auto` |
| 终端 `claude` | `my-knowledge-network-53` | cwd 目录名 + 序号，`nameSource: derived`，**不生成内容标题** |
| Remote Control 自动名 | `lilingdemacbook-pro-local-valiant-flurry` | **FQDN 小写、点换横杠** + 随机形容词-名词 |

auto 标题的三个特性（都能在本机数据里对上）：
1. **只看开局**：标题 `你好` / `我有哪些技能` 只可能来自第一句；聊到第十轮跑题了标题也不动
2. **早期版本固定英文**：4–5 月输入 220 条里 181 条中文，标题全英文；7 月后才跟随对话语言
3. **开场白模式化就撞车**：四条 `Analyze DataTester experiment …` 近义，标题层面分不出是哪个实验

Remote Control 自动名的 `local` 不是「本地」，是 macOS 主机名的 `.local` 后缀：`lilingdeMacBook-Pro.local → lilingdemacbook-pro-local`，同规则 `gvna01.cluster.com → gvna01-cluster-com`。**出现这种名字 = 那个会话压根没标题**（终端起的）。同一列表里有标题的会话正常显示标题。

## 三、Remote Control：客户端怎么「读到」远程服务器的会话

不读服务器文件，读的是 Anthropic 后端上挂在账号名下的注册表。数据流：`远端 CLI → Anthropic 后端 → 客户端`，两台机器之间无直连。

- CLI 开 Remote Control 时向后端开一条出站长连接（bridge），后端发一个 `bridgeSessionId`，记在 `~/.claude/sessions/<pid>.json`；`~/.claude.json` 的 `replBridgePlaceholders` 是本地缓存
- 归组依据是账号：CLI `~/.claude.json → oauthAccount.accountUuid` 与桌面 `config.json → lastKnownAccountUuid` 同一个 UUID。桌面元数据目录的两层名字就是「账号 UUID / 组织 UUID」
- `idle` / `offline` = 那个 CLI 进程的长连接是否还活着。offline 的注册记录仍在列表里显示，但**点了没用**，得回那台机器重新拉起 CLI 桥才会重连
- 在客户端里发消息：客户端 → 后端 → 桥 → 远端 CLI 执行 → 原路返回。**代码、文件、执行全在远端**，客户端只是远程显示器
- 安全：信任基础是账号 token，任何登录了该账号的客户端都能驱动远端 CLI，权限等同于在那台机器上的 shell。集群机器开 Remote Control 前确认会话权限模式

侧边栏因此是两个来源：本地列表（元数据）里没有终端会话；Remote Control 区（后端）里有，但顶着主机名乱码。

## 四、怎么让名字可读

根因是终端会话不自动生成标题。CLI 帮助文本：

```
--remote-control [name]                       启动带 Remote Control 的会话
--remote-control-session-name-prefix <prefix> 自动名前缀（默认 hostname）
```

- 服务器上起会话直接给名：`claude --remote-control "gvna01 训练数据清洗"`
- 桌面/终端会话中途改名：会话里让 Claude 用 `set_session_title` 改，auto 生成的标题改起来不需要二次确认，用户手动设过的才会弹确认
- 最省事：开局第一句话写成标题该有的样子（「分析 0509 DataTester 留存实验」而不是「帮我处理一下这个文件」）

## 五、把终端会话迁到桌面客户端：官方导入

Remote Control 区里看到的终端会话是「正在运行的进程」的注册记录，终端一退出就下线消失——它不是保存的会话，不能当迁移路径。正确入口：

- **屏幕顶部 macOS 菜单栏 → Help → Troubleshooting → `Import Claude Code CLI Sessions…`**（不在 app 窗口内的设置页里；和 Show Logs in Finder / Clear Cache and Restart 一组，无条件显示）
- 机制：扫描 `~/.claude/projects/`，把 jsonl 复制到 `claude-code-sessions/<账号>/<组织>/imported-staging/`，生成 `local_*.json` 元数据并标记 `importedFrom`；客户端恢复会话就是 `claude --resume=<cliSessionId>`，上下文能完整接上
- 全量导入、没有勾选；已存在的跳过；结束弹结果框（导入/跳过/失败计数）
- 导入的会话第一次恢复要点一次确认（"This session was imported. Resuming lets Claude act on its history"），是设计不是报错
- 代码里大会话警告阈值 **10 MB**：超过则「Resume may fail if this exceeds the SDK prompt-size cap」。预防：先在终端 `claude --resume <id>` → `/compact` → `/exit` 再导入
- 导入前先退出终端里还在跑的同一会话，否则两个进程写同一份 jsonl
- 代码里另有 `claude://code/…?session=<uuid>` 形式的 deep link 走同一个 `importCliSession`，但 path 分派没从混淆代码里读清，未验证

## 六、两端共享 / 不共享的路径

桌面 app 拉起自带 CLI 时传 `--setting-sources=user,project,local`，同一个 `$HOME`，所以：

- **共享**：`~/.claude/settings.json`（hooks 在桌面会话里照常触发）、`~/.claude/CLAUDE.md`、项目 CLAUDE.md、`~/.claude/skills/`、`~/.claude/plugins/`、`~/.claude/projects/`（记录 + 记忆）、`~/.claude.json`
- **桌面额外**：`…/Claude/local-agent-mode-sessions/skills-plugin/<组织>/<账号>/skills/`（docx/pptx/xlsx/pdf/morning/schedule 等，即 `anthropic-skills:*`）；桌面自带的 MCP（浏览器、会话管理、Claude Docs、visualize 等）由 app 注入
- **不共享**：① CLI 二进制——桌面用 `…/Claude/claude-code/<ver>/`（app 自带，随 app 更新，实测一天内 2.1.260→2.1.271），终端用 `~/.local/share/claude/versions/<ver>`，版本各自升级；② 「始终允许」落点——终端写项目 `.claude/settings.local.json`，桌面写进该会话的 `local_*.json`（`sessionPermissionUpdates`，destination=session），终端批的桌面认、桌面批的终端不认；③ 模型/effort 由桌面 UI 逐会话传参覆盖 settings.json；④ 「No folder」桌面会话的 cwd 是临时 scratch 目录，记忆池独立，看不到项目记忆

## 七、桌面端「权限不一样」与「更慢」的实测

**权限**：两端都是 auto 模式（`settings.json` 的 `permissions.defaultMode=auto`，桌面另传 `--permission-mode auto`），分类器拒绝（"denied by the Claude Code auto mode classifier"）在终端会话里同样出现（596bca1d 里 3+ 次），不是桌面独有。桌面独有的四样：
1. **文件夹作用域**：会话绑定选中的文件夹，「No folder」会话落在临时 scratch 目录，碰任何项目都要先 `change_directory`/`request_directory` 拿授权（授权后环境里出现 "Additional working directories added"）
2. **桌面注入的安全 system prompt**：一整段 Prohibited / Explicit-permission 规则（删文件、发消息、提交表单、下载、填凭证……），模型会主动说「需要你确认 / 不能做」——这是模型**自述**没权限，不是硬拦截，终端没有这段
3. `--disallowedTools SendMessage`；权限询问走 `--permission-prompt-tool stdio` 弹到桌面 UI
4. 「始终允许」只写进该会话的 `local_*.json`，不跨会话

**速度**：按 jsonl 时间戳算「用户消息 → 首条 assistant 记录」：桌面会话中位 4.3 s / p90 7.3 s（8 轮样本），终端三个会话中位 5.8–8.0 s / p90 7.5–22.9 s；首轮 prompt 规模两端相当（桌面 65.7K tok，终端 53–68K；`~/.claude/skills` 里 28 个 lark-* skill 两端都加载）。**数据不支持「桌面更慢」**。感知差异更可能来自桌面 `--thinking-display omitted` + `CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES=false`：思考和工具调用过程不显示，长思考的一轮在桌面看起来是空等，终端能看到 spinner/思考流。

## 排查时踩的坑

- **jsonl 里有二进制内容，文本模式 grep 会漏**。第一次 `grep -rl` 全盘没搜到 `valiant`，加 `-a` 才命中——差点下了「不落盘」的错误结论
- `~/.claude/history.jsonl`（带 sessionId 的输入历史）只记终端 CLI 的输入，桌面会话不写进去；jsonl 被清后无法用它还原桌面会话内容（21 条只对上 1 条）
- `~/.claude/sessions/<pid>.json` 是运行时注册文件，含 `name` / `nameSource` / `entrypoint`(cli | claude-desktop) / `bridgeSessionId`，是判断「这个会话是谁起的、有没有开远程」最快的入口
