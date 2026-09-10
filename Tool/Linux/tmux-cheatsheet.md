# tmux 常用命令与快捷键

## 基础概念

- **会话 (Session)**: 一组窗口的集合，断开 SSH 后会话仍在后台运行
- **窗口 (Window)**: 相当于标签页，一个会话可以有多个窗口
- **窗格 (Pane)**: 窗口内的分屏区域

所有快捷键都以 **前缀键** `Ctrl+b`（默认）开头，先按 `Ctrl+b` 松手，再按对应键。

---

## 一、会话管理（终端命令）

| 命令 | 说明 |
|------|------|
| `tmux` | 新建匿名会话 |
| `tmux new -s 名称` | 新建命名会话 |
| `tmux ls` | 列出所有会话 |
| `tmux attach -t 名称` | 连接到指定会话 |
| `tmux attach` | 连接到最近的会话 |
| `tmux kill-session -t 名称` | 杀掉指定会话 |
| `tmux kill-server` | 杀掉所有会话及 tmux 服务 |
| `tmux rename-session -t 旧名 新名` | 重命名会话 |

---

## 二、会话操作（快捷键）

| 快捷键 | 说明 |
|--------|------|
| `Ctrl+b d` | 分离当前会话（回到终端，会话后台运行） |
| `Ctrl+b s` | 列出所有会话并切换 |
| `Ctrl+b $` | 重命名当前会话 |

---

## 三、窗口操作

| 快捷键 | 说明 |
|--------|------|
| `Ctrl+b c` | 新建窗口 |
| `Ctrl+b ,` | 重命名当前窗口 |
| `Ctrl+b w` | 列出所有窗口并切换 |
| `Ctrl+b n` | 切换到下一个窗口 |
| `Ctrl+b p` | 切换到上一个窗口 |
| `Ctrl+b 0-9` | 切换到第 N 个窗口 |
| `Alt+←` | 切换到上一个窗口（自定义快捷键） |
| `Alt+→` | 切换到下一个窗口（自定义快捷键） |
| `Ctrl+b &` | 关闭当前窗口（会确认） |

---

## 四、窗格（分屏）操作

| 快捷键 | 说明 |
|--------|------|
| `Ctrl+b "` | 水平分屏（上下） |
| `Ctrl+b %` | 垂直分屏（左右） |
| `Ctrl+b 方向键` | 在窗格间切换 |
| `Ctrl+b x` | 关闭当前窗格（会确认） |
| `Ctrl+b z` | 当前窗格全屏/恢复 |
| `Ctrl+b {` | 当前窗格与上一个窗格交换 |
| `Ctrl+b }` | 当前窗格与下一个窗格交换 |
| `Ctrl+b Space` | 切换窗格布局 |
| `Ctrl+b Ctrl+方向键` | 调整窗格大小 |
| `Ctrl+b q` | 显示窗格编号，按数字跳转 |

---

## 五、复制模式（翻页/搜索/复制）

| 快捷键 | 说明 |
|--------|------|
| `Ctrl+b [` | 进入复制模式（可翻页、搜索、选择文本） |
| `q` 或 `Esc` | 退出复制模式 |
| `上下键` / `PageUp/PageDown` | 复制模式中翻页 |
| `/` | 复制模式中向下搜索 |
| `?` | 复制模式中向上搜索 |
| `Space` | 复制模式中开始选择 |
| `Enter` | 复制选中文本 |
| `Ctrl+b ]` | 粘贴复制的文本 |

---

## 六、其他实用操作

| 快捷键/命令 | 说明 |
|-------------|------|
| `Ctrl+b :` | 打开命令行（输入 tmux 命令） |
| `Ctrl+b t` | 显示时钟 |
| `Ctrl+b ?` | 列出所有快捷键 |
| `tmux source-file ~/.tmux.conf` | 重新加载配置 |

---

## 七、当前 ~/.tmux.conf 配置说明

```bash
set -g mouse on                    # 鼠标支持（点击切换、滚轮翻页）
setw -g automatic-rename on        # 窗口名自动跟踪当前命令
setw -g automatic-rename-format '#{pane_current_command}'
# 前缀键保持默认 Ctrl+b
bind -n M-Left previous-window     # Alt+← 切换窗口
bind -n M-Right next-window        # Alt+→ 切换窗口
set -g status-right '...'          # 状态栏右侧显示 Claude 额度 + 时间
set -g status-interval 30          # 状态栏每 30s 刷新
```

修改配置后执行：`tmux source-file ~/.tmux.conf`

---

## 八、Shell 别名（已配置在 ~/.shell_common）

| 别名 | 等效命令 | 说明 |
|------|----------|------|
| `tm` | `tmux` | 新建匿名会话 |
| `tn work` | `tmux new -s work` | 新建命名会话 |
| `tl` | `tmux ls` | 列出所有会话 |
| `ta work` | `tmux attach -t work` | 连接到指定会话 |
| `tk work` | `tmux kill-session -t work` | 删除指定会话 |
| `tp` | （自定义函数） | 列出所有 session 及其当前运行的进程 |

---

## 九、最常用场景

### 在服务器跑长时间任务
```bash
tmux new -s train          # 创建会话
python train.py            # 开始训练
# 按 Ctrl+b d 分离         # 断开 SSH 也没事
tmux attach -t train       # 下次回来接上
```

### 同时看代码和运行结果
```bash
tmux                       # 进入 tmux
# Ctrl+b %                 # 左右分屏
# 左边写代码，右边运行
```

### 多任务并行监控
```bash
tmux new -s work
# Ctrl+b c                 # 创建多个窗口
# Ctrl+b 0/1/2             # 数字键快速切换
```

### 推荐的 session 组织方式

按用途分 session，同类任务用窗口区分：

| Session | 用途 | 窗口数 |
|---------|------|--------|
| `claude` | 所有 claude 实例 | 多个，每个窗口一个 claude |
| `download` | 长时间下载/数据处理 | 单独 session，挂了不影响其他 |
| `train` | 训练任务 | 同上 |

```bash
# 1. 创建 claude session
tn claude

# 2. 在 session 内开多个窗口
Ctrl+b c                   # 新建窗口
Ctrl+b ,                   # 给窗口命名（如 "elo分析"）

# 3. 切换窗口
Alt+← / Alt+→              # 左右切换
Ctrl+b 0-9                 # 数字键跳转
Ctrl+b w                   # 列出所有窗口并选择

# 4. 状态栏自动显示每个窗口正在运行的命令（automatic-rename）
```

原则：**长时间跑的独立任务单独 session，交互式 claude 实例合并到一个 session 的多个窗口里。**

---

## 十、踩过的坑（2026-09-10，Mac + iTerm2 + tmux 3.7）

### 坑 1：在 tmux 里往上滚，最顶上是当前会话的内容，但跟记忆完全对不上

**现象**：Claude Code 跑在 tmux 里，往上滚，scrollback 顶部确实是本会话的内容，
但它是会话**中途**的某一段，前面的全没有 —— 看起来像"错乱"。

**根因**：会话不是在 tmux 里开始的，是 `claude --resume` 恢复进来的。
**resume 只把最近一段历史重放到终端，不重放全部**，所以 pane 的 scrollback 从会话中途开始。

**怎么确认**（三条证据同时看）：

```bash
# 1. pane 里 claude 进程跑了多久 —— 远小于会话实际时长就是 resume
tmux list-panes -t <sess>:<win> -F '#{pane_pid}' | xargs -I{} pgrep -P {} | xargs -I{} ps -o pid,lstart,etime,command -p {}

# 2. scrollback 里有没有启动横幅 —— 没有 = 不是从这里启动的
tmux capture-pane -p -S - -t <sess>:<win> | grep -i 'welcome to claude'

# 3. 会话真实长度看 transcript
wc -l ~/.claude/projects/<项目路径转义>/<session-id>.jsonl
```

**⚠️ 做上面三步之前，先确认「你查的 pane 里跑的到底是哪个会话」** —— 我第一次查就栽在这：
人在 tmux **外面**的窗口里，却去抓 tmux 里另一个会话的 scrollback，
再拿那个 pane 的进程启动时间跟**自己**会话的 transcript 行数配对，推出来的结论方向对、证据全错。

```bash
# 0. 先确认自己在不在 tmux 里（空 = 不在，那 tmux 里跑的就是别人）
echo "${TMUX:-不在 tmux}"

# 1. 从 scrollback 里的 scratchpad 路径反查该 pane 属于哪个 session id
tmux capture-pane -p -S - -t <sess>:<win> \
  | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' \
  | sort | uniq -c | sort -rn | head
# 出现次数最多的那个就是；若你自己的 session id 出现 0 次，说明这个 pane 不是你

# 2. 确认是同一个会话之后，再去比 transcript 创建时间 vs 进程启动时间
stat -f '%SB' -t '%H:%M:%S' ~/.claude/projects/<项目>/<那个 id>.jsonl
```

**真实案例的数据长这样**（2026-09-10）：

| | 值 | 说明 |
|---|---|---|
| transcript | 5656 行，**前一晚 19:11** 创建 | 会话真实体量 |
| pane 里进程启动 | **次日 14:06** | 差了一整晚 → 是 resume |
| pane scrollback | 1635 行 | 只有重放的一小段 + 之后的新内容 |

5656 行的会话若从头跑，scrollback 会有几万行；只剩 1635 行就是断层的量化证据。

**结论**：这不是 tmux 的问题，配置改不好。
- 要回看完整历史 → 读 transcript `~/.claude/projects/<项目>/<session>.jsonl`，或 `claude --resume` 后用界面自己的历史
- 想让 scrollback 完整 → **在 tmux 里从头启动** claude，别在外面开了再 resume 进来
- 对照：在服务器上习惯先 tmux 再 claude 的，就从来不会遇到这个断层

### 坑 2：macOS 上 `LC_CTYPE=C` 不会让 tmux 中文乱码（别照搬 Linux 经验）

Linux 上的常识是"locale 不是 UTF-8 → tmux 按单字节拆汉字 → 乱码"。
**macOS + tmux 3.7 实测不成立**：`LC_CTYPE=C` 时 tmux 仍然 `utf8=1`，
存进 scrollback 的中文字节完全正确（tmux 自己 fallback 到 UTF-8 了）。

```bash
# 判定 tmux 这层到底是不是 UTF-8
tmux list-clients -F 'term=#{client_termname} utf8=#{client_utf8}'

# 直接验字节：tmux 内部存的 vs 直接输出的，一致就说明 tmux 没问题
tmux new-session -d -s _probe -x 80 -y 10
tmux send-keys -t _probe 'printf "中文测试|abc\n"' Enter; sleep 1
tmux capture-pane -p -t _probe | grep '中文测试' | hexdump -C | head -3
tmux kill-session -t _probe
printf "中文测试|abc\n" | hexdump -C | head -3     # 对照
# 中 = e4 b8 ad，文 = e6 96 87
```

所以在 macOS 上看到 tmux 里中文有问题，**先别急着怪 locale**，用上面的字节对照法确认是哪一层。

顺带：oh-my-zsh 的 `.zshrc` 模板里 `# export LANG=en_US.UTF-8` 默认注释着，
装完基本没人打开（`locale` 会显示 `LANG=""`、`LC_CTYPE="C"`）。
这行**该开**（很多 CLI 工具、Python 输出、ssh 到 Linux 时都依赖它），只是它不是 tmux 乱码的解释。
设 `LANG` 就够，不要设 `LC_ALL`——它强制覆盖所有 `LC_*`，出问题时没法单项调。

注：非交互式 `ssh host 'locale'` 显示 `LANG=` 是正常的，那是没 source 交互式 rc 文件，
不代表登录进去也是空的（A100 的 `LANG` 就写在 `~/.shell_common` 里）。

### 附：`alternate_on` 是什么，什么时候真该看它

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} cmd=#{pane_current_command} alt=#{alternate_on} hist=#{history_size}'
```

- `alternate_on=1`（vim / htop / less）：程序用备用屏幕，退出后屏幕**还原**成进入前的样子，不污染 scrollback
- `alternate_on=0`（Claude Code）：输出直接进正常缓冲区，退出后内容留在 scrollback 里

真正会被它坑到的场景是**满屏刷新型 TUI**（进度条、watch）把每一帧都糊进 scrollback；
Claude Code 的对话正文是正常输出，翻起来是干净的，不要拿这个当排版错乱的解释。
