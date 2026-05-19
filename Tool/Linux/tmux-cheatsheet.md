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
