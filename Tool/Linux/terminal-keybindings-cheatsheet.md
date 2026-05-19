# Terminal 快捷键速查

终端命令行编辑用的是 **readline**（bash/zsh/python REPL 等都共享这套），快捷键通用。

---

## 一、删除

| 快捷键 | 说明 |
|---|---|
| `Ctrl+w` | 删除前一个单词（按**空格**切分，最常用） |
| `Alt+Backspace` | 删除前一个单词（按**标点**切分，更细粒度） |
| `Alt+d` | 删除后一个单词 |
| `Ctrl+u` | 删除光标前所有内容（到行首） |
| `Ctrl+k` | 删除光标后所有内容（到行尾） |
| `Ctrl+u` 后 `Ctrl+k` | 删除整行（或 `Ctrl+e` 跳行尾再 `Ctrl+u`） |
| `Ctrl+h` | 删除前一个字符（= Backspace） |
| `Ctrl+d` | 删除后一个字符（**空行时会退出 shell，小心**） |
| `Ctrl+y` | 粘贴最近一次删除的内容（找回误删） |

> readline 的删除会进入 kill-ring，相当于剪贴板。`Ctrl+u` 误删一长串时按 `Ctrl+y` 就能恢复。

---

## 二、光标移动

| 快捷键 | 说明 |
|---|---|
| `Ctrl+a` | 跳到行首 |
| `Ctrl+e` | 跳到行尾 |
| `Ctrl+b` | 左移一个字符 |
| `Ctrl+f` | 右移一个字符 |
| `Alt+b` | 左移一个单词 |
| `Alt+f` | 右移一个单词 |

---

## 三、历史命令

| 快捷键 | 说明 |
|---|---|
| `Ctrl+r` | 反向搜索历史（再按 `Ctrl+r` 翻上一个匹配） |
| `Ctrl+g` | 退出搜索（保持当前命令行不变） |
| `↑` / `↓` | 翻历史（已配置 `~/.inputrc` 后按前缀过滤） |
| `Ctrl+p` / `Ctrl+n` | 同 ↑ / ↓ |
| `!!` | 上一条命令 |
| `!$` | 上一条命令的最后一个参数 |
| `Alt+.` | 插入上一条命令的最后一个参数（按多次翻更早的） |

---

## 四、其它常用

| 快捷键 | 说明 |
|---|---|
| `Ctrl+l` | 清屏（= `clear`，但不清历史） |
| `Ctrl+c` | 中断当前命令 / 放弃当前输入 |
| `Ctrl+z` | 挂起前台任务（后续 `fg` 恢复，`bg` 转后台） |
| `Ctrl+s` / `Ctrl+q` | 暂停 / 恢复终端输出（容易误按导致「卡住」，按 `Ctrl+q` 解开） |
| `Tab` | 补全 |
| `Tab Tab` | 列出所有候选（已配置 `show-all-if-ambiguous` 后一次 Tab 即可） |

---

## 五、大小写转换

| 快捷键 | 说明 |
|---|---|
| `Alt+u` | 当前单词转大写 |
| `Alt+l` | 当前单词转小写 |
| `Alt+c` | 当前单词首字母大写 |

---

## 六、macOS Terminal 注意

`Alt` 键（即 `Option`）默认不发 Meta，需要在 Terminal/iTerm2 偏好设置里把 Option 设为 **"Meta key"** / **"Esc+"**。

或者用 `Esc` 替代：**先按 `Esc` 松开，再按字母**。例如 `Alt+d` ≡ `Esc` `d`。

---

## 七、tmux 前缀键冲突说明

tmux 默认前缀是 `Ctrl+b`，会拦截 readline 的 `Ctrl+b`（左移一字符）。在 tmux 里建议用 `Ctrl+f` / `Ctrl+b`+方向键 替代，或者把 tmux 前缀改成 `Ctrl+a`（再让 readline 行首改用 `Home`）。

---

## 配套配置

`~/.inputrc` 里这几条让补全和历史更顺手：
```
set completion-ignore-case on        # 大小写不敏感
set show-all-if-ambiguous on         # 一次 Tab 列候选
"\e[A": history-search-backward      # ↑ 按前缀过滤历史
"\e[B": history-search-forward       # ↓ 同上
```
