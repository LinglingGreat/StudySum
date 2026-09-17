---
title: lark-cli 飞书 CLI 安装与跨机迁移
created: 2026-09-17
tags: [lark-cli, 飞书, ClaudeCode, skill, 工具]
type: 工具笔记
---

# lark-cli 飞书 CLI 安装与跨机迁移

> 2026-06-12 在 A100 首次安装（v1.0.52），2026-09-17 在 Mac 复装（v1.0.96）时踩到「密钥不随配置文件走」的坑，顺手把 A100 上那份 `~/docx/lark-cli-cheatsheet.md` 合并进来。官方仓库 https://github.com/larksuite/cli （MIT，1.7 万 star，几乎每天更新）。

## 一、它是什么

- npm 包 `@larksuite/cli`：一个二进制 `lark-cli` + 28 个 `lark-*` Claude skill（doc / drive / wiki / im / base / sheets / calendar / mail / task …）
- skill 装在 `~/.agents/skills/lark-*`，软链进 `~/.claude/skills/`；Codex 那套软链（`codex-link`）不会自动同步，要的话手动加
- 两种身份：**user**（你本人，文档读写用这个，命令加 `--as user`）、**bot**（app 自己，需要 app secret）

## 二、安装（任何机器都一样）

```bash
npx -y @larksuite/cli@latest install   # 装全局包 + skills，升级也是这条
lark-cli --version
```

依赖 nvm 的 node（Mac v24.12、A100 v24.13）；换 node 版本要重装。

## 三、首次配置 vs 跨机复用

| 场景 | 做法 |
|---|---|
| 全新、没有 app | `lark-cli config init --new` → 浏览器里完成，CLI 自动建一个 app |
| 已有 app（如 A100 的 `cli_aaa11e6c697a9cda`） | 拷 `~/.lark-cli/config.json` 过来 **+ 手动灌 app secret** |

**坑：config.json 里的 appSecret 只是钥匙串引用**（`{"source":"keychain","id":"appsecret:cli_…"}`）。Linux 存在 `~/.local/share/lark-cli/`，macOS 存系统钥匙串，拷文件带不过来。症状：`auth status` 能认出 app 和用户名，但 bot 报 `missing app secret`，并且 **device flow 登录直接报 `missing a required parameter: client_secret`**，看着像网络问题，其实不是，别往代理上查。

补 secret（从开发者后台 https://open.feishu.cn/app → 该应用 → 凭证与基础信息 → App Secret 复制）：

```bash
read -rs "SECRET?粘贴 App Secret 后回车: " && printf '%s\n' "$SECRET" | lark-cli config init --app-id cli_aaa11e6c697a9cda --app-secret-stdin --brand feishu; unset SECRET
```

`--app-secret-stdin` 不打任何提示、静默等 stdin，**裸跑时粘进去的 secret 会明文回显在终端**（实犯过，事后 `clear` 并考虑在后台重置）。直接裸跑它再 Ctrl-C 会报 `read /dev/stdin: input/output error`，是正常现象。上面这条用 `read -rs` 包一层，有提示、不回显、不进 history。

## 四、登录与权限（device flow，不用去后台点权限）

```bash
lark-cli auth login --no-wait --json --domain docs,drive,wiki   # 拿 verification_url + device_code
# 浏览器打开 verification_url 授权（约 10 分钟有效）
lark-cli auth login --device-code <device_code>                 # 授权后再轮询完成
lark-cli auth status                                             # identities.user.scope 看已授列表
```

- 要精确 scope 用 `--scope "a b c"`（空格分隔；逗号整串会被当成一个 scope 误报 missing）。A100 当初授的 28 个 doc scope 清单在 A100 `~/work/DataPre/doc_scopes.json`
- **user token 的 refresh 只有 7 天**，过期重新 `auth login`；接 cron 的自动化每周得续一次
- device code 时效短，别在用户授权前提前轮询，会作废
- 代理：A100 集群注入代理时要 `export LARK_CLI_NO_PROXY=1`，否则凭据也过代理；Mac 本地 Clash（127.0.0.1:7897）带不带都通

## 五、文档读写常用命令

- 建：`lark-cli docs +create --doc-format markdown --content @file.md --parent-position my_library --json`（子命令带 `+` 前缀，`docs create` 会报 unknown）
- 改：`docs +update --doc <id> --command overwrite --doc-format markdown --content @file.md`（保持同一 URL）；局部改用 `--command str_replace --pattern`，markdown 模式可多行匹配
- 读回校验：`docs +fetch --doc <id> --doc-format xml`（flag 是 `--doc` 不是 `--document-id`）
- markdown 表格会转成飞书原生表格块，标题/列表正常 ⇒ 汇报文档直接写 md 灌进去即可
- `docs +search` 需要 `search:docs:read`，没授就不可用，不影响读写
- 删：`lark-cli drive +delete --file-token <id> --type docx --as user --yes`（`drive files` 子命令下没有 delete，走 `+delete` shortcut；高风险操作必须 `--yes`）
- 2026-09-17 Mac 实测：`+create`（md 含表格）→ `+fetch` 内容一致 → `+delete` 成功，`--domain docs,drive,wiki` 一次授了 134 个 scope

### 写 docx 的坑（A100 08-17 / 08-31 实测）

1. **`--command overwrite` 是整篇替换**，会冲掉别人在飞书上的手动编辑（实犯过：用户删了一节，overwrite 又写回去）。推之前先 `docs +fetch` 拉回比对，能局部改就别整篇覆盖
2. **飞书 md 解析器的加粗会串行**：同一段落 ≥2 处 `**…**`，第 1 个到第 2 对之间的普通文字一起变粗（16 段里 5 段错乱）。给飞书的 md 每段只留一处加粗；表格单元格各自独立不受影响
3. 直接用 block API 建表格**最多 9 行**，10 行起返回空 stdout + 退出码 1 且无报错；超 9 行拆多张表（走 md 导入没这个限制）
4. `block_type` 与字段名要匹配（type 4 = heading2，配 heading3 报 invalid param）
5. DELETE 成功时 stdout 为空，看退出码别看输出
6. 上传素材不认绝对路径（`cannot open file`），`cwd=父目录` + 只传文件名
7. 文件附件三段式：建 type=23 空 block（飞书自动包一层 view/33，返回的是外层 id，真 file block 是 `children[0]`）→ 上传素材 `parent_node` 填内层 id → PATCH 内层 `replace_file`。判断关联成功看 `file.token` 非空，别拿 block_type 23→33 当证据

## 六、安全审查记录

2026-09-17 用 skill-vetter 扫 28 个 skill 共 550 个文件（513 md / 16 py / 7 jsx / 6 html）：无非飞书域名的外部请求、无 base64 解码、无 eval、不碰 `~/.ssh`。命中的 `curl` 是画板文档里的下载示例，`.aws` 是「发布时默认拦截凭据文件」的说明，`exec(` 是 JS 正则的 `re.exec`。结论：可装。

## 参考

- 官方 README（中文）：https://github.com/larksuite/cli/blob/main/README.zh.md
- A100 原 cheatsheet：A100 `~/docx/lark-cli-cheatsheet.md`（内容已并入本文）
- Claude memory：`reference_lark_cli.md`（同一批坑的机器可读版）
