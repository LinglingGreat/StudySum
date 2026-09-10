---
title: Claude Code 记忆机制与跨机迁移
created: 2026-09-09
tags: [ClaudeCode, memory, 工具]
type: 工具笔记
---

# Claude Code 记忆机制与跨机迁移

> 2026-09-09 把 A100 集群上积累的 228 条记忆迁到 Mac 时摸清的。适用于任何「换机器 / 多目录起 Claude 但想共用一份经验」的场景。

## 记忆在物理上是什么

- 路径：`~/.claude/projects/<启动目录的路径 slug>/memory/`，slug 是把路径里的 `/` 换成 `-`（如 `/Users/liling/xinchen` → `-Users-liling-xinchen`）
- 一条记忆一个 md 文件，frontmatter 有 `name` / `description` / `metadata.type`（user / feedback / project / reference）。`description` 用于自动召回时判断相关性
- 同目录的 `MEMORY.md` 是索引，**每次起 Claude 全文注入上下文**，Claude 看到索引后按需打开具体文件
- 结论：记忆**按启动目录隔离**。从 `~/work/DataPre` 起和从 `~` 起，读的是两个池子；每个池子的 MEMORY.md 体积直接等于每个 session 的固定开销（A100 那份 48KB ≈ 2 万 token）

## 跨机迁移怎么做

1. **搬文件**：把源机各项目 `memory/*.md`（去掉 MEMORY.md）原样复制进目标池子。同名文件（多个项目都有 `user_profile.md`）改名并把 frontmatter `name` 去重，其余内容不改。留一份清单方便回滚
2. **多入口共用一个池子**：目标机上常用的启动目录各对应一个 project 目录，把它们的 `memory` 做成指向同一个真实目录的软链接。Claude 读写走软链，透明
3. **两级索引控成本**：几百条不要平铺进 MEMORY.md。按主题分成若干 `index_<类>.md`（自带 frontmatter，type: reference，description 里写关键词，这样自动召回也能命中分索引），MEMORY.md 只放一节分类指针。代价是 Claude 要主动开分索引，召回比平铺弱一点
4. **绑源机路径的条目不要改**：它们描述的是源机状态，通过 ssh 用源机时仍然有效；在分索引头部注明「路径以源机为准」即可
5. **源机全局 CLAUDE.md** 分两类处理：通用规则（开发习惯、git 习惯）合并进目标机全局 CLAUDE.md，先出 diff 再落地；绑源机路径的规则（docs 落点、worklog 台账、shell 配置位置）转成一条 reference 记忆

## 约束与取舍

- 源机出不了外网时 git 中转不可行；只能目标机趁 ssh 通着时 `rsync` 拉。真正的「同一份文件两边共享」做不到，能做的是「一次性迁 + 需要时再拉」
- 一起迁的还有 `~/.claude/skills/` 下自定义 skill（检查名字冲突；用 `codex-link` 给 Codex 也链上）、项目 docs 目录（md + 图，排除数据产物）
- 会话记录 `*.jsonl` 和 `history.jsonl` 不值得迁：体积大、不可读、经验已经沉淀在记忆里
- 分类 228 条用子 agent 读 description 归类，比关键词规则准；类别定成 8 个：评测 / 数据 / 训练 / 部署 / 业务 / 协作习惯 / 环境参考 / 用户画像
