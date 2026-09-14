---
title: 多语种词数统计：用 ICU BreakIterator 一套规则覆盖所有语种
created: 2026-09-11
tags:
  - 工具笔记/python
  - NLP/分词
type: 工具笔记
---

# 多语种词数统计：ICU BreakIterator（零安装 ctypes 版）

## 问题

要数「一段文本有多少个词」，而文本可能是任何语种。常见两种土办法都会在某些语种上**方向性出错**：

| 方法 | 泰语（97 字符的例句） | 中文（35 字） | 英文含破折号 |
|---|---|---|---|
| `text.split()` | **6**（低估 ~3×） | **1** | `smiles—warmly—and` 算 1 个 |
| 按字符数 | **97**（高估 ~4.5×） | 35 | — |
| **ICU** | **24** ✅ | **18** ✅ | **切成 3 个** ✅ |

泰语、老挝语、高棉语、缅甸语**不用空格分词**，`split()` 必然低估；但它们的词平均 4-6 个字符，
按字符数又会高估。中文按字符数是通行口径但也高估 ~1.7×（中文词平均 1.5-2 字）。
**韩语看着像 CJK，其实有空格**（띄어쓰기），误归到「按字符」会错得离谱。

⇒ 按语种写分支的做法，本质上是在维护一份永远不全的名单。

## 解法：ICU 的 BreakIterator

ICU（International Components for Unicode）是 Unicode 官方的实现，`UBRK_WORD` 模式：
- 泰/老/高棉/缅/中/日 用**内置词典**分词
- 有空格的语种走 UAX#29 规则
- 一个接口，不需要判断语种

验证（同一套代码）：

| 语种 | ICU | split | 说明 |
|---|---|---|---|
| 泰语 | 24 | 6 | 词典分词 |
| 中文 | 18 | 1 | 35 字 → 18 词，符合中文词长 |
| 日语 | 12 | 1 | 同理 |
| 英文（含 —） | 6 | 4 | 破折号被正确当作边界 |
| 俄语 / 阿拉伯 / 韩语 | 4 / 4 / 4 | 4 / 4 / 4 | **与 split 一致 ⇒ 没有过度切分** |

最后一行很重要：有空格的语种 ICU 和 `split()` 结果相同，说明换成 ICU **不会**把原本正确的语种改坏。

## 为什么用 ctypes 而不是 PyICU

`pip install PyICU` 需要编译，依赖 `libicu-dev` 头文件和 `icu-config`；在只装了运行时库的机器上
**装不上**（`Getting requirements to build wheel did not run successfully`）。

而 ICU 的 C API 是稳定的共享库导出符号，`ctypes` 直接调即可，**零安装**。
只要机器上有 `libicuuc.so.*` 就能用（大多数 Linux 发行版自带）。

## 代码

```python
import ctypes, ctypes.util, re

_UBRK_WORD = 1
_UBRK_DONE = -1
_UBRK_WORD_NONE_LIMIT = 100      # ruleStatus < 100 的片段是空白/标点，不算词

path = ctypes.util.find_library("icuuc")          # 如 'libicuuc.so.70'
suf  = "_" + re.search(r"\.so\.(\d+)", path).group(1)   # ICU 符号带版本后缀！
lib  = ctypes.CDLL(path)

_open = getattr(lib, "ubrk_open" + suf)
_open.restype  = ctypes.c_void_p
_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p,
                  ctypes.c_int32, ctypes.POINTER(ctypes.c_int)]   # ← 5 个参数
_setText = getattr(lib, "ubrk_setText" + suf)
_setText.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                     ctypes.POINTER(ctypes.c_int)]
_first  = getattr(lib, "ubrk_first"  + suf); _first.restype  = ctypes.c_int32
_next   = getattr(lib, "ubrk_next"   + suf); _next.restype   = ctypes.c_int32
_status = getattr(lib, "ubrk_getRuleStatus" + suf); _status.restype = ctypes.c_int32
for f in (_first, _next, _status):
    f.argtypes = [ctypes.c_void_p]

err = ctypes.c_int(0)
_BI = _open(_UBRK_WORD, b"", None, 0, ctypes.byref(err))   # 复用同一个迭代器

def count_words(text):
    if not text:
        return 0
    buf = text.encode("utf-16-le")                 # ICU 用 UTF-16
    arr = ctypes.create_string_buffer(buf, len(buf))
    e = ctypes.c_int(0)
    _setText(_BI, ctypes.cast(arr, ctypes.c_void_p), len(buf) // 2, ctypes.byref(e))
    n = 0
    _first(_BI)
    while _next(_BI) != _UBRK_DONE:
        if _status(_BI) >= _UBRK_WORD_NONE_LIMIT:  # 过滤空白与标点
            n += 1
    return n
```

## 三个坑（都实际踩过）

1. 🔴 **`ubrk_open` 是 5 个参数，不是 6 个**
   ```c
   UBreakIterator* ubrk_open(UBreakIteratorType type, const char *locale,
                             const UChar *text, int32_t textLength, UErrorCode *status);
   ```
   我按 6 个声明（和 `ubrk_openRules` 记混了），**直接 Segmentation fault**，python 进程没有任何
   错误输出就没了。ctypes 声明错参数不会报错，只会踩内存。

2. **ICU 的 C 符号带版本后缀**：实际导出的是 `ubrk_open_70`，不是 `ubrk_open`。
   必须从 `.so` 文件名里取版本号拼出来。不同机器版本可能不同（我这里一台 70、一台 60），
   所以要动态推导而不是写死。

3. **逐条 open/close 太慢**：用 `ubrk_setText` 复用同一个迭代器。
   实测 10 万次 1.5 秒（91 万条约 13 秒）；每次新建的话慢一个量级。
   ⚠️ BreakIterator **非线程安全**，多线程要各建一个。

## 用它的时候

- 要跨语种比较文本长度（长度约束、截断率、生成长度统计）
- 训练数据里写「reply between N and M words」这类指令，需要数字与真实长度对齐
- **不要做静默回退**：ICU 不可用时应直接报错。若一部分数据用 ICU、一部分用 `split()`，
  两种口径混进同一份统计，事后极难发现（这类「口径混用」的 bug 我在评测里踩过，
  见 [[../../../xinchen/docs/模型训练/gemma4_DPO_system字段修复重训评测_20260910|口径踩坑记录]]）。

## 参考

- ICU C API `ubrk_*`：https://unicode-org.github.io/icu-docs/apidoc/released/icu4c/ubrk_8h.html
- UAX#29 Text Segmentation：https://unicode.org/reports/tr29/
