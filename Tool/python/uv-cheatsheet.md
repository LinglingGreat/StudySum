# uv 使用说明

> 安装路径：~/.local/bin/uv
> uv 是极速 Python 包管理器（Rust 编写），可替代 pip / pip-tools / virtualenv。
> 注意：在 GPU 集群上，深度学习环境（PyTorch、CUDA 相关）仍推荐用 conda 管理，uv 适合轻量脚本和工具安装。

## 核心概念

- `uv pip` — 替代 pip，速度快 10-100x
- `uvx` — 临时运行工具，无需安装（类似 npx）
- `uv venv` — 创建虚拟环境
- `uv run` — 在指定环境中运行脚本

---

## 一、包安装（替代 pip）

```bash
uv pip install 包名              # 安装包
uv pip install 包名==1.2.3       # 安装指定版本
uv pip install -r requirements.txt  # 从文件安装
uv pip install -e .              # 可编辑安装（开发模式）
uv pip uninstall 包名            # 卸载
uv pip list                      # 列出已装包
uv pip show 包名                 # 查看包信息
uv pip freeze                    # 导出当前环境依赖
```

---

## 二、虚拟环境

```bash
uv venv                          # 在当前目录创建 .venv
uv venv myenv                    # 创建指定名称的环境
uv venv --python 3.11            # 指定 Python 版本

source .venv/bin/activate        # 激活（Linux/Mac）
deactivate                       # 退出
```

---

## 三、uvx — 临时运行工具

```bash
uvx ruff check .                 # 临时运行 ruff（不需要先安装）
uvx black .                      # 临时运行 black 格式化
uvx httpie GET http://...        # 临时用 httpie 发请求
uvx --from rich python -c "from rich import print; print('[bold]hi[/bold]')"
```

> uvx 会自动下载工具到隔离缓存，不污染当前环境，用完即走。

---

## 四、运行脚本

```bash
uv run script.py                 # 用当前环境运行脚本
uv run --with pandas script.py  # 临时加一个依赖再运行
```

---

## 五、工具管理（持久安装）

```bash
uv tool install ruff             # 全局安装工具（到 ~/.local/bin）
uv tool install black
uv tool list                     # 列出已安装工具
uv tool uninstall ruff           # 卸载
uv tool upgrade ruff             # 升级
uv tool upgrade --all            # 升级所有工具
```

---

## 六、常用场景

### 快速跑一个脚本（无需配置环境）
```bash
uvx --from requests python -c "import requests; print(requests.get('https://httpbin.org/ip').json())"
```

### 在 conda 环境里加速 pip 安装
```bash
conda activate ll_mixenv
uv pip install some_package      # 比 pip 快很多
```

### 一键安装 requirements.txt
```bash
uv pip install -r requirements.txt
```

---

## 七、更新 uv 自身

```bash
uv self update
```
