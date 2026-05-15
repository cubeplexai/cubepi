---
title: 安装
---

# 安装

CubePi 需要 **Python 3.11 或以上**。核心运行时只有三个依赖：`pydantic`、
`anthropic`、`openai`。可选功能（SQLite、Postgres、MCP）通过 extras
按需安装,不用的话不会被拉进来。

## 使用 pip

```bash
pip install cubepi
```

可选 extras:

```bash
pip install "cubepi[sqlite]"     # 安装 aiosqlite,启用 SQLiteCheckpointer
pip install "cubepi[postgres]"   # 安装 asyncpg + sqlalchemy + msgpack
pip install "cubepi[mcp]"        # 安装 MCP SDK,启用 MCP 工具加载器
pip install "cubepi[sqlite,mcp]" # 组合
```

## 使用 uv

[`uv`](https://github.com/astral-sh/uv) 比 pip 快很多,是推荐的工作流：

```bash
uv add cubepi
uv add "cubepi[sqlite,postgres,mcp]"
```

在已有 uv 项目里,改完 `pyproject.toml` 后 `uv sync` 会重新锁定环境。

## 使用 Poetry

```bash
poetry add cubepi
poetry add "cubepi[sqlite,postgres,mcp]"
```

## 验证安装

```bash
python -c "import cubepi; print(cubepi.__doc__)"
# cubepi — Pythonic async-native agent framework.
```

如果报 `ImportError`,大概率是解释器版本低于 3.11——用 `python --version`
确认一下。

## 配置 provider 凭据

CubePi 的 provider 从构造函数参数读取凭据。大多数部署会从环境变量
取出来：

```python
import os
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
```

你也可以传 `base_url=...` 指向自托管端点或兼容代理（如 Anthropic Bedrock、
LiteLLM、vLLM）。

[FauxProvider](../guides/providers/custom#using-fauxprovider-in-tests)
（用于测试）不需要任何凭据。

## extras 选择指南

| Extra | 拉入什么 | 什么时候装 |
|---|---|---|
| (无) | 仅核心 | 只需要内存里的状态,不用 MCP |
| `[sqlite]` | `aiosqlite` | 单进程应用需要落盘 |
| `[postgres]` | `asyncpg`、`sqlalchemy`、`msgpack` | 多实例 / 生产环境——见 [Postgres 指南](../guides/checkpointing/postgres) |
| `[mcp]` | `mcp` | 想把 MCP server 工具挂到 Agent 上 |
| `[docs]` | `griffe` | 仅文档站构建（贡献者用） |

## 下一步

- [快速开始](./quick-start) —— 五分钟内跑通你的第一个 Agent。
- [核心概念](./core-concepts) —— 在动手前先弄清 `Agent` / `Tool` /
  `Provider` / `Checkpointer` 各自是什么。
