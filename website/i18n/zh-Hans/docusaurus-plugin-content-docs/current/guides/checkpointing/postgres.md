---
title: Postgres 检查点
description: "使用 PostgresCheckpointer 实现生产级的 agent 状态持久化。"
---

# Postgres 检查点

`PostgresCheckpointer` 是生产级持久化后端。它使用 `asyncpg` 搭配连接池、
`msgpack` 编码 payload、以及每线程的 Postgres advisory lock，让多个进程
可以安全地写入同一个 `thread_id` 而不会互相干扰。

安装 extra：

```bash
pip install "cubepi[postgres]"
```

会拉入 `asyncpg`、`sqlalchemy` 和 `msgpack`。

## 基本用法

```python
import asyncio
from cubepi import Agent, Model
from cubepi.checkpointer import PostgresCheckpointer
from cubepi.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(api_key="…")
    async with PostgresCheckpointer("postgresql://user:pass@host/dbname") as cp:
        agent = Agent(
            provider=provider,
            model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("hello")


asyncio.run(main())
```

DSN 接受 `asyncpg.create_pool(...)` 支持的任何格式。连接池大小：

```python
async with PostgresCheckpointer(
    "postgresql://…",
    min_pool_size=2,
    max_pool_size=20,
) as cp:
    …
```

## Schema

checkpointer 需要三张表：`cubepi_threads`、`cubepi_messages` 和
`cubepi_schema_version`。与 SQLite 不同，CubePi **不会自动创建这些表**——
它只在 `__aenter__` 时验证它们是否存在且 `schema_version` 是否匹配预期。

如果表不存在，你会收到 `CubepiSchemaUninitialized`。如果版本与本版
CubePi 不匹配，你会收到 `CubepiSchemaMismatch`。

原因：生产数据库属于宿主应用的迁移系统（Alembic、Atlas……），
不属于一个可能与你现有迁移冲突的第三方库。

### 通过 Alembic 引导 {#bootstrapping-via-alembic}

CubePi 暴露了 SQLAlchemy `MetaData`，让你的迁移可以采用其 schema：

```python
# alembic/env.py
from cubepi.checkpointer.postgres import cubepi_metadata, EXPECTED_SCHEMA_VERSION

target_metadata = [my_app_metadata, cubepi_metadata]
```

然后生成一个 revision 并执行。迁移还必须在 `cubepi_schema_version` 中
INSERT schema 版本。使用辅助函数：

```python
# 在迁移的 upgrade() 中：
from cubepi.checkpointer.postgres.alembic_helpers import (
    create_message_partitions_op,
    write_schema_version_op,
)

def upgrade():
    op.create_table(...)                            # 从 cubepi_metadata 自动生成
    op.execute(create_message_partitions_op())      # 创建 64 个哈希分区
    op.execute(write_schema_version_op())           # 记录 EXPECTED_SCHEMA_VERSION
```

两个辅助函数都返回 SQL 字符串——你需要传入 `op.execute(...)`。
`write_schema_version_op()` 是幂等的：它会删除之前 CubePi 版本的所有行，
然后插入当前版本。

当 CubePi 后续升级并提升了 `EXPECTED_SCHEMA_VERSION` 时，生成一个新的
revision，再次调用 `op.execute(write_schema_version_op())`。

## 数据模型

```
cubepi_threads
    thread_id (PK)
    parent_thread_id   -- 用于 fork
    forked_at_seq      -- fork 点处的序列号
    extra              -- JSONB
    created_at / updated_at

cubepi_messages
    thread_id, seq     -- 复合 PK；按 HASH(thread_id) 分为 64 个分区
    role               -- "user" | "assistant" | "tool"
    metadata           -- JSONB（通过 GIN 索引）
    payload            -- bytea (msgpack)
    created_at

cubepi_schema_version
    version (PK)
```

重要属性：

- **`(thread_id, seq)` 是消息标识。** `seq` 在每线程中单调递增，
  在 `pg_advisory_xact_lock(hashtext(thread_id))` 下分配。
  两个对同一线程的并发写入者会干净地序列化。
- **`payload` 是 msgpack 编码的 `model.model_dump(mode="json")`。**
  CubePi 在读取时重建 Pydantic 模型。
- **`metadata` 是 JSONB，可查询。** 完整消息的 payload 内部也包含
  `metadata`，但这一列是 SQL 查询的规范视图。
- **表按 `HASH(thread_id)` 分为 64 个分区。** 跨分区均匀分布，
  无每线程瓶颈。

## 并发

advisory lock 让同一线程上的追加操作跨进程安全：

```python
# 进程 A 和进程 B 同时追加到线程 "user-42"。
# 它们通过 pg_advisory_xact_lock 序列化，各自获得连续的 seq。
```

读取（`load`）不取锁——它们在事务内是快照一致性的。

默认连接池 `min=1, max=10` 对大多数应用足够；如果你的并发 agent 数
很高，请调大 `max_pool_size`。

## `save_extra` 语义

`save_extra` 做的是 JSONB 合并，而不是替换：

```sql
extra = cubepi_threads.extra || EXCLUDED.extra
```

所以先写 `{"foo": 1}` 再写 `{"bar": 2}` 会得到 `{"foo": 1, "bar": 2}`。
中间件可以安全地写入部分 dict 而不会丢失先前的键。

## Fork

`parent_thread_id` + `forked_at_seq` 列是为将来的 fork 支持预留的。
CubePi v0.3 尚未暴露 fork API——现在写入它们是为了保持 schema
的前向兼容性。

## 常见坑

- **`CubepiSchemaUninitialized`** —— 数据库为空或迁移未运行。
  先执行宿主 alembic upgrade。
- **`CubepiSchemaMismatch`** —— 你升级了 cubepi 但未生成新的迁移。
  生成一个、执行它，CubePi 就会启动。

  :::info Schema v2（HITL）

  cubepi ≥ HITL 版本将 `EXPECTED_SCHEMA_VERSION` 从 1 提升到 2，
  并在 `cubepi_threads` 表中新增 `pending_request JSONB NULL` 列。
  宿主 alembic upgrade 必须在提升 schema_version 行之前调用
  `add_pending_request_column_op()`（来自
  `cubepi.checkpointer.postgres.alembic_helpers`）。
  完整的跨进程流程请参阅 [HITL 指南](../hitl)。
  :::
- **负载下连接池耗尽** —— 默认 `max_pool_size=10`。
  如果应用的并发 agent 数超过此值，请调大。
- **`asyncpg.exceptions.UndefinedTableError` 在 `__aenter__` 外部** ——
  表示你在 `async with` 之外使用了 checkpointer。连接池尚未连接。
  请用 context manager 包裹。
- **混用宿主 SQLAlchemy `MetaData`** —— CubePi 自带独立的
  `MetaData` 实例，正是为了能与你的应用模型共存而不冲突。
  不要将它们合并到你的全局 metadata 中——分别传给 Alembic。

## 另请参阅

- [SQLite 检查点](./sqlite) —— 单进程替代方案。
- [自定义后端](./custom) —— Protocol 详情。
- [配方 → Postgres + FastAPI 服务](../../recipes/postgres-fastapi)
  —— 一个可部署的 HTTP 前端 agent。
