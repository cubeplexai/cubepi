---
title: From cubepi
description: "CubePi 在 0.14 更名为 CubeLoop。如何更新安装、导入、checkpointer schema 和 traces。"
---

# 从 cubepi 迁移

0.14 把项目从 CubePi / `cubepi` 更名为 CubeLoop / `cubeloop`。
GitHub 仓库原地改名为 `cubeplexai/cubeloop`。文档站是 https://cubeloop.dev。

## 安装与导入

```bash
pip install cubeloop
# extras 不变：cubeloop[sqlite]、cubeloop[postgres]、cubeloop[mysql]、…
```

```python
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider
```

`pip install -U cubepi` 仍然可用：0.14+ 是一层包装，依赖
`cubeloop==0.14.0`，再导出公共 API，并把 `cubepi.*` 导入别名到新包。
第一次 `import cubepi` 会告警。请直接依赖 `cubeloop`。

CLI：`cubeloop trace`（包装层仍提供 `cubepi trace`，并告警）。

## Checkpointer schema v5 → v6

Postgres 和 MySQL 表名从 `cubepi_*` 改成 `cubeloop_*`。
SQLite 表名从未加前缀，仍是 `messages` / `runs` / …。

已有数据库：加一条 Alembic revision，先跑 `upgrade_v5_to_v6_op()`，再跑
`write_schema_version_op()`。用 0.14 打开未迁移的 v5 库会抛
`CubeloopSchemaMismatch`，并指向这个 helper — 不是「表不存在」。

如果有数据表但没有 `*_schema_version` 表，不要跑全新的 v6 CREATE TABLE。
先创建 `cubepi_schema_version` 并写入 version 5，再跑 v5→v6 helper。

包升级和这条 revision **必须一起上**。进程已经 import 0.14、却对着仍是
v5 的库打开 checkpointer，会拒绝启动。

### 不要把历史 helper 的 import 一刀切改掉

0.14 把 Alembic helper 分成两类：

| Helper | 0.14 发出的 SQL |
|---|---|
| `create_message_partitions_op()`、`create_runs_partitions_op()` | **当前**名字：`cubeloop_messages_pXX PARTITION OF cubeloop_messages`，runs 同理。仅 Postgres。 |
| `add_pending_request_column_op()`、`add_run_id_column_op()`、`upgrade_v3_to_v4_op()`、`upgrade_v4_to_v5_op()` | **历史**名字：仍是 `cubepi_*`。可以从 `cubeloop.checkpointer.postgres.alembic_helpers` 导入。 |
| `write_schema_version_op()` | 把 `EXPECTED_SCHEMA_VERSION`（现在是 **6**）写进已存在的 `cubeloop_schema_version`，否则写进 `cubepi_schema_version`。 |

常见的 host v1 revision 会先 `CREATE cubepi_messages`，再调用
`create_message_partitions_op()`。依赖 cubeloop 0.14 之后，同一次调用会发出
`PARTITION OF cubeloop_messages`。空库 `alembic upgrade head`（从头 replay
v1→v6）会在 v1 失败：cubeloop 父表还不存在。

**改法：** 历史 revision 里不要再调用 `create_message_partitions_op()`。
把原来的 SQL 内联到旧父表上：

```sql
CREATE TABLE cubepi_messages_p00
  PARTITION OF cubepi_messages
  FOR VALUES WITH (modulus 64, remainder 0);
-- … p01 … p63
```

如果某条历史 revision 调用过 `create_runs_partitions_op()`，同样会踩坑。
CubeLoop 自带的 `upgrade_v3_to_v4_op()` 已经内联了 `cubepi_runs_pXX`，可以安全导入。

内联要发生在去掉 `cubepi` 包 **之前**（或与 pin 同一批改动）。中间不要对空库跑
Alembic：v1 在你改完剩余 helper import 之前仍是 `import cubepi.checkpointer…`，
pin 之后这个模块就不在了。

`write_schema_version_op()` 在 v1 replay 时写入 6 是故意的 — 它一直写的是
*当前* expected version，`alembic upgrade head` 会一次跑完 v1→v6。不要改成手写
1/2/3/4/5。

### Autogenerate 可能 DROP `cubepi_threads`

`cubeloop_metadata` 描述的是 **新** 名字。Alembic `target_metadata` 指过去之后，
仍停在 v5 的库里却是 `cubepi_*`。Autogenerate 会提议 `DROP cubepi_*` 再
`CREATE cubeloop_*`（空表）。最常见的受害者是 `cubepi_threads`：很多 host 的 v1
是 autogen 建出来的，从未放进排除列表。

这次 DROP 会 CASCADE 掉 messages、runs 和 HITL answers。

**改法：生成 v6 revision 之前，先在 `env.py` 里把两代表都排除：**

- 表：`cubepi_threads`、`cubepi_messages`、`cubepi_runs`、
  `cubepi_hitl_answers`、`cubepi_schema_version`，以及对应的 `cubeloop_*`。
- Postgres 分区：前缀 `cubepi_messages_p`、`cubepi_runs_p`、
  `cubeloop_messages_p`、`cubeloop_runs_p`。

检查生成出来的 v6 文件：不能出现 `DROP TABLE cubepi_threads`，也不能出现全新的
`CREATE TABLE cubeloop_*`。正文应是 `upgrade_v5_to_v6_op()` 加上
`write_schema_version_op()`。

### Downgrade 是反向 rename，不是 DROP

`upgrade_v5_to_v6_op()` 会 rename 父表、**每一张** 64 张 message 分区和 64 张
run 分区（Postgres）、HITL 和 version 表，以及 `ix_cubepi_*` 索引。PostgreSQL
对分区父表做 `ALTER TABLE … RENAME TO` **不会** 改子分区名字。

从更早的 bump 抄 `DROP TABLE cubeloop_runs CASCADE` 会删掉全部会话。应该反向
rename（含每一张分区），再把 version 5 写回 `cubepi_schema_version`。

## Tracing

新 span 使用 `cubeloop.*` 属性和 `cubeloop.turn` span 名。
`cubeloop trace` 仍能读 0.13 里用 `cubepi.*` 的 JSONL。
你自己的 dashboard 需要改属性名。

默认 JSONL 目录是 `./cubeloop-traces`。不会自动回退到 `./cubepi-traces`；读旧文件请传
`--dir`（或沿用你原来的配置路径）。

## 环境变量

`CUBELOOP_TEST_PG_DSN`、`CUBELOOP_TEST_MYSQL_DSN`、`CUBELOOP_PG_DSN`、
`CUBELOOP_MYSQL_DSN` 替换原来的 `CUBEPI_*`。新名字未设置时，0.x 剩余周期内仍会读旧名字。
