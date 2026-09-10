---
title: 从 cubepi 迁移
description: "CubePi 在 0.14 更名为 CubeLoop；安装、导入、持久化和 trace 的迁移方法。"
---

# 从 cubepi 迁移

0.14 将项目从 CubePi / `cubepi` 更名为 CubeLoop / `cubeloop`。GitHub 仓库
原地更名为 `cubeplexai/cubeloop`，文档站为 https://cubeloop.dev。

## 安装与导入

```bash
pip install cubeloop
# extras 不变：cubeloop[sqlite]、cubeloop[postgres]、cubeloop[mysql]、…
```

```python
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider
```

`pip install -U cubepi` 仍可使用：0.14.1+ 是兼容包装，依赖同一 0.14 系列的
CubeLoop，重新导出公共 API，并为 `cubepi.*` 深层导入提供别名。首次导入会告警，
新项目应直接依赖 `cubeloop`。

CLI 改为 `cubeloop trace`；包装包仍提供会告警的 `cubepi trace`。

## Checkpointer schema 保持 v5

Postgres 和 MySQL 的物理表名、分区名和索引名继续使用 `cubepi_*`。这些名字属于
持久化协议，不是产品品牌。schema version 保持 5，因此已有 0.13.6 数据库升级到
0.14.1 不需要数据库迁移。SQLite 表名从未加品牌前缀，保持不变。

历史 Alembic revision 可以继续从 `cubeloop.checkpointer.*.alembic_helpers`
导入 helper；这些 helper 保持原来的 v1–v5 SQL。不要添加表名 rename revision。

### 从已撤回的 0.14.0 紧急恢复

0.14.0 曾短暂把数据库对象改成 `cubeloop_*`，因此已被 yank。只有确实执行过
那条迁移时才使用下面的恢复脚本。先停止全部应用实例并完成可验证备份，再用
数据库管理客户端执行对应脚本：

- [Postgres v6→v5 恢复脚本](/recovery/0.14.0/postgres-v6-to-v5.sql)
- [MySQL v6→v5 恢复脚本](/recovery/0.14.0/mysql-v6-to-v5.sql)

脚本会在修改前拒绝混合或冲突 schema。完成后核对表和行数，再部署 0.14.1。
正常 v5 数据库不要执行这些脚本。

## Tracing

新 span 使用 `cubeloop.*` 属性和 `cubeloop.turn` span 名。`cubeloop trace` 仍能读取
0.13 中使用 `cubepi.*` 的 JSONL；你自己的 dashboard 需要更新属性名。默认 JSONL
目录为 `./cubeloop-traces`。

## 环境变量

`CUBELOOP_TEST_PG_DSN`、`CUBELOOP_TEST_MYSQL_DSN`、`CUBELOOP_PG_DSN`、
`CUBELOOP_MYSQL_DSN` 替换对应的 `CUBEPI_*` 名称。新名称未设置时，0.x 的剩余
周期仍会读取旧名称。
