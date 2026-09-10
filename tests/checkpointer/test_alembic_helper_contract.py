"""Golden contracts for helpers embedded in host Alembic history."""

from __future__ import annotations

import hashlib

from cubeloop.checkpointer.mysql import alembic_helpers as mysql
from cubeloop.checkpointer.postgres import alembic_helpers as postgres


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_postgres_v5_helper_output_is_immutable() -> None:
    expected = {
        "create_message_partitions_op": "57e8dd437c937ebefc18cdb48b627319391b5b3448c928d331f52e8fb617f852",
        "add_pending_request_column_op": "318dd4212e5132f47042b78fe9be9b7e07af021e22b652fa8075dd34aa538e88",
        "add_run_id_column_op": "3cd63949435a8b140ee47de04c765a8b52e125cd9cfefec95067db0f1a8164bf",
        "create_runs_partitions_op": "8c20242b0b93aa78683d0f68b341483632031486b409157177822a54470d7968",
        "upgrade_v3_to_v4_op": "69a190dbaa3c9ba842504a45fd54fc5d858d202faa6e025f72bf7882aa9853c6",
        "upgrade_v4_to_v5_op": "a1d3efc24502a2564a8a5aba3045a0df0c13af3000c35f18407cf48811451e56",
        "write_schema_version_op": "b2260726818d8ab4ea437f15cd6c91e272a703057b9ef6a106fe5f68130a55bc",
    }
    assert {name: _sha(getattr(postgres, name)()) for name in expected} == expected


def test_mysql_v5_helper_output_and_split_contract_are_immutable() -> None:
    expected = {
        "messages_partition_clause": "5b81448a7b4b98b02bf81e3fbabe49de2e274c609f878aacfe1773d09cb1f83a",
        "add_pending_request_column_op": "666f5c76247cf46ad3e4aef2b57402593108ceefa4fcf71f59a528d3424ab9e3",
        "add_run_id_column_op": "f601e64528136e153c4e4e0c2f37b0067378c7eeef97dc5f2a544c17c3ea1f9b",
        "runs_partition_clause": "5b81448a7b4b98b02bf81e3fbabe49de2e274c609f878aacfe1773d09cb1f83a",
        "add_messages_run_id_column_op": "8c0406e13ccc0c027df75b98a6825cfa88bf750930eaeb59090fa3208cf91a69",
        "create_runs_table_op": "c2fd37e2e2eb1ec2f88f0bd41c866b25cbf965012a061c34ced90031e45faf32",
        "create_runs_partitions_op": "5b81448a7b4b98b02bf81e3fbabe49de2e274c609f878aacfe1773d09cb1f83a",
        "upgrade_v3_to_v4_op": "e6d0a93e5e3abf4b27805fe428b813a0ae8d420e01c78964dc20988e2628dc31",
        "create_hitl_answers_table_op": "7d92af42b539319943f0a4d23dd51c884294d1f4281f1848a09f11992776a47c",
        "upgrade_v4_to_v5_op": "e71d86abf7d5b2d1534c577051c97faecc2386d37a9831063d1c603c64bdeb86",
        "write_schema_version_op": "555d7168cf0880e2a821b1199dbf8f35ee953db5daca4ca9e177005b4674e624",
    }
    assert {name: _sha(getattr(mysql, name)()) for name in expected} == expected
    assert (
        len([s for s in mysql.write_schema_version_op().split(";") if s.strip()]) == 2
    )
