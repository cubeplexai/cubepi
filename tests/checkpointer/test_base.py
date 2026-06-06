from cubepi.checkpointer.base import CheckpointData


def test_checkpoint_data_default_parent_is_none():
    cd = CheckpointData()
    assert cd.parent_thread_id is None
    assert cd.messages == []
    assert cd.extra == {}


def test_checkpoint_data_with_parent():
    cd = CheckpointData(parent_thread_id="src_thread")
    assert cd.parent_thread_id == "src_thread"


def test_checkpoint_data_keyword_construction_unchanged():
    cd = CheckpointData(messages=[], extra={"k": "v"})
    assert cd.extra == {"k": "v"}
    assert cd.parent_thread_id is None
