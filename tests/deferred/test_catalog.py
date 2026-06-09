from __future__ import annotations

import pytest

from cubepi.deferred.types import DeferredToolGroup


class TestDeferredToolGroup:
    def test_basic_construction(self) -> None:
        async def _loader():
            return []

        group = DeferredToolGroup(
            group_id="mcp:github",
            display_name="GitHub",
            description="Code hosting: issues, PRs, repos",
            tool_names=["create_issue", "search_repos"],
            loader=_loader,
        )
        assert group.group_id == "mcp:github"
        assert group.display_name == "GitHub"
        assert group.description == "Code hosting: issues, PRs, repos"
        assert group.tool_names == ["create_issue", "search_repos"]
        assert group.loader is _loader
