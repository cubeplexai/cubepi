from __future__ import annotations

from cubepi.deferred._catalog import render_catalog, render_expanded_schemas
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


def _make_group(
    group_id: str,
    display_name: str,
    description: str,
    tool_names: list[str],
) -> DeferredToolGroup:
    async def _noop_loader():
        return []

    return DeferredToolGroup(
        group_id=group_id,
        display_name=display_name,
        description=description,
        tool_names=tool_names,
        loader=_noop_loader,
    )


class TestRenderCatalog:
    def test_no_groups_returns_empty(self) -> None:
        result = render_catalog(groups=[], expanded={})
        assert result == ""

    def test_single_group_no_expansion(self) -> None:
        groups = [
            _make_group(
                "mcp:github", "GitHub", "Code hosting", ["create_issue", "search_repos"]
            )
        ]
        result = render_catalog(groups=groups, expanded={})
        assert "mcp:github" in result
        assert "GitHub" in result
        assert "Code hosting" in result
        assert "2 tools" in result
        assert "create_issue" in result
        assert "search_repos" in result

    def test_sorted_by_group_id(self) -> None:
        groups = [
            _make_group("z:last", "Last", "desc", ["t1"]),
            _make_group("a:first", "First", "desc", ["t2"]),
        ]
        result = render_catalog(groups=groups, expanded={})
        a_pos = result.index("a:first")
        z_pos = result.index("z:last")
        assert a_pos < z_pos

    def test_byte_stable_across_input_orderings(self) -> None:
        g1 = _make_group("mcp:a", "A", "desc", ["t1"])
        g2 = _make_group("mcp:b", "B", "desc", ["t2"])
        result_ab = render_catalog(groups=[g1, g2], expanded={})
        result_ba = render_catalog(groups=[g2, g1], expanded={})
        assert result_ab == result_ba

    def test_fully_expanded_group_omitted(self) -> None:
        groups = [
            _make_group(
                "mcp:github", "GitHub", "Code hosting", ["create_issue", "search_repos"]
            ),
            _make_group("mcp:linear", "Linear", "Issues", ["create_issue"]),
        ]
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        result = render_catalog(groups=groups, expanded=expanded)
        assert "mcp:github" not in result
        assert "mcp:linear" in result

    def test_partially_expanded_shows_remaining(self) -> None:
        groups = [
            _make_group(
                "mcp:github",
                "GitHub",
                "Code hosting",
                ["create_issue", "search_repos", "create_pr"],
            )
        ]
        expanded: dict[str, list[str] | None] = {"mcp:github": ["create_issue"]}
        result = render_catalog(groups=groups, expanded=expanded)
        assert "2 remaining tools" in result
        assert "create_issue" not in result
        assert "search_repos" in result
        assert "create_pr" in result

    def test_all_groups_fully_expanded_returns_empty(self) -> None:
        groups = [_make_group("mcp:github", "GitHub", "desc", ["t1"])]
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        result = render_catalog(groups=groups, expanded=expanded)
        assert result == ""

    def test_custom_header(self) -> None:
        groups = [_make_group("mcp:a", "A", "desc", ["t1"])]
        result = render_catalog(groups=groups, expanded={}, header="Custom header text")
        assert "Custom header text" in result


class TestRenderExpandedSchemas:
    def test_no_expansions_returns_empty(self) -> None:
        result = render_expanded_schemas(expanded_schemas=[])
        assert result == ""

    def test_single_expansion(self) -> None:
        schemas = [
            (
                "mcp:github",
                [
                    {
                        "name": "create_issue",
                        "description": "Create an issue",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            )
        ]
        result = render_expanded_schemas(expanded_schemas=schemas)
        assert "mcp:github" in result
        assert "create_issue" in result
        assert "Create an issue" in result

    def test_expansion_order_preserved(self) -> None:
        schemas = [
            ("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}]),
            ("mcp:github", [{"name": "t2", "description": "d2", "parameters": {}}]),
        ]
        result = render_expanded_schemas(expanded_schemas=schemas)
        linear_pos = result.index("mcp:linear")
        github_pos = result.index("mcp:github")
        assert linear_pos < github_pos

    def test_append_only_prefix_stable(self) -> None:
        schemas_v1 = [
            ("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}])
        ]
        schemas_v2 = [
            ("mcp:linear", [{"name": "t1", "description": "d1", "parameters": {}}]),
            ("mcp:github", [{"name": "t2", "description": "d2", "parameters": {}}]),
        ]
        result_v1 = render_expanded_schemas(expanded_schemas=schemas_v1)
        result_v2 = render_expanded_schemas(expanded_schemas=schemas_v2)
        assert result_v2.startswith(result_v1)
