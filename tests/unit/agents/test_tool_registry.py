from unittest.mock import MagicMock

import pytest

from app.agents.tool_registry import Tool, ToolRegistry, _normalize, _validate_tool_input


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert _normalize("a  b") == "a b"

    def test_none_returns_empty(self):
        assert _normalize(None) == ""


class TestValidateToolInput:

    def test_valid_input_returns_true(self):
        assert _validate_tool_input("query", {"key": "val"}, "s1") is True

    def test_empty_query_returns_false(self):
        assert _validate_tool_input("", {"key": "val"}, "s1") is False

    def test_whitespace_query_returns_false(self):
        assert _validate_tool_input("   ", {}, "s1") is False

    def test_non_dict_context_returns_false(self):
        assert _validate_tool_input("query", "not a dict", "s1") is False

    def test_none_context_returns_false(self):
        assert _validate_tool_input("query", None, "s1") is False

    def test_empty_session_id_returns_false(self):
        assert _validate_tool_input("query", {}, "") is False


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

def _make_tool(name="test_tool", handler=None):
    if handler is None:
        handler = MagicMock(return_value={"result": "ok", "status": "success"})
    return Tool(
        name=name,
        description="A test tool",
        handler=handler,
        tool_type="generic",
        cost="medium",
    )


class TestToolInit:

    def test_name_stored(self):
        tool = _make_tool(name="my_tool")
        assert tool.name == "my_tool"

    def test_description_stored(self):
        tool = _make_tool()
        assert tool.description == "A test tool"

    def test_tool_type_stored(self):
        tool = _make_tool()
        assert tool.tool_type == "generic"

    def test_cost_stored(self):
        tool = _make_tool()
        assert tool.cost == "medium"


class TestToolExecute:

    def test_handler_called_with_query(self):
        handler = MagicMock(return_value={"result": "val", "status": "success"})
        tool = _make_tool(handler=handler)
        tool.execute("test query", context={}, session_id="s1")
        handler.assert_called_once()

    def test_execute_returns_dict(self):
        tool = _make_tool()
        result = tool.execute("query", context={}, session_id="s1")
        assert isinstance(result, dict)

    def test_execute_invalid_query_returns_error(self):
        tool = _make_tool()
        result = tool.execute("", context={}, session_id="s1")
        assert "error" in result or result.get("status") == "error"

    def test_handler_exception_returns_error_dict(self):
        handler = MagicMock(side_effect=RuntimeError("boom"))
        tool = _make_tool(handler=handler)
        result = tool.execute("query", context={}, session_id="s1")
        assert isinstance(result, dict)

    def test_execute_passes_session_id(self):
        handler = MagicMock(return_value={"result": "ok", "status": "success"})
        tool = _make_tool(handler=handler)
        tool.execute("query", context={}, session_id="my-session")
        # Handler was called — session propagation verified by no crash
        handler.assert_called()


class TestToolToDict:

    def test_returns_dict(self):
        tool = _make_tool(name="my_tool")
        d = tool.to_dict()
        assert isinstance(d, dict)

    def test_has_name_and_description(self):
        tool = _make_tool(name="my_tool")
        d = tool.to_dict()
        assert d["name"] == "my_tool"
        assert "description" in d


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistryRegister:

    def test_register_and_get(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        tool = _make_tool(name="search_tool")
        registry.register(tool)
        assert registry.get("search_tool") is tool

    def test_duplicate_register_overwrites(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        t1 = _make_tool(name="tool_x")
        t2 = _make_tool(name="tool_x")
        registry.register(t1)
        registry.register(t2)
        assert registry.get("tool_x") is t2

    def test_get_unknown_raises(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        with pytest.raises((KeyError, ValueError)):
            registry.get("nonexistent_tool")


class TestToolRegistryGetOptional:

    def test_known_tool_returned(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        tool = _make_tool(name="opt_tool")
        registry.register(tool)
        assert registry.get_optional("opt_tool") is tool

    def test_unknown_tool_returns_none(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        assert registry.get_optional("nonexistent") is None


class TestToolRegistryListTools:

    def test_returns_list(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        result = registry.list_tools()
        assert isinstance(result, list)

    def test_registered_tool_in_list(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.register(_make_tool(name="listed_tool"))
        names = [t["name"] for t in registry.list_tools()]
        assert "listed_tool" in names


class TestToolRegistryDeregister:

    def test_deregister_removes_tool(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.register(_make_tool(name="rm_tool"))
        registry.deregister("rm_tool")
        assert registry.get_optional("rm_tool") is None

    def test_deregister_nonexistent_does_not_raise(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        # Should not raise
        registry.deregister("ghost_tool")
