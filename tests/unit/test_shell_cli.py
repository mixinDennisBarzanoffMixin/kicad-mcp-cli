"""Tests for the shell-first KiCad interface."""

from __future__ import annotations

import argparse
import json

from mcp import types as mcp_types

from kicad_mcp.compact_server import server as compact_server
from kicad_mcp.shell_cli import parse_call_arguments, result_envelope


def test_parse_call_arguments_merges_json_and_set_values() -> None:
    args = argparse.Namespace(
        args_json='{"layer":"F.Cu","count":1}',
        set_values=["count=2", "enabled=true", "label=USB"],
    )

    assert parse_call_arguments(args) == {
        "layer": "F.Cu",
        "count": 2,
        "enabled": True,
        "label": "USB",
    }


def test_result_envelope_preserves_structured_and_text_content() -> None:
    result = mcp_types.CallToolResult(
        isError=False,
        structuredContent={"result": {"version": "10.0"}},
        content=[mcp_types.TextContent(type="text", text=json.dumps({"version": "10.0"}))],
    )

    payload = result_envelope("kicad_get_version", result)

    assert payload["ok"] is True
    assert payload["tool"] == "kicad_get_version"
    assert payload["result"] == {"result": {"version": "10.0"}}
    assert payload["content"] == [{"type": "text", "text": '{"version": "10.0"}'}]


async def test_compact_server_exposes_only_discover_run_and_grep() -> None:
    tool_names = {tool.name for tool in await compact_server.list_tools()}

    assert tool_names == {"kicad_catalog", "kicad_run", "kicad_grep"}
