"""Registration coverage for compact deep-inspection tools."""

from __future__ import annotations

from kicad_mcp.server import build_server


def test_full_profile_registers_deep_project_tools() -> None:
    names = {tool.name for tool in build_server("full").list_tools_sync()}

    assert {"project_get_deep_snapshot", "project_get_ascii_map", "pcb_get_route_plan"} <= names
