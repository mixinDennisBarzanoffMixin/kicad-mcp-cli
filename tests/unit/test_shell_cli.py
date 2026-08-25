"""Tests for the shell-first KiCad interface."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mcp import types as mcp_types

from kicad_mcp.compact_server import server as compact_server
from kicad_mcp.deep_inspection import ascii_map, filter_snapshot, placement_plan, route_plan
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


def _snapshot() -> dict[str, object]:
    return {
        "project": {"name": "demo", "directory": "demo"},
        "schematic": {
            "sheets": [{"number": 1, "name": "/Power/", "file": "power.kicad_sch"}],
            "components": [
                {"reference": "U1", "value": "MCU", "footprint": "QFN", "sheet": "/Power/"},
                {"reference": "R1", "value": "10k", "footprint": "R_0603", "sheet": "/Power/"},
            ],
            "nets": [
                {
                    "name": "GPIO",
                    "unconnected": False,
                    "nodes": [
                        {"reference": "U1", "pin": "1", "function": "IO", "type": "bidirectional"},
                        {"reference": "R1", "pin": "1", "function": "", "type": "passive"},
                    ],
                }
            ],
        },
        "board": {
            "bounds_mm": [0, 0, 20, 10],
            "footprints": [
                {
                    "reference": "U1",
                    "x_mm": 5,
                    "y_mm": 5,
                    "width_mm": 2,
                    "height_mm": 2,
                    "pads": [{"number": "1", "at": [5, 5], "net": "GPIO"}],
                },
                {
                    "reference": "R1",
                    "x_mm": 15,
                    "y_mm": 5,
                    "width_mm": 2,
                    "height_mm": 1,
                    "pads": [{"number": "1", "at": [15, 5], "net": "GPIO"}],
                },
            ],
            "tracks": [],
            "vias": [],
        },
    }


def test_deep_filter_and_ascii_zoom() -> None:
    snapshot = _snapshot()
    filtered = filter_snapshot(snapshot, reference="U1")

    assert [item["reference"] for item in filtered["schematic"]["components"]] == ["U1"]
    assert "COMPONENT MAP" in ascii_map(snapshot, zoom=1)
    assert "GPIO" in ascii_map(snapshot, zoom=2)
    assert "PCB MAP" in ascii_map(snapshot, zoom=3)


def test_route_plan_is_dry_and_refuses_critical_nets() -> None:
    snapshot = _snapshot()

    plan = route_plan(snapshot, "GPIO")
    refused = route_plan(snapshot, "+3V3")

    assert plan["status"] == "planned"
    assert len(plan["segments"]) == 1
    assert refused["status"] == "refused"


def test_route_plan_ignores_its_endpoint_footprints() -> None:
    plan = route_plan(_snapshot(), "GPIO")

    assert plan["collision_score"] == 0
    assert plan["routing_methods"] == ["direct-manhattan"]


def test_route_plan_uses_astar_around_footprint_obstacle() -> None:
    snapshot = copy.deepcopy(_snapshot())
    snapshot["board"]["footprints"].append(
        {
            "reference": "U2",
            "x_mm": 10,
            "y_mm": 5,
            "width_mm": 3,
            "height_mm": 3,
            "pads": [],
        }
    )

    plan = route_plan(snapshot, "GPIO", clearance_mm=0.5)

    assert plan["collision_score"] == 0
    assert plan["routing_methods"] == ["astar-grid"]
    assert len(plan["segments"]) >= 3


def test_placement_plan_is_deterministic_and_holds_fixed_references() -> None:
    first = placement_plan(_snapshot(), fixed_references=["U1"], iterations=20)
    second = placement_plan(_snapshot(), fixed_references=["U1"], iterations=20)

    assert first == second
    u1 = next(item for item in first["placements"] if item["reference"] == "U1")
    assert u1["fixed"] is True
    assert u1["from"] == u1["to"]


def test_shell_source_does_not_use_shell_execution() -> None:
    source = Path(__file__).parents[2] / "src" / "kicad_mcp" / "deep_inspection.py"
    text = source.read_text()
    assert "shell=True" not in text
